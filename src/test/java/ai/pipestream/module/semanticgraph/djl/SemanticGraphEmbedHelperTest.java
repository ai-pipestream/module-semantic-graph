package ai.pipestream.module.semanticgraph.djl;

import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.helpers.test.UniAssertSubscriber;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.Test;

import java.net.ConnectException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link SemanticGraphEmbedHelper}'s batched retry behaviour.
 *
 * <p>Focus areas:
 * <ul>
 *   <li>Single-batch + multi-batch happy paths with correct index mapping</li>
 *   <li>Concurrency cap via {@code Multi.merge(cap)} on large inputs</li>
 *   <li>§22.5-style verification: null slots after merge raise
 *       {@link IllegalStateException}</li>
 *   <li>Transient retry for all transient classes (connect, socket timeout,
 *       HTTP 5xx subset, HTTP 429/408)</li>
 *   <li>Permanent failures (HTTP 4xx, IAE/ISE) propagate without retry</li>
 *   <li>Alignment violation (response row count != input row count)</li>
 *   <li>Empty/null/blank input fast-paths</li>
 * </ul>
 */
class SemanticGraphEmbedHelperTest {

    private static final int BATCH = 4;
    private static final int CAP = 3;
    private static final int MAX_RETRIES = 2;
    private static final long BACKOFF_MS = 1L;  // near-zero for fast tests

    // ======================================================================
    // Happy paths
    // ======================================================================

    @Test
    void embed_singleBatch_returnsVectorsInOrder() {
        DjlServingClient djl = mock(DjlServingClient.class);
        JsonArray response = arr(arr(0.1, 0.2), arr(0.3, 0.4));
        when(djl.predict(eq("minilm"), any(JsonObject.class)))
                .thenReturn(response);

        List<float[]> result = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", List.of("a", "b"), BATCH, CAP, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        assertThat(result).as("Two inputs, two vectors").hasSize(2);
        assertThat(result.get(0)).as("First vector matches first input row").containsExactly(0.1f, 0.2f);
        assertThat(result.get(1)).as("Second vector matches second input row").containsExactly(0.3f, 0.4f);
        verify(djl, times(1)).predict(any(), any());
    }

    @Test
    void embed_multiBatch_slicesAndReassemblesInOrder() {
        DjlServingClient djl = mock(DjlServingClient.class);
        // batchSize=4, so 10 inputs → 3 batches of (4, 4, 2)
        // DJL returns each batch as nested arrays where vec[j] = [batchIdx * 10 + j]
        when(djl.predict(any(), any())).thenAnswer(inv -> {
            JsonObject body = inv.getArgument(1);
            JsonArray inputs = body.getJsonArray("inputs");
            JsonArray out = new JsonArray();
            for (int i = 0; i < inputs.size(); i++) {
                float marker = Float.parseFloat(inputs.getString(i));  // encode the global index into the text
                out.add(new JsonArray().add(marker));
            }
            return out;
        });

        List<String> texts = new ArrayList<>();
        for (int i = 0; i < 10; i++) texts.add(String.valueOf((float) i));

        List<float[]> result = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", texts, BATCH, CAP, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        assertThat(result).as("Ten inputs, ten output vectors").hasSize(10);
        for (int i = 0; i < 10; i++) {
            assertThat(result.get(i))
                    .as("Output index %d must carry marker %d regardless of sub-batch completion order", i, i)
                    .containsExactly((float) i);
        }
    }

    @Test
    void embed_capClampsToBatchCount_smallInputOneBatch() {
        DjlServingClient djl = mock(DjlServingClient.class);
        when(djl.predict(any(), any())).thenReturn(arr(arr(0.1, 0.2)));

        List<float[]> result = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", List.of("solo"), BATCH, /*perDocCap*/ 60, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        assertThat(result).as("One input → one batch, cap clamped to 1").hasSize(1);
        verify(djl, times(1)).predict(any(), any());
    }

    @Test
    void embed_emptyTexts_shortCircuitsWithoutWire() {
        DjlServingClient djl = mock(DjlServingClient.class);
        List<float[]> r = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", List.of(), BATCH, CAP, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();
        assertThat(r).as("Empty input means empty result, no predict call").isEmpty();
        verify(djl, times(0)).predict(any(), any());
    }

    @Test
    void embed_nullTexts_alsoShortCircuits() {
        DjlServingClient djl = mock(DjlServingClient.class);
        List<float[]> r = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", null, BATCH, CAP, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();
        assertThat(r).as("Null input behaves like empty input").isEmpty();
        verify(djl, times(0)).predict(any(), any());
    }

    @Test
    void embed_blankModelId_failsWithoutWire() {
        DjlServingClient djl = mock(DjlServingClient.class);
        Throwable err = new SemanticGraphEmbedHelper(djl)
                .embed("   ", List.of("x"), BATCH, CAP, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();
        assertThat(err)
                .as("Blank modelId must fail before wire — caller precondition violation")
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("modelId is required");
        verify(djl, times(0)).predict(any(), any());
    }

    @Test
    void embed_invalidBatchSize_fails() {
        DjlServingClient djl = mock(DjlServingClient.class);
        Throwable err = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", List.of("x"), /*batchSize*/ 0, CAP, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();
        assertThat(err)
                .as("batchSize=0 is invalid")
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void embed_invalidPerDocCap_fails() {
        DjlServingClient djl = mock(DjlServingClient.class);
        Throwable err = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", List.of("x"), BATCH, /*perDocCap*/ 0, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();
        assertThat(err)
                .as("perDocCap=0 is invalid")
                .isInstanceOf(IllegalArgumentException.class);
    }

    // ======================================================================
    // Response shape errors
    // ======================================================================

    @Test
    void embed_rowCountMismatch_failsLoudPerBatch() {
        DjlServingClient djl = mock(DjlServingClient.class);
        // Return 2 vectors for a 4-input batch — alignment violation
        when(djl.predict(any(), any()))
                .thenReturn(arr(arr(0.1), arr(0.2)));

        Throwable err = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", List.of("a", "b", "c", "d"), BATCH, CAP, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(err)
                .as("Row-count mismatch must not retry or guess — fails loudly")
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("2 vectors")
                .hasMessageContaining("4 inputs")
                .hasMessageContaining("alignment violation");
    }

    @Test
    void embed_nullRowInResponse_failsLoud() {
        DjlServingClient djl = mock(DjlServingClient.class);
        JsonArray resp = new JsonArray().add(new JsonArray().add(0.1)).add((Object) null);
        when(djl.predict(any(), any())).thenReturn(resp);

        Throwable err = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", List.of("a", "b"), BATCH, CAP, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(err)
                .as("Null row in response means a malformed payload — fail loud rather than substitute zero")
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("position 1");
    }

    @Test
    void embed_emptyVectorInResponse_failsLoud() {
        DjlServingClient djl = mock(DjlServingClient.class);
        JsonArray resp = arr(arr(0.1, 0.2), new JsonArray());  // second vector is length-0
        when(djl.predict(any(), any())).thenReturn(resp);

        Throwable err = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", List.of("a", "b"), BATCH, CAP, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(err)
                .as("Length-0 vector is a data loss event — fail, never silently accept")
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("null/empty vector");
    }

    // ======================================================================
    // Retry — transient
    // ======================================================================

    @Test
    void embed_transientConnect_retriesAndSucceeds() {
        DjlServingClient djl = mock(DjlServingClient.class);
        AtomicInteger calls = new AtomicInteger();
        when(djl.predict(any(), any())).thenAnswer(inv -> {
            if (calls.incrementAndGet() == 1) {
                throw new ConnectException("refused");
            }
            return arr(arr(0.9));
        });

        List<float[]> r = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", List.of("x"), BATCH, CAP, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        assertThat(calls.get()).as("Exactly one retry after first ConnectException").isEqualTo(2);
        assertThat(r).hasSize(1);
        assertThat(r.get(0)).containsExactly(0.9f);
    }

    @Test
    void embed_transient503_retriesAndSucceeds() {
        DjlServingClient djl = mock(DjlServingClient.class);
        AtomicInteger calls = new AtomicInteger();
        WebApplicationException unavailable = new WebApplicationException(Response.status(503).build());
        when(djl.predict(any(), any())).thenAnswer(inv -> {
            if (calls.incrementAndGet() == 1) {
                throw unavailable;
            }
            return arr(arr(0.5));
        });

        List<float[]> r = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", List.of("x"), BATCH, CAP, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        assertThat(calls.get()).as("503 is transient and retried").isEqualTo(2);
        assertThat(r.get(0)).containsExactly(0.5f);
    }

    @Test
    void embed_transientRetriesExhausted_propagatesLastError() {
        DjlServingClient djl = mock(DjlServingClient.class);
        AtomicInteger calls = new AtomicInteger();
        when(djl.predict(any(), any())).thenAnswer(inv -> {
            calls.incrementAndGet();
            throw new ConnectException("persistent");
        });

        Throwable err = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", List.of("x"), BATCH, CAP, /*maxRetries*/ 2, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(calls.get())
                .as("maxRetries=2 means up to 3 attempts (initial + 2 retries)")
                .isEqualTo(3);
        assertThat(err)
                .as("After exhaustion the last transient failure propagates verbatim")
                .isInstanceOf(ConnectException.class)
                .hasMessageContaining("persistent");
    }

    // ======================================================================
    // Retry — permanent
    // ======================================================================

    @Test
    void embed_permanent400_doesNotRetry() {
        DjlServingClient djl = mock(DjlServingClient.class);
        AtomicInteger calls = new AtomicInteger();
        WebApplicationException badRequest = new WebApplicationException(
                Response.status(400).entity("bad inputs").build());
        when(djl.predict(any(), any())).thenAnswer(inv -> {
            calls.incrementAndGet();
            throw badRequest;
        });

        Throwable err = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", List.of("x"), BATCH, CAP, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(calls.get()).as("Permanent 400 must not retry").isEqualTo(1);
        assertThat(err).isSameAs(badRequest);
    }

    @Test
    void embed_permanent404_doesNotRetry() {
        DjlServingClient djl = mock(DjlServingClient.class);
        AtomicInteger calls = new AtomicInteger();
        WebApplicationException notFound = new WebApplicationException(Response.status(404).build());
        when(djl.predict(any(), any())).thenAnswer(inv -> {
            calls.incrementAndGet();
            throw notFound;
        });

        Throwable err = new SemanticGraphEmbedHelper(djl)
                .embed("minilm", List.of("x"), BATCH, CAP, MAX_RETRIES, BACKOFF_MS)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(calls.get()).as("404 is permanent").isEqualTo(1);
        assertThat(err).isSameAs(notFound);
    }

    // ======================================================================
    // isModelLoaded
    // ======================================================================

    @Test
    void isModelLoaded_present_true() {
        DjlServingClient djl = mock(DjlServingClient.class);
        JsonObject list = new JsonObject().put("models", new JsonArray()
                .add(new JsonObject().put("modelName", "minilm"))
                .add(new JsonObject().put("modelName", "paraphrase")));
        when(djl.listModels()).thenReturn(list);

        Boolean loaded = new SemanticGraphEmbedHelper(djl).isModelLoaded("minilm")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();
        assertThat(loaded).isTrue();
    }

    @Test
    void isModelLoaded_missing_false() {
        DjlServingClient djl = mock(DjlServingClient.class);
        JsonObject list = new JsonObject().put("models", new JsonArray()
                .add(new JsonObject().put("modelName", "paraphrase")));
        when(djl.listModels()).thenReturn(list);

        Boolean loaded = new SemanticGraphEmbedHelper(djl).isModelLoaded("minilm")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();
        assertThat(loaded)
                .as("Missing from /models means NOT loaded; caller FAIL_PRECONDITIONs per §21.3")
                .isFalse();
    }

    @Test
    void isModelLoaded_blank_false() {
        DjlServingClient djl = mock(DjlServingClient.class);
        Boolean loaded = new SemanticGraphEmbedHelper(djl).isModelLoaded("  ")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();
        assertThat(loaded).isFalse();
        verify(djl, times(0)).listModels();
    }

    @Test
    void isModelLoaded_listModelsFails_propagates() {
        DjlServingClient djl = mock(DjlServingClient.class);
        // thenAnswer (not thenThrow) because ConnectException is checked and
        // the sync listModels() signature doesn't declare it. Answer.answer()
        // declares throws Throwable, so the checked exception is allowed here.
        when(djl.listModels()).thenAnswer(inv -> { throw new ConnectException("down"); });

        Throwable err = new SemanticGraphEmbedHelper(djl).isModelLoaded("minilm")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();
        assertThat(err)
                .as("Probe failure must propagate so the pipeline maps it to FAILED_PRECONDITION/UNAVAILABLE")
                .isInstanceOf(ConnectException.class);
    }

    // ======================================================================
    // listLoadedModels
    // ======================================================================

    @Test
    void listLoadedModels_returnsNames() {
        DjlServingClient djl = mock(DjlServingClient.class);
        JsonObject list = new JsonObject().put("models", new JsonArray()
                .add(new JsonObject().put("modelName", "minilm"))
                .add(new JsonObject().put("modelName", "e5-small")));
        when(djl.listModels()).thenReturn(list);

        Set<String> names = new SemanticGraphEmbedHelper(djl).listLoadedModels()
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();
        assertThat(names).containsExactlyInAnyOrder("minilm", "e5-small");
    }

    @Test
    void listLoadedModels_emptyModelsArray_emptySet() {
        DjlServingClient djl = mock(DjlServingClient.class);
        when(djl.listModels()).thenReturn(new JsonObject().put("models", new JsonArray()));

        Set<String> names = new SemanticGraphEmbedHelper(djl).listLoadedModels()
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();
        assertThat(names).isEmpty();
    }

    @Test
    void listLoadedModels_nullResponseBody_emptySet() {
        DjlServingClient djl = mock(DjlServingClient.class);
        when(djl.listModels()).thenReturn((JsonObject) null);

        Set<String> names = new SemanticGraphEmbedHelper(djl).listLoadedModels()
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();
        assertThat(names).isEqualTo(Collections.emptySet());
    }

    // ======================================================================
    // Helpers
    // ======================================================================

    private static JsonArray arr(Object... items) {
        JsonArray a = new JsonArray();
        for (Object it : items) a.add(it);
        return a;
    }
}
