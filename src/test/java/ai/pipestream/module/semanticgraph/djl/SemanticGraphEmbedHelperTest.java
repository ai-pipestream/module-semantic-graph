package ai.pipestream.module.semanticgraph.djl;

import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.helpers.test.UniAssertSubscriber;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.Test;

import java.net.ConnectException;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class SemanticGraphEmbedHelperTest {

    // --- happy path --------------------------------------------------------

    @Test
    void embed_singleCallReturnsAlignedFloatArrays() {
        DjlServingClient djl = mock(DjlServingClient.class);
        JsonArray response = new JsonArray()
                .add(new JsonArray().add(0.1).add(0.2).add(0.3))
                .add(new JsonArray().add(0.4).add(0.5).add(0.6));
        when(djl.predict(eq("minilm"), any(JsonObject.class)))
                .thenReturn(Uni.createFrom().item(response));

        SemanticGraphEmbedHelper helper = new SemanticGraphEmbedHelper(djl);

        List<float[]> result = helper.embed("minilm", List.of("foo", "bar"))
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        assertThat(result)
                .as("Two-text input produces two output vectors in input order")
                .hasSize(2);
        assertThat(result.get(0))
                .as("First vector matches first input row")
                .containsExactly(0.1f, 0.2f, 0.3f);
        assertThat(result.get(1))
                .as("Second vector matches second input row")
                .containsExactly(0.4f, 0.5f, 0.6f);
        verify(djl, times(1)).predict(eq("minilm"), any(JsonObject.class));
    }

    @Test
    void embed_emptyInput_shortCircuitsWithoutTouchingDjl() {
        DjlServingClient djl = mock(DjlServingClient.class);
        SemanticGraphEmbedHelper helper = new SemanticGraphEmbedHelper(djl);

        List<float[]> result = helper.embed("minilm", List.of())
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        assertThat(result)
                .as("Empty input must not roundtrip to DJL — fast-path returns empty list")
                .isEmpty();
        verify(djl, times(0)).predict(any(), any());
    }

    @Test
    void embed_nullInput_alsoShortCircuits() {
        DjlServingClient djl = mock(DjlServingClient.class);
        SemanticGraphEmbedHelper helper = new SemanticGraphEmbedHelper(djl);

        List<float[]> result = helper.embed("minilm", null)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        assertThat(result).as("Null input behaves like empty input").isEmpty();
        verify(djl, times(0)).predict(any(), any());
    }

    @Test
    void embed_blankModelId_failsWithoutCallingDjl() {
        DjlServingClient djl = mock(DjlServingClient.class);
        SemanticGraphEmbedHelper helper = new SemanticGraphEmbedHelper(djl);

        Throwable err = helper.embed("   ", List.of("foo"))
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(err)
                .as("Blank model id must fail before the wire — caller's bug, not DJL's")
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("modelId is required");
        verify(djl, times(0)).predict(any(), any());
    }

    // --- alignment / response-shape errors ---------------------------------

    @Test
    void embed_responseRowCountMismatch_failsLoud() {
        DjlServingClient djl = mock(DjlServingClient.class);
        // Sent 3 inputs but DJL returned 2 rows — alignment is broken
        JsonArray bad = new JsonArray()
                .add(new JsonArray().add(0.1))
                .add(new JsonArray().add(0.2));
        when(djl.predict(any(), any())).thenReturn(Uni.createFrom().item(bad));

        SemanticGraphEmbedHelper helper = new SemanticGraphEmbedHelper(djl);

        Throwable err = helper.embed("minilm", List.of("a", "b", "c"))
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(err)
                .as("Row-count mismatch is a contract violation; we never guess at alignment")
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("returned 2 vectors")
                .hasMessageContaining("3 were requested");
    }

    @Test
    void embed_responseHasNullRow_failsLoud() {
        DjlServingClient djl = mock(DjlServingClient.class);
        JsonArray bad = new JsonArray().add(new JsonArray().add(0.1)).add((Object) null);
        when(djl.predict(any(), any())).thenReturn(Uni.createFrom().item(bad));

        SemanticGraphEmbedHelper helper = new SemanticGraphEmbedHelper(djl);

        Throwable err = helper.embed("minilm", List.of("a", "b"))
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(err)
                .as("Null vector slot is a malformed response — fail loud, do not substitute zero")
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("position 1");
    }

    // --- transient retry --------------------------------------------------

    @Test
    void embed_transientFailureRetriesOnceAndSucceeds() {
        DjlServingClient djl = mock(DjlServingClient.class);
        AtomicInteger calls = new AtomicInteger();
        JsonArray ok = new JsonArray().add(new JsonArray().add(0.1).add(0.2));
        when(djl.predict(any(), any())).thenAnswer(inv -> {
            int n = calls.incrementAndGet();
            if (n == 1) {
                return Uni.createFrom().<JsonArray>failure(new ConnectException("boom"));
            }
            return Uni.createFrom().item(ok);
        });

        SemanticGraphEmbedHelper helper = new SemanticGraphEmbedHelper(djl);

        List<float[]> result = helper.embed("minilm", List.of("foo"))
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        assertThat(calls.get())
                .as("First call failed transient; helper retried exactly once")
                .isEqualTo(2);
        assertThat(result).as("Retry produced a vector").hasSize(1);
        assertThat(result.get(0)).containsExactly(0.1f, 0.2f);
    }

    @Test
    void embed_transientFailureExhaustsRetries_propagates() {
        DjlServingClient djl = mock(DjlServingClient.class);
        AtomicInteger calls = new AtomicInteger();
        when(djl.predict(any(), any())).thenAnswer(inv -> {
            calls.incrementAndGet();
            return Uni.createFrom().<JsonArray>failure(new ConnectException("permanent transport down"));
        });

        SemanticGraphEmbedHelper helper = new SemanticGraphEmbedHelper(djl);

        Throwable err = helper.embed("minilm", List.of("foo"))
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(calls.get())
                .as("With RETRY_ATTEMPTS=1 the helper makes exactly 2 attempts before giving up")
                .isEqualTo(1 + SemanticGraphEmbedHelper.RETRY_ATTEMPTS);
        assertThat(err)
                .as("After retries the original transport failure propagates — no swallowing")
                .isInstanceOf(ConnectException.class)
                .hasMessageContaining("permanent transport down");
    }

    // --- permanent failure: no retry --------------------------------------

    @Test
    void embed_permanent4xxFailure_doesNotRetry() {
        DjlServingClient djl = mock(DjlServingClient.class);
        AtomicInteger calls = new AtomicInteger();
        WebApplicationException badRequest = new WebApplicationException(
                Response.status(400).entity("bad inputs").build());
        when(djl.predict(any(), any())).thenAnswer(inv -> {
            calls.incrementAndGet();
            return Uni.createFrom().<JsonArray>failure(badRequest);
        });

        SemanticGraphEmbedHelper helper = new SemanticGraphEmbedHelper(djl);

        Throwable err = helper.embed("minilm", List.of("foo"))
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(calls.get())
                .as("HTTP 400 is a permanent contract error; helper does NOT retry")
                .isEqualTo(1);
        assertThat(err)
                .as("Permanent error propagates verbatim")
                .isSameAs(badRequest);
    }

    @Test
    void embed_503ServiceUnavailable_isTransientAndRetries() {
        DjlServingClient djl = mock(DjlServingClient.class);
        AtomicInteger calls = new AtomicInteger();
        WebApplicationException unavailable = new WebApplicationException(
                Response.status(503).build());
        JsonArray ok = new JsonArray().add(new JsonArray().add(0.42));
        when(djl.predict(any(), any())).thenAnswer(inv -> {
            int n = calls.incrementAndGet();
            return n == 1
                    ? Uni.createFrom().<JsonArray>failure(unavailable)
                    : Uni.createFrom().item(ok);
        });

        SemanticGraphEmbedHelper helper = new SemanticGraphEmbedHelper(djl);

        List<float[]> result = helper.embed("minilm", List.of("foo"))
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        assertThat(calls.get())
                .as("503 should be classified transient and retried once")
                .isEqualTo(2);
        assertThat(result.get(0)).containsExactly(0.42f);
    }

    // --- isModelLoaded ----------------------------------------------------

    @Test
    void isModelLoaded_modelPresent_returnsTrue() {
        DjlServingClient djl = mock(DjlServingClient.class);
        JsonObject listResp = new JsonObject().put("models", new JsonArray()
                .add(new JsonObject().put("modelName", "minilm"))
                .add(new JsonObject().put("modelName", "paraphrase")));
        when(djl.listModels()).thenReturn(Uni.createFrom().item(listResp));

        SemanticGraphEmbedHelper helper = new SemanticGraphEmbedHelper(djl);

        Boolean loaded = helper.isModelLoaded("minilm")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        assertThat(loaded).as("listModels listed the requested model").isTrue();
    }

    @Test
    void isModelLoaded_modelMissing_returnsFalse() {
        DjlServingClient djl = mock(DjlServingClient.class);
        JsonObject listResp = new JsonObject().put("models", new JsonArray()
                .add(new JsonObject().put("modelName", "paraphrase")));
        when(djl.listModels()).thenReturn(Uni.createFrom().item(listResp));

        SemanticGraphEmbedHelper helper = new SemanticGraphEmbedHelper(djl);

        Boolean loaded = helper.isModelLoaded("minilm")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        assertThat(loaded)
                .as("Missing-from-listModels means not loaded; caller will FAIL_PRECONDITION per §21.3")
                .isFalse();
    }

    @Test
    void isModelLoaded_blankModelId_returnsFalse() {
        DjlServingClient djl = mock(DjlServingClient.class);
        SemanticGraphEmbedHelper helper = new SemanticGraphEmbedHelper(djl);

        Boolean loaded = helper.isModelLoaded(" ")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        assertThat(loaded).as("Blank id is not a model id").isFalse();
        verify(djl, times(0)).listModels();
    }

    @Test
    void isModelLoaded_listModelsFails_propagates() {
        DjlServingClient djl = mock(DjlServingClient.class);
        when(djl.listModels()).thenReturn(Uni.createFrom().failure(new ConnectException("nope")));

        SemanticGraphEmbedHelper helper = new SemanticGraphEmbedHelper(djl);

        Throwable err = helper.isModelLoaded("minilm")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(err)
                .as("Probe failure must propagate so the caller can map to FAILED_PRECONDITION / INTERNAL")
                .isInstanceOf(ConnectException.class);
    }

    // --- isTransient classification -----------------------------------------

    @Test
    void isTransient_classification() {
        assertThat(SemanticGraphEmbedHelper.isTransient(new ConnectException("x")))
                .as("ConnectException is transient").isTrue();
        assertThat(SemanticGraphEmbedHelper.isTransient(new java.net.SocketTimeoutException("x")))
                .as("SocketTimeoutException is transient").isTrue();
        assertThat(SemanticGraphEmbedHelper.isTransient(new java.util.concurrent.TimeoutException("x")))
                .as("Mutiny TimeoutException is transient").isTrue();
        assertThat(SemanticGraphEmbedHelper.isTransient(new RuntimeException(new ConnectException("x"))))
                .as("Wrapped ConnectException is still transient").isTrue();
        assertThat(SemanticGraphEmbedHelper.isTransient(
                new WebApplicationException(Response.status(503).build())))
                .as("HTTP 503 is transient").isTrue();
        assertThat(SemanticGraphEmbedHelper.isTransient(
                new WebApplicationException(Response.status(429).build())))
                .as("HTTP 429 (Too Many Requests) is transient").isTrue();
        assertThat(SemanticGraphEmbedHelper.isTransient(
                new WebApplicationException(Response.status(400).build())))
                .as("HTTP 400 is permanent").isFalse();
        assertThat(SemanticGraphEmbedHelper.isTransient(
                new WebApplicationException(Response.status(404).build())))
                .as("HTTP 404 (model not found) is permanent").isFalse();
        assertThat(SemanticGraphEmbedHelper.isTransient(new IllegalStateException("misc")))
                .as("Generic IllegalStateException is not transient").isFalse();
    }
}
