package ai.pipestream.module.semanticgraph.djl;

import io.smallrye.mutiny.Uni;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.WebApplicationException;
import org.eclipse.microprofile.rest.client.inject.RestClient;
import org.jboss.logging.Logger;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * Thin Mutiny wrapper around {@link DjlServingClient#predict} for the
 * semantic-graph boundary re-embed (DESIGN.md §7.3 step 5; PLAN.md R3 step 3).
 *
 * <p>Stage-3 boundary detection produces at most
 * {@code SemanticGraphStepOptions.effectiveMaxSemanticChunksPerDoc()} grouped
 * sentence spans per doc (default 50, hard cap). Those grouped texts must be
 * re-embedded — they are not centroids of the original sentence vectors,
 * because boundary groups concatenate text first and then run the model over
 * the concatenated string. This helper sends the entire batch in one
 * {@code predict} call so the round-trip cost is independent of group count.
 *
 * <p>On {@link io.grpc.Status.Code#UNAVAILABLE UNAVAILABLE} /
 * {@code DEADLINE_EXCEEDED} / {@code RESOURCE_EXHAUSTED}-class transient
 * failures (HTTP 502, 503, 504, or any {@link java.net.ConnectException}-rooted
 * cause) the call is retried <b>once</b> with a small backoff. Permanent
 * failures (HTTP 400/404/422 — model unknown, malformed payload) propagate
 * immediately. Per DESIGN.md §10.1 the caller maps the propagated failure to
 * the right gRPC status; we never swallow.
 *
 * <p>Empty input ({@code texts.isEmpty()}) short-circuits to an empty result
 * without touching the wire — this is the only legal "fast path" and matches
 * the behavior of {@code DjlServingBackend.embed}.
 *
 * <p>Note on swap: when {@code quarkus-djl-embeddings} publishes as a
 * standalone artifact, swap the {@link DjlServingClient} import to
 * {@code ai.pipestream.quarkus.djl.serving.runtime.client.DjlServingClient}.
 * The {@code @RestClient} injection point and the
 * {@code quarkus.rest-client.djl-serving.url} property are unchanged.
 */
@ApplicationScoped
public class SemanticGraphEmbedHelper {

    private static final Logger log = Logger.getLogger(SemanticGraphEmbedHelper.class);

    /** One retry on transient errors per DESIGN.md §10.2 (semantic-graph is best-effort, not retry-heavy). */
    static final int RETRY_ATTEMPTS = 1;

    /** Base backoff between the first attempt and the retry. */
    static final Duration RETRY_BACKOFF = Duration.ofMillis(150);

    private final DjlServingClient djl;

    @Inject
    public SemanticGraphEmbedHelper(@RestClient DjlServingClient djl) {
        this.djl = djl;
    }

    /**
     * Embeds {@code texts} via DJL Serving's {@code /predictions/{modelId}}
     * endpoint and returns the vectors in input order.
     *
     * @param modelId DJL serving name (already validated as loaded by the
     *                caller via the listModels probe)
     * @param texts   the boundary-group texts to embed (≤
     *                {@code effectiveMaxSemanticChunksPerDoc()})
     * @return a Uni that succeeds with one {@code float[]} per input text in
     *         the same order, or fails with the underlying transport / DJL
     *         exception (caller maps to gRPC status)
     */
    public Uni<List<float[]>> embed(String modelId, List<String> texts) {
        if (modelId == null || modelId.isBlank()) {
            return Uni.createFrom().failure(new IllegalArgumentException(
                    "modelId is required; SemanticGraphStepOptions.requireBoundaryEmbeddingModelId " +
                    "should have caught this upstream"));
        }
        if (texts == null || texts.isEmpty()) {
            return Uni.createFrom().item(List.of());
        }
        JsonObject body = new JsonObject().put("inputs", new JsonArray(texts));
        // deferred(...) re-creates the underlying Uni on every subscription so
        // .retry() actually re-invokes djl.predict(...) instead of just
        // re-emitting the cached failure from a single shared upstream.
        return Uni.createFrom().deferred(() -> djl.predict(modelId, body))
                .onFailure(SemanticGraphEmbedHelper::isTransient)
                .retry().withBackOff(RETRY_BACKOFF).atMost(RETRY_ATTEMPTS)
                .map(response -> parseBatchResponse(response, texts.size(), modelId));
    }

    /**
     * Verifies that {@code modelId} is currently loaded by DJL Serving.
     * Returns a Uni that succeeds with {@code true} when the model is loaded,
     * {@code false} when the {@code /models} endpoint returns successfully but
     * does not list it, and fails when the {@code /models} call itself fails.
     *
     * <p>The caller invokes this once per processData call (only when boundary
     * detection is enabled) and maps a {@code false} result to
     * {@code FAILED_PRECONDITION} per DESIGN.md §21.3.
     */
    public Uni<Boolean> isModelLoaded(String modelId) {
        if (modelId == null || modelId.isBlank()) {
            return Uni.createFrom().item(Boolean.FALSE);
        }
        return djl.listModels().map(json -> containsModel(json, modelId));
    }

    private static List<float[]> parseBatchResponse(JsonArray response, int expected, String modelId) {
        if (response == null) {
            throw new IllegalStateException(
                    "DJL Serving returned a null response body for model '" + modelId + "'");
        }
        if (response.size() != expected) {
            throw new IllegalStateException(String.format(
                    "DJL Serving returned %d vectors for model '%s' but %d were requested; " +
                    "this indicates a serialization or batching bug — bailing rather than guessing alignment",
                    response.size(), modelId, expected));
        }
        List<float[]> results = new ArrayList<>(response.size());
        for (int i = 0; i < response.size(); i++) {
            JsonArray vec = response.getJsonArray(i);
            if (vec == null) {
                throw new IllegalStateException(String.format(
                        "DJL Serving response position %d for model '%s' was not a JSON array",
                        i, modelId));
            }
            float[] arr = new float[vec.size()];
            for (int j = 0; j < vec.size(); j++) {
                Number n = vec.getNumber(j);
                if (n == null) {
                    throw new IllegalStateException(String.format(
                            "DJL Serving response[%d][%d] was null for model '%s'", i, j, modelId));
                }
                arr[j] = n.floatValue();
            }
            results.add(arr);
        }
        return results;
    }

    private static boolean containsModel(JsonObject listResponse, String modelId) {
        if (listResponse == null) {
            return false;
        }
        // DJL Serving /models response shape: { "models": [ { "modelName": "...", ... }, ... ] }
        JsonArray models = listResponse.getJsonArray("models");
        if (models == null) {
            log.warnf("DJL /models response did not contain 'models' array — payload keys: %s",
                    listResponse.fieldNames());
            return false;
        }
        for (int i = 0; i < models.size(); i++) {
            JsonObject m = models.getJsonObject(i);
            if (m == null) continue;
            String name = m.getString("modelName");
            if (modelId.equals(name)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Visible for {@code .onFailure(predicate)} chaining and for unit-test
     * assertions on classification. A failure is transient when its root
     * cause looks like a connection / 5xx / timeout — anything where
     * "try again in a moment" is the right move.
     */
    static boolean isTransient(Throwable t) {
        Throwable cur = t;
        while (cur != null) {
            if (cur instanceof java.net.ConnectException
                    || cur instanceof java.net.SocketTimeoutException
                    || cur instanceof java.util.concurrent.TimeoutException
                    || cur instanceof io.vertx.core.VertxException
                    || cur instanceof io.netty.channel.ConnectTimeoutException) {
                return true;
            }
            if (cur instanceof WebApplicationException wae) {
                int s = wae.getResponse() != null ? wae.getResponse().getStatus() : -1;
                if (s == 502 || s == 503 || s == 504 || s == 408 || s == 429) {
                    return true;
                }
                // 4xx-other and 500 are treated as permanent — DJL returns 5xx for
                // overload but 500 alone is too ambiguous; DJL Serving sends 503
                // for "model not ready" / "no workers available".
                return false;
            }
            cur = cur.getCause();
        }
        return false;
    }
}
