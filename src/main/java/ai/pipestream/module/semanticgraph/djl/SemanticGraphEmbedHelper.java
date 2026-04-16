package ai.pipestream.module.semanticgraph.djl;

import ai.pipestream.module.semanticgraph.retry.SemanticGraphRetryPolicy;
import io.quarkus.cache.CacheResult;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.rest.client.inject.RestClient;
import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Batched, retrying Mutiny wrapper around {@link DjlServingClient#predict}
 * for the semantic-graph boundary re-embed (DESIGN.md §7.3 step 5).
 *
 * <h2>Batching — not optional</h2>
 *
 * <p>Slices the input text list into sub-batches of at most
 * {@code effectiveBatchSize} and fires them concurrently up to
 * {@code effectivePerDocCap}. This is the same {@code Multi.range →
 * transformToUni → merge(cap)} pattern used by
 * {@code module-embedder}'s {@code EmbedderPipelineService}, for the same
 * reason: DJL Serving rejects oversized inference batches (default max
 * batch ~32 per model), and a monolithic predict call for a 50-chunk
 * boundary pass either errors at DJL's ingress or crashes its worker.
 *
 * <p>For the typical boundary workload (≤50 groups per doc), this is
 * 1–2 sub-batches that complete in parallel. The concurrency cap clamps
 * to {@code [1, batchCount]} so small inputs don't try to spawn cap
 * parallel requests for nothing.
 *
 * <h2>§22.5-style final verification</h2>
 *
 * <p>After all sub-batches merge, every slot of the shared vectors array
 * must be populated. A null slot — impossible unless a sub-batch silently
 * dropped an index — raises {@link IllegalStateException} so the caller
 * maps it to {@code FAILED_PRECONDITION} / DLQ rather than emit a chunk
 * with no vector. This mirrors the §22.5 regression gate in the embedder
 * and enforces the same "never an empty vector" contract here.
 *
 * <h2>Transient vs permanent</h2>
 *
 * <p>Per-sub-batch retry via {@link SemanticGraphRetryPolicy} with
 * {@link ai.pipestream.module.semanticgraph.retry.SemanticGraphRetryClassifier}.
 * Transient (5xx, connect, timeout) retries up to the configured budget;
 * permanent (4xx, alignment mismatch) propagates immediately.
 *
 * <h2>Model-loaded probe</h2>
 *
 * <p>{@link #isModelLoaded(String)} calls {@code /models} and checks whether
 * the target model name appears. Cached under Quarkus
 * {@code @CacheResult("djl-models-loaded")} with a 30s TTL so repeated
 * pipeline calls don't hammer DJL Serving; cadence matches the scheduled refresh
 * in {@code pipestream-embedder-djl}'s {@code DjlModelRegistry}.
 *
 * <h2>Swap plan</h2>
 *
 * <p>When {@code ai.pipestream.module:pipestream-embedder-djl-runtime}
 * publishes to Maven Central, this helper can delegate to that extension's
 * {@code DjlServingBackend.embed(...)} instead of hand-rolling the batched
 * REST loop. Until then, this is the local implementation.
 */
@ApplicationScoped
public class SemanticGraphEmbedHelper {

    private static final Logger log = Logger.getLogger(SemanticGraphEmbedHelper.class);

    private final DjlServingClient djl;

    @Inject
    public SemanticGraphEmbedHelper(@RestClient DjlServingClient djl) {
        this.djl = djl;
    }

    /**
     * Embeds {@code texts} via DJL Serving and returns the vectors in input
     * order. Sliced into sub-batches of {@code batchSize} with up to
     * {@code perDocCap} concurrent in-flight requests, each retried up to
     * {@code maxRetries} times on transient failure.
     *
     * <p>Empty / null {@code texts} returns an empty list without touching
     * the wire. Blank {@code modelId} fails synchronously —
     * {@link ai.pipestream.module.semanticgraph.config.SemanticGraphStepOptions#requireBoundaryEmbeddingModelId}
     * should have caught this upstream.
     *
     * @throws IllegalArgumentException if modelId is null/blank
     */
    public Uni<List<float[]>> embed(
            String modelId,
            List<String> texts,
            int batchSize,
            int perDocCap,
            int maxRetries,
            long retryBackoffMs) {

        if (modelId == null || modelId.isBlank()) {
            return Uni.createFrom().failure(new IllegalArgumentException(
                    "modelId is required for SemanticGraphEmbedHelper.embed; " +
                    "SemanticGraphStepOptions.requireBoundaryEmbeddingModelId should have caught this upstream"));
        }
        if (texts == null || texts.isEmpty()) {
            return Uni.createFrom().item(List.of());
        }
        if (batchSize < 1) {
            return Uni.createFrom().failure(new IllegalArgumentException(
                    "batchSize must be >= 1, got " + batchSize));
        }
        if (perDocCap < 1) {
            return Uni.createFrom().failure(new IllegalArgumentException(
                    "perDocCap must be >= 1, got " + perDocCap));
        }

        final int total = texts.size();
        final int batchCount = (total + batchSize - 1) / batchSize;
        final int effectiveCap = Math.min(perDocCap, batchCount);
        final float[][] out = new float[total][];

        return Multi.createFrom().range(0, batchCount)
                .onItem().transformToUni(batchIdx -> {
                    int from = batchIdx * batchSize;
                    int to = Math.min(from + batchSize, total);
                    // Defensive copy — subList shares backing storage; safe today but brittle.
                    final List<String> slice = new ArrayList<>(texts.subList(from, to));
                    final int sliceStart = from;
                    final String label = modelId + "[batch " + batchIdx + "/" + batchCount + "]";

                    return SemanticGraphRetryPolicy.withRetry(
                            () -> djl.predict(modelId, new JsonObject().put("inputs", new JsonArray(slice))),
                            maxRetries, retryBackoffMs, label)
                            .map(response -> {
                                List<float[]> parsed = parseBatch(response);
                                if (parsed.size() != slice.size()) {
                                    throw new IllegalStateException(String.format(
                                            "DJL returned %d vectors for %d inputs on model '%s' batch %d/%d; " +
                                            "alignment violation — refusing to guess at index mapping",
                                            parsed.size(), slice.size(), modelId, batchIdx, batchCount));
                                }
                                for (int k = 0; k < parsed.size(); k++) {
                                    float[] vec = parsed.get(k);
                                    if (vec == null || vec.length == 0) {
                                        throw new IllegalStateException(String.format(
                                                "DJL returned null/empty vector at batch %d slot %d (global index %d) " +
                                                "for model '%s'", batchIdx, k, sliceStart + k, modelId));
                                    }
                                    out[sliceStart + k] = vec;
                                }
                                return (Void) null;
                            });
                })
                .merge(effectiveCap)
                .collect().asList()
                .map(ignored -> finalVerify(out, modelId));
    }

    /**
     * Overload that uses conservative hardcoded defaults for small inputs.
     * Callers with per-step tunables (SemanticGraphStepOptions) should use
     * the full-signature overload.
     */
    public Uni<List<float[]>> embed(String modelId, List<String> texts) {
        return embed(modelId, texts, /*batchSize*/ 32, /*perDocCap*/ 5,
                /*maxRetries*/ 2, /*retryBackoffMs*/ 150L);
    }

    /**
     * Returns {@code true} when {@code modelId} is currently registered in
     * DJL Serving's {@code /models} listing. Result cached 30s under
     * {@code @CacheResult("djl-models-loaded")} to avoid per-doc probing.
     *
     * <p>Returns {@code false} for blank/null ids without calling the wire.
     * Wire failures propagate — the caller maps them.
     */
    @CacheResult(cacheName = "djl-models-loaded")
    public Uni<Boolean> isModelLoaded(String modelId) {
        if (modelId == null || modelId.isBlank()) {
            return Uni.createFrom().item(Boolean.FALSE);
        }
        return djl.listModels().map(json -> containsModel(json, modelId));
    }

    /**
     * Returns the set of currently-loaded model names from DJL Serving.
     * Not cached — used by health/diagnostic paths, not the hot path.
     */
    public Uni<Set<String>> listLoadedModels() {
        return djl.listModels().map(json -> {
            if (json == null) return Collections.<String>emptySet();
            JsonArray models = json.getJsonArray("models");
            if (models == null || models.isEmpty()) return Collections.<String>emptySet();
            java.util.Set<String> names = new java.util.LinkedHashSet<>(models.size() * 2);
            for (int i = 0; i < models.size(); i++) {
                JsonObject m = models.getJsonObject(i);
                if (m == null) continue;
                String name = m.getString("modelName");
                if (name != null && !name.isEmpty()) names.add(name);
            }
            return (Set<String>) names;
        });
    }

    // -------------------- helpers --------------------

    private static List<float[]> finalVerify(float[][] out, String modelId) {
        for (int i = 0; i < out.length; i++) {
            if (out[i] == null || out[i].length == 0) {
                throw new IllegalStateException(String.format(
                        "SemanticGraphEmbedHelper.embed left slot %d null/empty for model '%s' — " +
                        "sub-batch slicing bug (§22.5 regression gate)", i, modelId));
            }
        }
        return Arrays.asList(out);
    }

    /**
     * Parses a DJL Serving batch response. DJL's handler for
     * sentence-transformers returns either nested arrays
     * ({@code [[f...], [f...]]}) for multi-input batches or a flat array
     * ({@code [f, f, ...]}) for a single input. This helper normalises
     * both to {@code List<float[]>}.
     */
    private static List<float[]> parseBatch(JsonArray response) {
        if (response == null) {
            throw new IllegalStateException("DJL Serving returned a null response body");
        }
        if (response.isEmpty()) {
            return List.of();
        }
        Object first = response.getValue(0);
        if (first instanceof JsonArray) {
            List<float[]> out = new ArrayList<>(response.size());
            for (int i = 0; i < response.size(); i++) {
                JsonArray vec = response.getJsonArray(i);
                if (vec == null) {
                    throw new IllegalStateException(
                            "DJL Serving response position " + i + " was not a JSON array");
                }
                float[] arr = new float[vec.size()];
                for (int j = 0; j < vec.size(); j++) {
                    Number n = vec.getNumber(j);
                    if (n == null) {
                        throw new IllegalStateException(
                                "DJL Serving response[" + i + "][" + j + "] was null");
                    }
                    arr[j] = n.floatValue();
                }
                out.add(arr);
            }
            return out;
        }
        // Flat single-vector response
        float[] arr = new float[response.size()];
        for (int i = 0; i < response.size(); i++) {
            Number n = response.getNumber(i);
            if (n == null) {
                throw new IllegalStateException("DJL Serving flat response[" + i + "] was null");
            }
            arr[i] = n.floatValue();
        }
        return List.of(arr);
    }

    private static boolean containsModel(JsonObject listResponse, String modelId) {
        if (listResponse == null) return false;
        JsonArray models = listResponse.getJsonArray("models");
        if (models == null) {
            log.warnf("DJL /models response missing 'models' array (keys: %s)",
                    listResponse.fieldNames());
            return false;
        }
        for (int i = 0; i < models.size(); i++) {
            JsonObject m = models.getJsonObject(i);
            if (m == null) continue;
            if (modelId.equals(m.getString("modelName"))) return true;
        }
        return false;
    }
}
