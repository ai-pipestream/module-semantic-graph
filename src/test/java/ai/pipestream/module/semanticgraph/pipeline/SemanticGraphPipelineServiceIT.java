package ai.pipestream.module.semanticgraph.pipeline;

import ai.pipestream.data.v1.ChunkEmbedding;
import ai.pipestream.data.v1.GranularityLevel;
import ai.pipestream.data.v1.NamedChunkerConfig;
import ai.pipestream.data.v1.NamedEmbedderConfig;
import ai.pipestream.data.v1.NlpDocumentAnalysis;
import ai.pipestream.data.v1.PipeDoc;
import ai.pipestream.data.v1.SearchMetadata;
import ai.pipestream.data.v1.SemanticChunk;
import ai.pipestream.data.v1.SemanticProcessingResult;
import ai.pipestream.data.v1.SourceFieldAnalytics;
import ai.pipestream.data.v1.VectorDirective;
import ai.pipestream.data.v1.VectorSetDirectives;
import ai.pipestream.module.semanticgraph.config.SemanticGraphStepOptions;
import ai.pipestream.module.semanticgraph.djl.DjlServingClient;
import ai.pipestream.module.semanticgraph.djl.SemanticGraphEmbedHelper;
import ai.pipestream.module.semanticgraph.invariants.SemanticPipelineInvariants;
import com.google.protobuf.Struct;
import com.google.protobuf.Value;
import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.junit.TestProfile;
import io.smallrye.mutiny.Uni;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import jakarta.inject.Inject;
import org.eclipse.microprofile.rest.client.inject.RestClient;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * End-to-end integration test for {@link SemanticGraphPipelineService}
 * against a real DJL Serving instance with an actual MiniLM model.
 *
 * <h2>Two ways to run</h2>
 *
 * <ol>
 *   <li><b>Testcontainer</b> (default): spins up
 *       {@code deepjavalibrary/djl-serving:0.36.0-cpu}, waits for ping,
 *       POSTs to {@code /models} to load {@code all-MiniLM-L6-v2}
 *       synchronously, then runs the tests. Requires docker on the host.</li>
 *   <li><b>External</b>: pass {@code -Ddjl.host=localhost -Ddjl.port=18080}
 *       (or similar) on the Gradle command line to bypass the container and
 *       hit an already-running DJL Serving instance. Useful when CI can't
 *       run docker-in-docker, or when the developer already has DJL up.</li>
 * </ol>
 *
 * <p>If neither docker nor an external endpoint is reachable,
 * {@link Assumptions#assumeTrue(boolean, String)} skips the tests with a
 * clear reason — never a silent pass.
 *
 * <h2>What this test proves end-to-end</h2>
 * <ul>
 *   <li>{@link SemanticGraphEmbedHelper} correctly packs batched predict
 *       calls and parses real 384-d MiniLM vectors back into {@code float[]}.</li>
 *   <li>{@link SemanticGraphPipelineService} consumes a Stage-2 {@link PipeDoc}
 *       whose vectors came from the live model, computes centroids on those
 *       vectors, re-embeds boundary groups via the live model, assembles
 *       a Stage-3 doc that satisfies
 *       {@link SemanticPipelineInvariants#assertPostSemanticGraph}.</li>
 *   <li>Stage-2 SPRs are preserved byte-for-byte through R3.</li>
 *   <li>p95 per-doc wall clock measurement against DESIGN.md §13
 *       semantic-graph gate (target ≤ 500 ms). Not enforced as an
 *       assertion — reported as a number the session can track.</li>
 * </ul>
 */
@QuarkusTest
@TestProfile(DjlExternalProfile.class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("SemanticGraphPipelineService IT (real DJL + MiniLM)")
class SemanticGraphPipelineServiceIT {

    private static final Logger log = LoggerFactory.getLogger(SemanticGraphPipelineServiceIT.class);

    private static final String DJL_SERVING_VERSION = "0.36.0";
    /** DJL model name. Defaults to {@code minilm} to match the
     *  embedder-pipedocs-court fixture's {@code embedding_config_id} naming
     *  convention. Override with {@code -Dr3.fixtures.boundary-model=...}
     *  if your DJL has it registered under a different name. */
    private static final String MODEL_NAME = System.getProperty(
            "r3.fixtures.boundary-model", "minilm");
    private static final String HF_MODEL_URL =
            "djl://ai.djl.huggingface.pytorch/sentence-transformers/all-MiniLM-L6-v2";
    private static final int EXPECTED_DIMS = 384;
    private static final Duration TIMEOUT = Duration.ofMinutes(2);

    /**
     * Default DJL endpoint. Port 18090 is the convention for module-semantic-graph's
     * test/dev DJL Serving instance — picked to avoid the 18080 collision with
     * SeaweedFS in the platform's docker-compose stack. Override via
     * {@code -Ddjl.host=... -Ddjl.port=...} on the Gradle command line.
     */
    private static final String EXTERNAL_HOST = System.getProperty("djl.host", "localhost");
    private static final int EXTERNAL_PORT = Integer.parseInt(System.getProperty("djl.port", "18090"));

    @Inject
    @RestClient
    DjlServingClient client;

    @Inject
    SemanticGraphEmbedHelper helper;

    @Inject
    SemanticGraphPipelineService service;

    @BeforeAll
    void setup() {
        // External DJL is the only supported mode for this IT — the testcontainer
        // path was removed in favor of @QuarkusTest + -Ddjl.host/port, so the
        // MicroProfile REST client infrastructure is available during the test
        // and we can @Inject @RestClient directly.
        //
        // When djl.host is unset, default to localhost:8080 (matches the Quarkus
        // profile default). Callers running tests against a docker container on
        // a different port pass -Ddjl.host=localhost -Ddjl.port=XXXX.
        String reason = verifyExternalDjl(EXTERNAL_HOST, EXTERNAL_PORT);
        Assumptions.assumeTrue(reason == null, reason);
        log.info("IT ready against DJL at {}:{}", EXTERNAL_HOST, EXTERNAL_PORT);
    }

    private static String verifyExternalDjl(String host, int httpPort) {
        try {
            HttpResponse<String> resp = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(2)).build()
                    .send(HttpRequest.newBuilder()
                                    .uri(URI.create("http://" + host + ":" + httpPort + "/ping"))
                                    .timeout(Duration.ofSeconds(3)).GET().build(),
                            HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() >= 500) {
                return "External DJL at " + host + ":" + httpPort + " returned HTTP "
                        + resp.statusCode() + " on /ping";
            }
        } catch (Exception e) {
            return "External DJL at " + host + ":" + httpPort + " unreachable (" + e.getClass().getSimpleName()
                    + ": " + e.getMessage() + "). Run with -Ddjl.host=<host> -Ddjl.port=<port> "
                    + "pointing at a running DJL Serving instance.";
        }
        // Verify the target model is loaded.
        try {
            HttpResponse<String> resp = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(2)).build()
                    .send(HttpRequest.newBuilder()
                                    .uri(URI.create("http://" + host + ":" + httpPort + "/models"))
                                    .timeout(Duration.ofSeconds(5)).GET().build(),
                            HttpResponse.BodyHandlers.ofString());
            if (!resp.body().contains("\"" + MODEL_NAME + "\"")) {
                return "DJL at " + host + ":" + httpPort + " reachable but model '" + MODEL_NAME
                        + "' is not loaded. POST to /models first: " +
                        "curl -X POST 'http://" + host + ":" + httpPort
                        + "/models?url=" + java.net.URLEncoder.encode(HF_MODEL_URL,
                                java.nio.charset.StandardCharsets.UTF_8)
                        + "&model_name=" + MODEL_NAME + "&engine=PyTorch&synchronous=true'";
            }
        } catch (Exception e) {
            return "Failed to query /models on " + host + ":" + httpPort + ": " + e.getMessage();
        }
        return null; // null = ready
    }

    @AfterAll
    void teardown() {
        // No-op: container lifecycle is external to this IT.
    }

    // ======================================================================
    // Happy path — full Stage-2 → Stage-3 with real MiniLM
    // ======================================================================

    @Test
    @DisplayName("full pipeline: Stage-2 with real MiniLM vectors → Stage-3 with centroids + boundaries")
    void fullPipeline_docCentroidAndBoundaries() {
        List<String> sentences = List.of(
                "The quick brown fox jumps over the lazy dog.",
                "Apples and oranges are both common fruits.",
                "Machine learning models learn from data.",
                "Photography captures moments in time.",
                "The weather today is sunny and warm.",
                "A recipe for success combines hard work and luck.");
        List<float[]> realVectors = embedViaDjl(sentences);

        PipeDoc input = buildStage2WithRealVectors(sentences, realVectors);

        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                /*paragraph*/ false, /*section*/ false, /*document*/ true,
                /*boundaries*/ true, MODEL_NAME,
                /*cap*/ null, /*simThr*/ null, /*pctThr*/ null,
                /*minSent*/ null, /*maxSent*/ null,
                /*batchSize*/ null, /*cap*/ null, /*maxRetry*/ null, /*backoff*/ null);

        PipeDoc out = service.process(input, opts, "it-step").await().atMost(TIMEOUT);

        // Invariant: output is valid Stage 3
        assertThat(SemanticPipelineInvariants.assertPostSemanticGraph(out))
                .as("Stage-3 output must satisfy assertPostSemanticGraph").isNull();

        // Stage-2 preservation
        for (SemanticProcessingResult inSpr : input.getSearchMetadata().getSemanticResultsList()) {
            assertThat(out.getSearchMetadata().getSemanticResultsList())
                    .as("Stage-2 SPR result_id=%s must appear unchanged", inSpr.getResultId())
                    .anyMatch(o -> o.equals(inSpr));
        }

        // Document centroid present, 384-d, granularity + metadata correct
        List<SemanticProcessingResult> doc = out.getSearchMetadata().getSemanticResultsList().stream()
                .filter(s -> "document_centroid".equals(s.getChunkConfigId())).toList();
        assertThat(doc).as("One document centroid per triple").hasSize(1);
        assertThat(doc.get(0).getGranularity()).isEqualTo(GranularityLevel.GRANULARITY_LEVEL_DOCUMENT);
        assertThat(doc.get(0).getCentroidMetadata().getSourceVectorCount())
                .isEqualTo(sentences.size());
        assertThat(doc.get(0).getChunks(0).getEmbeddingInfo().getVectorCount())
                .as("Centroid vector must carry MiniLM dim").isEqualTo(EXPECTED_DIMS);

        // Boundary SPR present, each chunk 384-d, semantic_config_id set
        List<SemanticProcessingResult> boundary = out.getSearchMetadata().getSemanticResultsList()
                .stream().filter(s -> "semantic".equals(s.getChunkConfigId())).toList();
        assertThat(boundary).as("One boundary SPR per (source, model)").hasSize(1);
        SemanticProcessingResult b = boundary.get(0);
        assertThat(b.getGranularity()).isEqualTo(GranularityLevel.GRANULARITY_LEVEL_SEMANTIC_CHUNK);
        assertThat(b.getSemanticConfigId()).isEqualTo("semantic:" + MODEL_NAME);
        assertThat(b.getChunksCount())
                .as("Boundary chunk count within [1, sentences.size()]")
                .isBetween(1, sentences.size());
        for (SemanticChunk sc : b.getChunksList()) {
            assertThat(sc.getEmbeddingInfo().getVectorCount())
                    .as("Every boundary chunk vector is 384-d MiniLM").isEqualTo(EXPECTED_DIMS);
            assertThat(sc.getEmbeddingInfo().getTextContent())
                    .as("Every boundary chunk carries grouped sentence text").isNotEmpty();
        }

        log.info("End-to-end happy path: {} Stage-2 SPR(s) + {} centroid(s) + {} boundary SPR(s) "
                + "with {} boundary chunk(s)",
                input.getSearchMetadata().getSemanticResultsCount(), doc.size(),
                boundary.size(), b.getChunksCount());
    }

    // ======================================================================
    // §13 gate measurement — p95 latency across 10 runs
    // ======================================================================

    @Test
    @DisplayName("§13 semantic-graph per-doc p95 latency measurement (target ≤ 500 ms)")
    void measurePerDocP95() {
        List<String> sentences = List.of(
                "Introduction paragraph with several sentences on the same theme.",
                "The quick brown fox jumps over the lazy dog on a sunny afternoon.",
                "Data structures and algorithms form the foundation of computer science.",
                "Machine learning models learn from observed examples to make predictions.",
                "Recipes for classic dishes often involve a handful of simple ingredients.",
                "The weather today is sunny with light breezes from the south.");
        List<float[]> realVectors = embedViaDjl(sentences);
        PipeDoc input = buildStage2WithRealVectors(sentences, realVectors);

        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, true, true, MODEL_NAME,
                null, null, null, null, null, null, null, null, null);

        // Warmup run — prime the model & Vert.x HttpClient pool.
        service.process(input, opts, "warmup").await().atMost(TIMEOUT);

        int iterations = 10;
        long[] timingsMs = new long[iterations];
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            PipeDoc out = service.process(input, opts, "it-measure-" + i).await().atMost(TIMEOUT);
            timingsMs[i] = (System.nanoTime() - start) / 1_000_000L;
            assertThat(SemanticPipelineInvariants.assertPostSemanticGraph(out))
                    .as("Every measured run must produce a valid Stage-3 doc")
                    .isNull();
        }
        Arrays.sort(timingsMs);
        long p50 = timingsMs[timingsMs.length / 2];
        long p95 = timingsMs[(int) Math.ceil(timingsMs.length * 0.95) - 1];
        long p99 = timingsMs[timingsMs.length - 1];

        log.info("R3 per-doc latency over {} runs: p50={}ms p95={}ms p99(max)={}ms",
                iterations, p50, p95, p99);
        log.info("DESIGN.md §13 gate: semantic-graph p95 ≤ 500ms — measured p95 = {}ms {}",
                p95, (p95 <= 500 ? "MET" : "NOT MET"));

        // Not enforced as an assertion (hardware-dependent, CI-dependent). The
        // log line is the deliverable; a regression alarm can be wired in CI
        // later using these timings as a baseline.
        assertThat(p95)
                .as("p95 must be strictly positive (sanity check, not the §13 gate)")
                .isGreaterThan(0);
    }

    // ======================================================================
    // §21.3 — model not loaded
    // ======================================================================

    @Test
    @DisplayName("§21.3: boundaries on with unknown model → FAILED_PRECONDITION")
    void unknownModel_failsFailedPrecondition() {
        List<String> sentences = List.of("first", "second", "third");
        List<float[]> realVectors = embedViaDjl(sentences);
        PipeDoc input = buildStage2WithRealVectors(sentences, realVectors);

        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, true, "not-a-real-model",
                null, null, null, null, null, null, null, null, null);

        Throwable err = null;
        try {
            service.process(input, opts, "it-missing-model").await().atMost(TIMEOUT);
        } catch (Throwable t) {
            err = t;
        }
        assertThat(err)
                .as("Unknown model must cause a failure the caller maps to FAILED_PRECONDITION")
                .isNotNull();
    }

    // ======================================================================
    // Fixture helpers — use REAL MiniLM vectors from the same DJL instance
    // ======================================================================

    private List<float[]> embedViaDjl(List<String> texts) {
        JsonObject body = new JsonObject().put("inputs", new JsonArray(texts));
        Uni<JsonArray> call = client.predict(MODEL_NAME, body);
        JsonArray resp = call.await().atMost(TIMEOUT);
        assertThat(resp).as("DJL predict returned %d rows for %d inputs", resp.size(), texts.size())
                .hasSize(texts.size());
        List<float[]> out = new ArrayList<>(texts.size());
        for (int i = 0; i < resp.size(); i++) {
            JsonArray v = resp.getJsonArray(i);
            assertThat(v).hasSize(EXPECTED_DIMS);
            float[] arr = new float[v.size()];
            for (int j = 0; j < v.size(); j++) arr[j] = v.getNumber(j).floatValue();
            out.add(arr);
        }
        return out;
    }

    private static PipeDoc buildStage2WithRealVectors(List<String> sentences, List<float[]> vectors) {
        SemanticProcessingResult.Builder spr = SemanticProcessingResult.newBuilder()
                .setResultId("stage2:it-doc:body:sentences_internal:" + MODEL_NAME)
                .setSourceFieldName("body")
                .setChunkConfigId("sentences_internal")
                .setEmbeddingConfigId(MODEL_NAME)
                .setNlpAnalysis(NlpDocumentAnalysis.getDefaultInstance())
                .putMetadata("directive_key",
                        Value.newBuilder().setStringValue("it-dk").build());

        StringBuilder body = new StringBuilder();
        int offset = 0;
        for (int i = 0; i < sentences.size(); i++) {
            String text = sentences.get(i);
            int start = offset;
            int end = start + text.length();
            body.append(text);
            if (i < sentences.size() - 1) { body.append(' '); end++; }
            offset = end;

            ChunkEmbedding.Builder emb = ChunkEmbedding.newBuilder()
                    .setTextContent(text)
                    .setChunkConfigId("sentences_internal")
                    .setOriginalCharStartOffset(start)
                    .setOriginalCharEndOffset(start + text.length());
            for (float f : vectors.get(i)) emb.addVector(f);
            spr.addChunks(SemanticChunk.newBuilder()
                    .setChunkId("s" + i)
                    .setChunkNumber(i)
                    .setEmbeddingInfo(emb.build())
                    .build());
        }

        VectorSetDirectives directives = VectorSetDirectives.newBuilder()
                .addDirectives(VectorDirective.newBuilder()
                        .setSourceLabel("body")
                        .setCelSelector("document.search_metadata.body")
                        .addChunkerConfigs(NamedChunkerConfig.newBuilder()
                                .setConfigId("sentences_internal")
                                .setConfig(Struct.newBuilder()
                                        .putFields("algorithm",
                                                Value.newBuilder().setStringValue("SENTENCE").build())
                                        .build())
                                .build())
                        .addEmbedderConfigs(NamedEmbedderConfig.newBuilder()
                                .setConfigId(MODEL_NAME).build())
                        .build())
                .build();

        return PipeDoc.newBuilder()
                .setDocId("it-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody(body.toString())
                        .addSemanticResults(spr.build())
                        .setVectorSetDirectives(directives)
                        .addSourceFieldAnalytics(SourceFieldAnalytics.newBuilder()
                                .setSourceField("body")
                                .setChunkConfigId("sentences_internal").build())
                        .build())
                .build();
    }

}
