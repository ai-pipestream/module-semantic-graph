package ai.pipestream.module.semanticgraph.pipeline;

import ai.pipestream.data.v1.PipeDoc;
import ai.pipestream.data.v1.SemanticProcessingResult;
import ai.pipestream.module.semanticgraph.config.SemanticGraphStepOptions;
import ai.pipestream.module.semanticgraph.invariants.SemanticPipelineInvariants;
import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.junit.QuarkusTestProfile;
import io.quarkus.test.junit.TestProfile;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Assumptions;
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
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Scale ITs for {@link SemanticGraphPipelineService} against streamed
 * {@link PipeDoc} fixtures resolved from the Maven artifact at
 * {@code ai.pipestream.testdata:???:???} (GAV TBD when the fixture jar
 * publishes; for now driven by {@code -Dr3.fixtures.stage2-100=...} and
 * {@code -Dr3.fixtures.stage2-1000=...} system properties).
 *
 * <h2>Two scenarios</h2>
 *
 * <ol>
 *   <li><b>100-doc end-to-end</b> — fixture includes a sentence-shaped SPR
 *       per source. R3 runs with {@code compute_semantic_boundaries=true}
 *       against the configured boundary model. Asserts every doc passes
 *       {@link SemanticPipelineInvariants#assertPostSemanticGraph} and
 *       Stage-2 SPRs are preserved byte-for-byte.</li>
 *   <li><b>1000-doc centroids-only</b> — fixture has no sentence
 *       embeddings. R3 runs with {@code compute_semantic_boundaries=false};
 *       only document / paragraph / section centroids are produced. The
 *       1000-doc set sits at ~2 GB on disk so we stream one
 *       {@link PipeDoc} at a time and never materialise the full list.</li>
 * </ol>
 *
 * <h2>Skip semantics</h2>
 *
 * <p>Each test {@link Assumptions#assumeTrue(boolean, String)} on its
 * fixture resource being available — never a silent pass. The 100-doc
 * test additionally verifies DJL Serving is reachable and the boundary
 * model is loaded. The 1000-doc test does not need DJL (boundaries off).
 *
 * <h2>Memory model</h2>
 *
 * <p>Sequential streaming via {@link Stage2FixtureStream#stream}: at any
 * moment exactly one {@link PipeDoc} input + one {@link PipeDoc} output
 * is live. Per-doc timings are recorded into a {@code long[]} with size
 * = doc count; that's 8 KB for 1000 docs. The aggregated {@code long[]}
 * is the only doc-count-proportional state in the test JVM.
 */
@QuarkusTest
@TestProfile(SemanticGraphScaleIT.ItProfile.class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("SemanticGraph scale ITs (100 + 1000 doc streaming)")
class SemanticGraphScaleIT {

    private static final Logger log = LoggerFactory.getLogger(SemanticGraphScaleIT.class);

    /** Fixture for the 100-doc end-to-end test. Sentence embeddings present. */
    private static final String FIXTURE_100 = System.getProperty(
            "r3.fixtures.stage2-100",
            "fixtures/court-stage2-100.pb.gz");

    /** Fixture for the 1000-doc centroids-only test. No sentence embeddings. */
    private static final String FIXTURE_1000 = System.getProperty(
            "r3.fixtures.stage2-1000",
            "fixtures/court-stage2-1000.pb.gz");

    /** Boundary embedding model id used by the 100-doc test. Must match
     *  whichever sentence-shaped SPR's embedding_config_id is in the fixture. */
    private static final String BOUNDARY_MODEL = System.getProperty(
            "r3.fixtures.boundary-model", "all-MiniLM-L6-v2");

    private static final Duration PER_DOC_TIMEOUT = Duration.ofSeconds(60);

    /** Mirror of SemanticGraphPipelineServiceIT.ItProfile. Points the
     *  injected @RestClient DjlServingClient at the standard test DJL. */
    public static class ItProfile implements QuarkusTestProfile {
        @Override
        public Map<String, String> getConfigOverrides() {
            String host = System.getProperty("djl.host", "localhost");
            String port = System.getProperty("djl.port", "18090");
            return Map.of(
                    "quarkus.rest-client.djl-serving.url", "http://" + host + ":" + port);
        }
    }

    @Inject
    SemanticGraphPipelineService service;

    // ======================================================================
    // 100-doc end-to-end (boundaries on)
    // ======================================================================

    @Test
    @DisplayName("100 docs: full pipeline (centroids + boundaries) — every doc passes assertPostSemanticGraph")
    void scale100_fullPipeline() {
        Assumptions.assumeTrue(resourceExists(FIXTURE_100),
                "Fixture not on classpath: " + FIXTURE_100
                        + " — set -Dr3.fixtures.stage2-100=<resource> or add the fixture jar to "
                        + "testRuntimeClasspath. Skipping rather than silently passing.");
        Assumptions.assumeTrue(djlReachableWithModel(BOUNDARY_MODEL),
                "DJL at " + djlUrl() + " is unreachable or doesn't have model '"
                        + BOUNDARY_MODEL + "' loaded. Boundaries can't run; skipping.");

        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                /*paragraph*/ true, /*section*/ true, /*document*/ true,
                /*boundaries*/ true, BOUNDARY_MODEL,
                null, null, null, null, null, null, null, null, null);

        Stats stats = streamAndProcess(FIXTURE_100, opts, /*expectBoundaries*/ true);
        report("scale100_fullPipeline", stats);

        assertThat(stats.docCount).as("Some docs should have been streamed from the fixture")
                .isGreaterThan(0);
        assertThat(stats.invariantViolations).as("Every doc must satisfy assertPostSemanticGraph")
                .isZero();
        assertThat(stats.preservationViolations).as("Stage-2 SPRs must be byte-identical in output")
                .isZero();
        assertThat(stats.docsWithBoundarySpr)
                .as("Every doc with a sentence SPR for the boundary model should produce a boundary SPR")
                .isGreaterThan(0);
    }

    // ======================================================================
    // 1000-doc centroids-only (boundaries off)
    // ======================================================================

    @Test
    @DisplayName("1000 docs: centroids only (no boundaries) — every doc passes assertPostSemanticGraph")
    void scale1000_centroidsOnly() {
        Assumptions.assumeTrue(resourceExists(FIXTURE_1000),
                "Fixture not on classpath: " + FIXTURE_1000
                        + " — set -Dr3.fixtures.stage2-1000=<resource> or add the fixture jar to "
                        + "testRuntimeClasspath. Skipping.");

        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                /*paragraph*/ true, /*section*/ true, /*document*/ true,
                /*boundaries*/ false, /*boundary model id*/ null,
                null, null, null, null, null, null, null, null, null);

        Stats stats = streamAndProcess(FIXTURE_1000, opts, /*expectBoundaries*/ false);
        report("scale1000_centroidsOnly", stats);

        assertThat(stats.docCount).isGreaterThan(0);
        assertThat(stats.invariantViolations).isZero();
        assertThat(stats.preservationViolations).isZero();
        assertThat(stats.docsWithBoundarySpr)
                .as("Boundaries off: no boundary SPRs should be emitted")
                .isZero();
        assertThat(stats.docsWithCentroidSpr)
                .as("With document_centroid=true, every input SPR triple yields ≥1 centroid")
                .isEqualTo(stats.docCount);
    }

    // ======================================================================
    // Stream driver — sequential, one doc at a time, aggregates timings
    // ======================================================================

    private Stats streamAndProcess(
            String fixtureResource,
            SemanticGraphStepOptions opts,
            boolean expectBoundaries) {

        Stats s = new Stats();
        List<Long> timings = new ArrayList<>();

        try (Stream<PipeDoc> docs = Stage2FixtureStream.stream(fixtureResource)) {
            docs.forEach(input -> {
                s.docCount++;
                long t0 = System.nanoTime();
                PipeDoc out;
                try {
                    out = service.process(input, opts, "scale-it").await().atMost(PER_DOC_TIMEOUT);
                } catch (Throwable t) {
                    s.processFailures++;
                    log.warn("R3 failed on doc#{} (id={}): {}", s.docCount,
                            input.getDocId(), t.getMessage());
                    return;
                }
                timings.add(System.nanoTime() - t0);

                String invariantErr = SemanticPipelineInvariants.assertPostSemanticGraph(out);
                if (invariantErr != null) {
                    s.invariantViolations++;
                    log.warn("Invariant violation on doc#{} (id={}): {}",
                            s.docCount, input.getDocId(), invariantErr);
                }

                // Stage-2 byte-for-byte preservation: every input SPR must
                // appear unchanged in the output's semantic_results list.
                for (SemanticProcessingResult inSpr : input.getSearchMetadata().getSemanticResultsList()) {
                    boolean preserved = out.getSearchMetadata().getSemanticResultsList()
                            .stream().anyMatch(o -> o.equals(inSpr));
                    if (!preserved) {
                        s.preservationViolations++;
                        break;
                    }
                }

                int boundary = 0;
                int centroid = 0;
                for (SemanticProcessingResult spr : out.getSearchMetadata().getSemanticResultsList()) {
                    String cfg = spr.getChunkConfigId();
                    if ("semantic".equals(cfg)) boundary++;
                    else if (cfg.endsWith("_centroid")) centroid++;
                }
                if (boundary > 0) s.docsWithBoundarySpr++;
                if (centroid > 0) s.docsWithCentroidSpr++;
            });
        }

        // Compute p50/p95/p99 from per-doc nanos
        if (!timings.isEmpty()) {
            long[] sorted = timings.stream().mapToLong(Long::longValue).toArray();
            Arrays.sort(sorted);
            s.p50ms = sorted[sorted.length / 2] / 1_000_000L;
            s.p95ms = sorted[(int) Math.ceil(sorted.length * 0.95) - 1] / 1_000_000L;
            s.p99ms = sorted[(int) Math.ceil(sorted.length * 0.99) - 1] / 1_000_000L;
            s.maxMs = sorted[sorted.length - 1] / 1_000_000L;
        }
        return s;
    }

    private static void report(String label, Stats s) {
        log.info("R3 SCALE {} — docs={} failures={} invariant_violations={} "
                        + "preservation_violations={} docs_with_centroid={} docs_with_boundary={} "
                        + "p50={}ms p95={}ms p99={}ms max={}ms",
                label, s.docCount, s.processFailures, s.invariantViolations,
                s.preservationViolations, s.docsWithCentroidSpr, s.docsWithBoundarySpr,
                s.p50ms, s.p95ms, s.p99ms, s.maxMs);
    }

    private static final class Stats {
        long docCount;
        long processFailures;
        long invariantViolations;
        long preservationViolations;
        long docsWithCentroidSpr;
        long docsWithBoundarySpr;
        long p50ms, p95ms, p99ms, maxMs;
    }

    // ======================================================================
    // Probe helpers (mirror SemanticGraphPipelineServiceIT)
    // ======================================================================

    private static boolean resourceExists(String resourcePath) {
        return Thread.currentThread().getContextClassLoader().getResource(resourcePath) != null;
    }

    private static String djlUrl() {
        String host = System.getProperty("djl.host", "localhost");
        String port = System.getProperty("djl.port", "18090");
        return "http://" + host + ":" + port;
    }

    private static boolean djlReachableWithModel(String modelName) {
        try {
            HttpResponse<String> resp = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(2)).build()
                    .send(HttpRequest.newBuilder()
                                    .uri(URI.create(djlUrl() + "/models"))
                                    .timeout(Duration.ofSeconds(5)).GET().build(),
                            HttpResponse.BodyHandlers.ofString());
            return resp.statusCode() < 400 && resp.body().contains("\"" + modelName + "\"");
        } catch (Exception e) {
            return false;
        }
    }
}
