package ai.pipestream.module.semanticmanager;

import ai.pipestream.data.module.v1.PipeStepProcessorServiceGrpc;
import ai.pipestream.data.module.v1.ProcessDataRequest;
import ai.pipestream.data.module.v1.ProcessDataResponse;
import ai.pipestream.data.module.v1.ProcessingOutcome;
import ai.pipestream.data.v1.PipeDoc;
import ai.pipestream.data.v1.SearchMetadata;
import ai.pipestream.data.v1.SemanticChunk;
import ai.pipestream.data.v1.SemanticProcessingResult;
import io.grpc.ManagedChannel;
import io.quarkus.test.junit.QuarkusIntegrationTest;
import io.quarkus.test.junit.TestProfile;
import org.eclipse.microprofile.config.ConfigProvider;
import org.jboss.logging.Logger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

import java.io.IOException;
import java.io.InputStream;
import java.net.Socket;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Integration tests for the semantic manager using REAL chunker and embedder services.
 * <p>
 * These tests require the chunker (port 19002) and embedder (port 19003) running
 * in dev mode. They are automatically skipped if either service is unreachable.
 * <p>
 * Exercises the full pipeline with real NLP tokenization, real sentence splitting,
 * and real embedding vectors — catches bugs that mock services cannot reproduce.
 */
@QuarkusIntegrationTest
@TestProfile(RealServicesTestProfile.class)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class SemanticManagerRealServicesIT {

    private static final Logger LOG = Logger.getLogger(SemanticManagerRealServicesIT.class);
    private static final int CHUNKER_PORT = 19002;
    private static final int EMBEDDER_PORT = 19003;

    private ManagedChannel channel;
    private PipeStepProcessorServiceGrpc.PipeStepProcessorServiceBlockingStub stub;

    @BeforeEach
    void setUp() {
        assumeTrue(isServiceReachable("localhost", CHUNKER_PORT),
                "Chunker not running on port " + CHUNKER_PORT + " — skipping real services test");
        assumeTrue(isServiceReachable("localhost", EMBEDDER_PORT),
                "Embedder not running on port " + EMBEDDER_PORT + " — skipping real services test");

        int port = ConfigProvider.getConfig().getValue("quarkus.http.test-port", Integer.class);
        LOG.infof("Real services IT: semantic-manager on port %d, chunker=%d, embedder=%d",
                port, CHUNKER_PORT, EMBEDDER_PORT);

        channel = io.grpc.netty.NettyChannelBuilder.forAddress("localhost", port)
                .usePlaintext()
                .maxInboundMessageSize(Integer.MAX_VALUE)
                .flowControlWindow(100 * 1024 * 1024)
                .build();
        stub = PipeStepProcessorServiceGrpc.newBlockingStub(channel)
                .withDeadlineAfter(5, TimeUnit.MINUTES);
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        if (channel != null) {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    // =========================================================================
    // Test 1: Simple 1×1 with real services
    // =========================================================================

    @Test
    @Order(1)
    void realScatterGather1x1() throws Exception {
        String text = SemanticManagerTestBase.loadResource("test-data/sample_article.txt");

        String jsonConfig = """
                {
                    "directives": [
                        {
                            "source_label": "body",
                            "cel_selector": "document.search_metadata.body",
                            "chunker_configs": [
                                {"config_id": "token_200_20", "config": {"algorithm": "TOKEN", "chunk_size": 200, "chunk_overlap": 20}}
                            ],
                            "embedder_configs": [
                                {"config_id": "minilm"}
                            ]
                        }
                    ]
                }
                """;

        ProcessDataRequest request = SemanticManagerTestBase.buildRequestWithJsonConfig(text, "Real 1x1 Article", jsonConfig);
        ProcessDataResponse response = stub.processData(request);

        SemanticManagerTestBase.assertSingleDirective(response, "minilm");

        // With real embedder, verify vectors are actually 384-dim
        SemanticProcessingResult result = response.getOutputDoc().getSearchMetadata().getSemanticResults(0);
        SemanticChunk firstChunk = result.getChunks(0);
        assertThat(firstChunk.getEmbeddingInfo().getVectorCount())
                .as("Real MiniLM embeddings should be 384-dimensional")
                .isEqualTo(384);

        // Verify vectors are not all zeros (real embeddings should be non-trivial)
        double sumSquared = firstChunk.getEmbeddingInfo().getVectorList().stream()
                .mapToDouble(Float::doubleValue)
                .map(v -> v * v)
                .sum();
        assertThat(sumSquared)
                .as("Real embedding vector should have non-zero magnitude")
                .isGreaterThan(0.01);

        LOG.infof("Real 1x1: %d chunks, vectorDim=%d, vectorNorm=%.4f",
                result.getChunksCount(), firstChunk.getEmbeddingInfo().getVectorCount(), Math.sqrt(sumSquared));
    }

    // =========================================================================
    // Test 2: 2×2 scatter-gather with real services
    // =========================================================================

    @Test
    @Order(2)
    void realScatterGather2x2() throws Exception {
        String text = SemanticManagerTestBase.loadResource("test-data/sample_article.txt");

        // Use two different source fields (body + title) × 1 embedder to get a valid 2×2
        // that doesn't depend on multiple embedding models being loaded.
        // 2 chunker configs × 2 embedder configs (same model, different config IDs)
        // won't work since model validation deduplicates by model name.
        // Instead: 2 directives (body + title) × 1 chunker × 1 embedder = 2 results,
        // BUT for a true 2×2 we use 2 chunker configs × 2 embedder configs with
        // both embedders pointing to the same model (different config_ids).
        String jsonConfig = """
                {
                    "directives": [
                        {
                            "source_label": "body",
                            "cel_selector": "document.search_metadata.body",
                            "chunker_configs": [
                                {"config_id": "sentence_5_1", "config": {"algorithm": "SENTENCE", "chunk_size": 5, "chunk_overlap": 1}},
                                {"config_id": "token_200_20", "config": {"algorithm": "TOKEN", "chunk_size": 200, "chunk_overlap": 20}}
                            ],
                            "embedder_configs": [
                                {"config_id": "minilm"},
                                {"config_id": "minilm"}
                            ]
                        }
                    ]
                }
                """;

        ProcessDataRequest request = SemanticManagerTestBase.buildRequestWithJsonConfig(text, "Real 2x2 Article", jsonConfig);
        long startMs = System.currentTimeMillis();
        ProcessDataResponse response = stub.processData(request);
        long elapsedMs = System.currentTimeMillis() - startMs;

        SemanticManagerTestBase.assertScatterGather2x2(response);

        LOG.infof("Real 2x2: 4 results in %dms", elapsedMs);
    }

    // =========================================================================
    // Test 3: Field-level embedding with real embedder
    // =========================================================================

    @Test
    @Order(3)
    void realFieldLevelEmbedding() throws Exception {
        String text = "This document tests field-level embedding with the real embedder service. "
                + "No chunking should occur — the entire text is embedded as a single vector.";

        String jsonConfig = """
                {
                    "source_field": "body",
                    "embedding_model": "minilm",
                    "skip_chunking": true
                }
                """;

        ProcessDataRequest request = SemanticManagerTestBase.buildRequestWithJsonConfig(text, "Real Field-Level", jsonConfig);
        ProcessDataResponse response = stub.processData(request);

        SemanticManagerTestBase.assertFieldLevelResult(response, "body");
    }

    // =========================================================================
    // Test 4: Court opinions throughput with real services
    // =========================================================================

    @Test
    @Order(4)
    void realCourtOpinionsThroughput() throws Exception {
        List<SemanticManagerTestBase.CourtOpinion> opinions = SemanticManagerTestBase.loadOpinions(10);

        assertThat(opinions)
                .as("Should load at least 10 court opinions")
                .hasSizeGreaterThanOrEqualTo(10);

        String jsonConfig = """
                {
                    "directives": [
                        {
                            "source_label": "body",
                            "cel_selector": "document.search_metadata.body",
                            "chunker_configs": [
                                {"config_id": "token_500_50", "config": {"algorithm": "TOKEN", "chunk_size": 500, "chunk_overlap": 50}}
                            ],
                            "embedder_configs": [
                                {"config_id": "minilm"}
                            ]
                        }
                    ]
                }
                """;

        long totalMs = 0;
        int totalChunks = 0;

        for (int i = 0; i < opinions.size(); i++) {
            SemanticManagerTestBase.CourtOpinion opinion = opinions.get(i);
            long docStart = System.currentTimeMillis();

            ProcessDataRequest request = SemanticManagerTestBase.buildRequestWithJsonConfig(
                    opinion.plainText,
                    opinion.caseName != null ? opinion.caseName : "Opinion-" + i,
                    jsonConfig);

            ProcessDataResponse response = stub.processData(request);
            long docMs = System.currentTimeMillis() - docStart;
            totalMs += docMs;

            assertThat(response.getOutcome())
                    .as("Court opinion %d ('%s') should succeed", i, opinion.caseName)
                    .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

            assertThat(response.hasOutputDoc())
                    .as("Court opinion %d should have output doc", i)
                    .isTrue();

            int chunks = response.getOutputDoc().getSearchMetadata().getSemanticResults(0).getChunksCount();
            totalChunks += chunks;

            LOG.infof("  Opinion %d: '%s' — %d chars, %d chunks, %dms",
                    i, opinion.caseName, opinion.plainText.length(), chunks, docMs);
        }

        LOG.infof("Real court opinions: %d docs, %d total chunks in %dms (avg %dms/doc)",
                opinions.size(), totalChunks, totalMs, totalMs / opinions.size());
    }

    // =========================================================================
    // Test 5: Constitution stress test with real services
    // =========================================================================

    @Test
    @Order(5)
    void realConstitutionStressTest() throws Exception {
        String text = SemanticManagerTestBase.loadResource("test-data/constitution.txt");
        LOG.infof("Constitution: %d chars", text.length());

        // Use only minilm to avoid dependency on multiple models being loaded
        String jsonConfig = """
                {
                    "directives": [
                        {
                            "source_label": "body",
                            "cel_selector": "document.search_metadata.body",
                            "chunker_configs": [
                                {"config_id": "sentence_10_2", "config": {"algorithm": "SENTENCE", "chunk_size": 10, "chunk_overlap": 2}},
                                {"config_id": "token_500_50", "config": {"algorithm": "TOKEN", "chunk_size": 500, "chunk_overlap": 50}}
                            ],
                            "embedder_configs": [
                                {"config_id": "minilm"},
                                {"config_id": "minilm"}
                            ]
                        }
                    ]
                }
                """;

        long startMs = System.currentTimeMillis();
        ProcessDataRequest request = SemanticManagerTestBase.buildRequestWithJsonConfig(text, "US Constitution", jsonConfig);
        ProcessDataResponse response = stub.processData(request);
        long elapsedMs = System.currentTimeMillis() - startMs;

        SemanticManagerTestBase.assertScatterGather2x2(response);

        int totalChunks = response.getOutputDoc().getSearchMetadata().getSemanticResultsList().stream()
                .mapToInt(SemanticProcessingResult::getChunksCount)
                .sum();

        LOG.infof("Real Constitution stress: 4 results, %d total chunks in %dms", totalChunks, elapsedMs);
    }

    // =========================================================================
    // Test 6: Parsed PipeDoc batch with real services
    // =========================================================================

    @Test
    @Order(6)
    void realParsedPipeDocBatch() throws Exception {
        String jsonConfig = """
                {
                    "directives": [
                        {
                            "source_label": "body",
                            "cel_selector": "document.search_metadata.body",
                            "chunker_configs": [
                                {"config_id": "token_300_30", "config": {"algorithm": "TOKEN", "chunk_size": 300, "chunk_overlap": 30}}
                            ],
                            "embedder_configs": [
                                {"config_id": "minilm"}
                            ]
                        }
                    ]
                }
                """;

        int[] docNums = {1, 5, 10, 20, 30, 50, 70, 80, 90, 100};
        int processed = 0;
        int skipped = 0;
        long totalMs = 0;

        for (int docNum : docNums) {
            String resourceName = String.format("test-data/parsed-docs/parsed_document_%03d.pb", docNum);
            try (InputStream is = getClass().getClassLoader().getResourceAsStream(resourceName)) {
                if (is == null) {
                    LOG.warnf("Resource not found: %s", resourceName);
                    skipped++;
                    continue;
                }

                PipeDoc doc = PipeDoc.parseFrom(is.readAllBytes());
                String body = "";
                if (doc.hasSearchMetadata() && doc.getSearchMetadata().hasBody()) {
                    body = doc.getSearchMetadata().getBody();
                }

                if (body.isBlank()) {
                    LOG.infof("Document %03d has no body, skipping", docNum);
                    skipped++;
                    continue;
                }

                long docStart = System.currentTimeMillis();
                ProcessDataRequest request = SemanticManagerTestBase.buildRequestWithJsonConfig(
                        body, "Parsed Doc " + docNum, jsonConfig);

                ProcessDataResponse response = stub.processData(request);
                long docMs = System.currentTimeMillis() - docStart;
                totalMs += docMs;

                assertThat(response.getOutcome())
                        .as("Parsed doc %03d should succeed or partially succeed", docNum)
                        .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

                assertThat(response.hasOutputDoc())
                        .as("Parsed doc %03d should have output doc", docNum)
                        .isTrue();

                int chunks = response.getOutputDoc().getSearchMetadata().getSemanticResultsCount() > 0
                        ? response.getOutputDoc().getSearchMetadata().getSemanticResults(0).getChunksCount()
                        : 0;

                LOG.infof("  Doc %03d: %d chars, %d chunks, %dms", docNum, body.length(), chunks, docMs);
                processed++;
            }
        }

        assertThat(processed)
                .as("Should process at least 5 parsed documents successfully")
                .isGreaterThanOrEqualTo(5);

        LOG.infof("Real parsed docs: %d processed, %d skipped, %dms total", processed, skipped, totalMs);
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static boolean isServiceReachable(String host, int port) {
        try (Socket socket = new Socket()) {
            socket.connect(new java.net.InetSocketAddress(host, port), 1000);
            return true;
        } catch (IOException e) {
            return false;
        }
    }
}
