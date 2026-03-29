package ai.pipestream.module.semanticmanager;

import ai.pipestream.data.module.v1.PipeStepProcessorServiceGrpc;
import ai.pipestream.data.module.v1.ProcessDataRequest;
import ai.pipestream.data.module.v1.ProcessDataResponse;
import ai.pipestream.data.module.v1.ProcessingOutcome;
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
import java.net.Socket;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Stress tests for the semantic manager with large real-world documents.
 * <p>
 * Excluded from default {@code quarkusIntTest} runs. Run explicitly:
 * <pre>
 *   ./gradlew quarkusIntTest -Dinclude.stress=true
 * </pre>
 * <p>
 * Requires real chunker (port 19002) and embedder (port 19003) running in dev mode.
 * Tests automatically skip if services are unreachable.
 */
@Stress
@QuarkusIntegrationTest
@TestProfile(RealServicesTestProfile.class)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class SemanticManagerStressIT {

    private static final Logger LOG = Logger.getLogger(SemanticManagerStressIT.class);
    private static final int CHUNKER_PORT = 19002;
    private static final int EMBEDDER_PORT = 19003;

    private ManagedChannel channel;
    private PipeStepProcessorServiceGrpc.PipeStepProcessorServiceBlockingStub stub;

    @BeforeEach
    void setUp() {
        assumeTrue(isServiceReachable("localhost", CHUNKER_PORT),
                "Chunker not running on port " + CHUNKER_PORT);
        assumeTrue(isServiceReachable("localhost", EMBEDDER_PORT),
                "Embedder not running on port " + EMBEDDER_PORT);

        int port = ConfigProvider.getConfig().getValue("quarkus.http.test-port", Integer.class);
        LOG.infof("Stress IT: semantic-manager on port %d", port);

        channel = io.grpc.netty.NettyChannelBuilder.forAddress("localhost", port)
                .usePlaintext()
                .maxInboundMessageSize(Integer.MAX_VALUE)
                .flowControlWindow(100 * 1024 * 1024)
                .build();
        stub = PipeStepProcessorServiceGrpc.newBlockingStub(channel)
                .withDeadlineAfter(10, TimeUnit.MINUTES);
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        if (channel != null) {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    // =========================================================================
    // Test 1: King James Bible — 4.5MB, ~800K tokens
    // =========================================================================

    @Test
    @Order(1)
    void bibleKjvSingleEmbedder() throws Exception {
        String text = SemanticManagerTestBase.loadResource("test-data/bible_kjv.txt");
        LOG.infof("Bible KJV: %d chars (~%.1f MB)", text.length(), text.length() / 1_000_000.0);

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

        long startMs = System.currentTimeMillis();
        ProcessDataRequest request = SemanticManagerTestBase.buildRequestWithJsonConfig(
                text, "King James Bible", jsonConfig);
        ProcessDataResponse response = stub.processData(request);
        long elapsedMs = System.currentTimeMillis() - startMs;

        assertThat(response.getOutcome())
                .as("Bible KJV should complete successfully")
                .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

        assertThat(response.hasOutputDoc())
                .as("Should return enriched document")
                .isTrue();

        SemanticProcessingResult result = response.getOutputDoc().getSearchMetadata().getSemanticResults(0);
        assertThat(result.getChunksCount())
                .as("Bible should produce many chunks")
                .isGreaterThan(100);

        SemanticManagerTestBase.assertAllChunksHaveEmbeddings(response.getOutputDoc());

        LOG.infof("╔═══════════════════════════════════════════════════╗");
        LOG.infof("║  BIBLE KJV STRESS TEST                           ║");
        LOG.infof("║  Text:    %,d chars (%.1f MB)               ║", text.length(), text.length() / 1_000_000.0);
        LOG.infof("║  Chunks:  %,d                                    ║", result.getChunksCount());
        LOG.infof("║  Vectors: %d-dim per chunk                       ║", result.getChunks(0).getEmbeddingInfo().getVectorCount());
        LOG.infof("║  Time:    %,d ms (%.1f s)                        ║", elapsedMs, elapsedMs / 1000.0);
        LOG.infof("╚═══════════════════════════════════════════════════╝");
    }

    // =========================================================================
    // Test 2: Bible KJV with 2×2 scatter-gather
    // =========================================================================

    @Test
    @Order(2)
    void bibleKjv2x2() throws Exception {
        String text = SemanticManagerTestBase.loadResource("test-data/bible_kjv.txt");

        String jsonConfig = """
                {
                    "directives": [
                        {
                            "source_label": "body",
                            "cel_selector": "document.search_metadata.body",
                            "chunker_configs": [
                                {"config_id": "token_500_50", "config": {"algorithm": "TOKEN", "chunk_size": 500, "chunk_overlap": 50}},
                                {"config_id": "sentence_10_2", "config": {"algorithm": "SENTENCE", "chunk_size": 10, "chunk_overlap": 2}}
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
        ProcessDataRequest request = SemanticManagerTestBase.buildRequestWithJsonConfig(
                text, "Bible KJV 2x2", jsonConfig);
        ProcessDataResponse response = stub.processData(request);
        long elapsedMs = System.currentTimeMillis() - startMs;

        assertThat(response.getOutcome())
                .as("Bible 2x2 should complete successfully")
                .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

        int resultCount = response.getOutputDoc().getSearchMetadata().getSemanticResultsCount();
        assertThat(resultCount)
                .as("2 chunkers × 2 embedders = 4 results")
                .isEqualTo(4);

        int totalChunks = response.getOutputDoc().getSearchMetadata().getSemanticResultsList().stream()
                .mapToInt(SemanticProcessingResult::getChunksCount)
                .sum();

        LOG.infof("Bible KJV 2x2: %d results, %,d total chunks in %.1fs", resultCount, totalChunks, elapsedMs / 1000.0);
    }

    // =========================================================================
    // Test 3: Gray's Anatomy — dense medical text (544KB)
    // =========================================================================

    @Test
    @Order(3)
    void graysAnatomyStress() throws Exception {
        String text = SemanticManagerTestBase.loadResource("test-data/grays_anatomy.txt");
        LOG.infof("Gray's Anatomy: %d chars", text.length());

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

        long startMs = System.currentTimeMillis();
        ProcessDataRequest request = SemanticManagerTestBase.buildRequestWithJsonConfig(
                text, "Gray's Anatomy", jsonConfig);
        ProcessDataResponse response = stub.processData(request);
        long elapsedMs = System.currentTimeMillis() - startMs;

        assertThat(response.getOutcome())
                .as("Gray's Anatomy should complete")
                .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

        SemanticProcessingResult result = response.getOutputDoc().getSearchMetadata().getSemanticResults(0);
        LOG.infof("Gray's Anatomy: %d chunks in %.1fs", result.getChunksCount(), elapsedMs / 1000.0);
    }

    // =========================================================================
    // Test 4: Blackstone's Commentaries — formal legal text (234KB)
    // =========================================================================

    @Test
    @Order(4)
    void blackstoneStress() throws Exception {
        String text = SemanticManagerTestBase.loadResource("test-data/blackstone_commentaries.txt");
        LOG.infof("Blackstone: %d chars", text.length());

        String jsonConfig = """
                {
                    "directives": [
                        {
                            "source_label": "body",
                            "cel_selector": "document.search_metadata.body",
                            "chunker_configs": [
                                {"config_id": "sentence_5_1", "config": {"algorithm": "SENTENCE", "chunk_size": 5, "chunk_overlap": 1}}
                            ],
                            "embedder_configs": [
                                {"config_id": "minilm"}
                            ]
                        }
                    ]
                }
                """;

        long startMs = System.currentTimeMillis();
        ProcessDataRequest request = SemanticManagerTestBase.buildRequestWithJsonConfig(
                text, "Blackstone's Commentaries", jsonConfig);
        ProcessDataResponse response = stub.processData(request);
        long elapsedMs = System.currentTimeMillis() - startMs;

        assertThat(response.getOutcome())
                .as("Blackstone should complete")
                .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

        SemanticProcessingResult result = response.getOutputDoc().getSearchMetadata().getSemanticResults(0);
        LOG.infof("Blackstone: %d chunks in %.1fs", result.getChunksCount(), elapsedMs / 1000.0);
    }

    // =========================================================================
    // Test 5: Court opinions batch — 50 opinions with 2x1
    // =========================================================================

    @Test
    @Order(5)
    void courtOpinions50DocBatch() throws Exception {
        List<SemanticManagerTestBase.CourtOpinion> opinions = SemanticManagerTestBase.loadOpinions(50);
        assertThat(opinions).as("Should load 50 opinions").hasSizeGreaterThanOrEqualTo(50);

        String jsonConfig = """
                {
                    "directives": [
                        {
                            "source_label": "body",
                            "cel_selector": "document.search_metadata.body",
                            "chunker_configs": [
                                {"config_id": "token_500_50", "config": {"algorithm": "TOKEN", "chunk_size": 500, "chunk_overlap": 50}},
                                {"config_id": "sentence_10_2", "config": {"algorithm": "SENTENCE", "chunk_size": 10, "chunk_overlap": 2}}
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
        int failures = 0;

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

            if (response.getOutcome() == ProcessingOutcome.PROCESSING_OUTCOME_FAILURE) {
                failures++;
                LOG.warnf("  Opinion %d FAILED: '%s' (%d chars)", i, opinion.caseName, opinion.plainText.length());
                continue;
            }

            int chunks = response.getOutputDoc().getSearchMetadata().getSemanticResultsList().stream()
                    .mapToInt(SemanticProcessingResult::getChunksCount)
                    .sum();
            totalChunks += chunks;

            if (i % 10 == 0) {
                LOG.infof("  Opinion %d/%d: '%s' — %d chars, %d chunks, %dms",
                        i, opinions.size(), opinion.caseName, opinion.plainText.length(), chunks, docMs);
            }
        }

        assertThat(failures)
                .as("No more than 5 failures allowed out of 50 opinions")
                .isLessThanOrEqualTo(5);

        LOG.infof("╔═══════════════════════════════════════════════════╗");
        LOG.infof("║  COURT OPINIONS BATCH STRESS TEST                ║");
        LOG.infof("║  Documents: %d                                   ║", opinions.size());
        LOG.infof("║  Failures:  %d                                   ║", failures);
        LOG.infof("║  Chunks:    %,d total                            ║", totalChunks);
        LOG.infof("║  Time:      %,d ms (%.1f s)                      ║", totalMs, totalMs / 1000.0);
        LOG.infof("║  Avg:       %d ms/doc                            ║", totalMs / opinions.size());
        LOG.infof("╚═══════════════════════════════════════════════════╝");
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
