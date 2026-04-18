package ai.pipestream.module.semanticgraph;

import ai.pipestream.data.module.v1.GetServiceRegistrationRequest;
import ai.pipestream.data.module.v1.GetServiceRegistrationResponse;
import ai.pipestream.data.module.v1.ProcessDataRequest;
import ai.pipestream.data.module.v1.ProcessDataResponse;
import ai.pipestream.data.module.v1.ProcessingOutcome;
import ai.pipestream.data.module.v1.PipeStepProcessorServiceGrpc;
import ai.pipestream.data.module.v1.ServiceMetadata;
import ai.pipestream.data.v1.PipeDoc;
import ai.pipestream.data.v1.ProcessConfiguration;
import ai.pipestream.data.v1.SearchMetadata;
import ai.pipestream.data.v1.SemanticChunk;
import ai.pipestream.data.v1.SemanticProcessingResult;
import com.google.protobuf.Struct;
import com.google.protobuf.util.JsonFormat;
import io.grpc.ManagedChannel;
import io.quarkus.test.common.QuarkusTestResource;
import io.quarkus.test.junit.QuarkusIntegrationTest;
import io.quarkus.test.junit.TestProfile;
import org.eclipse.microprofile.config.ConfigProvider;
import org.jboss.logging.Logger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration tests for the semantic-manager module running as a packaged JAR.
 * <p>
 * Tests the full scatter-gather orchestrator pipeline:
 * test client -> SemanticManagerGrpcImpl -> orchestrator -> mock chunker -> mock embedder -> enriched PipeDoc
 * <p>
 * MockServicesTestResource starts external gRPC servers for mock chunker and embedder.
 * Stork static discovery routes the orchestrator's outbound gRPC calls to those servers.
 */
@QuarkusIntegrationTest
@TestProfile(SemanticManagerIntegrationTestProfile.class)
@QuarkusTestResource(MockServicesTestResource.class)
class SemanticManagerIT {

    private static final Logger LOG = Logger.getLogger(SemanticManagerIT.class);

    private ManagedChannel channel;
    private PipeStepProcessorServiceGrpc.PipeStepProcessorServiceBlockingStub stub;

    @BeforeEach
    void setUp() {
        int port = ConfigProvider.getConfig().getValue("quarkus.http.test-port", Integer.class);
        LOG.infof("Connecting gRPC client to localhost:%d", port);

        channel = io.grpc.netty.NettyChannelBuilder.forAddress("localhost", port)
                .usePlaintext()
                .maxInboundMessageSize(Integer.MAX_VALUE)
                .flowControlWindow(100 * 1024 * 1024)
                .build();
        stub = PipeStepProcessorServiceGrpc.newBlockingStub(channel)
                .withDeadlineAfter(2, TimeUnit.MINUTES);
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        if (channel != null) {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    // =========================================================================
    // Test 1: 2x2 scatter-gather (2 chunkers x 2 embedders = 4 results)
    // =========================================================================

    @Test
    void scatterGather2x2ProducesAllResults() throws Exception {
        String text = loadResource("test-data/sample_article.txt");

        String jsonConfig = """
                {
                    "directives": [
                        {
                            "source_label": "body",
                            "cel_selector": "document.search_metadata.body",
                            "chunker_configs": [
                                {"config_id": "sentence_10_3", "config": {"algorithm": "SENTENCE", "chunk_size": 10, "chunk_overlap": 3}},
                                {"config_id": "token_500_50", "config": {"algorithm": "TOKEN", "chunk_size": 500, "chunk_overlap": 50}}
                            ],
                            "embedder_configs": [
                                {"config_id": "all-MiniLM-L6-v2"},
                                {"config_id": "all-mpnet-base-v2"}
                            ]
                        }
                    ]
                }
                """;

        ProcessDataRequest request = buildRequestWithJsonConfig(text, "2x2 Test Article", jsonConfig);
        ProcessDataResponse response = stub.processData(request);

        assertThat(response.getOutcome())
                .as("2x2 scatter-gather should succeed or partially succeed")
                .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

        assertThat(response.hasOutputDoc())
                .as("Response should contain an enriched output document")
                .isTrue();

        PipeDoc outputDoc = response.getOutputDoc();
        int resultCount = outputDoc.getSearchMetadata().getSemanticResultsCount();
        LOG.infof("2x2 test: %d SemanticProcessingResults", resultCount);

        assertThat(resultCount)
                .as("2 chunkers x 2 embedders should produce 4 SemanticProcessingResults")
                .isEqualTo(4);

        // Verify each result has chunks with embeddings
        for (SemanticProcessingResult result : outputDoc.getSearchMetadata().getSemanticResultsList()) {
            assertThat(result.getChunksCount())
                    .as("Result '%s' (chunker=%s, embedder=%s) should have at least 1 chunk",
                            result.getResultSetName(), result.getChunkConfigId(), result.getEmbeddingConfigId())
                    .isGreaterThan(0);

            for (SemanticChunk chunk : result.getChunksList()) {
                assertThat(chunk.getEmbeddingInfo().getVectorCount())
                        .as("Chunk %d in result '%s' should have embedding vectors",
                                chunk.getChunkNumber(), result.getResultSetName())
                        .isGreaterThan(0);
            }

            LOG.infof("  result='%s': chunker=%s, embedder=%s, chunks=%d, vectorDim=%d",
                    result.getResultSetName(),
                    result.getChunkConfigId(),
                    result.getEmbeddingConfigId(),
                    result.getChunksCount(),
                    result.getChunks(0).getEmbeddingInfo().getVectorCount());
        }

        // Verify result set names contain expected combinations
        List<String> resultSetNames = outputDoc.getSearchMetadata().getSemanticResultsList().stream()
                .map(SemanticProcessingResult::getResultSetName)
                .toList();
        LOG.infof("Result set names: %s", resultSetNames);

        assertThat(resultSetNames)
                .as("Result set names should contain all 4 combinations of chunker x embedder")
                .hasSize(4);
    }

    // =========================================================================
    // Test 2: Single directive, single embedder (1x1)
    // =========================================================================

    @Test
    void singleDirectiveSingleEmbedder() throws Exception {
        String text = "The quick brown fox jumps over the lazy dog. "
                + "This sentence provides enough text for the mock chunker to produce "
                + "at least a couple of ten-word chunks for embedding.";

        String jsonConfig = """
                {
                    "directives": [
                        {
                            "source_label": "body",
                            "cel_selector": "document.search_metadata.body",
                            "chunker_configs": [
                                {"config_id": "token_default"}
                            ],
                            "embedder_configs": [
                                {"config_id": "all-MiniLM-L6-v2"}
                            ]
                        }
                    ]
                }
                """;

        ProcessDataRequest request = buildRequestWithJsonConfig(text, "Simple 1x1 Doc", jsonConfig);
        ProcessDataResponse response = stub.processData(request);

        assertThat(response.getOutcome())
                .as("1x1 directive should succeed")
                .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

        assertThat(response.hasOutputDoc())
                .as("Response should contain output document")
                .isTrue();

        PipeDoc outputDoc = response.getOutputDoc();
        assertThat(outputDoc.getSearchMetadata().getSemanticResultsCount())
                .as("1 chunker x 1 embedder = 1 SemanticProcessingResult")
                .isEqualTo(1);

        SemanticProcessingResult result = outputDoc.getSearchMetadata().getSemanticResults(0);
        assertThat(result.getChunksCount())
                .as("Result should have chunks from the mock chunker")
                .isGreaterThan(0);

        assertThat(result.getEmbeddingConfigId())
                .as("Embedding config ID should be all-MiniLM-L6-v2")
                .isEqualTo("all-MiniLM-L6-v2");

        // Mock embedder returns 384-dim vectors for MiniLM
        SemanticChunk firstChunk = result.getChunks(0);
        assertThat(firstChunk.getEmbeddingInfo().getVectorCount())
                .as("MiniLM embeddings should be 384-dimensional")
                .isEqualTo(384);

        LOG.infof("1x1 test: %d chunks, vector dim=%d",
                result.getChunksCount(), firstChunk.getEmbeddingInfo().getVectorCount());
    }

    // =========================================================================
    // Test 3: Convenience fields (source_field, embedding_model, chunk_size)
    // =========================================================================

    @Test
    void convenienceFieldsWork() throws Exception {
        String text = "Convenience fields simplify configuration for common use cases. "
                + "Instead of specifying full directives with explicit chunker and embedder configs, "
                + "users can set source_field, embedding_model, and chunk_size directly. "
                + "The semantic manager converts these into an implicit directive internally.";

        String jsonConfig = """
                {
                    "source_field": "body",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                    "chunk_algorithm": "TOKEN"
                }
                """;

        ProcessDataRequest request = buildRequestWithJsonConfig(text, "Convenience Fields Doc", jsonConfig);
        ProcessDataResponse response = stub.processData(request);

        assertThat(response.getOutcome())
                .as("Convenience fields should produce a successful result")
                .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

        assertThat(response.hasOutputDoc())
                .as("Response should contain output document")
                .isTrue();

        PipeDoc outputDoc = response.getOutputDoc();
        assertThat(outputDoc.getSearchMetadata().getSemanticResultsCount())
                .as("Convenience fields should produce exactly 1 SemanticProcessingResult")
                .isEqualTo(1);

        SemanticProcessingResult result = outputDoc.getSearchMetadata().getSemanticResults(0);
        assertThat(result.getChunksCount())
                .as("Convenience fields result should have chunks")
                .isGreaterThan(0);

        assertThat(result.getSourceFieldName())
                .as("Source field should be 'body'")
                .isEqualTo("body");

        for (SemanticChunk chunk : result.getChunksList()) {
            assertThat(chunk.getEmbeddingInfo().getVectorCount())
                    .as("Chunk %d should have embedding vectors", chunk.getChunkNumber())
                    .isGreaterThan(0);
        }

        LOG.infof("Convenience fields test: %d chunks produced", result.getChunksCount());
    }

    // =========================================================================
    // Test 4: Multiple source fields (body + title)
    // =========================================================================

    @Test
    void multiSourceFields() throws Exception {
        String bodyText = "This is the body text of the document. It contains multiple sentences "
                + "that should be chunked and embedded by the semantic manager. "
                + "The body field is typically the largest source of text for embedding.";
        String titleText = "An Important Document Title for Multi-Source Testing";

        String jsonConfig = """
                {
                    "directives": [
                        {
                            "source_label": "body",
                            "cel_selector": "document.search_metadata.body",
                            "chunker_configs": [
                                {"config_id": "chunker-body"}
                            ],
                            "embedder_configs": [
                                {"config_id": "minilm"}
                            ]
                        },
                        {
                            "source_label": "title",
                            "cel_selector": "document.search_metadata.title",
                            "chunker_configs": [
                                {"config_id": "chunker-title"}
                            ],
                            "embedder_configs": [
                                {"config_id": "minilm"}
                            ]
                        }
                    ]
                }
                """;

        PipeDoc testDoc = PipeDoc.newBuilder()
                .setDocId(UUID.randomUUID().toString())
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody(bodyText)
                        .setTitle(titleText)
                        .build())
                .build();

        Struct.Builder configStruct = Struct.newBuilder();
        JsonFormat.parser().ignoringUnknownFields().merge(jsonConfig, configStruct);

        ProcessDataRequest request = ProcessDataRequest.newBuilder()
                .setDocument(testDoc)
                .setMetadata(ServiceMetadata.newBuilder()
                        .setPipelineName("integration-test")
                        .setPipeStepName("semantic-manager-step")
                        .setStreamId(UUID.randomUUID().toString())
                        .setCurrentHopNumber(1)
                        .build())
                .setConfig(ProcessConfiguration.newBuilder()
                        .setJsonConfig(configStruct.build())
                        .build())
                .build();

        ProcessDataResponse response = stub.processData(request);

        assertThat(response.getOutcome())
                .as("Multi-source directive should succeed")
                .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

        assertThat(response.hasOutputDoc())
                .as("Response should contain output document")
                .isTrue();

        PipeDoc outputDoc = response.getOutputDoc();
        assertThat(outputDoc.getSearchMetadata().getSemanticResultsCount())
                .as("Two source directives should produce 2 SemanticProcessingResults")
                .isEqualTo(2);

        // Verify both source fields are represented
        List<String> sourceFields = outputDoc.getSearchMetadata().getSemanticResultsList().stream()
                .map(SemanticProcessingResult::getSourceFieldName)
                .toList();

        assertThat(sourceFields)
                .as("Results should include both body and title source fields")
                .containsExactlyInAnyOrder("body", "title");

        // Both should have chunks with embeddings
        for (SemanticProcessingResult result : outputDoc.getSearchMetadata().getSemanticResultsList()) {
            assertThat(result.getChunksCount())
                    .as("Result for source '%s' should have chunks", result.getSourceFieldName())
                    .isGreaterThan(0);

            for (SemanticChunk chunk : result.getChunksList()) {
                assertThat(chunk.getEmbeddingInfo().getVectorCount())
                        .as("Chunk in '%s' result should have embeddings", result.getSourceFieldName())
                        .isGreaterThan(0);
            }
        }

        LOG.infof("Multi-source test: %s", sourceFields);
    }

    // =========================================================================
    // Test 5: Document with empty body
    // =========================================================================

    @Test
    void documentWithNoBody() throws Exception {
        // Document with empty body text
        String jsonConfig = """
                {
                    "directives": [
                        {
                            "source_label": "body",
                            "cel_selector": "document.search_metadata.body",
                            "chunker_configs": [
                                {"config_id": "default"}
                            ],
                            "embedder_configs": [
                                {"config_id": "minilm"}
                            ]
                        }
                    ]
                }
                """;

        PipeDoc testDoc = PipeDoc.newBuilder()
                .setDocId(UUID.randomUUID().toString())
                .setSearchMetadata(SearchMetadata.newBuilder()
                        // No body set — empty
                        .setTitle("Title Only Document")
                        .build())
                .build();

        Struct.Builder configStruct = Struct.newBuilder();
        JsonFormat.parser().ignoringUnknownFields().merge(jsonConfig, configStruct);

        ProcessDataRequest request = ProcessDataRequest.newBuilder()
                .setDocument(testDoc)
                .setMetadata(ServiceMetadata.newBuilder()
                        .setPipelineName("integration-test")
                        .setPipeStepName("semantic-manager-step")
                        .setStreamId(UUID.randomUUID().toString())
                        .setCurrentHopNumber(1)
                        .build())
                .setConfig(ProcessConfiguration.newBuilder()
                        .setJsonConfig(configStruct.build())
                        .build())
                .build();

        ProcessDataResponse response = stub.processData(request);

        // Should not crash -- either success with no results or partial/failure
        assertThat(response)
                .as("Response should not be null even for empty body")
                .isNotNull();

        assertThat(response.getOutcome())
                .as("Empty body should not cause an unhandled crash; outcome should be a valid enum value")
                .isNotNull();

        // The orchestrator should handle empty body gracefully
        if (response.hasOutputDoc()) {
            // If it returns a doc, semantic results should be 0 since body was empty
            int resultCount = response.getOutputDoc().getSearchMetadata().getSemanticResultsCount();
            LOG.infof("Empty body test: %d results (expected 0 or graceful handling)", resultCount);
        } else {
            LOG.info("Empty body test: no output doc returned (graceful handling)");
        }
    }

    // =========================================================================
    // Test 6: Court opinions throughput (5 docs x 2x2)
    // =========================================================================

    @Test
    void courtOpinionsThroughput() throws Exception {
        List<CourtOpinion> opinions = loadOpinions(5);

        assertThat(opinions)
                .as("Should load at least 5 court opinions from JSONL")
                .hasSizeGreaterThanOrEqualTo(5);

        String jsonConfig = """
                {
                    "directives": [
                        {
                            "source_label": "body",
                            "cel_selector": "document.search_metadata.body",
                            "chunker_configs": [
                                {"config_id": "sentence_10_3"},
                                {"config_id": "token_500_50"}
                            ],
                            "embedder_configs": [
                                {"config_id": "minilm"},
                                {"config_id": "mpnet"}
                            ]
                        }
                    ]
                }
                """;

        long totalStart = System.currentTimeMillis();

        for (int i = 0; i < opinions.size(); i++) {
            CourtOpinion opinion = opinions.get(i);
            long docStart = System.currentTimeMillis();

            ProcessDataRequest request = buildRequestWithJsonConfig(
                    opinion.plainText,
                    opinion.caseName != null ? opinion.caseName : "Opinion-" + i,
                    jsonConfig);

            ProcessDataResponse response = stub.processData(request);
            long docMs = System.currentTimeMillis() - docStart;

            assertThat(response.getOutcome())
                    .as("Court opinion %d ('%s') should succeed or partially succeed",
                            i, opinion.caseName)
                    .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

            assertThat(response.hasOutputDoc())
                    .as("Court opinion %d should have output doc", i)
                    .isTrue();

            int resultCount = response.getOutputDoc().getSearchMetadata().getSemanticResultsCount();
            assertThat(resultCount)
                    .as("Court opinion %d should produce 4 results (2 chunkers x 2 embedders)", i)
                    .isEqualTo(4);

            int totalChunks = response.getOutputDoc().getSearchMetadata().getSemanticResultsList().stream()
                    .mapToInt(SemanticProcessingResult::getChunksCount)
                    .sum();

            LOG.infof("  Opinion %d: '%s' — %d chars, %d results, %d total chunks, %dms",
                    i, opinion.caseName, opinion.plainText.length(),
                    resultCount, totalChunks, docMs);
        }

        long totalMs = System.currentTimeMillis() - totalStart;
        LOG.infof("Throughput test: %d opinions processed in %dms (avg %dms/doc)",
                opinions.size(), totalMs, totalMs / opinions.size());
    }

    // =========================================================================
    // Test 7: getServiceRegistration
    // =========================================================================

    @Test
    void getServiceRegistration() {
        GetServiceRegistrationRequest request = GetServiceRegistrationRequest.newBuilder().build();
        GetServiceRegistrationResponse response = stub.getServiceRegistration(request);

        assertThat(response)
                .as("Service registration response should not be null")
                .isNotNull();

        assertThat(response.getModuleName())
                .as("Module name should be 'semantic-manager'")
                .isEqualTo("semantic-manager");

        assertThat(response.getHealthCheckPassed())
                .as("Health check should pass")
                .isTrue();

        assertThat(response.getHealthCheckMessage())
                .as("Health check message should indicate readiness")
                .containsIgnoringCase("ready");

        assertThat(response.getDisplayName())
                .as("Display name should be set")
                .isNotEmpty();

        assertThat(response.getDescription())
                .as("Description should be set")
                .isNotEmpty();

        assertThat(response.getJsonConfigSchema())
                .as("JSON config schema should contain key properties")
                .contains("index_name", "directives", "source_field");

        assertThat(response.getVersion())
                .as("Version should be set")
                .isNotEmpty();

        LOG.infof("Registration: module=%s, version=%s, health=%s",
                response.getModuleName(), response.getVersion(), response.getHealthCheckMessage());
    }

    // =========================================================================
    // Helper methods
    // =========================================================================

    /**
     * Builds a ProcessDataRequest with body text and JSON config parsed into a Struct.
     */
    private ProcessDataRequest buildRequestWithJsonConfig(String bodyText, String title, String jsonConfig)
            throws Exception {
        PipeDoc testDoc = PipeDoc.newBuilder()
                .setDocId(UUID.randomUUID().toString())
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody(bodyText)
                        .setTitle(title)
                        .build())
                .build();

        Struct.Builder configStruct = Struct.newBuilder();
        JsonFormat.parser().ignoringUnknownFields().merge(jsonConfig, configStruct);

        return ProcessDataRequest.newBuilder()
                .setDocument(testDoc)
                .setMetadata(ServiceMetadata.newBuilder()
                        .setPipelineName("integration-test")
                        .setPipeStepName("semantic-manager-step")
                        .setStreamId(UUID.randomUUID().toString())
                        .setCurrentHopNumber(1)
                        .build())
                .setConfig(ProcessConfiguration.newBuilder()
                        .setJsonConfig(configStruct.build())
                        .build())
                .build();
    }

    private String loadResource(String path) {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream(path)) {
            if (is == null) {
                throw new IllegalStateException("Resource not found on classpath: " + path);
            }
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load resource: " + path, e);
        }
    }

    /**
     * Load the first N court opinions from the JSONL file.
     * Each line is a JSON object with "plain_text", "case_name", etc.
     */
    private List<CourtOpinion> loadOpinions(int count) throws IOException {
        List<CourtOpinion> opinions = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                getClass().getClassLoader().getResourceAsStream("test-data/opinions_1000.jsonl"),
                StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null && opinions.size() < count) {
                CourtOpinion opinion = new CourtOpinion();
                opinion.plainText = extractJsonField(line, "plain_text");
                opinion.caseName = extractJsonField(line, "case_name");
                if (opinion.plainText != null && !opinion.plainText.isEmpty()) {
                    opinions.add(opinion);
                }
            }
        }
        return opinions;
    }

    /**
     * Extract a string field from a JSON line using simple string parsing.
     * Avoids adding Jackson as a test dependency just for JSONL parsing.
     */
    private static String extractJsonField(String json, String fieldName) {
        String quotedKey = "\"" + fieldName + "\"";
        int keyIdx = json.indexOf(quotedKey);
        if (keyIdx < 0) {
            return null;
        }
        int valueStart = keyIdx + quotedKey.length();
        while (valueStart < json.length() && json.charAt(valueStart) == ' ') {
            valueStart++;
        }
        if (valueStart >= json.length() || json.charAt(valueStart) != ':') {
            return null;
        }
        valueStart++;
        while (valueStart < json.length() && json.charAt(valueStart) == ' ') {
            valueStart++;
        }
        if (valueStart >= json.length() || json.charAt(valueStart) != '"') {
            return null;
        }
        valueStart++;
        StringBuilder sb = new StringBuilder();
        for (int i = valueStart; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '\\' && i + 1 < json.length()) {
                char next = json.charAt(i + 1);
                switch (next) {
                    case '"': sb.append('"'); i++; break;
                    case '\\': sb.append('\\'); i++; break;
                    case 'n': sb.append('\n'); i++; break;
                    case 'r': sb.append('\r'); i++; break;
                    case 't': sb.append('\t'); i++; break;
                    default: sb.append(c); break;
                }
            } else if (c == '"') {
                break;
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    private static class CourtOpinion {
        String plainText;
        String caseName;
    }
}
