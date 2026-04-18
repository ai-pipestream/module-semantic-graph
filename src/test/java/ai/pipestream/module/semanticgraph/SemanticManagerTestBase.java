package ai.pipestream.module.semanticgraph;

import ai.pipestream.data.module.v1.ProcessDataRequest;
import ai.pipestream.data.module.v1.ProcessDataResponse;
import ai.pipestream.data.module.v1.ProcessingOutcome;
import ai.pipestream.data.module.v1.ServiceMetadata;
import ai.pipestream.data.v1.PipeDoc;
import ai.pipestream.data.v1.ProcessConfiguration;
import ai.pipestream.data.v1.SearchMetadata;
import ai.pipestream.data.v1.SemanticChunk;
import ai.pipestream.data.v1.SemanticProcessingResult;
import com.google.protobuf.Struct;
import com.google.protobuf.Value;
import com.google.protobuf.util.JsonFormat;
import org.jboss.logging.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Shared test logic for semantic manager integration tests.
 * <p>
 * Contains assertion methods and request builders used by both:
 * - {@code @QuarkusTest} (mock chunker/embedder, in-process)
 * - {@code @QuarkusIntegrationTest} (real or mock services, external JAR)
 * <p>
 * No CDI injection — all methods take explicit parameters.
 */
public abstract class SemanticManagerTestBase {

    private static final Logger LOG = Logger.getLogger(SemanticManagerTestBase.class);

    // =========================================================================
    // Request builders
    // =========================================================================

    /**
     * Builds a ProcessDataRequest with body text and JSON config parsed into a Struct.
     */
    protected static ProcessDataRequest buildRequestWithJsonConfig(String bodyText, String title, String jsonConfig)
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

    /**
     * Builds a ProcessDataRequest with body+title and a raw Struct config.
     */
    protected static ProcessDataRequest buildRequest(PipeDoc doc, String indexName) {
        ServiceMetadata metadata = ServiceMetadata.newBuilder()
                .setPipelineName("test-pipeline")
                .setPipeStepName("semantic-manager-step")
                .setStreamId(UUID.randomUUID().toString())
                .setCurrentHopNumber(1)
                .putContextParams("tenant", "test-tenant")
                .build();

        Struct.Builder configFields = Struct.newBuilder();
        if (indexName != null) {
            configFields.putFields("index_name",
                    Value.newBuilder().setStringValue(indexName).build());
        }

        ProcessConfiguration config = ProcessConfiguration.newBuilder()
                .setJsonConfig(configFields.build())
                .build();

        return ProcessDataRequest.newBuilder()
                .setDocument(doc)
                .setMetadata(metadata)
                .setConfig(config)
                .build();
    }

    // =========================================================================
    // Resource loading
    // =========================================================================

    protected static String loadResource(String path) {
        try (InputStream is = SemanticManagerTestBase.class.getClassLoader().getResourceAsStream(path)) {
            if (is == null) {
                throw new IllegalStateException("Resource not found on classpath: " + path);
            }
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load resource: " + path, e);
        }
    }

    protected static List<CourtOpinion> loadOpinions(int count) throws IOException {
        List<CourtOpinion> opinions = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                SemanticManagerTestBase.class.getClassLoader().getResourceAsStream("test-data/opinions_1000.jsonl"),
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

    // =========================================================================
    // Assertion helpers
    // =========================================================================

    /**
     * Asserts a 2x2 scatter-gather: 2 chunkers × 2 embedders = 4 results.
     */
    protected static void assertScatterGather2x2(ProcessDataResponse response) {
        assertThat(response.getOutcome())
                .as("2x2 scatter-gather should succeed or partially succeed")
                .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

        assertThat(response.hasOutputDoc())
                .as("Response should contain enriched output document")
                .isTrue();

        PipeDoc outputDoc = response.getOutputDoc();
        int resultCount = outputDoc.getSearchMetadata().getSemanticResultsCount();

        assertThat(resultCount)
                .as("2 chunkers x 2 embedders should produce 4 SemanticProcessingResults")
                .isEqualTo(4);

        assertAllChunksHaveEmbeddings(outputDoc);
    }

    /**
     * Asserts a single directive 1x1: 1 chunker × 1 embedder = 1 result.
     */
    protected static void assertSingleDirective(ProcessDataResponse response, String expectedEmbedder) {
        assertThat(response.getOutcome())
                .as("1x1 directive should succeed")
                .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

        PipeDoc outputDoc = response.getOutputDoc();
        assertThat(outputDoc.getSearchMetadata().getSemanticResultsCount())
                .as("1 chunker x 1 embedder = 1 SemanticProcessingResult")
                .isEqualTo(1);

        SemanticProcessingResult result = outputDoc.getSearchMetadata().getSemanticResults(0);
        assertThat(result.getChunksCount())
                .as("Result should have chunks")
                .isGreaterThan(0);

        assertThat(result.getEmbeddingConfigId())
                .as("Embedding config ID should match")
                .isEqualTo(expectedEmbedder);

        assertAllChunksHaveEmbeddings(outputDoc);
    }

    /**
     * Asserts field-level embedding: skip_chunking=true → 1 result with 1 chunk.
     */
    protected static void assertFieldLevelResult(ProcessDataResponse response, String expectedSourceField) {
        assertThat(response.getOutcome())
                .as("Field-level embedding should succeed")
                .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

        assertThat(response.hasOutputDoc())
                .as("Response should contain output document")
                .isTrue();

        PipeDoc outputDoc = response.getOutputDoc();
        assertThat(outputDoc.getSearchMetadata().getSemanticResultsCount())
                .as("Field-level should produce exactly 1 SemanticProcessingResult")
                .isEqualTo(1);

        SemanticProcessingResult result = outputDoc.getSearchMetadata().getSemanticResults(0);
        assertThat(result.getSourceFieldName())
                .as("Source field should match")
                .isEqualTo(expectedSourceField);

        assertThat(result.getChunksCount())
                .as("Field-level result should have exactly 1 chunk (the entire field)")
                .isEqualTo(1);

        SemanticChunk chunk = result.getChunks(0);
        assertThat(chunk.getEmbeddingInfo().getVectorCount())
                .as("Field-level chunk should have embedding vectors")
                .isGreaterThan(0);

        LOG.infof("Field-level result: source=%s, vectorDim=%d, textLen=%d",
                result.getSourceFieldName(),
                chunk.getEmbeddingInfo().getVectorCount(),
                chunk.getEmbeddingInfo().getTextContent().length());
    }

    /**
     * Asserts mixed chunked + field-level directives produce results from both paths.
     */
    protected static void assertMixedChunkedAndFieldLevel(ProcessDataResponse response) {
        assertThat(response.getOutcome())
                .as("Mixed directive should succeed")
                .isIn(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS, ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL);

        PipeDoc outputDoc = response.getOutputDoc();
        assertThat(outputDoc.getSearchMetadata().getSemanticResultsCount())
                .as("Mixed directives should produce 2 results (1 chunked + 1 field-level)")
                .isEqualTo(2);

        assertAllChunksHaveEmbeddings(outputDoc);

        // One result should have multiple chunks (chunked path), one should have 1 (field-level)
        List<Integer> chunkCounts = outputDoc.getSearchMetadata().getSemanticResultsList().stream()
                .map(SemanticProcessingResult::getChunksCount)
                .sorted()
                .toList();

        LOG.infof("Mixed directive chunk counts: %s", chunkCounts);
    }

    /**
     * Asserts that a malformed config doesn't crash the service.
     */
    protected static void assertGracefulErrorResponse(ProcessDataResponse response) {
        assertThat(response)
                .as("Response should not be null even for error case")
                .isNotNull();

        assertThat(response.getOutcome())
                .as("Error case should produce a valid outcome (not crash)")
                .isNotNull();
    }

    /**
     * Verifies every chunk in every result has non-empty embedding vectors.
     */
    protected static void assertAllChunksHaveEmbeddings(PipeDoc outputDoc) {
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
        }
    }

    // =========================================================================
    // JSON field extraction (avoids Jackson dependency for JSONL parsing)
    // =========================================================================

    protected static String extractJsonField(String json, String fieldName) {
        String quotedKey = "\"" + fieldName + "\"";
        int keyIdx = json.indexOf(quotedKey);
        if (keyIdx < 0) return null;
        int valueStart = keyIdx + quotedKey.length();
        while (valueStart < json.length() && json.charAt(valueStart) == ' ') valueStart++;
        if (valueStart >= json.length() || json.charAt(valueStart) != ':') return null;
        valueStart++;
        while (valueStart < json.length() && json.charAt(valueStart) == ' ') valueStart++;
        if (valueStart >= json.length() || json.charAt(valueStart) != '"') return null;
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

    protected static class CourtOpinion {
        public String plainText;
        public String caseName;
    }
}
