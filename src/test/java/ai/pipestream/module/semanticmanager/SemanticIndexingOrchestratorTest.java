package ai.pipestream.module.semanticmanager;

import ai.pipestream.data.v1.*;
import ai.pipestream.module.semanticmanager.config.SemanticManagerOptions;
import ai.pipestream.module.semanticmanager.service.ChunkerStreamClient;
import ai.pipestream.module.semanticmanager.service.EmbedderStreamClient;
import ai.pipestream.module.semanticmanager.service.SemanticIndexingOrchestrator;
import ai.pipestream.module.semanticmanager.service.VectorSetResolver;
import ai.pipestream.opensearch.v1.VectorSet;
import ai.pipestream.semantic.v1.StreamChunksResponse;
import ai.pipestream.semantic.v1.StreamEmbeddingsRequest;
import ai.pipestream.semantic.v1.StreamEmbeddingsResponse;
import com.google.protobuf.Struct;
import com.google.protobuf.Value;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class SemanticIndexingOrchestratorTest {

    private SemanticIndexingOrchestrator orchestrator;
    private VectorSetResolver vectorSetResolver;
    private ChunkerStreamClient chunkerStreamClient;
    private EmbedderStreamClient embedderStreamClient;

    @BeforeEach
    void setUp() throws Exception {
        orchestrator = new SemanticIndexingOrchestrator();
        vectorSetResolver = mock(VectorSetResolver.class);
        chunkerStreamClient = mock(ChunkerStreamClient.class);
        embedderStreamClient = mock(EmbedderStreamClient.class);

        setField(orchestrator, "vectorSetResolver", vectorSetResolver);
        setField(orchestrator, "chunkerStreamClient", chunkerStreamClient);
        setField(orchestrator, "embedderStreamClient", embedderStreamClient);
    }

    // =========================================================================
    // VectorSetService fallback tests
    // =========================================================================

    @Test
    void testFallback_noVectorSets_returnsUnchangedDoc() {
        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("test-doc-1")
                .setSearchMetadata(SearchMetadata.newBuilder().setBody("Hello world").build())
                .build();

        when(vectorSetResolver.resolveVectorSets(anyString()))
                .thenReturn(Uni.createFrom().item(List.of()));

        SemanticManagerOptions options = new SemanticManagerOptions("test-index", null, 4, 8, null);

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        assertEquals("test-doc-1", result.getDocId());
        assertEquals(0, result.getSearchMetadata().getSemanticResultsCount());
    }

    @Test
    void testFallback_withVectorSets_producesResults() {
        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("test-doc-2")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("This is a test document.")
                        .build())
                .build();

        VectorSet vs1 = VectorSet.newBuilder()
                .setId("vs-1").setName("body-minilm")
                .setChunkerConfigId("chunker-a")
                .setEmbeddingModelConfigId("all-MiniLM-L6-v2")
                .setIndexName("test-index").setFieldName("embeddings")
                .setResultSetName("body_minilm_results").setSourceField("body")
                .build();

        when(vectorSetResolver.resolveVectorSets("test-index"))
                .thenReturn(Uni.createFrom().item(List.of(vs1)));

        setupMockChunkerAndEmbedder();

        SemanticManagerOptions options = new SemanticManagerOptions("test-index", null, 4, 8, null);

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        assertEquals(1, result.getSearchMetadata().getSemanticResultsCount());
        assertEquals("chunker-a", result.getSearchMetadata().getSemanticResults(0).getChunkConfigId());
    }

    @Test
    void testFallback_deduplicatesChunking() {
        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("test-doc-3")
                .setSearchMetadata(SearchMetadata.newBuilder().setBody("Text").build())
                .build();

        VectorSet vs1 = VectorSet.newBuilder()
                .setId("vs-1").setName("vs1")
                .setChunkerConfigId("same-chunker")
                .setEmbeddingModelConfigId("model-a")
                .setIndexName("test-index").setFieldName("f").setResultSetName("r1").setSourceField("body")
                .build();

        VectorSet vs2 = VectorSet.newBuilder()
                .setId("vs-2").setName("vs2")
                .setChunkerConfigId("same-chunker")
                .setEmbeddingModelConfigId("model-b")
                .setIndexName("test-index").setFieldName("f").setResultSetName("r2").setSourceField("body")
                .build();

        when(vectorSetResolver.resolveVectorSets("test-index"))
                .thenReturn(Uni.createFrom().item(List.of(vs1, vs2)));

        setupMockChunkerAndEmbedder();

        SemanticManagerOptions options = new SemanticManagerOptions("test-index", null, 4, 8, null);

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        assertEquals(2, result.getSearchMetadata().getSemanticResultsCount());
        verify(chunkerStreamClient, times(1)).streamChunks(any());
        verify(embedderStreamClient, times(2)).streamEmbeddings(any());
    }

    // =========================================================================
    // Directive-based orchestration tests
    // =========================================================================

    @Test
    void testDirectives_cartesianProduct() {
        // 1 directive with 2 chunkers × 2 embedders = 4 results
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder()
                        .setConfigId("sentence_v1")
                        .setConfig(Struct.newBuilder()
                                .putFields("algorithm", Value.newBuilder().setStringValue("sentence").build())
                                .build())
                        .build())
                .addChunkerConfigs(NamedChunkerConfig.newBuilder()
                        .setConfigId("token_v1")
                        .setConfig(Struct.newBuilder()
                                .putFields("algorithm", Value.newBuilder().setStringValue("token").build())
                                .build())
                        .build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder()
                        .setConfigId("minilm")
                        .build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder()
                        .setConfigId("mpnet")
                        .build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("directive-doc-1")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Document text for cartesian product test.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive)
                                .build())
                        .build())
                .build();

        setupMockChunkerAndEmbedder();

        SemanticManagerOptions options = new SemanticManagerOptions();

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        // 2 chunkers × 2 embedders = 4 results
        assertEquals(4, result.getSearchMetadata().getSemanticResultsCount());
        // But chunker should only be called TWICE (one per chunker config)
        verify(chunkerStreamClient, times(2)).streamChunks(any());
        // Embedder called 4 times (cartesian product)
        verify(embedderStreamClient, times(4)).streamEmbeddings(any());
    }

    @Test
    void testDirectives_fieldNameTemplate() {
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .setFieldNameTemplate("{source_label}_{chunker_id}_{embedder_id}")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("sent").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("mini").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("template-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Some text.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        setupMockChunkerAndEmbedder();

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        assertEquals(1, result.getSearchMetadata().getSemanticResultsCount());
        assertEquals("body_sent_mini",
                result.getSearchMetadata().getSemanticResults(0).getResultSetName());
    }

    @Test
    void testDirectives_usedOverVectorSetService() {
        // When directives are present, VectorSetService should NOT be called
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("c1").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("e1").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("priority-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Priority test.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        setupMockChunkerAndEmbedder();

        orchestrator.orchestrate(inputDoc, new SemanticManagerOptions("some-index", null, 4, 8, null), "node-1")
                .await().indefinitely();

        // VectorSetResolver should never be called
        verify(vectorSetResolver, never()).resolveVectorSets(anyString());
    }

    @Test
    void testDirectives_chunkerDeduplication() {
        // Two directives using the same chunker + source = chunk only once
        VectorDirective d1 = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("shared_chunker").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("emb_a").build())
                .build();

        VectorDirective d2 = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("shared_chunker").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("emb_b").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("dedup-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Dedup test text.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(d1).addDirectives(d2).build())
                        .build())
                .build();

        setupMockChunkerAndEmbedder();

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        // 2 results (one per embedder)
        assertEquals(2, result.getSearchMetadata().getSemanticResultsCount());
        // But chunker called only ONCE (deduplication)
        verify(chunkerStreamClient, times(1)).streamChunks(any());
        verify(embedderStreamClient, times(2)).streamEmbeddings(any());
    }

    @Test
    void testDirectives_partialEmbedderFailure() {
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("c1").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("ok_emb").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("fail_emb").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("partial-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Partial failure test.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        StreamChunksResponse chunk = StreamChunksResponse.newBuilder()
                .setChunkId("c-0001").setChunkNumber(0).setTextContent("text")
                .setChunkConfigId("c1").setSourceFieldName("body").setIsLast(true)
                .build();

        when(chunkerStreamClient.streamChunks(any()))
                .thenReturn(Multi.createFrom().items(chunk));

        when(embedderStreamClient.streamEmbeddings(any()))
                .thenAnswer(invocation -> {
                    @SuppressWarnings("unchecked")
                    Multi<StreamEmbeddingsRequest> reqs = (Multi<StreamEmbeddingsRequest>) invocation.getArgument(0);
                    return reqs.map(req -> {
                        if ("fail_emb".equals(req.getEmbeddingModelId())) {
                            throw new RuntimeException("Embedder down");
                        }
                        return StreamEmbeddingsResponse.newBuilder()
                                .setRequestId(req.getRequestId()).setDocId(req.getDocId())
                                .setChunkId(req.getChunkId()).setChunkConfigId(req.getChunkConfigId())
                                .setEmbeddingModelId(req.getEmbeddingModelId())
                                .addVector(0.5f).setSuccess(true).build();
                    });
                });

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        // At least 1 result (partial failure tolerance)
        assertTrue(result.getSearchMetadata().getSemanticResultsCount() >= 1);
    }

    // =========================================================================
    // Field-level (no-chunk) orchestration tests
    // =========================================================================

    @Test
    void testFieldLevel_noChunkerConfigs_skipsChunkerCallsEmbedder() {
        // Directive with 0 chunker_configs and 1 embedder config → field-level path
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                // No chunker configs → field-level
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder()
                        .setConfigId("all-MiniLM-L6-v2")
                        .build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("field-level-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Full body text for field-level embedding.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        // Only set up the embedder mock (chunker should NOT be called)
        when(embedderStreamClient.streamEmbeddings(any()))
                .thenAnswer(invocation -> {
                    @SuppressWarnings("unchecked")
                    Multi<StreamEmbeddingsRequest> reqs = (Multi<StreamEmbeddingsRequest>) invocation.getArgument(0);
                    return reqs.map(req -> StreamEmbeddingsResponse.newBuilder()
                            .setRequestId(req.getRequestId()).setDocId(req.getDocId())
                            .setChunkId(req.getChunkId()).setChunkConfigId(req.getChunkConfigId())
                            .setEmbeddingModelId(req.getEmbeddingModelId())
                            .addVector(0.5f).addVector(0.6f).addVector(0.7f)
                            .setSuccess(true).build());
                });

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        // Chunker should NOT have been called
        verify(chunkerStreamClient, never()).streamChunks(any());
        // Embedder should have been called exactly once
        verify(embedderStreamClient, times(1)).streamEmbeddings(any());

        // Should have 1 semantic result
        assertEquals(1, result.getSearchMetadata().getSemanticResultsCount());
        SemanticProcessingResult spr = result.getSearchMetadata().getSemanticResults(0);
        assertEquals("body", spr.getSourceFieldName());
        assertEquals("all-MiniLM-L6-v2", spr.getEmbeddingConfigId());

        // Should have exactly 1 chunk (the full field)
        assertEquals(1, spr.getChunksCount());
        SemanticChunk chunk = spr.getChunks(0);
        assertEquals("field-level-doc_body_full", chunk.getChunkId());
        assertEquals(0, chunk.getChunkNumber());
        assertEquals("Full body text for field-level embedding.", chunk.getEmbeddingInfo().getTextContent());
        assertEquals(3, chunk.getEmbeddingInfo().getVectorCount());
    }

    @Test
    void testFieldLevel_multipleEmbeddersNoChunker() {
        // 1 directive with 0 chunkers × 2 embedders = 2 field-level results
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("model-a").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("model-b").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("multi-emb-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Text for multi-embedder field-level test.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        when(embedderStreamClient.streamEmbeddings(any()))
                .thenAnswer(invocation -> {
                    @SuppressWarnings("unchecked")
                    Multi<StreamEmbeddingsRequest> reqs = (Multi<StreamEmbeddingsRequest>) invocation.getArgument(0);
                    return reqs.map(req -> StreamEmbeddingsResponse.newBuilder()
                            .setRequestId(req.getRequestId()).setDocId(req.getDocId())
                            .setChunkId(req.getChunkId()).setChunkConfigId(req.getChunkConfigId())
                            .setEmbeddingModelId(req.getEmbeddingModelId())
                            .addVector(0.1f).setSuccess(true).build());
                });

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        verify(chunkerStreamClient, never()).streamChunks(any());
        verify(embedderStreamClient, times(2)).streamEmbeddings(any());
        assertEquals(2, result.getSearchMetadata().getSemanticResultsCount());
    }

    @Test
    void testFieldLevel_mixedDirectives_bothPathsRun() {
        // Directive 1: has chunker (standard path)
        VectorDirective d1 = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("chunker-1").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("emb-a").build())
                .build();

        // Directive 2: no chunker (field-level path)
        VectorDirective d2 = VectorDirective.newBuilder()
                .setSourceLabel("title")
                .setCelSelector("document.search_metadata.title")
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("emb-b").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("mixed-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Body text.")
                        .setTitle("Title text.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(d1).addDirectives(d2).build())
                        .build())
                .build();

        setupMockChunkerAndEmbedder();

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        // d1 uses chunker, d2 skips chunker
        verify(chunkerStreamClient, times(1)).streamChunks(any());
        verify(embedderStreamClient, times(2)).streamEmbeddings(any());

        // 2 total results (1 chunked + 1 field-level)
        assertEquals(2, result.getSearchMetadata().getSemanticResultsCount());
    }

    // =========================================================================
    // Convenience fields tests
    // =========================================================================

    @Test
    void testConvenienceFields_skipChunking_producesFieldLevelResult() {
        // Use convenience fields: skip_chunking=true, source_field=body, model=mini
        SemanticManagerOptions options = new SemanticManagerOptions(
                "test-index", null, 4, 8, null,
                "body", null, null, null, "all-MiniLM-L6-v2", true, null);

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("convenience-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Body text for convenience field test.")
                        .build())
                .build();

        when(embedderStreamClient.streamEmbeddings(any()))
                .thenAnswer(invocation -> {
                    @SuppressWarnings("unchecked")
                    Multi<StreamEmbeddingsRequest> reqs = (Multi<StreamEmbeddingsRequest>) invocation.getArgument(0);
                    return reqs.map(req -> StreamEmbeddingsResponse.newBuilder()
                            .setRequestId(req.getRequestId()).setDocId(req.getDocId())
                            .setChunkId(req.getChunkId()).setChunkConfigId(req.getChunkConfigId())
                            .setEmbeddingModelId(req.getEmbeddingModelId())
                            .addVector(0.4f).addVector(0.5f)
                            .setSuccess(true).build());
                });

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        // skip_chunking=true → chunker should not be called
        verify(chunkerStreamClient, never()).streamChunks(any());
        verify(embedderStreamClient, times(1)).streamEmbeddings(any());

        assertEquals(1, result.getSearchMetadata().getSemanticResultsCount());
        SemanticProcessingResult spr = result.getSearchMetadata().getSemanticResults(0);
        assertEquals("body", spr.getSourceFieldName());
        assertEquals("all-MiniLM-L6-v2", spr.getEmbeddingConfigId());
        assertEquals(1, spr.getChunksCount());
    }

    @Test
    void testConvenienceFields_withChunking_producesChunkedResult() {
        // Convenience fields with chunking (skip_chunking defaults to false)
        SemanticManagerOptions options = new SemanticManagerOptions(
                "test-index", null, 4, 8, null,
                "body", 200, 20, "SENTENCE", "all-MiniLM-L6-v2", false, null);

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("convenience-chunk-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Body text for chunked convenience test.")
                        .build())
                .build();

        setupMockChunkerAndEmbedder();

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        // With chunking → chunker should be called
        verify(chunkerStreamClient, times(1)).streamChunks(any());
        verify(embedderStreamClient, times(1)).streamEmbeddings(any());
        assertEquals(1, result.getSearchMetadata().getSemanticResultsCount());
    }

    @Test
    void testConvenienceFields_explicitDirectivesTakePriority() {
        // Both explicit directives AND convenience fields set — directives should win
        var directiveConfig = new ai.pipestream.module.semanticmanager.config.DirectiveConfig(
                "body", "document.search_metadata.body",
                List.of(new ai.pipestream.module.semanticmanager.config.DirectiveConfig.NamedConfig("chunker-x", null)),
                List.of(new ai.pipestream.module.semanticmanager.config.DirectiveConfig.NamedConfig("emb-x", null)),
                null);

        SemanticManagerOptions options = new SemanticManagerOptions(
                "test-index", null, 4, 8, List.of(directiveConfig),
                "title", null, null, null, "some-other-model", true, null);

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("priority-doc-2")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Body text.")
                        .setTitle("Title text.")
                        .build())
                .build();

        setupMockChunkerAndEmbedder();

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        // Explicit directives used — should chunk 'body' with chunker-x, not skip chunking on 'title'
        verify(chunkerStreamClient, times(1)).streamChunks(any());
        assertEquals(1, result.getSearchMetadata().getSemanticResultsCount());
        assertEquals("body", result.getSearchMetadata().getSemanticResults(0).getSourceFieldName());
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private void setupMockChunkerAndEmbedder() {
        StreamChunksResponse chunk = StreamChunksResponse.newBuilder()
                .setChunkId("chunk-0001").setChunkNumber(0).setTextContent("chunk text")
                .setChunkConfigId("default").setSourceFieldName("body").setIsLast(true)
                .build();

        when(chunkerStreamClient.streamChunks(any()))
                .thenReturn(Multi.createFrom().items(chunk));

        when(embedderStreamClient.streamEmbeddings(any()))
                .thenAnswer(invocation -> {
                    @SuppressWarnings("unchecked")
                    Multi<StreamEmbeddingsRequest> reqs = (Multi<StreamEmbeddingsRequest>) invocation.getArgument(0);
                    return reqs.map(req -> StreamEmbeddingsResponse.newBuilder()
                            .setRequestId(req.getRequestId()).setDocId(req.getDocId())
                            .setChunkId(req.getChunkId()).setChunkConfigId(req.getChunkConfigId())
                            .setEmbeddingModelId(req.getEmbeddingModelId())
                            .addVector(0.1f).addVector(0.2f).addVector(0.3f)
                            .setSuccess(true).build());
                });
    }

    private void setField(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }
}
