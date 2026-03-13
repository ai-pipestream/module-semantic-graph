package ai.pipestream.module.semanticmanager;

import ai.pipestream.data.v1.PipeDoc;
import ai.pipestream.data.v1.SearchMetadata;
import ai.pipestream.data.v1.SemanticProcessingResult;
import ai.pipestream.module.semanticmanager.config.SemanticManagerOptions;
import ai.pipestream.module.semanticmanager.service.ChunkerStreamClient;
import ai.pipestream.module.semanticmanager.service.EmbedderStreamClient;
import ai.pipestream.module.semanticmanager.service.SemanticIndexingOrchestrator;
import ai.pipestream.module.semanticmanager.service.VectorSetResolver;
import ai.pipestream.opensearch.v1.VectorSet;
import ai.pipestream.semantic.v1.StreamChunksResponse;
import ai.pipestream.semantic.v1.StreamEmbeddingsRequest;
import ai.pipestream.semantic.v1.StreamEmbeddingsResponse;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * Unit test for the SemanticIndexingOrchestrator with mock services.
 */
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

        // Inject mocks via reflection
        setField(orchestrator, "vectorSetResolver", vectorSetResolver);
        setField(orchestrator, "chunkerStreamClient", chunkerStreamClient);
        setField(orchestrator, "embedderStreamClient", embedderStreamClient);
    }

    @Test
    void testOrchestrate_noVectorSets_returnsUnchangedDoc() {
        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("test-doc-1")
                .setSearchMetadata(SearchMetadata.newBuilder().setBody("Hello world").build())
                .build();

        when(vectorSetResolver.resolveVectorSets(anyString()))
                .thenReturn(Uni.createFrom().item(List.of()));

        SemanticManagerOptions options = new SemanticManagerOptions("test-index", null, 4, 8);

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        assertEquals("test-doc-1", result.getDocId());
        assertEquals(0, result.getSearchMetadata().getSemanticResultsCount());
    }

    @Test
    void testOrchestrate_withVectorSets_producesResults() {
        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("test-doc-2")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("This is a test document with enough text to chunk properly.")
                        .build())
                .build();

        VectorSet vs1 = VectorSet.newBuilder()
                .setId("vs-1")
                .setName("body-minilm")
                .setChunkerConfigId("chunker-a")
                .setEmbeddingModelConfigId("all-MiniLM-L6-v2")
                .setIndexName("test-index")
                .setFieldName("embeddings")
                .setResultSetName("body_minilm_results")
                .setSourceField("body")
                .setVectorDimensions(384)
                .build();

        VectorSet vs2 = VectorSet.newBuilder()
                .setId("vs-2")
                .setName("body-mpnet")
                .setChunkerConfigId("chunker-a")
                .setEmbeddingModelConfigId("all-mpnet-base-v2")
                .setIndexName("test-index")
                .setFieldName("embeddings")
                .setResultSetName("body_mpnet_results")
                .setSourceField("body")
                .setVectorDimensions(768)
                .build();

        when(vectorSetResolver.resolveVectorSets("test-index"))
                .thenReturn(Uni.createFrom().item(List.of(vs1, vs2)));

        // Mock chunker: return 2 chunks
        StreamChunksResponse chunk1 = StreamChunksResponse.newBuilder()
                .setRequestId("req-1")
                .setDocId("test-doc-2")
                .setChunkId("chunk-0001")
                .setChunkNumber(0)
                .setTextContent("This is a test document")
                .setStartOffset(0)
                .setEndOffset(23)
                .setChunkConfigId("chunker-a")
                .setSourceFieldName("body")
                .setIsLast(false)
                .build();

        StreamChunksResponse chunk2 = StreamChunksResponse.newBuilder()
                .setRequestId("req-1")
                .setDocId("test-doc-2")
                .setChunkId("chunk-0002")
                .setChunkNumber(1)
                .setTextContent("with enough text to chunk properly.")
                .setStartOffset(24)
                .setEndOffset(59)
                .setChunkConfigId("chunker-a")
                .setSourceFieldName("body")
                .setIsLast(true)
                .build();

        when(chunkerStreamClient.streamChunks(any()))
                .thenReturn(Multi.createFrom().items(chunk1, chunk2));

        // Mock embedder: return vectors for each chunk
        when(embedderStreamClient.streamEmbeddings(any()))
                .thenAnswer(invocation -> {
                    @SuppressWarnings("unchecked")
                    Multi<StreamEmbeddingsRequest> requests = (Multi<StreamEmbeddingsRequest>) invocation.getArgument(0);
                    return requests.map(req -> StreamEmbeddingsResponse.newBuilder()
                            .setRequestId(req.getRequestId())
                            .setDocId(req.getDocId())
                            .setChunkId(req.getChunkId())
                            .setChunkConfigId(req.getChunkConfigId())
                            .setEmbeddingModelId(req.getEmbeddingModelId())
                            .addVector(0.1f)
                            .addVector(0.2f)
                            .addVector(0.3f)
                            .setSuccess(true)
                            .build());
                });

        SemanticManagerOptions options = new SemanticManagerOptions("test-index", null, 4, 8);

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        assertEquals("test-doc-2", result.getDocId());
        // Should have 2 SemanticProcessingResults (one per VectorSet, same chunker group)
        assertEquals(2, result.getSearchMetadata().getSemanticResultsCount());

        for (SemanticProcessingResult spr : result.getSearchMetadata().getSemanticResultsList()) {
            assertEquals("chunker-a", spr.getChunkConfigId());
            assertEquals("body", spr.getSourceFieldName());
            assertEquals(2, spr.getChunksCount());
            // Each chunk should have a 3-dim vector
            assertTrue(spr.getChunks(0).getEmbeddingInfo().getVectorCount() > 0);
        }
    }

    @Test
    void testOrchestrate_partialEmbedderFailure_returnsPartialResults() {
        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("test-doc-3")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Some text for partial failure test.")
                        .build())
                .build();

        VectorSet vs1 = VectorSet.newBuilder()
                .setId("vs-1")
                .setName("ok-embedder")
                .setChunkerConfigId("chunker-a")
                .setEmbeddingModelConfigId("model-ok")
                .setIndexName("test-index")
                .setFieldName("embeddings")
                .setResultSetName("ok_results")
                .setSourceField("body")
                .build();

        VectorSet vs2 = VectorSet.newBuilder()
                .setId("vs-2")
                .setName("failing-embedder")
                .setChunkerConfigId("chunker-a")
                .setEmbeddingModelConfigId("model-fail")
                .setIndexName("test-index")
                .setFieldName("embeddings")
                .setResultSetName("fail_results")
                .setSourceField("body")
                .build();

        when(vectorSetResolver.resolveVectorSets("test-index"))
                .thenReturn(Uni.createFrom().item(List.of(vs1, vs2)));

        StreamChunksResponse chunk = StreamChunksResponse.newBuilder()
                .setChunkId("chunk-0001")
                .setChunkNumber(0)
                .setTextContent("Some text")
                .setChunkConfigId("chunker-a")
                .setSourceFieldName("body")
                .setIsLast(true)
                .build();

        when(chunkerStreamClient.streamChunks(any()))
                .thenReturn(Multi.createFrom().items(chunk));

        // First embedder call succeeds, second fails
        when(embedderStreamClient.streamEmbeddings(any()))
                .thenAnswer(invocation -> {
                    @SuppressWarnings("unchecked")
                    Multi<StreamEmbeddingsRequest> requests = (Multi<StreamEmbeddingsRequest>) invocation.getArgument(0);
                    return requests.map(req -> {
                        if ("model-fail".equals(req.getEmbeddingModelId())) {
                            throw new RuntimeException("Embedder unavailable");
                        }
                        return StreamEmbeddingsResponse.newBuilder()
                                .setRequestId(req.getRequestId())
                                .setDocId(req.getDocId())
                                .setChunkId(req.getChunkId())
                                .setChunkConfigId(req.getChunkConfigId())
                                .setEmbeddingModelId(req.getEmbeddingModelId())
                                .addVector(0.5f)
                                .setSuccess(true)
                                .build();
                    });
                });

        SemanticManagerOptions options = new SemanticManagerOptions("test-index", null, 4, 8);

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        // Should have at least 1 result (partial failure tolerance)
        assertTrue(result.getSearchMetadata().getSemanticResultsCount() >= 1);
    }

    @Test
    void testOrchestrate_deduplicatesChunking() {
        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("test-doc-4")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Text for deduplication test.")
                        .build())
                .build();

        // Two VectorSets with the SAME chunker config + source field
        // Should result in only ONE chunker call, but TWO embedder calls
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

        StreamChunksResponse chunk = StreamChunksResponse.newBuilder()
                .setChunkId("c1").setChunkNumber(0).setTextContent("Text")
                .setChunkConfigId("same-chunker").setSourceFieldName("body").setIsLast(true)
                .build();

        when(chunkerStreamClient.streamChunks(any()))
                .thenReturn(Multi.createFrom().items(chunk));

        when(embedderStreamClient.streamEmbeddings(any()))
                .thenAnswer(invocation -> {
                    @SuppressWarnings("unchecked")
                    Multi<StreamEmbeddingsRequest> requests = (Multi<StreamEmbeddingsRequest>) invocation.getArgument(0);
                    return requests.map(req -> StreamEmbeddingsResponse.newBuilder()
                            .setRequestId(req.getRequestId()).setDocId(req.getDocId())
                            .setChunkId(req.getChunkId()).setChunkConfigId(req.getChunkConfigId())
                            .setEmbeddingModelId(req.getEmbeddingModelId())
                            .addVector(1.0f).setSuccess(true).build());
                });

        SemanticManagerOptions options = new SemanticManagerOptions("test-index", null, 4, 8);

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        // 2 results (one per VectorSet)
        assertEquals(2, result.getSearchMetadata().getSemanticResultsCount());
        // But chunker should only be called ONCE (deduplication)
        verify(chunkerStreamClient, times(1)).streamChunks(any());
        // Embedder should be called TWICE (one per VectorSet)
        verify(embedderStreamClient, times(2)).streamEmbeddings(any());
    }

    private void setField(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }
}
