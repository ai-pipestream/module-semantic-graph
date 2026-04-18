package ai.pipestream.module.semanticgraph;

import ai.pipestream.data.v1.NlpDocumentAnalysis;
import ai.pipestream.data.v1.SentenceSpan;
import ai.pipestream.semantic.v1.ChunkConfigEntry;
import ai.pipestream.semantic.v1.EmbeddingModelInfo;
import ai.pipestream.semantic.v1.EmbeddingModelStatus;
import ai.pipestream.semantic.v1.ListEmbeddingModelsRequest;
import ai.pipestream.semantic.v1.ListEmbeddingModelsResponse;
import ai.pipestream.semantic.v1.SemanticChunkerServiceGrpc;
import ai.pipestream.semantic.v1.SemanticEmbedderServiceGrpc;
import ai.pipestream.semantic.v1.StreamChunksRequest;
import ai.pipestream.semantic.v1.StreamChunksResponse;
import ai.pipestream.semantic.v1.StreamEmbeddingsRequest;
import ai.pipestream.semantic.v1.StreamEmbeddingsResponse;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import io.quarkus.test.common.QuarkusTestResourceLifecycleManager;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

/**
 * QuarkusTestResource that starts standalone gRPC servers for the mock chunker
 * and mock embedder services. These run OUTSIDE the application JAR and the
 * application's Stork config is pointed at them via config overrides.
 * <p>
 * The chunker splits text into 10-word chunks.
 * The embedder returns random 384-dim vectors (deterministic seed).
 */
public class MockServicesTestResource implements QuarkusTestResourceLifecycleManager {

    private static final Logger LOG = Logger.getLogger(MockServicesTestResource.class);

    private Server mockServer;
    private int serverPort;

    @Override
    public Map<String, String> start() {
        try {
            mockServer = ServerBuilder.forPort(0) // random port
                    .addService(new MockChunkerImpl())
                    .addService(new MockEmbedderImpl())
                    .build()
                    .start();
            serverPort = mockServer.getPort();
            LOG.infof("Mock gRPC services started on port %d", serverPort);
        } catch (IOException e) {
            throw new RuntimeException("Failed to start mock gRPC servers", e);
        }

        String address = "localhost:" + serverPort;
        return Map.of(
                // Stork static discovery for chunker and embedder
                "stork.chunker.service-discovery.type", "static",
                "stork.chunker.service-discovery.address-list", address,
                "stork.embedder.service-discovery.type", "static",
                "stork.embedder.service-discovery.address-list", address,
                // Direct address for dynamic-grpc (bypasses Consul)
                "quarkus.dynamic-grpc.service.chunker.address", address,
                "quarkus.dynamic-grpc.service.embedder.address", address
        );
    }

    @Override
    public void stop() {
        if (mockServer != null) {
            LOG.infof("Shutting down mock gRPC services on port %d", serverPort);
            mockServer.shutdownNow();
            try {
                mockServer.awaitTermination();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    // =========================================================================
    // Mock Chunker Service — splits text into 10-word chunks
    // =========================================================================

    private static class MockChunkerImpl extends SemanticChunkerServiceGrpc.SemanticChunkerServiceImplBase {

        private static final int WORDS_PER_CHUNK = 10;

        @Override
        public void streamChunks(StreamChunksRequest request,
                                 StreamObserver<StreamChunksResponse> responseObserver) {
            String text = request.getTextContent();
            String docId = request.getDocId();
            String sourceField = request.getSourceFieldName();
            String requestId = request.getRequestId();

            LOG.debugf("MockChunker: doc=%s, sourceField=%s, textLen=%d, configs=%d",
                    docId, sourceField, text.length(), request.getChunkConfigsCount());

            if (request.getChunkConfigsCount() > 0) {
                // Multi-config path
                for (int i = 0; i < request.getChunkConfigsCount(); i++) {
                    ChunkConfigEntry entry = request.getChunkConfigs(i);
                    String configId = entry.getChunkConfigId();
                    boolean isLastConfig = (i == request.getChunkConfigsCount() - 1);

                    List<StreamChunksResponse> chunks = splitIntoChunks(
                            text, requestId, docId, configId, sourceField, isLastConfig);
                    for (StreamChunksResponse chunk : chunks) {
                        responseObserver.onNext(chunk);
                    }
                }
            } else {
                // Legacy single-config path
                String configId = request.getChunkConfigId();
                List<StreamChunksResponse> chunks = splitIntoChunks(
                        text, requestId, docId, configId, sourceField, true);
                for (StreamChunksResponse chunk : chunks) {
                    responseObserver.onNext(chunk);
                }
            }

            responseObserver.onCompleted();
        }

        private List<StreamChunksResponse> splitIntoChunks(
                String text, String requestId, String docId, String configId,
                String sourceField, boolean includeNlpOnLast) {

            List<StreamChunksResponse> chunks = new ArrayList<>();
            String[] words = text.split("\\s+");

            int chunkNumber = 0;
            int charOffset = 0;

            for (int i = 0; i < words.length; i += WORDS_PER_CHUNK) {
                int end = Math.min(i + WORDS_PER_CHUNK, words.length);
                StringBuilder chunkText = new StringBuilder();
                for (int j = i; j < end; j++) {
                    if (j > i) chunkText.append(" ");
                    chunkText.append(words[j]);
                }

                String content = chunkText.toString();
                int startOffset = charOffset;
                int endOffset = startOffset + content.length();
                boolean isLast = (end >= words.length);

                StreamChunksResponse.Builder chunkBuilder = StreamChunksResponse.newBuilder()
                        .setRequestId(requestId)
                        .setDocId(docId)
                        .setChunkId(UUID.randomUUID().toString())
                        .setChunkNumber(chunkNumber)
                        .setTextContent(content)
                        .setStartOffset(startOffset)
                        .setEndOffset(endOffset)
                        .setChunkConfigId(configId)
                        .setSourceFieldName(sourceField)
                        .setIsLast(isLast);

                if (isLast && includeNlpOnLast) {
                    chunkBuilder.setNlpAnalysis(buildMockNlpAnalysis(text));
                }

                chunks.add(chunkBuilder.build());
                chunkNumber++;
                charOffset = endOffset + 1;
            }

            if (chunks.isEmpty()) {
                StreamChunksResponse.Builder emptyBuilder = StreamChunksResponse.newBuilder()
                        .setRequestId(requestId)
                        .setDocId(docId)
                        .setChunkId(UUID.randomUUID().toString())
                        .setChunkNumber(0)
                        .setTextContent("")
                        .setChunkConfigId(configId)
                        .setSourceFieldName(sourceField)
                        .setIsLast(true);

                if (includeNlpOnLast) {
                    emptyBuilder.setNlpAnalysis(buildMockNlpAnalysis(text));
                }
                chunks.add(emptyBuilder.build());
            }

            return chunks;
        }

        private NlpDocumentAnalysis buildMockNlpAnalysis(String text) {
            String[] rawSentences = text.split("(?<=[.?!])\\s+");
            List<SentenceSpan> sentences = new ArrayList<>();
            int offset = 0;
            for (String s : rawSentences) {
                if (!s.isEmpty()) {
                    int idx = text.indexOf(s, offset);
                    if (idx >= 0) {
                        sentences.add(SentenceSpan.newBuilder()
                                .setText(s)
                                .setStartOffset(idx)
                                .setEndOffset(idx + s.length())
                                .build());
                        offset = idx + s.length();
                    }
                }
            }

            String[] words = text.split("\\s+");
            int totalTokens = words.length;

            return NlpDocumentAnalysis.newBuilder()
                    .addAllSentences(sentences)
                    .setDetectedLanguage("eng")
                    .setLanguageConfidence(0.95f)
                    .setTotalTokens(totalTokens)
                    .setNounDensity(0.25f)
                    .setVerbDensity(0.15f)
                    .setAdjectiveDensity(0.08f)
                    .setAdverbDensity(0.05f)
                    .setContentWordRatio(0.55f)
                    .setUniqueLemmaCount((int) (totalTokens * 0.7))
                    .setLexicalDensity(0.55f)
                    .build();
        }
    }

    // =========================================================================
    // Mock Embedder Service — returns random 384-dim vectors
    // =========================================================================

    private static class MockEmbedderImpl extends SemanticEmbedderServiceGrpc.SemanticEmbedderServiceImplBase {

        private static final int DEFAULT_DIMENSIONS = 384;
        private final Random random = new Random(42);

        @Override
        public StreamObserver<StreamEmbeddingsRequest> streamEmbeddings(
                StreamObserver<StreamEmbeddingsResponse> responseObserver) {

            return new StreamObserver<>() {
                @Override
                public void onNext(StreamEmbeddingsRequest req) {
                    int dimensions = getDimensions(req.getEmbeddingModelId());

                    StreamEmbeddingsResponse.Builder builder = StreamEmbeddingsResponse.newBuilder()
                            .setRequestId(req.getRequestId())
                            .setDocId(req.getDocId())
                            .setChunkId(req.getChunkId())
                            .setChunkConfigId(req.getChunkConfigId())
                            .setEmbeddingModelId(req.getEmbeddingModelId())
                            .setSuccess(true);

                    for (int i = 0; i < dimensions; i++) {
                        builder.addVector((float) (random.nextGaussian() * 0.1));
                    }

                    responseObserver.onNext(builder.build());
                }

                @Override
                public void onError(Throwable t) {
                    LOG.warnf("MockEmbedder stream error: %s", t.getMessage());
                }

                @Override
                public void onCompleted() {
                    responseObserver.onCompleted();
                }
            };
        }

        @Override
        public void listEmbeddingModels(ListEmbeddingModelsRequest request,
                                        StreamObserver<ListEmbeddingModelsResponse> responseObserver) {
            ListEmbeddingModelsResponse response = ListEmbeddingModelsResponse.newBuilder()
                    .addModels(model("all-MiniLM-L6-v2", 384))
                    .addModels(model("minilm", 384))
                    .addModels(model("all-mpnet-base-v2", 768))
                    .addModels(model("mpnet", 768))
                    .addModels(model("e5-large", 1024))
                    .addModels(model("e5", 1024))
                    .build();

            responseObserver.onNext(response);
            responseObserver.onCompleted();
        }

        private static EmbeddingModelInfo model(String name, int dims) {
            return EmbeddingModelInfo.newBuilder()
                    .setModelName(name)
                    .setDimensions(dims)
                    .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                    .build();
        }

        private int getDimensions(String modelId) {
            if (modelId == null) return DEFAULT_DIMENSIONS;
            return switch (modelId.toLowerCase()) {
                case "all-minilm-l6-v2", "minilm" -> 384;
                case "all-mpnet-base-v2", "mpnet" -> 768;
                case "e5-large", "e5" -> 1024;
                default -> DEFAULT_DIMENSIONS;
            };
        }
    }
}
