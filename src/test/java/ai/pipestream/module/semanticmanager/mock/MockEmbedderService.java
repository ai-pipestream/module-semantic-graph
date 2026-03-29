package ai.pipestream.module.semanticmanager.mock;

import ai.pipestream.semantic.v1.EmbeddingModelInfo;
import ai.pipestream.semantic.v1.EmbeddingModelStatus;
import ai.pipestream.semantic.v1.ListEmbeddingModelsRequest;
import ai.pipestream.semantic.v1.ListEmbeddingModelsResponse;
import ai.pipestream.semantic.v1.SemanticEmbedderService;
import ai.pipestream.semantic.v1.StreamEmbeddingsRequest;
import ai.pipestream.semantic.v1.StreamEmbeddingsResponse;
import io.smallrye.mutiny.Uni;
import io.quarkus.grpc.GrpcService;
import io.smallrye.mutiny.Multi;
import jakarta.inject.Singleton;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

/**
 * In-process mock embedder that returns random vectors.
 * Simulates a real embedding service without needing ML models.
 * Default dimension is 384 (MiniLM-sized), configurable by embedding model ID.
 * <p>
 * Reports all known models as READY via listEmbeddingModels for fail-fast validation.
 */
@Singleton
@GrpcService
public class MockEmbedderService implements SemanticEmbedderService {

    private static final Logger log = LoggerFactory.getLogger(MockEmbedderService.class);
    private static final int DEFAULT_DIMENSIONS = 384;

    private final Random random = new Random(42); // deterministic seed for reproducibility

    @Override
    public Multi<StreamEmbeddingsResponse> streamEmbeddings(Multi<StreamEmbeddingsRequest> requests) {
        return requests.map(req -> {
            int dimensions = getDimensions(req.getEmbeddingModelId());

            StreamEmbeddingsResponse.Builder builder = StreamEmbeddingsResponse.newBuilder()
                    .setRequestId(req.getRequestId())
                    .setDocId(req.getDocId())
                    .setChunkId(req.getChunkId())
                    .setChunkConfigId(req.getChunkConfigId())
                    .setEmbeddingModelId(req.getEmbeddingModelId())
                    .setSuccess(true);

            // Generate random vector
            for (int i = 0; i < dimensions; i++) {
                builder.addVector((float) (random.nextGaussian() * 0.1));
            }

            log.debug("MockEmbedder: embedded chunk={} with model={}, dims={}",
                    req.getChunkId(), req.getEmbeddingModelId(), dimensions);

            return builder.build();
        });
    }

    @Override
    public Uni<ListEmbeddingModelsResponse> listEmbeddingModels(ListEmbeddingModelsRequest request) {
        log.info("MockEmbedder: listEmbeddingModels(readyOnly={})", request.getReadyOnly());

        ListEmbeddingModelsResponse.Builder response = ListEmbeddingModelsResponse.newBuilder()
                .addModels(EmbeddingModelInfo.newBuilder()
                        .setModelName("all-MiniLM-L6-v2")
                        .setDimensions(384)
                        .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                        .build())
                .addModels(EmbeddingModelInfo.newBuilder()
                        .setModelName("minilm")
                        .setDimensions(384)
                        .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                        .build())
                .addModels(EmbeddingModelInfo.newBuilder()
                        .setModelName("all-mpnet-base-v2")
                        .setDimensions(768)
                        .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                        .build())
                .addModels(EmbeddingModelInfo.newBuilder()
                        .setModelName("mpnet")
                        .setDimensions(768)
                        .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                        .build())
                .addModels(EmbeddingModelInfo.newBuilder()
                        .setModelName("e5-large")
                        .setDimensions(1024)
                        .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                        .build())
                .addModels(EmbeddingModelInfo.newBuilder()
                        .setModelName("e5")
                        .setDimensions(1024)
                        .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                        .build());

        return Uni.createFrom().item(response.build());
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
