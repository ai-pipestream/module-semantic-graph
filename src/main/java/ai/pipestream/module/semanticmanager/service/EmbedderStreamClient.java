package ai.pipestream.module.semanticmanager.service;

import ai.pipestream.quarkus.dynamicgrpc.GrpcClientFactory;
import ai.pipestream.semantic.v1.ListEmbeddingModelsRequest;
import ai.pipestream.semantic.v1.ListEmbeddingModelsResponse;
import ai.pipestream.semantic.v1.MutinySemanticEmbedderServiceGrpc;
import ai.pipestream.semantic.v1.StreamEmbeddingsRequest;
import ai.pipestream.semantic.v1.StreamEmbeddingsResponse;
import io.quarkus.cache.CacheResult;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Client wrapper for the SemanticEmbedderService bidirectional streaming RPC.
 * Uses DynamicGrpcClientFactory for service discovery via Stork/Consul.
 */
@ApplicationScoped
public class EmbedderStreamClient {

    private static final Logger log = LoggerFactory.getLogger(EmbedderStreamClient.class);
    private static final String SERVICE_NAME = "embedder";

    @Inject
    GrpcClientFactory grpcClientFactory;

    /**
     * Opens a bidirectional streaming embedding call. Sends chunks as they arrive,
     * receives embedded vectors back. The embedder batches internally for GPU efficiency.
     */
    public Multi<StreamEmbeddingsResponse> streamEmbeddings(Multi<StreamEmbeddingsRequest> requests) {
        log.info("Opening StreamEmbeddings bidirectional stream");

        return grpcClientFactory.getClient(SERVICE_NAME, MutinySemanticEmbedderServiceGrpc::newMutinyStub)
                .onItem().transformToMulti(stub -> stub.streamEmbeddings(requests));
    }

    /**
     * Lists available embedding models from the embedder service.
     * Cached for 30s (matches DjlModelRegistry refresh interval) to avoid
     * a gRPC round-trip per document during Phase 0 validation.
     *
     * @param readyOnly if true, only returns models with READY status
     * @return response containing available models with their status and dimensions
     */
    @CacheResult(cacheName = "embedding-models")
    public Uni<ListEmbeddingModelsResponse> listEmbeddingModels(boolean readyOnly) {
        log.info("Listing embedding models (readyOnly={}) — cache miss, calling embedder", readyOnly);

        ListEmbeddingModelsRequest request = ListEmbeddingModelsRequest.newBuilder()
                .setReadyOnly(readyOnly)
                .build();

        return grpcClientFactory.getClient(SERVICE_NAME, MutinySemanticEmbedderServiceGrpc::newMutinyStub)
                .chain(stub -> stub.listEmbeddingModels(request));
    }
}
