package ai.pipestream.module.semanticmanager.service;

import ai.pipestream.semantic.v1.MutinySemanticEmbedderServiceGrpc;
import ai.pipestream.semantic.v1.StreamEmbeddingsRequest;
import ai.pipestream.semantic.v1.StreamEmbeddingsResponse;
import io.quarkus.grpc.GrpcClient;
import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Client wrapper for the SemanticEmbedderService bidirectional streaming RPC.
 * Discovered via Stork/Consul with the "embedder-service" service name.
 */
@ApplicationScoped
public class EmbedderStreamClient {

    private static final Logger log = LoggerFactory.getLogger(EmbedderStreamClient.class);

    @GrpcClient("embedder-service")
    MutinySemanticEmbedderServiceGrpc.MutinySemanticEmbedderServiceStub embedderStub;

    /**
     * Opens a bidirectional streaming embedding call. Sends chunks as they arrive,
     * receives embedded vectors back. The embedder batches internally for GPU efficiency.
     */
    public Multi<StreamEmbeddingsResponse> streamEmbeddings(Multi<StreamEmbeddingsRequest> requests) {
        log.info("Opening StreamEmbeddings bidirectional stream");
        return embedderStub.streamEmbeddings(requests);
    }
}
