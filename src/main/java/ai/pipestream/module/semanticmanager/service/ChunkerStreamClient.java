package ai.pipestream.module.semanticmanager.service;

import ai.pipestream.semantic.v1.MutinySemanticChunkerServiceGrpc;
import ai.pipestream.semantic.v1.StreamChunksRequest;
import ai.pipestream.semantic.v1.StreamChunksResponse;
import io.quarkus.grpc.GrpcClient;
import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Client wrapper for the SemanticChunkerService streaming RPC.
 * Discovered via Stork/Consul with the "chunker-service" service name.
 */
@ApplicationScoped
public class ChunkerStreamClient {

    private static final Logger log = LoggerFactory.getLogger(ChunkerStreamClient.class);

    @GrpcClient("chunker-service")
    MutinySemanticChunkerServiceGrpc.MutinySemanticChunkerServiceStub chunkerStub;

    /**
     * Opens a server-streaming chunking call. Returns a Multi of chunk responses
     * that can be forwarded to embedder streams as they arrive.
     */
    public Multi<StreamChunksResponse> streamChunks(StreamChunksRequest request) {
        log.info("Opening StreamChunks: requestId={}, docId={}, configId={}, sourceField={}",
                request.getRequestId(), request.getDocId(),
                request.getChunkConfigId(), request.getSourceFieldName());

        return chunkerStub.streamChunks(request);
    }
}
