package ai.pipestream.module.semanticmanager.service;

import ai.pipestream.opensearch.v1.ListVectorSetsRequest;
import ai.pipestream.opensearch.v1.MutinyVectorSetServiceGrpc;
import ai.pipestream.opensearch.v1.VectorSet;
import io.quarkus.grpc.GrpcClient;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Resolves VectorSets from the opensearch-manager's VectorSetService.
 * VectorSets define the "recipe" — which chunker config + embedding model combinations
 * to use for a given index.
 */
@ApplicationScoped
public class VectorSetResolver {

    private static final Logger log = LoggerFactory.getLogger(VectorSetResolver.class);

    @GrpcClient("opensearch-manager")
    MutinyVectorSetServiceGrpc.MutinyVectorSetServiceStub vectorSetClient;

    /**
     * Lists all VectorSets for a given index name. Handles pagination to fetch all results.
     */
    public Uni<List<VectorSet>> resolveVectorSets(String indexName) {
        log.info("Resolving VectorSets for index: {}", indexName);

        return fetchAllPages(indexName, "", new ArrayList<>());
    }

    private Uni<List<VectorSet>> fetchAllPages(String indexName, String pageToken, List<VectorSet> accumulated) {
        ListVectorSetsRequest.Builder requestBuilder = ListVectorSetsRequest.newBuilder()
                .setIndexName(indexName)
                .setPageSize(100);

        if (pageToken != null && !pageToken.isEmpty()) {
            requestBuilder.setPageToken(pageToken);
        }

        return vectorSetClient.listVectorSets(requestBuilder.build())
                .chain(response -> {
                    accumulated.addAll(response.getVectorSetsList());
                    String nextToken = response.getNextPageToken();

                    if (nextToken != null && !nextToken.isEmpty()) {
                        return fetchAllPages(indexName, nextToken, accumulated);
                    }

                    log.info("Resolved {} VectorSets for index: {}", accumulated.size(), indexName);
                    return Uni.createFrom().item(accumulated);
                });
    }
}
