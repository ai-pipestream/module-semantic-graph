package ai.pipestream.module.semanticmanager.service;

import ai.pipestream.data.v1.ChunkEmbedding;
import ai.pipestream.data.v1.PipeDoc;
import ai.pipestream.data.v1.SearchMetadata;
import ai.pipestream.data.v1.SemanticChunk;
import ai.pipestream.data.v1.SemanticProcessingResult;
import ai.pipestream.module.semanticmanager.config.SemanticManagerOptions;
import ai.pipestream.opensearch.v1.VectorSet;
import ai.pipestream.semantic.v1.StreamChunksRequest;
import ai.pipestream.semantic.v1.StreamChunksResponse;
import ai.pipestream.semantic.v1.StreamEmbeddingsRequest;
import ai.pipestream.semantic.v1.StreamEmbeddingsResponse;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Core orchestrator for semantic indexing. Implements the fan-out/fan-in pattern:
 *
 * 1. Resolve VectorSets (the "recipe") from opensearch-manager
 * 2. Deduplicate chunking work — group by (chunker_config_id, source_field)
 * 3. Fan-out: open parallel chunker streams
 * 4. Pipeline: as chunks arrive, forward to appropriate embedder streams
 * 5. Fan-in: collect embedded chunks, build SemanticProcessingResult entries
 * 6. Return enriched PipeDoc
 */
@ApplicationScoped
public class SemanticIndexingOrchestrator {

    private static final Logger log = LoggerFactory.getLogger(SemanticIndexingOrchestrator.class);

    @Inject
    VectorSetResolver vectorSetResolver;

    @Inject
    ChunkerStreamClient chunkerStreamClient;

    @Inject
    EmbedderStreamClient embedderStreamClient;

    /**
     * Orchestrates the full semantic indexing pipeline for a document.
     *
     * @param inputDoc the input PipeDoc
     * @param options  configuration options
     * @param nodeId   the coordinator's node ID for audit tagging
     * @return Uni of enriched PipeDoc with SemanticProcessingResults populated
     */
    public Uni<PipeDoc> orchestrate(PipeDoc inputDoc, SemanticManagerOptions options, String nodeId) {
        String docId = inputDoc.getDocId();
        log.info("Starting semantic orchestration for doc: {}, index: {}", docId, options.effectiveIndexName());

        // Step 1: Resolve VectorSets
        return vectorSetResolver.resolveVectorSets(options.effectiveIndexName())
                .chain(vectorSets -> {
                    // Filter to specific IDs if requested
                    List<VectorSet> activeVectorSets = filterVectorSets(vectorSets, options.vectorSetIds());

                    if (activeVectorSets.isEmpty()) {
                        log.warn("No VectorSets found for index: {}. Returning doc unchanged.", options.effectiveIndexName());
                        return Uni.createFrom().item(inputDoc);
                    }

                    log.info("Processing {} VectorSets for doc: {}", activeVectorSets.size(), docId);

                    // Step 2: Deduplicate chunking work
                    Map<ChunkingKey, List<VectorSet>> chunkingGroups = groupByChunkingKey(activeVectorSets);
                    log.info("Deduplicated to {} chunking groups (from {} VectorSets)",
                            chunkingGroups.size(), activeVectorSets.size());

                    // Step 3-5: Fan-out, pipeline, fan-in
                    return processChunkingGroups(inputDoc, chunkingGroups, nodeId);
                });
    }

    private List<VectorSet> filterVectorSets(List<VectorSet> vectorSets, List<String> vectorSetIds) {
        if (vectorSetIds == null || vectorSetIds.isEmpty()) {
            return vectorSets;
        }
        Set<String> idSet = new HashSet<>(vectorSetIds);
        return vectorSets.stream()
                .filter(vs -> idSet.contains(vs.getId()))
                .collect(Collectors.toList());
    }

    /**
     * Groups VectorSets by (chunker_config_id, source_field) to avoid redundant chunking.
     */
    private Map<ChunkingKey, List<VectorSet>> groupByChunkingKey(List<VectorSet> vectorSets) {
        return vectorSets.stream()
                .collect(Collectors.groupingBy(
                        vs -> new ChunkingKey(vs.getChunkerConfigId(), vs.getSourceField())));
    }

    /**
     * Processes all chunking groups in parallel, pipelines chunks to embedders,
     * and assembles the enriched PipeDoc.
     */
    private Uni<PipeDoc> processChunkingGroups(PipeDoc inputDoc,
                                                Map<ChunkingKey, List<VectorSet>> chunkingGroups,
                                                String nodeId) {
        String docId = inputDoc.getDocId();

        // Collect all SemanticProcessingResults from all groups
        List<Uni<List<SemanticProcessingResult>>> groupUnis = new ArrayList<>();

        for (Map.Entry<ChunkingKey, List<VectorSet>> entry : chunkingGroups.entrySet()) {
            ChunkingKey key = entry.getKey();
            List<VectorSet> vectorSetsForChunker = entry.getValue();

            groupUnis.add(processOneChunkingGroup(inputDoc, key, vectorSetsForChunker, nodeId));
        }

        // Combine all group results
        return Uni.combine().all().unis(groupUnis)
                .with(results -> {
                    PipeDoc.Builder outputDocBuilder = inputDoc.toBuilder();
                    SearchMetadata.Builder searchMetadataBuilder = inputDoc.hasSearchMetadata()
                            ? inputDoc.getSearchMetadata().toBuilder()
                            : SearchMetadata.newBuilder();

                    for (Object resultObj : results) {
                        @SuppressWarnings("unchecked")
                        List<SemanticProcessingResult> semanticResults = (List<SemanticProcessingResult>) resultObj;
                        for (SemanticProcessingResult result : semanticResults) {
                            searchMetadataBuilder.addSemanticResults(result);
                        }
                    }

                    outputDocBuilder.setSearchMetadata(searchMetadataBuilder.build());

                    log.info("Semantic orchestration complete for doc: {}. Total semantic results: {}",
                            docId, searchMetadataBuilder.getSemanticResultsCount());

                    return outputDocBuilder.build();
                });
    }

    /**
     * Processes one chunking group: opens a chunker stream, then for each VectorSet
     * in the group, opens an embedder stream and forwards chunks.
     */
    private Uni<List<SemanticProcessingResult>> processOneChunkingGroup(
            PipeDoc inputDoc, ChunkingKey key, List<VectorSet> vectorSets, String nodeId) {

        String docId = inputDoc.getDocId();
        String sourceText = extractSourceText(inputDoc, key.sourceField());

        if (sourceText == null || sourceText.isEmpty()) {
            log.warn("No text found for source field '{}' in doc: {}", key.sourceField(), docId);
            return Uni.createFrom().item(Collections.emptyList());
        }

        String requestId = UUID.randomUUID().toString();

        // Build the chunking request
        StreamChunksRequest chunksRequest = StreamChunksRequest.newBuilder()
                .setRequestId(requestId)
                .setDocId(docId)
                .setSourceFieldName(key.sourceField())
                .setTextContent(sourceText)
                .setChunkConfigId(key.chunkerConfigId())
                .build();

        // Open the chunker stream — get all chunks first, then fan to embedders
        return chunkerStreamClient.streamChunks(chunksRequest)
                .collect().asList()
                .chain(chunks -> {
                    if (chunks.isEmpty()) {
                        log.warn("Chunker returned no chunks for configId={}, sourceField={}, doc={}",
                                key.chunkerConfigId(), key.sourceField(), docId);
                        return Uni.createFrom().item(Collections.<SemanticProcessingResult>emptyList());
                    }

                    log.info("Received {} chunks for configId={}, sourceField={}. Fanning out to {} embedders.",
                            chunks.size(), key.chunkerConfigId(), key.sourceField(), vectorSets.size());

                    // For each VectorSet (embedder), create an embedder stream
                    List<Uni<SemanticProcessingResult>> embedderUnis = new ArrayList<>();
                    for (VectorSet vs : vectorSets) {
                        embedderUnis.add(
                                embedChunks(chunks, vs, requestId, docId, nodeId)
                                        .onFailure().recoverWithItem(error -> {
                                            log.error("Embedder failed for VectorSet {}: {}",
                                                    vs.getName(), error.getMessage());
                                            return null; // partial failure tolerance
                                        })
                        );
                    }

                    return Uni.combine().all().unis(embedderUnis)
                            .with(results -> {
                                List<SemanticProcessingResult> nonNull = new ArrayList<>();
                                for (Object r : results) {
                                    if (r != null) {
                                        nonNull.add((SemanticProcessingResult) r);
                                    }
                                }
                                return nonNull;
                            });
                });
    }

    /**
     * Sends chunks to an embedder stream and assembles the SemanticProcessingResult.
     */
    private Uni<SemanticProcessingResult> embedChunks(
            List<StreamChunksResponse> chunks,
            VectorSet vectorSet,
            String requestId,
            String docId,
            String nodeId) {

        String embeddingModelId = vectorSet.getEmbeddingModelConfigId();
        String chunkConfigId = vectorSet.getChunkerConfigId();

        // Build embedding requests from chunks
        Multi<StreamEmbeddingsRequest> embeddingRequests = Multi.createFrom().iterable(chunks)
                .map(chunk -> StreamEmbeddingsRequest.newBuilder()
                        .setRequestId(requestId)
                        .setDocId(docId)
                        .setChunkId(chunk.getChunkId())
                        .setTextContent(chunk.getTextContent())
                        .setChunkConfigId(chunkConfigId)
                        .setEmbeddingModelId(embeddingModelId)
                        .build());

        // Open bidirectional stream and collect results
        return embedderStreamClient.streamEmbeddings(embeddingRequests)
                .collect().asList()
                .map(embeddingResponses -> {
                    // Build a map of chunkId -> embedding for fast lookup
                    Map<String, StreamEmbeddingsResponse> embeddingMap = new HashMap<>();
                    for (StreamEmbeddingsResponse resp : embeddingResponses) {
                        if (resp.getSuccess()) {
                            embeddingMap.put(resp.getChunkId(), resp);
                        } else {
                            log.warn("Embedding failed for chunk {}: {}", resp.getChunkId(), resp.getErrorMessage());
                        }
                    }

                    // Assemble SemanticProcessingResult
                    SemanticProcessingResult.Builder resultBuilder = SemanticProcessingResult.newBuilder()
                            .setResultId(UUID.randomUUID().toString())
                            .setSourceFieldName(vectorSet.getSourceField())
                            .setChunkConfigId(chunkConfigId)
                            .setEmbeddingConfigId(embeddingModelId)
                            .setResultSetName(vectorSet.getResultSetName());

                    // Add coordinator node_id as metadata
                    if (nodeId != null) {
                        resultBuilder.putMetadata("coordinator_node_id",
                                com.google.protobuf.Value.newBuilder().setStringValue(nodeId).build());
                    }
                    resultBuilder.putMetadata("vector_set_id",
                            com.google.protobuf.Value.newBuilder().setStringValue(vectorSet.getId()).build());
                    resultBuilder.putMetadata("vector_set_name",
                            com.google.protobuf.Value.newBuilder().setStringValue(vectorSet.getName()).build());

                    // Build SemanticChunks with embeddings
                    for (StreamChunksResponse chunk : chunks) {
                        ChunkEmbedding.Builder embeddingInfoBuilder = ChunkEmbedding.newBuilder()
                                .setTextContent(chunk.getTextContent())
                                .setChunkId(chunk.getChunkId())
                                .setOriginalCharStartOffset(chunk.getStartOffset())
                                .setOriginalCharEndOffset(chunk.getEndOffset())
                                .setChunkConfigId(chunkConfigId);

                        // Add vector if available
                        StreamEmbeddingsResponse embResp = embeddingMap.get(chunk.getChunkId());
                        if (embResp != null) {
                            embeddingInfoBuilder.addAllVector(embResp.getVectorList());
                        }

                        SemanticChunk semanticChunk = SemanticChunk.newBuilder()
                                .setChunkId(chunk.getChunkId())
                                .setChunkNumber(chunk.getChunkNumber())
                                .setEmbeddingInfo(embeddingInfoBuilder.build())
                                .putAllMetadata(chunk.getMetadataMap())
                                .build();

                        resultBuilder.addChunks(semanticChunk);
                    }

                    log.info("Assembled SemanticProcessingResult: vectorSet={}, chunks={}, embeddings={}",
                            vectorSet.getName(), chunks.size(), embeddingMap.size());

                    return resultBuilder.build();
                });
    }

    /**
     * Extracts text from PipeDoc for a given source field name.
     */
    private String extractSourceText(PipeDoc doc, String sourceField) {
        if (!doc.hasSearchMetadata()) {
            return null;
        }
        SearchMetadata sm = doc.getSearchMetadata();
        return switch (sourceField.toLowerCase()) {
            case "body" -> sm.hasBody() ? sm.getBody() : null;
            case "title" -> sm.hasTitle() ? sm.getTitle() : null;
            default -> {
                log.warn("Unsupported source field: {}", sourceField);
                yield null;
            }
        };
    }

    /**
     * Key for deduplicating chunking work.
     */
    record ChunkingKey(String chunkerConfigId, String sourceField) {
    }
}
