package ai.pipestream.module.semanticmanager.service;

import ai.pipestream.data.v1.*;
import ai.pipestream.module.semanticmanager.config.SemanticManagerOptions;
import ai.pipestream.opensearch.v1.VectorSet;
import ai.pipestream.semantic.v1.StreamChunksRequest;
import ai.pipestream.semantic.v1.StreamChunksResponse;
import ai.pipestream.semantic.v1.StreamEmbeddingsRequest;
import ai.pipestream.semantic.v1.StreamEmbeddingsResponse;
import com.google.protobuf.Struct;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Core orchestrator for semantic indexing. Supports two recipe sources:
 *
 * 1. **VectorSetDirectives on the doc** (primary) — populated by the mapper step
 *    via CEL selectors. Each directive specifies source text + arrays of chunker
 *    and embedder configs. The orchestrator produces the cartesian product.
 *
 * 2. **VectorSetService** (fallback) — queries opensearch-manager for VectorSets
 *    by index name. Used when no directives are present on the doc.
 *
 * In both cases, the orchestrator:
 * - Deduplicates chunking work by (chunker_config, source_text_key)
 * - Fans out to parallel chunker streams
 * - Pipelines chunks to embedder streams
 * - Assembles SemanticProcessingResult entries on the enriched PipeDoc
 */
@ApplicationScoped
public class SemanticIndexingOrchestrator {

    private static final Logger log = LoggerFactory.getLogger(SemanticIndexingOrchestrator.class);

    private static final String DEFAULT_FIELD_NAME_TEMPLATE = "{source_label}_{chunker_id}_{embedder_id}";

    @Inject
    VectorSetResolver vectorSetResolver;

    @Inject
    ChunkerStreamClient chunkerStreamClient;

    @Inject
    EmbedderStreamClient embedderStreamClient;

    /**
     * Orchestrates semantic indexing for a document. Checks for directives on the doc first,
     * falls back to VectorSetService resolution.
     */
    public Uni<PipeDoc> orchestrate(PipeDoc inputDoc, SemanticManagerOptions options, String nodeId) {
        String docId = inputDoc.getDocId();

        // Primary: use directives from the doc if present
        if (hasDirectives(inputDoc)) {
            log.info("Using VectorSetDirectives from doc: {} ({} directives)",
                    docId, inputDoc.getSearchMetadata().getVectorSetDirectives().getDirectivesCount());
            return orchestrateFromDirectives(inputDoc, nodeId);
        }

        // Fallback: resolve from VectorSetService
        log.info("No directives on doc: {}, falling back to VectorSetService for index: {}",
                docId, options.effectiveIndexName());
        return orchestrateFromVectorSetService(inputDoc, options, nodeId);
    }

    private boolean hasDirectives(PipeDoc doc) {
        return doc.hasSearchMetadata()
                && doc.getSearchMetadata().hasVectorSetDirectives()
                && doc.getSearchMetadata().getVectorSetDirectives().getDirectivesCount() > 0;
    }

    // =========================================================================
    // Directive-based orchestration (primary path)
    // =========================================================================

    private Uni<PipeDoc> orchestrateFromDirectives(PipeDoc inputDoc, String nodeId) {
        String docId = inputDoc.getDocId();
        VectorSetDirectives directives = inputDoc.getSearchMetadata().getVectorSetDirectives();

        // Build the work items: one per (directive, chunker_config, embedder_config) triple
        // but deduplicate chunking by (chunker_config_id, source_label)
        Map<DirectiveChunkingKey, DirectiveChunkingGroup> chunkingGroups = new LinkedHashMap<>();

        for (VectorDirective directive : directives.getDirectivesList()) {
            String sourceLabel = directive.getSourceLabel();
            String sourceText = extractTextByCelSelector(inputDoc, directive.getCelSelector(), sourceLabel);

            if (sourceText == null || sourceText.isEmpty()) {
                log.warn("CEL selector '{}' returned no text for source_label '{}' in doc: {}",
                        directive.getCelSelector(), sourceLabel, docId);
                continue;
            }

            String template = directive.hasFieldNameTemplate()
                    ? directive.getFieldNameTemplate() : DEFAULT_FIELD_NAME_TEMPLATE;

            for (NamedChunkerConfig chunkerCfg : directive.getChunkerConfigsList()) {
                DirectiveChunkingKey key = new DirectiveChunkingKey(chunkerCfg.getConfigId(), sourceLabel);

                DirectiveChunkingGroup group = chunkingGroups.computeIfAbsent(key,
                        k -> new DirectiveChunkingGroup(sourceText, sourceLabel,
                                chunkerCfg.getConfigId(), chunkerCfg.getConfig(),
                                new ArrayList<>()));

                // Each embedder config in this directive gets paired with this chunker
                for (NamedEmbedderConfig embedderCfg : directive.getEmbedderConfigsList()) {
                    group.embedderTargets().add(new EmbedderTarget(
                            embedderCfg.getConfigId(), embedderCfg.getConfig(), template));
                }
            }
        }

        if (chunkingGroups.isEmpty()) {
            log.warn("No valid directives produced work items for doc: {}", docId);
            return Uni.createFrom().item(inputDoc);
        }

        log.info("Directive orchestration: {} chunking groups for doc: {}", chunkingGroups.size(), docId);

        // Process each chunking group
        List<Uni<List<SemanticProcessingResult>>> groupUnis = new ArrayList<>();
        for (DirectiveChunkingGroup group : chunkingGroups.values()) {
            groupUnis.add(processDirectiveGroup(inputDoc, group, nodeId));
        }

        return combineResults(inputDoc, groupUnis);
    }

    private Uni<List<SemanticProcessingResult>> processDirectiveGroup(
            PipeDoc inputDoc, DirectiveChunkingGroup group, String nodeId) {

        String docId = inputDoc.getDocId();
        String requestId = UUID.randomUUID().toString();

        // Build chunking request with the chunker's native config
        StreamChunksRequest.Builder reqBuilder = StreamChunksRequest.newBuilder()
                .setRequestId(requestId)
                .setDocId(docId)
                .setSourceFieldName(group.sourceLabel())
                .setTextContent(group.sourceText())
                .setChunkConfigId(group.chunkerConfigId());

        if (group.chunkerConfig() != null) {
            reqBuilder.setChunkerConfig(group.chunkerConfig());
        }

        return chunkerStreamClient.streamChunks(reqBuilder.build())
                .collect().asList()
                .chain(chunks -> {
                    if (chunks.isEmpty()) {
                        log.warn("Chunker returned no chunks for configId={}, source={}, doc={}",
                                group.chunkerConfigId(), group.sourceLabel(), docId);
                        return Uni.createFrom().item(Collections.<SemanticProcessingResult>emptyList());
                    }

                    log.info("Received {} chunks for configId={}, source={}. Fanning out to {} embedders.",
                            chunks.size(), group.chunkerConfigId(), group.sourceLabel(),
                            group.embedderTargets().size());

                    List<Uni<SemanticProcessingResult>> embedderUnis = new ArrayList<>();
                    for (EmbedderTarget target : group.embedderTargets()) {
                        String resultSetName = target.template()
                                .replace("{source_label}", group.sourceLabel())
                                .replace("{chunker_id}", group.chunkerConfigId())
                                .replace("{embedder_id}", target.embedderConfigId());

                        embedderUnis.add(
                                embedChunksForDirective(chunks, group, target, resultSetName,
                                        requestId, docId, nodeId)
                                        .onFailure().recoverWithItem(error -> {
                                            log.error("Embedder {} failed for doc {}: {}",
                                                    target.embedderConfigId(), docId, error.getMessage());
                                            return null;
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

    private Uni<SemanticProcessingResult> embedChunksForDirective(
            List<StreamChunksResponse> chunks,
            DirectiveChunkingGroup group,
            EmbedderTarget target,
            String resultSetName,
            String requestId,
            String docId,
            String nodeId) {

        Multi<StreamEmbeddingsRequest> embeddingRequests = Multi.createFrom().iterable(chunks)
                .map(chunk -> {
                    StreamEmbeddingsRequest.Builder builder = StreamEmbeddingsRequest.newBuilder()
                            .setRequestId(requestId)
                            .setDocId(docId)
                            .setChunkId(chunk.getChunkId())
                            .setTextContent(chunk.getTextContent())
                            .setChunkConfigId(group.chunkerConfigId())
                            .setEmbeddingModelId(target.embedderConfigId());
                    if (target.embedderConfig() != null) {
                        builder.setEmbedderConfig(target.embedderConfig());
                    }
                    return builder.build();
                });

        return embedderStreamClient.streamEmbeddings(embeddingRequests)
                .collect().asList()
                .map(responses -> assembleResult(
                        chunks, responses, group.sourceLabel(), group.chunkerConfigId(),
                        target.embedderConfigId(), resultSetName, nodeId));
    }

    /**
     * Extracts text from the PipeDoc using a CEL selector or simple field path.
     * For now, supports common dot-paths; full CEL evaluation can be added later
     * by integrating the engine's CelEvaluatorService.
     */
    private String extractTextByCelSelector(PipeDoc doc, String celSelector, String sourceLabel) {
        if (celSelector == null || celSelector.isEmpty()) {
            // Fall back to source_label as a simple field name
            return extractSourceText(doc, sourceLabel);
        }

        // Support common dot-path patterns without full CEL
        String normalized = celSelector.toLowerCase().trim();
        if (normalized.startsWith("document.")) {
            normalized = normalized.substring("document.".length());
        }

        if (normalized.startsWith("search_metadata.")) {
            String field = normalized.substring("search_metadata.".length());
            return extractSourceText(doc, field);
        }

        // Try as a direct field name
        return extractSourceText(doc, celSelector);
    }

    // =========================================================================
    // VectorSetService-based orchestration (fallback path)
    // =========================================================================

    private Uni<PipeDoc> orchestrateFromVectorSetService(PipeDoc inputDoc,
                                                          SemanticManagerOptions options,
                                                          String nodeId) {
        String docId = inputDoc.getDocId();

        return vectorSetResolver.resolveVectorSets(options.effectiveIndexName())
                .chain(vectorSets -> {
                    List<VectorSet> activeVectorSets = filterVectorSets(vectorSets, options.vectorSetIds());

                    if (activeVectorSets.isEmpty()) {
                        log.warn("No VectorSets found for index: {}. Returning doc unchanged.",
                                options.effectiveIndexName());
                        return Uni.createFrom().item(inputDoc);
                    }

                    log.info("Processing {} VectorSets for doc: {}", activeVectorSets.size(), docId);

                    Map<ChunkingKey, List<VectorSet>> chunkingGroups = groupByChunkingKey(activeVectorSets);
                    log.info("Deduplicated to {} chunking groups (from {} VectorSets)",
                            chunkingGroups.size(), activeVectorSets.size());

                    List<Uni<List<SemanticProcessingResult>>> groupUnis = new ArrayList<>();
                    for (Map.Entry<ChunkingKey, List<VectorSet>> entry : chunkingGroups.entrySet()) {
                        groupUnis.add(processVectorSetGroup(inputDoc, entry.getKey(), entry.getValue(), nodeId));
                    }

                    return combineResults(inputDoc, groupUnis);
                });
    }

    private Uni<List<SemanticProcessingResult>> processVectorSetGroup(
            PipeDoc inputDoc, ChunkingKey key, List<VectorSet> vectorSets, String nodeId) {

        String docId = inputDoc.getDocId();
        String sourceText = extractSourceText(inputDoc, key.sourceField());

        if (sourceText == null || sourceText.isEmpty()) {
            log.warn("No text found for source field '{}' in doc: {}", key.sourceField(), docId);
            return Uni.createFrom().item(Collections.emptyList());
        }

        String requestId = UUID.randomUUID().toString();

        StreamChunksRequest chunksRequest = StreamChunksRequest.newBuilder()
                .setRequestId(requestId)
                .setDocId(docId)
                .setSourceFieldName(key.sourceField())
                .setTextContent(sourceText)
                .setChunkConfigId(key.chunkerConfigId())
                .build();

        return chunkerStreamClient.streamChunks(chunksRequest)
                .collect().asList()
                .chain(chunks -> {
                    if (chunks.isEmpty()) {
                        return Uni.createFrom().item(Collections.<SemanticProcessingResult>emptyList());
                    }

                    List<Uni<SemanticProcessingResult>> embedderUnis = new ArrayList<>();
                    for (VectorSet vs : vectorSets) {
                        embedderUnis.add(
                                embedChunksForVectorSet(chunks, vs, requestId, docId, nodeId)
                                        .onFailure().recoverWithItem(error -> {
                                            log.error("Embedder failed for VectorSet {}: {}",
                                                    vs.getName(), error.getMessage());
                                            return null;
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

    private Uni<SemanticProcessingResult> embedChunksForVectorSet(
            List<StreamChunksResponse> chunks, VectorSet vectorSet,
            String requestId, String docId, String nodeId) {

        String embeddingModelId = vectorSet.getEmbeddingModelConfigId();
        String chunkConfigId = vectorSet.getChunkerConfigId();

        Multi<StreamEmbeddingsRequest> embeddingRequests = Multi.createFrom().iterable(chunks)
                .map(chunk -> StreamEmbeddingsRequest.newBuilder()
                        .setRequestId(requestId)
                        .setDocId(docId)
                        .setChunkId(chunk.getChunkId())
                        .setTextContent(chunk.getTextContent())
                        .setChunkConfigId(chunkConfigId)
                        .setEmbeddingModelId(embeddingModelId)
                        .build());

        return embedderStreamClient.streamEmbeddings(embeddingRequests)
                .collect().asList()
                .map(responses -> {
                    SemanticProcessingResult result = assembleResult(
                            chunks, responses, vectorSet.getSourceField(), chunkConfigId,
                            embeddingModelId, vectorSet.getResultSetName(), nodeId);

                    // Add VectorSet-specific metadata
                    return result.toBuilder()
                            .putMetadata("vector_set_id",
                                    com.google.protobuf.Value.newBuilder()
                                            .setStringValue(vectorSet.getId()).build())
                            .putMetadata("vector_set_name",
                                    com.google.protobuf.Value.newBuilder()
                                            .setStringValue(vectorSet.getName()).build())
                            .build();
                });
    }

    // =========================================================================
    // Shared helpers
    // =========================================================================

    private SemanticProcessingResult assembleResult(
            List<StreamChunksResponse> chunks,
            List<StreamEmbeddingsResponse> embeddingResponses,
            String sourceFieldName,
            String chunkConfigId,
            String embeddingConfigId,
            String resultSetName,
            String nodeId) {

        Map<String, StreamEmbeddingsResponse> embeddingMap = new HashMap<>();
        for (StreamEmbeddingsResponse resp : embeddingResponses) {
            if (resp.getSuccess()) {
                embeddingMap.put(resp.getChunkId(), resp);
            } else {
                log.warn("Embedding failed for chunk {}: {}", resp.getChunkId(), resp.getErrorMessage());
            }
        }

        SemanticProcessingResult.Builder resultBuilder = SemanticProcessingResult.newBuilder()
                .setResultId(UUID.randomUUID().toString())
                .setSourceFieldName(sourceFieldName)
                .setChunkConfigId(chunkConfigId)
                .setEmbeddingConfigId(embeddingConfigId)
                .setResultSetName(resultSetName);

        if (nodeId != null) {
            resultBuilder.putMetadata("coordinator_node_id",
                    com.google.protobuf.Value.newBuilder().setStringValue(nodeId).build());
        }

        for (StreamChunksResponse chunk : chunks) {
            ChunkEmbedding.Builder embeddingInfoBuilder = ChunkEmbedding.newBuilder()
                    .setTextContent(chunk.getTextContent())
                    .setChunkId(chunk.getChunkId())
                    .setOriginalCharStartOffset(chunk.getStartOffset())
                    .setOriginalCharEndOffset(chunk.getEndOffset())
                    .setChunkConfigId(chunkConfigId);

            StreamEmbeddingsResponse embResp = embeddingMap.get(chunk.getChunkId());
            if (embResp != null) {
                embeddingInfoBuilder.addAllVector(embResp.getVectorList());
            }

            resultBuilder.addChunks(SemanticChunk.newBuilder()
                    .setChunkId(chunk.getChunkId())
                    .setChunkNumber(chunk.getChunkNumber())
                    .setEmbeddingInfo(embeddingInfoBuilder.build())
                    .putAllMetadata(chunk.getMetadataMap())
                    .build());
        }

        log.info("Assembled SemanticProcessingResult: resultSet={}, chunks={}, embeddings={}",
                resultSetName, chunks.size(), embeddingMap.size());

        return resultBuilder.build();
    }

    private Uni<PipeDoc> combineResults(PipeDoc inputDoc,
                                         List<Uni<List<SemanticProcessingResult>>> groupUnis) {
        String docId = inputDoc.getDocId();

        return Uni.combine().all().unis(groupUnis)
                .with(results -> {
                    PipeDoc.Builder outputDocBuilder = inputDoc.toBuilder();
                    SearchMetadata.Builder smBuilder = inputDoc.hasSearchMetadata()
                            ? inputDoc.getSearchMetadata().toBuilder()
                            : SearchMetadata.newBuilder();

                    for (Object resultObj : results) {
                        @SuppressWarnings("unchecked")
                        List<SemanticProcessingResult> semanticResults =
                                (List<SemanticProcessingResult>) resultObj;
                        for (SemanticProcessingResult result : semanticResults) {
                            smBuilder.addSemanticResults(result);
                        }
                    }

                    outputDocBuilder.setSearchMetadata(smBuilder.build());
                    log.info("Semantic orchestration complete for doc: {}. Total semantic results: {}",
                            docId, smBuilder.getSemanticResultsCount());
                    return outputDocBuilder.build();
                });
    }

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

    private List<VectorSet> filterVectorSets(List<VectorSet> vectorSets, List<String> vectorSetIds) {
        if (vectorSetIds == null || vectorSetIds.isEmpty()) {
            return vectorSets;
        }
        Set<String> idSet = new HashSet<>(vectorSetIds);
        return vectorSets.stream()
                .filter(vs -> idSet.contains(vs.getId()))
                .collect(Collectors.toList());
    }

    private Map<ChunkingKey, List<VectorSet>> groupByChunkingKey(List<VectorSet> vectorSets) {
        return vectorSets.stream()
                .collect(Collectors.groupingBy(
                        vs -> new ChunkingKey(vs.getChunkerConfigId(), vs.getSourceField())));
    }

    // =========================================================================
    // Internal records
    // =========================================================================

    record ChunkingKey(String chunkerConfigId, String sourceField) {}

    record DirectiveChunkingKey(String chunkerConfigId, String sourceLabel) {}

    record DirectiveChunkingGroup(
            String sourceText,
            String sourceLabel,
            String chunkerConfigId,
            Struct chunkerConfig,
            List<EmbedderTarget> embedderTargets
    ) {}

    record EmbedderTarget(String embedderConfigId, Struct embedderConfig, String template) {}
}
