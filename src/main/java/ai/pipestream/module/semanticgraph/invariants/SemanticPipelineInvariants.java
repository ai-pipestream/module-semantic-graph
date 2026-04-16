package ai.pipestream.module.semanticgraph.invariants;

import ai.pipestream.data.v1.CentroidMetadata;
import ai.pipestream.data.v1.ChunkEmbedding;
import ai.pipestream.data.v1.GranularityLevel;
import ai.pipestream.data.v1.NamedEmbedderConfig;
import ai.pipestream.data.v1.PipeDoc;
import ai.pipestream.data.v1.SearchMetadata;
import ai.pipestream.data.v1.SemanticChunk;
import ai.pipestream.data.v1.SemanticProcessingResult;
import ai.pipestream.data.v1.SourceFieldAnalytics;
import ai.pipestream.data.v1.VectorDirective;
import com.google.protobuf.Value;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Stage-invariant checkers for module-semantic-graph per DESIGN.md §5.2
 * (R3's input gate) and §5.3 (R3's output self-check).
 *
 * <h2>Contract</h2>
 * <p>Each method returns {@code null} when the doc satisfies the invariant,
 * or a concise one-line error message when it doesn't. The caller maps
 * non-null returns to gRPC status:
 * <ul>
 *   <li>Pre-condition failures (input gate) → {@code FAILED_PRECONDITION}</li>
 *   <li>Self-check failures on R3's own output → {@code FAILED_PRECONDITION}
 *       (defensive: means R3 produced an invalid shape and should be treated
 *       as a module bug, quarantine the doc, do not emit downstream)</li>
 * </ul>
 *
 * <p>Inlined per-consumer per the pattern in {@code pipestream-wiremock-server}'s
 * {@code SemanticPipelineInvariants} (tests). The test-source version uses
 * AssertJ and throws {@code AssertionError}; this main-source version returns
 * {@code String} so production code doesn't pull in a test framework.
 *
 * <p>All methods are pure. Safe for concurrent use.
 */
public final class SemanticPipelineInvariants {

    /** §21.8 lex comparator. */
    private static final Comparator<SemanticProcessingResult> LEX = Comparator
            .comparing(SemanticProcessingResult::getSourceFieldName)
            .thenComparing(SemanticProcessingResult::getChunkConfigId)
            .thenComparing(SemanticProcessingResult::getEmbeddingConfigId)
            .thenComparing(SemanticProcessingResult::getResultId);

    /** Metadata key carrying the §21.2 directive_key stamp. */
    public static final String DIRECTIVE_KEY_METADATA = "directive_key";

    /** Default cap on semantic-boundary chunks per doc per DESIGN.md §6.3. */
    public static final int MAX_SEMANTIC_CHUNKS_PER_DOC_DEFAULT = 50;

    /** {@code chunk_config_id} suffix marking a centroid SPR. */
    public static final String CENTROID_SUFFIX = "_centroid";

    /** {@code chunk_config_id} literal for a semantic-boundary SPR. */
    public static final String SEMANTIC_CHUNK_CONFIG_ID = "semantic";

    private SemanticPipelineInvariants() {}

    /**
     * Validates the post-embedder (Stage-2) shape per DESIGN.md §5.2. R3 calls
     * this on its input {@link PipeDoc} before processing; a non-null return
     * is mapped to {@code FAILED_PRECONDITION}.
     *
     * <p>Empty {@code semantic_results[]} is valid (per §5.2, when no source
     * text matched any directive the upstream chunker and embedder both
     * pass-through with zero SPRs). R3 inherits that interpretation; a
     * zero-SPR doc flows through R3 unchanged.
     *
     * @return {@code null} if valid; otherwise a one-line error message
     */
    public static String assertPostEmbedder(PipeDoc doc) {
        if (doc == null) {
            return "post-embedder: PipeDoc is null";
        }
        if (!doc.hasSearchMetadata()) {
            return "post-embedder: search_metadata not set";
        }
        SearchMetadata sm = doc.getSearchMetadata();
        List<SemanticProcessingResult> sprs = sm.getSemanticResultsList();

        // When directives are still present on the doc, cross-check that every
        // SPR's embedding_config_id was actually advertised. When they've been
        // cleared (some engine configs do this post-processing) skip the check.
        Set<String> advertisedEmbedderConfigIds = Collections.emptySet();
        if (sm.hasVectorSetDirectives()) {
            advertisedEmbedderConfigIds = new HashSet<>();
            for (VectorDirective d : sm.getVectorSetDirectives().getDirectivesList()) {
                for (NamedEmbedderConfig nec : d.getEmbedderConfigsList()) {
                    advertisedEmbedderConfigIds.add(nec.getConfigId());
                }
            }
        }

        Map<String, Boolean> nlpBySourceField = new HashMap<>();
        Set<String> sourceFieldConfigPairs = new HashSet<>();

        for (int i = 0; i < sprs.size(); i++) {
            SemanticProcessingResult spr = sprs.get(i);
            String sprCtx = "SPR[" + i + "]";
            String srcField = spr.getSourceFieldName();
            String chunkConfigId = spr.getChunkConfigId();
            String embedderConfigId = spr.getEmbeddingConfigId();

            if (srcField.isEmpty()) {
                return "post-embedder: " + sprCtx + " source_field_name is empty";
            }
            if (chunkConfigId.isEmpty()) {
                return "post-embedder: " + sprCtx + " (source='" + srcField
                        + "') chunk_config_id is empty";
            }
            // §5.2: every SPR MUST be fully embedded (non-empty embedding_config_id).
            if (embedderConfigId.isEmpty()) {
                return "post-embedder: " + sprCtx + " (source='" + srcField
                        + "', chunk='" + chunkConfigId
                        + "') has empty embedding_config_id — §5.2 requires "
                        + "all SPRs at Stage 2 to be fully embedded (no placeholders)";
            }
            if (!advertisedEmbedderConfigIds.isEmpty()
                    && !advertisedEmbedderConfigIds.contains(embedderConfigId)) {
                return "post-embedder: " + sprCtx + " (source='" + srcField
                        + "') embedding_config_id='" + embedderConfigId
                        + "' does not match any NamedEmbedderConfig advertised in "
                        + "vector_set_directives — §5.2 cross-check failed";
            }

            // §21.2 directive_key preserved from Stage 1.
            if (!spr.containsMetadata(DIRECTIVE_KEY_METADATA)) {
                return "post-embedder: " + sprCtx + " (source='" + srcField
                        + "') is missing metadata['" + DIRECTIVE_KEY_METADATA
                        + "'] — §21.2 directive_key MUST be preserved through all stages";
            }
            Value dkValue = spr.getMetadataMap().get(DIRECTIVE_KEY_METADATA);
            if (dkValue == null
                    || dkValue.getKindCase() != Value.KindCase.STRING_VALUE
                    || dkValue.getStringValue().isEmpty()) {
                return "post-embedder: " + sprCtx + " (source='" + srcField
                        + "') has empty or non-string metadata['" + DIRECTIVE_KEY_METADATA
                        + "'] — §21.2 directive_key MUST be a non-empty string";
            }

            List<SemanticChunk> chunks = spr.getChunksList();
            if (chunks.isEmpty()) {
                return "post-embedder: " + sprCtx + " (source='" + srcField
                        + "', chunk='" + chunkConfigId + "', embedder='" + embedderConfigId
                        + "') has zero chunks — §5.2 every SPR must carry ≥1 chunk";
            }
            for (int j = 0; j < chunks.size(); j++) {
                SemanticChunk chunk = chunks.get(j);
                String chunkCtx = sprCtx + " chunk[" + j + "]";
                if (chunk.getChunkId().isEmpty()) {
                    return "post-embedder: " + chunkCtx + " chunk_id is empty";
                }
                if (!chunk.hasEmbeddingInfo()) {
                    return "post-embedder: " + chunkCtx + " (id='" + chunk.getChunkId()
                            + "') has no embedding_info";
                }
                ChunkEmbedding emb = chunk.getEmbeddingInfo();
                if (emb.getTextContent().isEmpty()) {
                    return "post-embedder: " + chunkCtx + " (id='" + chunk.getChunkId()
                            + "') text_content is empty (must be preserved from stage 1)";
                }
                if (emb.getVectorCount() == 0) {
                    return "post-embedder: " + chunkCtx + " (id='" + chunk.getChunkId()
                            + "') has empty vector — §5.2 every chunk must be embedded "
                            + "at stage 2 (§22.5 regression gate)";
                }
                int start = emb.getOriginalCharStartOffset();
                int end = emb.getOriginalCharEndOffset();
                if (start < 0 || end < 0) {
                    return "post-embedder: " + chunkCtx + " (id='" + chunk.getChunkId()
                            + "') has negative offsets [" + start + ", " + end + "]";
                }
                if (start > end) {
                    return "post-embedder: " + chunkCtx + " (id='" + chunk.getChunkId()
                            + "') start > end [" + start + ", " + end + "]";
                }
            }

            nlpBySourceField.merge(srcField, spr.hasNlpAnalysis(), Boolean::logicalOr);
            sourceFieldConfigPairs.add(srcField + "|" + chunkConfigId);
        }

        // nlp_analysis preserved: at least one SPR per source_field must have it.
        for (Map.Entry<String, Boolean> e : nlpBySourceField.entrySet()) {
            if (!e.getValue()) {
                return "post-embedder: no SPR for source_field_name='" + e.getKey()
                        + "' carries nlp_analysis — §5.2 requires nlp_analysis "
                        + "preservation from stage 1";
            }
        }

        // source_field_analytics preserved.
        Set<String> analyticsPairs = new HashSet<>();
        for (SourceFieldAnalytics sfa : sm.getSourceFieldAnalyticsList()) {
            analyticsPairs.add(sfa.getSourceField() + "|" + sfa.getChunkConfigId());
        }
        for (String pair : sourceFieldConfigPairs) {
            if (!analyticsPairs.contains(pair)) {
                String[] parts = pair.split("\\|", 2);
                return "post-embedder: source_field_analytics[] is missing entry for "
                        + "(source_field='" + parts[0] + "', chunk_config_id='" + parts[1]
                        + "') — §5.2 requires analytics preservation from stage 1";
            }
        }

        // §21.8 lex sort.
        for (int i = 1; i < sprs.size(); i++) {
            if (LEX.compare(sprs.get(i - 1), sprs.get(i)) > 0) {
                return "post-embedder: semantic_results[] not lex-sorted (§21.8) at index " + i;
            }
        }
        return null;
    }

    /**
     * Validates the post-semantic-graph (Stage-3) shape per DESIGN.md §5.3. R3
     * calls this on its OUTPUT {@link PipeDoc} as a defensive self-check before
     * emitting to the engine. A non-null return indicates R3 produced an
     * invalid shape — that's a module bug; the caller maps it to
     * {@code FAILED_PRECONDITION} + quarantine.
     *
     * <p>This method cannot verify byte-identity of Stage-2 preservation (needs
     * the Stage-2 input as context). Callers that need that check must do it
     * themselves by diffing the pre-append portion against their input.
     *
     * @return {@code null} if valid; otherwise a one-line error message
     */
    public static String assertPostSemanticGraph(PipeDoc doc) {
        return assertPostSemanticGraph(doc, MAX_SEMANTIC_CHUNKS_PER_DOC_DEFAULT);
    }

    /**
     * Variant of {@link #assertPostSemanticGraph(PipeDoc)} that takes the
     * runtime-configured boundary chunk cap. R3's own self-check passes the
     * effective {@code max_semantic_chunks_per_doc} from
     * {@link ai.pipestream.module.semanticgraph.config.SemanticGraphStepOptions}
     * so a deployment that legitimately raises the cap (for long-document
     * corpora like court opinions) doesn't get its own valid output rejected
     * by an invariant frozen to the default.
     *
     * <p>External callers without R3's options context should use the no-arg
     * overload, which checks against {@link #MAX_SEMANTIC_CHUNKS_PER_DOC_DEFAULT}.
     */
    public static String assertPostSemanticGraph(PipeDoc doc, int boundaryChunkCap) {
        if (doc == null) {
            return "post-graph: PipeDoc is null";
        }
        if (!doc.hasSearchMetadata()) {
            return "post-graph: search_metadata not set";
        }
        SearchMetadata sm = doc.getSearchMetadata();
        List<SemanticProcessingResult> sprs = sm.getSemanticResultsList();

        Set<String> advertisedEmbedderConfigIds = Collections.emptySet();
        if (sm.hasVectorSetDirectives()) {
            advertisedEmbedderConfigIds = new HashSet<>();
            for (VectorDirective d : sm.getVectorSetDirectives().getDirectivesList()) {
                for (NamedEmbedderConfig nec : d.getEmbedderConfigsList()) {
                    advertisedEmbedderConfigIds.add(nec.getConfigId());
                }
            }
        }

        Map<String, Boolean> nlpBySourceField = new HashMap<>();
        Set<String> stage1LikePairs = new HashSet<>();

        for (int i = 0; i < sprs.size(); i++) {
            SemanticProcessingResult spr = sprs.get(i);
            String sprCtx = "SPR[" + i + "]";
            String srcField = spr.getSourceFieldName();
            String chunkConfigId = spr.getChunkConfigId();
            String embedderConfigId = spr.getEmbeddingConfigId();

            // Structural checks mirror post-embedder's.
            if (srcField.isEmpty()) {
                return "post-graph: " + sprCtx + " source_field_name is empty";
            }
            if (chunkConfigId.isEmpty()) {
                return "post-graph: " + sprCtx + " (source='" + srcField
                        + "') chunk_config_id is empty";
            }
            if (embedderConfigId.isEmpty()) {
                return "post-graph: " + sprCtx + " (source='" + srcField
                        + "') embedding_config_id is empty — §5.3 requires all stage-3 SPRs "
                        + "to carry the embedder config they were produced for";
            }
            if (!advertisedEmbedderConfigIds.isEmpty()
                    && !advertisedEmbedderConfigIds.contains(embedderConfigId)) {
                return "post-graph: " + sprCtx + " (source='" + srcField
                        + "') embedding_config_id='" + embedderConfigId
                        + "' does not match any NamedEmbedderConfig advertised in "
                        + "vector_set_directives — §5.3 cross-check failed";
            }

            if (!spr.containsMetadata(DIRECTIVE_KEY_METADATA)) {
                return "post-graph: " + sprCtx + " (source='" + srcField
                        + "') is missing metadata['" + DIRECTIVE_KEY_METADATA
                        + "'] — §21.2 directive_key MUST be preserved through all stages";
            }
            Value dkValue = spr.getMetadataMap().get(DIRECTIVE_KEY_METADATA);
            if (dkValue == null
                    || dkValue.getKindCase() != Value.KindCase.STRING_VALUE
                    || dkValue.getStringValue().isEmpty()) {
                return "post-graph: " + sprCtx + " (source='" + srcField
                        + "') has empty or non-string metadata['" + DIRECTIVE_KEY_METADATA + "']";
            }

            List<SemanticChunk> chunks = spr.getChunksList();
            if (chunks.isEmpty()) {
                return "post-graph: " + sprCtx + " (source='" + srcField
                        + "') has zero chunks";
            }
            // Centroid chunks MAY carry empty text_content per DESIGN.md §4.3
            // ("text_content: '' OR a representative excerpt"). For Stage-1-shaped
            // SPRs preserved from Stage 2, text_content MUST be non-empty because
            // §5.2 required it. Apply the relaxation only to centroid-shaped
            // chunks; offsets are also optional for centroids (they describe the
            // source chunks, not a new span).
            boolean isCentroidSpr = chunkConfigId.endsWith(CENTROID_SUFFIX);
            for (int j = 0; j < chunks.size(); j++) {
                SemanticChunk chunk = chunks.get(j);
                String chunkCtx = sprCtx + " chunk[" + j + "]";
                if (chunk.getChunkId().isEmpty()) {
                    return "post-graph: " + chunkCtx + " chunk_id is empty";
                }
                if (!chunk.hasEmbeddingInfo()) {
                    return "post-graph: " + chunkCtx + " has no embedding_info";
                }
                ChunkEmbedding emb = chunk.getEmbeddingInfo();
                if (!isCentroidSpr && emb.getTextContent().isEmpty()) {
                    return "post-graph: " + chunkCtx + " text_content is empty";
                }
                if (emb.getVectorCount() == 0) {
                    return "post-graph: " + chunkCtx + " has empty vector — §5.3 every "
                            + "stage-3 chunk must carry a populated vector";
                }
                int start = emb.getOriginalCharStartOffset();
                int end = emb.getOriginalCharEndOffset();
                if (start < 0 || end < 0) {
                    return "post-graph: " + chunkCtx + " negative offsets ["
                            + start + ", " + end + "]";
                }
                if (start > end) {
                    return "post-graph: " + chunkCtx + " start > end ["
                            + start + ", " + end + "]";
                }
            }

            // Stage-3-specific checks keyed on chunk_config_id shape.
            if (chunkConfigId.endsWith(CENTROID_SUFFIX)) {
                if (!spr.hasCentroidMetadata()) {
                    return "post-graph: " + sprCtx + " (centroid, config='" + chunkConfigId
                            + "') has no centroid_metadata";
                }
                CentroidMetadata cm = spr.getCentroidMetadata();
                if (cm.getGranularity() == GranularityLevel.GRANULARITY_LEVEL_UNSPECIFIED) {
                    return "post-graph: " + sprCtx + " (centroid) has UNSPECIFIED granularity";
                }
                if (cm.getSourceVectorCount() <= 0) {
                    return "post-graph: " + sprCtx + " (centroid) source_vector_count="
                            + cm.getSourceVectorCount() + " must be strictly positive";
                }
                if (spr.getChunksCount() != 1) {
                    return "post-graph: " + sprCtx + " (centroid) has " + spr.getChunksCount()
                            + " chunks — centroid SPRs must carry exactly 1 chunk";
                }
            } else if (SEMANTIC_CHUNK_CONFIG_ID.equals(chunkConfigId)) {
                if (!spr.hasGranularity()) {
                    return "post-graph: " + sprCtx + " (boundary) has no granularity set";
                }
                if (spr.getGranularity() != GranularityLevel.GRANULARITY_LEVEL_SEMANTIC_CHUNK) {
                    return "post-graph: " + sprCtx + " (boundary) granularity="
                            + spr.getGranularity() + " must be GRANULARITY_LEVEL_SEMANTIC_CHUNK";
                }
                if (spr.getSemanticConfigId().isEmpty()) {
                    return "post-graph: " + sprCtx + " (boundary) semantic_config_id is empty";
                }
                if (spr.getChunksCount() > boundaryChunkCap) {
                    return "post-graph: " + sprCtx + " (boundary) chunk count "
                            + spr.getChunksCount() + " exceeds boundary chunk cap "
                            + boundaryChunkCap;
                }
            } else {
                // Stage-1-shaped SPR (preserved from Stage 2) — contributes to
                // the analytics preservation check.
                stage1LikePairs.add(srcField + "|" + chunkConfigId);
            }

            nlpBySourceField.merge(srcField, spr.hasNlpAnalysis(), Boolean::logicalOr);
        }

        // nlp_analysis preservation (at least one per source_field).
        for (Map.Entry<String, Boolean> e : nlpBySourceField.entrySet()) {
            if (!e.getValue()) {
                return "post-graph: no SPR for source_field_name='" + e.getKey()
                        + "' carries nlp_analysis — §5.3 requires preservation from stage 1";
            }
        }

        // source_field_analytics preservation — only for Stage-1-shaped pairs.
        Set<String> analyticsPairs = new HashSet<>();
        for (SourceFieldAnalytics sfa : sm.getSourceFieldAnalyticsList()) {
            analyticsPairs.add(sfa.getSourceField() + "|" + sfa.getChunkConfigId());
        }
        for (String pair : stage1LikePairs) {
            if (!analyticsPairs.contains(pair)) {
                String[] parts = pair.split("\\|", 2);
                return "post-graph: source_field_analytics[] missing entry for "
                        + "(source_field='" + parts[0] + "', chunk_config_id='" + parts[1]
                        + "') — §5.3 requires analytics preservation from stage 1";
            }
        }

        // §21.8 lex sort.
        for (int i = 1; i < sprs.size(); i++) {
            if (LEX.compare(sprs.get(i - 1), sprs.get(i)) > 0) {
                return "post-graph: semantic_results[] not lex-sorted (§21.8) at index " + i;
            }
        }
        return null;
    }
}
