package ai.pipestream.module.semanticgraph.config;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.quarkus.runtime.annotations.RegisterForReflection;

/**
 * Typed parse of the semantic-graph step's {@code ProcessConfiguration.json_config}
 * (a {@code google.protobuf.Struct}), per DESIGN.md §6.3.
 *
 * <p>The semantic-graph step appends Stage-3 SPRs (centroids + optional
 * semantic-boundary chunks) to fully-embedded Stage-2 docs. This record holds
 * the small set of step-level knobs that every request reads. Per-directive
 * knobs (which boundary model, which centroid granularities) are pulled from
 * each {@code VectorDirective} and from this record's flags; infrastructure
 * knobs (DJL endpoint, retry budgets) live in {@code application.properties}.
 *
 * <p>Per DESIGN.md §21.1 there is no fallback. If JSON parse fails, the
 * caller MUST raise {@code INVALID_ARGUMENT}. Per DESIGN.md §21.3, if
 * {@code compute_semantic_boundaries} is true but {@code boundary_embedding_model_id}
 * is unset, the caller MUST raise {@code FAILED_PRECONDITION}; this record
 * exposes {@link #validateForUse()} so that check happens once, fail-fast,
 * before any downstream code touches the unset field.
 *
 * <p><b>Bidirectional JSON alias coverage:</b> every field carries both
 * {@code @JsonProperty(snake_case)} and {@code @JsonAlias(camelCase)}.
 * {@code ProcessConfiguration.json_config} is a {@code google.protobuf.Struct}
 * and {@code JsonFormat.printer()} passes Struct map keys through verbatim, so
 * whichever convention the caller used lands in Jackson unchanged. A
 * TypeScript admin form emitting {@code computeSemanticBoundaries} parses to
 * the same field as a Python client emitting {@code compute_semantic_boundaries}.
 *
 * <p>Defaults (§6.3): all four {@code compute_*} flags default to {@code true}
 * so an empty config produces full Stage-3 output; {@code maxSemanticChunksPerDoc}
 * defaults to {@value #DEFAULT_MAX_SEMANTIC_CHUNKS_PER_DOC} (the §6.3 hard cap).
 * {@code boundaryEmbeddingModelId} has <b>no default</b> — it is required when
 * boundaries are on and absent when they are off. The boundary-detection
 * thresholds default to the §6.3 values.
 *
 * <p>Usage:
 * <pre>{@code
 * String json = JsonFormat.printer().print(processConfig.getJsonConfig());
 * SemanticGraphStepOptions opts = mapper.readValue(json, SemanticGraphStepOptions.class);
 * opts.validateForUse();   // throws InvalidOptionsException on cross-field violations
 * }</pre>
 */
@RegisterForReflection
@JsonIgnoreProperties(ignoreUnknown = true)
public record SemanticGraphStepOptions(
        @JsonProperty("compute_paragraph_centroids") @JsonAlias("computeParagraphCentroids")
        Boolean computeParagraphCentroids,

        @JsonProperty("compute_section_centroids") @JsonAlias("computeSectionCentroids")
        Boolean computeSectionCentroids,

        @JsonProperty("compute_document_centroid") @JsonAlias("computeDocumentCentroid")
        Boolean computeDocumentCentroid,

        @JsonProperty("compute_semantic_boundaries") @JsonAlias("computeSemanticBoundaries")
        Boolean computeSemanticBoundaries,

        @JsonProperty("boundary_embedding_model_id") @JsonAlias("boundaryEmbeddingModelId")
        String boundaryEmbeddingModelId,

        @JsonProperty("max_semantic_chunks_per_doc") @JsonAlias("maxSemanticChunksPerDoc")
        Integer maxSemanticChunksPerDoc,

        @JsonProperty("boundary_similarity_threshold") @JsonAlias("boundarySimilarityThreshold")
        Float boundarySimilarityThreshold,

        @JsonProperty("boundary_percentile_threshold") @JsonAlias("boundaryPercentileThreshold")
        Integer boundaryPercentileThreshold,

        @JsonProperty("boundary_min_sentences_per_chunk") @JsonAlias("boundaryMinSentencesPerChunk")
        Integer boundaryMinSentencesPerChunk,

        @JsonProperty("boundary_max_sentences_per_chunk") @JsonAlias("boundaryMaxSentencesPerChunk")
        Integer boundaryMaxSentencesPerChunk
) {

    /** Default cap on semantic-boundary SPR chunks per doc (DESIGN.md §6.3). */
    public static final int DEFAULT_MAX_SEMANTIC_CHUNKS_PER_DOC = 50;

    /** Default cosine-similarity threshold below which a topic boundary is recorded. */
    public static final float DEFAULT_BOUNDARY_SIMILARITY_THRESHOLD = 0.5f;

    /** Default percentile-based threshold for boundary scoring (0–100). */
    public static final int DEFAULT_BOUNDARY_PERCENTILE_THRESHOLD = 20;

    /** Default minimum sentences merged into one semantic chunk. */
    public static final int DEFAULT_BOUNDARY_MIN_SENTENCES_PER_CHUNK = 2;

    /** Default maximum sentences merged into one semantic chunk. */
    public static final int DEFAULT_BOUNDARY_MAX_SENTENCES_PER_CHUNK = 30;

    public boolean effectiveComputeParagraphCentroids() {
        return computeParagraphCentroids == null || computeParagraphCentroids;
    }

    public boolean effectiveComputeSectionCentroids() {
        return computeSectionCentroids == null || computeSectionCentroids;
    }

    public boolean effectiveComputeDocumentCentroid() {
        return computeDocumentCentroid == null || computeDocumentCentroid;
    }

    public boolean effectiveComputeSemanticBoundaries() {
        return computeSemanticBoundaries == null || computeSemanticBoundaries;
    }

    /**
     * Returns the configured boundary-embedding model id verbatim. May be
     * {@code null} when {@link #effectiveComputeSemanticBoundaries()} is
     * {@code false}; when boundaries are enabled, callers MUST go through
     * {@link #requireBoundaryEmbeddingModelId()} or {@link #validateForUse()}.
     */
    public String rawBoundaryEmbeddingModelId() {
        return boundaryEmbeddingModelId;
    }

    /**
     * Returns the boundary-embedding model id, throwing
     * {@link InvalidOptionsException} if it is unset or blank. Use this at the
     * call site that needs the value, never store the unwrapped result before
     * boundaries are confirmed enabled.
     */
    public String requireBoundaryEmbeddingModelId() {
        if (boundaryEmbeddingModelId == null || boundaryEmbeddingModelId.isBlank()) {
            throw new InvalidOptionsException(
                    "compute_semantic_boundaries is true but boundary_embedding_model_id is unset; " +
                    "DESIGN.md §21.3 forbids 'first available model' fallback");
        }
        return boundaryEmbeddingModelId;
    }

    public int effectiveMaxSemanticChunksPerDoc() {
        return maxSemanticChunksPerDoc != null && maxSemanticChunksPerDoc > 0
                ? maxSemanticChunksPerDoc
                : DEFAULT_MAX_SEMANTIC_CHUNKS_PER_DOC;
    }

    public float effectiveBoundarySimilarityThreshold() {
        return boundarySimilarityThreshold != null
                ? boundarySimilarityThreshold
                : DEFAULT_BOUNDARY_SIMILARITY_THRESHOLD;
    }

    public int effectiveBoundaryPercentileThreshold() {
        return boundaryPercentileThreshold != null
                ? boundaryPercentileThreshold
                : DEFAULT_BOUNDARY_PERCENTILE_THRESHOLD;
    }

    public int effectiveBoundaryMinSentencesPerChunk() {
        return boundaryMinSentencesPerChunk != null && boundaryMinSentencesPerChunk > 0
                ? boundaryMinSentencesPerChunk
                : DEFAULT_BOUNDARY_MIN_SENTENCES_PER_CHUNK;
    }

    public int effectiveBoundaryMaxSentencesPerChunk() {
        return boundaryMaxSentencesPerChunk != null && boundaryMaxSentencesPerChunk > 0
                ? boundaryMaxSentencesPerChunk
                : DEFAULT_BOUNDARY_MAX_SENTENCES_PER_CHUNK;
    }

    /**
     * Asserts cross-field invariants per DESIGN.md §6.3 / §21.3. Callers
     * should invoke this once, immediately after parsing, and map any thrown
     * {@link InvalidOptionsException} to {@code Status.FAILED_PRECONDITION}.
     *
     * <p>Validations:
     * <ul>
     *   <li>{@code compute_semantic_boundaries=true} requires a non-blank
     *       {@code boundary_embedding_model_id}</li>
     *   <li>{@code max_semantic_chunks_per_doc}, when explicitly set, must
     *       be {@code > 0}</li>
     *   <li>{@code boundary_min_sentences_per_chunk <= boundary_max_sentences_per_chunk},
     *       when both are explicitly set</li>
     *   <li>{@code boundary_percentile_threshold}, when explicitly set, must
     *       be in {@code [0, 100]}</li>
     *   <li>{@code boundary_similarity_threshold}, when explicitly set, must
     *       be in {@code [-1.0, 1.0]} (cosine similarity range)</li>
     * </ul>
     */
    public void validateForUse() {
        if (effectiveComputeSemanticBoundaries()) {
            requireBoundaryEmbeddingModelId();
        }
        if (maxSemanticChunksPerDoc != null && maxSemanticChunksPerDoc <= 0) {
            throw new InvalidOptionsException(
                    "max_semantic_chunks_per_doc must be > 0 when explicitly set; got "
                            + maxSemanticChunksPerDoc);
        }
        if (boundaryMinSentencesPerChunk != null && boundaryMinSentencesPerChunk <= 0) {
            throw new InvalidOptionsException(
                    "boundary_min_sentences_per_chunk must be > 0 when explicitly set; got "
                            + boundaryMinSentencesPerChunk);
        }
        if (boundaryMaxSentencesPerChunk != null && boundaryMaxSentencesPerChunk <= 0) {
            throw new InvalidOptionsException(
                    "boundary_max_sentences_per_chunk must be > 0 when explicitly set; got "
                            + boundaryMaxSentencesPerChunk);
        }
        if (boundaryMinSentencesPerChunk != null
                && boundaryMaxSentencesPerChunk != null
                && boundaryMinSentencesPerChunk > boundaryMaxSentencesPerChunk) {
            throw new InvalidOptionsException(
                    "boundary_min_sentences_per_chunk (" + boundaryMinSentencesPerChunk
                            + ") must be <= boundary_max_sentences_per_chunk ("
                            + boundaryMaxSentencesPerChunk + ")");
        }
        if (boundaryPercentileThreshold != null
                && (boundaryPercentileThreshold < 0 || boundaryPercentileThreshold > 100)) {
            throw new InvalidOptionsException(
                    "boundary_percentile_threshold must be in [0, 100] when explicitly set; got "
                            + boundaryPercentileThreshold);
        }
        if (boundarySimilarityThreshold != null
                && (boundarySimilarityThreshold < -1.0f || boundarySimilarityThreshold > 1.0f)) {
            throw new InvalidOptionsException(
                    "boundary_similarity_threshold must be in [-1.0, 1.0] when explicitly set; got "
                            + boundarySimilarityThreshold);
        }
    }

    /**
     * Returns the canonical defaults instance — every field is {@code null},
     * so all {@code effective*()} accessors return their default values and
     * {@link #validateForUse()} fails because boundaries default to enabled
     * but no boundary model is set. Callers using {@link #defaults()} are
     * expected to override {@code boundaryEmbeddingModelId} OR set
     * {@code computeSemanticBoundaries=false} before calling
     * {@link #validateForUse()}.
     */
    public static SemanticGraphStepOptions defaults() {
        return new SemanticGraphStepOptions(null, null, null, null, null, null, null, null, null, null);
    }

    /**
     * Returns a JSON Schema draft-07 document describing the step config
     * shape per DESIGN.md §6.3. Exposed by
     * {@code SemanticGraphGrpcImpl.getServiceRegistration} so the admin UI /
     * testing-sidecar can auto-render a form. Every field appears under
     * snake_case (primary) and is tolerated under camelCase via
     * {@code @JsonAlias} at the parser level.
     */
    public static String getJsonV7Schema() {
        return """
                {
                  "$schema": "http://json-schema.org/draft-07/schema#",
                  "title": "SemanticGraphStepOptions",
                  "description": "Step-level config for module-semantic-graph (DESIGN.md §6.3). Toggles centroid + semantic-boundary outputs and configures the boundary-detection thresholds. boundary_embedding_model_id is REQUIRED when compute_semantic_boundaries=true; there is no model fallback (§21.3).",
                  "type": "object",
                  "additionalProperties": false,
                  "properties": {
                    "compute_paragraph_centroids": {
                      "type": "boolean",
                      "title": "Compute paragraph centroids",
                      "description": "Emit one paragraph_centroid SPR per (source_field, chunker, embedder) per paragraph. Detected from chunk offsets + NLP paragraph spans.",
                      "default": true
                    },
                    "compute_section_centroids": {
                      "type": "boolean",
                      "title": "Compute section centroids",
                      "description": "Emit one section_centroid SPR per (source_field, chunker, embedder) per Section in DocOutline. No-op when DocOutline is absent.",
                      "default": true
                    },
                    "compute_document_centroid": {
                      "type": "boolean",
                      "title": "Compute document centroid",
                      "description": "Emit one document_centroid SPR per (source_field, chunker, embedder).",
                      "default": true
                    },
                    "compute_semantic_boundaries": {
                      "type": "boolean",
                      "title": "Compute semantic boundaries",
                      "description": "Run topic-boundary detection on sentences_internal vectors and re-embed each detected group via boundary_embedding_model_id.",
                      "default": true
                    },
                    "boundary_embedding_model_id": {
                      "type": "string",
                      "title": "Boundary embedding model id",
                      "description": "DJL model id used to re-embed grouped boundary text. REQUIRED when compute_semantic_boundaries=true; FAILED_PRECONDITION if unset (DESIGN.md §21.3 forbids 'first available' fallback)."
                    },
                    "max_semantic_chunks_per_doc": {
                      "type": "integer",
                      "title": "Max semantic chunks per doc",
                      "description": "Hard cap on boundary SPR chunk count per document. Exceeding the cap raises INTERNAL — never silently truncates.",
                      "minimum": 1,
                      "default": 50
                    },
                    "boundary_similarity_threshold": {
                      "type": "number",
                      "title": "Boundary similarity threshold",
                      "description": "Cosine similarity between consecutive sentence vectors below which a boundary is recorded. Range [-1.0, 1.0].",
                      "minimum": -1.0,
                      "maximum": 1.0,
                      "default": 0.5
                    },
                    "boundary_percentile_threshold": {
                      "type": "integer",
                      "title": "Boundary percentile threshold",
                      "description": "Percentile-based threshold for boundary scoring (0–100). Used when the detector switches from absolute to percentile mode.",
                      "minimum": 0,
                      "maximum": 100,
                      "default": 20
                    },
                    "boundary_min_sentences_per_chunk": {
                      "type": "integer",
                      "title": "Boundary min sentences per chunk",
                      "description": "Minimum sentences merged into one semantic chunk.",
                      "minimum": 1,
                      "default": 2
                    },
                    "boundary_max_sentences_per_chunk": {
                      "type": "integer",
                      "title": "Boundary max sentences per chunk",
                      "description": "Maximum sentences merged into one semantic chunk. Must be >= boundary_min_sentences_per_chunk.",
                      "minimum": 1,
                      "default": 30
                    }
                  }
                }
                """;
    }

    /**
     * Thrown when {@link SemanticGraphStepOptions#validateForUse()} (or one of
     * the {@code requireXxx} accessors) detects a cross-field violation. The
     * caller maps this to {@code Status.FAILED_PRECONDITION} or
     * {@code INVALID_ARGUMENT} depending on which field tripped it (see
     * DESIGN.md §10.1).
     */
    public static final class InvalidOptionsException extends IllegalArgumentException {
        public InvalidOptionsException(String message) {
            super(message);
        }
    }
}
