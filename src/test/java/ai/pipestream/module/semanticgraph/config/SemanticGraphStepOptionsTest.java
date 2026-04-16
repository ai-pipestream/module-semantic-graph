package ai.pipestream.module.semanticgraph.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class SemanticGraphStepOptionsTest {

    private final ObjectMapper mapper = new ObjectMapper();

    // --- defaults ----------------------------------------------------------

    @Test
    void defaults_returnsAllNullsAndAllEffectiveAccessorsReturnSpecValues() {
        SemanticGraphStepOptions opts = SemanticGraphStepOptions.defaults();

        assertThat(opts.computeParagraphCentroids())
                .as("Raw paragraph-centroid flag is null in defaults")
                .isNull();
        assertThat(opts.computeSectionCentroids())
                .as("Raw section-centroid flag is null in defaults")
                .isNull();
        assertThat(opts.computeDocumentCentroid())
                .as("Raw document-centroid flag is null in defaults")
                .isNull();
        assertThat(opts.computeSemanticBoundaries())
                .as("Raw semantic-boundaries flag is null in defaults")
                .isNull();
        assertThat(opts.rawBoundaryEmbeddingModelId())
                .as("boundary_embedding_model_id has no default per §21.3")
                .isNull();

        assertThat(opts.effectiveComputeParagraphCentroids())
                .as("Paragraph centroids default to true per §6.3")
                .isTrue();
        assertThat(opts.effectiveComputeSectionCentroids())
                .as("Section centroids default to true per §6.3")
                .isTrue();
        assertThat(opts.effectiveComputeDocumentCentroid())
                .as("Document centroid defaults to true per §6.3")
                .isTrue();
        assertThat(opts.effectiveComputeSemanticBoundaries())
                .as("Semantic boundaries default to true per §6.3")
                .isTrue();
        assertThat(opts.effectiveMaxSemanticChunksPerDoc())
                .as("Max-semantic-chunks default is the §6.3 hard cap")
                .isEqualTo(SemanticGraphStepOptions.DEFAULT_MAX_SEMANTIC_CHUNKS_PER_DOC)
                .isEqualTo(50);
        assertThat(opts.effectiveBoundarySimilarityThreshold())
                .as("Boundary similarity threshold default per §6.3")
                .isEqualTo(0.5f);
        assertThat(opts.effectiveBoundaryPercentileThreshold())
                .as("Boundary percentile threshold default per §6.3")
                .isEqualTo(20);
        assertThat(opts.effectiveBoundaryMinSentencesPerChunk())
                .as("Boundary min sentences default per §6.3")
                .isEqualTo(2);
        assertThat(opts.effectiveBoundaryMaxSentencesPerChunk())
                .as("Boundary max sentences default per §6.3")
                .isEqualTo(30);
    }

    // --- snake_case parsing ------------------------------------------------

    @Test
    void parse_snakeCase_allFields() throws Exception {
        String json = """
                {
                  "compute_paragraph_centroids": false,
                  "compute_section_centroids": false,
                  "compute_document_centroid": true,
                  "compute_semantic_boundaries": true,
                  "boundary_embedding_model_id": "minilm",
                  "max_semantic_chunks_per_doc": 25,
                  "boundary_similarity_threshold": 0.42,
                  "boundary_percentile_threshold": 15,
                  "boundary_min_sentences_per_chunk": 3,
                  "boundary_max_sentences_per_chunk": 20
                }
                """;

        SemanticGraphStepOptions opts = mapper.readValue(json, SemanticGraphStepOptions.class);

        assertThat(opts.effectiveComputeParagraphCentroids()).as("paragraph centroids set to false").isFalse();
        assertThat(opts.effectiveComputeSectionCentroids()).as("section centroids set to false").isFalse();
        assertThat(opts.effectiveComputeDocumentCentroid()).as("document centroid set to true").isTrue();
        assertThat(opts.effectiveComputeSemanticBoundaries()).as("boundaries set to true").isTrue();
        assertThat(opts.rawBoundaryEmbeddingModelId()).as("model id parsed from snake_case").isEqualTo("minilm");
        assertThat(opts.effectiveMaxSemanticChunksPerDoc()).as("max-chunks override").isEqualTo(25);
        assertThat(opts.effectiveBoundarySimilarityThreshold()).as("similarity threshold override").isEqualTo(0.42f);
        assertThat(opts.effectiveBoundaryPercentileThreshold()).as("percentile threshold override").isEqualTo(15);
        assertThat(opts.effectiveBoundaryMinSentencesPerChunk()).as("min sentences override").isEqualTo(3);
        assertThat(opts.effectiveBoundaryMaxSentencesPerChunk()).as("max sentences override").isEqualTo(20);
    }

    // --- camelCase aliases -------------------------------------------------

    @Test
    void parse_camelCase_aliasesResolveToSameRecord() throws Exception {
        String json = """
                {
                  "computeParagraphCentroids": false,
                  "computeSectionCentroids": true,
                  "computeDocumentCentroid": false,
                  "computeSemanticBoundaries": true,
                  "boundaryEmbeddingModelId": "paraphrase-minilm",
                  "maxSemanticChunksPerDoc": 30,
                  "boundarySimilarityThreshold": 0.65,
                  "boundaryPercentileThreshold": 25,
                  "boundaryMinSentencesPerChunk": 4,
                  "boundaryMaxSentencesPerChunk": 12
                }
                """;

        SemanticGraphStepOptions opts = mapper.readValue(json, SemanticGraphStepOptions.class);

        assertThat(opts.effectiveComputeParagraphCentroids()).as("camelCase parses paragraph flag").isFalse();
        assertThat(opts.effectiveComputeSectionCentroids()).as("camelCase parses section flag").isTrue();
        assertThat(opts.effectiveComputeDocumentCentroid()).as("camelCase parses document flag").isFalse();
        assertThat(opts.effectiveComputeSemanticBoundaries()).as("camelCase parses boundaries flag").isTrue();
        assertThat(opts.rawBoundaryEmbeddingModelId()).as("camelCase parses model id").isEqualTo("paraphrase-minilm");
        assertThat(opts.effectiveMaxSemanticChunksPerDoc()).as("camelCase parses max-chunks").isEqualTo(30);
        assertThat(opts.effectiveBoundarySimilarityThreshold()).as("camelCase parses similarity").isEqualTo(0.65f);
        assertThat(opts.effectiveBoundaryPercentileThreshold()).as("camelCase parses percentile").isEqualTo(25);
        assertThat(opts.effectiveBoundaryMinSentencesPerChunk()).as("camelCase parses min sentences").isEqualTo(4);
        assertThat(opts.effectiveBoundaryMaxSentencesPerChunk()).as("camelCase parses max sentences").isEqualTo(12);
    }

    // --- ignore-unknown ----------------------------------------------------

    @Test
    void parse_ignoresUnknownFields() throws Exception {
        String json = """
                {
                  "compute_document_centroid": true,
                  "boundary_embedding_model_id": "minilm",
                  "compute_semantic_boundaries": true,
                  "future_field_we_havent_added_yet": "ignored",
                  "another_ghost": 42
                }
                """;

        SemanticGraphStepOptions opts = mapper.readValue(json, SemanticGraphStepOptions.class);

        assertThat(opts.effectiveComputeDocumentCentroid()).as("known field still parses").isTrue();
        assertThat(opts.rawBoundaryEmbeddingModelId()).as("known field still parses").isEqualTo("minilm");
    }

    // --- validateForUse: happy paths ---------------------------------------

    @Test
    void validateForUse_boundariesOnWithModelId_ok() {
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                true, true, true, true, "minilm",
                null, null, null, null, null, null, null, null, null);

        assertThat(opts.requireBoundaryEmbeddingModelId())
                .as("requireBoundaryEmbeddingModelId returns the configured value")
                .isEqualTo("minilm");
        opts.validateForUse(); // does not throw
    }

    @Test
    void validateForUse_boundariesOff_modelIdNotRequired() {
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                true, true, true, false, null,
                null, null, null, null, null, null, null, null, null);

        opts.validateForUse(); // does not throw — boundaries off, model id irrelevant
    }

    // --- validateForUse: §21.3 ---------------------------------------------

    @Test
    void validateForUse_boundariesOn_modelIdMissing_throws() {
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                true, true, true, true, null,
                null, null, null, null, null, null, null, null, null);

        assertThatExceptionOfType(SemanticGraphStepOptions.InvalidOptionsException.class)
                .as("§21.3 forbids running boundaries without an explicit model id")
                .isThrownBy(opts::validateForUse)
                .withMessageContaining("boundary_embedding_model_id is unset");
    }

    @Test
    void validateForUse_boundariesOn_modelIdBlank_throws() {
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                true, true, true, true, "   ",
                null, null, null, null, null, null, null, null, null);

        assertThatExceptionOfType(SemanticGraphStepOptions.InvalidOptionsException.class)
                .as("Blank model id is not a model id")
                .isThrownBy(opts::validateForUse);
    }

    @Test
    void validateForUse_boundariesDefaultOnByOmission_modelIdMissing_throws() {
        SemanticGraphStepOptions opts = SemanticGraphStepOptions.defaults();

        assertThatExceptionOfType(SemanticGraphStepOptions.InvalidOptionsException.class)
                .as("Boundaries default to true, so defaults() must fail validateForUse without a model id")
                .isThrownBy(opts::validateForUse);
    }

    // --- validateForUse: range checks --------------------------------------

    @Test
    void validateForUse_maxSemanticChunksPerDocZero_throws() {
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, false, null,
                0, null, null, null, null, null, null, null, null);

        assertThatExceptionOfType(SemanticGraphStepOptions.InvalidOptionsException.class)
                .as("Explicit zero is not a valid hard cap")
                .isThrownBy(opts::validateForUse)
                .withMessageContaining("max_semantic_chunks_per_doc");
    }

    @Test
    void validateForUse_minSentencesGreaterThanMax_throws() {
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, false, null,
                null, null, null, 30, 5, null, null, null, null);

        assertThatExceptionOfType(SemanticGraphStepOptions.InvalidOptionsException.class)
                .as("min > max is a contradiction the parser must catch")
                .isThrownBy(opts::validateForUse)
                .withMessageContaining("boundary_min_sentences_per_chunk");
    }

    @Test
    void validateForUse_percentileOutOfRange_throws() {
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, false, null,
                null, null, 150, null, null, null, null, null, null);

        assertThatExceptionOfType(SemanticGraphStepOptions.InvalidOptionsException.class)
                .as("Percentile must be in [0, 100]")
                .isThrownBy(opts::validateForUse)
                .withMessageContaining("boundary_percentile_threshold");
    }

    @Test
    void validateForUse_similarityThresholdOutOfRange_throws() {
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, false, null,
                null, 1.5f, null, null, null, null, null, null, null);

        assertThatExceptionOfType(SemanticGraphStepOptions.InvalidOptionsException.class)
                .as("Cosine similarity must be in [-1.0, 1.0]")
                .isThrownBy(opts::validateForUse)
                .withMessageContaining("boundary_similarity_threshold");
    }

    @Test
    void validateForUse_boundaryMinSentencesNegative_throws() {
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, false, null,
                null, null, null, -1, null, null, null, null, null);

        assertThatExceptionOfType(SemanticGraphStepOptions.InvalidOptionsException.class)
                .as("Negative min-sentences is invalid")
                .isThrownBy(opts::validateForUse)
                .withMessageContaining("boundary_min_sentences_per_chunk");
    }

    // --- requireBoundaryEmbeddingModelId direct -----------------------------

    @Test
    void requireBoundaryEmbeddingModelId_unsetThrows_setReturns() {
        SemanticGraphStepOptions unset = SemanticGraphStepOptions.defaults();
        assertThatThrownBy(unset::requireBoundaryEmbeddingModelId)
                .as("Unwrapping a missing model id must fail loudly")
                .isInstanceOf(SemanticGraphStepOptions.InvalidOptionsException.class);

        SemanticGraphStepOptions set = new SemanticGraphStepOptions(
                null, null, null, null, "minilm",
                null, null, null, null, null, null, null, null, null);
        assertThat(set.requireBoundaryEmbeddingModelId())
                .as("Set value flows through verbatim")
                .isEqualTo("minilm");
    }

    // --- effective accessors: explicit zero / negative override ------------

    @Test
    void effectiveMaxSemanticChunksPerDoc_negativeFallsBackToDefault() {
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                null, null, null, null, null,
                -10, null, null, null, null, null, null, null, null);

        assertThat(opts.effectiveMaxSemanticChunksPerDoc())
                .as("Negative value is not a meaningful cap; effective accessor falls back to the spec default")
                .isEqualTo(SemanticGraphStepOptions.DEFAULT_MAX_SEMANTIC_CHUNKS_PER_DOC);
    }

    // --- json schema sanity ------------------------------------------------

    @Test
    void getJsonV7Schema_isParseableAndReferencesEveryField() throws Exception {
        String schema = SemanticGraphStepOptions.getJsonV7Schema();
        var node = mapper.readTree(schema);

        assertThat(node.get("$schema").asText())
                .as("schema declares draft-07")
                .contains("draft-07");
        var properties = node.get("properties");
        assertThat(properties).as("schema has properties block").isNotNull();

        for (String field : new String[]{
                "compute_paragraph_centroids",
                "compute_section_centroids",
                "compute_document_centroid",
                "compute_semantic_boundaries",
                "boundary_embedding_model_id",
                "max_semantic_chunks_per_doc",
                "boundary_similarity_threshold",
                "boundary_percentile_threshold",
                "boundary_min_sentences_per_chunk",
                "boundary_max_sentences_per_chunk"
        }) {
            assertThat(properties.has(field))
                    .as("schema documents '%s'", field)
                    .isTrue();
        }
    }
}
