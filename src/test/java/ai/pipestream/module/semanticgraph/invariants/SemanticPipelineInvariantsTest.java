package ai.pipestream.module.semanticgraph.invariants;

import ai.pipestream.data.v1.CentroidMetadata;
import ai.pipestream.data.v1.ChunkEmbedding;
import ai.pipestream.data.v1.GranularityLevel;
import ai.pipestream.data.v1.NamedEmbedderConfig;
import ai.pipestream.data.v1.NlpDocumentAnalysis;
import ai.pipestream.data.v1.PipeDoc;
import ai.pipestream.data.v1.SearchMetadata;
import ai.pipestream.data.v1.SemanticChunk;
import ai.pipestream.data.v1.SemanticProcessingResult;
import ai.pipestream.data.v1.SourceFieldAnalytics;
import ai.pipestream.data.v1.VectorDirective;
import ai.pipestream.data.v1.VectorSetDirectives;
import com.google.protobuf.Value;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for {@link SemanticPipelineInvariants}. Covers every failure
 * path of {@link SemanticPipelineInvariants#assertPostEmbedder} and
 * {@link SemanticPipelineInvariants#assertPostSemanticGraph}, plus the
 * happy paths.
 */
class SemanticPipelineInvariantsTest {

    // ======================================================================
    // assertPostEmbedder — happy paths
    // ======================================================================

    @Test
    void assertPostEmbedder_validStage2Doc_returnsNull() {
        PipeDoc doc = validStage2();
        assertThat(SemanticPipelineInvariants.assertPostEmbedder(doc))
                .as("A well-formed Stage-2 doc must validate").isNull();
    }

    @Test
    void assertPostEmbedder_emptyResultsList_isValid() {
        // §5.2: when no source text matched a directive, zero SPRs is valid
        // and the downstream steps pass-through.
        PipeDoc doc = PipeDoc.newBuilder()
                .setDocId("empty-doc")
                .setSearchMetadata(SearchMetadata.getDefaultInstance())
                .build();
        assertThat(SemanticPipelineInvariants.assertPostEmbedder(doc))
                .as("Empty semantic_results[] is valid per §5.2").isNull();
    }

    // ======================================================================
    // assertPostEmbedder — failure paths
    // ======================================================================

    @Test
    void assertPostEmbedder_nullDoc() {
        assertThat(SemanticPipelineInvariants.assertPostEmbedder(null))
                .as("Null doc must yield a message").contains("PipeDoc is null");
    }

    @Test
    void assertPostEmbedder_noSearchMetadata() {
        PipeDoc doc = PipeDoc.newBuilder().setDocId("x").build();
        assertThat(SemanticPipelineInvariants.assertPostEmbedder(doc))
                .as("Missing search_metadata").contains("search_metadata not set");
    }

    @Test
    void assertPostEmbedder_sprWithEmptyEmbeddingConfigId() {
        SemanticProcessingResult placeholder = validStage2Spr().toBuilder()
                .setEmbeddingConfigId("")
                .build();
        PipeDoc doc = wrap(placeholder, advertisedEmbedder("minilm"));
        assertThat(SemanticPipelineInvariants.assertPostEmbedder(doc))
                .as("Placeholder SPR at Stage 2 is a violation")
                .contains("embedding_config_id").contains("§5.2");
    }

    @Test
    void assertPostEmbedder_sprChunkWithEmptyVector() {
        SemanticChunk bad = validChunk().toBuilder()
                .setEmbeddingInfo(validChunk().getEmbeddingInfo().toBuilder().clearVector().build())
                .build();
        SemanticProcessingResult spr = validStage2Spr().toBuilder()
                .clearChunks().addChunks(bad).build();
        PipeDoc doc = wrap(spr, advertisedEmbedder("minilm"));
        assertThat(SemanticPipelineInvariants.assertPostEmbedder(doc))
                .as("Chunk without a vector is a §22.5 regression")
                .contains("empty vector").contains("§22.5");
    }

    @Test
    void assertPostEmbedder_sprMissingDirectiveKey() {
        SemanticProcessingResult spr = validStage2Spr().toBuilder()
                .clearMetadata().build();
        PipeDoc doc = wrap(spr, advertisedEmbedder("minilm"));
        assertThat(SemanticPipelineInvariants.assertPostEmbedder(doc))
                .as("Missing directive_key violates §21.2")
                .contains("directive_key").contains("§21.2");
    }

    @Test
    void assertPostEmbedder_embeddingConfigIdNotAdvertised() {
        SemanticProcessingResult spr = validStage2Spr().toBuilder()
                .setEmbeddingConfigId("ghost").build();
        PipeDoc doc = wrap(spr, advertisedEmbedder("minilm"));
        assertThat(SemanticPipelineInvariants.assertPostEmbedder(doc))
                .as("Embedding config not advertised in directives")
                .contains("ghost").contains("does not match");
    }

    @Test
    void assertPostEmbedder_noNlpAnalysisOnAnySprForSourceField() {
        SemanticProcessingResult spr = validStage2Spr().toBuilder()
                .clearNlpAnalysis().build();
        PipeDoc doc = wrap(spr, advertisedEmbedder("minilm"));
        assertThat(SemanticPipelineInvariants.assertPostEmbedder(doc))
                .as("nlp_analysis must be preserved per §5.2")
                .contains("nlp_analysis").contains("body");
    }

    @Test
    void assertPostEmbedder_missingSourceFieldAnalytics() {
        SemanticProcessingResult spr = validStage2Spr();
        PipeDoc doc = PipeDoc.newBuilder()
                .setDocId("doc-x")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .addSemanticResults(spr)
                        .setVectorSetDirectives(advertisedEmbedder("minilm"))
                        // intentionally NO source_field_analytics
                        .build())
                .build();
        assertThat(SemanticPipelineInvariants.assertPostEmbedder(doc))
                .contains("source_field_analytics").contains("§5.2");
    }

    @Test
    void assertPostEmbedder_sprsNotLexSorted() {
        SemanticProcessingResult a = validStage2Spr().toBuilder()
                .setResultId("ra").setChunkConfigId("zzz").build();
        SemanticProcessingResult b = validStage2Spr().toBuilder()
                .setResultId("rb").setChunkConfigId("aaa").build();
        PipeDoc doc = PipeDoc.newBuilder()
                .setDocId("doc-x")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .addSemanticResults(a) // zzz first — violates lex
                        .addSemanticResults(b)
                        .setVectorSetDirectives(advertisedEmbedder("minilm"))
                        .addSourceFieldAnalytics(SourceFieldAnalytics.newBuilder()
                                .setSourceField("body").setChunkConfigId("zzz").build())
                        .addSourceFieldAnalytics(SourceFieldAnalytics.newBuilder()
                                .setSourceField("body").setChunkConfigId("aaa").build())
                        .build())
                .build();
        assertThat(SemanticPipelineInvariants.assertPostEmbedder(doc))
                .contains("not lex-sorted").contains("§21.8");
    }

    @Test
    void assertPostEmbedder_sprWithZeroChunks() {
        SemanticProcessingResult spr = validStage2Spr().toBuilder().clearChunks().build();
        PipeDoc doc = wrap(spr, advertisedEmbedder("minilm"));
        assertThat(SemanticPipelineInvariants.assertPostEmbedder(doc))
                .contains("zero chunks");
    }

    // ======================================================================
    // assertPostSemanticGraph — happy paths
    // ======================================================================

    @Test
    void assertPostSemanticGraph_validDocCentroidAppended_returnsNull() {
        SemanticProcessingResult stage2 = validStage2Spr();
        SemanticProcessingResult docCentroid = validCentroidSpr(
                "body", "document_centroid", "minilm",
                GranularityLevel.GRANULARITY_LEVEL_DOCUMENT, /*count*/ 1);

        PipeDoc doc = PipeDoc.newBuilder()
                .setDocId("doc-x")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .addSemanticResults(docCentroid)  // "document_centroid" sorts before "sentence_v1"
                        .addSemanticResults(stage2)
                        .setVectorSetDirectives(advertisedEmbedder("minilm"))
                        .addSourceFieldAnalytics(SourceFieldAnalytics.newBuilder()
                                .setSourceField("body").setChunkConfigId("sentence_v1").build())
                        .build())
                .build();

        assertThat(SemanticPipelineInvariants.assertPostSemanticGraph(doc))
                .as("Stage 2 + document centroid must validate").isNull();
    }

    @Test
    void assertPostSemanticGraph_validBoundaryAppended_returnsNull() {
        SemanticProcessingResult stage2 = validStage2Spr();
        SemanticProcessingResult boundary = validBoundarySpr("body", "minilm", 3);
        PipeDoc doc = PipeDoc.newBuilder()
                .setDocId("doc-x")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .addSemanticResults(boundary) // "semantic" sorts before "sentence_v1"
                        .addSemanticResults(stage2)
                        .setVectorSetDirectives(advertisedEmbedder("minilm"))
                        .addSourceFieldAnalytics(SourceFieldAnalytics.newBuilder()
                                .setSourceField("body").setChunkConfigId("sentence_v1").build())
                        .build())
                .build();
        assertThat(SemanticPipelineInvariants.assertPostSemanticGraph(doc)).isNull();
    }

    // ======================================================================
    // assertPostSemanticGraph — centroid failures
    // ======================================================================

    @Test
    void assertPostSemanticGraph_centroidMissingMetadata() {
        SemanticProcessingResult centroid = validCentroidSpr(
                "body", "document_centroid", "minilm",
                GranularityLevel.GRANULARITY_LEVEL_DOCUMENT, 1)
                .toBuilder().clearCentroidMetadata().build();
        PipeDoc doc = wrap(centroid, advertisedEmbedder("minilm"));
        assertThat(SemanticPipelineInvariants.assertPostSemanticGraph(doc))
                .contains("centroid").contains("centroid_metadata");
    }

    @Test
    void assertPostSemanticGraph_centroidGranularityUnspecified() {
        SemanticProcessingResult centroid = validCentroidSpr(
                "body", "document_centroid", "minilm",
                GranularityLevel.GRANULARITY_LEVEL_UNSPECIFIED, 1);
        PipeDoc doc = wrap(centroid, advertisedEmbedder("minilm"));
        assertThat(SemanticPipelineInvariants.assertPostSemanticGraph(doc))
                .contains("UNSPECIFIED granularity");
    }

    @Test
    void assertPostSemanticGraph_centroidSourceVectorCountZero() {
        SemanticProcessingResult centroid = validCentroidSpr(
                "body", "document_centroid", "minilm",
                GranularityLevel.GRANULARITY_LEVEL_DOCUMENT, 0);
        PipeDoc doc = wrap(centroid, advertisedEmbedder("minilm"));
        assertThat(SemanticPipelineInvariants.assertPostSemanticGraph(doc))
                .contains("source_vector_count=0");
    }

    @Test
    void assertPostSemanticGraph_centroidMoreThanOneChunk() {
        SemanticProcessingResult base = validCentroidSpr(
                "body", "document_centroid", "minilm",
                GranularityLevel.GRANULARITY_LEVEL_DOCUMENT, 1);
        SemanticProcessingResult twoChunks = base.toBuilder().addChunks(validChunk()).build();
        PipeDoc doc = wrap(twoChunks, advertisedEmbedder("minilm"));
        assertThat(SemanticPipelineInvariants.assertPostSemanticGraph(doc))
                .contains("2 chunks");
    }

    // ======================================================================
    // assertPostSemanticGraph — boundary failures
    // ======================================================================

    @Test
    void assertPostSemanticGraph_boundaryMissingGranularity() {
        SemanticProcessingResult boundary = validBoundarySpr("body", "minilm", 1)
                .toBuilder().clearGranularity().build();
        PipeDoc doc = wrap(boundary, advertisedEmbedder("minilm"));
        assertThat(SemanticPipelineInvariants.assertPostSemanticGraph(doc))
                .contains("no granularity");
    }

    @Test
    void assertPostSemanticGraph_boundaryWrongGranularity() {
        SemanticProcessingResult boundary = validBoundarySpr("body", "minilm", 1).toBuilder()
                .setGranularity(GranularityLevel.GRANULARITY_LEVEL_SENTENCE).build();
        PipeDoc doc = wrap(boundary, advertisedEmbedder("minilm"));
        assertThat(SemanticPipelineInvariants.assertPostSemanticGraph(doc))
                .contains("must be GRANULARITY_LEVEL_SEMANTIC_CHUNK");
    }

    @Test
    void assertPostSemanticGraph_boundaryMissingSemanticConfigId() {
        SemanticProcessingResult boundary = validBoundarySpr("body", "minilm", 1).toBuilder()
                .clearSemanticConfigId().build();
        PipeDoc doc = wrap(boundary, advertisedEmbedder("minilm"));
        assertThat(SemanticPipelineInvariants.assertPostSemanticGraph(doc))
                .contains("semantic_config_id is empty");
    }

    @Test
    void assertPostSemanticGraph_boundaryExceedsHardCap() {
        SemanticProcessingResult boundary = validBoundarySpr("body", "minilm",
                SemanticPipelineInvariants.MAX_SEMANTIC_CHUNKS_PER_DOC_DEFAULT + 1);
        PipeDoc doc = wrap(boundary, advertisedEmbedder("minilm"));
        assertThat(SemanticPipelineInvariants.assertPostSemanticGraph(doc))
                .contains("exceeds default hard cap");
    }

    // ======================================================================
    // Fixture helpers
    // ======================================================================

    private static PipeDoc wrap(SemanticProcessingResult spr, VectorSetDirectives directives) {
        return PipeDoc.newBuilder()
                .setDocId("doc-x")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .addSemanticResults(spr)
                        .setVectorSetDirectives(directives)
                        .addSourceFieldAnalytics(SourceFieldAnalytics.newBuilder()
                                .setSourceField(spr.getSourceFieldName())
                                .setChunkConfigId(spr.getChunkConfigId())
                                .build())
                        .build())
                .build();
    }

    private static VectorSetDirectives advertisedEmbedder(String configId) {
        return VectorSetDirectives.newBuilder()
                .addDirectives(VectorDirective.newBuilder()
                        .setSourceLabel("body")
                        .setCelSelector("document.body")
                        .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId(configId).build())
                        .build())
                .build();
    }

    private static SemanticChunk validChunk() {
        return SemanticChunk.newBuilder()
                .setChunkId("c0")
                .setChunkNumber(0)
                .setEmbeddingInfo(ChunkEmbedding.newBuilder()
                        .setTextContent("The quick brown fox.")
                        .setOriginalCharStartOffset(0)
                        .setOriginalCharEndOffset(20)
                        .addVector(0.1f).addVector(0.2f).addVector(0.3f).addVector(0.4f)
                        .build())
                .build();
    }

    private static SemanticProcessingResult validStage2Spr() {
        return SemanticProcessingResult.newBuilder()
                .setResultId("stage2:doc-x:body:sentence_v1:minilm")
                .setSourceFieldName("body")
                .setChunkConfigId("sentence_v1")
                .setEmbeddingConfigId("minilm")
                .addChunks(validChunk())
                .setNlpAnalysis(NlpDocumentAnalysis.getDefaultInstance())
                .putMetadata("directive_key",
                        Value.newBuilder().setStringValue("dk-fixture-001").build())
                .build();
    }

    private static PipeDoc validStage2() {
        return wrap(validStage2Spr(), advertisedEmbedder("minilm"));
    }

    private static SemanticProcessingResult validCentroidSpr(
            String source, String chunkConfig, String embedder,
            GranularityLevel granularity, int sourceVectorCount) {
        return SemanticProcessingResult.newBuilder()
                .setResultId("stage3-centroid-doc:doc-x:" + source + ":" + chunkConfig + ":" + embedder)
                .setSourceFieldName(source)
                .setChunkConfigId(chunkConfig)
                .setEmbeddingConfigId(embedder)
                .addChunks(validChunk())
                .putMetadata("directive_key",
                        Value.newBuilder().setStringValue("dk-fixture-001").build())
                .setCentroidMetadata(CentroidMetadata.newBuilder()
                        .setGranularity(granularity)
                        .setSourceVectorCount(sourceVectorCount)
                        .build())
                .build();
    }

    private static SemanticProcessingResult validBoundarySpr(
            String source, String embedder, int chunkCount) {
        SemanticProcessingResult.Builder b = SemanticProcessingResult.newBuilder()
                .setResultId("stage3-boundary:doc-x:" + source + ":semantic:" + embedder)
                .setSourceFieldName(source)
                .setChunkConfigId(SemanticPipelineInvariants.SEMANTIC_CHUNK_CONFIG_ID)
                .setEmbeddingConfigId(embedder)
                .setGranularity(GranularityLevel.GRANULARITY_LEVEL_SEMANTIC_CHUNK)
                .setSemanticConfigId("semantic:" + embedder)
                .putMetadata("directive_key",
                        Value.newBuilder().setStringValue("dk-fixture-001").build());
        for (int i = 0; i < chunkCount; i++) {
            b.addChunks(validChunk().toBuilder().setChunkId("semantic:" + i).build());
        }
        return b.build();
    }
}
