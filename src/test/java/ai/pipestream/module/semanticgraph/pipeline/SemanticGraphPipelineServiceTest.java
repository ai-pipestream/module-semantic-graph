package ai.pipestream.module.semanticgraph.pipeline;

import ai.pipestream.data.v1.ChunkEmbedding;
import ai.pipestream.data.v1.DocOutline;
import ai.pipestream.data.v1.GranularityLevel;
import ai.pipestream.data.v1.NamedChunkerConfig;
import ai.pipestream.data.v1.NamedEmbedderConfig;
import ai.pipestream.data.v1.NlpDocumentAnalysis;
import ai.pipestream.data.v1.PipeDoc;
import ai.pipestream.data.v1.SearchMetadata;
import ai.pipestream.data.v1.Section;
import ai.pipestream.data.v1.SemanticChunk;
import ai.pipestream.data.v1.SemanticProcessingResult;
import ai.pipestream.data.v1.SourceFieldAnalytics;
import ai.pipestream.data.v1.VectorDirective;
import ai.pipestream.data.v1.VectorSetDirectives;
import ai.pipestream.module.semanticgraph.config.SemanticGraphStepOptions;
import ai.pipestream.module.semanticgraph.djl.SemanticGraphEmbedHelper;
import ai.pipestream.module.semanticgraph.invariants.SemanticPipelineInvariants;
import ai.pipestream.module.semanticgraph.metrics.SemanticGraphMetrics;
import com.google.protobuf.Struct;
import com.google.protobuf.Value;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.helpers.test.UniAssertSubscriber;
import org.junit.jupiter.api.Test;

import java.net.ConnectException;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link SemanticGraphPipelineService}. Uses a mocked
 * {@link SemanticGraphEmbedHelper} so tests don't touch DJL Serving.
 *
 * <p>Coverage matrix (DESIGN.md §10.1 + §5.2 + §5.3 + §7.3):
 * <ul>
 *   <li>Happy paths: flags on/off, with/without boundaries, with DocOutline,
 *       with and without resolvable source text</li>
 *   <li>Stage-2 byte-identity preservation: every input SPR appears unchanged
 *       in the output's pre-append portion</li>
 *   <li>Input-gate failures: null doc, missing search_metadata, placeholder
 *       SPRs, empty vectors</li>
 *   <li>Options violations: missing boundary_embedding_model_id with
 *       boundaries on; cross-field invariants</li>
 *   <li>Async boundary failures: model not loaded, no sentence-shaped SPR
 *       for the model, hard cap exceeded, DJL permanent/transient failures</li>
 *   <li>Lex sort invariant on output</li>
 * </ul>
 */
class SemanticGraphPipelineServiceTest {

    // ======================================================================
    // Happy paths
    // ======================================================================

    @Test
    void process_allFlagsOff_passThroughWithLexSort() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());

        PipeDoc input = buildValidStage2();
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, false, null,
                null, null, null, null, null, null, null, null, null);

        PipeDoc out = svc.process(input, opts, "test-step")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        assertThat(out.getSearchMetadata().getSemanticResultsCount())
                .as("With all flags off, no SPRs appended")
                .isEqualTo(input.getSearchMetadata().getSemanticResultsCount());
        assertThat(SemanticPipelineInvariants.assertPostSemanticGraph(out))
                .as("Output must satisfy post-graph invariant").isNull();
        // Stage-2 preservation: every input SPR present in output.
        assertStage2Preserved(input, out);
        verify(djl, never()).embed(anyString(), any(), anyInt(), anyInt(), anyInt(), anyLong());
    }

    @Test
    void process_documentCentroidOnly_emitsOneCentroidPerTriple() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());

        PipeDoc input = buildValidStage2();  // 1 Stage-2 SPR
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, true, false, null,
                null, null, null, null, null, null, null, null, null);

        PipeDoc out = svc.process(input, opts, "step")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        long centroidCount = out.getSearchMetadata().getSemanticResultsList().stream()
                .filter(s -> "document_centroid".equals(s.getChunkConfigId())).count();
        assertThat(centroidCount)
                .as("One document centroid per Stage-2 triple").isEqualTo(1);

        SemanticProcessingResult centroid = out.getSearchMetadata().getSemanticResultsList()
                .stream()
                .filter(s -> "document_centroid".equals(s.getChunkConfigId()))
                .findFirst().orElseThrow();
        assertThat(centroid.hasCentroidMetadata()).isTrue();
        assertThat(centroid.getCentroidMetadata().getGranularity())
                .isEqualTo(GranularityLevel.GRANULARITY_LEVEL_DOCUMENT);
        assertThat(centroid.getCentroidMetadata().getSourceVectorCount())
                .as("Source vector count = number of Stage-2 chunks").isEqualTo(3);
        assertThat(centroid.getChunksCount())
                .as("Document centroid SPR must have exactly 1 chunk").isEqualTo(1);
        assertThat(centroid.getChunks(0).getEmbeddingInfo().getVectorCount())
                .as("Centroid vector must have same dim as source chunks").isEqualTo(4);
        assertThat(centroid.getParentResultId())
                .as("Centroid parent_result_id links back to source SPR")
                .isEqualTo("stage2:doc-x:body:sentence_v1:minilm");
        assertStage2Preserved(input, out);
    }

    @Test
    void process_sectionCentroidsWithDocOutline() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());

        PipeDoc input = buildStage2WithDocOutline();
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, true, false, false, null,
                null, null, null, null, null, null, null, null, null);

        PipeDoc out = svc.process(input, opts, "step")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        List<SemanticProcessingResult> sectionCentroids = out.getSearchMetadata()
                .getSemanticResultsList().stream()
                .filter(s -> "section_centroid".equals(s.getChunkConfigId())).toList();
        assertThat(sectionCentroids)
                .as("One section centroid per Section in DocOutline with matching chunks")
                .hasSizeGreaterThanOrEqualTo(1);
        for (SemanticProcessingResult sc : sectionCentroids) {
            assertThat(sc.getCentroidMetadata().getGranularity())
                    .isEqualTo(GranularityLevel.GRANULARITY_LEVEL_SECTION);
            assertThat(sc.getCentroidMetadata().getSourceVectorCount()).isGreaterThan(0);
            assertThat(sc.getChunksCount()).isEqualTo(1);
        }
        assertStage2Preserved(input, out);
    }

    @Test
    void process_paragraphCentroids_skippedWhenSourceTextUnresolvable() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());

        // source_label = "body" but SearchMetadata.body is unset → unresolvable
        PipeDoc input = buildValidStage2();  // default stage2 has no body text set
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                true, false, false, false, null,
                null, null, null, null, null, null, null, null, null);

        PipeDoc out = svc.process(input, opts, "step")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        long paragraphCount = out.getSearchMetadata().getSemanticResultsList().stream()
                .filter(s -> "paragraph_centroid".equals(s.getChunkConfigId())).count();
        assertThat(paragraphCount)
                .as("Paragraph centroids skipped when source text cannot be resolved")
                .isZero();
    }

    @Test
    void process_paragraphCentroids_usingBodyFieldWithParagraphBreaks() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());

        // Build stage2 where the SPR contains 2 chunks separated by "\n\n" in body.
        String body = "Sentence one.\n\nSentence two.";
        //            0   456789...12 13 14 15 16 17...
        // chunk 0: [0, 13)  (ends before the \n\n)
        // chunk 1: [15, 28) (starts after the \n\n)
        SemanticChunk c0 = chunk("c0", 0, "Sentence one.", 0, 13,
                new float[]{0.1f, 0.2f, 0.3f, 0.4f});
        SemanticChunk c1 = chunk("c1", 1, "Sentence two.", 15, 28,
                new float[]{0.5f, 0.6f, 0.7f, 0.8f});
        SemanticProcessingResult spr = SemanticProcessingResult.newBuilder()
                .setResultId("stage2:doc-x:body:sentence_v1:minilm")
                .setSourceFieldName("body").setChunkConfigId("sentence_v1")
                .setEmbeddingConfigId("minilm")
                .addChunks(c0).addChunks(c1)
                .setNlpAnalysis(NlpDocumentAnalysis.getDefaultInstance())
                .putMetadata("directive_key",
                        Value.newBuilder().setStringValue("dk").build())
                .build();
        PipeDoc input = PipeDoc.newBuilder()
                .setDocId("doc-x")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody(body)
                        .addSemanticResults(spr)
                        .setVectorSetDirectives(advertisedEmbedderMinilm())
                        .addSourceFieldAnalytics(SourceFieldAnalytics.newBuilder()
                                .setSourceField("body").setChunkConfigId("sentence_v1").build())
                        .build())
                .build();

        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                true, false, false, false, null,
                null, null, null, null, null, null, null, null, null);

        PipeDoc out = svc.process(input, opts, "step")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        List<SemanticProcessingResult> paragraphs = out.getSearchMetadata()
                .getSemanticResultsList().stream()
                .filter(s -> "paragraph_centroid".equals(s.getChunkConfigId())).toList();
        assertThat(paragraphs)
                .as("Body with '\\n\\n' between chunk offsets produces 2 paragraph centroids")
                .hasSize(2);
        assertStage2Preserved(input, out);
    }

    @Test
    void process_boundaries_happyPath() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        when(djl.isModelLoaded(eq("minilm"))).thenReturn(Uni.createFrom().item(Boolean.TRUE));
        when(djl.embed(eq("minilm"), any(), anyInt(), anyInt(), anyInt(), anyLong()))
                .thenAnswer(inv -> {
                    List<String> texts = inv.getArgument(1);
                    List<float[]> vecs = new ArrayList<>();
                    for (int i = 0; i < texts.size(); i++) {
                        vecs.add(new float[]{(float) i, (float) i, (float) i, (float) i});
                    }
                    return Uni.createFrom().item(vecs);
                });

        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());
        // stage2 with sentences_internal + minilm, 3 sentence chunks
        PipeDoc input = buildStage2WithSentencesInternal();

        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, true, "minilm",
                null, null, null, null, null, null, null, null, null);

        PipeDoc out = svc.process(input, opts, "step")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        List<SemanticProcessingResult> boundaries = out.getSearchMetadata()
                .getSemanticResultsList().stream()
                .filter(s -> "semantic".equals(s.getChunkConfigId())).toList();
        assertThat(boundaries).as("Boundaries on → one boundary SPR per source").hasSize(1);
        SemanticProcessingResult b = boundaries.get(0);
        assertThat(b.getGranularity()).isEqualTo(GranularityLevel.GRANULARITY_LEVEL_SEMANTIC_CHUNK);
        assertThat(b.getSemanticConfigId()).isEqualTo("semantic:minilm");
        assertThat(b.getChunksCount())
                .as("Boundary chunk count ≤ sentences; with low-signal vectors usually 1 group")
                .isBetween(1, 3);
        assertStage2Preserved(input, out);
    }

    // ======================================================================
    // Input-gate failures (sync)
    // ======================================================================

    @Test
    void process_nullDoc_throwsIAE() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, false, null,
                null, null, null, null, null, null, null, null, null);
        assertThatExceptionOfType(IllegalArgumentException.class)
                .as("Null PipeDoc → IAE (INVALID_ARGUMENT)")
                .isThrownBy(() -> svc.process(null, opts, "step"));
    }

    @Test
    void process_nullOptions_throwsIAE() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());
        PipeDoc input = buildValidStage2();
        assertThatExceptionOfType(IllegalArgumentException.class)
                .isThrownBy(() -> svc.process(input, null, "step"));
    }

    @Test
    void process_placeholderSprAtStage2_throwsISE() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());

        SemanticProcessingResult placeholder = validStage2Spr().toBuilder()
                .setEmbeddingConfigId("").build();
        PipeDoc input = PipeDoc.newBuilder()
                .setDocId("doc-x")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .addSemanticResults(placeholder)
                        .setVectorSetDirectives(advertisedEmbedderMinilm())
                        .addSourceFieldAnalytics(SourceFieldAnalytics.newBuilder()
                                .setSourceField("body").setChunkConfigId("sentence_v1").build())
                        .build())
                .build();

        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, false, null,
                null, null, null, null, null, null, null, null, null);

        assertThatExceptionOfType(IllegalStateException.class)
                .as("Placeholder SPR at Stage 2 → ISE (FAILED_PRECONDITION) per assertPostEmbedder")
                .isThrownBy(() -> svc.process(input, opts, "step"))
                .withMessageContaining("post-embedder");
    }

    @Test
    void process_duplicateSourceLabel_throwsIAE() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());

        VectorSetDirectives dup = VectorSetDirectives.newBuilder()
                .addDirectives(VectorDirective.newBuilder()
                        .setSourceLabel("body")
                        .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                        .build())
                .addDirectives(VectorDirective.newBuilder()
                        .setSourceLabel("body")  // duplicate!
                        .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                        .build())
                .build();
        PipeDoc input = PipeDoc.newBuilder()
                .setDocId("doc-x")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .addSemanticResults(validStage2Spr())
                        .setVectorSetDirectives(dup)
                        .addSourceFieldAnalytics(SourceFieldAnalytics.newBuilder()
                                .setSourceField("body").setChunkConfigId("sentence_v1").build())
                        .build())
                .build();

        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, false, null,
                null, null, null, null, null, null, null, null, null);

        assertThatExceptionOfType(IllegalArgumentException.class)
                .as("Duplicate source_label → IAE per §21.2")
                .isThrownBy(() -> svc.process(input, opts, "step"))
                .withMessageContaining("Duplicate source_label");
    }

    @Test
    void process_boundariesOnWithNoModelId_throwsIAEAtValidateForUse() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());

        PipeDoc input = buildStage2WithSentencesInternal();
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, true, null,  // boundaries on, model id null
                null, null, null, null, null, null, null, null, null);

        assertThatExceptionOfType(IllegalArgumentException.class)
                .as("§21.3: boundaries on with null model id → IAE from validateForUse")
                .isThrownBy(() -> svc.process(input, opts, "step"));
    }

    // ======================================================================
    // Async boundary failures
    // ======================================================================

    @Test
    void process_boundariesOnButModelNotLoaded_failsFailedPrecondition() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        when(djl.isModelLoaded(eq("minilm"))).thenReturn(Uni.createFrom().item(Boolean.FALSE));

        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());
        PipeDoc input = buildStage2WithSentencesInternal();
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, true, "minilm",
                null, null, null, null, null, null, null, null, null);

        Throwable err = svc.process(input, opts, "step")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(err).isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("not currently loaded")
                .hasMessageContaining("§21.3");
        verify(djl, never()).embed(any(), any(), anyInt(), anyInt(), anyInt(), anyLong());
    }

    @Test
    void process_boundariesOnButNoSentenceSprForModel_failsFailedPrecondition() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        when(djl.isModelLoaded(eq("unused"))).thenReturn(Uni.createFrom().item(Boolean.TRUE));

        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());
        PipeDoc input = buildStage2WithSentencesInternal();   // sentence SPR for 'minilm', not 'unused'
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, true, "unused",
                null, null, null, null, null, null, null, null, null);

        Throwable err = svc.process(input, opts, "step")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(err).isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("no sentence-shaped Stage-2 SPR");
    }

    @Test
    void process_boundaryHardCapExceeded_fails() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        when(djl.isModelLoaded(eq("minilm"))).thenReturn(Uni.createFrom().item(Boolean.TRUE));

        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());
        // Build 60 sentence chunks with alternating similarity — boundary detector
        // with percentile=100 will mark nearly every gap as a boundary.
        PipeDoc input = buildStage2WithManySentences(60);
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, true, "minilm",
                /* hardCap */ 10, /* similarity */ 1.0f, /* percentile */ 100,
                /* min */ 1, /* max */ 1,   // max=1 forces every sentence to its own group → 60 groups → exceeds 10
                null, null, null, null);

        Throwable err = svc.process(input, opts, "step")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(err).isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("exceeding max_semantic_chunks_per_doc");
        verify(djl, never()).embed(any(), any(), anyInt(), anyInt(), anyInt(), anyLong());
    }

    @Test
    void process_boundaryEmbedFails_propagates() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        when(djl.isModelLoaded(eq("minilm"))).thenReturn(Uni.createFrom().item(Boolean.TRUE));
        when(djl.embed(eq("minilm"), any(), anyInt(), anyInt(), anyInt(), anyLong()))
                .thenReturn(Uni.createFrom().failure(new ConnectException("djl down")));

        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());
        PipeDoc input = buildStage2WithSentencesInternal();
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, false, true, "minilm",
                null, null, null, null, null, null, null, null, null);

        Throwable err = svc.process(input, opts, "step")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure().getFailure();

        assertThat(err)
                .as("DJL transport failure propagates verbatim")
                .isInstanceOf(ConnectException.class)
                .hasMessageContaining("djl down");
    }

    // ======================================================================
    // Lex-sort invariant
    // ======================================================================

    @Test
    void process_centroidsAppendedInLexOrder() {
        SemanticGraphEmbedHelper djl = mock(SemanticGraphEmbedHelper.class);
        SemanticGraphPipelineService svc = new SemanticGraphPipelineService(djl, noopMetrics());

        PipeDoc input = buildValidStage2();
        SemanticGraphStepOptions opts = new SemanticGraphStepOptions(
                false, false, true, false, null,
                null, null, null, null, null, null, null, null, null);

        PipeDoc out = svc.process(input, opts, "step")
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem().getItem();

        List<SemanticProcessingResult> sprs = out.getSearchMetadata().getSemanticResultsList();
        // Verify lex sort: (source, chunker, embedder, result_id) ascending
        for (int i = 1; i < sprs.size(); i++) {
            SemanticProcessingResult prev = sprs.get(i - 1);
            SemanticProcessingResult curr = sprs.get(i);
            int cmp = prev.getSourceFieldName().compareTo(curr.getSourceFieldName());
            if (cmp == 0) cmp = prev.getChunkConfigId().compareTo(curr.getChunkConfigId());
            if (cmp == 0) cmp = prev.getEmbeddingConfigId().compareTo(curr.getEmbeddingConfigId());
            if (cmp == 0) cmp = prev.getResultId().compareTo(curr.getResultId());
            assertThat(cmp).as("SPR[%d] must be <= SPR[%d] in lex order", i - 1, i)
                    .isLessThanOrEqualTo(0);
        }
    }

    /** No-op metrics bean for unit tests. Mockito gives us zero-cost stubs
     *  for every SemanticGraphMetrics method without a real MeterRegistry. */
    private static SemanticGraphMetrics noopMetrics() {
        return mock(SemanticGraphMetrics.class);
    }

    // ======================================================================
    // Fixture helpers
    // ======================================================================

    private static SemanticChunk chunk(String id, int num, String text, int start, int end,
                                        float[] vec) {
        ChunkEmbedding.Builder emb = ChunkEmbedding.newBuilder()
                .setTextContent(text)
                .setOriginalCharStartOffset(start)
                .setOriginalCharEndOffset(end);
        for (float f : vec) emb.addVector(f);
        return SemanticChunk.newBuilder()
                .setChunkId(id)
                .setChunkNumber(num)
                .setEmbeddingInfo(emb.build())
                .build();
    }

    private static VectorSetDirectives advertisedEmbedderMinilm() {
        return VectorSetDirectives.newBuilder()
                .addDirectives(VectorDirective.newBuilder()
                        .setSourceLabel("body")
                        .setCelSelector("document.body")
                        .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                        .addChunkerConfigs(NamedChunkerConfig.newBuilder()
                                .setConfigId("sentence_v1")
                                .setConfig(Struct.newBuilder()
                                        .putFields("algorithm",
                                                Value.newBuilder().setStringValue("SENTENCE").build())
                                        .build())
                                .build())
                        .build())
                .build();
    }

    private static SemanticProcessingResult validStage2Spr() {
        return SemanticProcessingResult.newBuilder()
                .setResultId("stage2:doc-x:body:sentence_v1:minilm")
                .setSourceFieldName("body")
                .setChunkConfigId("sentence_v1")
                .setEmbeddingConfigId("minilm")
                .addChunks(chunk("c0", 0, "one two three", 0, 13,
                        new float[]{0.1f, 0.0f, 0.0f, 0.0f}))
                .addChunks(chunk("c1", 1, "four five six", 14, 27,
                        new float[]{0.0f, 0.1f, 0.0f, 0.0f}))
                .addChunks(chunk("c2", 2, "seven eight nine", 28, 44,
                        new float[]{0.0f, 0.0f, 0.1f, 0.0f}))
                .setNlpAnalysis(NlpDocumentAnalysis.getDefaultInstance())
                .putMetadata("directive_key",
                        Value.newBuilder().setStringValue("dk").build())
                .build();
    }

    private static PipeDoc buildValidStage2() {
        return PipeDoc.newBuilder()
                .setDocId("doc-x")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .addSemanticResults(validStage2Spr())
                        .setVectorSetDirectives(advertisedEmbedderMinilm())
                        .addSourceFieldAnalytics(SourceFieldAnalytics.newBuilder()
                                .setSourceField("body").setChunkConfigId("sentence_v1").build())
                        .build())
                .build();
    }

    private static PipeDoc buildStage2WithDocOutline() {
        SearchMetadata.Builder sm = buildValidStage2().getSearchMetadata().toBuilder()
                .setDocOutline(DocOutline.newBuilder()
                        .addSections(Section.newBuilder()
                                .setTitle("Intro").setDepth(0).setHeadingLevel(1)
                                .setCharStartOffset(0).setCharEndOffset(30))
                        .addSections(Section.newBuilder()
                                .setTitle("Body").setDepth(0).setHeadingLevel(1)
                                .setCharStartOffset(30).setCharEndOffset(80))
                        .build());
        return buildValidStage2().toBuilder().setSearchMetadata(sm.build()).build();
    }

    private static PipeDoc buildStage2WithSentencesInternal() {
        SemanticProcessingResult spr = SemanticProcessingResult.newBuilder()
                .setResultId("stage2:doc-x:body:sentences_internal:minilm")
                .setSourceFieldName("body")
                .setChunkConfigId("sentences_internal")
                .setEmbeddingConfigId("minilm")
                .addChunks(chunk("s0", 0, "first sentence.", 0, 15,
                        new float[]{1f, 0f, 0f, 0f}))
                .addChunks(chunk("s1", 1, "second sentence.", 16, 32,
                        new float[]{0.9f, 0.1f, 0f, 0f}))
                .addChunks(chunk("s2", 2, "third sentence.", 33, 48,
                        new float[]{0.8f, 0.2f, 0f, 0f}))
                .setNlpAnalysis(NlpDocumentAnalysis.getDefaultInstance())
                .putMetadata("directive_key",
                        Value.newBuilder().setStringValue("dk").build())
                .build();
        return PipeDoc.newBuilder()
                .setDocId("doc-x")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .addSemanticResults(spr)
                        .setVectorSetDirectives(advertisedEmbedderMinilm())
                        .addSourceFieldAnalytics(SourceFieldAnalytics.newBuilder()
                                .setSourceField("body")
                                .setChunkConfigId("sentences_internal").build())
                        .build())
                .build();
    }

    private static PipeDoc buildStage2WithManySentences(int count) {
        SemanticProcessingResult.Builder spr = SemanticProcessingResult.newBuilder()
                .setResultId("stage2:doc-x:body:sentences_internal:minilm")
                .setSourceFieldName("body")
                .setChunkConfigId("sentences_internal")
                .setEmbeddingConfigId("minilm")
                .setNlpAnalysis(NlpDocumentAnalysis.getDefaultInstance())
                .putMetadata("directive_key",
                        Value.newBuilder().setStringValue("dk").build());
        int off = 0;
        for (int i = 0; i < count; i++) {
            int len = 10;
            float[] v = new float[]{(i % 2 == 0) ? 1f : -1f, 0f, 0f, 0f};  // alternating high dissimilarity
            spr.addChunks(chunk("s" + i, i, "sent " + i, off, off + len, v));
            off += len + 1;
        }
        return PipeDoc.newBuilder()
                .setDocId("doc-x")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .addSemanticResults(spr.build())
                        .setVectorSetDirectives(advertisedEmbedderMinilm())
                        .addSourceFieldAnalytics(SourceFieldAnalytics.newBuilder()
                                .setSourceField("body")
                                .setChunkConfigId("sentences_internal").build())
                        .build())
                .build();
    }

    private static void assertStage2Preserved(PipeDoc input, PipeDoc output) {
        List<SemanticProcessingResult> inputSprs = input.getSearchMetadata().getSemanticResultsList();
        List<SemanticProcessingResult> outputSprs = output.getSearchMetadata().getSemanticResultsList();
        for (SemanticProcessingResult inSpr : inputSprs) {
            assertThat(outputSprs)
                    .as("Every input Stage-2 SPR must appear byte-identical in the output "
                            + "(result_id=%s)", inSpr.getResultId())
                    .anyMatch(out -> out.equals(inSpr));
        }
    }
}
