package ai.pipestream.module.semanticgraph.pipeline;

import ai.pipestream.data.v1.CentroidMetadata;
import ai.pipestream.data.v1.ChunkEmbedding;
import ai.pipestream.data.v1.DocOutline;
import ai.pipestream.data.v1.GranularityLevel;
import ai.pipestream.data.v1.NamedChunkerConfig;
import ai.pipestream.data.v1.PipeDoc;
import ai.pipestream.data.v1.PoolingMethod;
import ai.pipestream.data.v1.SearchMetadata;
import ai.pipestream.data.v1.Section;
import ai.pipestream.data.v1.SemanticChunk;
import ai.pipestream.data.v1.SemanticProcessingResult;
import ai.pipestream.data.v1.VectorDirective;
import ai.pipestream.module.semanticgraph.config.SemanticGraphStepOptions;
import ai.pipestream.module.semanticgraph.djl.SemanticGraphEmbedHelper;
import ai.pipestream.module.semanticgraph.invariants.SemanticPipelineInvariants;
import ai.pipestream.module.semanticgraph.metrics.SemanticGraphMetrics;
import ai.pipestream.module.semanticgraph.service.CentroidComputer;
import ai.pipestream.module.semanticgraph.service.SemanticBoundaryDetector;
import com.google.protobuf.Struct;
import com.google.protobuf.Value;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Transport-agnostic semantic-graph pipeline per DESIGN.md §7.3. Takes a Stage-2
 * {@link PipeDoc}, validates it, computes centroids + optional semantic
 * boundaries, appends new SPRs (Stage-2 SPRs preserved byte-for-byte),
 * lex-sorts, and returns Stage-3 shape.
 *
 * <h2>Error model</h2>
 * <ul>
 *   <li>{@link IllegalArgumentException} synchronously on caller-input
 *       violations (duplicate source_label, missing/malformed directive);
 *       caller maps to {@code INVALID_ARGUMENT}.</li>
 *   <li>{@link IllegalStateException} synchronously when the input doc
 *       fails {@link SemanticPipelineInvariants#assertPostEmbedder}; caller
 *       maps to {@code FAILED_PRECONDITION}.</li>
 *   <li>Async failures (DJL transport, hard-cap, model-not-loaded) propagate
 *       via the returned {@link Uni}'s failure channel; caller classifies via
 *       {@code SemanticGraphRetryClassifier}.</li>
 *   <li>Output self-check: before emit, calls
 *       {@link SemanticPipelineInvariants#assertPostSemanticGraph} on the
 *       assembled doc. Non-null message = module bug, Uni fails with ISE.</li>
 * </ul>
 *
 * <h2>Concurrency</h2>
 * <p>Per DESIGN.md §3: per-doc work only, no cross-doc state. Bean is
 * {@code @ApplicationScoped} but stateless — every field is either final
 * and injected or local to a single {@link #process} call.
 */
@ApplicationScoped
public class SemanticGraphPipelineService {

    private static final Logger log = Logger.getLogger(SemanticGraphPipelineService.class);

    /** §21.8 lex comparator on {@code semantic_results[]}. */
    private static final Comparator<SemanticProcessingResult> LEX = Comparator
            .comparing(SemanticProcessingResult::getSourceFieldName)
            .thenComparing(SemanticProcessingResult::getChunkConfigId)
            .thenComparing(SemanticProcessingResult::getEmbeddingConfigId)
            .thenComparing(SemanticProcessingResult::getResultId);

    /** Default {@code result_set_name} template. */
    private static final String DEFAULT_RESULT_SET_NAME_TEMPLATE =
            "{source_label}_{chunker_id}_{embedder_id}";

    /** Default template for boundary SPR {@code result_set_name}. */
    private static final String BOUNDARY_RESULT_SET_NAME_TEMPLATE =
            "{source_label}_semantic_{embedder_id}";

    /** The chunker's always-emit fallback config_id from DESIGN.md §4.1. */
    private static final String SENTENCES_INTERNAL = "sentences_internal";

    /** Chunker config {@code algorithm} value marking a sentence chunker. */
    private static final String ALGORITHM_SENTENCE = "SENTENCE";

    private final SemanticGraphEmbedHelper embedHelper;
    private final SemanticGraphMetrics metrics;

    @Inject
    public SemanticGraphPipelineService(
            SemanticGraphEmbedHelper embedHelper,
            SemanticGraphMetrics metrics) {
        this.embedHelper = embedHelper;
        this.metrics = metrics;
    }

    /**
     * Runs the §7.3 recipe for one {@link PipeDoc}.
     *
     * @param inputDoc     the Stage-2 doc (fully embedded)
     * @param options      effective step options (env-overrides already applied)
     * @param pipeStepName step name for logs/diagnostics
     * @return a {@link Uni} emitting the Stage-3 doc, or failing with the
     *         originating exception. See class javadoc for error semantics.
     * @throws IllegalArgumentException  synchronous: duplicate source_label,
     *         missing directives, options validation failure
     * @throws IllegalStateException     synchronous: {@link SemanticPipelineInvariants#assertPostEmbedder}
     *         violation
     */
    public Uni<PipeDoc> process(PipeDoc inputDoc, SemanticGraphStepOptions options, String pipeStepName) {
        final long processStartNs = System.nanoTime();

        // Phase 1 — synchronous validation + prep
        Prepared prepared = prepare(inputDoc, options, pipeStepName);
        metrics.recordStage2SprCount(prepared.byTriple.size());

        // Phase 2 — centroid pass (pure CPU; inline in Uni.createFrom().item
        // so every subscription evaluates lazily)
        Uni<List<SemanticProcessingResult>> centroidsUni = Uni.createFrom().item(() -> {
            long t0 = System.nanoTime();
            List<SemanticProcessingResult> out = computeCentroids(prepared);
            metrics.centroidCompleted(Duration.ofNanos(System.nanoTime() - t0));
            metrics.recordCentroidSprCount(out.size());
            return out;
        });

        // Phase 3 — boundary pass (may involve DJL I/O)
        final long boundaryStartNs = System.nanoTime();
        Uni<List<SemanticProcessingResult>> boundariesUni = prepared.options.effectiveComputeSemanticBoundaries()
                ? computeBoundaries(prepared)
                        .onTermination().invoke((ignored, err, cancelled) ->
                                metrics.boundaryCompleted(Duration.ofNanos(System.nanoTime() - boundaryStartNs)))
                : Uni.createFrom().item(List.<SemanticProcessingResult>of())
                        .onItem().invoke(ignored ->
                                metrics.boundaryCompleted(Duration.ofNanos(System.nanoTime() - boundaryStartNs)));

        // Phase 4 — assemble + self-check + emit
        return Uni.combine().all().unis(centroidsUni, boundariesUni)
                .asTuple()
                .map(tuple -> assembleOutputDoc(prepared, tuple.getItem1(), tuple.getItem2()))
                .onTermination().invoke((ignored, err, cancelled) -> {
                    // Per-doc wall clock — success AND failure both record so
                    // the p95 includes tail latencies even when they error.
                    // (docCompleted/docFailed live in the gRPC layer to
                    // bracket the full request including parseOptions.)
                    // Intentionally not recording here; gRPC layer does it.
                });
    }

    // =========================================================================
    // Phase 1 — synchronous validation + prep
    // =========================================================================

    private Prepared prepare(PipeDoc inputDoc, SemanticGraphStepOptions options, String pipeStepName) {
        if (inputDoc == null) {
            throw new IllegalArgumentException("PipeDoc must not be null");
        }
        if (options == null) {
            throw new IllegalArgumentException("SemanticGraphStepOptions must not be null");
        }
        // §21.3 — validate options up-front (boundary model id required when boundaries on, etc.).
        options.validateForUse();

        // Invariant gate per §5.2.
        String err = SemanticPipelineInvariants.assertPostEmbedder(inputDoc);
        if (err != null) {
            throw new IllegalStateException(
                    "Input doc failed post-embedder invariant (FAILED_PRECONDITION): " + err);
        }

        SearchMetadata sm = inputDoc.getSearchMetadata();

        // Directives may or may not be present by the time the pipeline runs (some engine
        // configs clear them after processing). If present, validate uniqueness
        // and use them to identify sentence-shaped chunker configs; if absent
        // we fall through to SENTENCES_INTERNAL-only detection.
        List<VectorDirective> directives = sm.hasVectorSetDirectives()
                ? sm.getVectorSetDirectives().getDirectivesList()
                : List.of();

        Set<String> seenLabels = new HashSet<>();
        Map<String, VectorDirective> directiveBySourceLabel = new LinkedHashMap<>();
        for (VectorDirective d : directives) {
            if (!seenLabels.add(d.getSourceLabel())) {
                throw new IllegalArgumentException(
                        "Duplicate source_label '" + d.getSourceLabel()
                                + "' in vector_set_directives — §21.2 INVALID_ARGUMENT");
            }
            directiveBySourceLabel.put(d.getSourceLabel(), d);
        }

        // Per-source set of sentence-shaped chunker config_ids.
        // Population: always include SENTENCES_INTERNAL; add any directive chunker
        // whose config.algorithm == "SENTENCE".
        Map<String, Set<String>> sentenceConfigsBySource = new HashMap<>();
        for (VectorDirective d : directives) {
            Set<String> set = new HashSet<>();
            set.add(SENTENCES_INTERNAL);
            for (NamedChunkerConfig ncc : d.getChunkerConfigsList()) {
                Struct cfg = ncc.hasConfig() ? ncc.getConfig() : null;
                if (cfg == null) continue;
                Value algo = cfg.getFieldsOrDefault("algorithm", null);
                if (algo != null
                        && algo.getKindCase() == Value.KindCase.STRING_VALUE
                        && ALGORITHM_SENTENCE.equalsIgnoreCase(algo.getStringValue())) {
                    set.add(ncc.getConfigId());
                }
            }
            sentenceConfigsBySource.put(d.getSourceLabel(), set);
        }

        String docHash = sha256b64url(inputDoc.getDocId());

        // Group Stage-2 SPRs by (source_field, chunker_config, embedder_config).
        Map<TripleKey, List<SemanticProcessingResult>> byTriple = new LinkedHashMap<>();
        for (SemanticProcessingResult spr : sm.getSemanticResultsList()) {
            TripleKey key = new TripleKey(
                    spr.getSourceFieldName(),
                    spr.getChunkConfigId(),
                    spr.getEmbeddingConfigId());
            byTriple.computeIfAbsent(key, k -> new ArrayList<>()).add(spr);
        }

        // Audit: surfaces the input shape in a grep-friendly line. Every
        // pipeline invocation logs exactly one of these at INFO so operators can
        // replay traffic shape from module logs alone, without
        // back-reading metrics.
        log.infof("AUDIT prepare doc=%s step=%s directives=%d stage2_triples=%d "
                        + "sentence_configs_by_source=%s hasDocOutline=%s",
                inputDoc.getDocId(), pipeStepName, directives.size(), byTriple.size(),
                sentenceConfigsBySource,
                inputDoc.getSearchMetadata().hasDocOutline());

        return new Prepared(inputDoc, options, pipeStepName, directives,
                directiveBySourceLabel, sentenceConfigsBySource, byTriple, docHash);
    }

    // =========================================================================
    // Phase 2 — centroid pass (pure CPU, no I/O)
    // =========================================================================

    private List<SemanticProcessingResult> computeCentroids(Prepared prepared) {
        List<SemanticProcessingResult> out = new ArrayList<>();
        SemanticGraphStepOptions opt = prepared.options;
        boolean doDoc = opt.effectiveComputeDocumentCentroid();
        boolean doParagraph = opt.effectiveComputeParagraphCentroids();
        boolean doSection = opt.effectiveComputeSectionCentroids();
        if (!doDoc && !doParagraph && !doSection) {
            return out;
        }

        // DocOutline is doc-level, shared across all groups.
        DocOutline outline = prepared.inputDoc.getSearchMetadata().hasDocOutline()
                ? prepared.inputDoc.getSearchMetadata().getDocOutline()
                : null;

        for (Map.Entry<TripleKey, List<SemanticProcessingResult>> e : prepared.byTriple.entrySet()) {
            TripleKey key = e.getKey();
            List<SemanticProcessingResult> groupSprs = e.getValue();
            // A (source, chunker, embedder) triple should only have 1 SPR per
            // Stage-2 invariant; we defensively concatenate all chunks across
            // same-triple SPRs if the upstream violated that.
            List<SemanticChunk> chunks = new ArrayList<>();
            String directiveKeyStr = "";
            boolean hasNlp = false;
            for (SemanticProcessingResult spr : groupSprs) {
                chunks.addAll(spr.getChunksList());
                if (directiveKeyStr.isEmpty()) {
                    Value v = spr.getMetadataMap().get(SemanticPipelineInvariants.DIRECTIVE_KEY_METADATA);
                    if (v != null && v.getKindCase() == Value.KindCase.STRING_VALUE) {
                        directiveKeyStr = v.getStringValue();
                    }
                }
                if (spr.hasNlpAnalysis()) hasNlp = true;
            }
            if (chunks.isEmpty()) continue;

            SemanticProcessingResult firstSourceSpr = groupSprs.get(0);

            if (doDoc) {
                out.add(buildDocumentCentroidSpr(prepared, key, chunks, firstSourceSpr, directiveKeyStr));
            }
            if (doSection && outline != null && !outline.getSectionsList().isEmpty()) {
                out.addAll(buildSectionCentroidSprs(prepared, key, chunks, firstSourceSpr,
                        directiveKeyStr, outline));
            }
            if (doParagraph) {
                out.addAll(buildParagraphCentroidSprs(prepared, key, chunks, firstSourceSpr,
                        directiveKeyStr));
            }
        }

        return out;
    }

    private SemanticProcessingResult buildDocumentCentroidSpr(
            Prepared prepared, TripleKey key, List<SemanticChunk> chunks,
            SemanticProcessingResult sourceSpr, String directiveKey) {
        List<float[]> vectors = extractVectors(chunks);
        float[] vec = CentroidComputer.averageAndNormalize(vectors);
        return buildCentroidSpr(prepared, key, vec, "",
                GranularityLevel.GRANULARITY_LEVEL_DOCUMENT, "document_centroid",
                vectors.size(), null, null,
                sourceSpr, directiveKey, 0);
    }

    private List<SemanticProcessingResult> buildSectionCentroidSprs(
            Prepared prepared, TripleKey key, List<SemanticChunk> chunks,
            SemanticProcessingResult sourceSpr, String directiveKey, DocOutline outline) {

        // Extract per-chunk vectors + offsets
        int n = chunks.size();
        List<float[]> vectors = extractVectors(chunks);
        List<String> texts = extractTexts(chunks);
        int[][] offsets = extractOffsets(chunks);
        List<CentroidComputer.SectionInfo> sections = new ArrayList<>();
        for (Section s : outline.getSectionsList()) {
            if (!s.hasCharStartOffset()) continue;
            int start = s.getCharStartOffset();
            int end = s.hasCharEndOffset() ? s.getCharEndOffset() : -1;
            sections.add(new CentroidComputer.SectionInfo(
                    s.hasTitle() ? s.getTitle() : "",
                    s.getDepth(),
                    start, end));
        }
        if (sections.isEmpty()) return List.of();

        List<CentroidComputer.CentroidResult> results = CentroidComputer.computeSectionCentroids(
                vectors, texts, offsets, sections);

        List<SemanticProcessingResult> sprs = new ArrayList<>(results.size());
        for (int i = 0; i < results.size(); i++) {
            CentroidComputer.CentroidResult r = results.get(i);
            sprs.add(buildCentroidSpr(prepared, key, r.vector(), r.text(),
                    GranularityLevel.GRANULARITY_LEVEL_SECTION, "section_centroid",
                    r.sourceVectorCount(), r.sectionTitle(), r.sectionDepth(),
                    sourceSpr, directiveKey, i));
        }
        return sprs;
    }

    private List<SemanticProcessingResult> buildParagraphCentroidSprs(
            Prepared prepared, TripleKey key, List<SemanticChunk> chunks,
            SemanticProcessingResult sourceSpr, String directiveKey) {
        // Paragraph detection relies on the ORIGINAL source text (to scan for
        // "\n\n" gaps between consecutive chunk offsets). Best-effort recovery:
        // for source_label == "body" / "title", use the matching SearchMetadata
        // field. Other labels fall back to skip (no false positives).
        Optional<String> maybeSourceText = resolveSourceText(prepared.inputDoc, key.sourceField());
        if (maybeSourceText.isEmpty()) {
            log.debugf("Paragraph centroids skipped for source='%s': cannot resolve original text " +
                    "(only 'body'/'title' conventions are supported without a CEL evaluator)",
                    key.sourceField());
            return List.of();
        }

        List<float[]> vectors = extractVectors(chunks);
        List<String> texts = extractTexts(chunks);
        int[][] offsets = extractOffsets(chunks);
        List<CentroidComputer.CentroidResult> results = CentroidComputer.computeParagraphCentroids(
                vectors, texts, maybeSourceText.get(), offsets);
        if (results.isEmpty()) {
            // No paragraph breaks found; not a failure.
            return List.of();
        }
        List<SemanticProcessingResult> sprs = new ArrayList<>(results.size());
        for (int i = 0; i < results.size(); i++) {
            CentroidComputer.CentroidResult r = results.get(i);
            sprs.add(buildCentroidSpr(prepared, key, r.vector(), r.text(),
                    GranularityLevel.GRANULARITY_LEVEL_PARAGRAPH, "paragraph_centroid",
                    r.sourceVectorCount(), null, null,
                    sourceSpr, directiveKey, i));
        }
        return sprs;
    }

    private SemanticProcessingResult buildCentroidSpr(
            Prepared prepared, TripleKey key, float[] vector, String text,
            GranularityLevel granularity, String chunkConfigId,
            int sourceVectorCount, String sectionTitle, Integer sectionDepth,
            SemanticProcessingResult sourceSpr, String directiveKey, int chunkIndex) {

        String granularityTag = granularity.name()
                .replace("GRANULARITY_LEVEL_", "").toLowerCase(java.util.Locale.ROOT);
        // §21.5: stage3-centroid-{granularity}:{docHash}:{source}:{chunkConfig}:{embedder}
        String resultId = "stage3-centroid-" + granularityTag + ":" + prepared.docHash
                + ":" + key.sourceField() + ":" + chunkConfigId + ":" + key.embedderConfig();
        String resultSetName = resolveResultSetNameWithGranularity(
                prepared.directiveBySourceLabel.get(key.sourceField()),
                key.chunkerConfig(), key.embedderConfig(), granularityTag);

        ChunkEmbedding.Builder emb = ChunkEmbedding.newBuilder()
                .setTextContent(text == null ? "" : text)
                .setChunkConfigId(chunkConfigId);
        for (float f : vector) emb.addVector(f);

        SemanticChunk chunk = SemanticChunk.newBuilder()
                .setChunkId(granularityTag + ":" + chunkIndex)
                .setChunkNumber(chunkIndex)
                .setEmbeddingInfo(emb.build())
                .build();

        CentroidMetadata.Builder cm = CentroidMetadata.newBuilder()
                .setGranularity(granularity)
                .setSourceVectorCount(sourceVectorCount);
        if (sectionTitle != null && !sectionTitle.isEmpty()) cm.setSectionTitle(sectionTitle);
        if (sectionDepth != null) cm.setSectionDepth(sectionDepth);

        SemanticProcessingResult.Builder b = SemanticProcessingResult.newBuilder()
                .setResultId(resultId)
                .setSourceFieldName(key.sourceField())
                .setChunkConfigId(chunkConfigId)
                .setEmbeddingConfigId(key.embedderConfig())
                .setResultSetName(resultSetName)
                .addChunks(chunk)
                .setCentroidMetadata(cm.build())
                .setGranularity(granularity)
                .setPoolingMethod(PoolingMethod.POOLING_METHOD_MEAN)
                .setParentResultId(sourceSpr.getResultId());

        if (!directiveKey.isEmpty()) {
            b.putMetadata(SemanticPipelineInvariants.DIRECTIVE_KEY_METADATA,
                    Value.newBuilder().setStringValue(directiveKey).build());
        }
        return b.build();
    }

    // =========================================================================
    // Phase 3 — boundary pass (reactive, may involve DJL)
    // =========================================================================

    private Uni<List<SemanticProcessingResult>> computeBoundaries(Prepared prepared) {
        final String modelId = prepared.options.requireBoundaryEmbeddingModelId();

        return embedHelper.isModelLoaded(modelId)
                .chain(loaded -> {
                    if (!Boolean.TRUE.equals(loaded)) {
                        return Uni.createFrom().<List<SemanticProcessingResult>>failure(
                                new IllegalStateException(
                                        "boundary_embedding_model_id='" + modelId
                                                + "' is not currently loaded in DJL Serving "
                                                + "(§21.3 forbids 'first available' fallback)"));
                    }
                    return runBoundaryPass(prepared, modelId);
                });
    }

    /** Builds one boundary SPR per source_field that has a sentence-shaped Stage-2 SPR for the model. */
    private Uni<List<SemanticProcessingResult>> runBoundaryPass(Prepared prepared, String modelId) {
        // Collect per-source sentence-shaped Stage-2 SPRs with embedding_config_id == modelId.
        List<BoundaryTask> tasks = new ArrayList<>();
        for (Map.Entry<TripleKey, List<SemanticProcessingResult>> e : prepared.byTriple.entrySet()) {
            TripleKey key = e.getKey();
            if (!modelId.equals(key.embedderConfig())) continue;
            Set<String> sentenceConfigs = prepared.sentenceConfigsBySource.get(key.sourceField());
            boolean isSentenceShaped =
                    SENTENCES_INTERNAL.equals(key.chunkerConfig())
                    || (sentenceConfigs != null && sentenceConfigs.contains(key.chunkerConfig()));
            if (!isSentenceShaped) continue;
            // Prefer sentences_internal over other sentence configs if both present for the
            // same source_field — deterministic selection.
            tasks.add(new BoundaryTask(key, e.getValue()));
        }
        if (tasks.isEmpty()) {
            // No sentence-shaped SPR for this model in any source. Per §21.3 this
            // is a FAILED_PRECONDITION: boundaries were requested, the model was
            // named and is loaded, but there's no sentence data to run boundary
            // detection over. Fail rather than silently emit nothing.
            return Uni.createFrom().failure(new IllegalStateException(
                    "compute_semantic_boundaries=true with boundary_embedding_model_id='" + modelId
                            + "' but no sentence-shaped Stage-2 SPR is present for any source_field "
                            + "with that embedding_config_id — either the chunker didn't emit "
                            + SENTENCES_INTERNAL + ", the directive didn't declare a SENTENCE "
                            + "chunker, or the embedder didn't run the sentence chunks through "
                            + "this model"));
        }
        // Deduplicate: if both sentences_internal AND a directive-declared sentence
        // chunker exist for the same source, prefer sentences_internal.
        Map<String, BoundaryTask> bestPerSource = new LinkedHashMap<>();
        for (BoundaryTask t : tasks) {
            BoundaryTask existing = bestPerSource.get(t.key.sourceField());
            if (existing == null) {
                bestPerSource.put(t.key.sourceField(), t);
            } else if (SENTENCES_INTERNAL.equals(t.key.chunkerConfig())
                    && !SENTENCES_INTERNAL.equals(existing.key.chunkerConfig())) {
                bestPerSource.put(t.key.sourceField(), t);
            }
        }

        // Build one Uni per source + merge.
        List<Uni<SemanticProcessingResult>> perSource = new ArrayList<>(bestPerSource.size());
        for (BoundaryTask t : bestPerSource.values()) {
            perSource.add(buildBoundarySpr(prepared, modelId, t));
        }
        if (perSource.isEmpty()) {
            return Uni.createFrom().item(List.of());
        }
        return Uni.combine().all().unis(perSource).with(objs -> {
            List<SemanticProcessingResult> out = new ArrayList<>(objs.size());
            for (Object o : objs) out.add((SemanticProcessingResult) o);
            return out;
        });
    }

    private Uni<SemanticProcessingResult> buildBoundarySpr(
            Prepared prepared, String modelId, BoundaryTask task) {
        // Flatten all sentence chunks for this (source, chunker, modelId) triple.
        List<SemanticChunk> sentences = new ArrayList<>();
        String directiveKeyStr = "";
        for (SemanticProcessingResult spr : task.sprs) {
            sentences.addAll(spr.getChunksList());
            if (directiveKeyStr.isEmpty()) {
                Value v = spr.getMetadataMap().get(SemanticPipelineInvariants.DIRECTIVE_KEY_METADATA);
                if (v != null && v.getKindCase() == Value.KindCase.STRING_VALUE) {
                    directiveKeyStr = v.getStringValue();
                }
            }
        }
        if (sentences.size() < 2) {
            // Not enough sentences to detect a boundary. Emit one group covering
            // all sentences (acts as a pass-through document chunk for the
            // boundary model), re-embed via DJL.
            return singleGroupBoundary(prepared, modelId, task, sentences, directiveKeyStr);
        }

        List<float[]> sentenceVecs = extractVectors(sentences);
        List<String> sentenceTexts = extractTexts(sentences);
        int[][] sentenceOffsets = extractOffsets(sentences);

        SemanticGraphStepOptions opt = prepared.options;
        float[] similarities = SemanticBoundaryDetector.computeConsecutiveSimilarities(sentenceVecs);
        List<Integer> boundaries = SemanticBoundaryDetector.findBoundaries(
                sentenceVecs,
                opt.effectiveBoundarySimilarityThreshold(),
                opt.effectiveBoundaryPercentileThreshold());

        // Create index lists per group. Work with Integer indices so we can
        // use the generic grouper for both texts AND offsets.
        List<Integer> indices = new ArrayList<>(sentences.size());
        for (int i = 0; i < sentences.size(); i++) indices.add(i);
        List<List<Integer>> groups = SemanticBoundaryDetector.groupByBoundaries(indices, boundaries);

        // Enforce min/max: min merges small groups with best-similarity neighbor,
        // max splits large groups at their lowest internal similarity.
        groups = SemanticBoundaryDetector.enforceMinChunkSize(
                groups, similarities, opt.effectiveBoundaryMinSentencesPerChunk());
        groups = SemanticBoundaryDetector.enforceMaxChunkSize(
                groups, similarities, boundaries, opt.effectiveBoundaryMaxSentencesPerChunk());

        // Record group count before the hard-cap check so even rejected docs
        // show up in the distribution — operators can tell when thresholds
        // drift toward pathological configs.
        metrics.recordBoundaryGroupCount(groups.size());

        // §21.x hard cap — fail with a clear message, never silently truncate.
        int hardCap = opt.effectiveMaxSemanticChunksPerDoc();
        if (groups.size() > hardCap) {
            return Uni.createFrom().failure(new IllegalStateException(
                    "semantic boundary detection produced " + groups.size() + " groups for source='"
                            + task.key.sourceField() + "' exceeding max_semantic_chunks_per_doc="
                            + hardCap + "; tune boundary_percentile_threshold / "
                            + "boundary_min_sentences_per_chunk upward or raise "
                            + "max_semantic_chunks_per_doc"));
        }

        // Build group texts + offset spans.
        List<String> groupTexts = new ArrayList<>(groups.size());
        List<int[]> groupSpans = new ArrayList<>(groups.size());
        for (List<Integer> g : groups) {
            StringBuilder sb = new StringBuilder();
            int spanStart = Integer.MAX_VALUE;
            int spanEnd = Integer.MIN_VALUE;
            for (int idx : g) {
                if (sb.length() > 0) sb.append(' ');
                sb.append(sentenceTexts.get(idx));
                spanStart = Math.min(spanStart, sentenceOffsets[idx][0]);
                spanEnd = Math.max(spanEnd, sentenceOffsets[idx][1]);
            }
            groupTexts.add(sb.toString());
            groupSpans.add(new int[]{spanStart, spanEnd});
        }

        final List<List<Integer>> finalGroups = groups;
        final String finalDirectiveKey = directiveKeyStr;
        return embedHelper.embed(
                modelId,
                groupTexts,
                opt.effectiveMaxBatchSize(),
                opt.effectiveMaxSubBatchesPerDoc(),
                opt.effectiveMaxRetryAttempts(),
                opt.effectiveRetryBackoffMs())
                .map(groupVectors -> assembleBoundarySpr(prepared, modelId, task,
                        finalGroups, groupTexts, groupSpans, groupVectors, finalDirectiveKey));
    }

    private Uni<SemanticProcessingResult> singleGroupBoundary(
            Prepared prepared, String modelId, BoundaryTask task,
            List<SemanticChunk> sentences, String directiveKey) {
        if (sentences.isEmpty()) {
            // Degenerate — sentence SPR exists but has 0 chunks (shouldn't happen
            // post assertPostEmbedder, but guard anyway).
            return Uni.createFrom().failure(new IllegalStateException(
                    "sentence-shaped SPR for source='" + task.key.sourceField()
                            + "' has zero chunks — cannot compute boundaries"));
        }
        List<String> texts = extractTexts(sentences);
        int[][] offsets = extractOffsets(sentences);
        StringBuilder sb = new StringBuilder();
        int spanStart = offsets[0][0];
        int spanEnd = offsets[offsets.length - 1][1];
        for (int i = 0; i < texts.size(); i++) {
            if (i > 0) sb.append(' ');
            sb.append(texts.get(i));
        }
        List<String> groupTexts = List.of(sb.toString());
        List<int[]> groupSpans = List.of(new int[]{spanStart, spanEnd});
        List<List<Integer>> groups = List.of(indicesOf(sentences.size()));
        SemanticGraphStepOptions opt = prepared.options;
        return embedHelper.embed(
                modelId, groupTexts,
                opt.effectiveMaxBatchSize(),
                opt.effectiveMaxSubBatchesPerDoc(),
                opt.effectiveMaxRetryAttempts(),
                opt.effectiveRetryBackoffMs())
                .map(groupVectors -> assembleBoundarySpr(prepared, modelId, task,
                        groups, groupTexts, groupSpans, groupVectors, directiveKey));
    }

    private static List<Integer> indicesOf(int n) {
        List<Integer> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) out.add(i);
        return out;
    }

    private SemanticProcessingResult assembleBoundarySpr(
            Prepared prepared, String modelId, BoundaryTask task,
            List<List<Integer>> groups, List<String> groupTexts,
            List<int[]> groupSpans, List<float[]> groupVectors, String directiveKey) {

        if (groupVectors.size() != groupTexts.size()) {
            throw new IllegalStateException(
                    "boundary re-embed returned " + groupVectors.size() + " vectors for "
                            + groupTexts.size() + " groups on model '" + modelId + "'");
        }

        SemanticProcessingResult.Builder b = SemanticProcessingResult.newBuilder()
                .setResultId("stage3-boundary:" + prepared.docHash + ":" + task.key.sourceField()
                        + ":" + SemanticPipelineInvariants.SEMANTIC_CHUNK_CONFIG_ID
                        + ":" + modelId)
                .setSourceFieldName(task.key.sourceField())
                .setChunkConfigId(SemanticPipelineInvariants.SEMANTIC_CHUNK_CONFIG_ID)
                .setEmbeddingConfigId(modelId)
                .setResultSetName(resolveBoundaryResultSetName(
                        prepared.directiveBySourceLabel.get(task.key.sourceField()),
                        task.key.sourceField(), modelId))
                .setGranularity(GranularityLevel.GRANULARITY_LEVEL_SEMANTIC_CHUNK)
                .setPoolingMethod(PoolingMethod.POOLING_METHOD_DIRECT)
                .setSemanticConfigId("semantic:" + modelId);

        if (!directiveKey.isEmpty()) {
            b.putMetadata(SemanticPipelineInvariants.DIRECTIVE_KEY_METADATA,
                    Value.newBuilder().setStringValue(directiveKey).build());
        }

        for (int i = 0; i < groups.size(); i++) {
            List<Integer> g = groups.get(i);
            int[] span = groupSpans.get(i);
            ChunkEmbedding.Builder emb = ChunkEmbedding.newBuilder()
                    .setTextContent(groupTexts.get(i))
                    .setChunkConfigId(SemanticPipelineInvariants.SEMANTIC_CHUNK_CONFIG_ID)
                    .setOriginalCharStartOffset(span[0])
                    .setOriginalCharEndOffset(span[1]);
            for (float f : groupVectors.get(i)) emb.addVector(f);
            SemanticChunk chunk = SemanticChunk.newBuilder()
                    .setChunkId(SemanticPipelineInvariants.SEMANTIC_CHUNK_CONFIG_ID + ":" + i)
                    .setChunkNumber(i)
                    .setEmbeddingInfo(emb.build())
                    .putMetadata("sentence_count",
                            Value.newBuilder().setStringValue(String.valueOf(g.size())).build())
                    .putMetadata("sentence_span",
                            Value.newBuilder()
                                    .setStringValue(g.get(0) + "-" + g.get(g.size() - 1))
                                    .build())
                    .build();
            b.addChunks(chunk);
        }
        return b.build();
    }

    // =========================================================================
    // Phase 4 — assemble output + self-check
    // =========================================================================

    private PipeDoc assembleOutputDoc(
            Prepared prepared,
            List<SemanticProcessingResult> centroidSprs,
            List<SemanticProcessingResult> boundarySprs) {

        // Preserve Stage 2 byte-for-byte. We read the input's SPRs and simply
        // append new SPRs. No mutation of existing SPRs.
        SearchMetadata inputSm = prepared.inputDoc.getSearchMetadata();
        List<SemanticProcessingResult> merged = new ArrayList<>(
                inputSm.getSemanticResultsCount() + centroidSprs.size() + boundarySprs.size());
        merged.addAll(inputSm.getSemanticResultsList());
        merged.addAll(centroidSprs);
        merged.addAll(boundarySprs);
        merged.sort(LEX);

        SearchMetadata.Builder smBuilder = inputSm.toBuilder();
        smBuilder.clearSemanticResults();
        smBuilder.addAllSemanticResults(merged);

        PipeDoc outputDoc = prepared.inputDoc.toBuilder()
                .setSearchMetadata(smBuilder.build())
                .build();

        // Defensive self-check before emit. Pass the runtime-configured
        // boundary cap so a deployment that legitimately raised
        // max_semantic_chunks_per_doc isn't rejected by an invariant
        // hard-coded to the default.
        String err = SemanticPipelineInvariants.assertPostSemanticGraph(
                outputDoc, prepared.options.effectiveMaxSemanticChunksPerDoc());
        if (err != null) {
            throw new IllegalStateException("the pipeline produced invalid Stage-3 output: " + err);
        }

        // Audit: per-doc output shape. Log per-granularity centroid counts +
        // per-source boundary group counts so log-only replay can reconstruct
        // Stage-3 shape without the PipeDoc. One grep for 'AUDIT emit
        // doc=<X>' gets the full pipeline timeline for a doc: prepare, centroid,
        // boundary, emit.
        int docCentroids = 0, paragraphCentroids = 0, sectionCentroids = 0;
        for (SemanticProcessingResult spr : centroidSprs) {
            switch (spr.getChunkConfigId()) {
                case "document_centroid"  -> docCentroids++;
                case "paragraph_centroid" -> paragraphCentroids++;
                case "section_centroid"   -> sectionCentroids++;
                default -> { /* forward-compat */ }
            }
        }
        int totalBoundaryChunks = 0;
        for (SemanticProcessingResult spr : boundarySprs) {
            totalBoundaryChunks += spr.getChunksCount();
        }
        log.infof("AUDIT emit doc=%s stage2_preserved=%d doc_centroids=%d "
                        + "paragraph_centroids=%d section_centroids=%d boundary_sprs=%d "
                        + "boundary_chunks_total=%d final_spr_count=%d",
                prepared.inputDoc.getDocId(), inputSm.getSemanticResultsCount(),
                docCentroids, paragraphCentroids, sectionCentroids,
                boundarySprs.size(), totalBoundaryChunks, merged.size());
        return outputDoc;
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static List<float[]> extractVectors(List<SemanticChunk> chunks) {
        List<float[]> out = new ArrayList<>(chunks.size());
        for (SemanticChunk c : chunks) {
            ChunkEmbedding emb = c.getEmbeddingInfo();
            int n = emb.getVectorCount();
            float[] arr = new float[n];
            for (int i = 0; i < n; i++) arr[i] = emb.getVector(i);
            out.add(arr);
        }
        return out;
    }

    private static List<String> extractTexts(List<SemanticChunk> chunks) {
        List<String> out = new ArrayList<>(chunks.size());
        for (SemanticChunk c : chunks) out.add(c.getEmbeddingInfo().getTextContent());
        return out;
    }

    private static int[][] extractOffsets(List<SemanticChunk> chunks) {
        int[][] out = new int[chunks.size()][2];
        for (int i = 0; i < chunks.size(); i++) {
            ChunkEmbedding emb = chunks.get(i).getEmbeddingInfo();
            out[i][0] = emb.getOriginalCharStartOffset();
            out[i][1] = emb.getOriginalCharEndOffset();
        }
        return out;
    }

    private static Optional<String> resolveSourceText(PipeDoc doc, String sourceField) {
        if (!doc.hasSearchMetadata()) return Optional.empty();
        SearchMetadata sm = doc.getSearchMetadata();
        if ("body".equals(sourceField) && sm.hasBody() && !sm.getBody().isEmpty()) {
            return Optional.of(sm.getBody());
        }
        if ("title".equals(sourceField) && sm.hasTitle() && !sm.getTitle().isEmpty()) {
            return Optional.of(sm.getTitle());
        }
        return Optional.empty();
    }

    private static String resolveResultSetNameWithGranularity(
            VectorDirective directive, String chunkerConfigId, String embedderConfigId,
            String granularityTag) {
        String template = (directive != null && directive.hasFieldNameTemplate()
                && !directive.getFieldNameTemplate().isBlank())
                ? directive.getFieldNameTemplate()
                : DEFAULT_RESULT_SET_NAME_TEMPLATE;
        String sourceLabel = directive != null ? directive.getSourceLabel() : "";
        String resolved = template
                .replace("{source_label}", sourceLabel)
                .replace("{chunker_id}", chunkerConfigId)
                .replace("{embedder_id}", embedderConfigId);
        // Append granularity suffix so e.g. body_sentence_v1_minilm → body_sentence_v1_minilm_document
        return sanitize(resolved + "_" + granularityTag);
    }

    private static String resolveBoundaryResultSetName(
            VectorDirective directive, String sourceLabel, String embedderConfigId) {
        String template = (directive != null && directive.hasFieldNameTemplate()
                && !directive.getFieldNameTemplate().isBlank())
                ? directive.getFieldNameTemplate()  // caller-chosen template wins
                : BOUNDARY_RESULT_SET_NAME_TEMPLATE;
        String resolved = template
                .replace("{source_label}", sourceLabel)
                .replace("{chunker_id}", SemanticPipelineInvariants.SEMANTIC_CHUNK_CONFIG_ID)
                .replace("{embedder_id}", embedderConfigId);
        return sanitize(resolved);
    }

    private static String sanitize(String raw) {
        return raw.replaceAll("[^a-zA-Z0-9_\\-]", "_");
    }

    private static String sha256b64url(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(input.getBytes(StandardCharsets.UTF_8));
            return Base64.getUrlEncoder().withoutPadding().encodeToString(digest);
        } catch (NoSuchAlgorithmException e) {
            // SHA-256 is part of every JDK we support; this is impossible.
            throw new IllegalStateException("SHA-256 not available", e);
        }
    }

    // =========================================================================
    // Record types
    // =========================================================================

    /** The {@code (source_field, chunker_config, embedder_config)} triple. */
    private record TripleKey(String sourceField, String chunkerConfig, String embedderConfig) {}

    /** Immutable per-request snapshot. */
    private record Prepared(
            PipeDoc inputDoc,
            SemanticGraphStepOptions options,
            String pipeStepName,
            List<VectorDirective> directives,
            Map<String, VectorDirective> directiveBySourceLabel,
            Map<String, Set<String>> sentenceConfigsBySource,
            Map<TripleKey, List<SemanticProcessingResult>> byTriple,
            String docHash) {}

    /** One boundary-detection task: all sentence-shaped SPRs for a (source, chunker, model). */
    private record BoundaryTask(TripleKey key, List<SemanticProcessingResult> sprs) {}
}
