# module-semantic-graph — Architecture

Scope: this document covers only the semantic-graph step (Stage 3). It does
not cover the sidecar rewire or the E2E gates that follow it. It is written
against:

- `pipestream-protos/docs/semantic-pipeline/DESIGN.md` (§3–§22.6, with §21 overrides)
- `pipestream-protos/docs/semantic-pipeline/PLAN.md` semantic-graph work packet
- The chunker and embedder modules as they exist on their respective `main`s
- The `pipestream-wiremock-server` invariants + fixtures + step mocks as
  currently published on its `main`

All research was read directly from source; nothing in this doc is paraphrased
from memory.

---

## 1. What the module actually does, in one paragraph

This module takes a Stage-2 `PipeDoc` (fully embedded by the embedder), walks its
`search_metadata.semantic_results[]`, and **appends** new SPRs without
touching the existing ones:

- For every distinct `(source_field, chunker_config, embedder_config)` triple
  present in Stage 2, optionally emit one document-centroid SPR, one
  paragraph-centroid-per-paragraph SPR, and one section-centroid-per-section
  SPR. These are **pure CPU math** over the existing chunk vectors — no
  inference, no I/O, no cache.
- For every `(source_field, boundaryEmbeddingModelId)` pair, optionally find
  the sentence-shaped Stage-2 SPR with that embedder, run topic-boundary
  detection on its sentence vectors, concatenate each resulting sentence
  group's text, **re-embed** the grouped text via DJL Serving, and emit one
  boundary SPR. Re-embedding is the only I/O the module does.
- Lex-sort `semantic_results[]` at the end.

Stage-2 SPRs are preserved byte-for-byte. The module only appends.

---

## 2. Proven facts from research (not opinion)

### 2.1 Batching — the real pattern the embedder uses

`module-embedder/.../pipeline/EmbedderPipelineService.java:388–587` is the
ground truth. The pattern:

```java
int batchSize      = perDirectiveOverride.orElse(modelsConfig.defaults().maxBatchSize());  // default 32
int batchCount     = ceilDiv(totalMisses, batchSize);
int perDocCap      = Math.max(1, Math.min(modelsConfig.defaults().maxSubBatchesPerDoc(), batchCount)); // default 5

Multi.createFrom().range(0, batchCount)
     .onItem().transformToUni(batchIdx -> {
         List<String> sliceTexts   = new ArrayList<>(missTexts.subList(from, to));   // defensive copy
         List<Integer> sliceIndex  = new ArrayList<>(missIndex.subList(from, to));
         return EmbeddingRetryPolicy.withRetry(
                 () -> binding.embed(sliceTexts),                                    // fresh Uni per attempt
                 maxRetries, retryBackoffMs, "model[batch " + batchIdx + "/" + batchCount + "]")
             .map(vectors -> {
                 if (vectors.size() != sliceTexts.size()) throw new IllegalStateException("alignment");
                 for (int k = 0; k < vectors.size(); k++) {
                     perChunkVectorsArray[sliceIndex.get(k)] = vectors.get(k);
                 }
                 return batchCacheMap;
             });
     })
     .merge(perDocCap)           // ← bounded concurrency lives on MERGE, not on transformToUni
     .collect().asList();        // ← happens-before barrier before reading the shared array
```

Key invariants:

1. `Multi.merge(cap)` is the concurrency bound. There is no single-call
   shortcut.
2. The retry loop takes a `Supplier<Uni<T>>`, not a `Uni<T>` — a new Uni per
   attempt. Re-subscribing to the same Uni just re-emits the same failure.
3. Per-sub-batch alignment is verified (`vectors.size() != sliceTexts.size()`
   → `IllegalStateException`). No silent truncation.
4. Sub-batches write to disjoint indices of a shared `float[totalChunks][]`
   array. `collect().asList()` provides the happens-before for safe read.
5. §22.5 regression gate: after all sub-batches return, before building the
   output SPR, verify EVERY slot of the array is populated. If any slot is
   null → `IllegalStateException("Chunk X has no vector... §22.5 regression")`.
   The retry path MUST NOT leave any chunk with a null vector.

### 2.2 Retry — `EmbeddingRetryPolicy` + `EmbeddingRetryClassifier`

- `withRetry(Supplier<Uni<T>>, maxAttempts, backoffMs, label)` is a recursive
  re-subscription loop with exponential backoff (`backoffMs * 2^attemptIndex`).
- `maxAttempts = 0` = fire once, never retry. `maxAttempts = 2` = up to 3 attempts.
- Transient classification (classifier walks the cause chain up to 8 frames):
  `StatusRuntimeException/StatusException` with `UNAVAILABLE`, `DEADLINE_EXCEEDED`,
  or `RESOURCE_EXHAUSTED`; `WebApplicationException` with 5xx; `TimeoutException`;
  `ConnectException`; `SocketTimeoutException`; `IOException`;
  `ProcessingException` (traverses cause).
- Permanent: everything else — 4xx, `IllegalArgumentException`,
  `IllegalStateException`, unknown types (conservative default).

### 2.3 Error model — exceptions at the service, `PROCESSING_OUTCOME_FAILURE` at the wire

Looked at `EmbedderGrpcImpl.java:120–181`. The pattern:

- Service layer throws:
  - `IllegalArgumentException` → INVALID_ARGUMENT class
  - `IllegalStateException` → FAILED_PRECONDITION class
  - Async failures propagate via the returned `Uni`'s failure channel
- gRPC layer:
  - Synchronous `try { parseOptions(); pipelineService.embed(...).map(...).onFailure().recoverWithItem(e -> error(e)); } catch (IAE/ISE sync) { ... }`
  - `recoverWithItem` builds `ProcessingOutcome.PROCESSING_OUTCOME_FAILURE`
    with `error_details = { grpc_status: classify(e).name(), error_message, error_type, error_label, error_cause }`
  - Does NOT throw `StatusRuntimeException` over the wire. The module always
    responds with a `ProcessDataResponse`; the engine reads `grpc_status`
    from `error_details` and routes to quarantine / DLQ accordingly.

### 2.4 SemanticPipelineInvariants shapes (from the wiremock 640-line version)

**`assertPostEmbedder(doc)`** — the module's **input** gate:

1. `search_metadata` set
2. Every SPR: non-empty `embedding_config_id`, `source_field_name`,
   `chunk_config_id`; non-empty `chunks[]`; each chunk has non-empty
   `text_content`, non-empty `vector`, non-empty `chunk_id`, non-negative
   `original_char_start_offset`, `original_char_end_offset >= start`;
   `metadata["directive_key"]` present
3. When directives still present: every SPR's `embedding_config_id` ∈
   advertised `NamedEmbedderConfig.config_id` values
4. `nlp_analysis` preserved: for every unique `source_field_name`, at least
   one SPR with that name has `nlp_analysis` set
5. `source_field_analytics[]` preserved: one entry per unique
   `(source_field, chunk_config_id)` pair in results
6. Lex-sorted on `(source_field_name, chunk_config_id, embedding_config_id, result_id)`

**`assertPostSemanticGraph(doc)`** — the module's **output** contract:

1. All of `assertPostEmbedder`'s per-SPR structural checks still hold on every SPR
2. Centroid SPRs (`chunk_config_id` ends in `_centroid`):
   - `centroid_metadata` set
   - `centroid_metadata.granularity` != `GRANULARITY_LEVEL_UNSPECIFIED`
   - `centroid_metadata.source_vector_count > 0`
   - Exactly 1 chunk
3. Boundary SPRs (`chunk_config_id == "semantic"`):
   - `hasGranularity() == true` AND `granularity == GRANULARITY_LEVEL_SEMANTIC_CHUNK`
   - Non-empty `semantic_config_id`
   - `chunkCount <= MAX_SEMANTIC_CHUNKS_PER_DOC_DEFAULT` (50 default)
4. `source_field_analytics[]` preserved only for NON-centroid, NON-boundary
   (source_field, chunk_config_id) pairs. Stage-3-added SPRs introduce new
   chunk_config_id values (`*_centroid`, `semantic`) that were never produced
   by the chunker step, so they don't contribute to this check.
5. Lex-sorted

**`assertPostSemanticGraph` does NOT** check deep-equal Stage-2 preservation —
the wiremock-server javadoc explicitly notes that this needs the Stage-2 input
as context and is the caller's responsibility. the module's tests must do the
deep-equal check themselves.

### 2.5 Stage-3 SPR shapes — from the wiremock fixture + proto definitions

**Centroid SPR**:
```
result_id:             "stage3-centroid-{granularity}:{docHash}:{source_label}:{chunk_config_id}:{embedder_config_id}"  (per §21.5)
source_field_name:     {source_label}                                       (preserved)
chunk_config_id:       "document_centroid" | "paragraph_centroid" | "section_centroid"
embedding_config_id:   {embedder_config_id}                                 (preserved from Stage-2 source SPR)
result_set_name:       "{source_label}_{embedder_id}_{granularity}"         (from directive template or default)
chunks:                EXACTLY 1
  [0].chunk_id:        "{granularity}:{index}"  (index=0 for doc centroid, per-paragraph/section for others)
  [0].chunk_number:    0
  [0].embedding_info:
      .text_content:   "" or representative excerpt (e.g. joined sentence texts for paragraph centroid)
      .vector:         averaged vector, L2-normalized, length = model dimension
      .chunk_config_id:    same as SPR.chunk_config_id
      (offsets unset OR 0..0; offsets don't meaningfully describe a centroid)
metadata:              {"directive_key": "..."}                             (preserved from source SPR)
granularity:           GRANULARITY_LEVEL_DOCUMENT | PARAGRAPH | SECTION
pooling_method:        POOLING_METHOD_MEAN                                   (L2-normalized but still mean)
parent_result_id:      result_id of the Stage-2 source SPR                   (hierarchy traversal)
centroid_metadata:
  .granularity:        same as SPR.granularity
  .source_vector_count: count of Stage-2 chunks averaged
  .section_title:      set ONLY for SECTION, from DocOutline.Section.title
  .section_depth:      set ONLY for SECTION, from DocOutline.Section.depth
```

**Boundary SPR** (one per (source_field, boundaryEmbeddingModelId)):
```
result_id:             "stage3-boundary:{docHash}:{source_label}:semantic:{boundaryEmbeddingModelId}"
source_field_name:     {source_label}                                       (preserved)
chunk_config_id:       "semantic"                                           (literal)
embedding_config_id:   {boundaryEmbeddingModelId}
result_set_name:       "{source_label}_semantic_{boundaryEmbeddingModelId}" (from template)
chunks:                1..N where N ≤ effectiveMaxSemanticChunksPerDoc()
  each: chunk_id:             "semantic:{index}"
        chunk_number:         0-based
        embedding_info:
            text_content:     space-joined sentence texts for this group
            vector:            re-embedded via boundary model
            chunk_config_id:   "semantic"
        metadata:             {"sentence_span": "start-end", "sentence_count": "N"}
metadata:              {"directive_key": "..."}                             (preserved from source SPR)
granularity:           GRANULARITY_LEVEL_SEMANTIC_CHUNK
pooling_method:        POOLING_METHOD_DIRECT                                 (freshly re-embedded, not pooled)
semantic_config_id:    "semantic:{boundaryEmbeddingModelId}"                 (deterministic, non-empty)
```

### 2.6 DJL extension availability — verified 2026-04-15

`pipestream-embedder-djl` (the intended standalone extension) exists on
GitHub at `ai-pipestream/pipestream-embedder-djl`. Latest commit
`5c6d085 Merge pull request #2 from .../port/http-rest-from-module-embedder`.
Package root: `ai.pipestream.quarkus.djl.runtime`. Contains:

- `client/DjlServingClient.java` — `@RegisterRestClient(configKey = "djl-serving")`,
  methods: `ping()`, `predict(modelName, input)`, `listModels()`, `registerModel(url, modelName, engine)`
- `DjlServingBackend.java` — implements `ai.pipestream.module.embedder.spi.EmbeddingBackend`
  with `name() = "djl-serving"`, `supports(servingName) -> Uni<Boolean>` via
  `DjlModelRegistry.isModelReady`, `embed(servingName, texts) -> Uni<List<float[]>>`
- `DjlModelRegistry.java` — scheduled poll of `/models` every 30s (configurable);
  per-model embedding probe; tracks `READY`/`LOADING`/`ERROR` + probe dims + last failure
- `DjlServingReadinessCheck.java` — MicroProfile readiness gated on
  `reachable && readyModels.nonEmpty()`
- `config/DjlServingRuntimeConfig.java` — `pipestream.djl-serving.*` config
  (`enabled`, `requestTimeout`, `refreshInterval`)

**Published?** No. Probed `ai.pipestream.module:{runtime, pipestream-embedder-djl-runtime, pipestream-embedder-djl}:0.0.1-SNAPSHOT` against both Maven Central and the Sonatype snapshots repo. All three return 404. The repo has no tags.

**SPI dep `module-embedder-api`** — lives in `module-embedder/module-embedder-api`
as a Quarkus extension. Not installed to local Maven (checked
`~/.m2/repository/ai/pipestream/module/`). Has single method set (`name`,
`supports`, `embed`) as noted above.

### 2.7 Wiremock mocks + showcase test — what the module must honor

- `ChunkerStepMock` → returns Stage-1 fixture on `x-module-name: chunker` header
- `EmbedderStepMock` → returns Stage-2 fixture on `x-module-name: embedder` header
- `SemanticGraphStepMock` → returns Stage-3 fixture on `x-module-name: semantic-graph` header
- `SemanticPipelineShowcaseTest` runs all three and asserts each `assertPost*` invariant

the module's integration test can use `EmbedderStepMock` as its **upstream** (to feed
a Stage-2 fixture in) but the module ITSELF replaces `SemanticGraphStepMock` with the
real gRPC impl. The real impl's output must pass `assertPostSemanticGraph`.

The canned `buildStage3PipeDoc()` fixture uses 4-dim deterministic vectors
`[0.1, 0.2, 0.3, 0.4]` and literal strings like `"semantic_v1"` for
`semantic_config_id`. the module's real output will have different values; the
contract is on the **invariant assertions**, not byte-equality to the fixture.

---

## 3. Package layout (matching the embedder repo's conventions)

```
ai.pipestream.module.semanticgraph
├── SemanticGraphGrpcImpl                      @GrpcService @Singleton — thin wire wrapper
├── config
│   ├── SemanticGraphStepOptions               already written in Phase B1; needs no changes
│   └── SemanticGraphStepDefaults              NEW — env-override merger mirror of EmbedderStepDefaults
├── directive
│   └── (nothing module-specific — directives are read from the doc, not resolved per-config)
├── djl
│   ├── DjlServingClient                       local stand-in; package path matches pipestream-embedder-djl
│   └── SemanticGraphEmbedHelper               NEW rewrite — batches + retries per §2.1/§2.2
├── invariants
│   └── SemanticPipelineInvariants             NEW main-source copy, return-String style (embedder pattern)
├── pipeline
│   └── SemanticGraphPipelineService           NEW — the whole algorithm, Mutiny end-to-end
├── retry
│   ├── SemanticGraphRetryPolicy               NEW or shared-copy of EmbeddingRetryPolicy
│   └── SemanticGraphRetryClassifier           NEW or shared-copy of EmbeddingRetryClassifier
└── service
    ├── CentroidComputer                       ALREADY moved in Phase A; unchanged
    └── SemanticBoundaryDetector               ALREADY moved in Phase A; unchanged
```

**Why copy RetryPolicy/RetryClassifier instead of depending on the embedder
jar?** Two reasons:

1. `module-embedder` publishes a `-runner.jar` Quarkus app, not a library jar.
   Its classes are not consumable by another module.
2. Pulling in `module-embedder-api` (the SPI jar) only gets us the
   `EmbeddingBackend` interface, not the retry helpers.

So either copy the two retry classes verbatim (~360 LOC total) or reimplement.
Copy + rename is honest, tracks a real dependency the codebase doesn't
currently manage. I'll note in the class javadoc that it mirrors the
embedder's copy and any divergence is intentional.

### 3.1 `invariants/SemanticPipelineInvariants` — return-String style (main-source pattern)

The wiremock 640-line version uses AssertJ (`assertThat(...).isX(...)`)
which throws `AssertionError`. Main sources cannot depend on AssertJ.

The embedder's main-source version at
`module-embedder/.../invariants/SemanticPipelineInvariants.java:83` uses:
```java
public static String assertPostChunker(PipeDoc doc) {
    if (!doc.hasSearchMetadata()) return "post-chunker: search_metadata not set";
    ...
    return null;
}
```

I'll follow that pattern. the module's main needs at minimum `assertPostEmbedder(doc)
-> String` for the input gate. `assertPostSemanticGraph(doc) -> String` is
useful as a self-check before returning the output (catches bugs early), and
I'll include it.

Tests copy the full AssertJ wiremock version verbatim into `src/test/java/.../invariants/` (not main) for their own pass/fail assertion style. Duplication is intentional per the wiremock javadoc.

### 3.2 `pipeline/SemanticGraphPipelineService`

Entry point:
```java
public Uni<PipeDoc> process(PipeDoc inputDoc, SemanticGraphStepOptions options, String pipeStepName);
```

Responsibilities:
- Assert `assertPostEmbedder(inputDoc)` → if non-null, throw `IllegalStateException`
- Resolve directives, compute `docHash`
- Group Stage-2 SPRs by `(source_field, chunker_config, embedder_config)` triple
- For each group: compute centroids per options flags (pure CPU; offload via `emitOn(workerPool)` if we're in a chain)
- For each `(source_field, boundaryEmbeddingModelId)` pair: find sentence-shaped Stage-2 SPR, run boundary detector, cap-check, batched re-embed
- Append all new SPRs, lex-sort, return

Errors thrown by this service:
- `IllegalArgumentException` — unparseable options, duplicate source_label, malformed directives
- `IllegalStateException` — stage assertion failed, boundary model not loaded, hard cap exceeded, no sentence-shaped SPR for (source, model)

Async failures propagate via the returned Uni's failure channel.

### 3.3 `SemanticGraphGrpcImpl`

Thin wrapper around `SemanticGraphPipelineService`. Mirrors
`EmbedderGrpcImpl` structure:

- Parse options (Jackson from `Struct`), apply env defaults → IAE maps to
  INVALID_ARGUMENT response on parse error
- Call `pipelineService.process(...)` wrapped in try/catch for IAE/ISE sync
  (from pipeline step 1–3) + `.onFailure().recoverWithItem(...)` for async
- Build `ProcessDataResponse`:
  - Success: `PROCESSING_OUTCOME_SUCCESS`, `setOutputDoc(stage3Doc)`, log
    entry summarizing appended SPRs
  - Failure: `PROCESSING_OUTCOME_FAILURE`, `error_details.grpc_status =
    classify(e).name()`, `error_details.error_message / error_type /
    error_label / error_cause`
- `getServiceRegistration` returns module metadata + `SemanticGraphStepOptions.getJsonV7Schema()`

### 3.4 `SemanticGraphStepDefaults` (env-override merger)

Mirror of `EmbedderStepDefaults`. Reads `semanticgraph.defaults.*` properties and applies them to a parsed `SemanticGraphStepOptions` record, filling in null fields. Lets operators override:
- `SEMANTIC_GRAPH_DEFAULTS_COMPUTE_PARAGRAPH_CENTROIDS=true/false`
- `SEMANTIC_GRAPH_DEFAULTS_COMPUTE_SECTION_CENTROIDS=true/false`
- `SEMANTIC_GRAPH_DEFAULTS_COMPUTE_DOCUMENT_CENTROID=true/false`
- `SEMANTIC_GRAPH_DEFAULTS_COMPUTE_SEMANTIC_BOUNDARIES=true/false`
- `SEMANTIC_GRAPH_DEFAULTS_BOUNDARY_EMBEDDING_MODEL_ID=...`
- `SEMANTIC_GRAPH_DEFAULTS_MAX_SEMANTIC_CHUNKS_PER_DOC=...`
- `SEMANTIC_GRAPH_DEFAULTS_MAX_BATCH_SIZE=...`
- `SEMANTIC_GRAPH_DEFAULTS_MAX_SUBBATCHES_PER_DOC=...` (per §21.6/§7.2 infra knob naming)
- `SEMANTIC_GRAPH_DEFAULTS_MAX_RETRY_ATTEMPTS=...`
- `SEMANTIC_GRAPH_DEFAULTS_RETRY_BACKOFF_MS=...`
- (+ the four boundary thresholds)

---

## 4. The Mutiny graph — full

```
SemanticGraphGrpcImpl.processData(request)
│
├── sync phase
│   ├── parseOptions(request.config) → SemanticGraphStepOptions
│   │   └── if parse fails → IAE → createErrorResponse(INVALID_ARGUMENT)
│   ├── stepDefaults.applyTo(raw)
│   └── options.validateForUse() (§21.3 check)
│       └── if violated → IAE (from options.InvalidOptionsException) → createErrorResponse
│
└── pipelineService.process(inputDoc, options, pipeStepName)
    │   — everything below is inside the pipeline service, returning Uni<PipeDoc>
    │
    ├── sync validation
    │   ├── invariants.assertPostEmbedder(inputDoc) → if non-null → ISE
    │   ├── read directives, check source_label uniqueness → IAE on dup
    │   ├── docHash = sha256b64url(docId)
    │   └── group Stage-2 SPRs by (source_field, chunker_config, embedder_config) triple
    │
    ├── centroid pass (pure CPU)
    │   for each group:
    │     if options.effectiveComputeDocumentCentroid():
    │       result = CentroidComputer.averageAndNormalize(allChunkVectors)
    │       stage3Sprs += buildCentroidSpr(group, DOCUMENT, result, ...)
    │     if options.effectiveComputeParagraphCentroids():
    │       paragraphs = CentroidComputer.detectParagraphBoundaries(fullText, chunkOffsets)
    │       if paragraphs.nonEmpty:
    │         results = CentroidComputer.computeParagraphCentroids(chunkVectors, chunkTexts, fullText, chunkOffsets)
    │         for each r: stage3Sprs += buildCentroidSpr(group, PARAGRAPH, r, ...)
    │     if options.effectiveComputeSectionCentroids() && doc.hasDocOutline():
    │       sections = extractSections(doc.docOutline)
    │       results = CentroidComputer.computeSectionCentroids(chunkVectors, chunkTexts, chunkOffsets, sections)
    │       for each r: stage3Sprs += buildCentroidSpr(group, SECTION, r, ...)
    │
    ├── boundary pass (reactive, batched)
    │   if options.effectiveComputeSemanticBoundaries():
    │     modelId = options.requireBoundaryEmbeddingModelId()
    │     // check loaded
    │     loadedCheckUni = embedHelper.isModelLoaded(modelId)
    │     then:
    │       if !loaded → Uni.failure(ISE "boundary model not loaded")
    │       else:
    │         for each source_field that has a sentence-shaped Stage-2 SPR with embedder_config_id == modelId:
    │           sentences = thatSpr.chunks
    │           vectors   = [c.vector for c in sentences]
    │           texts     = [c.text  for c in sentences]
    │           offsets   = [[c.start, c.end] for c in sentences]
    │           similarities = SemanticBoundaryDetector.computeConsecutiveSimilarities(vectors)
    │           boundaries   = SemanticBoundaryDetector.findBoundaries(vectors, thr, pct)
    │           groups       = SemanticBoundaryDetector.groupByBoundaries(range(sentences), boundaries)
    │           groups       = enforceMinChunkSize(groups, ...)
    │           groups       = enforceMaxChunkSize(groups, similarities, boundaries, maxSentences)
    │           // hard cap check
    │           if groups.size() > effectiveMaxSemanticChunksPerDoc():
    │             → Uni.failure(ISE "hard cap exceeded, got X need ≤ Y")
    │           groupedTexts = [join(sentenceTexts[indices]) for indices in groups]
    │           // BATCHED RE-EMBED (NOT one big predict call)
    │           groupedVectors = embedHelper.embed(modelId, groupedTexts)   // see §4.1 for batching
    │           stage3Sprs += buildBoundarySpr(source_field, modelId, groups, sentences, groupedTexts, groupedVectors)
    │
    ├── assemble output
    │   merged = stage2Sprs + stage3Sprs
    │   merged.sort(LEX_ORDER)
    │   outputDoc = inputDoc.toBuilder().setSearchMetadata(sm.toBuilder().clearSemanticResults().addAll(merged)).build()
    │
    ├── self-check (defensive; catches bugs before the wire)
    │   invariants.assertPostSemanticGraph(outputDoc) → if non-null → ISE "Module produced invalid output: ..."
    │
    └── emit outputDoc via Uni<PipeDoc>

gRPC layer:
├── Uni<PipeDoc> → map → buildSuccessResponse
└── .onFailure().recoverWithItem(e ->
      ErrorCategory cat = SemanticGraphRetryClassifier.classify(e);
      return ProcessDataResponse(FAILURE,
              error_details = { grpc_status: cat.name(), error_message, ... })
```

### 4.1 SemanticGraphEmbedHelper — batched per §2.1

**Not one predict call.** The helper internally slices into
`effectiveBatchSize`-sized sub-batches and uses `Multi.merge(cap)` exactly
like the embedder.

```java
public Uni<List<float[]>> embed(String modelId, List<String> texts, int batchSize, int perDocCap, int maxRetries, long retryBackoffMs) {
    if (texts.isEmpty()) return Uni.createFrom().item(List.of());
    int total = texts.size();
    int batchCount = (total + batchSize - 1) / batchSize;
    int cap = Math.max(1, Math.min(perDocCap, batchCount));
    float[][] out = new float[total][];

    return Multi.createFrom().range(0, batchCount)
        .onItem().transformToUni(batchIdx -> {
            int from = batchIdx * batchSize;
            int to   = Math.min(from + batchSize, total);
            List<String> slice     = new ArrayList<>(texts.subList(from, to));
            int sliceStart         = from;
            return SemanticGraphRetryPolicy.withRetry(
                    () -> djl.predict(modelId, new JsonObject().put("inputs", new JsonArray(slice))),
                    maxRetries, retryBackoffMs,
                    modelId + "[batch " + batchIdx + "/" + batchCount + "]")
                .map(response -> {
                    List<float[]> parsed = parseBatch(response);
                    if (parsed.size() != slice.size()) {
                        throw new IllegalStateException(
                            "DJL returned " + parsed.size() + " vectors for " + slice.size() +
                            " inputs on model '" + modelId + "' batch " + batchIdx);
                    }
                    for (int k = 0; k < parsed.size(); k++) {
                        out[sliceStart + k] = parsed.get(k);
                    }
                    return true;
                });
        })
        .merge(cap)
        .collect().asList()
        .map(ignored -> {
            // §22.5-style final verification
            for (int i = 0; i < out.length; i++) {
                if (out[i] == null || out[i].length == 0) {
                    throw new IllegalStateException(
                        "SemanticGraphEmbedHelper produced a null/empty vector at slot " + i +
                        " for model '" + modelId + "' — batch slice missed an index");
                }
            }
            return Arrays.asList(out);
        });
}
```

This is the batching the user called me out for missing. For a 50-group
boundary re-embed with default `batchSize=32` and `perDocCap=5`, the call
produces 2 sub-batches that can run in parallel (cap=min(5, 2)=2). For a
trivial doc with 5 groups, 1 sub-batch, cap clamps to 1. Concurrency is
self-bounded.

### 4.2 Model-loaded probe — how often to call `/models`

Options:

- **Per-doc probe** (naive): call `djl.listModels()` once per `processData`.
  Simple. Adds one HTTP round-trip per doc. Latency floor ~5ms on localhost.
  Burns connection-pool budget.
- **Cached probe with TTL** (Quarkus `@CacheResult` 30s TTL): one
  `listModels` per 30s window, shared across all docs. Matches the
  `pipestream-embedder-djl` `DjlModelRegistry` cadence without implementing
  the full scheduled poller.
- **Full `DjlModelRegistry` clone**: port the scheduled poller + per-model
  embedding probe. ~240 LOC. Correct long-term solution when the extension
  finally publishes.

**My recommendation: cached probe with 30s TTL via `@CacheResult("djl-models-loaded")`.**
Simple, matches the extension's cadence. When `pipestream-embedder-djl`
publishes and we swap to its `DjlModelRegistry.isModelReady()`, the cache
layer goes away.

---

## 5. Dependency decision for DJL integration

Three options for the `DjlServingClient` + `isModelLoaded` story. All three
honor the user's "thin REST client over djl-serve" framing:

### Option A — `includeBuild` the `pipestream-embedder-djl` repo as a Gradle composite

```groovy
// settings.gradle
includeBuild('../pipestream-embedder-djl')

// build.gradle
dependencies {
    implementation 'ai.pipestream.module:runtime'  // resolved from the included build
}
```

- Pro: zero duplication. The module always compiles against the real source.
- Pro: no publication dependency. Works in dev and in CI as long as the
  sibling repo is present.
- Con: CI has to clone + ./gradlew build the sibling repo first. Dockerfile
  + GitHub Actions need adjusting.
- Con: Two-repo dev workflow — changes in the extension require a rebuild
  here, which Gradle's composite build handles but isn't zero-friction.

### Option B — publish `pipestream-embedder-djl` SNAPSHOT first, depend normally

```groovy
implementation "ai.pipestream.module:pipestream-embedder-djl-runtime:${version}"
```

- Pro: cleanest; standard Maven dep.
- Con: requires first PR against `pipestream-embedder-djl` to add
  `publishAllPublicationsToCentralPortalSnapshots` and run it. Out of scope
  for this session unless you prioritize.

### Option C — local stand-in for now, swap when (A) or (B) lands

- Copy the interface `DjlServingClient` into
  `ai.pipestream.module.semanticgraph.djl` with the same `configKey =
  "djl-serving"`. Either keep the `pipestream-embedder-djl`'s
  `ai.pipestream.quarkus.djl.runtime.client` package (so swap = import
  rename only) or use the module's own package.
- Pro: zero blockers, the module lands standalone.
- Con: duplicated interface. Risks drift if `pipestream-embedder-djl`
  modifies its interface before we swap. Drift is mitigated by using the
  same method names + signatures, which have been stable for ~2 weeks.

**My recommendation: C for this PR, follow up with B (publish the extension) as a separate small PR later.** Gets the module unblocked. The local copy has a doc comment naming the swap plan and the upstream package path.

---

## 6. Error taxonomy — every code path mapped

| Condition | Where raised | Exception type | Classifier output | Response category |
|---|---|---|---|---|
| `request == null` or `!hasDocument` | `SemanticGraphGrpcImpl.processData` sync | `IllegalArgumentException` | `INVALID_ARGUMENT` | FAILURE + grpc_status=INVALID_ARGUMENT |
| `ProcessConfiguration.json_config` doesn't parse | `parseOptions` | IAE wrapping Jackson `JsonProcessingException` | `INVALID_ARGUMENT` | FAILURE + grpc_status=INVALID_ARGUMENT |
| `options.validateForUse()` fails (e.g. boundaries on + model-id blank) | options record | `InvalidOptionsException extends IAE` | `INVALID_ARGUMENT` | FAILURE + grpc_status=INVALID_ARGUMENT |
| Missing `vector_set_directives` OR empty directives list | pipeline step 1 | `IllegalStateException` | `FAILED_PRECONDITION` | FAILURE + grpc_status=FAILED_PRECONDITION |
| Duplicate `source_label` | pipeline step 1 | `IllegalArgumentException` | `INVALID_ARGUMENT` | FAILURE + grpc_status=INVALID_ARGUMENT |
| `assertPostEmbedder` violation (any) | pipeline step 2 | `IllegalStateException` | `FAILED_PRECONDITION` | FAILURE + grpc_status=FAILED_PRECONDITION |
| Boundaries on but `isModelLoaded` returns false | pipeline boundary pass | `IllegalStateException` | `FAILED_PRECONDITION` | FAILURE + grpc_status=FAILED_PRECONDITION |
| Boundaries on but no sentence-shaped Stage-2 SPR for `(source, model)` | pipeline boundary pass | `IllegalStateException` | `FAILED_PRECONDITION` | FAILURE + grpc_status=FAILED_PRECONDITION |
| Hard cap: `groups.size() > effectiveMaxSemanticChunksPerDoc()` | pipeline boundary pass | `IllegalStateException` | `FAILED_PRECONDITION` (arguable; could be INTERNAL) | FAILURE + grpc_status=FAILED_PRECONDITION |
| DJL alignment mismatch (`vectors.size() != texts.size()` per sub-batch) | embed helper | `IllegalStateException` | `FAILED_PRECONDITION` | FAILURE + grpc_status=FAILED_PRECONDITION |
| §22.5-style final gate: null vector slot after batches | embed helper | `IllegalStateException` | `FAILED_PRECONDITION` | FAILURE + grpc_status=FAILED_PRECONDITION |
| DJL transient (5xx, connect, timeout) after retry budget exhausted | embed helper Uni failure | pass-through transport exception | `UNAVAILABLE` | FAILURE + grpc_status=UNAVAILABLE |
| DJL permanent (4xx, model not found) | embed helper Uni failure | `WebApplicationException` or similar | `INVALID_ARGUMENT` | FAILURE + grpc_status=INVALID_ARGUMENT |
| Self-check: output fails `assertPostSemanticGraph` | pipeline emit | `IllegalStateException("Module produced invalid output: ...")` | `FAILED_PRECONDITION` | FAILURE + grpc_status=FAILED_PRECONDITION |
| Unknown exception bubbling up | catch-all in `onFailure().recoverWithItem` | anything | `INTERNAL` | FAILURE + grpc_status=INTERNAL |

**DESIGN.md §10.1 hard cap = INTERNAL** ambiguity: the spec classifies it
INTERNAL ("never silently truncate") because it's a contract violation of
the caller's input config (thresholds produce way too many groups). I lean
FAILED_PRECONDITION because the issue is the input config, not a runtime
internal error — but I'll follow DESIGN if you prefer the literal text.

---

## 7. Open questions I need you to decide before I code

1. **DJL dep strategy**: A (composite), B (publish first), or C (local stand-in for now)? My rec: C for this PR.
2. **Boundary source-SPR detection**: strict `chunk_config_id == "sentences_internal"` only, OR walk the directive's NamedChunkerConfig list and accept any chunker whose algorithm is SENTENCE? My rec: **both** — prefer `sentences_internal`, fall back to "algorithm=SENTENCE" match from the directive. If neither, FAIL.
3. **Paragraph/section centroids on non-sentence chunks**: DESIGN.md §7.3 reads "one per enabled granularity, per `(source_field, chunker_config, model)` combo" — which would mean running paragraph detection on token-chunker SPRs too, producing likely-degenerate results (token chunks don't have `\n\n` gaps between them). **My rec: restrict paragraph/section centroids to sentence-shaped SPRs** (where `\n\n` gap detection works). Document centroids produced for every Stage-2 SPR unconditionally.
4. **Hard-cap → INTERNAL vs FAILED_PRECONDITION**: literal DESIGN vs pragmatic? My rec: **FAILED_PRECONDITION** with a clear message naming the config knobs to tune.
5. **Retry classifier + policy**: copy the two embedder classes verbatim, rename-only, OR re-implement slimmer? My rec: **copy verbatim**, name `SemanticGraphRetryPolicy` / `SemanticGraphRetryClassifier`. ~360 LOC shared logic, zero re-invention risk.
6. **Output self-check (`assertPostSemanticGraph` before emit)**: include it or rely only on tests? My rec: **include it**. Catches bugs before they hit the engine, and the return-String version is cheap. If it ever fires in prod, it means our own code produced invalid output — that deserves FAILED_PRECONDITION + DLQ, not silent emission.
7. **Single-placeholder-SPR edge cases**: what if Stage 2 has ZERO SPRs (doc had no matching text for any directive)? Currently the chunker emits zero SPRs; embedder passes through; the semantic-graph step sees zero SPRs, emits zero new SPRs, lex-sorts empty, returns. That's valid per §5.2/5.3. Confirm.
8. **Result-set-name template for boundary SPR**: DESIGN §4.3 suggests `"{source_label}_semantic_{embedder_id}"`. The directive's `field_name_template` placeholders are `{source_label}, {chunker_id}, {embedder_id}`. For boundary, `{chunker_id}="semantic"` works naturally in the template. Confirm or override.
9. **`semantic_config_id` value**: I propose `"semantic:" + boundaryEmbeddingModelId`. Invariant only requires non-empty. Any preference?

---

## 8. Test matrix (Phase D)

### 8.1 Unit tests — `SemanticGraphStepOptions` (DONE, 17 tests)

Already covered in Phase B1.

### 8.2 Unit tests — `SemanticGraphEmbedHelper` (REPLACES current 15 tests)

- Happy path single batch, multi-batch, empty input, blank model id
- Row-count mismatch per batch → IllegalStateException
- Null row in response → IllegalStateException
- §22.5 post-assembly gate: force a sub-batch to return empty `float[]` → final check catches it
- Transient retry: ConnectException, SocketTimeoutException, HTTP 503, HTTP 504, HTTP 408, HTTP 429 (and cause-chain traversal)
- Permanent no-retry: HTTP 400, HTTP 404, HTTP 422
- Retry exhausted propagates last failure
- Concurrency cap: with perDocCap=1, batchCount=5, verify sequential execution (via mock latency + timing — or via a gate pattern)
- `isModelLoaded`: present, missing, blank id, listModels fails

### 8.3 Unit tests — `SemanticGraphPipelineService`

For each of the following, fixture-build a Stage-2 `PipeDoc` via the
wiremock `SemanticFixtureBuilder.buildStage2PipeDoc()` (or hand-rolled):

- **Happy path, all flags off** → pass-through with lex sort only, no new SPRs, `assertPostSemanticGraph` passes, Stage-2 prefix byte-identical
- **Doc centroid only** (others off, boundaries off): N Stage-2 SPRs in → N stage-2 + N centroid SPRs out
- **Paragraph centroids** with doc text containing `\n\n` gaps: verify paragraph SPR count = paragraph count
- **Section centroids** with doc having DocOutline.sections: verify section SPR count = sections count; `section_title` + `section_depth` populated
- **Boundaries** with a sentence-shaped SPR present and mock DJL returning deterministic vectors: verify 1 boundary SPR, chunks count matches detected groups, vectors have expected dim, `semantic_config_id` non-empty, `granularity=SEMANTIC_CHUNK`
- **Parse error** upstream (invalid JSON config) → IAE → caller maps INVALID_ARGUMENT
- **Missing directives** on input doc → ISE → FAILED_PRECONDITION
- **`assertPostEmbedder` fails** on input → ISE → FAILED_PRECONDITION
- **Boundaries on, model not loaded** (mock `isModelLoaded` returns false) → ISE → FAILED_PRECONDITION
- **Boundaries on, no sentence-shaped SPR** for (source, model) → ISE → FAILED_PRECONDITION
- **Hard cap exceeded**: force the boundary detector to produce 51 groups with `effectiveMaxSemanticChunksPerDoc()=50` → ISE
- **Transient DJL on boundary, retries, succeeds** → 1 boundary SPR emitted
- **Transient DJL retries exhausted** → Uni failure → UNAVAILABLE
- **Permanent DJL (400)** → Uni failure → INVALID_ARGUMENT
- **Lex-sort invariant**: shuffle input SPRs, verify output order matches `(source_field, chunker, embedder, result_id)` lex
- **Stage-2 preservation**: deep-equal the pre-append portion of output's `semantic_results[]` to input (byte-identical)
- **Output self-check catches a buggy impl**: inject a controller flag that skips `source_field_analytics` clone → self-check catches → ISE

### 8.4 Unit tests — `SemanticGraphGrpcImpl`

- Successful parse + successful pipeline → SUCCESS response, output doc set, log entry included
- Parse error → FAILURE + `grpc_status = INVALID_ARGUMENT`
- Pipeline sync ISE → FAILURE + `grpc_status = FAILED_PRECONDITION`
- Pipeline async ISE → FAILURE + `grpc_status = FAILED_PRECONDITION`
- Pipeline async UNAVAILABLE-class → FAILURE + `grpc_status = UNAVAILABLE`
- `getServiceRegistration` returns module name, version, schema

### 8.5 Integration test — real DJL + EmbedderStepMock upstream

`@QuarkusIntegrationTest` with:
- Testcontainer running `djl-serving` with `sentence-transformers/all-MiniLM-L6-v2` loaded, OR skip-with-reason via JUnit Assumptions when the container can't start
- `EmbedderStepMock` served by an in-test WireMock instance (optional — we can feed Stage-2 directly without a mock layer; the mock is more useful for sidecar-style plumbing tests)

Test: feed a Stage-2 doc whose `sentences_internal × minilm` SPR has 10
sentence chunks with real MiniLM vectors. Run the module with
`compute_semantic_boundaries=true`. Assert:
- `assertPostSemanticGraph` passes
- 1 boundary SPR with between 2 and 10 chunks (depends on thresholds)
- Each boundary chunk's vector has dim 384 (MiniLM-L6-v2)
- Stage-2 prefix byte-identical

---

## 9. Phases (rework)

Tasks #68, #69, #70 already done in previous session (Phase A, B1, B2) BUT
Phase B2's `SemanticGraphEmbedHelper` is now **superseded** by the batched
rewrite in §4.1. I'll rewrite it in the first code commit after this doc
is approved.

Proposed new phase ordering:

- **Phase B2-rewrite**: batched `SemanticGraphEmbedHelper` + `SemanticGraphRetryPolicy` + `SemanticGraphRetryClassifier` + tests
- **Phase C1**: `SemanticPipelineInvariants` (main-source return-String style for `assertPostEmbedder` + `assertPostSemanticGraph`)
- **Phase C1b**: `SemanticGraphStepDefaults` env override
- **Phase C2**: `SemanticGraphPipelineService` + unit tests per §8.3
- **Phase C3**: `SemanticGraphGrpcImpl` + unit tests per §8.4
- **Phase D2**: `@QuarkusIntegrationTest` per §8.5
- **Phase E**: full `./gradlew check`, measure p95, report numbers

---

## 10. Explicit non-goals for the semantic-graph module

- No OpenSearch interaction. This module emits a PipeDoc; the sink / indexing does
  the rest.
- No Redis. Centroids are CPU-local; boundary re-embed is ≤50 vectors per
  doc, not worth caching.
- No cross-doc state. Everything is per-doc.
- No engine changes. This module is a `PipeStepProcessorService` like every other
  module.
- No proto changes. Every field is existing (`CentroidMetadata`,
  `GranularityLevel`, `PoolingMethod`, `semantic_config_id`, etc.).
- No sidecar changes. R4 owns that.
- No R5 gates from this module. R5 is a separate worktree.

---

**End of architecture doc. Waiting on §7 answers before touching code.**
