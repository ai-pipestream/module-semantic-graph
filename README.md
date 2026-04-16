# module-semantic-graph

Stage 3 of the three-step semantic pipeline (DESIGN.md §7.3). Consumes
Stage-2 `PipeDoc`s (chunks fully embedded by `module-embedder`) and emits
Stage-3 `PipeDoc`s by appending:

- **document / paragraph / section centroids** computed from the existing
  Stage-2 chunk vectors (pure CPU, no inference); and
- **semantic-boundary SPRs** produced by detecting topic breaks in the
  sentence-shaped SPR's vectors and re-embedding each grouped text span
  via DJL Serving.

Stage-2 SPRs are preserved byte-for-byte; Stage-3 only appends.

This module is a per-doc stateless `PipeStepProcessorService`. Cross-doc
concurrency comes from the engine's slot config on the node, not in-process
scatter-gather.

The previous orchestrator (`module-semantic-manager`) was replaced by the
three-step split: `module-chunker` → `module-embedder` → `module-semantic-graph`.

## Configuration — `SemanticGraphStepOptions`

Parsed from `ProcessConfiguration.json_config` (a `google.protobuf.Struct`).
Every field accepts both snake_case and camelCase. Defaults match DESIGN.md
§6.3. Per DESIGN.md §21.1 there is no fallback: parse errors raise
`INVALID_ARGUMENT`.

| Field | Default | What it does |
|---|---|---|
| `compute_paragraph_centroids` | `true` | Emit one `paragraph_centroid` SPR per paragraph per Stage-2 triple. Paragraph boundaries detected from `\n\n` gaps in the source text (body / title). Skipped when source text cannot be resolved. |
| `compute_section_centroids` | `true` | Emit one `section_centroid` SPR per Section in `DocOutline` per Stage-2 triple. Requires `search_metadata.doc_outline` present with `Section.char_start_offset` populated. |
| `compute_document_centroid` | `true` | Emit one `document_centroid` SPR per Stage-2 triple. Always computable — averages all chunk vectors in the SPR. |
| `compute_semantic_boundaries` | `true` | Run topic-boundary detection on `sentences_internal` vectors and re-embed each group via `boundary_embedding_model_id`. |
| `boundary_embedding_model_id` | **REQUIRED** when `compute_semantic_boundaries=true` | DJL model id. §21.3: no "first available" fallback — absent / not-loaded → `FAILED_PRECONDITION`. |
| `max_semantic_chunks_per_doc` | `50` | Hard cap on boundary group count per doc. Exceeding it raises `INTERNAL`; R3 never silently truncates. |
| `boundary_similarity_threshold` | `0.5` | Cosine similarity below which a sentence break becomes a group boundary. Range `[-1.0, 1.0]`. |
| `boundary_percentile_threshold` | `20` | Percentile-based boundary trigger (0–100). |
| `boundary_min_sentences_per_chunk` | `2` | Merges small groups with the higher-similarity neighbor. |
| `boundary_max_sentences_per_chunk` | `30` | Splits large groups at their lowest internal similarity. Must be `>= min_sentences`. |
| `max_batch_size` | `32` | DJL predict call fan-in cap for the boundary re-embed. |
| `max_subbatches_per_doc` | `5` | `Multi.merge(cap)` concurrency on sub-batch dispatch. |
| `max_retry_attempts` | `2` | Retries on transient DJL failures (5xx, connect, timeout) per sub-batch. |
| `retry_backoff_ms` | `100` | Base for exponential backoff: `backoff × 2^attempt`. |

## Sentence-shaped SPR detection

Boundary detection needs a Stage-2 SPR whose `chunk_config_id` is sentence-
shaped AND whose `embedding_config_id == boundary_embedding_model_id`. R3
accepts either of the following as sentence-shaped for a given `source_label`:

1. `chunk_config_id == "sentences_internal"` — the chunker's always-emit
   sentence pass (DESIGN.md §4.1 alwaysEmitSentences).
2. Any `NamedChunkerConfig.config_id` declared on the directive whose
   `config.algorithm == "SENTENCE"` (case-insensitive).

When both are present for the same source, `sentences_internal` wins
(deterministic).

If no sentence-shaped SPR is found for `(source, boundary_model)`, R3 fails
with `FAILED_PRECONDITION`.

## Error semantics (gRPC status)

| Condition | Category stamped as `grpc_status` on `error_details` |
|---|---|
| Null request or missing doc | `INVALID_ARGUMENT` |
| Unparseable `json_config` | `INVALID_ARGUMENT` |
| `SemanticGraphStepOptions.validateForUse()` violation | `INVALID_ARGUMENT` |
| `assertPostEmbedder(doc)` violation | `FAILED_PRECONDITION` |
| Duplicate `source_label` on directive | `INVALID_ARGUMENT` |
| Boundaries on + model not loaded (§21.3) | `FAILED_PRECONDITION` |
| Boundaries on + no sentence SPR for the model | `FAILED_PRECONDITION` |
| Hard cap `max_semantic_chunks_per_doc` exceeded | `FAILED_PRECONDITION` (clear message naming the knobs) |
| DJL 4xx / unknown model | `INVALID_ARGUMENT` |
| DJL 5xx / connect / timeout after retry budget exhausted | `UNAVAILABLE` |
| Internal alignment mismatch / §22.5 regression gate | `FAILED_PRECONDITION` |
| Output self-check (`assertPostSemanticGraph`) fails | `FAILED_PRECONDITION` (module bug — quarantine) |
| Anything else | `INTERNAL` |

All failures produce `PROCESSING_OUTCOME_FAILURE` with the category stamped
onto `error_details.grpc_status`; no `StatusRuntimeException` is thrown over
the gRPC wire.

## Metrics — `/q/metrics`

Published via Micrometer + Prometheus. Meters:

- `semanticgraph.inflight.docs` — D(f) gauge
- `semanticgraph.docs.processed.total` / `docs.failed.total` — counters
- `semanticgraph.doc.processing` — per-doc end-to-end timer (§13 gate; p50/p95/p99)
- `semanticgraph.centroid.duration` — centroid pass timer
- `semanticgraph.boundary.duration` — boundary pass timer
- `semanticgraph.boundary.groups.per_doc` — group count distribution (recorded BEFORE hard-cap check)
- `semanticgraph.centroids.per_doc` — centroid SPR count per doc
- `semanticgraph.stage2.sprs.per_doc` — input Stage-2 SPR count per doc

## Running tests

### Unit tests

```bash
./gradlew test
```

107 unit tests across: options parse + validation, embed-helper batching + retry, invariants (post-embedder + post-graph), pipeline service happy/error paths, centroid math, boundary math.

### Integration test (`SemanticGraphPipelineServiceIT`) — requires a running DJL Serving

The IT is `@QuarkusTest`-based and hits a real DJL Serving instance with a
real MiniLM model. It skips with a clear reason when DJL isn't reachable.

**Bring up DJL (CPU variant, simplest):**

```bash
docker run -d --name r3-djl --rm -p 18080:8080 deepjavalibrary/djl-serving:0.36.0-cpu
# wait ~5 seconds for startup
curl -X POST 'http://localhost:18080/models?url=djl%3A%2F%2Fai.djl.huggingface.pytorch%2Fsentence-transformers%2Fall-MiniLM-L6-v2&model_name=all-MiniLM-L6-v2&engine=PyTorch&batch_size=1&max_batch_delay=0&min_worker=1&max_worker=2&job_queue_size=1000&translatorFactory=ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory&synchronous=true'
```

**Or GPU variant (matches module-embedder's tuned config — see
`/work/modules/module-embedder/docs/ai-slop/djl-serving-config.md`):**

```bash
docker run -d --name r3-djl --rm --gpus all -p 18080:8080 \
    -e JAVA_OPTS="-Xmx16g -Xms8g" \
    deepjavalibrary/djl-serving:0.36.0-pytorch-gpu
# same POST as above; add more models by repeating with different model_name / url
```

**Run the IT:**

```bash
./gradlew test --tests 'SemanticGraphPipelineServiceIT' \
    -Ddjl.host=localhost -Ddjl.port=18080
```

The `test` task forwards `-Ddjl.*` sysprops to the forked JVM (see
`build.gradle`'s `systemProperties.putAll(... findAll { startsWith("djl.") })`).

### Measured numbers — DESIGN.md §13 gate

On an RTX 4080 SUPER with MiniLM loaded at `batch_size=1 max_worker=2 max_batch_delay=0`, warm workers, 10 iterations on a 6-sentence fixture:

```
p50 = 7 ms, p95 = 9 ms, p99 = 9 ms   (§13 gate ≤ 500 ms — MET with 55× margin)
```

CPU variant (same fixture, MiniLM CPU-only): p50 = 26 ms, p95 = 27 ms.

## Models that need the Python-engine path

DJL's `djl://` traced-torchscript models have known bugs with certain
architectures on GPU:

- **all-mpnet-base-v2** — TorchScript device-placement bug: `two devices,
  cuda:0 and cpu!` on the relative-attention-bias embedding lookup.
- **intfloat/e5-small-v2** and **intfloat/e5-large-v2** — CUDA nvrtc compile
  error on the fp16 negative-infinity constants (`-3.402823466385289e+38.f`).

Workaround: register these via DJL's Python engine with a small custom
handler (`model.py` + `serving.properties`). The `bge-m3-model/` directory
at `/work/opensearch-grpc-knn/lucene-test-data/bge-m3-model/` is a working
example — the same pattern applies to mpnet and e5-* using `AutoTokenizer` +
`AutoModel` with mean pooling and L2 normalization.

## Relationship to other modules

- **Upstream:** `module-embedder` produces Stage 2 and passes the doc to
  this step. R3 asserts `assertPostEmbedder` on entry.
- **Downstream:** `opensearch-sink` (or whatever sink is wired in the
  graph) consumes the Stage-3 doc. R3's output SPRs carry `granularity`,
  `pooling_method`, `parent_result_id`, and `centroid_metadata` so the sink
  can route centroids to pooled fields and boundary SPRs to semantic fields
  without needing to parse `chunk_config_id`.
- **DJL:** this module depends on a DJL Serving instance reachable at
  `quarkus.rest-client.djl-serving.url`. Per §21.3 the model named by
  `boundary_embedding_model_id` MUST be loaded before R3 runs; R3 caches
  the `/models` probe for 30 s via `@CacheResult` to amortize the check.
