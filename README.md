# module-semantic-graph

Stage 3 of the three-step semantic pipeline (DESIGN.md §7.3).

Consumes Stage 2 PipeDocs (chunks fully embedded by `module-embedder`) and emits
Stage 3 PipeDocs by:

- computing centroid SemanticProcessingResults at document / paragraph / section
  granularity, and
- optionally running semantic-boundary detection on `sentences_internal` and
  re-embedding the grouped text via the `quarkus-djl-embeddings` extension.

Stage 2 SPRs are preserved byte-for-byte; Stage 3 only appends.

This module is a per-doc stateless `PipeStepProcessorService` — cross-doc
concurrency comes from the engine's slot config, not in-process scatter-gather.

The previous scatter-gather orchestrator (`module-semantic-manager`) was
replaced by the three-step split (`module-chunker` → `module-embedder` →
`module-semantic-graph`).
