# Semantic Chunking with Hierarchical Centroid Vectors

## Goal

Add semantic chunking to the Semantic Manager — find natural topic boundaries by measuring cosine similarity between consecutive sentence embeddings, then produce hierarchical centroid vectors (paragraph, section, document) from the sentence embeddings at near-zero extra cost.

## Architecture

**Orchestrator-side semantic logic, reusing existing chunker + embedder gRPC services.**

The semantic chunking algorithm lives in `SemanticIndexingOrchestrator`. The chunker stays mechanical (sentence splitting), the orchestrator adds intelligence (boundary detection via embeddings). No new gRPC services needed.

### Data Flow

```
Phase 1a: Sentence Split
    ChunkerStreamClient (SENTENCE algorithm) → List<StreamChunksResponse> sentences

Phase 1b: Sentence Embed (boundary detection pass)
    EmbedderStreamClient → Map<chunkId, float[]> sentenceVectors

Phase 1c: Boundary Detection (pure Java, orchestrator)
    cosine similarity between consecutive vectors → breakpoints
    group sentences between breakpoints → semantic chunks
    enforce min/max chunk size constraints

Phase 2a: Chunk Embed (final pass for grouped chunks)
    Concatenate grouped sentences → re-embed as chunks
    → SemanticProcessingResult with chunk_config_id="semantic"

Phase 2b: Centroid Computation (pure Java, averaging + L2 normalize)
    → SemanticProcessingResult "sentence" (reuse vectors from 1b)
    → SemanticProcessingResult "paragraph_centroid"
    → SemanticProcessingResult "section_centroid" (if DocOutline available)
    → SemanticProcessingResult "document_centroid"

Phase 3: Assemble (existing pattern)
    → Up to 5 SemanticProcessingResults on enriched PipeDoc
```

### Cost

For a 1000-sentence document with 384-dim embeddings:
- Sentence embedding pass: 1000 calls (batched, the real cost)
- Semantic chunk embedding pass: ~20 calls (re-embedded for quality)
- Paragraph centroid computation: ~500K float additions (negligible)
- Section/document centroids: trivial averaging
- **The pooling is free. Embedding cost dominated by sentence pass already needed for boundary detection.**

## Proto Changes (pipestream-protos)

### 1. ChunkAlgorithm enum

Verify `SEMANTIC = 3` exists in proto (already in Java ChunkingAlgorithm enum).

### 2. SemanticChunkingConfig message

New message in `semantic_indexing.proto`:

```protobuf
message SemanticChunkingConfig {
  // Absolute similarity cutoff (0.0-1.0). Break when consecutive similarity drops below.
  float similarity_threshold = 1;
  // Break at the bottom N% of similarity transitions (0-100).
  int32 percentile_threshold = 2;
  // Minimum sentences per semantic chunk (merge small chunks).
  int32 min_chunk_sentences = 3;
  // Maximum sentences per semantic chunk (force split large chunks).
  int32 max_chunk_sentences = 4;
  // Embedding model for sentence boundary detection (can differ from indexing model).
  string sentence_embedding_model = 5;
  // Whether to store sentence-level vectors as a separate result set.
  bool store_sentence_vectors = 6;
  // Whether to compute and store centroid vectors (paragraph, section, document).
  bool compute_centroids = 7;
}
```

### 3. ChunkerConfig extension

Add `semantic_config` field to existing `ChunkerConfig`:

```protobuf
message ChunkerConfig {
  ChunkAlgorithm algorithm = 1;
  int32 chunk_size = 2;
  int32 chunk_overlap = 3;
  bool clean_text = 4;
  bool preserve_urls = 5;
  // Semantic chunking configuration (only used when algorithm = SEMANTIC)
  optional SemanticChunkingConfig semantic_config = 6;
}
```

### 4. CentroidMetadata message

```protobuf
message CentroidMetadata {
  // Granularity level: "sentence", "paragraph_centroid", "section_centroid", "document_centroid"
  string granularity = 1;
  // How many source vectors were averaged to produce this centroid.
  int32 source_vector_count = 2;
  // Section title (for section_centroid granularity, from DocOutline).
  optional string section_title = 3;
  // Section heading depth (for section_centroid granularity).
  optional int32 section_depth = 4;
}
```

### 5. SemanticProcessingResult extension

Add optional field:

```protobuf
message SemanticProcessingResult {
  // ... existing fields ...
  // Centroid metadata (present when this result is a computed centroid, not a direct embedding).
  optional CentroidMetadata centroid_metadata = <next_field_number>;
}
```

## Module Changes

### module-semantic-manager

#### New: SemanticBoundaryDetector.java

Pure computation class, no CDI, no I/O.

```java
public class SemanticBoundaryDetector {
    /**
     * Find topic boundary indices using cosine similarity between consecutive sentence vectors.
     * Returns indices where breaks should occur (i.e., split BEFORE sentence at each index).
     */
    public static List<Integer> findBoundaries(
            List<float[]> sentenceVectors,
            float similarityThreshold,
            int percentileThreshold);

    /**
     * Group sentences by detected boundaries, enforcing min/max constraints.
     * Small groups are merged with their most-similar neighbor.
     * Large groups are split at the lowest-similarity internal transition.
     */
    public static List<List<StreamChunksResponse>> groupSentences(
            List<StreamChunksResponse> sentences,
            List<Integer> boundaries,
            int minChunkSentences,
            int maxChunkSentences);

    /** Cosine similarity between two vectors. */
    public static float cosineSimilarity(float[] a, float[] b);
}
```

**Boundary detection algorithm:**
1. Compute `sim[i] = cosine(vec[i], vec[i+1])` for all consecutive pairs
2. Find breakpoints where BOTH conditions met:
   - `sim[i] < similarityThreshold` (absolute cutoff)
   - `sim[i]` is in the bottom `percentileThreshold`% of all similarities
3. If only one condition configured (other is 0/default), use just that one
4. Group sentences between breakpoints

**Min/max enforcement:**
- Groups smaller than `minChunkSentences`: merge with the adjacent group that has highest boundary similarity
- Groups larger than `maxChunkSentences`: split at the lowest internal similarity point

#### New: CentroidComputer.java

Pure computation class.

```java
public class CentroidComputer {
    /**
     * Compute paragraph centroids by averaging sentence vectors within each paragraph.
     * Paragraph boundaries detected by double-newline in original text.
     */
    public static List<CentroidResult> computeParagraphCentroids(
            List<float[]> sentenceVectors,
            List<StreamChunksResponse> sentences,
            String originalText);

    /**
     * Compute section centroids by averaging paragraph centroids within each outline section.
     * Only when DocOutline is available.
     */
    public static List<CentroidResult> computeSectionCentroids(
            List<CentroidResult> paragraphCentroids,
            DocOutline outline);

    /**
     * Compute single document centroid by averaging all sentence vectors.
     */
    public static CentroidResult computeDocumentCentroid(
            List<float[]> sentenceVectors,
            String docId);

    /** Average vectors and L2 normalize. */
    static float[] averageAndNormalize(List<float[]> vectors);
}

record CentroidResult(
    float[] vector,
    String text,          // concatenated source text for this centroid
    int sourceVectorCount,
    String sectionTitle,  // null except for section centroids
    Integer sectionDepth  // null except for section centroids
) {}
```

#### Modified: SemanticIndexingOrchestrator.java

New method `processSemanticChunkingGroup()` that replaces the standard `chunkSourceText()` path when the directive uses semantic chunking.

**Detection:** When iterating `ChunkConfigWork` entries, check if `chunkConfigId` equals "semantic" or if the typed `ChunkerConfig` has `algorithm == SEMANTIC`.

**Flow:**
1. Build `StreamChunksRequest` with SENTENCE algorithm (reuse existing `chunkSourceText()` for this)
2. Collect all sentence responses
3. Build `StreamEmbeddingsRequest` for each sentence using the `sentence_embedding_model` from config
4. Collect sentence vectors into `List<float[]>`
5. Call `SemanticBoundaryDetector.findBoundaries()` + `groupSentences()`
6. Concatenate grouped sentence texts into chunk texts
7. Embed final chunks via existing embedder path (broadcast to all requested models)
8. If `store_sentence_vectors`: build sentence-level `SemanticProcessingResult`
9. If `compute_centroids`: call `CentroidComputer` methods, build centroid result sets
10. Return `List<AssemblyOutput>` with all result sets

### module-chunker

#### Modified: OverlapChunker.java

Add Caffeine cache for NLP analysis:

```java
private final Cache<String, NlpAnalysis> nlpCache = Caffeine.newBuilder()
    .maximumSize(100)
    .expireAfterWrite(Duration.ofMinutes(5))
    .build();

record NlpAnalysis(
    String[] sentences,
    Span[] sentenceSpans,
    String[] tokens,
    Span[] tokenSpans
) {}
```

- Cache key: SHA-256 of text content
- On sentence detection: check cache first, populate if miss
- On tokenization: check cache first, populate if miss
- The semantic manager sends SENTENCE split first, then potentially re-chunks oversized groups — cache ensures OpenNLP runs once per document text

#### build.gradle

Add Caffeine dependency (check if already present in BOM).

## Result Set Naming

For a semantic chunking directive on "body" with model "all-MiniLM-L6-v2":

| Result Set | chunk_config_id | Description |
|------------|----------------|-------------|
| `body-semantic-minilm` | `semantic` | Primary semantic chunks (re-embedded) |
| `body-sentence-minilm` | `sentence` | Sentence-level vectors (from detection pass) |
| `body-paragraph_centroid-minilm` | `paragraph_centroid` | Averaged sentence vectors per paragraph |
| `body-section_centroid-minilm` | `section_centroid` | Averaged paragraph centroids per section |
| `body-document_centroid-minilm` | `document_centroid` | Single document vector |

## Configuration Example

```json
{
  "directives": [{
    "source_label": "body",
    "cel_selector": "document.search_metadata.body",
    "chunker_configs": [{
      "config_id": "semantic",
      "config": {
        "algorithm": "SEMANTIC",
        "semantic_config": {
          "similarity_threshold": 0.5,
          "percentile_threshold": 20,
          "min_chunk_sentences": 2,
          "max_chunk_sentences": 30,
          "sentence_embedding_model": "all-MiniLM-L6-v2",
          "store_sentence_vectors": true,
          "compute_centroids": true
        }
      }
    }],
    "embedder_configs": [
      { "config_id": "all-MiniLM-L6-v2" }
    ]
  }]
}
```

## Edge Cases

- **Single sentence docs**: Return as one semantic chunk, sentence vector = document centroid
- **All similarities above threshold**: One big chunk (enforce max_chunk_sentences split)
- **All similarities below threshold**: Each sentence = chunk (enforce min_chunk_sentences merge)
- **Very long docs (10K+ sentences)**: Batch sentence embedding calls (existing embedder handles batching)
- **Empty/short text**: Fall through to field-level embedding (no chunking needed)
- **No DocOutline**: Skip section_centroid level, produce 4 result sets instead of 5

## Testing

### Unit Tests
- `SemanticBoundaryDetectorTest`: text with clear topic shifts, edge cases (all similar, all different, single sentence)
- `CentroidComputerTest`: averaging math, L2 normalization, empty inputs, paragraph boundary detection
- `NlpCacheTest`: cache hit/miss, expiration, key collision

### Integration Tests
- Full orchestrator flow with semantic config through MockServicesTestResource
- Verify 5 result sets produced with correct chunk_config_ids
- Verify centroid vectors are L2 normalized
- Verify sentence vectors match embedder output

## Files Affected

| Repo | File | Change |
|------|------|--------|
| pipestream-protos | `semantic_indexing.proto` | Add SemanticChunkingConfig, CentroidMetadata, ChunkerConfig.semantic_config |
| pipestream-protos | `search_metadata.proto` or `semantic_processing.proto` | Add centroid_metadata to SemanticProcessingResult |
| pipestream-platform | `gradle/libs.versions.toml` | BOM version bump after protos |
| module-semantic-manager | `SemanticBoundaryDetector.java` | NEW: boundary detection algorithm |
| module-semantic-manager | `CentroidComputer.java` | NEW: centroid computation |
| module-semantic-manager | `SemanticIndexingOrchestrator.java` | Add processSemanticChunkingGroup() |
| module-semantic-manager | `SemanticManagerOptions.java` | Recognize semantic config |
| module-chunker | `OverlapChunker.java` | Add NLP Caffeine cache |
| module-chunker | `build.gradle` | Add Caffeine dependency |
