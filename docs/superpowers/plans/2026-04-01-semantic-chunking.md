# Semantic Chunking with Hierarchical Centroid Vectors — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add semantic chunking (topic-boundary detection via sentence embedding similarity) and hierarchical centroid vector computation to the semantic manager, producing up to 5 vector sets from a single embedding pass.

**Architecture:** The semantic chunking algorithm lives in the SemanticIndexingOrchestrator (not the chunker). When a directive uses algorithm=SEMANTIC, the orchestrator: (1) calls the chunker with SENTENCE algorithm, (2) embeds sentences for boundary detection, (3) finds topic boundaries via cosine similarity, (4) re-embeds grouped chunks, (5) computes centroid vectors by averaging. The existing scatter-gather pattern is extended with a new code path. An NLP cache is added to the chunker module to avoid redundant OpenNLP processing.

**Tech Stack:** Java 21, Quarkus, Protobuf, gRPC (Mutiny), Caffeine cache, AssertJ

---

## Context

### Key Files

| File | Purpose |
|------|---------|
| `pipestream-protos/.../semantic_indexing.proto` | Chunker/embedder gRPC messages, ChunkerConfig, ChunkAlgorithm enum |
| `pipestream-protos/.../pipeline_core_types.proto` | SemanticProcessingResult, SemanticChunk, ChunkEmbedding, SearchMetadata |
| `module-semantic-manager/.../SemanticIndexingOrchestrator.java` | Main orchestrator — scatter-gather for chunk+embed |
| `module-semantic-manager/.../config/SemanticManagerOptions.java` | Module config with directives |
| `module-semantic-manager/.../config/DirectiveConfig.java` | Per-directive chunker+embedder config |
| `module-semantic-manager/.../service/ChunkerStreamClient.java` | gRPC client to chunker |
| `module-semantic-manager/.../service/EmbedderStreamClient.java` | gRPC client to embedder |
| `module-chunker/.../service/NlpPreprocessor.java` | OpenNLP sentence/token detection (NlpResult record) |
| `module-chunker/.../service/OverlapChunker.java` | Chunking algorithms |

### Repos & Branches

All work on `feat/semantic-chunking` branches:
1. `pipestream-protos` — proto changes (must be first, everything depends on it)
2. `pipestream-platform` — BOM bump after protos publish
3. `module-semantic-manager` — core implementation
4. `module-chunker` — NLP cache

---

## Task 1: Proto changes — SemanticChunkingConfig and CentroidMetadata

**Repo:** `pipestream-protos`
**Branch:** `feat/semantic-chunking`

**Files:**
- Modify: `semantic-indexing/proto/ai/pipestream/semantic/v1/semantic_indexing.proto`
- Modify: `common/proto/ai/pipestream/data/v1/pipeline_core_types.proto`

- [ ] **Step 1: Add SEMANTIC to ChunkAlgorithm enum**

In `semantic_indexing.proto`, add to the `ChunkAlgorithm` enum:

```protobuf
enum ChunkAlgorithm {
  CHUNK_ALGORITHM_UNSPECIFIED = 0;
  CHUNK_ALGORITHM_TOKEN = 1;
  CHUNK_ALGORITHM_SENTENCE = 2;
  CHUNK_ALGORITHM_CHARACTER = 3;
  CHUNK_ALGORITHM_SEMANTIC = 4;
}
```

- [ ] **Step 2: Add SemanticChunkingConfig message**

In `semantic_indexing.proto`, after the `ChunkAlgorithm` enum:

```protobuf
// Configuration for semantic (topic-boundary) chunking.
// Used when algorithm = CHUNK_ALGORITHM_SEMANTIC.
// The semantic manager uses sentence embeddings to detect topic shifts,
// then groups sentences into coherent chunks.
message SemanticChunkingConfig {
  // Absolute similarity cutoff (0.0-1.0). Break when consecutive sentence
  // similarity drops below this value. Default: 0.5.
  float similarity_threshold = 1;

  // Break at the bottom N% of similarity transitions (0-100).
  // Combined with similarity_threshold when both are set. Default: 20.
  int32 percentile_threshold = 2;

  // Minimum sentences per semantic chunk. Small chunks are merged with
  // their most-similar neighbor. Default: 2.
  int32 min_chunk_sentences = 3;

  // Maximum sentences per semantic chunk. Large chunks are split at
  // the lowest internal similarity point. Default: 30.
  int32 max_chunk_sentences = 4;

  // Embedding model for sentence boundary detection. Can differ from
  // the indexing model (use a fast small model here). Default: use
  // the first embedder_config from the directive.
  optional string sentence_embedding_model = 5;

  // Whether to store sentence-level vectors as a separate result set.
  // These are the raw sentence embeddings from the boundary detection pass.
  // Default: true.
  bool store_sentence_vectors = 6;

  // Whether to compute and store centroid vectors (paragraph, section, document).
  // Centroids are computed by averaging sentence vectors — zero extra embedding cost.
  // Default: true.
  bool compute_centroids = 7;
}
```

- [ ] **Step 3: Add semantic_config to ChunkerConfig**

In `semantic_indexing.proto`, modify `ChunkerConfig`:

```protobuf
message ChunkerConfig {
  ChunkAlgorithm algorithm = 1;
  int32 chunk_size = 2;
  int32 chunk_overlap = 3;
  bool clean_text = 4;
  bool preserve_urls = 5;
  // Semantic chunking configuration (only used when algorithm = CHUNK_ALGORITHM_SEMANTIC).
  optional SemanticChunkingConfig semantic_config = 6;
}
```

- [ ] **Step 4: Add CentroidMetadata message to pipeline_core_types.proto**

In `pipeline_core_types.proto`, before `SemanticProcessingResult`:

```protobuf
// Metadata for centroid (averaged) vector result sets.
// Present when a SemanticProcessingResult contains computed centroids
// rather than direct embeddings.
message CentroidMetadata {
  // Granularity level of this centroid.
  // Values: "sentence", "paragraph_centroid", "section_centroid", "document_centroid"
  string granularity = 1;

  // How many source vectors were averaged to produce this centroid.
  int32 source_vector_count = 2;

  // Section title (for section_centroid granularity, from DocOutline).
  optional string section_title = 3;

  // Section heading depth (for section_centroid granularity).
  optional int32 section_depth = 4;
}
```

- [ ] **Step 5: Add centroid_metadata to SemanticProcessingResult**

In `pipeline_core_types.proto`, add field 9 to `SemanticProcessingResult`:

```protobuf
message SemanticProcessingResult {
  string result_id = 1;
  string source_field_name = 2;
  string chunk_config_id = 3;
  string embedding_config_id = 4;
  optional string result_set_name = 5;
  repeated SemanticChunk chunks = 6;
  map<string, google.protobuf.Value> metadata = 7;
  optional NlpDocumentAnalysis nlp_analysis = 8;

  // Centroid metadata — present when this result set contains computed centroids
  // (averaged vectors) rather than direct embedding model output.
  optional CentroidMetadata centroid_metadata = 9;
}
```

- [ ] **Step 6: Build and verify protos compile**

```bash
cd /work/core-services/pipestream-protos && ./gradlew build
```

Expected: BUILD SUCCESSFUL

- [ ] **Step 7: Publish to local Maven and commit**

```bash
cd /work/core-services/pipestream-protos && ./gradlew publishToMavenLocal
git add -A && git commit -m "feat: add SemanticChunkingConfig, CentroidMetadata, SEMANTIC algorithm to protos"
```

---

## Task 2: BOM version bump

**Repo:** `pipestream-platform`
**Branch:** `feat/semantic-chunking`

- [ ] **Step 1: Bump protos version in BOM if needed**

Check if local snapshot resolves automatically. If pipestream-protos uses a snapshot version, no BOM change is needed — just rebuild platform:

```bash
cd /work/core-services/pipestream-platform && ./gradlew publishToMavenLocal
```

- [ ] **Step 2: Commit if changes were needed**

```bash
git add -A && git commit -m "build: bump protos for semantic chunking support"
```

---

## Task 3: SemanticBoundaryDetector — pure computation class

**Repo:** `module-semantic-manager`
**Branch:** `feat/semantic-chunking`

**Files:**
- Create: `src/main/java/ai/pipestream/module/semanticmanager/service/SemanticBoundaryDetector.java`
- Create: `src/test/java/ai/pipestream/module/semanticmanager/service/SemanticBoundaryDetectorTest.java`

- [ ] **Step 1: Write the test class**

```java
package ai.pipestream.module.semanticmanager.service;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.assertj.core.api.Assertions.assertThat;

class SemanticBoundaryDetectorTest {

    @Test
    void cosineSimilarity_identicalVectors_returnsOne() {
        float[] a = {1.0f, 0.0f, 0.0f};
        float[] b = {1.0f, 0.0f, 0.0f};
        assertThat(SemanticBoundaryDetector.cosineSimilarity(a, b))
                .as("Identical vectors should have similarity 1.0")
                .isEqualTo(1.0f, org.assertj.core.data.Offset.offset(0.001f));
    }

    @Test
    void cosineSimilarity_orthogonalVectors_returnsZero() {
        float[] a = {1.0f, 0.0f, 0.0f};
        float[] b = {0.0f, 1.0f, 0.0f};
        assertThat(SemanticBoundaryDetector.cosineSimilarity(a, b))
                .as("Orthogonal vectors should have similarity 0.0")
                .isEqualTo(0.0f, org.assertj.core.data.Offset.offset(0.001f));
    }

    @Test
    void findBoundaries_clearTopicShift_detectsBreak() {
        // 3 vectors about topic A (similar), then 2 about topic B (similar to each other, different from A)
        float[] topicA1 = {0.9f, 0.1f, 0.0f};
        float[] topicA2 = {0.85f, 0.15f, 0.0f};
        float[] topicA3 = {0.88f, 0.12f, 0.0f};
        float[] topicB1 = {0.1f, 0.9f, 0.0f};
        float[] topicB2 = {0.15f, 0.85f, 0.0f};

        List<float[]> vectors = List.of(topicA1, topicA2, topicA3, topicB1, topicB2);
        List<Integer> boundaries = SemanticBoundaryDetector.findBoundaries(vectors, 0.5f, 0);

        assertThat(boundaries)
                .as("Should detect boundary between sentence 2 and 3 (topic A→B)")
                .containsExactly(3);
    }

    @Test
    void findBoundaries_allSimilar_noBoundaries() {
        float[] v1 = {0.9f, 0.1f, 0.0f};
        float[] v2 = {0.88f, 0.12f, 0.0f};
        float[] v3 = {0.87f, 0.13f, 0.0f};

        List<Integer> boundaries = SemanticBoundaryDetector.findBoundaries(
                List.of(v1, v2, v3), 0.5f, 0);

        assertThat(boundaries)
                .as("All-similar vectors should produce no boundaries")
                .isEmpty();
    }

    @Test
    void findBoundaries_singleVector_noBoundaries() {
        float[] v1 = {1.0f, 0.0f, 0.0f};
        List<Integer> boundaries = SemanticBoundaryDetector.findBoundaries(
                List.of(v1), 0.5f, 0);

        assertThat(boundaries)
                .as("Single vector should have no boundaries")
                .isEmpty();
    }

    @Test
    void findBoundaries_percentileMode_detectsBottomPercentile() {
        // Similarities: 0.99, 0.98, 0.30, 0.97 — bottom 25% = index 2
        float[] v0 = {1.0f, 0.0f};
        float[] v1 = {0.99f, 0.01f};
        float[] v2 = {0.98f, 0.02f};
        float[] v3 = {0.3f, 0.7f};  // topic shift
        float[] v4 = {0.25f, 0.75f};

        List<Integer> boundaries = SemanticBoundaryDetector.findBoundaries(
                List.of(v0, v1, v2, v3, v4), 0.0f, 25);

        assertThat(boundaries)
                .as("Bottom 25% percentile should detect the topic shift at index 3")
                .containsExactly(3);
    }

    @Test
    void groupByBoundaries_basicGrouping() {
        List<String> items = List.of("s0", "s1", "s2", "s3", "s4");
        List<Integer> boundaries = List.of(3); // break before index 3

        List<List<String>> groups = SemanticBoundaryDetector.groupByBoundaries(items, boundaries);

        assertThat(groups).as("Should create 2 groups split at index 3")
                .hasSize(2);
        assertThat(groups.get(0)).as("First group is sentences 0-2")
                .containsExactly("s0", "s1", "s2");
        assertThat(groups.get(1)).as("Second group is sentences 3-4")
                .containsExactly("s3", "s4");
    }

    @Test
    void enforceMinSize_mergesSmallGroups() {
        // Groups: [s0], [s1,s2,s3], [s4] — min is 2, so [s0] and [s4] are too small
        List<List<String>> groups = List.of(
                List.of("s0"),
                List.of("s1", "s2", "s3"),
                List.of("s4"));

        // Similarity between consecutive groups: group0↔group1, group1↔group2
        float[] sims = {0.8f, 0.3f}; // s0 is closer to group1 than s4 is

        List<List<String>> enforced = SemanticBoundaryDetector.enforceMinChunkSize(groups, sims, 2);

        assertThat(enforced).as("Small groups should be merged; expect 2 groups")
                .hasSize(2);
        // s0 merges with its right neighbor (higher similarity)
        assertThat(enforced.get(0)).as("First group should include merged s0")
                .hasSize(4);
    }

    @Test
    void enforceMaxSize_splitsLargeGroups() {
        // Group of 6 sentences, max is 3
        float[] sims = {0.9f, 0.8f, 0.4f, 0.7f, 0.85f}; // lowest at index 2→3

        List<List<String>> groups = List.of(
                List.of("s0", "s1", "s2", "s3", "s4", "s5"));

        List<List<String>> enforced = SemanticBoundaryDetector.enforceMaxChunkSize(
                groups, List.of(sims), 3);

        assertThat(enforced).as("Group of 6 should be split into 2 groups of ≤3")
                .hasSizeGreaterThanOrEqualTo(2);
        for (List<String> group : enforced) {
            assertThat(group).as("Each group should have ≤3 sentences")
                    .hasSizeLessThanOrEqualTo(3);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /work/modules/module-semantic-manager && ./gradlew test --tests "*SemanticBoundaryDetectorTest*"
```

Expected: FAIL — class not found.

- [ ] **Step 3: Implement SemanticBoundaryDetector**

```java
package ai.pipestream.module.semanticmanager.service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Detects topic boundaries in a sequence of sentence embeddings using
 * cosine similarity between consecutive vectors.
 * <p>
 * Pure computation — no I/O, no CDI, no dependencies.
 */
public final class SemanticBoundaryDetector {

    private SemanticBoundaryDetector() {}

    /**
     * Cosine similarity between two vectors.
     */
    public static float cosineSimilarity(float[] a, float[] b) {
        float dot = 0f, normA = 0f, normB = 0f;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        float denom = (float) (Math.sqrt(normA) * Math.sqrt(normB));
        return denom == 0f ? 0f : dot / denom;
    }

    /**
     * Finds topic boundary indices using cosine similarity between
     * consecutive sentence vectors.
     *
     * @param sentenceVectors   embeddings for each sentence
     * @param similarityThreshold absolute cutoff — break when sim < threshold (0 to disable)
     * @param percentileThreshold bottom N% of transitions are breaks (0 to disable)
     * @return indices where breaks occur (split BEFORE the sentence at each index)
     */
    public static List<Integer> findBoundaries(
            List<float[]> sentenceVectors,
            float similarityThreshold,
            int percentileThreshold) {

        if (sentenceVectors.size() <= 1) {
            return List.of();
        }

        int n = sentenceVectors.size() - 1;
        float[] similarities = new float[n];
        for (int i = 0; i < n; i++) {
            similarities[i] = cosineSimilarity(sentenceVectors.get(i), sentenceVectors.get(i + 1));
        }

        // Compute percentile cutoff if requested
        float percentileCutoff = Float.MIN_VALUE;
        if (percentileThreshold > 0) {
            float[] sorted = similarities.clone();
            Arrays.sort(sorted);
            int cutoffIndex = Math.max(0, (int) Math.ceil(sorted.length * percentileThreshold / 100.0) - 1);
            percentileCutoff = sorted[Math.min(cutoffIndex, sorted.length - 1)];
        }

        List<Integer> boundaries = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            boolean belowThreshold = similarityThreshold > 0 && similarities[i] < similarityThreshold;
            boolean belowPercentile = percentileThreshold > 0 && similarities[i] <= percentileCutoff;

            // If both configured, require both. If only one configured, use that.
            boolean isBreak;
            if (similarityThreshold > 0 && percentileThreshold > 0) {
                isBreak = belowThreshold && belowPercentile;
            } else if (similarityThreshold > 0) {
                isBreak = belowThreshold;
            } else if (percentileThreshold > 0) {
                isBreak = belowPercentile;
            } else {
                isBreak = false;
            }

            if (isBreak) {
                boundaries.add(i + 1); // split BEFORE sentence i+1
            }
        }

        return boundaries;
    }

    /**
     * Groups items by boundary indices. Boundary at index N means split before item N.
     */
    public static <T> List<List<T>> groupByBoundaries(List<T> items, List<Integer> boundaries) {
        List<List<T>> groups = new ArrayList<>();
        int start = 0;
        for (int boundary : boundaries) {
            if (boundary > start && boundary <= items.size()) {
                groups.add(new ArrayList<>(items.subList(start, boundary)));
                start = boundary;
            }
        }
        if (start < items.size()) {
            groups.add(new ArrayList<>(items.subList(start, items.size())));
        }
        return groups;
    }

    /**
     * Merges groups smaller than minSize with their most-similar neighbor.
     *
     * @param groups the sentence groups
     * @param boundarySimilarities similarity at each boundary (length = groups.size()-1)
     * @param minSize minimum sentences per group
     */
    public static <T> List<List<T>> enforceMinChunkSize(
            List<List<T>> groups, float[] boundarySimilarities, int minSize) {

        if (minSize <= 0 || groups.size() <= 1) {
            return new ArrayList<>(groups);
        }

        List<List<T>> result = new ArrayList<>();
        for (List<T> g : groups) {
            result.add(new ArrayList<>(g));
        }

        boolean merged = true;
        while (merged) {
            merged = false;
            for (int i = 0; i < result.size(); i++) {
                if (result.get(i).size() < minSize && result.size() > 1) {
                    // Decide merge direction: merge with neighbor that has higher similarity
                    int mergeTarget;
                    if (i == 0) {
                        mergeTarget = 1;
                    } else if (i == result.size() - 1) {
                        mergeTarget = i - 1;
                    } else {
                        // Compare boundary similarities (approximation: use index-based)
                        int leftIdx = Math.min(i - 1, boundarySimilarities.length - 1);
                        int rightIdx = Math.min(i, boundarySimilarities.length - 1);
                        mergeTarget = (leftIdx >= 0 && boundarySimilarities[leftIdx] >= boundarySimilarities[rightIdx])
                                ? i - 1 : i + 1;
                        mergeTarget = Math.max(0, Math.min(mergeTarget, result.size() - 1));
                    }

                    if (mergeTarget < i) {
                        result.get(mergeTarget).addAll(result.get(i));
                        result.remove(i);
                    } else {
                        result.get(i).addAll(result.get(mergeTarget));
                        result.remove(mergeTarget);
                    }
                    merged = true;
                    break; // restart scan
                }
            }
        }
        return result;
    }

    /**
     * Splits groups larger than maxSize at their lowest internal similarity point.
     *
     * @param groups the sentence groups
     * @param groupInternalSimilarities per-group array of internal consecutive similarities
     * @param maxSize maximum sentences per group
     */
    public static <T> List<List<T>> enforceMaxChunkSize(
            List<List<T>> groups,
            List<float[]> groupInternalSimilarities,
            int maxSize) {

        if (maxSize <= 0) {
            return new ArrayList<>(groups);
        }

        List<List<T>> result = new ArrayList<>();
        for (int g = 0; g < groups.size(); g++) {
            List<T> group = groups.get(g);
            if (group.size() <= maxSize) {
                result.add(new ArrayList<>(group));
                continue;
            }

            // Split at lowest internal similarity
            float[] sims = g < groupInternalSimilarities.size()
                    ? groupInternalSimilarities.get(g) : new float[0];

            splitRecursive(group, sims, maxSize, result);
        }
        return result;
    }

    private static <T> void splitRecursive(List<T> group, float[] sims, int maxSize, List<List<T>> out) {
        if (group.size() <= maxSize) {
            out.add(new ArrayList<>(group));
            return;
        }

        // Find lowest similarity point
        int splitAt = group.size() / 2; // fallback: midpoint
        if (sims.length > 0) {
            float minSim = Float.MAX_VALUE;
            for (int i = 0; i < Math.min(sims.length, group.size() - 1); i++) {
                if (sims[i] < minSim) {
                    minSim = sims[i];
                    splitAt = i + 1;
                }
            }
        }

        splitAt = Math.max(1, Math.min(splitAt, group.size() - 1));
        List<T> left = group.subList(0, splitAt);
        List<T> right = group.subList(splitAt, group.size());
        float[] leftSims = splitAt - 1 <= sims.length ? Arrays.copyOfRange(sims, 0, Math.max(0, splitAt - 1)) : new float[0];
        float[] rightSims = splitAt < sims.length ? Arrays.copyOfRange(sims, splitAt, sims.length) : new float[0];

        splitRecursive(new ArrayList<>(left), leftSims, maxSize, out);
        splitRecursive(new ArrayList<>(right), rightSims, maxSize, out);
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /work/modules/module-semantic-manager && ./gradlew test --tests "*SemanticBoundaryDetectorTest*"
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add SemanticBoundaryDetector with cosine similarity boundary detection"
```

---

## Task 4: CentroidComputer — vector averaging and normalization

**Repo:** `module-semantic-manager`

**Files:**
- Create: `src/main/java/ai/pipestream/module/semanticmanager/service/CentroidComputer.java`
- Create: `src/test/java/ai/pipestream/module/semanticmanager/service/CentroidComputerTest.java`

- [ ] **Step 1: Write the test class**

```java
package ai.pipestream.module.semanticmanager.service;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.assertj.core.api.Assertions.assertThat;

class CentroidComputerTest {

    @Test
    void averageAndNormalize_twoVectors_returnsNormalizedMean() {
        float[] v1 = {1.0f, 0.0f};
        float[] v2 = {0.0f, 1.0f};

        float[] result = CentroidComputer.averageAndNormalize(List.of(v1, v2));

        // Mean is [0.5, 0.5], L2 norm = sqrt(0.25+0.25) = ~0.707
        // Normalized: [0.707, 0.707]
        float expected = (float) (0.5 / Math.sqrt(0.5));
        assertThat(result[0]).as("X component of normalized mean")
                .isCloseTo(expected, org.assertj.core.data.Offset.offset(0.01f));
        assertThat(result[1]).as("Y component of normalized mean")
                .isCloseTo(expected, org.assertj.core.data.Offset.offset(0.01f));

        // Verify L2 norm is 1.0
        float norm = 0f;
        for (float f : result) norm += f * f;
        assertThat((float) Math.sqrt(norm)).as("L2 norm should be 1.0")
                .isCloseTo(1.0f, org.assertj.core.data.Offset.offset(0.001f));
    }

    @Test
    void averageAndNormalize_singleVector_returnsNormalized() {
        float[] v = {3.0f, 4.0f};
        float[] result = CentroidComputer.averageAndNormalize(List.of(v));

        // L2 norm of [3,4] = 5, so normalized = [0.6, 0.8]
        assertThat(result[0]).as("X component").isCloseTo(0.6f, org.assertj.core.data.Offset.offset(0.01f));
        assertThat(result[1]).as("Y component").isCloseTo(0.8f, org.assertj.core.data.Offset.offset(0.01f));
    }

    @Test
    void averageAndNormalize_emptyList_returnsEmptyArray() {
        float[] result = CentroidComputer.averageAndNormalize(List.of());
        assertThat(result).as("Empty input should return empty array").isEmpty();
    }

    @Test
    void detectParagraphBoundaries_doubleNewline_splitsParagraphs() {
        String text = "Sentence one. Sentence two.\n\nSentence three. Sentence four.";
        // Sentence offsets: s0=[0,14], s1=[15,28], s2=[30,46], s3=[47,62]
        int[][] sentenceOffsets = {{0, 14}, {15, 28}, {30, 46}, {47, 62}};

        List<List<Integer>> paragraphs = CentroidComputer.detectParagraphBoundaries(text, sentenceOffsets);

        assertThat(paragraphs).as("Double newline should split into 2 paragraphs")
                .hasSize(2);
        assertThat(paragraphs.get(0)).as("First paragraph has sentences 0,1")
                .containsExactly(0, 1);
        assertThat(paragraphs.get(1)).as("Second paragraph has sentences 2,3")
                .containsExactly(2, 3);
    }
}
```

- [ ] **Step 2: Run tests — should fail**

```bash
cd /work/modules/module-semantic-manager && ./gradlew test --tests "*CentroidComputerTest*"
```

- [ ] **Step 3: Implement CentroidComputer**

```java
package ai.pipestream.module.semanticmanager.service;

import java.util.ArrayList;
import java.util.List;

/**
 * Computes centroid (averaged) vectors at various granularities:
 * paragraph, section, and document level.
 * <p>
 * Pure computation — no I/O, no CDI dependencies.
 * All centroids are L2 normalized after averaging.
 */
public final class CentroidComputer {

    private CentroidComputer() {}

    /**
     * Result of a centroid computation.
     */
    public record CentroidResult(
            float[] vector,
            String text,
            int sourceVectorCount,
            String sectionTitle,
            Integer sectionDepth
    ) {}

    /**
     * Averages vectors and L2-normalizes the result.
     * Returns empty array if input is empty.
     */
    public static float[] averageAndNormalize(List<float[]> vectors) {
        if (vectors.isEmpty()) {
            return new float[0];
        }

        int dim = vectors.get(0).length;
        float[] sum = new float[dim];
        for (float[] v : vectors) {
            for (int i = 0; i < dim; i++) {
                sum[i] += v[i];
            }
        }

        float count = vectors.size();
        for (int i = 0; i < dim; i++) {
            sum[i] /= count;
        }

        return l2Normalize(sum);
    }

    /**
     * L2-normalizes a vector in place and returns it.
     */
    public static float[] l2Normalize(float[] v) {
        float norm = 0f;
        for (float f : v) {
            norm += f * f;
        }
        norm = (float) Math.sqrt(norm);
        if (norm > 0f) {
            for (int i = 0; i < v.length; i++) {
                v[i] /= norm;
            }
        }
        return v;
    }

    /**
     * Detects paragraph boundaries in the original text by finding double-newline gaps.
     * Returns groups of sentence indices, where each group is one paragraph.
     *
     * @param originalText the source text
     * @param sentenceOffsets [i] = {startOffset, endOffset} for sentence i
     * @return list of paragraphs, each being a list of sentence indices
     */
    public static List<List<Integer>> detectParagraphBoundaries(String originalText, int[][] sentenceOffsets) {
        List<List<Integer>> paragraphs = new ArrayList<>();
        List<Integer> currentParagraph = new ArrayList<>();

        for (int i = 0; i < sentenceOffsets.length; i++) {
            currentParagraph.add(i);

            if (i < sentenceOffsets.length - 1) {
                int gapStart = sentenceOffsets[i][1];
                int gapEnd = sentenceOffsets[i + 1][0];
                String gap = originalText.substring(
                        Math.min(gapStart, originalText.length()),
                        Math.min(gapEnd, originalText.length()));

                // Double newline or blank line indicates paragraph boundary
                if (gap.contains("\n\n") || gap.contains("\r\n\r\n")) {
                    paragraphs.add(currentParagraph);
                    currentParagraph = new ArrayList<>();
                }
            }
        }

        if (!currentParagraph.isEmpty()) {
            paragraphs.add(currentParagraph);
        }

        return paragraphs;
    }

    /**
     * Computes paragraph centroids by averaging sentence vectors within each paragraph.
     */
    public static List<CentroidResult> computeParagraphCentroids(
            List<float[]> sentenceVectors,
            List<String> sentenceTexts,
            String originalText,
            int[][] sentenceOffsets) {

        List<List<Integer>> paragraphs = detectParagraphBoundaries(originalText, sentenceOffsets);
        List<CentroidResult> centroids = new ArrayList<>();

        for (List<Integer> paragraphIndices : paragraphs) {
            List<float[]> vecs = new ArrayList<>();
            StringBuilder text = new StringBuilder();
            for (int idx : paragraphIndices) {
                if (idx < sentenceVectors.size()) {
                    vecs.add(sentenceVectors.get(idx));
                }
                if (idx < sentenceTexts.size()) {
                    if (!text.isEmpty()) text.append(" ");
                    text.append(sentenceTexts.get(idx));
                }
            }
            if (!vecs.isEmpty()) {
                centroids.add(new CentroidResult(
                        averageAndNormalize(vecs),
                        text.toString(),
                        vecs.size(),
                        null, null));
            }
        }
        return centroids;
    }

    /**
     * Computes a single document centroid by averaging all sentence vectors.
     */
    public static CentroidResult computeDocumentCentroid(
            List<float[]> sentenceVectors,
            String fullText) {

        return new CentroidResult(
                averageAndNormalize(sentenceVectors),
                fullText,
                sentenceVectors.size(),
                null, null);
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cd /work/modules/module-semantic-manager && ./gradlew test --tests "*CentroidComputerTest*"
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add CentroidComputer for hierarchical vector averaging"
```

---

## Task 5: Orchestrator integration — processSemanticChunkingGroup

**Repo:** `module-semantic-manager`

**Files:**
- Modify: `src/main/java/ai/pipestream/module/semanticmanager/service/SemanticIndexingOrchestrator.java`
- Modify: `src/main/java/ai/pipestream/module/semanticmanager/config/SemanticManagerOptions.java`

- [ ] **Step 1: Add semantic config detection to SemanticManagerOptions**

In `SemanticManagerOptions.java`, add to `convertToTypedChunkerConfig()` or add a helper in the orchestrator that detects `algorithm: "SEMANTIC"` in the directive config and extracts `SemanticChunkingConfig` fields.

Add this constant to `SemanticIndexingOrchestrator`:
```java
private static final String SEMANTIC_CHUNK_CONFIG_ID = "semantic";
```

- [ ] **Step 2: Add isSemanticChunking detection method**

In `SemanticIndexingOrchestrator`, add:

```java
private boolean isSemanticChunking(String chunkConfigId, ChunkConfigWork work) {
    if (SEMANTIC_CHUNK_CONFIG_ID.equals(chunkConfigId)) return true;
    // Check if typed config has SEMANTIC algorithm
    if (work != null && work.chunkerConfig() != null) {
        Struct cfg = work.chunkerConfig();
        Value algoVal = cfg.getFieldsOrDefault("algorithm", null);
        if (algoVal != null && "SEMANTIC".equalsIgnoreCase(algoVal.getStringValue())) {
            return true;
        }
    }
    return false;
}
```

- [ ] **Step 3: Implement processSemanticChunkingGroup**

Add the main method that implements the 5-phase semantic chunking flow:

```java
/**
 * Processes a semantic chunking directive: sentence split → sentence embed →
 * boundary detection → chunk embed → centroid computation.
 * Returns up to 5 SemanticProcessingResults (sentence, semantic, paragraph, section, document).
 */
private Uni<List<AssemblyOutput>> processSemanticChunkingGroup(
        PipeDoc inputDoc,
        SourceTextWork work,
        String chunkConfigId,
        ChunkConfigWork chunkWork,
        String nodeId) {

    String docId = inputDoc.getDocId();
    String sourceText = work.sourceText();
    String sourceLabel = work.sourceLabel();

    // Parse semantic config from the Struct
    SemanticChunkingParams params = parseSemanticConfig(chunkWork.chunkerConfig());

    // Phase 1a: Sentence split — call chunker with SENTENCE algorithm
    StreamChunksRequest sentenceReq = StreamChunksRequest.newBuilder()
            .setRequestId(UUID.randomUUID().toString())
            .setDocId(docId)
            .setSourceFieldName(sourceLabel)
            .setTextContent(sourceText)
            .addChunkConfigs(ChunkConfigEntry.newBuilder()
                    .setChunkConfigId("__sentence_split__")
                    .setConfig(ChunkerConfig.newBuilder()
                            .setAlgorithm(ChunkAlgorithm.CHUNK_ALGORITHM_SENTENCE)
                            .setChunkSize(1) // one sentence per chunk
                            .setChunkOverlap(0)
                            .build())
                    .build())
            .build();

    return chunkerStreamClient.streamChunks(sentenceReq)
            .collect().asList()
            .chain(sentenceResponses -> {
                // Filter to our config
                List<StreamChunksResponse> sentences = sentenceResponses.stream()
                        .filter(r -> "__sentence_split__".equals(r.getChunkConfigId()))
                        .toList();

                if (sentences.isEmpty()) {
                    log.warn("Semantic chunking: no sentences from chunker for doc {}", docId);
                    return Uni.createFrom().item(List.<AssemblyOutput>of());
                }

                // Capture NLP analysis from last chunk
                NlpDocumentAnalysis nlpAnalysis = sentences.stream()
                        .filter(StreamChunksResponse::getIsLast)
                        .filter(StreamChunksResponse::hasNlpAnalysis)
                        .map(StreamChunksResponse::getNlpAnalysis)
                        .findFirst().orElse(null);

                // Phase 1b: Embed sentences for boundary detection
                String sentenceModel = params.sentenceEmbeddingModel();
                List<StreamEmbeddingsRequest> embedReqs = new ArrayList<>();
                for (StreamChunksResponse sent : sentences) {
                    embedReqs.add(StreamEmbeddingsRequest.newBuilder()
                            .setRequestId(UUID.randomUUID().toString())
                            .setDocId(docId)
                            .setChunkId(sent.getChunkId())
                            .setTextContent(sent.getTextContent())
                            .setChunkConfigId("__sentence_boundary__")
                            .setEmbeddingModelId(sentenceModel)
                            .build());
                }

                return embedderStreamClient.streamEmbeddings(Multi.createFrom().iterable(embedReqs))
                        .collect().asList()
                        .chain(embeddingResponses -> {
                            // Build sentence vector map
                            Map<String, float[]> sentenceVectorMap = new LinkedHashMap<>();
                            for (StreamEmbeddingsResponse resp : embeddingResponses) {
                                if (resp.getSuccess()) {
                                    float[] vec = new float[resp.getVectorCount()];
                                    for (int i = 0; i < vec.length; i++) {
                                        vec[i] = resp.getVector(i);
                                    }
                                    sentenceVectorMap.put(resp.getChunkId(), vec);
                                }
                            }

                            // Build ordered lists matching sentence order
                            List<float[]> sentenceVectors = new ArrayList<>();
                            List<String> sentenceTexts = new ArrayList<>();
                            List<int[]> sentenceOffsets = new ArrayList<>();
                            for (StreamChunksResponse sent : sentences) {
                                float[] vec = sentenceVectorMap.get(sent.getChunkId());
                                if (vec != null) {
                                    sentenceVectors.add(vec);
                                    sentenceTexts.add(sent.getTextContent());
                                    sentenceOffsets.add(new int[]{sent.getStartOffset(), sent.getEndOffset()});
                                }
                            }

                            // Phase 1c: Boundary detection
                            List<Integer> boundaries = SemanticBoundaryDetector.findBoundaries(
                                    sentenceVectors,
                                    params.similarityThreshold(),
                                    params.percentileThreshold());

                            List<List<StreamChunksResponse>> groups =
                                    SemanticBoundaryDetector.groupByBoundaries(sentences, boundaries);

                            // TODO: enforce min/max once we have internal similarity arrays

                            // Build semantic chunk texts by concatenating grouped sentences
                            List<String> chunkTexts = groups.stream()
                                    .map(g -> g.stream()
                                            .map(StreamChunksResponse::getTextContent)
                                            .collect(java.util.stream.Collectors.joining(" ")))
                                    .toList();

                            // Phase 2a: Embed final semantic chunks with each requested embedder
                            List<Uni<AssemblyOutput>> embedUnis = new ArrayList<>();
                            for (EmbedderTarget embedTarget : chunkWork.embedderTargets()) {
                                // Build synthetic StreamChunksResponse for each semantic chunk
                                List<StreamChunksResponse> syntheticChunks = new ArrayList<>();
                                for (int i = 0; i < chunkTexts.size(); i++) {
                                    syntheticChunks.add(StreamChunksResponse.newBuilder()
                                            .setRequestId(docId)
                                            .setDocId(docId)
                                            .setChunkId(docId + "_semantic_" + i)
                                            .setChunkNumber(i)
                                            .setTextContent(chunkTexts.get(i))
                                            .setChunkConfigId("semantic")
                                            .setSourceFieldName(sourceLabel)
                                            .setIsLast(i == chunkTexts.size() - 1)
                                            .build());
                                }

                                String resultSetName = sourceLabel + "-semantic-" + embedTarget.embedderConfigId();
                                embedUnis.add(embedChunks(
                                        docId, syntheticChunks, "semantic",
                                        embedTarget.embedderConfigId(),
                                        embedTarget.embedderConfig(),
                                        sourceLabel, resultSetName, nodeId, nlpAnalysis));
                            }

                            // Phase 2b: Build centroid result sets
                            List<AssemblyOutput> centroidOutputs = new ArrayList<>();

                            if (params.storeSentenceVectors()) {
                                // Sentence-level result: reuse vectors from boundary detection
                                for (EmbedderTarget embedTarget : chunkWork.embedderTargets()) {
                                    centroidOutputs.add(buildSentenceResult(
                                            sentences, embeddingResponses, sourceLabel,
                                            embedTarget.embedderConfigId(), nodeId, nlpAnalysis));
                                }
                            }

                            if (params.computeCentroids()) {
                                for (EmbedderTarget embedTarget : chunkWork.embedderTargets()) {
                                    centroidOutputs.addAll(buildCentroidResults(
                                            sentenceVectors, sentenceTexts,
                                            sentenceOffsets.toArray(new int[0][]),
                                            sourceText, inputDoc, sourceLabel,
                                            embedTarget.embedderConfigId(), nodeId));
                                }
                            }

                            // Combine semantic chunk embeddings with centroid results
                            return Uni.combine().all().unis(embedUnis)
                                    .with(results -> {
                                        List<AssemblyOutput> all = new ArrayList<>();
                                        for (Object r : results) {
                                            all.add((AssemblyOutput) r);
                                        }
                                        all.addAll(centroidOutputs);
                                        return all;
                                    });
                        });
            });
}
```

- [ ] **Step 4: Add helper methods**

```java
record SemanticChunkingParams(
        float similarityThreshold,
        int percentileThreshold,
        int minChunkSentences,
        int maxChunkSentences,
        String sentenceEmbeddingModel,
        boolean storeSentenceVectors,
        boolean computeCentroids
) {
    static SemanticChunkingParams defaults(String defaultModel) {
        return new SemanticChunkingParams(0.5f, 20, 2, 30, defaultModel, true, true);
    }
}

private SemanticChunkingParams parseSemanticConfig(Struct config) {
    String defaultModel = "all-MiniLM-L6-v2";
    if (config == null) return SemanticChunkingParams.defaults(defaultModel);

    Value scVal = config.getFieldsOrDefault("semantic_config", null);
    if (scVal == null || !scVal.hasStructValue()) {
        return SemanticChunkingParams.defaults(defaultModel);
    }
    Struct sc = scVal.getStructValue();

    return new SemanticChunkingParams(
            getFloat(sc, "similarity_threshold", 0.5f),
            getInt(sc, "percentile_threshold", 20),
            getInt(sc, "min_chunk_sentences", 2),
            getInt(sc, "max_chunk_sentences", 30),
            getString(sc, "sentence_embedding_model", defaultModel),
            getBool(sc, "store_sentence_vectors", true),
            getBool(sc, "compute_centroids", true)
    );
}

private float getFloat(Struct s, String key, float def) {
    Value v = s.getFieldsOrDefault(key, null);
    return v != null && v.hasNumberValue() ? (float) v.getNumberValue() : def;
}
private int getInt(Struct s, String key, int def) {
    Value v = s.getFieldsOrDefault(key, null);
    return v != null && v.hasNumberValue() ? (int) v.getNumberValue() : def;
}
private String getString(Struct s, String key, String def) {
    Value v = s.getFieldsOrDefault(key, null);
    return v != null && v.hasStringValue() ? v.getStringValue() : def;
}
private boolean getBool(Struct s, String key, boolean def) {
    Value v = s.getFieldsOrDefault(key, null);
    return v != null && v.hasBoolValue() ? v.getBoolValue() : def;
}
```

```java
private AssemblyOutput buildSentenceResult(
        List<StreamChunksResponse> sentences,
        List<StreamEmbeddingsResponse> embeddings,
        String sourceLabel, String embeddingConfigId,
        String nodeId, NlpDocumentAnalysis nlpAnalysis) {

    // Reuse the standard assembleResult by pairing sentences with their embeddings
    return assembleResult(sentences, embeddings, sourceLabel,
            "sentence", embeddingConfigId,
            sourceLabel + "-sentence-" + embeddingConfigId,
            nodeId, nlpAnalysis);
}

private List<AssemblyOutput> buildCentroidResults(
        List<float[]> sentenceVectors,
        List<String> sentenceTexts,
        int[][] sentenceOffsets,
        String originalText,
        PipeDoc inputDoc,
        String sourceLabel,
        String embeddingConfigId,
        String nodeId) {

    List<AssemblyOutput> results = new ArrayList<>();

    // Paragraph centroids
    List<CentroidComputer.CentroidResult> paragraphCentroids =
            CentroidComputer.computeParagraphCentroids(
                    sentenceVectors, sentenceTexts, originalText, sentenceOffsets);

    if (!paragraphCentroids.isEmpty()) {
        results.add(buildCentroidAssemblyOutput(
                paragraphCentroids, "paragraph_centroid",
                sourceLabel, embeddingConfigId, nodeId));
    }

    // Document centroid
    CentroidComputer.CentroidResult docCentroid =
            CentroidComputer.computeDocumentCentroid(sentenceVectors, originalText);
    results.add(buildCentroidAssemblyOutput(
            List.of(docCentroid), "document_centroid",
            sourceLabel, embeddingConfigId, nodeId));

    // Section centroids (only if DocOutline available)
    if (inputDoc.hasSearchMetadata() && inputDoc.getSearchMetadata().hasDocOutline()) {
        // Future: implement section centroid computation using DocOutline
        // For now, paragraph and document centroids cover the key use cases
    }

    return results;
}

private AssemblyOutput buildCentroidAssemblyOutput(
        List<CentroidComputer.CentroidResult> centroids,
        String granularity,
        String sourceLabel,
        String embeddingConfigId,
        String nodeId) {

    SemanticProcessingResult.Builder resultBuilder = SemanticProcessingResult.newBuilder()
            .setResultId(UUID.randomUUID().toString())
            .setSourceFieldName(sourceLabel)
            .setChunkConfigId(granularity)
            .setEmbeddingConfigId(embeddingConfigId)
            .setResultSetName(sourceLabel + "-" + granularity + "-" + embeddingConfigId)
            .setCentroidMetadata(CentroidMetadata.newBuilder()
                    .setGranularity(granularity)
                    .setSourceVectorCount(centroids.stream()
                            .mapToInt(CentroidComputer.CentroidResult::sourceVectorCount).sum())
                    .build());

    if (nodeId != null) {
        resultBuilder.putMetadata("coordinator_node_id", protoValue(nodeId));
    }

    for (int i = 0; i < centroids.size(); i++) {
        CentroidComputer.CentroidResult c = centroids.get(i);
        ChunkEmbedding.Builder emb = ChunkEmbedding.newBuilder()
                .setTextContent(c.text())
                .setChunkId(sourceLabel + "_" + granularity + "_" + i)
                .setChunkConfigId(granularity);
        for (float f : c.vector()) {
            emb.addVector(f);
        }

        SemanticChunk.Builder chunk = SemanticChunk.newBuilder()
                .setChunkId(sourceLabel + "_" + granularity + "_" + i)
                .setChunkNumber(i)
                .setEmbeddingInfo(emb.build());

        if (c.sectionTitle() != null) {
            chunk.putMetadata("section_title", protoValue(c.sectionTitle()));
        }

        resultBuilder.addChunks(chunk.build());
    }

    return new AssemblyOutput(resultBuilder.build(), null, centroids.size());
}
```

- [ ] **Step 5: Wire into the main orchestration flow**

In the existing `orchestrateFromDirectives()` method, after building `sourceTextWorkMap`, modify the Phase 1+2 loop to detect semantic chunking. Find the section where `chunkSourceText()` is called for each `SourceTextWork` and add a branch:

```java
// In the existing processing loop, before calling chunkSourceText:
for (Map.Entry<String, ChunkConfigWork> entry : work.chunkConfigs().entrySet()) {
    String cfgId = entry.getKey();
    ChunkConfigWork cfgWork = entry.getValue();

    if (isSemanticChunking(cfgId, cfgWork)) {
        // Semantic chunking: separate path with sentence embedding + boundary detection
        semanticUnis.add(processSemanticChunkingGroup(
                inputDoc, work, cfgId, cfgWork, nodeId));
    } else {
        // Standard chunking: existing path
        standardChunkConfigs.put(cfgId, cfgWork);
    }
}
```

The exact integration point depends on how the existing loop is structured. The key pattern is: check each chunkConfig, route to semantic path if applicable, standard path otherwise.

- [ ] **Step 6: Build and verify compilation**

```bash
cd /work/modules/module-semantic-manager && ./gradlew compileJava
```

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: integrate semantic chunking into SemanticIndexingOrchestrator"
```

---

## Task 6: NLP Cache in chunker module

**Repo:** `module-chunker`
**Branch:** `feat/semantic-chunking`

**Files:**
- Modify: `src/main/java/ai/pipestream/module/chunker/service/NlpPreprocessor.java`
- Modify: `build.gradle` (add Caffeine dependency if needed)

- [ ] **Step 1: Add Caffeine dependency to build.gradle**

Check if `caffeine` is already available via the BOM. If not, add:

```groovy
implementation libs.caffeine
```

Or if using direct coordinates:
```groovy
implementation 'com.github.ben-manes.caffeine:caffeine'
```

- [ ] **Step 2: Add cache to NlpPreprocessor**

At the top of `NlpPreprocessor`:

```java
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.HexFormat;
```

Add field:
```java
private final Cache<String, NlpResult> nlpCache = Caffeine.newBuilder()
        .maximumSize(100)
        .expireAfterWrite(Duration.ofMinutes(5))
        .recordStats()
        .build();
```

- [ ] **Step 3: Wrap preprocess() with cache lookup**

Modify the `preprocess(String text)` method:

```java
public NlpResult preprocess(String text) {
    if (text == null || text.isBlank()) {
        return NlpResult.empty();
    }

    String cacheKey = hashText(text);
    NlpResult cached = nlpCache.getIfPresent(cacheKey);
    if (cached != null) {
        LOG.debugf("NLP cache hit for text hash %s (length %d)", cacheKey.substring(0, 8), text.length());
        return cached;
    }

    NlpResult result = preprocessUncached(text);
    nlpCache.put(cacheKey, result);
    LOG.debugf("NLP cache miss for text hash %s (length %d), cached result with %d sentences, %d tokens",
            cacheKey.substring(0, 8), text.length(),
            result.sentences().length, result.tokens().length);
    return result;
}

private NlpResult preprocessUncached(String text) {
    // ... existing preprocess logic moved here ...
}
```

- [ ] **Step 4: Add hashText helper**

```java
private static String hashText(String text) {
    try {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] hash = md.digest(text.getBytes(StandardCharsets.UTF_8));
        return HexFormat.of().formatHex(hash);
    } catch (NoSuchAlgorithmException e) {
        // SHA-256 is guaranteed to be available
        throw new RuntimeException(e);
    }
}
```

- [ ] **Step 5: Build and test**

```bash
cd /work/modules/module-chunker && ./gradlew test
```

Expected: All existing tests pass (cache is transparent).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: add NLP analysis Caffeine cache to NlpPreprocessor"
```

---

## Task 7: Unit tests for orchestrator semantic path

**Repo:** `module-semantic-manager`

**Files:**
- Modify: `src/test/java/ai/pipestream/module/semanticmanager/SemanticIndexingOrchestratorTest.java`

- [ ] **Step 1: Add test for semantic chunking config detection**

```java
@Test
void testSemanticChunkingDetection() {
    // Build a directive with algorithm=SEMANTIC
    // Verify the orchestrator routes to the semantic path
    // This is a focused unit test — mock chunker returns sentences,
    // mock embedder returns known vectors, verify boundary detection + centroid output
}
```

The exact test depends on how the orchestrator's test harness is set up. Use the existing `MockChunkerService` and `MockEmbedderService` patterns from the test base.

- [ ] **Step 2: Add test verifying 5 result sets produced**

Verify that when semantic chunking is configured with `store_sentence_vectors=true` and `compute_centroids=true`, the output PipeDoc has result sets for: semantic, sentence, paragraph_centroid, document_centroid (4 minimum, 5 if DocOutline present).

- [ ] **Step 3: Run all tests**

```bash
cd /work/modules/module-semantic-manager && ./gradlew test
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "test: add semantic chunking orchestration tests"
```

---

## Verification

1. **Proto compilation**: `cd pipestream-protos && ./gradlew build` — SEMANTIC enum, SemanticChunkingConfig, CentroidMetadata all compile
2. **Boundary detection**: `./gradlew test --tests "*SemanticBoundaryDetectorTest*"` — all boundary/grouping tests pass
3. **Centroid computation**: `./gradlew test --tests "*CentroidComputerTest*"` — averaging + normalization correct
4. **Orchestrator integration**: `./gradlew test --tests "*SemanticIndexingOrchestratorTest*"` — semantic path produces expected result sets
5. **NLP cache**: `cd module-chunker && ./gradlew test` — all existing tests still pass with cache enabled
6. **Full build**: `./gradlew build` in each repo — all compile, all tests pass

## Future Work (separate tasks)

- Section centroid computation using DocOutline (when available from parser)
- NLP document caching in the chunker's gRPC service layer (StreamChunks level)
- Integration test with real chunker + embedder services via chain test sidecar
- Quality comparison: semantic vs token chunking on the same corpus
- Configuration UI in frontend for semantic chunking options
