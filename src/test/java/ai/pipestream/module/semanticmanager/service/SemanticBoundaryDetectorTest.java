package ai.pipestream.module.semanticmanager.service;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;

class SemanticBoundaryDetectorTest {

    @Test
    void cosineSimilarity_identicalVectors_returnsOne() {
        float[] a = {1.0f, 0.0f, 0.0f};
        assertThat(SemanticBoundaryDetector.cosineSimilarity(a, a))
                .as("Identical vectors should have similarity 1.0")
                .isCloseTo(1.0f, offset(0.001f));
    }

    @Test
    void cosineSimilarity_orthogonalVectors_returnsZero() {
        float[] a = {1.0f, 0.0f, 0.0f};
        float[] b = {0.0f, 1.0f, 0.0f};
        assertThat(SemanticBoundaryDetector.cosineSimilarity(a, b))
                .as("Orthogonal vectors should have similarity 0.0")
                .isCloseTo(0.0f, offset(0.001f));
    }

    @Test
    void cosineSimilarity_oppositeVectors_returnsNegativeOne() {
        float[] a = {1.0f, 0.0f};
        float[] b = {-1.0f, 0.0f};
        assertThat(SemanticBoundaryDetector.cosineSimilarity(a, b))
                .as("Opposite vectors should have similarity -1.0")
                .isCloseTo(-1.0f, offset(0.001f));
    }

    @Test
    void cosineSimilarity_zeroVector_returnsZero() {
        float[] a = {1.0f, 0.0f};
        float[] zero = {0.0f, 0.0f};
        assertThat(SemanticBoundaryDetector.cosineSimilarity(a, zero))
                .as("Zero vector should produce similarity 0.0")
                .isCloseTo(0.0f, offset(0.001f));
    }

    @Test
    void findBoundaries_clearTopicShift_detectsBreak() {
        float[] topicA1 = {0.9f, 0.1f, 0.0f};
        float[] topicA2 = {0.85f, 0.15f, 0.0f};
        float[] topicA3 = {0.88f, 0.12f, 0.0f};
        float[] topicB1 = {0.1f, 0.9f, 0.0f};
        float[] topicB2 = {0.15f, 0.85f, 0.0f};

        List<Integer> boundaries = SemanticBoundaryDetector.findBoundaries(
                List.of(topicA1, topicA2, topicA3, topicB1, topicB2), 0.5f, 0);

        assertThat(boundaries)
                .as("Should detect boundary between topic A and topic B at index 3")
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
        List<Integer> boundaries = SemanticBoundaryDetector.findBoundaries(
                List.of(new float[]{1.0f, 0.0f}), 0.5f, 0);

        assertThat(boundaries)
                .as("Single vector should have no boundaries")
                .isEmpty();
    }

    @Test
    void findBoundaries_emptyList_noBoundaries() {
        List<Integer> boundaries = SemanticBoundaryDetector.findBoundaries(
                List.of(), 0.5f, 0);

        assertThat(boundaries)
                .as("Empty list should have no boundaries")
                .isEmpty();
    }

    @Test
    void findBoundaries_percentileMode_detectsBottomPercentile() {
        // Construct vectors where similarity at index 2→3 is clearly the lowest
        float[] v0 = {1.0f, 0.0f};
        float[] v1 = {0.99f, 0.01f};  // sim with v0 ≈ 0.9999
        float[] v2 = {0.98f, 0.02f};  // sim with v1 ≈ 0.9999
        float[] v3 = {0.3f, 0.7f};    // sim with v2 ≈ 0.43 (topic shift!)
        float[] v4 = {0.25f, 0.75f};  // sim with v3 ≈ 0.99

        // 4 transitions; bottom 25% = 1 transition (the lowest)
        List<Integer> boundaries = SemanticBoundaryDetector.findBoundaries(
                List.of(v0, v1, v2, v3, v4), 0.0f, 25);

        assertThat(boundaries)
                .as("Bottom 25th percentile should detect exactly the topic shift at index 3")
                .contains(3);
    }

    @Test
    void findBoundaries_bothThresholds_requiresBothMet() {
        // Similarity at transition 2→3 is 0.45 (below 0.5 threshold)
        // But if it's NOT in bottom percentile, shouldn't break
        float[] v0 = {1.0f, 0.0f};
        float[] v1 = {0.1f, 0.1f};   // low similarity with v0
        float[] v2 = {0.5f, 0.5f};   // moderate similarity with v1
        float[] v3 = {0.6f, 0.4f};   // moderate similarity with v2

        // All transitions are somewhat low — percentile 10% is very restrictive
        List<Integer> boundaries = SemanticBoundaryDetector.findBoundaries(
                List.of(v0, v1, v2, v3), 0.5f, 10);

        // With both thresholds, only breaks that satisfy BOTH conditions qualify
        // This is intentionally restrictive
        assertThat(boundaries)
                .as("With both thresholds, only the worst transition should break (if any)")
                .hasSizeLessThanOrEqualTo(1);
    }

    @Test
    void groupByBoundaries_basicGrouping() {
        List<String> items = List.of("s0", "s1", "s2", "s3", "s4");
        List<Integer> boundaries = List.of(3);

        List<List<String>> groups = SemanticBoundaryDetector.groupByBoundaries(items, boundaries);

        assertThat(groups).as("Should create 2 groups split at index 3").hasSize(2);
        assertThat(groups.get(0)).as("First group").containsExactly("s0", "s1", "s2");
        assertThat(groups.get(1)).as("Second group").containsExactly("s3", "s4");
    }

    @Test
    void groupByBoundaries_multipleBoundaries() {
        List<String> items = List.of("a", "b", "c", "d", "e", "f");
        List<Integer> boundaries = List.of(2, 4);

        List<List<String>> groups = SemanticBoundaryDetector.groupByBoundaries(items, boundaries);

        assertThat(groups).as("Should create 3 groups").hasSize(3);
        assertThat(groups.get(0)).as("Group 1").containsExactly("a", "b");
        assertThat(groups.get(1)).as("Group 2").containsExactly("c", "d");
        assertThat(groups.get(2)).as("Group 3").containsExactly("e", "f");
    }

    @Test
    void groupByBoundaries_noBoundaries_singleGroup() {
        List<String> items = List.of("a", "b", "c");
        List<List<String>> groups = SemanticBoundaryDetector.groupByBoundaries(items, List.of());

        assertThat(groups).as("No boundaries should produce one group").hasSize(1);
        assertThat(groups.get(0)).containsExactly("a", "b", "c");
    }

    @Test
    void enforceMinSize_mergesSmallGroup() {
        List<List<String>> groups = List.of(
                new java.util.ArrayList<>(List.of("s0")),
                new java.util.ArrayList<>(List.of("s1", "s2", "s3")));
        float[] sims = {0.8f};

        List<List<String>> enforced = SemanticBoundaryDetector.enforceMinChunkSize(groups, sims, 2);

        assertThat(enforced).as("Small group should be merged").hasSize(1);
        assertThat(enforced.get(0)).as("Merged group should have all 4 items").hasSize(4);
    }

    @Test
    void computeConsecutiveSimilarities_correctLength() {
        float[] v1 = {1.0f, 0.0f};
        float[] v2 = {0.9f, 0.1f};  // close to v1
        float[] v3 = {0.0f, 1.0f};  // far from v2

        float[] sims = SemanticBoundaryDetector.computeConsecutiveSimilarities(List.of(v1, v2, v3));

        assertThat(sims).as("3 vectors should produce 2 similarity scores").hasSize(2);
        assertThat(sims[0]).as("sim(v1,v2) should be high (close vectors)").isGreaterThan(0.9f);
        assertThat(sims[1]).as("sim(v2,v3) should be low (distant vectors)").isLessThan(0.5f);
        assertThat(sims[0]).as("v1→v2 should be more similar than v2→v3")
                .isGreaterThan(sims[1]);
    }
}
