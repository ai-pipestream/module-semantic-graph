package ai.pipestream.module.semanticmanager.service;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;

class CentroidComputerTest {

    @Test
    void averageAndNormalize_twoVectors_returnsNormalizedMean() {
        float[] v1 = {1.0f, 0.0f};
        float[] v2 = {0.0f, 1.0f};

        float[] result = CentroidComputer.averageAndNormalize(List.of(v1, v2));

        // Mean is [0.5, 0.5], L2 norm = sqrt(0.25+0.25) ≈ 0.707
        // Normalized: [0.707, 0.707]
        float expected = (float) (0.5 / Math.sqrt(0.5));
        assertThat(result[0]).as("X component of normalized mean")
                .isCloseTo(expected, offset(0.01f));
        assertThat(result[1]).as("Y component of normalized mean")
                .isCloseTo(expected, offset(0.01f));
        assertL2Normalized(result);
    }

    @Test
    void averageAndNormalize_singleVector_returnsNormalized() {
        float[] v = {3.0f, 4.0f};
        float[] result = CentroidComputer.averageAndNormalize(List.of(v));

        assertThat(result[0]).as("X component").isCloseTo(0.6f, offset(0.01f));
        assertThat(result[1]).as("Y component").isCloseTo(0.8f, offset(0.01f));
        assertL2Normalized(result);
    }

    @Test
    void averageAndNormalize_emptyList_returnsEmptyArray() {
        float[] result = CentroidComputer.averageAndNormalize(List.of());
        assertThat(result).as("Empty input should return empty array").isEmpty();
    }

    @Test
    void averageAndNormalize_identicalVectors_returnsSameDirection() {
        float[] v = {0.6f, 0.8f};
        float[] result = CentroidComputer.averageAndNormalize(List.of(v, v, v));

        assertThat(result[0]).as("X component should preserve direction")
                .isCloseTo(0.6f, offset(0.01f));
        assertThat(result[1]).as("Y component should preserve direction")
                .isCloseTo(0.8f, offset(0.01f));
        assertL2Normalized(result);
    }

    @Test
    void l2Normalize_alreadyNormalized_unchanged() {
        float[] v = {0.6f, 0.8f}; // norm = 1.0
        float[] result = CentroidComputer.l2Normalize(v.clone());
        assertThat(result[0]).as("X unchanged").isCloseTo(0.6f, offset(0.001f));
        assertThat(result[1]).as("Y unchanged").isCloseTo(0.8f, offset(0.001f));
    }

    @Test
    void l2Normalize_zeroVector_returnsZero() {
        float[] v = {0.0f, 0.0f};
        float[] result = CentroidComputer.l2Normalize(v);
        assertThat(result[0]).as("Zero X stays zero").isEqualTo(0.0f);
        assertThat(result[1]).as("Zero Y stays zero").isEqualTo(0.0f);
    }

    @Test
    void detectParagraphBoundaries_doubleNewline_splitsParagraphs() {
        //                0         1         2         3         4         5
        //                0123456789012345678901234567890123456789012345678901234567890
        String text = "Sentence one. Sentence two.\n\nSentence three. Sentence four.";
        // "Sentence one." = [0,13], " Sentence two." = [14,27], gap = [27,29] = "\n\n"
        // "Sentence three." = [29,44], " Sentence four." = [45,59]
        int[][] offsets = {{0, 13}, {14, 27}, {29, 44}, {45, 59}};

        List<List<Integer>> paragraphs = CentroidComputer.detectParagraphBoundaries(text, offsets);

        assertThat(paragraphs).as("Double newline should split into 2 paragraphs").hasSize(2);
        assertThat(paragraphs.get(0)).as("First paragraph has sentences 0,1")
                .containsExactly(0, 1);
        assertThat(paragraphs.get(1)).as("Second paragraph has sentences 2,3")
                .containsExactly(2, 3);
    }

    @Test
    void detectParagraphBoundaries_noParagraphBreaks_singleParagraph() {
        String text = "Sentence one. Sentence two. Sentence three.";
        int[][] offsets = {{0, 14}, {15, 28}, {29, 44}};

        List<List<Integer>> paragraphs = CentroidComputer.detectParagraphBoundaries(text, offsets);

        assertThat(paragraphs).as("No paragraph breaks → single paragraph").hasSize(1);
        assertThat(paragraphs.get(0)).containsExactly(0, 1, 2);
    }

    @Test
    void detectParagraphBoundaries_multipleParagraphs() {
        String text = "A.\n\nB.\n\nC.";
        int[][] offsets = {{0, 2}, {4, 6}, {8, 10}};

        List<List<Integer>> paragraphs = CentroidComputer.detectParagraphBoundaries(text, offsets);

        assertThat(paragraphs).as("Three paragraphs").hasSize(3);
    }

    @Test
    void computeParagraphCentroids_averagesWithinParagraphs() {
        //                012345678901234567890123456
        String text = "Sent A1. Sent A2.\n\nSent B1.";
        // "Sent A1." = [0,8], " Sent A2." = [9,17], gap = text[17..19] = "\n\n"
        // "Sent B1." = [19,27]
        List<float[]> vecs = List.of(
                new float[]{1.0f, 0.0f},
                new float[]{0.0f, 1.0f},
                new float[]{0.5f, 0.5f}
        );
        List<String> texts = List.of("Sent A1.", "Sent A2.", "Sent B1.");
        int[][] offsets = {{0, 8}, {9, 17}, {19, 27}};

        List<CentroidComputer.CentroidResult> centroids =
                CentroidComputer.computeParagraphCentroids(vecs, texts, text, offsets);

        assertThat(centroids).as("2 paragraphs → 2 centroids").hasSize(2);
        assertThat(centroids.get(0).sourceVectorCount())
                .as("First paragraph has 2 sentences").isEqualTo(2);
        assertThat(centroids.get(1).sourceVectorCount())
                .as("Second paragraph has 1 sentence").isEqualTo(1);
        assertL2Normalized(centroids.get(0).vector());
        assertL2Normalized(centroids.get(1).vector());
    }

    @Test
    void computeDocumentCentroid_averagesAllVectors() {
        List<float[]> vecs = List.of(
                new float[]{1.0f, 0.0f},
                new float[]{0.0f, 1.0f}
        );

        CentroidComputer.CentroidResult result =
                CentroidComputer.computeDocumentCentroid(vecs, "full text");

        assertThat(result.sourceVectorCount()).as("Should average 2 vectors").isEqualTo(2);
        assertThat(result.text()).as("Should carry full text").isEqualTo("full text");
        assertL2Normalized(result.vector());
    }

    private void assertL2Normalized(float[] v) {
        float norm = 0f;
        for (float f : v) norm += f * f;
        assertThat((float) Math.sqrt(norm))
                .as("Vector should be L2 normalized (norm ≈ 1.0)")
                .isCloseTo(1.0f, offset(0.001f));
    }
}
