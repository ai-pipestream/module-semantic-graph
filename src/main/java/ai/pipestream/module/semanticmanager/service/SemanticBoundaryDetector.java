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
     * Computes cosine similarity between each pair of consecutive vectors.
     *
     * @return array of length (vectors.size() - 1), or empty if fewer than 2 vectors
     */
    public static float[] computeConsecutiveSimilarities(List<float[]> vectors) {
        if (vectors.size() <= 1) return new float[0];
        float[] sims = new float[vectors.size() - 1];
        for (int i = 0; i < sims.length; i++) {
            sims[i] = cosineSimilarity(vectors.get(i), vectors.get(i + 1));
        }
        return sims;
    }

    /**
     * Finds topic boundary indices using cosine similarity between consecutive
     * sentence vectors.
     *
     * @param sentenceVectors     embeddings for each sentence
     * @param similarityThreshold absolute cutoff — break when sim &lt; threshold (0 to disable)
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

        float[] similarities = computeConsecutiveSimilarities(sentenceVectors);

        // Compute percentile cutoff if requested
        float percentileCutoff = Float.MIN_VALUE;
        if (percentileThreshold > 0) {
            float[] sorted = similarities.clone();
            Arrays.sort(sorted);
            int cutoffIndex = Math.max(0, (int) Math.ceil(sorted.length * percentileThreshold / 100.0) - 1);
            percentileCutoff = sorted[Math.min(cutoffIndex, sorted.length - 1)];
        }

        List<Integer> boundaries = new ArrayList<>();
        for (int i = 0; i < similarities.length; i++) {
            boolean belowThreshold = similarityThreshold > 0 && similarities[i] < similarityThreshold;
            boolean belowPercentile = percentileThreshold > 0 && similarities[i] <= percentileCutoff;

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
     * @param groups               the sentence groups
     * @param boundarySimilarities similarity at each original boundary (length = groups.size()-1)
     * @param minSize              minimum sentences per group
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
                    int mergeTarget;
                    if (i == 0) {
                        mergeTarget = 1;
                    } else if (i == result.size() - 1) {
                        mergeTarget = i - 1;
                    } else {
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
                    break;
                }
            }
        }
        return result;
    }

    /**
     * Splits groups larger than maxSize at their lowest internal similarity point.
     *
     * @param groups       the sentence groups
     * @param similarities the full consecutive similarity array (across all sentences)
     * @param boundaries   the boundary indices that produced the groups
     * @param maxSize      maximum sentences per group
     */
    public static <T> List<List<T>> enforceMaxChunkSize(
            List<List<T>> groups,
            float[] similarities,
            List<Integer> boundaries,
            int maxSize) {

        if (maxSize <= 0) {
            return new ArrayList<>(groups);
        }

        List<List<T>> result = new ArrayList<>();
        int globalOffset = 0;

        for (int g = 0; g < groups.size(); g++) {
            List<T> group = groups.get(g);
            if (group.size() <= maxSize) {
                result.add(new ArrayList<>(group));
                globalOffset += group.size();
                continue;
            }

            // Extract internal similarities for this group
            int simStart = globalOffset;
            int simEnd = Math.min(globalOffset + group.size() - 1, similarities.length);
            float[] internalSims = simStart < simEnd
                    ? Arrays.copyOfRange(similarities, simStart, simEnd)
                    : new float[0];

            splitRecursive(group, internalSims, maxSize, result);
            globalOffset += group.size();
        }
        return result;
    }

    private static <T> void splitRecursive(List<T> group, float[] sims, int maxSize, List<List<T>> out) {
        if (group.size() <= maxSize) {
            out.add(new ArrayList<>(group));
            return;
        }

        int splitAt = group.size() / 2;
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
        List<T> left = new ArrayList<>(group.subList(0, splitAt));
        List<T> right = new ArrayList<>(group.subList(splitAt, group.size()));
        float[] leftSims = splitAt - 1 <= sims.length
                ? Arrays.copyOfRange(sims, 0, Math.max(0, splitAt - 1)) : new float[0];
        float[] rightSims = splitAt < sims.length
                ? Arrays.copyOfRange(sims, splitAt, sims.length) : new float[0];

        splitRecursive(left, leftSims, maxSize, out);
        splitRecursive(right, rightSims, maxSize, out);
    }
}
