package ai.pipestream.module.semanticgraph.retry;

import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;

import java.time.Duration;
import java.util.function.Supplier;

/**
 * Retry loop for semantic-graph DJL calls, per DESIGN.md §10.2. Mirrors
 * {@code EmbeddingRetryPolicy} in {@code module-embedder}.
 *
 * <p>Composes {@link SemanticGraphRetryClassifier#isTransient(Throwable)}
 * with a recursive re-subscription loop: each attempt creates a fresh
 * {@link Uni} via the {@link Supplier} argument (a cached Uni that has
 * already emitted a failure would just re-emit the same failure on retry —
 * this is the bug the embedder repo calls out and we inherited).
 *
 * <h2>Contract</h2>
 * <ul>
 *   <li>{@code maxAttempts} = retries AFTER the first failure. {@code 0} = fire once.</li>
 *   <li>Backoff is exponential: {@code backoffMs * 2^attemptIndex}.</li>
 *   <li>{@code backoffMs = 0} = retry immediately (valid).</li>
 * </ul>
 */
public final class SemanticGraphRetryPolicy {

    private static final Logger LOG = Logger.getLogger(SemanticGraphRetryPolicy.class);

    private SemanticGraphRetryPolicy() {}

    /**
     * Wraps a {@code Uni}-producing supplier with the §10.2 retry policy.
     *
     * @param producer    supplier that creates a fresh {@code Uni<T>} on each attempt
     * @param maxAttempts maximum retry attempts after first failure (≥ 0, 0 = no retry)
     * @param backoffMs   base backoff for exponential retry (≥ 0)
     * @param callContext short human-readable label for log messages
     */
    public static <T> Uni<T> withRetry(
            Supplier<Uni<T>> producer,
            int maxAttempts,
            long backoffMs,
            String callContext) {
        if (maxAttempts < 0) {
            throw new IllegalArgumentException("maxAttempts must be >= 0, got " + maxAttempts);
        }
        if (backoffMs < 0) {
            throw new IllegalArgumentException("backoffMs must be >= 0, got " + backoffMs);
        }
        return attempt(producer, maxAttempts, backoffMs, 0, callContext);
    }

    private static <T> Uni<T> attempt(
            Supplier<Uni<T>> producer,
            int maxAttempts,
            long backoffMs,
            int attemptIndex,
            String callContext) {

        return producer.get()
                .onFailure().recoverWithUni(failure -> {
                    boolean transientErr = SemanticGraphRetryClassifier.isTransient(failure);
                    boolean hasMore = attemptIndex < maxAttempts;
                    String label = SemanticGraphRetryClassifier.describe(failure);

                    if (!transientErr) {
                        LOG.debugf("DJL call '%s' failed permanent (%s) — no retry",
                                callContext, label);
                        return Uni.createFrom().failure(failure);
                    }
                    if (!hasMore) {
                        LOG.warnf("DJL call '%s' failed transient (%s) after %d retry attempt(s) — giving up: %s",
                                callContext, label, maxAttempts, failure.getMessage());
                        return Uni.createFrom().failure(failure);
                    }

                    long waitMs = backoffMs << attemptIndex;
                    LOG.warnf("DJL call '%s' failed transient (%s) — retry %d/%d after %dms backoff: %s",
                            callContext, label, attemptIndex + 1, maxAttempts, waitMs, failure.getMessage());

                    Uni<T> next = attempt(producer, maxAttempts, backoffMs, attemptIndex + 1, callContext);
                    if (waitMs == 0L) {
                        return next;
                    }
                    return Uni.createFrom().voidItem()
                            .onItem().delayIt().by(Duration.ofMillis(waitMs))
                            .onItem().transformToUni(ignored -> next);
                });
    }
}
