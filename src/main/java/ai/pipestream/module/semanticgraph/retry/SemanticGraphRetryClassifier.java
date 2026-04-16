package ai.pipestream.module.semanticgraph.retry;

import io.grpc.Status;
import io.grpc.StatusException;
import io.grpc.StatusRuntimeException;
import jakarta.ws.rs.ProcessingException;
import jakarta.ws.rs.WebApplicationException;

import java.io.IOException;
import java.net.ConnectException;
import java.net.SocketTimeoutException;
import java.util.Set;
import java.util.concurrent.TimeoutException;

/**
 * Pure-function failure classifier for the semantic-graph step. Mirrors
 * {@code EmbeddingRetryClassifier} in {@code module-embedder} — the two
 * modules share the same transient/permanent taxonomy because they share
 * the same DJL Serving REST backend.
 *
 * <p>Deliberately a verbatim re-implementation rather than a cross-repo
 * dep: pulling {@code module-embedder}'s classes would bring the whole
 * embedder app jar (not a library), and the classifier is small enough
 * that duplication is cleaner than a shared-lib refactor in this session.
 * The {@code module-embedder-api} SPI jar only carries {@link EmbeddingBackend}
 * style interfaces, not helpers like this one.
 *
 * <h2>Transient (retryable) — §10.2</h2>
 * <ul>
 *   <li>gRPC {@link Status.Code#UNAVAILABLE},
 *       {@link Status.Code#DEADLINE_EXCEEDED},
 *       {@link Status.Code#RESOURCE_EXHAUSTED}</li>
 *   <li>{@link WebApplicationException} with 5xx status</li>
 *   <li>{@link TimeoutException}, {@link ConnectException}, {@link SocketTimeoutException}</li>
 *   <li>{@link IOException} (cause traversal), {@link ProcessingException} wrappers</li>
 * </ul>
 *
 * <h2>Permanent — §10.1 fail-fast</h2>
 * <ul>
 *   <li>gRPC {@code NOT_FOUND}/{@code INVALID_ARGUMENT}/{@code FAILED_PRECONDITION}</li>
 *   <li>{@link WebApplicationException} with 4xx status</li>
 *   <li>{@link IllegalArgumentException} / {@link IllegalStateException} /
 *       {@link NullPointerException} — programmer errors</li>
 *   <li>Anything unrecognised — conservative default "don't retry"</li>
 * </ul>
 *
 * <p>The classifier walks the cause chain up to 8 frames before giving up.
 */
public final class SemanticGraphRetryClassifier {

    private static final int MAX_CAUSE_DEPTH = 8;

    private static final Set<Status.Code> TRANSIENT_GRPC_CODES = Set.of(
            Status.Code.UNAVAILABLE,
            Status.Code.DEADLINE_EXCEEDED,
            Status.Code.RESOURCE_EXHAUSTED
    );

    /**
     * §10.1 error category, stamped onto {@code ProcessDataResponse.error_details.grpc_status}
     * so the engine routes quarantine vs DLQ vs replay.
     */
    public enum ErrorCategory {
        /** Caller sent bad input. Engine routes → quarantine. */
        INVALID_ARGUMENT,
        /** Doc state wrong for this step (stage invariant, missing directive, model not loaded,
         *  hard cap exceeded). Engine routes → quarantine. */
        FAILED_PRECONDITION,
        /** Transient backend exhausted retries or unknown internal error. Engine → DLQ. */
        INTERNAL,
        /** Downstream backend unreachable after retries. Engine → DLQ + replay. */
        UNAVAILABLE
    }

    private SemanticGraphRetryClassifier() {}

    /**
     * Returns {@code true} if {@code t} (or any of its causes up to
     * {@value #MAX_CAUSE_DEPTH} frames) is transient and worth retrying.
     * Conservative default: unrecognised → not transient.
     */
    public static boolean isTransient(Throwable t) {
        Throwable cursor = t;
        int depth = 0;
        while (cursor != null && depth < MAX_CAUSE_DEPTH) {
            if (cursor instanceof StatusRuntimeException sre) {
                return TRANSIENT_GRPC_CODES.contains(sre.getStatus().getCode());
            }
            if (cursor instanceof StatusException se) {
                return TRANSIENT_GRPC_CODES.contains(se.getStatus().getCode());
            }
            if (cursor instanceof WebApplicationException wae) {
                int status = wae.getResponse() != null ? wae.getResponse().getStatus() : 0;
                return status >= 500 && status < 600;
            }
            if (cursor instanceof TimeoutException) return true;
            if (cursor instanceof ConnectException) return true;
            if (cursor instanceof SocketTimeoutException) return true;
            if (cursor instanceof ProcessingException) {
                cursor = cursor.getCause();
                depth++;
                continue;
            }
            if (cursor instanceof IOException) return true;
            cursor = cursor.getCause();
            depth++;
        }
        return false;
    }

    /**
     * Maps a failure to its {@link ErrorCategory} for {@code grpc_status} stamping.
     *
     * <p>Routing rules:
     * <ul>
     *   <li>gRPC {@code NOT_FOUND}/{@code INVALID_ARGUMENT}/{@code ALREADY_EXISTS}/{@code OUT_OF_RANGE} → INVALID_ARGUMENT</li>
     *   <li>gRPC {@code FAILED_PRECONDITION} → FAILED_PRECONDITION</li>
     *   <li>gRPC {@code UNAVAILABLE} → UNAVAILABLE</li>
     *   <li>gRPC {@code DEADLINE_EXCEEDED}/{@code RESOURCE_EXHAUSTED}/{@code INTERNAL}/{@code DATA_LOSS}/{@code ABORTED} → INTERNAL</li>
     *   <li>HTTP 4xx → INVALID_ARGUMENT</li>
     *   <li>HTTP 5xx → UNAVAILABLE</li>
     *   <li>{@link IllegalArgumentException} (ours) → INVALID_ARGUMENT</li>
     *   <li>{@link IllegalStateException} → FAILED_PRECONDITION</li>
     *   <li>{@link IOException}/{@link ConnectException}/{@link TimeoutException} → UNAVAILABLE</li>
     *   <li>anything else → INTERNAL</li>
     * </ul>
     */
    public static ErrorCategory classify(Throwable t) {
        Throwable cursor = t;
        int depth = 0;
        while (cursor != null && depth < MAX_CAUSE_DEPTH) {
            if (cursor instanceof StatusRuntimeException sre) {
                return fromGrpcCode(sre.getStatus().getCode());
            }
            if (cursor instanceof StatusException se) {
                return fromGrpcCode(se.getStatus().getCode());
            }
            if (cursor instanceof WebApplicationException wae) {
                int status = wae.getResponse() != null ? wae.getResponse().getStatus() : 0;
                if (status >= 400 && status < 500) return ErrorCategory.INVALID_ARGUMENT;
                if (status >= 500 && status < 600) return ErrorCategory.UNAVAILABLE;
                return ErrorCategory.INTERNAL;
            }
            if (cursor instanceof TimeoutException
                    || cursor instanceof ConnectException
                    || cursor instanceof SocketTimeoutException) {
                return ErrorCategory.UNAVAILABLE;
            }
            if (cursor instanceof IOException) {
                return ErrorCategory.UNAVAILABLE;
            }
            if (cursor instanceof IllegalArgumentException) {
                return ErrorCategory.INVALID_ARGUMENT;
            }
            if (cursor instanceof IllegalStateException) {
                return ErrorCategory.FAILED_PRECONDITION;
            }
            cursor = cursor.getCause();
            depth++;
        }
        return ErrorCategory.INTERNAL;
    }

    private static ErrorCategory fromGrpcCode(Status.Code code) {
        return switch (code) {
            case NOT_FOUND, INVALID_ARGUMENT, ALREADY_EXISTS, OUT_OF_RANGE
                    -> ErrorCategory.INVALID_ARGUMENT;
            case FAILED_PRECONDITION -> ErrorCategory.FAILED_PRECONDITION;
            case UNAVAILABLE -> ErrorCategory.UNAVAILABLE;
            case DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED, INTERNAL, DATA_LOSS, ABORTED
                    -> ErrorCategory.INTERNAL;
            default -> ErrorCategory.INTERNAL;
        };
    }

    /** Short human-readable label for log messages. */
    public static String describe(Throwable t) {
        Throwable cursor = t;
        int depth = 0;
        while (cursor != null && depth < MAX_CAUSE_DEPTH) {
            if (cursor instanceof StatusRuntimeException sre) {
                return sre.getStatus().getCode().name();
            }
            if (cursor instanceof StatusException se) {
                return se.getStatus().getCode().name();
            }
            if (cursor instanceof WebApplicationException wae) {
                int status = wae.getResponse() != null ? wae.getResponse().getStatus() : 0;
                return "HTTP_" + status;
            }
            cursor = cursor.getCause();
            depth++;
        }
        return t.getClass().getSimpleName();
    }
}
