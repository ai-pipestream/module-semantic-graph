package ai.pipestream.module.semanticgraph.metrics;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.DistributionSummary;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Central Micrometer metrics bean for module-semantic-graph. Mirrors the
 * D(f)-shaped observability model used by {@code module-embedder}'s
 * {@code EmbedderMetrics}: report how many docs are in flight on this
 * JVM right now, how many centroids / boundary groups R3 produces per
 * doc, and how long each phase takes.
 *
 * <h2>Meters published</h2>
 *
 * <table>
 *   <caption>Micrometer meters</caption>
 *   <tr><th>Meter</th><th>Type</th><th>What it measures</th></tr>
 *   <tr>
 *     <td>{@code semanticgraph.inflight.docs}</td>
 *     <td>Gauge</td>
 *     <td>D(f) on this JVM. Current count of
 *         {@code processData} calls actively inside the pipeline.
 *         Incremented at gRPC entry, decremented on response (success or
 *         failure).</td>
 *   </tr>
 *   <tr>
 *     <td>{@code semanticgraph.docs.processed.total}</td>
 *     <td>Counter</td>
 *     <td>Successful doc count since JVM start.</td>
 *   </tr>
 *   <tr>
 *     <td>{@code semanticgraph.docs.failed.total}</td>
 *     <td>Counter</td>
 *     <td>Failed doc count since JVM start (any path: parse error,
 *         invariant violation, DJL failure, hard-cap exceeded, etc.).</td>
 *   </tr>
 *   <tr>
 *     <td>{@code semanticgraph.doc.processing}</td>
 *     <td>Timer (p50/p95/p99)</td>
 *     <td>Per-doc end-to-end wall clock from pipeline entry to Stage-3
 *         assembly. This is the DESIGN.md §13 p95 gate metric (target ≤
 *         500 ms).</td>
 *   </tr>
 *   <tr>
 *     <td>{@code semanticgraph.centroid.duration}</td>
 *     <td>Timer (p50/p95/p99)</td>
 *     <td>Wall clock of the full centroid pass for one doc — pure-CPU
 *         averaging across all enabled granularities × all Stage-2
 *         triples.</td>
 *   </tr>
 *   <tr>
 *     <td>{@code semanticgraph.boundary.duration}</td>
 *     <td>Timer (p50/p95/p99)</td>
 *     <td>Wall clock of the boundary pass for one doc — includes
 *         {@link io.smallrye.mutiny.Uni} composition overhead,
 *         {@code isModelLoaded} probe (cached after first call),
 *         {@code SemanticBoundaryDetector} grouping, batched DJL
 *         re-embed, and boundary SPR assembly. A long p95 here with a
 *         flat centroid p95 points at DJL contention.</td>
 *   </tr>
 *   <tr>
 *     <td>{@code semanticgraph.boundary.groups.per_doc}</td>
 *     <td>DistributionSummary (p50/p95/p99, max)</td>
 *     <td>Boundary-group count per doc. p99 sitting at the configured
 *         {@code max_semantic_chunks_per_doc} means thresholds are set
 *         too aggressively; DESIGN.md §7.3 step 5 fails the doc in that
 *         case rather than truncating.</td>
 *   </tr>
 *   <tr>
 *     <td>{@code semanticgraph.centroids.per_doc}</td>
 *     <td>DistributionSummary (p50/p95/p99)</td>
 *     <td>Centroid SPR count appended per doc. Includes document +
 *         paragraph + section centroids across every Stage-2 triple,
 *         so a 2-directive × 2-embedder × full-flag config produces
 *         2×2×3 = 12 centroid SPRs before considering paragraph /
 *         section counts.</td>
 *   </tr>
 *   <tr>
 *     <td>{@code semanticgraph.stage2.sprs.per_doc}</td>
 *     <td>DistributionSummary</td>
 *     <td>Input Stage-2 SPR count, reported at pipeline entry after
 *         {@code assertPostEmbedder} validation succeeds. Reveals the
 *         upstream fan-out shape without re-reading the PipeDoc.</td>
 *   </tr>
 * </table>
 *
 * <h2>Scrape endpoint</h2>
 *
 * <p>Published via the {@code quarkus-micrometer-registry-prometheus}
 * extension at {@code /q/metrics}. Quarkus auto-wires the bean; no
 * explicit registration step needed in the module.
 *
 * <h2>Concurrency</h2>
 *
 * <p>{@link AtomicInteger} for the in-flight gauge; all Micrometer
 * primitives are thread-safe. No mutable state beyond that. Safe for
 * concurrent use from any Vert.x event loop thread.
 */
@ApplicationScoped
public class SemanticGraphMetrics {

    private static final Logger LOG = Logger.getLogger(SemanticGraphMetrics.class);

    @Inject
    MeterRegistry meterRegistry;

    private final AtomicInteger inflightDocs = new AtomicInteger(0);

    private Counter docsProcessedCounter;
    private Counter docsFailedCounter;
    private Timer docProcessingTimer;
    private Timer centroidDurationTimer;
    private Timer boundaryDurationTimer;
    private DistributionSummary boundaryGroupsSummary;
    private DistributionSummary centroidsPerDocSummary;
    private DistributionSummary stage2SprsSummary;

    @PostConstruct
    void init() {
        meterRegistry.gauge("semanticgraph.inflight.docs",
                inflightDocs, AtomicInteger::get);

        docsProcessedCounter = Counter.builder("semanticgraph.docs.processed.total")
                .description("Total successful doc count since JVM start")
                .register(meterRegistry);

        docsFailedCounter = Counter.builder("semanticgraph.docs.failed.total")
                .description("Total failed doc count since JVM start "
                        + "(parse error, assertPostEmbedder failure, DJL failure, hard-cap exceeded, etc.)")
                .register(meterRegistry);

        docProcessingTimer = Timer.builder("semanticgraph.doc.processing")
                .description("Per-doc pipeline wall clock from entry to Stage-3 assembly. "
                        + "§13 p95 gate metric (target ≤ 500 ms).")
                .publishPercentiles(0.5, 0.95, 0.99)
                .register(meterRegistry);

        centroidDurationTimer = Timer.builder("semanticgraph.centroid.duration")
                .description("Wall clock of the full centroid pass (document + paragraph + section "
                        + "across all Stage-2 triples). Pure CPU.")
                .publishPercentiles(0.5, 0.95, 0.99)
                .register(meterRegistry);

        boundaryDurationTimer = Timer.builder("semanticgraph.boundary.duration")
                .description("Wall clock of the boundary pass including isModelLoaded probe, "
                        + "grouping, batched DJL re-embed, and SPR assembly.")
                .publishPercentiles(0.5, 0.95, 0.99)
                .register(meterRegistry);

        boundaryGroupsSummary = DistributionSummary.builder("semanticgraph.boundary.groups.per_doc")
                .description("Boundary-group count per doc after min/max chunk-size enforcement, "
                        + "BEFORE the hard-cap check that can fail the doc.")
                .publishPercentiles(0.5, 0.95, 0.99)
                .register(meterRegistry);

        centroidsPerDocSummary = DistributionSummary.builder("semanticgraph.centroids.per_doc")
                .description("Centroid SPR count appended per doc across all enabled granularities "
                        + "and Stage-2 triples.")
                .publishPercentiles(0.5, 0.95, 0.99)
                .register(meterRegistry);

        stage2SprsSummary = DistributionSummary.builder("semanticgraph.stage2.sprs.per_doc")
                .description("Input Stage-2 SPR count (fan-out depth from upstream chunker × embedder).")
                .publishPercentiles(0.5, 0.95, 0.99)
                .register(meterRegistry);

        LOG.debug("SemanticGraphMetrics registered: inflight.docs, docs.processed.total, "
                + "docs.failed.total, doc.processing, centroid.duration, boundary.duration, "
                + "boundary.groups.per_doc, centroids.per_doc, stage2.sprs.per_doc");
    }

    // =========================================================================
    // D(f) gauge + success / failure counters
    // =========================================================================

    /** Call at gRPC entry. Pairs with exactly one
     *  {@link #docCompleted} or {@link #docFailed}. */
    public void docStarted() {
        inflightDocs.incrementAndGet();
    }

    /** Successful doc — decrement gauge, increment counter, record timer. */
    public void docCompleted(Duration duration) {
        inflightDocs.decrementAndGet();
        docsProcessedCounter.increment();
        if (duration != null && !duration.isNegative()) {
            docProcessingTimer.record(duration);
        }
    }

    /** Failed doc — decrement gauge, increment counter. Duration NOT recorded
     *  because a partial doc's latency isn't comparable to a successful doc's. */
    public void docFailed() {
        inflightDocs.decrementAndGet();
        docsFailedCounter.increment();
    }

    // =========================================================================
    // Phase timers + shape distributions
    // =========================================================================

    /** Records Stage-2 input SPR count at pipeline entry, after invariant pass. */
    public void recordStage2SprCount(int count) {
        stage2SprsSummary.record(count);
    }

    /** Records the centroid pass wall clock for one doc. */
    public void centroidCompleted(Duration duration) {
        if (duration != null && !duration.isNegative()) {
            centroidDurationTimer.record(duration);
        }
    }

    /** Records the boundary pass wall clock for one doc. Called even when
     *  boundaries are disabled — duration is then dominated by the short-circuit. */
    public void boundaryCompleted(Duration duration) {
        if (duration != null && !duration.isNegative()) {
            boundaryDurationTimer.record(duration);
        }
    }

    /** Records boundary-group count per doc (the value being hard-cap checked). */
    public void recordBoundaryGroupCount(int count) {
        boundaryGroupsSummary.record(count);
    }

    /** Records total centroid SPR count appended per doc (across all triples
     *  and granularities). */
    public void recordCentroidSprCount(int count) {
        centroidsPerDocSummary.record(count);
    }
}
