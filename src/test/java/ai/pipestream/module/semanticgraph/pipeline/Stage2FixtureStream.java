package ai.pipestream.module.semanticgraph.pipeline;

import ai.pipestream.data.v1.PipeDoc;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import java.util.zip.GZIPInputStream;

/**
 * Streaming loader for Stage-2 {@link PipeDoc} fixtures published as a Maven
 * artifact and resolved through {@code ~/.m2}.
 *
 * <p>R3's scale ITs need to drive 100- or 1000-document fixtures through the
 * pipeline without holding the whole set in memory (the 1000-doc court fixture
 * with 2x2 chunker×embedder fan-out lands at ~2 GB on disk). This class reads
 * one {@link PipeDoc} at a time from a classpath resource and presents the
 * results as a {@link Stream} so the IT can {@code .map} them through R3 and
 * drop each reference before the next one materialises.
 *
 * <h2>Supported resource formats</h2>
 *
 * <p>Auto-detected by the resource path's extension and contents:
 *
 * <ul>
 *   <li><b>{@code <name>.pb} or {@code <name>.pb.gz}</b> — single delimited
 *       protobuf stream. Repeated calls to
 *       {@link PipeDoc#parseDelimitedFrom(InputStream)} until null marks EOF.
 *       Standard Google protobuf framing: each message has a varint length
 *       prefix.</li>
 *   <li><b>{@code <name>.manifest} or {@code <name>.txt}</b> — one classpath
 *       resource path per line. Each referenced file is opened and streamed
 *       in turn; sub-files may themselves be {@code .pb} or {@code .pb.gz}.
 *       Lines starting with {@code #} are treated as comments.</li>
 * </ul>
 *
 * <h2>Memory profile</h2>
 *
 * <p>The returned {@code Stream} is lazy: at any moment exactly one
 * {@link PipeDoc} is fully materialised plus whatever the caller's downstream
 * pipeline holds. The upstream {@link InputStream} is opened on
 * {@code stream(...)} and closed when the stream is closed (auto-closing
 * {@code try (Stream<PipeDoc> s = ...)} works).
 *
 * <h2>Why not a {@code List<PipeDoc>}</h2>
 *
 * <p>The 1000-doc fixture would consume ~2 GB of heap if materialised whole.
 * The 100-doc fixture is small enough that {@code .toList()} works, but
 * keeping the API stream-based avoids a second loader for the big set.
 */
final class Stage2FixtureStream {

    private Stage2FixtureStream() {}

    /**
     * Opens {@code resourcePath} from the test thread's context classloader
     * and returns a lazy stream of {@link PipeDoc}s. The caller is responsible
     * for closing the stream so the underlying {@link InputStream} releases
     * its handle.
     *
     * @throws IllegalStateException if the resource is not found
     * @throws UncheckedIOException  on I/O errors during reading
     */
    static Stream<PipeDoc> stream(String resourcePath) {
        ClassLoader cl = Thread.currentThread().getContextClassLoader();
        InputStream raw = cl.getResourceAsStream(resourcePath);
        if (raw == null) {
            throw new IllegalStateException(
                    "Stage-2 fixture resource not found on classpath: '" + resourcePath
                            + "'. Either add the fixture jar to testRuntimeClasspath, set "
                            + "-Dr3.fixtures.stage2-100=<resource> / -Dr3.fixtures.stage2-1000=<resource> "
                            + "to a resource that exists, or skip the IT.");
        }
        try {
            if (resourcePath.endsWith(".manifest") || resourcePath.endsWith(".txt")) {
                return streamFromManifest(raw, cl);
            }
            return streamFromDelimitedFile(resourcePath, raw);
        } catch (IOException ioe) {
            try { raw.close(); } catch (IOException ignore) { /* best-effort */ }
            throw new UncheckedIOException(ioe);
        }
    }

    private static Stream<PipeDoc> streamFromDelimitedFile(String resourcePath, InputStream raw)
            throws IOException {
        InputStream wrapped = resourcePath.endsWith(".gz")
                ? new GZIPInputStream(new BufferedInputStream(raw))
                : new BufferedInputStream(raw);
        Iterator<PipeDoc> it = delimitedIterator(wrapped);
        return StreamSupport.stream(
                        Spliterators.spliteratorUnknownSize(it,
                                Spliterator.ORDERED | Spliterator.NONNULL),
                        /*parallel*/ false)
                .onClose(() -> {
                    try { wrapped.close(); } catch (IOException ignore) { /* best-effort */ }
                });
    }

    private static Stream<PipeDoc> streamFromManifest(InputStream rawManifest, ClassLoader cl)
            throws IOException {
        java.util.List<String> entries = new java.util.ArrayList<>();
        try (BufferedReader r = new BufferedReader(
                new InputStreamReader(rawManifest, StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                String t = line.trim();
                if (t.isEmpty() || t.startsWith("#")) continue;
                entries.add(t);
            }
        }
        // Lazy concat: open each sub-resource only when the previous one is exhausted.
        return entries.stream().flatMap(entry -> {
            InputStream sub = cl.getResourceAsStream(entry);
            if (sub == null) {
                throw new IllegalStateException(
                        "Manifest references missing resource: '" + entry + "'");
            }
            try {
                return streamFromDelimitedFile(entry, sub);
            } catch (IOException ioe) {
                try { sub.close(); } catch (IOException ignore) { /* best-effort */ }
                throw new UncheckedIOException(ioe);
            }
        });
    }

    private static Iterator<PipeDoc> delimitedIterator(InputStream in) {
        return new Iterator<>() {
            private PipeDoc next;
            private boolean fetched;

            @Override
            public boolean hasNext() {
                if (fetched) return next != null;
                try {
                    next = PipeDoc.parseDelimitedFrom(in);
                } catch (IOException ioe) {
                    throw new UncheckedIOException(ioe);
                }
                fetched = true;
                return next != null;
            }

            @Override
            public PipeDoc next() {
                if (!hasNext()) throw new NoSuchElementException();
                PipeDoc out = next;
                next = null;
                fetched = false;
                return out;
            }
        };
    }
}
