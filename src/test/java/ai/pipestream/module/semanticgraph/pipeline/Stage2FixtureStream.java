package ai.pipestream.module.semanticgraph.pipeline;

import ai.pipestream.data.v1.PipeDoc;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import java.util.zip.GZIPInputStream;

/**
 * Streaming loader for Stage-2 {@link PipeDoc} fixtures shipped by the
 * {@code ai.pipestream:embedder-pipedocs-court} Maven artifact. Each fixture
 * lives at {@code /fixtures/court/stage2_{variant}_{1000|full_100}/doc_NNNN.pb.gz}
 * — one non-delimited gzipped {@link PipeDoc} per file, stable
 * {@code doc_0001} ... {@code doc_NNNN} naming.
 *
 * <p>R3's scale ITs need to drive 100- or 1000-document fixtures through the
 * pipeline without holding the whole set in memory (the 1000-doc set lands at
 * ~99–186 MB on disk depending on chunker variant; the 100-doc full set adds
 * sentences_internal SPRs and lands at ~120–136 MB). This loader reads one
 * {@link PipeDoc} at a time and presents the results as a {@link Stream} so
 * the IT can {@code .map} them through R3 and drop each reference before the
 * next one materialises.
 *
 * <h2>Resource convention</h2>
 *
 * <p>Pass the <b>directory path</b> (without trailing slash, no leading slash —
 * classloader convention). The loader iterates {@code doc_NNNN.pb.gz} starting
 * at {@code 0001} and stops at the first absent file. The artifact's stable
 * 4-digit zero-padded naming makes this deterministic.
 *
 * <pre>{@code
 * try (Stream<PipeDoc> docs = Stage2FixtureStream.streamDir(
 *         "fixtures/court/stage2_token_full_100")) {
 *     docs.forEach(doc -> ...);
 * }
 * }</pre>
 *
 * <h2>Memory profile</h2>
 *
 * <p>The returned {@code Stream} is lazy: at any moment exactly one
 * {@link PipeDoc} is fully materialised plus whatever the caller's downstream
 * pipeline holds. Each per-doc {@link InputStream} is closed immediately after
 * its {@link PipeDoc#parseFrom} returns. Stream auto-closing is supported via
 * try-with-resources but largely a no-op since per-doc streams are
 * already closed.
 */
final class Stage2FixtureStream {

    private Stage2FixtureStream() {}

    /**
     * Streams every {@code doc_NNNN.pb.gz} file in {@code dirResourcePath},
     * starting at {@code doc_0001.pb.gz} and stopping at the first absent
     * index. The directory path is a classpath resource path (no leading
     * slash, no trailing slash).
     *
     * <p>Returns a lazy {@link Stream} — files are opened on demand.
     */
    static Stream<PipeDoc> streamDir(String dirResourcePath) {
        ClassLoader cl = Thread.currentThread().getContextClassLoader();
        String dir = dirResourcePath.endsWith("/") ? dirResourcePath : dirResourcePath + "/";

        // Sanity check: the artifact's first file is doc_0001.pb.gz. If it
        // isn't on the classpath, the artifact wasn't packaged in.
        if (cl.getResource(dir + "doc_0001.pb.gz") == null) {
            throw new IllegalStateException(
                    "Stage-2 fixture directory not found on classpath: '" + dir
                            + "doc_0001.pb.gz' missing. Add the embedder-pipedocs-court "
                            + "artifact to testRuntimeClasspath: "
                            + "testImplementation 'ai.pipestream:embedder-pipedocs-court:<version>'");
        }

        Iterator<PipeDoc> it = new DocIterator(cl, dir);
        return StreamSupport.stream(
                Spliterators.spliteratorUnknownSize(it,
                        Spliterator.ORDERED | Spliterator.NONNULL),
                /*parallel*/ false);
    }

    /**
     * Reads exactly one {@link PipeDoc} from a gzipped, non-delimited
     * classpath resource. Used for spot-checks; for bulk iteration use
     * {@link #streamDir}.
     */
    static PipeDoc readOne(String resourcePath) {
        ClassLoader cl = Thread.currentThread().getContextClassLoader();
        InputStream raw = cl.getResourceAsStream(resourcePath);
        if (raw == null) {
            throw new IllegalStateException("Resource not found: " + resourcePath);
        }
        try (InputStream in = raw; GZIPInputStream gz = new GZIPInputStream(in)) {
            return PipeDoc.parseFrom(gz);
        } catch (IOException ioe) {
            throw new UncheckedIOException(ioe);
        }
    }

    /**
     * Iterator that yields one {@link PipeDoc} per {@code doc_NNNN.pb.gz}
     * in a directory, terminating at the first absent index.
     */
    private static final class DocIterator implements Iterator<PipeDoc> {
        private final ClassLoader cl;
        private final String dir;
        private int nextIndex = 1;
        private PipeDoc next;
        private boolean fetched;
        private boolean exhausted;

        DocIterator(ClassLoader cl, String dir) {
            this.cl = cl;
            this.dir = dir;
        }

        @Override
        public boolean hasNext() {
            if (fetched) return next != null;
            if (exhausted) return false;
            String resource = dir + String.format("doc_%04d.pb.gz", nextIndex);
            InputStream raw = cl.getResourceAsStream(resource);
            if (raw == null) {
                next = null;
                fetched = true;
                exhausted = true;
                return false;
            }
            try (InputStream in = raw;
                 GZIPInputStream gz = new GZIPInputStream(new BufferedInputStream(in))) {
                next = PipeDoc.parseFrom(gz);
            } catch (IOException ioe) {
                throw new UncheckedIOException(
                        "Failed reading " + resource + ": " + ioe.getMessage(), ioe);
            }
            fetched = true;
            nextIndex++;
            return true;
        }

        @Override
        public PipeDoc next() {
            if (!hasNext()) throw new NoSuchElementException();
            PipeDoc out = next;
            next = null;
            fetched = false;
            return out;
        }
    }
}
