package ai.pipestream.module.semanticmanager.mock;

import ai.pipestream.data.v1.NlpDocumentAnalysis;
import ai.pipestream.data.v1.SentenceSpan;
import ai.pipestream.semantic.v1.ChunkConfigEntry;
import ai.pipestream.semantic.v1.SemanticChunkerService;
import ai.pipestream.semantic.v1.StreamChunksRequest;
import ai.pipestream.semantic.v1.StreamChunksResponse;
import io.quarkus.grpc.GrpcService;
import io.smallrye.mutiny.Multi;
import jakarta.inject.Singleton;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * In-process mock chunker that splits text into simple word-based chunks.
 * Supports multi-config requests (repeated ChunkConfigEntry) — produces
 * separate chunk sequences per config, each tagged with chunk_config_id.
 * Includes NlpDocumentAnalysis on the last chunk of the response.
 */
@Singleton
@GrpcService
public class MockChunkerService implements SemanticChunkerService {

    private static final Logger log = LoggerFactory.getLogger(MockChunkerService.class);
    private static final int WORDS_PER_CHUNK = 10;

    /**
     * Error simulation prefix for chunk_config_id values.
     * <ul>
     *   <li>{@code __error_crash} — fails the stream with INTERNAL error</li>
     *   <li>{@code __error_empty} — returns zero chunks (completes immediately)</li>
     * </ul>
     */
    private static final String ERROR_PREFIX = "__error_";

    @Override
    public Multi<StreamChunksResponse> streamChunks(StreamChunksRequest request) {
        String text = request.getTextContent();
        String docId = request.getDocId();
        String sourceField = request.getSourceFieldName();
        String requestId = request.getRequestId();

        // Multi-config path: chunk_configs field is populated
        if (request.getChunkConfigsCount() > 0) {
            // Check for error simulation triggers
            for (ChunkConfigEntry entry : request.getChunkConfigsList()) {
                if (entry.getChunkConfigId().startsWith(ERROR_PREFIX)) {
                    return handleChunkerError(entry.getChunkConfigId(), docId);
                }
            }

            log.info("MockChunker (multi-config): chunking doc={}, sourceField={}, textLen={}, configs={}",
                    docId, sourceField, text.length(), request.getChunkConfigsCount());

            List<StreamChunksResponse> allChunks = new ArrayList<>();

            for (int i = 0; i < request.getChunkConfigsCount(); i++) {
                ChunkConfigEntry entry = request.getChunkConfigs(i);
                String configId = entry.getChunkConfigId();
                boolean isLastConfig = (i == request.getChunkConfigsCount() - 1);

                List<StreamChunksResponse> configChunks = splitIntoChunks(
                        text, requestId, docId, configId, sourceField, isLastConfig);
                allChunks.addAll(configChunks);
            }

            log.info("MockChunker (multi-config): produced {} total chunks for doc={} across {} configs",
                    allChunks.size(), docId, request.getChunkConfigsCount());
            return Multi.createFrom().iterable(allChunks);
        }

        // Legacy single-config path
        String configId = request.getChunkConfigId();
        log.info("MockChunker (legacy): chunking doc={}, configId={}, sourceField={}, textLen={}",
                docId, configId, sourceField, text.length());

        List<StreamChunksResponse> chunks = splitIntoChunks(
                text, requestId, docId, configId, sourceField, true);

        log.info("MockChunker (legacy): produced {} chunks for doc={}", chunks.size(), docId);
        return Multi.createFrom().iterable(chunks);
    }

    private List<StreamChunksResponse> splitIntoChunks(
            String text, String requestId, String docId, String configId,
            String sourceField, boolean includeNlpOnLast) {

        List<StreamChunksResponse> chunks = new ArrayList<>();
        String[] words = text.split("\\s+");

        int chunkNumber = 0;
        int charOffset = 0;

        for (int i = 0; i < words.length; i += WORDS_PER_CHUNK) {
            int end = Math.min(i + WORDS_PER_CHUNK, words.length);
            StringBuilder chunkText = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) chunkText.append(" ");
                chunkText.append(words[j]);
            }

            String content = chunkText.toString();
            int startOffset = charOffset;
            int endOffset = startOffset + content.length();
            boolean isLast = (end >= words.length);

            StreamChunksResponse.Builder chunkBuilder = StreamChunksResponse.newBuilder()
                    .setRequestId(requestId)
                    .setDocId(docId)
                    .setChunkId(UUID.randomUUID().toString())
                    .setChunkNumber(chunkNumber)
                    .setTextContent(content)
                    .setStartOffset(startOffset)
                    .setEndOffset(endOffset)
                    .setChunkConfigId(configId)
                    .setSourceFieldName(sourceField)
                    .setIsLast(isLast);

            // Include NlpDocumentAnalysis on the last chunk if this is the last config
            if (isLast && includeNlpOnLast) {
                chunkBuilder.setNlpAnalysis(buildMockNlpAnalysis(text));
            }

            chunks.add(chunkBuilder.build());
            chunkNumber++;
            charOffset = endOffset + 1; // +1 for space between chunks
        }

        if (chunks.isEmpty()) {
            // Empty text -> single empty chunk
            StreamChunksResponse.Builder emptyBuilder = StreamChunksResponse.newBuilder()
                    .setRequestId(requestId)
                    .setDocId(docId)
                    .setChunkId(UUID.randomUUID().toString())
                    .setChunkNumber(0)
                    .setTextContent("")
                    .setChunkConfigId(configId)
                    .setSourceFieldName(sourceField)
                    .setIsLast(true);

            if (includeNlpOnLast) {
                emptyBuilder.setNlpAnalysis(buildMockNlpAnalysis(text));
            }

            chunks.add(emptyBuilder.build());
        }

        return chunks;
    }

    private Multi<StreamChunksResponse> handleChunkerError(String configId, String docId) {
        String errorType = configId.substring(ERROR_PREFIX.length());
        return switch (errorType) {
            case "crash" -> {
                log.info("MockChunker: simulating INTERNAL crash for doc={}", docId);
                yield Multi.createFrom().failure(
                        io.grpc.Status.INTERNAL.withDescription("Simulated chunker crash").asRuntimeException());
            }
            case "empty" -> {
                log.info("MockChunker: simulating empty response for doc={}", docId);
                yield Multi.createFrom().empty();
            }
            default -> {
                log.warn("MockChunker: unknown error type '{}', returning empty", errorType);
                yield Multi.createFrom().empty();
            }
        };
    }

    /**
     * Builds a mock NlpDocumentAnalysis with plausible values.
     */
    private NlpDocumentAnalysis buildMockNlpAnalysis(String text) {
        // Simple sentence detection: split on ". " or "? " or "! "
        String[] rawSentences = text.split("(?<=[.?!])\\s+");
        List<SentenceSpan> sentences = new ArrayList<>();
        int offset = 0;
        for (String s : rawSentences) {
            if (!s.isEmpty()) {
                int idx = text.indexOf(s, offset);
                if (idx >= 0) {
                    sentences.add(SentenceSpan.newBuilder()
                            .setText(s)
                            .setStartOffset(idx)
                            .setEndOffset(idx + s.length())
                            .build());
                    offset = idx + s.length();
                }
            }
        }

        String[] words = text.split("\\s+");
        int totalTokens = words.length;

        return NlpDocumentAnalysis.newBuilder()
                .addAllSentences(sentences)
                .setDetectedLanguage("eng")
                .setLanguageConfidence(0.95f)
                .setTotalTokens(totalTokens)
                .setNounDensity(0.25f)
                .setVerbDensity(0.15f)
                .setAdjectiveDensity(0.08f)
                .setAdverbDensity(0.05f)
                .setContentWordRatio(0.55f)
                .setUniqueLemmaCount((int) (totalTokens * 0.7))
                .setLexicalDensity(0.55f)
                .build();
    }
}
