package ai.pipestream.module.semanticmanager.config;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * Configuration options for the semantic manager, parsed from the ProcessDataRequest's jsonConfig.
 * Specifies which index to use for VectorSet resolution and optional overrides.
 */
public record SemanticManagerOptions(
        @JsonProperty("index_name") String indexName,
        @JsonProperty("vector_set_ids") List<String> vectorSetIds,
        @JsonProperty("max_concurrent_chunkers") Integer maxConcurrentChunkers,
        @JsonProperty("max_concurrent_embedders") Integer maxConcurrentEmbedders
) {
    public static final String DEFAULT_INDEX_NAME = "default-index";
    public static final int DEFAULT_MAX_CONCURRENT_CHUNKERS = 4;
    public static final int DEFAULT_MAX_CONCURRENT_EMBEDDERS = 8;

    public SemanticManagerOptions() {
        this(DEFAULT_INDEX_NAME, null, DEFAULT_MAX_CONCURRENT_CHUNKERS, DEFAULT_MAX_CONCURRENT_EMBEDDERS);
    }

    public String effectiveIndexName() {
        return indexName != null && !indexName.isEmpty() ? indexName : DEFAULT_INDEX_NAME;
    }

    public int effectiveMaxConcurrentChunkers() {
        return maxConcurrentChunkers != null ? maxConcurrentChunkers : DEFAULT_MAX_CONCURRENT_CHUNKERS;
    }

    public int effectiveMaxConcurrentEmbedders() {
        return maxConcurrentEmbedders != null ? maxConcurrentEmbedders : DEFAULT_MAX_CONCURRENT_EMBEDDERS;
    }

    public static String getJsonV7Schema() {
        return """
                {
                  "$schema": "http://json-schema.org/draft-07/schema#",
                  "type": "object",
                  "title": "Semantic Manager Options",
                  "properties": {
                    "index_name": {
                      "type": "string",
                      "description": "OpenSearch index name for VectorSet resolution",
                      "default": "default-index"
                    },
                    "vector_set_ids": {
                      "type": "array",
                      "items": { "type": "string" },
                      "description": "Optional list of specific VectorSet IDs to process (all if empty)"
                    },
                    "max_concurrent_chunkers": {
                      "type": "integer",
                      "description": "Maximum concurrent chunker streams",
                      "default": 4,
                      "minimum": 1
                    },
                    "max_concurrent_embedders": {
                      "type": "integer",
                      "description": "Maximum concurrent embedder streams",
                      "default": 8,
                      "minimum": 1
                    }
                  }
                }
                """;
    }
}
