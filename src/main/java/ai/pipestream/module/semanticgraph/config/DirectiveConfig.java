package ai.pipestream.module.semanticgraph.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

/**
 * JSON-friendly configuration for a single vector directive.
 * Maps to VectorDirective proto for the semantic indexing orchestrator.
 * <p>
 * The CEL selector points to a field on the PipeDoc (e.g., "document.search_metadata.body").
 * The semantic manager extracts text from that field, then applies the cartesian product
 * of chunker_configs × embedder_configs to produce semantic result sets.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record DirectiveConfig(
        @JsonProperty("source_label") String sourceLabel,
        @JsonProperty("cel_selector") String celSelector,
        @JsonProperty("chunker_configs") List<NamedConfig> chunkerConfigs,
        @JsonProperty("embedder_configs") List<NamedConfig> embedderConfigs,
        @JsonProperty("field_name_template") String fieldNameTemplate
) {

    /**
     * A named configuration with an ID and opaque JSON config.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record NamedConfig(
            @JsonProperty("config_id") String configId,
            @JsonProperty("config") Map<String, Object> config
    ) {
    }
}
