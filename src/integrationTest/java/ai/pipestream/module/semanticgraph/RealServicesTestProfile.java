package ai.pipestream.module.semanticgraph;

import java.util.HashMap;
import java.util.Map;

/**
 * Test profile for integration tests that use real chunker and embedder services
 * running on their standard dev ports (chunker=19002, embedder=19003).
 * <p>
 * Tests using this profile should call {@code assumeTrue(isServiceReachable(...))}
 * so they skip gracefully when the services aren't running (e.g., in CI).
 */
public class RealServicesTestProfile extends SemanticManagerIntegrationTestProfile {

    @Override
    public Map<String, String> getConfigOverrides() {
        Map<String, String> config = new HashMap<>(super.getConfigOverrides());
        // Point Stork at real dev services
        config.put("stork.chunker.service-discovery.type", "static");
        config.put("stork.chunker.service-discovery.address-list", "localhost:19002");
        config.put("stork.embedder.service-discovery.type", "static");
        config.put("stork.embedder.service-discovery.address-list", "localhost:19003");
        config.put("quarkus.dynamic-grpc.service.chunker.address", "localhost:19002");
        config.put("quarkus.dynamic-grpc.service.embedder.address", "localhost:19003");
        return config;
    }
}
