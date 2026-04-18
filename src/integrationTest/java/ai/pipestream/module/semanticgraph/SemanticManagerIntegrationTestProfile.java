package ai.pipestream.module.semanticgraph;

import io.quarkus.test.junit.QuarkusTestProfile;

import java.util.Map;

/**
 * Test profile for semantic-manager integration tests.
 * <p>
 * The JAR runs with prod profile by default. We must:
 * - Disable service registration (no Consul available in test)
 * - Share HTTP port for gRPC so ConfigProvider gives the correct port
 * - Set HTTP port to 0 for random port assignment
 * <p>
 * Stork discovery for chunker/embedder is configured by MockServicesTestResource,
 * which starts external mock gRPC servers and returns config overrides pointing
 * the application at those servers.
 */
public class SemanticManagerIntegrationTestProfile implements QuarkusTestProfile {

    @Override
    public Map<String, String> getConfigOverrides() {
        return Map.of(
                // Disable Consul-based service registration
                "pipestream.registration.enabled", "false",
                // Share HTTP port for gRPC so ConfigProvider gives the correct port
                "quarkus.grpc.server.use-separate-server", "false",
                // Random port for the HTTP/gRPC server
                "quarkus.http.port", "0",
                "quarkus.http.test-port", "0"
        );
    }
}
