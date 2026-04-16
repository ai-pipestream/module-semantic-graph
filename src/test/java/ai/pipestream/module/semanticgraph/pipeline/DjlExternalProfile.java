package ai.pipestream.module.semanticgraph.pipeline;

import io.quarkus.test.junit.QuarkusTestProfile;

import java.util.Map;

/**
 * Shared {@link QuarkusTestProfile} for every IT in this package that needs
 * the {@code @RestClient DjlServingClient} pointed at an external DJL Serving
 * instance (defaulted to {@code localhost:18090}, override with
 * {@code -Ddjl.host=... -Ddjl.port=...}).
 *
 * <p>One concrete class shared by every IT so Quarkus only spins up the
 * test app once per gradle test fork. Per-class {@code @TestProfile}
 * implementations cause Quarkus to silently skip one of the IT classes
 * when both are scheduled in the same fork.
 */
public class DjlExternalProfile implements QuarkusTestProfile {
    @Override
    public Map<String, String> getConfigOverrides() {
        String host = System.getProperty("djl.host", "localhost");
        String port = System.getProperty("djl.port", "18090");
        return Map.of(
                "quarkus.rest-client.djl-serving.url", "http://" + host + ":" + port);
    }
}
