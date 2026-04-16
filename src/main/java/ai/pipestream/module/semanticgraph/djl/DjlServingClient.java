package ai.pipestream.module.semanticgraph.djl;

import io.smallrye.mutiny.Uni;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

/**
 * MicroProfile REST client for the DJL Serving HTTP API.
 *
 * <p>This file is a <b>local copy</b> of the interface that lives in the
 * {@code quarkus-djl-embeddings} extension inside the {@code module-embedder}
 * monorepo (path:
 * {@code quarkus-djl-embeddings/runtime/src/main/java/ai/pipestream/quarkus/djl/serving/runtime/client/DjlServingClient.java}).
 *
 * <p>It is duplicated here only because the extension is not yet published as
 * a standalone Maven artifact (verified 2026-04-15: probing
 * {@code ai.pipestream.module:quarkus-djl-embeddings-runtime:0.0.1-SNAPSHOT}
 * against Sonatype snapshots returns 404). Once the extension publishes, this
 * file should be deleted and every {@code @RestClient DjlServingClient} in
 * this module re-pointed to
 * {@code ai.pipestream.quarkus.djl.serving.runtime.client.DjlServingClient}.
 * The {@code configKey} string ({@code "djl-serving"}) is intentionally
 * identical to the extension's so the application.properties entries
 * (e.g. {@code quarkus.rest-client.djl-serving.url}) survive the swap unchanged.
 *
 * <p>Only the methods this module actually uses are mirrored. The extension's
 * registerModel admin operation is intentionally omitted — semantic-graph never
 * registers models, it only consults / invokes already-loaded ones.
 */
@Path("/")
@RegisterRestClient(configKey = "djl-serving")
public interface DjlServingClient {

    /** Liveness probe. Returns "Healthy" when DJL Serving is up. */
    @GET
    @Path("/ping")
    Uni<String> ping();

    /**
     * Synchronous prediction. The {@code input} payload follows DJL Serving's
     * sentence-transformer convention: {@code {"inputs": ["text1", "text2", ...]}}.
     * The response is a JSON array of float arrays, one per input in order.
     */
    @POST
    @Path("/predictions/{modelName}")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Uni<JsonArray> predict(@PathParam("modelName") String modelName, JsonObject input);

    /**
     * Lists models registered with DJL Serving. Used by the boundary helper to
     * fail fast with FAILED_PRECONDITION when {@code boundary_embedding_model_id}
     * is not currently loaded — per DESIGN.md §21.3, no fallback to "first
     * available model" is allowed.
     */
    @GET
    @Path("/models")
    @Produces(MediaType.APPLICATION_JSON)
    Uni<JsonObject> listModels();
}
