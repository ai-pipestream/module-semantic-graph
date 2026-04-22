package ai.pipestream.module.semanticgraph.djl;

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
 * MicroProfile REST client for the DJL Serving HTTP API. Synchronous —
 * callers MUST be on a virtual / worker thread. Returning plain types
 * selects Quarkus REST Client's sync path, which hands I/O off to the
 * REST-client worker pool and blocks the caller. That breaks the
 * single-event-loop pinning the earlier {@code Uni<…>} surface produced:
 * Vert.x-bound Mutiny completions for an injected REST client always
 * re-entered the caller's event-loop context, so every in-flight call
 * serialised on one thread regardless of pool size.
 *
 * <p>This file is a <b>local copy</b> of the interface that lives in the
 * {@code quarkus-djl-embeddings} extension inside the {@code module-embedder}
 * monorepo (path:
 * {@code quarkus-djl-embeddings/runtime/src/main/java/ai/pipestream/quarkus/djl/serving/runtime/client/DjlServingClient.java}).
 *
 * <p>It is duplicated here only because the extension is not yet published as
 * a standalone Maven artifact. Once the extension publishes, this file should
 * be deleted and every {@code @RestClient DjlServingClient} in this module
 * re-pointed to
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
    String ping();

    /**
     * Synchronous prediction. The {@code input} payload follows DJL Serving's
     * sentence-transformer convention: {@code {"inputs": ["text1", "text2", ...]}}.
     * The response is a JSON array of float arrays, one per input in order.
     */
    @POST
    @Path("/predictions/{modelName}")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    JsonArray predict(@PathParam("modelName") String modelName, JsonObject input);

    /**
     * Lists models registered with DJL Serving. Used by the boundary helper to
     * fail fast with FAILED_PRECONDITION when {@code boundary_embedding_model_id}
     * is not currently loaded — per DESIGN.md §21.3, no fallback to "first
     * available model" is allowed.
     */
    @GET
    @Path("/models")
    @Produces(MediaType.APPLICATION_JSON)
    JsonObject listModels();
}
