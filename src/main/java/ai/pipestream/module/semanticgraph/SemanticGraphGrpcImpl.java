package ai.pipestream.module.semanticgraph;

import ai.pipestream.data.module.v1.Capabilities;
import ai.pipestream.data.module.v1.CapabilityType;
import ai.pipestream.data.module.v1.GetServiceRegistrationRequest;
import ai.pipestream.data.module.v1.GetServiceRegistrationResponse;
import ai.pipestream.data.module.v1.PipeStepProcessorService;
import ai.pipestream.data.module.v1.ProcessDataRequest;
import ai.pipestream.data.module.v1.ProcessDataResponse;
import ai.pipestream.data.module.v1.ProcessingOutcome;
import ai.pipestream.data.module.v1.ServiceMetadata;
import ai.pipestream.data.v1.LogEntry;
import ai.pipestream.data.v1.LogEntrySource;
import ai.pipestream.data.v1.LogLevel;
import ai.pipestream.data.v1.ModuleLogOrigin;
import ai.pipestream.data.v1.PipeDoc;
import ai.pipestream.data.v1.ProcessConfiguration;
import ai.pipestream.module.semanticgraph.config.SemanticGraphStepOptions;
import ai.pipestream.module.semanticgraph.pipeline.SemanticGraphPipelineService;
import ai.pipestream.module.semanticgraph.retry.SemanticGraphRetryClassifier;
import ai.pipestream.module.semanticgraph.retry.SemanticGraphRetryClassifier.ErrorCategory;
import ai.pipestream.server.meta.BuildInfoProvider;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.protobuf.Struct;
import com.google.protobuf.Value;
import com.google.protobuf.util.JsonFormat;
import io.quarkus.grpc.GrpcService;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import org.jboss.logging.Logger;

/**
 * Thin gRPC wrapper over {@link SemanticGraphPipelineService} per DESIGN.md §7.3.
 *
 * <h2>Responsibilities — wire format only</h2>
 * <ul>
 *   <li>Parse {@code ProcessConfiguration.json_config} into
 *       {@link SemanticGraphStepOptions} via Jackson; {@code INVALID_ARGUMENT}
 *       on parse error.</li>
 *   <li>Delegate to {@link SemanticGraphPipelineService#process} for the
 *       §7.3 algorithm.</li>
 *   <li>Wrap the resulting {@link PipeDoc} in a {@link ProcessDataResponse}
 *       with {@code PROCESSING_OUTCOME_SUCCESS} and a short info log entry.</li>
 *   <li>On any failure — sync or async — map the throwable to a §10.1
 *       {@link ErrorCategory} via {@link SemanticGraphRetryClassifier#classify}
 *       and stamp {@code grpc_status} onto {@code error_details} so the engine
 *       can route quarantine vs DLQ.</li>
 * </ul>
 *
 * <p>Mirrors the {@code EmbedderGrpcImpl} structure in the embedder repo: the
 * gRPC service owns wire-format concerns only; the actual algorithm lives in
 * a transport-agnostic service bean so it can be reused by testing-sidecar
 * REST endpoints, load harnesses, or admin-UI "try this config on this doc"
 * flows without a gRPC hop.
 */
@GrpcService
@Singleton
public class SemanticGraphGrpcImpl implements PipeStepProcessorService {

    private static final Logger log = Logger.getLogger(SemanticGraphGrpcImpl.class);

    /** §10.1 error-category stamp on {@code ProcessDataResponse.error_details}. */
    private static final String ERROR_DETAIL_GRPC_STATUS = "grpc_status";

    private static final String MODULE_NAME = "semantic-graph";

    @Inject
    ObjectMapper objectMapper;

    @Inject
    SemanticGraphPipelineService pipelineService;

    @Inject
    BuildInfoProvider buildInfoProvider;

    @Override
    public Uni<ProcessDataResponse> processData(ProcessDataRequest request) {
        long startMs = System.currentTimeMillis();

        // Sync-phase validation: null request, missing doc, options parse.
        final PipeDoc inputDoc;
        final String pipeStepName;
        final SemanticGraphStepOptions options;
        try {
            if (request == null || !request.hasDocument()) {
                throw new IllegalArgumentException("No document provided in the request");
            }
            inputDoc = request.getDocument();
            ServiceMetadata meta = request.getMetadata();
            String streamId = meta != null ? meta.getStreamId() : "";
            pipeStepName = meta != null ? meta.getPipeStepName() : MODULE_NAME;
            log.infof("Processing document ID: %s for step: %s in stream: %s",
                    inputDoc.getDocId(), pipeStepName, streamId);
            options = parseOptions(request.getConfig());
        } catch (IllegalArgumentException e) {
            log.warnf("Rejecting request: %s", e.getMessage());
            return Uni.createFrom().item(createErrorResponse(e.getMessage(), e));
        } catch (Exception e) {
            log.warnf("Error preparing request: %s", e.getMessage());
            return Uni.createFrom().item(
                    createErrorResponse("Invalid SemanticGraphStepOptions JSON: " + e.getMessage(), e));
        }

        // Pipeline invocation + response assembly. The service throws sync
        // for sync-phase invariant violations (IAE/ISE) and async-propagates
        // DJL / boundary failures via the returned Uni.
        try {
            return pipelineService.process(inputDoc, options, pipeStepName)
                    .map(outputDoc -> buildSuccessResponse(inputDoc, outputDoc, startMs))
                    .onFailure().recoverWithItem(throwable -> {
                        String msg = "Error in SemanticGraphService: " + throwable.getMessage();
                        log.errorf(throwable, "%s", msg);
                        return createErrorResponse(msg, throwable);
                    });
        } catch (IllegalArgumentException e) {
            log.warnf("INVALID_ARGUMENT: %s", e.getMessage());
            return Uni.createFrom().item(createErrorResponse(e.getMessage(), e));
        } catch (IllegalStateException e) {
            log.warnf("FAILED_PRECONDITION: %s", e.getMessage());
            return Uni.createFrom().item(createErrorResponse(e.getMessage(), e));
        }
    }

    /**
     * Parses {@code ProcessConfiguration.json_config} into
     * {@link SemanticGraphStepOptions}. Returns the canonical defaults when
     * the caller sent no json_config (accepting defaults is NOT a fallback
     * per §21.1 — the record's defaults are the spec's defaults).
     *
     * @throws IllegalArgumentException on JSON parse error
     */
    private SemanticGraphStepOptions parseOptions(ProcessConfiguration config) {
        Struct jsonConfig = config != null ? config.getJsonConfig() : null;
        if (jsonConfig == null || jsonConfig.getFieldsCount() == 0) {
            log.debug("No json_config provided — using canonical SemanticGraphStepOptions defaults");
            return SemanticGraphStepOptions.defaults();
        }
        try {
            String jsonStr = JsonFormat.printer().print(jsonConfig);
            SemanticGraphStepOptions parsed = objectMapper.readValue(
                    jsonStr, SemanticGraphStepOptions.class);
            log.debugf("Parsed SemanticGraphStepOptions: boundaries=%s, model='%s', cap=%d",
                    parsed.effectiveComputeSemanticBoundaries(),
                    parsed.rawBoundaryEmbeddingModelId() == null ? "(unset)"
                            : parsed.rawBoundaryEmbeddingModelId(),
                    parsed.effectiveMaxSemanticChunksPerDoc());
            return parsed;
        } catch (Exception e) {
            throw new IllegalArgumentException(
                    "Invalid SemanticGraphStepOptions JSON: " + e.getMessage(), e);
        }
    }

    private ProcessDataResponse buildSuccessResponse(
            PipeDoc inputDoc, PipeDoc outputDoc, long startMs) {
        long duration = System.currentTimeMillis() - startMs;
        int sprCount = outputDoc.getSearchMetadata().getSemanticResultsCount();
        int inputSprCount = inputDoc.hasSearchMetadata()
                ? inputDoc.getSearchMetadata().getSemanticResultsCount() : 0;
        int appended = sprCount - inputSprCount;
        return ProcessDataResponse.newBuilder()
                .setOutcome(ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS)
                .setOutputDoc(outputDoc)
                .addLogEntries(moduleLog(
                        "Appended " + appended + " stage-3 SPR(s) to document "
                                + inputDoc.getDocId() + " (total now " + sprCount + ") in "
                                + duration + "ms",
                        LogLevel.LOG_LEVEL_INFO))
                .build();
    }

    @Override
    public Uni<GetServiceRegistrationResponse> getServiceRegistration(
            GetServiceRegistrationRequest request) {
        log.info("Semantic-graph service registration requested");
        Capabilities capabilities = Capabilities.newBuilder()
                .addTypes(CapabilityType.CAPABILITY_TYPE_UNSPECIFIED)
                .build();
        return Uni.createFrom().item(GetServiceRegistrationResponse.newBuilder()
                .setModuleName(MODULE_NAME)
                .setVersion(buildInfoProvider.getVersion())
                .setDisplayName("Semantic Graph")
                .setDescription("Stage 3 of the semantic pipeline: centroids + semantic-boundary "
                        + "detection over fully-embedded chunks")
                .setJsonConfigSchema(SemanticGraphStepOptions.getJsonV7Schema())
                .setCapabilities(capabilities)
                .putAllMetadata(buildInfoProvider.registrationMetadata())
                .setHealthCheckPassed(true)
                .setHealthCheckMessage("module-semantic-graph is ready")
                .build());
    }

    // -------------------- helpers --------------------

    private static LogEntry moduleLog(String message, LogLevel level) {
        return LogEntry.newBuilder()
                .setSource(LogEntrySource.LOG_ENTRY_SOURCE_MODULE)
                .setLevel(level)
                .setMessage(message)
                .setTimestampEpochMs(System.currentTimeMillis())
                .setModule(ModuleLogOrigin.newBuilder().setModuleName(MODULE_NAME).build())
                .build();
    }

    /**
     * Builds a {@code PROCESSING_OUTCOME_FAILURE} response with §10.1
     * {@code grpc_status} stamped onto {@code error_details}. The engine
     * reads this field to decide quarantine / DLQ routing.
     */
    private ProcessDataResponse createErrorResponse(String errorMessage, Throwable e) {
        ErrorCategory category = e == null
                ? ErrorCategory.INTERNAL
                : SemanticGraphRetryClassifier.classify(e);

        Struct.Builder errorDetails = Struct.newBuilder();
        errorDetails.putFields(ERROR_DETAIL_GRPC_STATUS,
                Value.newBuilder().setStringValue(category.name()).build());
        errorDetails.putFields("error_message",
                Value.newBuilder().setStringValue(errorMessage).build());
        if (e != null) {
            errorDetails.putFields("error_type",
                    Value.newBuilder().setStringValue(e.getClass().getName()).build());
            errorDetails.putFields("error_label",
                    Value.newBuilder().setStringValue(SemanticGraphRetryClassifier.describe(e))
                            .build());
            if (e.getCause() != null) {
                errorDetails.putFields("error_cause",
                        Value.newBuilder()
                                .setStringValue(String.valueOf(e.getCause().getMessage()))
                                .build());
            }
        }

        return ProcessDataResponse.newBuilder()
                .setOutcome(ProcessingOutcome.PROCESSING_OUTCOME_FAILURE)
                .addLogEntries(moduleLog(errorMessage, LogLevel.LOG_LEVEL_ERROR))
                .setErrorDetails(errorDetails.build())
                .build();
    }
}
