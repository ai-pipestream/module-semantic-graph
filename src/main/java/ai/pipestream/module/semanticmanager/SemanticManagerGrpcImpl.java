package ai.pipestream.module.semanticmanager;

import ai.pipestream.data.module.v1.*;
import ai.pipestream.data.v1.PipeDoc;
import ai.pipestream.data.v1.ProcessConfiguration;
import ai.pipestream.module.semanticmanager.config.SemanticManagerOptions;
import ai.pipestream.module.semanticmanager.service.SemanticIndexingOrchestrator;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.protobuf.Struct;
import com.google.protobuf.Value;
import com.google.protobuf.util.JsonFormat;
import io.quarkus.grpc.GrpcService;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Semantic Manager gRPC service implementation. Implements PipeStepProcessorService
 * so the pipeline engine treats it as a normal module step.
 *
 * Receives a PipeDoc, resolves VectorSets (the semantic "recipe"), fans out to
 * chunker and embedder services via streaming RPCs, and returns an enriched PipeDoc
 * with all SemanticProcessingResults populated.
 */
@Singleton
@GrpcService
public class SemanticManagerGrpcImpl implements PipeStepProcessorService {

    private static final Logger log = LoggerFactory.getLogger(SemanticManagerGrpcImpl.class);

    @Inject
    ObjectMapper objectMapper;

    @Inject
    SemanticIndexingOrchestrator orchestrator;

    @Override
    public Uni<ProcessDataResponse> processData(ProcessDataRequest request) {
        if (request == null) {
            log.error("Received null request");
            return Uni.createFrom().item(createErrorResponse("Request cannot be null", null));
        }

        return Uni.createFrom().item(request)
                .chain(this::doProcess)
                .onFailure().recoverWithItem(throwable -> {
                    String errorMessage = "Error in SemanticManager: " + throwable.getMessage();
                    log.error(errorMessage, throwable);
                    return createErrorResponse(errorMessage, throwable);
                });
    }

    private Uni<ProcessDataResponse> doProcess(ProcessDataRequest request) {
        if (!request.hasDocument()) {
            return Uni.createFrom().item(
                    ProcessDataResponse.newBuilder()
                            .setSuccess(true)
                            .addProcessorLogs("Semantic manager: no document to process.")
                            .build());
        }

        PipeDoc inputDoc = request.getDocument();
        ServiceMetadata metadata = request.getMetadata();
        String nodeId = metadata.getPipeStepName();
        String streamId = metadata.getStreamId();

        log.info("SemanticManager processing doc: {}, step: {}, stream: {}",
                inputDoc.getDocId(), nodeId, streamId);

        // Parse configuration
        SemanticManagerOptions options = parseOptions(request.getConfig());

        // Run orchestration
        return orchestrator.orchestrate(inputDoc, options, nodeId)
                .map(enrichedDoc -> {
                    int semanticResultCount = enrichedDoc.hasSearchMetadata()
                            ? enrichedDoc.getSearchMetadata().getSemanticResultsCount()
                            : 0;

                    String logMsg = String.format(
                            "Semantic manager produced %d SemanticProcessingResults for doc: %s",
                            semanticResultCount, inputDoc.getDocId());
                    log.info(logMsg);

                    return ProcessDataResponse.newBuilder()
                            .setSuccess(true)
                            .setOutputDoc(enrichedDoc)
                            .addProcessorLogs(logMsg)
                            .build();
                });
    }

    private SemanticManagerOptions parseOptions(ProcessConfiguration config) {
        try {
            if (config != null && config.hasJsonConfig() && config.getJsonConfig().getFieldsCount() > 0) {
                String jsonStr = JsonFormat.printer().print(config.getJsonConfig());
                return objectMapper.readValue(jsonStr, SemanticManagerOptions.class);
            }
        } catch (Exception e) {
            log.warn("Failed to parse SemanticManagerOptions, using defaults: {}", e.getMessage());
        }
        return new SemanticManagerOptions();
    }

    @Override
    public Uni<GetServiceRegistrationResponse> getServiceRegistration(GetServiceRegistrationRequest request) {
        log.info("Semantic manager service registration requested");

        Capabilities capabilities = Capabilities.newBuilder()
                .addTypes(CapabilityType.CAPABILITY_TYPE_UNSPECIFIED)
                .build();

        GetServiceRegistrationResponse.Builder responseBuilder = GetServiceRegistrationResponse.newBuilder()
                .setModuleName("semantic-manager")
                .setVersion("1.0.0-SNAPSHOT")
                .setDisplayName("Semantic Manager")
                .setDescription("Semantic indexing coordinator - multiplexes chunking and embedding across VectorSets")
                .setJsonConfigSchema(SemanticManagerOptions.getJsonV7Schema())
                .setCapabilities(capabilities)
                .setHealthCheckPassed(true)
                .setHealthCheckMessage("Semantic manager module is ready");

        return Uni.createFrom().item(responseBuilder.build());
    }

    private ProcessDataResponse createErrorResponse(String errorMessage, Throwable e) {
        ProcessDataResponse.Builder responseBuilder = ProcessDataResponse.newBuilder();
        responseBuilder.setSuccess(false);
        responseBuilder.addProcessorLogs(errorMessage);

        Struct.Builder errorDetailsBuilder = Struct.newBuilder();
        errorDetailsBuilder.putFields("error_message",
                Value.newBuilder().setStringValue(errorMessage).build());
        if (e != null) {
            errorDetailsBuilder.putFields("error_type",
                    Value.newBuilder().setStringValue(e.getClass().getName()).build());
            if (e.getCause() != null) {
                errorDetailsBuilder.putFields("error_cause",
                        Value.newBuilder().setStringValue(e.getCause().getMessage()).build());
            }
        }
        responseBuilder.setErrorDetails(errorDetailsBuilder.build());
        return responseBuilder.build();
    }
}
