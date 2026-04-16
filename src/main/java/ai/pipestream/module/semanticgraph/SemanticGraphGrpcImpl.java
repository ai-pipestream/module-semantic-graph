package ai.pipestream.module.semanticgraph;

import ai.pipestream.data.module.v1.Capabilities;
import ai.pipestream.data.module.v1.CapabilityType;
import ai.pipestream.data.module.v1.GetServiceRegistrationRequest;
import ai.pipestream.data.module.v1.GetServiceRegistrationResponse;
import ai.pipestream.data.module.v1.PipeStepProcessorService;
import ai.pipestream.data.module.v1.ProcessDataRequest;
import ai.pipestream.data.module.v1.ProcessDataResponse;
import ai.pipestream.server.meta.BuildInfoProvider;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.quarkus.grpc.GrpcService;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;

/**
 * Stage 3 of the three-step semantic pipeline (DESIGN.md §7.3). Consumes
 * Stage 2 PipeDocs (chunks fully embedded) and emits Stage 3 PipeDocs
 * (centroids + optional semantic-boundary SPRs appended; Stage 2 SPRs
 * preserved byte-for-byte).
 *
 * <p>Phase A scaffold — every {@code processData} call returns
 * {@link Status#UNIMPLEMENTED}. The real Mutiny pipeline lands in Phase C
 * (tasks #71–#73).
 */
@Singleton
@GrpcService
public class SemanticGraphGrpcImpl implements PipeStepProcessorService {

    @Inject
    BuildInfoProvider buildInfoProvider;

    @Override
    public Uni<ProcessDataResponse> processData(ProcessDataRequest request) {
        return Uni.createFrom().failure(
                new StatusRuntimeException(Status.UNIMPLEMENTED.withDescription(
                        "module-semantic-graph processData is not yet implemented; " +
                        "Phase A rename only — Phase C will wire the Mutiny pipeline")));
    }

    @Override
    public Uni<GetServiceRegistrationResponse> getServiceRegistration(GetServiceRegistrationRequest request) {
        Capabilities capabilities = Capabilities.newBuilder()
                .addTypes(CapabilityType.CAPABILITY_TYPE_UNSPECIFIED)
                .build();

        return Uni.createFrom().item(
                GetServiceRegistrationResponse.newBuilder()
                        .setModuleName("semantic-graph")
                        .setVersion(buildInfoProvider.getVersion())
                        .setDisplayName("Semantic Graph")
                        .setDescription("Stage 3 of the semantic pipeline: centroids + semantic-boundary detection over fully-embedded chunks")
                        .setCapabilities(capabilities)
                        .putAllMetadata(buildInfoProvider.registrationMetadata())
                        .setHealthCheckPassed(false)
                        .setHealthCheckMessage("module-semantic-graph is in Phase A scaffold — processData not yet implemented")
                        .build());
    }
}
