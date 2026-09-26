use super::*;
use serde_json::Value;

pub const VERIFICATION_POLICY: &str = "jope.cpu-context-verification/v1:8-native-meta-train-cases:2-families:cuts0,1,2,3:all-component-and-context-maxabs<=0.0003:finite-f32:accepted-transforms:exact-source-program-assets:nine-controller-verified-execution-pins:threads1:each-job30s:original-proof-window:native-index6c81da15d75dff6abeaf1d40fb46ec77e98d6146f535392905bfaf1153574b9c:no-held-labels";
pub fn verification_policy_sha256() -> String {
    sha(VERIFICATION_POLICY.as_bytes())
}
pub const REFERENCE_REGISTRY_SHA256: &str =
    "22b5de5f2f847fb6e086d125d6fec945a1b4edb7c00b94b49e1ad95e7540cfc4";
pub const REFERENCE_INDEX_SHA256: &str =
    "6c81da15d75dff6abeaf1d40fb46ec77e98d6146f535392905bfaf1153574b9c";
const MAX_ABS: f32 = 3e-4;

/// Manifest must be selected and independently pinned before qualification runs.
/// Native bytes remain private evidence, never rewritten to imitate a new backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceManifest {
    pub schema: String,
    pub verification_policy_sha256: String,
    pub registry_bytes: Vec<u8>,
    pub index_bytes: Vec<u8>,
    pub cases: Vec<ReferenceCase>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceCase {
    pub id: String,
    /// Exact native, train-only preprocessed numeric rows supplied to the child.
    /// Features followed by the regression target. No source/corpus path.
    pub prepared_rows: Vec<Vec<f64>>,
    pub prepared_rows_sha256: String,
    pub context_receipt: Vec<u8>,
    pub context_vector: Vec<u8>,
    pub components: BTreeMap<String, NativeComponent>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeComponent {
    pub receipt: Vec<u8>,
    pub vector: Vec<u8>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducedCase {
    pub id: String,
    pub prepared_rows_sha256: String,
    /// Serialized claims; compared with the native references by [`compare`].
    pub components: ComponentVectors,
    pub vector: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qualification {
    pub schema: String,
    pub source_sha256: String,
    pub producer_sha256: String,
    pub verification_policy_sha256: String,
    pub reference_manifest_sha256: String,
    pub encoder: EncoderIdentity,
    /// Exact immutable bytes from eight independently bounded case workers.
    pub case_outputs: Vec<Vec<u8>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationCaseResult {
    pub schema: String,
    pub source_sha256: String,
    pub producer_sha256: String,
    pub reference_case_sha256: String,
    pub encoder: EncoderIdentity,
    pub produced: ProducedCase,
}

/// A *normalized attestation* made only by the controller after validating the
/// actual ExecutionPool receipt (program, input/output bytes, cgroup evidence,
/// completion, limits and cancellation). This type cannot establish process
/// execution by itself. Its digest must come through a separate trusted pin;
/// workers must never choose their own attestation/pins.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionAttestation {
    pub schema: String,
    pub supervisor_receipt_sha256: String,
    pub program_sha256: String,
    pub input_sha256: String,
    pub output_sha256: String,
    pub completed: bool,
    pub cpu_limit: u32,
    pub memory_max: u64,
    pub swap_max: u64,
    pub threads: u32,
    pub elapsed_seconds: f64,
    pub deadline_seconds: f64,
}
/// These pins are controller inputs, not fields trusted from artifact JSON.
/// Before calling this verifier the controller/child must validate the original
/// bounded execution receipts and derive all nine normalized attestation hashes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionPins {
    pub source_sha256: String,
    pub producer_sha256: String,
    pub artifact_sha256: String,
    pub qualification_sha256: String,
    pub reference_manifest_sha256: String,
    pub context_execution_sha256: String,
    /// One controller-derived attestation pin for every case, in manifest order.
    pub qualification_execution_sha256: Vec<String>,
    pub verification_policy_sha256: String,
    pub training: TrainingIdentity,
    pub input_sha256: String,
    pub preprocessing_sha256: String,
}
impl AdmissionPins {
    /// sha256 of these pins' canonical JSON (field order is fixed by the struct). A controller
    /// issues this digest in the policy request (`producer::Request::context_admission_sha256`)
    /// so a worker never verifies a bundle against pins the bundle chose itself.
    pub fn sha256(&self) -> Result<String> {
        Ok(sha(&serde_json::to_vec(self)?))
    }
}

/// A qualified CPU context: only [`verify_qualified`] constructs one (private fields, no
/// `Deserialize`). An unqualified [`CpuContextArtifact`] never becomes one, so a diagnostic
/// context cannot reach [`ContextHandoff`].
///
/// ```compile_fail,E0277
/// // An unqualified (diagnostic) artifact does not convert into a qualified context.
/// use jope_policy::context_producer::{CpuContextArtifact, VerifiedCpuContext};
/// fn mint(artifact: CpuContextArtifact) -> VerifiedCpuContext {
///     artifact.into()
/// }
/// ```
///
/// ```compile_fail
/// // Fields are private: no struct literal outside `verify_qualified`.
/// let _ = jope_policy::context_producer::VerifiedCpuContext { context: vec![0.0; 2048] };
/// ```
///
/// ```compile_fail,E0277
/// // A handoff cannot be deserialized into existence either.
/// let _: jope_policy::context_producer::ContextHandoff = serde_json::from_slice(b"{}").unwrap();
/// ```
pub struct VerifiedCpuContext {
    context: Vec<f32>,
    training: TrainingIdentity,
    input_sha256: String,
    preprocessing_sha256: String,
    original_labels_sha256: String,
    artifact_sha256: String,
    qualification_sha256: String,
    components: ComponentVectors,
    prepared_rows_sha256: String,
    encoder: EncoderIdentity,
    source_sha256: String,
    producer_sha256: String,
    reference_manifest_sha256: String,
    verification_policy_sha256: String,
    admission_pins_sha256: String,
}
impl VerifiedCpuContext {
    /// Unit-test support: the value `verify_qualified` returns for `artifact` under `pins`,
    /// without the reference-manifest and execution checks (whose pinned native bytes are not
    /// available to unit tests). Never compiled outside the crate's own tests.
    #[cfg(test)]
    pub(crate) fn for_testing(artifact: CpuContextArtifact, pins: &AdmissionPins) -> Self {
        Self {
            context: artifact.vector,
            training: artifact.training,
            input_sha256: artifact.input_sha256,
            preprocessing_sha256: artifact.preprocessing_sha256,
            original_labels_sha256: artifact.original_labels_sha256,
            artifact_sha256: pins.artifact_sha256.clone(),
            qualification_sha256: pins.qualification_sha256.clone(),
            components: artifact.components,
            prepared_rows_sha256: artifact.prepared_rows_sha256,
            encoder: artifact.encoder,
            source_sha256: pins.source_sha256.clone(),
            producer_sha256: pins.producer_sha256.clone(),
            reference_manifest_sha256: pins.reference_manifest_sha256.clone(),
            verification_policy_sha256: pins.verification_policy_sha256.clone(),
            admission_pins_sha256: pins.sha256().unwrap(),
        }
    }
    pub fn context(&self) -> &[f32] {
        &self.context
    }
    /// Digest of the controller pins this context was verified against.
    pub fn admission_pins_sha256(&self) -> &str {
        &self.admission_pins_sha256
    }
    /// Receiver replay of the frozen assembly over the qualified artifact's own components; the
    /// typed context carries the artifact's input scope.
    pub fn replay_assembly(&self, transforms: &FrozenTransforms) -> Result<FinalContext> {
        transforms.replay(&self.components, &self.context, &self.prepared_rows_sha256)
    }
    /// The producer-to-consumer handoff for `consumer`: release identity, typed transformation
    /// chain, input scope and final-context digest. `context` must be this context's replayed
    /// assembly ([`Self::replay_assembly`]).
    pub fn handoff(
        &self,
        context: &FinalContext,
        consumer: ContextConsumer,
    ) -> Result<ContextHandoff> {
        ensure!(
            context.values() == self.context.as_slice()
                && context.scope() == self.prepared_rows_sha256,
            "handoff context is not this qualified context"
        );
        Ok(ContextHandoff {
            schema: HANDOFF_SCHEMA.into(),
            consumer,
            release: ContextRelease {
                encoder: self.encoder.clone(),
                source_sha256: self.source_sha256.clone(),
                producer_sha256: self.producer_sha256.clone(),
                verification_policy_sha256: self.verification_policy_sha256.clone(),
                reference_manifest_sha256: self.reference_manifest_sha256.clone(),
            },
            transformations: transformation_chain(),
            scope: ContextScope {
                training: self.training.clone(),
                input_sha256: self.input_sha256.clone(),
                preprocessing_sha256: self.preprocessing_sha256.clone(),
                prepared_rows_sha256: self.prepared_rows_sha256.clone(),
            },
            artifact_sha256: self.artifact_sha256.clone(),
            qualification_sha256: self.qualification_sha256.clone(),
            admission_pins_sha256: self.admission_pins_sha256.clone(),
            context_sha256: context.sha256(),
        })
    }
    pub fn training(&self) -> &TrainingIdentity {
        &self.training
    }
    pub fn input_sha256(&self) -> &str {
        &self.input_sha256
    }
    pub fn preprocessing_sha256(&self) -> &str {
        &self.preprocessing_sha256
    }
    pub fn original_labels_sha256(&self) -> &str {
        &self.original_labels_sha256
    }
    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }
    pub fn qualification_sha256(&self) -> &str {
        &self.qualification_sha256
    }
}

pub const HANDOFF_SCHEMA: &str = "jope.cpu-context-handoff/v1";

/// The consumer a qualified context is handed to. A handoff names exactly one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextConsumer {
    /// The accepted step-60000 regression policy's proposal worker.
    AcceptedPolicyProposal,
    /// Teacher admission (W3 integrates the authority-bearing consumer).
    TeacherAdmission,
}

/// Static release identity of a qualified context: the pinned encoder closure, the producer
/// program and source, and the verification protocol it was qualified under.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ContextRelease {
    pub encoder: EncoderIdentity,
    pub source_sha256: String,
    pub producer_sha256: String,
    pub verification_policy_sha256: String,
    pub reference_manifest_sha256: String,
}

/// Request-specific input scope of a qualified context.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ContextScope {
    pub training: TrainingIdentity,
    pub input_sha256: String,
    pub preprocessing_sha256: String,
    pub prepared_rows_sha256: String,
}

/// The typed transformation chain of a CPU context, in execution order.
pub fn transformation_chain() -> Vec<String> {
    [
        RowEmbeddings::TRANSFORMATION,
        HyperionSignature::TRANSFORMATION,
        RawFoundation::TRANSFORMATION,
        TargetBlock::TRANSFORMATION,
        ProjectedFoundation::TRANSFORMATION,
        FinalContext::TRANSFORMATION,
    ]
    .map(String::from)
    .to_vec()
}

/// Producer-to-consumer handoff of one qualified CPU context (W2-T07): release identity, typed
/// transformation chain, input scope, final-context digest and the admission it came through.
/// Only [`VerifiedCpuContext::handoff`] constructs one; it serializes for receipts and cannot be
/// deserialized.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ContextHandoff {
    schema: String,
    consumer: ContextConsumer,
    release: ContextRelease,
    transformations: Vec<String>,
    scope: ContextScope,
    artifact_sha256: String,
    qualification_sha256: String,
    admission_pins_sha256: String,
    context_sha256: String,
}
impl ContextHandoff {
    pub fn consumer(&self) -> ContextConsumer {
        self.consumer
    }
    pub fn release(&self) -> &ContextRelease {
        &self.release
    }
    pub fn transformations(&self) -> &[String] {
        &self.transformations
    }
    pub fn scope(&self) -> &ContextScope {
        &self.scope
    }
    pub fn context_sha256(&self) -> &str {
        &self.context_sha256
    }
    pub fn admission_pins_sha256(&self) -> &str {
        &self.admission_pins_sha256
    }
    /// sha256 of this handoff's canonical JSON, for receipts that carry it onward.
    pub fn sha256(&self) -> Result<String> {
        Ok(sha(&serde_json::to_vec(self)?))
    }
    /// Deliver to the accepted policy proposal worker serving `request`. Refuses a handoff made
    /// for another consumer, another release or transformation chain, an admission the request
    /// did not issue, and another context, dataset, training cut or preprocessing.
    pub fn deliver_to_policy(&self, request: &crate::producer::Request) -> Result<()> {
        ensure!(
            self.schema == HANDOFF_SCHEMA
                && self.consumer == ContextConsumer::AcceptedPolicyProposal,
            "context was handed off for another consumer ({:?})",
            self.consumer
        );
        self.release.encoder.require_accepted()?;
        ensure!(
            self.release.verification_policy_sha256 == verification_policy_sha256()
                && self.transformations == transformation_chain(),
            "context release protocol or transformation chain is unknown"
        );
        ensure!(
            request.context_admission_sha256.as_deref()
                == Some(self.admission_pins_sha256.as_str()),
            "context admission was not issued by this request's controller"
        );
        ensure!(
            request.context_sha256 == self.context_sha256
                && request.prepared_sha256 == self.scope.training.prepared_sha256
                && request.training_sha256 == self.scope.training.training_sha256
                && request.cut == self.scope.training.cut as usize,
            "qualified context scope differs from the policy request"
        );
        Ok(())
    }
}

impl CpuProducer {
    /// One case per bounded job (30s, one numerical thread). The caller gives
    /// this worker only one canonical ReferenceCase payload from case_inputs.
    pub fn qualify_case(
        &self,
        case_bytes: &[u8],
        expected_case_sha256: &str,
        source_sha256: &str,
        producer_sha256: &str,
    ) -> Result<QualificationCaseResult> {
        let result = self.produce_reference_case(
            case_bytes,
            expected_case_sha256,
            source_sha256,
            producer_sha256,
        )?;
        validate_reference_case(case_bytes, &result)?;
        Ok(result)
    }

    /// Private unqualified computation. Only a successful comparison can yield
    /// the ordinary worker output; a failed diagnostic is never an admission.
    pub(super) fn produce_reference_case(
        &self,
        case_bytes: &[u8],
        expected_case_sha256: &str,
        source_sha256: &str,
        producer_sha256: &str,
    ) -> Result<QualificationCaseResult> {
        ensure!(
            case_bytes.len() <= 128 << 20 && sha(case_bytes) == expected_case_sha256,
            "qualification case differs from controller input pin"
        );
        validate_hex(source_sha256, 40)?;
        validate_hex(producer_sha256, 64)?;
        let reference: ReferenceCase = serde_json::from_slice(case_bytes)?;
        validate_rows(&reference.prepared_rows)?;
        ensure!(
            reference.prepared_rows_sha256 == sha(&serde_json::to_vec(&reference.prepared_rows)?),
            "qualification case numerical rows changed"
        );
        let components = self.encode_prepared(&reference.prepared_rows)?;
        let context = self.assemble(&components)?;
        ensure!(
            context.scope() == reference.prepared_rows_sha256,
            "qualification products belong to other prepared rows"
        );
        let produced = ProducedCase {
            id: reference.id.clone(),
            prepared_rows_sha256: reference.prepared_rows_sha256.clone(),
            components: components.to_vectors(),
            vector: context.values().to_vec(),
        };
        Ok(QualificationCaseResult {
            schema: "jope.cpu-context-case/v1".into(),
            source_sha256: source_sha256.into(),
            producer_sha256: producer_sha256.into(),
            reference_case_sha256: expected_case_sha256.into(),
            encoder: self.identity.clone(),
            produced,
        })
    }
}

pub(super) fn validate_reference_case(
    case_bytes: &[u8],
    result: &QualificationCaseResult,
) -> Result<()> {
    let reference: ReferenceCase = serde_json::from_slice(case_bytes)?;
    compare(&reference, &result.produced)
}

/// Validate all reference provenance before scheduling, then freeze these exact
/// case bytes and their SHA as the eight independent worker input artifacts.
/// Preserve the existing proof window across retries; this grants no new clock.
pub fn case_inputs(manifest_bytes: &[u8], expected_manifest_sha256: &str) -> Result<Vec<Vec<u8>>> {
    ensure!(
        sha(manifest_bytes) == expected_manifest_sha256,
        "reference manifest differs from pre-registered pin"
    );
    read_manifest(manifest_bytes)?
        .cases
        .iter()
        .map(|case| Ok(serde_json::to_vec(case)?))
        .collect()
}

/// Parent-only arithmetic validation/aggregation after all eight worker outputs
/// are immutable. Never executes an encoder or replaces timed-out/missing cases.
/// Execution attestations are separately checked by verify_qualified.
pub fn aggregate_qualification(
    manifest_bytes: &[u8],
    expected_manifest_sha256: &str,
    case_outputs: Vec<Vec<u8>>,
    source_sha256: &str,
    producer_sha256: &str,
) -> Result<Qualification> {
    ensure!(
        sha(manifest_bytes) == expected_manifest_sha256,
        "reference manifest differs from pre-registered pin"
    );
    validate_hex(source_sha256, 40)?;
    validate_hex(producer_sha256, 64)?;
    let manifest = read_manifest(manifest_bytes)?;
    ensure!(
        case_outputs.len() == manifest.cases.len(),
        "missing qualification case outcomes"
    );
    let first: QualificationCaseResult =
        serde_json::from_slice(case_outputs.first().context("no case outcomes")?)?;
    first.encoder.require_accepted()?;
    let result = Qualification {
        schema: "jope.cpu-context-qualification/v1".into(),
        source_sha256: source_sha256.into(),
        producer_sha256: producer_sha256.into(),
        verification_policy_sha256: verification_policy_sha256(),
        reference_manifest_sha256: expected_manifest_sha256.into(),
        encoder: first.encoder,
        case_outputs,
    };
    verify_case_outputs(&manifest, &result)?;
    Ok(result)
}

fn verify_case_outputs(manifest: &ReferenceManifest, qualification: &Qualification) -> Result<()> {
    // The qualified encoder is the release decision, not whatever the first case reported.
    qualification.encoder.require_accepted()?;
    ensure!(
        qualification.case_outputs.len() == manifest.cases.len(),
        "incomplete qualification cases"
    );
    for (reference, bytes) in manifest.cases.iter().zip(&qualification.case_outputs) {
        ensure!(
            bytes.len() <= 2 << 20,
            "qualification case output too large"
        );
        let result: QualificationCaseResult = serde_json::from_slice(bytes)?;
        ensure!(
            result.schema == "jope.cpu-context-case/v1"
                && result.source_sha256 == qualification.source_sha256
                && result.producer_sha256 == qualification.producer_sha256
                && result.encoder == qualification.encoder,
            "qualification case source/program/assets differ"
        );
        ensure!(
            result.reference_case_sha256 == sha(&serde_json::to_vec(reference)?),
            "qualification case input differs from frozen reference manifest"
        );
        compare(reference, &result.produced)?;
    }
    Ok(())
}

/// Recompute all hashes, native receipt chains and numerical comparisons. No
/// `passed:true` field is accepted. Execution attestations must have independent,
/// controller-derived pins; supplying self-attested pins is not qualification.
pub fn verify_qualified(
    artifact_bytes: &[u8],
    qualification_bytes: &[u8],
    manifest_bytes: &[u8],
    context_execution_bytes: &[u8],
    qualification_execution_bytes: &[Vec<u8>],
    pins: &AdmissionPins,
) -> Result<VerifiedCpuContext> {
    ensure!(
        pins.verification_policy_sha256 == verification_policy_sha256(),
        "verification policy differs"
    );
    for (bytes, pin, maximum) in [
        (artifact_bytes, &pins.artifact_sha256, 16 << 20),
        (qualification_bytes, &pins.qualification_sha256, 16 << 20),
        (manifest_bytes, &pins.reference_manifest_sha256, 128 << 20),
        (
            context_execution_bytes,
            &pins.context_execution_sha256,
            1 << 20,
        ),
    ] {
        ensure!(
            bytes.len() <= maximum && sha(bytes) == *pin,
            "context admission bytes differ from independent pin"
        );
    }
    let artifact: CpuContextArtifact = serde_json::from_slice(artifact_bytes)?;
    artifact.validate()?;
    let qualification: Qualification = serde_json::from_slice(qualification_bytes)?;
    ensure!(
        qualification.schema == "jope.cpu-context-qualification/v1"
            && qualification.verification_policy_sha256 == pins.verification_policy_sha256
            && qualification.reference_manifest_sha256 == pins.reference_manifest_sha256,
        "qualification protocol changed"
    );
    ensure!(
        qualification.source_sha256 == pins.source_sha256
            && artifact.source_sha256 == pins.source_sha256
            && qualification.producer_sha256 == pins.producer_sha256
            && artifact.producer_sha256 == pins.producer_sha256,
        "qualification source/program identity differs"
    );
    ensure!(
        artifact.training == pins.training
            && artifact.input_sha256 == pins.input_sha256
            && artifact.preprocessing_sha256 == pins.preprocessing_sha256,
        "context training input differs from controller cut"
    );
    ensure!(
        artifact.encoder == qualification.encoder,
        "qualified encoder assets/recipe differ from producer"
    );
    verify_execution(
        context_execution_bytes,
        &pins.producer_sha256,
        &pins.input_sha256,
        &pins.artifact_sha256,
        30.0,
    )?;
    let manifest = read_manifest(manifest_bytes)?;
    verify_case_outputs(&manifest, &qualification)?;
    ensure!(
        qualification_execution_bytes.len() == 8 && pins.qualification_execution_sha256.len() == 8,
        "missing bounded qualification execution evidence"
    );
    for (index, reference) in manifest.cases.iter().enumerate() {
        let evidence = &qualification_execution_bytes[index];
        ensure!(
            evidence.len() <= 1 << 20
                && sha(evidence) == pins.qualification_execution_sha256[index],
            "qualification execution differs from independent controller pin"
        );
        verify_execution(
            evidence,
            &pins.producer_sha256,
            &sha(&serde_json::to_vec(reference)?),
            &sha(&qualification.case_outputs[index]),
            30.0,
        )?;
    }
    Ok(VerifiedCpuContext {
        context: artifact.vector,
        training: artifact.training,
        input_sha256: artifact.input_sha256,
        preprocessing_sha256: artifact.preprocessing_sha256,
        original_labels_sha256: artifact.original_labels_sha256,
        artifact_sha256: pins.artifact_sha256.clone(),
        qualification_sha256: pins.qualification_sha256.clone(),
        components: artifact.components,
        prepared_rows_sha256: artifact.prepared_rows_sha256,
        encoder: artifact.encoder,
        source_sha256: pins.source_sha256.clone(),
        producer_sha256: pins.producer_sha256.clone(),
        reference_manifest_sha256: pins.reference_manifest_sha256.clone(),
        verification_policy_sha256: pins.verification_policy_sha256.clone(),
        admission_pins_sha256: pins.sha256()?,
    })
}
/// Independently pinned inputs for reusing native-regression component
/// qualification in an explicitly experimental feature-only classification route.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentQualificationPins {
    pub source_sha40: String,
    pub producer_sha256: String,
    pub qualification_sha256: String,
    pub reference_manifest_sha256: String,
    pub execution_sha256: Vec<String>,
}
/// Numerical component qualification only; this is not classification parity,
/// accepted-regression context admission, or proof of execution by itself.
pub struct VerifiedComponentQualification {
    encoder: EncoderIdentity,
}
impl VerifiedComponentQualification {
    pub fn encoder(&self) -> &EncoderIdentity {
        &self.encoder
    }
}
/// The parent must first rederive all execution attestations from original pool
/// job/started/finished/window files. No caller-supplied passed boolean is used.
pub fn verify_component_qualification(
    qualification_bytes: &[u8],
    manifest_bytes: &[u8],
    executions: &[Vec<u8>],
    pins: &ComponentQualificationPins,
) -> Result<VerifiedComponentQualification> {
    validate_hex(&pins.source_sha40, 40)?;
    validate_hex(&pins.producer_sha256, 64)?;
    ensure!(
        qualification_bytes.len() <= 16 << 20
            && sha(qualification_bytes) == pins.qualification_sha256
            && manifest_bytes.len() <= 128 << 20
            && sha(manifest_bytes) == pins.reference_manifest_sha256,
        "component qualification bytes changed"
    );
    let qualification: Qualification = serde_json::from_slice(qualification_bytes)?;
    ensure!(
        qualification.schema == "jope.cpu-context-qualification/v1"
            && qualification.verification_policy_sha256 == verification_policy_sha256()
            && qualification.reference_manifest_sha256 == pins.reference_manifest_sha256
            && qualification.source_sha256 == pins.source_sha40
            && qualification.producer_sha256 == pins.producer_sha256,
        "component qualification lineage changed"
    );
    let manifest = read_manifest(manifest_bytes)?;
    verify_case_outputs(&manifest, &qualification)?;
    ensure!(
        executions.len() == 8 && pins.execution_sha256.len() == 8,
        "component qualification requires all8 bounded executions"
    );
    for (index, reference) in manifest.cases.iter().enumerate() {
        let evidence = &executions[index];
        ensure!(
            evidence.len() <= 1 << 20 && sha(evidence) == pins.execution_sha256[index],
            "component execution differs from independent pin"
        );
        verify_execution(
            evidence,
            &pins.producer_sha256,
            &sha(&serde_json::to_vec(reference)?),
            &sha(&qualification.case_outputs[index]),
            30.,
        )?;
    }
    Ok(VerifiedComponentQualification {
        encoder: qualification.encoder,
    })
}

fn verify_execution(
    bytes: &[u8],
    program: &str,
    input: &str,
    output: &str,
    maximum_seconds: f64,
) -> Result<()> {
    ensure!(
        maximum_seconds.is_finite() && maximum_seconds > 0.0 && maximum_seconds <= 30.0,
        "invalid predeclared encoding qualification allocation"
    );
    let e: ExecutionAttestation = serde_json::from_slice(bytes)?;
    validate_hex(&e.supervisor_receipt_sha256, 64)?;
    ensure!(
        e.schema == "jope.cpu-context-execution/controller-v1"
            && e.completed
            && e.program_sha256 == program
            && e.input_sha256 == input
            && e.output_sha256 == output,
        "producer execution binding differs"
    );
    ensure!(
        (1..=20).contains(&e.cpu_limit)
            && e.threads == 1
            && e.memory_max > 0
            && e.memory_max <= 64u64 << 30
            && e.swap_max == 0,
        "producer execution exceeded resource envelope"
    );
    ensure!(
        e.elapsed_seconds.is_finite()
            && e.elapsed_seconds >= 0.0
            && e.deadline_seconds.is_finite()
            && e.deadline_seconds > 0.0
            && e.deadline_seconds <= maximum_seconds
            && e.elapsed_seconds <= e.deadline_seconds,
        "producer execution failed deadline"
    );
    Ok(())
}
fn read_manifest(bytes: &[u8]) -> Result<ReferenceManifest> {
    ensure!(bytes.len() <= 128 << 20, "qualification manifest too large");
    let m: ReferenceManifest = serde_json::from_slice(bytes)?;
    ensure!(
        m.schema == "jope.cpu-context-reference/v1"
            && m.verification_policy_sha256 == verification_policy_sha256()
            && m.cases.len() == 8,
        "qualification manifest protocol changed"
    );
    ensure!(
        sha(&m.registry_bytes) == REFERENCE_REGISTRY_SHA256,
        "reference family registry identity changed"
    );
    let registry: Value = serde_json::from_slice(&m.registry_bytes)?;
    ensure!(
        sha(&m.index_bytes) == REFERENCE_INDEX_SHA256,
        "native reference index changed"
    );
    let index: Value = serde_json::from_slice(&m.index_bytes)?;
    ensure!(
        index["schema"] == "parallel.context_index.v1"
            && index["assembly_sha256"] == ASSEMBLY_SHA256
            && index["registry"]["sha256"] == REFERENCE_REGISTRY_SHA256,
        "native reference index lineage changed"
    );

    let mut families: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut ids = BTreeSet::new();
    for case in &m.cases {
        ensure!(ids.insert(&case.id), "duplicate reference case");
        validate_rows(&case.prepared_rows)?;
        ensure!(
            sha(&serde_json::to_vec(&case.prepared_rows)?) == case.prepared_rows_sha256,
            "reference training rows changed"
        );
        let receipt: Value = serde_json::from_slice(&case.context_receipt)?;
        // Native training_sha256 is the headerless csv::Writer representation.
        // Finite f64 Display contains no CSV quoting characters.
        let native_csv = case
            .prepared_rows
            .iter()
            .map(|row| {
                let mut line = row
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                line.push('\n');
                line
            })
            .collect::<String>();
        ensure!(
            receipt["training_sha256"] == sha(native_csv.as_bytes()),
            "qualification numerical rows differ from retained native training cut"
        );
        let family = receipt["family"]
            .as_str()
            .context("reference family absent")?;
        let dataset = receipt["dataset"]
            .as_str()
            .context("reference dataset absent")?;
        let cut = receipt["cut"].as_u64().context("reference cut absent")?;
        let indexed = index["encodings"]
            .as_array()
            .context("native index entries absent")?
            .iter()
            .filter(|e| {
                e["dataset"] == dataset
                    && e["cut"] == cut
                    && e["prepared_sha256"] == receipt["prepared_sha256"]
            })
            .collect::<Vec<_>>();
        ensure!(
            indexed.len() == 1
                && indexed[0]["family"] == family
                && indexed[0]["partition"] == "meta_train"
                && indexed[0]["receipt_sha256"] == sha(&case.context_receipt)
                && indexed[0]["sha256"] == sha(&case.context_vector),
            "reference is not the retained native indexed receipt/vector"
        );

        ensure!(
            cut < 4
                && receipt["partition"] == "meta_train"
                && registry["families"][family]["partition"] == "meta_train",
            "reference is not a selected native training cut"
        );
        ensure!(
            case.id == format!("{dataset}/cut-{cut}"),
            "reference case identity differs"
        );
        let datasets = registry["families"][family]["datasets"]
            .as_array()
            .context("registry family datasets absent")?;
        ensure!(
            datasets.iter().any(|d| d["dataset"] == dataset
                && d["preparation_eligible"] == true
                && d["prepared"].as_array().is_some_and(|ps| ps
                    .iter()
                    .any(|p| p["sha256"] == receipt["prepared_sha256"]))),
            "reference prepared input not in training registry"
        );
        ensure!(
            registry["held_review_datasets"].get(dataset).is_none(),
            "reference dataset held by review"
        );
        ensure!(
            families.entry(family.into()).or_default().insert(cut),
            "duplicate family/cut reference"
        );
        ensure!(
            receipt["schema"] == "parallel.encoding.v1"
                && receipt["encoder_spec_sha256"] == ASSEMBLY_SHA256
                && receipt["dataset_dimensions"] == 1536
                && receipt["target_dimensions"] == 512,
            "native context lineage differs"
        );
        ensure!(
            receipt["output_sha256"] == sha(&case.context_vector),
            "native context vector bytes changed"
        );
        decode_f32(&case.context_vector, 2048)?;
        ensure!(
            case.components.len() == 3,
            "reference component inventory differs"
        );
        for (name, length) in [("hyperion", 512), ("foundation", 2048), ("target", 512)] {
            let component = case
                .components
                .get(name)
                .context("reference component absent")?;
            let r: Value = serde_json::from_slice(&component.receipt)?;
            let bound = &receipt["components"][name];
            ensure!(
                bound["receipt_sha256"] == sha(&component.receipt)
                    && bound["signature_sha256"] == sha(&component.vector),
                "native component receipt/vector chain differs"
            );
            ensure!(
                r["signature_sha256"] == sha(&component.vector)
                    && r["spec_sha256"] == bound["spec_sha256"]
                    && r["final_test_targets_used"] == false,
                "native component input closure differs"
            );
            for field in [
                "dataset",
                "family",
                "cut",
                "prepared_sha256",
                "training_sha256",
                "preprocessing_sha256",
            ] {
                ensure!(
                    r[field] == receipt[field] && !r[field].is_null(),
                    "native component training identity differs: {field}"
                );
            }
            decode_f32(&component.vector, length)?;
        }
    }
    ensure!(
        families.len() == 2 && families.values().all(|cuts| cuts.iter().copied().eq(0..4)),
        "qualification requires 2 training families and exactly cuts0..3"
    );
    Ok(m)
}
fn compare(reference: &ReferenceCase, produced: &ProducedCase) -> Result<()> {
    ensure!(
        reference.id == produced.id
            && reference.prepared_rows_sha256 == produced.prepared_rows_sha256,
        "qualification case/input identity differs"
    );
    produced.components.validate()?;
    for (name, actual, length) in [
        ("hyperion", &produced.components.hyperion, 512),
        ("foundation", &produced.components.foundation, 2048),
        ("target", &produced.components.target, 512),
    ] {
        let expected = decode_f32(
            &reference
                .components
                .get(name)
                .context("missing native component")?
                .vector,
            length,
        )?;
        max_abs(actual, &expected).with_context(|| format!("{name} component"))?;
    }
    max_abs(
        &produced.vector,
        &decode_f32(&reference.context_vector, 2048)?,
    )
    .context("assembled context")?;
    Ok(())
}
fn decode_f32(bytes: &[u8], length: usize) -> Result<Vec<f32>> {
    ensure!(bytes.len() == length * 4, "native FP32 vector size differs");
    let values = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    vector(&values, length)?;
    Ok(values)
}
fn max_abs(actual: &[f32], expected: &[f32]) -> Result<f32> {
    vector(actual, expected.len())?;
    vector(expected, actual.len())?;
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    ensure!(
        error.is_finite() && error <= MAX_ABS,
        "CPU/native parity failed: max_abs={error}, tolerance={MAX_ABS}"
    );
    Ok(error)
}
#[cfg(test)]
mod handoff_tests {
    use super::super::test_support::synthetic_artifact;
    use super::*;
    use crate::producer::Request;

    fn pins(artifact: &CpuContextArtifact) -> AdmissionPins {
        AdmissionPins {
            source_sha256: artifact.source_sha256.clone(),
            producer_sha256: artifact.producer_sha256.clone(),
            artifact_sha256: sha(&serde_json::to_vec(artifact).unwrap()),
            qualification_sha256: sha(b"qualification"),
            reference_manifest_sha256: sha(b"reference manifest"),
            context_execution_sha256: sha(b"context execution"),
            qualification_execution_sha256: vec![sha(b"case execution"); 8],
            verification_policy_sha256: verification_policy_sha256(),
            training: artifact.training.clone(),
            input_sha256: artifact.input_sha256.clone(),
            preprocessing_sha256: artifact.preprocessing_sha256.clone(),
        }
    }
    fn request(pins: &AdmissionPins, context: &FinalContext) -> Request {
        Request {
            worker_source_sha256: sha(b"worker source"),
            prepared_sha256: pins.training.prepared_sha256.clone(),
            training_sha256: pins.training.training_sha256.clone(),
            cut: pins.training.cut as usize,
            context_sha256: context.sha256(),
            checkpoint_sha256: crate::ACCEPTED_STATE_SHA256.into(),
            policy_safetensors_sha256:
                "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e".into(),
            vocabulary_sha256: crate::CHECKPOINT_VOCABULARY_SHA256.into(),
            sampling_seed: 1,
            fallback_seed: 2,
            exploration_seed: 3,
            sampling_rounds: 1,
            sampling_batch: 8,
            sampling_temperature: 1.0,
            challenger: None,
            context_admission_sha256: Some(pins.sha256().unwrap()),
        }
    }

    /// W2-T07: a qualified context carries its release identity, typed transformation chain,
    /// input scope and final-context digest to the intended consumer; the wrong consumer, scope
    /// or admission, a stale cached context and unknown historical receipts are refused.
    #[test]
    fn handoff_reaches_only_the_intended_consumer_for_the_issued_scope() {
        let transforms = FrozenTransforms::synthetic_for_testing(1.0);
        let artifact = synthetic_artifact(&transforms);
        let pins = pins(&artifact);
        let verified = VerifiedCpuContext::for_testing(artifact.clone(), &pins);
        let context = verified.replay_assembly(&transforms).unwrap();
        let request = request(&pins, &context);

        // Positive control: the correct assembly reaches the accepted policy consumer.
        let handoff = verified
            .handoff(&context, ContextConsumer::AcceptedPolicyProposal)
            .unwrap();
        handoff.deliver_to_policy(&request).unwrap();
        assert_eq!(handoff.release().encoder, EncoderIdentity::accepted());
        assert_eq!(handoff.transformations(), transformation_chain().as_slice());
        assert_eq!(handoff.transformations().len(), 6);
        assert_eq!(handoff.scope().training, artifact.training);
        assert_eq!(handoff.scope().input_sha256, pins.input_sha256);
        assert_eq!(
            handoff.scope().prepared_rows_sha256,
            artifact.prepared_rows_sha256
        );
        assert_eq!(
            handoff.context_sha256(),
            crate::context_sha256(&artifact.vector)
        );
        assert_eq!(handoff.admission_pins_sha256(), pins.sha256().unwrap());
        let json = serde_json::to_value(&handoff).unwrap();
        assert_eq!(json["schema"], HANDOFF_SCHEMA);
        assert_eq!(json["consumer"], "accepted_policy_proposal");

        let refused = |handoff: &ContextHandoff, request: &Request, needle: &str| {
            let error = format!("{:#}", handoff.deliver_to_policy(request).unwrap_err());
            assert!(error.contains(needle), "{needle}: {error}");
        };
        // Wrong consumer.
        let teacher = verified
            .handoff(&context, ContextConsumer::TeacherAdmission)
            .unwrap();
        refused(&teacher, &request, "another consumer");
        // Admission the request did not issue (none, or other pins).
        refused(
            &handoff,
            &Request {
                context_admission_sha256: None,
                ..request.clone()
            },
            "not issued by this request's controller",
        );
        let mut other = pins.clone();
        other.qualification_sha256 = sha(b"other qualification");
        refused(
            &handoff,
            &Request {
                context_admission_sha256: Some(other.sha256().unwrap()),
                ..request.clone()
            },
            "not issued by this request's controller",
        );
        // Wrong scope: another cut, training table, dataset preparation or context.
        for wrong in [
            Request {
                cut: 1,
                ..request.clone()
            },
            Request {
                training_sha256: sha(b"other training"),
                ..request.clone()
            },
            Request {
                prepared_sha256: sha(b"other prepared"),
                ..request.clone()
            },
            Request {
                context_sha256: sha(b"other context"),
                ..request.clone()
            },
        ] {
            refused(&handoff, &wrong, "scope differs");
        }
        // A handoff is only made for this context's own replayed assembly.
        let other_context = test_support_context(&transforms, 2.0);
        assert!(verified
            .handoff(&other_context, ContextConsumer::AcceptedPolicyProposal)
            .is_err());

        // Stale cache: a context cached under other frozen transforms does not replay, and a
        // cached artifact from another encoder release is refused before any handoff.
        assert!(verified
            .replay_assembly(&FrozenTransforms::synthetic_for_testing(1.5))
            .is_err());
        let mut stale = artifact.clone();
        stale.encoder.component_bundle = worker::COMPONENT_BUNDLE_MANIFEST_SHA256.into();
        assert!(stale.validate().is_err());
        let stale_verified = VerifiedCpuContext::for_testing(stale, &pins);
        let stale_handoff = stale_verified
            .handoff(&context, ContextConsumer::AcceptedPolicyProposal)
            .unwrap();
        refused(&stale_handoff, &request, "encoder lineage changed");

        // Unknown historical receipts: unknown artifact schema/backend/qualification labels and
        // an unknown transformation chain or verification protocol are refused.
        for (field, value) in [
            ("schema", "jope.cpu-context/v0"),
            ("backend", "Rust/Candle 0.10 CPU FP32"),
            ("qualification", "qualified"),
        ] {
            let mut json = serde_json::to_value(&artifact).unwrap();
            json[field] = serde_json::json!(value);
            let historical: CpuContextArtifact = serde_json::from_value(json).unwrap();
            assert!(historical.validate().is_err(), "{field}");
        }
        let mut unknown_chain = handoff.clone();
        unknown_chain.transformations.pop();
        refused(&unknown_chain, &request, "transformation chain is unknown");
        let mut unknown_protocol = handoff.clone();
        unknown_protocol.release.verification_policy_sha256 = sha(b"older protocol");
        refused(&unknown_protocol, &request, "release protocol");
        let mut unknown_schema = handoff;
        unknown_schema.schema = "jope.cpu-context-handoff/v0".into();
        refused(&unknown_schema, &request, "another consumer");
    }

    fn test_support_context(transforms: &FrozenTransforms, value: f32) -> FinalContext {
        transforms
            .assemble(&super::super::test_support::components(
                vec![value; 512],
                vec![value; 2048],
                vec![value; 512],
            ))
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_component_gate_refuses_wrong_values_shapes_and_nan() {
        assert!(max_abs(&[0.0, 1.0], &[0.0, 1.0]).is_ok());
        assert!(max_abs(&[0.0], &[0.0, 1.0]).is_err());
        assert!(max_abs(&[f32::NAN], &[0.0]).is_err());
        assert!(max_abs(&[0.001], &[0.0]).is_err());
    }
    #[test]
    fn cpu_manifest_never_accepts_unpinned_registry_or_fewer_cases() {
        let m = ReferenceManifest {
            schema: "jope.cpu-context-reference/v1".into(),
            verification_policy_sha256: verification_policy_sha256(),
            registry_bytes: b"{}".to_vec(),
            index_bytes: b"{}".to_vec(),
            cases: vec![],
        };
        assert!(read_manifest(&serde_json::to_vec(&m).unwrap()).is_err());
    }
    #[test]
    fn execution_requires_completion_limits_and_matching_outputs() {
        let mut e = ExecutionAttestation {
            schema: "jope.cpu-context-execution/controller-v1".into(),
            supervisor_receipt_sha256: "a".repeat(64),
            program_sha256: "b".repeat(64),
            input_sha256: "c".repeat(64),
            output_sha256: "d".repeat(64),
            completed: true,
            cpu_limit: 1,
            memory_max: 1 << 30,
            swap_max: 0,
            threads: 1,
            elapsed_seconds: 1.0,
            deadline_seconds: 2.0,
        };
        let check = |e: &ExecutionAttestation| {
            verify_execution(
                &serde_json::to_vec(e).unwrap(),
                &"b".repeat(64),
                &"c".repeat(64),
                &"d".repeat(64),
                30.0,
            )
        };
        assert!(check(&e).is_ok());
        e.swap_max = 1;
        assert!(check(&e).is_err());
        e.swap_max = 0;
        e.completed = false;
        assert!(check(&e).is_err());
        e.completed = true;
        e.elapsed_seconds = 3.0;
        assert!(check(&e).is_err());
    }
}
