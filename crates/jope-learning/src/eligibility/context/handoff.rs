use super::*;

/// The W2 handoff schema teacher admission consumes
/// (`jope_policy::context_producer::HANDOFF_SCHEMA`).
pub const HANDOFF_SCHEMA: &str = "jope.cpu-context-handoff/v1";
/// The W2 consumer a teaching context must be handed off to (`ContextConsumer::TeacherAdmission`,
/// as it serializes).
pub const TEACHER_ADMISSION: &str = "teacher_admission";

/// The W2 encoder closure (`jope_policy::context_producer::EncoderIdentity`), field for field.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffEncoder {
    /// Hyperion weights recorded for this handoff encoder.
    pub hyperion_weights: String,
    /// Hyperion manifest recorded for this handoff encoder.
    pub hyperion_manifest: String,
    /// Component bundle recorded for this handoff encoder.
    pub component_bundle: String,
    /// Foundation weights recorded for this handoff encoder.
    pub foundation_weights: String,
    /// Target weights recorded for this handoff encoder.
    pub target_weights: String,
    /// Frozen transforms recorded for this handoff encoder.
    pub frozen_transforms: String,
    /// Assembly recorded for this handoff encoder.
    pub assembly: String,
    /// Recipe recorded for this handoff encoder.
    pub recipe: String,
}
impl HandoffEncoder {
    fn digests(&self) -> [&str; 8] {
        [
            &self.hyperion_weights,
            &self.hyperion_manifest,
            &self.component_bundle,
            &self.foundation_weights,
            &self.target_weights,
            &self.frozen_transforms,
            &self.assembly,
            &self.recipe,
        ]
    }
}

/// The W2 release identity of a context producer, exactly as a `jope.cpu-context-handoff/v1`
/// record serializes its `release` (`jope_policy::context_producer::ContextRelease`): the encoder
/// closure, the producer's source commit and executable, the verification protocol and the pinned
/// native reference manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffRelease {
    /// Encoder recorded for this handoff release.
    pub encoder: HandoffEncoder,
    /// SHA-256 digest identifying the associated source.
    pub source_sha256: String,
    /// SHA-256 digest identifying the associated producer.
    pub producer_sha256: String,
    /// SHA-256 digest identifying the associated verification policy.
    pub verification_policy_sha256: String,
    /// SHA-256 digest identifying the associated reference manifest.
    pub reference_manifest_sha256: String,
}

/// The W2 context producer the controller preregisters with the freeze, before any candidate or
/// context exists: the release identity a teaching context must be handed off under, the typed
/// transformation chain and the W2 qualification the producer passed. A submission never supplies
/// these; the handoff it carries is checked against them (producer authentication).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerPlan {
    /// Release recorded for this producer plan.
    pub release: HandoffRelease,
    /// `jope_policy::context_producer::transformation_chain()`, in execution order.
    pub transformations: Vec<String>,
    /// SHA-256 digest identifying the associated qualification.
    pub qualification_sha256: String,
    /// The controller's admission pins fixed when the producer registered: its attestation of
    /// each W2 qualification case execution, in reference-manifest order
    /// (`jope_policy::context_producer::AdmissionPins::qualification_execution_sha256`, as
    /// `ProducerRegistration` records them). Teacher admission rebuilds a handoff's admission
    /// pins from these, the release and qualification above, and the run's own artifact, input
    /// scope and retained execution. Absent on a registration made before the pins were
    /// recorded: that freeze keeps its bytes, and no context under it can be authenticated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qualification_execution_sha256: Option<Vec<String>>,
}
impl ProducerPlan {
    pub(crate) fn validate(&self) -> Result<()> {
        let r = &self.release;
        ensure!(
            r.encoder.digests().iter().all(|d| is_hex64(d))
                && r.source_sha256.len() == 40
                && r.source_sha256
                    .bytes()
                    .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
                && is_hex64(&r.producer_sha256)
                && is_hex64(&r.verification_policy_sha256)
                && is_hex64(&r.reference_manifest_sha256)
                && is_hex64(&self.qualification_sha256)
                && !self.transformations.is_empty()
                && self.transformations.iter().all(|t| !t.trim().is_empty())
                && self
                    .qualification_execution_sha256
                    .as_ref()
                    .is_none_or(|pins| {
                        pins.len() == QUALIFICATION_CASES && pins.iter().all(|p| is_hex64(p))
                    }),
            "invalid preregistered W2 context producer"
        );
        Ok(())
    }
}

/// The W2 execution that produced a handoff's artifact, as retained pool receipts, each by
/// content address.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextExecution {
    /// `jope_pipeline::execution::WorkerJob`.
    pub job_sha256: String,
    /// `jope_pipeline::execution::WorkerOutcome`: the supervisor receipt.
    pub outcome_sha256: String,
    /// The context job's `jope_pipeline::experiment::Experiment` and its original
    /// `ExecutionWindow`.
    pub experiment_sha256: String,
    /// SHA-256 digest identifying the associated window.
    pub window_sha256: String,
}

/// The W2 handoff a submission carries with its teaching context: the retained
/// `jope.cpu-context-handoff/v1` record, as `ContextHandoff` serializes it, and the execution
/// that produced its artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffEvidence {
    /// SHA-256 digest identifying the associated handoff.
    pub handoff_sha256: String,
    /// Execution recorded for this handoff evidence.
    pub execution: ContextExecution,
}

/// A retained `jope.cpu-context-handoff/v1` record, field for field as `ContextHandoff`
/// serializes; anything else is refused.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HandoffRecord {
    pub(crate) schema: String,
    pub(crate) consumer: String,
    pub(crate) release: HandoffRelease,
    pub(crate) transformations: Vec<String>,
    pub(crate) scope: HandoffScope,
    pub(crate) artifact_sha256: String,
    pub(crate) qualification_sha256: String,
    pub(crate) admission_pins_sha256: String,
    pub(crate) context_sha256: String,
}

pub(crate) fn parse_handoff(bytes: &[u8]) -> Result<HandoffRecord> {
    serde_json::from_slice(bytes).context(
        "ContextHandoffMismatch: the retained handoff is not a jope.cpu-context-handoff/v1 record",
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HandoffScope {
    pub(crate) training: HandoffTraining,
    pub(crate) input_sha256: String,
    pub(crate) preprocessing_sha256: String,
    pub(crate) prepared_rows_sha256: String,
}

/// W2's training scope (`jope_policy::context_producer::TrainingIdentity`, field for field in its
/// serialization order). Its row digests are over W2's own prepared-row encoding, which is not
/// the W3 receipt's row encoding, so only their form is checked here; the dataset is joined.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HandoffTraining {
    pub(crate) dataset_sha256: String,
    pub(crate) prepared_sha256: String,
    pub(crate) training_sha256: String,
    pub(crate) fitting_rows_sha256: String,
    pub(crate) cut: u8,
}

/// The W2 context artifact (`jope_policy::context_producer::{SCHEMA, BACKEND}`).
pub(crate) const PRODUCED_ARTIFACT_SCHEMA: &str = "jope.cpu-context/v1";
pub(crate) const PRODUCED_ARTIFACT_BACKEND: &str = "Rust/Candle 0.11 CPU FP32; frozen accepted \
                                                    transforms; native-component qualification \
                                                    required";
/// `FinalContext::WIDTH` and the frozen clip `CpuContextArtifact::validate` holds it to.
pub(crate) const CONTEXT_WIDTH: usize = 2048;
pub(crate) const CONTEXT_CLIP: f32 = 32.0;
/// `verify_qualified`'s bound on the artifact bytes.
pub(crate) const MAX_ARTIFACT_BYTES: usize = 16 << 20;
/// The W2 verification policy's eight native meta-train qualification cases.
const QUALIFICATION_CASES: usize = 8;
/// The controller's execution attestation (`jope_pipeline::cpu_context_bridge`).
pub(crate) const EXECUTION_ATTESTATION_SCHEMA: &str = "jope.cpu-context-execution/controller-v1";

/// The artifact a W2 producer job writes (`jope_policy::context_producer::CpuContextArtifact`),
/// field for field; anything else is refused. The encoder products and the train-only
/// preprocessing are carried, not interpreted: teacher admission consumes the final context.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProducedArtifact {
    pub(crate) schema: String,
    pub(crate) backend: String,
    pub(crate) qualification: String,
    pub(crate) source_sha256: String,
    pub(crate) producer_sha256: String,
    pub(crate) input_sha256: String,
    pub(crate) training: HandoffTraining,
    pub(crate) selected_training_rows_sha256: String,
    pub(crate) original_labels_sha256: String,
    pub(crate) preprocessing_sha256: String,
    pub(crate) prepared_rows_sha256: String,
    pub(crate) preprocessing: serde_json::Value,
    pub(crate) encoder: HandoffEncoder,
    pub(crate) components: serde_json::Value,
    pub(crate) vector: Vec<f32>,
    pub(crate) final_test_targets_used: bool,
}

/// `jope_policy::context_producer::AdmissionPins`, field for field in its serialization order:
/// the sha256 of these bytes is the handoff's `admission_pins_sha256` (`AdmissionPins::sha256`).
#[derive(Serialize)]
pub(crate) struct AdmissionPinsRecord<'a> {
    pub(crate) source_sha256: &'a str,
    pub(crate) producer_sha256: &'a str,
    pub(crate) artifact_sha256: &'a str,
    pub(crate) qualification_sha256: &'a str,
    pub(crate) reference_manifest_sha256: &'a str,
    pub(crate) context_execution_sha256: &'a str,
    pub(crate) qualification_execution_sha256: &'a [String],
    pub(crate) verification_policy_sha256: &'a str,
    pub(crate) training: &'a HandoffTraining,
    pub(crate) input_sha256: &'a str,
    pub(crate) preprocessing_sha256: &'a str,
}

/// `jope_policy::context_producer::ExecutionAttestation`, field for field in its serialization
/// order, with the values the controller derives from the context job's pool receipts
/// (`jope_pipeline::cpu_context_bridge::ExecutionProof::attest`): the admission pins'
/// `context_execution_sha256` is the sha256 of these bytes.
#[derive(Serialize)]
pub(crate) struct ExecutionAttestationRecord<'a> {
    pub(crate) schema: &'a str,
    pub(crate) supervisor_receipt_sha256: &'a str,
    pub(crate) program_sha256: &'a str,
    pub(crate) input_sha256: &'a str,
    pub(crate) output_sha256: &'a str,
    pub(crate) completed: bool,
    pub(crate) cpu_limit: u32,
    pub(crate) memory_max: u64,
    pub(crate) swap_max: u64,
    pub(crate) threads: u32,
    pub(crate) elapsed_seconds: f64,
    pub(crate) deadline_seconds: f64,
}
