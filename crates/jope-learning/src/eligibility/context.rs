//! Mandatory teaching-context provenance. The receiver derives the encoder scope and
//! pins from the preoptimization freeze and verifies retained bytes with
//! [`check_teaching_binding`]; a submission supplies only content-addressed evidence,
//! never its own scope, encoder, consumer or product. Missing provenance never passes.
//!
//! W3-T02 / W2-T07: the context must also have been handed off to teacher admission by the W2
//! context producer the freeze preregistered, through a retained `jope.cpu-context-handoff/v1`
//! record, and the execution that produced its artifact must be proven from retained pool
//! receipts ([`verify_handoff`]). The encoder receipt's own claims are not enough, and neither
//! are the handoff's: the consumed context is decoded from the producer's artifact, and the
//! handoff's admission pins are rebuilt from the preregistration, that artifact and the retained
//! execution (PR #114 review round 1).
use super::*;
use crate::encoder_receipt::{
    check_teaching_binding, Encoder, EncoderCapability, EncoderReceipt, Inputs, Member, Product,
    Scope, TeachingBinding, SCHEMA as RECEIPT_SCHEMA,
};
use crate::revocation::Revocations;

const MAX_PARENT_DEPTH: usize = 8;
const MAX_PARENTS: usize = 64;

/// Global use must prove every retained context receipt, state, output, and handoff separately.
/// Existing producer/handoff/scope checks are performed by historical revalidation first.
pub(super) fn verify_global_exposure(
    state: &State,
    qualification: &Qualification,
    verifier: &crate::replay::admission::AdmissionVerifier<'_>,
    purpose: crate::replay::admission::Purpose,
    proofs: &BTreeMap<String, String>,
    at_ms: u64,
) -> Result<()> {
    verifier.require_state(state)?;
    let root = qualification
        .context_receipt_sha256
        .as_ref()
        .context("HELD: missing global context provenance")?;
    let handoff = qualification
        .context_handoff_sha256
        .as_ref()
        .context("HELD: missing global context handoff")?;
    let mut pending = vec![root.clone()];
    let mut seen = BTreeSet::new();
    while let Some(sha) = pending.pop() {
        if !seen.insert(sha.clone()) {
            continue;
        }
        ensure!(
            seen.len() <= MAX_PARENTS,
            "HELD: context ancestry exceeds its bound"
        );
        let context = receipt(state, &sha)?;
        for subject in [
            &sha,
            &context.fitted_state_sha256,
            &context.output_sha256,
            handoff,
        ] {
            let proof = proofs
                .get(subject)
                .context("HELD: context lacks global exposure proof")?;
            verifier.admit(subject, proof, purpose, None, at_ms)?;
        }
        pending.extend(context.inputs.parent_receipts);
    }
    Ok(())
}

/// Preregistered with the freeze, before any candidate or context exists.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextPlan {
    /// Source family recorded for this context plan.
    pub source_family: String,
    /// Encoder recorded for this context plan.
    pub encoder: Encoder,
    /// The W2 context producer a teaching context must be handed off by (W3-T02, W2-T07).
    /// Absent on freezes made before producer authentication existed: such a freeze keeps its
    /// bytes and identity, but no context under it can qualify a teacher (it stays quarantined;
    /// owner decision D4, 2026-09-26).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer: Option<ProducerPlan>,
}
impl ContextPlan {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            !self.source_family.trim().is_empty()
                && self.encoder.product == Product::TeachingContext2048
                && is_hex64(&self.encoder.artifact_sha256)
                && is_hex64(&self.encoder.configuration_sha256),
            "invalid preregistered teaching context encoder"
        );
        if let Some(producer) = &self.producer {
            producer.validate()?;
        }
        Ok(())
    }
}

/// Retained context for one submission: canonical `jope.encoder_receipt/v1` bytes, the
/// fitted transform state and the typed output vector, each by content address.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextEvidence {
    /// SHA-256 digest of the receipt that authenticates this result.
    pub receipt_sha256: String,
    /// SHA-256 digest identifying the associated fitted state.
    pub fitted_state_sha256: String,
    /// SHA-256 digest identifying the associated output.
    pub output_sha256: String,
    /// The W2 handoff of this context to teacher admission and the execution that produced it.
    /// Required for qualification; absent on submissions made before W3-T02, which stay
    /// quarantined.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handoff: Option<HandoffEvidence>,
}

/// The comparison's own fitting rows; inner-partition outer-held rows are forbidden.
pub(super) fn scope(state: &State, policy: &FrozenPolicy, plan: &ContextPlan) -> Result<Scope> {
    let members = |rows: &[String], groups: &[String]| -> Result<Vec<Member>> {
        ensure!(rows.len() == groups.len(), "context scope groups changed");
        Ok(rows
            .iter()
            .zip(groups)
            .map(|(row, group)| Member {
                row: row.clone(),
                group: group.clone(),
            })
            .collect())
    };
    let (outer_fold, fitting, forbidden) = match &policy.inner_partition {
        Some(p) => (
            Some(p.outer_fold),
            members(&p.fitting_rows, &p.fitting_groups)?,
            members(&p.hold_rows, &p.hold_groups)?,
        ),
        None => {
            let rows: Vec<TargetRow> = blob(state, &policy.targets_sha256)?;
            let (ids, groups): (Vec<_>, Vec<_>) =
                rows.into_iter().map(|row| (row.id, row.group)).unzip();
            (None, members(&ids, &groups)?, vec![])
        }
    };
    Ok(Scope {
        comparison: policy.comparison.clone(),
        source_family: plan.source_family.clone(),
        outer_fold,
        inner_fold: None,
        fitting,
        forbidden,
    })
}

fn receipt(state: &State, sha: &str) -> Result<EncoderReceipt> {
    let receipt: EncoderReceipt = blob(state, sha)?;
    ensure!(
        receipt.schema == RECEIPT_SCHEMA && canonical_sha256(&receipt)? == sha,
        "EncoderScopeMismatch: receipt is not canonical {RECEIPT_SCHEMA} content"
    );
    Ok(receipt)
}

fn retained(state: &State, sha: &str) -> Result<Vec<u8>> {
    ensure!(is_hex64(sha), "invalid context artifact identity");
    state
        .read_blob(sha)?
        .context("retained context artifact is missing")
}

/// Every parent is a retained receipt fitted in the same scope, with retained bytes, and none of
/// it is revoked evidence.
fn parents(state: &State, child: &EncoderReceipt, revocations: &Revocations) -> Result<()> {
    let mut pending: Vec<_> = child
        .inputs
        .parent_receipts
        .iter()
        .map(|sha| (sha.clone(), 1))
        .collect();
    let mut seen = BTreeSet::new();
    while let Some((sha, depth)) = pending.pop() {
        ensure!(
            depth <= MAX_PARENT_DEPTH && seen.len() < MAX_PARENTS,
            "context parent closure exceeds its bound"
        );
        if !seen.insert(sha.clone()) {
            continue;
        }
        revocations.refuse("a teaching context parent receipt", [sha.as_str()])?;
        let parent = receipt(state, &sha).context("unresolved context parent receipt")?;
        revocations.refuse(
            "a teaching context parent",
            [
                parent.fitted_state_sha256.as_str(),
                parent.output_sha256.as_str(),
            ],
        )?;
        ensure!(
            parent.scope_sha256 == child.scope_sha256
                && parent.comparison_sha256 == child.comparison_sha256
                && parent.ordered_fitting_rows_sha256 == child.ordered_fitting_rows_sha256,
            "EncoderScopeMismatch: parent was fitted outside the admitted scope"
        );
        retained(state, &parent.fitted_state_sha256)?;
        retained(state, &parent.output_sha256)?;
        pending.extend(
            parent
                .inputs
                .parent_receipts
                .iter()
                .map(|p| (p.clone(), depth + 1)),
        );
    }
    Ok(())
}

/// The teacher's authorized execution: the experiment and original window its OOF evidence names
/// (`OofEvidence::experiment_blob_sha256`, `window_sha256`), which `qualify` holds to the
/// preoptimization freeze. A teaching context must have been produced inside exactly this
/// execution (PR #114 review round 1).
pub(super) struct Teaching<'a> {
    pub(super) experiment: &'a Experiment,
    pub(super) window: &'a ExecutionWindow,
}

/// Runs before any teaching decision; returns the verified receipt digest and the digest of the
/// W2 handoff it was admitted under.
pub(super) fn verify(
    state: &State,
    policy: &FrozenPolicy,
    record: &HwmRecord,
    evidence: &OofEvidence,
    revocations: &Revocations,
    teaching: &Teaching<'_>,
) -> Result<(String, String)> {
    let plan = preregistered_plan(policy)?;
    let context = evidence.context.as_ref().with_context(|| {
        quarantined("MissingEncoderProvenance: submission carries no context receipt")
    })?;
    revocations.refuse(
        "the teaching context",
        [
            context.receipt_sha256.as_str(),
            context.fitted_state_sha256.as_str(),
            context.output_sha256.as_str(),
        ],
    )?;
    let receipt = receipt(state, &context.receipt_sha256)?;
    let capability = EncoderCapability::bind(
        scope(state, policy, plan)?,
        plan.encoder.clone(),
        receipt.inputs.clone(),
    )?;
    check_teaching_binding(
        &capability,
        Some(&receipt),
        &TeachingBinding {
            comparison_sha256: record.identity_sha256.clone(),
            program_sha256: record.genome.program_sha256.clone(),
            encoder_receipt_sha256: context.receipt_sha256.clone(),
            product: plan.encoder.product,
        },
        &record.genome.program_sha256,
        &retained(state, &context.fitted_state_sha256)?,
        &retained(state, &context.output_sha256)?,
    )?;
    parents(state, &receipt, revocations)?;
    let handoff = verify_handoff(state, policy, plan, record, context, revocations, teaching)?;
    Ok((context.receipt_sha256.clone(), handoff))
}

/// The teaching context encoder the freeze preregistered; without one no context can teach.
fn preregistered_plan(policy: &FrozenPolicy) -> Result<&ContextPlan> {
    let plan = policy.context.as_ref().with_context(|| {
        quarantined("MissingEncoderProvenance: freeze preregistered no teaching context encoder")
    })?;
    plan.validate()?;
    Ok(plan)
}

/// The W2 producer the freeze preregistered with its context encoder (W3-T02, W2-T07), and the
/// case-execution admission pins its registration recorded.
fn preregistered_producer(plan: &ContextPlan) -> Result<(&ProducerPlan, &[String])> {
    let producer = plan.producer.as_ref().with_context(|| {
        quarantined(
            "PreHandoffProvenance: the freeze preregistered no W2 context producer (release \
             identity, transformation chain and W2 qualification), so no teaching context under \
             it can be authenticated",
        )
    })?;
    producer.validate()?;
    let pins = producer
        .qualification_execution_sha256
        .as_deref()
        .with_context(|| {
            quarantined(
                "PreHandoffProvenance: the freeze's W2 producer registration records no \
                 controller admission pins (the W2 qualification's case-execution pins), so no \
                 handoff under it can be authenticated",
            )
        })?;
    Ok((producer, pins))
}

/// The W2 producer run behind one teaching context, as the exact bytes teacher admission reads
/// (W3-T02): the `jope.cpu-context-handoff/v1` record the run emitted for teacher admission; the
/// pool receipts of the context job (the job, its supervisor outcome, and the experiment and
/// original window it ran in); the context artifact the job wrote; and the context's
/// little-endian f32 bytes.
#[derive(Clone, Debug)]
pub struct ContextRun {
    /// Collection of handoff values carried by this context run.
    pub handoff: Vec<u8>,
    /// Collection of job values carried by this context run.
    pub job: Vec<u8>,
    /// Collection of outcome values carried by this context run.
    pub outcome: Vec<u8>,
    /// Collection of experiment values carried by this context run.
    pub experiment: Vec<u8>,
    /// Collection of window values carried by this context run.
    pub window: Vec<u8>,
    /// Collection of artifact values carried by this context run.
    pub artifact: Vec<u8>,
    /// Collection of context values carried by this context run.
    pub context: Vec<u8>,
}

/// The controller side of W3-T02: retain a W2 producer run where teacher admission looks for it
/// (the teaching state's content-addressed blobs) and issue the teaching context's
/// `jope.encoder_receipt/v1` under the encoder and scope the freeze preregistered. The receipt's
/// fitted state is the context artifact the job wrote, its input closure is the handoff's own
/// input (the W2 training input and its training rows), and it has no parents.
///
/// This decides nothing and changes no authority state. Teacher admission (`qualify` and every
/// revalidating route) authenticates the handoff against the preregistered producer and
/// re-verifies the execution from the retained receipts. Refused before anything is retained when
/// the freeze preregistered no context encoder or no W2 producer, or a producer registration
/// without its admission pins (the context stays quarantined), or the handoff is not a
/// `jope.cpu-context-handoff/v1` record.
pub fn retain_handed_off_context(
    state: &State,
    freeze: &FreezeReceipt,
    run: &ContextRun,
) -> Result<ContextEvidence> {
    let policy = load_freeze(state, freeze)?;
    let plan = preregistered_plan(&policy)?;
    preregistered_producer(plan)?;
    let handoff = parse_handoff(&run.handoff)?;
    let capability = EncoderCapability::bind(
        scope(state, &policy, plan)?,
        plan.encoder.clone(),
        Inputs {
            features_sha256: handoff.scope.input_sha256,
            labels_sha256: Some(handoff.scope.training.training_sha256),
            parent_receipts: vec![],
        },
    )?;
    let receipt = capability.receipt(&run.artifact, &run.context)?;
    let put = |bytes: &[u8]| state.put_blob(bytes);
    Ok(ContextEvidence {
        receipt_sha256: put(&canonical_json(&receipt)?)?,
        fitted_state_sha256: put(&run.artifact)?,
        output_sha256: put(&run.context)?,
        handoff: Some(HandoffEvidence {
            handoff_sha256: put(&run.handoff)?,
            execution: ContextExecution {
                job_sha256: put(&run.job)?,
                outcome_sha256: put(&run.outcome)?,
                experiment_sha256: put(&run.experiment)?,
                window_sha256: put(&run.window)?,
            },
        }),
    })
}

// ------------------------------------------------------------------------------------------
// W2 producer authentication and the context handoff (W3-T02, W2-T07)
// ------------------------------------------------------------------------------------------

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
    fn validate(&self) -> Result<()> {
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
struct HandoffRecord {
    schema: String,
    consumer: String,
    release: HandoffRelease,
    transformations: Vec<String>,
    scope: HandoffScope,
    artifact_sha256: String,
    qualification_sha256: String,
    admission_pins_sha256: String,
    context_sha256: String,
}

fn parse_handoff(bytes: &[u8]) -> Result<HandoffRecord> {
    serde_json::from_slice(bytes).context(
        "ContextHandoffMismatch: the retained handoff is not a jope.cpu-context-handoff/v1 record",
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HandoffScope {
    training: HandoffTraining,
    input_sha256: String,
    preprocessing_sha256: String,
    prepared_rows_sha256: String,
}

/// W2's training scope (`jope_policy::context_producer::TrainingIdentity`, field for field in its
/// serialization order). Its row digests are over W2's own prepared-row encoding, which is not
/// the W3 receipt's row encoding, so only their form is checked here; the dataset is joined.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HandoffTraining {
    pub(super) dataset_sha256: String,
    pub(super) prepared_sha256: String,
    pub(super) training_sha256: String,
    pub(super) fitting_rows_sha256: String,
    pub(super) cut: u8,
}

/// The W2 context artifact (`jope_policy::context_producer::{SCHEMA, BACKEND}`).
pub(super) const PRODUCED_ARTIFACT_SCHEMA: &str = "jope.cpu-context/v1";
pub(super) const PRODUCED_ARTIFACT_BACKEND: &str = "Rust/Candle 0.11 CPU FP32; frozen accepted \
                                                    transforms; native-component qualification \
                                                    required";
/// `FinalContext::WIDTH` and the frozen clip `CpuContextArtifact::validate` holds it to.
const CONTEXT_WIDTH: usize = 2048;
const CONTEXT_CLIP: f32 = 32.0;
/// `verify_qualified`'s bound on the artifact bytes.
const MAX_ARTIFACT_BYTES: usize = 16 << 20;
/// The W2 verification policy's eight native meta-train qualification cases.
const QUALIFICATION_CASES: usize = 8;
/// The controller's execution attestation (`jope_pipeline::cpu_context_bridge`).
pub(super) const EXECUTION_ATTESTATION_SCHEMA: &str = "jope.cpu-context-execution/controller-v1";

/// The artifact a W2 producer job writes (`jope_policy::context_producer::CpuContextArtifact`),
/// field for field; anything else is refused. The encoder products and the train-only
/// preprocessing are carried, not interpreted: teacher admission consumes the final context.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProducedArtifact {
    pub(super) schema: String,
    pub(super) backend: String,
    pub(super) qualification: String,
    pub(super) source_sha256: String,
    pub(super) producer_sha256: String,
    pub(super) input_sha256: String,
    pub(super) training: HandoffTraining,
    pub(super) selected_training_rows_sha256: String,
    pub(super) original_labels_sha256: String,
    pub(super) preprocessing_sha256: String,
    pub(super) prepared_rows_sha256: String,
    pub(super) preprocessing: serde_json::Value,
    pub(super) encoder: HandoffEncoder,
    pub(super) components: serde_json::Value,
    pub(super) vector: Vec<f32>,
    pub(super) final_test_targets_used: bool,
}

/// `jope_policy::context_producer::AdmissionPins`, field for field in its serialization order:
/// the sha256 of these bytes is the handoff's `admission_pins_sha256` (`AdmissionPins::sha256`).
#[derive(Serialize)]
pub(super) struct AdmissionPinsRecord<'a> {
    pub(super) source_sha256: &'a str,
    pub(super) producer_sha256: &'a str,
    pub(super) artifact_sha256: &'a str,
    pub(super) qualification_sha256: &'a str,
    pub(super) reference_manifest_sha256: &'a str,
    pub(super) context_execution_sha256: &'a str,
    pub(super) qualification_execution_sha256: &'a [String],
    pub(super) verification_policy_sha256: &'a str,
    pub(super) training: &'a HandoffTraining,
    pub(super) input_sha256: &'a str,
    pub(super) preprocessing_sha256: &'a str,
}

/// `jope_policy::context_producer::ExecutionAttestation`, field for field in its serialization
/// order, with the values the controller derives from the context job's pool receipts
/// (`jope_pipeline::cpu_context_bridge::ExecutionProof::attest`): the admission pins'
/// `context_execution_sha256` is the sha256 of these bytes.
#[derive(Serialize)]
pub(super) struct ExecutionAttestationRecord<'a> {
    pub(super) schema: &'a str,
    pub(super) supervisor_receipt_sha256: &'a str,
    pub(super) program_sha256: &'a str,
    pub(super) input_sha256: &'a str,
    pub(super) output_sha256: &'a str,
    pub(super) completed: bool,
    pub(super) cpu_limit: u32,
    pub(super) memory_max: u64,
    pub(super) swap_max: u64,
    pub(super) threads: u32,
    pub(super) elapsed_seconds: f64,
    pub(super) deadline_seconds: f64,
}

/// The context the W2 producer job wrote, decoded as the producer decodes it
/// (`jope_policy::context_producer::verify_qualified`): the retained, digest-verified artifact
/// bytes parsed as `CpuContextArtifact`, held to the envelope and context layout
/// `CpuContextArtifact::validate` checks, and produced under the handoff's release and input
/// scope. Returns the context's little-endian f32 bytes.
fn produced_context(state: &State, handoff: &HandoffRecord) -> Result<Vec<u8>> {
    let bytes = retained(state, &handoff.artifact_sha256)?;
    ensure!(
        bytes.len() <= MAX_ARTIFACT_BYTES,
        "ContextHandoffMismatch: the producer artifact exceeds its bound"
    );
    let artifact: ProducedArtifact = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "ContextHandoffMismatch: the producer artifact {} is not a {PRODUCED_ARTIFACT_SCHEMA} \
             context artifact",
            handoff.artifact_sha256
        )
    })?;
    ensure!(
        artifact.schema == PRODUCED_ARTIFACT_SCHEMA
            && artifact.backend == PRODUCED_ARTIFACT_BACKEND
            && artifact.qualification == "unqualified"
            && !artifact.final_test_targets_used,
        "ContextHandoffMismatch: the producer artifact is not the W2 CPU context envelope"
    );
    let (release, scope) = (&handoff.release, &handoff.scope);
    ensure!(
        artifact.source_sha256 == release.source_sha256
            && artifact.producer_sha256 == release.producer_sha256
            && artifact.encoder == release.encoder
            && artifact.input_sha256 == scope.input_sha256
            && artifact.training == scope.training
            && artifact.selected_training_rows_sha256 == scope.training.fitting_rows_sha256
            && artifact.preprocessing_sha256 == scope.preprocessing_sha256
            && artifact.prepared_rows_sha256 == scope.prepared_rows_sha256,
        "ContextHandoffMismatch: the producer artifact was produced under another release or \
         input scope than the handoff names"
    );
    ensure!(
        artifact.vector.len() == CONTEXT_WIDTH
            && artifact
                .vector
                .iter()
                .all(|v| v.is_finite() && v.abs() <= CONTEXT_CLIP),
        "ContextHandoffMismatch: the producer artifact's context is not a finite, clipped \
         {CONTEXT_WIDTH}-f32 vector"
    );
    Ok(artifact
        .vector
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect())
}

/// Teacher admission consumes the W2 handoff instead of trusting the receipt's own claims. The
/// handoff must:
/// - exist, under a freeze that preregistered a W2 producer (otherwise the teaching context is
///   quarantined, never upgraded in place);
/// - be handed off to teacher admission, under exactly the preregistered release identity,
///   transformation chain and W2 qualification (producer authentication);
/// - hand off exactly the teacher's context bytes (`context_sha256` is the sha256 of the
///   receipt's retained output, the context's little-endian bytes), scoped to the comparison's
///   dataset;
/// - name no revoked evidence (W12-T05);
/// - have its artifact produced by a completed execution of the preregistered producer
///   executable inside the teacher's own authorized experiment and window, proven from retained
///   pool receipts ([`verify_execution`]);
/// - hand off exactly the context that artifact contains, decoded as the W2 producer decodes it
///   ([`produced_context`]): the consumed vector is the artifact's vector byte for byte, and
///   `context_sha256` is its digest;
/// - name the controller's admission pins for this run: `admission_pins_sha256` is rebuilt from
///   the preregistered release, W2 qualification and case-execution pins, the artifact, its input
///   scope and the controller's attestation of the retained context execution.
///
/// Returns the handoff digest the qualification records.
fn verify_handoff(
    state: &State,
    policy: &FrozenPolicy,
    plan: &ContextPlan,
    record: &HwmRecord,
    context: &ContextEvidence,
    revocations: &Revocations,
    teaching: &Teaching<'_>,
) -> Result<String> {
    let (producer, case_execution_pins) = preregistered_producer(plan)?;
    let evidence = context.handoff.as_ref().with_context(|| {
        quarantined(
            "MissingContextHandoff: the submission carries no W2 context handoff to teacher \
             admission; a self-declared encoder receipt is not a produced context",
        )
    })?;
    let ex = &evidence.execution;
    revocations.refuse(
        "the W2 context handoff",
        [
            evidence.handoff_sha256.as_str(),
            ex.job_sha256.as_str(),
            ex.outcome_sha256.as_str(),
            ex.experiment_sha256.as_str(),
            ex.window_sha256.as_str(),
        ],
    )?;
    let handoff = parse_handoff(&retained(state, &evidence.handoff_sha256)?)?;
    ensure!(
        handoff.schema == HANDOFF_SCHEMA,
        "ContextHandoffMismatch: handoff schema {:?}",
        handoff.schema
    );
    ensure!(
        handoff.consumer == TEACHER_ADMISSION,
        "ContextHandoffMismatch: the context was handed off for consumer {:?}, not \
         {TEACHER_ADMISSION}",
        handoff.consumer
    );
    ensure!(
        handoff.release == producer.release,
        "ProducerNotAuthenticated: the handoff's release identity (encoder closure, source, \
         producer executable, verification policy, reference manifest) is not the one the freeze \
         preregistered"
    );
    ensure!(
        handoff.qualification_sha256 == producer.qualification_sha256,
        "ProducerNotAuthenticated: the handoff names W2 qualification {}, not the preregistered {}",
        handoff.qualification_sha256,
        producer.qualification_sha256
    );
    ensure!(
        handoff.transformations == producer.transformations,
        "ContextHandoffMismatch: the handoff's transformation chain is not the preregistered chain"
    );
    ensure!(
        handoff.context_sha256 == context.output_sha256,
        "ContextHandoffMismatch: the teacher's context {} is not the handed-off context {}",
        context.output_sha256,
        handoff.context_sha256
    );
    let t = &handoff.scope.training;
    ensure!(
        t.dataset_sha256 == record.identity.dataset_hash,
        "ContextHandoffMismatch: the handoff's input scope is dataset {}, not the comparison's {}",
        t.dataset_sha256,
        record.identity.dataset_hash
    );
    for digest in [
        &t.prepared_sha256,
        &t.training_sha256,
        &t.fitting_rows_sha256,
        &handoff.scope.input_sha256,
        &handoff.scope.preprocessing_sha256,
        &handoff.scope.prepared_rows_sha256,
        &handoff.artifact_sha256,
        &handoff.admission_pins_sha256,
    ] {
        ensure!(
            is_hex64(digest),
            "ContextHandoffMismatch: invalid handoff digest"
        );
    }
    revocations.refuse(
        "the W2 context handoff",
        [
            handoff.artifact_sha256.as_str(),
            handoff.qualification_sha256.as_str(),
        ],
    )?;
    let context_execution_sha256 = verify_execution(state, policy, &handoff, ex, record, teaching)?;
    let produced = produced_context(state, &handoff)?;
    let produced_sha256 = sha256_hex(&produced);
    ensure!(
        produced == retained(state, &context.output_sha256)?
            && produced_sha256 == handoff.context_sha256,
        "ContextHandoffMismatch: the handed-off context {} is not the context the producer \
         artifact contains ({produced_sha256})",
        handoff.context_sha256
    );
    let r = &producer.release;
    let pins = AdmissionPinsRecord {
        source_sha256: &r.source_sha256,
        producer_sha256: &r.producer_sha256,
        artifact_sha256: &handoff.artifact_sha256,
        qualification_sha256: &producer.qualification_sha256,
        reference_manifest_sha256: &r.reference_manifest_sha256,
        context_execution_sha256: &context_execution_sha256,
        qualification_execution_sha256: case_execution_pins,
        verification_policy_sha256: &r.verification_policy_sha256,
        training: &handoff.scope.training,
        input_sha256: &handoff.scope.input_sha256,
        preprocessing_sha256: &handoff.scope.preprocessing_sha256,
    };
    let pins_sha256 = sha256_hex(&serde_json::to_vec(&pins)?);
    ensure!(
        handoff.admission_pins_sha256 == pins_sha256,
        "ProducerNotAuthenticated: the handoff's admission pins {} are not the controller's \
         admission pins {pins_sha256} for this run (the preregistered release, W2 qualification \
         and case-execution pins, the producer artifact, its input scope and the retained \
         context execution)",
        handoff.admission_pins_sha256
    );
    Ok(evidence.handoff_sha256.clone())
}

/// The W2 execution proof, re-verified from retained pool receipts: the checks
/// `jope_pipeline::cpu_context_bridge::execution_attestation` makes, on retained bytes instead of
/// live files. The preregistered producer executable ran the job to completion, inside its
/// original window (which opened after the freeze, and is the teacher's own authorized
/// experiment and window), under the pool's per-fit memory cap, on the handoff's input, and wrote
/// the handoff's artifact.
///
/// Returns the sha256 of the controller's attestation of this execution (the admission pins'
/// `context_execution_sha256`), derived from the verified receipts as the controller derives it.
fn verify_execution(
    state: &State,
    policy: &FrozenPolicy,
    handoff: &HandoffRecord,
    ex: &ContextExecution,
    record: &HwmRecord,
    teaching: &Teaching<'_>,
) -> Result<String> {
    const RECEIPTS: &str = "UnprovenContextExecution: a retained pool receipt of the context job \
                            is missing or unreadable";
    let job: WorkerJob = blob(state, &ex.job_sha256).context(RECEIPTS)?;
    let outcome: WorkerOutcome = blob(state, &ex.outcome_sha256).context(RECEIPTS)?;
    let experiment: Experiment = blob(state, &ex.experiment_sha256).context(RECEIPTS)?;
    let window: ExecutionWindow = blob(state, &ex.window_sha256).context(RECEIPTS)?;
    ensure!(
        job.command.program.sha256 == handoff.release.producer_sha256,
        "ProducerNotAuthenticated: the context job ran executable {}, not the preregistered \
         producer {}",
        job.command.program.sha256,
        handoff.release.producer_sha256
    );
    experiment.validate()?;
    window.validate(&experiment)?;
    ensure!(
        experiment.resources.per_fit_seconds <= 30 && experiment.resources.threads_per_worker == 1,
        "UnprovenContextExecution: the context job allowance differs from the pool fit limit"
    );
    ensure!(
        window.started_unix_ms >= policy.created_at_unix * 1000,
        "UnprovenContextExecution: the context was produced before the freeze that preregistered \
         its producer"
    );
    // PR #114 review round 1: the context was produced inside the teacher's own authorized
    // execution, the experiment and original window `qualify` holds to the freeze, not in any
    // self-consistent later window. With the timing check below (exit by the window's deadline
    // less its reporting allowance), a context produced after the teaching deadline never teaches.
    let (context_experiment, teacher_experiment) =
        (experiment.identity()?, teaching.experiment.identity()?);
    let authorized = teaching.window;
    ensure!(
        context_experiment == teacher_experiment
            && window.experiment_sha256 == authorized.experiment_sha256
            && window.started_unix_ms == authorized.started_unix_ms
            && window.deadline_unix_ms == authorized.deadline_unix_ms,
        "UnprovenContextExecution: the context was produced in experiment {context_experiment} \
         over [{}, {}] ms, not in the teacher's authorized experiment {teacher_experiment} and \
         window [{}, {}] ms",
        window.started_unix_ms,
        window.deadline_unix_ms,
        authorized.started_unix_ms,
        authorized.deadline_unix_ms
    );
    let job_sha = job.identity()?;
    ensure!(
        job.request.experiment_sha256 == experiment.identity()?
            && job.request.dataset_sha256 == record.identity.dataset_hash,
        "UnprovenContextExecution: the context job belongs to another experiment or dataset"
    );
    ensure!(
        outcome.schema == EXECUTION_SCHEMA
            && outcome.job_sha256 == job_sha
            && outcome.request_sha256 == job.request.identity()?
            && outcome.command_status == EvaluationStatus::Completed
            && outcome.exit_code == Some(0)
            && outcome.signal.is_none()
            && outcome.reason.is_none(),
        "UnprovenContextExecution: the context job did not complete cleanly under its own identity"
    );
    ensure!(
        job.command
            .inputs
            .iter()
            .any(|a| a.sha256 == handoff.scope.input_sha256),
        "UnprovenContextExecution: the handoff's input is not an input of the context job"
    );
    ensure!(
        outcome.outputs.iter().any(|a| {
            a.sha256 == handoff.artifact_sha256
                && job.outputs.iter().any(|p| job.directory.join(p) == a.path)
        }),
        "UnprovenContextExecution: the handoff's artifact is not a declared output of the context \
         job"
    );
    let c = &outcome.consumption;
    let (Some(started), Some(exited), Some(wall), Some(cpu)) = (
        c.gate_released_unix_ms,
        c.observed_exit_unix_ms,
        c.wall_seconds,
        c.cpu_seconds,
    ) else {
        bail!("UnprovenContextExecution: the context job's timing or CPU accounting is unknown");
    };
    ensure!(
        started >= window.started_unix_ms
            && exited >= started
            && exited
                <= window
                    .deadline_unix_ms
                    .saturating_sub(experiment.resources.reporting_seconds * 1000)
            && wall.is_finite()
            && wall >= 0.
            && wall <= experiment.resources.per_fit_seconds as f64
            && cpu.is_finite()
            && cpu >= 0.,
        "UnprovenContextExecution: the context job escaped its original window or lacks finite \
         accounting"
    );
    let allocation = jope_pipeline::execution::fit_memory_allocation(&experiment)?;
    ensure!(
        c.fit_memory_limit_bytes == Some(allocation.per_fit_bytes)
            && c.cgroup_peak_memory_bytes
                .is_some_and(|n| n > 0 && n <= allocation.per_fit_bytes),
        "UnprovenContextExecution: the context job's memory cap is absent or changed, or its peak \
         is outside the allocation"
    );
    // The supervisor receipt is the retained outcome file (`ExecutionProof::attest` pins the
    // file's digest), the input and output are the handoff's, verified above.
    let attestation = ExecutionAttestationRecord {
        schema: EXECUTION_ATTESTATION_SCHEMA,
        supervisor_receipt_sha256: &ex.outcome_sha256,
        program_sha256: &job.command.program.sha256,
        input_sha256: &handoff.scope.input_sha256,
        output_sha256: &handoff.artifact_sha256,
        completed: true,
        cpu_limit: 1,
        memory_max: allocation.per_fit_bytes,
        swap_max: experiment.resources.swap_bytes,
        threads: 1,
        elapsed_seconds: wall,
        deadline_seconds: experiment.resources.per_fit_seconds as f64,
    };
    Ok(sha256_hex(&serde_json::to_vec(&attestation)?))
}
