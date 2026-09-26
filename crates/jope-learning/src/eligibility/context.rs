//! Mandatory teaching-context provenance. The receiver derives the encoder scope and
//! pins from the preoptimization freeze and verifies retained bytes with
//! [`check_teaching_binding`]; a submission supplies only content-addressed evidence,
//! never its own scope, encoder, consumer or product. Missing provenance never passes.
//!
//! W3-T02 / W2-T07: the context must also have been handed off to teacher admission by the W2
//! context producer the freeze preregistered, through a retained `jope.cpu-context-handoff/v1`
//! record, and the execution that produced its artifact must be proven from retained pool
//! receipts ([`verify_handoff`]). The encoder receipt's own claims are not enough.
use super::*;
use crate::encoder_receipt::{
    check_teaching_binding, Encoder, EncoderCapability, EncoderReceipt, Member, Product, Scope,
    TeachingBinding, SCHEMA as RECEIPT_SCHEMA,
};
use crate::revocation::Revocations;

const MAX_PARENT_DEPTH: usize = 8;
const MAX_PARENTS: usize = 64;

/// Preregistered with the freeze, before any candidate or context exists.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextPlan {
    pub source_family: String,
    pub encoder: Encoder,
    /// The W2 context producer a teaching context must be handed off by (W3-T02, W2-T07).
    /// Absent on freezes made before producer authentication existed: such a freeze keeps its
    /// bytes and identity, but no context under it can qualify a teacher (it stays quarantined;
    /// owner decision (d), 2026-09-26).
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
    pub receipt_sha256: String,
    pub fitted_state_sha256: String,
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

/// Runs before any teaching decision; returns the verified receipt digest and the digest of the
/// W2 handoff it was admitted under.
pub(super) fn verify(
    state: &State,
    policy: &FrozenPolicy,
    record: &HwmRecord,
    evidence: &OofEvidence,
    revocations: &Revocations,
) -> Result<(String, String)> {
    let plan = policy
        .context
        .as_ref()
        .context("MissingEncoderProvenance: freeze preregistered no teaching context encoder")?;
    plan.validate()?;
    let context = evidence
        .context
        .as_ref()
        .context("MissingEncoderProvenance: submission carries no context receipt")?;
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
    let handoff = verify_handoff(state, policy, plan, record, context, revocations)?;
    Ok((context.receipt_sha256.clone(), handoff))
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
    pub hyperion_weights: String,
    pub hyperion_manifest: String,
    pub component_bundle: String,
    pub foundation_weights: String,
    pub target_weights: String,
    pub frozen_transforms: String,
    pub assembly: String,
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
    pub encoder: HandoffEncoder,
    pub source_sha256: String,
    pub producer_sha256: String,
    pub verification_policy_sha256: String,
    pub reference_manifest_sha256: String,
}

/// The W2 context producer the controller preregisters with the freeze, before any candidate or
/// context exists: the release identity a teaching context must be handed off under, the typed
/// transformation chain and the W2 qualification the producer passed. A submission never supplies
/// these; the handoff it carries is checked against them (producer authentication).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerPlan {
    pub release: HandoffRelease,
    /// `jope_policy::context_producer::transformation_chain()`, in execution order.
    pub transformations: Vec<String>,
    pub qualification_sha256: String,
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
                && self.transformations.iter().all(|t| !t.trim().is_empty()),
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
    pub window_sha256: String,
}

/// The W2 handoff a submission carries with its teaching context: the retained
/// `jope.cpu-context-handoff/v1` record, as `ContextHandoff` serializes it, and the execution
/// that produced its artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffEvidence {
    pub handoff_sha256: String,
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HandoffScope {
    training: HandoffTraining,
    input_sha256: String,
    preprocessing_sha256: String,
    prepared_rows_sha256: String,
}

/// W2's training scope. Its row digests are over W2's own prepared-row encoding, which is not
/// the W3 receipt's row encoding, so only their form is checked here; the dataset is joined.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HandoffTraining {
    dataset_sha256: String,
    prepared_sha256: String,
    training_sha256: String,
    fitting_rows_sha256: String,
    #[allow(dead_code)]
    cut: u8,
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
///   executable, proven from retained pool receipts ([`verify_execution`]).
///
/// Returns the handoff digest the qualification records.
fn verify_handoff(
    state: &State,
    policy: &FrozenPolicy,
    plan: &ContextPlan,
    record: &HwmRecord,
    context: &ContextEvidence,
    revocations: &Revocations,
) -> Result<String> {
    let producer = plan.producer.as_ref().with_context(|| {
        quarantined(
            "PreHandoffProvenance: the freeze preregistered no W2 context producer (release \
             identity, transformation chain and W2 qualification), so no teaching context under \
             it can be authenticated",
        )
    })?;
    producer.validate()?;
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
    let handoff: HandoffRecord =
        serde_json::from_slice(&retained(state, &evidence.handoff_sha256)?).context(
            "ContextHandoffMismatch: the retained handoff is not a jope.cpu-context-handoff/v1 \
             record",
        )?;
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
    verify_execution(state, policy, &handoff, ex, record)?;
    Ok(evidence.handoff_sha256.clone())
}

/// The W2 execution proof, re-verified from retained pool receipts: the checks
/// `jope_pipeline::cpu_context_bridge::execution_attestation` makes, on retained bytes instead of
/// live files. The preregistered producer executable ran the job to completion, inside its
/// original window (which opened after the freeze), under the pool's per-fit memory cap, on the
/// handoff's input, and wrote the handoff's artifact.
fn verify_execution(
    state: &State,
    policy: &FrozenPolicy,
    handoff: &HandoffRecord,
    ex: &ContextExecution,
    record: &HwmRecord,
) -> Result<()> {
    let job: WorkerJob = blob(state, &ex.job_sha256)?;
    let outcome: WorkerOutcome = blob(state, &ex.outcome_sha256)?;
    let experiment: Experiment = blob(state, &ex.experiment_sha256)?;
    let window: ExecutionWindow = blob(state, &ex.window_sha256)?;
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
    retained(state, &handoff.artifact_sha256)?;
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
    Ok(())
}
