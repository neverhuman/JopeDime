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

mod handoff;
mod handoff_verify;

#[cfg(test)]
pub(super) use handoff::HandoffTraining;
pub(super) use handoff::{
    parse_handoff, AdmissionPinsRecord, ExecutionAttestationRecord, HandoffRecord,
    ProducedArtifact, CONTEXT_CLIP, CONTEXT_WIDTH, EXECUTION_ATTESTATION_SCHEMA,
    MAX_ARTIFACT_BYTES, PRODUCED_ARTIFACT_BACKEND, PRODUCED_ARTIFACT_SCHEMA,
};
pub use handoff::{
    ContextExecution, HandoffEncoder, HandoffEvidence, HandoffRelease, ProducerPlan,
    HANDOFF_SCHEMA, TEACHER_ADMISSION,
};
pub(super) use handoff_verify::verify_handoff;
