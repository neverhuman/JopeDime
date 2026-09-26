//! Mandatory teaching-context provenance. The receiver derives the encoder scope and
//! pins from the preoptimization freeze and verifies retained bytes with
//! [`check_teaching_binding`]; a submission supplies only content-addressed evidence,
//! never its own scope, encoder, consumer or product. Missing provenance never passes.
use super::*;
use crate::encoder_receipt::{
    check_teaching_binding, Encoder, EncoderCapability, EncoderReceipt, Member, Product, Scope,
    TeachingBinding, SCHEMA as RECEIPT_SCHEMA,
};

const MAX_PARENT_DEPTH: usize = 8;
const MAX_PARENTS: usize = 64;

/// Preregistered with the freeze, before any candidate or context exists.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextPlan {
    pub source_family: String,
    pub encoder: Encoder,
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

/// Every parent is a retained receipt fitted in the same scope, with retained bytes.
fn parents(state: &State, child: &EncoderReceipt) -> Result<()> {
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
        let parent = receipt(state, &sha).context("unresolved context parent receipt")?;
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

/// Runs before any teaching decision; returns the verified receipt digest.
pub(super) fn verify(
    state: &State,
    policy: &FrozenPolicy,
    record: &HwmRecord,
    evidence: &OofEvidence,
) -> Result<String> {
    let plan = policy
        .context
        .as_ref()
        .context("MissingEncoderProvenance: freeze preregistered no teaching context encoder")?;
    plan.validate()?;
    let context = evidence
        .context
        .as_ref()
        .context("MissingEncoderProvenance: submission carries no context receipt")?;
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
    parents(state, &receipt)?;
    Ok(context.receipt_sha256.clone())
}
