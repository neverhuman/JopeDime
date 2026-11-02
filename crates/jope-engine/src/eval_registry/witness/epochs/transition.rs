use super::*;
use crate::native_loop::lineage::{DescendantLineagePolicy, DESCENDANT_POLICY_SCHEMA};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EpochTransitionMode {
    InitialDelegationBootstrap,
    SameWindowRecovery,
    ReviewedNewRunSuccessor,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalBudgetAccounting {
    pub schema: String,
    pub original_policy_sha256: String,
    pub original_static_grant_sha256: String,
    pub original_approved_candidate_budget: u64,
    pub original_authorization_count: u64,
    pub physical_starts: Option<u64>,
    /// Unresolved original identities are retained; see
    /// `tests::initial_bootstrap_preserves_original_static_budget_and_unknown_identities`.
    pub unknown_or_held: BTreeMap<String, NativeGrantArtifact>,
    pub source_closure: Vec<NativeGrantArtifact>,
    pub authority_recovery_receipt_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessEpochTransitionBody {
    pub schema: String,
    pub immutable_base_policy_sha256: String,
    pub previous_effective_policy_sha256: String,
    pub previous_transition_sha256: String,
    pub previous_witness_sequence: u64,
    pub previous_witness_tip_sha256: String,
    pub authority_recovery_receipt: NativeGrantArtifact,
    pub run_approval: NativeGrantArtifact,
    pub frozen_config: NativeGrantArtifact,
    pub previous_run_completion: Option<NativeGrantArtifact>,
    pub mode: EpochTransitionMode,
    pub next_policy: WitnessPolicy,
    pub next_update_sequence: u64,
    pub previous_update_sha256: String,
    pub original_reserved_evaluations: u64,
    pub historical_authorization_count: u64,
    pub historical_budget_accounting: Option<NativeGrantArtifact>,
    pub historical_accounting: Option<HistoricalBudgetAccounting>,
    pub cumulative_evaluation_limit: u64,
    pub not_before_unix_ms: u64,
    pub expires_unix_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedWitnessEpochTransition {
    pub authorization_public_key: String,
    pub body: WitnessEpochTransitionBody,
    pub signature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessEpochTransitionAckBody {
    pub schema: String,
    pub immutable_base_policy_sha256: String,
    pub witness_instance: String,
    pub request_sha256: String,
    pub effective_policy_sha256: String,
    pub witness_sequence: u64,
    pub previous_record_sha256: String,
    pub committed_unix_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedWitnessEpochTransitionAck {
    pub body: WitnessEpochTransitionAckBody,
    pub signature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessEpochProof {
    pub epoch_transition: SignedWitnessEpochTransition,
    pub ack: SignedWitnessEpochTransitionAck,
}
// Called only after the original base policy and reviewed transition chain have
// been checked. Ordinary unextended base policies retain the initial60000 rule.
pub(super) fn validate_effective_shape(policy: &WitnessPolicy) -> Result<()> {
    let mut normalized = policy.clone();
    if let Some(d) = &mut normalized.native_grant_delegation {
        if let Some(lineage) = descendant_policy(policy)? {
            validate_descendant_window(&lineage, d)?;
        } else {
            ensure!(
                d.initial_optimizer_step >= 60000 && (d.initial_optimizer_step - 60000) % 6000 == 0,
                "invalid reviewed successor optimizer window"
            );
        }
        d.initial_optimizer_step = 60000;
    } else {
        ensure!(
            descendant_policy(policy)?.is_none(),
            "lineage opt-in requires a finite delegation"
        );
    }
    normalized.validate()
}

// Coordinator adds ONLY this optional field to the signed WitnessPolicy DTO.
// Until that hunk is integrated, its strict reader refuses opt-ins and this
// adapter always returns None. Source/executable rotation cannot enable it.
fn descendant_policy(policy: &WitnessPolicy) -> Result<Option<DescendantLineagePolicy>> {
    serde_json::to_value(policy)?
        .get("descendant_lineage_policy")
        .filter(|v| !v.is_null())
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(Into::into)
}

fn validate_descendant_window(
    policy: &DescendantLineagePolicy,
    delegation: &NativeGrantDelegation,
) -> Result<crate::native_loop::lineage::VerifiedLineage> {
    ensure!(
        policy.schema == DESCENDANT_POLICY_SCHEMA
            && policy.run == delegation.run
            && policy.scientific_policy_sha256 == delegation.scientific_policy_sha256
            && valid_hash(&policy.previous_completion_sha256),
        "lineage opt-in differs from signed run/science/completion scope"
    );
    let verified = policy.bundle.verify()?;
    ensure!(
        verified.lineage().anchor.optimizer_step == 60000
            && verified.local_step() == delegation.initial_optimizer_step,
        "reviewed descendant optimizer window differs from committed lineage"
    );
    Ok(verified)
}

fn validate_descendant_transition(
    next: &DescendantLineagePolicy,
    prior: Option<&DescendantLineagePolicy>,
    d: &NativeGrantDelegation,
    p: &NativeGrantDelegation,
    completion: Option<&NativeGrantArtifact>,
) -> Result<()> {
    let verified = validate_descendant_window(next, d)?;
    ensure!(
        completion.is_some_and(|c| c.sha256 == next.previous_completion_sha256),
        "descendant successor lacks exact separate completion"
    );
    let prefix = if let Some(prior) = prior {
        let previous = validate_descendant_window(prior, p)?;
        ensure!(
            verified.lineage().anchor == previous.lineage().anchor
                && verified.lineage().protocol == previous.lineage().protocol
                && verified
                    .lineage()
                    .edges
                    .starts_with(&previous.lineage().edges),
            "descendant transition changes immutable ancestry"
        );
        previous.edges().len()
    } else {
        ensure!(
            p.initial_optimizer_step == 60000,
            "first descendant must bind original anchor window"
        );
        0
    };
    let added = verified
        .edges()
        .get(prefix..)
        .context("missing descendant successor edges")?;
    let initial = if prior.is_some() {
        p.initial_optimizer_step
    } else {
        0
    };
    ensure!(
        added.len() == 2
            && added[0].intent.initial_step == initial
            && added
                .iter()
                .all(|e| e.intent.target_step.checked_sub(e.intent.initial_step) == Some(3000))
            && added[0].intent.target_step == added[1].intent.initial_step,
        "descendant successor requires both committed 3000-update blocks"
    );
    Ok(())
}
fn artifact(a: &NativeGrantArtifact) -> Result<()> {
    ensure!(
        valid_hash(&a.sha256)
            && a.path.is_absolute()
            && a.path.as_os_str().len() <= 4096
            && a.path.components().all(|c| matches!(
                c,
                std::path::Component::RootDir | std::path::Component::Normal(_)
            )),
        "invalid transition evidence identity"
    );
    Ok(())
}
fn validate_transition_with_key(
    base: &WitnessPolicy,
    previous: &WitnessPolicy,
    request: &SignedWitnessEpochTransition,
    when: u64,
    reviewer: &str,
) -> Result<()> {
    ensure!(
        bytes(request)?.len() <= 1024 * 1024,
        "epoch transition byte bound"
    );
    let b = &request.body;
    ensure!(
        request.authorization_public_key == reviewer,
        "untrusted epoch reviewer"
    );
    verify(reviewer, TRANSITION_DOMAIN, b, &request.signature)?;
    ensure!(
        b.schema == "parallel.witness_epoch_transition.v1"
            && b.immutable_base_policy_sha256 == base.sha256()?
            && b.previous_effective_policy_sha256 == previous.sha256()?
            && valid_hash(&b.previous_transition_sha256)
            && valid_hash(&b.previous_witness_tip_sha256)
            && b.next_update_sequence > 0
            && valid_hash(&b.previous_update_sha256)
            && ((b.next_update_sequence == 1) == (b.previous_update_sha256 == ZERO))
            && b.not_before_unix_ms <= when
            && when < b.expires_unix_ms,
        "stale epoch policy/window"
    );
    for a in [
        &b.authority_recovery_receipt,
        &b.run_approval,
        &b.frozen_config,
    ] {
        artifact(a)?;
    }
    if let Some(a) = &b.previous_run_completion {
        artifact(a)?;
    }
    let n = &b.next_policy;
    validate_effective_shape(n)?;
    let next_lineage = descendant_policy(n)?;
    let prior_lineage = descendant_policy(previous)?;
    ensure!(
        prior_lineage.is_none() || next_lineage.is_some(),
        "signed lineage opt-in cannot be stripped"
    );
    ensure!(
        n.witness_instance == base.witness_instance
            && n.authority_instance == base.authority_instance
            && n.witness_public_key == base.witness_public_key
            && n.authority_public_key == base.authority_public_key
            && n.ssh_known_hosts_sha256 == base.ssh_known_hosts_sha256
            && n.ssh_host_alias == base.ssh_host_alias
            && n.ssh_destination == base.ssh_destination
            && n.authority_epoch != previous.authority_epoch
            && n.authority_epoch != base.authority_epoch
            && n.native_fit_trust.is_none()
            && n.funding_trust.is_none(),
        "transition changes original witness/authority custody or reopens legacy trust"
    );
    let prior = previous.native_grant_delegation.as_ref();
    let d = n
        .native_grant_delegation
        .as_ref()
        .context("successor delegation absent")?;
    ensure!(
        reviewer != n.authority_public_key
            && reviewer != n.witness_public_key
            && reviewer != d.provisioner_public_key
            && prior.is_none_or(|p| reviewer != p.provisioner_public_key),
        "epoch reviewer cannot be authority, witness, or provisioner"
    );
    ensure!(
        d.cycles == 2
            && d.total_evaluation_budget == b.cumulative_evaluation_limit
            && b.original_reserved_evaluations <= b.cumulative_evaluation_limit
            && prior.is_none_or(|p| d.grant_template.authority_root
                == p.grant_template.authority_root
                && d.grant_template.witness == p.grant_template.witness
                && d.grant_template.root == p.grant_template.root
                && d.grant_template.authority_manifest_sha256
                    == p.grant_template.authority_manifest_sha256),
        "transition resets cumulative budget or physical authority/data identity"
    );
    if b.mode == EpochTransitionMode::InitialDelegationBootstrap {
        ensure!(
            prior.is_none()
                && previous.sha256()? == base.sha256()?
                && base.native_fit_trust.is_some()
                && base.funding_trust.is_none()
                && b.previous_transition_sha256 == ZERO
                && b.previous_run_completion.is_none()
                && b.original_reserved_evaluations == 0
                && next_lineage.is_none()
                && d.initial_optimizer_step == 60000,
            "bootstrap must be first separately reviewed production window on original static ledger"
        );
        let evidence = b
            .historical_budget_accounting
            .as_ref()
            .context("bootstrap original budget evidence absent")?;
        artifact(evidence)?;
        let accounting = b
            .historical_accounting
            .as_ref()
            .context("bootstrap original budget accounting absent")?;
        ensure!(
            accounting.schema == "jope.original_static_grant_budget.v1"
                && accounting.original_policy_sha256 == base.sha256()?
                && accounting.original_static_grant_sha256
                    == base.native_fit_trust.as_ref().unwrap().grant_sha256
                && accounting.original_approved_candidate_budget > 0
                && accounting.original_approved_candidate_budget <= 100_000_000
                && accounting.original_authorization_count == b.historical_authorization_count
                && accounting.original_authorization_count
                    <= accounting.original_approved_candidate_budget
                && accounting
                    .physical_starts
                    .is_none_or(|n| n <= accounting.original_approved_candidate_budget)
                && accounting.unknown_or_held.len() <= 100_000
                && !accounting.source_closure.is_empty()
                && accounting.source_closure.len() <= 4096
                && accounting.authority_recovery_receipt_sha256
                    == b.authority_recovery_receipt.sha256
                && sha(&bytes(accounting)?) == evidence.sha256,
            "bootstrap cannot omit original grant accounting/unknown provenance"
        );
        for (identity, evidence) in &accounting.unknown_or_held {
            ensure!(valid_hash(identity), "original held identity malformed");
            artifact(evidence)?;
        }
        for evidence in &accounting.source_closure {
            artifact(evidence)?;
        }
        return Ok(());
    }
    ensure!(
        b.historical_budget_accounting.is_none() && b.historical_accounting.is_none(),
        "original bootstrap accounting cannot be replaced by later transition"
    );
    let p = prior.context("original finite delegation absent")?;
    match b.mode {
        EpochTransitionMode::InitialDelegationBootstrap => {
            unreachable!("bootstrap validated above")
        }
        EpochTransitionMode::SameWindowRecovery => ensure!(
            b.previous_run_completion.is_none()
                && d.run == p.run
                && d.initial_optimizer_step == p.initial_optimizer_step
                && d.scientific_policy_sha256 == p.scientific_policy_sha256
                && d.deadline_unix_ms <= p.deadline_unix_ms
                && d.grant_template.period_start_unix_ms >= p.grant_template.period_start_unix_ms
                && d.grant_template.period_end_unix_ms <= p.grant_template.period_end_unix_ms
                && b.cumulative_evaluation_limit == p.total_evaluation_budget,
            "same-window recovery extends original science/window/budget"
        ),
        EpochTransitionMode::ReviewedNewRunSuccessor => {
            if let Some(lineage) = &next_lineage {
                validate_descendant_transition(
                    lineage,
                    prior_lineage.as_ref(),
                    d,
                    p,
                    b.previous_run_completion.as_ref(),
                )?;
            } else {
                ensure!(
                    d.initial_optimizer_step
                        == p.initial_optimizer_step
                            .checked_add(6000)
                            .context("optimizer overflow")?,
                    "new run lacks exact finite successor optimizer window"
                );
            }
            ensure!(
                b.previous_run_completion.is_some()
                    && d.run != p.run
                    && d.grant_template.period_start_unix_ms
                        >= p.grant_template.period_start_unix_ms
                    && b.cumulative_evaluation_limit >= p.total_evaluation_budget,
                "new run lacks exact finite successor identity/cumulative accounting"
            );
        }
    }
    if b.mode == EpochTransitionMode::SameWindowRecovery {
        ensure!(
            next_lineage == prior_lineage,
            "same-window recovery changes signed lineage scope"
        );
    }
    Ok(())
}
pub(super) fn validate_transition(
    base: &WitnessPolicy,
    previous: &WitnessPolicy,
    request: &SignedWitnessEpochTransition,
    when: u64,
) -> Result<()> {
    crate::native_review::require_signer(&request.authorization_public_key)?;
    validate_transition_with_key(
        base,
        previous,
        request,
        when,
        &request.authorization_public_key,
    )
}
/// Verify original proofs without minting a new ACK or consulting current time.
/// Old grants retain the exact policy/epoch under which they were witnessed.
pub fn verify_epoch_chain(
    base: &WitnessPolicy,
    proofs: &[WitnessEpochProof],
) -> Result<WitnessPolicy> {
    for proof in proofs {
        crate::native_review::require_signer(&proof.epoch_transition.authorization_public_key)?;
    }
    verify_epoch_chain_checked(base, proofs, None)
}
pub(super) fn verify_epoch_chain_checked(
    base: &WitnessPolicy,
    proofs: &[WitnessEpochProof],
    reviewer: Option<&str>,
) -> Result<WitnessPolicy> {
    base.validate()?;
    ensure!(
        proofs.len() <= MAX_EPOCH_TRANSITIONS && bytes(&proofs)?.len() <= 1024 * 1024,
        "epoch proof chain bound"
    );
    let mut policy = base.clone();
    let mut previous = ZERO.to_string();
    let mut last_sequence = 0;
    let mut epochs = std::collections::BTreeSet::from([base.authority_epoch.clone()]);
    let mut reserved = 0;
    let mut update_sequence = 1;
    for proof in proofs {
        let r = &proof.epoch_transition;
        let b = &proof.ack.body;
        if let Some(reviewer) = reviewer {
            validate_transition_with_key(base, &policy, r, b.committed_unix_ms, reviewer)?;
        } else {
            validate_transition(base, &policy, r, b.committed_unix_ms)?;
        }
        ensure!(
            r.body.previous_transition_sha256 == previous
                && r.body.previous_witness_sequence >= last_sequence
                && r.body.original_reserved_evaluations >= reserved
                && r.body.next_update_sequence >= update_sequence
                && epochs.insert(r.body.next_policy.authority_epoch.clone())
                && b.schema == "parallel.witness_epoch_transition_ack.v1"
                && b.immutable_base_policy_sha256 == base.sha256()?
                && b.witness_instance == base.witness_instance
                && b.request_sha256 == sha(&bytes(r)?)
                && b.effective_policy_sha256 == r.body.next_policy.sha256()?
                && b.witness_sequence
                    == r.body
                        .previous_witness_sequence
                        .checked_add(1)
                        .context("witness sequence overflow")?
                && b.previous_record_sha256 == r.body.previous_witness_tip_sha256,
            "epoch ACK/chain identity mismatch"
        );
        verify(
            &base.witness_public_key,
            TRANSITION_ACK_DOMAIN,
            b,
            &proof.ack.signature,
        )?;
        previous = sha(&bytes(proof)?);
        last_sequence = b.witness_sequence;
        reserved = r.body.original_reserved_evaluations;
        update_sequence = r.body.next_update_sequence;
        policy = r.body.next_policy.clone();
    }
    Ok(policy)
}
