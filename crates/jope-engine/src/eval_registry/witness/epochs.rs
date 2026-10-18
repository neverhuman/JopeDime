//! Root-reviewed transitions in the original witness journal. No ledger reset.
use super::*;
use crate::native_loop::lineage::{DescendantLineagePolicy, DESCENDANT_POLICY_SCHEMA};
const TRANSITION_DOMAIN: &str = "parallel.witness_epoch_transition.v1";
const TRANSITION_ACK_DOMAIN: &str = "parallel.witness_epoch_transition_ack.v1";
pub const MAX_EPOCH_TRANSITIONS: usize = 64;
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
    /// Explicit unresolved originals, not fabricated zero-cost/completed work.
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
fn verify_epoch_chain_checked(
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
impl WitnessStore {
    pub fn effective_policy(&self) -> &WitnessPolicy {
        self.epoch_proofs
            .last()
            .map_or(&self.policy, |p| &p.epoch_transition.body.next_policy)
    }
    pub fn epoch_proofs(&self) -> &[WitnessEpochProof] {
        &self.epoch_proofs
    }
    pub(super) fn check_epoch_next(
        &self,
        request: &SignedWitnessEpochTransition,
        when: u64,
    ) -> Result<()> {
        validate_transition(&self.policy, self.effective_policy(), request, when)?;
        let (next_update, previous_update) = self.native_grants.update_cursor()?;
        ensure!(
            request.body.next_update_sequence == next_update
                && request.body.previous_update_sha256 == previous_update,
            "epoch transition original grant cursor differs"
        );
        ensure!(
            self.epoch_proofs.len() < MAX_EPOCH_TRANSITIONS
                && request.body.previous_witness_sequence == self.sequence
                && request.body.previous_witness_tip_sha256 == self.tip
                && request.body.original_reserved_evaluations
                    == self.native_grants.reserved_evaluations
                && request.body.historical_authorization_count == self.records.len() as u64
                && request.body.previous_transition_sha256
                    == match self.epoch_proofs.last() {
                        Some(p) => sha(&bytes(p)?),
                        None => ZERO.into(),
                    }
                && self.epoch_proofs.iter().all(|p| p
                    .epoch_transition
                    .body
                    .next_policy
                    .authority_epoch
                    != request.body.next_policy.authority_epoch),
            "stale epoch transition/head/budget"
        );
        Ok(())
    }
    pub(super) fn replay_epoch_transition(
        &mut self,
        proof: WitnessEpochProof,
        digest: String,
    ) -> Result<()> {
        self.check_epoch_next(&proof.epoch_transition, proof.ack.body.committed_unix_ms)?;
        let mut chain = self.epoch_proofs.clone();
        chain.push(proof.clone());
        verify_epoch_chain(&self.policy, &chain)?;
        self.sequence = proof.ack.body.witness_sequence;
        self.tip = digest;
        self.epoch_proofs.push(proof);
        // NativeGrantState is deliberately not replaced: all original grant,
        // numerical, task/candidate identities and cumulative cost survive.
        Ok(())
    }
    pub fn add_epoch_transition(
        &mut self,
        request: &SignedWitnessEpochTransition,
    ) -> Result<WitnessEpochProof> {
        ensure!(!self.read_only, "read-only witness cannot transition");
        self.verify_custody()?;
        // Original retry wins before current-epoch/window checks.
        for original in &self.epoch_proofs {
            if original.epoch_transition.body.next_policy.authority_epoch
                == request.body.next_policy.authority_epoch
            {
                ensure!(
                    bytes(&original.epoch_transition)? == bytes(request)?,
                    "conflicting epoch retry"
                );
                let data = read(
                    &self
                        .root
                        .join("segments")
                        .join(format!("{:020}.json", original.ack.body.witness_sequence)),
                )?;
                ensure!(data == bytes(original)?, "original epoch record changed");
                return Ok(original.clone());
            }
        }
        let now = super::super::native_scq::policy::now_ms()?;
        ensure!(
            sha(&fs::read("/proc/self/exe")?) == request.body.next_policy.witness_executable_sha256
                && source_sha256() == request.body.next_policy.witness_source_sha256,
            "transition must execute in independently reviewed successor witness image"
        );
        self.check_epoch_next(request, now)?;
        let b = WitnessEpochTransitionAckBody {
            schema: "parallel.witness_epoch_transition_ack.v1".into(),
            immutable_base_policy_sha256: self.policy.sha256()?,
            witness_instance: self.policy.witness_instance.clone(),
            request_sha256: sha(&bytes(request)?),
            effective_policy_sha256: request.body.next_policy.sha256()?,
            witness_sequence: self
                .sequence
                .checked_add(1)
                .context("witness sequence overflow")?,
            previous_record_sha256: self.tip.clone(),
            committed_unix_ms: now,
        };
        let proof = WitnessEpochProof {
            epoch_transition: request.clone(),
            ack: SignedWitnessEpochTransitionAck {
                signature: sign(
                    self.signing
                        .as_ref()
                        .context("public verifier cannot sign")?,
                    TRANSITION_ACK_DOMAIN,
                    &b,
                )?,
                body: b,
            },
        };
        let data = bytes(&proof)?;
        ensure!(data.len() <= MAX_FRAME, "epoch record bound");
        let pending = tempfile::Builder::new()
            .prefix(".epoch-transition-pending-")
            .tempfile_in(&self.root)?;
        let (mut file, path) = pending.keep()?;
        file.write_all(&data)?;
        file.sync_all()?;
        sync_dir(&self.root)?;
        crate::native_persistence::publish_staged(
            &path,
            &self
                .root
                .join("segments")
                .join(format!("{:020}.json", self.sequence + 1)),
        )?;
        sync_dir(&self.root.join("segments"))?;
        self.replay_epoch_transition(proof.clone(), sha(&data))?;
        self.publish_head()?;
        crate::native_persistence::discard_staging(&path)?;
        sync_dir(&self.root)?;
        Ok(proof)
    }
}

impl WitnessStore {
    /// Only this endpoint can bootstrap the first reviewed runtime successor.
    /// Exact next-image signature is checked before acquiring mutable custody.
    pub fn open_for_epoch_transition(
        root: &Path,
        base: &WitnessPolicy,
        key_path: &Path,
        request: &SignedWitnessEpochTransition,
    ) -> Result<Self> {
        Self::open_mode_transition(root, base, key_path, false, Some(request))
    }
}
pub fn serve_stdio_epoch_transition(
    root: &Path,
    policy_path: &Path,
    key_path: &Path,
) -> Result<()> {
    let base: WitnessPolicy = serde_json::from_slice(&read(policy_path)?)?;
    let request: SignedWitnessEpochTransition = read_frame(std::io::stdin().lock())?;
    let mut store = WitnessStore::open_for_epoch_transition(root, &base, key_path, &request)?;
    let proof = store.add_epoch_transition(&request)?;
    write_frame(std::io::stdout().lock(), &proof)
}

/// Metadata only, from the completely replayed original journal.
impl WitnessStore {
    pub fn epoch_status(&self) -> Result<Value> {
        let (next, previous) = self.native_grants.update_cursor()?;
        Ok(
            serde_json::json!({"schema":"parallel.witness_epoch_status.v1",
            "immutable_base_policy_sha256":self.policy.sha256()?,
            "effective_policy":self.effective_policy(),"epoch_proofs":self.epoch_proofs,
            "witness_sequence":self.sequence,"witness_tip_sha256":self.tip,
            "reserved_evaluations":self.native_grants.reserved_evaluations,
            "next_update_sequence":next,"previous_update_sha256":previous}),
        )
    }
}
pub fn validate_native_grant_request_for_epoch(
    base: &WitnessPolicy,
    proofs: &[WitnessEpochProof],
    request: &SignedAddNativeGrant,
) -> Result<NativeWitnessTrust> {
    let effective = verify_epoch_chain(base, proofs)?;
    native_grants::validate_update_effective(&effective, request)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn lineage_fixture() -> DescendantLineagePolicy {
        use crate::native_loop::lineage::*;
        fn put(b: &mut LineageBundle, schema: &str, value: &Value) -> String {
            let bytes = canonical_bytes(value).unwrap();
            let id = object_id(schema, &bytes).unwrap();
            b.evidence
                .insert(id.clone(), String::from_utf8(bytes).unwrap());
            id
        }
        let h = |s: &str| sha(s.as_bytes());
        let protocol = serde_json::json!({"objective_sha256":h("objective"),"recipe_sha256":h("recipe"),"architecture_sha256":h("architecture"),"vocabulary_sha256":h("vocabulary"),"support_sha256":h("support")});
        let mut bundle: LineageBundle = serde_json::from_value(serde_json::json!({
            "lineage":{"schema":LINEAGE_SCHEMA,"anchor":{"checkpoint_sha256":h("anchor"),"weights_sha256":h("anchor weights"),"optimizer_step":60000,"objective_sha256":null,"committed_updates":null,"architecture_sha256":h("architecture"),"vocabulary_sha256":h("vocabulary"),"support_sha256":h("support")},"protocol":protocol,"edges":[]},"evidence":{}})).unwrap();
        let epoch = OptimizerEpoch {
            schema: EPOCH_SCHEMA.into(),
            origin_checkpoint_sha256: h("anchor"),
            reset_nonce_sha256: h("reset"),
            recipe_sha256: h("recipe"),
        };
        for i in 0..2 {
            let parent = if i == 0 {
                h("anchor")
            } else {
                h("checkpoint0")
            };
            let child = h(&format!("checkpoint{i}"));
            let intent = serde_json::json!({"schema":INTENT_SCHEMA,"parent_checkpoint_sha256":parent,"protocol":protocol,"corpus_sha256":h("corpus"),"optimizer_epoch_sha256":epoch.sha256().unwrap(),"initial_step":i*3000,"target_step":(i+1)*3000,"execution_partition_sha256":h("partition"),"sampler_sha256":h("sampler")});
            let intent_id = object_id(INTENT_SCHEMA, &canonical_bytes(&intent).unwrap()).unwrap();
            let eligible = put(
                &mut bundle,
                ELIGIBILITY_SCHEMA,
                &serde_json::json!({"schema":ELIGIBILITY_SCHEMA,"purpose":"training","intent_sha256":intent_id,"corpus_sha256":h("corpus"),"role_exposure_closure_sha256":h("fixture role closure"),"source_roles":["train"]}),
            );
            let credit = put(
                &mut bundle,
                CREDIT_SCHEMA,
                &serde_json::json!({"schema":CREDIT_SCHEMA,"commit_sha256":h(&format!("commit{i}")),"parent_checkpoint_sha256":parent,"checkpoint_sha256":child,"optimizer_epoch_sha256":epoch.sha256().unwrap(),"intent_sha256":intent_id,"eligibility_sha256":eligible,"rank":0,"world":1,"initial_update":i*3000,"target_update":(i+1)*3000}),
            );
            let edge = serde_json::json!({"schema":EDGE_SCHEMA,"previous_edge_sha256":bundle.lineage.edges.last(),"parent_checkpoint_sha256":parent,"checkpoint_sha256":child,"weights_sha256":h(&format!("weights{i}")),"optimizer_sha256":h(&format!("optimizer{i}")),"mode":if i==0 {"weight_warm_start"} else {"new_corpus_transition"},"epoch":epoch,"intent":intent,"committed_credit_sha256":[credit]});
            let id = put(&mut bundle, EDGE_SCHEMA, &edge);
            bundle.lineage.edges.push(id);
        }
        DescendantLineagePolicy {
            schema: DESCENDANT_POLICY_SCHEMA.into(),
            run: "descendant run".into(),
            scientific_policy_sha256: h("science"),
            previous_completion_sha256: h("completion"),
            bundle,
        }
    }

    #[test]
    fn descendant_verifier_requires_both_committed_blocks_and_scoped_opt_in() {
        let (_t, _root, _key, base, _provisioner) = native_grants::tests::fixture();
        let prior = base.native_grant_delegation.as_ref().unwrap();
        let mut next = prior.clone();
        let policy = lineage_fixture();
        next.run = policy.run.clone();
        next.scientific_policy_sha256 = policy.scientific_policy_sha256.clone();
        next.initial_optimizer_step = 6000;
        let completion = NativeGrantArtifact {
            path: "/data/completion.json".into(),
            sha256: policy.previous_completion_sha256.clone(),
        };
        let v = validate_descendant_window(&policy, &next).unwrap();
        assert_eq!(
            (
                v.local_step(),
                v.committed_updates(),
                v.cumulative_updates()
            ),
            (6000, 6000, None)
        );
        validate_descendant_transition(&policy, None, &next, prior, Some(&completion)).unwrap();
        assert!(validate_descendant_transition(&policy, None, &next, prior, None).is_err());
        let mut short = policy.clone();
        short.bundle.lineage.edges.truncate(1);
        next.initial_optimizer_step = 3000;
        validate_descendant_window(&short, &next).unwrap();
        assert!(
            validate_descendant_transition(&short, None, &next, prior, Some(&completion)).is_err()
        );
        next.run = "wrong scope".into();
        assert!(validate_descendant_window(&short, &next).is_err());
        next.run = policy.run.clone();
        next.initial_optimizer_step = 70000;
        assert!(validate_descendant_window(&policy, &next).is_err());
    }

    #[test]
    fn validly_signed_current_digest_rotation_without_opt_in_keeps_legacy_window() {
        let (_t, _root, _key, base, _provisioner) = native_grants::tests::fixture();
        let reviewer = SigningKey::from_bytes(&[81; 32]);
        let pk = hex::encode(reviewer.verifying_key().to_bytes());
        let mut r = transition(
            &base,
            &reviewer,
            EpochTransitionMode::ReviewedNewRunSuccessor,
        );
        assert_eq!(r.body.next_policy.witness_source_sha256, source_sha256());
        assert_eq!(
            r.body.next_policy.authority_source_sha256,
            authority_source_sha256()
        );
        assert!(descendant_policy(&r.body.next_policy).unwrap().is_none());
        r.body
            .next_policy
            .native_grant_delegation
            .as_mut()
            .unwrap()
            .initial_optimizer_step = 3000;
        r.signature = sign(&reviewer, TRANSITION_DOMAIN, &r.body).unwrap();
        verify(&pk, TRANSITION_DOMAIN, &r.body, &r.signature).unwrap();
        let error = validate_transition_with_key(&base, &base, &r, 50, &pk).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid reviewed successor optimizer window"),
            "{error}"
        );
        // Neither a forged body nor an expired/foreign issuer obtains a new window.
        r.body
            .next_policy
            .native_grant_delegation
            .as_mut()
            .unwrap()
            .initial_optimizer_step = 66000;
        assert!(validate_transition_with_key(&base, &base, &r, 50, &pk).is_err());
        r.signature = sign(&reviewer, TRANSITION_DOMAIN, &r.body).unwrap();
        assert!(validate_transition_with_key(&base, &base, &r, 100, &pk).is_err());
        assert!(
            validate_transition_with_key(&base, &base, &r, 50, &base.authority_public_key).is_err()
        );
    }

    #[test]
    fn forged_or_stripped_opt_in_projection_breaks_the_transition_signature() {
        let (_t, _root, _key, base, _provisioner) = native_grants::tests::fixture();
        let reviewer = SigningKey::from_bytes(&[81; 32]);
        let pk = hex::encode(reviewer.verifying_key().to_bytes());
        let r = transition(
            &base,
            &reviewer,
            EpochTransitionMode::ReviewedNewRunSuccessor,
        );
        // The coordinator's future field is signed inside next_policy. This
        // projection test does not claim the absent DTO hunk is integrated.
        let mut body = serde_json::to_value(&r.body).unwrap();
        body["next_policy"]["descendant_lineage_policy"] =
            serde_json::to_value(lineage_fixture()).unwrap();
        let signature = sign(&reviewer, TRANSITION_DOMAIN, &body).unwrap();
        verify(&pk, TRANSITION_DOMAIN, &body, &signature).unwrap();
        let mut forged = body.clone();
        forged["next_policy"]["descendant_lineage_policy"]["run"] =
            Value::String("foreign scope".into());
        assert!(verify(&pk, TRANSITION_DOMAIN, &forged, &signature).is_err());
        body["next_policy"]
            .as_object_mut()
            .unwrap()
            .remove("descendant_lineage_policy");
        assert!(verify(&pk, TRANSITION_DOMAIN, &body, &signature).is_err());
    }
    fn transition(
        base: &WitnessPolicy,
        reviewer: &SigningKey,
        mode: EpochTransitionMode,
    ) -> SignedWitnessEpochTransition {
        let mut next = base.clone();
        next.authority_epoch = sha(b"next epoch");
        let d = next.native_grant_delegation.as_mut().unwrap();
        d.authority_epoch = next.authority_epoch.clone();
        d.grant_template.authority_epoch = next.authority_epoch.clone();
        if mode == EpochTransitionMode::ReviewedNewRunSuccessor {
            d.run = "next reviewed run".into();
            d.initial_optimizer_step = 66000;
            d.total_evaluation_budget = 8;
        }
        let a = NativeGrantArtifact {
            path: "/data/review/evidence.json".into(),
            sha256: sha(b"exact evidence"),
        };
        let body = WitnessEpochTransitionBody {
            schema: "parallel.witness_epoch_transition.v1".into(),
            immutable_base_policy_sha256: base.sha256().unwrap(),
            previous_effective_policy_sha256: base.sha256().unwrap(),
            previous_transition_sha256: ZERO.into(),
            previous_witness_sequence: 7,
            previous_witness_tip_sha256: sha(b"original head"),
            authority_recovery_receipt: a.clone(),
            run_approval: a.clone(),
            frozen_config: a.clone(),
            previous_run_completion: if mode == EpochTransitionMode::ReviewedNewRunSuccessor {
                Some(a)
            } else {
                None
            },
            mode,
            next_update_sequence: 2,
            previous_update_sha256: sha(b"original grant"),
            original_reserved_evaluations: 3,
            historical_authorization_count: 0,
            historical_budget_accounting: None,
            historical_accounting: None,
            cumulative_evaluation_limit: next
                .native_grant_delegation
                .as_ref()
                .unwrap()
                .total_evaluation_budget,
            not_before_unix_ms: 1,
            expires_unix_ms: 100,
            next_policy: next,
        };
        SignedWitnessEpochTransition {
            authorization_public_key: hex::encode(reviewer.verifying_key().to_bytes()),
            signature: sign(reviewer, TRANSITION_DOMAIN, &body).unwrap(),
            body,
        }
    }
    #[test]
    fn same_window_cannot_extend_budget_deadline_or_relabel_initial_optimizer() {
        let (_t, _root, _key, base, _provisioner) = native_grants::tests::fixture();
        let reviewer = SigningKey::from_bytes(&[81; 32]);
        let pk = hex::encode(reviewer.verifying_key().to_bytes());
        let original = transition(&base, &reviewer, EpochTransitionMode::SameWindowRecovery);
        validate_transition_with_key(&base, &base, &original, 50, &pk).unwrap();
        for change in 0..4 {
            let mut r = original.clone();
            let d = r.body.next_policy.native_grant_delegation.as_mut().unwrap();
            match change {
                0 => {
                    d.total_evaluation_budget += 1;
                    r.body.cumulative_evaluation_limit += 1;
                }
                1 => d.deadline_unix_ms += 1,
                2 => d.initial_optimizer_step = 66000,
                _ => r.body.next_policy.authority_instance = sha(b"empty replacement authority"),
            };
            r.signature = sign(&reviewer, TRANSITION_DOMAIN, &r.body).unwrap();
            assert!(validate_transition_with_key(&base, &base, &r, 50, &pk).is_err());
        }
        assert!(validate_transition_with_key(&base, &base, &original, 100, &pk).is_err());
        assert!(validate_transition_with_key(
            &base,
            &base,
            &original,
            50,
            &base.authority_public_key
        )
        .is_err());
    }
    #[test]
    fn successor_requires_separate_completion_and_exact6000_window() {
        let (_t, _root, _key, base, _provisioner) = native_grants::tests::fixture();
        let reviewer = SigningKey::from_bytes(&[81; 32]);
        let pk = hex::encode(reviewer.verifying_key().to_bytes());
        let r = transition(
            &base,
            &reviewer,
            EpochTransitionMode::ReviewedNewRunSuccessor,
        );
        validate_transition_with_key(&base, &base, &r, 50, &pk).unwrap();
        assert!(r.body.next_policy.validate().is_err()); // not an unextended base
        let mut bad = r.clone();
        bad.body.previous_run_completion = None;
        bad.signature = sign(&reviewer, TRANSITION_DOMAIN, &bad.body).unwrap();
        assert!(validate_transition_with_key(&base, &base, &bad, 50, &pk).is_err());
        let mut bad = r;
        bad.body
            .next_policy
            .native_grant_delegation
            .as_mut()
            .unwrap()
            .initial_optimizer_step = 63000;
        bad.signature = sign(&reviewer, TRANSITION_DOMAIN, &bad.body).unwrap();
        assert!(validate_transition_with_key(&base, &base, &bad, 50, &pk).is_err());
    }
    #[test]
    fn signed_original_transition_ack_binds_head_budget_policy_and_offline_time() {
        let (_t, _root, _key, base, _provisioner) = native_grants::tests::fixture();
        let reviewer = SigningKey::from_bytes(&[81; 32]);
        let pk = hex::encode(reviewer.verifying_key().to_bytes());
        let r = transition(&base, &reviewer, EpochTransitionMode::SameWindowRecovery);
        let ack_key = SigningKey::from_bytes(&[42; 32]);
        let body = WitnessEpochTransitionAckBody {
            schema: "parallel.witness_epoch_transition_ack.v1".into(),
            immutable_base_policy_sha256: base.sha256().unwrap(),
            witness_instance: base.witness_instance.clone(),
            request_sha256: sha(&bytes(&r).unwrap()),
            effective_policy_sha256: r.body.next_policy.sha256().unwrap(),
            witness_sequence: 8,
            previous_record_sha256: r.body.previous_witness_tip_sha256.clone(),
            committed_unix_ms: 50,
        };
        let p = WitnessEpochProof {
            epoch_transition: r,
            ack: SignedWitnessEpochTransitionAck {
                signature: sign(&ack_key, TRANSITION_ACK_DOMAIN, &body).unwrap(),
                body,
            },
        };
        let effective =
            verify_epoch_chain_checked(&base, std::slice::from_ref(&p), Some(&pk)).unwrap();
        assert_eq!(
            effective.authority_epoch,
            p.epoch_transition.body.next_policy.authority_epoch
        );
        let mut altered = p.clone();
        altered.ack.body.previous_record_sha256 = sha(b"other ledger");
        assert!(verify_epoch_chain_checked(&base, &[altered], Some(&pk)).is_err());
        assert!(verify_epoch_chain_checked(&base, &[p.clone(), p], Some(&pk)).is_err());
    }
    #[test]
    fn initial_bootstrap_preserves_original_static_budget_and_unknown_identities() {
        let (_t, _root, _key, mut base, _provisioner) = native_grants::tests::fixture();
        let reviewer = SigningKey::from_bytes(&[81; 32]);
        let pk = hex::encode(reviewer.verifying_key().to_bytes());
        let mut r = transition(&base, &reviewer, EpochTransitionMode::SameWindowRecovery);
        base.native_grant_delegation = None;
        base.native_fit_trust = Some(NativeWitnessTrust {
            authorization_public_key: hex::encode(
                SigningKey::from_bytes(&[82; 32]).verifying_key().to_bytes(),
            ),
            grant_sha256: sha(b"original static grant"),
        });
        r.body.immutable_base_policy_sha256 = base.sha256().unwrap();
        r.body.previous_effective_policy_sha256 = base.sha256().unwrap();
        r.body.mode = EpochTransitionMode::InitialDelegationBootstrap;
        r.body.original_reserved_evaluations = 0;
        r.body.next_update_sequence = 1;
        r.body.previous_update_sha256 = ZERO.into();
        r.body.historical_authorization_count = 15;
        let accounting = HistoricalBudgetAccounting {
            schema: "jope.original_static_grant_budget.v1".into(),
            original_policy_sha256: base.sha256().unwrap(),
            original_static_grant_sha256: sha(b"original static grant"),
            original_approved_candidate_budget: 16,
            original_authorization_count: 15,
            physical_starts: None,
            unknown_or_held: BTreeMap::from([(
                sha(b"retained ambiguous ridge"),
                r.body.authority_recovery_receipt.clone(),
            )]),
            source_closure: vec![r.body.authority_recovery_receipt.clone()],
            authority_recovery_receipt_sha256: r.body.authority_recovery_receipt.sha256.clone(),
        };
        r.body.historical_budget_accounting = Some(NativeGrantArtifact {
            path: "/data/review/original-budget.json".into(),
            sha256: sha(&bytes(&accounting).unwrap()),
        });
        r.body.historical_accounting = Some(accounting);
        r.signature = sign(&reviewer, TRANSITION_DOMAIN, &r.body).unwrap();
        validate_transition_with_key(&base, &base, &r, 50, &pk).unwrap();
        let mut missing = r.clone();
        missing.body.historical_accounting = None;
        missing.signature = sign(&reviewer, TRANSITION_DOMAIN, &missing.body).unwrap();
        assert!(validate_transition_with_key(&base, &base, &missing, 50, &pk).is_err());
        let mut changed = r;
        changed
            .body
            .historical_accounting
            .as_mut()
            .unwrap()
            .unknown_or_held
            .clear();
        changed.signature = sign(&reviewer, TRANSITION_DOMAIN, &changed.body).unwrap();
        assert!(validate_transition_with_key(&base, &base, &changed, 50, &pk).is_err());
    }
}
