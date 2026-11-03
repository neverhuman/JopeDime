use super::*;

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
        let now = crate::eval_registry::native_scq::policy::now_ms()?;
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
