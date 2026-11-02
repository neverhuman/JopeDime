use super::*;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordCore {
    schema: String,
    sequence: u64,
    previous: String,
    request: SignedRequest,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    core: RecordCore,
    ack: SignedAck,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Head {
    sequence: u64,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalEvent {
    instance: String,
    seq: u64,
    previous: String,
    kind: String,
    payload: Value,
    effects: Vec<OriginalEffect>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalEffect {
    table: String,
    key: String,
    row: Value,
}
pub(super) struct EventIdentity {
    pub(super) seq: u64,
    pub(super) deadline: u64,
}
#[cfg(test)]
pub(super) fn event_identity(policy: &WitnessPolicy, event: &[u8]) -> Result<EventIdentity> {
    event_identity_effective(policy, policy, event)
}
pub(super) fn event_identity_effective(
    base: &WitnessPolicy,
    policy: &WitnessPolicy,
    event: &[u8],
) -> Result<EventIdentity> {
    ensure!(
        !event.is_empty() && event.len() < MAX_FRAME / 2,
        "invalid authorization event size"
    );
    let value: OriginalEvent = serde_json::from_slice(event)?;
    ensure!(
        value.instance == policy.authority_instance
            && value.seq > 0
            && valid_hash(&value.previous)
            && matches!(
                value.kind.as_str(),
                "authorize" | "funding_start" | "funding_participant"
            )
            && value.effects.len() == 1,
        "not an original authorization event"
    );
    let effect = &value.effects[0];
    let row = &effect.row;
    if value.kind != "authorize" {
        let trust = policy
            .funding_trust
            .as_ref()
            .context("funding witness trust not installed")?;
        let original: super::funding::FundingEvent =
            serde_json::from_value(value.payload["original"].clone())?;
        let (event_key, deadline) = super::funding::validate_witness_event(
            trust,
            &original,
            &policy.authority_instance,
            &policy.authority_epoch,
        )?;
        let admission_deadline = value.payload["admission_deadline_monotonic_ns"]
            .as_u64()
            .context("original funding admission deadline absent")?;
        ensure!(
            original.claimed_monotonic_ns < admission_deadline && admission_deadline <= deadline,
            "funding admission deadline is outside original task window"
        );
        ensure!(
            value.payload["event_key"] == event_key
                && value.payload["reservation_id"] == original.reservation_id(),
            "funding journal key mismatch"
        );
        match original.participant_slot {
            None => {
                let reservation: super::funding::SignedReservation = serde_json::from_str(
                    row["json"].as_str().context("signed reservation absent")?,
                )?;
                let stored: super::funding::FundingEvent = serde_json::from_str(
                    row["start_json"]
                        .as_str()
                        .context("original funding start absent")?,
                )?;
                ensure!(
                    value.kind == "funding_start"
                        && effect.table == "funding_reservations"
                        && effect.key == original.reservation_id()
                        && row["id"] == effect.key
                        && row["state"] == "started"
                        && row["task_sha256"] == original.demand.task_sha256
                        && reservation == original.reservation
                        && row["authorization_id"]
                            == super::funding::digest(&original.authorization.body)?
                        && stored == original,
                    "funding start postimage differs from original"
                );
            }
            Some(slot) => {
                let id = original.participant_id()?;
                let stored: super::funding::FundingEvent = serde_json::from_str(
                    row["json"]
                        .as_str()
                        .context("original funding participant absent")?,
                )?;
                ensure!(
                    value.kind == "funding_participant"
                        && effect.table == "funding_participants"
                        && effect.key == id
                        && row["id"] == id
                        && value.payload["participant_id"] == id
                        && row["reservation_id"] == original.reservation_id()
                        && row["slot"].as_str() == Some(slot.to_string().as_str())
                        && stored == original,
                    "funding participant postimage differs from original"
                );
            }
        }
        return Ok(EventIdentity {
            seq: value.seq,
            deadline: admission_deadline,
        });
    }
    ensure!(
        effect.table == "entries"
            && row["key"] == effect.key
            && row["state"] == "authorized"
            && row["epoch"] == policy.authority_epoch,
        "authorization epoch/state mismatch"
    );
    let start: super::StartToken = serde_json::from_str(
        row["start_json"]
            .as_str()
            .context("original full start absent")?,
    )?;
    let identity: super::FitIdentity = serde_json::from_str(
        row["identity_json"]
            .as_str()
            .context("original fit identity absent")?,
    )?;
    if let Some(native) = value.payload.get("native_fit") {
        let delegated;
        let trust = if let Some(value) = native.get("native_grant_proof").filter(|v| !v.is_null()) {
            let proof: NativeGrantProof = serde_json::from_value(value.clone())?;
            delegated = verify_native_grant_proof(base, &proof)?;
            ensure!(
                verify_epoch_chain(base, &proof.epoch_proofs)?.sha256()? == policy.sha256()?,
                "native grant epoch differs from event epoch"
            );
            ensure!(
                native["signed_grant"] == serde_json::to_value(&proof.request.body.signed_grant)?,
                "native event grant differs from witnessed delegation"
            );
            &delegated
        } else {
            policy
                .native_fit_trust
                .as_ref()
                .context("native witness trust unavailable")?
        };
        super::native_scq::verify_witness_event(trust, native, &identity, &start, policy)?;
    }
    ensure!(
        start.fit_key == effect.key
            && sha(&bytes(&identity)?) == effect.key
            && identity.executable_sha256 == policy.authority_executable_sha256
            && value.payload["fit_key"] == effect.key
            && value.payload["token"] == start.token
            && value.payload["executor"] == serde_json::to_value(&start.executor)?
            && value.payload["claim_token"] == row["claim_token"]
            && start.task_id == row["task_id"]
            && start.deadline_monotonic_ns > start.authorized_monotonic_ns,
        "authorization original identity/token/custody mismatch"
    );
    Ok(EventIdentity {
        seq: value.seq,
        deadline: start.deadline_monotonic_ns,
    })
}
fn verify_request(base: &WitnessPolicy, request: &SignedRequest) -> Result<EventIdentity> {
    let effective = verify_epoch_chain(base, &request.body.epoch_proofs)?;
    let policy = &effective;
    ensure!(
        request.body.schema == "parallel.fit_event_witness_request.v1"
            && request.body.policy_sha256 == policy.sha256()?,
        "witness request policy mismatch"
    );
    verify(
        &policy.authority_public_key,
        "parallel.fit_event_witness.request.v1",
        &request.body,
        &request.signature,
    )?;
    event_identity_effective(base, policy, &request.body.event)
}
impl SignedAck {
    pub fn verify(&self, policy: &WitnessPolicy, request: &SignedRequest) -> Result<()> {
        let event = verify_request(policy, request)?;
        let base = policy;
        let effective = verify_epoch_chain(base, &request.body.epoch_proofs)?;
        let policy = &effective;
        ensure!(
            self.body.schema == "parallel.fit_event_witness_ack.v1"
                && self.body.policy_sha256 == policy.sha256()?
                && self.body.witness_instance == policy.witness_instance
                && self.body.authority_instance == policy.authority_instance
                && self.body.authority_epoch == policy.authority_epoch
                && self.body.authority_sequence == event.seq
                && self.body.event_sha256 == sha(&request.body.event)
                && self.body.request_sha256 == sha(&bytes(request)?)
                && self.body.witness_sequence > 0
                && valid_hash(&self.body.previous_record_sha256)
                && valid_hash(&self.body.committed_record_core_sha256),
            "witness ACK does not bind exact original authorization"
        );
        let original: OriginalEvent = serde_json::from_slice(&request.body.event)?;
        if let Some(value) = original
            .payload
            .get("native_fit")
            .and_then(|v| v.get("native_grant_proof"))
            .filter(|v| !v.is_null())
        {
            let proof: NativeGrantProof = serde_json::from_value(value.clone())?;
            verify_native_grant_proof(base, &proof)?;
            ensure!(
                proof.ack.body.witness_sequence < self.body.witness_sequence,
                "native grant update does not precede authorization ACK"
            );
        }
        verify(
            &policy.witness_public_key,
            "parallel.fit_event_witness.ack.v1",
            &self.body,
            &self.signature,
        )
    }
}

/// Storage stays outside the authority database restore domain. No startup
/// creates storage, repairs missing committed tails, or erases unknown files.
pub struct WitnessStore {
    pub(super) root: PathBuf,
    pub(super) policy: WitnessPolicy,
    pub(super) signing: Option<SigningKey>,
    pub(super) _lease: File,
    pub(super) records: BTreeMap<u64, (String, SignedAck, String)>,
    pub(super) native_grants: native_grants::NativeGrantState,
    pub(super) epoch_proofs: Vec<WitnessEpochProof>,
    pub(super) sequence: u64,
    pub(super) tip: String,
    pub(super) root_identity: (u64, u64),
    pub(super) segment_identity: (u64, u64),
    pub(super) lease_identity: (u64, u64),
    pub(super) read_only: bool,
}
impl WitnessStore {
    /// Verify a frozen public copy without access to an ACK key or the original executable.
    /// Original signed records, exact head and immutable policy remain mandatory.
    pub fn verify_public_history(root: &Path, policy: &WitnessPolicy) -> Result<Value> {
        Self::verify_public_history_with_acks(root, policy, &[])
    }
    /// Exact membership; extra committed witness-before-local-ACK events remain.
    pub fn verify_public_history_with_acks(
        root: &Path,
        policy: &WitnessPolicy,
        required: &[SignedAck],
    ) -> Result<Value> {
        ensure!(
            required.len() <= 100_000 && bytes(&required)?.len() <= 64 * 1024 * 1024,
            "public witness membership bound"
        );
        policy.validate()?;
        ensure!(
            root.is_absolute() && fs::canonicalize(root)? == root,
            "public witness snapshot must be physical"
        );
        ensure!(
            read(&root.join("policy.json"))? == bytes(policy)?,
            "public witness original policy changed"
        );
        let lease = checked_file(&root.join("lease"))?;
        ensure!(
            lease.try_lock_shared().is_ok(),
            "public witness snapshot still has an active writer"
        );
        let identity = |p: &Path| -> Result<(u64, u64)> {
            let m = fs::symlink_metadata(p)?;
            ensure!(
                !m.file_type().is_symlink(),
                "public witness snapshot symlink"
            );
            Ok((m.dev(), m.ino()))
        };
        let head = read(&root.join("head.json"))?;
        let mut store = Self {
            root: root.into(),
            policy: policy.clone(),
            signing: None,
            _lease: lease,
            records: BTreeMap::new(),
            native_grants: native_grants::NativeGrantState::default(),
            epoch_proofs: Vec::new(),
            sequence: 0,
            tip: ZERO.into(),
            root_identity: identity(root)?,
            segment_identity: identity(&root.join("segments"))?,
            lease_identity: identity(&root.join("lease"))?,
            read_only: true,
        };
        store.replay()?;
        store.verify_custody()?;
        let expected: Head = serde_json::from_slice(&head)?;
        ensure!(
            expected.sequence == store.sequence
                && expected.sha256 == store.tip
                && read(&root.join("head.json"))? == head,
            "public witness snapshot is incomplete or changed during verification"
        );
        let mut required_sequences = std::collections::BTreeSet::new();
        for ack in required {
            ensure!(
                required_sequences.insert(ack.body.authority_sequence),
                "duplicate required original ACK"
            );
            let (_, original, _) = store
                .records
                .get(&ack.body.authority_sequence)
                .context("public witness snapshot omits original authority ACK")?;
            ensure!(
                original == ack,
                "public witness ACK differs from original authority receipt"
            );
        }
        Ok(
            serde_json::json!({"schema":"parallel.public_witness_history_verification.v1",
            "policy_sha256":policy.sha256()?,"authority_instance":policy.authority_instance,
            "authority_epoch":policy.authority_epoch,"witness_instance":policy.witness_instance,
            "sequence":store.sequence,"tip_sha256":store.tip,"head_sha256":sha(&head),
            "original_authorization_events":store.records.len(),
            "required_ack_count":required.len(),"required_acks_sha256":sha(&bytes(&required)?),
            "effective_policy_sha256":store.effective_policy().sha256()?,"epoch_proofs":store.epoch_proofs,"epoch_status":store.epoch_status()?,"private_key_accessed":false,
            "new_acknowledgement_issued":false}),
        )
    }
    pub fn initialize(root: &Path, policy: &WitnessPolicy) -> Result<()> {
        policy.validate()?;
        ensure!(!root.exists(), "witness already exists");
        fs::create_dir(root)?;
        fs::set_permissions(root, fs::Permissions::from_mode(0o700))?;
        fs::create_dir(root.join("segments"))?;
        publish(&root.join("policy.json"), &bytes(policy)?)?;
        publish(&root.join("lease"), b"")?;
        publish(
            &root.join("head.json"),
            &bytes(&Head {
                sequence: 0,
                sha256: ZERO.into(),
            })?,
        )?;
        sync_dir(root)?;
        sync_dir(root.parent().context("witness root parent absent")?)
    }
    pub fn open(root: &Path, policy: &WitnessPolicy, key_path: &Path) -> Result<Self> {
        Self::open_mode(root, policy, key_path, false)
    }
    pub fn open_read_only(root: &Path, policy: &WitnessPolicy, key_path: &Path) -> Result<Self> {
        Self::open_mode(root, policy, key_path, true)
    }
    fn open_mode(
        root: &Path,
        policy: &WitnessPolicy,
        key_path: &Path,
        read_only: bool,
    ) -> Result<Self> {
        Self::open_mode_transition(root, policy, key_path, read_only, None)
    }
    fn open_mode_transition(
        root: &Path,
        policy: &WitnessPolicy,
        key_path: &Path,
        read_only: bool,
        transition: Option<&SignedWitnessEpochTransition>,
    ) -> Result<Self> {
        policy.validate()?;
        if let Some(request) = transition {
            crate::native_review::require_signer(&request.authorization_public_key)?;
            verify(
                &request.authorization_public_key,
                "parallel.witness_epoch_transition.v1",
                &request.body,
                &request.signature,
            )?;
            ensure!(
                request.body.immutable_base_policy_sha256 == policy.sha256()?,
                "transition immutable base differs"
            );
        }
        ensure!(
            !fs::symlink_metadata(root)?.file_type().is_symlink(),
            "witness root symlink"
        );
        let root = fs::canonicalize(root)?;
        let key_path = fs::canonicalize(key_path)?;
        ensure!(
            !key_path.starts_with(&root),
            "ACK private key must remain outside recovery copy"
        );
        ensure!(
            read(&root.join("policy.json"))? == bytes(policy)?,
            "witness policy changed"
        );

        let signing = private_key(&key_path)?;
        ensure!(
            hex::encode(signing.verifying_key().to_bytes()) == policy.witness_public_key,
            "wrong witness ACK key"
        );
        let lease = checked_file(&root.join("lease"))?;
        ensure!(lease.try_lock().is_ok(), "witness already owned");
        let identity = |p: &Path| -> Result<(u64, u64)> {
            let m = fs::symlink_metadata(p)?;
            Ok((m.dev(), m.ino()))
        };
        let root_identity = identity(&root)?;
        let segment_identity = identity(&root.join("segments"))?;
        let lease_identity = identity(&root.join("lease"))?;
        let mut store = Self {
            root,
            policy: policy.clone(),
            signing: Some(signing),
            _lease: lease,
            records: BTreeMap::new(),
            native_grants: native_grants::NativeGrantState::default(),
            epoch_proofs: Vec::new(),
            sequence: 0,
            tip: ZERO.into(),
            root_identity,
            segment_identity,
            lease_identity,
            read_only: true,
        };
        store.replay()?;
        let runtime = if let Some(request) = transition {
            if let Some(original) = store.epoch_proofs.iter().find(|p| {
                p.epoch_transition.body.next_policy.authority_epoch
                    == request.body.next_policy.authority_epoch
            }) {
                ensure!(
                    bytes(&original.epoch_transition)? == bytes(request)?,
                    "conflicting transition runtime retry"
                );
                store.effective_policy()
            } else {
                store.check_epoch_next(request, super::native_scq::policy::now_ms()?)?;
                &request.body.next_policy
            }
        } else {
            store.effective_policy()
        };
        ensure!(
            sha(&fs::read("/proc/self/exe")?) == runtime.witness_executable_sha256
                && source_sha256() == runtime.witness_source_sha256,
            "unapproved effective witness executable/source"
        );
        store.read_only = read_only;
        let head: Head = serde_json::from_slice(&read(&store.root.join("head.json"))?)?;
        if !read_only && head.sequence != store.sequence {
            store.publish_head()?;
        }
        Ok(store)
    }
    fn replay(&mut self) -> Result<()> {
        let directory = self.root.join("segments");
        ensure!(
            fs::symlink_metadata(&directory)?.is_dir()
                && !fs::symlink_metadata(&directory)?.file_type().is_symlink(),
            "witness segment custody changed"
        );
        let head: Head = serde_json::from_slice(&read(&self.root.join("head.json"))?)?;
        let mut paths = Vec::new();
        for entry in fs::read_dir(&directory)? {
            ensure!(
                paths.len() < 1_000_000,
                "witness history segment count bound"
            );
            paths.push(entry?.path());
        }
        paths.sort();
        let mut matched = head.sequence == 0 && head.sha256 == ZERO;
        for path in paths {
            let expected = format!("{:020}.json", self.sequence + 1);
            ensure!(
                path.file_name().and_then(|v| v.to_str()) == Some(expected.as_str()),
                "unresolved witness segment/gap"
            );
            let data = read(&path)?;
            let shape: Value = serde_json::from_slice(&data)?;
            if shape.get("epoch_transition").is_some() {
                self.replay_epoch_transition(serde_json::from_slice(&data)?, sha(&data))?;
                if head.sequence == self.sequence {
                    ensure!(
                        head.sha256 == self.tip,
                        "witness committed epoch head changed"
                    );
                    matched = true;
                }
                continue;
            }
            if shape.get("request").is_some() {
                self.replay_native_grant(serde_json::from_slice(&data)?, sha(&data))?;
                if head.sequence == self.sequence {
                    ensure!(
                        head.sha256 == self.tip,
                        "witness committed grant head changed"
                    );
                    matched = true;
                }
                continue;
            }
            let record: Record = serde_json::from_slice(&data)?;
            ensure!(
                bytes(&record.core.request.body.epoch_proofs)? == bytes(&self.epoch_proofs)?,
                "authorization epoch not active at original journal position"
            );
            let event = verify_request(&self.policy, &record.core.request)?;
            record.ack.verify(&self.policy, &record.core.request)?;
            ensure!(
                record.core.schema == "parallel.fit_event_witness_record.v1"
                    && record.core.sequence == self.sequence + 1
                    && record.core.previous == self.tip
                    && record.ack.body.witness_sequence == record.core.sequence
                    && record.ack.body.previous_record_sha256 == record.core.previous
                    && record.ack.body.committed_record_core_sha256 == sha(&bytes(&record.core)?),
                "witness record chain mismatch"
            );
            ensure!(
                self.records
                    .last_key_value()
                    .is_none_or(|(seq, _)| event.seq > *seq),
                "replayed/conflicting authority sequence"
            );
            let work =
                self.native_grants
                    .check_work(&self.policy, &record.core.request, self.sequence)?;
            self.native_grants.commit_work(work, event.seq);
            self.sequence = record.core.sequence;
            self.tip = sha(&data);
            self.records.insert(
                event.seq,
                (sha(&record.core.request.body.event), record.ack, sha(&data)),
            );
            if head.sequence == self.sequence {
                ensure!(head.sha256 == self.tip, "witness committed head changed");
                matched = true;
            }
        }
        ensure!(matched, "witness committed tail missing");
        // A valid durable segment may precede a lost head publication. Adopt the
        // original signed record; no authorization is generated by this replay.
        if head.sequence != self.sequence && !self.read_only {
            self.publish_head()?;
        }
        Ok(())
    }
    fn publish_head(&self) -> Result<()> {
        let temporary = self.root.join("head.pending");
        // An interrupted head update remains explicit until it is proven to be
        // the exact derived head, then it is safe to finish its rename.
        let expected = bytes(&Head {
            sequence: self.sequence,
            sha256: self.tip.clone(),
        })?;
        if temporary.exists() {
            ensure!(
                read(&temporary)? == expected,
                "conflicting pending witness head"
            );
        } else {
            publish(&temporary, &expected)?;
        }
        fs::rename(temporary, self.root.join("head.json"))?;
        sync_dir(&self.root)
    }
    fn verify_custody(&self) -> Result<()> {
        for (path, expected) in [
            (&self.root, self.root_identity),
            (&self.root.join("segments"), self.segment_identity),
            (&self.root.join("lease"), self.lease_identity),
        ] {
            let m = fs::symlink_metadata(path)?;
            ensure!(
                !m.file_type().is_symlink() && (m.dev(), m.ino()) == expected,
                "witness owned storage replaced"
            );
        }
        ensure!(
            read(&self.root.join("policy.json"))? == bytes(&self.policy)?,
            "witness policy replaced"
        );
        Ok(())
    }
    fn verify_retained(&self, ack: &SignedAck, digest: &str) -> Result<()> {
        self.verify_custody()?;
        ensure!(
            sha(&read(
                &self
                    .root
                    .join("segments")
                    .join(format!("{:020}.json", ack.body.witness_sequence))
            )?) == digest,
            "original witness segment changed"
        );
        Ok(())
    }
    pub fn record_authorization(&mut self, request: &SignedRequest) -> Result<SignedAck> {
        ensure!(
            !self.read_only,
            "read-only witness cannot record an authorization"
        );
        self.verify_custody()?;
        let event = verify_request(&self.policy, request)?;
        ensure!(
            read(&self.root.join("policy.json"))? == bytes(&self.policy)?,
            "witness policy replaced"
        );
        if let Some((original, ack, record_sha)) = self.records.get(&event.seq) {
            self.verify_retained(ack, record_sha)?;
            ensure!(
                *original == sha(&request.body.event)
                    && ack.body.request_sha256 == sha(&bytes(request)?),
                "conflicting authorization retransmission"
            );
            return Ok(ack.clone());
        }
        ensure!(
            bytes(&request.body.epoch_proofs)? == bytes(&self.epoch_proofs)?,
            "new authorization uses stale epoch"
        );
        ensure!(
            self.records
                .last_key_value()
                .is_none_or(|(seq, _)| event.seq > *seq),
            "out-of-order original authorization"
        );
        let work = self
            .native_grants
            .check_work(&self.policy, request, self.sequence)?;
        let core = RecordCore {
            schema: "parallel.fit_event_witness_record.v1".into(),
            sequence: self.sequence + 1,
            previous: self.tip.clone(),
            request: request.clone(),
        };
        let effective = self.effective_policy();
        let body = AckBody {
            schema: "parallel.fit_event_witness_ack.v1".into(),
            policy_sha256: effective.sha256()?,
            witness_instance: effective.witness_instance.clone(),
            authority_instance: effective.authority_instance.clone(),
            authority_epoch: effective.authority_epoch.clone(),
            authority_sequence: event.seq,
            event_sha256: sha(&request.body.event),
            request_sha256: sha(&bytes(request)?),
            witness_sequence: core.sequence,
            previous_record_sha256: core.previous.clone(),
            committed_record_core_sha256: sha(&bytes(&core)?),
        };
        let ack = SignedAck {
            signature: sign(
                self.signing
                    .as_ref()
                    .context("public witness verifier cannot sign")?,
                "parallel.fit_event_witness.ack.v1",
                &body,
            )?,
            body,
        };
        let record = Record {
            core,
            ack: ack.clone(),
        };
        let data = bytes(&record)?;
        publish(
            &self
                .root
                .join("segments")
                .join(format!("{:020}.json", record.core.sequence)),
            &data,
        )?;
        self.native_grants.commit_work(work, event.seq);
        self.sequence = record.core.sequence;
        self.tip = sha(&data);
        self.records.insert(
            event.seq,
            (sha(&request.body.event), ack.clone(), sha(&data)),
        );
        self.publish_head()?;
        Ok(ack)
    }
    /// An ACK query is an exact signed retransmission, never a request to create
    /// a different event or restamp its original authorization.
    pub fn get_ack(&self, request: &SignedRequest) -> Result<Option<SignedAck>> {
        let event = verify_request(&self.policy, request)?;
        match self.records.get(&event.seq) {
            Some((digest, ack, record_sha)) => {
                self.verify_retained(ack, record_sha)?;
                ensure!(
                    *digest == sha(&request.body.event)
                        && ack.body.request_sha256 == sha(&bytes(request)?),
                    "conflicting ACK query"
                );
                Ok(Some(ack.clone()))
            }
            None => Ok(None),
        }
    }
}
