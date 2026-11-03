use super::*;

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
                store.check_epoch_next(
                    request,
                    crate::eval_registry::native_scq::policy::now_ms()?,
                )?;
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
