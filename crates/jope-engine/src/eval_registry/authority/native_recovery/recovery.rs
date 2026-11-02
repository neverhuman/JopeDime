use super::*;

impl Authority {
    /// Read-only, journal-verified epoch status. Normal in-flight claims are retained,
    /// not interpreted as recovery failures. No database or witness writes occur.
    pub fn native_epoch_status(root: &Path, witness: &Path) -> Result<Value> {
        ensure!(
            fs::canonicalize(root)? == root && fs::canonicalize(witness)? == witness,
            "epoch status requires original physical authority/witness paths"
        );
        ensure!(
            !root.join("recovery.pending").exists(),
            "reconciliation hold blocks all learning work"
        );
        let _lock = FileLock::acquire(&root.join("authority.lock"), false, false)?;
        let bytes = policy::read_file(&root.join("authority.json"))?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        ensure!(
            manifest.schema == NATIVE_SCHEMA && manifest.witness_path == witness,
            "epoch status requires original native authority binding"
        );
        regular(&root.join("registry.sqlite"))?;
        let db = Connection::open_with_flags(
            root.join("registry.sqlite"),
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        db.busy_timeout(Duration::from_secs(2))?;
        ensure!(
            db.query_row("PRAGMA user_version", [], |r| r.get::<_, u32>(0))? == 6,
            "native epoch schema changed"
        );
        let initial = Checkpoint {
            instance: manifest.instance.clone(),
            seq: 0,
            tip: ZERO.into(),
            length: 0,
        };
        let mut authority = Self {
            root: root.into(),
            witness: witness.into(),
            db,
            manifest,
            db_inode: identity(&root.join("registry.sqlite"))?,
            journal_inode: identity(&root.join("dispatch.jsonl"))?,
            lock_inode: identity(&root.join("authority.lock"))?,
            checkpoint: initial,
            manifest_sha: hash(&bytes),
            data_version: 0,
            recovery_mode: false,
            event_witness: None,
            funding_installation: None,
            #[cfg(test)]
            failpoint: None,
        };
        authority.verify_tail()?;
        authority.verify_full_state()?;
        let current: Option<(String, String)> = authority
            .db
            .query_row(
                "SELECT epoch,report_sha FROM recovery_epochs WHERE id='current'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let (epoch, receipt) = if let Some((epoch, sha)) = current {
            let bytes = authority.read_blob(&sha)?;
            let receipt: Receipt = serde_json::from_slice(&bytes)?;
            let (signed, original) = load_policy(&receipt.policy.path, receipt.committed_unix_ms)?;
            ensure!(
                original.sha256 == receipt.policy.sha256
                    && signed.body.new_epoch == epoch
                    && signed.body.authority_instance == authority.manifest.instance
                    && signed.body.authority_manifest_sha256 == authority.manifest_sha,
                "current recovery epoch is not independently reviewed"
            );
            (epoch, Some(sha))
        } else {
            (authority.manifest.instance.clone(), None)
        };
        ensure!(
            !root.join("recovery.pending").exists(),
            "reconciliation hold changed during verification"
        );
        Ok(
            json!({"schema":"jope.native_epoch_status.v1","authority_instance":authority.manifest.instance,
            "authority_manifest_sha256":authority.manifest_sha,"authority_epoch":epoch,
            "recovery_receipt_sha256":receipt,"checkpoint":authority.checkpoint,"read_only":true}),
        )
    }
    pub fn native_recovery_candidate(root: &Path, witness: &Path) -> Result<Value> {
        let mut authority = Self::open_storage(root, witness, true)?;
        let _lock = FileLock::acquire(&authority.root.join("authority.lock"), false, false)?;
        ensure!(
            authority.manifest.schema == NATIVE_SCHEMA,
            "native recovery requires v6"
        );
        authority.verify_tail()?;
        authority.verify_full_state()?;
        let evidence = authority.recovery_evidence()?;
        let report = authority.recovery_report(&evidence)?;
        let requests = original_requests(&authority)?;
        let mut held = BTreeMap::new();
        for item in evidence.expected.iter()? {
            let ((table, key), row) = item?;
            let (table, key, row) = (table.as_str(), key.as_str(), &row);
            if table == "entries"
                && matches!(
                    row["state"].as_str(),
                    Some("claimed" | "authorized" | "running")
                )
            {
                let request = requests.get(key).context("original claim request absent")?;
                held.insert(key.to_owned(),json!({"row_sha256":hash(&encoded(row)?),"request_sha256":hash(&encoded(request)?),
                    "original_entry":row,"original_request":request}));
            }
        }
        let epoch:String=authority.db.query_row("SELECT COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),(SELECT instance FROM meta))",[],|r|r.get(0))?;
        Ok(
            json!({"schema":"jope.native_held_recovery_candidate.v1","authority_root":authority.root,"witness":authority.witness,
            "authority_instance":authority.manifest.instance,"authority_manifest_sha256":authority.manifest_sha,"parent_epoch":epoch,
            "report_sha256":hash(&encoded(&report)?),"report":report,"held":held,"activation_performed":false}),
        )
    }
    /// Commit only the separately signed native transition; old rows/tokens stay unchanged.
    pub fn commit_native_held_epoch(
        root: &Path,
        witness: &Path,
        policy_path: &Path,
    ) -> Result<String> {
        let now = policy::now_ms()?;
        let bytes = policy::protected(policy_path)?;
        let signed: SignedNativeRecovery = serde_json::from_slice(&bytes)?;
        // Signature authenticity is checked separately from fresh-commit validity:
        // an already committed original retry may arrive after the approval window.
        crate::native_review::require_signer(&signed.authorization_public_key)?;
        verify_signed(&signed, signed.body.not_before_unix_ms)?;
        let policy_reference = RecoveryArtifact {
            path: policy_path.into(),
            sha256: hash(&bytes),
        };
        let body = &signed.body;
        let mut authority = Self::open_storage(root, witness, true)?;
        let _lock = FileLock::acquire(&authority.root.join("authority.lock"), true, false)?;
        ensure!(
            authority.manifest.schema == NATIVE_SCHEMA
                && authority.root == body.authority_root
                && authority.witness == body.witness
                && authority.manifest.instance == body.authority_instance
                && authority.manifest_sha == body.authority_manifest_sha256,
            "native recovery authority binding differs"
        );
        authority.verify_tail()?;
        authority.verify_full_state()?;
        let epoch:String=authority.db.query_row("SELECT COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),(SELECT instance FROM meta))",[],|r|r.get(0))?;
        if epoch == body.new_epoch {
            let sha: String = authority.db.query_row(
                "SELECT report_sha FROM recovery_epochs WHERE id='current'",
                [],
                |r| r.get(0),
            )?;
            let receipt: Receipt = serde_json::from_slice(&authority.read_blob(&sha)?)?;
            ensure!(
                receipt.schema == "jope.native_held_epoch_receipt.v1"
                    && receipt.policy.sha256 == policy_reference.sha256,
                "conflicting duplicate epoch transition"
            );
            verify_signed(&signed, receipt.committed_unix_ms)?;
            if authority.root.join("recovery.pending").exists() {
                authority.native_held_start_allowed()?;
                fs::remove_file(authority.root.join("recovery.pending"))?;
                sync_dir(&authority.root)?;
            }
            return Ok(epoch);
        }
        verify_signed(&signed, now)?;
        ensure!(
            epoch == body.parent_epoch && authority.root.join("recovery.pending").exists(),
            "original reconciliation hold/parent epoch absent"
        );
        ensure!(
            body.new_epoch != authority.manifest.instance,
            "cannot reuse original authority epoch"
        );
        for line in BufReader::new(File::open(authority.root.join("dispatch.jsonl"))?).lines() {
            let event: Event = serde_json::from_str(&line?)?;
            ensure!(
                !matches!(
                    event.kind.as_str(),
                    "recovery_commit_epoch" | "native_recovery_commit_held_epoch"
                ) || event.payload["epoch"] != body.new_epoch,
                "recovery epoch was already used in original history"
            );
        }
        let witness_policy: super::super::witness::WitnessPolicy =
            serde_json::from_slice(&artifact(&body.independent_witness.policy)?)?;
        ensure!(
            witness_policy.authority_instance == body.authority_instance,
            "independent witness snapshot belongs to another original authority/epoch"
        );
        let mut required_acks = Vec::new();
        for query in [
            "SELECT ack_sha FROM authorization_witness ORDER BY ack_sha",
            "SELECT ack_sha FROM funding_witness ORDER BY ack_sha",
        ] {
            let mut statement = authority.db.prepare(query)?;
            let rows = statement.query_map([], |r| r.get::<_, String>(0))?;
            for row in rows {
                ensure!(
                    required_acks.len() < 100_000,
                    "public ACK membership bound exceeded"
                );
                let ack: super::super::witness::SignedAck =
                    serde_json::from_slice(&authority.read_blob(&row?)?)?;
                required_acks.push(ack);
            }
        }
        let independent_witness_verification =
            super::super::witness::WitnessStore::verify_public_history_with_acks(
                &body.independent_witness.root,
                &witness_policy,
                &required_acks,
            )?;
        let epochs: Vec<super::super::witness::WitnessEpochProof> =
            serde_json::from_value(independent_witness_verification["epoch_proofs"].clone())?;
        ensure!(
            super::super::witness::verify_epoch_chain(&witness_policy, &epochs)?.authority_epoch
                == body.parent_epoch,
            "public witness effective original epoch differs"
        );
        ensure!(
            independent_witness_verification["head_sha256"] == body.independent_witness.head_sha256,
            "independent witness original snapshot head changed"
        );
        let evidence = authority.recovery_evidence()?;
        ensure!(
            evidence.database == evidence.journal
                && evidence.journal == evidence.witness
                && evidence.database_on_chain
                && authority.recovery_artifact_errors(&evidence)?.is_empty(),
            "original recovery journal/witness/artifacts incomplete"
        );
        let report = authority.recovery_report(&evidence)?;
        ensure!(
            hash(&encoded(&report)?) == body.report_sha256
                && report["blockers"].as_array().is_some_and(|rows| rows
                    .iter()
                    .all(|row| row == "recovery epoch not committed")),
            "signed original reconciliation report is stale"
        );
        ensure!(
            report["unresolved_original_funding"]
                .as_array()
                .is_some_and(Vec::is_empty)
                && report["unresolved_original_queries"]
                    .as_array()
                    .is_some_and(Vec::is_empty),
            "non-fit unresolved work requires separate recovery"
        );
        // The display report may be truncated. Admission must compare every
        // original held identity from authenticated retained postimages.
        let mut keys = BTreeSet::new();
        let mut rows = BTreeMap::new();
        let mut row_bytes = 0u64;
        for item in evidence.expected.iter()? {
            let ((table, key), row) = item?;
            if table == "entries"
                && matches!(
                    row["state"].as_str(),
                    Some("claimed" | "authorized" | "running")
                )
            {
                ensure!(
                    keys.len() < 100_000,
                    "original held set exceeds native recovery bound"
                );
                row_bytes += encoded(&row)?.len() as u64;
                ensure!(
                    row_bytes <= 64 << 20,
                    "original held postimages exceed bounded receipt; retain all evidence"
                );
                keys.insert(key.clone());
                rows.insert(key, row);
            }
        }
        ensure!(
            keys == body.held.keys().cloned().collect(),
            "signed recovery omits/adds original held identities"
        );
        let requests = original_requests(&authority)?;
        for (key, binding) in &body.held {
            let row = rows.get(key).context("held original entry missing")?;
            let request = requests
                .get(key)
                .context("held original journal provenance missing")?;
            check_binding(key, row, request, binding)?;
        }
        let drain_evidence = drains(body)?;
        let committed_at = policy::now_ms()?;
        let (again, _) = load_policy(policy_path, committed_at)?;
        ensure!(
            encoded(&again)? == encoded(&signed)?,
            "recovery policy changed during verification"
        );
        let receipt = Receipt {
            schema: "jope.native_held_epoch_receipt.v1".into(),
            policy: policy_reference,
            committed_unix_ms: committed_at,
            original_report: report,
            original_rows: rows,
            drain_evidence,
            independent_witness_verification,
        };
        let receipt_sha = authority.store_blob(&encoded(&receipt)?)?;
        let tip = authority.checkpoint.tip.clone();
        authority.mutation_under_lock("native_recovery_commit_held_epoch",&json!({"epoch":body.new_epoch,"parent_epoch":body.parent_epoch,"journal_tip":tip,"report_sha256":receipt_sha}),|db|{
            db.execute("INSERT INTO recovery_epochs VALUES('current',?,?,?,?) ON CONFLICT(id) DO UPDATE SET epoch=excluded.epoch,parent_epoch=excluded.parent_epoch,journal_tip=excluded.journal_tip,report_sha=excluded.report_sha",
                params![body.new_epoch,body.parent_epoch,tip,receipt_sha])?;Ok(((),true))})?;
        fs::remove_file(authority.root.join("recovery.pending"))?;
        sync_dir(&authority.root)?;
        Ok(body.new_epoch.clone())
    }
    pub(super) fn native_held_start_allowed(&self) -> Result<()> {
        ensure!(
            self.manifest.schema == NATIVE_SCHEMA,
            "legacy unresolved work remains blocked"
        );
        let (epoch, sha): (String, String) = self.db.query_row(
            "SELECT epoch,report_sha FROM recovery_epochs WHERE id='current'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        let receipt: Receipt = serde_json::from_slice(&self.read_blob(&sha)?)?;
        ensure!(
            receipt.schema == "jope.native_held_epoch_receipt.v1",
            "unreviewed recovery epoch cannot enable held work"
        );
        let (signed, reference) = load_policy(&receipt.policy.path, receipt.committed_unix_ms)?;
        ensure!(
            reference.sha256 == receipt.policy.sha256
                && signed.body.new_epoch == epoch
                && signed.body.authority_instance == self.manifest.instance
                && signed.body.authority_manifest_sha256 == self.manifest_sha
                && signed.body.authority_root == self.root
                && signed.body.witness == self.witness
                && hash(&encoded(&receipt.original_report)?) == signed.body.report_sha256,
            "held epoch approval changed"
        );
        let evidence = self.recovery_evidence()?;
        let requests = original_requests(self)?;
        for item in evidence.expected.iter()? {
            let ((table, key), row) = item?;
            let (table, key, row) = (table.as_str(), key.as_str(), &row);
            if table == "entries"
                && matches!(
                    row["state"].as_str(),
                    Some("claimed" | "authorized" | "running")
                )
            {
                ensure!(
                    row["epoch"] != epoch && receipt.original_rows.get(key) == Some(row),
                    "new/uncovered/changed held entry blocks restart"
                );
                check_binding(
                    key,
                    row,
                    requests
                        .get(key)
                        .context("original held journal provenance absent")?,
                    signed.body.held.get(key).context("held approval missing")?,
                )?;
            }
        }
        // This immutable receipt was published only after authenticated old-tree
        // drain. Do not reinterpret historical custody through a later daemon
        // executable or pruned process path on every authority restart.
        verify_recorded_drains(&signed.body, &receipt.drain_evidence)?;
        Ok(())
    }
}
