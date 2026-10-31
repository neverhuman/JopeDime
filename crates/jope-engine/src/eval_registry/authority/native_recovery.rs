//! Native recovery retains every original uncertain identity; it never releases a claim.
use super::*;
use crate::eval_registry::native_scq::{self, policy};
use ed25519_dalek::{Signature, VerifyingKey};
use jain_smartcluster::{FencedAttempt, GetRequest, InspectAttemptRequest, Request, Response};
const DOMAIN: &[u8] = b"jope.native_held_epoch.signature.v1\0";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryArtifact {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OriginalCustody {
    pub authority_attempt: FencedAttempt,
    pub target_attempt: FencedAttempt,
    pub authority_leader: ProcessIdentity,
    pub target_leader: ProcessIdentity,
    pub requesting_process: ProcessIdentity,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeldBinding {
    pub row_sha256: String,
    pub request_sha256: String,
    pub custody: OriginalCustody,
    /// Older claims require separately reviewed, immutable historical custody evidence.
    /// This is not written into, or substituted for, their original provenance.
    pub historical_custody: Option<RecoveryArtifact>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicWitnessSnapshot {
    pub root: PathBuf,
    pub policy: RecoveryArtifact,
    pub head_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRecoveryBody {
    pub schema: String,
    pub authority_root: PathBuf,
    pub witness: PathBuf,
    pub authority_instance: String,
    pub authority_manifest_sha256: String,
    pub parent_epoch: String,
    pub new_epoch: String,
    pub report_sha256: String,
    pub decision: RecoveryArtifact,
    pub not_before_unix_ms: u64,
    pub expires_unix_ms: u64,
    pub daemon_socket: PathBuf,
    pub daemon_uid: u32,
    pub daemon_executable_sha256: String,
    pub cgroup_root: PathBuf,
    pub held: BTreeMap<String, HeldBinding>,
    pub independent_witness: PublicWitnessSnapshot,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedNativeRecovery {
    pub authorization_public_key: String,
    pub body_sha256: String,
    pub body: NativeRecoveryBody,
    pub signature: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema: String,
    policy: RecoveryArtifact,
    committed_unix_ms: u64,
    original_report: Value,
    original_rows: BTreeMap<String, Value>,
    drain_evidence: Value,
    independent_witness_verification: Value,
}
fn artifact(reference: &RecoveryArtifact) -> Result<Vec<u8>> {
    ensure!(
        reference.path.is_absolute() && valid_sha(&reference.sha256),
        "invalid recovery artifact"
    );
    let bytes = policy::read_file(&reference.path)?;
    ensure!(
        hash(&bytes) == reference.sha256,
        "recovery artifact changed"
    );
    Ok(bytes)
}
fn verify_signed(signed: &SignedNativeRecovery, when: u64) -> Result<()> {
    let body = &signed.body;
    ensure!(
        body.schema == "jope.native_held_epoch.v1",
        "unknown held-work recovery schema"
    );
    ensure!(
        body.not_before_unix_ms <= when && when < body.expires_unix_ms,
        "recovery decision is outside its reviewed validity window"
    );
    ensure!(
        !body.held.is_empty() && body.held.len() <= 100_000,
        "unbounded or empty held set"
    );
    for value in [
        &body.authority_instance,
        &body.authority_manifest_sha256,
        &body.parent_epoch,
        &body.new_epoch,
        &body.report_sha256,
        &body.daemon_executable_sha256,
        &signed.body_sha256,
    ] {
        ensure!(valid_sha(value), "malformed recovery hash/epoch");
    }
    ensure!(
        body.parent_epoch != body.new_epoch,
        "recovery cannot reuse its parent epoch"
    );
    for path in [
        &body.authority_root,
        &body.witness,
        &body.daemon_socket,
        &body.cgroup_root,
    ] {
        ensure!(
            path.is_absolute()
                && path.components().all(|c| matches!(
                    c,
                    std::path::Component::RootDir | std::path::Component::Normal(_)
                )),
            "unsafe recovery path"
        );
    }
    let bytes = encoded(body)?;
    ensure!(
        hash(&bytes) == signed.body_sha256,
        "signed recovery body changed"
    );
    let raw: [u8; 32] = hex::decode(&signed.authorization_public_key)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("recovery public key width"))?;
    let key = VerifyingKey::from_bytes(&raw)?;
    ensure!(!key.is_weak(), "weak recovery signing key");
    let mut message = DOMAIN.to_vec();
    message.extend(bytes);
    key.verify_strict(
        &message,
        &Signature::from_slice(&hex::decode(&signed.signature)?)?,
    )?;
    let decision: Value = serde_json::from_slice(&artifact(&body.decision)?)?;
    ensure!(
        decision["schema"] == "jope.native_held_epoch_review.v1"
            && decision["authority_instance"] == body.authority_instance
            && decision["parent_epoch"] == body.parent_epoch
            && decision["new_epoch"] == body.new_epoch
            && decision["report_sha256"] == body.report_sha256
            && decision["held_bindings_sha256"] == hash(&encoded(&body.held)?)
            && decision["original_identities_remain_held"] == true
            && decision["original_outcomes_are_not_inferred"] == true,
        "independent recovery decision does not bind this exact held transition"
    );
    Ok(())
}
fn load_policy(path: &Path, at: u64) -> Result<(SignedNativeRecovery, RecoveryArtifact)> {
    // The separately installed root-controlled file pins both signer and exact body.
    // An authority process or planner cannot supply its own self-signed approval.
    let bytes = policy::protected(path)?;
    let signed: SignedNativeRecovery = serde_json::from_slice(&bytes)?;
    crate::native_review::require_signer(&signed.authorization_public_key)?;
    verify_signed(&signed, at)?;
    Ok((
        signed,
        RecoveryArtifact {
            path: path.into(),
            sha256: hash(&bytes),
        },
    ))
}
fn original_requests(authority: &Authority) -> Result<BTreeMap<String, Value>> {
    let mut claims = BTreeMap::new();
    let mut bytes = 0u64;
    let mut reader = BufReader::new(File::open(authority.root.join("dispatch.jsonl"))?);
    loop {
        let line = read_event_line(&mut reader)?;
        if line.is_empty() {
            break;
        }
        let event: Event = serde_json::from_slice(&line)?;
        if event.kind == "claim" {
            let key = event.payload["claim"]["fit_key"]
                .as_str()
                .context("original journal claim key")?
                .to_owned();
            bytes = bytes
                .checked_add(encoded(&event.payload["request"])?.len() as u64)
                .context("claim provenance byte overflow")?;
            ensure!(claims.len()<100_000 && bytes <= 64 << 20, "original claim provenance exceeds bounded native recovery; retain all original evidence");
            ensure!(
                claims
                    .insert(key, event.payload["request"].clone())
                    .is_none(),
                "duplicate original claim journal identity"
            );
        }
    }
    Ok(claims)
}

fn check_binding(key: &str, row: &Value, request: &Value, binding: &HeldBinding) -> Result<()> {
    ensure!(
        hash(&encoded(row)?) == binding.row_sha256
            && hash(&encoded(request)?) == binding.request_sha256,
        "original held entry/provenance changed"
    );
    let owner: ProcessIdentity = serde_json::from_str(
        row["owner_json"]
            .as_str()
            .context("original claim owner absent")?,
    )?;
    let peer: ProcessIdentity =
        serde_json::from_value(request["details"]["requesting_process"].clone())?;
    ensure!(
        owner == binding.custody.authority_leader && peer == binding.custody.requesting_process,
        "recovery custody differs from original committed owner/requester"
    );
    let native = &request["details"]["native_allocation"];
    if native.is_null() {
        let reference = binding
            .historical_custody
            .as_ref()
            .context("original claim lacks native fences and historical evidence")?;
        let history: Value = serde_json::from_slice(&artifact(reference)?)?;
        ensure!(
            history["schema"] == "jope.native_historical_claim_custody.v1"
                && history["fit_key"] == key
                && history["entry_sha256"] == binding.row_sha256
                && history["request_sha256"] == binding.request_sha256
                && history["custody"] == serde_json::to_value(&binding.custody)?
                && history["reviewed_original_ancestry"] == true,
            "historical original custody evidence differs"
        );
        let sources: Vec<RecoveryArtifact> =
            serde_json::from_value(history["original_sources"].clone())?;
        ensure!(
            !sources.is_empty() && sources.len() <= 1024,
            "historical custody original sources absent"
        );
        for source in sources {
            artifact(&source)?;
        }
    } else {
        ensure!(
            binding.historical_custody.is_none()
                && native["schema"] == "jope.native_claim_allocation.v1",
            "native preclaim provenance cannot be replaced by historical assertions"
        );
        ensure!(
            native["authority_attempt"] == serde_json::to_value(binding.custody.authority_attempt)?
                && native["target_attempt"]
                    == serde_json::to_value(binding.custody.target_attempt)?
                && native["authority_leader"]
                    == serde_json::to_value(&binding.custody.authority_leader)?
                && native["target_leader"] == serde_json::to_value(&binding.custody.target_leader)?,
            "original authenticated claim fences changed"
        );
    }
    Ok(())
}
fn process_gone(original: &ProcessIdentity) -> Result<()> {
    let boot = fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    if boot.trim() != original.boot_id {
        return Ok(());
    }
    let directory = PathBuf::from(format!("/proc/{}", original.pid));
    if !directory.exists() {
        return Ok(());
    }
    ensure!(
        ProcessIdentity::read(original.pid)? != *original,
        "original native process is still alive"
    );
    Ok(())
}
fn drains(body: &NativeRecoveryBody) -> Result<Value> {
    ensure!(
        body.cgroup_root.starts_with("/sys/fs/cgroup")
            && fs::canonicalize(&body.cgroup_root)? == body.cgroup_root,
        "recovery requires physical delegated cgroup root"
    );
    let path = std::ffi::CString::new(body.cgroup_root.as_os_str().as_encoded_bytes())?;
    let mut stats = std::mem::MaybeUninit::<libc::statfs>::uninit();
    ensure!(
        // SAFETY: `path` is a live NUL-terminated CString and `stats` is aligned
        // writable storage for one statfs result.
        unsafe { libc::statfs(path.as_ptr(), stats.as_mut_ptr()) } == 0,
        "cgroup filesystem unavailable"
    );
    // SAFETY: statfs returned 0, so it initialized every field of `stats`.
    ensure!(
        unsafe { stats.assume_init() }.f_type == libc::CGROUP2_SUPER_MAGIC,
        "recovery cgroup is not kernel cgroup v2"
    );
    let mut checked = BTreeMap::new();
    for binding in body.held.values() {
        process_gone(&binding.custody.requesting_process)?;
        for (attempt, leader) in [
            (
                binding.custody.authority_attempt,
                &binding.custody.authority_leader,
            ),
            (
                binding.custody.target_attempt,
                &binding.custody.target_leader,
            ),
        ] {
            process_gone(leader)?;
            let key = format!("{}:{}:{}", attempt.job_id, attempt.number, attempt.fence);
            if checked.contains_key(&key) {
                continue;
            }
            let response = native_scq::authenticated_exchange(
                &body.daemon_socket,
                body.daemon_uid,
                &body.daemon_executable_sha256,
                None,
                Request::InspectAttempt(InspectAttemptRequest { attempt }),
            )?;
            let inspection = match response {
                Response::AttemptInspected(Some(i)) => i,
                _ => anyhow::bail!("original native attempt evidence absent"),
            };
            let expected = body.cgroup_root.join(format!(
                "scq-{}-attempt-{}-fence-{}",
                attempt.job_id, attempt.number, attempt.fence
            ));
            ensure!(
                inspection.record.attempt == attempt
                    && inspection.record.pid == Some(leader.pid)
                    && inspection.record.finished_at.is_some()
                    && inspection.active_process.is_none()
                    && inspection.record.cgroup_path.as_deref() == expected.to_str(),
                "original native attempt is not a terminal fully drained allocation"
            );
            let response = native_scq::authenticated_exchange(
                &body.daemon_socket,
                body.daemon_uid,
                &body.daemon_executable_sha256,
                None,
                Request::Get(GetRequest {
                    job_id: attempt.job_id,
                }),
            )?;
            let job = match response {
                Response::Job(Some(j)) => j,
                _ => anyhow::bail!("original native job absent"),
            };
            ensure!(
                job.active_attempt.is_none() && job.state.is_terminal(),
                "old native job can still execute/retry"
            );
            if expected.exists() {
                ensure!(
                    fs::canonicalize(&expected)? == expected,
                    "old native cgroup path replaced"
                );
                let events = fs::read_to_string(expected.join("cgroup.events"))?;
                ensure!(
                    events.lines().any(|line| line == "populated 0"),
                    "original cgroup subtree remains populated"
                );
            }
            checked.insert(key, serde_json::to_value(inspection)?);
        }
    }
    Ok(serde_json::to_value(checked)?)
}
fn verify_recorded_drains(body: &NativeRecoveryBody, evidence: &Value) -> Result<()> {
    let mut expected = BTreeSet::new();
    for binding in body.held.values() {
        for (attempt, leader) in [
            (
                binding.custody.authority_attempt,
                &binding.custody.authority_leader,
            ),
            (
                binding.custody.target_attempt,
                &binding.custody.target_leader,
            ),
        ] {
            let key = format!("{}:{}:{}", attempt.job_id, attempt.number, attempt.fence);
            expected.insert(key.clone());
            let inspection: jain_smartcluster::AttemptInspection = serde_json::from_value(
                evidence
                    .get(&key)
                    .context("original committed drain evidence missing")?
                    .clone(),
            )?;
            let path = body.cgroup_root.join(format!(
                "scq-{}-attempt-{}-fence-{}",
                attempt.job_id, attempt.number, attempt.fence
            ));
            ensure!(
                inspection.record.attempt == attempt
                    && inspection.record.pid == Some(leader.pid)
                    && inspection.record.finished_at.is_some()
                    && inspection.record.state.is_terminal()
                    && inspection.record.lease.attempt == attempt
                    && inspection.active_process.is_none()
                    && inspection.record.cgroup_path.as_deref() == path.to_str(),
                "original committed drain evidence differs from reviewed held allocation"
            );
        }
    }
    ensure!(
        evidence
            .as_object()
            .context("drain evidence map")?
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
            == expected,
        "unexpected committed drain evidence identity"
    );
    Ok(())
}
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
        for table in ["authorization_witness", "funding_witness"] {
            let mut statement = authority
                .db
                .prepare(&format!("SELECT ack_sha FROM {table} ORDER BY ack_sha"))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    fn attempt() -> FencedAttempt {
        serde_json::from_value(
            json!({"job_id":jain_smartcluster::JobId::new(),"number":1,"fence":1}),
        )
        .unwrap()
    }
    #[test]
    fn original_preclaim_fences_cannot_be_replaced_by_same_process_or_historical_assertion() {
        let process = ProcessIdentity::current().unwrap();
        let custody = OriginalCustody {
            authority_attempt: attempt(),
            target_attempt: attempt(),
            authority_leader: process.clone(),
            target_leader: process.clone(),
            requesting_process: process.clone(),
        };
        let row = json!({"owner_json":serde_json::to_string(&process).unwrap(),"state":"claimed"});
        let request = json!({"details":{"requesting_process":process,"native_allocation":{"schema":"jope.native_claim_allocation.v1",
            "authority_attempt":custody.authority_attempt,"target_attempt":custody.target_attempt,
            "authority_leader":custody.authority_leader,"target_leader":custody.target_leader}}});
        let mut binding = HeldBinding {
            row_sha256: hash(&encoded(&row).unwrap()),
            request_sha256: hash(&encoded(&request).unwrap()),
            custody,
            historical_custody: None,
        };
        check_binding("key", &row, &request, &binding).unwrap();
        binding.custody.target_attempt = attempt();
        assert!(check_binding("key", &row, &request, &binding).is_err());
        let mut altered = row.clone();
        altered["state"] = json!("running");
        assert!(check_binding("key", &altered, &request, &binding).is_err());
    }
    #[test]
    fn original_live_process_blocks_recovery_but_distinct_pid_birth_is_not_the_original() {
        assert!(process_gone(&ProcessIdentity::current().unwrap()).is_err());
        let mut original = ProcessIdentity::current().unwrap();
        original.birth_ticks += 1;
        process_gone(&original).unwrap();
    }
    #[test]
    fn operator_writable_self_signed_policy_is_not_installed_authorization() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("policy.json");
        fs::write(&path, b"{}").unwrap();
        assert!(load_policy(&path, 0).is_err());
    }
}

#[cfg(test)]
#[path = "native_recovery_tests.rs"]
mod signature_tests;
