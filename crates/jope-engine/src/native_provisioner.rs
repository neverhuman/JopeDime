//! Root-owned, finite grant provisioner. No task dispatch or numerical execution.
mod service;
use crate::{
    eval_registry::{
        native_scq::policy::{NativeWitnessTrust, SignedNativeGrant},
        witness::{
            self, AddNativeGrantBody, NativeGrantArtifact, NativeGrantProof, NativeGrantRole,
            SignedAddNativeGrant, WitnessClientConfig, WitnessPolicy,
        },
    },
    manifest::hash,
    native_graph::Artifact,
    native_loop::{
        Config as LoopConfig, GrantProofInputs, Loop, NativeGrantEligibilityProof, Role,
    },
};
use anyhow::{ensure, Context, Result};
use ed25519_dalek::{Signer, SigningKey};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
pub use service::serve;
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
};
const LIMIT: u64 = 1024 * 1024;
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema: String,
    pub executable_sha256: String,
    pub root: PathBuf,
    pub loop_state: PathBuf,
    pub graph_state: PathBuf,
    pub worker_state: PathBuf,
    pub tenant: String,
    pub project: String,
    pub socket: PathBuf,
    pub loop_config: Artifact,
    pub witness_policy: Artifact,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predecessor_update_proof: Option<Artifact>,
    pub transport: Artifact,
    pub release: Artifact,
    pub coverage: Artifact,
    pub state: PathBuf,
    pub task_policy_directory: PathBuf,
    pub signing_key: PathBuf,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskInput {
    pub request: Artifact,
    pub admission: Artifact,
    pub task_manifest: Artifact,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Intent {
    schema: String,
    config_sha256: String,
    proof: NativeGrantEligibilityProof,
    request: SignedAddNativeGrant,
}
fn bytes<T: Serialize>(v: &T) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(v)?)
}
fn now() -> Result<u64> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}
fn protected(path: &Path, directory: bool) -> Result<()> {
    ensure!(
        path.is_absolute()
            && !path.components().any(|x| matches!(
                x,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )),
        "absolute protected path required"
    );
    for (i, p) in path.ancestors().enumerate() {
        let m = fs::symlink_metadata(p)?;
        ensure!(
            !m.file_type().is_symlink() && m.uid() == 0 && m.mode() & 0o022 == 0,
            "provisioner path custody invalid"
        );
        if i == 0 {
            ensure!(
                if directory { m.is_dir() } else { m.is_file() },
                "provisioner path type"
            );
        } else {
            ensure!(m.is_dir(), "provisioner ancestor type");
        }
    }
    Ok(())
}
fn read(path: &Path) -> Result<Vec<u8>> {
    let mut f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)?;
    let a = f.metadata()?;
    ensure!(
        a.is_file() && a.len() <= LIMIT,
        "bounded regular provisioner artifact required"
    );
    let mut b = Vec::new();
    Read::by_ref(&mut f).take(LIMIT + 1).read_to_end(&mut b)?;
    let z = f.metadata()?;
    let named = fs::symlink_metadata(path)?;
    ensure!(
        b.len() as u64 == a.len()
            && a.dev() == named.dev()
            && a.ino() == named.ino()
            && !named.file_type().is_symlink()
            && a.mtime() == z.mtime()
            && a.mtime_nsec() == z.mtime_nsec()
            && a.ctime() == z.ctime()
            && a.ctime_nsec() == z.ctime_nsec(),
        "provisioner artifact changed"
    );
    Ok(b)
}
fn checked(root: &Path, a: &Artifact) -> Result<Vec<u8>> {
    let p = if a.path.is_absolute() {
        a.path.clone()
    } else {
        root.join(&a.path)
    };
    ensure!(
        fs::canonicalize(&p)? == p,
        "physical artifact path required"
    );
    let b = read(&p)?;
    ensure!(hash(&b) == a.sha256, "provisioner artifact digest mismatch");
    Ok(b)
}
fn install(dir: &Path, name: &str, b: &[u8]) -> Result<Artifact> {
    protected(dir, true)?;
    let result = install_bytes(dir, name, b)?;
    protected(&result.path, false)?;
    Ok(result)
}
// File transaction core; callers must establish parent ownership before entry.
// Kept separate so actual interrupted-file recovery can be exercised without
// granting test processes root privileges or access to real signing keys.
fn install_bytes(dir: &Path, name: &str, b: &[u8]) -> Result<Artifact> {
    ensure!(
        !name.is_empty() && !name.contains('/') && b.len() as u64 <= LIMIT,
        "bounded immutable artifact name/bytes required"
    );
    let p = dir.join(name);
    if p.exists() {
        ensure!(read(&p)? == b, "immutable provisioner artifact conflict");
        return Ok(Artifact {
            path: p,
            sha256: hash(b),
        });
    }
    let pending = tempfile::Builder::new()
        .prefix(".pending-")
        .tempfile_in(dir)?;
    // Keep each failed staging identity; retries never collide on reused PIDs.
    let (mut f, tmp) = pending.keep()?;
    use std::os::unix::fs::PermissionsExt;
    f.set_permissions(fs::Permissions::from_mode(0o444))?;
    f.write_all(b)?;
    f.sync_all()?;
    // Original failed staging is retained; link never replaces another identity.
    fs::hard_link(&tmp, &p)?;
    File::open(dir)?.sync_all()?;
    fs::remove_file(tmp)?;
    File::open(dir)?.sync_all()?;
    Ok(Artifact {
        path: p,
        sha256: hash(b),
    })
}
fn sign<T: Serialize>(key: &SigningKey, domain: &[u8], body: &T) -> Result<String> {
    let mut m = domain.to_vec();
    m.extend(bytes(body)?);
    Ok(hex::encode(key.sign(&m).to_bytes()))
}
fn native_artifact(root: &Path, a: &Artifact) -> Result<NativeGrantArtifact> {
    checked(root, a)?;
    let path = if a.path.is_absolute() {
        a.path.clone()
    } else {
        root.join(&a.path)
    };
    Ok(NativeGrantArtifact {
        path,
        sha256: a.sha256.clone(),
    })
}
/// Inventory the finite root journal before walking it. A missing earlier intent
/// or orphan ACK must never make later historical reservations disappear.
#[cfg(test)]
fn journal_length(state: &Path) -> Result<u64> {
    journal_length_from(state, 1)
}
fn journal_length_from(state: &Path, first_sequence: u64) -> Result<u64> {
    let mut intents = BTreeSet::new();
    let mut artifacts = BTreeSet::new();
    for entry in fs::read_dir(state)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 journal entry"))?;
        if name == "lock" || name.starts_with(".pending-") {
            continue;
        }
        let (prefix, suffix) = name
            .split_once('.')
            .context("unknown provisioner journal entry")?;
        ensure!(
            prefix.len() == 8
                && prefix.bytes().all(|b| b.is_ascii_digit())
                && matches!(
                    suffix,
                    "intent.json"
                        | "proof.json"
                        | "grant.json"
                        | "eligibility.json"
                        | "installed.json"
                ),
            "unknown provisioner journal artifact"
        );
        let seq: u64 = prefix.parse()?;
        ensure!(
            (1..=100_000).contains(&seq),
            "provisioner journal sequence bound"
        );
        artifacts.insert(seq);
        if suffix == "intent.json" {
            intents.insert(seq);
        }
    }
    let last = first_sequence
        .checked_add(intents.len() as u64)
        .context("journal sequence overflow")?
        - 1;
    ensure!(
        intents.iter().copied().eq(first_sequence..=last) && artifacts.is_subset(&intents),
        "provisioner journal missing intent or contains orphan committed artifacts"
    );
    Ok(last)
}
fn validate_intent(
    config: &Config,
    config_bytes: &[u8],
    witness: &WitnessPolicy,
    epoch_proofs: &[witness::WitnessEpochProof],
    intent: &Intent,
    sequence: u64,
    previous: &str,
) -> Result<()> {
    ensure!(
        intent.schema == "jope.native_grant_intent.v1"
            && intent.config_sha256 == hash(config_bytes)
            && intent.request.body.sequence == sequence
            && intent.request.body.previous_update_sha256 == previous,
        "provisioner journal chain/config changed"
    );
    witness::validate_native_grant_request_for_epoch(witness, epoch_proofs, &intent.request)?;
    let p = &intent.proof;
    let i = &p.inputs;
    let body = &intent.request.body;
    ensure!(
        p.schema == "jope.native_grant_eligibility_proof.v1"
            && !p.authorization_issued
            && p.scientific_policy.sha256
                == witness
                    .native_grant_delegation
                    .as_ref()
                    .context("delegation absent")?
                    .scientific_policy_sha256
            && body.signed_grant.body.period_end_unix_ms <= p.task_deadline.as_unix_millis()
            && p.verifier_executable_sha256 == config.executable_sha256
            && i.root == config.root
            && i.loop_state == config.loop_state
            && i.graph_state == config.graph_state
            && i.worker_state == config.worker_state
            && i.tenant == config.tenant
            && i.project == config.project
            && i.config == config.loop_config
            && p.run == body.run
            && p.generation == body.generation
            && u64::try_from(p.evaluations)? == u64::from(body.evaluations)
            && p.task_manifest_sha256 == body.task_manifest.sha256
            && hash(&bytes(p)?) == body.signed_grant.body.decision_receipt_sha256,
        "root intent differs from original scientific proof/config"
    );
    let expected_role = match p.role {
        Role::Direct => NativeGrantRole::Direct,
        Role::Continuation => NativeGrantRole::Continuation,
        _ => anyhow::bail!("non-fit intent role"),
    };
    ensure!(body.role == expected_role, "intent scientific role differs");
    // Compare physical identity representations without reopening mutable source
    // files: ACK recovery must retain the ORIGINAL proof, not rerun eligibility.
    for (a, expected) in [
        (&i.request, &body.loop_request),
        (&i.admission, &body.graph_admission),
        (&i.task_manifest, &body.task_manifest),
    ] {
        let path = if a.path.is_absolute() {
            a.path.clone()
        } else {
            config.root.join(&a.path)
        };
        ensure!(
            path == expected.path && a.sha256 == expected.sha256,
            "intent artifact proof binding differs"
        );
    }
    Ok(())
}
fn finish_acknowledged(
    config: &Config,
    witness: &WitnessPolicy,
    intent: &Intent,
    proof: &NativeGrantProof,
) -> Result<Value> {
    let sequence = intent.request.body.sequence;
    let requested_sha = &intent.request.body.task_manifest.sha256;
    let trust: NativeWitnessTrust = witness::verify_native_grant_proof(witness, proof)?;
    ensure!(
        bytes(&proof.request)? == bytes(&intent.request)?,
        "remote proof differs from durable signed intent"
    );
    let proof_artifact = install(
        &config.state,
        &format!("{sequence:08}.proof.json"),
        &bytes(proof)?,
    )?;
    let decision = install(
        &config.state,
        &format!("{sequence:08}.eligibility.json"),
        &bytes(&intent.proof)?,
    )?;
    ensure!(
        decision.sha256
            == intent
                .request
                .body
                .signed_grant
                .body
                .decision_receipt_sha256,
        "original eligibility decision changed"
    );
    let grant_artifact = install(
        &config.state,
        &format!("{sequence:08}.grant.json"),
        &bytes(&intent.request.body.signed_grant)?,
    )?;
    ensure!(
        config.release.sha256 == intent.request.body.signed_grant.body.release_receipt_sha256
            && config.coverage.sha256
                == intent
                    .request
                    .body
                    .signed_grant
                    .body
                    .coverage_receipt_sha256,
        "installed release/coverage differs from original grant"
    );
    checked(&config.root, &config.release)?;
    checked(&config.root, &config.coverage)?;
    let policy = json!({"schema":"jope.smartcluster_fit_policy.v1","grant":grant_artifact,"trust":trust,
        "decision":decision,"release":config.release,"coverage":config.coverage,
        "native_grant_proof":proof_artifact,"witness_policy":config.witness_policy});
    let installed = install(
        &config.task_policy_directory,
        &format!("{requested_sha}.json"),
        &bytes(&policy)?,
    )?;
    let receipt = json!({"schema":"jope.native_grant_provisioned.v1","sequence":sequence,
        "task_manifest_sha256":requested_sha,"policy":installed,"numerical_completion":false,
        "original_update_sha256":hash(&bytes(&intent.request)?),"config_sha256":intent.config_sha256});
    // Only this deterministic original installation receipt closes the frontier.
    install(
        &config.state,
        &format!("{sequence:08}.installed.json"),
        &bytes(&receipt)?,
    )?;
    Ok(receipt)
}

/// One durable update at a time. An uncertain SSH response leaves the original
/// intent intact; a later invocation retries its exact bytes before any new work.
pub async fn provision(config_path: &Path, input: &TaskInput) -> Result<Value> {
    ensure!(
        crate::host_os::effective_uid() == 0,
        "provisioner requires root custody"
    );
    protected(config_path, false)?;
    let config_bytes = read(config_path)?;
    let config: Config = serde_json::from_slice(&config_bytes)?;
    ensure!(
        config.schema == "jope.native_grant_provisioner.v1"
            && crate::manifest::file_hash(Path::new("/proc/self/exe"))? == config.executable_sha256,
        "unreviewed provisioner executable/config"
    );
    protected(&config.state, true)?;
    protected(&config.task_policy_directory, true)?;
    protected(&config.signing_key, false)?;
    let key_meta = fs::metadata(&config.signing_key)?;
    ensure!(key_meta.mode() & 0o077 == 0, "provisioner key permissions");
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(config.state.join("lock"))?;
    ensure!(lock.try_lock().is_ok(), "provisioner busy");
    let loop_config: LoopConfig =
        serde_json::from_slice(&checked(&config.root, &config.loop_config)?)?;
    protected(&config.witness_policy.path, false)?;
    let witness: WitnessPolicy =
        serde_json::from_slice(&checked(&config.root, &config.witness_policy)?)?;
    protected(&config.transport.path, false)?;
    let transport =
        WitnessClientConfig::from_transport_file(&config.transport.path, &config.transport.sha256)?;
    ensure!(
        transport.policy.sha256()? == witness.sha256()?,
        "transport immutable base differs"
    );
    let effective = witness::verify_epoch_chain(&witness, &transport.epoch_proofs)?;
    let delegation = effective
        .native_grant_delegation
        .as_ref()
        .context("finite grant delegation absent")?;
    ensure!(
        loop_config.run == delegation.run
            && loop_config.initial_optimizer_step == delegation.initial_optimizer_step
            && loop_config.cycles == 2
            && delegation.cycles == 2
            && loop_config.scientific_policy.sha256 == delegation.scientific_policy_sha256
            && config.executable_sha256 == delegation.controller_sha256
            && config.root == delegation.grant_template.root
            && config.socket == delegation.grant_template.daemon_socket
            && config.tenant == delegation.grant_template.tenant
            && config.project == delegation.grant_template.project,
        "frozen scientific/window/runtime roots differ from delegation"
    );
    let raw: [u8; 32] = read(&config.signing_key)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("provisioner key width"))?;
    let key = SigningKey::from_bytes(&raw);
    ensure!(
        hex::encode(key.verifying_key().to_bytes()) == delegation.provisioner_public_key,
        "provisioner key differs from immutable delegation"
    );

    ensure!(
        config.release.sha256 == delegation.grant_template.release_receipt_sha256
            && config.coverage.sha256 == delegation.grant_template.coverage_receipt_sha256,
        "provisioner template release/coverage mismatch"
    );
    checked(&config.root, &config.release)?;
    checked(&config.root, &config.coverage)?;
    let (first_sequence, initial_previous, initial_spent) = match transport.epoch_proofs.last() {
        None => (1, "0".repeat(64), 0),
        Some(proof) => {
            let body = &proof.epoch_transition.body;
            (
                body.next_update_sequence,
                body.previous_update_sha256.clone(),
                body.original_reserved_evaluations,
            )
        }
    };
    match (&config.predecessor_update_proof, first_sequence) {
        (None, 1) => {}
        (Some(artifact), sequence) if sequence > 1 => {
            protected(&artifact.path, false)?;
            let prior: NativeGrantProof =
                serde_json::from_slice(&checked(&config.root, artifact)?)?;
            witness::verify_native_grant_proof(&witness, &prior)?;
            ensure!(
                prior.request.body.sequence.checked_add(1) == Some(sequence)
                    && hash(&bytes(&prior.request)?) == initial_previous,
                "original predecessor update differs from signed epoch cursor"
            );
        }
        _ => anyhow::bail!("original predecessor grant proof required for nonempty global ledger"),
    }
    let journal_count = journal_length_from(&config.state, first_sequence)?;
    let requested_sha = input.task_manifest.sha256.clone();
    ensure!(
        requested_sha.len() == 64
            && requested_sha
                .bytes()
                .all(|x| x.is_ascii_hexdigit() && !x.is_ascii_uppercase()),
        "task digest invalid"
    );
    let mut sequence = first_sequence;
    let mut previous = initial_previous;
    let mut spent = initial_spent;
    let mut existing = None;
    loop {
        ensure!(sequence <= 100_000, "finite provisioner journal exhausted");
        let name = format!("{sequence:08}.intent.json");
        let p = config.state.join(&name);
        if !p.exists() {
            break;
        }
        protected(&p, false)?;
        let intent: Intent = serde_json::from_slice(&read(&p)?)?;
        validate_intent(
            &config,
            &config_bytes,
            &witness,
            &transport.epoch_proofs,
            &intent,
            sequence,
            &previous,
        )?;
        let ackpath = config.state.join(format!("{sequence:08}.proof.json"));
        if !ackpath.exists() {
            ensure!(
                sequence == journal_count,
                "unsettled intent precedes later journal updates"
            );
            ensure!(
                intent.request.body.task_manifest.sha256 == requested_sha,
                "original pending grant must be settled first"
            );
            existing = Some(intent);
            break;
        }
        protected(&ackpath, false)?;
        let ack: NativeGrantProof = serde_json::from_slice(&read(&ackpath)?)?;
        witness::verify_native_grant_proof(&witness, &ack)?;
        ensure!(
            bytes(&ack.request)? == bytes(&intent.request)?,
            "journal ACK differs from original intent"
        );
        finish_acknowledged(&config, &witness, &intent, &ack)?;
        if intent.request.body.task_manifest.sha256 == requested_sha {
            existing = Some(intent);
            break;
        }
        spent = spent
            .checked_add(u64::from(intent.request.body.evaluations))
            .context("budget overflow")?;
        ensure!(
            spent <= delegation.total_evaluation_budget,
            "stored journal exceeds delegated evaluation budget"
        );
        previous = hash(&bytes(&intent.request)?);
        sequence += 1;
    }
    let intent = if let Some(v) = existing {
        v
    } else {
        let proof_inputs = GrantProofInputs {
            root: config.root.clone(),
            loop_state: config.loop_state.clone(),
            graph_state: config.graph_state.clone(),
            worker_state: config.worker_state.clone(),
            tenant: config.tenant.clone(),
            project: config.project.clone(),
            config: config.loop_config.clone(),
            request: input.request.clone(),
            admission: input.admission.clone(),
            task_manifest: input.task_manifest.clone(),
        };
        let proof = Loop::prove_native_grant_request(
            &proof_inputs,
            &jain_smartcluster::Client::new(config.socket.clone()),
        )
        .await?;
        let evaluations: u32 = proof.evaluations.try_into()?;
        ensure!(
            !proof.authorization_issued
                && proof.verifier_executable_sha256 == config.executable_sha256
                && proof.run == delegation.run
                && proof.generation < 2
                && proof.task_manifest_sha256 == requested_sha
                && proof.scientific_policy.sha256 == delegation.scientific_policy_sha256
                && spent
                    .checked_add(u64::from(evaluations))
                    .is_some_and(|x| x <= delegation.total_evaluation_budget),
            "proof exceeds immutable provisioner delegation"
        );
        let issued = now()?;
        ensure!(
            issued < delegation.deadline_unix_ms
                && issued < proof.task_deadline.as_unix_millis()
                && issued >= delegation.grant_template.period_start_unix_ms
                && issued < delegation.grant_template.period_end_unix_ms
                && evaluations > 0
                && evaluations <= delegation.grant_template.max_candidates_per_task,
            "original task/delegation expired"
        );
        // Identity is computed without exposing a partial transaction. The
        // complete signed intent is the FIRST durable final artifact.
        let decision = Artifact {
            path: config.state.join(format!("{sequence:08}.eligibility.json")),
            sha256: hash(&bytes(&proof)?),
        };
        let mut grant = delegation.grant_template.clone();
        grant.approved_task_sha256 = BTreeSet::from([requested_sha.clone()]);
        grant.max_candidates_per_task = evaluations;
        grant.decision_receipt_sha256 = decision.sha256;
        grant.period_start_unix_ms = issued.max(grant.period_start_unix_ms);
        grant.period_end_unix_ms = grant
            .period_end_unix_ms
            .min(proof.task_deadline.as_unix_millis());
        let grant_sha256 = hash(&bytes(&grant)?);
        let signed_grant = SignedNativeGrant {
            signature: sign(&key, b"jope.smartcluster_fit_grant.signature.v1\0", &grant)?,
            body: grant,
        };
        let body = AddNativeGrantBody {
            schema: "parallel.add_native_grant.v1".into(),
            base_policy_sha256: effective.sha256()?,
            delegation_sha256: delegation.sha256()?,
            sequence,
            previous_update_sha256: previous,
            signed_grant,
            grant_sha256,
            loop_request: native_artifact(&config.root, &input.request)?,
            graph_admission: native_artifact(&config.root, &input.admission)?,
            task_manifest: native_artifact(&config.root, &input.task_manifest)?,
            run: proof.run.clone(),
            generation: proof.generation,
            role: match proof.role {
                Role::Direct => NativeGrantRole::Direct,
                Role::Continuation => NativeGrantRole::Continuation,
                _ => anyhow::bail!("non-numerical grant role"),
            },
            evaluations,
            issued_unix_ms: issued,
        };
        let request = SignedAddNativeGrant {
            signature: sign(&key, b"parallel.add_native_grant.request.v1\0", &body)?,
            body,
        };
        // Reject every bound violation before creating a durable frontier that
        // the witness could never acknowledge.
        witness::validate_native_grant_request_for_epoch(
            &witness,
            &transport.epoch_proofs,
            &request,
        )?;
        let intent = Intent {
            schema: "jope.native_grant_intent.v1".into(),
            config_sha256: hash(&config_bytes),
            proof,
            request,
        };
        install(
            &config.state,
            &format!("{sequence:08}.intent.json"),
            &bytes(&intent)?,
        )?;
        intent
    };
    ensure!(
        intent.request.body.loop_request == native_artifact(&config.root, &input.request)?
            && intent.request.body.graph_admission
                == native_artifact(&config.root, &input.admission)?
            && intent.request.body.task_manifest
                == native_artifact(&config.root, &input.task_manifest)?,
        "retry differs from original exact artifacts"
    );
    let decision = install(
        &config.state,
        &format!("{sequence:08}.eligibility.json"),
        &bytes(&intent.proof)?,
    )?;
    ensure!(
        decision.sha256
            == intent
                .request
                .body
                .signed_grant
                .body
                .decision_receipt_sha256,
        "original eligibility decision differs"
    );
    let ackname = format!("{sequence:08}.proof.json");
    let ackpath = config.state.join(&ackname);
    let proof: NativeGrantProof = if ackpath.exists() {
        serde_json::from_slice(&read(&ackpath)?)?
    } else {
        let deadline = crate::host_os::monotonic_ns()
            .context("monotonic clock")?
            .checked_add(30_000_000_000)
            .context("transport deadline overflow")?;
        witness::add_native_grant_remote(&transport, &intent.request, deadline)?
    };
    finish_acknowledged(&config, &witness, &intent, &proof)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crash_after_intent_keeps_original_eligibility_identity() {
        let temp = tempfile::tempdir().unwrap();
        let original = br#"{"verified_unix_ms":111,"authorization_issued":false}"#;
        install_bytes(
            temp.path(),
            "00000001.intent.json",
            b"durable original intent includes exact proof",
        )
        .unwrap();
        assert_eq!(journal_length(temp.path()).unwrap(), 1);
        assert!(!temp.path().join("00000001.eligibility.json").exists());
        let first = install_bytes(temp.path(), "00000001.eligibility.json", original).unwrap();
        // Retry is a byte-for-byte reconstruction, never a fresh timestamp.
        let second = install_bytes(temp.path(), "00000001.eligibility.json", original).unwrap();
        assert_eq!(first, second);
        assert!(install_bytes(
            temp.path(),
            "00000001.eligibility.json",
            br#"{"verified_unix_ms":222}"#
        )
        .is_err());
        assert_eq!(read(&first.path).unwrap(), original);
    }
    #[test]
    fn ack_without_policy_is_a_retained_incomplete_transaction_not_an_orphan() {
        let temp = tempfile::tempdir().unwrap();
        install_bytes(temp.path(), "00000001.intent.json", b"original intent").unwrap();
        install_bytes(temp.path(), "00000001.proof.json", b"original remote ACK").unwrap();
        assert_eq!(journal_length(temp.path()).unwrap(), 1);
        assert!(!temp.path().join("00000001.installed.json").exists());
        // A missing intent must not hide a later reservation or turn the ACK
        // into a new sequence. Every original byte remains available to restore.
        fs::remove_file(temp.path().join("00000001.intent.json")).unwrap();
        assert!(journal_length(temp.path()).is_err());
        assert_eq!(
            read(&temp.path().join("00000001.proof.json")).unwrap(),
            b"original remote ACK"
        );
        install_bytes(temp.path(), "00000001.intent.json", b"original intent").unwrap();
        install_bytes(
            temp.path(),
            "00000003.intent.json",
            b"unexpected later sequence",
        )
        .unwrap();
        assert!(journal_length(temp.path()).is_err());
    }
    #[test]
    fn retained_failed_staging_does_not_collide_with_same_process_retry() {
        let temp = tempfile::tempdir().unwrap();
        let prior = temp.path().join(format!(
            ".pending-{}-{}",
            std::process::id(),
            hash(b"same bytes")
        ));
        fs::write(&prior, b"unfinished original bytes").unwrap();
        let installed = install_bytes(temp.path(), "00000001.intent.json", b"same bytes").unwrap();
        assert_eq!(read(&installed.path).unwrap(), b"same bytes");
        assert_eq!(fs::read(prior).unwrap(), b"unfinished original bytes");
        assert_eq!(journal_length(temp.path()).unwrap(), 1);
    }
}
