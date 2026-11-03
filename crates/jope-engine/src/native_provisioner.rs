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

mod finish;
use finish::finish_acknowledged;
pub use finish::provision;

#[cfg(test)]
mod tests;
