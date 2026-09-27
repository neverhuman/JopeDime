//! A bounded-run rank worker for the W7-T07 CPU phase (`jope-native-loop participant`).
//!
//! It runs as the SmartCluster trusted service its rank's supervisor submitted, and:
//! 1. loads only the frozen plan (by digest) and its launch record, and checks its own device
//!    visibility before anything else (W7-T03 delivery);
//! 2. observes its own host identity exactly as the census collector does, with the running
//!    binary as the executable;
//! 3. joins the authenticated rendezvous over the pinned-TLS transport, presenting this host's
//!    certificate and accepting only the plan's coordinator certificate (W7-T04);
//! 4. starts no work before it holds a verified full-group admission;
//! 5. performs the CPU proof workload, one update round per coordinator heartbeat (strictly
//!    increasing delivery ids), then publishes a signed result in its outbox.
//!
//! Losing control (no heartbeat within the plan's heartbeat timeout, end of stream, or a
//! replayed or regressed heartbeat) stops it with [`EXIT_LOST_CONTROL`] and no result. The
//! CPU proof is a lifecycle workload over the verified inputs; it is not training.
use super::{
    collect::{self, CollectSpec, CommandRunner},
    controller::EXIT_LOST_CONTROL,
    launch,
    rendezvous::{self, publish_immutable, ControlFrame, ControlStream, HelloBody, ResultBody},
    transport::{self, HostCredential, Timeouts},
};
use crate::manifest::hash;
use anyhow::{ensure, Context, Result};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    net::ToSocketAddrs,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    time::Duration,
};

pub const INSTALLATION_SCHEMA: &str = "jope.native_host_installation.v1";
pub const CPU_PROOF_SCHEMA: &str = "jope.native_cpu_proof_result.v1";
const MAX_FILE_BYTES: u64 = 64 * 1024;

/// How one host is installed: what its census collector and every participant on it observe.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Installation {
    pub schema: String,
    /// The host's address, assigned to a local interface.
    pub address: String,
    pub certificate: PathBuf,
    pub certificate_key: PathBuf,
    /// The installer-declared commit of the installed `jope-native-loop`.
    pub source_commit: String,
    /// The installed `jope-native-loop`, the executable the plan pins.
    pub executable: PathBuf,
    pub runtime_libraries: Vec<PathBuf>,
    pub compiler: PathBuf,
    #[serde(default)]
    pub cuda_libraries: Vec<PathBuf>,
    #[serde(default)]
    pub nccl_libraries: Vec<PathBuf>,
    pub installed_policies: Vec<PathBuf>,
    /// Whether the host's census records its GPUs. `false` is a CPU-only census, as
    /// collected with `CUDA_VISIBLE_DEVICES=""`.
    pub gpu_inventory: bool,
    /// Where this host keeps its copy of the plan's input closure.
    pub input_root: PathBuf,
}
fn read_small(path: &Path, private: bool) -> Result<Vec<u8>> {
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let meta = f.metadata()?;
    ensure!(meta.is_file(), "{} is not a regular file", path.display());
    ensure!(
        !private || meta.mode() & 0o077 == 0,
        "private key {} must be readable by its owner only",
        path.display()
    );
    let mut b = Vec::new();
    f.take(MAX_FILE_BYTES + 1).read_to_end(&mut b)?;
    ensure!(
        b.len() as u64 <= MAX_FILE_BYTES,
        "{} exceeds {MAX_FILE_BYTES} bytes",
        path.display()
    );
    Ok(b)
}
impl Installation {
    pub fn load(path: &Path) -> Result<Self> {
        let i: Self = serde_json::from_slice(&read_small(path, false)?)
            .with_context(|| format!("installation {}", path.display()))?;
        ensure!(
            i.schema == INSTALLATION_SCHEMA,
            "unsupported installation schema `{}`",
            i.schema
        );
        Ok(i)
    }
    /// The collector's view of this installation. `executable` overrides the installed path:
    /// a participant reports the binary it is actually running.
    pub fn collect_spec(&self, executable: Option<&Path>) -> CollectSpec {
        CollectSpec {
            address: self.address.clone(),
            certificate: self.certificate.clone(),
            source_commit: self.source_commit.clone(),
            executable: executable
                .map(Path::to_path_buf)
                .unwrap_or_else(|| self.executable.clone()),
            runtime_libraries: self.runtime_libraries.clone(),
            compiler: self.compiler.clone(),
            cuda_libraries: self.cuda_libraries.clone(),
            nccl_libraries: self.nccl_libraries.clone(),
            installed_policies: self.installed_policies.clone(),
            cuda_visible_devices: (!self.gpu_inventory).then(std::ffi::OsString::new),
        }
    }
}

/// The PEM label of a PKCS#8 private key. The armour lines are joined at run time so that this
/// parser is not itself mistaken for an embedded key by the repository secret scan.
const PKCS8_LABEL: &str = "PRIVATE KEY";

/// An Ed25519 signing key from a PKCS#8 v1 private key (PEM `PRIVATE KEY` or DER), as
/// `openssl genpkey -algorithm ed25519` writes it. The file must be readable by its owner
/// only and must not be a symlink.
pub fn load_signing_key(path: &Path) -> Result<SigningKey> {
    let bytes = read_small(path, true)?;
    let der = if bytes.starts_with(b"-----BEGIN") {
        let text = std::str::from_utf8(&bytes).context("PEM key is not UTF-8")?;
        let begin = format!("-----BEGIN {PKCS8_LABEL}-----");
        let end = format!("-----END {PKCS8_LABEL}-----");
        let body = text
            .split_once(begin.as_str())
            .and_then(|(_, rest)| rest.split_once(end.as_str()))
            .map(|(body, _)| body)
            .context("PEM file holds no PKCS#8 `PRIVATE KEY` block")?;
        collect::base64_decode(body)?
    } else {
        bytes
    };
    // SEQUENCE { INTEGER 0, SEQUENCE { OID 1.3.101.112 }, OCTET STRING { OCTET STRING seed } }
    const PREFIX: [u8; 16] = [
        0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04,
        0x20,
    ];
    ensure!(
        der.len() == 48 && der[..16] == PREFIX,
        "{} is not a PKCS#8 v1 Ed25519 private key",
        path.display()
    );
    let seed: [u8; 32] = der[16..].try_into()?;
    Ok(SigningKey::from_bytes(&seed))
}

/// Everything a participant process is given: frozen, host-local artifacts only.
#[derive(Clone, Debug)]
pub struct ParticipantArgs {
    pub plan: PathBuf,
    pub plan_sha256: String,
    pub launch_record: PathBuf,
    pub outbox: PathBuf,
    /// The coordinator's rendezvous address, `host:port`.
    pub coordinator: String,
    pub participant_key: PathBuf,
    pub installation: PathBuf,
}

/// The CPU proof's deterministic update: one SHA-256 round over the chain, the round number,
/// the verified input closure digest and this rank's admission.
pub fn cpu_proof_round(chain: &[u8; 32], round: u64, closure: &str, admission: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(chain);
    h.update(round.to_be_bytes());
    h.update(closure.as_bytes());
    h.update(admission.as_bytes());
    h.finalize().into()
}
fn now_ms() -> u64 {
    crate::resident_protocol::now_ms().unwrap_or(0)
}
/// Best effort: publish why this participant refused (`refusal.txt`, at most 4 KiB) in its
/// outbox, which its supervisor retains. It is a diagnostic, never a result.
pub fn record_refusal(outbox: &Path, error: &anyhow::Error) {
    let text: String = format!("{error:#}").chars().take(4096).collect();
    let _ = publish_immutable(outbox, Path::new("refusal.txt"), text.as_bytes());
}

/// Run one participant to completion. Returns the process exit code: 0 after publishing a
/// signed result (or on an explicit stop), [`EXIT_LOST_CONTROL`] when control is lost. Every
/// refusal before admission is an error, and then no work was started.
pub fn run(args: &ParticipantArgs, runner: &dyn CommandRunner) -> Result<i32> {
    let (contract, record) =
        launch::load_launch(&args.plan, &args.plan_sha256, &args.launch_record)?;
    launch::verify_worker_visibility(
        &record,
        std::env::var_os(launch::VISIBILITY_VARIABLE).as_deref(),
    )?;
    let rank = record.global_rank;
    let planned = contract.participant(rank).context("rank")?;
    ensure!(
        planned.device.is_none(),
        "rank {rank} plans a GPU; the CPU-phase participant runs CPU ranks only"
    );
    let key = load_signing_key(&args.participant_key)?;
    ensure!(
        hex::encode(key.verifying_key().to_bytes()) == planned.public_key,
        "participant key is not rank {rank}'s planned key"
    );
    let installation = Installation::load(&args.installation)?;
    let identity = collect::observe_identity(
        &installation.collect_spec(Some(Path::new("/proc/self/exe"))),
        runner,
    )?;
    let closure =
        rendezvous::observed_closure_digest(&installation.input_root, &contract.artifact_closure)?;
    let credential =
        HostCredential::load(&installation.certificate, &installation.certificate_key)?;
    let coordinator = args
        .coordinator
        .to_socket_addrs()
        .with_context(|| format!("coordinator address `{}`", args.coordinator))?
        .next()
        .context("coordinator address resolves to nothing")?;
    let pin = transport::coordinator_certificate(&contract)?;
    let mut stream = transport::connect(
        coordinator,
        &credential,
        (&pin.0, &pin.1),
        Timeouts::from_plan(&contract),
    )?;
    // Rank 0 carries the group bootstrap. The CPU phase creates no communicator, so it is a
    // fresh 32-byte group token, never an NCCL id (which is exactly 128 bytes).
    let bootstrap = if rank == 0 {
        let mut token = rendezvous::fresh_nonce()?.to_vec();
        token.extend(rendezvous::fresh_nonce()?);
        Some(hex::encode(token))
    } else {
        None
    };
    let grant_sha256 = super::digest(&planned.grant)?;
    let member = rendezvous::join(
        &mut stream,
        &contract,
        &args.plan_sha256,
        &key,
        |challenge| HelloBody {
            schema: rendezvous::HELLO_SCHEMA.into(),
            run: contract.run.clone(),
            attempt: contract.attempt,
            plan_sha256: args.plan_sha256.clone(),
            rendezvous_id: challenge.body.rendezvous_id.clone(),
            global_rank: rank,
            delivery_id: 1,
            sent_unix_ms: now_ms(),
            host: identity.host.clone(),
            machine_id: identity.machine_id.clone(),
            boot_id: identity.boot_id.clone(),
            device_uuid: None,
            source_commit: identity.source_commit.clone(),
            executable_sha256: identity.executable_sha256.clone(),
            runtime: identity.runtime.clone(),
            installed_policy_sha256: identity.installed_policy_sha256.clone(),
            artifact_closure_sha256: closure.clone(),
            grant_sha256: grant_sha256.clone(),
            communicator_bootstrap: bootstrap.clone(),
        },
        &now_ms,
    )?;
    publish_immutable(
        &args.outbox,
        Path::new("admitted"),
        member.admission_sha256().as_bytes(),
    )?;
    // Admitted: work may start. One update round per heartbeat.
    let heartbeat = Some(Duration::from_millis(
        contract.rendezvous.heartbeat_timeout_ms,
    ));
    stream.set_timeouts(heartbeat, heartbeat)?;
    let limit = contract.rendezvous.max_message_bytes;
    let started = now_ms();
    let mut chain = [0u8; 32];
    let (mut last, mut rounds) = (0u64, 0u64);
    while rounds < contract.counters.planned_updates {
        let Ok(frame) = rendezvous::read_frame(&mut stream, limit) else {
            return Ok(EXIT_LOST_CONTROL);
        };
        match serde_json::from_slice::<ControlFrame>(&frame) {
            Ok(ControlFrame::Heartbeat { delivery_id }) if delivery_id > last => {
                last = delivery_id;
                rounds += 1;
                chain = cpu_proof_round(&chain, rounds, &closure, member.admission_sha256());
            }
            Ok(ControlFrame::Stop { .. }) => return Ok(0),
            // A replayed or regressed heartbeat, or anything else, is not control.
            _ => return Ok(EXIT_LOST_CONTROL),
        }
    }
    let result = serde_json::to_vec(&json!({
        "schema": CPU_PROOF_SCHEMA,
        "run": contract.run,
        "attempt": contract.attempt,
        "global_rank": rank,
        "host": identity.host,
        "machine_id": identity.machine_id,
        "boot_id": identity.boot_id,
        "admission_sha256": member.admission_sha256(),
        "input_closure_sha256": closure,
        "updates": rounds,
        "chain_sha256": hex::encode(chain),
        "started_unix_ms": started,
        "numerical": false,
    }))?;
    let envelope = rendezvous::sign_result(
        &key,
        ResultBody {
            schema: rendezvous::RESULT_SCHEMA.into(),
            run: contract.run.clone(),
            attempt: contract.attempt,
            plan_sha256: args.plan_sha256.clone(),
            admission_sha256: member.admission_sha256().into(),
            global_rank: rank,
            delivery_id: 1,
            result_sha256: hash(&result),
            result_bytes: result.len() as u64,
            updates: rounds,
            completed_unix_ms: now_ms(),
        },
    )?;
    // The result bytes first, the signed envelope last: an envelope names complete bytes.
    publish_immutable(&args.outbox, Path::new("result.bin"), &result)?;
    publish_immutable(
        &args.outbox,
        Path::new("envelope.json"),
        &serde_json::to_vec(&envelope)?,
    )?;
    Ok(0)
}
