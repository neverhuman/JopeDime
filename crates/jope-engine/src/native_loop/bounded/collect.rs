//! Local host census collection and per-host census merge (W7-T01).
//!
//! [`collect`] observes the **local host only** and produces exactly the typed
//! `jope.native_host_census.v1` record that [`super::census::preflight`] validates. Through its
//! [`HostSources`] it reads the machine id, the kernel boot id, the host name and the local
//! interface addresses; the configured certificate; the SHA-256 of the executables, runtime
//! libraries and installed policy files it is given; online CPUs and available memory; and,
//! unless GPUs are hidden, the GPU inventory and device-owning processes from `nvidia-smi`. It
//! never contacts another host, never opens a device and never allocates or reserves anything:
//! a census is an observation, not a reservation, and admission re-checks it.
//!
//! Nothing is invented. An absent `nvidia-smi` or `CUDA_VISIBLE_DEVICES=""` records no GPU; a
//! failing or unparseable probe, a value that is not available, or an identity input that cannot
//! be verified locally refuses the whole collection. A machine id that is not 32 lowercase hex,
//! an empty or oversized boot id and a host name that is not a census name are refused where they
//! are read, naming the file; no identity is ever derived or defaulted.
//!
//! [`HostSources`] is the one observation provider the collector and every rendezvous
//! participant read the host through. The production entrypoints (`jope-native-loop census
//! collect` and `jope-native-loop participant`) pass [`HostSources::system`], the real
//! `/etc/machine-id`, `/proc/sys/kernel/random/boot_id`, `/proc/sys/kernel/hostname`,
//! `getifaddrs` and `nvidia-smi`; nothing else in the product constructs one. No argument or
//! environment variable substitutes a source, and the type has no `Default`. A fixture passes
//! files and functions of its own, and they are read by exactly the code that reads the real host.
//!
//! [`merge`] combines per-host documents into one multi-host census. Every document is pinned by
//! the SHA-256 its collector reported, so a document changed in transit is refused. Stale or
//! future-dated documents, duplicate hosts or identities and mixed or unsupported schema versions
//! are refused too.
use super::{
    census::{
        GpuObservation, HostCensus, HostIdentity, HostObservation, OwnerKind, Ownership,
        RuntimeIdentity, CENSUS_SCHEMA, MAX_CENSUS_BYTES,
    },
    digest, is_commit, is_digest, is_name,
};
use crate::manifest::hash;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fs,
    io::Read,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

pub const COLLECTION_SCHEMA: &str = "jope.native_host_census_collection.v1";
pub const MERGE_SCHEMA: &str = "jope.native_host_census_merge.v1";
/// Digest domain for a set of files (runtime libraries, CUDA, NCCL, installed policies).
pub const FILE_CLOSURE_SCHEMA: &str = "jope.native_file_closure.v1";
const MAX_CERTIFICATE_BYTES: u64 = 64 * 1024;
const MAX_HASHED_FILE_BYTES: u64 = 16 << 30;
const MAX_FILES_PER_CLOSURE: usize = 256;
const MAX_COMMAND_OUTPUT: u64 = 1024 * 1024;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
const MIB: u64 = 1024 * 1024;
pub const GPU_QUERY: [&str; 2] = [
    "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free",
    "--format=csv,noheader,nounits",
];
pub const APPS_QUERY: [&str; 2] = [
    "--query-compute-apps=gpu_uuid,pid,process_name",
    "--format=csv,noheader,nounits",
];

/// Output of one read-only external probe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandOutput {
    pub success: bool,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}
/// Runs read-only inventory programs. Injectable so fixtures never need a real GPU.
pub trait CommandRunner {
    /// `Ok(None)` when `program` is not installed; an error when it cannot be run or bounded.
    fn run(&self, program: &str, args: &[&str]) -> Result<Option<CommandOutput>>;
}
/// The host's own programs, resolved on `PATH`, run with a minimal environment, bounded
/// output and a timeout.
pub struct SystemCommands;
impl CommandRunner for SystemCommands {
    fn run(&self, program: &str, args: &[&str]) -> Result<Option<CommandOutput>> {
        let Some(path) = resolve_program(program) else {
            return Ok(None);
        };
        let mut child = Command::new(&path)
            .args(args)
            .env_clear()
            .env("LC_ALL", "C")
            .env("PATH", "/usr/bin:/bin")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("run {}", path.display()))?;
        let bounded = |r: Option<Box<dyn Read + Send>>| {
            std::thread::spawn(move || -> std::io::Result<Vec<u8>> {
                let mut out = Vec::new();
                if let Some(r) = r {
                    r.take(MAX_COMMAND_OUTPUT + 1).read_to_end(&mut out)?;
                }
                Ok(out)
            })
        };
        let stdout = bounded(
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
        );
        let stderr = bounded(
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
        );
        let start = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if start.elapsed() >= COMMAND_TIMEOUT {
                let _ = child.kill();
                let _ = child.wait();
                bail!("{program} did not finish within {COMMAND_TIMEOUT:?}");
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let join = |h: std::thread::JoinHandle<std::io::Result<Vec<u8>>>| -> Result<Vec<u8>> {
            let out = h
                .join()
                .map_err(|_| anyhow::anyhow!("{program} output reader panicked"))??;
            ensure!(
                out.len() as u64 <= MAX_COMMAND_OUTPUT,
                "{program} output exceeds {MAX_COMMAND_OUTPUT} bytes"
            );
            Ok(out)
        };
        Ok(Some(CommandOutput {
            success: status.success(),
            stdout: join(stdout)?,
            stderr: join(stderr)?,
        }))
    }
}
/// The host name file of a real host.
pub const SYSTEM_HOST_NAME: &str = "/proc/sys/kernel/hostname";
/// The machine id file of a real host.
pub const SYSTEM_MACHINE_ID: &str = "/etc/machine-id";
/// The boot id file of a real host.
pub const SYSTEM_BOOT_ID: &str = "/proc/sys/kernel/random/boot_id";
/// Every address assigned to a local interface, observed once.
pub type InterfaceSource = dyn Fn() -> Result<BTreeSet<IpAddr>>;

/// Where the local host is observed from: the observation provider of [`collect`],
/// [`observe_identity`] and every rendezvous participant (see the module documentation).
/// Each identity file is read bounded and regular, symlinks resolved, and its value is checked
/// where it is read.
pub struct HostSources<'a> {
    /// A file holding the host name ([`SYSTEM_HOST_NAME`] on a real host).
    pub host_name: PathBuf,
    /// A file holding the 32-hex machine id ([`SYSTEM_MACHINE_ID`] on a real host).
    pub machine_id: PathBuf,
    /// A file holding the boot id ([`SYSTEM_BOOT_ID`] on a real host).
    pub boot_id: PathBuf,
    /// The local interface addresses ([`local_addresses`] on a real host).
    pub interfaces: &'a InterfaceSource,
    /// The read-only inventory programs ([`SystemCommands`] on a real host).
    pub commands: &'a dyn CommandRunner,
}
impl HostSources<'static> {
    /// The real host. Only the production entrypoints construct this.
    pub fn system() -> Self {
        Self {
            host_name: PathBuf::from(SYSTEM_HOST_NAME),
            machine_id: PathBuf::from(SYSTEM_MACHINE_ID),
            boot_id: PathBuf::from(SYSTEM_BOOT_ID),
            interfaces: &local_addresses,
            commands: &SystemCommands,
        }
    }
}
/// The host's own name, machine id and boot id, as read from `sources` and checked where read.
fn observe_host(sources: &HostSources) -> Result<(String, String, String)> {
    let host = read_line(&sources.host_name)?;
    ensure!(
        is_name(&host),
        "host name `{host}` read from {} is not a census host name",
        sources.host_name.display()
    );
    let machine_id = read_line(&sources.machine_id)?;
    ensure!(
        super::census::is_machine_id(&machine_id),
        "machine id `{machine_id}` read from {} is not 32 lowercase hex",
        sources.machine_id.display()
    );
    let boot_id = read_line(&sources.boot_id)?;
    ensure!(
        super::census::is_boot_id(&boot_id),
        "boot id `{boot_id}` read from {} is empty or longer than 64 bytes",
        sources.boot_id.display()
    );
    Ok((host, machine_id, boot_id))
}

fn resolve_program(program: &str) -> Option<PathBuf> {
    if program.contains('/') {
        let p = PathBuf::from(program);
        return p.is_file().then_some(p);
    }
    let path = std::env::var_os("PATH").unwrap_or_else(|| OsString::from("/usr/bin:/bin"));
    std::env::split_paths(&path)
        .map(|d| d.join(program))
        .find(|p| {
            fs::metadata(p)
                .map(|m| m.is_file() && m.mode() & 0o111 != 0)
                .unwrap_or(false)
        })
}

/// What the operator names for this host's collection. Every path is hashed as installed; the
/// source commit is the installer-declared commit whose bytes `executable` binds.
#[derive(Clone, Debug)]
pub struct CollectSpec {
    /// The address this host is reached at; it must be assigned to a local interface.
    pub address: String,
    /// The host's transport certificate (DER, or PEM holding exactly one certificate).
    pub certificate: PathBuf,
    pub source_commit: String,
    /// The rank worker executable the plan pins (`executable_sha256` is its file SHA-256).
    pub executable: PathBuf,
    pub runtime_libraries: Vec<PathBuf>,
    pub compiler: PathBuf,
    /// Empty: the host has no CUDA runtime component (`cuda_sha256` is recorded as absent).
    pub cuda_libraries: Vec<PathBuf>,
    /// Empty: the host has no NCCL component (`nccl_sha256` is recorded as absent).
    pub nccl_libraries: Vec<PathBuf>,
    pub installed_policies: Vec<PathBuf>,
    /// The collector's own `CUDA_VISIBLE_DEVICES`: set and empty means a CPU-only census that
    /// records no GPU and runs no GPU probe; unset means the host-wide GPU inventory.
    pub cuda_visible_devices: Option<OsString>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GpuProbe {
    /// `CUDA_VISIBLE_DEVICES` was set and empty: no GPU recorded, nothing run.
    NotQueriedCudaVisibleDevicesEmpty,
    /// `nvidia-smi` is not installed: no GPU recorded.
    NvidiaSmiAbsent,
    NvidiaSmi,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CollectionReport {
    pub schema: String,
    pub census_sha256: String,
    pub host: String,
    pub machine_id: String,
    pub boot_id: String,
    pub address: String,
    pub observed_unix_ms: u64,
    pub gpu_probe: GpuProbe,
    pub gpus: u32,
    pub device_owners: u32,
    pub allocation_held: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MergedDocument {
    pub host: String,
    pub sha256: String,
    pub observed_unix_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MergeReport {
    pub schema: String,
    pub census_sha256: String,
    pub merged_unix_ms: u64,
    pub max_age_ms: u64,
    pub clock_skew_ms: u64,
    pub documents: Vec<MergedDocument>,
    pub allocation_held: bool,
}

mod collection;
mod merge;

pub use collection::{
    certificate_der, collect, file_closure_digest, hash_installed_file, local_addresses,
    observe_identity,
};
pub use merge::{merge, write_census, MergePolicy, PinnedDocument};

#[cfg(test)]
mod tests;
