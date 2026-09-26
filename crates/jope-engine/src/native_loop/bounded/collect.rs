//! Local host census collection and per-host census merge (W7-T01).
//!
//! [`collect`] observes the **local host only** and produces exactly the typed
//! `jope.native_host_census.v1` record that [`super::census::preflight`] validates. It reads
//! `/etc/machine-id`, the kernel boot id, the host name and the local interface addresses; the
//! configured certificate; the SHA-256 of the executables, runtime libraries and installed
//! policy files it is given; online CPUs and available memory; and, unless GPUs are hidden, the
//! GPU inventory and device-owning processes from `nvidia-smi`. It never contacts another host,
//! never opens a device and never allocates or reserves anything: a census is an observation,
//! not a reservation, and admission re-checks it.
//!
//! Nothing is invented. An absent `nvidia-smi` or `CUDA_VISIBLE_DEVICES=""` records no GPU; a
//! failing or unparseable probe, a value that is not available, or an identity input that cannot
//! be verified locally refuses the whole collection.
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

/// Read a small local file (bounded, regular). A symlink resolves to its installed target,
/// as `/etc/machine-id` or a certificate may legitimately be one.
fn read_small(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let canonical =
        fs::canonicalize(path).with_context(|| format!("resolve {}", path.display()))?;
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(&canonical)
        .with_context(|| format!("open {}", path.display()))?;
    let meta = f.metadata()?;
    // procfs files report length 0; they are regular files all the same.
    ensure!(meta.is_file(), "{} is not a regular file", path.display());
    let mut b = Vec::new();
    f.take(limit + 1).read_to_end(&mut b)?;
    ensure!(
        b.len() as u64 <= limit,
        "{} exceeds {limit} bytes",
        path.display()
    );
    Ok(b)
}
fn read_line(path: &Path) -> Result<String> {
    let text = String::from_utf8(read_small(path, 256)?)
        .with_context(|| format!("{} is not UTF-8", path.display()))?;
    Ok(text.trim().to_string())
}

/// SHA-256 of one installed file, streamed, with its size and identity re-checked afterwards.
/// Symlinks resolve to the installed file, whose canonical path is returned.
pub fn hash_installed_file(path: &Path) -> Result<(PathBuf, String, u64)> {
    ensure!(path.is_absolute(), "{} must be absolute", path.display());
    let canonical =
        fs::canonicalize(path).with_context(|| format!("resolve {}", path.display()))?;
    let mut f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(&canonical)
        .with_context(|| format!("open {}", canonical.display()))?;
    let before = f.metadata()?;
    ensure!(
        before.is_file() && before.len() <= MAX_HASHED_FILE_BYTES,
        "{} is not a bounded regular file",
        canonical.display()
    );
    let mut state = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut count = 0u64;
    loop {
        let n = f.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        count += n as u64;
        ensure!(
            count <= before.len(),
            "{} grew while hashing",
            canonical.display()
        );
        state.update(&buffer[..n]);
    }
    let after = f.metadata()?;
    ensure!(
        count == before.len()
            && after.len() == before.len()
            && after.ino() == before.ino()
            && after.mtime() == before.mtime()
            && after.mtime_nsec() == before.mtime_nsec(),
        "{} changed while hashing",
        canonical.display()
    );
    Ok((canonical, hex::encode(state.finalize()), count))
}
/// Digest of a set of installed files: `(canonical path, sha256, bytes)` sorted by path.
pub fn file_closure_digest(paths: &[PathBuf]) -> Result<String> {
    ensure!(
        !paths.is_empty() && paths.len() <= MAX_FILES_PER_CLOSURE,
        "file closure must name 1..={MAX_FILES_PER_CLOSURE} files"
    );
    let mut entries = BTreeMap::new();
    for p in paths {
        let (canonical, sha, bytes) = hash_installed_file(p)?;
        ensure!(
            entries.insert(canonical.clone(), (sha, bytes)).is_none(),
            "{} named twice",
            canonical.display()
        );
    }
    let entries: Vec<_> = entries.into_iter().map(|(p, (s, b))| (p, s, b)).collect();
    digest(&(FILE_CLOSURE_SCHEMA, entries))
}

/// Strict standard base64 (RFC 4648 with padding), ignoring ASCII whitespace.
fn base64_decode(text: &str) -> Result<Vec<u8>> {
    let clean: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    ensure!(
        !clean.is_empty() && clean.len() % 4 == 0,
        "base64 length is not a positive multiple of 4"
    );
    let value = |c: u8| -> Result<u32> {
        Ok(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => bail!("invalid base64 byte"),
        } as u32)
    };
    let mut out = Vec::with_capacity(clean.len() / 4 * 3);
    let chunks = clean.chunks(4).count();
    for (i, chunk) in clean.chunks(4).enumerate() {
        let pad = chunk.iter().rev().take_while(|&&c| c == b'=').count();
        ensure!(
            pad <= 2 && (pad == 0 || i + 1 == chunks),
            "misplaced base64 padding"
        );
        let mut n = 0u32;
        for &c in &chunk[..4 - pad] {
            n = (n << 6) | value(c)?;
        }
        n <<= 6 * pad as u32;
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&bytes[..3 - pad]);
    }
    Ok(out)
}
/// DER bytes of one certificate: DER as given, or the single `CERTIFICATE` block of a PEM file.
/// Only the outer framing is checked here (a SEQUENCE whose first element is a SEQUENCE); the
/// transport verifies the certificate itself.
pub fn certificate_der(bytes: &[u8]) -> Result<Vec<u8>> {
    const BEGIN: &str = "-----BEGIN CERTIFICATE-----";
    const END: &str = "-----END CERTIFICATE-----";
    let der = if bytes.starts_with(b"-----BEGIN") {
        let text = std::str::from_utf8(bytes).context("PEM certificate is not UTF-8")?;
        ensure!(
            text.matches(BEGIN).count() == 1 && text.matches("-----BEGIN").count() == 1,
            "PEM file must hold exactly one CERTIFICATE block"
        );
        let body = text
            .split_once(BEGIN)
            .and_then(|(_, rest)| rest.split_once(END))
            .map(|(body, _)| body)
            .context("PEM CERTIFICATE block is not terminated")?;
        base64_decode(body)?
    } else {
        bytes.to_vec()
    };
    let (outer, rest) = der_element(&der)?;
    ensure!(
        rest.is_empty() && outer.0 == 0x30,
        "certificate is not a single DER SEQUENCE"
    );
    let (inner, _) = der_element(outer.1)?;
    ensure!(
        inner.0 == 0x30,
        "certificate does not start with a TBSCertificate SEQUENCE"
    );
    Ok(der)
}
/// One DER TLV: `((tag, contents), remainder)`, definite lengths only.
pub(crate) fn der_element(b: &[u8]) -> Result<((u8, &[u8]), &[u8])> {
    ensure!(b.len() >= 2, "truncated DER element");
    let tag = b[0];
    let (len, header) = match b[1] {
        n if n < 0x80 => (n as usize, 2),
        0x81..=0x84 => {
            let k = (b[1] & 0x7f) as usize;
            ensure!(b.len() >= 2 + k, "truncated DER length");
            let mut len = 0usize;
            for &x in &b[2..2 + k] {
                len = (len << 8) | x as usize;
            }
            ensure!(len >= 0x80, "non-minimal DER length");
            (len, 2 + k)
        }
        _ => bail!("unsupported DER length form"),
    };
    ensure!(b.len() >= header + len, "truncated DER contents");
    Ok(((tag, &b[header..header + len]), &b[header + len..]))
}

/// Every IP address assigned to a local interface.
pub fn local_addresses() -> Result<BTreeSet<IpAddr>> {
    let mut head: *mut libc::ifaddrs = std::ptr::null_mut();
    ensure!(
        unsafe { libc::getifaddrs(&mut head) } == 0,
        "getifaddrs: {}",
        std::io::Error::last_os_error()
    );
    let mut out = BTreeSet::new();
    let mut cursor = head;
    while !cursor.is_null() {
        // SAFETY: getifaddrs returned a valid list that is freed only below.
        let ifa = unsafe { &*cursor };
        if !ifa.ifa_addr.is_null() {
            match i32::from(unsafe { (*ifa.ifa_addr).sa_family }) {
                libc::AF_INET => {
                    let sa = unsafe { &*(ifa.ifa_addr as *const libc::sockaddr_in) };
                    out.insert(IpAddr::V4(Ipv4Addr::from(u32::from_be(sa.sin_addr.s_addr))));
                }
                libc::AF_INET6 => {
                    let sa = unsafe { &*(ifa.ifa_addr as *const libc::sockaddr_in6) };
                    out.insert(IpAddr::V6(Ipv6Addr::from(sa.sin6_addr.s6_addr)));
                }
                _ => {}
            }
        }
        cursor = ifa.ifa_next;
    }
    unsafe { libc::freeifaddrs(head) };
    Ok(out)
}

fn online_cpu_millicores() -> Result<u32> {
    let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    ensure!(n > 0, "online CPU count unavailable");
    u32::try_from(n)
        .ok()
        .and_then(|n| n.checked_mul(1000))
        .context("online CPU count out of range")
}
/// `MemAvailable` from `/proc/meminfo`: memory available now, not total capacity.
fn available_memory_bytes() -> Result<u64> {
    let text = String::from_utf8(read_small(Path::new("/proc/meminfo"), 64 * 1024)?)?;
    let kib: u64 = text
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().strip_suffix("kB"))
        .context("MemAvailable absent from /proc/meminfo")?
        .trim()
        .parse()?;
    kib.checked_mul(1024).context("MemAvailable overflow")
}

fn csv_fields(line: &str, n: usize) -> Result<Vec<&str>> {
    let fields: Vec<&str> = line.splitn(n, ',').map(str::trim).collect();
    ensure!(
        fields.len() == n && fields.iter().all(|f| !f.is_empty()),
        "unexpected nvidia-smi record `{line}`"
    );
    ensure!(
        !fields.iter().any(|f| f.starts_with('[')),
        "nvidia-smi reports an unavailable value in `{line}`; nothing is invented"
    );
    Ok(fields)
}
fn probe_text(out: &CommandOutput, what: &str) -> Result<String> {
    ensure!(
        out.success,
        "nvidia-smi {what} failed: {}",
        String::from_utf8_lossy(&out.stderr)
            .chars()
            .take(400)
            .collect::<String>()
    );
    String::from_utf8(out.stdout.clone()).with_context(|| format!("nvidia-smi {what} output"))
}
/// GPUs, the common driver version and device-owning processes, host-wide.
fn gpu_inventory(
    runner: &dyn CommandRunner,
    visible: Option<&OsString>,
) -> Result<(
    GpuProbe,
    Vec<GpuObservation>,
    Option<String>,
    Vec<Ownership>,
)> {
    match visible {
        Some(v) if v.is_empty() => {
            return Ok((
                GpuProbe::NotQueriedCudaVisibleDevicesEmpty,
                Vec::new(),
                None,
                Vec::new(),
            ))
        }
        Some(_) => bail!(
            "the census is host-wide: unset CUDA_VISIBLE_DEVICES, or set it to the empty string \
             for a CPU-only census that records no GPU"
        ),
        None => {}
    }
    let Some(out) = runner.run("nvidia-smi", &GPU_QUERY)? else {
        return Ok((GpuProbe::NvidiaSmiAbsent, Vec::new(), None, Vec::new()));
    };
    let mut gpus = Vec::new();
    let mut drivers = BTreeSet::new();
    for line in probe_text(&out, "GPU query")?
        .lines()
        .filter(|l| !l.trim().is_empty())
    {
        let f = csv_fields(line, 6)?;
        let uuid = f[1].to_string();
        ensure!(
            uuid.starts_with("GPU-") && is_name(&uuid),
            "nvidia-smi GPU UUID `{uuid}` is not in GPU-<uuid> form"
        );
        ensure!(
            f[3].len() <= 64 && f[3].bytes().all(|b| b.is_ascii_digit() || b == b'.'),
            "nvidia-smi driver version `{}`",
            f[3]
        );
        drivers.insert(f[3].to_string());
        let mib = |s: &str| -> Result<u64> {
            s.parse::<u64>()
                .ok()
                .and_then(|m| m.checked_mul(MIB))
                .with_context(|| format!("nvidia-smi memory value `{s}`"))
        };
        gpus.push(GpuObservation {
            uuid,
            model: f[2].to_string(),
            local_index: f[0]
                .parse()
                .with_context(|| format!("nvidia-smi index `{}`", f[0]))?,
            total_vram_bytes: mib(f[4])?,
            available_vram_bytes: mib(f[5])?,
        });
    }
    ensure!(
        drivers.len() <= 1,
        "GPUs on one host report different driver versions"
    );
    let known: BTreeSet<&str> = gpus.iter().map(|g| g.uuid.as_str()).collect();
    let apps = runner
        .run("nvidia-smi", &APPS_QUERY)?
        .context("nvidia-smi disappeared between the GPU and process queries")?;
    let mut owners = Vec::new();
    for line in probe_text(&apps, "compute-apps query")?
        .lines()
        .filter(|l| !l.trim().is_empty() && l.trim() != "No running processes found")
    {
        let f = csv_fields(line, 3)?;
        ensure!(
            known.contains(f[0]),
            "a device process names GPU `{}`, which the GPU query did not report",
            f[0]
        );
        let pid: u32 = f[1]
            .parse()
            .with_context(|| format!("nvidia-smi pid `{}`", f[1]))?;
        // An observed device process is an owner the collector cannot attribute to a managed
        // run; it is recorded as a non-managed owner, which conflicts with any plan for it.
        owners.push(Ownership {
            owner: format!("gpu-process-{pid}"),
            kind: OwnerKind::User,
            device_uuid: Some(f[0].to_string()),
            cpu_millicores: 0,
            ram_bytes: 0,
            run: None,
            attempt: None,
        });
    }
    Ok((
        GpuProbe::NvidiaSmi,
        gpus,
        drivers.into_iter().next(),
        owners,
    ))
}

/// Observe the local host. Refuses rather than inventing any value.
pub fn collect(
    spec: &CollectSpec,
    runner: &dyn CommandRunner,
    now_unix_ms: u64,
) -> Result<(HostCensus, CollectionReport)> {
    let host = read_line(Path::new("/proc/sys/kernel/hostname"))?;
    ensure!(
        is_name(&host),
        "host name `{host}` is not a census host name"
    );
    let machine_id = read_line(Path::new("/etc/machine-id"))?;
    let boot_id = read_line(Path::new("/proc/sys/kernel/random/boot_id"))?;
    let address: IpAddr = spec
        .address
        .parse()
        .with_context(|| format!("address `{}` is not an IP address", spec.address))?;
    ensure!(
        local_addresses()?.contains(&address),
        "address {address} is not assigned to a local interface"
    );
    ensure!(
        is_commit(&spec.source_commit),
        "source commit must be 40 lowercase hex"
    );
    let certificate = read_small(&spec.certificate, MAX_CERTIFICATE_BYTES)
        .and_then(|b| certificate_der(&b))
        .with_context(|| format!("certificate {}", spec.certificate.display()))?;
    let closure = |paths: &[PathBuf]| -> Result<Option<String>> {
        if paths.is_empty() {
            Ok(None)
        } else {
            file_closure_digest(paths).map(Some)
        }
    };
    ensure!(
        !spec.runtime_libraries.is_empty() && !spec.installed_policies.is_empty(),
        "at least one runtime library and one installed policy file are required"
    );
    let (gpu_probe, gpus, driver_version, ownership) =
        gpu_inventory(runner, spec.cuda_visible_devices.as_ref())?;
    let identity = HostIdentity {
        host,
        machine_id,
        boot_id,
        address: address.to_string(),
        certificate_sha256: hash(&certificate),
        source_commit: spec.source_commit.clone(),
        executable_sha256: hash_installed_file(&spec.executable)?.1,
        installed_policy_sha256: file_closure_digest(&spec.installed_policies)?,
        runtime: RuntimeIdentity {
            runtime_sha256: file_closure_digest(&spec.runtime_libraries)?,
            compiler_sha256: hash_installed_file(&spec.compiler)?.1,
            cuda_sha256: closure(&spec.cuda_libraries)?,
            nccl_sha256: closure(&spec.nccl_libraries)?,
            driver_version,
        },
    };
    let census = HostCensus {
        schema: CENSUS_SCHEMA.into(),
        captured_unix_ms: now_unix_ms,
        hosts: vec![HostObservation {
            identity,
            observed_unix_ms: now_unix_ms,
            cpu_millicores_allocatable: online_cpu_millicores()?,
            ram_bytes_allocatable: available_memory_bytes()?,
            gpus,
            ownership,
        }],
    };
    census.validate()?;
    let bytes = serde_json::to_vec(&census)?;
    ensure!(bytes.len() <= MAX_CENSUS_BYTES, "collected census bound");
    let h = &census.hosts[0];
    let report = CollectionReport {
        schema: COLLECTION_SCHEMA.into(),
        census_sha256: hash(&bytes),
        host: h.identity.host.clone(),
        machine_id: h.identity.machine_id.clone(),
        boot_id: h.identity.boot_id.clone(),
        address: h.identity.address.clone(),
        observed_unix_ms: now_unix_ms,
        gpu_probe,
        gpus: h.gpus.len() as u32,
        device_owners: h.ownership.len() as u32,
        allocation_held: false,
    };
    Ok((census, report))
}

/// Freshness window applied to every per-host document; the same bounds as a plan's telemetry
/// policy (`max_age_ms` 1..=600000, `clock_skew_ms` <= 10000).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MergePolicy {
    pub max_age_ms: u64,
    pub clock_skew_ms: u64,
}
/// One per-host document and the SHA-256 its collector reported for it.
#[derive(Clone, Debug)]
pub struct PinnedDocument {
    pub label: String,
    pub bytes: Vec<u8>,
    pub sha256: String,
}

/// Combine pinned per-host census documents into one multi-host census.
pub fn merge(
    documents: &[PinnedDocument],
    policy: MergePolicy,
    now_unix_ms: u64,
) -> Result<(HostCensus, MergeReport)> {
    ensure!(
        (1..=600_000).contains(&policy.max_age_ms) && policy.clock_skew_ms <= 10_000,
        "merge freshness policy must be max_age_ms 1..=600000 and clock_skew_ms <= 10000"
    );
    ensure!(
        !documents.is_empty() && documents.len() <= 64,
        "merge takes 1..=64 per-host documents"
    );
    // Pin every document before interpreting any of it.
    let mut digests = BTreeSet::new();
    for d in documents {
        ensure!(
            is_digest(&d.sha256),
            "document {} pin must be 64 lowercase hex",
            d.label
        );
        ensure!(
            d.bytes.len() <= MAX_CENSUS_BYTES,
            "document {} exceeds {MAX_CENSUS_BYTES} bytes",
            d.label
        );
        let actual = hash(&d.bytes);
        ensure!(
            actual == d.sha256,
            "document {} was altered: its SHA-256 {actual} differs from the pinned {}",
            d.label,
            d.sha256
        );
        ensure!(
            digests.insert(actual),
            "document {} is given twice",
            d.label
        );
    }
    let mut schemas = BTreeSet::new();
    for d in documents {
        let v: serde_json::Value = serde_json::from_slice(&d.bytes)
            .with_context(|| format!("document {} is not JSON", d.label))?;
        schemas.insert(
            v.get("schema")
                .and_then(|s| s.as_str())
                .unwrap_or("<none>")
                .to_string(),
        );
    }
    ensure!(
        schemas.len() == 1 && schemas.contains(CENSUS_SCHEMA),
        "mixed or unsupported census schema versions {schemas:?}; every document must be `{CENSUS_SCHEMA}`"
    );
    let mut hosts: Vec<(HostObservation, MergedDocument)> = Vec::new();
    let mut captured = 0u64;
    let mut seen: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    for d in documents {
        let census =
            HostCensus::from_bytes(&d.bytes).with_context(|| format!("document {}", d.label))?;
        ensure!(
            census.hosts.len() == 1,
            "document {} must describe exactly one host",
            d.label
        );
        let h = census.hosts.into_iter().next().context("host")?;
        for t in [census.captured_unix_ms, h.observed_unix_ms] {
            ensure!(
                t <= now_unix_ms.saturating_add(policy.clock_skew_ms),
                "document {} is dated {} ms in the future",
                d.label,
                t - now_unix_ms
            );
            ensure!(
                now_unix_ms.saturating_sub(t) <= policy.max_age_ms,
                "document {} is stale: {} ms old, beyond {} ms",
                d.label,
                now_unix_ms - t,
                policy.max_age_ms
            );
        }
        let i = &h.identity;
        for (what, value) in [
            ("host", i.host.clone()),
            ("machine id", i.machine_id.clone()),
            ("boot id", i.boot_id.clone()),
            ("address", i.address.clone()),
            ("certificate", i.certificate_sha256.clone()),
        ] {
            ensure!(
                seen.entry(what).or_default().insert(value.clone()),
                "document {} duplicates the {what} `{value}` of another document",
                d.label
            );
        }
        captured = captured.max(census.captured_unix_ms);
        hosts.push((
            h.clone(),
            MergedDocument {
                host: h.identity.host.clone(),
                sha256: d.sha256.clone(),
                observed_unix_ms: h.observed_unix_ms,
            },
        ));
    }
    hosts.sort_by(|a, b| a.0.identity.host.cmp(&b.0.identity.host));
    let merged = HostCensus {
        schema: CENSUS_SCHEMA.into(),
        captured_unix_ms: captured,
        hosts: hosts.iter().map(|(h, _)| h.clone()).collect(),
    };
    merged.validate()?;
    let bytes = serde_json::to_vec(&merged)?;
    ensure!(bytes.len() <= MAX_CENSUS_BYTES, "merged census bound");
    let report = MergeReport {
        schema: MERGE_SCHEMA.into(),
        census_sha256: hash(&bytes),
        merged_unix_ms: now_unix_ms,
        max_age_ms: policy.max_age_ms,
        clock_skew_ms: policy.clock_skew_ms,
        documents: hosts.into_iter().map(|(_, d)| d).collect(),
        allocation_held: false,
    };
    Ok((merged, report))
}

/// Publish census bytes at `out` without replacement (an existing different file is refused
/// and kept). The parent directory must already exist.
pub fn write_census(out: &Path, census: &HostCensus) -> Result<Vec<u8>> {
    ensure!(out.is_absolute(), "--out must be an absolute path");
    let parent = out.parent().context("--out parent")?;
    let name = out.file_name().context("--out file name")?;
    ensure!(
        fs::symlink_metadata(parent)
            .map(|m| m.is_dir())
            .unwrap_or(false),
        "--out parent {} must be an existing directory",
        parent.display()
    );
    ensure!(
        fs::symlink_metadata(out).is_err(),
        "{} already exists; a census is never replaced",
        out.display()
    );
    let bytes = serde_json::to_vec(census)?;
    super::rendezvous::publish_immutable(parent, Path::new(name), &bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn base64_and_pem_framing_are_strict() {
        assert_eq!(base64_decode("TWFu").unwrap(), b"Man");
        assert_eq!(base64_decode("TWE=").unwrap(), b"Ma");
        assert_eq!(base64_decode("TQ==\n").unwrap(), b"M");
        for bad in ["TQ=", "T===", "TQ==TWFu", "TW!u", ""] {
            assert!(base64_decode(bad).is_err(), "{bad}");
        }
        let der = [0x30, 0x03, 0x30, 0x01, 0x00];
        assert_eq!(certificate_der(&der).unwrap(), der);
        let pem = "-----BEGIN CERTIFICATE-----\nMAMwAQA=\n-----END CERTIFICATE-----\n";
        assert_eq!(certificate_der(pem.as_bytes()).unwrap(), der);
        assert!(certificate_der(format!("{pem}{pem}").as_bytes()).is_err());
        assert!(certificate_der(&[0x30, 0x03, 0x02, 0x01, 0x00]).is_err());
        assert!(certificate_der(&[0x30, 0x03, 0x30, 0x01]).is_err());
        assert!(certificate_der(&[0x30, 0x03, 0x30, 0x01, 0x00, 0x00]).is_err());
    }
}
