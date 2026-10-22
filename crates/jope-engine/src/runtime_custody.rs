//! CPU-only observation of an authenticated Linux process's loaded images.
//! This observes backing files and startup environment, not live instruction
//! pages, a CUDA context, lazily loaded plugins, or a positive execution grant.
use crate::eval_registry::ProcessIdentity;
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, Metadata, OpenOptions},
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock, TryLockError},
    time::{Duration, Instant},
};

const MAX_PROC_BYTES: u64 = 4 * 1024 * 1024;
const MAX_IMAGE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_IMAGES: usize = 1024;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
struct FileIdentity {
    device: u64,
    inode: u64,
    bytes: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}
impl FileIdentity {
    fn read(metadata: &Metadata) -> Result<Self> {
        ensure!(metadata.is_file(), "loaded image is not a regular file");
        Ok(Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            bytes: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        })
    }
}

static IMAGE_HASHES: OnceLock<Mutex<BTreeMap<FileIdentity, String>>> = OnceLock::new();

fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn bounded_read(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(MAX_PROC_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_PROC_BYTES,
        "process metadata exceeds observation bound"
    );
    Ok(bytes)
}

fn image_hash(file: &mut File, expected: &FileIdentity, end: Instant) -> Result<String> {
    ensure!(Instant::now() < end, "runtime observation deadline elapsed");
    ensure!(
        FileIdentity::read(&file.metadata()?)? == *expected,
        "image changed before hash"
    );
    // Serialize misses so eight peer observations hash immutable shared images
    // once in this authority process. A cache hit still rechecks file metadata.
    let cache_mutex = IMAGE_HASHES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut cache = loop {
        ensure!(
            Instant::now() < end,
            "runtime observation deadline elapsed while waiting for image cache"
        );
        match cache_mutex.try_lock() {
            Ok(cache) => break cache,
            Err(TryLockError::Poisoned(_)) => anyhow::bail!("runtime image cache poisoned"),
            Err(TryLockError::WouldBlock) => std::thread::sleep(Duration::from_millis(1)),
        }
    };
    let digest = if let Some(digest) = cache.get(expected) {
        digest.clone()
    } else {
        ensure!(
            expected.bytes <= MAX_IMAGE_BYTES,
            "loaded image exceeds hashing bound"
        );
        let mut digest = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        let mut count = 0u64;
        loop {
            ensure!(Instant::now() < end, "runtime observation deadline elapsed");
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            count = count
                .checked_add(n as u64)
                .context("runtime image byte overflow")?;
            ensure!(count <= expected.bytes, "loaded image grew during hash");
            digest.update(&buffer[..n]);
        }
        ensure!(
            count == expected.bytes,
            "loaded image truncated during hash"
        );
        hex::encode(digest.finalize())
    };
    ensure!(
        FileIdentity::read(&file.metadata()?)? == *expected,
        "image changed while hashing"
    );
    if cache.len() >= 4096 {
        cache.clear();
    }
    cache.insert(expected.clone(), digest.clone());
    Ok(digest)
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Mapping {
    path: PathBuf,
    major: u64,
    minor: u64,
    inode: u64,
}

fn executable_mappings(bytes: &[u8]) -> Result<BTreeSet<Mapping>> {
    let text = std::str::from_utf8(bytes).context("non-UTF8 process mappings")?;
    let mut images = BTreeSet::new();
    for line in text.lines() {
        let mut tail = line;
        let mut fields = Vec::with_capacity(5);
        for _ in 0..5 {
            tail = tail.trim_start();
            let end = tail.find(char::is_whitespace).unwrap_or(tail.len());
            ensure!(end > 0, "malformed process mapping");
            fields.push(&tail[..end]);
            tail = &tail[end..];
        }
        if !fields[1].contains('x') {
            continue;
        }
        let path = tail.trim_start();
        // vDSO/vsyscall are kernel-generated; boot/kernel custody is separate.
        if path == "[vdso]" || path == "[vsyscall]" {
            continue;
        }
        ensure!(
            path.starts_with('/') && !path.ends_with(" (deleted)") && !fields[1].contains('w'),
            "unverifiable anonymous/deleted/writable executable mapping"
        );
        let (major, minor) = fields[3].split_once(':').context("mapping device absent")?;
        images.insert(Mapping {
            path: PathBuf::from(path),
            major: u64::from_str_radix(major, 16)?,
            minor: u64::from_str_radix(minor, 16)?,
            inode: fields[4].parse()?,
        });
        ensure!(images.len() <= MAX_IMAGES, "too many loaded images");
    }
    ensure!(!images.is_empty(), "no observable executable mappings");
    Ok(images)
}

fn numerical_environment(bytes: &[u8]) -> Result<BTreeMap<String, String>> {
    let mut selected = BTreeMap::new();
    for item in bytes.split(|b| *b == 0).filter(|item| !item.is_empty()) {
        let split = item
            .iter()
            .position(|b| *b == b'=')
            .context("malformed process environment")?;
        let key = std::str::from_utf8(&item[..split]).context("non-UTF8 environment key")?;
        let relevant = [
            "CUDA_",
            "CUBLAS_",
            "CUDNN_",
            "NCCL_",
            "TORCH_",
            "PYTORCH_",
            "OMP_",
            "MKL_",
            "OPENBLAS_",
            "BLIS_",
            "RAYON_",
            "KMP_",
        ]
        .iter()
        .any(|prefix| key.starts_with(prefix))
            || [
                "LD_PRELOAD",
                "LD_LIBRARY_PATH",
                "LD_AUDIT",
                "GLIBC_TUNABLES",
            ]
            .contains(&key);
        if relevant {
            let value = std::str::from_utf8(&item[split + 1..])
                .context("non-UTF8 numerical environment")?;
            ensure!(
                selected
                    .insert(key.to_string(), value.to_string())
                    .is_none(),
                "duplicate numerical environment key"
            );
        }
    }
    Ok(selected)
}

/// Observe a PID only after authenticating it independently (for example using
/// SO_PEERCRED). The same image/environment identity yields the same digest for
/// all ranks; process identity, ASLR, cgroup and affinity are separate evidence.
/// The caller must independently enforce the grant's deadline/resource caps and
/// obtain a post-context runtime binding before any qualified numerical work.
pub fn loaded_runtime_descriptor(process: &ProcessIdentity) -> Result<Value> {
    let deadline = monotonic_ns()?
        .checked_add(60_000_000_000)
        .context("runtime deadline overflow")?;
    loaded_runtime_descriptor_until(process, deadline)
}

fn monotonic_ns() -> Result<u64> {
    crate::host_os::monotonic_ns().context("runtime monotonic clock unavailable")
}

/// Use the original admission/task CLOCK_MONOTONIC deadline for both snapshots.
/// Hash loops and contended cache waits check this bound. A stuck filesystem
/// syscall still requires the caller's independent process watchdog; no expired
/// observation is returned as a successful runtime binding.
pub fn loaded_runtime_descriptor_until(
    process: &ProcessIdentity,
    deadline_monotonic_ns: u64,
) -> Result<Value> {
    let remaining = deadline_monotonic_ns
        .checked_sub(monotonic_ns()?)
        .filter(|value| *value > 0)
        .context("runtime observation deadline elapsed")?;
    let end = Instant::now()
        .checked_add(Duration::from_nanos(remaining))
        .context("runtime deadline overflow")?;
    ensure!(
        ProcessIdentity::read(process.pid)? == *process,
        "runtime peer birth changed"
    );
    let proc = PathBuf::from(format!("/proc/{}", process.pid));
    let maps_before = bounded_read(&proc.join("maps"))?;
    let mappings = executable_mappings(&maps_before)?;
    let env_before = bounded_read(&proc.join("environ"))?;
    let environment = numerical_environment(&env_before)?;
    let executable = FileIdentity::read(&fs::metadata(proc.join("exe"))?)?;
    let mut images = Vec::new();
    let mut open_images = Vec::new();
    let mut executable_sha = None;
    let mut total_bytes = 0u64;
    for mapping in &mappings {
        ensure!(Instant::now() < end, "runtime observation deadline elapsed");
        let through_root = proc.join("root").join(mapping.path.strip_prefix("/")?);
        let canonical_path = fs::canonicalize(&through_root)?;
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(&through_root)?;
        let identity = FileIdentity::read(&file.metadata()?)?;
        ensure!(
            identity.inode == mapping.inode
                && libc::major(identity.device) as u64 == mapping.major
                && libc::minor(identity.device) as u64 == mapping.minor,
            "loaded mapping backing file changed"
        );
        total_bytes = total_bytes
            .checked_add(identity.bytes)
            .context("runtime image size overflow")?;
        ensure!(
            total_bytes <= MAX_IMAGE_BYTES,
            "loaded images exceed observation byte bound"
        );
        let sha256 = image_hash(&mut file, &identity, end)?;
        if identity == executable {
            executable_sha = Some(sha256.clone());
        }
        images.push(json!({"path":canonical_path,"identity":identity,"sha256":sha256}));
        open_images.push((through_root, file, identity));
    }
    images.sort_by_key(|image| image["path"].as_str().unwrap_or("").to_string());
    let executable_sha256 = executable_sha.context("executable not observed in loaded mappings")?;
    let normalized = json!({"schema":"parallel.loaded_runtime_identity.v1", "images":images,
        "numerical_environment":environment, "executable_sha256":executable_sha256});
    let status = String::from_utf8(bounded_read(&proc.join("status"))?)?;
    let affinity: Vec<_> = status
        .lines()
        .filter(|line| {
            line.starts_with("Cpus_allowed_list:") || line.starts_with("Mems_allowed_list:")
        })
        .map(str::to_string)
        .collect();
    let cgroup = String::from_utf8(bounded_read(&proc.join("cgroup"))?)?;
    let maps_after = bounded_read(&proc.join("maps"))?;
    ensure!(
        executable_mappings(&maps_after)? == mappings
            && numerical_environment(&bounded_read(&proc.join("environ"))?)? == environment,
        "runtime images/environment changed during observation"
    );
    for (path, file, identity) in open_images {
        ensure!(
            FileIdentity::read(&file.metadata()?)? == identity
                && FileIdentity::read(&fs::metadata(path)?)? == identity,
            "loaded image changed after hash"
        );
    }
    ensure!(
        FileIdentity::read(&fs::metadata(proc.join("exe"))?)? == executable
            && ProcessIdentity::read(process.pid)? == *process
            && Instant::now() < end
            && monotonic_ns()? < deadline_monotonic_ns,
        "runtime peer changed or observation deadline elapsed"
    );
    Ok(
        json!({"schema":"parallel.loaded_runtime_observation.v1", "process":process,
        "runtime_sha256":hash(&serde_json::to_vec(&normalized)?), "executable_sha256":executable_sha256,
        "normalized":normalized, "maps_before_sha256":hash(&maps_before),
        "maps_after_sha256":hash(&maps_after), "cgroup":cgroup, "affinity":affinity,
        "observation":"CPU-only loaded backing files and proc startup numerical environment",
        "live_instruction_pages_verified":false, "cuda_context_verified":false,
        "lazy_plugins_verified":false, "post_context_binding_required":true,
        "execution_authority":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_expired_deadline_rejects_before_process_or_image_observation() {
        let absent = ProcessIdentity {
            boot_id: "absent".into(),
            pid: u32::MAX,
            birth_ticks: 1,
        };
        let error = loaded_runtime_descriptor_until(&absent, 0).unwrap_err();
        assert!(error.to_string().contains("deadline elapsed"));
    }
    #[test]
    fn contended_image_cache_does_not_extend_original_deadline() {
        let mut file = tempfile::tempfile().unwrap();
        let identity = FileIdentity::read(&file.metadata().unwrap()).unwrap();
        let cache = IMAGE_HASHES.get_or_init(|| Mutex::new(BTreeMap::new()));
        let guard = cache.lock().unwrap();
        let error = image_hash(
            &mut file,
            &identity,
            Instant::now() + Duration::from_millis(10),
        )
        .unwrap_err();
        drop(guard);
        assert!(error
            .to_string()
            .contains("deadline elapsed while waiting for image cache"));
    }
    #[test]
    fn rank_and_aslr_do_not_change_normalized_observations() {
        let first =
            b"1000-2000 r-xp 00000000 08:01 3 /lib/a.so\n3000-4000 r-xp 00000000 00:00 0 [vdso]\n";
        let second = b"9000-a000 r-xp 00000000 08:01 3 /lib/a.so\n";
        assert_eq!(
            executable_mappings(first).unwrap(),
            executable_mappings(second).unwrap()
        );
        assert_eq!(
            numerical_environment(b"RANK=0\0CUDA_VISIBLE_DEVICES=0,1\0NCCL_DEBUG=WARN\0").unwrap(),
            numerical_environment(b"RANK=1\0NCCL_DEBUG=WARN\0CUDA_VISIBLE_DEVICES=0,1\0").unwrap()
        );
        assert_ne!(
            numerical_environment(b"OMP_NUM_THREADS=2\0").unwrap(),
            numerical_environment(b"OMP_NUM_THREADS=3\0").unwrap()
        );
    }
    #[test]
    fn unobservable_executable_images_and_duplicate_environment_reject() {
        for path in ["", "[anon:jit]", "/lib/a.so (deleted)"] {
            assert!(
                executable_mappings(format!("1000-2000 r-xp 0 08:01 3 {path}\n").as_bytes())
                    .is_err()
            );
        }
        assert!(executable_mappings(b"1000-2000 rwxp 0 08:01 3 /lib/a.so\n").is_err());
        assert!(numerical_environment(b"NCCL_DEBUG=WARN\0NCCL_DEBUG=INFO\0").is_err());
    }
    #[test]
    fn cached_image_digest_invalidates_when_backing_file_changes() {
        let path = std::env::temp_dir().join(format!("jope-runtime-cache-{}", std::process::id()));
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        use std::io::{Seek, SeekFrom, Write};
        file.write_all(b"old").unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        let identity = FileIdentity::read(&file.metadata().unwrap()).unwrap();
        let end = Instant::now() + Duration::from_secs(5);
        let first = image_hash(&mut file, &identity, end).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        assert_eq!(image_hash(&mut file, &identity, end).unwrap(), first);
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(b"changed").unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        assert!(image_hash(&mut file, &identity, end).is_err());
        let changed = FileIdentity::read(&file.metadata().unwrap()).unwrap();
        assert_ne!(image_hash(&mut file, &changed, end).unwrap(), first);
        fs::remove_file(path).unwrap();
    }
}
