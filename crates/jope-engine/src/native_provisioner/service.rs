//! Bounded serial outbox consumer. `provision` remains the only signer/transaction path.
use super::*;
use std::{
    collections::BTreeMap,
    ffi::CString,
    os::fd::{FromRawFd, OwnedFd},
    path::Component,
    time::{Duration, Instant},
};
const MAX_ENTRIES: usize = 100_000;
const MAX_SCAN_BYTES: usize = 100 * 1024 * 1024;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Outbox {
    schema: String,
    run: String,
    generation: u64,
    role: Role,
    frozen_config: Artifact,
    request: Artifact,
    admission: Artifact,
    task_manifest: Artifact,
    authorization_issued: bool,
}
fn directory(path: &Path) -> Result<OwnedFd> {
    ensure!(path.is_absolute(), "absolute outbox directory required");
    let slash = CString::new("/")?;
    let fd = unsafe {
        libc::open(
            slash.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
        )
    };
    ensure!(fd >= 0, "cannot open directory root");
    let mut fd = unsafe { OwnedFd::from_raw_fd(fd) };
    for part in path.components() {
        let Component::Normal(name) = part else {
            ensure!(
                matches!(part, Component::RootDir),
                "unsafe outbox component"
            );
            continue;
        };
        let name = CString::new(name.as_encoded_bytes())?;
        let next = unsafe {
            libc::openat(
                fd.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if next < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        fd = unsafe { OwnedFd::from_raw_fd(next) };
    }
    Ok(fd)
}
fn at(fd: &OwnedFd, name: &str) -> Result<Vec<u8>> {
    ensure!(
        !name.contains('/') && !matches!(name, "." | ".."),
        "invalid outbox entry name"
    );
    let name = CString::new(name)?;
    let raw = unsafe {
        libc::openat(
            fd.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
        )
    };
    if raw < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let mut file = unsafe { File::from_raw_fd(raw) };
    let before = file.metadata()?;
    ensure!(
        before.is_file() && before.len() <= LIMIT,
        "bounded regular outbox record required"
    );
    let mut b = Vec::new();
    Read::by_ref(&mut file)
        .take(LIMIT + 1)
        .read_to_end(&mut b)?;
    let after = file.metadata()?;
    ensure!(
        b.len() as u64 == before.len()
            && after.len() == before.len()
            && before.ctime() == after.ctime()
            && before.ctime_nsec() == after.ctime_nsec()
            && before.mtime() == after.mtime()
            && before.mtime_nsec() == after.mtime_nsec(),
        "outbox record changed during read"
    );
    Ok(b)
}
fn names(fd: &OwnedFd) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(format!("/proc/self/fd/{}", fd.as_raw_fd()))? {
        ensure!(
            out.len() < MAX_ENTRIES * 5 + 32,
            "bounded outbox/journal entry count exceeded"
        );
        out.push(
            entry?
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("nonUTF8 outbox entry"))?,
        );
    }
    out.sort();
    Ok(out)
}
fn selected_name(name: &str) -> bool {
    name.strip_suffix(".json").is_some_and(|x| {
        x.len() == 64
            && x.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    })
}
fn scan(config: &Config, run: &str) -> Result<Vec<TaskInput>> {
    ensure!(
        config.loop_state.starts_with(&config.root),
        "outbox outside pinned data root"
    );
    let path = config.loop_state.join("authority-requests");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let fd = directory(&path)?;
    let mut rows = Vec::new();
    let mut total = 0usize;
    for name in names(&fd)? {
        if !selected_name(&name) {
            continue;
        }
        let b = at(&fd, &name)?;
        total = total.checked_add(b.len()).context("outbox size overflow")?;
        ensure!(total <= MAX_SCAN_BYTES, "bounded outbox bytes exceeded");
        let o: Outbox = serde_json::from_slice(&b)?;
        ensure!(
            o.schema == "jope.native_grant_request_outbox.v1"
                && o.run == run
                && o.generation < 2
                && matches!(o.role, Role::Direct | Role::Continuation)
                && !o.authorization_issued
                && o.frozen_config == config.loop_config,
            "outbox is not original pinned run eligibility"
        );
        for a in [&o.request, &o.admission, &o.task_manifest] {
            crate::manifest::relative(&a.path)?;
            ensure!(
                a.sha256.len() == 64
                    && a.sha256
                        .bytes()
                        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
                "outbox hash invalid"
            );
        }
        rows.push((
            o.generation,
            if o.role == Role::Direct { 0 } else { 1 },
            name,
            TaskInput {
                request: o.request,
                admission: o.admission,
                task_manifest: o.task_manifest,
            },
        ));
    }
    // A path replacement cannot silently move the scan to a different directory.
    let original = File::open(format!("/proc/self/fd/{}", fd.as_raw_fd()))?.metadata()?;
    let named = fs::symlink_metadata(&path)?;
    ensure!(
        !named.file_type().is_symlink()
            && named.dev() == original.dev()
            && named.ino() == original.ino(),
        "outbox directory replaced"
    );
    rows.sort_by(|a, b| (&a.0, &a.1, &a.2).cmp(&(&b.0, &b.1, &b.2)));
    Ok(rows.into_iter().map(|x| x.3).collect())
}
/// Return original unsettled input even when its public outbox has disappeared.
fn pending(config: &Config) -> Result<Option<TaskInput>> {
    protected(&config.state, true)?;
    let fd = directory(&config.state)?;
    let entries = names(&fd)?;
    for name in &entries {
        let Some(prefix) = name.strip_suffix(".intent.json") else {
            continue;
        };
        ensure!(
            prefix.len() == 8 && prefix.bytes().all(|b| b.is_ascii_digit()),
            "invalid original intent name"
        );
        if entries
            .binary_search(&format!("{prefix}.installed.json"))
            .is_ok()
        {
            continue;
        }
        let path = config.state.join(name);
        protected(&path, false)?;
        let intent: Intent = serde_json::from_slice(&at(&fd, name)?)?;
        // Full signature, original config, sequence and receipt validation happens inside provision().
        return Ok(Some(TaskInput {
            request: intent.proof.inputs.request,
            admission: intent.proof.inputs.admission,
            task_manifest: intent.proof.inputs.task_manifest,
        }));
    }
    Ok(None)
}
/// Explicitly bounded process. No new service can extend a signed run deadline.
pub async fn serve(config_path: &Path, poll_ms: u64, max_runtime_seconds: u64) -> Result<()> {
    ensure!(
        (500..=60_000).contains(&poll_ms) && (1..=172_800).contains(&max_runtime_seconds),
        "invalid bounded provisioner service interval/runtime"
    );
    ensure!(
        crate::host_os::effective_uid() == 0,
        "root provisioner custody required"
    );
    protected(config_path, false)?;
    let original = read(config_path)?;
    let config: Config = serde_json::from_slice(&original)?;
    ensure!(
        config.schema == "jope.native_grant_provisioner.v1"
            && crate::manifest::file_hash(Path::new("/proc/self/exe"))? == config.executable_sha256,
        "unreviewed provisioner service image/config"
    );
    let loop_config: LoopConfig =
        serde_json::from_slice(&checked(&config.root, &config.loop_config)?)?;
    let start = Instant::now();
    let mut done = BTreeMap::<String, Artifact>::new();
    while start.elapsed() < Duration::from_secs(max_runtime_seconds) {
        protected(config_path, false)?;
        ensure!(
            read(config_path)? == original,
            "provisioner config changed during service"
        );
        let original_pending = pending(&config)?;
        // Pending intents are reconciled even after task expiry; no fresh work is selected then.
        if original_pending.is_none() && now()? >= loop_config.task_deadline.as_unix_millis() {
            break;
        }
        let retrying_original = original_pending.is_some();
        let candidates = match original_pending {
            Some(p) => Ok(vec![p]),
            None => scan(&config, &loop_config.run),
        };
        let result: Result<()> = async {
            for input in candidates? {
                let identity = hash(&bytes(&input)?);
                if let Some(policy) = done.get(&identity).filter(|_| !retrying_original) {
                    protected(&policy.path, false)?;
                    if hash(&read(&policy.path)?) == policy.sha256 {
                        continue;
                    }
                    anyhow::bail!("previously verified installed policy changed");
                }
                let receipt = provision(config_path, &input).await?;
                let installed: Artifact = serde_json::from_value(receipt["policy"].clone())?;
                ensure!(
                    done.len() < MAX_ENTRIES,
                    "bounded provisioner completed map exhausted"
                );
                done.insert(identity, installed);
                println!("{}", serde_json::to_string(&receipt)?);
                if start.elapsed() >= Duration::from_secs(max_runtime_seconds) {
                    break;
                }
            }
            Ok(())
        }
        .await;
        if let Err(e) = result {
            println!(
                "{}",
                json!({"schema":"jope.native_provisioner_wait.v1","run":loop_config.run,"at_unix_ms":now()?,"reason":e.to_string(),"new_authorization_claimed":false})
            );
        }
        tokio::time::sleep(Duration::from_millis(poll_ms)).await;
    }
    println!(
        "{}",
        json!({"schema":"jope.native_provisioner_stopped.v1","run":loop_config.run,"bounded_runtime_seconds":max_runtime_seconds,"pending_intents_preserved":true})
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn descriptor_reads_refuse_symlinks_fifo_and_ancestor_escape() {
        let t = tempfile::tempdir().unwrap();
        fs::create_dir(t.path().join("d")).unwrap();
        fs::write(t.path().join("d/a"), b"original").unwrap();
        let d = directory(&t.path().join("d")).unwrap();
        assert_eq!(at(&d, "a").unwrap(), b"original");
        std::os::unix::fs::symlink("a", t.path().join("d/link")).unwrap();
        assert!(at(&d, "link").is_err());
        assert!(at(&d, "../outside").is_err());
        std::os::unix::fs::symlink(t.path().join("d"), t.path().join("alias")).unwrap();
        assert!(directory(&t.path().join("alias")).is_err());
        let path = CString::new(t.path().join("d/fifo").as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
        assert!(at(&d, "fifo").is_err());
    }
    #[test]
    fn held_directory_descriptor_cannot_redirect_reads_after_rename() {
        let t = tempfile::tempdir().unwrap();
        let original = t.path().join("d");
        fs::create_dir(&original).unwrap();
        fs::write(original.join("a"), b"original").unwrap();
        let d = directory(&original).unwrap();
        fs::rename(&original, t.path().join("old")).unwrap();
        fs::create_dir(&original).unwrap();
        fs::write(original.join("a"), b"replacement").unwrap();
        assert_eq!(at(&d, "a").unwrap(), b"original");
    }
    #[test]
    fn only_exact_request_names_are_selected() {
        assert!(selected_name(&format!("{}.json", "a".repeat(64))));
        for name in [
            "../a.json",
            "a.json",
            "authority.admission.json",
            ".pending-1",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.json",
        ] {
            assert!(!selected_name(name));
        }
    }
}
