//! Qualification borrows existing GP slots after their current searches finish.
use crate::manifest::{file_hash, read_json, resolve};
use anyhow::{ensure, Result};
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    fs::{File, OpenOptions},
    os::fd::AsRawFd,
    path::Path,
};

#[derive(Deserialize)]
struct Reservation {
    schema: String,
    manifest: std::path::PathBuf,
    manifest_sha256: String,
    slots: Vec<usize>,
    created_at: u64,
    expires_at: u64,
}

pub fn reserved(root: &Path, workers: usize, threads: usize, now: u64) -> Result<BTreeSet<usize>> {
    let path = root.join("runs/gp-qualification-reserve.json");
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    let request: Reservation = read_json(&path)?;
    ensure!(
        request.schema == "parallel.gp_slot_reservation.v1",
        "unknown GP slot reservation"
    );
    ensure!(
        request.created_at <= now
            && request.expires_at > request.created_at
            && request.expires_at - request.created_at <= 7200,
        "invalid GP reservation lifetime"
    );
    // Expiry releases admission, but cannot release a flock still held by a
    // surviving qualification process.
    if now >= request.expires_at {
        return Ok(BTreeSet::new());
    }
    ensure!(
        threads == 1 && !request.slots.is_empty() && request.slots.len() <= 16,
        "qualification needs at most sixteen single-core GP slots"
    );
    let slots = request.slots.iter().copied().collect::<BTreeSet<_>>();
    ensure!(
        slots.len() == request.slots.len() && slots.iter().all(|slot| *slot < workers),
        "invalid or duplicate qualification slots"
    );
    ensure!(
        file_hash(&resolve(root, &request.manifest)?)? == request.manifest_sha256,
        "qualification reservation manifest changed"
    );
    Ok(slots)
}

pub fn try_acquire(root: &Path, slot: usize) -> Result<Option<File>> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.join(format!("runs/gp-slot-{slot}.lock")))?;
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(Some(file));
    }
    let error = std::io::Error::last_os_error();
    if error.kind() == std::io::ErrorKind::WouldBlock {
        return Ok(None);
    }
    Err(error.into())
}

/// Called in pre_exec: the search owns the slot until it exits. Dropping the
/// supervisor's copy must not release it.
pub fn inherit(fd: std::os::fd::RawFd) -> std::io::Result<()> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::write_json;
    use std::{
        os::unix::process::CommandExt,
        process::{Command, Stdio},
    };

    #[test]
    fn reservation_is_bounded_identity_valid_and_expiring() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        std::fs::create_dir(root.join("runs")).unwrap();
        write_json(&root.join("trial.json"), &serde_json::json!({"trial":1})).unwrap();
        let mut value = serde_json::json!({"schema":"parallel.gp_slot_reservation.v1",
            "manifest":"trial.json","manifest_sha256":file_hash(&root.join("trial.json")).unwrap(),
            "slots":[88,103],"created_at":100,"expires_at":200});
        let path = root.join("runs/gp-qualification-reserve.json");
        write_json(&path, &value).unwrap();
        assert_eq!(
            reserved(root, 104, 1, 150).unwrap(),
            BTreeSet::from([88, 103])
        );
        assert!(reserved(root, 104, 2, 150).is_err());
        assert!(reserved(root, 104, 1, 200).unwrap().is_empty());
        value["slots"] = serde_json::json!([88, 88]);
        write_json(&path, &value).unwrap();
        assert!(reserved(root, 104, 1, 150).is_err());
        value["slots"] = serde_json::json!((0..17).collect::<Vec<_>>());
        write_json(&path, &value).unwrap();
        assert!(reserved(root, 104, 1, 150).is_err());
        value["slots"] = serde_json::json!([103]);
        write_json(&path, &value).unwrap();
        std::fs::write(root.join("trial.json"), "tampered").unwrap();
        assert!(reserved(root, 104, 1, 150).is_err());
    }

    #[test]
    fn inherited_lease_survives_supervisor_copy_and_releases_at_child_exit() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        std::fs::create_dir(root.join("runs")).unwrap();
        let lease = try_acquire(root, 103).unwrap().unwrap();
        let fd = lease.as_raw_fd();
        let mut command = Command::new("cat");
        command.stdin(Stdio::piped()).stdout(Stdio::null());
        unsafe {
            command.pre_exec(move || inherit(fd));
        }
        let mut child = command.spawn().unwrap();
        drop(lease);
        assert!(try_acquire(root, 103).unwrap().is_none());
        drop(child.stdin.take());
        assert!(child.wait().unwrap().success());
        assert!(try_acquire(root, 103).unwrap().is_some());
    }
}
