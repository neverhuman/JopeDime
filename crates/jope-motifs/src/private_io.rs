//! Private and shareable output paths, following the immutable private-artifact pattern of
//! the frozen panel (`crates/jope-bench/src/panel/worker.rs:66-76`).
//!
//! Private outputs must be absolute, new, outside every temporary root, outside any
//! version-controlled checkout and outside every state root; directories are created 0700 and
//! files 0600. The shareable summary must be absolute, new and outside temporary and state
//! roots, may live in a checkout, and is written 0644. Unix only; other platforms fail closed.
//!
//! The state roots are the product's resolved default and, when a verb names one with
//! `--state`, that root too: an output is refused under either.

use crate::{Code, Coded};
use anyhow::{ensure, Result};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub(crate) struct Policy {
    temp_roots: Vec<PathBuf>,
    refuse_git_ancestors: bool,
    state_roots: Vec<PathBuf>,
}

impl Policy {
    /// `/tmp`, `/var/tmp`, `/dev/shm` and the process temporary directory are temporary
    /// roots; version-controlled checkouts are refused for private outputs; nothing may be
    /// written under any of `state_roots`.
    pub(crate) fn standard(state_roots: Vec<PathBuf>) -> Self {
        Self {
            temp_roots: vec![
                PathBuf::from("/tmp"),
                PathBuf::from("/var/tmp"),
                PathBuf::from("/dev/shm"),
                std::env::temp_dir(),
            ],
            refuse_git_ancestors: true,
            state_roots,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(temp_roots: Vec<PathBuf>, refuse_git_ancestors: bool) -> Self {
        Self {
            temp_roots,
            refuse_git_ancestors,
            state_roots: Vec::new(),
        }
    }

    /// The same policy with one more state root refused.
    pub(crate) fn with_state_root(mut self, state_root: PathBuf) -> Self {
        self.state_roots.push(state_root);
        self
    }

    /// A new private output path. Returns the path under its canonical parent.
    pub(crate) fn check_private(&self, path: &Path) -> Result<PathBuf> {
        let parent = self.checked_parent(path)?;
        if self.refuse_git_ancestors {
            let versioned = parent
                .ancestors()
                .any(|dir| dir.join(".git").symlink_metadata().is_ok());
            ensure!(!versioned, Code("output_path_refused"));
        }
        Ok(parent.join(file_name(path)?))
    }

    /// A new shareable output path. Returns the path under its canonical parent.
    pub(crate) fn check_public(&self, path: &Path) -> Result<PathBuf> {
        let parent = self.checked_parent(path)?;
        Ok(parent.join(file_name(path)?))
    }

    fn checked_parent(&self, path: &Path) -> Result<PathBuf> {
        let refused = Code("output_path_refused");
        ensure!(path.is_absolute(), refused);
        ensure!(path.symlink_metadata().is_err(), refused);
        let parent = path
            .parent()
            .ok_or(refused)?
            .canonicalize()
            .map_err(|_| refused)?;
        for root in &self.temp_roots {
            let root = root.canonicalize().unwrap_or_else(|_| root.clone());
            ensure!(!parent.starts_with(root), refused);
        }
        for state in &self.state_roots {
            let state = state.canonicalize().unwrap_or_else(|_| state.clone());
            ensure!(!parent.starts_with(state), refused);
        }
        Ok(parent)
    }
}

fn file_name(path: &Path) -> Result<&OsStr> {
    Ok(path.file_name().ok_or(Code("output_path_refused"))?)
}

/// Create a new private directory, mode 0700.
#[cfg(unix)]
pub(crate) fn create_private_dir(policy: &Policy, path: &Path) -> Result<PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    let target = policy.check_private(path)?;
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&target)
        .coded("output_path_refused")?;
    std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o700)).coded("io")?;
    Ok(target)
}

#[cfg(not(unix))]
pub(crate) fn create_private_dir(_policy: &Policy, _path: &Path) -> Result<PathBuf> {
    anyhow::bail!(Code("output_path_refused"))
}

/// Write a new private file, mode 0600.
pub(crate) fn write_private_file(policy: &Policy, path: &Path, bytes: &[u8]) -> Result<()> {
    let target = policy.check_private(path)?;
    write_new(&target, bytes, 0o600)
}

/// Write a new shareable file, mode 0644.
pub(crate) fn write_public_file(policy: &Policy, path: &Path, bytes: &[u8]) -> Result<()> {
    let target = policy.check_public(path)?;
    write_new(&target, bytes, 0o644)
}

/// Create `target` new with `mode` and write it completely. If anything after creation
/// fails, the partial file is removed, so a failed write never leaves a truncated report or
/// summary where a complete one is expected.
#[cfg(unix)]
fn write_new(target: &Path, bytes: &[u8], mode: u32) -> Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(mode)
        .open(target)
        .coded("output_path_refused")?;
    if let Err(error) = fill(&mut file, bytes, mode) {
        drop(file);
        // Best effort: the removal itself failing cannot be reported more usefully.
        let _ = std::fs::remove_file(target);
        return Err::<(), _>(error).coded("io");
    }
    Ok(())
}

#[cfg(unix)]
fn fill(file: &mut std::fs::File, bytes: &[u8], mode: u32) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    file.set_permissions(std::fs::Permissions::from_mode(mode))?;
    file.write_all(bytes)?;
    file.sync_all()
}

#[cfg(not(unix))]
fn write_new(_target: &Path, _bytes: &[u8], _mode: u32) -> Result<()> {
    anyhow::bail!(Code("output_path_refused"))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::code_of;
    use std::os::unix::fs::PermissionsExt;

    fn refused(result: Result<PathBuf>) -> bool {
        result
            .err()
            .is_some_and(|error| code_of(&error) == Some("output_path_refused"))
    }

    #[test]
    fn private_paths_reject_relative_temporary_git_checkout_and_existing_targets() {
        let root = tempfile::tempdir().expect("temporary directory");
        let plain = root.path().join("plain");
        let checkout = root.path().join("checkout");
        std::fs::create_dir(&plain).expect("plain directory");
        std::fs::create_dir_all(checkout.join(".git")).expect("checkout marker");
        std::fs::create_dir(checkout.join("nested")).expect("nested directory");
        std::fs::write(plain.join("existing"), b"x").expect("existing file");

        let open = Policy::for_test(Vec::new(), false);
        assert!(open.check_private(&plain.join("report.json")).is_ok());
        assert!(refused(
            open.check_private(Path::new("relative/report.json"))
        ));
        assert!(refused(open.check_private(&plain.join("existing"))));
        assert!(refused(
            open.check_private(&plain.join("absent/report.json"))
        ));

        let temporary = Policy::for_test(vec![plain.clone()], false);
        assert!(refused(temporary.check_private(&plain.join("report.json"))));
        assert!(temporary
            .check_private(&checkout.join("nested/report.json"))
            .is_ok());

        let versioned = Policy::for_test(Vec::new(), true);
        assert!(refused(
            versioned.check_private(&checkout.join("nested/report.json"))
        ));
        assert!(refused(
            versioned.check_private(&checkout.join("report.json"))
        ));

        // The standard policy treats the process temporary directory as temporary.
        let standard = Policy::standard(Vec::new());
        assert!(refused(standard.check_private(&plain.join("report.json"))));
        assert!(refused(standard.check_public(&plain.join("summary.json"))));
    }

    #[test]
    fn private_files_are_0600_in_a_0700_directory() {
        let root = tempfile::tempdir().expect("temporary directory");
        let policy = Policy::for_test(Vec::new(), false);
        let directory = create_private_dir(&policy, &root.path().join("private")).expect("dir");
        let mode = std::fs::metadata(&directory)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o700);
        let file = directory.join("report.json");
        write_private_file(&policy, &file, b"{}").expect("write");
        let mode = std::fs::metadata(&file)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
        assert_eq!(std::fs::read(&file).expect("read"), b"{}");
        let again = write_private_file(&policy, &file, b"{}").expect_err("never overwritten");
        assert_eq!(code_of(&again), Some("output_path_refused"));
        let again = create_private_dir(&policy, &directory).expect_err("never reused");
        assert_eq!(code_of(&again), Some("output_path_refused"));
    }

    #[test]
    fn public_out_may_live_in_a_checkout_but_not_in_temporary_or_state_paths() {
        let root = tempfile::tempdir().expect("temporary directory");
        let checkout = root.path().join("checkout");
        let state = root.path().join("state");
        std::fs::create_dir_all(checkout.join(".git")).expect("checkout marker");
        std::fs::create_dir(&state).expect("state directory");

        let policy = Policy::for_test(Vec::new(), true);
        let summary = checkout.join("summary.json");
        assert!(refused(policy.check_private(&summary)));
        write_public_file(&policy, &summary, b"{}").expect("public write in a checkout");
        let mode = std::fs::metadata(&summary)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o644);
        assert!(refused(policy.check_public(&summary)));

        let temporary = Policy::for_test(vec![checkout.clone()], true);
        assert!(refused(
            temporary.check_public(&checkout.join("other.json"))
        ));

        let stateful = Policy::for_test(Vec::new(), false).with_state_root(state.clone());
        assert!(refused(stateful.check_public(&state.join("summary.json"))));
        assert!(refused(stateful.check_private(&state.join("report.json"))));
        assert!(stateful
            .check_public(&root.path().join("summary.json"))
            .is_ok());

        // Every state root is refused: the resolved default and an explicit one alike.
        let explicit = root.path().join("explicit");
        std::fs::create_dir(&explicit).expect("explicit state directory");
        let both = stateful.clone().with_state_root(explicit.clone());
        assert!(refused(both.check_private(&state.join("report.json"))));
        assert!(refused(both.check_private(&explicit.join("report.json"))));
        assert!(refused(both.check_public(&explicit.join("summary.json"))));
        assert!(both.check_private(&root.path().join("report.json")).is_ok());
        assert!(stateful
            .check_private(&explicit.join("report.json"))
            .is_ok());
    }
}
