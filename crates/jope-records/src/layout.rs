//! Where the committed records live in a repository checkout.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{Error, Result};

/// The core-IP record tree, relative to the repository root.
pub const CORE_IP: &str = "docs/internal/core-ip";

/// `docs/internal/core-ip` under `repo`.
pub fn core_ip(repo: &Path) -> PathBuf {
    repo.join(CORE_IP)
}

/// The findings register directory (`register.json`, `sources/`, `dispositions.json`).
pub fn findings(repo: &Path) -> PathBuf {
    core_ip(repo).join("findings")
}

/// The maintained regression test inventory.
pub fn test_inventory(repo: &Path) -> PathBuf {
    findings(repo).join("test-inventory.json")
}

/// The claims record.
pub fn claims(repo: &Path) -> PathBuf {
    core_ip(repo).join("claims").join("claims.json")
}

/// The work orders (`W*.tasks.md`, `task-status.json`, `STATUS.md`, `PROGRESS.md`).
pub fn work_orders(repo: &Path) -> PathBuf {
    core_ip(repo).join("work-orders")
}

/// The top level of the git checkout containing `dir` (`git rev-parse --show-toplevel`), as the
/// bytes git printed (without the line end): a checkout whose path is not UTF-8 is that path,
/// never a lossy conversion of it, which names a directory that does not exist.
pub fn git_toplevel(dir: &Path) -> Result<PathBuf> {
    use std::os::unix::ffi::OsStringExt;
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|e| Error::io(Path::new("git"), e))?;
    if !output.status.success() {
        return Err(Error::Command(format!(
            "{} is not inside a git checkout; pass --repo ({})",
            crate::py::os_display(dir.as_os_str()),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let mut top = output.stdout;
    while matches!(top.last(), Some(b'\n' | b'\r')) {
        top.pop();
    }
    Ok(PathBuf::from(std::ffi::OsString::from_vec(top)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Review round six of PR #117: the repository of a checkout whose path is not UTF-8 is found
    /// as the bytes it is (it was read lossily, U+FFFD for the byte, a directory that does not
    /// exist, so every guard run without `--repo` there failed).
    #[test]
    fn a_checkout_whose_path_is_not_utf8_is_found_as_it_is() {
        use std::os::unix::ffi::OsStrExt;
        let temp = tempfile::tempdir().unwrap();
        let repo = temp
            .path()
            .canonicalize()
            .unwrap()
            .join(std::ffi::OsStr::from_bytes(b"r\xff"));
        match std::fs::create_dir_all(repo.join("sub")) {
            Ok(()) => {
                let init = Command::new("git")
                    .arg("init")
                    .arg("-q")
                    .arg(&repo)
                    .status()
                    .unwrap();
                assert!(init.success());
                assert_eq!(git_toplevel(&repo.join("sub")).unwrap(), repo);
            }
            // APFS on macOS rejects non-UTF-8 filenames at creation time with EILSEQ (92).
            Err(e) if cfg!(target_os = "macos") && e.raw_os_error() == Some(92) => {}
            Err(e) => panic!("unexpected create_dir_all error: {e:?}"),
        }
        let outside = git_toplevel(temp.path()).map(|_| ());
        // The temporary directory may itself be inside a checkout; when it is not, the refusal
        // names it.
        if let Err(Error::Command(message)) = outside {
            assert!(
                message.contains("is not inside a git checkout"),
                "{message}"
            );
        }
    }
}
