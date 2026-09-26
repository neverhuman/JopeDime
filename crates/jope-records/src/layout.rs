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

/// The top level of the git checkout containing `dir` (`git rev-parse --show-toplevel`).
pub fn git_toplevel(dir: &Path) -> Result<PathBuf> {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|e| Error::io(Path::new("git"), e))?;
    if !output.status.success() {
        return Err(Error::Command(format!(
            "{} is not inside a git checkout; pass --repo ({})",
            dir.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let top = String::from_utf8_lossy(&output.stdout)
        .trim_end_matches(['\n', '\r'])
        .to_owned();
    Ok(PathBuf::from(top))
}
