//! Installed trust is custody, not a client-supplied declaration.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs::{self, OpenOptions}, io::Read, os::unix::fs::{MetadataExt, OpenOptionsExt}, path::{Component, Path, PathBuf}};
use super::funding::FundingTrust;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientConfig {
    pub schema: String,
    pub client_executable: PathBuf,
    pub client_executable_sha256: String,
    pub socket: PathBuf,
    pub authority_root: PathBuf,
    pub witness_path: PathBuf,
    pub trust: FundingTrust,
    pub witness_policy_sha256: String,
}

#[derive(Clone)]
pub struct InstalledFunding {
    path: PathBuf,
    sha256: String,
    config: ClientConfig,
    #[cfg(test)]
    private_fixture: bool,
}

fn protected_read(path: &Path) -> Result<Vec<u8>> {
    ensure!(path.is_absolute(), "funding configuration must be an absolute installed path");
    let mut current = PathBuf::new();
    for component in path.components() {
        ensure!(matches!(component, Component::RootDir | Component::Normal(_)), "noncanonical funding configuration path");
        current.push(component);
        let meta = fs::symlink_metadata(&current)?;
        ensure!(!meta.file_type().is_symlink() && meta.uid() == 0 && meta.mode() & 0o022 == 0,
            "funding configuration and ancestors must be root-owned, nonsymlink and not group/other writable");
        ensure!(if current == path { meta.is_file() } else { meta.is_dir() },
            "funding configuration must be a regular file under real directories");
    }
    let mut file = OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(path)?;
    let before = file.metadata()?;
    ensure!(before.is_file() && before.len() <= 1024 * 1024 && before.uid() == 0 && before.mode() & 0o022 == 0,
        "invalid installed funding configuration file");
    let mut bytes = Vec::new();
    (&mut file).take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    let named = fs::symlink_metadata(path)?;
    ensure!(before.dev() == named.dev() && before.ino() == named.ino()
        && before.len() == after.len() && before.mtime_nsec() == after.mtime_nsec()
        && before.mtime() == after.mtime() && before.ctime() == after.ctime()
        && before.ctime_nsec() == after.ctime_nsec() && bytes.len() as u64 == before.len(),
        "installed funding configuration changed during observation");
    Ok(bytes)
}

impl InstalledFunding {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = protected_read(path)?;
        let config: ClientConfig = serde_json::from_slice(&bytes)?;
        ensure!(config.schema == "parallel.funding_client.v1", "unsupported funding client configuration");
        for path in [&config.client_executable, &config.socket, &config.authority_root, &config.witness_path] {
            ensure!(path.is_absolute(), "funding runtime paths must be absolute");
        }
        ensure!(config.witness_policy_sha256.len() == 64 && config.client_executable_sha256.len() == 64,
            "funding executable/witness pins absent");
        Ok(Self { path: path.into(), sha256: hex::encode(Sha256::digest(&bytes)), config,
            #[cfg(test)] private_fixture: false })
    }
    /// Temporary unit-test configuration only. This constructor is absent from
    /// production binaries; signatures, authority pins and witness checks still
    /// execute normally, and changing the original fixture bytes still fails.
    #[cfg(test)]
    pub(crate) fn for_test(path: &Path) -> Result<Self> {
        ensure!(fs::metadata(path)?.len() <= 1024 * 1024, "fixture config exceeds bound");
        let bytes = fs::read(path)?;
        let config: ClientConfig = serde_json::from_slice(&bytes)?;
        ensure!(config.schema == "parallel.funding_client.v1", "unsupported fixture config");
        Ok(Self { path:path.into(), sha256:hex::encode(Sha256::digest(&bytes)), config, private_fixture:true })
    }
    pub fn revalidate(&self) -> Result<&ClientConfig> {
        #[cfg(test)]
        let bytes = if self.private_fixture {
            ensure!(fs::metadata(&self.path)?.len() <= 1024 * 1024, "fixture config exceeds bound");
            fs::read(&self.path)?
        } else { protected_read(&self.path)? };
        #[cfg(not(test))]
        let bytes = protected_read(&self.path)?;
        ensure!(hex::encode(Sha256::digest(bytes)) == self.sha256,
            "installed funding configuration replaced; reconcile before admission");
        Ok(&self.config)
    }
    pub fn trust(&self) -> Result<&FundingTrust> { Ok(&self.revalidate()?.trust) }
    pub fn verify_client(&self) -> Result<()> {
        let config = self.revalidate()?;
        let current = fs::canonicalize("/proc/self/exe")?;
        ensure!(current == fs::canonicalize(&config.client_executable)?, "unapproved funding client executable path");
        let bytes = fs::read("/proc/self/exe").context("funding client executable unavailable")?;
        ensure!(hex::encode(Sha256::digest(bytes)) == config.client_executable_sha256,
            "funding client executable digest differs from installation");
        Ok(())
    }
}
