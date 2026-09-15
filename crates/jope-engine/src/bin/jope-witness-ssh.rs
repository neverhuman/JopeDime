//! Forced OpenSSH entrypoint: two fixed witness operations, no shell evaluation.
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use std::{
    fs,
    io::Read,
    os::{
        fd::AsRawFd,
        unix::{
            fs::{MetadataExt, OpenOptionsExt},
            process::CommandExt,
        },
    },
    path::PathBuf,
    process::Command,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    executable: PathBuf,
    executable_sha256: String,
    root: PathBuf,
    policy: PathBuf,
    key: PathBuf,
}
fn main() -> Result<()> {
    let path = PathBuf::from("/etc/jope-fit-witness/ssh-launcher.json");
    for ancestor in path.ancestors() {
        let meta = fs::symlink_metadata(ancestor)?;
        ensure!(
            !meta.file_type().is_symlink() && meta.uid() == 0 && meta.mode() & 0o022 == 0,
            "launcher config custody changed"
        );
    }
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    ensure!(
        file.metadata()?.is_file(),
        "regular launcher config required"
    );
    let mut bytes = Vec::new();
    file.take(65537).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 65536, "launcher config too large");
    let c: Config = serde_json::from_slice(&bytes)?;
    for path in [&c.executable, &c.root, &c.policy, &c.key] {
        ensure!(
            path.is_absolute() && fs::canonicalize(path)? == *path,
            "physical witness path required"
        );
        ensure!(
            path.to_str()
                .context("non UTF8 witness path")?
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"/._-".contains(&b)),
            "witness path not representable in fixed transport grammar"
        );
    }
    use sha2::{Digest, Sha256};
    let mut executable = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&c.executable)?;
    let executable_meta = executable.metadata()?;
    ensure!(
        executable_meta.is_file()
            && executable_meta.uid() == 0
            && executable_meta.mode() & 0o022 == 0,
        "root-owned immutable witness executable required"
    );
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = executable.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    ensure!(
        hex::encode(hasher.finalize()) == c.executable_sha256,
        "witness executable changed"
    );
    let original = std::env::var("SSH_ORIGINAL_COMMAND").context("forced SSH command required")?;
    let operation = [
        "registry-witness-request",
        "registry-witness-get-ack",
        "registry-witness-add-native-grant",
        "registry-witness-epoch-transition",
    ]
    .into_iter()
    .find(|op| {
        original
            == format!(
                "{} {} --root {} --policy {} --key {}",
                c.executable.display(),
                op,
                c.root.display(),
                c.policy.display(),
                c.key.display()
            )
    })
    .context("unrecognized witness command")?;
    let error = Command::new(format!("/proc/self/fd/{}", executable.as_raw_fd()))
        .arg(operation)
        .arg("--root")
        .arg(&c.root)
        .arg("--policy")
        .arg(&c.policy)
        .arg("--key")
        .arg(&c.key)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1")
        .exec();
    Err(error.into())
}
