use crate::manifest::*;
use anyhow::{ensure, Context, Result};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    thread,
};

pub fn verify(manifest: &Path, root: &Path) -> Result<usize> {
    let m: TransferManifest = read_json(manifest)?;
    m.validate()?;
    for f in &m.files {
        let p = resolve(root, &f.destination)?;
        ensure!(
            fs::metadata(&p)?.len() == f.bytes && file_hash(&p)? == f.sha256,
            "transfer mismatch: {}",
            p.display()
        );
    }
    Ok(m.files.len())
}
pub fn stage(manifest: &Path, root: &Path) -> Result<()> {
    let m: TransferManifest = read_json(manifest)?;
    m.validate()?;
    fs::create_dir_all(root)?;
    for f in &m.files {
        ensure!(
            !fs::symlink_metadata(&f.source)?.file_type().is_symlink(),
            "source symlink not allowed"
        );
        ensure!(
            fs::metadata(&f.source)?.len() == f.bytes && file_hash(&f.source)? == f.sha256,
            "source changed: {}",
            f.source.display()
        );
        let dst = resolve(root, &f.destination)?;
        if dst.exists() {
            ensure!(
                file_hash(&dst)? == f.sha256,
                "immutable staged path collision"
            );
            continue;
        }
        fs::create_dir_all(dst.parent().unwrap())?;
        let mutable_source = f.role.contains("source")
            || f.role.contains("provenance")
            || f.destination.starts_with("manifests")
            || f.destination.starts_with("native")
            || f.destination.starts_with("src")
            || f.destination.starts_with("tools");
        if mutable_source || fs::hard_link(&f.source, &dst).is_err() {
            let tmp = dst.with_extension("partial");
            fs::copy(&f.source, &tmp)?;
            ensure!(file_hash(&tmp)? == f.sha256, "source changed during copy");
            fs::rename(tmp, dst)?;
        }
    }
    Ok(())
}
pub fn transfer(
    manifest: &Path,
    stage_root: &Path,
    host: &str,
    identity: &Path,
    destination: &str,
) -> Result<()> {
    ensure!(
        !host.is_empty()
            && host
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "@.-_".contains(c)),
        "invalid SSH destination"
    );
    ensure!(
        destination.starts_with('/')
            && destination
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "/.-_".contains(c)),
        "destination must be an absolute shell-safe path"
    );
    ensure!(
        identity
            .to_string_lossy()
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "/.-_".contains(c)),
        "identity path contains unsupported characters"
    );
    stage(manifest, stage_root)?;
    let m: TransferManifest = read_json(manifest)?;
    let lists = stage_root.join(".transfer-lists");
    fs::create_dir_all(&lists)?;
    let mut shards: [Vec<&FileEntry>; 4] = Default::default();
    let mut sizes = [0u64; 4];
    let mut files = m.files.iter().collect::<Vec<_>>();
    files.sort_by_key(|f| std::cmp::Reverse(f.bytes));
    for f in files {
        let shard = (0..4).min_by_key(|&i| sizes[i]).unwrap();
        shards[shard].push(f);
        sizes[shard] += f.bytes;
    }
    let mut paths: Vec<PathBuf> = Vec::new();
    for (i, entries) in shards.into_iter().enumerate() {
        let path = lists.join(format!("{i}.files0"));
        let mut file = fs::File::create(&path)?;
        for f in entries {
            file.write_all(f.destination.as_os_str().as_encoded_bytes())?;
            file.write_all(&[0])?;
        }
        paths.push(path);
    }
    let status = Command::new("ssh")
        .args(["-i"])
        .arg(identity)
        .args([
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            host,
            "mkdir",
            "-p",
            destination,
        ])
        .status()?;
    ensure!(status.success(), "remote directory creation failed");
    let shell = format!(
        "ssh -i {} -o BatchMode=yes -o ConnectTimeout=20",
        identity.display()
    );
    thread::scope(|scope| -> Result<()> {
        let workers = paths
            .iter()
            .map(|path| {
                let shell = &shell;
                scope.spawn(move || -> Result<()> {
                    let status = Command::new("rsync")
                        .args([
                            "-a",
                            "--partial",
                            "--partial-dir=.rsync-partial",
                            "--from0",
                            "--protect-args",
                            "--files-from",
                        ])
                        .arg(path)
                        .args(["-e", shell])
                        .arg(format!("{}/", stage_root.display()))
                        .arg(format!("{host}:{destination}/"))
                        .status()?;
                    ensure!(status.success(), "rsync stream failed: {}", path.display());
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        let mut errors = Vec::new();
        for w in workers {
            if let Err(e) = w
                .join()
                .map_err(|_| anyhow::anyhow!("transfer thread panicked"))
                .and_then(|x| x)
            {
                errors.push(e.to_string());
            }
        }
        ensure!(errors.is_empty(), "{}", errors.join("; "));
        Ok(())
    })?;
    // Build a checksum list from the manifest, and independently hash remote bytes.
    let mut sums = String::new();
    for f in &m.files {
        let path = f.destination.to_str().context("non-UTF8 transfer path")?;
        ensure!(
            !path.contains(['\n', '\r', '\\']),
            "unsupported checksum filename"
        );
        sums.push_str(&format!("{}  {}\n", f.sha256, path));
    }
    use std::process::Stdio;
    let mut child = Command::new("ssh")
        .arg("-i")
        .arg(identity)
        .args(["-o", "BatchMode=yes", host])
        .arg(format!("cd {destination} && sha256sum --quiet -c -"))
        .stdin(Stdio::piped())
        .spawn()?;
    child.stdin.take().unwrap().write_all(sums.as_bytes())?;
    ensure!(
        child.wait()?.success(),
        "remote SHA-256 reconciliation failed"
    );
    write_json(
        &lists.join("verified.json"),
        &serde_json::json!({"schema":"parallel.transfer_receipt.v1","manifest_sha256":file_hash(manifest)?,"files":m.files.len(),"bytes":sizes.iter().sum::<u64>(),"host":host,"destination":destination,"verified":true}),
    )?;
    Ok(())
}
