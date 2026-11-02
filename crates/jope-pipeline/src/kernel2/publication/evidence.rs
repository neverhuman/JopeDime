use super::*;

pub(super) fn read_evidence(root: &Path, relative: &Path) -> Result<Vec<u8>> {
    let mut path = root.to_path_buf();
    ensure!(!relative.as_os_str().is_empty(), "empty evidence path");
    for component in relative.components() {
        let Component::Normal(name) = component else {
            anyhow::bail!("unsafe evidence path");
        };
        path.push(name);
        ensure!(
            !std::fs::symlink_metadata(&path)?.file_type().is_symlink(),
            "symlink evidence path"
        );
    }
    std::fs::read(&path).with_context(|| format!("read {}", path.display()))
}

pub(super) fn read_file(root: &Path, relative: &Path, digest: &str) -> Result<Vec<u8>> {
    let bytes = read_evidence(root, relative)?;
    ensure!(
        sha256_hex(&bytes) == digest,
        "{}: evidence digest mismatch",
        root.join(relative).display()
    );
    Ok(bytes)
}
