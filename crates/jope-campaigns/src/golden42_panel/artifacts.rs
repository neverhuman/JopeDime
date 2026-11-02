use super::*;

pub(super) fn source_tree_hash() -> Result<String> {
    let workspace = workspace_root();
    let mut paths = vec![
        workspace.join("Cargo.toml"),
        workspace.join("Cargo.lock"),
        workspace.join("scripts/run_golden42_panel_max_xbabe2.sh"),
    ];
    for root in [workspace.join("src"), workspace.join("vendor")] {
        collect_source_files(&root, &mut paths)?;
    }
    paths.sort();
    let mut hasher = Sha256::new();
    for path in paths {
        let relative = path.strip_prefix(&workspace)?.to_string_lossy();
        let bytes = fs::read(&path)?;
        hasher.update((relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub(super) fn collect_source_files(root: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            collect_source_files(&path, output)?;
        } else if kind.is_file() {
            output.push(path);
        }
    }
    Ok(())
}

pub(super) fn atomic_immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        if fs::read(path)? == bytes {
            return Ok(());
        }
        bail!("immutable artifact collision at {}", path.display());
    }
    let parent = path.parent().context("artifact path has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("artifact"),
        std::process::id(),
        unix_ms()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path)?;
    OpenOptions::new().read(true).open(parent)?.sync_all()?;
    Ok(())
}

pub(super) fn atomic_replace(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("artifact path has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("artifact"),
        std::process::id(),
        unix_ms()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path)?;
    OpenOptions::new().read(true).open(parent)?.sync_all()?;
    Ok(())
}

pub(super) fn json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub(super) fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub(super) fn command_text(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    if !output.status.success() {
        bail!(
            "{program} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8(output.stdout)?)
}

pub(super) fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}
