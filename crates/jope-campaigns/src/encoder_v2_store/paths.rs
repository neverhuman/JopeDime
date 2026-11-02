use super::*;

pub(crate) fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn validate_transaction_id(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty()
            && value.len() <= 128
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "transaction ID is noncanonical"
    );
    Ok(())
}

pub(crate) fn prefix(digest: &str) -> &str {
    &digest[..2]
}

pub(crate) fn object_relative_path(output_sha256: &str) -> String {
    format!("objects/sha256/{}/{}", prefix(output_sha256), output_sha256)
}

pub(crate) fn metadata_relative_path(output_sha256: &str) -> String {
    format!("metadata/{}/{}", prefix(output_sha256), output_sha256)
}

pub(crate) fn binding_relative_path(cache_key_sha256: &str) -> String {
    format!(
        "bindings/{}/{}/binding.v2.json",
        prefix(cache_key_sha256),
        cache_key_sha256
    )
}

pub(crate) fn namespace_relative_path(encoder: &str, input: &str) -> String {
    format!("namespaces/{encoder}/{input}.pointer.json")
}

pub(crate) fn ensure_parent(path: &Path) -> Result<()> {
    fs::create_dir_all(path.parent().context("publication path has no parent")?)?;
    Ok(())
}

pub(crate) fn write_new_synced(path: &Path, bytes: &[u8], mode: u32) -> Result<()> {
    ensure_parent(path)?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    sync_directory(path.parent().context("new file has no parent")?)
}

pub(crate) fn sync_file(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

pub(crate) fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(hex::encode(hash.finalize()))
}

pub(crate) fn publish_file_exclusive(staged: &Path, target: &Path, digest: &str) -> Result<()> {
    match fs::hard_link(staged, target) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            ensure!(
                sha256_file(target)? == digest,
                "existing V2 object content drift"
            );
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

fn publish_json_exclusive<T: Serialize>(staging: &Path, target: &Path, value: &T) -> Result<()> {
    ensure_parent(target)?;
    let digest = canonical_json_sha256(&serde_json::to_value(value)?)?;
    let staged = staging.join(format!("json-{digest}"));
    if !staged.exists() {
        write_new_synced(&staged, &serde_json::to_vec(value)?, 0o444)?;
    }
    fs::hard_link(&staged, target)
        .with_context(|| format!("exclusive publish {}", target.display()))?;
    sync_directory(target.parent().context("JSON target has no parent")?)
}

fn publish_json_convergent<T>(staging: &Path, target: &Path, value: &T) -> Result<()>
where
    T: Serialize + DeserializeOwned + PartialEq,
{
    if target.is_file() {
        let existing: T = read_json(target)?;
        ensure!(&existing == value, "immutable JSON publication collision");
        return Ok(());
    }
    match publish_json_exclusive(staging, target, value) {
        Ok(()) => Ok(()),
        Err(error) if target.is_file() => {
            let existing: T = read_json(target)?;
            ensure!(
                &existing == value,
                "immutable JSON publication collision: {error:#}"
            );
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_reader(std::io::BufReader::new(File::open(path)?))
        .with_context(|| format!("decode {}", path.display()))
}

pub fn relative_path_is_v2_only(path: &Path) -> bool {
    !path.is_absolute()
        && path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
        && !path.to_string_lossy().contains("expanded-v1")
}
