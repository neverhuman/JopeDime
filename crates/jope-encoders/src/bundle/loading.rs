use super::*;

mod loaded;
mod manifest;

pub(super) fn verify_notices(root: &File, manifest: &ModelBundleV1) -> Result<()> {
    for (relative, expected) in &manifest.notices {
        let bytes = read_regular_bytes(root, relative, MAX_NOTICE_BYTES)?;
        if sha256_bytes(&bytes) != *expected {
            bail!("model notice sha256 mismatch: {relative}");
        }
    }
    Ok(())
}

pub(super) fn open_bundle_root(root: &Path) -> Result<(PathBuf, File)> {
    let descriptor = open(
        root,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .with_context(|| {
        format!(
            "open model bundle root {} as a real directory without following symlinks",
            root.display()
        )
    })?;
    let directory = File::from(descriptor);
    let canonical = std::fs::canonicalize(root)
        .with_context(|| format!("canonicalize model bundle root {}", root.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let opened = directory.metadata()?;
        let named = std::fs::metadata(&canonical)?;
        if opened.dev() != named.dev() || opened.ino() != named.ino() {
            bail!("model bundle root changed while it was opened");
        }
    }
    Ok((canonical, directory))
}

pub(super) fn read_manifest(root: &File) -> Result<Vec<u8>> {
    for relative in ["model-bundle.v1.json", "artifacts/model-bundle.v1.json"] {
        if let Some(bytes) = read_regular_bytes_optional(root, relative, MAX_MANIFEST_BYTES)? {
            return Ok(bytes);
        }
    }
    bail!("model bundle manifest is missing")
}

pub(super) fn read_regular_bytes(root: &File, relative: &str, max_bytes: u64) -> Result<Vec<u8>> {
    read_regular_bytes_optional(root, relative, max_bytes)?
        .with_context(|| format!("model bundle path missing: {relative}"))
}

pub(super) fn read_regular_bytes_optional(
    root: &File,
    relative: &str,
    max_bytes: u64,
) -> Result<Option<Vec<u8>>> {
    validate_relative_path(relative)?;
    let components = Path::new(relative).components().collect::<Vec<_>>();
    let mut parent = None::<File>;
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(component) = component else {
            bail!("invalid model bundle path {relative:?}");
        };
        let final_component = index + 1 == components.len();
        let flags = OFlags::RDONLY
            | OFlags::NOFOLLOW
            | OFlags::CLOEXEC
            | if final_component {
                OFlags::NONBLOCK
            } else {
                OFlags::DIRECTORY
            };
        let directory = parent.as_ref().unwrap_or(root);
        let descriptor = match openat(directory.as_fd(), *component, flags, Mode::empty()) {
            Ok(descriptor) => descriptor,
            Err(Errno::NOENT) => return Ok(None),
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "open model bundle path {relative} without following component {:?}",
                        component
                    )
                })
            }
        };
        let opened = File::from(descriptor);
        if !final_component {
            parent = Some(opened);
            continue;
        }
        let metadata = opened.metadata()?;
        if !metadata.is_file() {
            bail!("model bundle path is not a regular file: {relative}");
        }
        if metadata.len() > max_bytes {
            bail!(
                "model bundle path {relative} exceeds its {} byte read bound",
                max_bytes
            );
        }
        let capacity = usize::try_from(metadata.len()).context("bundle file size exceeds usize")?;
        let mut bytes = Vec::with_capacity(capacity);
        opened
            .take(max_bytes.saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > max_bytes {
            bail!(
                "model bundle path {relative} grew beyond its {} byte read bound",
                max_bytes
            );
        }
        return Ok(Some(bytes));
    }
    if components.is_empty() {
        bail!("model bundle path is not a regular file: {relative}");
    }
    unreachable!("validated model bundle path has a final component")
}
