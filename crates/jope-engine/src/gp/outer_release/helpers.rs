use super::*;

pub const AUTHORITY_SCHEMA: &str = "parallel.outer_release_authority.v1";
pub const GRANT_SCHEMA: &str = "parallel.outer_release_grant.v1";
pub const EVENT_SCHEMA: &str = "parallel.outer_release_event.v1";
pub const RESULT_SCHEMA: &str = "parallel.outer_score.v1";
pub const RECEIPT_SCHEMA: &str = "parallel.outer_score_receipt.v1";
/// Original label role of the scored labels (the release contract's term).
pub const LABEL_ROLE: &str = "sealed_final";
/// The CLI `score` command finds its release authority here; nothing else does.
pub const AUTHORITY_ENV: &str = "JOPE_OUTER_RELEASE_AUTHORITY";
pub(crate) const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

pub(crate) fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A path under `root` as its root-relative form, refusing any symlink
/// component (including the leaf) and anything outside the root.
pub(crate) fn rooted(root: &Path, path: &Path) -> Result<(PathBuf, PathBuf)> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let relative = absolute
        .strip_prefix(root)
        .context("sealed label file outside the granted root")?
        .to_path_buf();
    Ok((resolve(root, &relative)?, relative))
}

/// Read `path` once and record it as `relative_artifact` does (its path under
/// the canonical `root` and the digest of exactly the bytes returned).
pub(crate) fn read_artifact(root: &Path, path: &Path) -> Result<(Value, Vec<u8>)> {
    let absolute = path.canonicalize()?;
    let relative = absolute
        .strip_prefix(root)
        .context("original fit artifact outside workspace")?
        .to_path_buf();
    let bytes = fs::read(&absolute)?;
    Ok((json!({"path":relative,"sha256":hash(&bytes)}), bytes))
}
