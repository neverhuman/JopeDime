use super::*;

pub(super) fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub(super) fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub(super) fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        bail!("{label} is not a lowercase SHA-256");
    }
    Ok(())
}

pub(super) fn validate_hashes<'a>(
    values: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<()> {
    for (label, value) in values {
        validate_sha256(label, value)?;
    }
    Ok(())
}

pub(super) fn require_unique<T: Ord>(
    label: &str,
    values: impl IntoIterator<Item = T>,
) -> Result<()> {
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value) {
            bail!("{label} contains a duplicate");
        }
    }
    Ok(())
}

pub(super) fn validate_track_history(
    label: &str,
    entries: &[TrackHistoryEntryV2],
    required_authority: SelectionAuthorityV2,
) -> Result<()> {
    let mut previous_time = 0;
    for (index, entry) in entries.iter().enumerate() {
        if entry.sequence != index as u64 + 1 || entry.authority != required_authority {
            bail!("{label} history has invalid sequence or authority");
        }
        entry.candidate.validate()?;
        validate_sha256(&format!("{label} receipt"), &entry.receipt_sha256)?;
        if !entry.robust_fitness.is_finite() || entry.committed_at_unix_seconds < previous_time {
            bail!("{label} history contains invalid fitness or time");
        }
        previous_time = entry.committed_at_unix_seconds;
    }
    Ok(())
}
