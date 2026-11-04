use super::*;

/// Verify that the immutable v4 packet (`PACKET_MANIFEST.v4.json` and all 39 payload files)
/// remains intact and unmodified in `repo`.
pub fn verify_immutable_packet_v4(repo: &Path) -> Result<()> {
    let manifest_path = repo.join(PACKET_MANIFEST_V4_PATH);
    let manifest_bytes = fs::read(&manifest_path).map_err(|e| Error::io(&manifest_path, e))?;
    let actual_manifest_sha = sha256_hex(&manifest_bytes);
    require!(
        actual_manifest_sha == EXPECTED_PACKET_MANIFEST_V4_SHA256,
        "PACKET_MANIFEST.v4.json SHA-256 mismatch: expected {EXPECTED_PACKET_MANIFEST_V4_SHA256}, got {actual_manifest_sha}"
    );
    let manifest_text = String::from_utf8(manifest_bytes)
        .map_err(|e| Error::parse(&manifest_path, e.to_string()))?;
    let manifest_val =
        pyjson::parse_str(&manifest_text).map_err(|e| Error::parse(&manifest_path, e))?;
    let manifest = dict_of(&manifest_val, "PACKET_MANIFEST.v4.json")?;
    let files = req_list(manifest, "files", "PACKET_MANIFEST.v4.json")?;
    let file_count = files.len();
    require!(
        file_count == V4_PAYLOAD_COUNT,
        "PACKET_MANIFEST.v4.json must list {V4_PAYLOAD_COUNT} payload files, found {file_count}"
    );
    let plan_dir = repo.join(PLAN_DIR);
    for (idx, entry_val) in files.iter().enumerate() {
        let ctx = format!("PACKET_MANIFEST.v4.json files[{idx}]");
        let entry = dict_of(entry_val, &ctx)?;
        let rel = req_str(entry, "path", &ctx)?;
        validate_repo_relative_path(rel)?;
        let expected_sha = req_str(entry, "sha256", &ctx)?;
        let expected_bytes = req_u64(entry, "bytes", &ctx)?;
        let target = plan_dir.join(rel);
        let data = fs::read(&target).map_err(|e| Error::io(&target, e))?;
        require!(
            u64::try_from(data.len()).ok() == Some(expected_bytes),
            "immutable packet payload `{rel}` byte length mismatch: expected {expected_bytes}, got {}",
            data.len()
        );
        let actual_sha = sha256_hex(&data);
        require!(
            actual_sha == expected_sha,
            "immutable packet payload `{rel}` SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        );
    }
    Ok(())
}
