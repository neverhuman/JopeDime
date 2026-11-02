use super::*;

/// Pure preflight of a frozen `parallel.label_admission.v2` policy. It inspects only the
/// policy fields, before any evaluator, witness or authority file is opened.
pub fn preflight_policy(expected: &Value) -> Result<()> {
    ensure!(
        expected["schema"] == POLICY_SCHEMA,
        "unknown admission policy schema"
    );
    ensure!(
        expected["evaluator_protocol"] == PROTOCOL,
        "admission evaluator protocol differs"
    );
    for key in [
        "authority_instance",
        "runtime_sha256",
        "numerical_contract_sha256",
        "evaluator_sha256",
    ] {
        ensure!(
            is_sha256(&expected[key]),
            "admission identity malformed: {key}"
        );
    }
    for key in ["evaluator", "authority_root", "authority_witness"] {
        ensure!(
            expected[key].as_str().is_some_and(|s| !s.trim().is_empty()),
            "admission path absent: {key}"
        );
    }
    if let Some(binding) = expected.get("native_fit_policy").filter(|v| !v.is_null()) {
        ensure!(
            binding["path"]
                .as_str()
                .is_some_and(|s| !s.trim().is_empty()),
            "native fit policy path absent"
        );
        ensure!(
            is_sha256(&binding["sha256"]),
            "native fit policy sha256 malformed"
        );
    }
    Ok(())
}

/// Key and length closure of the historical shard census. It deliberately does not claim
/// payload hashes: a census with `all_7877_payload_hashes_verified: false` stays unverified.
pub fn census_closure(census: &Value) -> Result<()> {
    ensure!(
        census["directories_declared"] == 7877,
        "shard directory count differs"
    );
    ensure!(
        census["expected_member_keys"] == 23631,
        "member key count differs"
    );
    ensure!(
        census["located_member_keys"] == census["expected_member_keys"],
        "member keys are missing"
    );
    ensure!(
        census["missing_member_keys"] == 0,
        "missing member keys present"
    );
    Ok(())
}

/// Byte-level check of one historical shard directory. Each member's SHA-256 must equal its
/// expected digest. A missing member or any mismatch refuses. This verifies payload bytes
/// only; it infers no label or training semantics.
pub fn verify_shard_directory_payload_hashes(
    shard_dir: &Path,
    expected_shard_sha256: &str,
    expected_records_sha256: &str,
    expected_labels_sha256: &str,
) -> Result<()> {
    for (member, expected) in [
        ("shard.json", expected_shard_sha256),
        ("records.bin", expected_records_sha256),
        ("labels.jsonl", expected_labels_sha256),
    ] {
        let actual = file_hash(&shard_dir.join(member))
            .with_context(|| format!("shard member absent or unreadable: {member}"))?;
        ensure!(actual == expected, "shard member payload changed: {member}");
    }
    Ok(())
}

/// The census sample payload checks: each sampled member matched its index digest and its
/// metadata matched the parent shard.
pub fn verify_census_payload_samples(census: &Value) -> Result<()> {
    let samples = census["sample_exact_version_payload_verification"]
        .as_array()
        .context("census payload samples absent")?;
    ensure!(!samples.is_empty(), "census payload samples empty");
    for sample in samples {
        ensure!(
            sample["hash_matches"] == true,
            "census sample hash mismatch"
        );
        ensure!(
            sample["manifest_metadata_matches_parent"] == true,
            "census sample metadata mismatch"
        );
    }
    Ok(())
}

/// Promotion gate: refuse unless the census states that every 7,877 shard payload was verified.
pub fn require_full_payload_closure(census: &Value) -> Result<()> {
    ensure!(
        census["all_7877_payload_hashes_verified"] == true,
        "full 7877 payload closure not verified"
    );
    Ok(())
}
