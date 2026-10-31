//! Fail-closed root and training-artifact-registry verifiers for committed research waves (`MR03`).

use jope_records::research::wave_git::validate_index_rows;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq)]
/// `ProspectiveCohortRootsSummary` is a record for the prospective cohort roots summary contract in this module: Fail-closed root and training-artifact-registry verifiers for committed research waves (MR03).
pub struct ProspectiveCohortRootsSummary {
    /// Identifier for the associated wave.
    pub wave_id: String,
    /// SHA-256 digest identifying the associated manifest.
    pub manifest_sha256: String,
    /// SHA-256 digest identifying the associated card.
    pub card_sha256: String,
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: String,
    /// SHA-256 digest identifying the associated metrics.
    pub metrics_sha256: String,
    /// SHA-256 digest identifying the associated archive manifest.
    pub archive_manifest_sha256: String,
    /// Number of evidence root represented here.
    pub evidence_root_count: usize,
    /// Number of assertion represented here.
    pub assertion_count: usize,
    /// Scored contexts total recorded for this prospective cohort roots summary.
    pub scored_contexts_total: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// `ArtifactRegistryRootsSummary` is a record for the artifact registry roots summary contract in this module: Fail-closed root and training-artifact-registry verifiers for committed research waves (MR03).
pub struct ArtifactRegistryRootsSummary {
    /// SHA-256 digest identifying the associated registry.
    pub registry_sha256: String,
    /// SHA-256 digest identifying the associated runbook.
    pub runbook_sha256: String,
    /// Number of artifact represented here.
    pub artifact_count: usize,
    /// Declared shards recorded for this artifact registry roots summary.
    pub declared_shards: u64,
    /// Shard member keys located recorded for this artifact registry roots summary.
    pub shard_member_keys_located: u64,
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn read_bytes(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))
}

/// Verifies every row in `docs/research/waves/INDEX.jsonl` against `wave_git::validate_index_rows`
/// and checks that every indexed wave directory's 4 emitted files match their recorded SHA-256 digests.
pub fn verify_all_indexed_wave_projections(
    repo_root: &Path,
) -> Result<BTreeMap<String, Value>, String> {
    let index_path = repo_root.join("docs/research/waves/INDEX.jsonl");
    let index_bytes = read_bytes(&index_path)?;
    let rows: Vec<Vec<u8>> = index_bytes
        .split_inclusive(|&b| b == b'\n')
        .filter(|r| !r.is_empty())
        .map(|r| r.to_vec())
        .collect();
    validate_index_rows(&rows).map_err(|e| format!("INDEX.jsonl validation failed: {e}"))?;

    let mut by_wave_id = BTreeMap::new();
    for row_bytes in &rows {
        let row: Value = serde_json::from_slice(row_bytes)
            .map_err(|e| format!("invalid JSON row in INDEX.jsonl: {e}"))?;
        let wave_id = row["wave_id"]
            .as_str()
            .ok_or_else(|| "missing wave_id in INDEX.jsonl row".to_string())?;
        let wave_dir = repo_root.join("docs/research/waves").join(wave_id);
        for (field, filename) in [
            ("manifest_sha256", "wave.manifest.json"),
            ("card_sha256", "card.json"),
            ("context_sha256", "context.md"),
            ("metrics_sha256", "metrics.json"),
        ] {
            let expected = row[field]
                .as_str()
                .ok_or_else(|| format!("missing {field} for {wave_id}"))?;
            let actual = sha256_hex(&read_bytes(&wave_dir.join(filename))?);
            if actual != expected {
                return Err(format!(
                    "{wave_id}/{filename} sha256 mismatch: index={expected} actual={actual}"
                ));
            }
        }
        by_wave_id.insert(wave_id.to_string(), row);
    }
    Ok(by_wave_id)
}

/// Verifies the 3-arm prospective full-qhat CPU cohort compact roots (`1136/1136` contexts per arm)
/// under `docs/research/waves/prospective-full-qhat-cpu-20261018T1516Z`.
pub fn verify_prospective_3arm_cohort_roots(
    repo_root: &Path,
) -> Result<ProspectiveCohortRootsSummary, String> {
    let by_wave_id = verify_all_indexed_wave_projections(repo_root)?;
    let wave_id = "prospective-full-qhat-cpu-20261018T1516Z";
    let index_row = by_wave_id
        .get(wave_id)
        .ok_or_else(|| format!("{wave_id} missing from INDEX.jsonl"))?;

    if index_row["capture_status"].as_str() != Some("COMPLETE") {
        return Err(format!(
            "unexpected capture_status for {wave_id}: {:?}",
            index_row["capture_status"]
        ));
    }

    let wave_dir = repo_root.join("docs/research/waves").join(wave_id);
    let manifest_bytes = read_bytes(&wave_dir.join("wave.manifest.json"))?;
    let card_bytes = read_bytes(&wave_dir.join("card.json"))?;
    let context_bytes = read_bytes(&wave_dir.join("context.md"))?;
    let metrics_bytes = read_bytes(&wave_dir.join("metrics.json"))?;

    let manifest: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| format!("parse wave.manifest.json: {e}"))?;
    if manifest["wave_id"].as_str() != Some(wave_id) {
        return Err("wave.manifest.json wave_id mismatch".to_string());
    }
    if manifest["stage"].as_str() != Some("shadow") {
        return Err("prospective cohort wave stage must be shadow".to_string());
    }
    if manifest["disposition"].as_str() != Some("inconclusive") {
        return Err("prospective cohort wave disposition must be inconclusive".to_string());
    }
    let archive_manifest_sha256 = manifest["archive_manifest_sha256"]
        .as_str()
        .ok_or_else(|| "missing archive_manifest_sha256".to_string())?;
    if archive_manifest_sha256 != "76be65856630031714f10de9abc88183fab055baeff84a37228ceb910bd85cc7"
    {
        return Err(format!(
            "unexpected archive_manifest_sha256: {archive_manifest_sha256}"
        ));
    }
    let evidence_roots = manifest["evidence_roots"]
        .as_array()
        .ok_or_else(|| "evidence_roots must be an array".to_string())?;
    if evidence_roots.len() != 11 {
        return Err(format!(
            "expected 11 evidence_roots, found {}",
            evidence_roots.len()
        ));
    }

    let card: Value =
        serde_json::from_slice(&card_bytes).map_err(|e| format!("parse card.json: {e}"))?;
    if card["evidence_root_count"].as_u64() != Some(11) {
        return Err("card.json evidence_root_count != 11".to_string());
    }
    if card["assertion_count"].as_u64() != Some(15) {
        return Err("card.json assertion_count != 15".to_string());
    }
    if card["interpretation"].as_str() != Some("descriptive; no promotion authority") {
        return Err("card.json interpretation mismatch".to_string());
    }

    let metrics: Value =
        serde_json::from_slice(&metrics_bytes).map_err(|e| format!("parse metrics.json: {e}"))?;
    let assertions = metrics["assertions"]
        .as_array()
        .ok_or_else(|| "metrics.json assertions must be an array".to_string())?;
    if assertions.len() != 15 {
        return Err(format!(
            "expected 15 assertions, found {}",
            assertions.len()
        ));
    }

    let by_id: BTreeMap<String, &Value> = assertions
        .iter()
        .filter_map(|a| Some((a["id"].as_str()?.to_string(), a)))
        .collect();

    for (arm_id, expected_status) in [
        ("arm_a_mean_realized_gain", "complete_1136_255_xbabe1"),
        ("arm_b_mean_realized_gain", "complete_1136_255_xbabe2"),
        ("arm_d_mean_realized_gain", "complete_1136_255_xbabe3"),
    ] {
        let arm = by_id
            .get(arm_id)
            .ok_or_else(|| format!("missing assertion {arm_id}"))?;
        if arm["numerator"].as_u64() != Some(1136) || arm["denominator"].as_u64() != Some(1136) {
            return Err(format!("{arm_id} must have 1136/1136 contexts"));
        }
        if arm["status"].as_str() != Some(expected_status) {
            return Err(format!("{arm_id} status mismatch: {:?}", arm["status"]));
        }
        if arm["evidence_sha256"].as_str()
            != Some("2dd27e2b4c4c03007cae4a7421cfac484d81bd6f118b6b1d3b8743951903cbf5")
        {
            return Err(format!("{arm_id} evidence_sha256 mismatch"));
        }
        let val = arm["value"]
            .as_f64()
            .ok_or_else(|| format!("{arm_id} value not f64"))?;
        if !(val > 0.88 && val < 0.91) {
            return Err(format!("{arm_id} value {val} outside (0.88, 0.91)"));
        }
    }

    let cov = by_id
        .get("cohort_scored_context_coverage")
        .ok_or_else(|| "missing cohort_scored_context_coverage".to_string())?;
    if cov["numerator"].as_u64() != Some(3408) || cov["denominator"].as_u64() != Some(3408) {
        return Err("cohort_scored_context_coverage must be 3408/3408".to_string());
    }
    if cov["value"].as_f64() != Some(1.0) {
        return Err("cohort_scored_context_coverage value must be 1.0".to_string());
    }
    if cov["status"].as_str() != Some("three_arms_complete_zero_excluded_failed") {
        return Err("cohort_scored_context_coverage status mismatch".to_string());
    }

    Ok(ProspectiveCohortRootsSummary {
        wave_id: wave_id.to_string(),
        manifest_sha256: sha256_hex(&manifest_bytes),
        card_sha256: sha256_hex(&card_bytes),
        context_sha256: sha256_hex(&context_bytes),
        metrics_sha256: sha256_hex(&metrics_bytes),
        archive_manifest_sha256: archive_manifest_sha256.to_string(),
        evidence_root_count: evidence_roots.len(),
        assertion_count: assertions.len(),
        scored_contexts_total: 3408,
    })
}

/// Verifies `catalogs/training-artifacts.json` and `docs/runbooks/training-artifact-locations.md`
/// and cross-checks their archive SHA-256 and version bindings against the committed research waves.
pub fn verify_training_artifact_location_registry(
    repo_root: &Path,
) -> Result<ArtifactRegistryRootsSummary, String> {
    let registry_bytes = read_bytes(&repo_root.join("catalogs/training-artifacts.json"))?;
    let runbook_bytes =
        read_bytes(&repo_root.join("docs/runbooks/training-artifact-locations.md"))?;
    let runbook_text = String::from_utf8(runbook_bytes.clone())
        .map_err(|e| format!("invalid utf8 in training-artifact-locations.md: {e}"))?;

    let registry: Value = serde_json::from_slice(&registry_bytes)
        .map_err(|e| format!("parse training-artifacts.json: {e}"))?;
    if registry["schema"].as_str() != Some("jope.training-artifact-location-registry/v1") {
        return Err("unexpected training-artifacts.json schema".to_string());
    }
    if registry["git_home"].as_str() != Some("catalogs/training-artifacts.json") {
        return Err("unexpected training-artifacts.json git_home".to_string());
    }
    if registry["runbook"].as_str() != Some("docs/runbooks/training-artifact-locations.md") {
        return Err("unexpected training-artifacts.json runbook".to_string());
    }

    let artifacts = registry["artifacts"]
        .as_array()
        .ok_or_else(|| "artifacts must be an array".to_string())?;
    if artifacts.len() < 8 {
        return Err(format!(
            "expected at least 8 registered artifacts, found {}",
            artifacts.len()
        ));
    }

    let by_id: BTreeMap<String, &Value> = artifacts
        .iter()
        .filter_map(|a| Some((a["id"].as_str()?.to_string(), a)))
        .collect();

    let cohort_results = by_id
        .get("prospective-full-qhat-cpu-results")
        .ok_or_else(|| "missing prospective-full-qhat-cpu-results in registry".to_string())?;
    if cohort_results["sha256"].as_str()
        != Some("f1457513f3406f0701745a1855e2a4c9dcdb2250c03dcd405c608b18329d3199")
    {
        return Err("prospective-full-qhat-cpu-results sha256 mismatch".to_string());
    }
    if cohort_results["decoded_tar_sha256"].as_str()
        != Some("4a01417a9a460644d8c67f8f5d34cade6fb799ea8ca35d70a1e6dc87320cdf58")
    {
        return Err("prospective-full-qhat-cpu-results decoded_tar_sha256 mismatch".to_string());
    }
    if cohort_results["index_sha256"].as_str()
        != Some("76be65856630031714f10de9abc88183fab055baeff84a37228ceb910bd85cc7")
    {
        return Err("prospective-full-qhat-cpu-results index_sha256 mismatch".to_string());
    }
    if cohort_results["version_id"].as_str() != Some("TPQ20h7N_ObtMLI4vSkXBwHY1RRb6Ew.") {
        return Err("prospective-full-qhat-cpu-results version_id mismatch".to_string());
    }

    let corpus_index = by_id
        .get("historical-accepted60000-corpus-index")
        .ok_or_else(|| "missing historical-accepted60000-corpus-index in registry".to_string())?;
    let declared_shards = corpus_index["declared_shards"]
        .as_u64()
        .ok_or_else(|| "missing declared_shards".to_string())?;
    let expected_keys = corpus_index["shard_member_keys_expected"]
        .as_u64()
        .ok_or_else(|| "missing shard_member_keys_expected".to_string())?;
    let located_keys = corpus_index["shard_member_keys_located"]
        .as_u64()
        .ok_or_else(|| "missing shard_member_keys_located".to_string())?;
    if declared_shards != 7877 || expected_keys != 23631 || located_keys != 23631 {
        return Err(format!(
            "unexpected historical corpus shard counts: shards={declared_shards} expected={expected_keys} located={located_keys}"
        ));
    }

    let wave_manifest: Value =
        serde_json::from_slice(&read_bytes(&repo_root.join(
            "docs/research/waves/prospective-full-qhat-cpu-20261018T1516Z/wave.manifest.json",
        ))?)
        .map_err(|e| format!("parse wave.manifest.json: {e}"))?;
    if wave_manifest["archive_manifest_sha256"] != cohort_results["index_sha256"] {
        return Err(
            "prospective wave archive_manifest_sha256 != registry index_sha256".to_string(),
        );
    }
    let root_f145 = wave_manifest["evidence_roots"]
        .as_array()
        .ok_or_else(|| "evidence_roots not array".to_string())?
        .iter()
        .find(|r| r["sha256"] == cohort_results["sha256"])
        .ok_or_else(|| {
            "prospective-full-qhat-cpu-results blob missing in wave evidence_roots".to_string()
        })?;
    if root_f145["member_sha256"] != cohort_results["decoded_tar_sha256"]
        || root_f145["physical_version"] != cohort_results["version_id"]
    {
        return Err("prospective wave evidence_root member/version mismatch".to_string());
    }

    for required_needle in [
        "catalogs/training-artifacts.json",
        "1c840597554f76598b7c173a0de6f5d13e1b29bfa971f4fa49777d9d1da47798",
        "b7d9e669b7be6259aa5e63d7ec1a54b6341e2bc7d5667b283a9903d2c4b82175",
        "53a5bba05bc7d16e9788fc67a7060bfc36b11aadbca900f2e7d6b270617b3976",
    ] {
        if !runbook_text.contains(required_needle) {
            return Err(format!(
                "training-artifact-locations.md missing required needle {required_needle}"
            ));
        }
    }

    Ok(ArtifactRegistryRootsSummary {
        registry_sha256: sha256_hex(&registry_bytes),
        runbook_sha256: sha256_hex(&runbook_bytes),
        artifact_count: artifacts.len(),
        declared_shards,
        shard_member_keys_located: located_keys,
    })
}

/// Combined MR03 root and training-artifact-registry verifier.
pub fn verify_3arm_cohort_and_artifact_registry_roots(
    repo_root: &Path,
) -> Result<(ProspectiveCohortRootsSummary, ArtifactRegistryRootsSummary), String> {
    let cohort = verify_prospective_3arm_cohort_roots(repo_root)?;
    let registry = verify_training_artifact_location_registry(repo_root)?;
    Ok((cohort, registry))
}
