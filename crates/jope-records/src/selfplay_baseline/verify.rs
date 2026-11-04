use super::workspace::{verify_baseline_md_consistency, verify_file_inventories_and_pins};
use super::*;

/// Verify `BASELINE.md` and `FILE_MAP.i01.json` in `repo`.
pub fn verify_selfplay_baseline(repo: &Path) -> Result<SelfplayBaselineSummary> {
    verify_immutable_packet_v4(repo)?;

    let file_map_v1_bytes = fs::read(repo.join(FILE_MAP_V1_PATH))
        .map_err(|e| Error::io(&repo.join(FILE_MAP_V1_PATH), e))?;
    let file_map_v1_sha = sha256_hex(&file_map_v1_bytes);
    require!(
        file_map_v1_sha == EXPECTED_FILE_MAP_V1_SHA256,
        "immutable FILE_MAP.json SHA-256 mismatch: expected {EXPECTED_FILE_MAP_V1_SHA256}, got {file_map_v1_sha}"
    );

    let baseline_md_path = repo.join(BASELINE_MD_PATH);
    let baseline_md = read_utf8(&baseline_md_path)?;

    let i01_path = repo.join(FILE_MAP_I01_PATH);
    let i01_val = read_json(&i01_path)?;
    let i01 = dict_of(&i01_val, "FILE_MAP.i01.json")?;

    let schema = req_str(i01, "schema", "FILE_MAP.i01.json")?;
    require!(
        schema == "jope.unified-workplan.file-map/v2",
        "FILE_MAP.i01.json schema must be `jope.unified-workplan.file-map/v2`, got `{schema}`"
    );
    let task_id = req_str(i01, "task_id", "FILE_MAP.i01.json")?;
    require!(
        task_id == "I01",
        "FILE_MAP.i01.json task_id must be `I01`, got `{task_id}`"
    );
    let source_sha = req_str(i01, "source_sha", "FILE_MAP.i01.json")?;
    let source_tree = req_str(i01, "source_tree", "FILE_MAP.i01.json")?;
    let parent_sha = req_str(i01, "parent_sha", "FILE_MAP.i01.json")?;
    require!(
        source_sha == RECEIVING_MAIN_HEAD_SHA
            && source_tree == RECEIVING_MAIN_TREE_SHA
            && parent_sha == RECEIVING_MAIN_PARENT_SHA,
        "FILE_MAP.i01.json receiving main commit/tree/parent pin mismatch"
    );

    let mut snapshot = BaselineSnapshot::new(repo, source_sha)?;
    snapshot.verify_commit_object(
        source_sha,
        source_tree,
        parent_sha,
        "FILE_MAP.i01.json root pin",
    )?;

    let packet = req_dict(i01, "immutable_v4_packet", "FILE_MAP.i01.json")?;
    let sup_sha = req_str(packet, "previous_file_map_sha256", "immutable_v4_packet")?;
    let sup_manifest_sha = req_str(packet, "manifest_sha256", "immutable_v4_packet")?;
    let modified = req_bool(
        packet,
        "immutable_payload_files_modified",
        "immutable_v4_packet",
    )?;
    require!(
        sup_sha == EXPECTED_FILE_MAP_V1_SHA256,
        "immutable_v4_packet.previous_file_map_sha256 mismatch: got `{sup_sha}`"
    );
    require!(
        sup_manifest_sha == EXPECTED_PACKET_MANIFEST_V4_SHA256,
        "immutable_v4_packet.manifest_sha256 mismatch: got `{sup_manifest_sha}`"
    );
    require!(
        !modified,
        "immutable_v4_packet.immutable_payload_files_modified must be false"
    );

    verify_baseline_md_consistency(&baseline_md, i01)?;
    verify_file_inventories_and_pins(&mut snapshot, i01)?;

    let rec_val = i01.get("profile_and_model_reconciliation").ok_or_else(|| {
        Error::Invalid("FILE_MAP.i01.json missing `profile_and_model_reconciliation`".to_owned())
    })?;
    verify_profile_identity_reconciliation_with_snapshot(&mut snapshot, rec_val)?;

    let sym_val = i01
        .get("symbols")
        .ok_or_else(|| Error::Invalid("FILE_MAP.i01.json missing `symbols`".to_owned()))?;
    let (symbol_count, held_gpu_route_count) =
        verify_symbol_inventory_with_snapshot(&mut snapshot, sym_val)?;

    let (
        workspace_crate_count,
        neural_gated_test_count,
        partially_neural_gated_test_count,
        schema_file_count,
    ) = verify_workspace_and_schemas_with_snapshot(&mut snapshot, i01)?;

    let findings_val = i01
        .get("source_findings")
        .ok_or_else(|| Error::Invalid("FILE_MAP.i01.json missing `source_findings`".to_owned()))?;
    let (confirmed_finding_count, rejected_finding_count, unknown_finding_count) =
        verify_finding_classifications_with_snapshot(&mut snapshot, findings_val)?;

    let runtime_val = i01.get("skipped_and_unavailable_runtimes").ok_or_else(|| {
        Error::Invalid("FILE_MAP.i01.json missing `skipped_and_unavailable_runtimes`".to_owned())
    })?;
    let runtime_combination_count =
        verify_runtime_combinations_with_snapshot(&mut snapshot, i01, runtime_val)?;

    Ok(SelfplayBaselineSummary {
        schema: schema.to_owned(),
        task_id: task_id.to_owned(),
        workspace_crate_count,
        neural_gated_test_count,
        partially_neural_gated_test_count,
        schema_file_count,
        symbol_count,
        held_gpu_route_count,
        confirmed_finding_count,
        rejected_finding_count,
        unknown_finding_count,
        runtime_combination_count,
        git_object_store_verified: snapshot.has_git_commit(),
        git_object_store_status: snapshot.status_label().to_owned(),
    })
}
