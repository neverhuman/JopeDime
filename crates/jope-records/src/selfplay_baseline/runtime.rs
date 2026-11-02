use super::*;

/// Acceptance 3 validator: verify all skipped and unavailable runtime combinations (`RT01`–`RT07`)
/// dynamically against workspace manifests, GPU hold routes, `docs/parity/test-inventory.json`,
/// and `deploy/ci/coverage.json`.
pub fn verify_runtime_combinations(repo: &Path, runtime_val: &Value) -> Result<usize> {
    let mut snapshot = BaselineSnapshot::new(repo, RECEIVING_MAIN_HEAD_SHA)?;
    let i01_val = read_json(&repo.join(FILE_MAP_I01_PATH))?;
    let i01 = dict_of(&i01_val, "FILE_MAP.i01.json")?;
    verify_runtime_combinations_with_snapshot(&mut snapshot, i01, runtime_val)
}

pub(super) fn verify_runtime_combinations_with_snapshot(
    snapshot: &mut BaselineSnapshot<'_>,
    file_map: &Dict,
    runtime_val: &Value,
) -> Result<usize> {
    let entries = list_of(runtime_val, "skipped_and_unavailable_runtimes")?;
    let required_ids = ["RT01", "RT02", "RT03", "RT04", "RT05", "RT06", "RT07"];
    let entry_count = entries.len();
    let required_count = required_ids.len();
    require!(
        entry_count >= required_count,
        "skipped_and_unavailable_runtimes must record all {required_count} combinations (RT01..RT07), got {entry_count}"
    );

    // Derive expected sets dynamically from file_map and repository inventories.
    let members = req_list(file_map, "workspace_packages", "FILE_MAP.i01.json")?;
    let mut expected_whole_neural = Vec::new();
    let mut expected_partial_neural = Vec::new();
    let mut expected_testing_gated = Vec::new();
    let mut expected_req_feature_bins = Vec::new();

    for m_val in members {
        let m = dict_of(m_val, "workspace_packages")?;
        let pkg = req_str(m, "package_name", "workspace_packages")?;
        for t in req_list(m, "neural_gated_integration_tests", "workspace_packages")? {
            if let Some(s) = t.as_str() {
                expected_whole_neural.push(s.to_owned());
            }
        }
        for t in req_list(
            m,
            "partially_neural_gated_integration_tests",
            "workspace_packages",
        )? {
            if let Some(s) = t.as_str() {
                expected_partial_neural.push(s.to_owned());
            }
        }
        if pkg == "jope-policy" {
            for t in req_list(m, "testing_gated_integration_tests", "workspace_packages")? {
                if let Some(s) = t.as_str() {
                    expected_testing_gated.push(s.to_owned());
                }
            }
        }
        let req_targets = req_dict(m, "required_feature_targets", "workspace_packages")?;
        for k in req_targets.keys() {
            if let Some(bin_name) = k.strip_prefix("bin:") {
                expected_req_feature_bins.push(format!("{pkg}::bin:{bin_name}"));
            }
        }
    }

    let symbols = req_list(file_map, "symbols", "FILE_MAP.i01.json")?;
    let gpu_sym = symbols
        .iter()
        .filter_map(|v| v.as_dict())
        .find(|d| d.get("symbol").and_then(Value::as_str) == Some("require_original_gpu_grant"))
        .ok_or_else(|| Error::Invalid("symbols missing `require_original_gpu_grant`".into()))?;
    let expected_direct_gpu_routes: Vec<String> =
        req_list(gpu_sym, "direct_routes", "require_original_gpu_grant")?
            .iter()
            .filter_map(|v| v.as_dict()?.get("route")?.as_str().map(str::to_owned))
            .collect();
    let expected_indirect_gpu_routes: Vec<String> = req_list(
        gpu_sym,
        "indirect_routes_via_require_qualified_gpu_experiment",
        "require_original_gpu_grant",
    )?
    .iter()
    .filter_map(|v| v.as_dict()?.get("route")?.as_str().map(str::to_owned))
    .collect();

    let (expected_total_ignored, expected_ignored_binaries, expected_non_default_features) =
        if snapshot.has_git_commit() {
            let test_inv_text = snapshot.blob_utf8("docs/parity/test-inventory.json")?;
            let test_inv_val = pyjson::parse_str(&test_inv_text)
                .map_err(|e| Error::Invalid(format!("docs/parity/test-inventory.json: {e}")))?;
            let test_inv = dict_of(&test_inv_val, "docs/parity/test-inventory.json")?;
            let rust_inv = req_dict(test_inv, "rust", "docs/parity/test-inventory.json")?;
            let packages = req_dict(rust_inv, "packages", "test-inventory.rust")?;
            let mut total_ign = 0u64;
            let mut ign_bins = 0usize;
            for (_pkg_key, p_val) in packages.iter() {
                let p_dict = dict_of(p_val, "test-inventory.rust.packages")?;
                let binaries = req_dict(p_dict, "test-inventory.rust.packages", "binaries")
                    .or_else(|_| req_dict(p_dict, "binaries", "test-inventory.rust.packages"))?;
                for (_bin_key, b_val) in binaries.iter() {
                    let b_dict = dict_of(b_val, "test-inventory.binaries")?;
                    let ign = req_u64(b_dict, "ignored", "test-inventory.binaries")?;
                    if ign > 0 {
                        total_ign += ign;
                        ign_bins += 1;
                    }
                }
            }

            let cov_text = snapshot.blob_utf8("deploy/ci/coverage.json")?;
            let cov_val = pyjson::parse_str(&cov_text)
                .map_err(|e| Error::Invalid(format!("deploy/ci/coverage.json: {e}")))?;
            let cov = dict_of(&cov_val, "deploy/ci/coverage.json")?;
            let cov_items = req_dict(cov, "items", "deploy/ci/coverage.json")?;
            let nd_feats: Vec<String> = cov_items
                .keys()
                .filter_map(|k| k.strip_prefix("feature:"))
                .filter(|rest| !rest.ends_with("/default"))
                .map(str::to_owned)
                .collect();
            (Some(total_ign), Some(ign_bins), Some(nd_feats))
        } else {
            (None, None, None)
        };

    let mut seen_ids = BTreeSet::new();
    for (idx, entry_val) in entries.iter().enumerate() {
        let rctx = format!("skipped_and_unavailable_runtimes[{idx}]");
        let entry = dict_of(entry_val, &rctx)?;
        let rid = req_str(entry, "id", &rctx)?;
        require!(
            seen_ids.insert(rid.to_owned()),
            "{rctx}: duplicate runtime combination id `{rid}`"
        );
        let status = req_str(entry, "status", &rctx)?;
        require!(
            matches!(
                status,
                "SKIPPED_ZERO_TESTS_WITHOUT_FEATURE"
                    | "GATED_BY_TESTING_FEATURE"
                    | "HELD_ORIGINAL_GPU_GRANT_UNAVAILABLE"
                    | "IGNORED_BY_DEFAULT_TEST_HARNESS"
                    | "FEATURE_GATED_NON_DEFAULT_COMBINATIONS"
                    | "SKIPPED_BY_PATH_NOT_EXECUTED_COVERAGE"
                    | "BLOCKED_PROVIDER_CREDITS_HTTP_402"
            ),
            "{rctx} (`{rid}`): unrecognized runtime status `{status}` (passed/active statuses are forbidden here)"
        );
        let required_flag = req_str(entry, "required_flag", &rctx)?;
        require!(
            !required_flag.trim().is_empty(),
            "{rctx} (`{rid}`): required_flag must not be empty"
        );
        let targets = req_list(entry, "affected_targets", &rctx)?;
        require!(
            !targets.is_empty(),
            "{rctx} (`{rid}`): affected_targets must not be empty"
        );
        let source_path = req_str(entry, "source_path", &rctx)?;
        validate_repo_relative_path(source_path)?;
        if snapshot.has_git_commit() {
            require!(
                snapshot.file_exists(source_path),
                "{rctx} (`{rid}`): source_path `{source_path}` does not exist in baseline commit"
            );
        }
        let source_sha = req_str(entry, "source_sha", &rctx)?;
        require!(
            is_lower_hex_sha40(source_sha),
            "{rctx} (`{rid}`): source_sha must be 40-hex, got `{source_sha}`"
        );

        match rid {
            "RT01" => {
                let whole = req_list(entry, "whole_file_gated_targets", &rctx)?;
                let partial = req_list(entry, "partially_gated_targets", &rctx)?;
                let whole_len = whole.len();
                let exp_whole_len = expected_whole_neural.len();
                require!(
                    whole_len == exp_whole_len,
                    "RT01 whole_file_gated_targets count ({whole_len}) must match workspace neural_gated_integration_tests count ({exp_whole_len})"
                );
                let partial_len = partial.len();
                let exp_partial_len = expected_partial_neural.len();
                require!(
                    partial_len == exp_partial_len,
                    "RT01 partially_gated_targets count ({partial_len}) must match workspace partially_neural_gated_integration_tests count ({exp_partial_len})"
                );
                for p_val in partial {
                    let p_str = p_val.as_str().ok_or_else(|| {
                        Error::Invalid("RT01 partially_gated_targets entry must be string".into())
                    })?;
                    let (rel_path, line_str) = p_str.split_once(':').ok_or_else(|| {
                        Error::Invalid(format!(
                            "RT01 partially_gated_targets entry `{p_str}` must be `path:line`"
                        ))
                    })?;
                    let line_no: usize = line_str.parse().map_err(|e| {
                        Error::Invalid(format!("RT01 invalid line number in `{p_str}`: {e}"))
                    })?;
                    if snapshot.has_git_commit() {
                        let src = snapshot.blob_utf8(rel_path)?;
                        let lines: Vec<&str> = src.lines().collect();
                        let line_text = lines.get(line_no.saturating_sub(1)).copied().unwrap_or("");
                        require!(
                            line_no >= 1
                                && line_no <= lines.len()
                                && (line_text.contains("#[cfg(feature = \"neural\")]")
                                    || line_text.contains("#[cfg(not(feature = \"neural\"))]")),
                            "RT01 `{p_str}` does not contain `#[cfg(feature = \"neural\")]` or `#[cfg(not(feature = \"neural\"))]`"
                        );
                    }
                }
                let expected_total = exp_whole_len + exp_partial_len;
                let target_len = targets.len();
                require!(
                    target_len == expected_total,
                    "RT01 affected_targets must list all {expected_total} neural-gated integration test targets in jope-parallel ({exp_whole_len} whole-file + {exp_partial_len} partially gated), got {target_len}"
                );
            }
            "RT02" => {
                let target_len = targets.len();
                let exp_test_len = expected_testing_gated.len();
                require!(
                    target_len == exp_test_len,
                    "RT02 affected_targets count ({target_len}) must match jope-policy testing_gated_integration_tests count ({exp_test_len})"
                );
                for t_name in &expected_testing_gated {
                    let present = targets
                        .iter()
                        .filter_map(Value::as_str)
                        .any(|s| s.starts_with(t_name));
                    require!(
                        present,
                        "RT02 affected_targets missing testing-gated target `{t_name}`"
                    );
                }
                let note = req_str(entry, "default_lane_note", &rctx)?;
                require!(
                    note.contains("challenger_tests.rs") && note.contains("rust-test"),
                    "RT02 default_lane_note must document that challenger_tests.rs executes in default rust-test"
                );
            }
            "RT03" => {
                let direct = req_list(entry, "direct_routes", &rctx)?;
                let indirect = req_list(
                    entry,
                    "indirect_routes_via_require_qualified_gpu_experiment",
                    &rctx,
                )?;
                let direct_len = direct.len();
                let exp_direct_len = expected_direct_gpu_routes.len();
                require!(
                    direct_len == exp_direct_len,
                    "RT03 direct_routes count ({direct_len}) must match symbol inventory direct_routes count ({exp_direct_len})"
                );
                let indirect_len = indirect.len();
                let exp_indirect_len = expected_indirect_gpu_routes.len();
                require!(
                    indirect_len == exp_indirect_len,
                    "RT03 indirect_routes_via_require_qualified_gpu_experiment count ({indirect_len}) must match symbol inventory indirect count ({exp_indirect_len})"
                );
                let expected_total_routes = exp_direct_len + exp_indirect_len;
                let target_len = targets.len();
                require!(
                    target_len == expected_total_routes,
                    "RT03 affected_targets must list all {expected_total_routes} held GPU routes ({exp_direct_len} direct + {exp_indirect_len} via require_qualified_gpu_experiment), got {target_len}"
                );
            }
            "RT04" => {
                let total_ign = req_u64(entry, "total_ignored_count", &rctx)?;
                if let Some(exp_ign) = expected_total_ignored {
                    require!(
                        total_ign == exp_ign,
                        "RT04 total_ignored_count ({total_ign}) must match docs/parity/test-inventory.json sum ({exp_ign})"
                    );
                }
                if let Some(exp_bins) = expected_ignored_binaries {
                    let target_len = targets.len();
                    require!(
                        target_len == exp_bins,
                        "RT04 affected_targets count ({target_len}) must match test-inventory binaries with ignored > 0 ({exp_bins})"
                    );
                }
            }
            "RT05" => {
                let comb = req_str(entry, "combination", &rctx)?;
                require!(
                    comb.contains("hyperion/batch-worker")
                        && comb.contains("hyperion/hyperion-cpu")
                        && comb.contains("jope-cli/qualified-training-context")
                        && !comb.contains("jope-bench/asm"),
                    "RT05 combination must name the 3 crate-default-enabled features (hyperion/batch-worker, hyperion/hyperion-cpu, jope-cli/qualified-training-context)"
                );
                let nd_feats = req_list(entry, "non_default_features", &rctx)?;
                let nd_len = nd_feats.len();
                let target_len = targets.len();
                require!(
                    nd_len == target_len,
                    "RT05 non_default_features ({nd_len}) and affected_targets ({target_len}) must agree"
                );
                if let Some(ref exp_nd) = expected_non_default_features {
                    let exp_nd_len = exp_nd.len();
                    require!(
                        nd_len == exp_nd_len,
                        "RT05 non_default_features ({nd_len}) must match deploy/ci/coverage.json features count ({exp_nd_len})"
                    );
                }
                let req_bins = req_list(entry, "required_feature_binaries", &rctx)?;
                let req_bins_len = req_bins.len();
                let exp_req_bins_len = expected_req_feature_bins.len();
                require!(
                    req_bins_len == exp_req_bins_len,
                    "RT05 required_feature_binaries count ({req_bins_len}) must match workspace required_feature_targets bin count ({exp_req_bins_len})"
                );
                for exp_bin in &expected_req_feature_bins {
                    let found = req_bins
                        .iter()
                        .filter_map(Value::as_str)
                        .any(|s| s.starts_with(exp_bin));
                    require!(found, "RT05 required_feature_binaries missing `{exp_bin}`");
                }
            }
            _ => {}
        }
    }

    for expected_id in required_ids {
        require!(
            seen_ids.contains(expected_id),
            "skipped_and_unavailable_runtimes missing required entry `{expected_id}`"
        );
    }

    Ok(entry_count)
}
