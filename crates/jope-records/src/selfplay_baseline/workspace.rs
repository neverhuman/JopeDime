use super::*;

/// Verify the Cargo workspace crates, package-name mismatches, `jope-parallel` neural-gated
/// integration tests (whole-file and partial), `jope-policy` testing-gated integration tests, and
/// all 29 committed `*.schema.json` files recorded in `FILE_MAP.i01.json`.
pub fn verify_workspace_and_schemas(repo: &Path, file_map: &Dict) -> Result<(usize, usize, usize)> {
    let mut snapshot = BaselineSnapshot::new(repo, RECEIVING_MAIN_HEAD_SHA)?;
    let (members, whole_neural, _partial_neural, schemas) =
        verify_workspace_and_schemas_with_snapshot(&mut snapshot, file_map)?;
    Ok((members, whole_neural, schemas))
}

pub(super) fn verify_workspace_and_schemas_with_snapshot(
    snapshot: &mut BaselineSnapshot<'_>,
    file_map: &Dict,
) -> Result<(usize, usize, usize, usize)> {
    let members = req_list(file_map, "workspace_packages", "FILE_MAP.i01.json")?;
    let member_count = members.len();
    if snapshot.has_git_commit() {
        let root_toml_text = snapshot.blob_utf8("Cargo.toml")?;
        let root_toml: toml::Value = toml::from_str(&root_toml_text)
            .map_err(|e| Error::Invalid(format!("Cargo.toml parse error: {e}")))?;
        let root_members = root_toml
            .get("workspace")
            .and_then(|w| w.get("members"))
            .and_then(toml::Value::as_array)
            .ok_or_else(|| Error::Invalid("Cargo.toml missing [workspace].members".to_owned()))?;
        let root_member_count = root_members.len();
        require!(
            root_member_count == member_count,
            "Cargo.toml [workspace].members count ({root_member_count}) != FILE_MAP.i01.json workspace_packages count ({member_count})"
        );
    }

    let mut total_neural_gated = 0usize;
    let mut total_partial_neural_gated = 0usize;

    for (idx, m_val) in members.iter().enumerate() {
        let mctx = format!("workspace_packages[{idx}]");
        let m = dict_of(m_val, &mctx)?;
        let member_dir = req_str(m, "member_dir", &mctx)?;
        validate_repo_relative_path(member_dir)?;
        let manifest_rel = req_str(m, "cargo_toml_path", &mctx)?;
        validate_repo_relative_path(manifest_rel)?;
        let pkg_name = req_str(m, "package_name", &mctx)?;
        let expected_sha = req_str(m, "cargo_toml_sha256", &mctx)?;
        require!(
            is_lower_hex_sha64(expected_sha),
            "{mctx}: cargo_toml_sha256 must be 64-hex"
        );
        if snapshot.has_git_commit() {
            let manifest_bytes = snapshot.blob_bytes(manifest_rel)?;
            require!(
                sha256_hex(&manifest_bytes) == expected_sha,
                "{mctx}: cargo_toml_sha256 mismatch for `{manifest_rel}`"
            );
            let manifest_text = String::from_utf8(manifest_bytes)
                .map_err(|e| Error::Invalid(format!("{mctx}: non-UTF-8 `{manifest_rel}`: {e}")))?;
            let parsed: toml::Value = toml::from_str(&manifest_text)
                .map_err(|e| Error::Invalid(format!("{manifest_rel}: {e}")))?;
            let actual_pkg = parsed
                .get("package")
                .and_then(|p| p.get("name"))
                .and_then(toml::Value::as_str)
                .ok_or_else(|| Error::Invalid(format!("{manifest_rel}: missing [package].name")))?;
            require!(
                actual_pkg == pkg_name,
                "{mctx}: package_name mismatch for `{member_dir}`: expected `{actual_pkg}`, got `{pkg_name}`"
            );

            // Verify integration_tests includes both tests/<name>.rs and tests/<name>/main.rs.
            let recorded_tests: Vec<String> = req_list(m, "integration_tests", &mctx)?
                .iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
            let tests_prefix = format!("{member_dir}/tests/");
            let all_test_rs = snapshot.matching_paths(&tests_prefix, ".rs");
            let mut actual_targets = BTreeSet::new();
            for p in &all_test_rs {
                if let Some(rest) = p.strip_prefix(&tests_prefix) {
                    if !rest.contains('/') {
                        if let Some(stem) = rest.strip_suffix(".rs") {
                            actual_targets.insert(stem.to_owned());
                        }
                    } else if let Some(dir_name) = rest.strip_suffix("/main.rs") {
                        if !dir_name.contains('/') {
                            actual_targets.insert(dir_name.to_owned());
                        }
                    }
                }
            }
            let recorded_set: BTreeSet<String> = recorded_tests.iter().cloned().collect();
            require!(
                recorded_set == actual_targets,
                "{mctx} (`{pkg_name}`): integration_tests mismatch: recorded {:?} vs actual {:?}",
                recorded_set,
                actual_targets
            );
        }

        let neural_tests = req_list(m, "neural_gated_integration_tests", &mctx)?;
        let mut recorded_whole_neural = BTreeSet::new();
        for nt_val in neural_tests {
            let nt_name = nt_val.as_str().ok_or_else(|| {
                Error::Invalid(format!(
                    "{mctx}: neural_gated_integration_tests entry must be string"
                ))
            })?;
            recorded_whole_neural.insert(nt_name.to_owned());
            total_neural_gated += 1;
        }

        let partial_neural = req_list(m, "partially_neural_gated_integration_tests", &mctx)?;
        let mut recorded_partial_neural = BTreeSet::new();
        for pnt_val in partial_neural {
            let pnt_name = pnt_val.as_str().ok_or_else(|| {
                Error::Invalid(format!(
                    "{mctx}: partially_neural_gated_integration_tests entry must be string"
                ))
            })?;
            recorded_partial_neural.insert(pnt_name.to_owned());
            total_partial_neural_gated += 1;
        }

        let testing_gated = req_list(m, "testing_gated_integration_tests", &mctx)?;
        let mut recorded_testing_gated = BTreeSet::new();
        for tg_val in testing_gated {
            let tg_name = tg_val.as_str().ok_or_else(|| {
                Error::Invalid(format!(
                    "{mctx}: testing_gated_integration_tests entry must be string"
                ))
            })?;
            recorded_testing_gated.insert(tg_name.to_owned());
        }

        if snapshot.has_git_commit() {
            let recorded_tests: Vec<String> = req_list(m, "integration_tests", &mctx)?
                .iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
            let mut actual_whole_neural = BTreeSet::new();
            let mut actual_partial_neural = BTreeSet::new();
            let mut actual_testing_gated = BTreeSet::new();
            for t_name in &recorded_tests {
                let direct_rel = format!("{member_dir}/tests/{t_name}.rs");
                let main_rel = format!("{member_dir}/tests/{t_name}/main.rs");
                let test_rel = if snapshot.file_exists(&direct_rel) {
                    direct_rel
                } else {
                    main_rel
                };
                let test_src = snapshot.blob_utf8(&test_rel)?;
                if test_src.contains("#![cfg(feature = \"neural\")]") {
                    actual_whole_neural.insert(t_name.clone());
                } else if test_src.contains("#[cfg(feature = \"neural\")]")
                    || test_src.contains("#[cfg(not(feature = \"neural\"))]")
                {
                    actual_partial_neural.insert(t_name.clone());
                }
                if test_src.contains("#![cfg(feature = \"testing\")]") {
                    actual_testing_gated.insert(t_name.clone());
                }
            }
            require!(
                recorded_whole_neural == actual_whole_neural,
                "{mctx} (`{pkg_name}`): neural_gated_integration_tests completeness mismatch: recorded {:?} vs actual {:?}",
                recorded_whole_neural,
                actual_whole_neural
            );
            require!(
                recorded_partial_neural == actual_partial_neural,
                "{mctx} (`{pkg_name}`): partially_neural_gated_integration_tests completeness mismatch: recorded {:?} vs actual {:?}",
                recorded_partial_neural,
                actual_partial_neural
            );
            require!(
                recorded_testing_gated == actual_testing_gated,
                "{mctx} (`{pkg_name}`): testing_gated_integration_tests completeness mismatch: recorded {:?} vs actual {:?}",
                recorded_testing_gated,
                actual_testing_gated
            );
        }
    }

    // Verify all committed *.schema.json files across the repository.
    let schema_files = req_list(file_map, "schemas", "FILE_MAP.i01.json")?;
    let schema_count = schema_files.len();
    let mut recorded_schema_paths = BTreeSet::new();
    for (idx, sf_val) in schema_files.iter().enumerate() {
        let sctx = format!("schemas[{idx}]");
        let sf = dict_of(sf_val, &sctx)?;
        let spath = req_str(sf, "path", &sctx)?;
        validate_repo_relative_path(spath)?;
        let expected_sha = req_str(sf, "sha256", &sctx)?;
        require!(
            is_lower_hex_sha64(expected_sha),
            "{sctx} (`{spath}`): sha256 must be 64-hex"
        );
        let expected_bytes = req_u64(sf, "bytes", &sctx)?;
        if snapshot.has_git_commit() {
            let bytes = snapshot.blob_bytes(spath)?;
            require!(
                u64::try_from(bytes.len()).ok() == Some(expected_bytes),
                "{sctx} (`{spath}`): byte count mismatch: expected {expected_bytes}, got {}",
                bytes.len()
            );
            let actual_sha = sha256_hex(&bytes);
            require!(
                actual_sha == expected_sha,
                "{sctx} (`{spath}`): SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
            );
        }
        recorded_schema_paths.insert(spath.to_owned());
    }
    if snapshot.has_git_commit() {
        let all_repo_schemas = snapshot.matching_paths("", ".schema.json");
        let repo_schema_count = all_repo_schemas.len();
        require!(
            schema_count == repo_schema_count,
            "schemas must list all {repo_schema_count} committed *.schema.json files across the repository, got {schema_count}"
        );
        let expected_schema_set: BTreeSet<String> = all_repo_schemas.into_iter().collect();
        require!(
            recorded_schema_paths == expected_schema_set,
            "schemas path set does not match repository *.schema.json files"
        );
    }

    Ok((
        member_count,
        total_neural_gated,
        total_partial_neural_gated,
        schema_count,
    ))
}

pub(super) fn verify_file_inventories_and_pins(
    snapshot: &mut BaselineSnapshot<'_>,
    file_map: &Dict,
) -> Result<()> {
    let pins = req_list(file_map, "historical_baseline_pins", "FILE_MAP.i01.json")?;
    let pin_count = pins.len();
    require!(
        pin_count >= MIN_HISTORICAL_PIN_COUNT,
        "historical_baseline_pins must contain at least {MIN_HISTORICAL_PIN_COUNT} pins, got {pin_count}"
    );
    for (idx, p_val) in pins.iter().enumerate() {
        let pctx = format!("historical_baseline_pins[{idx}]");
        let p = dict_of(p_val, &pctx)?;
        let commit_sha = req_str(p, "source_sha", &pctx)?;
        let tree_sha = req_str(p, "tree_sha", &pctx)?;
        let parent_sha = req_str(p, "parent_sha", &pctx)?;
        snapshot.verify_commit_object(commit_sha, tree_sha, parent_sha, &pctx)?;
    }

    let existing = req_list(file_map, "verified_existing_files", "FILE_MAP.i01.json")?;
    let existing_count = existing.len();
    require!(
        existing_count == EXPECTED_EXISTING_FILE_COUNT,
        "verified_existing_files must list all {EXPECTED_EXISTING_FILE_COUNT} existing baseline files from FILE_MAP.json, got {existing_count}"
    );
    if snapshot.has_git_commit() {
        let existing_paths: Vec<String> = existing
            .iter()
            .filter_map(|v| v.as_dict()?.get("path")?.as_str().map(str::to_owned))
            .collect();
        snapshot.preload_paths(&existing_paths)?;
    }
    for (idx, e_val) in existing.iter().enumerate() {
        let ectx = format!("verified_existing_files[{idx}]");
        let e = dict_of(e_val, &ectx)?;
        let path = req_str(e, "path", &ectx)?;
        validate_repo_relative_path(path)?;
        let exp_bytes = req_u64(e, "bytes", &ectx)?;
        let exp_sha = req_str(e, "sha256", &ectx)?;
        require!(
            is_lower_hex_sha64(exp_sha),
            "{ectx} (`{path}`): sha256 must be 64-hex"
        );
        if snapshot.has_git_commit() {
            let data = snapshot.blob_bytes(path)?;
            require!(
                u64::try_from(data.len()).ok() == Some(exp_bytes) && sha256_hex(&data) == exp_sha,
                "{ectx} (`{path}`): byte length or SHA-256 mismatch against baseline commit"
            );
        }
    }

    let v4_files = req_list(file_map, "verified_v4_packet_files", "FILE_MAP.i01.json")?;
    let v4_count = v4_files.len();
    require!(
        v4_count == V4_TOTAL_FILE_COUNT,
        "verified_v4_packet_files must list all {V4_TOTAL_FILE_COUNT} v4 packet files (PACKET_MANIFEST.v4.json + 39 payloads), got {v4_count}"
    );
    if snapshot.has_git_commit() {
        let v4_paths: Vec<String> = v4_files
            .iter()
            .filter_map(|v| v.as_dict()?.get("path")?.as_str().map(str::to_owned))
            .collect();
        snapshot.preload_paths(&v4_paths)?;
    }
    for (idx, vf_val) in v4_files.iter().enumerate() {
        let vctx = format!("verified_v4_packet_files[{idx}]");
        let vf = dict_of(vf_val, &vctx)?;
        let path = req_str(vf, "path", &vctx)?;
        validate_repo_relative_path(path)?;
        let exp_bytes = req_u64(vf, "bytes", &vctx)?;
        let exp_sha = req_str(vf, "sha256", &vctx)?;
        require!(
            is_lower_hex_sha64(exp_sha),
            "{vctx} (`{path}`): sha256 must be 64-hex"
        );
        if snapshot.has_git_commit() {
            let data = snapshot.blob_bytes(path)?;
            require!(
                u64::try_from(data.len()).ok() == Some(exp_bytes) && sha256_hex(&data) == exp_sha,
                "{vctx} (`{path}`): byte length or SHA-256 mismatch against baseline commit"
            );
        }
    }

    let write_paths = req_list(file_map, "proposed_task_write_paths", "FILE_MAP.i01.json")?;
    let write_path_count = write_paths.len();
    require!(
        write_path_count == EXPECTED_WRITE_PATH_COUNT,
        "proposed_task_write_paths must list all {EXPECTED_WRITE_PATH_COUNT} unique task write-path mappings, got {write_path_count}"
    );
    if snapshot.has_git_commit() {
        let existing_write_paths: Vec<String> = write_paths
            .iter()
            .filter_map(|v| {
                let d = v.as_dict()?;
                if d.get("exists_at_receiving_main") == Some(&Value::Bool(true)) {
                    d.get("path")?.as_str().map(str::to_owned)
                } else {
                    None
                }
            })
            .collect();
        snapshot.preload_paths(&existing_write_paths)?;
    }
    let mut covered_tasks = BTreeSet::new();
    for (idx, wp_val) in write_paths.iter().enumerate() {
        let wctx = format!("proposed_task_write_paths[{idx}]");
        let wp = dict_of(wp_val, &wctx)?;
        let path = req_str(wp, "path", &wctx)?;
        validate_repo_relative_path(path)?;
        let exists = req_bool(wp, "exists_at_receiving_main", &wctx)?;
        let tasks = req_list(wp, "tasks", &wctx)?;
        require!(
            !tasks.is_empty(),
            "{wctx} (`{path}`): tasks must not be empty"
        );
        for t_val in tasks {
            let tid = t_val
                .as_str()
                .ok_or_else(|| Error::Invalid(format!("{wctx}: task ID must be string")))?;
            covered_tasks.insert(tid.to_owned());
        }
        if snapshot.has_git_commit() {
            let actual_exists = snapshot.file_exists(path);
            require!(
                actual_exists == exists,
                "{wctx} (`{path}`): exists_at_receiving_main={exists} does not match baseline commit (actual_exists={actual_exists})"
            );
        }
        if exists {
            let sha = req_str(wp, "sha256_at_receiving_main", &wctx)?;
            require!(
                is_lower_hex_sha64(sha),
                "{wctx} (`{path}`): expected 64-hex sha256_at_receiving_main"
            );
            if snapshot.has_git_commit() {
                let data = snapshot.blob_bytes(path)?;
                let actual_sha = sha256_hex(&data);
                require!(
                    actual_sha == sha,
                    "{wctx} (`{path}`): sha256_at_receiving_main mismatch: expected {actual_sha}, got {sha}"
                );
            }
        }
    }
    let covered_task_count = covered_tasks.len();
    require!(
        covered_task_count == EXPECTED_TASK_COUNT,
        "proposed_task_write_paths must cover all {EXPECTED_TASK_COUNT} tasks (I01..I66), covered {covered_task_count}"
    );

    Ok(())
}

pub(super) fn verify_baseline_md_consistency(baseline_md: &str, file_map: &Dict) -> Result<()> {
    let mut table_cells: Vec<String> = Vec::new();
    for (line_idx, line) in baseline_md.lines().enumerate() {
        if line.trim_start().starts_with('|') {
            require!(
                !line.contains("\\n"),
                "BASELINE.md line {} contains literal `\\n` inside a Markdown table row",
                line_idx + 1
            );
            for cell in line.split('|') {
                let trimmed = cell.trim();
                if !trimmed.is_empty() {
                    table_cells.push(trimmed.to_owned());
                }
            }
        }
    }
    let cell_contains = |needle: &str| table_cells.iter().any(|c| c.contains(needle));

    for required_heading in [
        "# Executable baseline and capability gap map (`I01`)",
        "## 1. Receiving `main` pin and preserved historical baseline pins",
        "## 2. Core `I01` contract symbols, callers, and capability gaps at `22939ca1c67bdcc4beb54c5a4fbdb9e24073db2d`",
        "## 3. Reconciling benchmark/product profile identities and model lineages",
        "## 4. Workspace Cargo package inventory and feature gates at `22939ca1c67bdcc4beb54c5a4fbdb9e24073db2d`",
        "## 5. Committed JSON schema catalogue at `22939ca1c67bdcc4beb54c5a4fbdb9e24073db2d`",
        "## 6. Classification of source findings (`confirmed` / `rejected` / `unknown`) and separation of CI, historical runs, and proposed protocols",
        "## 7. Skipped and unavailable runtime combinations",
        "## 8. Verified baseline path inventory summary",
    ] {
        require!(
            baseline_md.contains(required_heading),
            "BASELINE.md missing required section heading `{required_heading}`"
        );
    }

    for p_val in req_list(file_map, "historical_baseline_pins", "FILE_MAP.i01.json")? {
        let p = dict_of(p_val, "historical_baseline_pins")?;
        let commit_sha = req_str(p, "source_sha", "historical_baseline_pins")?;
        let tree_sha = req_str(p, "tree_sha", "historical_baseline_pins")?;
        require!(
            cell_contains(commit_sha) && cell_contains(tree_sha),
            "BASELINE.md table cells missing historical pin commit `{commit_sha}` or tree `{tree_sha}`"
        );
    }

    for s_val in req_list(file_map, "symbols", "FILE_MAP.i01.json")? {
        let s = dict_of(s_val, "symbols")?;
        let sym = req_str(s, "symbol", "symbols")?;
        let path = req_str(s, "defining_path", "symbols")?;
        let line = req_u64(s, "defining_line", "symbols")?;
        let anchor = format!("{path}:{line}");
        require!(
            cell_contains(sym) && cell_contains(&anchor),
            "BASELINE.md table cells missing symbol `{sym}` or anchor `{anchor}`"
        );
    }

    for m_val in req_list(file_map, "workspace_packages", "FILE_MAP.i01.json")? {
        let m = dict_of(m_val, "workspace_packages")?;
        let member_dir = req_str(m, "member_dir", "workspace_packages")?;
        let pkg_name = req_str(m, "package_name", "workspace_packages")?;
        require!(
            cell_contains(member_dir) && cell_contains(pkg_name),
            "BASELINE.md table cells missing workspace package `{member_dir}` (`{pkg_name}`)"
        );
    }

    for sf_val in req_list(file_map, "schemas", "FILE_MAP.i01.json")? {
        let sf = dict_of(sf_val, "schemas")?;
        let spath = req_str(sf, "path", "schemas")?;
        let ssha = req_str(sf, "sha256", "schemas")?;
        require!(
            cell_contains(spath) && cell_contains(ssha),
            "BASELINE.md table cells missing schema `{spath}` or its SHA-256 `{ssha}`"
        );
    }

    for f_val in req_list(file_map, "source_findings", "FILE_MAP.i01.json")? {
        let f = dict_of(f_val, "source_findings")?;
        let fid = req_str(f, "id", "source_findings")?;
        let fclass = req_str(f, "classification", "source_findings")?;
        let fcat = req_str(f, "category", "source_findings")?;
        require!(
            cell_contains(fid) && cell_contains(fclass) && cell_contains(fcat),
            "BASELINE.md table cells missing finding `{fid}` (`{fclass}` / `{fcat}`)"
        );
    }

    for r_val in req_list(
        file_map,
        "skipped_and_unavailable_runtimes",
        "FILE_MAP.i01.json",
    )? {
        let r = dict_of(r_val, "skipped_and_unavailable_runtimes")?;
        let rid = req_str(r, "id", "skipped_and_unavailable_runtimes")?;
        let status = req_str(r, "status", "skipped_and_unavailable_runtimes")?;
        require!(
            cell_contains(rid) && cell_contains(status),
            "BASELINE.md table cells missing runtime ID `{rid}` or status `{status}`"
        );
    }

    Ok(())
}
