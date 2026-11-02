use super::*;

/// Verify the 4 required `I01` symbol anchors and all 23 held native GPU routes in
/// `FILE_MAP.i01.json` against the baseline Rust source files.
pub fn verify_symbol_inventory(repo: &Path, symbols_val: &Value) -> Result<usize> {
    let mut snapshot = BaselineSnapshot::new(repo, RECEIVING_MAIN_HEAD_SHA)?;
    let (sym_count, _route_count) =
        verify_symbol_inventory_with_snapshot(&mut snapshot, symbols_val)?;
    Ok(sym_count)
}

pub(super) fn verify_symbol_inventory_with_snapshot(
    snapshot: &mut BaselineSnapshot<'_>,
    symbols_val: &Value,
) -> Result<(usize, usize)> {
    let symbols = list_of(symbols_val, "symbols")?;
    let required: [(&str, &str, usize, &str); 4] = [
        (
            "native_loop::verify_successor",
            "crates/jope-engine/src/native_loop.rs",
            82,
            "fn verify_successor(",
        ),
        (
            "CheckpointBinding::require_challenger_identity",
            "crates/jope-policy/src/config.rs",
            584,
            "pub fn require_challenger_identity(",
        ),
        (
            "require_original_gpu_grant",
            "crates/jope-engine/src/task_contract.rs",
            313,
            "pub fn require_original_gpu_grant(",
        ),
        (
            "RoundQualReport",
            "crates/jope-learning/src/clover/round_qual/model.rs",
            600,
            "pub struct RoundQualReport",
        ),
    ];

    let sym_len = symbols.len();
    let req_len = required.len();
    require!(
        sym_len == req_len,
        "symbols array must contain {req_len} entries, got {sym_len}"
    );

    let mut held_gpu_route_count = 0usize;

    for (idx, (expected_sym, expected_path, expected_line, expected_sig)) in
        required.iter().enumerate()
    {
        let sctx = format!("symbols[{idx}]");
        let entry = dict_of(&symbols[idx], &sctx)?;
        let sym = req_str(entry, "symbol", &sctx)?;
        require!(
            sym == *expected_sym,
            "{sctx}: expected symbol `{expected_sym}`, got `{sym}`"
        );
        let decl_path = req_str(entry, "defining_path", &sctx)?;
        validate_repo_relative_path(decl_path)?;
        require!(
            decl_path == *expected_path,
            "{sctx}: defining_path must be `{expected_path}`, got `{decl_path}`"
        );
        let decl_line = req_u64(entry, "defining_line", &sctx)? as usize;
        require!(
            decl_line == *expected_line,
            "{sctx}: defining_line must be {expected_line}, got {decl_line}"
        );
        let file_sha = req_str(entry, "defining_file_sha256", &sctx)?;
        require!(
            is_lower_hex_sha64(file_sha),
            "{sctx}: defining_file_sha256 must be 64-hex"
        );
        if snapshot.has_git_commit() {
            let raw = snapshot.blob_bytes(decl_path)?;
            require!(
                sha256_hex(&raw) == file_sha,
                "{sctx}: defining_file_sha256 mismatch for `{decl_path}`"
            );
            let src = String::from_utf8(raw)
                .map_err(|e| Error::Invalid(format!("{sctx}: non-UTF-8 `{decl_path}`: {e}")))?;
            let lines: Vec<&str> = src.lines().collect();
            require!(
                decl_line >= 1 && decl_line <= lines.len(),
                "{sctx}: defining_line {decl_line} out of range (1..={})",
                lines.len()
            );
            let actual_line = lines[decl_line - 1];
            require!(
                actual_line.contains(expected_sig),
                "{sctx}: `{decl_path}:{decl_line}` does not contain `{expected_sig}` (found `{actual_line}`)"
            );
        }

        if sym == "CheckpointBinding::require_challenger_identity" {
            let callers = req_list(entry, "callers", &sctx)?;
            let first_caller = callers.first().and_then(Value::as_str).unwrap_or_default();
            require!(
                first_caller.contains("CheckpointBinding::parse")
                    && !first_caller.contains("CheckpointBinding::verify"),
                "{sctx}: caller at config.rs:576 must be recorded as `CheckpointBinding::parse`, got `{first_caller}`"
            );
        }

        if sym == "require_original_gpu_grant" {
            let direct = req_list(entry, "direct_routes", &sctx)?;
            let indirect = req_list(
                entry,
                "indirect_routes_via_require_qualified_gpu_experiment",
                &sctx,
            )?;

            let mut recorded_direct = BTreeSet::new();
            for (r_idx, r_val) in direct.iter().enumerate() {
                let r_ctx = format!("{sctx}.direct_routes[{r_idx}]");
                let r = dict_of(r_val, &r_ctx)?;
                let r_path = req_str(r, "path", &r_ctx)?;
                let r_line = req_u64(r, "line", &r_ctx)? as usize;
                let r_route = req_str(r, "route", &r_ctx)?;
                recorded_direct.insert((r_path.to_owned(), r_line, r_route.to_owned()));
            }

            let mut recorded_indirect = BTreeSet::new();
            for (r_idx, r_val) in indirect.iter().enumerate() {
                let r_ctx =
                    format!("{sctx}.indirect_routes_via_require_qualified_gpu_experiment[{r_idx}]");
                let r = dict_of(r_val, &r_ctx)?;
                let r_path = req_str(r, "path", &r_ctx)?;
                let r_line = req_u64(r, "line", &r_ctx)? as usize;
                let r_route = req_str(r, "route", &r_ctx)?;
                recorded_indirect.insert((r_path.to_owned(), r_line, r_route.to_owned()));
            }

            if snapshot.has_git_commit() {
                let engine_rs_paths = snapshot.matching_paths("crates/jope-engine/src/", ".rs");
                snapshot.preload_paths(&engine_rs_paths)?;
                let mut actual_direct = BTreeSet::new();
                let mut actual_indirect = BTreeSet::new();
                for rs_path in &engine_rs_paths {
                    let rs_text = snapshot.blob_utf8(rs_path)?;
                    for (l_idx, l_str) in rs_text.lines().enumerate() {
                        let l_no = l_idx + 1;
                        if let Some(route) =
                            extract_string_arg_call(l_str, "require_original_gpu_grant(")
                        {
                            actual_direct.insert((rs_path.clone(), l_no, route));
                        }
                        if let Some(route) =
                            extract_string_arg_call(l_str, "require_qualified_gpu_experiment(")
                        {
                            actual_indirect.insert((rs_path.clone(), l_no, route));
                        }
                    }
                }
                require!(
                    recorded_direct == actual_direct,
                    "{sctx}: direct_routes mismatch against crates/jope-engine/src/ call sites: recorded {} vs actual {}",
                    recorded_direct.len(),
                    actual_direct.len()
                );
                require!(
                    recorded_indirect == actual_indirect,
                    "{sctx}: indirect_routes_via_require_qualified_gpu_experiment mismatch against crates/jope-engine/src/ call sites: recorded {} vs actual {}",
                    recorded_indirect.len(),
                    actual_indirect.len()
                );
            }

            held_gpu_route_count = recorded_direct.len() + recorded_indirect.len();
        }
    }

    Ok((sym_len, held_gpu_route_count))
}

fn extract_string_arg_call(line: &str, call_prefix: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.starts_with("//") || trimmed.starts_with("pub fn ") || trimmed.starts_with("fn ") {
        return None;
    }
    let idx = trimmed.find(call_prefix)?;
    let rest = &trimmed[idx + call_prefix.len()..];
    let rest = rest.strip_prefix('"')?;
    let end_quote = rest.find('"')?;
    Some(rest[..end_quote].to_owned())
}
