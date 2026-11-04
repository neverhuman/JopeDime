use super::*;

/// Acceptance 1 validator: reconcile benchmark and product search/model/tokenizer profile
/// identities (`SearchProfile` vs `SearchStrategy`, `401`-token native PolicyLM vs `282`-token
/// CLOVER prefix trie vs `native_loop_v2_weight_warm_start`).
pub fn verify_profile_identity_reconciliation(
    repo: &Path,
    reconciliation_val: &Value,
) -> Result<()> {
    let mut snapshot = BaselineSnapshot::new(repo, RECEIVING_MAIN_HEAD_SHA)?;
    verify_profile_identity_reconciliation_with_snapshot(&mut snapshot, reconciliation_val)
}

pub(super) fn verify_profile_identity_reconciliation_with_snapshot(
    snapshot: &mut BaselineSnapshot<'_>,
    reconciliation_val: &Value,
) -> Result<()> {
    let rec = dict_of(reconciliation_val, "profile_and_model_reconciliation")?;
    let vs = req_dict(
        rec,
        "search_vocabulary_vs_strategy",
        "profile_and_model_reconciliation",
    )?;

    let vocab_symbol = req_str(vs, "vocabulary_symbol", "search_vocabulary_vs_strategy")?;
    require!(
        vocab_symbol == "SearchProfile",
        "vocabulary_symbol must be `SearchProfile`, got `{vocab_symbol}`"
    );
    let vocab_path = req_str(vs, "vocabulary_path", "search_vocabulary_vs_strategy")?;
    validate_repo_relative_path(vocab_path)?;
    let vocab_line = req_u64(vs, "vocabulary_line", "search_vocabulary_vs_strategy")? as usize;
    let vocab_sha = req_str(
        vs,
        "vocabulary_file_sha256",
        "search_vocabulary_vs_strategy",
    )?;
    require!(
        is_lower_hex_sha64(vocab_sha),
        "`vocabulary_file_sha256` must be 64-hex"
    );

    let vocab_lines: Option<Vec<String>> = if snapshot.has_git_commit() {
        let vocab_bytes = snapshot.blob_bytes(vocab_path)?;
        require!(
            sha256_hex(&vocab_bytes) == vocab_sha,
            "`{vocab_path}` SHA-256 mismatch against `vocabulary_file_sha256`"
        );
        let vocab_src = String::from_utf8(vocab_bytes)
            .map_err(|e| Error::Invalid(format!("non-UTF-8 `{vocab_path}`: {e}")))?;
        let lines: Vec<String> = vocab_src.lines().map(str::to_owned).collect();
        require!(
            vocab_line >= 1
                && vocab_line <= lines.len()
                && lines[vocab_line - 1].contains("pub struct SearchProfile"),
            "`{vocab_path}:{vocab_line}` does not declare `pub struct SearchProfile`"
        );
        Some(lines)
    } else {
        None
    };

    let registered = req_list(vs, "registered_profiles", "search_vocabulary_vs_strategy")?;
    let expected_registry: [(&str, u64, &str, usize, bool); 8] = [
        ("legacy-wrappers-v1", 1, "legacy(false)", 729, false),
        ("legacy-wrappers-v2", 1, "legacy(true)", 730, false),
        (
            "kernel2-baseline",
            1,
            "kernel2(\"kernel2-baseline\")",
            731,
            false,
        ),
        (
            "kernel2-experimental",
            1,
            "kernel2(\"kernel2-experimental\")",
            732,
            false,
        ),
        (
            "kernel2-baseline",
            2,
            "expression_operators(\"kernel2-baseline\")",
            733,
            false,
        ),
        (
            "kernel2-experimental",
            2,
            "expression_operators(\"kernel2-experimental\")",
            734,
            false,
        ),
        (
            "kernel2-baseline",
            3,
            "pipeline_operators(\"kernel2-baseline\")",
            735,
            true,
        ),
        (
            "kernel2-experimental",
            3,
            "pipeline_operators(\"kernel2-experimental\")",
            736,
            true,
        ),
    ];
    let registered_count = registered.len();
    let expected_registry_count = expected_registry.len();
    require!(
        registered_count == expected_registry_count,
        "registered_profiles must match all {expected_registry_count} entries of feat_math::kernel::profile::profiles(), got {registered_count}"
    );
    let mut registered_pairs = BTreeSet::new();
    for (idx, (exp_id, exp_rev, exp_builder, exp_line, exp_shipped)) in
        expected_registry.iter().enumerate()
    {
        let pctx = format!("registered_profiles[{idx}]");
        let p = dict_of(&registered[idx], &pctx)?;
        let id = req_str(p, "id", &pctx)?;
        let rev = req_u64(p, "revision", &pctx)?;
        let builder = req_str(p, "builder", &pctx)?;
        let line = req_u64(p, "line", &pctx)? as usize;
        let shipped = req_bool(p, "in_shipped_kernel2_profiles", &pctx)?;
        require!(
            id == *exp_id
                && rev == *exp_rev
                && builder == *exp_builder
                && line == *exp_line
                && shipped == *exp_shipped,
            "{pctx}: expected ({exp_id}, rev {exp_rev}, {exp_builder}, line {exp_line}, shipped={exp_shipped}), got ({id}, rev {rev}, {builder}, line {line}, shipped={shipped})"
        );
        if let Some(ref vlines) = vocab_lines {
            require!(
                line >= 1 && line <= vlines.len() && vlines[line - 1].contains(exp_builder),
                "{pctx}: `{vocab_path}:{line}` does not contain `{exp_builder}`"
            );
        }
        registered_pairs.insert((id.to_owned(), rev));
    }

    let mappings = req_list(
        vs,
        "product_and_benchmark_profile_mapping",
        "search_vocabulary_vs_strategy",
    )?;
    let mapping_count = mappings.len();
    require!(
        mapping_count == EXPECTED_MAPPING_COUNT,
        "product_and_benchmark_profile_mapping must record the {EXPECTED_MAPPING_COUNT} CLI/benchmark profile options (legacy, kernel2-baseline, kernel2-experimental), got {mapping_count}"
    );
    if snapshot.has_git_commit() {
        let pipeline_messages =
            snapshot.blob_utf8("crates/jope-pipeline/src/kernel2/messages.rs")?;
        let pipeline_cli = snapshot.blob_utf8("crates/jope-pipeline/src/kernel2/cli.rs")?;
        let pipeline_search = snapshot.blob_utf8("crates/jope-pipeline/src/kernel2/search.rs")?;
        let bench_ablation = snapshot.blob_utf8("crates/jope-bench/src/ablation.rs")?;
        require!(
            pipeline_messages.contains("DEFAULT_SEARCH_PROFILE: &str = \"legacy\"")
                && pipeline_cli.contains("[\"kernel2-baseline\", \"kernel2-experimental\"]")
                && pipeline_search.contains("shipped_kernel2_profiles()"),
            "jope-pipeline kernel2 CLI/search profile wiring changed unexpectedly"
        );
        require!(
            bench_ablation.contains("Profile::Legacy => \"legacy\"")
                && bench_ablation.contains("Profile::Kernel2Baseline => \"kernel2-baseline\"")
                && bench_ablation
                    .contains("Profile::Kernel2Experimental => \"kernel2-experimental\""),
            "jope-bench ablation Profile::id() strings changed unexpectedly"
        );
    }
    for (idx, m_val) in mappings.iter().enumerate() {
        let mctx = format!("product_and_benchmark_profile_mapping[{idx}]");
        let m = dict_of(m_val, &mctx)?;
        let cli_id = req_str(m, "cli_and_bench_id", &mctx)?;
        let reg_id = req_str(m, "resolved_registry_id", &mctx)?;
        let reg_rev = req_u64(m, "resolved_registry_revision", &mctx)?;
        require!(
            registered_pairs.contains(&(reg_id.to_owned(), reg_rev)),
            "{mctx} (`{cli_id}`): resolved ({reg_id}, {reg_rev}) is not in registered_profiles"
        );
        match cli_id {
            "legacy" => {
                require!(
                    reg_id == "legacy-wrappers-v1" && reg_rev == 1,
                    "legacy must map to (`legacy-wrappers-v1`, 1), got (`{reg_id}`, {reg_rev})"
                );
            }
            "kernel2-baseline" => {
                require!(
                    reg_id == "kernel2-baseline" && reg_rev == 3,
                    "kernel2-baseline must map to (`kernel2-baseline`, 3), got (`{reg_id}`, {reg_rev})"
                );
            }
            "kernel2-experimental" => {
                require!(
                    reg_id == "kernel2-experimental" && reg_rev == 3,
                    "kernel2-experimental must map to (`kernel2-experimental`, 3), got (`{reg_id}`, {reg_rev})"
                );
            }
            other => {
                return Err(Error::Invalid(format!(
                    "{mctx}: unexpected cli_and_bench_id `{other}`"
                )));
            }
        }
    }

    let strategy_symbol = req_str(vs, "strategy_axis_symbol", "search_vocabulary_vs_strategy")?;
    require!(
        strategy_symbol == "SearchStrategy",
        "strategy_axis_symbol must be `SearchStrategy`, got `{strategy_symbol}`"
    );
    let strategy_status = req_str(vs, "strategy_axis_status", "search_vocabulary_vs_strategy")?;
    require!(
        strategy_status.contains("PROPOSED_SEPARATE_AXIS_I17_D08"),
        "strategy_axis_status must contain `PROPOSED_SEPARATE_AXIS_I17_D08`, got `{strategy_status}`"
    );
    let rule = req_str(vs, "reconciliation_rule", "search_vocabulary_vs_strategy")?;
    require!(
        rule.contains("SearchProfile")
            && rule.contains("SearchStrategy")
            && rule.contains("never be conflated"),
        "reconciliation_rule must explicitly forbid conflating SearchProfile and SearchStrategy"
    );

    let lineages = req_list(rec, "model_lineages", "profile_and_model_reconciliation")?;
    let lineage_count = lineages.len();
    require!(
        lineage_count == EXPECTED_LINEAGE_COUNT,
        "model_lineages must contain {EXPECTED_LINEAGE_COUNT} distinct lineage records, got {lineage_count}"
    );
    let mut saw_native = false;
    let mut saw_clover = false;
    let mut saw_warm_start = false;
    for (idx, lin_val) in lineages.iter().enumerate() {
        let lctx = format!("model_lineages[{idx}]");
        let lin = dict_of(lin_val, &lctx)?;
        let lid = req_str(lin, "lineage_id", &lctx)?;
        let def_path = req_str(lin, "defining_path", &lctx)?;
        validate_repo_relative_path(def_path)?;
        let def_sha = req_str(lin, "defining_file_sha256", &lctx)?;
        require!(
            is_lower_hex_sha64(def_sha),
            "{lctx}: defining_file_sha256 must be 64-hex"
        );
        if snapshot.has_git_commit() {
            let actual_def_bytes = snapshot.blob_bytes(def_path)?;
            require!(
                sha256_hex(&actual_def_bytes) == def_sha,
                "{lctx} (`{lid}`): defining_file_sha256 mismatch for `{def_path}`"
            );
        }
        match lid {
            "lineage_b_native_accepted_step60000" => {
                let width = req_u64(lin, "width", &lctx)?;
                let layers = req_u64(lin, "layers", &lctx)?;
                let heads = req_u64(lin, "heads", &lctx)?;
                let ffn = req_u64(lin, "ffn", &lctx)?;
                let max_seq = req_u64(lin, "max_seq", &lctx)?;
                require!(
                    width == 512 && layers == 6 && heads == 4 && ffn == 2048 && max_seq == 128,
                    "native PolicyLM architecture must be width=512, layers=6, heads=4, ffn=2048, max_seq=128"
                );
                let arch_path = req_str(lin, "architecture_witness_path", &lctx)?;
                let vocab_tokens = req_u64(lin, "vocabulary_tokens", &lctx)?;
                require!(
                    vocab_tokens == 401,
                    "native PolicyLM vocabulary_tokens must be 401, got {vocab_tokens}"
                );
                let vocab_wpath = req_str(lin, "vocabulary_witness_path", &lctx)?;
                let vocab_wsha = req_str(lin, "vocabulary_sha256", &lctx)?;
                require!(
                    is_lower_hex_sha64(vocab_wsha),
                    "native PolicyLM vocabulary_sha256 must be 64-hex"
                );
                let context_abi = req_u64(lin, "context_abi_values", &lctx)?;
                require!(
                    context_abi == 2048,
                    "native PolicyLM context_abi_values must be 2048, got {context_abi}"
                );
                let ctx_wpath = req_str(lin, "context_witness_path", &lctx)?;
                let accepted_step = req_u64(lin, "accepted_step", &lctx)?;
                let rejected_step = req_u64(lin, "rejected_checkpoint_step", &lctx)?;
                let hist_step = req_u64(lin, "historical_rtx3090_continuation_step", &lctx)?;
                let state_sha = req_str(lin, "state_sha256", &lctx)?;
                let status = req_str(lin, "status", &lctx)?;
                require!(
                    accepted_step == 60000 && rejected_step == 69000 && hist_step == 60100,
                    "native PolicyLM steps must be accepted_step=60000, rejected_checkpoint_step=69000, historical_rtx3090_continuation_step=60100"
                );
                require!(
                    state_sha == "8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6",
                    "native PolicyLM state_sha256 mismatch against ACCEPTED_STATE_SHA256: got `{state_sha}`"
                );
                require!(
                    status == "CONFIRMED_ACCEPTED_SERVING_BASELINE",
                    "native PolicyLM status must be `CONFIRMED_ACCEPTED_SERVING_BASELINE`, got `{status}`"
                );
                if snapshot.has_git_commit() {
                    let arch_src = snapshot.blob_utf8(arch_path)?;
                    require!(
                        arch_src.contains("width: 512") && arch_src.contains("layers: 6"),
                        "`{arch_path}` must witness width: 512 and layers: 6"
                    );
                    let grammar_bytes = snapshot.blob_bytes(vocab_wpath)?;
                    require!(
                        sha256_hex(&grammar_bytes) == vocab_wsha,
                        "`{vocab_wpath}` SHA-256 mismatch against `vocabulary_sha256`"
                    );
                    let ctx_src = snapshot.blob_utf8(ctx_wpath)?;
                    require!(
                        ctx_src.contains("pub const VALUES: usize = 2048;"),
                        "`{ctx_wpath}` must witness `pub const VALUES: usize = 2048;`"
                    );
                    let policy_lib_src = snapshot.blob_utf8("crates/jope-policy/src/lib.rs")?;
                    require!(
                        policy_lib_src.contains("pub const ACCEPTED_STEP: u64 = 60000;")
                            && policy_lib_src.contains("pub const REJECTED_STEP: u64 = 69000;")
                            && policy_lib_src.contains(state_sha),
                        "crates/jope-policy/src/lib.rs must witness ACCEPTED_STEP, REJECTED_STEP, and ACCEPTED_STATE_SHA256"
                    );
                }
                saw_native = true;
            }
            "clover_rlfw_prefix_trie_282" => {
                let vocab_tokens = req_u64(lin, "vocabulary_tokens", &lctx)?;
                let context_width = req_u64(lin, "context_width", &lctx)?;
                let trie_schema = req_str(lin, "trie_schema", &lctx)?;
                let trie_nodes = req_str(lin, "trie_nodes", &lctx)?;
                let status = req_str(lin, "status", &lctx)?;
                require!(
                    vocab_tokens == 282,
                    "CLOVER prefix-trie vocabulary_tokens must be 282, got {vocab_tokens}"
                );
                require!(
                    context_width == 2051,
                    "CLOVER PolicyLM context_width must be 2051, got {context_width}"
                );
                if snapshot.has_git_commit() {
                    let tok_src =
                        snapshot.blob_utf8("crates/jope-learning/src/support/tokenizer.rs")?;
                    require!(
                        tok_src.contains("282"),
                        "crates/jope-learning/src/support/tokenizer.rs must witness 282"
                    );
                    let exp_src =
                        snapshot.blob_utf8("crates/jope-learning/src/clover/expansion.rs")?;
                    require!(
                        exp_src.contains("self.context_width == 2051")
                            && exp_src.contains("PolicyLM/Kumo-2051"),
                        "crates/jope-learning/src/clover/expansion.rs must witness context_width == 2051 and PolicyLM/Kumo-2051"
                    );
                }
                require!(
                    trie_schema == "jope.flywheel-full-catalog-trie/v1",
                    "CLOVER prefix-trie trie_schema must be `jope.flywheel-full-catalog-trie/v1`, got `{trie_schema}`"
                );
                require!(
                    trie_nodes == "unknown",
                    "CLOVER prefix-trie trie_nodes must be `unknown` (dynamic Trie<'a> has no compiled node count constant), got `{trie_nodes}`"
                );
                require!(
                    status == "CONFIRMED_RESEARCH_SHADOW_NON_SERVING",
                    "CLOVER prefix-trie status must be `CONFIRMED_RESEARCH_SHADOW_NON_SERVING`, got `{status}`"
                );
                saw_clover = true;
            }
            "native_loop_v2_weight_warm_start" => {
                let block_updates = req_u64(lin, "block_updates", &lctx)?;
                let override_dims = req_str(lin, "warm_start_override_width_layers", &lctx)?;
                let status = req_str(lin, "status", &lctx)?;
                require!(
                    block_updates == 3000,
                    "native_loop_v2_weight_warm_start must record block_updates=3000"
                );
                if snapshot.has_git_commit() {
                    let ws_src =
                        snapshot.blob_utf8("crates/jope-engine/src/native_loop/warm_start.rs")?;
                    require!(
                        ws_src.contains("pub const BLOCK_UPDATES: u64 = 3000;"),
                        "warm_start.rs must witness BLOCK_UPDATES = 3000"
                    );
                }
                require!(
                    override_dims == "unknown",
                    "native_loop_v2_weight_warm_start warm_start_override_width_layers must be `unknown`, got `{override_dims}`"
                );
                require!(
                    status == "CONFIRMED_EXPERIMENTAL_NO_SUCCESSOR_PROTOCOL",
                    "native_loop_v2_weight_warm_start status must be `CONFIRMED_EXPERIMENTAL_NO_SUCCESSOR_PROTOCOL`, got `{status}`"
                );
                saw_warm_start = true;
            }
            other => {
                return Err(Error::Invalid(format!(
                    "unexpected model lineage_id `{other}`"
                )));
            }
        }
    }
    require!(
        saw_native && saw_clover && saw_warm_start,
        "model_lineages must include native PolicyLM (401), CLOVER prefix-trie (282), and v2 warm-start"
    );
    Ok(())
}
