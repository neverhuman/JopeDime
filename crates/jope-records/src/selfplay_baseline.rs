//! Verify the `I01` executable baseline (`docs/plans/algorithmic-selfplay-20261017/BASELINE.md`)
//! and its file-map successor (`docs/plans/algorithmic-selfplay-20261017/FILE_MAP.i01.json`)
//! against the repository checkout, Cargo workspace manifests, committed JSON schemas, and the
//! immutable v4 unified workplan packet (`PACKET_MANIFEST.v4.json`).
//!
//! Covers the three `I01` acceptance criteria:
//! 1. `reconcile_benchmark_and_product_profile_identities`: reconciles benchmark search
//!    vocabularies (`SearchProfile` vs `SearchStrategy`) and model/tokenizer lineage identities
//!    (`lineage_b_native_accepted_step60000` 401-token native PolicyLM vs `clover_rlfw_prefix_trie_282`
//!    282-token CLOVER prefix trie vs `native_loop_v2_weight_warm_start`), rejecting conflation
//!    or unverified challenger promotion.
//! 2. `distinguish_current_ci_historical_runs_and_proposed_protocols`: classifies every finding as
//!    `confirmed`, `rejected`, or `unknown` with explicit evidence domain (`current_ci`,
//!    `historical_run`, or `proposed_protocol`), rejecting missing 40-hex commit / repo-relative
//!    witness bindings on `confirmed` findings, coerced `unknown` -> `confirmed` transitions, or
//!    historical/skipped/proposed runs promoted to live qualification.
//! 3. `record_all_skipped_and_unavailable_runtime_combinations`: inventories skipped/unavailable
//!    runtime combinations (`RT01`–`RT05`), rejecting 0-collected test filters or unverified GPU
//!    grant routes treated as passed or active.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path};

use sha2::{Digest, Sha256};

use crate::pyjson::{self, Dict, Value};
use crate::{require, Error, Result};

/// Relative directory of the unified algorithmic self-play workplan packet.
pub const PLAN_DIR: &str = "docs/plans/algorithmic-selfplay-20261017";

/// Relative path to the `I01` executable baseline document.
pub const BASELINE_MD_PATH: &str = "docs/plans/algorithmic-selfplay-20261017/BASELINE.md";

/// Relative path to the `I01` machine-readable file map successor.
pub const FILE_MAP_I01_PATH: &str = "docs/plans/algorithmic-selfplay-20261017/FILE_MAP.i01.json";

/// Relative path to the immutable v4 packet file map.
pub const FILE_MAP_V1_PATH: &str = "docs/plans/algorithmic-selfplay-20261017/FILE_MAP.json";

/// Relative path to the immutable v4 packet manifest.
pub const PACKET_MANIFEST_V4_PATH: &str =
    "docs/plans/algorithmic-selfplay-20261017/PACKET_MANIFEST.v4.json";

/// Expected SHA-256 of `PACKET_MANIFEST.v4.json`.
pub const EXPECTED_PACKET_MANIFEST_V4_SHA256: &str =
    "921643bc3ae0b7d75bbab13434f4d8211bc3c8638154c45ac00fc7df184750d5";

/// Expected SHA-256 of immutable `FILE_MAP.json`.
pub const EXPECTED_FILE_MAP_V1_SHA256: &str =
    "7c2eca800309a520aacf38c6fdf0af0929d9a8c497efcc08d03d9d98954cb2c2";

/// Expected receiving `main` commit SHA after PR #356 merge.
pub const RECEIVING_MAIN_HEAD_SHA: &str = "22939ca1c67bdcc4beb54c5a4fbdb9e24073db2d";

/// Summary returned by [`verify_selfplay_baseline`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelfplayBaselineSummary {
    pub schema: String,
    pub task_id: String,
    pub workspace_crate_count: usize,
    pub neural_gated_test_count: usize,
    pub schema_file_count: usize,
    pub symbol_count: usize,
    pub confirmed_finding_count: usize,
    pub rejected_finding_count: usize,
    pub unknown_finding_count: usize,
    pub runtime_combination_count: usize,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn read_utf8(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|e| Error::io(path, e))?;
    String::from_utf8(bytes).map_err(|e| Error::parse(path, e.to_string()))
}

fn read_json(path: &Path) -> Result<Value> {
    let text = read_utf8(path)?;
    pyjson::parse_str(&text).map_err(|e| Error::parse(path, e))
}

fn dict_of<'a>(val: &'a Value, ctx: &str) -> Result<&'a Dict> {
    val.as_dict()
        .ok_or_else(|| Error::Invalid(format!("{ctx}: expected JSON object")))
}

fn list_of<'a>(val: &'a Value, ctx: &str) -> Result<&'a Vec<Value>> {
    val.as_list()
        .ok_or_else(|| Error::Invalid(format!("{ctx}: expected JSON array")))
}

fn req_str<'a>(dict: &'a Dict, key: &str, ctx: &str) -> Result<&'a str> {
    dict.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Invalid(format!("{ctx}: missing or non-string `{key}`")))
}

fn req_u64(dict: &Dict, key: &str, ctx: &str) -> Result<u64> {
    let v = dict
        .get(key)
        .and_then(Value::as_int)
        .ok_or_else(|| Error::Invalid(format!("{ctx}: missing or non-integer `{key}`")))?;
    u64::try_from(v).map_err(|_| Error::Invalid(format!("{ctx}: negative integer `{key}` ({v})")))
}

fn req_bool(dict: &Dict, key: &str, ctx: &str) -> Result<bool> {
    match dict.get(key) {
        Some(Value::Bool(b)) => Ok(*b),
        _ => Err(Error::Invalid(format!(
            "{ctx}: missing or non-boolean `{key}`"
        ))),
    }
}

fn req_dict<'a>(dict: &'a Dict, key: &str, ctx: &str) -> Result<&'a Dict> {
    dict.get(key)
        .and_then(Value::as_dict)
        .ok_or_else(|| Error::Invalid(format!("{ctx}: missing or non-object `{key}`")))
}

fn req_list<'a>(dict: &'a Dict, key: &str, ctx: &str) -> Result<&'a Vec<Value>> {
    dict.get(key)
        .and_then(Value::as_list)
        .ok_or_else(|| Error::Invalid(format!("{ctx}: missing or non-array `{key}`")))
}

/// Check that `s` is a 40-character lowercase hexadecimal Git SHA-1.
pub fn is_lower_hex_sha40(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Check that `s` is a 64-character lowercase hexadecimal SHA-256 digest.
pub fn is_lower_hex_sha64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Ensure `rel` is a clean repository-relative path with no `..`, root, or prefix components.
pub fn validate_repo_relative_path(rel: &str) -> Result<()> {
    require!(
        !rel.is_empty(),
        "repository-relative path must not be empty"
    );
    require!(
        !rel.starts_with('/'),
        "path `{rel}` must be repository-relative, not absolute"
    );
    let path = Path::new(rel);
    for comp in path.components() {
        match comp {
            Component::Normal(_) => {}
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                return Err(Error::Invalid(format!(
                    "path `{rel}` contains disallowed component `{comp:?}`"
                )));
            }
        }
    }
    Ok(())
}

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
    require!(
        files.len() == 39,
        "PACKET_MANIFEST.v4.json must list 39 payload files, found {}",
        files.len()
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

/// Acceptance 1 validator: reconcile benchmark and product search/model/tokenizer profile
/// identities (`SearchProfile` vs `SearchStrategy`, `401`-token native PolicyLM vs `282`-token
/// CLOVER prefix trie vs `native_loop_v2_weight_warm_start`).
pub fn verify_profile_identity_reconciliation(
    repo: &Path,
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
    let vocab_full = repo.join(vocab_path);
    let vocab_bytes = fs::read(&vocab_full).map_err(|e| Error::io(&vocab_full, e))?;
    let vocab_sha = req_str(
        vs,
        "vocabulary_file_sha256",
        "search_vocabulary_vs_strategy",
    )?;
    require!(
        sha256_hex(&vocab_bytes) == vocab_sha,
        "`{vocab_path}` SHA-256 mismatch against `vocabulary_file_sha256`"
    );
    let vocab_src =
        String::from_utf8(vocab_bytes).map_err(|e| Error::parse(&vocab_full, e.to_string()))?;
    require!(
        vocab_src.contains("pub struct SearchProfile"),
        "`{vocab_path}` does not declare `pub struct SearchProfile`"
    );

    let factories = req_list(
        vs,
        "shipped_profile_factories",
        "search_vocabulary_vs_strategy",
    )?;
    require!(
        factories.len() == 7,
        "shipped_profile_factories must list 7 factories, got {}",
        factories.len()
    );

    let strategy_status = req_str(vs, "strategy_axis_status", "search_vocabulary_vs_strategy")?;
    require!(
        strategy_status == "PROPOSED_SEPARATE_AXIS_I17_D08",
        "strategy_axis_status must be `PROPOSED_SEPARATE_AXIS_I17_D08`, got `{strategy_status}`"
    );
    let rule = req_str(vs, "reconciliation_rule", "search_vocabulary_vs_strategy")?;
    require!(
        rule.contains("SearchProfile")
            && rule.contains("SearchStrategy")
            && rule.contains("never be conflated"),
        "reconciliation_rule must explicitly forbid conflating SearchProfile and SearchStrategy"
    );

    let lineages = req_list(rec, "model_lineages", "profile_and_model_reconciliation")?;
    require!(
        lineages.len() == 3,
        "model_lineages must contain 3 distinct lineage records, got {}",
        lineages.len()
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
        let actual_def_bytes =
            fs::read(repo.join(def_path)).map_err(|e| Error::io(&repo.join(def_path), e))?;
        require!(
            sha256_hex(&actual_def_bytes) == def_sha,
            "{lctx} (`{lid}`): defining_file_sha256 mismatch for `{def_path}`"
        );
        match lid {
            "lineage_b_native_accepted_step60000" => {
                let vocab_tokens = req_u64(lin, "vocabulary_tokens", &lctx)?;
                let context_len = req_u64(lin, "context_length", &lctx)?;
                let accepted_step = req_u64(lin, "accepted_step", &lctx)?;
                let rejected_step = req_u64(lin, "rejected_continuation_step", &lctx)?;
                let state_sha = req_str(lin, "state_sha256", &lctx)?;
                let status = req_str(lin, "status", &lctx)?;
                require!(
                    vocab_tokens == 401,
                    "native PolicyLM vocabulary_tokens must be 401, got {vocab_tokens}"
                );
                require!(
                    context_len == 2048,
                    "native PolicyLM context_length must be 2048, got {context_len}"
                );
                require!(
                    accepted_step == 60000 && rejected_step == 60100,
                    "native PolicyLM must record accepted_step=60000 and rejected_continuation_step=60100"
                );
                require!(
                    state_sha == "8ac80c26856734696935852f29500263a9703802505966a58e13a9769699e083",
                    "native PolicyLM state_sha256 mismatch: got `{state_sha}`"
                );
                require!(
                    status == "CONFIRMED_ACCEPTED_SERVING_BASELINE",
                    "native PolicyLM status must be `CONFIRMED_ACCEPTED_SERVING_BASELINE`, got `{status}`"
                );
                saw_native = true;
            }
            "clover_rlfw_prefix_trie_282" => {
                let vocab_tokens = req_u64(lin, "vocabulary_tokens", &lctx)?;
                let trie_nodes = req_u64(lin, "trie_nodes", &lctx)?;
                let status = req_str(lin, "status", &lctx)?;
                require!(
                    vocab_tokens == 282,
                    "CLOVER prefix-trie vocabulary_tokens must be 282, got {vocab_tokens}"
                );
                require!(
                    trie_nodes == 2051,
                    "CLOVER prefix-trie trie_nodes must be 2051, got {trie_nodes}"
                );
                require!(
                    status == "CONFIRMED_RESEARCH_SHADOW_NON_SERVING",
                    "CLOVER prefix-trie status must be `CONFIRMED_RESEARCH_SHADOW_NON_SERVING`, got `{status}`"
                );
                saw_clover = true;
            }
            "native_loop_v2_weight_warm_start" => {
                let vocab_tokens = req_u64(lin, "vocabulary_tokens", &lctx)?;
                let status = req_str(lin, "status", &lctx)?;
                require!(
                    vocab_tokens == 401,
                    "native_loop_v2_weight_warm_start vocabulary_tokens must be 401, got {vocab_tokens}"
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

/// Acceptance 2 validator: classify every finding in `source_findings` as `confirmed`, `rejected`,
/// or `unknown` and separate `current_ci`, `historical_run`, and `proposed_protocol` categories.
pub fn verify_finding_classifications(
    repo: &Path,
    findings_val: &Value,
) -> Result<(usize, usize, usize)> {
    let entries = list_of(findings_val, "source_findings")?;
    require!(!entries.is_empty(), "source_findings must not be empty");

    let mut confirmed_count = 0usize;
    let mut rejected_count = 0usize;
    let mut unknown_count = 0usize;
    let mut seen_ids = BTreeSet::new();

    for (idx, entry_val) in entries.iter().enumerate() {
        let ectx = format!("source_findings[{idx}]");
        let entry = dict_of(entry_val, &ectx)?;
        let fid = req_str(entry, "id", &ectx)?;
        require!(
            seen_ids.insert(fid.to_owned()),
            "{ectx}: duplicate finding id `{fid}`"
        );
        let classification = req_str(entry, "classification", &ectx)?;
        let category = req_str(entry, "category", &ectx)?;
        require!(
            matches!(
                category,
                "current_ci" | "historical_run" | "proposed_protocol"
            ),
            "{ectx} (`{fid}`): invalid category `{category}` (expected current_ci, historical_run, or proposed_protocol)"
        );
        let statement = req_str(entry, "statement", &ectx)?;
        require!(
            !statement.trim().is_empty(),
            "{ectx} (`{fid}`): empty statement"
        );

        let live_qualified = entry
            .get("live_qualification_active")
            .and_then(Value::as_str)
            .unwrap_or("false");
        require!(
            live_qualified != "true",
            "{ectx} (`{fid}`): historical, skipped, or proposed protocol records must never claim live_qualification_active=true"
        );

        let commit_sha = req_str(entry, "source_sha", &ectx)?;
        require!(
            is_lower_hex_sha40(commit_sha),
            "{ectx} (`{fid}`): finding must bind a 40-hex source_sha, got `{commit_sha}`"
        );
        let path = req_str(entry, "path", &ectx)?;
        validate_repo_relative_path(path)?;
        require!(
            repo.join(path).is_file(),
            "{ectx} (`{fid}`): path `{path}` does not exist on disk"
        );

        match classification {
            "confirmed" => {
                require!(
                    category != "proposed_protocol",
                    "{ectx} (`{fid}`): a `proposed_protocol` finding cannot be classified as `confirmed`"
                );
                confirmed_count += 1;
            }
            "rejected" => {
                rejected_count += 1;
            }
            "unknown" => {
                require!(
                    entry.get("coerced_to_confirmed").is_none(),
                    "{ectx} (`{fid}`): unknown finding must never set coerced_to_confirmed"
                );
                unknown_count += 1;
            }
            other => {
                return Err(Error::Invalid(format!(
                    "{ectx} (`{fid}`): invalid classification `{other}` (expected confirmed, rejected, or unknown)"
                )));
            }
        }
    }

    require!(
        confirmed_count > 0 && rejected_count > 0 && unknown_count > 0,
        "source_findings must include at least one confirmed ({confirmed_count}), rejected ({rejected_count}), and unknown ({unknown_count}) finding"
    );
    Ok((confirmed_count, rejected_count, unknown_count))
}

/// Acceptance 3 validator: verify all skipped and unavailable runtime combinations (`RT01`–`RT05`),
/// ensuring zero-test filters and unverified GPU grant routes are never marked passed or active.
pub fn verify_runtime_combinations(repo: &Path, runtime_val: &Value) -> Result<usize> {
    let entries = list_of(runtime_val, "skipped_and_unavailable_runtimes")?;
    require!(
        entries.len() >= 5,
        "skipped_and_unavailable_runtimes must record at least 5 combinations (RT01..RT05), got {}",
        entries.len()
    );

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
        require!(
            repo.join(source_path).is_file(),
            "{rctx} (`{rid}`): source_path `{source_path}` does not exist on disk"
        );
        let source_sha = req_str(entry, "source_sha", &rctx)?;
        require!(
            is_lower_hex_sha40(source_sha),
            "{rctx} (`{rid}`): source_sha must be 40-hex, got `{source_sha}`"
        );

        if rid == "RT01" {
            require!(
                targets.len() == 18,
                "RT01 must list all 18 neural-gated integration test targets in jope-parallel, got {}",
                targets.len()
            );
        }
        if rid == "RT03" {
            require!(
                targets.len() == 17,
                "RT03 must list all 17 native GPU routes held by require_original_gpu_grant, got {}",
                targets.len()
            );
        }
    }

    for expected_id in ["RT01", "RT02", "RT03", "RT04", "RT05"] {
        require!(
            seen_ids.contains(expected_id),
            "skipped_and_unavailable_runtimes missing required entry `{expected_id}`"
        );
    }

    Ok(entries.len())
}

/// Verify the 4 required `I01` symbol anchors in `FILE_MAP.i01.json` against actual Rust source files.
pub fn verify_symbol_inventory(repo: &Path, symbols_val: &Value) -> Result<usize> {
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
            "crates/jope-learning/src/clover/round_qual.rs",
            537,
            "pub struct RoundQualReport",
        ),
    ];

    require!(
        symbols.len() == required.len(),
        "symbols array must contain {} entries, got {}",
        required.len(),
        symbols.len()
    );

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
        let full_path = repo.join(decl_path);
        let raw = fs::read(&full_path).map_err(|e| Error::io(&full_path, e))?;
        let file_sha = req_str(entry, "defining_file_sha256", &sctx)?;
        require!(
            sha256_hex(&raw) == file_sha,
            "{sctx}: defining_file_sha256 mismatch for `{decl_path}`"
        );
        let src = String::from_utf8(raw).map_err(|e| Error::parse(&full_path, e.to_string()))?;
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

    Ok(symbols.len())
}

/// Verify the Cargo workspace crates, package-name mismatches, `jope-parallel` neural-gated
/// integration tests, and committed schemas recorded in `FILE_MAP.i01.json` against the live repository.
pub fn verify_workspace_and_schemas(repo: &Path, file_map: &Dict) -> Result<(usize, usize, usize)> {
    let members = req_list(file_map, "workspace_packages", "FILE_MAP.i01.json")?;
    require!(
        members.len() == 26,
        "workspace_packages must list all 26 workspace members, got {}",
        members.len()
    );

    let root_toml_path = repo.join("Cargo.toml");
    let root_toml_text = read_utf8(&root_toml_path)?;
    let root_toml: toml::Value = toml::from_str(&root_toml_text)
        .map_err(|e| Error::parse(&root_toml_path, e.to_string()))?;
    let root_members = root_toml
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(toml::Value::as_array)
        .ok_or_else(|| Error::Invalid("Cargo.toml missing [workspace].members".to_owned()))?;
    require!(
        root_members.len() == members.len(),
        "Cargo.toml [workspace].members count ({}) != FILE_MAP.i01.json workspace_packages count ({})",
        root_members.len(),
        members.len()
    );

    let mut total_neural_gated = 0usize;
    for (idx, m_val) in members.iter().enumerate() {
        let mctx = format!("workspace_packages[{idx}]");
        let m = dict_of(m_val, &mctx)?;
        let member_dir = req_str(m, "member_dir", &mctx)?;
        validate_repo_relative_path(member_dir)?;
        let manifest_rel = req_str(m, "cargo_toml_path", &mctx)?;
        validate_repo_relative_path(manifest_rel)?;
        let pkg_name = req_str(m, "package_name", &mctx)?;
        let manifest_path = repo.join(manifest_rel);
        let manifest_bytes = fs::read(&manifest_path).map_err(|e| Error::io(&manifest_path, e))?;
        let expected_sha = req_str(m, "cargo_toml_sha256", &mctx)?;
        require!(
            sha256_hex(&manifest_bytes) == expected_sha,
            "{mctx}: cargo_toml_sha256 mismatch for `{manifest_rel}`"
        );
        let manifest_text = String::from_utf8(manifest_bytes)
            .map_err(|e| Error::parse(&manifest_path, e.to_string()))?;
        let parsed: toml::Value = toml::from_str(&manifest_text)
            .map_err(|e| Error::parse(&manifest_path, e.to_string()))?;
        let actual_pkg = parsed
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(toml::Value::as_str)
            .ok_or_else(|| Error::Invalid(format!("{manifest_rel}: missing [package].name")))?;
        require!(
            actual_pkg == pkg_name,
            "{mctx}: package_name mismatch for `{member_dir}`: expected `{actual_pkg}`, got `{pkg_name}`"
        );

        let neural_tests = req_list(m, "neural_gated_integration_tests", &mctx)?;
        for nt_val in neural_tests {
            let nt_name = nt_val.as_str().ok_or_else(|| {
                Error::Invalid(format!(
                    "{mctx}: neural_gated_integration_tests entry must be string"
                ))
            })?;
            let test_file = repo
                .join(member_dir)
                .join("tests")
                .join(format!("{nt_name}.rs"));
            let test_src = read_utf8(&test_file)?;
            require!(
                test_src.contains("#![cfg(feature = \"neural\")]"),
                "`{}` must contain `#![cfg(feature = \"neural\")]`",
                test_file.display()
            );
            total_neural_gated += 1;
        }
    }
    require!(
        total_neural_gated == 18,
        "workspace_packages must record 18 neural-gated integration tests in jope-parallel, got {total_neural_gated}"
    );

    let schema_files = req_list(file_map, "schemas", "FILE_MAP.i01.json")?;
    require!(
        schema_files.len() == 14,
        "schemas must list 14 schema files, got {}",
        schema_files.len()
    );
    for (idx, sf_val) in schema_files.iter().enumerate() {
        let sctx = format!("schemas[{idx}]");
        let sf = dict_of(sf_val, &sctx)?;
        let spath = req_str(sf, "path", &sctx)?;
        validate_repo_relative_path(spath)?;
        let expected_sha = req_str(sf, "sha256", &sctx)?;
        let expected_bytes = req_u64(sf, "bytes", &sctx)?;
        let full = repo.join(spath);
        let bytes = fs::read(&full).map_err(|e| Error::io(&full, e))?;
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

    Ok((members.len(), total_neural_gated, schema_files.len()))
}

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
    for (line_idx, line) in baseline_md.lines().enumerate() {
        if line.trim_start().starts_with('|') {
            require!(
                !line.contains("\\n"),
                "BASELINE.md line {} contains literal `\\n` inside a Markdown table row",
                line_idx + 1
            );
        }
    }
    for required_heading in [
        "# Executable baseline and capability gap map (`I01`)",
        "## 1. Receiving `main` pin and preserved historical baseline pins",
        "## 2. Core `I01` contract symbols, callers, and capability gaps at `22939ca1c67bdcc4beb54c5a4fbdb9e24073db2d`",
        "## 3. Reconciling benchmark/product profile identities and model lineages",
        "## 4. Workspace Cargo package inventory and feature gates at `22939ca1c67bdcc4beb54c5a4fbdb9e24073db2d`",
        "## 5. Committed JSON schema catalogue at `22939ca1c67bdcc4beb54c5a4fbdb9e24073db2d`",
        "## 6. Classification of source findings (`confirmed` / `rejected` / `unknown`) and separation of CI, historical runs, and proposed protocols",
        "## 7. Skipped and unavailable runtime combinations",
    ] {
        require!(
            baseline_md.contains(required_heading),
            "BASELINE.md missing required section heading `{required_heading}`"
        );
    }

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
    require!(
        source_sha == RECEIVING_MAIN_HEAD_SHA && is_lower_hex_sha40(source_sha),
        "FILE_MAP.i01.json source_sha must be `{RECEIVING_MAIN_HEAD_SHA}`, got `{source_sha}`"
    );

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

    let rec_val = i01.get("profile_and_model_reconciliation").ok_or_else(|| {
        Error::Invalid("FILE_MAP.i01.json missing `profile_and_model_reconciliation`".to_owned())
    })?;
    verify_profile_identity_reconciliation(repo, rec_val)?;

    let sym_val = i01
        .get("symbols")
        .ok_or_else(|| Error::Invalid("FILE_MAP.i01.json missing `symbols`".to_owned()))?;
    let symbol_count = verify_symbol_inventory(repo, sym_val)?;

    let (workspace_crate_count, neural_gated_test_count, schema_file_count) =
        verify_workspace_and_schemas(repo, i01)?;

    let findings_val = i01
        .get("source_findings")
        .ok_or_else(|| Error::Invalid("FILE_MAP.i01.json missing `source_findings`".to_owned()))?;
    let (confirmed_finding_count, rejected_finding_count, unknown_finding_count) =
        verify_finding_classifications(repo, findings_val)?;

    let runtime_val = i01.get("skipped_and_unavailable_runtimes").ok_or_else(|| {
        Error::Invalid("FILE_MAP.i01.json missing `skipped_and_unavailable_runtimes`".to_owned())
    })?;
    let runtime_combination_count = verify_runtime_combinations(repo, runtime_val)?;

    Ok(SelfplayBaselineSummary {
        schema: schema.to_owned(),
        task_id: task_id.to_owned(),
        workspace_crate_count,
        neural_gated_test_count,
        schema_file_count,
        symbol_count,
        confirmed_finding_count,
        rejected_finding_count,
        unknown_finding_count,
        runtime_combination_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root")
            .to_path_buf()
    }

    fn load_i01_json() -> Value {
        let root = repo_root();
        read_json(&root.join(FILE_MAP_I01_PATH)).expect("parse FILE_MAP.i01.json")
    }

    #[test]
    fn reconcile_benchmark_and_product_profile_identities() {
        let root = repo_root();
        let summary = verify_selfplay_baseline(&root).expect("I01 baseline must verify");
        assert_eq!(summary.schema, "jope.unified-workplan.file-map/v2");
        assert_eq!(summary.task_id, "I01");
        assert_eq!(summary.workspace_crate_count, 26);
        assert_eq!(summary.neural_gated_test_count, 18);
        assert_eq!(summary.schema_file_count, 14);
        assert_eq!(summary.symbol_count, 4);

        // Negative case 1a: conflating SearchProfile and SearchStrategy must fail.
        let mut conflated = load_i01_json();
        conflated
            .as_dict_mut()
            .unwrap()
            .get_mut("profile_and_model_reconciliation")
            .unwrap()
            .as_dict_mut()
            .unwrap()
            .get_mut("search_vocabulary_vs_strategy")
            .unwrap()
            .as_dict_mut()
            .unwrap()
            .insert("vocabulary_symbol", Value::from("SearchStrategy"));
        let err = verify_profile_identity_reconciliation(
            &root,
            conflated.get("profile_and_model_reconciliation").unwrap(),
        )
        .expect_err("conflating SearchProfile and SearchStrategy must be rejected");
        assert!(
            err.to_string().contains("SearchProfile"),
            "unexpected error: {err}"
        );

        // Negative case 1b: conflating 401-token native PolicyLM with 282-token CLOVER vocabulary must fail.
        let mut wrong_vocab = load_i01_json();
        let lineages = wrong_vocab
            .as_dict_mut()
            .unwrap()
            .get_mut("profile_and_model_reconciliation")
            .unwrap()
            .as_dict_mut()
            .unwrap()
            .get_mut("model_lineages")
            .unwrap()
            .as_list_mut()
            .unwrap();
        lineages[0]
            .as_dict_mut()
            .unwrap()
            .insert("vocabulary_tokens", Value::from(282_i64));
        let err = verify_profile_identity_reconciliation(
            &root,
            wrong_vocab.get("profile_and_model_reconciliation").unwrap(),
        )
        .expect_err("conflating 401-token native PolicyLM and 282-token CLOVER vocab must fail");
        assert!(err.to_string().contains("401"), "unexpected error: {err}");

        // Negative case 1c: promoting unaccepted v2 warm-start as accepted serving must fail.
        let mut unaccepted_promoted = load_i01_json();
        let lineages = unaccepted_promoted
            .as_dict_mut()
            .unwrap()
            .get_mut("profile_and_model_reconciliation")
            .unwrap()
            .as_dict_mut()
            .unwrap()
            .get_mut("model_lineages")
            .unwrap()
            .as_list_mut()
            .unwrap();
        lineages[2]
            .as_dict_mut()
            .unwrap()
            .insert("status", Value::from("CONFIRMED_ACCEPTED_SERVING_BASELINE"));
        let err = verify_profile_identity_reconciliation(
            &root,
            unaccepted_promoted
                .get("profile_and_model_reconciliation")
                .unwrap(),
        )
        .expect_err("promoting unaccepted warm-start checkpoint must fail");
        assert!(
            err.to_string()
                .contains("CONFIRMED_EXPERIMENTAL_NO_SUCCESSOR_PROTOCOL"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn distinguish_current_ci_historical_runs_and_proposed_protocols() {
        let root = repo_root();
        let doc = load_i01_json();
        let findings = doc.get("source_findings").unwrap();
        let (confirmed, rejected, unknown) =
            verify_finding_classifications(&root, findings).expect("findings must verify");
        assert_eq!(confirmed, 6);
        assert_eq!(rejected, 2);
        assert_eq!(unknown, 2);

        // Negative case 2a: confirmed finding with missing/invalid 40-hex commit SHA must fail.
        let mut bad_sha = doc.clone();
        bad_sha
            .as_dict_mut()
            .unwrap()
            .get_mut("source_findings")
            .unwrap()
            .as_list_mut()
            .unwrap()[0]
            .as_dict_mut()
            .unwrap()
            .insert("source_sha", Value::from("not-a-40-hex-sha"));
        let err = verify_finding_classifications(&root, bad_sha.get("source_findings").unwrap())
            .expect_err("confirmed finding without 40-hex SHA must fail");
        assert!(
            err.to_string().contains("40-hex source_sha"),
            "unexpected error: {err}"
        );

        // Negative case 2b: unknown proposed_protocol finding coerced to confirmed must fail.
        let mut coerced_unknown = doc.clone();
        let entries = coerced_unknown
            .as_dict_mut()
            .unwrap()
            .get_mut("source_findings")
            .unwrap()
            .as_list_mut()
            .unwrap();
        let unknown_entry = entries
            .iter_mut()
            .find(|e| e.get("classification").and_then(Value::as_str) == Some("unknown"))
            .expect("at least one unknown entry");
        unknown_entry
            .as_dict_mut()
            .unwrap()
            .insert("classification", Value::from("confirmed"));
        let err =
            verify_finding_classifications(&root, coerced_unknown.get("source_findings").unwrap())
                .expect_err("coercing unknown proposed_protocol finding to confirmed must fail");
        assert!(
            err.to_string()
                .contains("`proposed_protocol` finding cannot be classified as `confirmed`"),
            "unexpected error: {err}"
        );

        // Negative case 2c: claiming live_qualification_active=true on a historical or proposed protocol must fail.
        let mut false_live = doc;
        false_live
            .as_dict_mut()
            .unwrap()
            .get_mut("source_findings")
            .unwrap()
            .as_list_mut()
            .unwrap()[0]
            .as_dict_mut()
            .unwrap()
            .insert("live_qualification_active", Value::from("true"));
        let err = verify_finding_classifications(&root, false_live.get("source_findings").unwrap())
            .expect_err("claiming live_qualification_active=true must fail");
        assert!(
            err.to_string().contains("live_qualification_active=true"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn record_all_skipped_and_unavailable_runtime_combinations() {
        let root = repo_root();
        let doc = load_i01_json();
        let runtime = doc.get("skipped_and_unavailable_runtimes").unwrap();
        let count =
            verify_runtime_combinations(&root, runtime).expect("runtime combinations must verify");
        assert_eq!(count, 5);

        // Negative case 3a: marking a skipped/zero-test/GPU-gated runtime combination as passed/active must fail.
        let mut false_pass = doc.clone();
        false_pass
            .as_dict_mut()
            .unwrap()
            .get_mut("skipped_and_unavailable_runtimes")
            .unwrap()
            .as_list_mut()
            .unwrap()[0]
            .as_dict_mut()
            .unwrap()
            .insert("status", Value::from("PASSED_DEFAULT_LANE"));
        let err = verify_runtime_combinations(
            &root,
            false_pass.get("skipped_and_unavailable_runtimes").unwrap(),
        )
        .expect_err("counting a 0-collected or skipped runtime combination as passed must fail");
        assert!(
            err.to_string().contains("unrecognized runtime status"),
            "unexpected error: {err}"
        );

        // Negative case 3b: under-reporting jope-parallel neural-gated test files must fail.
        let mut wrong_neural_count = doc;
        wrong_neural_count
            .as_dict_mut()
            .unwrap()
            .get_mut("skipped_and_unavailable_runtimes")
            .unwrap()
            .as_list_mut()
            .unwrap()[0]
            .as_dict_mut()
            .unwrap()
            .get_mut("affected_targets")
            .unwrap()
            .as_list_mut()
            .unwrap()
            .pop();
        let err = verify_runtime_combinations(
            &root,
            wrong_neural_count
                .get("skipped_and_unavailable_runtimes")
                .unwrap(),
        )
        .expect_err("under-reporting neural-gated test files must fail");
        assert!(
            err.to_string().contains("18 neural-gated"),
            "unexpected error: {err}"
        );
    }
}
