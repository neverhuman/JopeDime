//! Verify the `I01` executable baseline (`docs/plans/algorithmic-selfplay-20261017/BASELINE.md`)
//! and its file-map successor (`docs/plans/algorithmic-selfplay-20261017/FILE_MAP.i01.json`)
//! against the pinned receiving `main` commit (`22939ca1c67bdcc4beb54c5a4fbdb9e24073db2d`),
//! Cargo workspace manifests, committed JSON schemas, and the immutable v4 unified workplan packet
//! (`PACKET_MANIFEST.v4.json`).
//!
//! Covers the three `I01` acceptance criteria:
//! 1. `reconcile_benchmark_and_product_profile_identities`: reconciles the 8-entry `SearchProfile`
//!    registry (`crates/feat-math/src/kernel/profile.rs:727-738`), the 2-entry
//!    `shipped_kernel2_profiles()` registry (`:748-753`), the product (`jope-pipeline`) and
//!    benchmark (`jope-bench`) profile mappings (`legacy`, `kernel2-baseline`,
//!    `kernel2-experimental`), the planned orthogonal `SearchStrategy` axis (`I17`–`I19`, `D08`),
//!    and the three model/tokenizer lineages (`lineage_b_native_accepted_step60000` 401-token
//!    native PolicyLM vs `clover_rlfw_prefix_trie_282` 282-token CLOVER prefix trie vs
//!    `native_loop_v2_weight_warm_start`).
//! 2. `distinguish_current_ci_historical_runs_and_proposed_protocols`: classifies every finding as
//!    `confirmed`, `rejected`, or `unknown` with explicit evidence domain (`current_ci`,
//!    `historical_run`, or `proposed_protocol`), rejecting missing 40-hex commit / repo-relative
//!    witness bindings on `confirmed` findings, coerced `unknown` -> `confirmed` transitions across
//!    any category, or historical/skipped/proposed runs promoted to live qualification.
//! 3. `record_all_skipped_and_unavailable_runtime_combinations`: inventories skipped/unavailable
//!    runtime combinations (`RT01`–`RT07`) dynamically against the workspace manifests,
//!    `crates/jope-engine/src/` GPU guard call sites, `docs/parity/test-inventory.json` (`72`
//!    ignored tests), and `deploy/ci/coverage.json` (`22` non-default features), rejecting
//!    0-collected test filters or unverified GPU grant routes treated as passed or active.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Component, Path};
use std::process::{Command, Stdio};

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

/// Expected receiving `main` tree SHA after PR #356 merge.
pub const RECEIVING_MAIN_TREE_SHA: &str = "885189fe077e4a23acd7333370c1e95b57193bc4";

/// Expected receiving `main` parent SHA after PR #356 merge.
pub const RECEIVING_MAIN_PARENT_SHA: &str = "0e063f4ce110dd142f18f8482867c2d3d237598c";

/// Summary returned by [`verify_selfplay_baseline`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelfplayBaselineSummary {
    pub schema: String,
    pub task_id: String,
    pub workspace_crate_count: usize,
    pub neural_gated_test_count: usize,
    pub partially_neural_gated_test_count: usize,
    pub schema_file_count: usize,
    pub symbol_count: usize,
    pub held_gpu_route_count: usize,
    pub confirmed_finding_count: usize,
    pub rejected_finding_count: usize,
    pub unknown_finding_count: usize,
    pub runtime_combination_count: usize,
}

/// Snapshot reader that resolves pinned baseline files from the Git commit object (`source_sha`)
/// when available in the local Git store, falling back to `repo` on disk only when running outside
/// a Git checkout containing `source_sha`.
///
/// Reading pinned baseline files from `source_sha` ensures that subsequent tasks (`I02`–`I66`)
/// modifying `crates/**`, `Cargo.toml`, or `schemas/**` do not break `jope-records` unit tests in
/// the default `rust-test` CI lane, while `BASELINE.md`, `FILE_MAP.i01.json`, and the immutable v4
/// packet are always verified directly from the working tree (`repo`).
pub struct BaselineSnapshot<'a> {
    repo: &'a Path,
    source_sha: String,
    has_git_commit: bool,
    tree_files: BTreeSet<String>,
    blob_cache: BTreeMap<String, Vec<u8>>,
}

impl<'a> BaselineSnapshot<'a> {
    /// Construct a snapshot bound to `repo` and `source_sha`, pre-loading `ls-tree` when
    /// `source_sha` is present in the local Git object database.
    pub fn new(repo: &'a Path, source_sha: &str) -> Result<Self> {
        let has_git_commit = Command::new("git")
            .arg("-C")
            .arg(repo)
            .arg("cat-file")
            .arg("-e")
            .arg(format!("{source_sha}^{{commit}}"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        let mut tree_files = BTreeSet::new();
        if has_git_commit {
            let output = Command::new("git")
                .arg("-C")
                .arg(repo)
                .arg("ls-tree")
                .arg("-r")
                .arg("--name-only")
                .arg(source_sha)
                .output()
                .map_err(|e| Error::io(repo, e))?;
            require!(
                output.status.success(),
                "git ls-tree -r --name-only {source_sha} failed"
            );
            let text =
                String::from_utf8(output.stdout).map_err(|e| Error::parse(repo, e.to_string()))?;
            for line in text.lines() {
                if !line.is_empty() {
                    tree_files.insert(line.to_owned());
                }
            }
        }

        Ok(Self {
            repo,
            source_sha: source_sha.to_owned(),
            has_git_commit,
            tree_files,
            blob_cache: BTreeMap::new(),
        })
    }

    /// Preload a batch of repository-relative paths from `source_sha` in a single
    /// `git cat-file --batch` invocation for fast verification.
    pub fn preload_paths<I, S>(&mut self, paths: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if !self.has_git_commit {
            return Ok(());
        }
        let mut to_load = Vec::new();
        for p in paths {
            let rel = p.as_ref();
            if !self.blob_cache.contains_key(rel) && self.tree_files.contains(rel) {
                to_load.push(rel.to_owned());
            }
        }
        to_load.sort();
        to_load.dedup();
        if to_load.is_empty() {
            return Ok(());
        }

        let mut child = Command::new("git")
            .arg("-C")
            .arg(self.repo)
            .arg("cat-file")
            .arg("--batch")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| Error::io(self.repo, e))?;

        {
            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| Error::Invalid("missing stdin for git cat-file --batch".into()))?;
            for rel in &to_load {
                writeln!(stdin, "{}:{rel}", self.source_sha)
                    .map_err(|e| Error::io(self.repo, e))?;
            }
        }

        let out = child
            .wait_with_output()
            .map_err(|e| Error::io(self.repo, e))?;
        require!(out.status.success(), "git cat-file --batch failed");

        let buf = &out.stdout;
        let mut pos = 0usize;
        for rel in to_load {
            let nl = buf[pos..].iter().position(|&b| b == b'\n').ok_or_else(|| {
                Error::Invalid(format!("truncated git cat-file --batch header for `{rel}`"))
            })?;
            let header = std::str::from_utf8(&buf[pos..pos + nl])
                .map_err(|e| Error::Invalid(format!("invalid utf8 header for `{rel}`: {e}")))?;
            pos += nl + 1;
            let parts: Vec<&str> = header.split_whitespace().collect();
            require!(
                parts.len() >= 3 && parts[1] == "blob",
                "unexpected git cat-file --batch header `{header}` for `{rel}`"
            );
            let size: usize = parts[2]
                .parse()
                .map_err(|e| Error::Invalid(format!("invalid blob size in `{header}`: {e}")))?;
            require!(
                pos + size < buf.len(),
                "truncated git cat-file --batch body for `{rel}`"
            );
            let data = buf[pos..pos + size].to_vec();
            pos += size + 1; // trailing LF after blob payload
            self.blob_cache.insert(rel, data);
        }
        Ok(())
    }

    /// Check whether `rel` exists at the pinned baseline commit (or on disk if outside Git).
    pub fn file_exists(&self, rel: &str) -> bool {
        if self.has_git_commit {
            self.tree_files.contains(rel)
        } else {
            self.repo.join(rel).is_file()
        }
    }

    /// Read raw bytes of `rel` from the pinned baseline commit (or from disk if outside Git).
    pub fn read_bytes(&mut self, rel: &str) -> Result<Vec<u8>> {
        if let Some(cached) = self.blob_cache.get(rel) {
            return Ok(cached.clone());
        }
        if self.has_git_commit {
            self.preload_paths([rel])?;
            self.blob_cache.get(rel).cloned().ok_or_else(|| {
                Error::Invalid(format!(
                    "path `{rel}` does not exist in baseline commit `{}`",
                    self.source_sha
                ))
            })
        } else {
            let full = self.repo.join(rel);
            fs::read(&full).map_err(|e| Error::io(&full, e))
        }
    }

    /// Read UTF-8 text of `rel` from the pinned baseline commit.
    pub fn read_utf8(&mut self, rel: &str) -> Result<String> {
        let bytes = self.read_bytes(rel)?;
        String::from_utf8(bytes).map_err(|e| Error::Invalid(format!("non-UTF-8 file `{rel}`: {e}")))
    }

    /// Return all repository-relative file paths tracked in the pinned baseline commit under
    /// `prefix` that end with `suffix`.
    pub fn matching_paths(&self, prefix: &str, suffix: &str) -> Vec<String> {
        if self.has_git_commit {
            self.tree_files
                .iter()
                .filter(|p| p.starts_with(prefix) && p.ends_with(suffix))
                .cloned()
                .collect()
        } else {
            let mut out = Vec::new();
            collect_disk_paths(self.repo, &self.repo.join(prefix), suffix, &mut out);
            out.sort();
            out
        }
    }

    /// Verify a commit's `tree_sha` and `parent_sha` in the local Git store when the commit object
    /// is available.
    pub fn verify_commit_object(
        &self,
        commit_sha: &str,
        expected_tree: &str,
        expected_parent: &str,
        ctx: &str,
    ) -> Result<()> {
        require!(
            is_lower_hex_sha40(commit_sha),
            "{ctx}: invalid 40-hex commit SHA `{commit_sha}`"
        );
        require!(
            is_lower_hex_sha40(expected_tree),
            "{ctx}: invalid 40-hex tree SHA `{expected_tree}`"
        );
        require!(
            is_lower_hex_sha40(expected_parent),
            "{ctx}: invalid 40-hex parent SHA `{expected_parent}`"
        );
        if !self.has_git_commit {
            return Ok(());
        }
        let output = Command::new("git")
            .arg("-C")
            .arg(self.repo)
            .arg("cat-file")
            .arg("-p")
            .arg(commit_sha)
            .output()
            .map_err(|e| Error::io(self.repo, e))?;
        if !output.status.success() {
            return Ok(());
        }
        let text =
            String::from_utf8(output.stdout).map_err(|e| Error::parse(self.repo, e.to_string()))?;
        let actual_tree = text
            .lines()
            .find_map(|l| l.strip_prefix("tree "))
            .ok_or_else(|| Error::Invalid(format!("{ctx}: commit `{commit_sha}` missing tree")))?;
        let actual_parent = text
            .lines()
            .find_map(|l| l.strip_prefix("parent "))
            .ok_or_else(|| {
                Error::Invalid(format!("{ctx}: commit `{commit_sha}` missing parent"))
            })?;
        require!(
            actual_tree == expected_tree,
            "{ctx} (`{commit_sha}`): tree_sha mismatch: expected {expected_tree}, got {actual_tree}"
        );
        require!(
            actual_parent == expected_parent,
            "{ctx} (`{commit_sha}`): parent_sha mismatch: expected {expected_parent}, got {actual_parent}"
        );
        Ok(())
    }
}

fn collect_disk_paths(repo: &Path, dir: &Path, suffix: &str, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_disk_paths(repo, &p, suffix, out);
        } else if p.is_file() {
            if let Ok(rel) = p.strip_prefix(repo) {
                let rel_str = rel.to_string_lossy().replace('\\', "/");
                if rel_str.ends_with(suffix) {
                    out.push(rel_str);
                }
            }
        }
    }
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
    let mut snapshot = BaselineSnapshot::new(repo, RECEIVING_MAIN_HEAD_SHA)?;
    verify_profile_identity_reconciliation_with_snapshot(&mut snapshot, reconciliation_val)
}

fn verify_profile_identity_reconciliation_with_snapshot(
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
    let vocab_bytes = snapshot.read_bytes(vocab_path)?;
    let vocab_sha = req_str(
        vs,
        "vocabulary_file_sha256",
        "search_vocabulary_vs_strategy",
    )?;
    require!(
        sha256_hex(&vocab_bytes) == vocab_sha,
        "`{vocab_path}` SHA-256 mismatch against `vocabulary_file_sha256`"
    );
    let vocab_src = String::from_utf8(vocab_bytes)
        .map_err(|e| Error::Invalid(format!("non-UTF-8 `{vocab_path}`: {e}")))?;
    let vocab_lines: Vec<&str> = vocab_src.lines().collect();
    require!(
        vocab_line >= 1
            && vocab_line <= vocab_lines.len()
            && vocab_lines[vocab_line - 1].contains("pub struct SearchProfile"),
        "`{vocab_path}:{vocab_line}` does not declare `pub struct SearchProfile`"
    );

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
    require!(
        registered.len() == expected_registry.len(),
        "registered_profiles must match all {} entries of feat_math::kernel::profile::profiles(), got {}",
        expected_registry.len(),
        registered.len()
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
        require!(
            line >= 1 && line <= vocab_lines.len() && vocab_lines[line - 1].contains(exp_builder),
            "{pctx}: `{vocab_path}:{line}` does not contain `{exp_builder}`"
        );
        registered_pairs.insert((id.to_owned(), rev));
    }

    let mappings = req_list(
        vs,
        "product_and_benchmark_profile_mapping",
        "search_vocabulary_vs_strategy",
    )?;
    require!(
        mappings.len() == 3,
        "product_and_benchmark_profile_mapping must record the 3 CLI/benchmark profile options (legacy, kernel2-baseline, kernel2-experimental), got {}",
        mappings.len()
    );
    let pipeline_messages = snapshot.read_utf8("crates/jope-pipeline/src/kernel2/messages.rs")?;
    let pipeline_cli = snapshot.read_utf8("crates/jope-pipeline/src/kernel2/cli.rs")?;
    let pipeline_search = snapshot.read_utf8("crates/jope-pipeline/src/kernel2/search.rs")?;
    let bench_ablation = snapshot.read_utf8("crates/jope-bench/src/ablation.rs")?;
    require!(
        pipeline_messages.contains("DEFAULT_SEARCH_PROFILE: &str = \"legacy\"")
            && pipeline_cli.contains("[\"kernel2-baseline\", \"kernel2-experimental\"]")
            && pipeline_search.contains("shipped_kernel2_profiles()"),
        "jope-pipeline kernel2 CLI/search profile wiring changed unexpectedly"
    );
    require!(
        bench_ablation.contains("Profile::Legacy => \"legacy\"")
            && bench_ablation.contains("Profile::Kernel2Baseline => \"kernel2-baseline\"")
            && bench_ablation.contains("Profile::Kernel2Experimental => \"kernel2-experimental\""),
        "jope-bench ablation Profile::id() strings changed unexpectedly"
    );
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
        let actual_def_bytes = snapshot.read_bytes(def_path)?;
        require!(
            sha256_hex(&actual_def_bytes) == def_sha,
            "{lctx} (`{lid}`): defining_file_sha256 mismatch for `{def_path}`"
        );
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
                let arch_src = snapshot.read_utf8(arch_path)?;
                require!(
                    arch_src.contains("width: 512") && arch_src.contains("layers: 6"),
                    "`{arch_path}` must witness width: 512 and layers: 6"
                );

                let vocab_tokens = req_u64(lin, "vocabulary_tokens", &lctx)?;
                require!(
                    vocab_tokens == 401,
                    "native PolicyLM vocabulary_tokens must be 401, got {vocab_tokens}"
                );
                let vocab_wpath = req_str(lin, "vocabulary_witness_path", &lctx)?;
                let vocab_wsha = req_str(lin, "vocabulary_sha256", &lctx)?;
                let grammar_bytes = snapshot.read_bytes(vocab_wpath)?;
                require!(
                    sha256_hex(&grammar_bytes) == vocab_wsha,
                    "`{vocab_wpath}` SHA-256 mismatch against `vocabulary_sha256`"
                );

                let context_len = req_u64(lin, "context_length", &lctx)?;
                require!(
                    context_len == 2048,
                    "native PolicyLM context_length must be 2048, got {context_len}"
                );
                let ctx_wpath = req_str(lin, "context_witness_path", &lctx)?;
                let ctx_src = snapshot.read_utf8(ctx_wpath)?;
                require!(
                    ctx_src.contains("pub const VALUES: usize = 2048;"),
                    "`{ctx_wpath}` must witness `pub const VALUES: usize = 2048;`"
                );

                let accepted_step = req_u64(lin, "accepted_step", &lctx)?;
                let rejected_step = req_u64(lin, "rejected_checkpoint_step", &lctx)?;
                let hist_step = req_u64(lin, "historical_rtx3090_continuation_step", &lctx)?;
                let state_sha = req_str(lin, "state_sha256", &lctx)?;
                let status = req_str(lin, "status", &lctx)?;
                let policy_lib_src = snapshot.read_utf8("crates/jope-policy/src/lib.rs")?;
                require!(
                    accepted_step == 60000
                        && policy_lib_src.contains("pub const ACCEPTED_STEP: u64 = 60000;"),
                    "native PolicyLM must record accepted_step=60000 witnessed in crates/jope-policy/src/lib.rs"
                );
                require!(
                    rejected_step == 69000
                        && policy_lib_src.contains("pub const REJECTED_STEP: u64 = 69000;"),
                    "native PolicyLM must record rejected_checkpoint_step=69000 witnessed in crates/jope-policy/src/lib.rs"
                );
                require!(
                    hist_step == 60100,
                    "native PolicyLM must record historical_rtx3090_continuation_step=60100"
                );
                require!(
                    state_sha == "8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6"
                        && policy_lib_src.contains(state_sha),
                    "native PolicyLM state_sha256 mismatch against ACCEPTED_STATE_SHA256 in crates/jope-policy/src/lib.rs: got `{state_sha}`"
                );
                require!(
                    status == "CONFIRMED_ACCEPTED_SERVING_BASELINE",
                    "native PolicyLM status must be `CONFIRMED_ACCEPTED_SERVING_BASELINE`, got `{status}`"
                );
                saw_native = true;
            }
            "clover_rlfw_prefix_trie_282" => {
                let vocab_tokens = req_u64(lin, "vocabulary_tokens", &lctx)?;
                let trie_schema = req_str(lin, "trie_schema", &lctx)?;
                let trie_nodes = req_str(lin, "trie_nodes", &lctx)?;
                let status = req_str(lin, "status", &lctx)?;
                let tok_src =
                    snapshot.read_utf8("crates/jope-learning/src/support/tokenizer.rs")?;
                require!(
                    vocab_tokens == 282 && tok_src.contains("282"),
                    "CLOVER prefix-trie vocabulary_tokens must be 282, got {vocab_tokens}"
                );
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
                let ws_src =
                    snapshot.read_utf8("crates/jope-engine/src/native_loop/warm_start.rs")?;
                require!(
                    block_updates == 3000
                        && ws_src.contains("pub const BLOCK_UPDATES: u64 = 3000;"),
                    "native_loop_v2_weight_warm_start must record block_updates=3000"
                );
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

/// Acceptance 2 validator: classify every finding in `source_findings` as `confirmed`, `rejected`,
/// or `unknown` and separate `current_ci`, `historical_run`, and `proposed_protocol` categories.
pub fn verify_finding_classifications(
    repo: &Path,
    findings_val: &Value,
) -> Result<(usize, usize, usize)> {
    let mut snapshot = BaselineSnapshot::new(repo, RECEIVING_MAIN_HEAD_SHA)?;
    verify_finding_classifications_with_snapshot(&mut snapshot, findings_val)
}

fn verify_finding_classifications_with_snapshot(
    snapshot: &mut BaselineSnapshot<'_>,
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

        match entry.get("live_qualification_active") {
            Some(Value::Bool(false)) => {}
            Some(Value::Bool(true)) | Some(Value::Str(_)) => {
                return Err(Error::Invalid(format!(
                    "{ectx} (`{fid}`): historical, skipped, or proposed protocol records must never claim live_qualification_active=true (must be boolean false)"
                )));
            }
            _ => {
                return Err(Error::Invalid(format!(
                    "{ectx} (`{fid}`): missing or non-boolean `live_qualification_active`"
                )));
            }
        }

        let commit_sha = req_str(entry, "source_sha", &ectx)?;
        require!(
            is_lower_hex_sha40(commit_sha),
            "{ectx} (`{fid}`): finding must bind a 40-hex source_sha, got `{commit_sha}`"
        );
        let path = req_str(entry, "path", &ectx)?;
        validate_repo_relative_path(path)?;
        require!(
            snapshot.file_exists(path),
            "{ectx} (`{fid}`): path `{path}` does not exist in baseline commit"
        );
        let witness_line = req_u64(entry, "witness_line", &ectx)? as usize;
        let file_bytes = snapshot.read_bytes(path)?;
        let file_text = String::from_utf8(file_bytes.clone())
            .map_err(|e| Error::Invalid(format!("{ectx}: non-UTF-8 `{path}`: {e}")))?;
        let line_count = file_text.lines().count().max(1);
        require!(
            witness_line >= 1 && witness_line <= line_count,
            "{ectx} (`{fid}`): witness_line {witness_line} out of range (1..={line_count}) for `{path}`"
        );
        let evidence_sha = req_str(entry, "evidence_sha256", &ectx)?;

        match classification {
            "confirmed" => {
                require!(
                    category != "proposed_protocol",
                    "{ectx} (`{fid}`): a `proposed_protocol` finding cannot be classified as `confirmed`"
                );
                require!(
                    evidence_sha != "unknown" && is_lower_hex_sha64(evidence_sha),
                    "{ectx} (`{fid}`): confirmed finding cannot have `evidence_sha256 = \"{evidence_sha}\"` (unknown findings must never be coerced to confirmed)"
                );
                require!(
                    entry.get("resolution_gate").is_none()
                        && entry.get("coerced_to_confirmed").is_none(),
                    "{ectx} (`{fid}`): unresolved/unknown finding with resolution_gate must never be coerced to `confirmed`"
                );
                let actual_sha = sha256_hex(&file_bytes);
                require!(
                    actual_sha == evidence_sha,
                    "{ectx} (`{fid}`): evidence_sha256 mismatch for `{path}`: expected {actual_sha}, got {evidence_sha}"
                );
                confirmed_count += 1;
            }
            "rejected" => {
                let reason = req_str(entry, "rejection_reason", &ectx)?;
                require!(
                    !reason.trim().is_empty(),
                    "{ectx} (`{fid}`): rejected finding must include a non-empty `rejection_reason`"
                );
                require!(
                    is_lower_hex_sha64(evidence_sha) && sha256_hex(&file_bytes) == evidence_sha,
                    "{ectx} (`{fid}`): rejected finding evidence_sha256 mismatch for `{path}`"
                );
                rejected_count += 1;
            }
            "unknown" => {
                require!(
                    evidence_sha == "unknown",
                    "{ectx} (`{fid}`): unknown finding must set `evidence_sha256: \"unknown\"`, got `{evidence_sha}`"
                );
                let gate = req_str(entry, "resolution_gate", &ectx)?;
                require!(
                    !gate.trim().is_empty(),
                    "{ectx} (`{fid}`): unknown finding must specify a non-empty `resolution_gate`"
                );
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

/// Acceptance 3 validator: verify all skipped and unavailable runtime combinations (`RT01`–`RT07`)
/// dynamically against workspace manifests, GPU hold routes, `docs/parity/test-inventory.json`,
/// and `deploy/ci/coverage.json`.
pub fn verify_runtime_combinations(repo: &Path, runtime_val: &Value) -> Result<usize> {
    let mut snapshot = BaselineSnapshot::new(repo, RECEIVING_MAIN_HEAD_SHA)?;
    let i01_val = read_json(&repo.join(FILE_MAP_I01_PATH))?;
    let i01 = dict_of(&i01_val, "FILE_MAP.i01.json")?;
    verify_runtime_combinations_with_snapshot(&mut snapshot, i01, runtime_val)
}

fn verify_runtime_combinations_with_snapshot(
    snapshot: &mut BaselineSnapshot<'_>,
    file_map: &Dict,
    runtime_val: &Value,
) -> Result<usize> {
    let entries = list_of(runtime_val, "skipped_and_unavailable_runtimes")?;
    let required_ids = ["RT01", "RT02", "RT03", "RT04", "RT05", "RT06", "RT07"];
    require!(
        entries.len() >= required_ids.len(),
        "skipped_and_unavailable_runtimes must record all {} combinations (RT01..RT07), got {}",
        required_ids.len(),
        entries.len()
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

    let test_inv_text = snapshot.read_utf8("docs/parity/test-inventory.json")?;
    let test_inv_val = pyjson::parse_str(&test_inv_text)
        .map_err(|e| Error::Invalid(format!("docs/parity/test-inventory.json: {e}")))?;
    let test_inv = dict_of(&test_inv_val, "docs/parity/test-inventory.json")?;
    let rust_inv = req_dict(test_inv, "rust", "docs/parity/test-inventory.json")?;
    let packages = req_dict(rust_inv, "packages", "test-inventory.rust")?;
    let mut expected_total_ignored = 0u64;
    let mut expected_ignored_binaries = 0usize;
    for (_pkg_key, p_val) in packages.iter() {
        let p_dict = dict_of(p_val, "test-inventory.rust.packages")?;
        let binaries = req_dict(p_dict, "binaries", "test-inventory.rust.packages")?;
        for (_bin_key, b_val) in binaries.iter() {
            let b_dict = dict_of(b_val, "test-inventory.binaries")?;
            let ign = req_u64(b_dict, "ignored", "test-inventory.binaries")?;
            if ign > 0 {
                expected_total_ignored += ign;
                expected_ignored_binaries += 1;
            }
        }
    }

    let cov_text = snapshot.read_utf8("deploy/ci/coverage.json")?;
    let cov_val = pyjson::parse_str(&cov_text)
        .map_err(|e| Error::Invalid(format!("deploy/ci/coverage.json: {e}")))?;
    let cov = dict_of(&cov_val, "deploy/ci/coverage.json")?;
    let cov_items = req_dict(cov, "items", "deploy/ci/coverage.json")?;
    let expected_non_default_features: Vec<String> = cov_items
        .keys()
        .filter_map(|k| k.strip_prefix("feature:"))
        .filter(|rest| !rest.ends_with("/default"))
        .map(str::to_owned)
        .collect();

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
        require!(
            snapshot.file_exists(source_path),
            "{rctx} (`{rid}`): source_path `{source_path}` does not exist in baseline commit"
        );
        let source_sha = req_str(entry, "source_sha", &rctx)?;
        require!(
            is_lower_hex_sha40(source_sha),
            "{rctx} (`{rid}`): source_sha must be 40-hex, got `{source_sha}`"
        );

        match rid {
            "RT01" => {
                let whole = req_list(entry, "whole_file_gated_targets", &rctx)?;
                let partial = req_list(entry, "partially_gated_targets", &rctx)?;
                require!(
                    whole.len() == expected_whole_neural.len(),
                    "RT01 whole_file_gated_targets count ({}) must match workspace neural_gated_integration_tests count ({})",
                    whole.len(),
                    expected_whole_neural.len()
                );
                require!(
                    partial.len() == expected_partial_neural.len(),
                    "RT01 partially_gated_targets count ({}) must match workspace partially_neural_gated_integration_tests count ({})",
                    partial.len(),
                    expected_partial_neural.len()
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
                    let src = snapshot.read_utf8(rel_path)?;
                    let lines: Vec<&str> = src.lines().collect();
                    require!(
                        line_no >= 1
                            && line_no <= lines.len()
                            && lines[line_no - 1].contains("#[cfg(feature = \"neural\")]"),
                        "RT01 `{p_str}` does not contain `#[cfg(feature = \"neural\")]`"
                    );
                }
                let expected_total = expected_whole_neural.len() + expected_partial_neural.len();
                require!(
                    targets.len() == expected_total,
                    "RT01 affected_targets must list all {expected_total} neural-gated integration test targets in jope-parallel ({} whole-file + {} partially gated), got {}",
                    expected_whole_neural.len(),
                    expected_partial_neural.len(),
                    targets.len()
                );
            }
            "RT02" => {
                require!(
                    targets.len() == expected_testing_gated.len(),
                    "RT02 affected_targets count ({}) must match jope-policy testing_gated_integration_tests count ({})",
                    targets.len(),
                    expected_testing_gated.len()
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
                require!(
                    direct.len() == expected_direct_gpu_routes.len(),
                    "RT03 direct_routes count ({}) must match symbol inventory direct_routes count ({})",
                    direct.len(),
                    expected_direct_gpu_routes.len()
                );
                require!(
                    indirect.len() == expected_indirect_gpu_routes.len(),
                    "RT03 indirect_routes_via_require_qualified_gpu_experiment count ({}) must match symbol inventory indirect count ({})",
                    indirect.len(),
                    expected_indirect_gpu_routes.len()
                );
                let expected_total_routes =
                    expected_direct_gpu_routes.len() + expected_indirect_gpu_routes.len();
                require!(
                    targets.len() == expected_total_routes,
                    "RT03 affected_targets must list all {expected_total_routes} held GPU routes ({} direct + {} via require_qualified_gpu_experiment), got {}",
                    expected_direct_gpu_routes.len(),
                    expected_indirect_gpu_routes.len(),
                    targets.len()
                );
            }
            "RT04" => {
                let total_ign = req_u64(entry, "total_ignored_count", &rctx)?;
                require!(
                    total_ign == expected_total_ignored,
                    "RT04 total_ignored_count ({total_ign}) must match docs/parity/test-inventory.json sum ({expected_total_ignored})"
                );
                require!(
                    targets.len() == expected_ignored_binaries,
                    "RT04 affected_targets count ({}) must match test-inventory binaries with ignored > 0 ({expected_ignored_binaries})",
                    targets.len()
                );
            }
            "RT05" => {
                let nd_feats = req_list(entry, "non_default_features", &rctx)?;
                require!(
                    nd_feats.len() == expected_non_default_features.len()
                        && targets.len() == expected_non_default_features.len(),
                    "RT05 non_default_features ({}) and affected_targets ({}) must match deploy/ci/coverage.json features count ({})",
                    nd_feats.len(),
                    targets.len(),
                    expected_non_default_features.len()
                );
                let req_bins = req_list(entry, "required_feature_binaries", &rctx)?;
                require!(
                    req_bins.len() == expected_req_feature_bins.len(),
                    "RT05 required_feature_binaries count ({}) must match workspace required_feature_targets bin count ({})",
                    req_bins.len(),
                    expected_req_feature_bins.len()
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

    Ok(entries.len())
}

/// Verify the 4 required `I01` symbol anchors and all 23 held native GPU routes in
/// `FILE_MAP.i01.json` against the baseline Rust source files.
pub fn verify_symbol_inventory(repo: &Path, symbols_val: &Value) -> Result<usize> {
    let mut snapshot = BaselineSnapshot::new(repo, RECEIVING_MAIN_HEAD_SHA)?;
    let (sym_count, _route_count) =
        verify_symbol_inventory_with_snapshot(&mut snapshot, symbols_val)?;
    Ok(sym_count)
}

fn verify_symbol_inventory_with_snapshot(
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
        let raw = snapshot.read_bytes(decl_path)?;
        let file_sha = req_str(entry, "defining_file_sha256", &sctx)?;
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

            // Scan all crates/jope-engine/src/**/*.rs in the snapshot to discover every real call site.
            let engine_rs_paths = snapshot.matching_paths("crates/jope-engine/src/", ".rs");
            snapshot.preload_paths(&engine_rs_paths)?;
            let mut actual_direct = BTreeSet::new();
            let mut actual_indirect = BTreeSet::new();
            for rs_path in &engine_rs_paths {
                let rs_text = snapshot.read_utf8(rs_path)?;
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

            let mut recorded_direct = BTreeSet::new();
            for (r_idx, r_val) in direct.iter().enumerate() {
                let r_ctx = format!("{sctx}.direct_routes[{r_idx}]");
                let r = dict_of(r_val, &r_ctx)?;
                let r_path = req_str(r, "path", &r_ctx)?;
                let r_line = req_u64(r, "line", &r_ctx)? as usize;
                let r_route = req_str(r, "route", &r_ctx)?;
                recorded_direct.insert((r_path.to_owned(), r_line, r_route.to_owned()));
            }
            require!(
                recorded_direct == actual_direct,
                "{sctx}: direct_routes mismatch against crates/jope-engine/src/ call sites: recorded {} vs actual {}",
                recorded_direct.len(),
                actual_direct.len()
            );

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
            require!(
                recorded_indirect == actual_indirect,
                "{sctx}: indirect_routes_via_require_qualified_gpu_experiment mismatch against crates/jope-engine/src/ call sites: recorded {} vs actual {}",
                recorded_indirect.len(),
                actual_indirect.len()
            );

            held_gpu_route_count = recorded_direct.len() + recorded_indirect.len();
        }
    }

    Ok((symbols.len(), held_gpu_route_count))
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

/// Verify the Cargo workspace crates, package-name mismatches, `jope-parallel` neural-gated
/// integration tests (whole-file and partial), `jope-policy` testing-gated integration tests, and
/// all 29 committed `*.schema.json` files recorded in `FILE_MAP.i01.json`.
pub fn verify_workspace_and_schemas(repo: &Path, file_map: &Dict) -> Result<(usize, usize, usize)> {
    let mut snapshot = BaselineSnapshot::new(repo, RECEIVING_MAIN_HEAD_SHA)?;
    let (members, whole_neural, _partial_neural, schemas) =
        verify_workspace_and_schemas_with_snapshot(&mut snapshot, file_map)?;
    Ok((members, whole_neural, schemas))
}

fn verify_workspace_and_schemas_with_snapshot(
    snapshot: &mut BaselineSnapshot<'_>,
    file_map: &Dict,
) -> Result<(usize, usize, usize, usize)> {
    let members = req_list(file_map, "workspace_packages", "FILE_MAP.i01.json")?;
    let root_toml_text = snapshot.read_utf8("Cargo.toml")?;
    let root_toml: toml::Value = toml::from_str(&root_toml_text)
        .map_err(|e| Error::Invalid(format!("Cargo.toml parse error: {e}")))?;
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
    let mut total_partial_neural_gated = 0usize;

    for (idx, m_val) in members.iter().enumerate() {
        let mctx = format!("workspace_packages[{idx}]");
        let m = dict_of(m_val, &mctx)?;
        let member_dir = req_str(m, "member_dir", &mctx)?;
        validate_repo_relative_path(member_dir)?;
        let manifest_rel = req_str(m, "cargo_toml_path", &mctx)?;
        validate_repo_relative_path(manifest_rel)?;
        let pkg_name = req_str(m, "package_name", &mctx)?;
        let manifest_bytes = snapshot.read_bytes(manifest_rel)?;
        let expected_sha = req_str(m, "cargo_toml_sha256", &mctx)?;
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

        let neural_tests = req_list(m, "neural_gated_integration_tests", &mctx)?;
        for nt_val in neural_tests {
            let nt_name = nt_val.as_str().ok_or_else(|| {
                Error::Invalid(format!(
                    "{mctx}: neural_gated_integration_tests entry must be string"
                ))
            })?;
            let test_rel = format!("{member_dir}/tests/{nt_name}.rs");
            let test_src = snapshot.read_utf8(&test_rel)?;
            require!(
                test_src.contains("#![cfg(feature = \"neural\")]"),
                "`{test_rel}` must contain `#![cfg(feature = \"neural\")]`"
            );
            total_neural_gated += 1;
        }

        let partial_neural = req_list(m, "partially_neural_gated_integration_tests", &mctx)?;
        for pnt_val in partial_neural {
            let pnt_name = pnt_val.as_str().ok_or_else(|| {
                Error::Invalid(format!(
                    "{mctx}: partially_neural_gated_integration_tests entry must be string"
                ))
            })?;
            let test_rel = format!("{member_dir}/tests/{pnt_name}.rs");
            let test_src = snapshot.read_utf8(&test_rel)?;
            require!(
                !test_src.contains("#![cfg(feature = \"neural\")]")
                    && test_src.contains("#[cfg(feature = \"neural\")]"),
                "`{test_rel}` must be partially gated with `#[cfg(feature = \"neural\")]` (not whole-file `#![cfg]`)"
            );
            total_partial_neural_gated += 1;
        }

        let testing_gated = req_list(m, "testing_gated_integration_tests", &mctx)?;
        for tg_val in testing_gated {
            let tg_name = tg_val.as_str().ok_or_else(|| {
                Error::Invalid(format!(
                    "{mctx}: testing_gated_integration_tests entry must be string"
                ))
            })?;
            let test_rel = format!("{member_dir}/tests/{tg_name}.rs");
            let test_src = snapshot.read_utf8(&test_rel)?;
            require!(
                test_src.contains("#![cfg(feature = \"testing\")]"),
                "`{test_rel}` must contain `#![cfg(feature = \"testing\")]`"
            );
        }
    }

    // Verify all committed *.schema.json files across the repository.
    let all_repo_schemas = snapshot.matching_paths("", ".schema.json");
    let schema_files = req_list(file_map, "schemas", "FILE_MAP.i01.json")?;
    require!(
        schema_files.len() == all_repo_schemas.len(),
        "schemas must list all {} committed *.schema.json files across the repository, got {}",
        all_repo_schemas.len(),
        schema_files.len()
    );
    snapshot.preload_paths(&all_repo_schemas)?;
    let mut recorded_schema_paths = BTreeSet::new();
    for (idx, sf_val) in schema_files.iter().enumerate() {
        let sctx = format!("schemas[{idx}]");
        let sf = dict_of(sf_val, &sctx)?;
        let spath = req_str(sf, "path", &sctx)?;
        validate_repo_relative_path(spath)?;
        let expected_sha = req_str(sf, "sha256", &sctx)?;
        let expected_bytes = req_u64(sf, "bytes", &sctx)?;
        let bytes = snapshot.read_bytes(spath)?;
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
        recorded_schema_paths.insert(spath.to_owned());
    }
    let expected_schema_set: BTreeSet<String> = all_repo_schemas.into_iter().collect();
    require!(
        recorded_schema_paths == expected_schema_set,
        "schemas path set does not match repository *.schema.json files"
    );

    Ok((
        members.len(),
        total_neural_gated,
        total_partial_neural_gated,
        schema_files.len(),
    ))
}

fn verify_file_inventories_and_pins(
    snapshot: &mut BaselineSnapshot<'_>,
    file_map: &Dict,
) -> Result<()> {
    let pins = req_list(file_map, "historical_baseline_pins", "FILE_MAP.i01.json")?;
    require!(
        pins.len() >= 4,
        "historical_baseline_pins must contain at least 4 pins, got {}",
        pins.len()
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
    require!(
        existing.len() == 134,
        "verified_existing_files must list all 134 existing baseline files from FILE_MAP.json, got {}",
        existing.len()
    );
    let existing_paths: Vec<String> = existing
        .iter()
        .filter_map(|v| v.as_dict()?.get("path")?.as_str().map(str::to_owned))
        .collect();
    snapshot.preload_paths(&existing_paths)?;
    for (idx, e_val) in existing.iter().enumerate() {
        let ectx = format!("verified_existing_files[{idx}]");
        let e = dict_of(e_val, &ectx)?;
        let path = req_str(e, "path", &ectx)?;
        validate_repo_relative_path(path)?;
        let exp_bytes = req_u64(e, "bytes", &ectx)?;
        let exp_sha = req_str(e, "sha256", &ectx)?;
        let data = snapshot.read_bytes(path)?;
        require!(
            u64::try_from(data.len()).ok() == Some(exp_bytes) && sha256_hex(&data) == exp_sha,
            "{ectx} (`{path}`): byte length or SHA-256 mismatch against baseline commit"
        );
    }

    let v4_files = req_list(file_map, "verified_v4_packet_files", "FILE_MAP.i01.json")?;
    require!(
        v4_files.len() == 40,
        "verified_v4_packet_files must list all 40 v4 packet files (PACKET_MANIFEST.v4.json + 39 payloads), got {}",
        v4_files.len()
    );
    let v4_paths: Vec<String> = v4_files
        .iter()
        .filter_map(|v| v.as_dict()?.get("path")?.as_str().map(str::to_owned))
        .collect();
    snapshot.preload_paths(&v4_paths)?;
    for (idx, vf_val) in v4_files.iter().enumerate() {
        let vctx = format!("verified_v4_packet_files[{idx}]");
        let vf = dict_of(vf_val, &vctx)?;
        let path = req_str(vf, "path", &vctx)?;
        validate_repo_relative_path(path)?;
        let exp_bytes = req_u64(vf, "bytes", &vctx)?;
        let exp_sha = req_str(vf, "sha256", &vctx)?;
        let data = snapshot.read_bytes(path)?;
        require!(
            u64::try_from(data.len()).ok() == Some(exp_bytes) && sha256_hex(&data) == exp_sha,
            "{vctx} (`{path}`): byte length or SHA-256 mismatch against baseline commit"
        );
    }

    let write_paths = req_list(file_map, "proposed_task_write_paths", "FILE_MAP.i01.json")?;
    require!(
        write_paths.len() == 133,
        "proposed_task_write_paths must list all 133 unique task write-path mappings, got {}",
        write_paths.len()
    );
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
        if exists {
            let sha = req_str(wp, "sha256_at_receiving_main", &wctx)?;
            require!(
                is_lower_hex_sha64(sha),
                "{wctx} (`{path}`): expected 64-hex sha256_at_receiving_main"
            );
        }
    }
    require!(
        covered_tasks.len() == 66,
        "proposed_task_write_paths must cover all 66 tasks (I01..I66), covered {}",
        covered_tasks.len()
    );

    Ok(())
}

fn verify_baseline_md_consistency(baseline_md: &str, file_map: &Dict) -> Result<()> {
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
            baseline_md.contains(commit_sha) && baseline_md.contains(tree_sha),
            "BASELINE.md missing historical pin commit `{commit_sha}` or tree `{tree_sha}`"
        );
    }

    for s_val in req_list(file_map, "symbols", "FILE_MAP.i01.json")? {
        let s = dict_of(s_val, "symbols")?;
        let sym = req_str(s, "symbol", "symbols")?;
        let path = req_str(s, "defining_path", "symbols")?;
        let line = req_u64(s, "defining_line", "symbols")?;
        let anchor = format!("{path}:{line}");
        require!(
            baseline_md.contains(sym) && baseline_md.contains(&anchor),
            "BASELINE.md missing symbol `{sym}` or anchor `{anchor}`"
        );
    }

    for sf_val in req_list(file_map, "schemas", "FILE_MAP.i01.json")? {
        let sf = dict_of(sf_val, "schemas")?;
        let spath = req_str(sf, "path", "schemas")?;
        let ssha = req_str(sf, "sha256", "schemas")?;
        require!(
            baseline_md.contains(spath) && baseline_md.contains(ssha),
            "BASELINE.md missing schema `{spath}` or its SHA-256 `{ssha}`"
        );
    }

    for f_val in req_list(file_map, "source_findings", "FILE_MAP.i01.json")? {
        let f = dict_of(f_val, "source_findings")?;
        let fid = req_str(f, "id", "source_findings")?;
        require!(
            baseline_md.contains(fid),
            "BASELINE.md missing finding ID `{fid}`"
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
            baseline_md.contains(rid) && baseline_md.contains(status),
            "BASELINE.md missing runtime ID `{rid}` or status `{status}`"
        );
    }

    Ok(())
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
        assert_eq!(summary.partially_neural_gated_test_count, 3);
        assert_eq!(summary.schema_file_count, 29);
        assert_eq!(summary.symbol_count, 4);
        assert_eq!(summary.held_gpu_route_count, 23);

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

        // Negative case 1b: recording an unverified profile ID not in feat_math::kernel::profile::profiles() must fail.
        let mut bogus_profile = load_i01_json();
        bogus_profile
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
            .get_mut("registered_profiles")
            .unwrap()
            .as_list_mut()
            .unwrap()[0]
            .as_dict_mut()
            .unwrap()
            .insert("id", Value::from("legacy_v1"));
        let err = verify_profile_identity_reconciliation(
            &root,
            bogus_profile
                .get("profile_and_model_reconciliation")
                .unwrap(),
        )
        .expect_err("mismatching profile ID against profile.rs registry must fail");
        assert!(
            err.to_string().contains("legacy-wrappers-v1"),
            "unexpected error: {err}"
        );

        // Negative case 1c: conflating 401-token native PolicyLM with 282-token CLOVER vocabulary must fail.
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

        // Negative case 1d: promoting unaccepted v2 warm-start as accepted serving must fail.
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

        // Negative case 2c: unknown finding in current_ci category coerced to confirmed must also fail.
        let mut coerced_ci_unknown = doc.clone();
        let entries = coerced_ci_unknown
            .as_dict_mut()
            .unwrap()
            .get_mut("source_findings")
            .unwrap()
            .as_list_mut()
            .unwrap();
        let unknown_entry = entries
            .iter_mut()
            .find(|e| e.get("classification").and_then(Value::as_str) == Some("unknown"))
            .expect("at least one unknown entry")
            .as_dict_mut()
            .unwrap();
        unknown_entry.insert("category", Value::from("current_ci"));
        unknown_entry.insert("classification", Value::from("confirmed"));
        let err = verify_finding_classifications(
            &root,
            coerced_ci_unknown.get("source_findings").unwrap(),
        )
        .expect_err("coercing unknown finding in current_ci to confirmed must fail");
        assert!(
            err.to_string().contains("unknown"),
            "unexpected error: {err}"
        );

        // Negative case 2d: claiming live_qualification_active=true (boolean or string) must fail.
        let mut false_live_bool = doc.clone();
        false_live_bool
            .as_dict_mut()
            .unwrap()
            .get_mut("source_findings")
            .unwrap()
            .as_list_mut()
            .unwrap()[0]
            .as_dict_mut()
            .unwrap()
            .insert("live_qualification_active", Value::Bool(true));
        let err =
            verify_finding_classifications(&root, false_live_bool.get("source_findings").unwrap())
                .expect_err("claiming boolean live_qualification_active=true must fail");
        assert!(
            err.to_string().contains("live_qualification_active"),
            "unexpected error: {err}"
        );

        let mut false_live_str = doc;
        false_live_str
            .as_dict_mut()
            .unwrap()
            .get_mut("source_findings")
            .unwrap()
            .as_list_mut()
            .unwrap()[0]
            .as_dict_mut()
            .unwrap()
            .insert("live_qualification_active", Value::from("true"));
        let err =
            verify_finding_classifications(&root, false_live_str.get("source_findings").unwrap())
                .expect_err("claiming string live_qualification_active=\"true\" must fail");
        assert!(
            err.to_string().contains("live_qualification_active"),
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
        assert_eq!(count, 7);

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
        let mut wrong_neural_count = doc.clone();
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
            err.to_string().contains("neural-gated"),
            "unexpected error: {err}"
        );

        // Negative case 3c: omitting one of the 6 indirect require_qualified_gpu_experiment routes in RT03 must fail.
        let mut missing_gpu_route = doc;
        missing_gpu_route
            .as_dict_mut()
            .unwrap()
            .get_mut("skipped_and_unavailable_runtimes")
            .unwrap()
            .as_list_mut()
            .unwrap()[2]
            .as_dict_mut()
            .unwrap()
            .get_mut("affected_targets")
            .unwrap()
            .as_list_mut()
            .unwrap()
            .pop();
        let err = verify_runtime_combinations(
            &root,
            missing_gpu_route
                .get("skipped_and_unavailable_runtimes")
                .unwrap(),
        )
        .expect_err("omitting a held GPU route in RT03 must fail");
        assert!(
            err.to_string().contains("held GPU routes"),
            "unexpected error: {err}"
        );
    }
}
