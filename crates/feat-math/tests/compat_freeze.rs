//! Compatibility freeze of the shipped math surfaces (workplan WP-0.2) and the shared scalar
//! boundary corpus (WP-0.3).
//!
//! One test per surface, so a failure names the surface, and every mismatch prints the first
//! differing records with their old and new values.
//!
//! | test | fixture under `tests/fixtures/` | frozen surface |
//! |---|---|---|
//! | `boundary_corpus_v1_is_well_formed_and_matches_its_definition` | `math_boundary/v1.json` | the scalar boundary inputs (WP-0.3) |
//! | `expression_values_match_the_frozen_fixture` | `compat_freeze/mathexpr_values.v1.json` | `MathExpr` value bits and error kinds: interpreter and compiled, row and map context |
//! | `stage_outputs_match_the_frozen_fixture` | `compat_freeze/stage_values.v1.json` | `map`/`derive` stage outputs and the predictions behind them |
//! | `emitted_artifact_bytes_match_the_frozen_fixture` | `compat_freeze/emitted_bytes.v1.json` | Rust, Python and manifest bytes under the default and the `jope emit` options; C++17 recipe and JavaScript bytes, or the emitter's refusal, under the default options; fitted C++17 bytes, or the recorded skip; the ops each recipe emitter accepts |
//! | `gained_support_fails_until_recorded_and_its_mode_changes_nothing_else` | an in-memory copy of `compat_freeze/emitted_bytes.v1.json` (the file is only read) | the rules for bytes where the fixture records a refusal |
//! | `fitted_cpp_revision_records_only_fitted_values_and_changes_nothing_else` | an in-memory copy of `compat_freeze/emitted_bytes.v1.json` (the file is only read) | the rules for a new revision of the fitted C++17 emitter |
//! | `serialized_models_match_the_frozen_fixture` | `compat_freeze/model_serde.v1.json` | `serde_json` bytes of fitted models, their predictions, and loading of stored bytes |
//! | `vocabulary_registry_and_lineage_pins_match` | `compat_freeze/frozen_pins.v1.json` | grammar assets, `grammar.rs`, jope-policy pins, registry and emitter versions |
//! | `evaluator_identity_inputs_match_the_recorded_identity` | `compat_freeze/evaluator_identity.v1.json` | the ADR-0004 evaluator identity inputs |
//! | `lineage_sources_match_the_recorded_sources` | `compat_freeze/lineage_sources.v1.json` | the sources jope-engine hashes into its teaching, corpus and query identities |
//! | `seeded_generator_streams_match_the_frozen_fixture` | `compat_freeze/generator_streams.v1.json` | seeded program generator streams and their vocabularies |
//! | `seeded_search_runs_match_the_frozen_fixture` | `compat_freeze/search_runs.v1.json` | whole seeded searches: the surrogate `evolve()` and the lite race, `rank_robust` off and on |
//! | `fixture_provenance_is_recorded` | `compat_freeze/meta.json` | generation commit, toolchain, host and libm |
//!
//! Modes. By default every test only compares. `JOPE_COMPAT_FREEZE_REGENERATE` selects one of four
//! regeneration modes; any other value panics.
//!
//! - `JOPE_COMPAT_FREEZE_REGENERATE=1` rewrites every fixture. Regenerating a behaviour surface is
//!   a rebaseline: it needs the PR-body line
//!   `compat-freeze: REBASELINE approved by Jepson <date> reason <...>`, and without it a red
//!   surface is a hard stop (revert the change, do not rebaseline).
//! - `JOPE_COMPAT_FREEZE_REGENERATE=identity` rewrites only the announced surfaces, which planned
//!   work moves: the ADR-0004 evaluator identity and the lineage sources. Every other test still
//!   compares, so the run also proves that nothing else moved. A change to a file in the ADR-0004
//!   list (or to `Cargo.lock`, which the identity hashes), or to a source jope-engine hashes into
//!   a lineage, announces the new identity in the PR body and rebaselines this way in the same
//!   change.
//! - `JOPE_COMPAT_FREEZE_REGENERATE=gained-support` records bytes that an emitter newly writes. In
//!   `emitted_bytes.v1.json` it replaces each C++17 recipe, JavaScript or fitted C++17 value that
//!   is recorded as unsupported, and for which the emitter now writes bytes, with those bytes. It
//!   changes nothing else: no recorded bytes, no refusal whose element kinds changed, no header,
//!   note or op list, and no other fixture. Every test still compares, and the mode writes nothing
//!   unless every other emitted-bytes record matches. A second run records nothing.
//! - `JOPE_COMPAT_FREEZE_REGENERATE=fitted-cpp-revision` records a new revision of the fitted
//!   C++17 emitter. In `emitted_bytes.v1.json` it replaces each fitted C++17 source and codec value
//!   that is recorded as bytes, and for which the emitter now writes other bytes, with those bytes,
//!   and sets that file's `generated_at_sha` to the commit it records. It changes nothing else: no
//!   refusal, error or skip (bytes for a value recorded as unsupported are still `gained-support`'s
//!   to record), no other value, no note or op list, and no other fixture. Every test still
//!   compares, and the mode writes nothing unless every other emitted-bytes record matches. A
//!   second run records nothing and leaves the header alone. New fitted bytes are a rebaseline of
//!   that surface: the pull request that records them carries the PR-body line
//!   `compat-freeze: REBASELINE approved by Jepson <date> reason <...>`.
//!
//! `=1`, `=identity` and `=fitted-cpp-revision` need `JOPE_COMPAT_FREEZE_SHA=<full commit id>`,
//! which names the commit the regeneration records; it must carry exactly the sources in the
//! working tree (checked with `git diff --quiet`). `=gained-support` records no commit id: the
//! fixture keeps its header.
//!
//! In compare mode the emitted-bytes test passes, and reports, two additive changes: a refusal that
//! narrows to a strict subset of its recorded element kinds (the fixture keeps the recorded
//! refusal), and an op that a recipe emitter newly accepts. Every other difference fails, among
//! them lost support, a refusal whose element kinds change otherwise, and bytes for a value
//! recorded as unsupported. The last names the program, the language and the `gained-support`
//! mode, so that bytes an emitter gains are pinned from the change that adds the support.
//!
//! Which bits depend on the platform libm. A surface that fails only on another host is a libm
//! difference only if it is one of these:
//! - MathExpr and stage values of the transcendental functions (exp, log, log1p, sin, cos, tanh,
//!   sigmoid, asinh, atan, erf and `^`);
//! - everything fitted: the stage predictions, the serialized models and their predictions, the
//!   fitted C++17 source and codec bytes, and the search runs. Their training rows come from
//!   `toy_data`, which calls `sin` and `Pcg32::gaussian` (`ln` and `cos`);
//! - the streams of `mutate_canonical_pipeline_v2` and of `mutate_canonical_pipeline_with_arm`
//!   with the parameter-jitter arm (their log-scale jitter calls `ln` and `exp`), and of
//!   `optimize_constants_v1` (it fits `sin` data).
//!
//! Free of libm: the arithmetic guards and clamps, the boundary corpus, the emitted recipe bytes
//! (Rust, Python, manifest, C++17 and JavaScript), the pins, and the other streams (the v1
//! wrappers, the other `with_arm` arms, `full_rust`, MathExpr `mutate_v1` and `crossover_v1`,
//! `PipelineSlots` and `advanced_v3`). `meta.json` records the host the fixtures were generated
//! on: target triple, glibc version, CPU model and FMA/AVX flags (glibc picks its transcendental
//! code by CPU feature), and hostname. Failure messages print the same for the failing run.
//!
//! The workplan asks for the corpus to parse in Python too. The stack is Rust-only (ADR-0006), so
//! the second, independent parser is `jq -e . tests/fixtures/math_boundary/v1.json`.

use feat_math::advanced_v3::ARTIFACT_SCHEMA;
use feat_math::dsl::{parse_dsl, Stage};
use feat_math::evolve::{
    crossover_canonical_pipelines, evolve, evolve_lite_race, mutate_canonical_pipeline,
    mutate_canonical_pipeline_v2, mutate_canonical_pipeline_with_arm, random_canonical_pipeline,
    CanonicalMutationArm, EvolveConfig, FitnessMode, GenStats, Observer, Scored,
    EXPANSION_STARTERS, EXPANSION_TRANSFORMS, HEAD_OPS, LITE_HEAD_BUDGETS,
};
use feat_math::full_rust::{self, EvolvedGenome, OperationKind, ParamDomain};
use feat_math::kernel::pipeline::PipelineSlots;
use feat_math::math_expr::{ExpressionScope, MathEvalError, MathExpr};
use feat_math::ops::{apply_op, fit_op};
use feat_math::registry::{operator_registry, registry_hash, REGISTRY_ENGINE, REGISTRY_VERSION};
use feat_math::transpile::{
    emit, emit_cpp17, emit_javascript, supports_cpp17, supports_javascript, EmitOptions, Emitted,
    EMITTER_VERSION,
};
use feat_math::{
    crossover_candidates, deterministic_seed_enumeration, emit_fitted_cpp17, fit,
    fit_legacy_numeric_bound, initial_population, mutate_candidate, next_generation, sanitize,
    GenomeIr, Model, Pcg32, V3Archive, V3EvolutionConfig, V3ScoredCandidate, CHAMPION_GENOME,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------------------------
// Modes, paths and fixture I/O
// ---------------------------------------------------------------------------------------------

const REGENERATE_ENV: &str = "JOPE_COMPAT_FREEZE_REGENERATE";
const SHA_ENV: &str = "JOPE_COMPAT_FREEZE_SHA";
/// The mode that records bytes for values recorded as unsupported (see the module doc).
const GAINED_SUPPORT_MODE: &str = "gained-support";
/// The mode that records a new revision of the fitted C++17 emitter (see the module doc).
const FITTED_CPP_REVISION_MODE: &str = "fitted-cpp-revision";
const REGENERATE_COMMAND: &str = "JOPE_COMPAT_FREEZE_REGENERATE=1 JOPE_COMPAT_FREEZE_SHA=<full commit id> cargo nextest run --locked -P pr -p feat-math --test compat_freeze";
const IDENTITY_REGENERATE_COMMAND: &str = "JOPE_COMPAT_FREEZE_REGENERATE=identity JOPE_COMPAT_FREEZE_SHA=<full commit id> cargo nextest run --locked -P pr -p feat-math --test compat_freeze";
const GAINED_SUPPORT_COMMAND: &str = "JOPE_COMPAT_FREEZE_REGENERATE=gained-support cargo nextest run --locked -P pr -p feat-math --test compat_freeze";
const REBASELINE_POLICY: &str = "A red compat_freeze is a hard stop: revert the change that moved a frozen surface. \
Rebaselining needs JOPE_COMPAT_FREEZE_REGENERATE=1 JOPE_COMPAT_FREEZE_SHA=<commit> and the PR-body line \
`compat-freeze: REBASELINE approved by Jepson <date> reason <...>` (workplan section 4.3).";

/// Every fixture file this binary owns, in the order `meta.json` lists them.
const FIXTURE_FILES: [&str; 9] = [
    "mathexpr_values.v1.json",
    "stage_values.v1.json",
    "emitted_bytes.v1.json",
    "model_serde.v1.json",
    "frozen_pins.v1.json",
    "evaluator_identity.v1.json",
    "generator_streams.v1.json",
    "search_runs.v1.json",
    "lineage_sources.v1.json",
];

/// Sources whose bytes the fixtures depend on. Regeneration refuses to record a commit id whose
/// copies of these differ from the working tree's.
const SOURCE_PATHSPECS: &[&str] = &[
    "crates/feat-math/src",
    "crates/feat-math/Cargo.toml",
    "crates/feat-math/assets",
    "crates/feat-math/cpp",
    "crates/feat-math/tests/fixtures/emitted",
    "crates/jope-engine/assets/grammar",
    "crates/jope-engine/src",
    "crates/jope-policy/src",
    "crates/jope-policy/release",
    "crates/jope-campaigns/src",
    "crates/jope-campaigns/Cargo.toml",
    "crates/jope-campaigns/known-evaluator-identities.json",
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
];

/// Which regeneration mode may rewrite a fixture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Surface {
    /// Behaviour surfaces and never-rebaselined pins: rewritten only by `REGENERATE=1`, an
    /// approved rebaseline.
    Frozen,
    /// Identities that planned changes move and announce: rewritten by `REGENERATE=1` and by
    /// `REGENERATE=identity`.
    Announced,
}

/// Whether this run rewrites the fixture of `surface`. `gained-support` and `fitted-cpp-revision`
/// rewrite none of them whole: only the emitted-bytes test writes in those modes (see
/// `recording_gained_support` and `recording_fitted_cpp_revision`).
fn regenerating(surface: Surface) -> bool {
    match std::env::var(REGENERATE_ENV).as_deref() {
        Err(_) | Ok("") | Ok("0") | Ok(GAINED_SUPPORT_MODE) | Ok(FITTED_CPP_REVISION_MODE) => false,
        Ok("1") => true,
        Ok("identity") => surface == Surface::Announced,
        Ok(other) => panic!(
            "{REGENERATE_ENV}={other:?} is not a mode: use 1 to rewrite every fixture (an approved \
             rebaseline), identity to rewrite only the announced identity fixtures, \
             {GAINED_SUPPORT_MODE} to record the bytes of emitted-bytes values recorded as \
             unsupported that an emitter now writes, or {FITTED_CPP_REVISION_MODE} to record the \
             fitted C++17 bytes of a new emitter revision; in the last three every other surface \
             still compares"
        ),
    }
}

/// Whether this run records gained emitter support (`JOPE_COMPAT_FREEZE_REGENERATE=gained-support`).
fn recording_gained_support() -> bool {
    matches!(
        std::env::var(REGENERATE_ENV).as_deref(),
        Ok(GAINED_SUPPORT_MODE)
    )
}

/// Whether this run records a new revision of the fitted C++17 emitter
/// (`JOPE_COMPAT_FREEZE_REGENERATE=fitted-cpp-revision`).
fn recording_fitted_cpp_revision() -> bool {
    matches!(
        std::env::var(REGENERATE_ENV).as_deref(),
        Ok(FITTED_CPP_REVISION_MODE)
    )
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(relative)
}

fn read_repo_file(relative: &str) -> Vec<u8> {
    std::fs::read(repo_root().join(relative)).unwrap_or_else(|error| panic!("{relative}: {error}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn hex_bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn decode_token(token: &str) -> String {
    match u64::from_str_radix(token, 16) {
        Ok(bits) if token.len() == 16 => format!("{:?}", f64::from_bits(bits)),
        _ => token.to_string(),
    }
}

fn text_identity(text: &str) -> String {
    format!("{} {}", text.len(), sha256_hex(text.as_bytes()))
}

fn is_hex(text: &str, len: usize) -> bool {
    text.len() == len
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The commit the regenerated fixtures are recorded against (regenerate mode only).
fn generation_sha() -> String {
    let sha = std::env::var(SHA_ENV).unwrap_or_else(|_| {
        panic!("{REGENERATE_ENV}=1 needs {SHA_ENV}=<full commit id> naming the sources the fixtures come from")
    });
    assert!(
        is_hex(&sha, 40),
        "{SHA_ENV} must be a full lowercase 40-hex commit id, got {sha:?}"
    );
    let status = Command::new("git")
        .arg("-C")
        .arg(repo_root())
        .args(["diff", "--quiet", &sha, "--"])
        .args(SOURCE_PATHSPECS)
        .status()
        .expect("git is required to regenerate the compat-freeze fixtures");
    assert!(
        status.success(),
        "the working tree's sources differ from {SHA_ENV}={sha} (git diff --quiet {sha} -- {}); \
         regenerate only against a commit that carries exactly these sources",
        SOURCE_PATHSPECS.join(" ")
    );
    sha
}

fn header_sha(surface: Surface) -> String {
    if regenerating(surface) {
        generation_sha()
    } else {
        String::new()
    }
}

fn write_atomically(path: &Path, text: &str) {
    let dir = path.parent().expect("fixture directory");
    std::fs::create_dir_all(dir).expect("create fixture directory");
    let name = path.file_name().expect("fixture name").to_string_lossy();
    let partial = dir.join(format!(".{name}.{}.partial", std::process::id()));
    std::fs::write(&partial, text).expect("write fixture");
    std::fs::rename(&partial, path).expect("install fixture");
}

/// The text of a fixture file, as every mode writes it.
fn fixture_text<T: Serialize>(value: &T) -> String {
    serde_json::to_string_pretty(value).expect("serialize fixture") + "\n"
}

/// Regenerate mode writes `actual` and returns `None`; compare mode returns the committed fixture.
fn committed_or_write<T: Serialize + DeserializeOwned>(
    surface: Surface,
    file: &str,
    actual: &T,
) -> Option<T> {
    let relative = format!("compat_freeze/{file}");
    let path = fixture_path(&relative);
    if regenerating(surface) {
        let text = fixture_text(actual);
        write_atomically(&path, &text);
        eprintln!("compat-freeze: wrote tests/fixtures/{relative}");
        return None;
    }
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "tests/fixtures/{relative} is missing ({error}); generate it with {REGENERATE_COMMAND}"
        )
    });
    Some(serde_json::from_str(&text).unwrap_or_else(|error| {
        panic!("tests/fixtures/{relative} does not parse as this surface's fixture ({error})")
    }))
}

/// What this build runs on, for failure messages about host-specific bits.
fn toolchain_note() -> String {
    let meta = std::fs::read_to_string(fixture_path("compat_freeze/meta.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Meta>(&text).ok());
    let rustc = command_line("rustc", &["-V"]);
    let libm = command_line("getconf", &["GNU_LIBC_VERSION"]);
    let (cpu, cpu_flags) = cpu_identity();
    let hostname = hostname();
    let this_run = format!("{rustc}, {libm}, CPU {cpu} ({cpu_flags}) on {hostname}");
    match meta {
        Some(meta) => format!(
            "fixtures: {} on {} with {}, CPU {} ({}) on {}, at {}; this run: {this_run}. glibc picks \
             its exp/log/pow/sin/cos code by CPU feature, so on another host only the libm-dependent \
             surfaces listed in the compat_freeze.rs module doc may differ.",
            meta.rustc,
            meta.host,
            meta.libm,
            meta.cpu,
            meta.cpu_flags,
            meta.hostname,
            meta.generated_at_sha
        ),
        None => format!("this run: {this_run}"),
    }
}

/// The CPU model and which of the FMA/AVX2/AVX-512 features it has, from /proc/cpuinfo.
fn cpu_identity() -> (String, String) {
    let text = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let field = |name: &str| {
        text.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim() == name).then(|| value.trim().to_string())
        })
    };
    let model = field("model name").unwrap_or_else(|| "unknown".to_string());
    let flags = field("flags").unwrap_or_default();
    let present: Vec<&str> = ["fma", "avx2", "avx512f"]
        .into_iter()
        .filter(|feature| flags.split_whitespace().any(|flag| flag == *feature))
        .collect();
    let features = if flags.is_empty() {
        "flags unknown".to_string()
    } else if present.is_empty() {
        "no fma, avx2 or avx512f".to_string()
    } else {
        present.join(" ")
    };
    (model, features)
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|name| name.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

fn command_line(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|| format!("{program} unavailable"))
}

/// Compare ordered `(key, value)` records and fail with the first differences.
fn assert_records(
    surface: &str,
    fixture: &[(String, String)],
    actual: &[(String, String)],
    advice: &str,
) {
    let problems = record_problems(surface, fixture, actual);
    if !problems.is_empty() {
        panic!("{}", drift_message(surface, &problems, advice));
    }
}

/// The first twelve items of a failure list, one per indented line.
fn first_lines(items: &[String]) -> String {
    items
        .iter()
        .take(12)
        .cloned()
        .collect::<Vec<_>>()
        .join("\n  ")
}

/// The failure message for records of `surface` that differ from the fixture.
fn drift_message(surface: &str, problems: &[String], advice: &str) -> String {
    format!(
        "compat-freeze: {surface} drifted ({} differing records; first {} below)\n  {}\n{}\n{advice}",
        problems.len(),
        problems.len().min(12),
        first_lines(problems),
        toolchain_note()
    )
}

/// The differences between ordered `(key, value)` records: removed, changed and added keys, then a
/// changed order.
fn record_problems(
    surface: &str,
    fixture: &[(String, String)],
    actual: &[(String, String)],
) -> Vec<String> {
    let old: BTreeMap<&str, &str> = fixture
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let new: BTreeMap<&str, &str> = actual
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    assert_eq!(
        old.len(),
        fixture.len(),
        "{surface}: duplicate fixture keys"
    );
    assert_eq!(new.len(), actual.len(), "{surface}: duplicate actual keys");
    let mut problems = Vec::new();
    for (key, value) in fixture {
        match new.get(key.as_str()) {
            None => problems.push(format!("removed {key}: fixture {value}")),
            Some(now) if *now != value => problems.push(format!(
                "{key}:\n      fixture {value}\n      now     {now}"
            )),
            _ => {}
        }
    }
    for (key, value) in actual {
        if !old.contains_key(key.as_str()) {
            problems.push(format!("added {key}: {value}"));
        }
    }
    if problems.is_empty()
        && fixture
            .iter()
            .map(|record| &record.0)
            .ne(actual.iter().map(|record| &record.0))
    {
        problems.push("the record order changed".to_string());
    }
    problems
}

fn record(key: impl Into<String>, value: impl Into<String>) -> (String, String) {
    (key.into(), value.into())
}

// ---------------------------------------------------------------------------------------------
// Shared data: the boundary corpus (WP-0.3), the toy data and the program set
// ---------------------------------------------------------------------------------------------

const CORPUS_FILE: &str = "math_boundary/v1.json";
const CORPUS_SCHEMA: &str = "feat-math.math-boundary.v1";
const CORPUS_NOTE: &str = "Scalar boundary inputs shared by the Rust tests and the emitted-artifact harnesses (workplan WP-0.3). bits is the IEEE-754 binary64 pattern; value_repr is Rust's shortest round-trip form and parses back to exactly bits, except that NaN payloads live only in bits. Neighbours are exact: bits plus or minus one. A value that the workplan lists in two groups appears once, under the first of them. Generated by crates/feat-math/tests/compat_freeze.rs, which also proves the file equals its definition.";
const CORPUS_GROUPS: [&str; 9] = [
    "specials",
    "division_guard",
    "log_guard",
    "log1p_guard",
    "unity_small_integers",
    "rounding",
    "exp_sigmoid_clamps",
    "magnitude_clamp",
    "trig_erf_mid",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusEntry {
    name: String,
    group: String,
    bits: String,
    value_repr: String,
}

impl CorpusEntry {
    fn value(&self) -> f64 {
        f64::from_bits(corpus_bits(&self.bits))
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusFile {
    schema: String,
    note: String,
    groups: Vec<String>,
    entries: Vec<CorpusEntry>,
}

fn corpus_bits(text: &str) -> u64 {
    let digits = text
        .strip_prefix("0x")
        .unwrap_or_else(|| panic!("corpus bits {text:?} must start with 0x"));
    assert!(
        is_hex(digits, 16),
        "corpus bits {text:?} must be 16 lowercase hex digits"
    );
    u64::from_str_radix(digits, 16).expect("hex bits")
}

/// The next representable value toward zero (finite, nonzero input).
fn toward_zero(value: f64) -> f64 {
    f64::from_bits(value.to_bits() - 1)
}

/// The next representable value away from zero (finite input).
fn away_from_zero(value: f64) -> f64 {
    f64::from_bits(value.to_bits() + 1)
}

fn parse_f64(text: &str) -> f64 {
    text.parse().expect("decimal literal")
}

fn corpus_entry(group: &str, name: &str, bits: u64) -> CorpusEntry {
    CorpusEntry {
        name: name.to_string(),
        group: group.to_string(),
        bits: format!("0x{bits:016x}"),
        value_repr: format!("{:?}", f64::from_bits(bits)),
    }
}

/// The corpus, with every neighbour computed by exact bit arithmetic.
fn corpus_definition() -> Vec<CorpusEntry> {
    let mut entries = Vec::new();
    let mut add = |group: &str, name: &str, value: f64| {
        entries.push(corpus_entry(group, name, value.to_bits()));
    };
    let g = "specials";
    add(g, "pos_zero", 0.0);
    add(g, "neg_zero", -0.0);
    add(g, "pos_inf", f64::INFINITY);
    add(g, "neg_inf", f64::NEG_INFINITY);
    add(g, "nan_quiet", f64::from_bits(0x7ff8_0000_0000_0000));
    add(g, "nan_payload", f64::from_bits(0x7ff8_0000_0000_0001));
    let g = "division_guard";
    for (name, v) in [("div_guard_pos", 1e-12), ("div_guard_neg", -1e-12)] {
        add(g, name, v);
        add(g, &format!("{name}_toward_zero"), toward_zero(v));
        add(g, &format!("{name}_away_from_zero"), away_from_zero(v));
    }
    let g = "log_guard";
    add(g, "log_guard", 1e-300);
    add(g, "log_guard_toward_zero", toward_zero(1e-300));
    add(g, "log_guard_away_from_zero", away_from_zero(1e-300));
    add(g, "min_subnormal", f64::from_bits(1));
    add(g, "min_normal", f64::MIN_POSITIVE);
    let g = "log1p_guard";
    add(g, "neg_one", -1.0);
    add(g, "neg_one_toward_zero", toward_zero(-1.0));
    add(
        g,
        "neg_one_two_ulps_toward_zero",
        toward_zero(toward_zero(-1.0)),
    );
    add(g, "neg_one_away_from_zero", away_from_zero(-1.0));
    add(g, "neg_half", -0.5);
    add(g, "pos_half", 0.5);
    let g = "unity_small_integers";
    add(g, "pos_one", 1.0);
    add(g, "pos_two", 2.0);
    add(g, "neg_two", -2.0);
    add(g, "pos_three", 3.0);
    add(g, "neg_three", -3.0);
    let g = "rounding";
    add(g, "pos_one_and_half", 1.5);
    add(g, "neg_one_and_half", -1.5);
    add(g, "pos_two_and_half", 2.5);
    add(g, "below_half", toward_zero(0.5));
    add(g, "two_pow_52_plus_one", 4_503_599_627_370_497.0);
    let g = "exp_sigmoid_clamps";
    add(g, "pos_27", 27.0);
    add(g, "neg_27", -27.0);
    add(g, "ln_1e12", parse_f64("27.631021115928547"));
    add(g, "pos_27_64", 27.64);
    add(g, "pos_28", 28.0);
    add(g, "neg_28", -28.0);
    add(g, "pos_30", 30.0);
    add(g, "neg_30", -30.0);
    add(g, "pos_31", 31.0);
    add(g, "neg_31", -31.0);
    let g = "magnitude_clamp";
    for (name, v) in [("max_abs", 1e12), ("neg_max_abs", -1e12)] {
        add(g, name, v);
        add(g, &format!("{name}_toward_zero"), toward_zero(v));
        add(g, &format!("{name}_away_from_zero"), away_from_zero(v));
    }
    let sqrt_max = parse_f64("1.3407807929942596e154");
    for (name, v) in [
        ("pos_1e154", 1e154),
        ("neg_1e154", -1e154),
        ("sqrt_max", sqrt_max),
        ("neg_sqrt_max", -sqrt_max),
        ("pos_1_34e154", 1.34e154),
        ("neg_1_34e154", -1.34e154),
        ("pos_1_35e154", 1.35e154),
        ("neg_1_35e154", -1.35e154),
        ("pos_1e155", 1e155),
        ("neg_1e155", -1e155),
        ("max_finite", f64::MAX),
        ("neg_max_finite", f64::MIN),
    ] {
        add(g, name, v);
    }
    let g = "trig_erf_mid";
    add(g, "tenth", 0.1);
    add(g, "tiny", 1e-8);
    add(g, "six", 6.0);
    add(g, "pi", std::f64::consts::PI);
    add(g, "neg_pi", -std::f64::consts::PI);
    entries
}

/// Every value WP-0.3 lists, by group (a value listed twice is stored once, see `CORPUS_NOTE`).
fn corpus_requirements() -> Vec<(&'static str, f64)> {
    let mut required = vec![
        ("specials", 0.0),
        ("specials", -0.0),
        ("specials", f64::INFINITY),
        ("specials", f64::NEG_INFINITY),
        ("specials", f64::from_bits(0x7ff8_0000_0000_0000)),
        ("specials", f64::from_bits(0x7ff8_0000_0000_0001)),
    ];
    for v in [1e-12, -1e-12] {
        required.extend([
            ("division_guard", v),
            ("division_guard", toward_zero(v)),
            ("division_guard", away_from_zero(v)),
        ]);
    }
    required.extend([
        ("log_guard", 1e-300),
        ("log_guard", toward_zero(1e-300)),
        ("log_guard", away_from_zero(1e-300)),
        ("log_guard", f64::from_bits(1)),
        ("log_guard", f64::MIN_POSITIVE),
        ("log1p_guard", -1.0),
        ("log1p_guard", -1.0 + f64::EPSILON / 2.0),
        ("log1p_guard", -1.0 - f64::EPSILON),
        ("log1p_guard", -0.5),
        ("log1p_guard", 0.5),
    ]);
    for v in [1.0, -1.0, 2.0, -2.0, 3.0, -3.0] {
        required.push(("unity_small_integers", v));
    }
    for v in [
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        toward_zero(0.5),
        4_503_599_627_370_497.0,
    ] {
        required.push(("rounding", v));
    }
    for v in [
        27.0,
        -27.0,
        parse_f64("27.631021115928547"),
        27.64,
        28.0,
        -28.0,
        30.0,
        -30.0,
        31.0,
        -31.0,
    ] {
        required.push(("exp_sigmoid_clamps", v));
    }
    for v in [
        1e12,
        -1e12,
        toward_zero(1e12),
        away_from_zero(1e12),
        toward_zero(-1e12),
        away_from_zero(-1e12),
        1e154,
        -1e154,
        parse_f64("1.3407807929942596e154"),
        -parse_f64("1.3407807929942596e154"),
        1e155,
        -1e155,
        f64::MAX,
        f64::MIN,
    ] {
        required.push(("magnitude_clamp", v));
    }
    for v in [0.1, 1e-8, 6.0, std::f64::consts::PI, -std::f64::consts::PI] {
        required.push(("trig_erf_mid", v));
    }
    required
}

fn render_corpus(entries: &[CorpusEntry]) -> String {
    let quote = |text: &str| serde_json::to_string(text).expect("quote");
    let mut out = String::from("{\n");
    out.push_str(&format!("  \"schema\": {},\n", quote(CORPUS_SCHEMA)));
    out.push_str(&format!("  \"note\": {},\n", quote(CORPUS_NOTE)));
    out.push_str(&format!(
        "  \"groups\": [{}],\n",
        CORPUS_GROUPS
            .iter()
            .map(|group| quote(group))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str("  \"entries\": [\n");
    for (i, entry) in entries.iter().enumerate() {
        out.push_str(&format!(
            "    {{\"name\": {}, \"group\": {}, \"bits\": {}, \"value_repr\": {}}}{}\n",
            quote(&entry.name),
            quote(&entry.group),
            quote(&entry.bits),
            quote(&entry.value_repr),
            if i + 1 < entries.len() { "," } else { "" }
        ));
    }
    out.push_str("  ]\n}\n");
    out
}

/// Deterministic regression data, copied from `tests/golden.rs` (that file is hash-pinned by
/// `legacy_v2_freeze.rs`, so it is copied rather than included).
fn toy_data(n: usize, cols: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Pcg32::new(seed, 17);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..n {
        let row: Vec<f64> = (0..cols).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        let target = 1.7 * row[0] - 0.9 * row[1]
            + 0.5 * (3.0 * row[2]).sin()
            + 0.3 * row[3] * row[4]
            + 0.1 * rng.gaussian();
        x.push(row);
        y.push(target);
    }
    (x, y)
}

type Rows = Vec<Vec<f64>>;

/// Training rows (the first 64) and held-out rows (the last 16) for the fitted surfaces.
fn fitted_surface_data() -> (Rows, Vec<f64>, Rows) {
    let (x, y) = toy_data(80, 5, 42);
    (x[..64].to_vec(), y[..64].to_vec(), x[64..].to_vec())
}

const FITTED_DATA_NOTE: &str = "toy_data(80 rows, 5 columns, seed 42) as in tests/golden.rs: the first 64 rows train, the last 16 are predicted";

/// Programs with expressions and every expansion op, beside the champion and the ip_scrub set.
const NAMED_PROGRAMS: &[(&str, &str)] = &[
    ("golden_map_derive_ridge", "(pipe X (map expr=\"sigmoid(x) + 0.1*col(0)\" append=true) (derive expr=\"row_mean + row_std - c1/(c0 + 3)\" append=true) (ridge))"),
    ("golden_map_pow_elasticnet", "(pipe X (map expr=\"x^2 - min(x, c2) + tanh(c3)\") (elasticnet alpha=0.005 l1_ratio=0.7))"),
    ("golden_step_map_expr", "(pipe X (map expr=\"step(col(0), 0.0) + 0.5*step(row_mean, c1)\" append=true) (ridge))"),
    ("golden_scalar_batch_map_ridge", "(pipe X (map expr=\"exp(clip(col(0),-2,2)) + log(abs(col(1)) + 1.0) + asinh(col(2)) + atan(col(3))\") (ridge))"),
    ("golden_scalar_batch2_derive_ridge", "(pipe X (derive append=true expr=\"erf(col(0)) + sign(col(1)) * floor(col(2)) + round(col(3))\") (ridge))"),
    ("golden_symbolic_affine", "(sym affine \"sigmoid(col(0)) + 0.5*col(1) - log1p(abs(col(2))) + max(c3, c4^2)\")"),
    ("golden_symbolic_raw", "(sym raw \"clip(row_mean + 0.3*row_std - c0/(c1 + 5), -2, 2)\")"),
    ("golden_step_symbolic", "(sym affine \"step(c0, 0.0) + step(c1, c2) - 0.5\")"),
    ("golden_symbolic_scalar_batch", "(sym affine \"erf(c0) + asinh(c1) - exp(clip(c2, -1, 1)) + sign(c3) * round(c4)\")"),
    ("expressions_1", "(pipe X (map expr=\"abs(x) + sqrt(abs(c0)) - log1p(abs(x))\" append=true) (derive expr=\"sin(c0) + cos(c1) * tanh(c2)\" append=true) (ridge))"),
    ("expressions_2", "(pipe X (map expr=\"sigmoid(x) - min(x, c1) + max(x, c2)\") (derive expr=\"clip(c0, -1, 1) + step(c1, 0.5)\" append=true) (ridge))"),
    ("expressions_3", "(pipe X (map expr=\"exp(clip(x, -3, 3)) + log(abs(x) + 1)\" append=true) (derive expr=\"asinh(c0) + atan(c1) - erf(c2)\" append=true) (ridge))"),
    ("expressions_4", "(pipe X (map expr=\"sign(x) * floor(x) + round(x)\") (derive expr=\"row_sum - row_l1 + row_l2\" append=true) (ridge))"),
    ("expressions_5", "(pipe X (map expr=\"x - row_mean + row_std\" append=true) (derive expr=\"row_min * row_max + c0 / (c1 + 3)\" append=true) (ridge))"),
    ("expressions_6", "(pipe X (map expr=\"x^2 - (-x) * 0.5 + x / (c0 + 2)\") (derive expr=\"c3^3 - c4*c4*c4 + c2^0.5\" append=true) (elasticnet alpha=0.005 l1_ratio=0.7))"),
    ("expansion_symlog", "(pipe X (symlog c=0.7) (ridge))"),
    ("expansion_select_corr", "(pipe X (select_corr k=3 by=corr) (ridge))"),
    ("expansion_impute", "(pipe X (impute strategy=median add_indicator=true) (standardize) (ridge))"),
    ("expansion_pair_ratio", "(pipe X (pair_ratio topk=4 append=true) (ridge))"),
    ("expansion_spline", "(pipe X (spline knots=4) (ridge))"),
    // With the ip_scrub pipelines these three emit a C++17 recipe and JavaScript for every op
    // those emitters support, so every op body and shared helper they write is pinned.
    ("recipe_random_features", "(pipe X (rff_cos n=8) (rff_sin n=8) (rand_proj k=4) (landmark_dist k=4) (ridge))"),
    ("recipe_row_features", "(pipe X (oblique_hinge n=4) (row_stats) (pair_prod) (target_bin_encode) (sigmoid) (ridge))"),
    ("recipe_indicator_features", "(pipe X (step t=0.5 append=true) (norm_row) (row_diff_stats) (elasticnet))"),
    // `logistic` is an alias of the ridge head, and the recipe emitters accept it.
    ("recipe_logistic_alias", "(pipe X (standardize) (logistic))"),
    // The target stage in each kind (tests/golden.rs programs).
    ("target_rank_gauss", "(pipe X (target kind=rank_gauss) (ridge))"),
    ("target_log1p", "(pipe X (target kind=log1p) (ridge lambda_frac=0.02))"),
    ("target_winsor", "(pipe X (target kind=winsor q=0.05) (standardize) (ridge))"),
    ("target_zscore", "(pipe X (target kind=zscore) (standardize) (ridge))"),
    // Data paths the toy rows never reach on their own: step makes exact zeros, which impute
    // treats as missing (fill, indicator columns, the mean and const strategies).
    ("step_impute_mean", "(pipe X (step t=0.0) (impute strategy=mean add_indicator=true) (ridge))"),
    ("step_impute_const", "(pipe X (step t=0.0 append=true) (impute strategy=const value=0.5) (ridge))"),
    ("select_corr_by_var", "(pipe X (select_corr k=3 by=var) (ridge))"),
    // 96 features on 64 training rows: ridge_gcv takes its dual (n < p) route.
    ("ridge_gcv_dual_route", "(pipe X (rff_cos n=96 seed=3) (ridge_gcv))"),
    // One column leaves pair_ratio no pair, so its rows are empty and padded to [0.0].
    ("empty_stage_output", "(pipe X (select_corr k=1) (pair_ratio topk=4) (ridge))"),
    // `x` outside a map: the emitters write its constant 0.0 body.
    ("derive_x_body", "(pipe X (derive expr=\"x\" append=true) (ridge))"),
    ("symbolic_x", "(sym raw \"x\")"),
];

/// `tests/ip_scrub.rs` PIPELINES and SYMBOLIC, verbatim.
const IP_SCRUB_PIPELINES: &[&str] = &[
    "(pipe X (portfolio preset=core))",
    "(pipe X (standardize) (rank) (blom) (quantile_bins) (bin_onehot bins=3) (ridge))",
    "(pipe X (hinge) (winsor_clip) (cheb) (poly_interact append=true) (elasticnet))",
    "(pipe X (rff_cos n=8) (rff_sin n=8) (rand_proj k=4) (landmark_dist k=4) (histgbm rounds=5))",
    "(pipe X (oblique_hinge n=4) (row_stats) (pair_prod) (target_bin_encode) (ensemble rounds=5))",
    "(pipe X (sigmoid) (norm_row) (polar) (lag k=1) (roll k=2) (fft_mag k=2) (ridge))",
    "(pipe X (row_spectral) (row_diff_stats) (svd_project k=2 iters=5) (nystrom k=4) (ridge))",
    "(pipe X (map expr=\"clip(x, -1, 1)\") (derive expr=\"row_l1 + row_l2 + row_min + row_max\") (ridge))",
    "(pipe X (histgbm2 leaves=15 lr=0.08 bags=2))",
    "(pipe X (standardize) (krr m=200))",
    "(pipe X (gpr m=200 steps=10))",
    "(pipe X (gpr kernel=rq+linear m=200 steps=10))",
    "(pipe X (gpr kernel=rbf*periodic m=200 steps=10))",
    "(pipe X (ksmooth k=16 gamma=2.0))",
    "(pipe X (standardize) (cheb degree=3 append=true) (ridge_gcv))",
    "(pipe X (stack))",
];
const IP_SCRUB_SYMBOLIC: &[&str] = &[
    "(sym affine \"tanh(c0) * sqrt(abs(c1)) + min(c2, 1) / max(c3, 1) + sin(c4) + cos(c0)\")",
    "(sym raw \"sigmoid(c0) + 0.5*c1\")",
];

/// The program set G of WP-0.2 step 4, named and in a fixed order.
fn program_set() -> Vec<(String, String)> {
    let mut set = vec![("champion".to_string(), CHAMPION_GENOME.trim().to_string())];
    set.extend(
        NAMED_PROGRAMS
            .iter()
            .map(|(name, source)| (name.to_string(), source.to_string())),
    );
    set.extend(
        IP_SCRUB_PIPELINES
            .iter()
            .enumerate()
            .map(|(i, source)| (format!("ip_scrub_pipeline_{i:02}"), source.to_string())),
    );
    set.extend(
        IP_SCRUB_SYMBOLIC
            .iter()
            .enumerate()
            .map(|(i, source)| (format!("ip_scrub_symbolic_{i}"), source.to_string())),
    );
    let portable = std::fs::read_to_string(fixture_path("emitted/portable-4lang.dsl"))
        .expect("tests/fixtures/emitted/portable-4lang.dsl");
    set.push(("portable_4lang".to_string(), portable.trim().to_string()));
    set
}

fn parse_program(name: &str, source: &str) -> GenomeIr {
    GenomeIr::parse(source)
        .unwrap_or_else(|error| panic!("{name}: {source} no longer parses: {error}"))
}

/// Portfolio fits carry wall-clock `fit_ms` diagnostics (portfolio.rs) and take minutes in a
/// debug build, so their fitted surfaces are recorded as skipped; their emission is frozen.
fn fit_skip_reason(source: &str) -> Option<&'static str> {
    source.contains("(portfolio").then_some(
        "skipped: portfolio diagnostics carry wall-clock fit_ms and a debug-build fit takes minutes",
    )
}

// ---------------------------------------------------------------------------------------------
// WP-0.3: boundary corpus
// ---------------------------------------------------------------------------------------------

#[test]
fn boundary_corpus_v1_is_well_formed_and_matches_its_definition() {
    let definition = corpus_definition();
    let path = fixture_path(CORPUS_FILE);
    if regenerating(Surface::Frozen) {
        let _ = header_sha(Surface::Frozen);
        write_atomically(&path, &render_corpus(&definition));
        eprintln!("compat-freeze: wrote tests/fixtures/{CORPUS_FILE}");
        return;
    }
    let text = std::fs::read_to_string(&path).expect("tests/fixtures/math_boundary/v1.json");
    let file: CorpusFile = serde_json::from_str(&text).expect("the corpus parses with serde_json");
    assert_eq!(file.schema, CORPUS_SCHEMA);
    assert_eq!(file.groups, CORPUS_GROUPS);

    let mut names = BTreeSet::new();
    let mut bits = BTreeSet::new();
    for entry in &file.entries {
        assert!(
            names.insert(entry.name.as_str()),
            "duplicate corpus name {}",
            entry.name
        );
        assert!(
            bits.insert(corpus_bits(&entry.bits)),
            "duplicate corpus value {} ({})",
            entry.bits,
            entry.name
        );
        assert!(
            CORPUS_GROUPS.contains(&entry.group.as_str()),
            "{}: unknown group {}",
            entry.name,
            entry.group
        );
        let value = entry.value();
        let parsed: f64 = entry.value_repr.parse().unwrap_or_else(|_| {
            panic!(
                "{}: value_repr {:?} does not parse",
                entry.name, entry.value_repr
            )
        });
        if value.is_nan() {
            assert!(
                parsed.is_nan(),
                "{}: value_repr of a NaN must be NaN",
                entry.name
            );
        } else {
            assert_eq!(
                parsed.to_bits(),
                value.to_bits(),
                "{}: value_repr {} does not round-trip to {}",
                entry.name,
                entry.value_repr,
                entry.bits
            );
        }
    }
    for group in CORPUS_GROUPS {
        assert!(
            file.entries.iter().any(|entry| entry.group == group),
            "group {group} is empty"
        );
    }
    for (group, value) in corpus_requirements() {
        assert!(
            bits.contains(&value.to_bits()),
            "WP-0.3 {group} value {value:?} (0x{:016x}) is missing",
            value.to_bits()
        );
    }
    // Neighbours are one ulp away: bits - 1 toward zero, bits + 1 away from zero, either sign.
    let by_name: BTreeMap<&str, u64> = file
        .entries
        .iter()
        .map(|entry| (entry.name.as_str(), corpus_bits(&entry.bits)))
        .collect();
    for base in [
        "div_guard_pos",
        "div_guard_neg",
        "log_guard",
        "neg_one",
        "max_abs",
        "neg_max_abs",
    ] {
        let bits = by_name[base];
        assert_eq!(by_name[format!("{base}_toward_zero").as_str()], bits - 1);
        assert_eq!(by_name[format!("{base}_away_from_zero").as_str()], bits + 1);
    }
    assert_eq!(
        file.entries, definition,
        "the committed corpus differs from its definition in this file; a corpus change is a new version (v2.json), not an edit"
    );
    assert_eq!(
        text,
        render_corpus(&definition),
        "the committed corpus is not in canonical form; regenerate it with {REGENERATE_COMMAND}"
    );
}

// ---------------------------------------------------------------------------------------------
// WP-0.2 (a): MathExpr values on the corpus rows
// ---------------------------------------------------------------------------------------------

const ERR: &str = "ERR";
const SAME_AS_INTERPRETER_ROW: &str = "=interpreter_row";
const SAME_AS_INTERPRETER_MAP: &str = "=interpreter_map with every ERR token as 0000000000000000";
const WIDE_ROW: [f64; 8] = [0.1, -2.0, 3.0, 0.5, -1.5, 2.5, 1e-12, 6.0];

/// Row-context expressions: every function alone on c0 (with its arity), every variable, every
/// binary operator and negation, the workplan combinations, and the golden and ip_scrub `derive`
/// and closed-form expressions.
const ROW_EXPRESSIONS: &[&str] = &[
    "abs(col(0))",
    "sqrt(col(0))",
    "log1p(col(0))",
    "sin(col(0))",
    "cos(col(0))",
    "tanh(col(0))",
    "sigmoid(col(0))",
    "min(col(0), col(1))",
    "max(col(0), col(1))",
    "clip(col(0), col(1), col(2))",
    "step(col(0), 0)",
    "step(col(0), col(1))",
    "exp(col(0))",
    "log(col(0))",
    "asinh(col(0))",
    "atan(col(0))",
    "erf(col(0))",
    "sign(col(0))",
    "floor(col(0))",
    "round(col(0))",
    "col(0)",
    "c1",
    "row_mean",
    "row_sum",
    "row_l1",
    "row_l2",
    "row_min",
    "row_max",
    "row_std",
    "col(7)",
    "col(0) + col(1)",
    "col(0) - col(1)",
    "col(0) * col(1)",
    "col(0)/col(1)",
    "col(0)^col(1)",
    "col(0)^0.5",
    "col(0)^-9",
    "col(0)^9",
    "2^col(0)",
    "-col(0)",
    "(col(0)*col(0))*col(0)",
    "asinh(col(0)*col(0))",
    "exp(col(0))-exp(col(1))",
    "1/col(0)",
    "col(1)/col(0)",
    "log(col(0)) + log1p(col(0)) + sqrt(col(0))",
    "clip(1, col(0) - col(0), col(0) - col(0))",
    "clip(1,(-1)^0.5,(-1)^0.5)",
    "min(clip(1,(-1)^0.5,(-1)^0.5),7)+1",
    "clip(1,(-1)^0.5,3)",
    "-2^2",
    "2^100",
    "0.5",
    "row_mean + row_std - c1/(c0 + 3)",
    "row_l1 + row_l2 + row_min + row_max",
    "erf(col(0)) + sign(col(1)) * floor(col(2)) + round(col(3))",
    "sigmoid(col(0)) + 0.5*col(1) - log1p(abs(col(2))) + max(c3, c4^2)",
    "clip(row_mean + 0.3*row_std - c0/(c1 + 5), -2, 2)",
    "tanh(c0) * sqrt(abs(c1)) + min(c2, 1) / max(c3, 1) + sin(c4) + cos(c0)",
    "sigmoid(c0) + 0.5*c1",
    "step(c0, 0.0) + step(c1, c2) - 0.5",
    "erf(c0) + asinh(c1) - exp(clip(c2, -1, 1)) + sign(c3) * round(c4)",
];

/// Map-context expressions (`x` is each element): the golden and ip_scrub `map` expressions and
/// elementwise forms of every function.
const MAP_EXPRESSIONS: &[&str] = &[
    "x",
    "-x",
    "x/col(0)",
    "sigmoid(x) + 0.1*col(0)",
    "x^2 - min(x, c2) + tanh(c3)",
    "step(col(0), 0.0) + 0.5*step(row_mean, c1)",
    "exp(clip(col(0),-2,2)) + log(abs(col(1)) + 1.0) + asinh(col(2)) + atan(col(3))",
    "clip(x, -1, 1)",
    "x - row_mean + row_std",
    "x-row_mean+row_std-row_sum",
    "clip(x,c0,c1)",
    "sign(x)+floor(x)+round(x)",
    "erf(x)+asinh(x)+atan(x)+exp(x)+log(x)",
    "abs(x) + sqrt(x) + log1p(x)",
    "sin(x) * cos(x) - tanh(x)",
    "min(x, row_min) + max(x, row_max)",
    "x / row_l2",
    "c7 + x",
    "x^x",
    "step(x, row_mean)",
];

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MathExprValues {
    schema: String,
    generated_at_sha: String,
    encoding: String,
    rows: Vec<RowRecord>,
    expressions: Vec<ExpressionRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RowRecord {
    label: String,
    values: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpressionRecord {
    source: String,
    context: String,
    interpreter_row: String,
    compiled_row: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    interpreter_map: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compiled_map: Option<String>,
}

/// Full token lists of one expression (markers expanded).
struct ExpressionValues {
    source: String,
    context: &'static str,
    paths: Vec<(&'static str, Vec<String>)>,
}

fn corpus_rows() -> Vec<(String, Vec<f64>)> {
    let mut rows = Vec::new();
    for entry in corpus_definition() {
        let b = entry.value();
        rows.push((format!("{}:[b]", entry.name), vec![b]));
        rows.push((format!("{}:[b,1]", entry.name), vec![b, 1.0]));
        rows.push((format!("{}:[b,0,-1]", entry.name), vec![b, 0.0, -1.0]));
        rows.push((format!("{}:[1,b,2]", entry.name), vec![1.0, b, 2.0]));
    }
    rows.push(("empty:[]".to_string(), Vec::new()));
    rows.push(("wide8".to_string(), WIDE_ROW.to_vec()));
    rows
}

fn row_text(row: &[f64]) -> String {
    row.iter()
        .map(|value| hex_bits(*value))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `ERR:<kind>`, a stable tag this file gives each evaluation error: the typed error by its
/// variant, and the two scope errors by what their message names.
fn error_token(error: &anyhow::Error) -> String {
    if let Some(typed) = error.downcast_ref::<MathEvalError>() {
        return format!("{ERR}:{typed:?}");
    }
    let message = error.to_string();
    let kind = if message.starts_with("variable x is only valid") {
        "x-scope"
    } else if message.starts_with("column index") {
        "column"
    } else {
        "other"
    };
    format!("{ERR}:{kind}")
}

/// The token of an expression whose compiled form failed to build.
const COMPILE_ERR: &str = "ERR:compile";

fn token(result: anyhow::Result<f64>) -> String {
    match result {
        Ok(value) => hex_bits(value),
        Err(error) => error_token(&error),
    }
}

fn expression_values(
    source: &str,
    context: &'static str,
    rows: &[(String, Vec<f64>)],
) -> ExpressionValues {
    let expr = MathExpr::parse(source)
        .unwrap_or_else(|error| panic!("expression {source:?} no longer parses: {error}"));
    let compiled = expr.compile();
    let interpreter_row = rows
        .iter()
        .map(|(_, row)| token(expr.eval_row(row)))
        .collect();
    let compiled_row = rows
        .iter()
        .map(|(_, row)| match &compiled {
            Ok(compiled) => token(compiled.eval_row(row)),
            Err(_) => COMPILE_ERR.to_string(),
        })
        .collect();
    let mut paths = vec![
        ("interpreter_row", interpreter_row),
        ("compiled_row", compiled_row),
    ];
    if context == "map" {
        let interpreter_map = rows
            .iter()
            .flat_map(|(_, row)| {
                row.iter()
                    .map(|x| token(expr.eval_element(*x, row)))
                    .collect::<Vec<_>>()
            })
            .collect();
        let compiled_map = rows
            .iter()
            .flat_map(|(_, row)| match &compiled {
                Ok(compiled) => compiled.map_row(row).into_iter().map(hex_bits).collect(),
                Err(_) => vec![COMPILE_ERR.to_string(); row.len()],
            })
            .collect();
        paths.push(("interpreter_map", interpreter_map));
        paths.push(("compiled_map", compiled_map));
    }
    ExpressionValues {
        source: source.to_string(),
        context,
        paths,
    }
}

fn err_as_zero(tokens: &[String]) -> Vec<String> {
    tokens
        .iter()
        .map(|t| {
            if t.starts_with(ERR) {
                hex_bits(0.0)
            } else {
                t.clone()
            }
        })
        .collect()
}

fn path_tokens<'a>(values: &'a ExpressionValues, name: &str) -> &'a [String] {
    &values
        .paths
        .iter()
        .find(|(n, _)| *n == name)
        .expect("recorded path")
        .1
}

fn encode_expression(values: &ExpressionValues) -> ExpressionRecord {
    let interpreter_row = path_tokens(values, "interpreter_row");
    let compiled_row = path_tokens(values, "compiled_row");
    let (interpreter_map, compiled_map) = if values.context == "map" {
        let interpreter_map = path_tokens(values, "interpreter_map");
        let compiled_map = path_tokens(values, "compiled_map");
        let compiled_map = if compiled_map == err_as_zero(interpreter_map).as_slice() {
            SAME_AS_INTERPRETER_MAP.to_string()
        } else {
            compiled_map.join(" ")
        };
        (Some(interpreter_map.join(" ")), Some(compiled_map))
    } else {
        (None, None)
    };
    ExpressionRecord {
        source: values.source.clone(),
        context: values.context.to_string(),
        interpreter_row: interpreter_row.join(" "),
        compiled_row: if compiled_row == interpreter_row {
            SAME_AS_INTERPRETER_ROW.to_string()
        } else {
            compiled_row.join(" ")
        },
        interpreter_map,
        compiled_map,
    }
}

fn split_tokens(text: &str) -> Vec<String> {
    text.split(' ').map(str::to_string).collect()
}

/// Expand a fixture record's equivalence markers into full token lists.
fn decode_expression(record: &ExpressionRecord) -> Vec<(&'static str, Vec<String>)> {
    let interpreter_row = split_tokens(&record.interpreter_row);
    let compiled_row = if record.compiled_row == SAME_AS_INTERPRETER_ROW {
        interpreter_row.clone()
    } else {
        split_tokens(&record.compiled_row)
    };
    let mut paths = vec![
        ("interpreter_row", interpreter_row),
        ("compiled_row", compiled_row),
    ];
    if let (Some(interpreter_map), Some(compiled_map)) =
        (&record.interpreter_map, &record.compiled_map)
    {
        let interpreter_map = split_tokens(interpreter_map);
        let compiled_map = if compiled_map == SAME_AS_INTERPRETER_MAP {
            err_as_zero(&interpreter_map)
        } else {
            split_tokens(compiled_map)
        };
        paths.push(("interpreter_map", interpreter_map));
        paths.push(("compiled_map", compiled_map));
    }
    paths
}

fn expression_catalogue() -> Vec<(&'static str, &'static str)> {
    ROW_EXPRESSIONS
        .iter()
        .map(|source| (*source, "row"))
        .chain(MAP_EXPRESSIONS.iter().map(|source| (*source, "map")))
        .collect()
}

#[test]
fn expression_values_match_the_frozen_fixture() {
    let rows = corpus_rows();
    let catalogue = expression_catalogue();
    assert!(
        catalogue.len() >= 60,
        "WP-0.2 needs at least 60 expressions"
    );
    let actual: Vec<ExpressionValues> = catalogue
        .iter()
        .map(|&(source, context)| expression_values(source, context, &rows))
        .collect();
    let fixture = MathExprValues {
        schema: "feat-math.compat-freeze.mathexpr-values.v1".to_string(),
        generated_at_sha: header_sha(Surface::Frozen),
        encoding: format!(
            "Each path is one token per row (row paths) or per row element (map paths), in row order, separated by single spaces: 16 hex digits of the f64 bits, or {ERR}:<kind> when the call returned an error (InvalidClipBounds: the typed clip error; x-scope: x outside a map; column: a column index past the row; other: any other error; compile: the compiled form did not build). compiled_row may be \"{SAME_AS_INTERPRETER_ROW}\" and compiled_map may be \"{SAME_AS_INTERPRETER_MAP}\" when the compiled path agreed with the interpreter at generation. Rows are built from math_boundary/v1.json: [b], [b,1], [b,0,-1], [1,b,2] for each entry b, then [] and one 8-wide row."
        ),
        rows: rows
            .iter()
            .map(|(label, row)| RowRecord {
                label: label.clone(),
                values: row_text(row),
            })
            .collect(),
        expressions: actual.iter().map(encode_expression).collect(),
    };
    let Some(committed) = committed_or_write(Surface::Frozen, FIXTURE_FILES[0], &fixture) else {
        return;
    };
    assert_eq!(
        committed.schema, fixture.schema,
        "mathexpr_values fixture schema"
    );
    if committed.rows != fixture.rows {
        let first = committed
            .rows
            .iter()
            .zip(&fixture.rows)
            .position(|(a, b)| a != b)
            .unwrap_or(committed.rows.len().min(fixture.rows.len()));
        panic!(
            "compat-freeze: the corpus rows changed (fixture {} rows, now {}; first difference at row {first}: fixture {:?}, now {:?}). The corpus is versioned: add math_boundary/v2.json instead of editing v1.",
            committed.rows.len(),
            fixture.rows.len(),
            committed.rows.get(first),
            fixture.rows.get(first)
        );
    }
    let committed_sources: Vec<(&str, &str)> = committed
        .expressions
        .iter()
        .map(|e| (e.source.as_str(), e.context.as_str()))
        .collect();
    assert_eq!(
        committed_sources, catalogue,
        "the expression catalogue changed; the fixture and this file must list the same expressions"
    );
    let row_lengths: Vec<usize> = rows.iter().map(|(_, row)| row.len()).collect();
    let locate = |flat: usize| -> (usize, usize) {
        let mut remaining = flat;
        for (i, len) in row_lengths.iter().enumerate() {
            if remaining < *len {
                return (i, remaining);
            }
            remaining -= len;
        }
        (row_lengths.len(), remaining)
    };
    let mut problems = Vec::new();
    let mut differing = 0usize;
    for (record, values) in committed.expressions.iter().zip(&actual) {
        let expected_paths = decode_expression(record);
        let expected_names: Vec<&str> = expected_paths.iter().map(|(n, _)| *n).collect();
        let actual_names: Vec<&str> = values.paths.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            expected_names, actual_names,
            "{}: recorded paths differ",
            values.source
        );
        for ((path, old), (_, new)) in expected_paths.iter().zip(&values.paths) {
            if old.len() != new.len() {
                problems.push(format!(
                    "{:?} [{path}]: fixture has {} values, now {}",
                    values.source,
                    old.len(),
                    new.len()
                ));
                differing += 1;
                continue;
            }
            for (i, (a, b)) in old.iter().zip(new).enumerate() {
                if a == b {
                    continue;
                }
                differing += 1;
                if problems.len() < 12 {
                    let (row_index, element) = if path.ends_with("_row") {
                        (i, None)
                    } else {
                        let (r, e) = locate(i);
                        (r, Some(e))
                    };
                    let (label, row) = &rows[row_index];
                    problems.push(format!(
                        "{:?} [{path}] row {row_index} {label} = [{}]{}: fixture {a} ({}) -> now {b} ({})",
                        values.source,
                        row.iter()
                            .map(|v| format!("{v:?}"))
                            .collect::<Vec<_>>()
                            .join(", "),
                        element
                            .map(|e| format!(" element {e}"))
                            .unwrap_or_default(),
                        decode_token(a),
                        decode_token(b)
                    ));
                }
            }
        }
    }
    if differing > 0 {
        panic!(
            "compat-freeze: MathExpr values drifted: {differing} differing values (first {} below)\n  {}\n{}\n{REBASELINE_POLICY}",
            problems.len(),
            problems.join("\n  "),
            toolchain_note()
        );
    }
}

// ---------------------------------------------------------------------------------------------
// WP-0.2 (a, stage level): map/derive stage outputs through apply_op and fit/predict
// ---------------------------------------------------------------------------------------------

/// The 5x4 stage matrix: ordinary values, signed zeros and guard neighbours, non-finite values,
/// magnitudes beyond the clamp, and rounding cases.
fn stage_matrix() -> Rows {
    vec![
        vec![1.0, -2.0, 0.5, 3.0],
        vec![0.0, -0.0, 1e-12, -1e-12],
        vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 2.0],
        vec![1e155, -parse_f64("1.3407807929942596e154"), 27.0, -30.0],
        vec![toward_zero(0.5), 2.5, -1.0, 4_503_599_627_370_497.0],
    ]
}

const STAGES: &[&str] = &[
    "(map expr=\"sigmoid(x) + 0.1*col(0)\" append=true)",
    "(map expr=\"x^2 - min(x, c2) + tanh(c3)\")",
    "(map expr=\"step(col(0), 0.0) + 0.5*step(row_mean, c1)\" append=true)",
    "(map expr=\"exp(clip(col(0),-2,2)) + log(abs(col(1)) + 1.0) + asinh(col(2)) + atan(col(3))\")",
    "(map expr=\"clip(x, -1, 1)\")",
    "(map expr=\"x - row_mean + row_std\" append=true)",
    "(map expr=\"c7 + x\")",
    "(map expr=\"clip(1,(-1)^0.5,(-1)^0.5)\")",
    "(map expr=\"x/c0\")",
    "(derive expr=\"row_mean + row_std - c1/(c0 + 3)\" append=true)",
    "(derive expr=\"row_l1 + row_l2 + row_min + row_max\")",
    "(derive expr=\"erf(col(0)) + sign(col(1)) * floor(col(2)) + round(col(3))\" append=true)",
    "(derive expr=\"1/c0 + log(c1) + sqrt(c2) + log1p(c3)\" append=true)",
    "(derive expr=\"asinh(c0*c0) + exp(c3)\")",
    "(derive expr=\"c7\")",
    "(derive expr=\"x\")",
];

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StageValues {
    schema: String,
    generated_at_sha: String,
    training: String,
    matrix: Vec<String>,
    cases: Vec<StageCase>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StageCase {
    stage: String,
    apply_op_raw: Vec<String>,
    predict_stage_output: Vec<String>,
    predictions: String,
}

fn parse_stage(text: &str) -> Stage {
    parse_dsl(&format!("(pipe X {text})"))
        .unwrap_or_else(|error| panic!("{text}: {error}"))
        .stages
        .remove(0)
}

fn stage_case(text: &str, train: &Rows, target: &[f64], matrix: &Rows) -> StageCase {
    let stage = parse_stage(text);
    let apply_op_raw = match fit_op(&stage, matrix, target) {
        Ok(op) => apply_op(&op, matrix)
            .unwrap()
            .iter()
            .map(|row| row_text(row))
            .collect(),
        Err(error) => vec![format!("ERR {error}")],
    };
    let genome = parse_program(text, &format!("(pipe X {text} (ridge))"));
    let (predict_stage_output, predictions) = match fit(&genome, train, target) {
        Ok(model) => {
            let Model::Pipeline { ops, head, target } = &model else {
                panic!("{text}: a pipeline program fits a pipeline model")
            };
            assert!(target.is_none());
            let mut current = matrix.clone();
            sanitize(&mut current);
            for op in ops {
                current = apply_op(op, &current).unwrap();
            }
            let predictions = model.predict(matrix).unwrap();
            let replayed = head.predict(&current).unwrap();
            assert_eq!(
                predictions.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                replayed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{text}: the per-stage replay must reproduce Model::predict"
            );
            (
                current.iter().map(|row| row_text(row)).collect(),
                row_text(&predictions),
            )
        }
        Err(error) => (vec![format!("ERR {error}")], format!("ERR {error}")),
    };
    StageCase {
        stage: text.to_string(),
        apply_op_raw,
        predict_stage_output,
        predictions,
    }
}

#[test]
fn stage_outputs_match_the_frozen_fixture() {
    let (train, target) = toy_data(40, 5, 42);
    let train: Rows = train.iter().map(|row| row[..4].to_vec()).collect();
    let matrix = stage_matrix();
    let fixture = StageValues {
        schema: "feat-math.compat-freeze.stage-values.v1".to_string(),
        generated_at_sha: header_sha(Surface::Frozen),
        training: "Each stage is fitted as (pipe X <stage> (ridge)) on the first four columns of toy_data(40 rows, 5 columns, seed 42) as in tests/golden.rs and applied to the matrix below. apply_op_raw is apply_op on the raw matrix (no input sanitizing); predict_stage_output is the fitted stage on the sanitized matrix, as Model::predict runs it; rows are space-separated f64 bits.".to_string(),
        matrix: matrix.iter().map(|row| row_text(row)).collect(),
        cases: STAGES
            .iter()
            .map(|text| stage_case(text, &train, &target, &matrix))
            .collect(),
    };
    let Some(committed) = committed_or_write(Surface::Frozen, FIXTURE_FILES[1], &fixture) else {
        return;
    };
    let flatten = |values: &StageValues| -> Vec<(String, String)> {
        let mut records = vec![record("schema", values.schema.clone())];
        for (i, row) in values.matrix.iter().enumerate() {
            records.push(record(format!("matrix[{i}]"), row.clone()));
        }
        for case in &values.cases {
            for (i, row) in case.apply_op_raw.iter().enumerate() {
                records.push(record(
                    format!("{} apply_op_raw[{i}]", case.stage),
                    row.clone(),
                ));
            }
            for (i, row) in case.predict_stage_output.iter().enumerate() {
                records.push(record(
                    format!("{} predict_stage_output[{i}]", case.stage),
                    row.clone(),
                ));
            }
            records.push(record(
                format!("{} predictions", case.stage),
                case.predictions.clone(),
            ));
        }
        records
    };
    assert_records(
        "stage outputs (stage_values.v1.json)",
        &flatten(&committed),
        &flatten(&fixture),
        REBASELINE_POLICY,
    );
}

// ---------------------------------------------------------------------------------------------
// WP-0.2 (c): emitted artifact bytes
// ---------------------------------------------------------------------------------------------

const EMITTED_FILE: &str = FIXTURE_FILES[2];
const EMITTED_SURFACE: &str = "emitted artifact bytes (emitted_bytes.v1.json)";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmittedBytes {
    schema: String,
    generated_at_sha: String,
    note: String,
    cpp17_recipe_ops: Vec<String>,
    javascript_ops: Vec<String>,
    programs: Vec<EmittedProgram>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmittedProgram {
    name: String,
    canonical: String,
    rust: String,
    python: String,
    manifest: String,
    manifest_emitter_version: String,
    rust_jope_emit: String,
    python_jope_emit: String,
    manifest_jope_emit: String,
    cpp17: String,
    javascript: String,
    fitted_cpp17_source: String,
    fitted_cpp17_codec: String,
}

/// The values an emitter can record as unsupported: each field, and the language a message names.
const REFUSABLE: [(&str, &str); 4] = [
    ("cpp17", "C++17 recipe"),
    ("javascript", "JavaScript"),
    ("fitted_cpp17_source", "fitted C++17 source"),
    ("fitted_cpp17_codec", "fitted C++17 codec"),
];

impl EmittedProgram {
    /// The values of the `REFUSABLE` fields, in that order.
    fn refusable(&self) -> [&String; 4] {
        [
            &self.cpp17,
            &self.javascript,
            &self.fitted_cpp17_source,
            &self.fitted_cpp17_codec,
        ]
    }

    fn refusable_mut(&mut self) -> [&mut String; 4] {
        [
            &mut self.cpp17,
            &mut self.javascript,
            &mut self.fitted_cpp17_source,
            &mut self.fitted_cpp17_codec,
        ]
    }
}

const UNSUPPORTED: &str = "unsupported: ";

/// "unsupported: <kinds>", keeping each reason's element kind and dropping the `Debug` payload
/// of internal plan structs (their text is not a shipped surface).
fn unsupported(reasons: &[String]) -> String {
    let kinds: Vec<&str> = reasons
        .iter()
        .map(|reason| reason[..reason.find(['(', '{']).unwrap_or(reason.len())].trim_end())
        .collect();
    format!("{UNSUPPORTED}{}", kinds.join("; "))
}

/// The element kinds of an "unsupported: a; b" value, or `None` for anything else.
fn unsupported_kinds(value: &str) -> Option<BTreeSet<&str>> {
    value
        .strip_prefix(UNSUPPORTED)
        .map(|kinds| kinds.split("; ").collect())
}

/// Whether the emitter still refuses a program recorded as unsupported, but for a strict subset of
/// the recorded element kinds. That is additive and passes; the fixture keeps the recorded refusal,
/// because no mode rewrites a refusal whose element kinds changed. Any other change of kinds (an
/// element that loses support while the program stays unsupported, a different kind) is drift.
fn narrows_refusal(recorded: &str, now: &str) -> bool {
    match (unsupported_kinds(recorded), unsupported_kinds(now)) {
        (Some(before), Some(after)) => after.len() < before.len() && after.is_subset(&before),
        _ => false,
    }
}

/// Whether the emitter now writes bytes for a value recorded as unsupported. Compare mode fails on
/// it, so the new bytes are pinned from the change that adds the support, and only
/// `JOPE_COMPAT_FREEZE_REGENERATE=gained-support` records them. (Where the emitter now errors or
/// the value is skipped, the change is drift.)
fn gains_bytes(recorded: &str, now: &str) -> bool {
    unsupported_kinds(recorded).is_some() && is_emitted(now)
}

/// Whether a value is emitted bytes ("<length> <sha256>") rather than a refusal, an error or a
/// skip.
fn is_emitted(value: &str) -> bool {
    !(value.starts_with(UNSUPPORTED) || value.starts_with("ERR") || value.starts_with("skipped"))
}

/// The stage ops of a pipeline program (its target, transforms and head); none for a formula.
fn program_ops(source: &str) -> BTreeSet<String> {
    match GenomeIr::parse(source) {
        Ok(GenomeIr::Pipeline(pipe)) => pipe.stages.iter().map(|stage| stage.op.clone()).collect(),
        _ => BTreeSet::new(),
    }
}

/// The ops the C++17 recipe and JavaScript emitters accept, probed one op at a time: every op of
/// `full_rust::operation_schemas()` and every head of `evolve::HEAD_OPS`, each in the smallest
/// program that carries it (a transform or target before `(ridge)`, a head alone).
fn recipe_supported_ops() -> (Vec<String>, Vec<String>) {
    let mut probes: BTreeMap<String, String> = BTreeMap::new();
    for schema in full_rust::operation_schemas() {
        let op = schema.op;
        let expression = schema
            .params
            .iter()
            .any(|param| matches!(param.domain, ParamDomain::Expression));
        let probe = match schema.kind {
            OperationKind::Head => format!("(pipe X ({op}))"),
            OperationKind::Target => format!("(pipe X ({op} kind=log1p) (ridge))"),
            OperationKind::Transform if expression => {
                format!("(pipe X ({op} expr=\"c0\") (ridge))")
            }
            OperationKind::Transform => format!("(pipe X ({op}) (ridge))"),
        };
        probes.insert(op.to_string(), probe);
    }
    for head in HEAD_OPS {
        probes
            .entry(head.to_string())
            .or_insert_with(|| format!("(pipe X ({head}))"));
    }
    let (mut cpp17, mut javascript) = (Vec::new(), Vec::new());
    for (op, probe) in &probes {
        let Ok(genome) = GenomeIr::parse(probe) else {
            continue;
        };
        if supports_cpp17(&genome).is_ok() {
            cpp17.push(op.clone());
        }
        if supports_javascript(&genome).is_ok() {
            javascript.push(op.clone());
        }
    }
    (cpp17, javascript)
}

/// Reads one emitter's recorded value from a program record.
type RecordedValue = fn(&EmittedProgram) -> &str;

fn cpp17_bytes(program: &EmittedProgram) -> &str {
    &program.cpp17
}

fn javascript_bytes(program: &EmittedProgram) -> &str {
    &program.javascript
}

/// Every op an emitter accepted must be used by a program whose bytes from that emitter are
/// recorded, so each op body and shared helper the emitter writes is pinned.
fn recipe_coverage_gaps(values: &EmittedBytes) -> Vec<String> {
    let recipes: [(&str, &Vec<String>, RecordedValue); 2] = [
        ("C++17 recipe", &values.cpp17_recipe_ops, cpp17_bytes),
        ("JavaScript", &values.javascript_ops, javascript_bytes),
    ];
    let mut gaps = Vec::new();
    for (language, supported, bytes) in recipes {
        let pinned: BTreeSet<String> = values
            .programs
            .iter()
            .filter(|program| is_emitted(bytes(program)))
            .flat_map(|program| program_ops(&program.canonical))
            .collect();
        for op in supported {
            if !pinned.contains(op) {
                gaps.push(format!(
                    "{op}: the {language} emitter accepts it, but no program in the set records {language} bytes that use it; add one to NAMED_PROGRAMS"
                ));
            }
        }
    }
    gaps
}

/// The options `jope emit` writes Rust, Python and the manifest with
/// (`crates/jope-cli/src/kernel.rs` `emit_genome`; its C++ and JavaScript do not embed the title).
fn jope_emit_options() -> EmitOptions {
    EmitOptions {
        title: Some("jope emit".to_string()),
        ..EmitOptions::default()
    }
}

/// `tests/ip_scrub.rs` lints every artifact the emitters write for the frozen programs (plan N6).
const IP_SCRUB_SOURCE: &str = include_str!("ip_scrub.rs");

/// Frozen programs that `tests/ip_scrub.rs` does not list verbatim.
fn ip_scrub_gaps() -> Vec<String> {
    NAMED_PROGRAMS
        .iter()
        .map(|(_, source)| *source)
        .chain(IP_SCRUB_PIPELINES.iter().copied())
        .chain(IP_SCRUB_SYMBOLIC.iter().copied())
        .filter(|source| !IP_SCRUB_SOURCE.contains(&format!("{source:?}")))
        .map(str::to_string)
        .collect()
}

fn emitted_program(name: &str, source: &str, train: &Rows, target: &[f64]) -> EmittedProgram {
    let genome = parse_program(name, source);
    let options = EmitOptions::default();
    let (rust_jope_emit, python_jope_emit, manifest_jope_emit) =
        match emit(&genome, &jope_emit_options()) {
            Ok(artifact) => (
                text_identity(&artifact.rust),
                text_identity(&artifact.python),
                text_identity(&artifact.manifest),
            ),
            Err(error) => {
                let error = format!("ERR {error:#}");
                (error.clone(), error.clone(), error)
            }
        };
    let (rust, python, manifest, manifest_emitter_version) = match emit(&genome, &options) {
        Ok(artifact) => {
            let manifest: serde_json::Value =
                serde_json::from_str(&artifact.manifest).expect("manifest JSON");
            (
                text_identity(&artifact.rust),
                text_identity(&artifact.python),
                text_identity(&artifact.manifest),
                manifest["emitter_version"]
                    .as_str()
                    .expect("emitter_version")
                    .to_string(),
            )
        }
        Err(error) => {
            let error = format!("ERR {error:#}");
            (error.clone(), error.clone(), error.clone(), error)
        }
    };
    let cpp17 = match supports_cpp17(&genome) {
        Ok(()) => emit_cpp17(&genome, &options)
            .map(|source| text_identity(&source))
            .unwrap_or_else(|error| format!("ERR {error:#}")),
        Err(missing) => unsupported(&missing),
    };
    let javascript = match supports_javascript(&genome) {
        Ok(()) => emit_javascript(&genome, &options)
            .map(|source| text_identity(&source))
            .unwrap_or_else(|error| format!("ERR {error:#}")),
        Err(missing) => unsupported(&missing),
    };
    let (fitted_cpp17_source, fitted_cpp17_codec) = match fit_skip_reason(source) {
        Some(reason) => (reason.to_string(), reason.to_string()),
        None => match fit_legacy_numeric_bound(&genome, train, target) {
            Ok(model) => match emit_fitted_cpp17(&model) {
                Ok(artifact) => {
                    assert_eq!(
                        artifact.source_sha256,
                        sha256_hex(artifact.source.as_bytes())
                    );
                    assert_eq!(artifact.codec_sha256, sha256_hex(artifact.codec.as_bytes()));
                    (
                        text_identity(&artifact.source),
                        text_identity(&artifact.codec),
                    )
                }
                Err(error) => {
                    let error = unsupported(&[format!("{error:#}")]);
                    (error.clone(), error)
                }
            },
            Err(error) => {
                let error = format!("ERR fit: {error:#}");
                (error.clone(), error)
            }
        },
    };
    EmittedProgram {
        name: name.to_string(),
        canonical: genome.canonical(),
        rust,
        python,
        manifest,
        manifest_emitter_version,
        rust_jope_emit,
        python_jope_emit,
        manifest_jope_emit,
        cpp17,
        javascript,
        fitted_cpp17_source,
        fitted_cpp17_codec,
    }
}

/// The emitted-bytes records compare mode checks, in fixture order.
fn flatten_emitted(values: &EmittedBytes) -> Vec<(String, String)> {
    let mut records = vec![record("schema", values.schema.clone())];
    for p in &values.programs {
        let n = &p.name;
        records.extend([
            record(format!("{n}.canonical"), p.canonical.clone()),
            record(format!("{n}.rust"), p.rust.clone()),
            record(format!("{n}.python"), p.python.clone()),
            record(format!("{n}.manifest"), p.manifest.clone()),
            record(
                format!("{n}.manifest_emitter_version"),
                p.manifest_emitter_version.clone(),
            ),
            record(format!("{n}.rust_jope_emit"), p.rust_jope_emit.clone()),
            record(format!("{n}.python_jope_emit"), p.python_jope_emit.clone()),
            record(
                format!("{n}.manifest_jope_emit"),
                p.manifest_jope_emit.clone(),
            ),
            record(format!("{n}.cpp17"), p.cpp17.clone()),
            record(format!("{n}.javascript"), p.javascript.clone()),
            record(
                format!("{n}.fitted_cpp17_source"),
                p.fitted_cpp17_source.clone(),
            ),
            record(
                format!("{n}.fitted_cpp17_codec"),
                p.fitted_cpp17_codec.clone(),
            ),
        ]);
    }
    records
}

/// How the emitters' output compares with the emitted-bytes fixture (see the module doc).
#[derive(Debug, Default)]
struct EmittedCheck {
    /// Values recorded as unsupported for which an emitter now writes bytes. Compare mode fails on
    /// them, and only `JOPE_COMPAT_FREEZE_REGENERATE=gained-support` records them.
    gained: Vec<String>,
    /// Additive changes, which pass and are reported: narrowed refusals and newly accepted ops.
    additive: Vec<String>,
    /// Every other difference. Each one fails, and only an approved rebaseline records it.
    drift: Vec<String>,
}

/// Compares the emitters' output with the committed fixture: every recorded op stays pinned and
/// supported, and every program record holds, apart from the additive changes and gained bytes.
fn check_emitted(committed: &EmittedBytes, actual: &EmittedBytes) -> EmittedCheck {
    let mut check = EmittedCheck {
        drift: recipe_coverage_gaps(committed),
        ..EmittedCheck::default()
    };
    for (language, recorded, now) in [
        (
            "C++17 recipe",
            &committed.cpp17_recipe_ops,
            &actual.cpp17_recipe_ops,
        ),
        (
            "JavaScript",
            &committed.javascript_ops,
            &actual.javascript_ops,
        ),
    ] {
        for op in recorded {
            if !now.contains(op) {
                check.drift.push(format!("{op} lost {language} support"));
            }
        }
        for op in now {
            if !recorded.contains(op) {
                check.additive.push(format!(
                    "the {language} emitter now accepts {op} (additive; pin it with an approved rebaseline and a program that uses it)"
                ));
            }
        }
    }
    // An additive or gained value is listed above and compared below as its recorded value, so
    // the records report only the rest.
    let mut expected = actual.clone();
    for program in &mut expected.programs {
        let Some(old) = committed.programs.iter().find(|p| p.name == program.name) else {
            continue;
        };
        let name = program.name.clone();
        for (&(field, language), (now, recorded)) in REFUSABLE
            .iter()
            .zip(program.refusable_mut().into_iter().zip(old.refusable()))
        {
            if narrows_refusal(recorded, now) {
                check.additive.push(format!(
                    "{name}.{field} was recorded as {recorded:?} and is now {now:?} (additive: the {language} emitter refuses fewer element kinds; the fixture keeps the recorded refusal)"
                ));
                now.clone_from(recorded);
            } else if gains_bytes(recorded, now) {
                check.gained.push(format!(
                    "{name}.{field} ({language}): recorded as {recorded:?}, now emits {now}"
                ));
                now.clone_from(recorded);
            }
        }
    }
    check.drift.extend(record_problems(
        EMITTED_SURFACE,
        &flatten_emitted(committed),
        &flatten_emitted(&expected),
    ));
    check
}

/// Compare mode's failure message for the emitted bytes, or `None` when the fixture holds.
fn emitted_failure(check: &EmittedCheck) -> Option<String> {
    let mut sections = Vec::new();
    if !check.gained.is_empty() {
        sections.push(format!(
            "compat-freeze: {} values that {EMITTED_FILE} records as unsupported now emit bytes, and nothing pins those bytes yet (first {} below)\n  {}\nRecord them with `{GAINED_SUPPORT_COMMAND}` in the change that adds the support. That mode replaces only these values, and writes only when every other record matches.",
            check.gained.len(),
            check.gained.len().min(12),
            first_lines(&check.gained)
        ));
    }
    if !check.drift.is_empty() {
        sections.push(drift_message(
            EMITTED_SURFACE,
            &check.drift,
            REBASELINE_POLICY,
        ));
    }
    (!sections.is_empty()).then(|| sections.join("\n"))
}

/// `JOPE_COMPAT_FREEZE_REGENERATE=gained-support` applied to the text of the emitted-bytes fixture.
/// Each value recorded as unsupported, for which an emitter now writes bytes, takes those bytes, and
/// nothing else changes. Returns the new text and the `<program>.<field>` values it recorded, or
/// `None` when no value gained support. It refuses, writing nothing, when any other record differs
/// from the emitters' output, or when the text is not in the form this test writes (rewriting it
/// would then move bytes that the mode must not touch).
fn gained_support_update(
    committed_text: &str,
    actual: &EmittedBytes,
) -> Result<Option<(String, Vec<String>)>, String> {
    let mut updated: EmittedBytes = serde_json::from_str(committed_text).map_err(|error| {
        format!("tests/fixtures/compat_freeze/{EMITTED_FILE} does not parse ({error})")
    })?;
    if fixture_text(&updated) != committed_text {
        return Err(format!(
            "compat-freeze: {GAINED_SUPPORT_MODE} wrote nothing: tests/fixtures/compat_freeze/{EMITTED_FILE} is not in the form this test writes, so rewriting it would change more than the values it records"
        ));
    }
    let mut recorded = Vec::new();
    for program in &mut updated.programs {
        let Some(current) = actual.programs.iter().find(|p| p.name == program.name) else {
            continue;
        };
        let name = program.name.clone();
        for (&(field, _), (value, now)) in REFUSABLE
            .iter()
            .zip(program.refusable_mut().into_iter().zip(current.refusable()))
        {
            if gains_bytes(value, now) {
                value.clone_from(now);
                recorded.push(format!("{name}.{field}"));
            }
        }
    }
    let check = check_emitted(&updated, actual);
    assert!(
        check.gained.is_empty(),
        "every gained value is recorded: {:?}",
        check.gained
    );
    if !check.drift.is_empty() {
        return Err(format!(
            "compat-freeze: {GAINED_SUPPORT_MODE} wrote nothing: it records only values recorded as unsupported for which an emitter now writes bytes, and other records differ.\n{}",
            drift_message(EMITTED_SURFACE, &check.drift, REBASELINE_POLICY)
        ));
    }
    Ok((!recorded.is_empty()).then(|| (fixture_text(&updated), recorded)))
}

/// `JOPE_COMPAT_FREEZE_REGENERATE=fitted-cpp-revision` applied to the text of the emitted-bytes
/// fixture. Each fitted C++17 source or codec value that is recorded as bytes, and for which the
/// emitter now writes other bytes, takes those bytes, and the header's `generated_at_sha` becomes
/// `sha`; nothing else changes. Returns the new text and the `<program>.<field>` values it
/// recorded, or `None` when no fitted value changed (the header then stays as it is). It refuses,
/// writing nothing, when any other record differs from the emitters' output (among them a fitted
/// value that becomes a refusal, an error or a skip, and a skip or error that becomes bytes), when
/// a value recorded as unsupported now emits bytes (that is `gained-support`'s to record), or when
/// the text is not in the form this test writes.
fn fitted_cpp_revision_update(
    committed_text: &str,
    actual: &EmittedBytes,
    sha: &str,
) -> Result<Option<(String, Vec<String>)>, String> {
    let mut updated: EmittedBytes = serde_json::from_str(committed_text).map_err(|error| {
        format!("tests/fixtures/compat_freeze/{EMITTED_FILE} does not parse ({error})")
    })?;
    if fixture_text(&updated) != committed_text {
        return Err(format!(
            "compat-freeze: {FITTED_CPP_REVISION_MODE} wrote nothing: tests/fixtures/compat_freeze/{EMITTED_FILE} is not in the form this test writes, so rewriting it would change more than the values it records"
        ));
    }
    let mut recorded = Vec::new();
    for program in &mut updated.programs {
        let Some(current) = actual.programs.iter().find(|p| p.name == program.name) else {
            continue;
        };
        let name = program.name.clone();
        for (field, value, now) in [
            (
                "fitted_cpp17_source",
                &mut program.fitted_cpp17_source,
                &current.fitted_cpp17_source,
            ),
            (
                "fitted_cpp17_codec",
                &mut program.fitted_cpp17_codec,
                &current.fitted_cpp17_codec,
            ),
        ] {
            if is_emitted(value) && is_emitted(now) && value != now {
                value.clone_from(now);
                recorded.push(format!("{name}.{field}"));
            }
        }
    }
    let check = check_emitted(&updated, actual);
    if !check.gained.is_empty() {
        return Err(format!(
            "compat-freeze: {FITTED_CPP_REVISION_MODE} wrote nothing: {} values recorded as unsupported now emit bytes (first {} below), and only {GAINED_SUPPORT_MODE} records those\n  {}",
            check.gained.len(),
            check.gained.len().min(12),
            first_lines(&check.gained)
        ));
    }
    if !check.drift.is_empty() {
        return Err(format!(
            "compat-freeze: {FITTED_CPP_REVISION_MODE} wrote nothing: it records only fitted C++17 values recorded as bytes for which the emitter now writes other bytes, and other records differ.\n{}",
            drift_message(EMITTED_SURFACE, &check.drift, REBASELINE_POLICY)
        ));
    }
    if recorded.is_empty() {
        return Ok(None);
    }
    updated.generated_at_sha = sha.to_string();
    Ok(Some((fixture_text(&updated), recorded)))
}

#[test]
fn emitted_artifact_bytes_match_the_frozen_fixture() {
    let gaps = ip_scrub_gaps();
    assert!(
        gaps.is_empty(),
        "compat-freeze: these frozen programs are not linted by tests/ip_scrub.rs (plan N6); add them to its COMPAT_FREEZE_PROGRAMS verbatim:\n  {}",
        gaps.join("\n  ")
    );
    let (train, target, _) = fitted_surface_data();
    let (cpp17_recipe_ops, javascript_ops) = recipe_supported_ops();
    let fixture = EmittedBytes {
        schema: "feat-math.compat-freeze.emitted-bytes.v1".to_string(),
        generated_at_sha: header_sha(Surface::Frozen),
        note: format!(
            "Values are \"<byte length> <sha256>\" of the artifact text (EmitOptions::default()), \"unsupported: <element kinds>\" where the emitter refuses the program, or the recorded error or skip. The *_jope_emit values use the options `jope emit` writes Rust, Python and the manifest with (title \"jope emit\"); C++17 recipe and JavaScript are recorded under the default options only. C++20 is the same C++17 source compiled with -std=c++20. Fitted C++17 comes from fit_legacy_numeric_bound on {FITTED_DATA_NOTE}. cpp17_recipe_ops and javascript_ops are the ops each emitter accepted at generation, probed one op at a time; every one is used by a program whose bytes from that emitter are recorded here. Recorded bytes never change. Two changes are additive and pass: fewer unsupported element kinds (the recorded refusal stays) and a newly accepted op. Bytes for a value recorded as unsupported fail until JOPE_COMPAT_FREEZE_REGENERATE=gained-support records them."
        ),
        cpp17_recipe_ops,
        javascript_ops,
        programs: program_set()
            .iter()
            .map(|(name, source)| emitted_program(name, source, &train, &target))
            .collect(),
    };
    if regenerating(Surface::Frozen) {
        let gaps = recipe_coverage_gaps(&fixture);
        assert!(
            gaps.is_empty(),
            "compat-freeze: the program set leaves recipe ops unpinned:\n  {}",
            gaps.join("\n  ")
        );
    }
    if recording_gained_support() {
        let relative = format!("compat_freeze/{EMITTED_FILE}");
        let path = fixture_path(&relative);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("tests/fixtures/{relative}: {error}"));
        match gained_support_update(&text, &fixture) {
            Err(message) => panic!("{message}"),
            Ok(None) => eprintln!(
                "compat-freeze: {GAINED_SUPPORT_MODE}: no value recorded as unsupported emits bytes now; tests/fixtures/{relative} is unchanged"
            ),
            Ok(Some((updated, recorded))) => {
                write_atomically(&path, &updated);
                eprintln!(
                    "compat-freeze: {GAINED_SUPPORT_MODE} recorded the bytes of {} values in tests/fixtures/{relative}:\n  {}",
                    recorded.len(),
                    recorded.join("\n  ")
                );
            }
        }
        return;
    }
    if recording_fitted_cpp_revision() {
        let relative = format!("compat_freeze/{EMITTED_FILE}");
        let path = fixture_path(&relative);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("tests/fixtures/{relative}: {error}"));
        let sha = generation_sha();
        match fitted_cpp_revision_update(&text, &fixture, &sha) {
            Err(message) => panic!("{message}"),
            Ok(None) => eprintln!(
                "compat-freeze: {FITTED_CPP_REVISION_MODE}: no fitted C++17 value recorded as bytes changed; tests/fixtures/{relative} is unchanged"
            ),
            Ok(Some((updated, recorded))) => {
                write_atomically(&path, &updated);
                eprintln!(
                    "compat-freeze: {FITTED_CPP_REVISION_MODE} recorded {} fitted C++17 values at {sha} in tests/fixtures/{relative}:\n  {}",
                    recorded.len(),
                    recorded.join("\n  ")
                );
            }
        }
        return;
    }
    let Some(committed) = committed_or_write(Surface::Frozen, EMITTED_FILE, &fixture) else {
        return;
    };
    let check = check_emitted(&committed, &fixture);
    for note in &check.additive {
        eprintln!("compat-freeze: {note}");
    }
    if let Some(failure) = emitted_failure(&check) {
        panic!("{failure}");
    }
}

/// The rules for bytes where the emitted-bytes fixture records a refusal, proved on an in-memory
/// copy of the committed fixture; the file itself is only read. Compare mode fails, and names the
/// program, the language and the `gained-support` mode. That mode records the bytes and changes no
/// other byte of the file, and a second run records nothing. Bytes it recorded that change later
/// still fail, as do changed bytes recorded before, lost support and a refusal whose element kinds
/// change other than by narrowing; the mode records none of them. A narrowed refusal passes and
/// keeps its record.
#[test]
fn gained_support_fails_until_recorded_and_its_mode_changes_nothing_else() {
    fn first(programs: &[EmittedProgram], what: &str, wanted: impl Fn(&str) -> bool) -> usize {
        programs
            .iter()
            .position(|program| wanted(&program.cpp17))
            .unwrap_or_else(|| panic!("the fixture records no {what} to simulate with"))
    }
    let relative = format!("compat_freeze/{EMITTED_FILE}");
    let text = std::fs::read_to_string(fixture_path(&relative))
        .unwrap_or_else(|error| panic!("tests/fixtures/{relative}: {error}"));
    let committed: EmittedBytes = serde_json::from_str(&text).expect("the fixture parses");
    let key = |index: usize| format!("{}.cpp17", committed.programs[index].name);
    let names =
        |lines: &[String], index: usize| lines.iter().any(|line| line.contains(&key(index)));
    let quoted = |value: &str| serde_json::to_string(value).expect("quote");

    // The emitters' output, simulated: the fixture's values, except that the C++17 recipe emitter
    // now writes bytes for a program the fixture records as refused.
    let gained = first(&committed.programs, "C++17 recipe refusal", |value| {
        unsupported_kinds(value).is_some()
    });
    let refusal = committed.programs[gained].cpp17.clone();
    let bytes = text_identity("// the C++17 recipe source the emitter would now write\n");
    let mut actual = committed.clone();
    actual.programs[gained].cpp17.clone_from(&bytes);

    // Compare mode fails, and names the program, the language and the mode.
    let check = check_emitted(&committed, &actual);
    assert!(check.drift.is_empty(), "{:?}", check.drift);
    assert_eq!(check.gained.len(), 1, "{:?}", check.gained);
    let failure = emitted_failure(&check).expect("bytes for a refused value fail compare mode");
    for needle in [
        key(gained).as_str(),
        "C++17 recipe",
        "JOPE_COMPAT_FREEZE_REGENERATE=gained-support",
    ] {
        assert!(
            failure.contains(needle),
            "the failure does not name {needle:?}:\n{failure}"
        );
    }

    // The mode records those bytes and changes no other byte: the new text is the committed text
    // with the one line of that value replaced.
    let (updated_text, recorded) = gained_support_update(&text, &actual)
        .expect("no other record differs")
        .expect("one value gained support");
    assert_eq!(recorded, [key(gained)]);
    let old_lines: Vec<&str> = text.split('\n').collect();
    let new_lines: Vec<&str> = updated_text.split('\n').collect();
    assert_eq!(old_lines.len(), new_lines.len());
    let changed: Vec<(&str, &str)> = old_lines
        .iter()
        .zip(&new_lines)
        .filter(|(old, new)| old != new)
        .map(|(old, new)| (*old, *new))
        .collect();
    assert_eq!(changed.len(), 1, "{changed:#?}");
    let (old_line, new_line) = changed[0];
    assert!(
        old_line.trim_start().starts_with("\"cpp17\": "),
        "{old_line}"
    );
    assert_eq!(
        old_line.replace(&quoted(&refusal), &quoted(&bytes)),
        new_line
    );

    // Recorded, the fixture holds, and a second run records nothing.
    let updated: EmittedBytes = serde_json::from_str(&updated_text).expect("the new text parses");
    assert_eq!(emitted_failure(&check_emitted(&updated, &actual)), None);
    assert_eq!(gained_support_update(&updated_text, &actual), Ok(None));

    // The review's scenario: the bytes recorded this way change in a later PR. That fails, and
    // the mode refuses to record it.
    let mut later = actual.clone();
    later.programs[gained].cpp17 = text_identity("// that source, changed in a later PR\n");
    let failure = emitted_failure(&check_emitted(&updated, &later)).expect("changed bytes fail");
    assert!(failure.contains(&key(gained)), "{failure}");
    let refused = gained_support_update(&updated_text, &later).expect_err("recorded bytes stay");
    assert!(refused.contains(&key(gained)), "{refused}");

    // Bytes recorded before that change in the same run as a gain: compare mode names both, and
    // the mode writes nothing.
    let pinned = first(&committed.programs, "C++17 recipe bytes", is_emitted);
    let mut both = actual.clone();
    both.programs[pinned].cpp17 = text_identity("// a recorded source, changed\n");
    let failure = emitted_failure(&check_emitted(&committed, &both)).expect("both fail");
    assert!(
        failure.contains(&key(gained)) && failure.contains(&key(pinned)),
        "{failure}"
    );
    let refused = gained_support_update(&text, &both).expect_err("recorded bytes never change");
    assert!(refused.contains(&key(pinned)), "{refused}");

    // Lost support fails, and the mode does not record the refusal.
    let mut lost = committed.clone();
    lost.programs[pinned].cpp17 = format!("{UNSUPPORTED}transform Simulated");
    let check = check_emitted(&committed, &lost);
    assert!(names(&check.drift, pinned), "{:?}", check.drift);
    assert!(gained_support_update(&text, &lost).is_err());

    // A refusal narrowed to a strict subset of its element kinds passes, keeps its record and is
    // left alone by the mode; a refusal whose kinds change otherwise fails.
    let several = first(
        &committed.programs,
        "C++17 recipe refusal of several element kinds",
        |value| unsupported_kinds(value).is_some_and(|kinds| kinds.len() > 1),
    );
    let kept = unsupported_kinds(&committed.programs[several].cpp17)
        .and_then(|kinds| kinds.into_iter().next().map(str::to_string))
        .expect("an element kind");
    let mut narrowed = committed.clone();
    narrowed.programs[several].cpp17 = format!("{UNSUPPORTED}{kept}");
    let check = check_emitted(&committed, &narrowed);
    assert_eq!(emitted_failure(&check), None);
    assert!(names(&check.additive, several), "{:?}", check.additive);
    assert_eq!(gained_support_update(&text, &narrowed), Ok(None));
    let mut other_kinds = committed.clone();
    other_kinds.programs[several].cpp17 = format!("{UNSUPPORTED}{kept}; transform Simulated");
    let check = check_emitted(&committed, &other_kinds);
    assert!(names(&check.drift, several), "{:?}", check.drift);
    assert!(gained_support_update(&text, &other_kinds).is_err());
}

/// The rules for a new revision of the fitted C++17 emitter, proved on an in-memory copy of the
/// committed fixture; the file itself is only read, and no git is needed (the commit id is a
/// stand-in). A revision that changes every fitted source and codec fails compare mode. The
/// `fitted-cpp-revision` mode records exactly those values and the header's commit, changing no
/// other line of the file; the result compares clean, and a second run records nothing and keeps
/// the header. The mode refuses, and writes nothing, when anything else differs: a changed Rust
/// value, a fitted value that becomes a refusal, a skip that becomes bytes, a value recorded as
/// unsupported that now emits bytes (that stays `gained-support`'s), or a fixture text that is not
/// in the form this test writes.
#[test]
fn fitted_cpp_revision_records_only_fitted_values_and_changes_nothing_else() {
    let relative = format!("compat_freeze/{EMITTED_FILE}");
    let text = std::fs::read_to_string(fixture_path(&relative))
        .unwrap_or_else(|error| panic!("tests/fixtures/{relative}: {error}"));
    let committed: EmittedBytes = serde_json::from_str(&text).expect("the fixture parses");
    let sha = "0123456789abcdef0123456789abcdef01234567";
    assert_ne!(committed.generated_at_sha, sha);
    let find = |what: &str, wanted: &dyn Fn(&EmittedProgram) -> bool| {
        committed
            .programs
            .iter()
            .position(wanted)
            .unwrap_or_else(|| panic!("the fixture records no {what} to simulate with"))
    };

    // The emitters' output, simulated: the fixture's values, except that a new emitter revision
    // writes other bytes for every fitted source and codec recorded as bytes.
    let mut actual = committed.clone();
    let mut revised = BTreeSet::new();
    for program in &mut actual.programs {
        for (field, value) in [
            ("fitted_cpp17_source", &mut program.fitted_cpp17_source),
            ("fitted_cpp17_codec", &mut program.fitted_cpp17_codec),
        ] {
            if is_emitted(value) {
                *value = text_identity(&format!("// revised {field} of {}\n", program.name));
                revised.insert(format!("{}.{field}", program.name));
            }
        }
    }
    let fitted_programs = committed
        .programs
        .iter()
        .filter(|program| is_emitted(&program.fitted_cpp17_source))
        .count();
    assert!(fitted_programs > 0, "the fixture records fitted bytes");
    assert_eq!(revised.len(), 2 * fitted_programs);

    // Compare mode fails on each revised value, and on nothing else.
    let check = check_emitted(&committed, &actual);
    assert!(check.gained.is_empty(), "{:?}", check.gained);
    assert_eq!(check.drift.len(), revised.len(), "{:?}", check.drift);
    assert!(emitted_failure(&check).is_some());

    // The mode records exactly those values and the commit: the new text is the committed text
    // with those lines and the `generated_at_sha` line replaced.
    let (updated_text, recorded) = fitted_cpp_revision_update(&text, &actual, sha)
        .expect("no other record differs")
        .expect("the fitted values changed");
    assert_eq!(recorded.into_iter().collect::<BTreeSet<_>>(), revised);
    let old_lines: Vec<&str> = text.split('\n').collect();
    let new_lines: Vec<&str> = updated_text.split('\n').collect();
    assert_eq!(old_lines.len(), new_lines.len());
    let changed: Vec<(&str, &str)> = old_lines
        .iter()
        .zip(&new_lines)
        .filter(|(old, new)| old != new)
        .map(|(old, new)| (*old, *new))
        .collect();
    assert_eq!(changed.len(), revised.len() + 1, "{changed:#?}");
    let header: Vec<_> = changed
        .iter()
        .filter(|(old, _)| old.trim_start().starts_with("\"generated_at_sha\": "))
        .collect();
    assert_eq!(header.len(), 1, "{changed:#?}");
    assert!(header[0].1.contains(sha), "{:?}", header[0]);
    for (old, new) in &changed {
        let key = old.trim_start();
        assert!(
            key.starts_with("\"generated_at_sha\": ")
                || key.starts_with("\"fitted_cpp17_source\": ")
                || key.starts_with("\"fitted_cpp17_codec\": "),
            "a line the mode must not change: {old:?} became {new:?}"
        );
    }

    // Recorded, the fixture holds, and a second run records nothing and keeps the header.
    let updated: EmittedBytes = serde_json::from_str(&updated_text).expect("the new text parses");
    assert_eq!(updated.generated_at_sha, sha);
    assert_eq!(emitted_failure(&check_emitted(&updated, &actual)), None);
    assert_eq!(
        fitted_cpp_revision_update(
            &updated_text,
            &actual,
            "fedcba9876543210fedcba9876543210fedcba98"
        ),
        Ok(None)
    );
    // Nothing changed: nothing to record, and the header stays as it is.
    assert_eq!(fitted_cpp_revision_update(&text, &committed, sha), Ok(None));

    // Any other difference refuses the mode, naming the record.
    let fitted = find("fitted C++17 bytes", &|program| {
        is_emitted(&program.fitted_cpp17_source)
    });
    let fitted_name = &committed.programs[fitted].name;
    let mut rust = actual.clone();
    rust.programs[fitted].rust = text_identity("// a Rust program that changed too\n");
    let refused = fitted_cpp_revision_update(&text, &rust, sha).expect_err("a Rust value moved");
    assert!(
        refused.contains(&format!("{fitted_name}.rust")),
        "{refused}"
    );

    // A fitted value that becomes a refusal is lost support, not a revision.
    let mut lost = actual.clone();
    lost.programs[fitted].fitted_cpp17_source = format!("{UNSUPPORTED}head Simulated");
    let refused = fitted_cpp_revision_update(&text, &lost, sha).expect_err("lost support");
    assert!(
        refused.contains(&format!("{fitted_name}.fitted_cpp17_source")),
        "{refused}"
    );

    // A skip that becomes bytes is not a revision of recorded bytes.
    let skipped = find("fitted C++17 skip", &|program| {
        program.fitted_cpp17_source.starts_with("skipped")
    });
    let mut unskipped = actual.clone();
    unskipped.programs[skipped].fitted_cpp17_source =
        text_identity("// fitted bytes for a skipped program\n");
    let refused = fitted_cpp_revision_update(&text, &unskipped, sha).expect_err("a skip moved");
    assert!(
        refused.contains(&format!(
            "{}.fitted_cpp17_source",
            committed.programs[skipped].name
        )),
        "{refused}"
    );

    // Bytes for a value recorded as unsupported stay gained-support's to record.
    let refusal = find("C++17 recipe refusal", &|program| {
        unsupported_kinds(&program.cpp17).is_some()
    });
    let mut gained = actual.clone();
    gained.programs[refusal].cpp17 = text_identity("// the C++17 recipe source now written\n");
    let refused = fitted_cpp_revision_update(&text, &gained, sha).expect_err("gained support");
    assert!(
        refused.contains(GAINED_SUPPORT_MODE)
            && refused.contains(&format!("{}.cpp17", committed.programs[refusal].name)),
        "{refused}"
    );

    // A fixture text in another form would move bytes the mode must not touch.
    let reformatted = text.replacen("\n  ", "\n   ", 1);
    assert!(fitted_cpp_revision_update(&reformatted, &actual, sha).is_err());
}

// ---------------------------------------------------------------------------------------------
// WP-0.2 (d): serialized models
// ---------------------------------------------------------------------------------------------

/// Small models whose complete serialized bytes are stored, to prove stored bytes still load.
const STORED_MODELS: [&str; 5] = [
    "golden_map_derive_ridge",
    "golden_symbolic_affine",
    "expansion_spline",
    "target_zscore",
    "step_impute_mean",
];

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelSerde {
    schema: String,
    generated_at_sha: String,
    note: String,
    models: Vec<ModelRecord>,
    stored: Vec<StoredModel>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelRecord {
    name: String,
    serde_json: String,
    predictions: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredModel {
    name: String,
    json: String,
}

#[test]
fn serialized_models_match_the_frozen_fixture() {
    let (train, target, test) = fitted_surface_data();
    let mut models = Vec::new();
    let mut stored = Vec::new();
    for (name, source) in program_set() {
        let genome = parse_program(&name, &source);
        let (serde_json_identity, predictions) = match fit_skip_reason(&source) {
            Some(reason) => (reason.to_string(), reason.to_string()),
            None => match fit(&genome, &train, &target) {
                Ok(model) => {
                    let bytes = serde_json::to_vec(&model).expect("serialize model");
                    if STORED_MODELS.contains(&name.as_str()) {
                        stored.push(StoredModel {
                            name: name.clone(),
                            json: String::from_utf8(bytes.clone()).expect("UTF-8 JSON"),
                        });
                    }
                    (
                        format!("{} {}", bytes.len(), sha256_hex(&bytes)),
                        row_text(&model.predict(&test).unwrap()),
                    )
                }
                Err(error) => {
                    let error = format!("ERR {error:#}");
                    (error.clone(), error)
                }
            },
        };
        models.push(ModelRecord {
            name,
            serde_json: serde_json_identity,
            predictions,
        });
    }
    let fixture = ModelSerde {
        schema: "feat-math.compat-freeze.model-serde.v1".to_string(),
        generated_at_sha: header_sha(Surface::Frozen),
        note: format!(
            "serde_json is \"<byte length> <sha256>\" of serde_json::to_vec(&fit(program)) on {FITTED_DATA_NOTE}; predictions are the f64 bits of Model::predict on the held-out rows. stored keeps the complete bytes of five small models, which must still load and predict the recorded values (to within serde_json's float-parsing precision, which is not exact without its float_roundtrip feature)."
        ),
        models,
        stored,
    };
    let Some(committed) = committed_or_write(Surface::Frozen, FIXTURE_FILES[3], &fixture) else {
        return;
    };
    // Stored bytes from the generating build must still load and predict the recorded values.
    // Only to within parsing precision: serde_json parses floats exactly only with its
    // `float_roundtrip` feature, which feat-math does not enable (other workspace crates do, and
    // Cargo unifies it in multi-package builds), so a `-p feat-math` build reloads weights that
    // can differ in the last bits and predictions a few ulps away. A schema or semantic change
    // moves predictions far more than the bound.
    for stored in &committed.stored {
        let model: Model = serde_json::from_str(&stored.json).unwrap_or_else(|error| {
            panic!(
                "compat-freeze: stored model {} no longer deserializes ({error}); serialized models are a frozen surface.\n{REBASELINE_POLICY}",
                stored.name
            )
        });
        let recorded = committed
            .models
            .iter()
            .find(|m| m.name == stored.name)
            .expect("a stored model has a record");
        let reloaded = model.predict(&test).unwrap();
        let expected: Vec<f64> = recorded
            .predictions
            .split(' ')
            .map(|token| f64::from_bits(u64::from_str_radix(token, 16).expect("hex bits")))
            .collect();
        assert_eq!(
            reloaded.len(),
            expected.len(),
            "{}: prediction count",
            stored.name
        );
        for (i, (now, then)) in reloaded.iter().zip(&expected).enumerate() {
            assert!(
                (now - then).abs() <= 1e-9 * now.abs().max(then.abs()).max(1.0),
                "compat-freeze: stored model {} loads but row {i} predicts {now:?}, recorded {then:?}.\n{REBASELINE_POLICY}",
                stored.name
            );
        }
    }
    let flatten = |values: &ModelSerde| -> Vec<(String, String)> {
        let mut records = vec![record("schema", values.schema.clone())];
        for m in &values.models {
            records.push(record(
                format!("{}.serde_json", m.name),
                m.serde_json.clone(),
            ));
            records.push(record(
                format!("{}.predictions", m.name),
                m.predictions.clone(),
            ));
        }
        for s in &values.stored {
            records.push(record(
                format!("{}.stored_json", s.name),
                text_identity(&s.json),
            ));
        }
        records
    };
    assert_records(
        "serialized models (model_serde.v1.json)",
        &flatten(&committed),
        &flatten(&fixture),
        REBASELINE_POLICY,
    );
}

// ---------------------------------------------------------------------------------------------
// WP-0.2 (e): vocabulary, registry, lineage and emitter-version pins (never rebaselined)
// ---------------------------------------------------------------------------------------------

const EMITTED_PIN_WHY: &str = "an emitters-lane pin (deploy/ci/emitters.sh, jope-motifs validate.rs); changing it changes shipped artifact bytes";

/// Files whose bytes are frozen, with what a change to each one breaks.
const FROZEN_FILES: &[(&str, &str)] = &[
    ("crates/jope-engine/assets/grammar/featmath.json", "the v3 action vocabulary the step-60000 policy checkpoint is bound to (jope-policy CHECKPOINT_VOCABULARY_SHA256, parity.json vocab_sha256); a new vocabulary is a new file, never an edit"),
    ("crates/jope-engine/assets/grammar/featmath-jain-v1.json", "the product vocabulary (jope-policy PRODUCT_VOCABULARY_SHA256)"),
    ("crates/jope-engine/assets/grammar/parity.json", "the v3 automaton parity fixtures (jope-engine tests/grammar.rs)"),
    ("crates/jope-engine/assets/grammar/jain-math-executable-9fa594e2.json", "the jain registry snapshot (ADR-0002 registry_dump byte check; featmath-jain-v1 restriction.executable_ops_sha256)"),
    ("crates/jope-engine/src/grammar.rs", "this changes the teaching/corpus lineage: teaching_store.rs and corpus.rs hash include_str!(\"grammar.rs\") into every lineage (lineage_sources.v1.json lists every hashed source); put new automaton code in a new module"),
    ("crates/jope-policy/release/policy-step-60000.semantic-v1.json", "the accepted semantic manifest (jope-policy ACCEPTED_SEMANTIC_MANIFEST_SHA256)"),
    ("crates/feat-math/src/registry.rs", "the wire registry (registry_hash) and the line numbers registry_dump recomputes for the ADR-0002 snapshot check; the registry is not edited this release (plan OQ-1)"),
    ("crates/feat-math/assets/champions/8944cbdf.dsl", "the champion program (CHAMPION_GENOME)"),
    ("crates/feat-math/tests/fixtures/emitted/8944cbdf.predictions.sha256", EMITTED_PIN_WHY),
    ("crates/feat-math/tests/fixtures/emitted/8944cbdf.py.sha256", EMITTED_PIN_WHY),
    ("crates/feat-math/tests/fixtures/emitted/8944cbdf.rs.sha256", EMITTED_PIN_WHY),
    ("crates/feat-math/tests/fixtures/emitted/8944cbdf.targets.sha256", EMITTED_PIN_WHY),
    ("crates/feat-math/tests/fixtures/emitted/portable-4lang.cpp.sha256", EMITTED_PIN_WHY),
    ("crates/feat-math/tests/fixtures/emitted/portable-4lang.dsl", EMITTED_PIN_WHY),
    ("crates/feat-math/tests/fixtures/emitted/portable-4lang.js.sha256", EMITTED_PIN_WHY),
    ("crates/feat-math/tests/fixtures/emitted/portable-4lang.predictions.sha256", EMITTED_PIN_WHY),
    ("crates/feat-math/tests/fixtures/emitted/portable-4lang.py.sha256", EMITTED_PIN_WHY),
    ("crates/feat-math/tests/fixtures/emitted/portable-4lang.rs.sha256", EMITTED_PIN_WHY),
    ("crates/feat-math/tests/fixtures/emitted/portable-4lang.targets.sha256", EMITTED_PIN_WHY),
    ("crates/feat-math/tests/fixtures/fitted_runtime_oracle.cpp", "the frozen C++ parity oracle identity recorded in fitted containers (runtime_source_sha256)"),
];

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrozenPins {
    schema: String,
    generated_at_sha: String,
    files: Vec<FilePin>,
    values: Vec<ValuePin>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FilePin {
    path: String,
    sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ValuePin {
    name: String,
    value: String,
}

/// The string literal assigned to `pub const <name>: &str =` in Rust source text.
fn const_literal(source: &str, name: &str) -> String {
    let marker = format!("pub const {name}: &str =");
    let start = source
        .find(&marker)
        .unwrap_or_else(|| panic!("{marker} not found"))
        + marker.len();
    let rest = source[start..].trim_start();
    let rest = rest
        .strip_prefix('"')
        .unwrap_or_else(|| panic!("{name} is not a string literal"));
    rest[..rest.find('"').expect("closing quote")].to_string()
}

fn json_file(relative: &str) -> serde_json::Value {
    serde_json::from_slice(&read_repo_file(relative)).unwrap_or_else(|e| panic!("{relative}: {e}"))
}

fn json_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

#[test]
fn vocabulary_registry_and_lineage_pins_match() {
    let files: Vec<FilePin> = FROZEN_FILES
        .iter()
        .map(|(path, _)| FilePin {
            path: path.to_string(),
            sha256: sha256_hex(&read_repo_file(path)),
        })
        .collect();
    let file_sha = |path: &str| -> String {
        files
            .iter()
            .find(|pin| pin.path == path)
            .expect("pinned file")
            .sha256
            .clone()
    };
    let policy_lib = String::from_utf8(read_repo_file("crates/jope-policy/src/lib.rs")).unwrap();
    let policy_semantic =
        String::from_utf8(read_repo_file("crates/jope-policy/src/semantic.rs")).unwrap();
    let featmath = json_file("crates/jope-engine/assets/grammar/featmath.json");
    let jain = json_file("crates/jope-engine/assets/grammar/featmath-jain-v1.json");
    let parity = json_file("crates/jope-engine/assets/grammar/parity.json");
    let snapshot =
        json_file("crates/jope-engine/assets/grammar/jain-math-executable-9fa594e2.json");
    let manifest = json_file("crates/jope-policy/release/policy-step-60000.semantic-v1.json");
    let checkpoint = const_literal(&policy_lib, "CHECKPOINT_VOCABULARY_SHA256");
    let product = const_literal(&policy_lib, "PRODUCT_VOCABULARY_SHA256");
    let config = const_literal(&policy_lib, "ACCEPTED_CONFIG_SHA256");
    let semantic = const_literal(&policy_semantic, "ACCEPTED_SEMANTIC_MANIFEST_SHA256");
    let restriction = &jain["restriction"];
    let values: Vec<ValuePin> = [
        ("registry_hash", registry_hash().to_string()),
        ("registry_engine", REGISTRY_ENGINE.to_string()),
        ("registry_version", REGISTRY_VERSION.to_string()),
        (
            "registry_operator_count",
            operator_registry().len().to_string(),
        ),
        ("transpile::EMITTER_VERSION", EMITTER_VERSION.to_string()),
        ("advanced_v3::ARTIFACT_SCHEMA", ARTIFACT_SCHEMA.to_string()),
        (
            "jope_policy::CHECKPOINT_VOCABULARY_SHA256",
            checkpoint.clone(),
        ),
        ("jope_policy::PRODUCT_VOCABULARY_SHA256", product.clone()),
        ("jope_policy::ACCEPTED_CONFIG_SHA256", config.clone()),
        (
            "jope_policy::ACCEPTED_SEMANTIC_MANIFEST_SHA256",
            semantic.clone(),
        ),
        ("featmath.json version", json_text(&featmath["version"])),
        (
            "featmath.json token count",
            featmath["tokens"]
                .as_array()
                .expect("tokens")
                .len()
                .to_string(),
        ),
        (
            "parity.json vocab_sha256",
            json_text(&parity["vocab_sha256"]),
        ),
        (
            "parity.json fixture count",
            parity["fixtures"]
                .as_array()
                .expect("fixtures")
                .len()
                .to_string(),
        ),
        (
            "featmath-jain-v1.json restriction.base_vocabulary_sha256",
            json_text(&restriction["base_vocabulary_sha256"]),
        ),
        (
            "featmath-jain-v1.json restriction.executable_ops_sha256",
            json_text(&restriction["executable_ops_sha256"]),
        ),
        (
            "featmath-jain-v1.json restriction.jain_math_registry_hash",
            json_text(&restriction["jain_math_registry_hash"]),
        ),
        (
            "jain snapshot provenance.registry_hash",
            json_text(&snapshot["provenance"]["registry_hash"]),
        ),
        (
            "semantic manifest vocabulary.checkpoint_sha256",
            json_text(&manifest["vocabulary"]["checkpoint_sha256"]),
        ),
        (
            "semantic manifest vocabulary.product_restriction.sha256",
            json_text(&manifest["vocabulary"]["product_restriction"]["sha256"]),
        ),
        (
            "semantic manifest vocabulary.tokens",
            json_text(&manifest["vocabulary"]["tokens"]),
        ),
        (
            "semantic manifest policy.config_sha256",
            json_text(&manifest["policy"]["config_sha256"]),
        ),
    ]
    .into_iter()
    .map(|(name, value)| ValuePin {
        name: name.to_string(),
        value,
    })
    .collect();

    // The pin chain must be internally consistent in every build, fixture or not.
    let featmath_sha = file_sha("crates/jope-engine/assets/grammar/featmath.json");
    let jain_sha = file_sha("crates/jope-engine/assets/grammar/featmath-jain-v1.json");
    let value = |name: &str| -> String {
        values
            .iter()
            .find(|pin| pin.name == name)
            .expect("value pin")
            .value
            .clone()
    };
    for (name, expected) in [
        ("jope_policy::CHECKPOINT_VOCABULARY_SHA256", &featmath_sha),
        ("parity.json vocab_sha256", &featmath_sha),
        (
            "featmath-jain-v1.json restriction.base_vocabulary_sha256",
            &featmath_sha,
        ),
        (
            "semantic manifest vocabulary.checkpoint_sha256",
            &featmath_sha,
        ),
        ("jope_policy::PRODUCT_VOCABULARY_SHA256", &jain_sha),
        (
            "semantic manifest vocabulary.product_restriction.sha256",
            &jain_sha,
        ),
    ] {
        assert_eq!(&value(name), expected, "pin chain: {name}");
    }
    assert_eq!(
        value("featmath-jain-v1.json restriction.executable_ops_sha256"),
        file_sha("crates/jope-engine/assets/grammar/jain-math-executable-9fa594e2.json"),
        "pin chain: executable_ops_sha256"
    );
    for name in [
        "featmath-jain-v1.json restriction.jain_math_registry_hash",
        "jain snapshot provenance.registry_hash",
    ] {
        assert_eq!(value(name), registry_hash(), "pin chain: {name}");
    }
    assert_eq!(
        value("jope_policy::ACCEPTED_SEMANTIC_MANIFEST_SHA256"),
        file_sha("crates/jope-policy/release/policy-step-60000.semantic-v1.json"),
        "pin chain: the semantic manifest"
    );
    assert_eq!(
        value("semantic manifest policy.config_sha256"),
        config,
        "pin chain: config"
    );
    assert_eq!(
        value("semantic manifest vocabulary.tokens"),
        value("featmath.json token count"),
        "pin chain: token count"
    );

    let fixture = FrozenPins {
        schema: "feat-math.compat-freeze.frozen-pins.v1".to_string(),
        generated_at_sha: header_sha(Surface::Frozen),
        files,
        values,
    };
    let Some(committed) = committed_or_write(Surface::Frozen, FIXTURE_FILES[4], &fixture) else {
        return;
    };
    let mut problems = Vec::new();
    for pin in &committed.files {
        let why = FROZEN_FILES
            .iter()
            .find(|(path, _)| *path == pin.path)
            .map(|(_, why)| *why)
            .unwrap_or("no longer listed in FROZEN_FILES");
        match fixture.files.iter().find(|now| now.path == pin.path) {
            Some(now) if now.sha256 == pin.sha256 => {}
            Some(now) => problems.push(format!(
                "{} changed (fixture {}, now {}): {why}",
                pin.path, pin.sha256, now.sha256
            )),
            None => problems.push(format!("{} is no longer pinned: {why}", pin.path)),
        }
    }
    for now in &fixture.files {
        if !committed.files.iter().any(|pin| pin.path == now.path) {
            problems.push(format!(
                "{} is pinned here but not in the fixture",
                now.path
            ));
        }
    }
    let old: Vec<(String, String)> = committed
        .values
        .iter()
        .map(|pin| (pin.name.clone(), pin.value.clone()))
        .collect();
    let new: Vec<(String, String)> = fixture
        .values
        .iter()
        .map(|pin| (pin.name.clone(), pin.value.clone()))
        .collect();
    if old != new {
        for (name, value) in &old {
            match new.iter().find(|(n, _)| n == name) {
                Some((_, now)) if now == value => {}
                Some((_, now)) => problems.push(format!("{name}: fixture {value}, now {now}")),
                None => problems.push(format!("{name} is no longer recorded")),
            }
        }
        for (name, value) in &new {
            if !old.iter().any(|(n, _)| n == name) {
                problems.push(format!("{name} = {value} is new"));
            }
        }
    }
    assert!(
        problems.is_empty(),
        "compat-freeze: frozen pins moved. These bytes and values are never rebaselined (plan N1); revert the change.\n  {}",
        problems.join("\n  ")
    );
}

// ---------------------------------------------------------------------------------------------
// WP-0.2 (e): ADR-0004 evaluator identity inputs (rebaselined on purpose by identity changes)
// ---------------------------------------------------------------------------------------------

const EVALUATOR_SOURCE: &str = "crates/jope-campaigns/src/evaluation_v2.rs";
const KNOWN_IDENTITIES: &str = "crates/jope-campaigns/known-evaluator-identities.json";
/// How an announced surface is rebaselined: in the change that moves it, after committing it.
fn announced_rebaseline() -> String {
    format!(
        "Rebaseline in the same change, after committing it, with `{IDENTITY_REGENERATE_COMMAND}` \
         (the commit id is the one that carries the change). That rewrites only the announced \
         identity fixtures and still compares every other surface: no other fixture may change in \
         an identity-only rebaseline."
    )
}

fn identity_advice() -> String {
    format!(
        "This change mints a new ADR-0004 evaluator identity (a Cargo.lock change does too: the \
         identity hashes it). Announce it in the PR body (old and new implementation_sha256 and \
         identity digest) and record it as docs/adr/ADR-0004-evaluator-identity-known-table.md \
         requires (crates/jope-campaigns/known-evaluator-identities.json rows are appended only \
         with the receipt that proves an identity existed). {}",
        announced_rebaseline()
    )
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluatorIdentity {
    schema: String,
    generated_at_sha: String,
    note: String,
    domain: String,
    inputs: Vec<IdentityInput>,
    implementation_sha256: String,
    dsl_abi_sha256: String,
    cargo_lock_sha256: String,
    identity_digest: String,
    known_identities_sha256: String,
    known_identity_rows: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IdentityInput {
    label: String,
    path: String,
    sha256: String,
}

/// `evaluation_v2.rs` `SemanticHasher`: every label and value is framed by its u64 LE length.
struct SemanticHasher(Sha256);

impl SemanticHasher {
    fn new(domain: &str) -> Self {
        let mut hasher = Self(Sha256::new());
        hasher.bytes("domain", domain.as_bytes());
        hasher
    }

    fn bytes(&mut self, label: &str, value: &[u8]) {
        self.0.update((label.len() as u64).to_le_bytes());
        self.0.update(label.as_bytes());
        self.0.update((value.len() as u64).to_le_bytes());
        self.0.update(value);
    }

    fn finish(self) -> String {
        self.0
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

/// String literals in `text`, with the offset of each opening quote.
fn string_literals(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut chars = text.char_indices();
    while let Some((start, ch)) = chars.next() {
        if ch != '"' {
            continue;
        }
        let mut literal = String::new();
        while let Some((_, c)) = chars.next() {
            match c {
                '\\' => {
                    if let Some((_, escaped)) = chars.next() {
                        literal.push(escaped);
                    }
                }
                '"' => break,
                other => literal.push(other),
            }
        }
        out.push((start, literal));
    }
    out
}

/// Lexically normalize `base/relative` into a repository-relative path.
fn normalize(base: &str, relative: &str) -> String {
    let mut out = PathBuf::new();
    for component in Path::new(base).join(relative).components() {
        match component {
            Component::ParentDir => {
                assert!(out.pop(), "{relative} leaves the repository");
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out.to_string_lossy().replace('\\', "/")
}

/// The hashed `(label, include_bytes! path)` list, read from `compute_implementation_sha256`.
fn evaluator_inputs(source: &str) -> (String, Vec<(String, String)>) {
    let start = source
        .find("fn compute_implementation_sha256()")
        .expect("compute_implementation_sha256 is defined");
    let body = &source[start..];
    let body = &body[..body.find("hash.finish()").expect("hash.finish()")];
    let mut domain = None;
    let mut pending = None;
    let mut inputs = Vec::new();
    for (offset, literal) in string_literals(body) {
        let before = body[..offset].trim_end();
        if before.ends_with("include_bytes!(") {
            let label = pending
                .take()
                .unwrap_or_else(|| panic!("no label before include_bytes!(\"{literal}\")"));
            inputs.push((label, normalize("crates/jope-campaigns/src", &literal)));
        } else if before.ends_with("SemanticHasher::new(") {
            domain = Some(literal);
        } else {
            pending = Some(literal);
        }
    }
    (domain.expect("hasher domain"), inputs)
}

#[test]
fn evaluator_identity_inputs_match_the_recorded_identity() {
    let source = String::from_utf8(read_repo_file(EVALUATOR_SOURCE)).unwrap();
    let (domain, listed) = evaluator_inputs(&source);
    assert_eq!(domain, "jope-dime/evaluator-implementation/v1");
    let mut hasher = SemanticHasher::new(&domain);
    let mut inputs = Vec::new();
    for (label, path) in &listed {
        let bytes = read_repo_file(path);
        hasher.bytes(label, &bytes);
        inputs.push(IdentityInput {
            label: label.clone(),
            path: path.clone(),
            sha256: sha256_hex(&bytes),
        });
    }
    let implementation_sha256 = hasher.finish();
    let mut dsl = SemanticHasher::new("jope-dime/dsl-abi/v1");
    dsl.bytes(
        "value",
        b"vendored-feat-math.GenomeIr+fit+Model::predict.v1",
    );
    let dsl_abi_sha256 = dsl.finish();
    let cargo_lock_sha256 = sha256_hex(&read_repo_file("Cargo.lock"));
    for literal in [
        "\"JopeDimeEvaluatorIdentity.v1\"",
        "\"prime-jain.native-feat-math-f64.evaluation-v2.v1\"",
        "\"jope-dime/evaluator-identity/v1\"",
        "\"jope-dime/dsl-abi/v1\"",
        "b\"vendored-feat-math.GenomeIr+fit+Model::predict.v1\"",
    ] {
        assert!(
            source.contains(literal),
            "{EVALUATOR_SOURCE} no longer contains {literal}; update this replica of the identity"
        );
    }
    let mut identity = SemanticHasher::new("jope-dime/evaluator-identity/v1");
    for (label, value) in [
        ("schema", "JopeDimeEvaluatorIdentity.v1"),
        (
            "evaluator_abi",
            "prime-jain.native-feat-math-f64.evaluation-v2.v1",
        ),
        ("implementation_sha256", implementation_sha256.as_str()),
        ("dsl_abi_sha256", dsl_abi_sha256.as_str()),
        ("cargo_lock_sha256", cargo_lock_sha256.as_str()),
    ] {
        identity.bytes(label, value.as_bytes());
    }
    let identity_digest = identity.finish();

    // The runtime table parser silently drops malformed rows; refuse them here instead.
    let known_bytes = read_repo_file(KNOWN_IDENTITIES);
    let known: serde_json::Value = serde_json::from_slice(&known_bytes).expect("known identities");
    assert_eq!(known["schema"], "JopeDimeKnownEvaluatorIdentities.v1");
    let rows = known["identities"].as_array().expect("identities array");
    for (i, row) in rows.iter().enumerate() {
        for field in ["implementation_sha256", "cargo_lock_sha256"] {
            let value = row[field].as_str().unwrap_or_default();
            assert!(
                is_hex(value, 64),
                "{KNOWN_IDENTITIES} row {i}: {field} must be 64 lowercase hex digits (the runtime parser would silently drop this row)"
            );
        }
    }

    let fixture = EvaluatorIdentity {
        schema: "feat-math.compat-freeze.evaluator-identity.v1".to_string(),
        generated_at_sha: header_sha(Surface::Announced),
        note: "A replica of evaluation_v2.rs compute_implementation_sha256 (the hashed list is read from that function) and EvaluatorIdentityV1::digest. Planned changes move this fixture: the change that mints the identity announces it and rebaselines it with JOPE_COMPAT_FREEZE_REGENERATE=identity.".to_string(),
        domain,
        inputs,
        implementation_sha256,
        dsl_abi_sha256,
        cargo_lock_sha256,
        identity_digest,
        known_identities_sha256: sha256_hex(&known_bytes),
        known_identity_rows: rows.len(),
    };
    let Some(committed) = committed_or_write(Surface::Announced, FIXTURE_FILES[5], &fixture) else {
        return;
    };
    let mut problems = Vec::new();
    let old_labels: Vec<&str> = committed.inputs.iter().map(|i| i.label.as_str()).collect();
    let new_labels: Vec<&str> = fixture.inputs.iter().map(|i| i.label.as_str()).collect();
    if old_labels != new_labels {
        problems.push(format!(
            "the hashed input list changed:\n      fixture {old_labels:?}\n      now     {new_labels:?}"
        ));
    }
    for input in &committed.inputs {
        if let Some(now) = fixture.inputs.iter().find(|i| i.label == input.label) {
            if now.path != input.path || now.sha256 != input.sha256 {
                problems.push(format!(
                    "{} ({}) changed: fixture {}, now {} ({})",
                    input.label, input.path, input.sha256, now.sha256, now.path
                ));
            }
        }
    }
    for (name, old, new) in [
        (
            "implementation_sha256",
            &committed.implementation_sha256,
            &fixture.implementation_sha256,
        ),
        (
            "dsl_abi_sha256",
            &committed.dsl_abi_sha256,
            &fixture.dsl_abi_sha256,
        ),
        (
            "cargo_lock_sha256",
            &committed.cargo_lock_sha256,
            &fixture.cargo_lock_sha256,
        ),
        (
            "identity_digest",
            &committed.identity_digest,
            &fixture.identity_digest,
        ),
        (
            "known identity table sha256",
            &committed.known_identities_sha256,
            &fixture.known_identities_sha256,
        ),
        ("domain", &committed.domain, &fixture.domain),
    ] {
        if old != new {
            problems.push(format!("{name}: fixture {old}, now {new}"));
        }
    }
    if committed.known_identity_rows != fixture.known_identity_rows {
        problems.push(format!(
            "known identity rows: fixture {}, now {}",
            committed.known_identity_rows, fixture.known_identity_rows
        ));
    }
    assert!(
        problems.is_empty(),
        "compat-freeze: the ADR-0004 evaluator identity inputs changed:\n  {}\n{}",
        problems.join("\n  "),
        identity_advice()
    );
}

// ---------------------------------------------------------------------------------------------
// Lineage sources: what jope-engine hashes into its identities (rebaselined on purpose)
// ---------------------------------------------------------------------------------------------

/// Where jope-engine hashes source files into an identity: the identity, the file that hashes,
/// the text that opens the hashed `include_str!`/`include_bytes!` list and the text after it, and
/// what a change to a listed source does.
const LINEAGES: &[(&str, &str, &str, &str, &str)] = &[
    (
        "teaching store semantic_dependencies_sha256",
        "crates/jope-engine/src/teaching_store.rs",
        "let semantic_sources = concat!(",
        ");",
        "every teaching store packed afterwards gets a new lineage",
    ),
    (
        "corpus scan_policy",
        "crates/jope-engine/src/corpus.rs",
        "let scan_policy = hash(",
        "file_hash(Path::new(",
        "every corpus version written afterwards gets a new scan policy",
    ),
    (
        "eval-registry query execution identity",
        "crates/jope-engine/src/eval_registry/query.rs",
        "fn execution_query_key(",
        "\n}",
        "query keys change, and verify_artifacts refuses every query receipt written before it (it compares these source hashes)",
    ),
];

const LINEAGE_NOTE: &str = "Source files that jope-engine hashes into identities, read from the include_str!/include_bytes! lists in the named code: the teaching-store semantic_dependencies_sha256 (teaching_store.rs), the corpus scan_policy (corpus.rs) and the eval-registry query execution identity (eval_registry/query.rs execution_query_key; verify_artifacts refuses receipts whose recorded source hashes differ). Each identity also hashes the running executable (/proc/self/exe), so any change linked into that binary, every feat-math change included, mints new identities even when no listed source changes; this fixture makes the listed sources visible. A change here is announced in the PR body and rebaselined with JOPE_COMPAT_FREEZE_REGENERATE=identity in the same change.";

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LineageSources {
    schema: String,
    generated_at_sha: String,
    note: String,
    lineages: Vec<Lineage>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Lineage {
    name: String,
    hashed_in: String,
    sources: Vec<FilePin>,
}

/// The repository paths of the `include_str!`/`include_bytes!` literals between `start` and the
/// next `end` in `file`'s text, resolved against `file`'s directory.
fn hashed_sources(text: &str, file: &str, start: &str, end: &str) -> Vec<String> {
    let from = text
        .find(start)
        .unwrap_or_else(|| panic!("{file}: `{start}` not found; update LINEAGES"));
    let body = &text[from..];
    let body = &body[..body
        .find(end)
        .unwrap_or_else(|| panic!("{file}: `{end}` not found after `{start}`"))];
    let dir = Path::new(file)
        .parent()
        .expect("a file in a directory")
        .to_string_lossy()
        .into_owned();
    string_literals(body)
        .into_iter()
        .filter(|(offset, _)| {
            let before = body[..*offset].trim_end();
            before.ends_with("include_str!(") || before.ends_with("include_bytes!(")
        })
        .map(|(_, literal)| normalize(&dir, &literal))
        .collect()
}

#[test]
fn lineage_sources_match_the_recorded_sources() {
    let lineages: Vec<Lineage> = LINEAGES
        .iter()
        .map(|(name, file, start, end, _)| {
            let text = String::from_utf8(read_repo_file(file)).expect("UTF-8 source");
            let sources = hashed_sources(&text, file, start, end);
            assert!(
                !sources.is_empty(),
                "{file}: no hashed sources found for {name}"
            );
            Lineage {
                name: name.to_string(),
                hashed_in: file.to_string(),
                sources: sources
                    .into_iter()
                    .map(|path| FilePin {
                        sha256: sha256_hex(&read_repo_file(&path)),
                        path,
                    })
                    .collect(),
            }
        })
        .collect();
    let fixture = LineageSources {
        schema: "feat-math.compat-freeze.lineage-sources.v1".to_string(),
        generated_at_sha: header_sha(Surface::Announced),
        note: LINEAGE_NOTE.to_string(),
        lineages,
    };
    let Some(committed) = committed_or_write(Surface::Announced, FIXTURE_FILES[8], &fixture) else {
        return;
    };
    let mut problems = Vec::new();
    for old in &committed.lineages {
        let consequence = LINEAGES
            .iter()
            .find(|lineage| lineage.0 == old.name)
            .map_or("it is no longer listed in LINEAGES", |lineage| lineage.4);
        let Some(new) = fixture.lineages.iter().find(|l| l.name == old.name) else {
            problems.push(format!("{} is no longer checked", old.name));
            continue;
        };
        let old_paths: Vec<&str> = old.sources.iter().map(|pin| pin.path.as_str()).collect();
        let new_paths: Vec<&str> = new.sources.iter().map(|pin| pin.path.as_str()).collect();
        if old_paths != new_paths {
            problems.push(format!(
                "{} ({}) hashes a different source list, so {consequence}:\n      fixture {old_paths:?}\n      now     {new_paths:?}",
                old.name, new.hashed_in
            ));
        }
        for pin in &old.sources {
            if let Some(now) = new.sources.iter().find(|p| p.path == pin.path) {
                if now.sha256 != pin.sha256 {
                    problems.push(format!(
                        "{} changed (fixture {}, now {}): {} hashes it, so {consequence}",
                        pin.path, pin.sha256, now.sha256, old.name
                    ));
                }
            }
        }
    }
    for new in &fixture.lineages {
        if !committed.lineages.iter().any(|l| l.name == new.name) {
            problems.push(format!("{} is new", new.name));
        }
    }
    assert!(
        problems.is_empty(),
        "compat-freeze: sources that jope-engine hashes into its identities changed:\n  {}\nAnnounce the new identities in the PR body. {}",
        problems.join("\n  "),
        announced_rebaseline()
    );
}

// ---------------------------------------------------------------------------------------------
// WP-0.2 (f): seeded generator streams and their vocabularies
// ---------------------------------------------------------------------------------------------

const SEEDS: std::ops::Range<u64> = 0..64;
const EVOLVE_SOURCE: &str = include_str!("../src/evolve.rs");

/// Parent programs for the stage-list generators: the champion, a plain pipeline, the golden
/// map/derive pipeline (expression strings must be carried unchanged), an expansion starter, and
/// an elasticnet pipeline whose `l1_ratio` reaches the logit branch of the v2 parameter jitter.
fn stream_parents() -> Vec<String> {
    vec![
        GenomeIr::parse(CHAMPION_GENOME).unwrap().canonical(),
        "(pipe X (standardize) (ridge))".to_string(),
        NAMED_PROGRAMS[0].1.to_string(),
        EXPANSION_STARTERS[2].to_string(),
        "(pipe X (winsor_clip q=0.05) (standardize) (elasticnet alpha=0.001 l1_ratio=0.5))"
            .to_string(),
    ]
}

/// The first four stream parents, which seed the advanced_v3 initial population.
const V3_SEED_PARENTS: usize = 4;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratorStreams {
    schema: String,
    generated_at_sha: String,
    note: String,
    vocabularies: Vec<Vocabulary>,
    streams: Vec<StreamRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Vocabulary {
    name: String,
    items: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamRecord {
    name: String,
    inputs: String,
    count: usize,
    sha256: String,
    first: String,
    per_item: String,
}

fn stream(name: &str, inputs: &str, items: Vec<String>) -> (StreamRecord, Vec<String>) {
    let record = StreamRecord {
        name: name.to_string(),
        inputs: inputs.to_string(),
        count: items.len(),
        sha256: sha256_hex(items.join("\n").as_bytes()),
        first: items.first().cloned().unwrap_or_default(),
        per_item: items
            .iter()
            .map(|item| sha256_hex(item.as_bytes())[..16].to_string())
            .collect::<Vec<_>>()
            .join(" "),
    };
    (record, items)
}

fn shown<E: std::fmt::Display>(result: Result<String, E>) -> String {
    result.unwrap_or_else(|error| format!("ERR {error}"))
}

fn evolved(result: Result<EvolvedGenome, String>) -> String {
    match result {
        Ok(g) => format!(
            "{} | {} | {} | {}",
            g.canonical, g.semantic_sha256, g.complexity, g.rust_source_len
        ),
        Err(error) => format!("ERR {error}"),
    }
}

/// A private `[&str; N]` array of `evolve.rs`, read from the source text.
fn source_array(header: &str) -> Vec<String> {
    let start = EVOLVE_SOURCE.find(header).unwrap_or_else(|| {
        panic!("`{header}` not found in src/evolve.rs: the legacy arrays must not change (their lengths are the modulus of every recorded stream)")
    }) + header.len();
    let end = start + EVOLVE_SOURCE[start..].find("];").expect("array end");
    string_literals(&EVOLVE_SOURCE[start..end])
        .into_iter()
        .map(|(_, literal)| literal)
        .collect()
}

fn vocabularies() -> Vec<Vocabulary> {
    let list = |items: &[&str]| items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    vec![
        Vocabulary {
            name: "evolve::TRANSFORMS (private)".to_string(),
            items: source_array("const TRANSFORMS: [&str; 18] = ["),
        },
        Vocabulary {
            name: "evolve::HEADS (private)".to_string(),
            items: source_array("const HEADS: [&str; 9] = ["),
        },
        Vocabulary {
            name: "evolve::STARTERS (private)".to_string(),
            items: source_array("const STARTERS: [&str; 8] = ["),
        },
        Vocabulary {
            name: "evolve::HARD_TEN_STARTERS (private)".to_string(),
            items: source_array("const HARD_TEN_STARTERS: [&str; 8] = ["),
        },
        Vocabulary {
            name: "evolve::EXPANSION_TRANSFORMS".to_string(),
            items: list(&EXPANSION_TRANSFORMS),
        },
        Vocabulary {
            name: "evolve::HEAD_OPS".to_string(),
            items: list(&HEAD_OPS),
        },
        Vocabulary {
            name: "evolve::EXPANSION_STARTERS".to_string(),
            items: list(&EXPANSION_STARTERS),
        },
        Vocabulary {
            name: "evolve::LITE_HEAD_BUDGETS".to_string(),
            items: LITE_HEAD_BUDGETS
                .iter()
                .map(|(head, budget)| format!("{head} {budget:?}"))
                .collect(),
        },
        Vocabulary {
            name: "full_rust::TRANSFORMS".to_string(),
            items: list(&full_rust::TRANSFORMS),
        },
        Vocabulary {
            name: "full_rust::EXPANSION_TRANSFORMS".to_string(),
            items: list(&full_rust::EXPANSION_TRANSFORMS),
        },
        Vocabulary {
            name: "full_rust::HEADS".to_string(),
            items: list(&full_rust::HEADS),
        },
        Vocabulary {
            name: "full_rust::TARGET_KINDS".to_string(),
            items: list(&full_rust::TARGET_KINDS),
        },
        Vocabulary {
            name: "full_rust::VALID_TARGET_KINDS".to_string(),
            items: list(&full_rust::VALID_TARGET_KINDS),
        },
        Vocabulary {
            name: "full_rust limits (MAX_TRANSFORMS, MAX_EXPRESSION_DEPTH, MAX_EXPRESSION_NODES)"
                .to_string(),
            items: vec![
                full_rust::MAX_TRANSFORMS.to_string(),
                full_rust::MAX_EXPRESSION_DEPTH.to_string(),
                full_rust::MAX_EXPRESSION_NODES.to_string(),
            ],
        },
        Vocabulary {
            name: "full_rust::operation_schemas()".to_string(),
            items: full_rust::operation_schemas()
                .iter()
                .map(|schema| format!("{schema:?}"))
                .collect(),
        },
    ]
}

const ARMS: [(&str, CanonicalMutationArm); 6] = [
    ("parameter_jitter", CanonicalMutationArm::ParameterJitter),
    ("insert", CanonicalMutationArm::Insert),
    ("delete", CanonicalMutationArm::Delete),
    ("adjacent_swap", CanonicalMutationArm::AdjacentSwap),
    ("head_replacement", CanonicalMutationArm::HeadReplacement),
    ("simplify", CanonicalMutationArm::Simplify),
];

fn evolve_streams(parents: &[String]) -> Vec<(StreamRecord, Vec<String>)> {
    let parents_note = "parents: champion, (pipe X (standardize) (ridge)), golden_map_derive_ridge, EXPANSION_STARTERS[2], (pipe X (winsor_clip q=0.05) (standardize) (elasticnet alpha=0.001 l1_ratio=0.5)); seeds 0..64 per parent";
    let per_parent = |f: &dyn Fn(&str, u64) -> String| -> Vec<String> {
        parents
            .iter()
            .flat_map(|parent| SEEDS.map(move |seed| (parent.as_str(), seed)))
            .map(|(parent, seed)| f(parent, seed))
            .collect()
    };
    let mut out = vec![stream(
        "evolve::random_canonical_pipeline(seed, 6)",
        "seeds 0..64",
        SEEDS
            .map(|seed| random_canonical_pipeline(seed, 6))
            .collect(),
    )];
    out.push(stream(
        "evolve::mutate_canonical_pipeline(parent, seed, 6)",
        parents_note,
        per_parent(&|parent, seed| shown(mutate_canonical_pipeline(parent, seed, 6))),
    ));
    let pairs = [(0usize, 1usize), (2, 3), (1, 2), (3, 0), (4, 2)];
    out.push(stream(
        "evolve::crossover_canonical_pipelines(left, right, seed, 6)",
        "parent pairs (0,1) (2,3) (1,2) (3,0) (4,2) of the mutation parents; seeds 0..64 per pair",
        pairs
            .iter()
            .flat_map(|(l, r)| SEEDS.map(move |seed| (*l, *r, seed)))
            .map(|(l, r, seed)| {
                shown(crossover_canonical_pipelines(
                    &parents[l],
                    &parents[r],
                    seed,
                    6,
                ))
            })
            .collect(),
    ));
    for expansion in [false, true] {
        out.push(stream(
            &format!("evolve::mutate_canonical_pipeline_v2(parent, seed, 6, {expansion})"),
            parents_note,
            per_parent(&|parent, seed| {
                shown(mutate_canonical_pipeline_v2(parent, seed, 6, expansion))
            }),
        ));
        for (arm_name, arm) in ARMS {
            out.push(stream(
                &format!(
                    "evolve::mutate_canonical_pipeline_with_arm(parent, seed, 6, {expansion}, {arm_name})"
                ),
                parents_note,
                per_parent(&|parent, seed| {
                    shown(mutate_canonical_pipeline_with_arm(
                        parent, seed, 6, expansion, arm,
                    ))
                }),
            ));
        }
    }
    out
}

fn full_rust_streams(parents: &[String]) -> Vec<(StreamRecord, Vec<String>)> {
    // The third parent carries a target stage (so the target arm drops or replaces it) and ends
    // in ridge_gcv, whose empty parameter schema sends the parameter arm to stage replacement.
    let fr_parents = [
        parents[1].clone(),
        parents[2].clone(),
        "(pipe X (target kind=log1p) (standardize append=true) (ridge_gcv))".to_string(),
    ];
    let note = "parents: (pipe X (standardize) (ridge)), golden_map_derive_ridge, (pipe X (target kind=log1p) (standardize append=true) (ridge_gcv)); feature_count 5; seeds 0..64 per parent";
    let per_parent = |f: &dyn Fn(&str, u64) -> String| -> Vec<String> {
        fr_parents
            .iter()
            .flat_map(|parent| SEEDS.map(move |seed| (parent.as_str(), seed)))
            .map(|(parent, seed)| f(parent, seed))
            .collect()
    };
    vec![
        stream(
            "full_rust::random_pipeline(seed, 5)",
            "seeds 0..64",
            SEEDS
                .map(|seed| evolved(full_rust::random_pipeline(seed, 5)))
                .collect(),
        ),
        stream(
            "full_rust::mutate_pipeline(parent, seed, 5)",
            note,
            per_parent(&|parent, seed| evolved(full_rust::mutate_pipeline(parent, seed, 5))),
        ),
        stream(
            "full_rust::mutate_pipeline_expanded(parent, seed, 5)",
            note,
            per_parent(&|parent, seed| {
                evolved(full_rust::mutate_pipeline_expanded(parent, seed, 5))
            }),
        ),
        stream(
            "full_rust::crossover_pipelines(left, right, seed)",
            "left (pipe X (standardize) (ridge)), right golden_map_derive_ridge; seeds 0..64",
            SEEDS
                .map(|seed| {
                    evolved(full_rust::crossover_pipelines(
                        &fr_parents[0],
                        &fr_parents[1],
                        seed,
                    ))
                })
                .collect(),
        ),
        stream(
            "full_rust::random_expression(&mut Pcg32::new(seed, 0), 5)",
            "seeds 0..64",
            SEEDS
                .map(|seed| full_rust::random_expression(&mut Pcg32::new(seed, 0), 5))
                .collect(),
        ),
        stream(
            "full_rust::random_expression_expanded(&mut Pcg32::new(seed, 0), 5, false)",
            "seeds 0..64",
            SEEDS
                .map(|seed| {
                    full_rust::random_expression_expanded(&mut Pcg32::new(seed, 0), 5, false)
                })
                .collect(),
        ),
        stream(
            "full_rust::random_expression_expanded(&mut Pcg32::new(seed, 0), 5, true)",
            "seeds 0..64",
            SEEDS
                .map(|seed| {
                    full_rust::random_expression_expanded(&mut Pcg32::new(seed, 0), 5, true)
                })
                .collect(),
        ),
    ]
}

fn scope(columns: usize, elementwise: bool) -> ExpressionScope {
    ExpressionScope {
        columns,
        elementwise,
        max_nodes: 64,
        max_depth: 16,
    }
}

fn edit_text(result: anyhow::Result<feat_math::math_expr::ExpressionEdit>) -> String {
    match result {
        Ok(edit) => format!(
            "{} | changed={}",
            shown(edit.expression.to_source()),
            edit.changed
        ),
        Err(error) => format!("ERR {error}"),
    }
}

fn expression_streams() -> Vec<(StreamRecord, Vec<String>)> {
    // The last parent has negations, so the edits visit, replace and unwrap unary nodes.
    let parents = [
        ("sigmoid(x) + 0.1*c0", scope(3, true)),
        ("row_mean + row_std - c1/(c0 + 3)", scope(5, false)),
        ("0.1", scope(0, true)),
        ("-(c0) + sin(-x)", scope(3, true)),
    ];
    let note = "parents and scopes: sigmoid(x) + 0.1*c0 {columns 3, map}, row_mean + row_std - c1/(c0 + 3) {columns 5, row}, 0.1 {columns 0, map}, -(c0) + sin(-x) {columns 3, map}; limits 64 nodes, depth 16; seeds 0..64 per parent";
    let mut out = vec![stream(
        "MathExpr::mutate_v1(seed, scope)",
        note,
        parents
            .iter()
            .flat_map(|(source, scope)| SEEDS.map(move |seed| (*source, *scope, seed)))
            .map(|(source, scope, seed)| {
                edit_text(MathExpr::parse(source).unwrap().mutate_v1(seed, scope))
            })
            .collect(),
    )];
    let crosses = [
        (parents[0].0, parents[1].0, scope(3, true)),
        (parents[1].0, parents[0].0, scope(5, false)),
        (parents[3].0, parents[1].0, scope(3, true)),
    ];
    out.push(stream(
        "MathExpr::crossover_v1(donor, seed, scope)",
        "recipient sigmoid(x) + 0.1*c0 {columns 3, map} with donor row_mean + row_std - c1/(c0 + 3), the reverse {columns 5, row}, and recipient -(c0) + sin(-x) {columns 3, map} with donor row_mean + row_std - c1/(c0 + 3); seeds 0..64 per pair",
        crosses
            .iter()
            .flat_map(|(recipient, donor, scope)| {
                SEEDS.map(move |seed| (*recipient, *donor, *scope, seed))
            })
            .map(|(recipient, donor, scope, seed)| {
                edit_text(MathExpr::parse(recipient).unwrap().crossover_v1(
                    &MathExpr::parse(donor).unwrap(),
                    seed,
                    scope,
                ))
            })
            .collect(),
    ));
    // A chain: each child is the next parent, with a crossover every third seed.
    let donor = MathExpr::parse("sin(c3)+c0*0.5").unwrap();
    let mut parent = MathExpr::parse("0.1").unwrap();
    let mut chain = Vec::new();
    for seed in SEEDS {
        let edit = if seed % 3 == 0 {
            parent.crossover_v1(&donor, seed, scope(4, true))
        } else {
            parent.mutate_v1(seed, scope(4, true))
        }
        .expect("a valid parent stays editable");
        chain.push(format!(
            "{} | changed={}",
            shown(edit.expression.to_source()),
            edit.changed
        ));
        parent = edit.expression;
    }
    out.push(stream(
        "MathExpr edit chain (mutate_v1, crossover_v1 every third seed)",
        "start 0.1, donor sin(c3)+c0*0.5, scope {columns 4, map}; seeds 0..64 in order",
        chain,
    ));
    let linear_rows: Vec<Vec<f64>> = (-4..=4).map(|i| vec![f64::from(i)]).collect();
    let linear_targets: Vec<f64> = linear_rows.iter().map(|r| 2.5 * r[0] - 0.75).collect();
    let wave_rows: Vec<Vec<f64>> = (-6..=6).map(|i| vec![f64::from(i) * 0.5]).collect();
    let wave_targets: Vec<f64> = wave_rows
        .iter()
        .map(|r| 3.0 * (1.7 * r[0]).sin() + 0.5)
        .collect();
    let fitted = |source: &str, rows: &[Vec<f64>], targets: &[f64], budget: usize| -> String {
        match MathExpr::parse(source)
            .unwrap()
            .optimize_constants_v1(rows, targets, budget)
        {
            Ok((expression, count)) => {
                format!("{} | evaluations={count}", shown(expression.to_source()))
            }
            Err(error) => format!("ERR {error}"),
        }
    };
    out.push(stream(
        "MathExpr::optimize_constants_v1(rows, targets, budget)",
        "0.1*c0+0.1 on x=-4..4, y=2.5x-0.75 with budgets 1, 10, 400; 1.3*sin(0.5*c0)+0.2 on x=-3..3 step 0.5, y=3 sin(1.7x)+0.5 with budget 300; and an empty training set",
        vec![
            fitted("0.1*c0+0.1", &linear_rows, &linear_targets, 1),
            fitted("0.1*c0+0.1", &linear_rows, &linear_targets, 10),
            fitted("0.1*c0+0.1", &linear_rows, &linear_targets, 400),
            fitted("1.3*sin(0.5*c0)+0.2", &wave_rows, &wave_targets, 300),
            fitted("0.1*c0+0.1", &[], &[], 10),
        ],
    ));
    out
}

fn slots(source: &str) -> PipelineSlots {
    PipelineSlots::from_pipeline(&parse_dsl(source).unwrap()).unwrap()
}

fn slots_text(result: feat_math::kernel::Result<PipelineSlots>) -> String {
    match result.and_then(|child| child.to_pipeline()) {
        Ok(pipe) => pipe.to_string(),
        Err(error) => format!("ERR {error}"),
    }
}

fn pipeline_slot_streams() -> Vec<(StreamRecord, Vec<String>)> {
    // The last parent's winsor_clip has a float parameter, so the edit reaches the float jitter.
    let parents = [
        ("(pipe X (target kind=log1p) (standardize) (ridge))", 3usize),
        (
            "(pipe X (derive expr=\"c0\" append=true) (map expr=\"c2+1\") (ridge))",
            2,
        ),
        (NAMED_PROGRAMS[0].1, 5),
        ("(pipe X (winsor_clip q=0.05) (standardize) (ridge))", 4),
    ];
    // Crossover donor of each parent above (the first three as the parent two places later).
    let donors = [2usize, 0, 1, 0];
    let note = "parents (pipe X (target kind=log1p) (standardize) (ridge)) {3 columns}, (pipe X (derive expr=\"c0\" append=true) (map expr=\"c2+1\") (ridge)) {2 columns}, golden_map_derive_ridge {5 columns}, (pipe X (winsor_clip q=0.05) (standardize) (ridge)) {4 columns}; rows 32, max_stages 12; seeds 0..64 per parent";
    let mutations = parents
        .iter()
        .flat_map(|(source, columns)| SEEDS.map(move |seed| (*source, *columns, seed)))
        .map(|(source, columns, seed)| slots_text(slots(source).mutate_v1(seed, 32, columns, 12)))
        .collect();
    let crosses = parents
        .iter()
        .zip(donors)
        .flat_map(|((source, columns), donor)| {
            let donor = parents[donor].0;
            SEEDS.map(move |seed| (*source, donor, *columns, seed))
        })
        .map(|(source, donor, columns, seed)| {
            slots_text(slots(source).crossover_v1(&slots(donor), seed, 32, columns, 12))
        })
        .collect();
    vec![
        stream(
            "PipelineSlots::mutate_v1(seed, rows, columns, max_stages)",
            note,
            mutations,
        ),
        stream(
            "PipelineSlots::crossover_v1(donor, seed, rows, columns, max_stages)",
            "each parent above with a donor parent: 0 with 2, 1 with 0, 2 with 1, 3 with 0; seeds 0..64 per parent",
            crosses,
        ),
    ]
}

fn json_line<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("serialize")
}

fn advanced_v3_streams(parents: &[String]) -> Vec<(StreamRecord, Vec<String>)> {
    let config = V3EvolutionConfig::default();
    let population = initial_population(&config, &[]);
    let seeded = initial_population(&config, &parents[..V3_SEED_PARENTS]);
    let scored: Vec<V3ScoredCandidate> = population
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, candidate)| V3ScoredCandidate {
            candidate,
            rmse: 0.1 + index as f64 * 1e-6,
            novelty: (index % 7) as f64,
        })
        .collect();
    // Every fifth candidate shares a score, so ranking and selection fall through to their
    // complexity and hash tie-breaks.
    let tied: Vec<V3ScoredCandidate> = population
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, candidate)| V3ScoredCandidate {
            candidate,
            rmse: 0.1 + (index % 5) as f64 * 1e-6,
            novelty: (index % 3) as f64,
        })
        .collect();
    let mutations = SEEDS
        .map(|seed| {
            let index = seed as usize * 7 % population.len();
            let parent = &population[index];
            shown(
                mutate_candidate(parent, seed, 1 + index / 20, parent.island as usize)
                    .map(|c| json_line(&c))
                    .map_err(|e| e.to_string()),
            )
        })
        .collect();
    let crosses = SEEDS
        .map(|seed| {
            let index = seed as usize * 7 % population.len();
            let left = &population[index];
            let right = &population[(index * 37 + 11) % population.len()];
            shown(
                crossover_candidates(left, right, seed, 1 + index / 20, left.island as usize)
                    .map(|c| json_line(&c))
                    .map_err(|e| e.to_string()),
            )
        })
        .collect();
    let next =
        |generation: usize, scores: &[V3ScoredCandidate], stagnation: &[usize]| -> Vec<String> {
            match next_generation(&config, generation, scores, stagnation) {
                Ok(candidates) => candidates.iter().map(json_line).collect(),
                Err(error) => vec![format!("ERR {error}")],
            }
        };
    let mut archive = V3Archive::default();
    archive.update(scored.iter().cloned());
    // The same candidates again: the better score replaces an entry, the worse one does not.
    let mut twice = V3Archive::default();
    twice.update(scored.iter().cloned());
    twice.update(
        scored
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, mut again)| {
                again.rmse += if index % 2 == 0 { -0.05 } else { 0.05 };
                again
            }),
    );
    vec![
        stream(
            "advanced_v3::initial_population(default, [])",
            "V3EvolutionConfig::default(), no seeds",
            population.iter().map(json_line).collect(),
        ),
        stream(
            "advanced_v3::initial_population(default, parents)",
            "V3EvolutionConfig::default(), seeds = the first four mutation parents",
            seeded.iter().map(json_line).collect(),
        ),
        stream(
            "advanced_v3::deterministic_seed_enumeration(default, [])",
            "V3EvolutionConfig::default(), no seeds",
            deterministic_seed_enumeration(&config, &[]),
        ),
        stream(
            "advanced_v3::mutate_candidate(parent, seed, generation, island)",
            "parent = initial population[seed*7 mod 80], generation 1 + index/20; seeds 0..64",
            mutations,
        ),
        stream(
            "advanced_v3::crossover_candidates(left, right, seed, generation, island)",
            "left = initial population[seed*7 mod 80], right = [(index*37+11) mod 80]; seeds 0..64",
            crosses,
        ),
        stream(
            "advanced_v3::next_generation(default, 1, scored, [0,0,0,0])",
            "scored = initial population with rmse 0.1 + i*1e-6 and novelty i mod 7",
            next(1, &scored, &[0, 0, 0, 0]),
        ),
        stream(
            "advanced_v3::next_generation(default, 5, scored, [8,0,9,0])",
            "the same scores; generation 5 migrates and two islands are stagnant",
            next(5, &scored, &[8, 0, 9, 0]),
        ),
        stream(
            "advanced_v3::V3Archive::update(scored)",
            "an empty archive updated with the scored initial population",
            archive.entries.iter().map(json_line).collect(),
        ),
        stream(
            "advanced_v3::next_generation(default, 2, tied, [0,0,0,0])",
            "tied = initial population with rmse 0.1 + (i mod 5)*1e-6 and novelty i mod 3",
            next(2, &tied, &[0, 0, 0, 0]),
        ),
        stream(
            "advanced_v3::V3Archive::update(scored), then update(rescored)",
            "an empty archive updated with the scored initial population, then with the same candidates rescored: even i 0.05 lower rmse, odd i 0.05 higher",
            twice.entries.iter().map(json_line).collect(),
        ),
    ]
}

#[test]
fn seeded_generator_streams_match_the_frozen_fixture() {
    let parents = stream_parents();
    let mut all = evolve_streams(&parents);
    all.extend(full_rust_streams(&parents));
    all.extend(expression_streams());
    all.extend(pipeline_slot_streams());
    all.extend(advanced_v3_streams(&parents));
    let fixture = GeneratorStreams {
        schema: "feat-math.compat-freeze.generator-streams.v1".to_string(),
        generated_at_sha: header_sha(Surface::Frozen),
        note: "Each stream is the ordered output of one seeded generator over fixed inputs: sha256 of the items joined by newlines, the first item verbatim, and the first 16 hex digits of each item's sha256. The legacy wrappers, full_rust, MathExpr, PipelineSlots and advanced_v3 streams must stay bit-identical (plan N1).".to_string(),
        vocabularies: vocabularies(),
        streams: all.iter().map(|(record, _)| record.clone()).collect(),
    };
    let Some(committed) = committed_or_write(Surface::Frozen, FIXTURE_FILES[6], &fixture) else {
        return;
    };
    let mut problems = Vec::new();
    let vocab = |values: &GeneratorStreams| -> Vec<(String, String)> {
        values
            .vocabularies
            .iter()
            .map(|v| (v.name.clone(), v.items.join(" | ")))
            .collect()
    };
    let (old_vocab, new_vocab) = (vocab(&committed), vocab(&fixture));
    for (name, items) in &old_vocab {
        match new_vocab.iter().find(|(n, _)| n == name) {
            Some((_, now)) if now == items => {}
            Some((_, now)) => problems.push(format!(
                "vocabulary {name}:\n      fixture {items}\n      now     {now}"
            )),
            None => problems.push(format!("vocabulary {name} is gone")),
        }
    }
    for (name, items) in &new_vocab {
        if !old_vocab.iter().any(|(n, _)| n == name) {
            problems.push(format!("vocabulary {name} is new: {items}"));
        }
    }
    if problems.is_empty() && old_vocab != new_vocab {
        problems.push("the vocabulary order changed".to_string());
    }
    for old in &committed.streams {
        let Some((new, items)) = all.iter().find(|(r, _)| r.name == old.name) else {
            problems.push(format!("stream {} is gone", old.name));
            continue;
        };
        if old == new {
            continue;
        }
        let old_items: Vec<&str> = old.per_item.split(' ').collect();
        let new_items: Vec<&str> = new.per_item.split(' ').collect();
        let first = old_items
            .iter()
            .zip(&new_items)
            .position(|(a, b)| a != b)
            .unwrap_or(old_items.len().min(new_items.len()));
        let digest = |digests: &[&str]| digests.get(first).copied().unwrap_or("(none)").to_string();
        problems.push(format!(
            "stream {} ({}): {} items in the fixture, {} now; first differing item {first}: \
             fixture digest {}, now digest {}{}\n      now: {}",
            old.name,
            old.inputs,
            old.count,
            new.count,
            digest(&old_items),
            digest(&new_items),
            if first == 0 {
                format!("\n      fixture: {}", old.first)
            } else {
                String::new()
            },
            items.get(first).map(String::as_str).unwrap_or("(none)")
        ));
    }
    for (new, _) in &all {
        if !committed.streams.iter().any(|old| old.name == new.name) {
            problems.push(format!("stream {} is new", new.name));
        }
    }
    assert!(
        problems.is_empty(),
        "compat-freeze: seeded generator streams drifted:\n  {}\n{}\n{REBASELINE_POLICY}",
        problems.join("\n  "),
        toolchain_note()
    );
}

// ---------------------------------------------------------------------------------------------
// WP-0.2 (f, whole searches): the seeded searches end to end
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchRuns {
    schema: String,
    generated_at_sha: String,
    note: String,
    runs: Vec<SearchRun>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchRun {
    name: String,
    inputs: String,
    /// The ranked result: program, cv_mean bits and fold score bits.
    ranked: Vec<String>,
    /// Lite race only: every raced program and its per-rung scores and bookkeeping.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    raced: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    summary: String,
}

struct Quiet;

impl Observer for Quiet {
    fn on_generation(&self, _: &GenStats) {}
    fn on_improvement(&self, _: &Scored, _: &Emitted, _: &GenStats) {}
}

fn bits_list(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| hex_bits(*v))
        .collect::<Vec<_>>()
        .join(" ")
}

fn ranked_line(scored: &Scored) -> String {
    format!(
        "{} | cv_mean={} | folds={}",
        scored.dsl,
        hex_bits(scored.cv_mean),
        bits_list(&scored.fold_scores)
    )
}

fn or_dash<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "-".to_string(), |value| value.to_string())
}

fn index_list_digest(values: &[usize]) -> String {
    let text = values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    format!("{} {}", values.len(), &sha256_hex(text.as_bytes())[..16])
}

/// The surrogate `evolve()` with default flags, which `evolve()` callers get.
fn surrogate_search_run() -> SearchRun {
    let (x, y) = toy_data(96, 5, 42);
    let metric = |truth: &[f64], predicted: &[f64]| feat_math::r2(truth, predicted);
    let config = EvolveConfig {
        pop: 16,
        gens: 3,
        folds: 3,
        seed: 12345,
        max_transforms: 3,
        ..EvolveConfig::default()
    };
    let ranked = evolve(&x, &y, &metric, &config, &Quiet);
    SearchRun {
        name: "evolve::evolve (surrogate scoring)".to_string(),
        inputs: "evolve() on toy_data(96 rows, 5 columns, seed 42) with EvolveConfig { pop 16, gens 3, folds 3, seed 12345, max_transforms 3, every other field default (surrogate scoring, opt-in flags off, no time budget) }; each line is the program, the cv_mean bits and the fold score bits".to_string(),
        ranked: ranked.iter().map(ranked_line).collect(),
        raced: Vec::new(),
        summary: String::new(),
    }
}

/// The lite race: `rank_robust` off is its default, on is how jope-bench `search_program` (the
/// shipped `jope optimize` search) runs it. No time budget, so no rung cap or wall clock can cut a
/// generation short; wall-clock timings are not recorded.
fn lite_race_run(rank_robust: bool) -> SearchRun {
    let (x, y) = toy_data(160, 5, 42);
    let config = EvolveConfig {
        pop: 16,
        gens: 3,
        folds: 3,
        seed: 41,
        max_transforms: 3,
        time_budget_secs: 0.0,
        fitness: FitnessMode::LiteRace,
        rank_robust,
        ..EvolveConfig::default()
    };
    let outcome = evolve_lite_race(&x, &y, &feat_math::r2, &config, &Quiet);
    let raced = outcome
        .candidates
        .iter()
        .map(|candidate| {
            let rungs: Vec<String> = candidate
                .rung_fold_scores
                .iter()
                .map(|scores| scores.as_deref().map_or_else(|| "-".to_string(), bits_list))
                .collect();
            let routes = candidate.r0_fold_routes.as_ref().map_or_else(
                || "-".to_string(),
                |routes| {
                    routes
                        .iter()
                        .map(|receipt| format!("{:?}", receipt.route))
                        .collect::<Vec<_>>()
                        .join(",")
                },
            );
            let reuse = candidate
                .r0_exact_surrogate_dedup
                .as_ref()
                .map_or_else(|| "-".to_string(), |receipt| format!("{receipt:?}"));
            format!(
                "{} | r0={} | r1={} | r2={} | seq={} generation={} slot={} | routes={} | reuse={} | seen={} first={} last={}",
                candidate.dsl,
                rungs[0],
                rungs[1],
                rungs[2],
                or_dash(candidate.r0_seq),
                or_dash(candidate.r0_generation),
                or_dash(candidate.r0_generation_slot),
                routes,
                reuse,
                candidate.recurrence_count,
                candidate.first_generation,
                candidate.last_generation
            )
        })
        .collect();
    SearchRun {
        name: format!("evolve::evolve_lite_race (rank_robust {rank_robust})"),
        inputs: format!(
            "evolve_lite_race on toy_data(160 rows, 5 columns, seed 42) with feat_math::r2 and EvolveConfig {{ pop 16, gens 3, folds 3, seed 41, max_transforms 3, time_budget_secs 0 (no caps), fitness LiteRace, rank_robust {rank_robust}, every other field default }}. ranked: program | cv_mean bits | fold score bits. raced: every raced program | fold score bits at rungs 0, 1 and 2 (- when not scored there) | rung-0 sequence, generation and slot | rung-0 fold routes | exact-reuse receipt | occurrences, first and last generation. summary: evaluations per rung, whether the target exit fired, and the rung-0 rows and fold assignment (count and sha256 prefix). Wall-clock timings are not recorded."
        ),
        ranked: outcome.ranked.iter().map(ranked_line).collect(),
        raced,
        summary: format!(
            "rung_evals {} {} {} | target_hit {} | rung0_rows {} | rung0_assignments {}",
            outcome.rung_evals[0],
            outcome.rung_evals[1],
            outcome.rung_evals[2],
            outcome.target_hit,
            index_list_digest(&outcome.rung0_rows),
            index_list_digest(&outcome.rung0_assignments)
        ),
    }
}

#[test]
fn seeded_search_runs_match_the_frozen_fixture() {
    let fixture = SearchRuns {
        schema: "feat-math.compat-freeze.search-runs.v1".to_string(),
        generated_at_sha: header_sha(Surface::Frozen),
        note: "Whole seeded searches, end to end: the surrogate evolve() with default flags, and the lite race with rank_robust off (its default) and on (as jope-bench search_program runs it). The race lines cover its rung scoring, promotion and ranking. Every line must stay bit-identical (plan N1).".to_string(),
        runs: vec![
            surrogate_search_run(),
            lite_race_run(false),
            lite_race_run(true),
        ],
    };
    let Some(committed) = committed_or_write(Surface::Frozen, FIXTURE_FILES[7], &fixture) else {
        return;
    };
    let flatten = |values: &SearchRuns| -> Vec<(String, String)> {
        let mut records = vec![record("schema", values.schema.clone())];
        for run in &values.runs {
            let name = &run.name;
            records.push(record(format!("{name}: inputs"), run.inputs.clone()));
            for (i, line) in run.ranked.iter().enumerate() {
                records.push(record(format!("{name}: ranked[{i}]"), line.clone()));
            }
            for (i, line) in run.raced.iter().enumerate() {
                records.push(record(format!("{name}: raced[{i}]"), line.clone()));
            }
            if !run.summary.is_empty() {
                records.push(record(format!("{name}: summary"), run.summary.clone()));
            }
        }
        records
    };
    assert_records(
        "search runs (search_runs.v1.json)",
        &flatten(&committed),
        &flatten(&fixture),
        REBASELINE_POLICY,
    );
}

// ---------------------------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Meta {
    schema: String,
    generated_at_sha: String,
    rustc: String,
    host: String,
    libm: String,
    cpu: String,
    cpu_flags: String,
    hostname: String,
    math_kernel: String,
    fixtures: Vec<String>,
    boundary_corpus: String,
    regenerate: String,
    regenerate_identity: String,
    rebaseline_policy: String,
}

#[test]
fn fixture_provenance_is_recorded() {
    let path = fixture_path("compat_freeze/meta.json");
    if regenerating(Surface::Frozen) {
        let verbose = command_line("rustc", &["-vV"]);
        let host = verbose
            .lines()
            .find_map(|line| line.strip_prefix("host: "))
            .expect("rustc -vV reports the host")
            .to_string();
        let (cpu, cpu_flags) = cpu_identity();
        let meta = Meta {
            schema: "feat-math.compat-freeze.meta.v1".to_string(),
            generated_at_sha: generation_sha(),
            rustc: command_line("rustc", &["-V"]),
            host,
            libm: command_line("getconf", &["GNU_LIBC_VERSION"]),
            cpu,
            cpu_flags,
            hostname: hostname(),
            math_kernel: EmitOptions::default().math_kernel.policy_id().to_string(),
            fixtures: FIXTURE_FILES.iter().map(|f| f.to_string()).collect(),
            boundary_corpus: CORPUS_FILE.to_string(),
            regenerate: REGENERATE_COMMAND.to_string(),
            regenerate_identity: IDENTITY_REGENERATE_COMMAND.to_string(),
            rebaseline_policy: REBASELINE_POLICY.to_string(),
        };
        write_atomically(
            &path,
            &(serde_json::to_string_pretty(&meta).unwrap() + "\n"),
        );
        eprintln!("compat-freeze: wrote tests/fixtures/compat_freeze/meta.json");
        return;
    }
    let meta: Meta = serde_json::from_str(
        &std::fs::read_to_string(&path).expect("tests/fixtures/compat_freeze/meta.json"),
    )
    .expect("meta.json parses");
    assert_eq!(meta.schema, "feat-math.compat-freeze.meta.v1");
    assert!(
        is_hex(&meta.generated_at_sha, 40),
        "meta.json generated_at_sha"
    );
    assert_eq!(meta.fixtures, FIXTURE_FILES);
    assert_eq!(meta.boundary_corpus, CORPUS_FILE);
    assert_eq!(meta.math_kernel, "platform-libm-v1");
    for file in FIXTURE_FILES {
        let text = std::fs::read_to_string(fixture_path(&format!("compat_freeze/{file}")))
            .unwrap_or_else(|error| panic!("{file}: {error}"));
        let value: serde_json::Value = serde_json::from_str(&text).expect("fixture JSON");
        assert!(
            is_hex(value["generated_at_sha"].as_str().unwrap_or_default(), 40),
            "{file}: generated_at_sha must name a full commit id"
        );
        assert!(
            text.len() < 1 << 20,
            "{file} is {} bytes; keep each fixture under 1 MiB",
            text.len()
        );
    }
}
