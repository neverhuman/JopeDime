//! Compatibility freeze of the shipped math surfaces (workplan WP-0.2), starting with the shared
//! scalar boundary corpus (WP-0.3) that the freeze and the emitted-artifact harnesses consume.
//!
//! `boundary_corpus_v1_is_well_formed_and_matches_its_definition` checks
//! `tests/fixtures/math_boundary/v1.json` against its definition below: the file parses with
//! serde_json, every `value_repr` parses back to exactly its `bits` (NaN payloads live only in
//! `bits`), names and values are unique, every workplan group is present, and every value the
//! workplan lists is there. Neighbours are computed by exact bit arithmetic, never by decimal
//! guessing.
//!
//! Modes. By default the test only compares. With `JOPE_COMPAT_FREEZE_REGENERATE=1` and
//! `JOPE_COMPAT_FREEZE_SHA=<full commit id>` it rewrites the file instead; the commit must carry
//! exactly the sources in the working tree (checked with `git diff --quiet`). A changed corpus is
//! a new version (`v2.json`), never an edit of v1.
//!
//! The workplan asks for the corpus to parse in Python too. The stack is Rust-only (ADR-0006), so
//! the second, independent parser is `jq -e . tests/fixtures/math_boundary/v1.json`.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------------------------
// Modes, paths and fixture I/O
// ---------------------------------------------------------------------------------------------

const REGENERATE_ENV: &str = "JOPE_COMPAT_FREEZE_REGENERATE";
const SHA_ENV: &str = "JOPE_COMPAT_FREEZE_SHA";
const REGENERATE_COMMAND: &str = "JOPE_COMPAT_FREEZE_REGENERATE=1 JOPE_COMPAT_FREEZE_SHA=<full commit id> cargo nextest run --locked -P pr -p feat-math --test compat_freeze";

/// Sources whose bytes the fixtures depend on. Regeneration refuses to record a commit id whose
/// copies of these differ from the working tree's.
const SOURCE_PATHSPECS: &[&str] = &[
    "crates/feat-math/src",
    "crates/feat-math/Cargo.toml",
    "crates/feat-math/assets",
    "crates/feat-math/cpp",
    "crates/feat-math/tests/fixtures/emitted",
    "crates/jope-engine/assets/grammar",
    "crates/jope-engine/src/grammar.rs",
    "crates/jope-policy/src",
    "crates/jope-policy/release",
    "crates/jope-campaigns/src",
    "crates/jope-campaigns/Cargo.toml",
    "crates/jope-campaigns/known-evaluator-identities.json",
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
];

fn regenerating() -> bool {
    std::env::var(REGENERATE_ENV).as_deref() == Ok("1")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(relative)
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

fn header_sha() -> String {
    if regenerating() {
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

// ---------------------------------------------------------------------------------------------
// The boundary corpus (WP-0.3)
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

#[test]
fn boundary_corpus_v1_is_well_formed_and_matches_its_definition() {
    let definition = corpus_definition();
    let path = fixture_path(CORPUS_FILE);
    if regenerating() {
        let _ = header_sha();
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
