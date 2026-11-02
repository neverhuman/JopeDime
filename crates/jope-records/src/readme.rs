//! `readme-benchmarks --check|--write`: render the README's measured-results blocks from
//! committed evidence.
//!
//! Rust port of `deploy/ci/readme-benchmarks.py` (archived at `8a55c4fef`; see
//! `docs/archive/python-archive-manifest.json`) producing the same bytes. Three independent marker
//! pairs, so regenerating one can never clobber another:
//!
//! ```text
//! <!-- arena:begin --> ... <!-- arena:end -->       the Golden45 ladders
//! <!-- samecut:begin --> ... <!-- samecut:end -->   the same-cut ladder this tree reproduces
//! <!-- bench:begin --> ... <!-- bench:end -->       emit-path latency
//! ```
//!
//! Nothing here is hand-typed. Every number is read from a file that is already under test:
//! `crates/jope-arena/tests/elo_regression.rs` asserts that both ladders recompute from
//! `scores.csv`, so this renderer does not re-derive them — it formats what those tests already
//! guarantee, records where every number came from, and refuses to render when the committed
//! evidence disagrees with itself. **benchmark-valid** is the number this repository claims;
//! **published** is what the benchmark ranked, rendered as the best known program per dataset
//! and labelled as not a held-out result.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use regex::Regex;

use crate::py;
use crate::pycsv::{self, Newlines};
use crate::pyjson::{Dict, Value};
use crate::{Error, Result};

mod arena;
mod numeric;
mod samecut_load;
mod samecut_render;
mod values;
pub(super) use arena::*;
pub(super) use numeric::*;
pub(super) use samecut_load::*;
pub(super) use samecut_render::*;
pub(super) use values::*;

const REGRESSION: &str = "assets/regression";
const ARENA: &str = "assets/regression/sotarena-tabular-regression-v2";
const TRACK: &str =
    "assets/regression/sotarena-tabular-regression-v2/tracks/tabular-regression-v2/track.json";
const LEADERBOARD: &str =
    "assets/regression/sotarena-tabular-regression-v2/results/tabular-regression-v2/leaderboard.csv";
const SNAPSHOT: &str = "assets/regression/elo-tabular-regression-v2.json";
const VALID_LEADERBOARD: &str =
    "assets/regression/leaderboard-tabular-regression-v2.benchmark-valid.csv";
const VALID_SNAPSHOT: &str = "assets/regression/elo-tabular-regression-v2.benchmark-valid.json";
const CORRECTIONS: &str = "assets/regression/protocol-corrections-tabular-regression-v2.json";
const ARCHIVE_MANIFEST: &str = "docs/archive/python-archive-manifest.json";
const BENCH: &str = "assets/regression/bench-baseline.json";
const README: &str = "README.md";
const SAMECUT: &str = "assets/same-cut/v2";
const SAMECUT_ADAPTERS: [&str; 4] = ["catboost", "lightgbm", "rf", "xgboost"];
const SAMECUT_TRACK_ID: &str = "jope-same-cut-v1";
const SAMECUT_DOC: &str = "docs/same-cut-bench.md";
const REPORT_SCHEMA: &str = "jope.bench-report/v2";
const SEARCHED: &str = "afm.lite-race.v1.resampled";
const LLM_REFERENCE_HOURS: i64 = 1;
const LLM_REFERENCE_TOKENS: i64 = 10_000_000;
const SNAPSHOT_SCHEMA: &str = "jope.arena-elo-snapshot/v2";
const CORRECTIONS_SCHEMA: &str = "jope.arena-protocol-corrections/v1";
const OURS: [&str; 2] = ["jope-dime", "jope-dime-regressor"];
const ARENA_BEGIN: &str = "<!-- arena:begin -->";
const ARENA_END: &str = "<!-- arena:end -->";
const SAMECUT_BEGIN: &str = "<!-- samecut:begin -->";
const SAMECUT_END: &str = "<!-- samecut:end -->";
const BENCH_BEGIN: &str = "<!-- bench:begin -->";
const BENCH_END: &str = "<!-- bench:end -->";
const ANCHOR: &str = "## Install and check";
const TOP_N: usize = 5;

fn samecut_path(name: &str) -> String {
    format!("{SAMECUT}/{name}")
}

pub(crate) struct Paths {
    root: PathBuf,
}

impl Paths {
    fn at(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }
    fn sha256(&self, rel: &str) -> Result<String> {
        Ok(py::sha256_hex(&py::read_bytes(&self.at(rel))?))
    }
    fn json(&self, rel: &str) -> Result<Value> {
        py::load_json(&self.at(rel))
    }
}

/// A `path:lines` evidence reference whose file the Python removal archived, written
/// `<commit>:path:lines` so a reader can `git show` it; any other reference unchanged. The
/// SHA-pinned corrections file keeps the paths it was written with (`assets/regression/SOURCES.md`).
/// A tree without the archive manifest (the Python-parity fixtures) renders the reference as
/// written, as the Python renderer did.
fn archived(p: &Paths, reference: &str) -> Result<String> {
    let path = reference.split(':').next().unwrap_or(reference);
    if path.is_empty() || p.at(path).exists() || !p.at(ARCHIVE_MANIFEST).is_file() {
        return Ok(reference.to_owned());
    }
    let manifest = p.json(ARCHIVE_MANIFEST)?;
    let commit = manifest
        .get("files")
        .and_then(Value::as_list)
        .into_iter()
        .flatten()
        .find(|f| f.get("path").and_then(Value::as_str) == Some(path))
        .and_then(|f| f.get("source_commit").and_then(Value::as_str));
    Ok(match commit {
        Some(c) => format!("{}:{reference}", &c[..c.len().min(9)]),
        None => reference.to_owned(),
    })
}

/// `refuse(why)`: the committed evidence disagrees with itself.
fn render_bench(p: &Paths) -> Result<String> {
    let b = p.json(BENCH)?;
    let num = |key: &str| -> Result<f64> { float_of(at(&b, key)?, &format!("{BENCH} {key}")) };
    let git_sha = first_nine(at(&b, "git_sha")?, &format!("{BENCH} git_sha"))?;
    Ok([
        BENCH_BEGIN.to_owned(),
        "### Emit-path latency".to_owned(),
        String::new(),
        format!(
            "`{}` — {} runs of the release `emit` example on `{}`.",
            s(at(&b, "bench")?),
            s(at(&b, "runs")?),
            s(at(&b, "genome")?)
        ),
        String::new(),
        "| min | median | mean | p95 |".to_owned(),
        "| ---: | ---: | ---: | ---: |".to_owned(),
        format!(
            "| {} ms | {} ms | {} ms | {} ms |",
            fixed(num("min_ms")?, 2),
            fixed(num("median_ms")?, 2),
            fixed(num("mean_ms")?, 2),
            fixed(num("p95_ms")?, 2)
        ),
        String::new(),
        format!(
            "**This is not an engine-speed number.** Its own `what` field says it measures \"{}\". The min-to-p95 spread is dominated by process spawn and scheduling, not by emission. Treat it as a regression tripwire (alert at {}% drift), not as a performance claim.",
            s(at(&b, "what")?),
            fixed(num("alert_pct")?, 0)
        ),
        String::new(),
        format!(
            "Recorded {} on `{}` ({}, {} cores, kernel {}) at `{}` with {}. Baselines are host-pinned; `bash deploy/ci/regression-suite.sh bench` compares against this file.",
            s(at(&b, "recorded")?),
            s(at(&b, "host")?),
            s(at(&b, "cpu")?),
            s(at(&b, "nproc")?),
            s(at(&b, "kernel")?),
            git_sha,
            s(at(&b, "rustc")?)
        ),
        BENCH_END.to_owned(),
    ]
    .join("\n"))
}

fn splice(text: &str, begin: &str, end: &str, block: &str) -> Result<String> {
    match (text.find(begin), text.find(end)) {
        (Some(i), Some(j)) => Ok(format!("{}{block}{}", &text[..i], &text[j + end.len()..])),
        _ => Err(Error::Invalid(format!(
            "markers {begin}/{end} not found; cannot splice"
        ))),
    }
}

/// The README with every block re-rendered (inserting the section on first use).
pub fn build(root: &Path, text: &str) -> Result<String> {
    let p = Paths {
        root: root.to_path_buf(),
    };
    let mut text = text.to_owned();
    if !text.contains(ARENA_BEGIN) {
        let Some(anchor) = text.find(ANCHOR) else {
            return Err(Error::Invalid(format!(
                "anchor {} not found in README",
                crate::pyjson::str_repr(ANCHOR)
            )));
        };
        let section = format!(
            "## Measured results\n\nGenerated by `jope-records readme-benchmarks` from committed evidence. Do not edit between the markers by hand.\n\n{ARENA_BEGIN}\n{ARENA_END}\n\n{SAMECUT_BEGIN}\n{SAMECUT_END}\n\n{BENCH_BEGIN}\n{BENCH_END}\n\n"
        );
        text = format!("{}{section}{}", &text[..anchor], &text[anchor..]);
    }
    if !text.contains(SAMECUT_BEGIN) {
        // The same-cut block was added after the other two: slot it between them.
        let Some(i) = text.find(BENCH_BEGIN) else {
            return Err(Error::Invalid(format!(
                "markers {BENCH_BEGIN} not found; cannot place the same-cut block"
            )));
        };
        text = format!(
            "{}{SAMECUT_BEGIN}\n{SAMECUT_END}\n\n{}",
            &text[..i],
            &text[i..]
        );
    }
    let text = splice(&text, ARENA_BEGIN, ARENA_END, &render_arena(&p)?)?;
    let text = splice(&text, SAMECUT_BEGIN, SAMECUT_END, &render_samecut(&p)?)?;
    splice(&text, BENCH_BEGIN, BENCH_END, &render_bench(&p)?)
}

/// Every input the render reads; a missing one refuses before anything is read.
pub fn required_inputs() -> Vec<String> {
    vec![
        LEADERBOARD.to_owned(),
        TRACK.to_owned(),
        SNAPSHOT.to_owned(),
        VALID_LEADERBOARD.to_owned(),
        VALID_SNAPSHOT.to_owned(),
        CORRECTIONS.to_owned(),
        BENCH.to_owned(),
        README.to_owned(),
        samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-deep.csv")),
        samecut_path(&format!("elo-{SAMECUT_TRACK_ID}-deep.json")),
        samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-screen.csv")),
        samecut_path(&format!("elo-{SAMECUT_TRACK_ID}-screen.json")),
        samecut_path(&format!("track/tracks/{SAMECUT_TRACK_ID}-deep/track.json")),
        samecut_path(&format!("track/results/{SAMECUT_TRACK_ID}-deep/scores.csv")),
        samecut_path("reports"),
        SAMECUT_DOC.to_owned(),
    ]
}

/// `--check` (write = false) or `--write`: the message, and whether the command succeeded.
pub fn run(root: &Path, write: bool) -> Result<(String, bool)> {
    for input in required_inputs() {
        let path = root.join(&input);
        // `p.exists()`: a missing input refuses with the Python's message; any other error (a
        // directory that may not be searched) refuses as the Python raised it.
        if !py::path_exists(&path).map_err(|e| Error::io(&path, e))? {
            return Err(Error::Invalid(format!(
                "missing required input: {}",
                py::os_display(path.as_os_str())
            )));
        }
    }
    let readme = root.join(README);
    let current = py::read_text(&readme)?;
    let updated = build(root, &current)?;
    if write {
        if updated != current {
            std::fs::write(&readme, &updated).map_err(|e| Error::io(&readme, e))?;
            return Ok(("README.md updated\n".to_owned(), true));
        }
        return Ok(("README.md already current\n".to_owned(), true));
    }
    if updated != current {
        return Ok((
            "README.md is out of date with the committed benchmark evidence.\nRun: cargo run -q --locked -p jope-records -- readme-benchmarks --write\n".to_owned(),
            false,
        ));
    }
    Ok(("README.md benchmark blocks are current\n".to_owned(), true))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python's `format` for the specs the blocks use.
    #[test]
    fn number_formats_match_python() {
        assert_eq!(fixed(0.125, 2), "0.12");
        assert_eq!(fixed(0.375, 2), "0.38");
        assert_eq!(fixed(2.5, 0), "2");
        assert_eq!(fixed(1.05, 1), "1.1");
        assert_eq!(thousands(10_000_000), "10,000,000");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(8763), "8,763");
        assert_eq!(general4(0.123456), "0.1235");
        assert_eq!(general4(1234.5), "1234");
        assert_eq!(general4(12345.0), "1.234e+04");
        assert_eq!(general4(0.0001234), "0.0001234");
        assert_eq!(general4(0.00001234), "1.234e-05");
        assert_eq!(general4(2.0), "2");
        assert_eq!(general4(0.5), "0.5");
        assert_eq!(general4(9999.5), "1e+04");
    }

    /// Integer true division rounds once, to the double nearest the exact quotient, ties to even
    /// (CPython's `long_true_divide`). For operands a double holds exactly that is one IEEE
    /// division; beyond 2^53 the integer is not rounded first.
    #[test]
    fn integer_true_division_rounds_once() {
        let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
        for _ in 0..20_000 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let a = i128::from((seed >> 11) as i64) - (1_i128 << 52);
            let b = (seed & 0xffff) + 1;
            assert_eq!(int_true_div(a, b), a as f64 / b as f64, "{a} / {b}");
        }
        let two_53 = 1_i128 << 53;
        // The captured same-cut mean (`seconds-int-true-division`), and its negative.
        assert_eq!(
            int_true_div(5_000_000_000_000_000_321, 10),
            500_000_000_000_000_064.0
        );
        assert_eq!(
            int_true_div(-5_000_000_000_000_000_321, 10),
            -500_000_000_000_000_064.0
        );
        // 2^53 + 1 and 2^53 + 3 are ties, to the even neighbour; 2^53 + 1.5 is not.
        assert_eq!(int_true_div(2 * (two_53 + 1), 2), 9_007_199_254_740_992.0);
        assert_eq!(int_true_div(2 * (two_53 + 3), 2), 9_007_199_254_740_996.0);
        assert_eq!(int_true_div(2 * two_53 + 3, 2), 9_007_199_254_740_994.0);
        assert_eq!(
            int_true_div(i128::from(i64::MAX) * 2 - 1, 2),
            9_223_372_036_854_775_808.0
        );
        assert_eq!(int_true_div(0, 7), 0.0);
        assert_eq!(
            int_true_div(1, u64::MAX),
            1.0 / 18_446_744_073_709_551_615.0
        );
    }

    /// Numbers compare as Python compares them: exactly, an int against a float included.
    #[test]
    fn numbers_compare_exactly() {
        use std::cmp::Ordering::{Equal, Greater, Less};
        let two_53 = 1_i128 << 53;
        let two_127 = 2.0_f64.powi(127);
        for (a, b, want) in [
            (
                Num::Int(two_53 + 1),
                Num::Float(9_007_199_254_740_992.0),
                Some(Greater),
            ),
            (
                Num::Int(two_53),
                Num::Float(9_007_199_254_740_992.0),
                Some(Equal),
            ),
            (
                Num::Int(-two_53 - 1),
                Num::Float(-9_007_199_254_740_992.0),
                Some(Less),
            ),
            (Num::Int(3), Num::Float(2.5), Some(Greater)),
            (Num::Int(2), Num::Float(2.5), Some(Less)),
            (Num::Int(-2), Num::Float(-2.5), Some(Greater)),
            (Num::Int(-3), Num::Float(-2.5), Some(Less)),
            (Num::Int(0), Num::Float(-0.0), Some(Equal)),
            (Num::Int(i128::MAX), Num::Float(two_127), Some(Less)),
            (Num::Int(i128::MIN), Num::Float(-two_127), Some(Equal)),
            (
                Num::Int(i128::MIN),
                Num::Float(f64::NEG_INFINITY),
                Some(Greater),
            ),
            (Num::Int(i128::MAX), Num::Float(f64::INFINITY), Some(Less)),
            (Num::Int(1), Num::Float(f64::NAN), None),
            (Num::Int(two_53 + 1), Num::Int(two_53), Some(Greater)),
            (Num::Float(0.5), Num::Float(0.25), Some(Greater)),
        ] {
            assert_eq!(py_cmp(a, b), want, "{a:?} {b:?}");
            assert_eq!(
                py_cmp(b, a),
                want.map(std::cmp::Ordering::reverse),
                "{b:?} {a:?}"
            );
        }
    }

    /// `sum()`'s three paths: the C `long` fast path, exact integers and plain float addition
    /// after it overflows, and the compensated float path (integers added plainly).
    #[test]
    fn sums_follow_cpython() {
        let max = i128::from(i64::MAX);
        let sum = |values: &[Num]| py_sum(values, "test").unwrap();
        let int = |n: Num| match n {
            Num::Int(i) => i,
            Num::Float(f) => panic!("a float {f}"),
        };
        let float = |n: Num| match n {
            Num::Float(f) => f.to_bits(),
            Num::Int(i) => panic!("an int {i}"),
        };
        assert_eq!(int(sum(&[Num::Int(max), Num::Int(1), Num::Int(-max)])), 1);
        assert_eq!(
            int(sum(&[Num::Int(max), Num::Int(max), Num::Int(2)])),
            1 << 64
        );
        let (big, one) = (Num::Float(1e16), Num::Float(1.0));
        let generic = [
            Num::Int(max),
            Num::Int(1),
            Num::Int(-max),
            big,
            one,
            Num::Float(-1e16),
        ];
        assert_eq!(float(sum(&generic)), 0.0_f64.to_bits());
        let compensated = [Num::Int(1), big, one, Num::Float(-1e16)];
        assert_eq!(float(sum(&compensated)), 1.0_f64.to_bits());
        let ints_plain = [big, Num::Int(1), Num::Int(1), Num::Float(-1e16)];
        assert_eq!(float(sum(&ints_plain)), 0.0_f64.to_bits());
        assert_eq!(float(sum(&[Num::Float(-0.0)])), 0.0_f64.to_bits());
    }
}
