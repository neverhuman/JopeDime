//! The damped least-squares refinement's pinned winner, emitted and compared bit for bit
//! (laneD:B17-lm-refinement).
//!
//! [`PINNED_LM_PROGRAM`] is the program the opt-in damped least-squares refinement makes of the
//! jope-pipeline sine fixture. The refinement only rewrites a literal of a `derive` stage, so the
//! winner is an ordinary pipeline over the existing elements and needs no emitter change. This
//! module runs it through the two existing bit-parity procedures:
//!
//! * the `parity_bits.rs` lanes: the runtime (`feat_math::fit`, then `Model::predict`) against the
//!   emitted Rust program (`rustc -O`), the emitted Python program (`python3`) and the fitted C++17
//!   program (`emit_fitted_cpp17`, g++ 13.3), on that test's training rows and on its rows for a
//!   pipeline (ordinary rows, every boundary-corpus value in four row shapes, and the mixed
//!   signed-zero rows);
//! * the recipe C++17 driver of `mathexpr_boundary_parity.rs`: `expression_recipe::emit_cpp17`,
//!   fitted and predicted in C++ (`g++ -std=c++17 -O2`) against `fit_predict`, on that test's rows
//!   and the non-finite training and prediction cells it adds for a pipeline.
//!
//! Every prediction is compared with `f64::to_bits`, so the sign of zero counts and a NaN is always
//! a divergence, exactly as those tests compare. No allowlist applies: the parity allowlist names
//! only `parity_bits.rs`'s own programs, so any divergence here fails.
//!
//! The helpers of both files are private to their own test targets, so each procedure is restated
//! here step for step. The two C++ drivers are read verbatim from those files
//! ([`driver_source`]), so their bytes cannot drift from the drivers the parity tests run.
//!
//! As in `parity_bits.rs`, `JOPE_PARITY_STRICT=1` turns a missing `rustc`, `python3` or C++
//! compiler (`CXX`, default `g++`), or a fitted-lane compiler other than g++ 13.3, into a failure;
//! without it such a lane prints a `parity_bits SKIP` line and returns. The recipe lane, like
//! `mathexpr_boundary_parity.rs`, always runs `g++`. Scratch files live under `TMPDIR` when it is
//! set, otherwise under Cargo's target temporary directory, never an implicit system default.

use feat_math::cxx::CxxStd;
use feat_math::dsl::ParamValue;
use feat_math::kernel::expression_recipe::emit_cpp17 as emit_recipe_cpp17;
use feat_math::kernel::Program as KernelProgram;
use feat_math::math_expr::MathExpr;
use feat_math::transpile::{emit, supports, EmitOptions};
use feat_math::{emit_fitted_cpp17, fit, fit_legacy_numeric_bound, fit_predict, GenomeIr, Pcg32};
use std::fs;
use std::io::{ErrorKind, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// The damped least-squares refinement of the jope-pipeline sine fixture
/// `(pipe X (derive expr="sin(0.5*c0)" append=false) (ridge lambda_frac=0.000000001))`, 60 trials,
/// on `3 sin(1.7 x) + 0.5`. It is produced and pinned by the jope-pipeline test
/// `kernel2::refinement_lm_tests::the_lm_refined_fixture_program_is_pinned`
/// (`crates/jope-pipeline/src/kernel2/refinement_lm_tests.rs`), and
/// [`the_pinned_winner_is_the_jope_pipeline_pin_and_round_trips`] binds this copy to that pin.
const PINNED_LM_PROGRAM: &str = "(pipe X (derive append=false expr=\"sin((1.6999999999997006*c0))\") (ridge lambda_frac=0.000000001))";

/// The jope-pipeline test file that produces and pins [`PINNED_LM_PROGRAM`].
const JOPE_PIPELINE_PIN: &str =
    include_str!("../../../jope-pipeline/src/kernel2/refinement_lm_tests.rs");

/// The parity tests whose procedures and C++ drivers this module reuses.
const PARITY_BITS: &str = include_str!("../parity_bits.rs");
const BOUNDARY_PARITY: &str = include_str!("../mathexpr_boundary_parity.rs");

/// The shared scalar boundary corpus `parity_bits.rs` reads.
const BOUNDARY_V1: &str = include_str!("../fixtures/math_boundary/v1.json");

/// `parity_bits.rs`: feature columns, training rows, ordinary prediction rows, the fill of a row
/// that carries a boundary value in one column, and the compiler its fitted lane requires.
const WIDTH: usize = 3;
const TRAIN_ROWS: usize = 64;
const ORDINARY_ROWS: usize = 8;
const FILL: [f64; WIDTH] = [0.75, -1.25, 1.5];
const FITTED_CXX_VERSION: &str = "13.3";

/// The raw-string constant `name` of a parity test's source, verbatim.
fn driver_source<'a>(source: &'a str, name: &str) -> &'a str {
    let start = format!("const {name}: &str = r#\"");
    let mut parts = source.split(start.as_str());
    parts.next();
    let rest = parts
        .next()
        .unwrap_or_else(|| panic!("the parity test no longer defines {name}"));
    assert!(parts.next().is_none(), "{name} is defined more than once");
    let (driver, _) = rest
        .split_once("\"#;")
        .unwrap_or_else(|| panic!("{name} is not a closed raw string"));
    driver
}

/// The `parity_bits.rs` fitted-lane driver: reads the input bits from a file named on its command
/// line and prints each prediction's bits.
fn fitted_driver() -> &'static str {
    let driver = driver_source(PARITY_BITS, "DRIVER_CPP");
    assert!(driver.contains("jope_fitted_predict"), "{driver}");
    driver
}

/// The `mathexpr_boundary_parity.rs` recipe driver: fits and predicts in C++ from bits on stdin.
fn recipe_driver() -> &'static str {
    let driver = driver_source(BOUNDARY_PARITY, "DRIVER");
    assert!(
        driver.contains("model_fit_numeric") && driver.contains("model_predict_numeric_batch"),
        "{driver}"
    );
    driver
}

fn strict() -> bool {
    std::env::var_os("JOPE_PARITY_STRICT").is_some_and(|value| value == "1")
}

/// `TMPDIR` when it is set, otherwise Cargo's per-target temporary directory.
fn scratch_root() -> PathBuf {
    let root = match std::env::var_os("TMPDIR") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from(env!("CARGO_TARGET_TMPDIR")),
    };
    fs::create_dir_all(&root).unwrap();
    root
}

fn run_checked(what: &str, command: &mut Command) -> Output {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("{what}: cannot start: {error}"));
    assert!(
        output.status.success(),
        "{what} failed with {}:\n{}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

/// The pinned winner after the checks `parity_bits.rs` makes before any lane fits or emits a
/// program: every scalar tree reparses to itself, the canonical text is a fixed point, and neither
/// the canonical text nor the typed envelope changes the executable tree.
fn winner() -> GenomeIr {
    let original = GenomeIr::parse(PINNED_LM_PROGRAM).unwrap();
    let GenomeIr::Pipeline(dsl) = &original else {
        panic!("the pinned winner is not a pipeline");
    };
    let mut expressions = 0;
    for stage in dsl
        .stages
        .iter()
        .filter(|stage| matches!(stage.op.as_str(), "map" | "derive"))
    {
        let Some(ParamValue::Str(source)) = stage.params.get("expr") else {
            panic!("{}: expression parameter", stage.op);
        };
        let expression = MathExpr::parse(source).unwrap();
        let text = expression.to_source().unwrap();
        let reparsed = MathExpr::parse(&text).unwrap();
        assert_eq!(reparsed, expression, "scalar tree changed");
        assert_eq!(reparsed.to_source().unwrap(), text);
        expressions += 1;
    }
    assert_eq!(expressions, 1, "one refined derive stage");
    let canonical = original.canonical();
    let reparsed = GenomeIr::parse(&canonical).unwrap();
    assert_eq!(reparsed.canonical(), canonical, "fixed point");
    let bytes = KernelProgram::import_legacy(&original).to_bytes().unwrap();
    assert_eq!(
        KernelProgram::import_legacy(&reparsed).to_bytes().unwrap(),
        bytes,
        "canonical text changed the executable tree"
    );
    let envelope = KernelProgram::import_legacy(&reparsed)
        .to_lossless_text()
        .unwrap();
    assert_eq!(
        KernelProgram::parse_lossless_text(&envelope)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes,
        "typed envelope changed the executable tree"
    );
    reparsed
}

#[test]
fn the_pinned_winner_is_the_jope_pipeline_pin_and_round_trips() {
    let pinned = format!("const PINNED_LM_PROGRAM: &str = {PINNED_LM_PROGRAM:?};");
    assert_eq!(
        JOPE_PIPELINE_PIN.matches(pinned.as_str()).count(),
        1,
        "jope-pipeline's the_lm_refined_fixture_program_is_pinned pins another program; copy its \
         PINNED_LM_PROGRAM here"
    );
    assert!(JOPE_PIPELINE_PIN.contains("fn the_lm_refined_fixture_program_is_pinned()"));
    let genome = winner();
    assert_eq!(genome.canonical(), PINNED_LM_PROGRAM);
    supports(&genome).unwrap_or_else(|missing| panic!("{missing:?}"));
    // The winner is the refined program, not the fixture's start.
    let start = GenomeIr::parse(
        "(pipe X (derive expr=\"sin(0.5*c0)\" append=false) (ridge lambda_frac=0.000000001))",
    )
    .unwrap();
    assert_ne!(
        KernelProgram::import_legacy(&start).to_bytes().unwrap(),
        KernelProgram::import_legacy(&genome).to_bytes().unwrap()
    );
}

// ------------------------------------------------------------------------------------------------
// The parity_bits.rs lanes
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
mod execution;
mod recipe;
