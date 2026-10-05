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
enum Lane {
    Rust,
    Python,
    Cpp17Fitted,
}

impl Lane {
    fn id(self) -> &'static str {
        match self {
            Lane::Rust => "rust",
            Lane::Python => "python",
            Lane::Cpp17Fitted => "cpp17-fitted",
        }
    }

    fn tool(self) -> String {
        match self {
            Lane::Rust => "rustc".to_owned(),
            Lane::Python => "python3".to_owned(),
            Lane::Cpp17Fitted => feat_math::cxx::compiler(),
        }
    }
}

/// The lane's tool version, or `None` after a loud skip (a failure under `JOPE_PARITY_STRICT=1`).
fn toolchain(lane: Lane) -> Option<String> {
    let version = match lane {
        Lane::Cpp17Fitted => feat_math::cxx::version(),
        Lane::Rust | Lane::Python => {
            Command::new(lane.tool())
                .arg("--version")
                .output()
                .map(|output| {
                    String::from_utf8_lossy(&output.stdout)
                        .lines()
                        .next()
                        .unwrap_or_default()
                        .to_owned()
                })
        }
    };
    match version {
        Ok(version) if lane == Lane::Cpp17Fitted && !version.contains(FITTED_CXX_VERSION) => {
            assert!(
                !strict(),
                "emitted winner: the {} lane needs g++ {FITTED_CXX_VERSION} under \
                 JOPE_PARITY_STRICT=1, and {:?} reports {version:?}",
                lane.id(),
                lane.tool()
            );
            eprintln!(
                "parity_bits SKIP: emitted winner, {} lane, {:?} reports {version:?}, not g++ \
                 {FITTED_CXX_VERSION}",
                lane.id(),
                lane.tool()
            );
            None
        }
        Ok(version) => Some(version),
        Err(error) if error.kind() == ErrorKind::NotFound => {
            assert!(
                !strict(),
                "emitted winner: {} tool {:?} is unavailable and JOPE_PARITY_STRICT=1",
                lane.id(),
                lane.tool()
            );
            eprintln!(
                "parity_bits SKIP: emitted winner, {} lane, tool {:?} is unavailable",
                lane.id(),
                lane.tool()
            );
            None
        }
        Err(error) => panic!(
            "emitted winner: cannot start {} tool {:?}: {error}",
            lane.id(),
            lane.tool()
        ),
    }
}

/// `parity_bits.rs` `training_data`: finite rows every lane fits on.
fn training_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Pcg32::new(20_260_929, 7);
    let mut x = Vec::with_capacity(TRAIN_ROWS);
    let mut y = Vec::with_capacity(TRAIN_ROWS);
    for _ in 0..TRAIN_ROWS {
        let row: Vec<f64> = (0..WIDTH).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        let target =
            1.3 * row[0] - 0.7 * row[1] + 0.4 * (2.0 * row[2]).sin() + 0.1 * rng.gaussian();
        x.push(row);
        y.push(target);
    }
    (x, y)
}

fn parse_bits(text: &str) -> u64 {
    let digits = text.strip_prefix("0x").unwrap();
    assert_eq!(digits.len(), 16, "{text}");
    u64::from_str_radix(digits, 16).unwrap()
}

/// `parity_bits.rs` `input_rows(Inputs::All)`, the rows its pipeline programs are compared on:
/// ordinary rows, then every boundary-corpus value in every column (`all:`) and in one column with
/// [`FILL`] in the others (`c0:` ..), then rows of mixed signed zeros. `(id, values)` in order.
fn prediction_rows() -> Vec<(String, Vec<f64>)> {
    let mut rng = Pcg32::new(20_260_930, 11);
    let mut rows: Vec<(String, Vec<f64>)> = (0..ORDINARY_ROWS)
        .map(|index| {
            (
                format!("ordinary:{index:02}"),
                (0..WIDTH).map(|_| rng.next_f64() * 6.0 - 3.0).collect(),
            )
        })
        .collect();
    let corpus: serde_json::Value = serde_json::from_str(BOUNDARY_V1).unwrap();
    assert_eq!(corpus["schema"], "feat-math.math-boundary.v1");
    let entries = corpus["entries"].as_array().unwrap();
    assert!(!entries.is_empty());
    for entry in entries {
        let name = entry["name"].as_str().unwrap();
        let value = f64::from_bits(parse_bits(entry["bits"].as_str().unwrap()));
        rows.push((format!("all:{name}"), vec![value; WIDTH]));
        for column in 0..WIDTH {
            let mut values = FILL.to_vec();
            values[column] = value;
            rows.push((format!("c{column}:{name}"), values));
        }
    }
    let (p, n) = (0.0_f64, -0.0_f64);
    for (id, values) in [
        ("zeros:pos_pos_neg", [p, p, n]),
        ("zeros:neg_neg_pos", [n, n, p]),
        ("zeros:pos_neg_pos", [p, n, p]),
        ("zeros:neg_pos_neg", [n, p, n]),
    ] {
        rows.push((id.to_owned(), values.to_vec()));
    }
    assert!(rows.iter().any(|(_, v)| v.iter().any(|x| x.is_nan())));
    assert!(rows.iter().any(|(_, v)| v.iter().any(|x| x.is_infinite())));
    rows
}

fn csv_line(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:?}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_decimal_predictions(what: &str, text: &str) -> Vec<u64> {
    text.lines()
        .map(|line| {
            line.trim()
                .parse::<f64>()
                .unwrap_or_else(|error| panic!("{what}: prediction {line:?}: {error}"))
                .to_bits()
        })
        .collect()
}

/// `parity_bits.rs` `assert_manifest_round_trip`: the manifest's program is the emitted tree.
fn assert_manifest_round_trip(manifest: &str, genome: &GenomeIr, what: &str) {
    let value: serde_json::Value = serde_json::from_str(manifest).unwrap();
    let source = value["program"]
        .as_str()
        .unwrap_or_else(|| panic!("{what}: missing manifest program"));
    let parsed = GenomeIr::parse(source).unwrap();
    assert_eq!(parsed.canonical(), source, "{what}: manifest fixed point");
    assert_eq!(
        KernelProgram::import_legacy(&parsed).to_bytes().unwrap(),
        KernelProgram::import_legacy(genome).to_bytes().unwrap(),
        "{what}: manifest changed the executable tree"
    );
}

/// The winner's predictions in `lane`, as `parity_bits.rs` `lane_predictions` produces them.
fn lane_predictions(
    lane: Lane,
    genome: &GenomeIr,
    train: (&[Vec<f64>], &[f64]),
    rows: &[(String, Vec<f64>)],
    dir: &Path,
) -> Vec<u64> {
    let what = format!("emitted winner {}", lane.id());
    let train_csv = dir.join("train.csv");
    let features_csv = dir.join("features-all.csv");
    let features_bits = dir.join("features-all.bits");
    let mut text = String::new();
    for (row, target) in train.0.iter().zip(train.1) {
        let mut cells = row.clone();
        cells.push(*target);
        text.push_str(&csv_line(&cells));
        text.push('\n');
    }
    fs::write(&train_csv, text).unwrap();
    let mut text = String::new();
    for (_, values) in rows {
        text.push_str(&csv_line(values));
        text.push('\n');
    }
    fs::write(&features_csv, text).unwrap();
    let mut text = format!("{} {WIDTH}\n", rows.len());
    for (_, values) in rows {
        for value in values {
            text.push_str(&format!("{:016x}\n", value.to_bits()));
        }
    }
    fs::write(&features_bits, text).unwrap();
    let bits = match lane {
        Lane::Rust => {
            let artifact = emit(genome, &EmitOptions::default()).unwrap();
            assert_manifest_round_trip(&artifact.manifest, genome, &what);
            let source = dir.join("model.rs");
            let binary = dir.join("model_bin");
            let predictions = dir.join("predictions.txt");
            fs::write(&source, &artifact.rust).unwrap();
            run_checked(
                &format!("{what}: rustc"),
                Command::new("rustc")
                    .arg("-O")
                    .arg("-o")
                    .arg(&binary)
                    .arg(&source)
                    .env("TMPDIR", dir),
            );
            run_checked(
                &format!("{what}: run"),
                Command::new(&binary)
                    .arg(&train_csv)
                    .arg(&features_csv)
                    .arg(&predictions),
            );
            parse_decimal_predictions(&what, &fs::read_to_string(&predictions).unwrap())
        }
        Lane::Python => {
            let artifact = emit(genome, &EmitOptions::default()).unwrap();
            assert_manifest_round_trip(&artifact.manifest, genome, &what);
            let source = dir.join("model.py");
            let predictions = dir.join("predictions.txt");
            fs::write(&source, &artifact.python).unwrap();
            run_checked(
                &format!("{what}: python3"),
                Command::new("python3")
                    .arg(&source)
                    .arg(&train_csv)
                    .arg(&features_csv)
                    .arg(&predictions)
                    .env("PYTHONDONTWRITEBYTECODE", "1")
                    .env("TMPDIR", dir),
            );
            parse_decimal_predictions(&what, &fs::read_to_string(&predictions).unwrap())
        }
        Lane::Cpp17Fitted => {
            let bound = fit_legacy_numeric_bound(genome, train.0, train.1)
                .unwrap_or_else(|error| panic!("{what}: bind: {error:#}"));
            let artifact =
                emit_fitted_cpp17(&bound).unwrap_or_else(|error| panic!("{what}: emit: {error:#}"));
            let model = dir.join("model.cpp");
            let driver = dir.join("driver.cpp");
            let binary = dir.join("model_bin");
            fs::write(&model, &artifact.source).unwrap();
            fs::write(&driver, fitted_driver()).unwrap();
            run_checked(
                &format!("{what}: {}", lane.tool()),
                feat_math::cxx::command(Some(CxxStd::Cxx17))
                    .args([
                        "-O2",
                        "-ffp-contract=off",
                        "-fno-fast-math",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                    ])
                    .arg(&model)
                    .arg(&driver)
                    .arg("-o")
                    .arg(&binary)
                    .env("TMPDIR", dir),
            );
            let output = run_checked(
                &format!("{what}: run"),
                Command::new(&binary).arg(&features_bits),
            );
            String::from_utf8(output.stdout)
                .unwrap()
                .lines()
                .map(|line| {
                    u64::from_str_radix(line.trim(), 16)
                        .unwrap_or_else(|error| panic!("{what}: prediction {line:?}: {error}"))
                })
                .collect()
        }
    };
    assert_eq!(bits.len(), rows.len(), "{what}: one prediction per row");
    bits
}

/// The runtime's predictions against `lane`'s, every row bit for bit; a NaN on either side is a
/// divergence, as in `parity_bits.rs`.
fn check_lane(lane: Lane) {
    let Some(version) = toolchain(lane) else {
        return;
    };
    let genome = winner();
    let (train_x, train_y) = training_data();
    let rows = prediction_rows();
    let features: Vec<Vec<f64>> = rows.iter().map(|(_, values)| values.clone()).collect();
    let reference: Vec<u64> = fit(&genome, &train_x, &train_y)
        .unwrap()
        .predict(&features)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect();
    let scratch = tempfile::Builder::new()
        .prefix(&format!("emitted-winner-{}-", lane.id()))
        .tempdir_in(scratch_root())
        .unwrap();
    let emitted = lane_predictions(lane, &genome, (&train_x, &train_y), &rows, scratch.path());
    let mut report = String::new();
    for (((id, _), runtime), emitted) in rows.iter().zip(&reference).zip(&emitted) {
        let nan = f64::from_bits(*runtime).is_nan() || f64::from_bits(*emitted).is_nan();
        if runtime != emitted || nan {
            report.push_str(&format!(
                "  {id}: runtime 0x{runtime:016x} ({:?}), emitted 0x{emitted:016x} ({:?})\n",
                f64::from_bits(*runtime),
                f64::from_bits(*emitted)
            ));
        }
    }
    eprintln!(
        "emitted winner [{}] {version}: {} predictions compared",
        lane.id(),
        rows.len()
    );
    assert!(
        report.is_empty(),
        "emitted winner [{}]: the runtime and the emitted program disagree:\n{report}",
        lane.id()
    );
    scratch.close().unwrap();
}

#[test]
fn the_refined_winner_matches_emitted_rust_bit_for_bit() {
    check_lane(Lane::Rust);
}

#[test]
fn the_refined_winner_matches_emitted_python_bit_for_bit() {
    check_lane(Lane::Python);
}

#[test]
fn the_refined_winner_matches_fitted_cpp17_bit_for_bit() {
    check_lane(Lane::Cpp17Fitted);
}

// ------------------------------------------------------------------------------------------------
// The recipe C++17 driver of mathexpr_boundary_parity.rs
// ------------------------------------------------------------------------------------------------

/// `mathexpr_boundary_parity.rs` `rows`: training rows, targets and prediction rows.
fn recipe_rows() -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let train: Vec<Vec<f64>> = (0..32)
        .map(|i| {
            vec![
                i as f64 / 8.0 - 2.0,
                (i % 7) as f64 / 3.0 - 1.0,
                (i % 11) as f64 / 5.0 - 1.0,
            ]
        })
        .collect();
    let targets = train
        .iter()
        .map(|r| 0.7 * r[0] - 0.2 * r[1] + 0.1 * r[2])
        .collect();
    let mut test = vec![
        vec![0.125, -0.75, 1.5],
        vec![-2.0, -1.0, -0.5],
        vec![0.0, -0.0, 0.0],
        vec![f64::from_bits(1), 1e-12, 1.0],
    ];
    for value in [1e-12_f64.to_bits() - 1, 1e-12_f64.to_bits() + 1] {
        test.push(vec![f64::from_bits(value), 0.5, 1.0]);
    }
    (train, targets, test)
}

/// `mathexpr_boundary_parity.rs` `execute(program, "g++", "-std=c++17", "-O2", false)`.
#[test]
fn the_refined_winner_matches_the_cpp17_recipe_bit_for_bit() {
    let original = GenomeIr::parse(PINNED_LM_PROGRAM).unwrap();
    let canonical = original.canonical();
    let genome = GenomeIr::parse(&canonical).unwrap();
    assert_eq!(canonical, genome.canonical());
    let bytes = KernelProgram::import_legacy(&genome).to_bytes().unwrap();
    assert_eq!(
        bytes,
        KernelProgram::from_bytes(&bytes)
            .unwrap()
            .to_bytes()
            .unwrap()
    );
    let recipe = emit_recipe_cpp17(&genome, &EmitOptions::default()).unwrap();
    let (mut train, targets, mut test) = recipe_rows();
    assert!(matches!(genome, GenomeIr::Pipeline(_)));
    // Pipeline entry points sanitize features, so a pipeline also receives non-finite cells.
    for (index, value) in [
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::INFINITY,
        f64::NEG_INFINITY,
    ]
    .into_iter()
    .enumerate()
    {
        train[index][index] = value;
        let mut row = vec![0.75, -1.25, 1.5];
        row[index] = value;
        test.push(row);
    }
    let expected: Vec<u64> = fit_predict(&genome, &train, &targets, &test)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect();
    let scratch = tempfile::Builder::new()
        .prefix("emitted-winner-recipe-")
        .tempdir_in(scratch_root())
        .unwrap();
    let source = scratch.path().join("recipe.cpp");
    let binary = scratch.path().join("recipe");
    fs::write(&source, recipe.source + recipe_driver()).unwrap();
    run_checked(
        "emitted winner recipe: g++",
        Command::new("g++")
            .args(["-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math"])
            .arg(&source)
            .arg("-o")
            .arg(&binary)
            .env("TMPDIR", scratch.path()),
    );
    let mut input = format!("{} {} 3\n", train.len(), test.len());
    for value in train
        .into_iter()
        .flatten()
        .chain(targets)
        .chain(test.into_iter().flatten())
    {
        input.push_str(&format!("{:016x}\n", value.to_bits()));
    }
    let mut child = Command::new(&binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "emitted winner recipe: status {:?}: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Vec<u64> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| u64::from_str_radix(line, 16).unwrap())
        .collect();
    assert_eq!(actual, expected, "emitted winner: g++ -std=c++17");
    scratch.close().unwrap();
}
