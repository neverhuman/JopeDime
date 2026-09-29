//! Per-stage fault telemetry (kernel2 WP-1.4) and the zero-width refusal of the traced entry
//! points (WP-1.8), from `feat_math::kernel::telemetry`.
//!
//! - `apply_op_traced` returns `apply_op`'s rows, and `apply_op_train`'s, bit for bit: at every
//!   op of every fitted program of PR 01's corpus, on the training pass and on held-out rows; for
//!   every expression of PR 01's compatibility corpus as a `map` and a `derive` stage, with and
//!   without append, over every corpus row; for compat_freeze's stage set on its raw stage
//!   matrix; for every op, with append on and off, over raw boundary rows; and past the width cap.
//!   The counts of every expression stage are recounted independently, cell by cell, from the
//!   interpreter's traced calls.
//! - `fit_traced` returns a model whose `serde_json` bytes are `fit`'s, or `fit`'s error, and
//!   `predict_traced` returns `Model::predict`'s bits, for every program of PR 01's corpus (the
//!   portfolio program is skipped, as compat_freeze skips it) and for the five stored models. The
//!   telemetry of the training pass equals `predict_traced`'s on the training rows, and two runs
//!   give identical telemetry.
//! - The workplan's cases: `log(col(0))` on negative and on positive data, `col(9)` at width 4,
//!   `step(col(0), 100)`; and the epilogue count, a compile failure, the symbolic path, an
//!   explicit portfolio, the empty-row identities and the zero-width refusal.
//!
//! PR 01's program corpus is defined in `tests/compat_freeze.rs`. It is copied here, and
//! `the_program_corpus_is_compat_freezes` proves the copy entry for entry against that file and
//! against the names `compat_freeze/model_serde.v1.json` records.

use feat_math::dsl::{parse_dsl, Stage};
use feat_math::kernel::telemetry::{
    apply_op_traced, fit_traced, predict_traced, PipelineTelemetry, StageFaultCounts,
    ZeroWidthInput, FAULT_SLOTS,
};
use feat_math::math_expr::{FaultClass, FaultSet, FaultSeverity, MathExpr};
use feat_math::ops::{apply_op, apply_op_train, fit_op, FittedOp};
use feat_math::{fit, sanitize, GenomeIr, Model, Pcg32, CHAMPION_GENOME, MAX_WIDTH};
use std::collections::BTreeSet;

type Rows = Vec<Vec<f64>>;

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn row_bits(rows: &Rows) -> Vec<Vec<u64>> {
    rows.iter().map(|row| bits(row)).collect()
}

fn fixture_text(relative: &str) -> String {
    let path = format!("{}/tests/fixtures/{relative}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{path}: {error}"))
}

fn fixture_json(relative: &str) -> serde_json::Value {
    serde_json::from_str(&fixture_text(relative))
        .unwrap_or_else(|error| panic!("{relative}: {error}"))
}

fn parse_program(source: &str) -> GenomeIr {
    GenomeIr::parse(source).unwrap_or_else(|error| panic!("{source}: {error}"))
}

fn parse_stage(text: &str) -> Stage {
    parse_dsl(&format!("(pipe X {text})"))
        .unwrap_or_else(|error| panic!("{text}: {error}"))
        .stages
        .remove(0)
}

fn serialized(model: &Model) -> Vec<u8> {
    serde_json::to_vec(model).expect("serialize the model")
}

// ---------------------------------------------------------------------------------------------
// PR 01's program corpus, copied from tests/compat_freeze.rs
// ---------------------------------------------------------------------------------------------

/// Deterministic regression data, as in `tests/golden.rs` and `tests/compat_freeze.rs`.
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

/// Training rows (the first 64) and held-out rows (the last 16), as compat_freeze fits them.
fn fitted_surface_data() -> (Rows, Vec<f64>, Rows) {
    let (x, y) = toy_data(80, 5, 42);
    (x[..64].to_vec(), y[..64].to_vec(), x[64..].to_vec())
}

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
    ("recipe_random_features", "(pipe X (rff_cos n=8) (rff_sin n=8) (rand_proj k=4) (landmark_dist k=4) (ridge))"),
    ("recipe_row_features", "(pipe X (oblique_hinge n=4) (row_stats) (pair_prod) (target_bin_encode) (sigmoid) (ridge))"),
    ("recipe_indicator_features", "(pipe X (step t=0.5 append=true) (norm_row) (row_diff_stats) (elasticnet))"),
    ("recipe_logistic_alias", "(pipe X (standardize) (logistic))"),
    ("target_rank_gauss", "(pipe X (target kind=rank_gauss) (ridge))"),
    ("target_log1p", "(pipe X (target kind=log1p) (ridge lambda_frac=0.02))"),
    ("target_winsor", "(pipe X (target kind=winsor q=0.05) (standardize) (ridge))"),
    ("target_zscore", "(pipe X (target kind=zscore) (standardize) (ridge))"),
    ("step_impute_mean", "(pipe X (step t=0.0) (impute strategy=mean add_indicator=true) (ridge))"),
    ("step_impute_const", "(pipe X (step t=0.0 append=true) (impute strategy=const value=0.5) (ridge))"),
    ("select_corr_by_var", "(pipe X (select_corr k=3 by=var) (ridge))"),
    ("ridge_gcv_dual_route", "(pipe X (rff_cos n=96 seed=3) (ridge_gcv))"),
    ("empty_stage_output", "(pipe X (select_corr k=1) (pair_ratio topk=4) (ridge))"),
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

/// compat_freeze's program set, named and in its order.
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
    let portable = fixture_text("emitted/portable-4lang.dsl");
    set.push(("portable_4lang".to_string(), portable.trim().to_string()));
    set
}

/// compat_freeze skips fitting portfolio programs: their diagnostics carry wall-clock `fit_ms`,
/// so their bytes differ from run to run, and a debug-build fit takes minutes.
fn fit_skipped(source: &str) -> bool {
    source.contains("(portfolio")
}

const COMPAT_FREEZE: &str = include_str!("compat_freeze.rs");

/// The entry lines of the array `const <name>` in compat_freeze.rs, without its comment lines.
fn compat_freeze_entries(name: &str) -> Vec<&'static str> {
    let start = COMPAT_FREEZE
        .find(&format!("const {name}: "))
        .unwrap_or_else(|| panic!("compat_freeze.rs no longer defines {name}"));
    let body = &COMPAT_FREEZE[start..];
    let open = body.find("= &[\n").expect("an array literal") + "= &[\n".len();
    let close = body[open..].find("\n];").expect("the end of the array") + open;
    body[open..close]
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect()
}

#[test]
fn the_program_corpus_is_compat_freezes() {
    let named: Vec<String> = NAMED_PROGRAMS
        .iter()
        .map(|(name, source)| format!("    ({name:?}, {source:?}),"))
        .collect();
    assert_eq!(named, compat_freeze_entries("NAMED_PROGRAMS"));
    for (copy, name) in [
        (IP_SCRUB_PIPELINES, "IP_SCRUB_PIPELINES"),
        (IP_SCRUB_SYMBOLIC, "IP_SCRUB_SYMBOLIC"),
    ] {
        let lines: Vec<String> = copy
            .iter()
            .map(|source| format!("    {source:?},"))
            .collect();
        assert_eq!(lines, compat_freeze_entries(name), "{name}");
    }
    let fixture = fixture_json("compat_freeze/model_serde.v1.json");
    let models = fixture["models"].as_array().expect("models");
    let recorded: Vec<&str> = models
        .iter()
        .map(|model| model["name"].as_str().expect("name"))
        .collect();
    let set = program_set();
    let names: Vec<&str> = set.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, recorded);
    // The fixture records exactly the programs skipped here as skipped.
    for ((name, source), model) in set.iter().zip(models) {
        let skipped = model["serde_json"]
            .as_str()
            .expect("serde_json")
            .starts_with("skipped:");
        assert_eq!(skipped, fit_skipped(source), "{name}");
    }
    assert_eq!(set.len(), 55);
}

// ---------------------------------------------------------------------------------------------
// Independent recount of an expression stage
// ---------------------------------------------------------------------------------------------

/// The fields of `StageFaultCounts`, comparable with an expectation built here.
#[derive(Debug, Default, PartialEq, Eq)]
struct Fields {
    evaluated_cells: u64,
    faulted_cells: [u64; FAULT_SLOTS],
    hard_rows: u64,
    compile_failed: bool,
    constant_output: bool,
    epilogue_sanitized: u64,
}

fn fields(counts: &StageFaultCounts) -> Fields {
    Fields {
        evaluated_cells: counts.evaluated_cells,
        faulted_cells: counts.faulted_cells,
        hard_rows: counts.hard_rows,
        compile_failed: counts.compile_failed,
        constant_output: counts.constant_output,
        epilogue_sanitized: counts.epilogue_sanitized,
    }
}

/// The counts of a `map` or `derive` stage whose expression compiles, recounted cell by cell from
/// the interpreter's traced calls (PR 03 proved the compiled form's values and faults equal
/// theirs), with the epilogue applied to the generated values by hand.
fn recount(op: &FittedOp, x: &Rows) -> Fields {
    let (expr, elementwise, append) = match op {
        FittedOp::MapExpr { expr, append, .. } => (expr, true, *append),
        FittedOp::DeriveExpr { expr, append, .. } => (expr, false, *append),
        other => panic!("not an expression stage: {other:?}"),
    };
    assert!(
        expr.compile().is_ok(),
        "recount covers compiled expressions"
    );
    let hard = FaultSeverity::Hard.classes();
    let mut expected = Fields::default();
    let mut first: Option<u64> = None;
    let mut varied = false;
    for row in x {
        let cells: Vec<(f64, FaultSet)> = if elementwise {
            row.iter()
                .map(|value| expr.eval_element_traced(*value, row))
                .collect()
        } else {
            vec![expr.eval_row_traced(row)]
        };
        let mut row_is_hard = false;
        for (value, faults) in &cells {
            expected.evaluated_cells += 1;
            for class in faults.iter() {
                expected.faulted_cells[class.index()] += 1;
            }
            row_is_hard |= faults.intersects(hard);
            match first {
                None => first = Some(value.to_bits()),
                Some(bits) => varied |= value.to_bits() != bits,
            }
        }
        if row_is_hard {
            expected.hard_rows += 1;
        }
        let mut merged: Vec<f64> = if append { row.clone() } else { Vec::new() };
        merged.extend(cells.iter().map(|(value, _)| *value));
        merged.truncate(MAX_WIDTH);
        expected.epilogue_sanitized +=
            merged.iter().filter(|value| !value.is_finite()).count() as u64;
    }
    expected.constant_output = first.is_some() && !varied;
    expected
}

/// `apply_op_traced` against `apply_op` and `apply_op_train` on `x`, and its counts against the
/// recount for an expression stage or against "epilogue only" for any other op. Returns the
/// counts.
fn check_op(at: &str, op: &FittedOp, x: &Rows) -> StageFaultCounts {
    let legacy = apply_op(op, x);
    let (traced, counts) = apply_op_traced(op, x);
    assert_eq!(row_bits(&traced), row_bits(&legacy), "{at}: apply_op");
    assert_eq!(
        row_bits(&traced),
        row_bits(&apply_op_train(op, x)),
        "{at}: apply_op_train"
    );
    match op {
        FittedOp::MapExpr { .. } | FittedOp::DeriveExpr { .. } => {
            assert_eq!(fields(&counts), recount(op, x), "{at}: counts");
        }
        _ => {
            let expression = Fields {
                epilogue_sanitized: counts.epilogue_sanitized,
                ..Fields::default()
            };
            assert_eq!(fields(&counts), expression, "{at}: a non-expression op");
        }
    }
    counts
}

/// Every op of a fitted pipeline, replayed on `rows` as `Model::predict` runs them.
fn replay_ops(at: &str, model: &Model, rows: &Rows) -> Vec<StageFaultCounts> {
    let Model::Pipeline { ops, .. } = model else {
        return Vec::new();
    };
    let mut current = rows.clone();
    sanitize(&mut current);
    let mut counts = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        counts.push(check_op(&format!("{at} op {i}"), op, &current));
        current = apply_op(op, &current);
    }
    counts
}

/// The number of telemetry entries a model's pass has.
fn stage_count(model: &Model) -> usize {
    match model {
        Model::Pipeline { ops, .. } => ops.len(),
        Model::Symbolic { .. } => 1,
        Model::Portfolio(_) => 0,
    }
}

// ---------------------------------------------------------------------------------------------
// The corpus: models, predictions and every op
// ---------------------------------------------------------------------------------------------

#[test]
fn fit_traced_serializes_to_fits_bytes_on_every_corpus_program() {
    let (train, target, test) = fitted_surface_data();
    let mut fitted = 0;
    let mut expression_stages = 0;
    for (name, source) in program_set() {
        if fit_skipped(&source) {
            continue;
        }
        let program = parse_program(&source);
        match (
            fit(&program, &train, &target),
            fit_traced(&program, &train, &target),
        ) {
            (Ok(model), Ok((traced, telemetry))) => {
                assert_eq!(
                    serialized(&traced),
                    serialized(&model),
                    "{name}: model bytes"
                );
                assert_eq!(telemetry.stages.len(), stage_count(&model), "{name}");
                assert_eq!(telemetry.input_sanitized, 0, "{name}: finite training rows");
                for (rows, which) in [(&test, "held-out"), (&train, "training")] {
                    let expected = bits(&model.predict(rows));
                    for (fitted_by, m) in [("fit", &model), ("fit_traced", &traced)] {
                        let (predictions, _) = predict_traced(m, rows).expect("finite rows");
                        assert_eq!(
                            bits(&predictions),
                            expected,
                            "{name}: {which} predictions of the {fitted_by} model"
                        );
                    }
                }
                // The training pass is Model::predict's pass over the training rows.
                let (_, replayed) = predict_traced(&model, &train).expect("finite rows");
                assert_eq!(replayed, telemetry, "{name}: training-pass telemetry");
                let training = replay_ops(&format!("{name} training pass"), &model, &train);
                if matches!(model, Model::Pipeline { .. }) {
                    assert_eq!(training, telemetry.stages, "{name}: per-op replay");
                }
                replay_ops(&format!("{name} held-out rows"), &model, &test);
                expression_stages += telemetry
                    .stages
                    .iter()
                    .filter(|stage| stage.evaluated_cells > 0)
                    .count();
                fitted += 1;
            }
            (Err(legacy), Err(traced)) => {
                assert_eq!(
                    format!("{traced:#}"),
                    format!("{legacy:#}"),
                    "{name}: error"
                );
            }
            (legacy, traced) => panic!(
                "{name}: fit {} but fit_traced {}",
                if legacy.is_ok() { "succeeds" } else { "fails" },
                if traced.is_ok() { "succeeds" } else { "fails" }
            ),
        }
    }
    eprintln!("CORPUS fitted={fitted} expression_stages={expression_stages}");
    assert_eq!(fitted, 54);
    assert!(expression_stages > 20);
}

#[test]
fn two_runs_give_identical_telemetry() {
    let (train, target, test) = fitted_surface_data();
    for (name, source) in program_set() {
        if fit_skipped(&source) {
            continue;
        }
        let program = parse_program(&source);
        let (first_model, first) = fit_traced(&program, &train, &target).expect("fits");
        let (second_model, second) = fit_traced(&program, &train, &target).expect("fits");
        assert_eq!(first, second, "{name}: training telemetry");
        assert_eq!(
            serialized(&first_model),
            serialized(&second_model),
            "{name}"
        );
        let first_pass = predict_traced(&first_model, &test).expect("finite rows");
        let second_pass = predict_traced(&second_model, &test).expect("finite rows");
        assert_eq!(bits(&first_pass.0), bits(&second_pass.0), "{name}");
        assert_eq!(first_pass.1, second_pass.1, "{name}: inference telemetry");
        // The telemetry itself round-trips through serde_json.
        let text = serde_json::to_string(&first).expect("serialize the telemetry");
        let back: PipelineTelemetry = serde_json::from_str(&text).expect("parse the telemetry");
        assert_eq!(back, first, "{name}: serde round trip");
    }
}

#[test]
fn predict_traced_equals_predict_on_the_stored_models() {
    let (_, _, test) = fitted_surface_data();
    let fixture = fixture_json("compat_freeze/model_serde.v1.json");
    let stored = fixture["stored"].as_array().expect("stored");
    assert_eq!(stored.len(), 5);
    for entry in stored {
        let name = entry["name"].as_str().expect("name");
        let model: Model =
            serde_json::from_str(entry["json"].as_str().expect("json")).expect("a stored model");
        let (predictions, telemetry) = predict_traced(&model, &test).expect("finite rows");
        assert_eq!(bits(&predictions), bits(&model.predict(&test)), "{name}");
        assert_eq!(telemetry.stages.len(), stage_count(&model), "{name}");
        replay_ops(name, &model, &test);
    }
}

// ---------------------------------------------------------------------------------------------
// Boundary rows
// ---------------------------------------------------------------------------------------------

/// PR 01's compatibility corpus, `compat_freeze/mathexpr_values.v1.json`: every row (the empty
/// row among them), in its order, and every expression source.
fn corpus() -> (Rows, Vec<String>) {
    let file = fixture_json("compat_freeze/mathexpr_values.v1.json");
    let rows: Rows = file["rows"]
        .as_array()
        .expect("rows")
        .iter()
        .map(|row| {
            // The empty row is recorded as "": no cells, not one empty token.
            row["values"]
                .as_str()
                .expect("row values")
                .split(' ')
                .filter(|token| !token.is_empty())
                .map(|token| {
                    f64::from_bits(
                        u64::from_str_radix(token, 16)
                            .unwrap_or_else(|error| panic!("{token}: {error}")),
                    )
                })
                .collect()
        })
        .collect();
    let expressions: Vec<String> = file["expressions"]
        .as_array()
        .expect("expressions")
        .iter()
        .map(|expression| expression["source"].as_str().expect("source").to_string())
        .collect();
    assert!(
        rows.iter().any(Vec::is_empty) && !expressions.is_empty(),
        "the corpus has an empty row and at least one expression"
    );
    (rows, expressions)
}

#[test]
fn expression_stages_equal_apply_op_on_every_corpus_expression_and_row() {
    let (rows, expressions) = corpus();
    let nonfinite = rows.iter().flatten().filter(|v| !v.is_finite()).count() as u64;
    assert!(nonfinite > 0, "the corpus has non-finite cells");
    let mut stages = 0;
    let mut faulted_stages = 0;
    let mut hard_stages = 0;
    for source in &expressions {
        let expr = MathExpr::parse(source).expect("a corpus expression parses");
        for append in [false, true] {
            let ops = [
                FittedOp::MapExpr {
                    source: source.clone(),
                    expr: expr.clone(),
                    append,
                },
                FittedOp::DeriveExpr {
                    source: source.clone(),
                    expr: expr.clone(),
                    append,
                },
            ];
            for op in &ops {
                let counts = check_op(&format!("{source} append={append}"), op, &rows);
                // Expression outputs are finite, so only merged input cells reach the epilogue.
                let merged = if append { nonfinite } else { 0 };
                assert_eq!(
                    counts.epilogue_sanitized, merged,
                    "{source} append={append}"
                );
                stages += 1;
                faulted_stages += usize::from(counts.faulted_cells.iter().any(|&n| n > 0));
                hard_stages += usize::from(counts.hard_rows > 0);
            }
        }
    }
    eprintln!(
        "BOUNDARY expressions={} rows={} stages={stages} faulted={faulted_stages} hard={hard_stages}",
        expressions.len(),
        rows.len()
    );
    assert!(faulted_stages > 0 && hard_stages > 0);
}

/// compat_freeze's 5x4 stage matrix: ordinary values, signed zeros and guard neighbours,
/// non-finite values, magnitudes beyond the clamp, and rounding cases.
fn stage_matrix() -> Rows {
    vec![
        vec![1.0, -2.0, 0.5, 3.0],
        vec![0.0, -0.0, 1e-12, -1e-12],
        vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 2.0],
        vec![1e155, -parse_f64("1.3407807929942596e154"), 27.0, -30.0],
        vec![toward_zero(0.5), 2.5, -1.0, 4_503_599_627_370_497.0],
    ]
}

/// The next representable value toward zero (finite, nonzero input).
fn toward_zero(value: f64) -> f64 {
    f64::from_bits(value.to_bits() - 1)
}

fn parse_f64(text: &str) -> f64 {
    text.parse().expect("decimal literal")
}

/// compat_freeze's `STAGES`, verbatim.
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

#[test]
fn compat_freeze_stages_equal_apply_op_on_the_raw_stage_matrix() {
    let lines: Vec<String> = STAGES
        .iter()
        .map(|stage| format!("    {stage:?},"))
        .collect();
    assert_eq!(lines, compat_freeze_entries("STAGES"));
    let (train, target) = toy_data(40, 5, 42);
    let train: Rows = train.iter().map(|row| row[..4].to_vec()).collect();
    let matrix = stage_matrix();
    for text in STAGES {
        // As compat_freeze's stage_case: fitted on the raw matrix, applied to it.
        let op = fit_op(&parse_stage(text), &matrix, &target).expect("fits");
        check_op(text, &op, &matrix);
        // Fitted as a program on the training rows, then predicting the matrix.
        let program = parse_program(&format!("(pipe X {text} (ridge))"));
        let model = fit(&program, &train, &target).expect("fits");
        let (traced, telemetry) = fit_traced(&program, &train, &target).expect("fits");
        assert_eq!(serialized(&traced), serialized(&model), "{text}");
        assert_eq!(telemetry.stages.len(), 1);
        let (predictions, pass) = predict_traced(&model, &matrix).expect("no zero-width row");
        assert_eq!(bits(&predictions), bits(&model.predict(&matrix)), "{text}");
        // NaN, inf and -inf: the entry sanitize replaces them before the stage.
        assert_eq!(pass.input_sanitized, 3, "{text}");
        assert_eq!(pass.stages, replay_ops(text, &model, &matrix), "{text}");
    }
}

/// Raw rows of width 5 with what the entry sanitize would remove and what overflows in the ops.
fn boundary_rows() -> Rows {
    vec![
        vec![0.5, -1.0, 1.5, -0.25, 2.0],
        vec![0.0, -0.0, 1e-300, -1e-300, 5e-324],
        vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0, -1.0],
        vec![1e200, -1e200, 1e155, -1e155, 3.0],
        vec![-2.0, 2.0, 0.0, 1e12, -1e12],
    ]
}

/// One stage for every `FittedOp` variant, as the DSL names them, with the variant it fits.
const EVERY_OP: [(&str, &str); 36] = [
    ("standardize", "Standardize"),
    ("robust_scale", "RobustScale"),
    ("rank", "Rank"),
    ("blom", "Blom"),
    ("quantile_bins", "QuantileBins"),
    ("bin_onehot bins=3", "BinOnehot"),
    ("hinge", "Hinge"),
    ("winsor_clip", "WinsorClip"),
    ("cheb degree=3", "Cheb"),
    ("poly_interact", "PolyInteract"),
    ("rff_cos n=8", "RffCos"),
    ("rff_sin n=8", "RffSin"),
    ("rand_proj k=4", "RandProj"),
    ("landmark_dist k=4", "LandmarkDist"),
    ("oblique_hinge n=4", "ObliqueHinge"),
    ("row_stats", "RowStats"),
    ("pair_prod", "PairProd"),
    ("target_bin_encode", "TargetBinEncode"),
    ("sigmoid", "Sigmoid"),
    ("step t=0.5", "Step"),
    ("norm_row", "NormRow"),
    ("polar", "Polar"),
    ("lag k=1", "Lag"),
    ("roll k=2", "Roll"),
    ("fft_mag k=2", "FftMag"),
    ("row_spectral", "RowSpectral"),
    ("row_diff_stats", "RowDiffStats"),
    ("svd_project k=2 iters=5", "SvdProject"),
    ("nystrom k=4", "Nystrom"),
    ("map expr=\"x*x - c1 + row_mean\"", "MapExpr"),
    ("derive expr=\"c0/c1 + log(c2)\"", "DeriveExpr"),
    ("symlog c=0.7", "Symlog"),
    ("select_corr k=3", "SelectCorr"),
    ("impute strategy=median add_indicator=true", "Impute"),
    ("pair_ratio topk=4", "PairRatio"),
    ("spline knots=4", "Spline"),
];

#[test]
fn every_op_equals_apply_op_with_append_on_and_off() {
    let (mut train, target) = toy_data(64, 5, 42);
    // Exact zeros, so impute fits a fill and an indicator column.
    train[3][2] = 0.0;
    train[11][4] = 0.0;
    let raw = boundary_rows();
    let mut variants = BTreeSet::new();
    for (text, variant) in EVERY_OP {
        for append in [false, true] {
            let stage = parse_stage(&format!("({text} append={append})"));
            let op = fit_op(&stage, &train, &target).unwrap_or_else(|e| panic!("{text}: {e}"));
            let json = serde_json::to_value(&op).expect("serialize the op");
            let tag = json
                .as_object()
                .and_then(|object| object.keys().next())
                .expect("an externally tagged variant");
            assert_eq!(tag, variant, "{text}");
            variants.insert(tag.clone());
            let appended = json[variant]["append"].as_bool();
            assert_eq!(appended, Some(append), "{text}: the fitted append flag");
            check_op(&format!("{text} append={append} (train)"), &op, &train);
            check_op(&format!("{text} append={append} (raw)"), &op, &raw);
        }
    }
    assert_eq!(variants.len(), 36, "one stage per FittedOp variant");
}

#[test]
fn the_width_cap_comes_before_the_epilogue_count() {
    // 300 columns appended to their 300 transforms make 600, cut to MAX_WIDTH = 512.
    let mut row = vec![1.0; 300];
    row[0] = 1e200;
    row[1] = 1e200;
    row[298] = 1e200;
    row[299] = 1e200;
    let x = vec![row];
    // polar's first radius, at merged column 300, overflows and is kept; its last, at merged
    // column 598, overflows and is cut.
    let polar = fit_op(&parse_stage("(polar append=true)"), &x, &[0.0]).expect("fits");
    let counts = check_op("polar append=true, 300 columns", &polar, &x);
    assert_eq!(counts.epilogue_sanitized, 1);
    let (rows, _) = apply_op_traced(&polar, &x);
    assert_eq!(rows[0].len(), MAX_WIDTH);
    // An expression's counts cover every cell it generated, the cut ones included.
    let map = fit_op(&parse_stage("(map expr=\"x\" append=true)"), &x, &[0.0]).expect("fits");
    let counts = check_op("map append=true, 300 columns", &map, &x);
    assert_eq!(counts.evaluated_cells, 300);
    assert_eq!(counts.epilogue_sanitized, 0);
}

// ---------------------------------------------------------------------------------------------
// The workplan's cases (WP-1.4)
// ---------------------------------------------------------------------------------------------

/// The first stage of a training pass.
fn first_stage(source: &str, x: &Rows, y: &[f64]) -> (Model, StageFaultCounts) {
    let program = parse_program(source);
    let (model, telemetry) = fit_traced(&program, x, y).expect("fits");
    assert_eq!(
        serialized(&model),
        serialized(&fit(&program, x, y).expect("fits")),
        "{source}"
    );
    (model, telemetry.stages[0].clone())
}

fn only_class(counts: &StageFaultCounts, class: FaultClass, n: u64) {
    let mut expected = [0; FAULT_SLOTS];
    expected[class.index()] = n;
    assert_eq!(counts.faulted_cells, expected, "only {class}");
}

#[test]
fn log_of_a_negative_column_counts_log_domain_and_of_a_positive_one_nothing() {
    let (x, y) = toy_data(64, 5, 42);
    let source = "(pipe X (derive expr=\"log(col(0))\") (ridge))";
    let (_, stage) = first_stage(source, &x, &y);
    let outside = x.iter().filter(|row| row[0] <= 1e-300).count() as u64;
    assert!(outside > 0, "toy_data has non-positive values in column 0");
    assert_eq!(stage.faulted(FaultClass::LogDomain), outside);
    only_class(&stage, FaultClass::LogDomain, outside);
    assert_eq!(stage.evaluated_cells, 64);
    assert_eq!(stage.hard_rows, 0);
    assert!(!stage.compile_failed && !stage.constant_output);

    let positive: Rows = x
        .iter()
        .map(|row| row.iter().map(|v| v.abs() + 1.0).collect())
        .collect();
    let (_, stage) = first_stage(source, &positive, &y);
    assert_eq!(stage.faulted(FaultClass::LogDomain), 0);
    assert_eq!(stage.faulted_cells, [0; FAULT_SLOTS]);
    assert_eq!(stage.evaluated_cells, 64);
}

#[test]
fn a_column_past_the_width_is_a_hard_row_and_the_legacy_zeros() {
    let (x, y) = toy_data(40, 5, 42);
    let x: Rows = x.iter().map(|row| row[..4].to_vec()).collect();
    let (model, stage) = first_stage("(pipe X (map expr=\"col(9)\") (ridge))", &x, &y);
    assert_eq!(stage.hard_rows, 40);
    assert_eq!(stage.evaluated_cells, 160);
    only_class(&stage, FaultClass::ColumnOob, 160);
    assert!(stage.constant_output && !stage.compile_failed);
    let Model::Pipeline { ops, .. } = &model else {
        panic!("a pipeline")
    };
    let (rows, _) = apply_op_traced(&ops[0], &x);
    assert_eq!(row_bits(&rows), row_bits(&apply_op(&ops[0], &x)));
    assert!(rows.iter().flatten().all(|value| value.to_bits() == 0));
    assert_eq!(rows.len(), 40);
    assert!(rows.iter().all(|row| row.len() == 4));
}

#[test]
fn a_step_above_the_data_is_a_constant_output() {
    let (x, y) = toy_data(64, 5, 42);
    let (_, stage) = first_stage(
        "(pipe X (derive expr=\"step(col(0), 100)\") (ridge))",
        &x,
        &y,
    );
    assert!(stage.constant_output);
    assert_eq!(stage.faulted_cells, [0; FAULT_SLOTS]);
    assert_eq!(stage.hard_rows, 0);
    let (_, stage) = first_stage("(pipe X (derive expr=\"step(col(0), 0)\") (ridge))", &x, &y);
    assert!(!stage.constant_output);
    // Constant means bit-identical: +0.0 and -0.0 differ.
    let derive = fit_op(&parse_stage("(derive expr=\"c0\")"), &x, &y).expect("fits");
    let zeros: Rows = vec![vec![0.0], vec![-0.0]];
    assert!(!check_op("c0 on +0.0 and -0.0", &derive, &zeros).constant_output);
    let zeros: Rows = vec![vec![-0.0], vec![-0.0]];
    assert!(check_op("c0 on -0.0 twice", &derive, &zeros).constant_output);
}

#[test]
fn a_stage_that_emits_a_non_finite_value_counts_the_epilogue() {
    let (mut x, y) = toy_data(32, 5, 42);
    // symlog's asinh overflows to ±inf above about 1.34e154; the epilogue replaces it with 0.0.
    x[3][1] = 1e200;
    x[7][4] = -1e200;
    let (model, stage) = first_stage("(pipe X (symlog c=1) (ridge))", &x, &y);
    assert_eq!(stage.epilogue_sanitized, 2);
    assert_eq!(stage.evaluated_cells, 0);
    assert_eq!(stage.faulted_cells, [0; FAULT_SLOTS]);
    let (_, pass) = predict_traced(&model, &x).expect("no zero-width row");
    assert_eq!(pass.stages[0].epilogue_sanitized, 2);
    // A merged input cell counts too: an expression stage on raw rows, with append.
    let op = fit_op(&parse_stage("(map expr=\"x\" append=true)"), &x, &y).expect("fits");
    let raw = vec![vec![f64::NAN, 1.0]];
    let counts = check_op("map x append on a raw NaN", &op, &raw);
    assert_eq!(counts.epilogue_sanitized, 1);
    assert_eq!(counts.faulted(FaultClass::NonfiniteInput), 1);
    assert_eq!(counts.faulted(FaultClass::NonfiniteSanitized), 1);
}

#[test]
fn a_structurally_invalid_expression_is_a_compile_failure() {
    // parse and from_ast refuse this arity; serde decoding, a compatibility interface, does not.
    let bad: MathExpr = serde_json::from_value(serde_json::json!({
        "root": {"Call": ["Sin", [{"Const": 1.0}, {"Const": 2.0}]]}
    }))
    .expect("the historical AST encoding");
    assert!(bad.compile().is_err());
    let rows: Rows = vec![vec![1.0, 2.0, 3.0], vec![], vec![-1.0]];
    for append in [false, true] {
        let map = FittedOp::MapExpr {
            source: "sin(1, 2)".to_string(),
            expr: bad.clone(),
            append,
        };
        let (out, counts) = apply_op_traced(&map, &rows);
        assert_eq!(row_bits(&out), row_bits(&apply_op(&map, &rows)));
        assert!(counts.compile_failed && counts.constant_output);
        assert_eq!(counts.evaluated_cells, 4);
        only_class(&counts, FaultClass::CompileFailed, 4);
        // The empty row generates no cell, so it is not a hard row.
        assert_eq!(counts.hard_rows, 2);
        assert_eq!(counts.epilogue_sanitized, 0);

        let derive = FittedOp::DeriveExpr {
            source: "sin(1, 2)".to_string(),
            expr: bad.clone(),
            append,
        };
        let (out, counts) = apply_op_traced(&derive, &rows);
        assert_eq!(row_bits(&out), row_bits(&apply_op(&derive, &rows)));
        assert!(counts.compile_failed && counts.constant_output);
        assert_eq!(counts.evaluated_cells, 3);
        only_class(&counts, FaultClass::CompileFailed, 3);
        assert_eq!(counts.hard_rows, 3);
    }
    // A symbolic program does not compile: fit and predict call the interpreter, whose arity check
    // raises COMPILE_FAILED on each row. The stage's compile flag stays false.
    let (x, y) = toy_data(8, 5, 42);
    let program = GenomeIr::Symbolic {
        source: "sin(1, 2)".to_string(),
        expr: bad,
        affine: true,
    };
    let model = fit(&program, &x, &y).expect("fits");
    let (traced, telemetry) = fit_traced(&program, &x, &y).expect("fits");
    assert_eq!(serialized(&traced), serialized(&model));
    let stage = &telemetry.stages[0];
    assert!(!stage.compile_failed && stage.constant_output);
    assert_eq!(stage.hard_rows, 8);
    only_class(stage, FaultClass::CompileFailed, 8);
    let (predictions, pass) = predict_traced(&model, &x).expect("no zero-width row");
    assert_eq!(bits(&predictions), bits(&model.predict(&x)));
    assert_eq!(pass, telemetry);
}

#[test]
fn the_symbolic_path_reads_raw_inputs_and_a_pipeline_sanitizes_them() {
    let (mut x, y) = toy_data(32, 5, 42);
    x[2][0] = f64::NAN;
    x[9][0] = f64::INFINITY;
    let (model, stage) = first_stage("(sym raw \"c0 + c1\")", &x, &y);
    assert_eq!(stage.evaluated_cells, 32);
    assert_eq!(stage.faulted(FaultClass::NonfiniteInput), 2);
    assert_eq!(stage.faulted(FaultClass::NonfiniteSanitized), 2);
    assert_eq!(stage.hard_rows, 0);
    assert_eq!(stage.epilogue_sanitized, 0);
    let (predictions, pass) = predict_traced(&model, &x).expect("no zero-width row");
    assert_eq!(bits(&predictions), bits(&model.predict(&x)));
    assert_eq!(pass.input_sanitized, 0);
    assert_eq!(pass.stages[0], stage);

    // Through a pipeline, the entry sanitize replaces both cells first: the stage reads zeros.
    let program = parse_program("(pipe X (derive expr=\"c0 + c1\") (ridge))");
    let (_, telemetry) = fit_traced(&program, &x, &y).expect("fits");
    assert_eq!(telemetry.input_sanitized, 2);
    assert_eq!(telemetry.stages[0].faulted_cells, [0; FAULT_SLOTS]);

    // `x` outside a map: every row is a Hard fallback, and the affine fit takes its flat branch.
    let (model, stage) = first_stage("(sym affine \"x\")", &x, &y);
    assert_eq!(stage.hard_rows, 32);
    only_class(&stage, FaultClass::IllegalScope, 32);
    assert!(stage.constant_output);
    let Model::Symbolic { a, .. } = model else {
        panic!("a symbolic model")
    };
    assert_eq!(a.to_bits(), 0.0f64.to_bits());
}

#[test]
fn an_explicit_portfolio_is_fitted_by_fit_and_reports_no_stage() {
    let (train, target, test) = fitted_surface_data();
    let program = parse_program(
        "(portfolio (member (pipe X (ridge))) (member (pipe X (standardize) (ridge))) \
         (blend kind=nnls folds=5 repeats=1 guard=clip))",
    );
    let model = fit(&program, &train, &target).expect("fits");
    let (traced, telemetry) = fit_traced(&program, &train, &target).expect("fits");
    assert_eq!(telemetry, PipelineTelemetry::default());
    // Serialized portfolios carry wall-clock fit times, so only the predictions are compared.
    let expected = bits(&model.predict(&test));
    for m in [&model, &traced] {
        let (predictions, pass) = predict_traced(m, &test).expect("no zero-width row");
        assert_eq!(bits(&predictions), expected);
        assert_eq!(pass, PipelineTelemetry::default());
    }
}

#[test]
fn stage_level_empty_row_identities() {
    // WP-1.8 at stage level: the empty sum, l1 and l2 are identities; mean, std, min and max of
    // an empty row raise EMPTY_ROW_REDUCTION (min and max also reach the output filter).
    let empty: Rows = vec![Vec::new()];
    let cases = [
        ("row_sum", FaultSet::EMPTY),
        ("row_l1", FaultSet::EMPTY),
        ("row_l2", FaultSet::EMPTY),
        ("row_mean", FaultSet::EMPTY_ROW_REDUCTION),
        ("row_std", FaultSet::EMPTY_ROW_REDUCTION),
        (
            "row_min",
            FaultSet::EMPTY_ROW_REDUCTION.union(FaultSet::NONFINITE_SANITIZED),
        ),
        (
            "row_max",
            FaultSet::EMPTY_ROW_REDUCTION.union(FaultSet::NONFINITE_SANITIZED),
        ),
    ];
    for (source, faults) in cases {
        let expr = MathExpr::parse(source).expect("parses");
        let derive = FittedOp::DeriveExpr {
            source: source.to_string(),
            expr: expr.clone(),
            append: false,
        };
        let counts = check_op(source, &derive, &empty);
        assert_eq!(counts.evaluated_cells, 1, "{source}");
        let mut expected = [0; FAULT_SLOTS];
        for class in faults.iter() {
            expected[class.index()] = 1;
        }
        assert_eq!(counts.faulted_cells, expected, "{source}");
        assert_eq!(counts.hard_rows, 0, "{source}");
        // A map over an empty row generates nothing; the epilogue pads the row with one 0.0.
        let map = FittedOp::MapExpr {
            source: source.to_string(),
            expr,
            append: false,
        };
        let (rows, counts) = apply_op_traced(&map, &empty);
        assert_eq!(row_bits(&rows), vec![vec![0]]);
        assert_eq!(counts.evaluated_cells, 0);
        assert!(!counts.constant_output);
        assert_eq!(counts.epilogue_sanitized, 0);
    }
}

// ---------------------------------------------------------------------------------------------
// The zero-width refusal (WP-1.8)
// ---------------------------------------------------------------------------------------------

#[test]
fn the_traced_entry_points_refuse_a_zero_width_input() {
    let ragged: Rows = vec![vec![0.5, 1.0], Vec::new(), vec![2.0, -1.0]];
    let empty_rows: Rows = vec![Vec::new(); 3];
    let y = vec![1.0, 2.0, 3.0];
    let programs = [
        "(pipe X (ridge))",
        "(pipe X (standardize) (ridge))",
        "(sym raw \"row_sum\")",
        "(portfolio (member (pipe X (ridge))) (member (pipe X (standardize) (ridge))) \
         (blend kind=nnls folds=5 repeats=1 guard=clip))",
    ];
    for source in programs {
        let program = parse_program(source);
        for (x, row) in [(&ragged, 1), (&empty_rows, 0)] {
            let error = fit_traced(&program, x, &y).expect_err("refused");
            assert_eq!(
                error.downcast_ref::<ZeroWidthInput>(),
                Some(&ZeroWidthInput { row }),
                "{source}"
            );
            assert_eq!(
                error.to_string(),
                format!("zero-width input: row {row} has no columns")
            );
            assert!(error.to_string().starts_with(ZeroWidthInput::REASON));
        }
        // fit's own check comes first, with fit's message.
        let error = fit_traced(&program, &Vec::new(), &[]).expect_err("refused");
        let legacy = fit(&program, &Vec::new(), &[]).expect_err("refused");
        assert_eq!(error.to_string(), legacy.to_string());
        let error = fit_traced(&program, &empty_rows, &y[..2]).expect_err("refused");
        assert_eq!(
            error.to_string(),
            "training data is empty or target is short"
        );
    }
    // The legacy fit, for contrast: a program without transforms accepts zero-width rows, and a
    // transform refuses them itself (the domain table records the difference).
    assert!(fit(&parse_program("(pipe X (ridge))"), &empty_rows, &y).is_ok());
    assert!(fit(&parse_program("(sym raw \"row_sum\")"), &empty_rows, &y).is_ok());
    let legacy = fit(
        &parse_program("(pipe X (standardize) (ridge))"),
        &empty_rows,
        &y,
    )
    .expect_err("refused by the stage");
    assert_eq!(legacy.to_string(), "empty input to stage standardize");

    let (train, target, _) = fitted_surface_data();
    for source in ["(pipe X (standardize) (ridge))", "(sym raw \"c0 + c1\")"] {
        let model = fit(&parse_program(source), &train, &target).expect("fits");
        let error = predict_traced(&model, &ragged).expect_err("refused");
        assert_eq!(
            error.downcast_ref::<ZeroWidthInput>(),
            Some(&ZeroWidthInput { row: 1 })
        );
        // A set with no rows is not zero-width.
        let (predictions, pass) = predict_traced(&model, &Vec::new()).expect("accepted");
        assert!(predictions.is_empty() && model.predict(&Vec::new()).is_empty());
        assert_eq!(pass.stages.len(), 1);
        assert_eq!(pass.stages[0].evaluated_cells, 0);
        assert!(!pass.stages[0].constant_output);
    }
    assert_eq!(
        ZeroWidthInput::find(&ragged),
        Some(ZeroWidthInput { row: 1 })
    );
    assert_eq!(ZeroWidthInput::find(&train), None);
    assert_eq!(ZeroWidthInput::find(&Vec::new()), None);
}

#[test]
fn the_telemetry_types_stay_shareable_across_threads() {
    fn shareable<T: Send + Sync + 'static>() {}
    shareable::<StageFaultCounts>();
    shareable::<PipelineTelemetry>();
    shareable::<ZeroWidthInput>();
}
