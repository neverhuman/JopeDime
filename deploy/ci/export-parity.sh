#!/usr/bin/env bash
# Local, fail-closed spec 10.5 qualification; no GitHub dispatch or new Cargo workspace.
set -euo pipefail
cd "$(dirname "$0")/../.."
fault=0
strict_calibration=0
case ${1:-} in
  --fault-emitter) fault=1; shift ;;
  --fault-routing) fault=2; shift ;;
  # A mutated emitted Platt map must fail the calibration goldens, admitted or not.
  --fault-calibration) fault=3; strict_calibration=1; shift ;;
  # Before admitting a calibration row: every calibration golden must pass.
  --require-calibration) strict_calibration=1; shift ;;
esac
[[ $# -le 1 ]] || { echo 'usage: export-parity.sh [--fault-emitter|--fault-routing|--fault-calibration|--require-calibration] [NEW_REPORT_DIR]' >&2; exit 2; }
report=${1:-"$PWD/target/export-parity/$(date -u +%Y%m%dT%H%M%SZ)-$$"}
[[ ! -e "$report" && ! -L "$report" ]] || { echo 'report directory must be new' >&2; exit 2; }
mkdir -p -- "$report"
report=$(realpath -- "$report")
cp -- deploy/ci/export-parity.sh "$report/harness.sh"
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} RAYON_NUM_THREADS=1
[[ $CARGO_BUILD_JOBS =~ ^[1-8]$ ]] || { echo 'build jobs must be 1..8' >&2; exit 2; }
export TMPDIR="${CARGO_TARGET_DIR:-$PWD/target}/tmp"
mkdir -p -- "$TMPDIR"
export JOPE_EXPORT_PARITY_REPORT="$report" JOPE_EXPORT_PARITY_FAULT="$fault"
export JOPE_EXPORT_PARITY_STRICT_CALIBRATION="$strict_calibration"
cat > "$report/generate.rs" <<'RUST'
use feat_math::{
    evolve::api::capability::{CapabilityKind, Language, Registry},
    genome::GenomeIr,
    model::{fit, Model},
};
use jope_runtime::{fitted_js, fitted_py, fitted_rs};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};
const WIDTH: usize = 12;
/// Calibration rows with dedicated goldens below; they run whether or not they are admitted.
const CALIBRATION_GOLDENS: [&str; 2] = ["affine_floor", "platt"];
fn rows(n: usize, mut seed: u64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|_| {
            (0..WIDTH)
                .map(|j| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    if j == WIDTH - 1 {
                        0.25
                    } else {
                        (seed >> 11) as f64 / (1u64 << 53) as f64
                    }
                })
                .collect()
        })
        .collect()
}
fn probe(kind: CapabilityKind, name: &str) -> String {
    if matches!(
        name,
        "gbm" | "histgbm" | "histgbm2" | "ensemble" | "ensemble_ridge_gbm"
    ) {
        return format!("(pipe X ({name} rounds=4 depth=2 leaves=4 min_leaf=2 bags=1))");
    }
    let op = match name {
        "target" => "target kind=winsor",
        "map" | "derive" => return format!("(pipe X ({name} expr=\"col(0)\") (ridge))"),
        "gpr" => "gpr m=50 steps=1",
        "krr" => "krr m=50",
        "portfolio" => "portfolio preset=core",
        _ => name,
    };
    if matches!(kind, CapabilityKind::Calibration | CapabilityKind::Encoding) {
        return "(pipe X (ridge))".into();
    }
    if kind == CapabilityKind::Head {
        format!("(pipe X ({op}))")
    } else {
        format!("(pipe X ({op}) (ridge))")
    }
}
fn save_case(
    out: &Path,
    cases: &mut Vec<Value>,
    key: String,
    dsl: &str,
    genome: &GenomeIr,
    model: &Model,
    test: &[Vec<f64>],
    languages: &[Language],
    decision: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let index = cases.len();
    let dir = out.join(format!("case-{index}"));
    fs::create_dir(&dir)?;
    let expected = model.predict(&test.to_vec())?;
    assert!(
        expected.len() == test.len() && expected.iter().all(|v| v.is_finite()),
        "non-finite native golden: {key}"
    );
    fs::write(dir.join("model.json"), serde_json::to_vec(model)?)?;
    let mut emissions = Vec::new();
    for language in languages {
        let (name, ext) = match language {
            Language::Rust => ("rust", "rs"),
            Language::Python => ("python", "py"),
            Language::Javascript => ("javascript", "mjs"),
            _ => unreachable!(),
        };
        let emit = || -> Result<String, String> {
            match language {
                Language::Rust => fitted_rs::emit_with_checks(
                    genome,
                    model,
                    WIDTH,
                    &test[..64]
                        .iter()
                        .zip(&expected)
                        .map(|(r, y)| fitted_rs::CheckRow {
                            inputs: r.clone(),
                            expected: *y,
                        })
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| format!("{e:#}")),
                Language::Python => fitted_py::emit_with_checks(
                    genome,
                    model,
                    WIDTH,
                    &test[..64]
                        .iter()
                        .zip(&expected)
                        .map(|(r, y)| fitted_py::CheckRow {
                            inputs: r.clone(),
                            expected: *y,
                        })
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| format!("{e:#}")),
                Language::Javascript => fitted_js::emit_with_checks(
                    genome,
                    model,
                    WIDTH,
                    &test[..64]
                        .iter()
                        .zip(&expected)
                        .map(|(r, y)| fitted_js::CheckRow {
                            inputs: r.clone(),
                            expected: *y,
                        })
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| format!("{e:#}")),
                _ => unreachable!(),
            }
        };
        match emit() {
            Ok(source) => {
                assert_eq!(source, emit()?, "nondeterministic emitter: {key}");
                fs::write(dir.join(format!("predictor.{ext}")), source)?;
                emissions.push(json!({"language":name,"status":"generated","file":format!("case-{index}/predictor.{ext}")}));
            }
            Err(error) => {
                emissions.push(json!({"language":name,"status":"emit_error","error":error}))
            }
        }
    }
    cases.push(json!({"key":key,"dsl":dsl,"directory":format!("case-{index}"),"rows":test,"expected":expected,"decision":decision,"golden_rows":64,"emissions":emissions}));
    Ok(())
}
/// Emit one calibrated golden in every primary language. `fixture` carries the native
/// reference; `admitted` records the registry row, which decides qualification only.
#[allow(clippy::too_many_arguments)]
fn save_calibrated(
    out: &Path,
    calibrated: &mut Vec<Value>,
    key: &str,
    registry: &Registry,
    name: &str,
    genome: &GenomeIr,
    model: &Model,
    calibration: &fitted_js::ExportCalibration,
    checks: &[(Vec<f64>, f64)],
    mut fixture: Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let index = calibrated.len();
    let dir = out.join(format!("calibration-{index}"));
    fs::create_dir(&dir)?;
    let mut emissions = Vec::new();
    for language in [Language::Rust, Language::Python, Language::Javascript] {
        let (lang, ext) = match language {
            Language::Rust => ("rust", "rs"),
            Language::Python => ("python", "py"),
            _ => ("javascript", "mjs"),
        };
        let emit = || -> Result<String, String> {
            match language {
                Language::Rust => fitted_rs::emit_with_checks_and_platt(
                    genome,
                    model,
                    WIDTH,
                    calibration,
                    &checks
                        .iter()
                        .map(|(inputs, expected)| fitted_rs::CheckRow {
                            inputs: inputs.clone(),
                            expected: *expected,
                        })
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| format!("{e:#}")),
                Language::Python => fitted_py::emit_with_checks_and_platt(
                    genome,
                    model,
                    WIDTH,
                    calibration,
                    &checks
                        .iter()
                        .map(|(inputs, expected)| fitted_py::CheckRow {
                            inputs: inputs.clone(),
                            expected: *expected,
                        })
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| format!("{e:#}")),
                _ => fitted_js::emit_with_checks_and_platt(
                    genome,
                    model,
                    WIDTH,
                    calibration,
                    &checks
                        .iter()
                        .map(|(inputs, expected)| fitted_js::CheckRow {
                            inputs: inputs.clone(),
                            expected: *expected,
                        })
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| format!("{e:#}")),
            }
        };
        let admitted = registry.supports(CapabilityKind::Calibration, name, language);
        match emit() {
            Ok(source) => {
                assert_eq!(source, emit()?, "nondeterministic emitter: {key}");
                fs::write(dir.join(format!("predictor.{ext}")), source)?;
                emissions.push(json!({"language":lang,"admitted":admitted,"status":"generated","file":format!("calibration-{index}/predictor.{ext}")}));
            }
            Err(error) => emissions.push(
                json!({"language":lang,"admitted":admitted,"status":"emit_error","error":error}),
            ),
        }
    }
    fixture["key"] = json!(key);
    fixture["capability"] = json!(name);
    fixture["directory"] = json!(format!("calibration-{index}"));
    fixture["golden_rows"] = json!(checks.len());
    fixture["emissions"] = json!(emissions);
    calibrated.push(fixture);
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args().nth(1).expect("report directory");
    let out = Path::new(&directory);
    let registry = Registry::current();
    fs::write(
        out.join("registry.json"),
        serde_json::to_vec_pretty(&registry)?,
    )?;
    let train = rows(160, 7);
    let y: Vec<f64> = train
        .iter()
        .map(|r| 2.0 * r[0] - r[1] * r[2] + (3.0 * r[3]).sin() + 0.5 * r[4] + 2.0)
        .collect();
    let mut golden = rows(57, 99);
    golden.extend([
        vec![0.0; WIDTH],
        vec![-0.0; WIDTH],
        vec![-f64::from_bits(1); WIDTH],
        vec![0.25; WIDTH],
        vec![-0.5; WIDTH],
        vec![1.5; WIDTH],
        vec![1e6; WIDTH],
    ]);
    let mut cases = Vec::new();
    let mut cells = Vec::new();
    for entry in &registry.entries {
        let key = format!("{:?}/{}", entry.kind, entry.name);
        for language in Language::ALL {
            cells.push(json!({"key":key,"language":language,"supported":entry.supported[&language],"status":if entry.supported[&language]{"pending"}else{"unsupported_registry"}}));
        }
        if entry.kind == CapabilityKind::Calibration && CALIBRATION_GOLDENS.contains(&entry.name)
        {
            continue;
        }
        let languages: Vec<_> = [Language::Rust, Language::Python, Language::Javascript]
            .into_iter()
            .filter(|l| entry.supported[l])
            .collect();
        if languages.is_empty() {
            continue;
        }
        assert!(
            matches!(entry.kind, CapabilityKind::Operation | CapabilityKind::Head)
                || (entry.kind == CapabilityKind::Calibration && entry.name == "none")
                || (entry.kind == CapabilityKind::Encoding && entry.name == "dense_numeric"),
            "new supported entry needs a golden: {key}"
        );
        let dsl = probe(entry.kind, entry.name);
        let genome = GenomeIr::parse(&dsl)?;
        let model = fit(&genome, &train, &y)?;
        save_case(
            out, &mut cases, key, &dsl, &genome, &model, &golden, &languages, false,
        )?;
    }
    // Observe real routing: label each actual fitted tree's leaves with node IDs.
    let mut decision_rows = golden.clone();
    decision_rows.extend(
        rows(2000, 2026)
            .into_iter()
            .map(|r| r.into_iter().map(|v| v * 1.5 - 0.25).collect()),
    );
    for dsl in [
        "(pipe X (gbm rounds=4 depth=2 min_leaf=2))",
        "(pipe X (histgbm2 rounds=4 leaves=4 min_leaf=2 bags=1))",
    ] {
        use feat_math::{
            boost2::BoostNode,
            heads::{FittedHead, GbmNode},
        };
        let genome = GenomeIr::parse(dsl)?;
        let model = fit(&genome, &train, &y)?;
        let Model::Pipeline { head, .. } = &model else {
            unreachable!()
        };
        let probes: Vec<_> = match head.as_ref() {
            FittedHead::Gbm(h) => h
                .trees
                .iter()
                .map(|tree| {
                    let mut h = h.clone();
                    let mut tree = tree.clone();
                    for (i, node) in tree.iter_mut().enumerate() {
                        if let GbmNode::Leaf(v) = node {
                            *v = i as f64;
                        }
                    }
                    h.init = 0.0;
                    h.lr = 1.0;
                    h.trees = vec![tree];
                    FittedHead::Gbm(h)
                })
                .collect(),
            FittedHead::HistGbm2(h) => h
                .bags
                .iter()
                .flatten()
                .map(|tree| {
                    let mut h = h.clone();
                    let mut tree = tree.clone();
                    for (i, node) in tree.iter_mut().enumerate() {
                        if let BoostNode::Leaf(v) = node {
                            *v = i as f64;
                        }
                    }
                    h.init = 0.0;
                    h.lr = 1.0;
                    h.bags = vec![vec![tree]];
                    FittedHead::HistGbm2(h)
                })
                .collect(),
            _ => unreachable!(),
        };
        assert!(!probes.is_empty());
        for (i, head) in probes.into_iter().enumerate() {
            let boundaries: Vec<(usize, f64)> = match &head {
                FittedHead::Gbm(h) => h
                    .trees
                    .iter()
                    .flatten()
                    .filter_map(|n| {
                        if let GbmNode::Split {
                            feature, thr_bin, ..
                        } = n
                        {
                            Some((*feature, h.edges[*feature][*thr_bin as usize]))
                        } else {
                            None
                        }
                    })
                    .collect(),
                FittedHead::HistGbm2(h) => h
                    .bags
                    .iter()
                    .flatten()
                    .flatten()
                    .filter_map(|n| {
                        if let BoostNode::Split {
                            feature, threshold, ..
                        } = n
                        {
                            Some((*feature, *threshold))
                        } else {
                            None
                        }
                    })
                    .collect(),
                _ => unreachable!(),
            };
            let margin = boundaries
                .iter()
                .flat_map(|(f, t)| golden.iter().map(move |r| (r[*f] - *t).abs()))
                .fold(f64::INFINITY, f64::min);
            assert!(
                margin.is_finite() && margin > 1e-9,
                "golden lies on a split threshold"
            );
            let model = Model::Pipeline {
                target: None,
                ops: vec![],
                head: Box::new(head),
            };
            save_case(
                out,
                &mut cases,
                format!("routing/{dsl}/{i}"),
                dsl,
                &genome,
                &model,
                &decision_rows,
                &[Language::Rust, Language::Python],
                true,
            )?;
            cases.last_mut().unwrap()["golden_threshold_margin"] = json!(margin);
        }
    }
    // Calibration maps, fitted through the production APIs. The registry row decides only
    // whether a passing golden qualifies a cell; unadmitted rows stay unqualified.
    use jope_runtime::{
        calibration::{self as output, OutputTransform},
        fitness::{Columns, Fitness},
        model_object::calibration_platt,
    };
    let mut calibrated = Vec::new();
    let genome = GenomeIr::parse("(pipe X (ridge))")?;
    let model = fit(&genome, &train, &y)?;
    let raw_train = model.predict(&train)?;
    let raw_golden = model.predict(&golden)?;
    // Wider than the training range, so the learned floors are reached on held-out rows.
    let mut floor_rows = golden.clone();
    floor_rows.extend(
        rows(400, 4242)
            .into_iter()
            .map(|r| r.into_iter().map(|v| v * 4.0 - 1.5).collect()),
    );
    let raw_floor = model.predict(&floor_rows)?;
    // A nonnegative target the raw predictions overshoot: the map is learned and its floor binds.
    let shifted: Vec<f64> = y.iter().map(|v| (1.7 * v - 4.0).max(0.0)).collect();
    let positive: Vec<f64> = y.iter().map(|v| (v - 2.5).max(0.0)).collect();
    let columns = Columns::new();
    let maps: [(&str, OutputTransform); 4] = [
        (
            "Calibration/affine_floor",
            output::fit(
                &Fitness::Msle {},
                output::Calibration::Affine,
                &shifted,
                &raw_train,
                &columns,
            ),
        ),
        // What a model keeps when its raw predictions win under a log-error objective.
        (
            "calibration/affine_floor/floor_only",
            output::floor_only(&Fitness::Msle {}, &shifted).expect("log errors need a floor"),
        ),
        // The shift family keeps a unit slope; deviances need a strictly positive floor.
        (
            "calibration/affine_floor/shift_positive",
            output::fit(
                &Fitness::PoissonDeviance {},
                output::Calibration::Shift,
                &positive,
                &raw_train,
                &columns,
            ),
        ),
        (
            "calibration/affine_floor/affine_unfloored",
            output::fit(
                &Fitness::Mae {},
                output::Calibration::Affine,
                &shifted,
                &raw_train,
                &columns,
            ),
        ),
    ];
    for (key, map) in maps {
        map.validate()?;
        assert!(!map.is_identity(), "identity calibration golden: {key}");
        let expected = map.apply_all(&raw_floor);
        assert!(
            expected.iter().all(|v| v.is_finite()),
            "non-finite calibrated golden: {key}"
        );
        let binding: Vec<usize> = raw_floor
            .iter()
            .enumerate()
            .filter(|(_, p)| map.lower.is_some_and(|lower| map.intercept + map.slope * **p < lower))
            .map(|(i, _)| i)
            .collect();
        if key == "Calibration/affine_floor" {
            assert!(
                map.slope != 1.0 && map.intercept != 0.0,
                "primary affine golden must move both coefficients"
            );
        }
        if map.lower.is_some() {
            assert!(
                !binding.is_empty() && binding.len() < floor_rows.len(),
                "floor must bind on some golden rows but not all: {key}"
            );
        }
        let checks: Vec<_> = golden[..64]
            .iter()
            .cloned()
            .zip(expected.iter().copied())
            .collect();
        save_calibrated(
            out,
            &mut calibrated,
            key,
            &registry,
            "affine_floor",
            &genome,
            &model,
            &fitted_js::ExportCalibration {
                output: Some(map),
                platt: None,
            },
            &checks,
            json!({"kind":"affine_floor","rows":floor_rows,"expected":expected,"decision":false,"transform":map,"floor_binding_rows":binding}),
        )?;
    }
    // Binary Platt maps, cross-fitted on explicit native positive probabilities. Math-pipeline
    // predictors have no probability head, so the emitted map is checked on scores directly.
    let fit_rows = rows(200, 31);
    let scores: Vec<f64> = fit_rows.iter().map(|r| r[0]).collect();
    let folds: Vec<usize> = (0..scores.len()).map(|i| i % 5).collect();
    let mut grid = vec![
        0.0,
        -0.0,
        f64::from_bits(1),
        1e-300,
        1e-9,
        0.25,
        0.5,
        0.75,
        1.0 - f64::EPSILON / 2.0,
        1.0,
    ];
    grid.extend(rows(54, 77).iter().map(|r| r[0]));
    let refused = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -f64::from_bits(1),
        -1.0,
        1.0 + f64::EPSILON,
        2.0,
    ];
    for (key, order) in [
        ("Calibration/platt", ["no", "yes"]),
        ("calibration/platt/class_order", ["cold \"0\" \\ é", "hot 🔥"]),
    ] {
        let labels: Vec<&str> = fit_rows
            .iter()
            .map(|r| order[usize::from(r[0] + 0.35 * (r[1] - 0.5) > 0.55)])
            .collect();
        let calibration = calibration_platt::cross_fit(&scores, &labels, order, &folds)?.calibration;
        let sigmoid = &calibration.sigmoid;
        assert!(sigmoid.slope > 0.0, "flat Platt golden: {key}");
        let mut test = grid.clone();
        // The score where the map crosses one half, where a class label is decided.
        let half = sigmoid.scale * (sigmoid.center - sigmoid.intercept / sigmoid.slope);
        if (0.0..=1.0).contains(&half) {
            test.push(half);
        }
        let mut expected = Vec::new();
        let mut decided = Vec::new();
        for score in &test {
            let p = calibration.apply(*score)?;
            // The runtime's label rule: the later class only when strictly more probable.
            decided.push(order[usize::from(p > 1.0 - p)]);
            expected.extend([1.0 - p, p]);
        }
        for score in refused {
            assert!(calibration.apply(score).is_err(), "runtime accepted {score:?}");
        }
        let checks: Vec<_> = golden[..64]
            .iter()
            .cloned()
            .zip(raw_golden.iter().copied())
            .collect();
        save_calibrated(
            out,
            &mut calibrated,
            key,
            &registry,
            "platt",
            &genome,
            &model,
            &fitted_js::ExportCalibration {
                output: None,
                platt: Some(calibration.clone()),
            },
            &checks,
            json!({"kind":"platt","scores":test,"score_bits":test.iter().map(|v| format!("{:016x}", v.to_bits())).collect::<Vec<_>>(),"expected":expected,"labels":decided,"class_order":order,"refused_bits":refused.iter().map(|v| format!("{:016x}", v.to_bits())).collect::<Vec<_>>(),"sigmoid":sigmoid,"decision":false}),
        )?;
    }
    // Raw numeric encoding is a distinct contract, not categorical support.
    use jope_runtime::math_pipeline::{
        InputContract, MathPipeline, NumericColumn, PipelineState, TargetScale, INPUT_PART,
        PIPELINE_PART,
    };
    let dsl = "(pipe X (robust_scale append=true) (ridge))";
    let genome = GenomeIr::parse(dsl)?;
    let model = fit(&genome, &train, &y)?;
    let columns = (0..WIDTH)
        .map(|i| NumericColumn {
            name: format!("c{i}"),
            min: 0.0,
            max: 1.0,
            constant: i == WIDTH - 1,
        })
        .collect();
    let input = InputContract::new(
        columns,
        (0..WIDTH).rev().collect(),
        TargetScale {
            name: "target".into(),
            min: -2.0,
            scale: 3.0,
        },
    )?;
    let parts = BTreeMap::from([
        (INPUT_PART.to_owned(), serde_json::to_vec(&input)?),
        (
            PIPELINE_PART.to_owned(),
            serde_json::to_vec(&PipelineState {
                dsl: dsl.into(),
                model,
            })?,
        ),
    ]);
    let pipeline = MathPipeline::decode(&parts, PIPELINE_PART)?;
    let mut raw: Vec<Vec<Option<String>>> = golden
        .iter()
        .map(|r| r.iter().map(|v| Some(v.to_string())).collect())
        .collect();
    for value in [
        None,
        Some(""),
        Some("NaN"),
        Some("inf"),
        Some("-inf"),
        Some("unknown-category"),
        Some("true"),
        Some("false"),
    ] {
        raw.push(vec![value.map(str::to_string); WIDTH]);
    }
    let borrowed: Vec<Vec<Option<&str>>> = raw
        .iter()
        .map(|r| r.iter().map(|v| v.as_deref()).collect())
        .collect();
    let expected = pipeline.predict_cells(&borrowed)?;
    fs::write(out.join("raw.rs"), fitted_rs::emit_pipeline(&pipeline)?)?;
    fs::write(
        out.join("raw.py"),
        fitted_py::emit_math_pipeline(&pipeline)?,
    )?;
    fs::write(out.join("raw.mjs"), pipeline.export_javascript()?)?;
    fs::write(
        out.join("fixtures.json"),
        serde_json::to_vec(
            &json!({"cells":cells,"cases":cases,"calibration":calibrated,"raw":{"rows":raw,"expected":expected,"columns":pipeline.input.columns.iter().map(|c|&c.name).collect::<Vec<_>>()}}),
        )?,
    )?;
    Ok(())
}
RUST
nice -n 10 python3 -I -B - "$report" <<'PY'
import datetime, hashlib, json, math, os, pathlib, platform, struct, subprocess, sys, time, traceback

root = pathlib.Path.cwd()
out = pathlib.Path(sys.argv[1])
started = time.monotonic()
report = {
    "schema": "jope.export-parity/v1",
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "running",
    "passed": False,
    "cases": [],
    "cells": [],
    "toolchains": {},
    "fixture_kind": "deterministic numerical contract fixtures, not quality/training qualification",
    "host": platform.platform(),
    "environment": {
        "CARGO_BUILD_JOBS": os.environ["CARGO_BUILD_JOBS"],
        "CARGO_INCREMENTAL": os.environ["CARGO_INCREMENTAL"],
        "CARGO_PROFILE_DEV_DEBUG": os.environ["CARGO_PROFILE_DEV_DEBUG"],
        "RAYON_NUM_THREADS": os.environ["RAYON_NUM_THREADS"],
        "niceness": os.getpriority(os.PRIO_PROCESS, 0),
    },
    "ci_executed": os.environ.get("GITHUB_ACTIONS") == "true",
    "ci_context": {
        key: os.environ.get(key)
        for key in [
            "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT",
            "GITHUB_WORKFLOW_REF",
            "GITHUB_REPOSITORY",
            "GITHUB_EVENT_NAME",
        ]
    },
    "scope": "Advertised primary-language registry cells. Unsupported calibration, categorical and class-label contracts remain unqualified; calibration goldens for unadmitted rows are evidence, not qualification.",
}


def digest(p):
    with pathlib.Path(p).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(args, name, timeout=900):
    begin = time.monotonic()
    with (
        (out / (name + ".stdout")).open("w") as stdout,
        (out / (name + ".stderr")).open("w") as stderr,
    ):
        result = subprocess.run(
            args, stdout=stdout, stderr=stderr, text=True, timeout=timeout
        )
    report.setdefault("commands", []).append(
        {
            "argv": [str(x) for x in args],
            "exit": result.returncode,
            "seconds": time.monotonic() - begin,
            "stdout": name + ".stdout",
            "stderr": name + ".stderr",
        }
    )
    if result.returncode:
        raise RuntimeError(f"{name} exited {result.returncode}; see {name}.stderr")
    return (out / (name + ".stdout")).read_text()


def compare(case, language, actual):
    expected = case["expected"]
    assert len(actual) == len(expected), f"{case['key']}: shape mismatch"
    absolute, relative = (1e-9, 1e-7) if language == "javascript" else (1e-12, 1e-10)
    differences = []
    for i, (a, b) in enumerate(zip(actual, expected)):
        assert not (
            isinstance(a, bool)
            or not isinstance(a, (int, float))
            or not math.isfinite(a)
            or not math.isfinite(b)
        ), f"{case['key']}/{language}: invalid numeric output at row {i}"
        if abs(a - b) > absolute + relative * abs(b):
            differences.append(i)
    if case.get("decision"):
        flips = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
        golden = sum(i < 64 for i in flips)
        random = len(flips) - golden
        assert not golden and random * 1000 <= len(expected) - 64, (
            f"{case['key']}: decision flips golden={golden} randomized={random}"
        )
    else:
        flips = []
        assert not differences, (
            f"{case['key']}/{language}: {len(differences)} numeric mismatches, first row {differences[:1]}"
        )
    return {
        "status": "pass",
        "language": language,
        "rows": len(expected),
        "atol": absolute,
        "rtol": relative,
        "max_abs_error": max(abs(a - b) for a, b in zip(actual, expected)),
        "decision_flip_rows": flips,
        "golden_flip_rate": sum(i < 64 for i in flips) / 64
        if case.get("decision")
        else None,
        "randomized_flip_rate": sum(i >= 64 for i in flips) / (len(expected) - 64)
        if case.get("decision")
        else None,
        "decision_measurement": "actual fitted leaf node ID"
        if case.get("decision")
        else None,
        "golden_threshold_margin": case.get("golden_threshold_margin"),
        "numeric_mismatch_rows": differences,
        "self_check_rows": 64,
    }


try:
    report["source_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    report["source_dirty"] = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    )
    report["harness_sha256"] = digest(out / "harness.sh")
    report["cargo_lock_sha256"] = digest(root / "Cargo.lock")
    for name, args, pin in [
        ("rustc", ["rustc", "--version"], "rustc 1.97.1"),
        ("python", ["python3", "--version"], "Python 3.12.3"),
        ("node", ["node", "--version"], "v26.1.0"),
    ]:
        version = command(args, "version-" + name).strip()
        assert version == pin or (name == "rustc" and version.startswith(pin + " ")), (
            f"{name}: expected {pin}, got {version}"
        )
        report["toolchains"][name] = version
    assert sys.version_info[:3] == (3, 12, 3), (
        "driver interpreter differs from pinned Python"
    )
    report["load_ceiling"] = min(85, 0.85 * len(os.sched_getaffinity(0)))
    while os.getloadavg()[0] > report["load_ceiling"]:
        time.sleep(10)
    build = command(
        [
            "cargo",
            "build",
            "--locked",
            "-p",
            "jope-runtime",
            "--lib",
            "--message-format=json",
        ],
        "cargo-build",
    )
    libraries = {}
    for line in build.splitlines():
        message = json.loads(line)
        if message.get("reason") == "compiler-artifact":
            for filename in message["filenames"]:
                if filename.endswith(".rlib"):
                    libraries[message["target"]["name"]] = filename
    library_dir = pathlib.Path(libraries["jope_runtime"]).parent
    dependency_dir = library_dir if library_dir.name == "deps" else library_dir / "deps"
    args = [
        "rustc",
        "--edition=2021",
        "-C",
        "debuginfo=line-tables-only",
        "-C",
        "codegen-units=" + os.environ["CARGO_BUILD_JOBS"],
        "-L",
        "dependency=" + str(dependency_dir),
    ]
    for name in ["jope_runtime", "feat_math", "serde_json"]:
        args += ["--extern", name + "=" + libraries[name]]
    command(
        args + [str(out / "generate.rs"), "-o", str(out / "generate")],
        "compile-generator",
    )
    command([str(out / "generate"), str(out)], "generate-fixtures")
    fixtures = json.loads((out / "fixtures.json").read_text())
    report["cells"] = fixtures["cells"]
    report["registry_sha256"] = digest(out / "registry.json")
    report["fixture_sha256"] = digest(out / "fixtures.json")
    cases = fixtures["cases"]
    assert cases, "zero collected golden models"
    expected_cells = sum(c["supported"] for c in report["cells"])
    report["expected_supported_cells"] = expected_cells
    errors = [
        (c["key"], emission)
        for c in cases
        for emission in c["emissions"]
        if emission["status"] != "generated"
    ]
    assert not errors, f"advertised support lacks fitted export: {errors}"
    fault = os.environ["JOPE_EXPORT_PARITY_FAULT"]
    report["fault_injected"] = fault != "0"
    report["fault_scope"] = {
        "0": None,
        "1": "primary",
        "2": "routing",
        "3": "calibration",
    }[fault]
    if fault == "3":
        victim = next(
            e
            for c in fixtures["calibration"]
            if c["key"] == "Calibration/platt"
            for e in c["emissions"]
            if e["language"] == "python"
        )
        p = out / victim["file"]
        text = p.read_text()
        old = "    return [1 - p, p]\n"
        assert text.count(old) == 1
        report["fault_original_sha256"] = digest(p)
        p.write_text(text.replace(old, "    return [1 - p, p * (1 + 1e-6)]\n"))
        report["fault_file"] = victim["file"]
        report["fault_mutated_sha256"] = digest(p)
    elif fault != "0":
        victim = next(
            e
            for c in cases
            if fault != "2" or c["key"].startswith("routing/")
            for e in c["emissions"]
            if e["language"] == "python"
        )
        p = out / victim["file"]
        text = p.read_text()
        old = "return fitted_model().predict(rows)"
        assert text.count(old) == 1
        report["fault_original_sha256"] = digest(p)
        p.write_text(
            text.replace(
                old, "return [value + 1.0 for value in fitted_model().predict(rows)]"
            )
        )
        report["fault_file"] = victim["file"]
        report["fault_mutated_sha256"] = digest(p)
    # Execute Python and JavaScript separately from the native reference generator.
    for language in ["python", "javascript"]:
        for i, case in enumerate(cases):
            emission = next(
                (e for e in case["emissions"] if e["language"] == language), None
            )
            if not emission:
                continue
            file = out / emission["file"]
            stem = f"{language}-{i}"
            try:
                command(
                    (["python3", "-I", "-B"] if language == "python" else ["node"])
                    + [str(file), "--self-check"],
                    stem + "-self-check",
                )
                if language == "python":
                    driver = file.parent / "run.py"
                    driver.write_text(
                        'import runpy,json,sys\nm=runpy.run_path(sys.argv[1]); rows=json.load(open(sys.argv[2]))["cases"][int(sys.argv[3])]["rows"];print(json.dumps(m["fitted_predict"](rows),allow_nan=False))\n'
                    )
                    values = command(
                        [
                            "python3",
                            "-I",
                            "-B",
                            str(driver),
                            str(file),
                            str(out / "fixtures.json"),
                            str(i),
                        ],
                        stem,
                    )
                else:
                    driver = file.parent / "run.mjs"
                    driver.write_text(
                        file.read_text()
                        + "\nconst rows="
                        + json.dumps(case["rows"])
                        + ';const h=modelFittedHandle(),out=new Float64Array(rows.length),flat=Float64Array.from(rows.flat());if(modelPredictNumericBatch(h,flat,flat.length,rows.length,12,null,0,out,out.length)!==0)throw Error("status");console.log(JSON.stringify(Array.from(out)));\n'
                    )
                    values = command(["node", str(driver)], stem)
                actual = json.loads(values)
                record = compare(case, language, actual)
                record.update(key=case["key"], emitted_sha256=digest(file))
                report["cases"].append(record)
                (file.parent / (language + "-rows.json")).write_text(
                    json.dumps(
                        {
                            "expected": case["expected"],
                            "actual": actual,
                            "decision": case["decision"],
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            except Exception as error:
                report["cases"].append(
                    {
                        "key": case["key"],
                        "language": language,
                        "status": "fail",
                        "error": str(error),
                        "emitted_sha256": digest(file),
                    }
                )
    # Batch real std-only Rust modules; every module includes a populated self_check test.
    rust_cases = [
        (i, c)
        for i, c in enumerate(cases)
        if any(e["language"] == "rust" for e in c["emissions"])
    ]
    program = out / "rust-predictors.rs"
    chunks = []
    calls = []
    for i, c in rust_cases:
        source = (out / c["directory"] / "predictor.rs").read_text()
        (out / c["directory"] / "rows.tsv").write_text(
            "".join(" ".join(repr(v) for v in row) + "\n" for row in c["rows"])
        )
        chunks.append(f"mod c{i} {{\n{source}\n}}\n")
        calls.append(
            f'let rows=read_rows(&format!("{{root}}/{c["directory"]}/rows.tsv"));for value in c{i}::predict_numeric(&rows).unwrap() {{println!("{{:.17e}}",value);}}'
        )
    loader = "fn read_rows(path:&str)->Vec<Vec<f64>>{std::fs::read_to_string(path).unwrap().lines().map(|line|line.split_whitespace().map(|v|v.parse::<f64>().unwrap()).collect()).collect()}\n"
    program.write_text(
        "".join(chunks)
        + "\n"
        + loader
        + 'fn main(){let root=std::env::args().nth(1).expect("fixture directory");'
        + "".join(calls)
        + "}\n"
    )
    command(
        [
            "rustc",
            "--edition=2021",
            "-C",
            "codegen-units=" + os.environ["CARGO_BUILD_JOBS"],
            str(program),
            "-o",
            str(out / "rust-predictors"),
        ],
        "compile-predictors",
    )
    actual = [
        float(v)
        for v in command(
            [str(out / "rust-predictors"), str(out)], "rust-predictors"
        ).splitlines()
    ]
    offset = 0
    for i, c in rust_cases:
        n = len(c["expected"])
        values = actual[offset : offset + n]
        offset += n
        record = compare(c, "rust", values)
        record.update(
            key=c["key"], emitted_sha256=digest(out / c["directory"] / "predictor.rs")
        )
        report["cases"].append(record)
        (out / c["directory"] / "rust-rows.json").write_text(
            json.dumps(
                {
                    "expected": c["expected"],
                    "actual": values,
                    "decision": c["decision"],
                },
                allow_nan=False,
            )
            + "\n"
        )
    assert offset == len(actual), "unexpected Rust output rows"
    command(
        [
            "rustc",
            "--edition=2021",
            "--test",
            "-C",
            "codegen-units=" + os.environ["CARGO_BUILD_JOBS"],
            str(program),
            "-o",
            str(out / "rust-self-checks"),
        ],
        "compile-self-checks",
    )
    self_check = command(
        [str(out / "rust-self-checks"), "--test-threads=1"], "rust-self-checks"
    )
    assert f"{len(rust_cases)} passed; 0 failed" in self_check, (
        "zero/missing Rust self-checks"
    )
    # Raw-cell encoding, non-finite/missing tokens, constant columns and explicit column order.
    raw = fixtures["raw"]
    raw_case = {"key": "raw_numeric_contract", "expected": raw["expected"]}
    raw_results = []
    for language in ["python", "javascript", "rust"]:
        if language == "python":
            p = out / "raw-run.py"
            p.write_text(
                'import runpy,json,sys\nm=runpy.run_path(sys.argv[1]);f=json.load(open(sys.argv[2]))["raw"];assert list(m["MODEL_COLUMNS"])==f["columns"];print(json.dumps(m["predict_rows"](f["rows"]),allow_nan=False))\n'
            )
            text = command(
                [
                    "python3",
                    "-I",
                    "-B",
                    str(p),
                    str(out / "raw.py"),
                    str(out / "fixtures.json"),
                ],
                "raw-python",
            )
        elif language == "javascript":
            p = out / "raw-run.mjs"
            p.write_text(
                (out / "raw.mjs").read_text()
                + "\nif(JSON.stringify(MODEL_COLUMNS)!=="
                + json.dumps(json.dumps(raw["columns"], separators=(",", ":")))
                + ')throw Error("column order");console.log(JSON.stringify(predictRows('
                + json.dumps(raw["rows"])
                + ")));\n"
            )
            text = command(["node", str(p)], "raw-javascript")
        else:
            rows = (
                "vec!["
                + ",".join(
                    "vec!["
                    + ",".join(
                        "None" if v is None else "Some(" + json.dumps(v) + ")"
                        for v in row
                    )
                    + "]"
                    for row in raw["rows"]
                )
                + "]"
            )
            columns = "&[" + ",".join(json.dumps(v) for v in raw["columns"]) + "]"
            p = out / "raw-run.rs"
            p.write_text(
                (out / "raw.rs").read_text()
                + f'\nfn main(){{let rows={rows};let headers={columns};let actual=predict_cells(&rows).unwrap();assert_eq!(actual,predict_table(headers,&rows).unwrap());let reversed_headers:Vec<_>=headers.iter().rev().copied().collect();let reversed_rows:Vec<Vec<_>>=rows.iter().map(|r|r.iter().rev().copied().collect()).collect();assert_eq!(actual,predict_table(&reversed_headers,&reversed_rows).unwrap());for value in actual{{println!("{{:.17e}}",value);}}}}\n'
            )
            command(
                [
                    "rustc",
                    "--edition=2021",
                    "-C",
                    "codegen-units=" + os.environ["CARGO_BUILD_JOBS"],
                    str(p),
                    "-o",
                    str(out / "raw-run"),
                ],
                "compile-raw",
            )
            text = command([str(out / "raw-run")], "raw-rust")
            text = json.dumps([float(v) for v in text.splitlines()])
        record = compare(raw_case, language, json.loads(text))
        record["self_check_rows"] = 0
        raw_results.append(record)
    report["raw_numeric_contract"] = raw_results
    # Calibration goldens. Each emitted module runs on its own, so one defect cannot hide
    # another, with the same pinned toolchains and tolerances as the advertised cells.
    strict_calibration = os.environ["JOPE_EXPORT_PARITY_STRICT_CALIBRATION"] == "1"
    rust_helpers = (
        "fn read_rows(path:&str)->Vec<Vec<f64>>{std::fs::read_to_string(path).unwrap().lines().map(|line|line.split_whitespace().map(|v|v.parse::<f64>().unwrap()).collect()).collect()}\n"
        "fn read_bits(path:&str)->Vec<f64>{std::fs::read_to_string(path).unwrap().lines().map(|v|f64::from_bits(u64::from_str_radix(v,16).unwrap())).collect()}\n"
        'fn hex(s:&str)->String{s.bytes().map(|b|format!("{b:02x}")).collect()}\n'
    )
    python_driver = """import json, runpy, struct, sys
m = runpy.run_path(sys.argv[1])
c = json.load(open(sys.argv[2]))["calibration"][int(sys.argv[3])]
bits = lambda b: struct.unpack(">d", bytes.fromhex(b))[0]
if c["kind"] == "platt":
    refused = []
    for b in c["refused_bits"]:
        try:
            m["fitted_platt"](bits(b))
            refused.append(False)
        except ValueError:
            refused.append(True)
    result = {"values": [v for b in c["score_bits"] for v in m["fitted_platt"](bits(b))], "class_order": list(m["PLATT_CLASS_ORDER"]), "refused": refused}
else:
    result = {"values": m["fitted_predict"](c["rows"])}
print(json.dumps(result, allow_nan=False))
"""
    javascript_driver = """function goldenBits(b){const v=new DataView(new ArrayBuffer(8));v.setBigUint64(0,BigInt("0x"+b));return v.getFloat64(0);}
let goldenResult;
if(GOLDEN_CASE.kind==="platt"){
  const values=[];for(const b of GOLDEN_CASE.score_bits){const p=fittedPlatt(goldenBits(b));values.push(p[0],p[1]);}
  const refused=GOLDEN_CASE.refused_bits.map(b=>{try{fittedPlatt(goldenBits(b));return false;}catch(_error){return true;}});
  goldenResult={values,class_order:Array.from(PLATT_CLASS_ORDER),refused};
}else{
  const rows=GOLDEN_CASE.rows,h=modelFittedHandle(),out=new Float64Array(rows.length),flat=Float64Array.from(rows.flat());
  if(modelPredictNumericBatch(h,flat,flat.length,rows.length,rows[0].length,null,0,out,out.length)!==0)throw Error("status");
  goldenResult={values:Array.from(out)};
}
console.log(JSON.stringify(goldenResult));
"""

    def run_calibrated(index, case, language, file):
        directory = file.parent
        stem = f"calibration-{index}-{language}"
        if language == "python":
            command(["python3", "-I", "-B", str(file), "--self-check"], stem + "-self-check")
            driver = directory / "run.py"
            driver.write_text(python_driver)
            argv = [str(driver), str(file), str(out / "fixtures.json"), str(index)]
            return json.loads(command(["python3", "-I", "-B", *argv], stem))
        if language == "javascript":
            command(["node", str(file), "--self-check"], stem + "-self-check")
            driver = directory / "run.mjs"
            driver.write_text(
                file.read_text()
                + "\nconst GOLDEN_CASE="
                + json.dumps(case)
                + ";\n"
                + javascript_driver
            )
            return json.loads(command(["node", str(driver)], stem))
        if case["kind"] == "platt":
            for name in ["score_bits", "refused_bits"]:
                (directory / (name + ".hex")).write_text("".join(b + "\n" for b in case[name]))
            body = (
                "for name in m::PLATT_CLASS_ORDER{println!(\"class {}\",hex(name));}"
                'for score in read_bits(&format!("{root}/score_bits.hex")){for value in m::fitted_platt(score).unwrap(){println!("value {:016x}",value.to_bits());}}'
                'for score in read_bits(&format!("{root}/refused_bits.hex")){println!("refused {}",m::fitted_platt(score).is_err());}'
            )
        else:
            (directory / "rows.tsv").write_text(
                "".join(" ".join(repr(v) for v in row) + "\n" for row in case["rows"])
            )
            body = 'for value in m::predict_numeric(&read_rows(&format!("{root}/rows.tsv"))).unwrap(){println!("value {:016x}",value.to_bits());}'
        program = directory / "run.rs"
        program.write_text(
            "#![allow(dead_code)]\nmod m {\n"
            + file.read_text()
            + "\n}\n"
            + rust_helpers
            + 'fn main(){let root=std::env::args().nth(1).expect("case directory");'
            + body
            + "}\n"
        )
        for flags, binary in [([], "run"), (["--test"], "self-check")]:
            command(
                ["rustc", "--edition=2021", *flags, "-C"]
                + ["codegen-units=" + os.environ["CARGO_BUILD_JOBS"], str(program)]
                + ["-o", str(directory / ("rust-" + binary))],
                stem + "-compile-" + binary,
            )
        checked = command(
            [str(directory / "rust-self-check"), "--test-threads=1"], stem + "-self-check"
        )
        assert "1 passed; 0 failed" in checked, "zero/missing Rust self-check"
        lines = command([str(directory / "rust-run"), str(directory)], stem).splitlines()
        field = lambda tag: [line.split()[1] for line in lines if line.startswith(tag + " ")]
        result = {"values": [struct.unpack(">d", bytes.fromhex(v))[0] for v in field("value")]}
        if case["kind"] == "platt":
            result["class_order"] = [bytes.fromhex(v).decode() for v in field("class")]
            result["refused"] = [v == "true" for v in field("refused")]
        return result

    def check_calibrated(case, language, result):
        record = compare(
            {"key": case["key"], "expected": case["expected"], "decision": False},
            language,
            result["values"],
        )
        record["self_check_rows"] = case["golden_rows"]
        if case["kind"] == "platt":
            assert result["class_order"] == case["class_order"], (
                f"{case['key']}/{language}: class order differs"
            )
            assert len(result["refused"]) == len(case["refused_bits"]) and all(
                result["refused"]
            ), f"{case['key']}/{language}: accepted an invalid score"
            values = result["values"]
            # The runtime's label rule: the later class only when strictly more probable.
            labels = [
                case["class_order"][int(values[2 * i + 1] > values[2 * i])]
                for i in range(len(values) // 2)
            ]
            flips = [i for i, (a, b) in enumerate(zip(labels, case["labels"])) if a != b]
            assert not flips, f"{case['key']}/{language}: class label flips at scores {flips}"
            record.update(
                scores=len(labels), class_label_flips=0, refused_scores=len(result["refused"])
            )
        else:
            lower = case["transform"].get("lower")
            loose = [i for i in case["floor_binding_rows"] if result["values"][i] != lower]
            assert not loose, f"{case['key']}/{language}: floor not exact at rows {loose}"
            record.update(
                floor_binding_rows=len(case["floor_binding_rows"]), transform=case["transform"]
            )
        return record

    calibration_records = []
    for index, case in enumerate(fixtures["calibration"]):
        for emission in case["emissions"]:
            record = {
                "key": case["key"],
                "capability": case["capability"],
                "language": emission["language"],
                "status": "fail",
            }
            try:
                assert emission["status"] == "generated", "emit error: " + emission["error"]
                file = out / emission["file"]
                record["emitted_sha256"] = digest(file)
                result = run_calibrated(index, case, emission["language"], file)
                record.update(check_calibrated(case, emission["language"], result))
            except Exception as error:
                record["error"] = str(error)
            record["admitted"] = emission["admitted"]
            calibration_records.append(record)
    assert {c["capability"] for c in fixtures["calibration"]} == {"affine_floor", "platt"}
    assert all(len(c["emissions"]) == 3 for c in fixtures["calibration"])
    report["calibration"] = {
        "scope": "Calibrated golden predictors for affine_floor and platt, fitted with the production calibration APIs. A row the registry does not admit stays unqualified even when its golden passes.",
        "strict": strict_calibration,
        "expected_modules": 3 * len(fixtures["calibration"]),
        "executed_modules": len(calibration_records),
        "passed_modules": sum(r["status"] == "pass" for r in calibration_records),
        "failed": [
            (r["key"], r["language"]) for r in calibration_records if r["status"] != "pass"
        ],
        "cases": calibration_records,
    }
    blocking = [
        (r["key"], r["language"])
        for r in calibration_records
        if r["status"] != "pass" and (strict_calibration or r["admitted"])
    ]
    for cell in report["cells"]:
        if cell["supported"]:
            matches = [
                c
                for c in report["cases"] + calibration_records
                if c["key"] == cell["key"] and c["language"] == cell["language"]
            ]
            assert len(matches) == 1, cell
            cell["status"] = matches[0]["status"]
    executed = sum(c["status"] == "pass" for c in report["cells"])
    report["executed_supported_cells"] = executed
    assert expected_cells > 0 and executed == expected_cells, (
        f"{executed}/{expected_cells} supported cells passed; see failed case records"
    )
    expected_emissions = [
        (case["key"], emission["language"])
        for case in cases
        for emission in case["emissions"]
    ]
    observed_emissions = [(case["key"], case["language"]) for case in report["cases"]]
    report["expected_emitted_modules"] = len(expected_emissions)
    report["executed_emitted_modules"] = len(observed_emissions)
    assert any(case["key"].startswith("routing/") for case in cases), "no routing probes"
    assert sorted(observed_emissions) == sorted(expected_emissions), (
        "missing or duplicate emitted predictor execution"
    )
    failed = [
        (case["key"], case["language"])
        for case in report["cases"]
        if case["status"] != "pass"
    ]
    assert not failed, f"emitted predictors failed: {failed}"
    assert not blocking, f"calibration goldens failed: {blocking}"
    report.update(
        status="pass",
        passed=True,
        executed_supported_cells=executed,
        unsupported_cells=sum(not c["supported"] for c in report["cells"]),
    )
    identity = {
        key: report[key]
        for key in [
            "source_sha",
            "harness_sha256",
            "registry_sha256",
            "fixture_sha256",
            "toolchains",
        ]
    }
    report["qualification_id"] = (
        "sha256:"
        + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    )
except Exception as error:
    report.update(status="fail", passed=False, error=str(error))
    (out / "failure.txt").write_text(traceback.format_exc())
finally:
    report["seconds"] = time.monotonic() - started
    report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["source_end"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    report["source_dirty_end"] = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    )
    report["source_stable"] = (
        report["source_end"] == report.get("source_sha")
        and report["source_dirty_end"] == report.get("source_dirty")
        and digest(root / "deploy/ci/export-parity.sh") == report.get("harness_sha256")
    )
    if not report["source_stable"]:
        report.update(
            status="source_changed",
            passed=False,
            error="source changed during qualification",
        )
    (out / "summary.md").write_text(
        f"Export parity: **{report['status']}**\n\nSource: {report.get('source_sha', 'unavailable')}\n\nSupported cells: {report.get('executed_supported_cells', 0)}/{report.get('expected_supported_cells', 'unavailable')}\n\n{report.get('error', 'All advertised supported cells executed with real pinned toolchains. Unsupported registry cells remain unqualified; hosted E2 is not claimed.')}\n"
        + (
            f"\nCalibration goldens: {report['calibration']['passed_modules']}/{report['calibration']['expected_modules']} emitted modules passed ({'strict' if report['calibration']['strict'] else 'unadmitted rows stay unqualified'}); failed: {report['calibration']['failed']}\n"
            if "calibration" in report
            else ""
        )
    )
    report["files"] = {
        str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "report.json"
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps({k: report[k] for k in ["status", "passed", "seconds"]})
        + " report="
        + str(out / "report.json")
    )
sys.exit(0 if report["passed"] else 1)
PY
