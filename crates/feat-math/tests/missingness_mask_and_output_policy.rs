//! MR15 (G01 / P02 / GP-03 / WP02): Missingness mask and linear output policy.
//!
//! Verifies:
//! 1. `nan_mask_distinct_from_zero`: `f64::NAN` and `0.0` remain distinct in `NumericModelV2`
//!    imputation and prediction (unlike legacy `sanitize` + `impute` which conflates `0.0` and `NaN`).
//! 2. `linear_output_unclamped_at_1e4`: linear targets at `1e4` scale predict unclamped `1e4`
//!    outputs (unlike legacy `feat_math::fit` which silently clamps to `±60.0`), while clamping
//!    only outside the training target envelope (`clamp_lo..=clamp_hi`).
//! 3. `post_expansion_width_admitted_exact`: exact post-expansion widths up to `MAX_WIDTH` (512)
//!    are admitted and width mismatches or `> 512` expansions fail closed before evaluation.
//! 4. `wp15_mask_and_clamp_export_subset_parity`: Rust, Python, and C++17 emitted predictors
//!    propagate missingness and clamp semantics identically across the export subset.

use feat_math::dsl::{ParamValue, Stage};
use feat_math::kernel::linear_scaling::LinearScaledWinner;
use feat_math::kernel::width::{check_input_width, output_width, required_input_width, WidthReq};
use feat_math::math_expr::MathExpr;
use feat_math::numeric_v2::{NumericModelV2, NumericProfileV2};
use feat_math::numeric_v2_emit::{self, OptionsV2, TargetV2};
use feat_math::ops::{apply_op, fit_op};
use feat_math::{GenomeIr, Rows, MAX_WIDTH};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn profile(n: usize) -> NumericProfileV2 {
    NumericProfileV2::new((0..n).map(|i| format!("c{i}")).collect())
}

fn genome(s: &str) -> GenomeIr {
    GenomeIr::parse(s).expect("valid genome")
}

#[test]
fn nan_mask_distinct_from_zero() {
    // Legacy sanitize + impute conflates 0.0 and NaN into identical imputed cells.
    let mut legacy_rows = vec![vec![0.0], vec![f64::NAN], vec![6.0]];
    feat_math::sanitize(&mut legacy_rows);
    let legacy_stage = Stage::new("impute");
    let legacy_fitted = fit_op(&legacy_stage, &legacy_rows, &[]).unwrap();
    let legacy_applied = apply_op(&legacy_fitted, &legacy_rows).unwrap();
    assert_eq!(legacy_applied[0], legacy_applied[1]);

    // NumericModelV2 preserves 0.0 as a finite observation and imputes only non-finite cells.
    let train_x = vec![
        vec![0.0, f64::NAN],
        vec![6.0, f64::NAN],
        vec![12.0, f64::NAN],
        vec![f64::NAN, f64::NAN],
    ];
    let train_y = [0.0, 600.0, 1200.0, 600.0];
    let model = NumericModelV2::fit(
        profile(2),
        &genome("(pipe X (impute) (ridge))"),
        &train_x,
        &train_y,
    )
    .unwrap();
    // Mean of finite values in col 0 is (0 + 6 + 12) / 3 = 6.0; all-NaN col 1 falls back to 0.0.
    assert_eq!(model.fill, vec![6.0, 0.0]);

    let probe_rows = vec![vec![0.0, 0.0], vec![f64::NAN, 0.0], vec![12.0, 0.0]];
    let preds = model.predict(&model.profile.schema, &probe_rows).unwrap();
    assert!(
        (preds[1] - preds[0]).abs() > 200.0,
        "NaN row must use imputed mean (6.0) rather than 0.0: {:?}",
        preds
    );
    assert!(preds[0] < preds[1] && preds[1] < preds[2]);
}

#[test]
fn linear_output_unclamped_at_1e4() {
    let x: Rows = (0..21).map(|i| vec![(i as f64) - 10.0]).collect();
    let y: Vec<f64> = x.iter().map(|r| 1_000.0 * r[0]).collect(); // spans [-1e4, +1e4]

    // Legacy fit silently clamps linear predictions to [-60.0, +60.0].
    let legacy_preds = feat_math::fit(&genome("(pipe X (ridge))"), &x, &y)
        .unwrap()
        .predict(&x)
        .unwrap();
    assert_eq!(legacy_preds[0], -60.0);
    assert_eq!(legacy_preds[20], 60.0);

    // NumericModelV2 preserves the 1e4 linear scale without the +-60 clamp.
    let model = NumericModelV2::fit(profile(1), &genome("(pipe X (ridge))"), &x, &y).unwrap();
    let preds = model.predict(&model.profile.schema, &x).unwrap();
    assert!(
        preds[0] < -9_500.0 && preds[20] > 9_500.0,
        "expected unclamped 1e4 scale predictions, got {:?}",
        (preds[0], preds[20])
    );

    // Extreme out-of-envelope input clamps to the training target envelope +-5% margin.
    let extreme = model
        .predict(&model.profile.schema, &vec![vec![-1e9], vec![1e9]])
        .unwrap();
    assert_eq!(extreme[0], model.clamp_lo);
    assert_eq!(extreme[1], model.clamp_hi);
    assert!((model.clamp_lo - (-11_000.0)).abs() < 1e-9);
    assert!((model.clamp_hi - 11_000.0).abs() < 1e-9);

    // Scalar linear scaling also preserves exact 1e4 outputs without clamping.
    let expr = MathExpr::parse("c0").unwrap();
    let fit = LinearScaledWinner::fit_training(&expr, &x, &y).unwrap();
    let scaled_preds = fit.winner.predict(&[vec![-10.0], vec![10.0]]).unwrap();
    assert!((scaled_preds[0] - (-10_000.0)).abs() < 1e-9);
    assert!((scaled_preds[1] - 10_000.0).abs() < 1e-9);
}

#[test]
fn post_expansion_width_admitted_exact() {
    // 1. Exact post-expansion width up to MAX_WIDTH (512) is admitted.
    let x_256 = vec![vec![1.0; 256], vec![2.0; 256], vec![3.0; 256]];
    let y_256 = [10.0, 20.0, 30.0];
    let exact_512 = NumericModelV2::fit(
        profile(256),
        &genome("(pipe X (poly_interact topk=256 append=true) (ridge))"),
        &x_256,
        &y_256,
    )
    .expect("256 + 256 == 512 must be admitted");
    assert_eq!(exact_512.ridge.means.len(), MAX_WIDTH);
    assert_eq!(exact_512.ridge.w.len(), MAX_WIDTH + 1);
    assert_eq!(
        exact_512
            .predict(&exact_512.profile.schema, &x_256)
            .unwrap()
            .len(),
        3
    );

    // 2. Post-expansion width exceeding MAX_WIDTH (256 + 257 = 513 > 512) is refused.
    assert!(NumericModelV2::fit(
        profile(256),
        &genome("(pipe X (poly_interact topk=257 append=true) (ridge))"),
        &x_256,
        &y_256,
    )
    .is_err());

    // 3. Kernel fitted-width contract checks exact input width and refuses mismatched widths.
    let mut stage = Stage::new("standardize");
    stage.params.insert("append".into(), ParamValue::Bool(true));
    let fitted = fit_op(&stage, &vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &[]).unwrap();
    assert_eq!(required_input_width(&fitted), WidthReq::Exactly(3));
    assert_eq!(output_width(&fitted, 3).unwrap(), 6);
    assert!(output_width(&fitted, 2).is_err());
    assert!(output_width(&fitted, 4).is_err());

    let legacy_model = feat_math::fit(
        &genome("(pipe X (standardize) (ridge))"),
        &vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        &[1.0, 2.0, 3.0],
    )
    .unwrap();
    assert!(check_input_width(&legacy_model, 2).is_ok());
    assert!(check_input_width(&legacy_model, 1).is_err());
    assert!(check_input_width(&legacy_model, 3).is_err());
    assert!(legacy_model.try_predict(&vec![vec![1.0, 2.0]]).is_ok());
    assert!(legacy_model.try_predict(&vec![vec![1.0]]).is_err());
    assert!(legacy_model
        .try_predict(&vec![vec![1.0, 2.0, 3.0]])
        .is_err());
}

#[test]
fn wp15_mask_and_clamp_export_subset_parity() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(&root).unwrap();

    // Part A: NumericModelV2 Rust + Python emitted predictor parity on NaN mask + 1e4 clamp range.
    let train_x = vec![
        vec![0.0, 1.0],
        vec![2.0, 2.0],
        vec![4.0, 3.0],
        vec![6.0, 4.0],
        vec![8.0, 5.0],
    ];
    let train_y = [-10_000.0, -5_000.0, 0.0, 5_000.0, 10_000.0];
    let model = NumericModelV2::fit(
        profile(2),
        &genome("(pipe X (quantile_bins bins=4) (ridge))"),
        &train_x,
        &train_y,
    )
    .unwrap();
    let probes = vec![
        vec![0.0, 1.0],
        vec![f64::NAN, 3.0],
        vec![8.0, 5.0],
        vec![1e9, 1e9],
    ];
    let want = model.predict(&model.profile.schema, &probes).unwrap();
    assert!(want[0] < -5_000.0 && want[2] > 5_000.0);
    assert!((want[1] - want[0]).abs() > 100.0);

    let py_art = numeric_v2_emit::emit(&model, OptionsV2::new(TargetV2::Python)).unwrap();
    let py_src = dir.path().join("v2_model.py");
    fs::write(&py_src, &py_art.source).unwrap();
    let py_out = Command::new("python3")
        .arg("-c")
        .arg("import runpy,sys; m=runpy.run_path(sys.argv[1]); print(*[f'{v:.17g}' for v in m['predict']([[0.0,1.0],[float('nan'),3.0],[8.0,5.0],[1e9,1e9]])], sep='\\n')")
        .arg(&py_src)
        .output()
        .unwrap();
    assert!(
        py_out.status.success(),
        "{}",
        String::from_utf8_lossy(&py_out.stderr)
    );
    let py_vals: Vec<f64> = String::from_utf8(py_out.stdout)
        .unwrap()
        .lines()
        .map(|s| s.parse().unwrap())
        .collect();

    let rs_art = numeric_v2_emit::emit(&model, OptionsV2::new(TargetV2::Rust)).unwrap();
    let rs_src = dir.path().join("v2_model.rs");
    let rs_drv = dir.path().join("v2_driver.rs");
    let rs_bin = dir.path().join("v2_driver");
    fs::write(&rs_src, &rs_art.source).unwrap();
    fs::write(
        &rs_drv,
        format!(
            "include!({rs_src:?}); fn main() {{ for v in predict(&[vec![0.0,1.0],vec![f64::NAN,3.0],vec![8.0,5.0],vec![1e9,1e9]]).unwrap() {{ println!(\"{{:.17}}\", v); }} }}"
        ),
    )
    .unwrap();
    let rs_compile = Command::new("rustc")
        .args(["--edition=2021", "-Awarnings"])
        .arg(&rs_drv)
        .arg("-o")
        .arg(&rs_bin)
        .output()
        .unwrap();
    assert!(
        rs_compile.status.success(),
        "{}",
        String::from_utf8_lossy(&rs_compile.stderr)
    );
    let rs_out = Command::new(&rs_bin).output().unwrap();
    assert!(rs_out.status.success());
    let rs_vals: Vec<f64> = String::from_utf8(rs_out.stdout)
        .unwrap()
        .lines()
        .map(|s| s.parse().unwrap())
        .collect();

    for i in 0..want.len() {
        assert!((py_vals[i] - want[i]).abs() < 1e-10);
        assert!((rs_vals[i] - want[i]).abs() < 1e-10);
    }

    // Part B: Rust, Python, and C++17 3-language recipe export parity at 1e4 linear scale
    // and non-finite input refusal via LinearScaledWinner.
    let expr = MathExpr::parse("c0").unwrap();
    let fit = LinearScaledWinner::fit_training(
        &expr,
        &[vec![-10.0], vec![0.0], vec![10.0]],
        &[-10_000.0, 0.0, 10_000.0],
    )
    .unwrap();
    let emitted = fit.winner.emit().unwrap();

    // Rust
    let scaled_rs = dir.path().join("scaled.rs");
    let scaled_rlib = dir.path().join("libscaled.rlib");
    let scaled_drv = dir.path().join("scaled_drv.rs");
    let scaled_bin = dir.path().join("scaled_bin");
    fs::write(&scaled_rs, &emitted.rust).unwrap();
    assert!(Command::new("rustc")
        .args(["-O", "--crate-type=lib", "--crate-name=scaled"])
        .arg(&scaled_rs)
        .arg("-o")
        .arg(&scaled_rlib)
        .status()
        .unwrap()
        .success());
    fs::write(
        &scaled_drv,
        "extern crate scaled; fn main() { let p = scaled::scaled_predict(&[vec![-10.0], vec![0.0], vec![10.0]]).unwrap(); for v in p { println!(\"{:016x}\", v.to_bits()); } assert!(scaled::scaled_predict(&[vec![f64::NAN]]).is_err()); }",
    )
    .unwrap();
    assert!(Command::new("rustc")
        .args(["-O", "--extern"])
        .arg(format!("scaled={}", scaled_rlib.display()))
        .arg(&scaled_drv)
        .arg("-o")
        .arg(&scaled_bin)
        .status()
        .unwrap()
        .success());
    let rust_hex = String::from_utf8(Command::new(&scaled_bin).output().unwrap().stdout).unwrap();

    // Python
    let scaled_py = dir.path().join("model.py");
    let scaled_py_drv = dir.path().join("driver.py");
    fs::write(&scaled_py, &emitted.python).unwrap();
    fs::write(
        &scaled_py_drv,
        "from struct import pack\nfrom model import scaled_predict\nfor v in scaled_predict([[-10.0],[0.0],[10.0]]): print(pack('>d', v).hex())\ntry:\n    scaled_predict([[float('nan')]])\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('NaN not refused')\n",
    )
    .unwrap();
    let py_hex = String::from_utf8(
        Command::new("python3")
            .arg(&scaled_py_drv)
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap();

    // C++17
    let scaled_cpp = dir.path().join("scaled.cpp");
    let scaled_cpp_bin = dir.path().join("scaled_cpp_bin");
    fs::write(
        &scaled_cpp,
        format!(
            "{}\n#include <cstdio>\nint main() {{ auto p = scaled_predict({{{{-10.0}},{{0.0}},{{10.0}}}}); for (double v : p) {{ std::uint64_t b; std::memcpy(&b, &v, 8); std::printf(\"%016llx\\n\", static_cast<unsigned long long>(b)); }} try {{ scaled_predict({{{{std::numeric_limits<double>::quiet_NaN()}}}}); return 1; }} catch (const std::invalid_argument&) {{ return 0; }} }}",
            emitted.cpp17
        ),
    )
    .unwrap();
    let cpp_compile = Command::new("g++")
        .args(["-std=c++17", "-O2"])
        .arg(&scaled_cpp)
        .arg("-o")
        .arg(&scaled_cpp_bin)
        .output()
        .unwrap();
    assert!(
        cpp_compile.status.success(),
        "{}",
        String::from_utf8_lossy(&cpp_compile.stderr)
    );
    let cpp_out = Command::new(&scaled_cpp_bin).output().unwrap();
    assert!(cpp_out.status.success());
    let cpp_hex = String::from_utf8(cpp_out.stdout).unwrap();

    assert_eq!(rust_hex, py_hex);
    assert_eq!(rust_hex, cpp_hex);
}

#[test]
fn nan_distinct_from_zero_and_linear_unclamped() {
    nan_mask_distinct_from_zero();
    linear_output_unclamped_at_1e4();
    post_expansion_width_admitted_exact();
    wp15_mask_and_clamp_export_subset_parity();
}
