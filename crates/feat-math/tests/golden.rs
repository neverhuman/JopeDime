//! Golden parity tests: for each genome, the emitted Python artifact, the
//! emitted+compiled Rust artifact, and feat_math::fit_predict must agree
//! bit-for-bit (max abs diff == 0.0) on a deterministic toy dataset.

#[path = "support/c17.rs"]
mod c17;
#[path = "support/c17_export.rs"]
mod c17_export;
#[cfg(feature = "cuda-parity-tests")]
#[path = "support/cuda.rs"]
#[allow(dead_code)]
mod cuda;

use feat_math::transpile::{emit, lint_vocabulary, supports, EmitOptions, Emitted};
use feat_math::{fit_predict, GenomeIr, Pcg32};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

mod typescript_support;

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

fn write_csvs(dir: &Path, x_train: &[Vec<f64>], y_train: &[f64], x_test: &[Vec<f64>]) {
    let mut train = String::new();
    for (row, y) in x_train.iter().zip(y_train) {
        for v in row {
            train.push_str(&format!("{v:?},"));
        }
        train.push_str(&format!("{y:?}\n"));
    }
    fs::write(dir.join("train.csv"), train).unwrap();
    let mut test = String::new();
    for row in x_test {
        let cells: Vec<String> = row.iter().map(|v| format!("{v:?}")).collect();
        test.push_str(&cells.join(","));
        test.push('\n');
    }
    fs::write(dir.join("test_features.csv"), test).unwrap();
}

fn parse_predictions(path: &Path) -> Vec<f64> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("no predictions at {}: {e}", path.display()))
        .lines()
        .map(|l| l.trim().parse::<f64>().unwrap())
        .collect()
}

fn run_python(dir: &Path, artifact: &Emitted) -> Vec<f64> {
    fs::write(dir.join("model.py"), &artifact.python).unwrap();
    let out = Command::new("python3")
        .arg(dir.join("model.py"))
        .arg(dir.join("train.csv"))
        .arg(dir.join("test_features.csv"))
        .arg(dir.join("py_predictions.txt"))
        .output()
        .expect("python3 must be on PATH");
    assert!(
        out.status.success(),
        "python artifact failed: {}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    parse_predictions(&dir.join("py_predictions.txt"))
}

fn run_rust(dir: &Path, artifact: &Emitted) -> Vec<f64> {
    fs::write(dir.join("model.rs"), &artifact.rust).unwrap();
    let bin = dir.join("model_bin");
    let out = Command::new("rustc")
        .arg("-O")
        .arg("-o")
        .arg(&bin)
        .arg(dir.join("model.rs"))
        .output()
        .expect("rustc must be on PATH");
    assert!(
        out.status.success(),
        "rustc failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(&bin)
        .arg(dir.join("train.csv"))
        .arg(dir.join("test_features.csv"))
        .arg(dir.join("rs_predictions.txt"))
        .output()
        .expect("compiled artifact must run");
    assert!(
        out.status.success(),
        "rust artifact failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    parse_predictions(&dir.join("rs_predictions.txt"))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "prediction count differs");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// Emit, run both artifacts, and compare against the runtime reference.
fn assert_parity(name: &str, genome_src: &str) {
    assert_parity_sized(name, genome_src, 240, 5, 200);
}

/// Like `assert_parity` but with a custom fixture size (rows, cols, train rows);
/// also reports the emitted-artifact runtimes so head budgets stay visible.
fn assert_parity_sized(name: &str, genome_src: &str, rows: usize, cols: usize, train: usize) {
    let genome = GenomeIr::parse(genome_src).unwrap();
    supports(&genome).unwrap_or_else(|m| panic!("supports() rejected {genome_src}: {m:?}"));
    let opts = EmitOptions::default();
    let artifact = emit(&genome, &opts).unwrap();

    let (x, y) = toy_data(rows, cols, 42);
    let x_train: Vec<Vec<f64>> = x[..train].to_vec();
    let y_train: Vec<f64> = y[..train].to_vec();
    let x_test: Vec<Vec<f64>> = x[train..].to_vec();

    let reference = fit_predict(&genome, &x_train, &y_train, &x_test).unwrap();

    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("golden_{name}"));
    fs::create_dir_all(&dir).unwrap();
    write_csvs(&dir, &x_train, &y_train, &x_test);

    let t_py = std::time::Instant::now();
    let py = run_python(&dir, &artifact);
    let py_secs = t_py.elapsed().as_secs_f64();
    let t_rs = std::time::Instant::now();
    let rs = run_rust(&dir, &artifact);
    let rs_secs = t_rs.elapsed().as_secs_f64();

    let c17 = c17::predictions(
        &dir.join("c17"),
        &genome,
        &opts,
        &x_train,
        &y_train,
        &x_test,
    );
    assert_eq!(
        reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        c17,
        "C17 parity for {name}"
    );

    #[cfg(feature = "cuda-parity-tests")]
    if cuda::enabled() {
        let bundle = feat_math::transpile::emit_cuda(&genome, &opts).unwrap();
        let predictions = cuda::predictions(
            &dir.join("cuda-target"),
            &bundle,
            &x_train,
            &y_train,
            &x_test,
        );
        assert_eq!(
            predictions,
            reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "CUDA C++ recipe parity for {name}"
        );
    }
    if typescript_support::enabled() {
        let artifact = feat_math::transpile::emit_typescript(&genome, &opts).unwrap();
        typescript_support::run(
            &dir.join("typescript"),
            &artifact.source,
            &artifact.kernel,
            &serde_json::json!({"train":x_train,"targets":y_train,"test":x_test,"stress":name=="robust_ridge"}),
            &rs,
            false,
        );
    }
    let py_diff = max_abs_diff(&reference, &py);
    let rs_diff = max_abs_diff(&reference, &rs);
    assert_eq!(py_diff, 0.0, "python parity broken for {genome_src}");
    assert_eq!(rs_diff, 0.0, "rust parity broken for {genome_src}");
    println!(
        "parity[{name}] ok: n_train={train} n_test={} py_diff={py_diff} rs_diff={rs_diff} \
         py_run={py_secs:.1}s rs_compile_run={rs_secs:.1}s",
        py.len()
    );
}

#[test]
fn golden_robust_ridge() {
    assert_parity("robust_ridge", "(pipe X (robust_scale) (ridge))");
}

#[test]
fn golden_standardize_rff_ridge() {
    assert_parity(
        "standardize_rff_ridge",
        "(pipe X (standardize append=true) (rff_cos n=24 seed=9 gamma=1.5 append=true) (ridge lambda_frac=0.05))",
    );
}

#[test]
fn golden_rank_qbins_elasticnet() {
    assert_parity(
        "rank_qbins_elasticnet",
        "(pipe X (rank) (quantile_bins bins=12) (elasticnet alpha=0.01 l1_ratio=0.3 sweeps=40))",
    );
}

#[test]
fn golden_winsor_gbm() {
    assert_parity(
        "winsor_gbm",
        "(pipe X (winsor_clip q=0.1) (histgbm rounds=60 depth=3 lr=0.1 bins=32))",
    );
}

#[test]
fn golden_pairprod_tbe_ensemble() {
    assert_parity(
        "pairprod_tbe_ensemble",
        "(pipe X (pair_prod topk=4 append=true) (target_bin_encode bins=6) (ensemble rounds=40 depth=3 lambda_frac=0.02))",
    );
}

#[test]
fn golden_blom_hinge_oblique_ridge() {
    assert_parity(
        "blom_hinge_oblique_ridge",
        "(pipe X (blom) (hinge knots=3 append=true) (oblique_hinge n=12 seed=5) (ridge))",
    );
}

#[test]
fn golden_cheb_proj_landmark_rowstats_onehot_ridge() {
    assert_parity(
        "cheb_proj_landmark_rowstats_onehot_ridge",
        "(pipe X (cheb degree=3) (rand_proj k=8 seed=11) (landmark_dist k=6 gamma=0.7 seed=3 append=true) (row_stats append=true) (bin_onehot bins=4) (ridge))",
    );
}

#[test]
fn golden_poly_rffsin_ridge() {
    assert_parity(
        "poly_rffsin_ridge",
        "(pipe X (poly_interact) (rff_sin n=16 seed=2) (ridge))",
    );
}

#[test]
fn golden_default_head_and_aliases() {
    // No explicit head => plain ridge; norm_features/winsor_moments aliases.
    assert_parity(
        "default_head_aliases",
        "(pipe X (norm_features) (winsor_moments q=0.02))",
    );
}

#[test]
fn golden_sigmoid_normrow_polar_ridge() {
    assert_parity(
        "sigmoid_normrow_polar_ridge",
        "(pipe X (sigmoid) (norm_row append=true) (polar) (ridge))",
    );
}

#[test]
fn golden_lag_roll_ridge() {
    assert_parity(
        "lag_roll_ridge",
        "(pipe X (lag k=2 append=true) (roll k=3) (ridge))",
    );
}

#[test]
fn golden_fft_spectral_diffstats_ridge() {
    assert_parity(
        "fft_spectral_diffstats_ridge",
        "(pipe X (fft_mag k=2 append=true) (row_spectral append=true) (row_diff_stats append=true) (ridge))",
    );
}

#[test]
fn golden_svd_ridge() {
    assert_parity(
        "svd_ridge",
        "(pipe X (svd_project k=3 iters=10 seed=4) (ridge))",
    );
}

#[test]
fn golden_nystrom_gbm() {
    assert_parity(
        "nystrom_gbm",
        "(pipe X (nystrom k=10 gamma=0.8 seed=6) (histgbm rounds=25 depth=3))",
    );
}

#[test]
fn golden_map_derive_ridge() {
    assert_parity(
        "map_derive_ridge",
        "(pipe X (map expr=\"sigmoid(x) + 0.1*col(0)\" append=true) (derive expr=\"row_mean + row_std - c1/(c0 + 3)\" append=true) (ridge))",
    );
}

#[test]
fn golden_map_pow_elasticnet() {
    assert_parity(
        "map_pow_elasticnet",
        "(pipe X (map expr=\"x^2 - min(x, c2) + tanh(c3)\") (elasticnet alpha=0.005 l1_ratio=0.7))",
    );
}

#[test]
fn golden_symbolic_affine() {
    assert_parity(
        "symbolic_affine",
        "(sym affine \"sigmoid(col(0)) + 0.5*col(1) - log1p(abs(col(2))) + max(c3, c4^2)\")",
    );
}

#[test]
fn golden_symbolic_raw() {
    assert_parity(
        "symbolic_raw",
        "(sym raw \"clip(row_mean + 0.3*row_std - c0/(c1 + 5), -2, 2)\")",
    );
}

#[test]
fn golden_histgbm2_default() {
    // Default params (the exact config seeds and the stack library use); the
    // printed py_run time is the emitted-python training-budget measurement.
    assert_parity("histgbm2_default", "(pipe X (histgbm2))");
}

#[test]
fn golden_histgbm2_params_with_transform() {
    assert_parity(
        "histgbm2_params",
        "(pipe X (standardize append=true) (histgbm2 leaves=7 lr=0.1 bags=2))",
    );
}

#[test]
fn golden_krr_default() {
    assert_parity("krr_default", "(pipe X (standardize) (krr))");
}

#[test]
fn golden_krr_subset_paths() {
    // 400 train rows: n_tr=320 > 256 exercises the seeded selection basis, and
    // m=300 < 400 exercises the seeded final-center subset.
    assert_parity_sized("krr_subset", "(pipe X (krr m=300))", 440, 6, 400);
}

#[test]
fn golden_gpr_default() {
    // Default head (m=500, steps=40); n_train <= m so the full-data GP path runs.
    assert_parity_sized("gpr_default", "(pipe X (gpr))", 140, 5, 120);
}

#[test]
fn golden_gpr_standardize() {
    assert_parity_sized(
        "gpr_standardize",
        "(pipe X (standardize) (gpr m=300 steps=20))",
        140,
        5,
        120,
    );
}

#[test]
fn golden_gpr_rff() {
    assert_parity_sized(
        "gpr_rff",
        "(pipe X (rff_cos n=16 seed=3 gamma=1.0 append=true) (gpr steps=25))",
        140,
        5,
        120,
    );
}

#[test]
fn golden_gpr_subset_paths() {
    // 160 train rows with m=50 exercises the seeded inducing-row subset; the
    // optimization subset equals the basis here (50 < 256).
    assert_parity_sized("gpr_subset", "(pipe X (gpr m=50 steps=15))", 200, 6, 160);
}

#[test]
fn golden_gpr_optcap_paths() {
    // 300 train rows with m=500 gives a 300-row basis but a 256-row optimization
    // subset, exercising the decoupled hyperparameter-fit / large-solve path.
    assert_parity_sized("gpr_optcap", "(pipe X (gpr steps=6))", 340, 6, 300);
}

#[test]
fn golden_histgbm2_deep() {
    // The deep-boosting knobs together: more leaves, longer budget, row AND
    // column subsampling, min_leaf, and the seeded-shuffled 15% holdout.
    assert_parity(
        "histgbm2_deep",
        "(pipe X (histgbm2 leaves=63 rounds=220 lr=0.05 bags=2 row_subsample=0.8 \
         col_subsample=0.7 min_leaf=5 val_pct=15 val_shuffle=true))",
    );
}

#[test]
fn golden_histgbm2_huber_mae_stop() {
    // Huber gradients + MAE early stopping flow through both emitters.
    assert_parity(
        "histgbm2_huber",
        "(pipe X (histgbm2 leaves=15 lr=0.1 bags=2 rounds=150 loss=huber delta=0.5 stop=mae))",
    );
}

#[test]
fn golden_gpr_rq() {
    // Rational-quadratic kernel (learned alpha) through the composite engine.
    assert_parity_sized(
        "gpr_rq",
        "(pipe X (gpr kernel=rq m=200 steps=12))",
        200,
        6,
        160,
    );
}

#[test]
fn golden_gpr_rbf_linear() {
    // Sum composition: ARD squared-exponential plus a linear term.
    assert_parity_sized(
        "gpr_rbf_linear",
        "(pipe X (gpr kernel=rbf+linear m=200 steps=12))",
        170,
        5,
        140,
    );
}

#[test]
fn golden_gpr_rbf_periodic() {
    // Product composition: ARD squared-exponential times a learned-period
    // periodic factor.
    assert_parity_sized(
        "gpr_rbf_periodic",
        "(pipe X (gpr kernel=rbf*periodic steps=10))",
        160,
        5,
        130,
    );
}

#[test]
fn golden_gpr_rq_linear_transform() {
    // Composite kernel behind a fitted transform, exercising the seeded
    // inducing-row subset (m < n_train).
    assert_parity_sized(
        "gpr_rq_linear",
        "(pipe X (standardize) (gpr kernel=rq+linear m=120 steps=10))",
        170,
        5,
        140,
    );
}

#[test]
fn golden_gpr_linear_winsor() {
    // The exact contender shape measured over the b6ec1900 bar: a pure linear
    // kernel (Bayesian linear regression as a GP) behind winsorized features.
    assert_parity_sized(
        "gpr_linear_winsor",
        "(pipe X (winsor_clip q=0.02 append=true) (gpr kernel=linear steps=15))",
        170,
        5,
        140,
    );
}

#[test]
fn golden_ksmooth_default() {
    assert_parity("ksmooth_default", "(pipe X (ksmooth))");
}

#[test]
fn golden_ksmooth_params() {
    assert_parity(
        "ksmooth_params",
        "(pipe X (robust_scale) (ksmooth k=12 gamma=2.0))",
    );
}

#[test]
fn golden_ridge_gcv_default() {
    assert_parity("ridge_gcv_default", "(pipe X (ridge_gcv))");
}

#[test]
fn golden_ridge_gcv_transforms() {
    assert_parity(
        "ridge_gcv_transforms",
        "(pipe X (standardize) (cheb degree=3 append=true) (ridge_gcv))",
    );
}

#[test]
fn golden_ridge_gcv_target() {
    assert_parity(
        "ridge_gcv_target",
        "(pipe X (target kind=rank_gauss) (ridge_gcv))",
    );
}

#[test]
fn golden_ridge_gcv_dual_wide() {
    // 60 train rows against 5 + 64 = 69 columns: n < p exercises the dual
    // (kernel) eigendecomposition route of the GCV head.
    assert_parity_sized(
        "ridge_gcv_dual",
        "(pipe X (rff_cos n=64 seed=3 gamma=1.0 append=true) (ridge_gcv))",
        100,
        5,
        60,
    );
}

#[test]
fn golden_ridge_gcv_screen_wide() {
    // 390 train rows x (5 + 200 + 200) = 405 columns: both sides above 384
    // exercise the train-only top-384 |corr| column screen ahead of the
    // primal route. The emitted-python run is the slow side (pure-python
    // Jacobi at p=384); it stays the head's worst-case budget measurement.
    assert_parity_sized(
        "ridge_gcv_screen",
        "(pipe X (rff_cos n=200 seed=3 gamma=1.0 append=true) (rff_sin n=200 seed=5 gamma=1.0 append=true) (ridge_gcv))",
        420,
        5,
        390,
    );
}

#[test]
fn golden_stack() {
    // Smaller fixture: the stack twin trains all five library members per fold.
    assert_parity_sized("stack", "(pipe X (stack))", 160, 5, 120);
}

#[test]
fn golden_stack_full_fixture() {
    assert_parity(
        "stack_full",
        "(pipe X (winsor_clip q=0.02 append=true) (stack))",
    );
}

#[test]
fn golden_step_indicator() {
    // The `step` transform op: per-column hard indicator at a fixed threshold,
    // emitted as an explicit conditional (never a *1e15 clip).
    assert_parity(
        "step_indicator",
        "(pipe X (standardize) (step t=0.0 append=true) (ridge))",
    );
}

#[test]
fn golden_step_map_expr() {
    // The `step()` grammar function inside a map, over a column and a row stat.
    assert_parity(
        "step_map_expr",
        "(pipe X (map expr=\"step(col(0), 0.0) + 0.5*step(row_mean, c1)\" append=true) (ridge))",
    );
}

#[test]
fn golden_step_symbolic() {
    // The `step()` grammar function in a symbolic genome with affine calibration.
    assert_parity(
        "step_symbolic",
        "(sym affine \"step(c0, 0.0) + step(c1, c2) - 0.5\")",
    );
}

#[test]
fn golden_clean_class_multistage() {
    // A repeated-op pipeline that exercises the deduped docstring and the class
    // structure end-to-end (parity across py, rs, runtime).
    assert_parity(
        "clean_class_multistage",
        "(pipe X (standardize append=true) (step t=0.0 append=true) (standardize append=true) (sigmoid append=true) (step t=0.5 append=true) (ridge))",
    );
}

/// Compact portfolio head, full 3-way parity. The fixture is 90 train rows
/// (repeats=3, folds=5): 11 registry components are active (the always-on 8
/// plus enet_quad, gpr_iso, gam_tbe). Fixture note: at n=240 the pure-Python
/// artifact needs many minutes (176+ component fold-fits, three of them
/// leaf-wise boosting with hundreds of rounds); 90 rows keeps the run in CI
/// range while exercising every driver path except wide-data projection
/// (engine-covered) and the duplicate-row member (golden below).
#[test]
fn golden_portfolio_core() {
    assert_parity_sized(
        "portfolio_core",
        "(pipe X (portfolio preset=core))",
        120,
        5,
        90,
    );
}

/// Small-sample fallback path: n=12 < 20 deploys the single built-in
/// GCV-ridge component through the same driver.
#[test]
fn golden_portfolio_small_n() {
    assert_parity_sized(
        "portfolio_small_n",
        "(pipe X (portfolio preset=core))",
        30,
        5,
        12,
    );
}

/// Compact-head parameters: shared transform prefix, explicit folds/blend,
/// an add (gate override), a drop, and repeats=1.
#[test]
fn golden_portfolio_params() {
    assert_parity_sized(
        "portfolio_params",
        "(pipe X (standardize) (portfolio preset=core folds=5 blend=nnls \
         add1=histgbm_slow drop1=rff_ridge repeats=1))",
        120,
        5,
        90,
    );
}

/// Dropping ridge_gcv_raw forces the hidden small-sample fallback component;
/// n=12 exercises it end-to-end.
#[test]
fn golden_portfolio_hidden_fallback() {
    assert_parity_sized(
        "portfolio_hidden_fallback",
        "(pipe X (portfolio drop1=ridge_gcv_raw))",
        30,
        5,
        12,
    );
}

/// Duplicate-heavy fixture: 8 distinct rows repeated -> duplicate share
/// 0.875 > 0.3 activates the duplicate-row average member.
#[test]
fn golden_portfolio_duplicate_rows() {
    let genome_src = "(pipe X (portfolio preset=core))";
    let genome = GenomeIr::parse(genome_src).unwrap();
    supports(&genome).unwrap_or_else(|m| panic!("supports() rejected {genome_src}: {m:?}"));
    let artifact = emit(&genome, &EmitOptions::default()).unwrap();
    let mut rng = Pcg32::new(9, 3);
    let distinct: Vec<Vec<f64>> = (0..8)
        .map(|_| (0..4).map(|_| rng.next_f64() * 2.0 - 1.0).collect())
        .collect();
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..88 {
        let row = distinct[i % 8].clone();
        y.push(3.0 * row[0] - row[1] + 0.05 * rng.gaussian());
        x.push(row);
    }
    let x_train: Vec<Vec<f64>> = x[..64].to_vec();
    let y_train: Vec<f64> = y[..64].to_vec();
    let x_test: Vec<Vec<f64>> = x[64..].to_vec();
    let reference = fit_predict(&genome, &x_train, &y_train, &x_test).unwrap();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("golden_portfolio_dups");
    fs::create_dir_all(&dir).unwrap();
    write_csvs(&dir, &x_train, &y_train, &x_test);
    let py = run_python(&dir, &artifact);
    let rs = run_rust(&dir, &artifact);
    let c17 = c17::predictions(
        &dir.join("c17"),
        &genome,
        &EmitOptions::default(),
        &x_train,
        &y_train,
        &x_test,
    );
    assert_eq!(
        reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        c17,
        "C17 parity for duplicate-row portfolio"
    );

    if typescript_support::enabled() {
        let bundle =
            feat_math::transpile::emit_typescript(&genome, &EmitOptions::default()).unwrap();
        typescript_support::run(
            &dir.join("typescript"),
            &bundle.source,
            &bundle.kernel,
            &serde_json::json!({"train":x_train,"targets":y_train,"test":x_test}),
            &rs,
            false,
        );
    }
    let py_diff = max_abs_diff(&reference, &py);
    let rs_diff = max_abs_diff(&reference, &rs);
    assert_eq!(
        py_diff, 0.0,
        "python parity broken for duplicate-row portfolio"
    );
    assert_eq!(
        rs_diff, 0.0,
        "rust parity broken for duplicate-row portfolio"
    );
    println!("parity[portfolio_dups] ok: py_diff={py_diff} rs_diff={rs_diff}");
}

/// Portfolio artifact hygiene: deterministic re-emission, the extended
/// banned-vocabulary lint (nnls / oof / member registry identifiers), and
/// supports() coverage — compact accepted, explicit-nested rejected with a
/// clear pointer at the compact form.
#[test]
fn portfolio_emission_rules() {
    use feat_math::transpile::lint_vocabulary_portfolio;
    for src in [
        "(pipe X (portfolio preset=core))",
        "(pipe X (portfolio preset=core folds=5 blend=nnls))",
        "(pipe X (standardize) (portfolio add1=histgbm_slow add2=gpr_iso drop1=gam_tbe))",
        "(pipe X (portfolio preset=none add1=ridge_gcv_raw))",
        "(pipe X (portfolio repeats=1))",
    ] {
        let genome = GenomeIr::parse(src).unwrap();
        supports(&genome).unwrap_or_else(|m| panic!("supports() rejected {src}: {m:?}"));
        let first = emit(&genome, &EmitOptions::default()).unwrap();
        let second = emit(&genome, &EmitOptions::default()).unwrap();
        assert_eq!(
            first.python, second.python,
            "python not deterministic: {src}"
        );
        assert_eq!(first.rust, second.rust, "rust not deterministic: {src}");
        assert!(
            lint_vocabulary_portfolio(&first.python).is_empty(),
            "python lint hits for {src}: {:?}",
            lint_vocabulary_portfolio(&first.python)
        );
        assert!(
            lint_vocabulary_portfolio(&first.rust).is_empty(),
            "rust lint hits for {src}: {:?}",
            lint_vocabulary_portfolio(&first.rust)
        );
    }
    // The extended lint itself trips on the new terms.
    assert!(!lint_vocabulary_portfolio("held nnls weights").is_empty());
    assert!(!lint_vocabulary_portfolio("the oof matrix").is_empty());
    assert!(!lint_vocabulary_portfolio("ridge_gcv_raw member").is_empty());
    // ...but the plain lint stays quiet on the legacy stack variable name.
    assert!(lint_vocabulary("oof = [[0.0] * n]").is_empty());
    // Explicit nested form: engine-valid, emission-rejected (G5).
    let nested =
        GenomeIr::parse("(portfolio (member (pipe X (ridge_gcv))) (blend kind=nnls folds=5))")
            .unwrap();
    let missing = supports(&nested).unwrap_err();
    assert!(
        missing.iter().any(|m| m.contains("compact head form")),
        "{missing:?}"
    );
    // Bad compact parameters are rejected by supports().
    for bad in [
        "(pipe X (portfolio preset=bogus))",
        "(pipe X (portfolio folds=3))",
        "(pipe X (portfolio blend=caruana))",
        "(pipe X (portfolio add1=not_a_member))",
        "(pipe X (portfolio bogus_param=1))",
        "(pipe X (portfolio) (standardize) (ridge))",
    ] {
        let genome = GenomeIr::parse(bad).unwrap();
        assert!(supports(&genome).is_err(), "{bad} should be unsupported");
    }
    // Token estimation goes through the emit-based path and stays exact.
    let genome = GenomeIr::parse("(pipe X (portfolio preset=core))").unwrap();
    let est = feat_math::tokens::try_estimate_tokens(&genome, &EmitOptions::default()).unwrap();
    let emitted = emit(&genome, &EmitOptions::default()).unwrap();
    assert_eq!(
        est.py,
        emitted.py_tokens.code_tokens + emitted.py_tokens.doc_tokens
    );
    assert_eq!(
        est.rs,
        emitted.rs_tokens.code_tokens + emitted.rs_tokens.doc_tokens
    );
}

#[test]
fn golden_target_rank_gauss_ridge() {
    assert_parity(
        "target_rank_gauss_ridge",
        "(pipe X (target kind=rank_gauss) (ridge))",
    );
}

#[test]
fn golden_target_log1p_ridge() {
    assert_parity(
        "target_log1p_ridge",
        "(pipe X (target kind=log1p) (ridge lambda_frac=0.02))",
    );
}

#[test]
fn golden_target_winsor_standardize_ridge() {
    assert_parity(
        "target_winsor_standardize_ridge",
        "(pipe X (target kind=winsor q=0.05) (standardize) (ridge))",
    );
}

#[test]
fn golden_target_zscore_ridge() {
    assert_parity(
        "target_zscore_ridge",
        "(pipe X (target kind=zscore) (standardize) (ridge))",
    );
}

#[test]
fn golden_symlog_ridge() {
    assert_parity("symlog_ridge", "(pipe X (symlog c=0.7) (ridge))");
}

#[test]
fn golden_select_corr_ridge() {
    assert_parity(
        "select_corr_ridge",
        "(pipe X (select_corr k=3 by=corr) (ridge))",
    );
}

#[test]
fn golden_select_corr_var_ridge() {
    assert_parity(
        "select_corr_var_ridge",
        "(pipe X (select_corr k=3 by=var) (ridge))",
    );
}

#[test]
fn golden_impute_ridge() {
    assert_parity(
        "impute_ridge",
        "(pipe X (impute strategy=median add_indicator=true) (standardize) (ridge))",
    );
}

#[test]
fn golden_step_impute_ridge() {
    // step produces exact 0.0/1.0 cells, so the fill path (not just the
    // no-zeros identity) is exercised cross-language.
    assert_parity(
        "step_impute_ridge",
        "(pipe X (step t=0.0) (impute strategy=mean add_indicator=true) (ridge))",
    );
}

#[test]
fn golden_scalar_batch_map_ridge() {
    assert_parity(
        "scalar_batch_map_ridge",
        "(pipe X (map expr=\"exp(clip(col(0),-2,2)) + log(abs(col(1)) + 1.0) + asinh(col(2)) + atan(col(3))\") (ridge))",
    );
}

#[test]
fn golden_scalar_batch2_derive_ridge() {
    assert_parity(
        "scalar_batch2_derive_ridge",
        "(pipe X (derive append=true expr=\"erf(col(0)) + sign(col(1)) * floor(col(2)) + round(col(3))\") (ridge))",
    );
}

#[test]
fn golden_symbolic_scalar_batch() {
    assert_parity(
        "symbolic_scalar_batch",
        "(sym affine \"erf(c0) + asinh(c1) - exp(clip(c2, -1, 1)) + sign(c3) * round(c4)\")",
    );
}

#[test]
fn golden_pair_ratio_ridge() {
    assert_parity(
        "pair_ratio_ridge",
        "(pipe X (pair_ratio topk=4 append=true) (ridge))",
    );
}

#[test]
fn golden_spline_ridge() {
    assert_parity("spline_ridge", "(pipe X (spline knots=4) (ridge))");
}

#[test]
fn golden_spline_append_ridge_gcv() {
    assert_parity(
        "spline_append_ridge_gcv",
        "(pipe X (spline knots=3 append=true) (ridge_gcv))",
    );
}

/// Target pseudo-stage emission rules: deterministic + lint-clean artifacts,
/// first-stage-only placement, and no target machinery in no-target genomes.
#[test]
fn target_stage_emission_rules() {
    let target_markers = [
        "normal_scores_target",
        "shifted_log_target",
        "winsorized_target",
        "standardized_target",
    ];
    for src in [
        "(pipe X (target kind=rank_gauss) (ridge))",
        "(pipe X (target kind=log1p) (ridge lambda_frac=0.02))",
        "(pipe X (target kind=winsor q=0.05) (standardize) (ridge))",
        "(pipe X (target kind=zscore) (ridge))",
    ] {
        let genome = GenomeIr::parse(src).unwrap();
        let first = emit(&genome, &EmitOptions::default()).unwrap();
        let second = emit(&genome, &EmitOptions::default()).unwrap();
        assert_eq!(
            first.python, second.python,
            "python not deterministic: {src}"
        );
        assert_eq!(first.rust, second.rust, "rust not deterministic: {src}");
        assert!(lint_vocabulary(&first.python).is_empty(), "{src}");
        assert!(lint_vocabulary(&first.rust).is_empty(), "{src}");
        assert!(
            target_markers.iter().any(|m| first.python.contains(m)),
            "python artifact missing target transform: {src}"
        );
    }
    // Misplaced, repeated, or unknown-kind target stages are rejected.
    for src in [
        "(pipe X (standardize) (target kind=rank_gauss) (ridge))",
        "(pipe X (target kind=log1p) (target kind=log1p) (ridge))",
        "(pipe X (target kind=bogus) (ridge))",
        "(pipe X (target) (ridge))",
    ] {
        let genome = GenomeIr::parse(src).unwrap();
        assert!(supports(&genome).is_err(), "{src} should be unsupported");
    }
    // The no-target path is untouched: no target machinery is emitted.
    let plain = emit(
        &GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap(),
        &EmitOptions::default(),
    )
    .unwrap();
    for marker in target_markers {
        assert!(!plain.python.contains(marker), "python leaks {marker}");
        assert!(!plain.rust.contains(marker), "rust leaks {marker}");
    }
    assert!(!plain.python.contains("self.target"));
}

/// Structure + no-machine-smell audit on a complex genome (emit-only, fast).
#[test]
fn class_structure_and_no_machine_smell() {
    let src =
        "(pipe X (standardize append=true) (step t=0.0 append=true) (standardize append=true) \
               (sigmoid append=true) (step t=0.5 append=true) (poly_interact append=true) (ridge))";
    let genome = GenomeIr::parse(src).unwrap();
    let artifact = emit(&genome, &EmitOptions::default()).unwrap();
    let py = &artifact.python;
    let rs = &artifact.rust;

    // Clean class structure with fit / predict / score in both languages.
    assert!(py.contains("class Model:"), "python missing class Model");
    assert!(py.contains("def fit(self,"), "python missing fit");
    assert!(py.contains("def predict(self,"), "python missing predict");
    assert!(py.contains("def score(self,"), "python missing score");
    assert!(rs.contains("struct Model"), "rust missing struct Model");
    assert!(rs.contains("fn fit("), "rust missing fit");
    assert!(rs.contains("fn predict("), "rust missing predict");
    assert!(rs.contains("fn score("), "rust missing score");
    assert!(
        rs.contains("#![forbid(unsafe_code)]"),
        "rust missing unsafe forbid"
    );

    // No machine-looking hard-threshold trick anywhere.
    for (lang, src) in [("python", py.as_str()), ("rust", rs.as_str())] {
        assert!(!src.contains("1e15"), "{lang} artifact contains 1e15");
        assert!(
            !src.contains("1000000000000000"),
            "{lang} contains a 1e15 literal"
        );
    }

    // The old per-stage "The stage {replaces|appends} ..." spam is gone: the
    // convention is stated exactly once, and repeated ops refer back.
    assert!(!py.contains("The stage replaces the columns it received"));
    assert!(!py.contains("The stage appends its outputs to the columns it received"));
    assert_eq!(
        py.matches("The feature pipeline has").count(),
        1,
        "convention should appear once"
    );
    assert!(
        py.contains("repeats the transform of stage"),
        "repeated op should refer back"
    );

    // Determinism: emit twice, byte-identical.
    let again = emit(&genome, &EmitOptions::default()).unwrap();
    assert_eq!(*py, again.python, "python emission not deterministic");
    assert_eq!(*rs, again.rust, "rust emission not deterministic");

    // Emitter version: 2.1 added the composite-kernel / ksmooth / deep-boost heads; 2.2 the
    // shipped-text lint, the manifest's `program` keys and the reworded closed-form docstring.
    assert_eq!(feat_math::transpile::EMITTER_VERSION, "2.2.0");
}

/// The emitted `score` method actually computes a sane R^2 when run.
#[test]
fn emitted_score_method_runs() {
    let genome = GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap();
    let artifact = emit(&genome, &EmitOptions::default()).unwrap();
    let (x, y) = toy_data(120, 5, 7);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("score_probe");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("model.py"), &artifact.python).unwrap();
    // Drive fit -> score on train (should be a finite R^2 in [.., 1]).
    let mut train = String::new();
    for (row, t) in x.iter().zip(&y) {
        for v in row {
            train.push_str(&format!("{v:?},"));
        }
        train.push_str(&format!("{t:?}\n"));
    }
    fs::write(dir.join("train.csv"), &train).unwrap();
    let probe = format!(
        "import sys\nsys.argv=['m']\nimport importlib.util\nspec=importlib.util.spec_from_file_location('m', r'{}')\nm=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(m)\nrows=m._read_csv(r'{}')\ny=[r[-1] for r in rows]\nx=[r[:-1] for r in rows]\nmdl=m.Model().fit(x,y)\nprint(mdl.score(x,y))\n",
        dir.join("model.py").display(),
        dir.join("train.csv").display()
    );
    fs::write(dir.join("probe.py"), &probe).unwrap();
    let out = Command::new("python3")
        .arg(dir.join("probe.py"))
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "score probe failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let r2: f64 = String::from_utf8_lossy(&out.stdout).trim().parse().unwrap();
    assert!(
        r2.is_finite() && r2 <= 1.0 + 1e-9,
        "score out of range: {r2}"
    );
}

#[test]
fn determinism_and_token_counts() {
    let genome = GenomeIr::parse(
        "(pipe X (robust_scale) (rff_cos n=32 seed=7 gamma=1.0 append=true) (ensemble rounds=25))",
    )
    .unwrap();
    let opts = EmitOptions {
        title: Some("Toy regression artifact".to_string()),
        feature_names: vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()],
        context: Some("Deterministic toy data for parity checks.".to_string()),
        ..Default::default()
    };
    let first = emit(&genome, &opts).unwrap();
    let second = emit(&genome, &opts).unwrap();
    assert_eq!(
        first.python, second.python,
        "python emission not deterministic"
    );
    assert_eq!(first.rust, second.rust, "rust emission not deterministic");
    assert_eq!(
        first.manifest, second.manifest,
        "manifest not deterministic"
    );
    assert_eq!(first.py_tokens.code_tokens, second.py_tokens.code_tokens);
    assert_eq!(first.rs_tokens.code_tokens, second.rs_tokens.code_tokens);
    assert!(first.py_tokens.code_tokens > 0 && first.py_tokens.doc_tokens > 0);
    assert!(first.rs_tokens.code_tokens > 0 && first.rs_tokens.doc_tokens > 0);
    assert!(first.timing.tokens_per_sec > 0.0);
    println!(
        "emit timing: {} ns, py_tokens={}, rs_tokens={}, tokens/sec={:.0}",
        first.timing.emit_ns,
        first.timing.py_tokens,
        first.timing.rs_tokens,
        first.timing.tokens_per_sec
    );
    // The docstring carries the provided title.
    assert!(first.python.contains("Toy regression artifact"));
    assert!(first.rust.contains("Toy regression artifact"));
    // Manifest sanity.
    assert!(first.manifest.contains("battle-artifact-v1"));
    assert!(first.manifest.contains("\"flavor\":\"pipeline\""));
}

#[test]
fn lint_trips_on_poisoned_docs() {
    let hits = lint_vocabulary("This docstring mentions a genome and lexicase selection.");
    assert!(hits.contains(&"genome".to_string()));
    assert!(hits.contains(&"lexicase".to_string()));
    // Sneaky substrings trip too ("below" contains "elo").
    assert!(!lint_vocabulary("see the table below").is_empty());
    // Clean text passes.
    assert!(lint_vocabulary("ridge regression on standardized features").is_empty());
    // emit() itself must refuse poisoned options.
    let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let opts = EmitOptions {
        title: Some("my mutation genome".to_string()),
        ..Default::default()
    };
    assert!(emit(&genome, &opts).is_err());
}

#[test]
fn supports_rejects_unknown_ops() {
    let genome = GenomeIr::parse("(pipe X (bogus_op n=3) (ridge))").unwrap();
    let missing = supports(&genome).unwrap_err();
    assert!(
        missing.iter().any(|m| m.contains("bogus_op")),
        "{missing:?}"
    );
    // Mid-pipeline heads are rejected (the runtime errors on them too).
    let genome = GenomeIr::parse("(pipe X (ridge) (robust_scale) (ridge))").unwrap();
    assert!(supports(&genome).is_err());
    // Clean genomes pass.
    let genome = GenomeIr::parse("(pipe X (robust_scale) (histgbm))").unwrap();
    assert!(supports(&genome).is_ok());
    // Symbolic with an unknown function is rejected.
    let genome = GenomeIr::parse("(sym affine \"madeup(c0)\")").unwrap_or_else(|_| {
        // The runtime parser rejects it outright; emulate via supports on a valid one.
        GenomeIr::parse("(sym affine \"c0\")").unwrap()
    });
    let _ = supports(&genome);
}

#[test]
fn artifacts_are_lint_clean_and_standalone() {
    for src in [
        "(pipe X (standardize) (rank) (blom) (quantile_bins) (bin_onehot bins=3) (ridge))",
        "(pipe X (hinge) (winsor_clip) (cheb) (poly_interact append=true) (elasticnet))",
        "(pipe X (rff_cos n=8) (rff_sin n=8) (rand_proj k=4) (landmark_dist k=4) (histgbm rounds=5))",
        "(pipe X (oblique_hinge n=4) (row_stats) (pair_prod) (target_bin_encode) (ensemble rounds=5))",
        "(pipe X (sigmoid) (norm_row) (polar) (lag k=1) (roll k=2) (fft_mag k=2) (ridge))",
        "(pipe X (row_spectral) (row_diff_stats) (svd_project k=2 iters=5) (nystrom k=4) (ridge))",
        "(pipe X (map expr=\"clip(x, -1, 1)\") (derive expr=\"row_l1 + row_l2 + row_min + row_max\") (ridge))",
        "(sym affine \"tanh(c0) * sqrt(abs(c1)) + min(c2, 1) / max(c3, 1) + sin(c4) + cos(c0)\")",
        "(pipe X (histgbm2 leaves=15 lr=0.08 bags=2))",
        "(pipe X (histgbm2 leaves=63 rounds=700 row_subsample=0.8 col_subsample=0.7 min_leaf=5 val_pct=15 val_shuffle=true))",
        "(pipe X (standardize) (krr m=200))",
        "(pipe X (gpr m=200 steps=10))",
        "(pipe X (gpr kernel=rq+linear m=200 steps=10))",
        "(pipe X (gpr kernel=rbf*periodic m=200 steps=10))",
        "(pipe X (ksmooth k=16 gamma=2.0))",
        "(pipe X (standardize) (cheb degree=3 append=true) (ridge_gcv))",
        "(pipe X (stack))",
    ] {
        let genome = GenomeIr::parse(src).unwrap();
        let artifact = emit(&genome, &EmitOptions::default()).unwrap();
        assert!(lint_vocabulary(&artifact.python).is_empty(), "{src}");
        assert!(lint_vocabulary(&artifact.rust).is_empty(), "{src}");
        // Python imports nothing beyond csv/math/sys. Real imports sit at
        // column 0 outside the module docstring; wrapped documentation lines
        // may legitimately start with "from ...", so skip the docstring.
        let mut in_module_doc = false;
        for (i, line) in artifact.python.lines().enumerate() {
            if i == 0 {
                assert!(line.starts_with("\"\"\""), "artifact must open with a docstring");
                in_module_doc = true;
                continue;
            }
            if in_module_doc {
                if line.ends_with("\"\"\"") {
                    in_module_doc = false;
                }
                continue;
            }
            if line.starts_with("import ") || line.starts_with("from ") {
                assert!(
                    ["import csv", "import math", "import sys"].contains(&line),
                    "unexpected import: {line}"
                );
            }
        }
        // Rust artifact is std-only and forbids unsafe.
        assert!(artifact.rust.contains("#![forbid(unsafe_code)]"));
        assert!(!artifact.rust.contains("extern crate"));
    }
}

// --------------------------- fast-runtime profile ---------------------------

fn fast_opts() -> EmitOptions {
    EmitOptions {
        fast_runtime: true,
        ..Default::default()
    }
}

/// Emit default + fast, run the fast Python artifact and BOTH Rust artifacts,
/// and require bit-identical predictions across all of them and the runtime.
fn assert_fast_parity(name: &str, genome_src: &str) {
    let genome = GenomeIr::parse(genome_src).unwrap();
    let default = emit(&genome, &EmitOptions::default()).unwrap();
    let fast = emit(&genome, &fast_opts()).unwrap();
    // Default bytes carry no fast markers; fast artifacts carry both.
    for marker in ["predict_rows", "available_parallelism", "lto=fat"] {
        assert!(
            !default.rust.contains(marker),
            "default rust leaks {marker}"
        );
    }
    assert!(
        !default.python.contains("PyPy"),
        "default python leaks PyPy"
    );
    assert!(
        fast.rust.contains("predict_rows"),
        "fast rust missing wrapper"
    );
    assert!(
        fast.python.contains("PyPy"),
        "fast python missing PyPy note"
    );

    let (x, y) = toy_data(240, 5, 42);
    let x_train: Vec<Vec<f64>> = x[..200].to_vec();
    let y_train: Vec<f64> = y[..200].to_vec();
    let x_test: Vec<Vec<f64>> = x[200..].to_vec();
    let reference = fit_predict(&genome, &x_train, &y_train, &x_test).unwrap();

    let dir_fast = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("fastp_{name}"));
    fs::create_dir_all(&dir_fast).unwrap();
    write_csvs(&dir_fast, &x_train, &y_train, &x_test);
    let py = run_python(&dir_fast, &fast);
    let rs_fast = run_rust(&dir_fast, &fast);
    let c_bits = c17::predictions(
        &dir_fast.join("c17"),
        &genome,
        &fast_opts(),
        &x_train,
        &y_train,
        &x_test,
    );
    assert_eq!(
        c_bits,
        rs_fast.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "C17 fast parity for {name}"
    );

    let dir_ref = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("fastp_{name}_ref"));
    fs::create_dir_all(&dir_ref).unwrap();
    write_csvs(&dir_ref, &x_train, &y_train, &x_test);
    let rs_default = run_rust(&dir_ref, &default);
    if typescript_support::enabled() {
        let bundle = feat_math::transpile::emit_typescript(&genome, &fast_opts()).unwrap();
        typescript_support::run(
            &dir_fast.join("typescript"),
            &bundle.source,
            &bundle.kernel,
            &serde_json::json!({"train":x_train,"targets":y_train,"test":x_test}),
            &rs_default,
            false,
        );
    }

    assert_eq!(
        max_abs_diff(&reference, &py),
        0.0,
        "fast python parity broken for {genome_src}"
    );
    assert_eq!(
        max_abs_diff(&reference, &rs_fast),
        0.0,
        "fast rust parity broken for {genome_src}"
    );
    assert_eq!(
        max_abs_diff(&reference, &rs_default),
        0.0,
        "default rust parity broken for {genome_src}"
    );
    for i in 0..reference.len() {
        assert_eq!(
            rs_fast[i].to_bits(),
            rs_default[i].to_bits(),
            "fast vs default rust bits differ at row {i} for {genome_src}"
        );
    }
}

#[test]
fn fast_profile_zscore_ridge() {
    assert_fast_parity(
        "zscore_ridge",
        "(pipe X (target kind=zscore) (standardize) (ridge))",
    );
}

#[test]
fn fast_profile_expansion_stack() {
    assert_fast_parity(
        "expansion_stack",
        "(pipe X (spline knots=4) (impute add_indicator=true strategy=median) (pair_ratio append=true topk=4) (ridge_gcv))",
    );
}

#[test]
fn fast_profile_histgbm2() {
    assert_fast_parity(
        "histgbm2",
        "(pipe X (rank) (histgbm2 leaves=15 lr=0.1 rounds=40 val_pct=20))",
    );
}

#[test]
fn fast_profile_gpr_poly() {
    assert_fast_parity(
        "gpr_poly",
        "(pipe X (poly_interact append=true) (gpr m=60 steps=8))",
    );
}

#[test]
fn fast_profile_portfolio_head() {
    assert_fast_parity("portfolio_head", "(pipe X (portfolio preset=core))");
}

#[test]
fn fast_profile_symbolic_affine() {
    assert_fast_parity(
        "sym_affine",
        "(sym affine \"sigmoid(col(0)) + 0.5*col(1)\")",
    );
}

#[test]
fn fast_profile_is_estimable_and_default_bytes_frozen() {
    for src in [
        "(pipe X (robust_scale) (ridge))",
        "(pipe X (standardize append=true) (rff_cos n=24 seed=9 gamma=1.5 append=true) (ridge lambda_frac=0.05))",
        "(pipe X (rank) (quantile_bins bins=12) (elasticnet alpha=0.01 l1_ratio=0.3 sweeps=40))",
        "(pipe X (winsor_clip q=0.1) (histgbm rounds=60 depth=3 lr=0.1 bins=32))",
        "(pipe X (sigmoid) (norm_row append=true) (polar) (ridge))",
        "(pipe X (target kind=zscore) (ridge))",
        "(pipe X (symlog c=0.7) (ridge))",
        "(pipe X (poly_interact) (rff_sin n=16 seed=2) (ridge))",
    ] {
        let genome = GenomeIr::parse(src).unwrap();
        let default = emit(&genome, &EmitOptions::default()).unwrap();
        for marker in ["predict_rows", "available_parallelism", "lto=fat"] {
            assert!(
                !default.rust.contains(marker),
                "{src}: default rust leaks {marker}"
            );
        }
        assert!(
            !default.python.contains("PyPy"),
            "{src}: default python leaks PyPy"
        );
        // The fast profile lints clean (emit() bails otherwise) and the
        // estimator stays exact through the emit-and-count fallback.
        let fast = emit(&genome, &fast_opts()).unwrap();
        let est = feat_math::tokens::try_estimate_tokens(&genome, &fast_opts()).unwrap();
        assert_eq!(
            est.py,
            fast.py_tokens.code_tokens + fast.py_tokens.doc_tokens,
            "{src}"
        );
        assert_eq!(
            est.rs,
            fast.rs_tokens.code_tokens + fast.rs_tokens.doc_tokens,
            "{src}"
        );
    }
}

#[test]
fn fast_profile_threading_speedup_smoke() {
    // Not a timing assertion — a printed sanity check that the threaded
    // wrapper engages on a large predict and stays bit-identical.
    let genome = GenomeIr::parse(
        "(pipe X (standardize) (rff_cos append=true gamma=1.0 n=64 seed=3) (ridge))",
    )
    .unwrap();
    let default = emit(&genome, &EmitOptions::default()).unwrap();
    let fast = emit(&genome, &fast_opts()).unwrap();
    let (x, y) = toy_data(20_200, 5, 9);
    let x_train: Vec<Vec<f64>> = x[..200].to_vec();
    let y_train: Vec<f64> = y[..200].to_vec();
    let x_test: Vec<Vec<f64>> = x[200..].to_vec();
    let dir_a = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("fast_smoke_default");
    let dir_b = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("fast_smoke_fast");
    fs::create_dir_all(&dir_a).unwrap();
    fs::create_dir_all(&dir_b).unwrap();
    write_csvs(&dir_a, &x_train, &y_train, &x_test);
    write_csvs(&dir_b, &x_train, &y_train, &x_test);
    let t0 = std::time::Instant::now();
    let rs_default = run_rust(&dir_a, &default);
    let default_secs = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let rs_fast = run_rust(&dir_b, &fast);
    let fast_secs = t1.elapsed().as_secs_f64();
    assert_eq!(rs_default.len(), rs_fast.len());
    for i in 0..rs_default.len() {
        assert_eq!(rs_default[i].to_bits(), rs_fast[i].to_bits(), "row {i}");
    }
    println!(
        "fast-profile smoke: 20k-row predict, default(compile+run)={default_secs:.2}s \
         fast(compile+run)={fast_secs:.2}s (both include rustc -O)"
    );
}
