//! X6 acceptance: the static token estimator must EXACTLY match the real
//! emitted token counts (code + doc, per language) on every golden fixture
//! genome and on hundreds of random pipeline/symbolic genomes, and it must be
//! far cheaper than emission.

use feat_math::tokens::{estimate_tokens, try_estimate_tokens, TokenEstimate};
use feat_math::transpile::{emit, EmitOptions, GenomeIr};
use feat_math::Pcg32;

/// Every genome exercised by tests/golden.rs (parity fixtures) plus the
/// lint-audit set: the estimator must be exact on all of them.
const GOLDEN_GENOMES: &[&str] = &[
    "(pipe X (robust_scale) (ridge))",
    "(pipe X (standardize append=true) (rff_cos n=24 seed=9 gamma=1.5 append=true) (ridge lambda_frac=0.05))",
    "(pipe X (rank) (quantile_bins bins=12) (elasticnet alpha=0.01 l1_ratio=0.3 sweeps=40))",
    "(pipe X (winsor_clip q=0.1) (histgbm rounds=60 depth=3 lr=0.1 bins=32))",
    "(pipe X (pair_prod topk=4 append=true) (target_bin_encode bins=6) (ensemble rounds=40 depth=3 lambda_frac=0.02))",
    "(pipe X (blom) (hinge knots=3 append=true) (oblique_hinge n=12 seed=5) (ridge))",
    "(pipe X (cheb degree=3) (rand_proj k=8 seed=11) (landmark_dist k=6 gamma=0.7 seed=3 append=true) (row_stats append=true) (bin_onehot bins=4) (ridge))",
    "(pipe X (poly_interact) (rff_sin n=16 seed=2) (ridge))",
    "(pipe X (norm_features) (winsor_moments q=0.02))",
    "(pipe X (sigmoid) (norm_row append=true) (polar) (ridge))",
    "(pipe X (lag k=2 append=true) (roll k=3) (ridge))",
    "(pipe X (fft_mag k=2 append=true) (row_spectral append=true) (row_diff_stats append=true) (ridge))",
    "(pipe X (svd_project k=3 iters=10 seed=4) (ridge))",
    "(pipe X (nystrom k=10 gamma=0.8 seed=6) (histgbm rounds=25 depth=3))",
    "(pipe X (map expr=\"sigmoid(x) + 0.1*col(0)\" append=true) (derive expr=\"row_mean + row_std - c1/(c0 + 3)\" append=true) (ridge))",
    "(pipe X (map expr=\"x^2 - min(x, c2) + tanh(c3)\") (elasticnet alpha=0.005 l1_ratio=0.7))",
    "(sym affine \"sigmoid(col(0)) + 0.5*col(1) - log1p(abs(col(2))) + max(c3, c4^2)\")",
    "(sym raw \"clip(row_mean + 0.3*row_std - c0/(c1 + 5), -2, 2)\")",
    "(pipe X (histgbm2))",
    "(pipe X (standardize append=true) (histgbm2 leaves=7 lr=0.1 bags=2))",
    "(pipe X (standardize) (krr))",
    "(pipe X (krr m=300))",
    "(pipe X (gpr))",
    "(pipe X (standardize) (gpr m=300 steps=20))",
    "(pipe X (rff_cos n=16 seed=3 gamma=1.0 append=true) (gpr steps=25))",
    "(pipe X (gpr m=50 steps=15))",
    "(pipe X (gpr steps=6))",
    "(pipe X (histgbm2 leaves=63 rounds=220 lr=0.05 bags=2 row_subsample=0.8 col_subsample=0.7 min_leaf=5 val_pct=15 val_shuffle=true))",
    "(pipe X (histgbm2 leaves=15 lr=0.1 bags=2 rounds=150 loss=huber delta=0.5 stop=mae))",
    "(pipe X (gpr kernel=rq m=200 steps=12))",
    "(pipe X (gpr kernel=rbf+linear m=200 steps=12))",
    "(pipe X (gpr kernel=rbf*periodic steps=10))",
    "(pipe X (standardize) (gpr kernel=rq+linear m=120 steps=10))",
    "(pipe X (winsor_clip q=0.02 append=true) (gpr kernel=linear steps=15))",
    "(pipe X (ksmooth))",
    "(pipe X (robust_scale) (ksmooth k=12 gamma=2.0))",
    "(pipe X (ridge_gcv))",
    "(pipe X (standardize) (cheb degree=3 append=true) (ridge_gcv))",
    "(pipe X (rff_cos n=64 seed=3 gamma=1.0 append=true) (ridge_gcv))",
    "(pipe X (rff_cos n=200 seed=3 gamma=1.0 append=true) (rff_sin n=200 seed=5 gamma=1.0 append=true) (ridge_gcv))",
    "(pipe X (stack))",
    "(pipe X (winsor_clip q=0.02 append=true) (stack))",
    "(pipe X (standardize) (step t=0.0 append=true) (ridge))",
    "(pipe X (map expr=\"step(col(0), 0.0) + 0.5*step(row_mean, c1)\" append=true) (ridge))",
    "(sym affine \"step(c0, 0.0) + step(c1, c2) - 0.5\")",
    "(pipe X (standardize append=true) (step t=0.0 append=true) (standardize append=true) (sigmoid append=true) (step t=0.5 append=true) (ridge))",
    "(pipe X (standardize append=true) (step t=0.0 append=true) (standardize append=true) (sigmoid append=true) (step t=0.5 append=true) (poly_interact append=true) (ridge))",
    "(pipe X (standardize) (ridge))",
    "(pipe X (robust_scale) (rff_cos n=32 seed=7 gamma=1.0 append=true) (ensemble rounds=25))",
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
    // Degenerate / structural corners.
    "(pipe X (ridge))",
    "(pipe X (logistic))",
    "(pipe X (standardize))",
    "(pipe X (hinge knots=3) (hinge knots=9) (ridge))",
    "(pipe X (winsor_clip q=0.00001) (winsor_clip q=0.2 append=true) (ridge))",
    "(pipe X (step t=-0.75 append=true) (step t=0.00001) (ridge))",
    "(pipe X (map expr=\"x\") (map expr=\"x/2\") (derive expr=\"x\") (ridge))",
    "(sym raw \"x\")",
    "(sym affine \"1.5\")",
];

fn assert_exact(src: &str, opts: &EmitOptions) -> (usize, usize) {
    let genome = GenomeIr::parse(src).unwrap_or_else(|e| panic!("parse failed: {src}: {e}"));
    let emitted = emit(&genome, opts).unwrap_or_else(|e| panic!("emit failed: {src}: {e}"));
    let est = try_estimate_tokens(&genome, opts)
        .unwrap_or_else(|e| panic!("estimate failed: {src}: {e:?}"));
    let real_py = emitted.py_tokens.code_tokens + emitted.py_tokens.doc_tokens;
    let real_rs = emitted.rs_tokens.code_tokens + emitted.rs_tokens.doc_tokens;
    assert_eq!(
        (est.py, est.rs),
        (real_py, real_rs),
        "estimate mismatch for {src}: est=({}, {}) real=({real_py}, {real_rs}) \
         diff=({}, {})",
        est.py,
        est.rs,
        est.py as i64 - real_py as i64,
        est.rs as i64 - real_rs as i64,
    );
    (real_py, real_rs)
}

#[test]
fn estimator_exact_on_golden_fixtures() {
    let opts = EmitOptions::default();
    let mut total = 0usize;
    for src in GOLDEN_GENOMES {
        let (py, rs) = assert_exact(src, &opts);
        total += py + rs;
    }
    println!(
        "estimator exact on {} golden fixtures ({} total tokens predicted, 0 diff)",
        GOLDEN_GENOMES.len(),
        total
    );
}

#[test]
fn estimator_exact_with_custom_options() {
    let opts = EmitOptions {
        title: Some("Toy regression artifact".to_string()),
        feature_names: vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()],
        context: Some("Deterministic toy data for parity checks.".to_string()),
        ..Default::default()
    };
    for src in [
        "(pipe X (robust_scale) (rff_cos n=32 seed=7 gamma=1.0 append=true) (ensemble rounds=25))",
        "(pipe X (standardize) (histgbm2 loss=huber delta=0.25 stop=mae))",
        "(sym affine \"sigmoid(c0) - 0.5*c1\")",
    ] {
        assert_exact(src, &opts);
    }
}

// ------------------------- random genome generation -------------------------

const OPS: &[&str] = &[
    "standardize",
    "robust_scale",
    "rank",
    "blom",
    "quantile_bins",
    "bin_onehot",
    "hinge",
    "winsor_clip",
    "cheb",
    "poly_interact",
    "rff_cos",
    "rff_sin",
    "rand_proj",
    "landmark_dist",
    "oblique_hinge",
    "row_stats",
    "pair_prod",
    "target_bin_encode",
    "sigmoid",
    "step",
    "norm_row",
    "polar",
    "lag",
    "roll",
    "fft_mag",
    "row_spectral",
    "row_diff_stats",
    "svd_project",
    "nystrom",
    "map",
    "derive",
];

const EXPRS: &[&str] = &[
    "x",
    "x/2",
    "x^2 - 0.5",
    "sigmoid(x) + 0.1*col(0)",
    "row_mean + row_std - c1/(c0 + 3)",
    "clip(x, -1, 1)",
    "step(col(2), 0.25) * log1p(abs(x))",
    "min(x, row_max) + max(x, row_min)",
    "sqrt(abs(c0)) + sin(c1) - cos(c2)",
    "row_l1 + row_l2 - row_sum",
    "tanh(x) * 0.00001",
];

const F64_POOL: &[&str] = &["0.5", "1.5", "0.02", "0.00001", "12.0", "2.75"];

fn pick<'a>(rng: &mut Pcg32, list: &[&'a str]) -> &'a str {
    list[(rng.next_u32() as usize) % list.len()]
}

fn random_stage(rng: &mut Pcg32) -> String {
    let op = pick(rng, OPS);
    let mut s = format!("({op}");
    match op {
        "map" | "derive" => {
            s.push_str(&format!(" expr=\"{}\"", pick(rng, EXPRS)));
        }
        "quantile_bins" | "bin_onehot" => {
            s.push_str(&format!(" bins={}", 2 + rng.next_u32() % 12));
        }
        "hinge" => s.push_str(&format!(" knots={}", 1 + rng.next_u32() % 8)),
        "winsor_clip" => s.push_str(&format!(" q={}", pick(rng, &["0.02", "0.1", "0.00001"]))),
        "cheb" => s.push_str(&format!(" degree={}", 1 + rng.next_u32() % 6)),
        "rff_cos" | "rff_sin" | "oblique_hinge" => s.push_str(&format!(
            " n={} gamma={} seed={}",
            1 + rng.next_u32() % 48,
            pick(rng, F64_POOL),
            rng.next_u32() % 100
        )),
        "rand_proj" | "svd_project" => s.push_str(&format!(
            " k={} seed={}",
            1 + rng.next_u32() % 16,
            rng.next_u32() % 100
        )),
        "landmark_dist" | "nystrom" => s.push_str(&format!(
            " k={} gamma={} seed={}",
            1 + rng.next_u32() % 16,
            pick(rng, F64_POOL),
            rng.next_u32() % 100
        )),
        "pair_prod" => s.push_str(&format!(" topk={}", 2 + rng.next_u32() % 10)),
        "target_bin_encode" => s.push_str(&format!(
            " bins={} prior={}",
            2 + rng.next_u32() % 12,
            pick(rng, F64_POOL)
        )),
        "step" => s.push_str(&format!(
            " t={}",
            pick(rng, &["0.0", "-0.75", "0.5", "0.00001"])
        )),
        "lag" | "roll" | "fft_mag" => s.push_str(&format!(" k={}", 1 + rng.next_u32() % 6)),
        _ => {}
    }
    if rng.next_u32().is_multiple_of(2) {
        s.push_str(" append=true");
    }
    s.push(')');
    s
}

fn random_head(rng: &mut Pcg32) -> String {
    match rng.next_u32() % 12 {
        0 => format!("(ridge lambda_frac={})", pick(rng, F64_POOL)),
        1 => format!(
            "(elasticnet alpha={} l1_ratio={} sweeps={})",
            pick(rng, F64_POOL),
            pick(rng, &["0.3", "0.00001", "0.9"]),
            10 + rng.next_u32() % 90
        ),
        2 => format!(
            "(histgbm rounds={} depth={} lr={} lambda={})",
            5 + rng.next_u32() % 100,
            1 + rng.next_u32() % 6,
            pick(rng, F64_POOL),
            pick(rng, F64_POOL)
        ),
        3 => format!(
            "(ensemble rounds={} lambda_frac={} lr={})",
            5 + rng.next_u32() % 60,
            pick(rng, F64_POOL),
            pick(rng, F64_POOL)
        ),
        4 => {
            let mut h = format!(
                "(histgbm2 leaves={} lr={} bags={}",
                2 + rng.next_u32() % 60,
                pick(rng, F64_POOL),
                1 + rng.next_u32() % 3
            );
            if rng.next_u32().is_multiple_of(2) {
                h.push_str(&format!(
                    " loss=huber delta={}",
                    pick(rng, &["0.5", "0.00001"])
                ));
            }
            if rng.next_u32().is_multiple_of(2) {
                h.push_str(" stop=mae val_shuffle=true");
            }
            h.push(')');
            h
        }
        5 => format!("(krr m={})", 50 + rng.next_u32() % 500),
        6 => format!(
            "(gpr m={} steps={})",
            50 + rng.next_u32() % 400,
            1 + rng.next_u32() % 40
        ),
        7 => format!(
            "(gpr kernel={} m={} steps={})",
            pick(
                rng,
                &[
                    "rq",
                    "linear",
                    "periodic",
                    "rbf+linear",
                    "rq+linear",
                    "rbf*periodic",
                    "rq+periodic+linear"
                ]
            ),
            50 + rng.next_u32() % 400,
            1 + rng.next_u32() % 30
        ),
        8 => format!(
            "(ksmooth k={} gamma={})",
            1 + rng.next_u32() % 64,
            pick(rng, &["1.0", "2.0", "0.00001"])
        ),
        9 => "(stack)".to_string(),
        10 => "(ridge_gcv)".to_string(),
        _ => String::new(), // no explicit head: defaults to ridge
    }
}

fn random_genome(rng: &mut Pcg32) -> String {
    if rng.next_u32().is_multiple_of(10) {
        let kind = if rng.next_u32().is_multiple_of(2) {
            "affine"
        } else {
            "raw"
        };
        return format!("(sym {kind} \"{}\")", pick(rng, EXPRS));
    }
    let n_stages = (rng.next_u32() % 7) as usize;
    let mut g = String::from("(pipe X");
    for _ in 0..n_stages {
        g.push(' ');
        g.push_str(&random_stage(rng));
    }
    let mut head = random_head(rng);
    if head.is_empty() && n_stages == 0 {
        // "(pipe X)" itself is not parseable; keep the implicit-ridge case
        // only when at least one stage exists.
        head = "(ridge)".to_string();
    }
    if !head.is_empty() {
        g.push(' ');
        g.push_str(&head);
    }
    g.push(')');
    g
}

#[test]
fn estimator_exact_on_200_random_genomes() {
    let opts = EmitOptions::default();
    let mut rng = Pcg32::new(0x7061727469616c, 99);
    let mut checked = 0usize;
    let mut total_tokens = 0usize;
    while checked < 200 {
        let src = random_genome(&mut rng);
        if GenomeIr::parse(&src).is_err() {
            panic!("random generator produced unparseable genome: {src}");
        }
        let (py, rs) = assert_exact(&src, &opts);
        total_tokens += py + rs;
        checked += 1;
    }
    println!(
        "estimator exact on {checked} random genomes ({total_tokens} total tokens predicted, 0 diff)"
    );
}

#[test]
fn estimator_is_fast_and_reports_cost() {
    // Representative 8-stage genome from the emitter benchmark.
    let src = "(pipe X (standardize append=true) (rff_cos n=32 seed=7 gamma=1.0 append=true) \
               (cheb degree=3 append=true) (rank append=true) (winsor_clip q=0.02 append=true) \
               (poly_interact append=true) (row_stats append=true) (ridge))";
    let genome = GenomeIr::parse(src).unwrap();
    let opts = EmitOptions::default();

    // Force one-time calibration outside the timed loop, and check agreement.
    let est0 = estimate_tokens(&genome);
    let emitted = emit(&genome, &opts).unwrap();
    assert_eq!(
        TokenEstimate {
            py: emitted.py_tokens.code_tokens + emitted.py_tokens.doc_tokens,
            rs: emitted.rs_tokens.code_tokens + emitted.rs_tokens.doc_tokens,
        },
        est0
    );

    let iters = 2000usize;
    let mut acc = 0usize;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        acc += estimate_tokens(&genome).py;
    }
    let est_ns = t0.elapsed().as_nanos() / iters as u128;
    let t1 = std::time::Instant::now();
    for _ in 0..200 {
        let e = emit(&genome, &opts).unwrap();
        acc += e.py_tokens.code_tokens;
    }
    let emit_ns = t1.elapsed().as_nanos() / 200;
    println!(
        "ESTIMATOR-BENCH stages=8 estimate_ns={est_ns} emit_ns={emit_ns} \
         speedup={:.1}x (acc={acc})",
        emit_ns as f64 / est_ns.max(1) as f64
    );
    assert!(
        est_ns * 5 < emit_ns,
        "estimator should be >5x faster than emission: {est_ns}ns vs {emit_ns}ns"
    );
}
