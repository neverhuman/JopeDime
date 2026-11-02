use super::*;
use std::fs;
use std::path::Path;
use std::process::Command;

fn rustc(directory: &Path, args: &[&str]) {
    let output = Command::new("rustc")
        .current_dir(directory)
        .args([
            "--edition=2021",
            "-F",
            "unsafe_code",
            "-C",
            "codegen-units=1",
            "-C",
            "panic=abort",
            "-C",
            "link-arg=-Wl,--threads=1",
        ])
        .args(args)
        .output()
        .expect("run rustc");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn predictions(directory: &Path, binary: &str, grant: &str) -> Vec<u64> {
    let output = Command::new(directory.join(binary))
        .current_dir(directory)
        .args(["train.csv", "hold.csv", "predictions.csv"])
        .env("JOPE_CPU_THREADS", grant)
        .output()
        .expect("run recipe");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::read_to_string(directory.join("predictions.csv"))
        .unwrap()
        .lines()
        .map(|line| line.parse::<f64>().unwrap().to_bits())
        .collect()
}

#[test]
fn runtime_is_plan_independent_and_unsupported_ir_is_explicit() {
    let source = operator_runtime_source();
    assert!(source.contains("#![forbid(unsafe_code)]"));
    assert!(!source.contains("struct Model"));
    assert!(!source.contains("fn main()"));
    assert!(!source.contains("#[inline]"));
    let plan = Plan::Symbolic {
        source: "1".into(),
        expr: crate::mathexpr::MathAst::Const(1.0),
        affine: false,
    };
    let opts = EmitOptions {
        hpc_v2: true,
        partition_point_quantile_bounds: true,
        ..EmitOptions::default()
    };
    assert!(emit_glue(&plan, &opts).unwrap_err().contains("symbolic"));

    let valid_genome = crate::GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap();
    let valid_plan = crate::spec::build_plan(&valid_genome).unwrap();
    let no_pp_opts = EmitOptions {
        hpc_v2: true,
        partition_point_quantile_bounds: false,
        ..EmitOptions::default()
    };
    let no_pp_err = emit_glue(&valid_plan, &no_pp_opts).unwrap_err();
    assert!(
        no_pp_err.contains("partition_point_quantile_bounds"),
        "unexpected error: {no_pp_err}"
    );
    let portable_opts = EmitOptions {
        hpc_v2: true,
        partition_point_quantile_bounds: true,
        math_kernel: crate::transpile::MathKernel::Portable,
        ..EmitOptions::default()
    };
    let portable_err = emit_glue(&valid_plan, &portable_opts).unwrap_err();
    assert!(
        portable_err.contains("portable math emission is not available"),
        "unexpected error: {portable_err}"
    );
    let default_glue = emit_glue(&valid_plan, &opts).unwrap();
    assert!(default_glue.contains("impl jope_recipe_ops::RecipeModel for Model"));
}

#[test]
fn one_runtime_and_structured_unseen_glue_match_whole_recipes_bit_for_bit() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target/recipe_bundle_tests");
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("lib.rs"), operator_runtime_source()).unwrap();
    rustc(
        &directory,
        &[
            "-C",
            "opt-level=3",
            "--crate-name",
            "jope_recipe_ops",
            "--crate-type=rlib",
            "lib.rs",
            "-o",
            "libjope_recipe_ops.rlib",
        ],
    );
    let train = (0..24)
        .map(|i| {
            let a = (i as f64 * 0.17).sin();
            let b = (i as f64 * 0.31).cos();
            format!(
                "{a:?},{b:?},{:?},{:?}\n",
                (i % 5) as f64 / 3.0,
                a * 0.4 - b * 0.2
            )
        })
        .collect::<String>();
    let hold = (0..70)
        .map(|i| {
            format!(
                "{:?},{:?},{:?}\n",
                (i as f64 * 0.23).sin(),
                (i as f64 * 0.19).cos(),
                (i % 7) as f64 / 4.0
            )
        })
        .collect::<String>();
    fs::write(directory.join("train.csv"), train).unwrap();
    fs::write(directory.join("hold.csv"), hold).unwrap();
    let programs = [
            "(pipe X (standardize append=true) (robust_scale) (rank) (blom) (quantile_bins bins=4) (bin_onehot bins=3) (ridge lambda_frac=0.03125))",
            "(pipe X (hinge append=true knots=3) (cheb degree=2) (poly_interact) (ridge_gcv))",
            "(pipe X (rff_cos n=7 gamma=1.3 seed=41) (rff_sin n=5 gamma=0.7 seed=43) (rand_proj k=4 seed=47) (landmark_dist k=3 gamma=0.8 seed=53) (oblique_hinge n=5 seed=59) (histgbm rounds=4 depth=2))",
            "(pipe X (row_stats append=true) (pair_prod topk=3) (target_bin_encode bins=4 prior=2) (sigmoid) (elasticnet alpha=0.01 l1_ratio=0.4 sweeps=5))",
            "(pipe X (symlog c=0.7) (select_corr k=2) (impute strategy=median add_indicator=true) (pair_ratio topk=3) (spline knots=3) (ensemble rounds=3 depth=2))",
            "(pipe X (step t=0.2 append=true) (norm_row) (polar) (lag k=1) (roll k=1) (fft_mag k=2) (row_spectral append=true) (row_diff_stats append=true) (ksmooth k=4 gamma=0.75))",
            "(pipe X (svd_project k=2 iters=3 seed=61) (nystrom k=4 gamma=0.6 seed=67) (krr m=8))",
            "(pipe X (winsor_clip q=0.1) (gpr m=8 steps=2))",
            "(pipe X (standardize) (gpr m=8 steps=2 kernel=\"rq+linear\"))",
            "(pipe X (histgbm2 rounds=3 leaves=4 bags=1))",
            "(pipe X (stack))",
            "(pipe X (target kind=zscore) (ridge))",
            "(pipe X (target kind=rank_gauss) (ridge))",
            "(pipe X (target kind=log1p) (ridge))",
            "(pipe X (target kind=winsor q=0.1) (ridge))",
        ];
    let opts = EmitOptions {
        hpc_v1: true,
        hpc_v2: true,
        fast_runtime: true,
        partition_point_quantile_bounds: true,
        guard_zero_width_roll: true,
        ..EmitOptions::default()
    };
    for (index, dsl) in programs.iter().enumerate() {
        let genome = crate::GenomeIr::parse(dsl).unwrap_or_else(|err| panic!("{dsl}: {err}"));
        let plan = crate::spec::build_plan(&genome).unwrap_or_else(|err| panic!("{dsl}: {err:?}"));
        let glue = emit_glue(&plan, &opts).unwrap();
        assert!(!glue.contains("use jope_recipe_ops::*"));
        fs::write(directory.join("whole.rs"), emit_rust(&plan, &opts)).unwrap();
        fs::write(directory.join("glue.rs"), glue).unwrap();
        rustc(
            &directory,
            &["-C", "opt-level=3", "whole.rs", "-o", "whole"],
        );
        rustc(
            &directory,
            &[
                "-C",
                "opt-level=0",
                "glue.rs",
                "--extern",
                "jope_recipe_ops=libjope_recipe_ops.rlib",
                "-o",
                "glue",
            ],
        );
        for grant in ["1", "3"] {
            assert_eq!(
                predictions(&directory, "whole", grant),
                predictions(&directory, "glue", grant),
                "program {index}, grant {grant}: {dsl}"
            );
        }
    }

    // Wide-512 G-6 + G-8 + A09 parity against hpc_v1 and reference Model::fit
    let mut wide_train = String::new();
    let mut x_train_ref = Vec::new();
    let mut y_train_ref = Vec::new();
    for i in 0..48 {
        let mut row = Vec::with_capacity(34);
        for j in 0..34 {
            row.push(((i * 7 + j * 13) as f64 * 0.11).sin() + (j as f64) * 0.03);
        }
        let y = row[0] * 0.5 - row[1] * 0.3 + row[2] * row[3] * 0.2;
        let mut line = row
            .iter()
            .map(|v| format!("{v:?}"))
            .collect::<Vec<_>>()
            .join(",");
        line.push_str(&format!(",{y:?}\n"));
        wide_train.push_str(&line);
        x_train_ref.push(row);
        y_train_ref.push(y);
    }
    let mut wide_hold = String::new();
    let mut x_hold_ref = Vec::new();
    for i in 0..32 {
        let mut row = Vec::with_capacity(34);
        for j in 0..34 {
            row.push(((i * 11 + j * 5) as f64 * 0.13).cos() - (j as f64) * 0.02);
        }
        let line = row
            .iter()
            .map(|v| format!("{v:?}"))
            .collect::<Vec<_>>()
            .join(",")
            + "\n";
        wide_hold.push_str(&line);
        x_hold_ref.push(row);
    }
    fs::write(directory.join("train.csv"), wide_train).unwrap();
    fs::write(directory.join("hold.csv"), wide_hold).unwrap();
    let wide_programs = [
            "(pipe X (poly_interact) (hinge knots=3 append=true) (quantile_bins bins=6 append=true) (cheb degree=2 append=true) (ridge_gcv))",
            "(pipe X (poly_interact) (poly_interact append=true) (hinge knots=3 append=true) (quantile_bins bins=6 append=true) (cheb degree=2 append=true) (ridge_gcv))",
            "(pipe X (poly_interact) (hinge knots=3 append=true) (quantile_bins bins=6 append=true) (cheb degree=2 append=true) (histgbm rounds=6 depth=3))",
        ];
    let opts_leg = EmitOptions {
        hpc_v1: false,
        hpc_v2: false,
        fast_runtime: false,
        partition_point_quantile_bounds: true,
        guard_zero_width_roll: true,
        ..EmitOptions::default()
    };
    let opts_v1 = EmitOptions {
        hpc_v1: true,
        hpc_v2: false,
        fast_runtime: true,
        partition_point_quantile_bounds: true,
        guard_zero_width_roll: true,
        ..EmitOptions::default()
    };
    for dsl in wide_programs {
        let genome = crate::GenomeIr::parse(dsl).unwrap();
        let model_ref = crate::model::fit(&genome, &x_train_ref, &y_train_ref).unwrap();
        let ref_bits: Vec<u64> = model_ref
            .predict(&x_hold_ref)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect();
        let plan = crate::spec::build_plan(&genome).unwrap();
        let glue = emit_glue(&plan, &opts).unwrap();
        fs::write(directory.join("whole_leg.rs"), emit_rust(&plan, &opts_leg)).unwrap();
        fs::write(directory.join("whole_v1.rs"), emit_rust(&plan, &opts_v1)).unwrap();
        fs::write(directory.join("whole_v2.rs"), emit_rust(&plan, &opts)).unwrap();
        fs::write(directory.join("glue.rs"), glue).unwrap();
        rustc(
            &directory,
            &["-C", "opt-level=3", "whole_leg.rs", "-o", "whole_leg"],
        );
        rustc(
            &directory,
            &["-C", "opt-level=3", "whole_v1.rs", "-o", "whole_v1"],
        );
        rustc(
            &directory,
            &["-C", "opt-level=3", "whole_v2.rs", "-o", "whole_v2"],
        );
        rustc(
            &directory,
            &[
                "-C",
                "opt-level=0",
                "glue.rs",
                "--extern",
                "jope_recipe_ops=libjope_recipe_ops.rlib",
                "-o",
                "glue",
            ],
        );
        let leg_bits = predictions(&directory, "whole_leg", "1");
        assert_eq!(leg_bits, ref_bits, "legacy vs ref on {dsl}");
        let v1_bits = predictions(&directory, "whole_v1", "1");
        assert_eq!(v1_bits, ref_bits, "v1 vs ref on {dsl}");
        for grant in ["1", "3"] {
            assert_eq!(
                predictions(&directory, "whole_v2", grant),
                ref_bits,
                "v2 whole vs ref (grant {grant}) on {dsl}"
            );
            assert_eq!(
                predictions(&directory, "glue", grant),
                ref_bits,
                "v2 glue vs ref (grant {grant}) on {dsl}"
            );
        }
    }

    // Empty holdout (0 prediction rows) under legacy, hpc_v1, hpc_v2 whole, and hpc_v2 glue (R5)
    fs::write(directory.join("hold.csv"), "").unwrap();
    for bin in ["whole_leg", "whole_v1", "whole_v2", "glue"] {
        assert!(
            predictions(&directory, bin, "1").is_empty(),
            "empty holdout must produce 0 predictions without panicking for {bin}"
        );
    }

    // Partial pre-cap path (0 < cap < generated width): 31 input cols -> poly_interact produces
    // 31 + 31*30/2 = 496 cols, leaving cap = 512 - 496 = 16 cols for the next append=true stage.
    let mut partial_train = String::new();
    let mut x_partial_train = Vec::new();
    let mut y_partial_train = Vec::new();
    for i in 0..32 {
        let row: Vec<f64> = (0..31)
            .map(|j| ((i * 9 + j * 7) as f64 * 0.13).sin() + (j as f64) * 0.02)
            .collect();
        let y = row[0] * 0.4 - row[1] * 0.25 + row[2] * 0.15;
        let mut line = row
            .iter()
            .map(|v| format!("{v:?}"))
            .collect::<Vec<_>>()
            .join(",");
        line.push_str(&format!(",{y:?}\n"));
        partial_train.push_str(&line);
        x_partial_train.push(row);
        y_partial_train.push(y);
    }
    let mut partial_hold = String::new();
    let mut x_partial_hold = Vec::new();
    for i in 0..16 {
        let row: Vec<f64> = (0..31)
            .map(|j| ((i * 5 + j * 11) as f64 * 0.17).cos() - (j as f64) * 0.01)
            .collect();
        let line = row
            .iter()
            .map(|v| format!("{v:?}"))
            .collect::<Vec<_>>()
            .join(",")
            + "\n";
        partial_hold.push_str(&line);
        x_partial_hold.push(row);
    }
    fs::write(directory.join("train.csv"), partial_train).unwrap();
    fs::write(directory.join("hold.csv"), partial_hold).unwrap();
    for stage in [
        "poly_interact append=true",
        "hinge knots=3 append=true",
        "cheb degree=2 append=true",
        "quantile_bins bins=6 append=true",
    ] {
        let dsl = format!("(pipe X (poly_interact) ({stage}) (ridge))");
        let genome = crate::GenomeIr::parse(&dsl).unwrap();
        let model_ref = crate::model::fit(&genome, &x_partial_train, &y_partial_train).unwrap();
        let ref_bits: Vec<u64> = model_ref
            .predict(&x_partial_hold)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect();
        let plan = crate::spec::build_plan(&genome).unwrap();
        fs::write(directory.join("p_leg.rs"), emit_rust(&plan, &opts_leg)).unwrap();
        fs::write(directory.join("p_v1.rs"), emit_rust(&plan, &opts_v1)).unwrap();
        fs::write(directory.join("p_v2.rs"), emit_rust(&plan, &opts)).unwrap();
        rustc(
            &directory,
            &["-C", "opt-level=2", "p_leg.rs", "-o", "p_leg"],
        );
        rustc(&directory, &["-C", "opt-level=2", "p_v1.rs", "-o", "p_v1"]);
        rustc(&directory, &["-C", "opt-level=2", "p_v2.rs", "-o", "p_v2"]);
        assert_eq!(
            predictions(&directory, "p_leg", "1"),
            ref_bits,
            "partial leg: {dsl}"
        );
        assert_eq!(
            predictions(&directory, "p_v1", "1"),
            ref_bits,
            "partial v1: {dsl}"
        );
        assert_eq!(
            predictions(&directory, "p_v2", "1"),
            ref_bits,
            "partial v2: {dsl}"
        );
    }

    // Zero-width input rows for hinge with append=false and append=true under hpc_v2 vs legacy (R4)
    let zero_w_train = "1.0\n2.0\n3.0\n4.0\n";
    let zero_w_hold = "0.0\n0.0\n0.0\n";
    fs::write(directory.join("train.csv"), zero_w_train).unwrap();
    fs::write(directory.join("hold.csv"), zero_w_hold).unwrap();
    for dsl in [
        "(pipe X (hinge knots=3) (ridge))",
        "(pipe X (hinge knots=3 append=true) (ridge))",
    ] {
        let genome = crate::GenomeIr::parse(dsl).unwrap();
        let plan = crate::spec::build_plan(&genome).unwrap();
        fs::write(directory.join("zw_leg.rs"), emit_rust(&plan, &opts_leg)).unwrap();
        fs::write(directory.join("zw_v2.rs"), emit_rust(&plan, &opts)).unwrap();
        rustc(
            &directory,
            &["-C", "opt-level=1", "zw_leg.rs", "-o", "zw_leg"],
        );
        rustc(
            &directory,
            &["-C", "opt-level=1", "zw_v2.rs", "-o", "zw_v2"],
        );
        let leg_zw = predictions(&directory, "zw_leg", "1");
        let v2_zw = predictions(&directory, "zw_v2", "1");
        assert_eq!(leg_zw.len(), 3);
        assert_eq!(v2_zw, leg_zw, "zero-width hinge parity on {dsl}");
    }
}
