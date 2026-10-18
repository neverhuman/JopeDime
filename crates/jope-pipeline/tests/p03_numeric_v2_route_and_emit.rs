//! MR44 (G01, G02 / SPEC_A P03 / GP-03 / WP02): Numeric-v2 route completion
//! (`numeric_v2`, `numeric_v2_emit`, pipeline/CLI kernel adapters; supported caller/target
//! subset executes with old default unchanged).

use clap::Parser;
use feat_math::kernel::profile::shipped_kernel2_profiles;
use feat_math::numeric_v2::NumericProfileV2;
use feat_math::numeric_v2_emit::{OptionsV2, TargetV2};
use feat_math::transpile::EmitOptions;
use jope_pipeline::cli::{Cli, Cmd};
use jope_pipeline::emit::{self, Language};
use jope_pipeline::kernel2::Kernel2Evaluator;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn scratch() -> tempfile::TempDir {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}

#[test]
fn p03_numeric_v2_and_numeric_v2_emit_execute_on_supported_subset() {
    let dir = scratch();
    let schema = vec!["c0".to_string(), "c1".to_string()];
    let profile = NumericProfileV2::new(schema.clone());
    let x = vec![
        vec![0.0, 1.0],
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 4.0],
        vec![4.0, 5.0],
    ];
    let y = vec![-500.0, -250.0, 0.0, 250.0, 500.0];

    for (idx, recipe) in [
        "(pipe X (ridge))",
        "(pipe X (impute) (poly_interact topk=2 append=true) (ridge))",
        "(pipe X (quantile_bins bins=4) (ridge))",
        "(pipe X (bin_onehot bins=4) (ridge))",
    ]
    .iter()
    .enumerate()
    {
        for target in [TargetV2::Python, TargetV2::Rust] {
            let sub = dir.path().join(format!("{idx}-{target:?}"));
            let opts = OptionsV2::new(target);
            let art =
                emit::emit_numeric_v2(recipe, profile.clone(), &x, &y, opts.clone(), &sub).unwrap();
            let reloaded = emit::reload_numeric_v2(&sub, &opts, &art.model_sha256).unwrap();
            let want = reloaded.predict(&schema, &x).unwrap();
            assert!(want[0] < -100.0 && want[4] > 100.0);

            let src_path = sub.join(if target == TargetV2::Python {
                "model.py"
            } else {
                "model.rs"
            });
            let out = if target == TargetV2::Python {
                Command::new("python3")
                    .arg("-c")
                    .arg(format!(
                        "import runpy,sys; print(*runpy.run_path(sys.argv[1])['predict']({}),sep='\\n')",
                        serde_json::to_string(&x).unwrap()
                    ))
                    .arg(&src_path)
                    .output()
                    .unwrap()
            } else {
                let driver = sub.join("driver.rs");
                let rows_lit = x
                    .iter()
                    .map(|r| format!("vec!{r:?}"))
                    .collect::<Vec<_>>()
                    .join(",");
                fs::write(
                    &driver,
                    format!(
                        "include!({src_path:?}); fn main() {{ for v in predict(&[{rows_lit}]).unwrap() {{ println!(\"{{:.17}}\", v); }} }}"
                    ),
                )
                .unwrap();
                let bin = sub.join("driver");
                let comp = Command::new("rustc")
                    .args(["--edition=2021", "-Awarnings"])
                    .arg(&driver)
                    .arg("-o")
                    .arg(&bin)
                    .output()
                    .unwrap();
                assert!(
                    comp.status.success(),
                    "{}",
                    String::from_utf8_lossy(&comp.stderr)
                );
                Command::new(&bin).output().unwrap()
            };
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            let got: Vec<f64> = String::from_utf8(out.stdout)
                .unwrap()
                .lines()
                .map(|s| s.parse().unwrap())
                .collect();
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn pipeline_and_cli_kernel_adapters_wired() {
    let dir = scratch();

    // 1. CLI default route parses as `legacy` (`Ok(None)` kernel2 options), leaving old default unchanged.
    let cli_default = Cli::try_parse_from([
        "jope-pipeline",
        "run",
        "--dataset",
        "0123456789abcdef",
        "--output",
        dir.path().to_str().unwrap(),
    ])
    .unwrap();
    let Cmd::Run {
        search: default_search,
        profile: default_profile,
        output: default_out,
        ..
    } = cli_default.cmd
    else {
        panic!("expected Cmd::Run");
    };
    assert_eq!(default_profile.search_profile, "legacy");
    assert!(default_profile
        .options(&default_search, default_out, false)
        .unwrap()
        .is_none());

    // 2. CLI opt-in kernel2 adapter (`--search-profile kernel2-baseline`) binds Kernel2Evaluator.
    let cli_k2 = Cli::try_parse_from([
        "jope-pipeline",
        "run",
        "--dataset",
        "0123456789abcdef",
        "--search-profile",
        "kernel2-baseline",
        "--refine-top-k",
        "1",
        "--refine-trials",
        "2",
        "--output",
        dir.path().to_str().unwrap(),
    ])
    .unwrap();
    let Cmd::Run {
        search: k2_search,
        profile: k2_profile,
        output: k2_out,
        ..
    } = cli_k2.cmd
    else {
        panic!("expected Cmd::Run");
    };
    let k2_opts = k2_profile
        .options(&k2_search, k2_out, false)
        .unwrap()
        .expect("kernel2 options present");
    assert!(k2_opts.run.languages.contains(&Language::Cpp17Fitted));
    let baseline = shipped_kernel2_profiles()
        .into_iter()
        .find(|p| p.id == "kernel2-baseline")
        .unwrap();
    let eval = Kernel2Evaluator::new(baseline).unwrap();
    assert_eq!(eval.profile().revision, 3);

    // 3. Pipeline numeric_v2 emit + reload adapter and tamper rejection.
    let x = vec![
        vec![0.25, -0.75],
        vec![0.50, -0.50],
        vec![0.75, -0.25],
        vec![1.00, 0.00],
    ];
    let y = vec![-200.0, -50.0, 100.0, 250.0];
    let schema = vec!["f0".to_string(), "f1".to_string()];
    let opts = OptionsV2::new(TargetV2::Rust);
    let v2_dir = dir.path().join("v2_adapter");
    let artifact = emit::emit_numeric_v2(
        "(pipe X (ridge))",
        NumericProfileV2::new(schema.clone()),
        &x,
        &y,
        opts.clone(),
        &v2_dir,
    )
    .unwrap();
    let reloaded = emit::reload_numeric_v2(&v2_dir, &opts, &artifact.model_sha256).unwrap();
    let v2_preds = reloaded.predict(&schema, &x).unwrap();
    assert!(v2_preds[0] < -100.0 && v2_preds[3] > 100.0);
    assert!(emit::reload_numeric_v2(
        &v2_dir,
        &OptionsV2::new(TargetV2::Python),
        &artifact.model_sha256
    )
    .is_err());
    assert!(emit::reload_numeric_v2(&v2_dir, &opts, &"0".repeat(64)).is_err());
}

#[test]
fn old_default_route_outputs_unchanged() {
    let dir = scratch();
    let genome = "(pipe X (quantile_bins bins=4) (ridge))";
    let default_dir = dir.path().join("default_emit");
    let explicit_dir = dir.path().join("explicit_default_emit");
    let opt_in_dir = dir.path().join("opt_in_emit");

    emit::emit(genome, &Language::ALL, &default_dir).unwrap();
    emit::emit_with_options(
        genome,
        &Language::ALL,
        &explicit_dir,
        &EmitOptions::default(),
    )
    .unwrap();
    emit::emit_with_options(
        genome,
        &Language::ALL,
        &opt_in_dir,
        &EmitOptions {
            partition_point_quantile_bounds: true,
            ..EmitOptions::default()
        },
    )
    .unwrap();

    for file in [
        "model.py",
        "model.rs",
        "model.cpp",
        "model.js",
        "manifest.json",
    ] {
        assert_eq!(
            fs::read(default_dir.join(file)).unwrap(),
            fs::read(explicit_dir.join(file)).unwrap(),
            "old default route output for {file} must remain byte-for-byte unchanged"
        );
        assert_ne!(
            fs::read(default_dir.join(file)).unwrap(),
            fs::read(opt_in_dir.join(file)).unwrap(),
            "opt-in partition-point route must be distinct from legacy default for {file}"
        );
    }
}

#[test]
fn numeric_v2_and_emit_adapters_execute_with_legacy_default_unchanged() {
    p03_numeric_v2_and_numeric_v2_emit_execute_on_supported_subset();
    pipeline_and_cli_kernel_adapters_wired();
    old_default_route_outputs_unchanged();
}
