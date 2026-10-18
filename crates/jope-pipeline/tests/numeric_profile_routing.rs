use feat_math::{
    numeric_v2::NumericProfileV2,
    numeric_v2_emit::{OptionsV2, TargetV2},
    transpile::EmitOptions,
};
use jope_pipeline::emit::{self, Language};
use std::{fs, process::Command};
fn scratch() -> tempfile::TempDir {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}
#[test]
fn fitted_product_emit_reload_and_execute() {
    let dir = scratch();
    let schema = vec!["c0".into()];
    let p = NumericProfileV2::new(schema.clone());
    let x = vec![vec![0.], vec![1.], vec![2.]];
    let y = vec![-200., 0., 200.];
    for target in [TargetV2::Python, TargetV2::Rust] {
        let root = dir.path().join(format!("{target:?}"));
        let opts = OptionsV2::new(target);
        let a = emit::emit_numeric_v2(
            "(pipe X (quantile_bins bins=4) (ridge))",
            p.clone(),
            &x,
            &y,
            opts.clone(),
            &root,
        )
        .unwrap();
        let m = emit::reload_numeric_v2(&root, &opts, &a.model_sha256).unwrap();
        let want = m.predict(&schema, &x).unwrap();
        assert!(want[2] > 60. && want[0] < -60.);
        assert!(emit::reload_numeric_v2(
            &root,
            &OptionsV2::new(if target == TargetV2::Rust {
                TargetV2::Python
            } else {
                TargetV2::Rust
            }),
            &a.model_sha256
        )
        .is_err());
        let source = root.join(if target == TargetV2::Python {
            "model.py"
        } else {
            "model.rs"
        });
        let out = if target == TargetV2::Python {
            Command::new("python3").arg("-c").arg("import runpy,sys; print(*runpy.run_path(sys.argv[1])['predict']([[0.],[1.],[2.]]),sep='\\n')").arg(&source).output().unwrap()
        } else {
            let driver = root.join("driver.rs");
            fs::write(&driver,format!("include!({source:?}); fn main() {{for v in predict(&[vec![0.],vec![1.],vec![2.]]).unwrap(){{println!(\"{{:.17}}\",v);}}}}")).unwrap();
            let bin = root.join("driver");
            let c = Command::new("rustc")
                .args(["--edition=2021", "-Awarnings"])
                .arg(&driver)
                .arg("-o")
                .arg(&bin)
                .output()
                .unwrap();
            assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
            Command::new(bin).output().unwrap()
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
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() < 1e-10);
        }
        let mut bytes = fs::read(&source).unwrap();
        bytes[0] ^= 1;
        fs::write(&source, &bytes).unwrap();
        assert!(emit::reload_numeric_v2(&root, &opts, &a.model_sha256).is_err());
        fs::write(&source, &a.source).unwrap();
        fs::write(
            root.join("model.codec.json"),
            a.codec.replace("numeric-profile-v2", "numeric-profile-v1"),
        )
        .unwrap();
        assert!(emit::reload_numeric_v2(&root, &opts, &a.model_sha256).is_err());
    }
}
#[test]
fn actual_recipe_options_preserve_default_and_route_quantile_opt_in() {
    let dir = scratch();
    let g = "(pipe X (quantile_bins bins=4) (ridge))";
    let old = dir.path().join("old");
    let explicit = dir.path().join("explicit");
    let new = dir.path().join("new");
    emit::emit(g, &Language::ALL, &old).unwrap();
    emit::emit_with_options(g, &Language::ALL, &explicit, &EmitOptions::default()).unwrap();
    let opts = EmitOptions {
        partition_point_quantile_bounds: true,
        ..EmitOptions::default()
    };
    let result = emit::emit_with_options(g, &Language::ALL, &new, &opts).unwrap();
    assert!(result.exclusions.is_empty());
    assert_eq!(result.artifacts.len(), 5);
    for file in [
        "model.py",
        "model.rs",
        "model.cpp",
        "model.js",
        "manifest.json",
    ] {
        assert_eq!(
            fs::read(old.join(file)).unwrap(),
            fs::read(explicit.join(file)).unwrap()
        );
        assert_ne!(
            fs::read(old.join(file)).unwrap(),
            fs::read(new.join(file)).unwrap()
        );
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(new.join("manifest.json")).unwrap()).unwrap();
    assert!(manifest["emitter_version"]
        .as_str()
        .unwrap()
        .contains("quantile-partition-point"));
    // Hand-derived plateau: edges [0,0,1], equality maps to the first equal edge.
    let out=Command::new("python3").arg("-c").arg("import runpy,sys; m=runpy.run_path(sys.argv[1]); e=m['quantile_bin_indexer_fit']([[0.],[0.],[0.],[1.],[1.]],4); assert e == [[0.,0.,1.]]; p=m['quantile_bin_indexer_apply'](e,[[0.],[1.],[2.]],4,False); assert p == [[0.],[2/3],[1.]],p").arg(new.join("model.py")).output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
#[test]
fn unsupported_options_and_heads_fail_before_export() {
    let dir = scratch();
    let x = vec![vec![1.], vec![2.]];
    let mut opts = OptionsV2::new(TargetV2::Python);
    opts.numeric_profile = "legacy".into();
    assert!(emit::emit_numeric_v2(
        "(pipe X (ridge))",
        NumericProfileV2::new(vec!["a".into()]),
        &x,
        &[1., 2.],
        opts,
        dir.path()
    )
    .is_err());
    assert!(emit::emit_numeric_v2(
        "(pipe X (histgbm))",
        NumericProfileV2::new(vec!["a".into()]),
        &x,
        &[1., 2.],
        OptionsV2::new(TargetV2::Rust),
        dir.path()
    )
    .is_err());
    assert!(!dir.path().join("numeric-manifest.json").exists());
    assert!(serde_json::from_str::<TargetV2>("\"Cpp17\"").is_err());
}

#[test]
fn near_tie_fitted_quantiles_use_independent_oracle_through_product_route() {
    let dir = scratch();
    let a = 0.10000000000000002_f64;
    let b = f64::from_bits(a.to_bits() + 1);
    let x = vec![vec![a], vec![a], vec![a], vec![b], vec![b]];
    let y = [-200., -100., 0., 100., 200.];
    let schema = vec!["value".into()];
    let options = OptionsV2::new(TargetV2::Python);
    let artifact = emit::emit_numeric_v2(
        "(pipe X (quantile_bins bins=13) (ridge))",
        NumericProfileV2::new(schema.clone()),
        &x,
        &y,
        options.clone(),
        dir.path(),
    )
    .unwrap();
    let model = emit::reload_numeric_v2(dir.path(), &options, &artifact.model_sha256).unwrap();
    let feat_math::numeric_v2::TransformV2::Quantile { edges, .. } = &model.transforms[0] else {
        panic!()
    };
    assert!(edges[0].windows(2).all(|w| w[0] <= w[1]));
    let feature = |v: f64| edges[0].iter().filter(|e| **e < v).count() as f64 / 12.0;
    let features: Vec<_> = x.iter().map(|r| feature(r[0])).collect();
    let mean = features.iter().sum::<f64>() / 5.0;
    let scale = (features.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 5.0)
        .sqrt()
        .max(1e-9);
    let coefficient = features
        .iter()
        .zip(y)
        .map(|(v, t)| (v - mean) / scale * t)
        .sum::<f64>()
        / 5.1;
    let probes = vec![vec![a], vec![b], vec![f64::from_bits(b.to_bits() + 1)]];
    let expected: Vec<_> = probes
        .iter()
        .map(|r| (coefficient * (feature(r[0]) - mean) / scale).clamp(-220., 220.))
        .collect();
    for (p, q) in model
        .predict(&schema, &probes)
        .unwrap()
        .iter()
        .zip(&expected)
    {
        assert!((p - q).abs() < 1e-10);
    }
    let driver=format!("import runpy,sys; m=runpy.run_path(sys.argv[1]); print(*m['predict']([[{a:?}],[{b:?}],[{:?}]]),sep='\\n')",probes[2][0]);
    let out = Command::new("python3")
        .arg("-c")
        .arg(driver)
        .arg(dir.path().join("model.py"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let values: Vec<f64> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(values.len(), expected.len());
    for (p, q) in values.iter().zip(expected) {
        assert!((p - q).abs() < 1e-10);
    }
}
