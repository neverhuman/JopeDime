use feat_math::{
    numeric_v2::{NumericModelV2, NumericProfileV2},
    numeric_v2_emit::{self, OptionsV2, TargetV2},
    GenomeIr, Rows,
};
fn profile(n: usize) -> NumericProfileV2 {
    NumericProfileV2::new((0..n).map(|i| format!("c{i}")).collect())
}
fn genome(s: &str) -> GenomeIr {
    GenomeIr::parse(s).unwrap()
}
fn fit(s: &str, x: &Rows, y: &[f64]) -> NumericModelV2 {
    NumericModelV2::fit(profile(x[0].len()), &genome(s), x, y).unwrap()
}

#[test]
fn legacy_reproducers_and_successor_raw_targets() {
    let x: Rows = (0..20).map(|i| vec![i as f64]).collect();
    let y: Vec<f64> = (0..20).map(|i| 200.0 * i as f64 - 1900.0).collect();
    let old = feat_math::fit(&genome("(pipe X (ridge))"), &x, &y)
        .unwrap()
        .predict(&x)
        .unwrap();
    assert_eq!(old[0], -60.0);
    assert_eq!(old[19], 60.0);
    let m = fit("(pipe X (ridge))", &x, &y);
    let p = m.predict(&m.profile.schema, &x).unwrap();
    assert!(p[0] < -1800.0 && p[19] > 1800.0);
    // Independent normal equations for this one standardized feature.
    for i in 0..20 {
        let expected = y[i] / 1.02;
        assert!((p[i] - expected).abs() < 1e-9);
    }
    let z = feat_math::fit(&genome("(pipe X (target kind=zscore) (ridge))"), &x, &y)
        .unwrap()
        .predict(&x)
        .unwrap();
    assert!(
        z[19] > 60.0,
        "zscore was already an explicit legacy alternative"
    );
    let extreme = m.predict(&m.profile.schema, &vec![vec![1e100]]).unwrap();
    assert_eq!(extreme[0], m.clamp_hi);
}
#[test]
fn zeros_and_missing_are_distinct_and_train_only() {
    let mut old = vec![vec![0.0], vec![f64::NAN], vec![6.0]];
    feat_math::sanitize(&mut old);
    let stage = feat_math::dsl::Stage::new("impute");
    let fitted = feat_math::ops::fit_op(&stage, &old, &[]).unwrap();
    let old = feat_math::ops::apply_op(&fitted, &old).unwrap();
    assert_eq!(old[0], old[1]);
    let m = fit(
        "(pipe X (impute) (ridge))",
        &vec![
            vec![0.0, f64::NAN],
            vec![6.0, f64::NAN],
            vec![f64::NAN, f64::NAN],
        ],
        &[0.0, 6.0, 3.0],
    );
    assert_eq!(m.fill, vec![3.0, 0.0]);
    let p = m
        .predict(
            &m.profile.schema,
            &vec![vec![0.0, 0.0], vec![f64::NAN, 0.0], vec![600.0, 0.0]],
        )
        .unwrap();
    assert!(p[0] < p[1]);
    assert_eq!(m.fill, vec![3.0, 0.0]);
}
#[test]
fn meaningful_topk_and_no_silent_width_loss() {
    let x = vec![vec![2.0, 3.0], vec![4.0, 5.0]];
    let mut a = feat_math::dsl::Stage::new("poly_interact");
    a.params
        .insert("topk".into(), feat_math::dsl::ParamValue::Int(1));
    let mut b = a.clone();
    b.params
        .insert("topk".into(), feat_math::dsl::ParamValue::Int(2));
    let old_a = feat_math::ops::fit_op(&a, &x, &[]).unwrap();
    let old_b = feat_math::ops::fit_op(&b, &x, &[]).unwrap();
    assert_eq!(
        feat_math::ops::apply_op(&old_a, &x).unwrap(),
        feat_math::ops::apply_op(&old_b, &x).unwrap()
    );
    let a = fit("(pipe X (poly_interact topk=1) (ridge))", &x, &[1., 2.]);
    let b = fit("(pipe X (poly_interact topk=2) (ridge))", &x, &[1., 2.]);
    assert_eq!(a.transforms[0].apply(&x).unwrap()[0], vec![4.0]);
    assert_eq!(b.transforms[0].apply(&x).unwrap()[0], vec![4.0, 9.0]);
    let wide = vec![vec![1.0; 512]; 2];
    assert!(NumericModelV2::fit(
        profile(512),
        &genome("(pipe X (poly_interact topk=1 append=true) (ridge))"),
        &wide,
        &[1., 2.]
    )
    .is_err());
    for s in [
        "(pipe X (histgbm))",
        "(pipe X (ridge typo=2))",
        "(pipe X (poly_interact topk=1.5) (ridge))",
        "(pipe X (target kind=zscore) (ridge))",
    ] {
        assert!(
            NumericModelV2::fit(profile(2), &genome(s), &x, &[1., 2.]).is_err(),
            "{s}"
        );
    }
}
#[test]
fn named_high_columns_and_schema_order() {
    let mut p = profile(64);
    p.selected_columns = vec!["c17".into(), "c63".into()];
    let x: Rows = (0..8)
        .map(|i| {
            let mut r = vec![0.; 64];
            r[17] = i as f64;
            r[63] = (i * i) as f64;
            r
        })
        .collect();
    let y: Vec<_> = (0..8).map(|i| (i * i * 100) as f64).collect();
    let m = NumericModelV2::fit(p.clone(), &genome("(pipe X (ridge))"), &x, &y).unwrap();
    assert_eq!(m.indices, vec![17, 63]);
    assert!(m.predict(&p.schema, &x).unwrap()[7] > 4000.);
    let mut shuffled = p.schema.clone();
    shuffled.swap(0, 63);
    assert!(m.predict(&shuffled, &x).is_err());
    let mut moved = p.clone();
    moved.schema = shuffled.clone();
    let moved_x: Rows = x
        .iter()
        .map(|r| {
            let mut row = r.clone();
            row.swap(0, 63);
            row
        })
        .collect();
    let moved_model =
        NumericModelV2::fit(moved.clone(), &genome("(pipe X (ridge))"), &moved_x, &y).unwrap();
    assert_eq!(
        m.predict(&p.schema, &x).unwrap(),
        moved_model.predict(&moved.schema, &moved_x).unwrap()
    );
    for name in ["c17", "c63"] {
        let expr = feat_math::math_expr::numeric_v2_scope::column(&p, name).unwrap();
        let index = if name == "c17" { 17 } else { 63 };
        assert_eq!(
            expr.compile().unwrap().eval_row(&x[7]).unwrap(),
            x[7][index]
        );
    }
    assert!(feat_math::math_expr::numeric_v2_scope::column(&p, "c64").is_err());
}
#[test]
fn iid_oof_refits_without_held_labels() {
    let x: Rows = (0..18)
        .map(|i| vec![if i == 0 { f64::NAN } else { i as f64 }])
        .collect();
    let y: Vec<_> = (0..18).map(|i| i as f64 * 100.).collect();
    let a = feat_math::stack::numeric_v2_iid_folds(18, 3, 42).unwrap();
    assert_eq!(
        a,
        feat_math::stack::numeric_v2_iid_folds(18, 3, 42).unwrap()
    );
    assert_ne!(
        a,
        feat_math::stack::numeric_v2_iid_folds(18, 3, 43).unwrap()
    );
    assert_ne!(a, (0..18).map(|i| i / 6).collect::<Vec<_>>());
    let p = feat_math::numeric_v2::predict_oof_iid(
        profile(1),
        &genome("(pipe X (ridge))"),
        &x,
        &y,
        3,
        42,
    )
    .unwrap();
    let mut poisoned = y.clone();
    for (i, v) in poisoned.iter_mut().enumerate() {
        if a[i] == a[0] {
            *v += 1e6;
        }
    }
    let q = feat_math::numeric_v2::predict_oof_iid(
        profile(1),
        &genome("(pipe X (ridge))"),
        &x,
        &poisoned,
        3,
        42,
    )
    .unwrap();
    assert_eq!(p.assignment, a);
    assert_eq!(p.fold_identity_sha256, q.fold_identity_sha256);
    assert_eq!(p.fold_identity_sha256.len(), 64);
    let other = feat_math::numeric_v2::predict_oof_iid(
        profile(1),
        &genome("(pipe X (ridge))"),
        &x,
        &y,
        3,
        43,
    )
    .unwrap();
    assert_ne!(p.fold_identity_sha256, other.fold_identity_sha256);
    for i in 0..18 {
        if a[i] == a[0] {
            assert_eq!(p.predictions[i], q.predictions[i]);
        }
    }
}
#[test]
fn artifact_identity_mismatch_and_rollback() {
    let m = fit(
        "(pipe X (ridge))",
        &vec![vec![0.], vec![1.]],
        &[-100., 100.],
    );
    let opts = OptionsV2::new(TargetV2::Python);
    let a = numeric_v2_emit::emit(&m, opts.clone()).unwrap();
    assert!(a.reload(&opts, &a.model_sha256).is_ok());
    assert!(a
        .reload(&OptionsV2::new(TargetV2::Rust), &a.model_sha256)
        .is_err());
    assert!(a.reload(&opts, &"0".repeat(64)).is_err());
    let mut corrupt = a.clone();
    corrupt.source.replace_range(0..1, "x");
    assert!(corrupt.reload(&opts, &a.model_sha256).is_err());
    corrupt.source_sha256 = numeric_v2_emit::hash(corrupt.source.as_bytes());
    assert!(corrupt.reload(&opts, &a.model_sha256).is_err());
    let mut bad = opts.clone();
    bad.numeric_profile = "legacy".into();
    assert!(numeric_v2_emit::emit(&m, bad).is_err());
    // Rollback selects the retained legacy caller rather than relabeling the new scores.
    let old = feat_math::fit(
        &genome("(pipe X (ridge))"),
        &vec![vec![0.], vec![1.]],
        &[-100., 100.],
    )
    .unwrap();
    assert_eq!(old.predict(&vec![vec![1.]]).unwrap(), vec![60.]);
}
#[test]
fn fitted_python_and_rust_execute_exact_emitted_bytes() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(&root).unwrap();
    let train = vec![
        vec![0., 2.],
        vec![1., 2.],
        vec![2., 3.],
        vec![3., 4.],
        vec![4., 4.],
    ];
    for recipe in [
        "(pipe X (ridge))",
        "(pipe X (poly_interact topk=3 append=true) (ridge))",
        "(pipe X (quantile_bins bins=4) (ridge))",
        "(pipe X (bin_onehot bins=4) (ridge))",
    ] {
        let m = fit(recipe, &train, &[-200., -100., 0., 100., 200.]);
        let probes = vec![vec![0., 2.], vec![f64::NAN, 3.], vec![4., 4.]];
        let want = m.predict(&m.profile.schema, &probes).unwrap();
        for target in [TargetV2::Python, TargetV2::Rust] {
            let artifact = numeric_v2_emit::emit(&m, OptionsV2::new(target)).unwrap();
            let evidence = root
                .join("numeric-v2-artifacts")
                .join(format!("{}-{target:?}", artifact.model_sha256));
            std::fs::create_dir_all(&evidence).unwrap();
            std::fs::write(
                evidence.join("artifact.json"),
                serde_json::to_vec_pretty(&artifact).unwrap(),
            )
            .unwrap();
            let source = dir.path().join(if target == TargetV2::Python {
                "model.py"
            } else {
                "model.rs"
            });
            std::fs::write(&source, &artifact.source).unwrap();
            let output = if target == TargetV2::Python {
                std::process::Command::new("python3").arg("-c").arg("import runpy,sys; m=runpy.run_path(sys.argv[1]); print(*m['predict']([[0.,2.],[float('nan'),3.],[4.,4.]]),sep='\\n')").arg(&source).output().unwrap()
            } else {
                let driver = dir.path().join("driver.rs");
                std::fs::write(&driver,format!("include!({:?}); fn main(){{for v in predict(&[vec![0.,2.],vec![f64::NAN,3.],vec![4.,4.]]).unwrap(){{println!(\"{{:.17}}\",v);}}}}",source)).unwrap();
                let bin = dir.path().join("driver");
                let compile = std::process::Command::new("rustc")
                    .args(["--edition=2021", "-Awarnings"])
                    .arg(&driver)
                    .arg("-o")
                    .arg(&bin)
                    .output()
                    .unwrap();
                assert!(
                    compile.status.success(),
                    "{}",
                    String::from_utf8_lossy(&compile.stderr)
                );
                std::process::Command::new(bin).output().unwrap()
            };
            std::fs::write(evidence.join("stdout.txt"), &output.stdout).unwrap();
            std::fs::write(evidence.join("stderr.txt"), &output.stderr).unwrap();
            std::fs::write(
                evidence.join("runtime.txt"),
                format!(
                    "target={target:?} source={} compiler={} exit={:?}",
                    source.display(),
                    if target == TargetV2::Rust {
                        "rustc --edition=2021 -Awarnings driver.rs -o driver; driver"
                    } else {
                        "python3 -c <recorded inline predictor driver> model.py"
                    },
                    output.status.code()
                ),
            )
            .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let got: Vec<f64> = String::from_utf8(output.stdout)
                .unwrap()
                .lines()
                .map(|v| v.parse().unwrap())
                .collect();
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 1e-10, "{target:?}: {a} {b}");
            }
            assert_eq!(got.len(), want.len());
        }
    }
}

#[test]
fn legacy_width_and_gbm_prefix_are_characterized_not_relabelled() {
    let x = vec![vec![1.0; 512]; 4];
    let mut stage = feat_math::dsl::Stage::new("poly_interact");
    stage
        .params
        .insert("append".into(), feat_math::dsl::ParamValue::Bool(true));
    let op = feat_math::ops::fit_op(&stage, &x, &[]).unwrap();
    let old = feat_math::ops::apply_op(&op, &x).unwrap();
    assert_eq!(old[0].len(), 512);
    assert_eq!(old, x);
    let mut rows = vec![vec![0.0; 64]; 8];
    for (i, r) in rows.iter_mut().enumerate() {
        r[63] = i as f64;
    }
    let y: Vec<f64> = (0..8).map(|i| 100.0 * i as f64).collect();
    let params = feat_math::heads::GbmParams {
        max_features: 17,
        rounds: 2,
        min_leaf: 1,
        ..Default::default()
    };
    let m = feat_math::heads::GbmModel::fit(&rows, &y, &params);
    assert_eq!(m.n_feat, 17);
    let pred = m.predict(&rows);
    assert!(pred.iter().all(|v| *v == pred[0]));
}
#[test]
fn invalid_fitted_state_and_ordered_shuffle_are_refused() {
    let x = vec![vec![0.], vec![1.], vec![2.]];
    let mut p = profile(1);
    p.row_semantics = feat_math::RowSemantics::Ordered {
        group_by: vec![],
        order_by: vec!["c0".into()],
    };
    assert!(feat_math::numeric_v2::predict_oof_iid(
        p,
        &genome("(pipe X (ridge))"),
        &x,
        &[0., 1., 2.],
        2,
        0
    )
    .is_err());
    let mut m = fit("(pipe X (ridge))", &x, &[0., 1., 2.]);
    m.ridge.w.clear();
    assert!(numeric_v2_emit::emit(&m, OptionsV2::new(TargetV2::Rust)).is_err());
    assert!(m.predict(&m.profile.schema, &x).is_err());
}
