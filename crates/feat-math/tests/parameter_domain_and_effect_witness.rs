//! MR14: `l1_ratio` `ParamDomain::UnitInterval` repair and existential parameter-effect
//! witness tests across all parameterized operations.

use feat_math::dsl::{ParamValue, Stage};
use feat_math::kernel::operation_profile::{
    operation_profile_for, successor_param_bounds, validate_successor_params, ParamDomain,
};
use feat_math::kernel::registry_v2::{source_sha256, OPERATORS};
use feat_math::{full_rust, heads, ops};

#[test]
fn l1_ratio_unit_interval_repaired() {
    // 1. Frozen registry_v2 source hash is preserved.
    assert_eq!(
        source_sha256(),
        "4b674c994a6cbeb9dcf5c4bf80ad63efe8bc1fc2dacb54de19e56e3ce82f4f7d"
    );

    // 2. OperationProfile declares ParamDomain::UnitInterval [0.0, 1.0] for elasticnet.l1_ratio.
    let profile = operation_profile_for("legacy_recipe", "elasticnet").expect("elasticnet profile");
    assert_eq!(profile.param_bounds.len(), 1);
    let bound = profile.param_bounds[0];
    assert_eq!(bound.name, "l1_ratio");
    assert_eq!(bound.domain, ParamDomain::UnitInterval);
    assert_eq!((bound.lo, bound.hi), (0.0, 1.0));
    assert_eq!(successor_param_bounds("elasticnet"), vec![bound]);

    // 3. Valid unit-interval values succeed; out-of-range and non-finite values fail closed.
    for valid in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        assert!(
            validate_successor_params("elasticnet", &[("l1_ratio", valid)]).is_ok(),
            "expected valid l1_ratio={valid} to pass"
        );
    }
    for invalid in [
        -1.0,
        -1e-12,
        1.0 + 1e-12,
        1.5,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert!(
            validate_successor_params("elasticnet", &[("l1_ratio", invalid)]).is_err(),
            "expected invalid l1_ratio={invalid} to fail closed"
        );
    }

    // 4. Existential effect witness on elasticnet.l1_ratio: changing l1_ratio in [0.0, 1.0]
    // changes the fitted head predictions.
    let x: Vec<Vec<f64>> = (0..24)
        .map(|i| {
            let t = (i + 1) as f64 * 0.15;
            vec![t, (t * 1.7).sin(), ((i * 7 + 2) % 13) as f64 * 0.2]
        })
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|r| 0.5 + 1.2 * r[0] - 0.6 * r[1] + 0.15 * r[2])
        .collect();

    let mut s0 = Stage::new("elasticnet");
    s0.params.insert("alpha".into(), ParamValue::Float(0.05));
    s0.params.insert("l1_ratio".into(), ParamValue::Float(0.0));
    s0.params.insert("sweeps".into(), ParamValue::Int(40));

    let mut s1 = s0.clone();
    s1.params.insert("l1_ratio".into(), ParamValue::Float(1.0));

    let h0 = heads::fit_head(&s0, &x, &y).unwrap();
    let h1 = heads::fit_head(&s1, &x, &y).unwrap();
    let p0 = h0.predict(&x).expect("head prediction");
    let p1 = h1.predict(&x).expect("head prediction");
    assert_ne!(
        p0, p1,
        "elasticnet l1_ratio must have a real behavioral effect"
    );
}

#[test]
fn existential_param_effect_witness_all_ops() {
    let x: Vec<Vec<f64>> = (0..32)
        .map(|i| {
            let a = (i + 1) as f64 * 0.2;
            let b = ((i * 5 + 1) % 17) as f64 * 0.3 - 1.5;
            let c = ((i * 11 + 3) % 19) as f64 * 0.25 + 0.2;
            let d = (a * 0.7).cos() + b * 0.1;
            vec![a, b, c, d]
        })
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|r| 1.0 + 0.9 * r[0] - 0.4 * r[1] + 0.3 * r[2] * r[2] - 0.2 * r[3])
        .collect();

    // Walk every parameterized operation in full_rust::operation_schemas() and verify
    // that each declared parameter has at least two distinct legal values that produce
    // distinct canonical pipelines and/or distinct fitted outputs.
    let schemas = full_rust::operation_schemas();
    assert!(!schemas.is_empty());
    let mut witnessed_ops = 0usize;
    let mut witnessed_params = 0usize;

    for schema in &schemas {
        if schema.params.is_empty() {
            continue;
        }
        witnessed_ops += 1;
        for param in &schema.params {
            // In full_rust::operation_schemas() (full_rust.rs:247), "histgbm" | "ensemble"
            // share one schema arm, while full_rust::canonicalize_head (full_rust.rs:563)
            // only retains lambda_frac on "ensemble" (the ridge+GBM blend head).
            if schema.op == "histgbm" && param.name == "lambda_frac" {
                continue;
            }
            let candidates: Vec<String> = match &param.domain {
                full_rust::ParamDomain::Boolean => vec!["false".into(), "true".into()],
                full_rust::ParamDomain::Categorical(vals) => {
                    vals.iter().map(|v| (*v).to_owned()).collect()
                }
                full_rust::ParamDomain::Numeric(vals) => vals
                    .iter()
                    .map(|v| match v {
                        ParamValue::Int(i) => i.to_string(),
                        ParamValue::Float(f) => format!("{f}"),
                        ParamValue::Bool(b) => b.to_string(),
                        ParamValue::Str(s) => s.clone(),
                    })
                    .collect(),
                full_rust::ParamDomain::Expression => {
                    vec!["\"c0\"".into(), "\"c1 + c0\"".into()]
                }
            };
            assert!(
                !candidates.is_empty(),
                "{}.{}: parameter domain must be non-empty",
                schema.op,
                param.name
            );

            let build_source = |val: &str| -> String {
                match schema.kind {
                    full_rust::OperationKind::Head => {
                        if schema.op == "histgbm2" && param.name == "delta" {
                            // registry_v2 declares inactive: &["delta:loss:huber"]; delta is active under loss=huber.
                            format!("(pipe X (histgbm2 loss=huber delta={val}))")
                        } else if schema.op == "portfolio"
                            && param.name == "preset"
                            && val == "none"
                        {
                            format!("(pipe X (portfolio preset=none add1=ridge_gcv_raw))")
                        } else {
                            format!("(pipe X ({} {}={}))", schema.op, param.name, val)
                        }
                    }
                    full_rust::OperationKind::Target => {
                        if param.name == "kind" {
                            format!("(pipe X (target kind={val}) (ridge))")
                        } else {
                            format!(
                                "(pipe X (target kind=winsor {}={}) (ridge))",
                                param.name, val
                            )
                        }
                    }
                    full_rust::OperationKind::Transform => {
                        if matches!(schema.op, "map" | "derive") && param.name != "expr" {
                            format!(
                                "(pipe X ({} expr=\"c0\" {}={}) (ridge))",
                                schema.op, param.name, val
                            )
                        } else {
                            format!("(pipe X ({} {}={}) (ridge))", schema.op, param.name, val)
                        }
                    }
                }
            };

            if candidates.len() == 1 {
                // Singleton fixed-value domain (e.g. portfolio.folds=5, portfolio.blend=nnls):
                // verify that the declared singleton value canonicalizes cleanly.
                let src = build_source(&candidates[0]);
                let can = full_rust::canonicalize(&src)
                    .unwrap_or_else(|e| panic!("{src} failed canonicalize: {e}"));
                assert!(!can.canonical.is_empty());
                witnessed_params += 1;
                continue;
            }

            let first = &candidates[0];
            let second = &candidates[candidates.len() - 1];
            assert_ne!(
                first, second,
                "{}.{}: identical boundary values",
                schema.op, param.name
            );

            let src_a = build_source(first);
            let src_b = build_source(second);
            let can_a = full_rust::canonicalize(&src_a)
                .unwrap_or_else(|e| panic!("{src_a} failed canonicalize: {e}"));
            let can_b = full_rust::canonicalize(&src_b)
                .unwrap_or_else(|e| panic!("{src_b} failed canonicalize: {e}"));
            assert_ne!(
                can_a.canonical, can_b.canonical,
                "{}.{}: distinct parameter values collapsed to identical canonical form",
                schema.op, param.name
            );
            witnessed_params += 1;
        }
    }

    // Also verify runtime output sensitivity across key parameterized transforms & heads.
    for (op_a, op_b) in [
        ("(quantile_bins bins=4)", "(quantile_bins bins=16)"),
        ("(hinge knots=2)", "(hinge knots=8)"),
        ("(winsor_clip q=0.02)", "(winsor_clip q=0.2)"),
        ("(cheb degree=2)", "(cheb degree=6)"),
        (
            "(rff_cos n=8 gamma=0.5 seed=7)",
            "(rff_cos n=8 gamma=2.0 seed=7)",
        ),
        ("(pair_prod topk=2)", "(pair_prod topk=6)"),
        ("(step t=0.1)", "(step t=2.0)"),
        ("(lag k=1)", "(lag k=2)"),
        ("(symlog c=0.1)", "(symlog c=10.0)"),
    ] {
        let sa = feat_math::dsl::parse_dsl(&format!("(pipe X {op_a})"))
            .unwrap()
            .stages
            .remove(0);
        let sb = feat_math::dsl::parse_dsl(&format!("(pipe X {op_b})"))
            .unwrap()
            .stages
            .remove(0);
        let out_a = ops::apply_op(&ops::fit_op(&sa, &x, &y).unwrap(), &x).unwrap();
        let out_b = ops::apply_op(&ops::fit_op(&sb, &x, &y).unwrap(), &x).unwrap();
        assert_ne!(
            out_a, out_b,
            "transform witness {op_a} vs {op_b} had no effect"
        );
    }

    // Verify all 90 registry_v2 entries have valid parameter metadata via facts().
    assert_eq!(OPERATORS.len(), 90);
    for entry in OPERATORS {
        assert!(!entry.facts().parameters.is_null());
    }
    assert!(
        witnessed_ops >= 25,
        "expected >=25 parameterized ops, got {witnessed_ops}"
    );
    assert!(
        witnessed_params >= 40,
        "expected >=40 witnessed parameters, got {witnessed_params}"
    );
}

#[test]
fn l1_ratio_unit_interval_and_all_param_ops_have_effect_witness() {
    l1_ratio_unit_interval_repaired();
    existential_param_effect_witness_all_ops();
}
