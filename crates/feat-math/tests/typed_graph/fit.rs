use super::*;

#[test]
fn fit_plan_pins_source_split_schema_operator_seed_and_monotone_row_closure() {
    let s = schema(2);
    let g = graph("(pipe X (target_bin_encode) (ridge))", &s);
    let d = dataset(&s, 16);
    let p = FitPlan::bind(&g, &d, closure(16), u64::MAX, &limits()).unwrap();
    assert!(p.effects().fit_y && p.effects().target_derived);
    let child = p
        .narrow(
            &g,
            &d,
            RowClosure {
                fit_rows: (0..8).collect(),
                predict_rows: vec![12, 13],
            },
            &limits(),
        )
        .unwrap();
    assert_ne!(p.identity().unwrap(), child.identity().unwrap());
    assert!(p
        .narrow(
            &g,
            &d,
            RowClosure {
                fit_rows: (0..13).collect(),
                predict_rows: vec![14, 15]
            },
            &limits()
        )
        .is_err());
    assert!(FitPlan::bind(
        &g,
        &d,
        RowClosure {
            fit_rows: vec![0, 0],
            predict_rows: vec![12]
        },
        1,
        &limits()
    )
    .is_err());
    assert!(FitPlan::bind(
        &g,
        &d,
        RowClosure {
            fit_rows: vec![0, 12],
            predict_rows: vec![12]
        },
        1,
        &limits()
    )
    .is_err());
    let mut changed = d.clone();
    changed.identity.split_sha256 = canon::sha256(b"different split");
    let mut stats = ExecutionStats::default();
    assert!(g.fit(&p, &changed, &limits(), &mut stats).is_err());
    assert_eq!(stats.fit_calls, 0);
    let p2 = FitPlan::bind(&g, &d, closure(16), 1, &limits()).unwrap();
    assert_ne!(p.identity().unwrap(), p2.identity().unwrap());
    let pure = scalar(
        &s,
        Expression::Column {
            column: s.column(0).unwrap(),
        },
    );
    let no_labels = FitPlan::bind(&pure, &d, closure(16), 1, &limits()).unwrap();
    assert!(!no_labels.effects().fit_y);
    assert!(no_labels.narrow(&g, &d, closure(16), &limits()).is_err());
    let mut forbidden_labels = d.clone();
    forbidden_labels.y[0] = f64::NAN;
    assert!(no_labels
        .narrow(&g, &forbidden_labels, closure(16), &limits())
        .unwrap_err()
        .contains("broadens schema/target access"));
    let mut no_fit = ExecutionStats::default();
    assert!(g
        .fit(&no_labels, &forbidden_labels, &limits(), &mut no_fit)
        .err()
        .unwrap()
        .contains("before data access"));
    assert_eq!(no_fit.fit_calls, 0);
}

#[test]
fn target_aware_heldout_poison_leaves_fitted_state_and_prescore_predictions_unchanged() {
    let s = schema(2);
    let g = graph("(pipe X (target_bin_encode bins=4 prior=1) (ridge))", &s);
    let d = dataset(&s, 32);
    let c = closure(32);
    let p = FitPlan::bind(&g, &d, c.clone(), 7, &limits()).unwrap();
    let mut stats = ExecutionStats::default();
    let fitted = g.fit(&p, &d, &limits(), &mut stats).unwrap();
    let before = fitted.state_bytes().unwrap();
    let honest = fitted.predict(&d, &c.predict_rows, &mut stats).unwrap();
    let mut poison = d.clone();
    for i in &c.predict_rows {
        poison.x[*i] = vec![1e9, -1e9];
        poison.y[*i] = 1e9;
    }
    let mut after_stats = ExecutionStats::default();
    let again = g.fit(&p, &poison, &limits(), &mut after_stats).unwrap();
    assert_eq!(before, again.state_bytes().unwrap());
    assert_eq!(
        bits(&honest),
        bits(
            &again
                .predict(&d, &c.predict_rows, &mut after_stats)
                .unwrap()
        )
    );
    assert_eq!(stats.target_fit_rows, c.fit_rows.iter().copied().collect());
    let GenomeIr::Pipeline(pipe) =
        GenomeIr::parse("(pipe X (target_bin_encode bins=4 prior=1) (ridge))").unwrap()
    else {
        panic!()
    };
    let old = GenomeIr::Pipeline(pipe);
    let score: Rows = c.predict_rows.iter().map(|i| d.x[*i].clone()).collect();
    let leaked = feat_math::fit_predict(&old, &d.x, &d.y, &score).unwrap();
    let poisoned_leak = feat_math::fit_predict(&old, &poison.x, &poison.y, &score).unwrap();
    assert_ne!(
        bits(&leaked),
        bits(&poisoned_leak),
        "deliberately leaky control did not detect poison"
    );
    let stack = GenomeIr::parse("(pipe X (target_bin_encode bins=4 prior=1) (stack))").unwrap();
    let stack_leak = feat_math::fit_predict(&stack, &d.x, &d.y, &score).unwrap();
    let stack_poison = feat_math::fit_predict(&stack, &poison.x, &poison.y, &score).unwrap();
    assert_ne!(
        bits(&stack_leak),
        bits(&stack_poison),
        "deliberately leaky target-aware stack control did not detect poison"
    );
}

#[test]
fn target_derived_stack_and_residual_are_unavailable_before_any_fit() {
    let s = schema(2);
    let g = graph("(pipe X (target_bin_encode) (ridge))", &s);
    let d = dataset(&s, 16);
    let plan = FitPlan::bind(&g, &d, closure(16), 1, &limits()).unwrap();
    for node in [
        json!({"kind":"stack","members":[2]}),
        json!({"kind":"residual","baseline":2,"correction":2}),
    ] {
        let unsafe_graph = forged(&g, |v| {
            v["nodes"].as_array_mut().unwrap().push(node);
            v["output"] = json!(3);
        });
        let mut stats = ExecutionStats::default();
        let err = unsafe_graph
            .fit(&plan, &d, &limits(), &mut stats)
            .err()
            .unwrap();
        assert!(err.contains("cross-fitting"));
        assert_eq!(stats, ExecutionStats::default());
    }
    for head in ["stack", "portfolio", "ensemble"] {
        let e = Edit::new(
            &g,
            EditKind::ReplaceHead {
                site: g.output(),
                operator: OperatorRef::new("legacy_recipe", head).unwrap(),
                parameters: Default::default(),
            },
            &limits(),
        )
        .unwrap();
        assert!(apply(&g, &e, &s, profile(), &limits()).is_err());
    }
}

#[test]
fn seeded_width_effect_executes_and_training_content_binding_precedes_fit() {
    let s = schema(3);
    let g = graph("(pipe X (rff_cos n=4 gamma=0.5 seed=7) (ridge))", &s);
    let d = dataset(&s, 16);
    assert!(g.validate(&limits()).unwrap().seeded);
    let (a, _, _) = run(&g, &d);
    let (b, _, _) = run(&g, &d);
    assert_eq!(bits(&a), bits(&b));
    let p = FitPlan::bind(&g, &d, closure(16), 1, &limits()).unwrap();
    let mut poison = d.clone();
    poison.y[0] += 10.;
    let mut stats = ExecutionStats::default();
    assert!(g.fit(&p, &poison, &limits(), &mut stats).is_err());
    assert_eq!(stats.fit_calls, 0);
    let mut stats = ExecutionStats::default();
    let fit = g.fit(&p, &d, &limits(), &mut stats).unwrap();
    assert!(fit.predict(&d, &[0], &mut stats).is_err());
}

#[test]
fn forged_bridge_cannot_attach_a_different_valid_graph_to_old_identity() {
    let s = schema(2);
    let mut bridge = LegacyBridge::import(
        &Program::parse_legacy("(pipe X (ridge))").unwrap(),
        &s,
        &limits(),
    )
    .unwrap();
    let other = graph("(pipe X (elasticnet))", &s);
    bridge.bridge = BridgeStatus::Typed {
        graph_id: other.identity(&limits()).unwrap(),
        graph: other,
    };
    assert!(bridge.to_bytes().unwrap_err().contains("forged"));
}
