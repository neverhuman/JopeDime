use super::*;

pub(super) fn branch_facts() -> Vec<(LineageFact, Links)> {
    let mut rows = Vec::new();
    let mut add = |name: &str, parent: &str, child: &str| {
        let mut fact = draw(parent, child);
        if let LineageFact::Drawn { coordinate, .. } = &mut fact {
            *coordinate = format!("synthetic/branch/{name}");
        }
        rows.push((fact, Links::default()));
    };
    for (name, parent, child) in [
        ("unchanged", "(pipe X (ridge))", "(pipe X (ridge))"),
        (
            "whole_program",
            "(sym raw \"c0 + 1\")",
            "(sym affine \"c0 * 2\")",
        ),
        ("insert", "(pipe X (ridge))", "(pipe X (rank) (ridge))"),
        ("delete", "(pipe X (rank) (ridge))", "(pipe X (ridge))"),
        (
            "replace",
            "(pipe X (rank) (ridge))",
            "(pipe X (robust_scale) (ridge))",
        ),
        (
            "swap",
            "(pipe X (rank) (robust_scale) (ridge))",
            "(pipe X (robust_scale) (rank) (ridge))",
        ),
        (
            "clamped",
            "(pipe X (hinge knots=40) (ridge))",
            "(pipe X (hinge knots=50) (ridge))",
        ),
        (
            "resolved",
            "(pipe X (cheb degree=2) (ridge))",
            "(pipe X (cheb degree=8) (ridge))",
        ),
        (
            "clipped",
            "(pipe X (winsor_clip q=0.5) (ridge))",
            "(pipe X (winsor_clip q=0.8) (ridge))",
        ),
        (
            "clip_change",
            "(pipe X (winsor_clip q=0.05) (ridge))",
            "(pipe X (winsor_clip q=0.1) (ridge))",
        ),
        (
            "seed",
            "(pipe X (rand_proj k=4 seed=1) (ridge))",
            "(pipe X (rand_proj k=4 seed=2) (ridge))",
        ),
        (
            "recipe",
            "(pipe X (rank) (ridge))",
            "(pipe X (cheb degree=4) (ridge lambda_frac=0.1))",
        ),
        ("logistic", "(pipe X (ridge))", "(pipe X (logistic))"),
        ("gbm", "(pipe X (ridge))", "(pipe X (gbm rounds=10))"),
        ("ensemble", "(pipe X (ridge))", "(pipe X (ensemble))"),
        // Invalid captured syntax is a limitation, never a guessed exact change.
        (
            "analysis_unavailable",
            "recorded unsupported syntax",
            "(pipe X (ridge))",
        ),
    ] {
        add(name, parent, child);
    }
    let before = format!("(pipe X {} (ridge))", "(cheb degree=1) ".repeat(7));
    let after = format!("(pipe X {} (ridge))", "(cheb degree=8) ".repeat(7));
    add("many_edits", &before, &after);
    for equal in [false, true] {
        let mut fact = draw("(pipe X (ridge))", "(pipe X (rank) (ridge))");
        if let LineageFact::Drawn {
            parents,
            origin,
            coordinate,
            ..
        } = &mut fact
        {
            parents.push(if equal {
                parents[0].clone()
            } else {
                program("(pipe X (robust_scale) (ridge))")
            });
            *origin = "crossover".into();
            *coordinate = format!("synthetic/branch/crossover/{equal}");
        }
        rows.push((fact, Links::default()));
    }
    let mut no_parent = draw("(pipe X (ridge))", "(pipe X (ridge))");
    if let LineageFact::Drawn {
        parents, origin, ..
    } = &mut no_parent
    {
        parents.clear();
        *origin = "random".into();
    }
    rows.push((no_parent, Links::default()));
    let mut comparison = draw("(pipe X (ridge))", "(pipe X (ridge lambda_frac=0.1))");
    if let LineageFact::Drawn { phase, .. } = &mut comparison {
        *phase = "comparison".into();
    }
    rows.push((comparison, Links::default()));
    let mut failed = draw("(pipe X (ridge))", "(pipe X (ridge))");
    if let LineageFact::Drawn { child, outcome, .. } = &mut failed {
        *child = None;
        *outcome = "operator_failed".into();
    }
    rows.push((failed, Links::default()));
    for failed in [false, true] {
        rows.push((
            LineageFact::Measured {
                cut: 0,
                phase: "comparison".into(),
                generation: 1,
                island: None,
                program_sha256: program("(pipe X (ridge))").sha256,
                origin: "remeasure".into(),
                status: if failed { "fit_failed" } else { "measured" }.into(),
                rmse: (!failed).then_some(0.25),
                hold_r2: None,
                train_r2: None,
                objective_loss: None,
                seconds: 0.02,
                failure: failed.then(|| "synthetic failure".into()),
            },
            Links {
                observation_sha256: Some("c".repeat(64)),
                ..Links::default()
            },
        ));
    }
    rows.push((
        LineageFact::PredictionBound {
            status: "failed".into(),
            model_sha256: "a".repeat(64),
            input_sha256: "b".repeat(64),
            rows: 3,
            seconds: Some(0.01),
        },
        Links::default(),
    ));
    rows
}

pub(super) fn branch_fixture() {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/renderer-v1-branches.jsonl");
    let page = read_journal(&fixture, None, 1024, 16 << 20).unwrap();
    assert!(!page.partial_tail);
    assert_eq!(page.events.len(), branch_facts().len());
    let text = page
        .events
        .iter()
        .map(|e| e.narration.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    for clause in [
        "comparison pool",
        "unchanged from",
        "semantic analysis is unavailable",
        "additional exact edits",
        "first two ordered parent roles",
        "unique crossover splice",
        "without parent references",
        "construction returned no program",
        "draw outcome",
        "retained boundary data",
        "insert ",
        "remove ",
        "replaces ",
        "swap stage[",
        "seeded construction",
        "changed recipe",
        "both requests resolve",
        "planner resolves",
        "logistic head to ridge",
        "gbm head to histgbm",
        "ensemble head to ensemble_ridge_gbm",
        "exact structural analysis is unavailable",
        "prediction failed",
    ] {
        assert!(text.contains(clause), "branch golden missing {clause}");
    }
    assert_eq!(
        jope_narrator::read_commit(&fixture).unwrap().records,
        page.events.len().to_string()
    );
}
