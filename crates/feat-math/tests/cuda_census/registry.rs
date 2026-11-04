#[test]
fn complete_recipe_registry_executes_or_refuses_identically() {
    if !support::enabled() {
        return;
    }
    let root = std::env::current_dir()
        .unwrap()
        .join("../../target/cuda-census/recipe");
    let mut executed = BTreeSet::new();
    let mut refused = BTreeSet::new();
    let rows = OPERATORS
        .iter()
        .filter(|r| r.dialect == "legacy_recipe")
        .collect::<Vec<_>>();
    for row in &rows {
        let stage = match row.name {
            "target" => "target kind=zscore".to_string(),
            "map" => "map expr=\"sin(x)+row_mean\"".to_string(),
            "derive" => "derive expr=\"col(0)*col(1)+row_std\"".to_string(),
            name => name.to_string(),
        };
        let source = if row.facts().role == "head" {
            format!("(pipe X ({stage}))")
        } else {
            format!("(pipe X ({stage}) (ridge))")
        };
        let genome = GenomeIr::parse(&source).unwrap();
        let rust = transpile::emit(&genome, &EmitOptions::default());
        let ts = emit(&genome, &EmitOptions::default());
        assert_eq!(rust.is_ok(), ts.is_ok(), "admission {}", row.name);
        assert_eq!(
            rust.is_ok(),
            row.targets.contains(&"rust_recipe"),
            "registry drift {}",
            row.name
        );
        if let Err(error) = ts {
            assert!(!error.to_string().is_empty());
            refused.insert(row.name);
            continue;
        }
        recipe(&root.join(row.name), &source);
        executed.insert(row.name);
        eprintln!("CUDA C++ registry recipe: {} executed", row.name);
    }
    assert!(!executed.is_empty());
    assert_eq!(executed.len() + refused.len(), rows.len());
    eprintln!(
        "CUDA C++ census: {}",
        serde_json::json!({"kind":"recipe","total":rows.len(),"executed":executed,"refused":refused,"unaccounted":0})
    );
}

#[test]
fn complete_bound_registry_and_codec_modes_execute() {
    use feat_math::advanced_v3::{
        fit_table, ColumnSpec, DataSchema, Objective, RawTable, RowSemantics, TaskSpec,
    };
    if !support::enabled() {
        return;
    }
    let root = std::env::current_dir()
        .unwrap()
        .join("../../target/cuda-census/bound");
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::numeric("a"),
            ColumnSpec::numeric("b"),
            ColumnSpec::categorical("category"),
        ],
        row_semantics: RowSemantics::Iid,
    };
    let raw = RawTable::new(
        vec!["a".into(), "b".into(), "category".into()],
        (0..32)
            .map(|i| {
                vec![
                    (i as f64 / 8.0 - 2.0).to_string(),
                    ((i % 5) as f64 - 2.0).to_string(),
                    format!("group,\"{}\"\n雪", i % 3),
                ]
            })
            .collect(),
    )
    .unwrap();
    let probe = RawTable::new(
        raw.headers.clone(),
        vec![
            raw.rows[3].clone(),
            vec!["".into(), "NaN".into(), "unknown\r\ncategory".into()],
            vec!["inf".into(), "-0".into(), "".into()],
        ],
    )
    .unwrap();
    let mut completed = BTreeSet::new();
    let mut refused = BTreeSet::new();
    // Registry exclusions describe generation policy, not an emitter promise.
    // They have separate ordered-tail/refusal tests in advanced_v3::emit.
    let rows = OPERATORS
        .iter()
        .filter(|r| r.dialect == "advanced_v3" && r.targets.contains(&"rust_bound"))
        .collect::<Vec<_>>();
    for row in &rows {
        let binary = matches!(row.name, "logistic" | "logit_ridge");
        let stage = if row.name == "target" {
            "target kind=zscore"
        } else {
            row.name
        };
        let source = if row.name == "input" {
            "(pipe X (input) (ridge))".into()
        } else if row.facts().role == "head" {
            format!("(pipe X (input) ({stage}))")
        } else if row.name == "target" {
            format!("(pipe X ({stage}) (input) (ridge))")
        } else {
            format!("(pipe X (input) ({stage}) (ridge))")
        };
        let task = if binary {
            TaskSpec::binary(Objective::LogLoss)
        } else {
            TaskSpec::regression(Objective::R2)
        }
        .unwrap();
        let y = (0..32)
            .map(|i| {
                if binary {
                    f64::from(i >= 16)
                } else {
                    i as f64 / 8.0 - 2.0
                }
            })
            .collect::<Vec<_>>();
        let model = fit_table(&GenomeIr::parse(&source).unwrap(), &task, &schema, &raw, &y)
            .unwrap_or_else(|e| panic!("{} fit: {e}", row.name));
        let expected = model.predict_table(&probe).unwrap();
        let rust = feat_math::advanced_v3::emit_bound(&model, &EmitOptions::default());
        let bundle = emit_bound(&model, &EmitOptions::default());
        assert_eq!(rust.is_ok(), bundle.is_ok(), "bound admission {}", row.name);
        let bundle = match bundle {
            Ok(bundle) => bundle,
            Err(error) => {
                assert_eq!(row.name, "portfolio", "unexpected bound refusal: {error}");
                assert!(error.to_string().contains("fitted legacy head"));
                refused.insert(row.name);
                continue;
            }
        };
        support::run(
            &root.join(row.name),
            &bundle,
            &serde_json::json!({"rows":probe.rows}),
            &expected,
            true,
        );
        completed.insert(row.name);
        eprintln!("CUDA C++ registry bound: {} executed", row.name);
    }
    assert_eq!(completed.len() + refused.len(), rows.len());
    for (index, input) in [
        "input categorical=onehot",
        "input categorical=frequency",
        "input categorical=ordered",
        "input categorical=hybrid max_onehot=2",
        "input categorical=onehot numeric=median missing=indicator",
    ]
    .iter()
    .enumerate()
    {
        let model = fit_table(
            &GenomeIr::parse(&format!(
                "(pipe X ({input}) (logit_ridge calibration=platt))"
            ))
            .unwrap(),
            &TaskSpec::binary(Objective::LogLoss).unwrap(),
            &schema,
            &raw,
            &(0..32).map(|i| f64::from(i >= 16)).collect::<Vec<_>>(),
        )
        .unwrap();
        let bundle = emit_bound(&model, &EmitOptions::default()).unwrap();
        support::run(
            &root.join(format!("codec-{index}")),
            &bundle,
            &serde_json::json!({"rows":probe.rows}),
            &model.predict_table(&probe).unwrap(),
            true,
        );
    }
    eprintln!(
        "CUDA C++ census: {}",
        serde_json::json!({"kind":"bound","total":rows.len(),"executed":completed,"refused":refused,"unaccounted":0})
    );
}
