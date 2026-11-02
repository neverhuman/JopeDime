#[test]
fn fitted_legacy_registry_states_execute_or_match_rust_refusal() {
    use feat_math::advanced_v3::{
        fit_legacy_numeric_bound, fit_table, ColumnSpec, DataSchema, Objective, RawTable,
        RowSemantics, TaskSpec,
    };
    if !support::enabled() {
        return;
    }
    let root = std::env::current_dir()
        .unwrap()
        .join("../../target/cuda-census/bound");
    let (x, y, probe) = data();
    let headers = (0..5).map(|i| format!("c{i}")).collect::<Vec<_>>();
    let schema = DataSchema {
        columns: headers
            .iter()
            .map(|h| ColumnSpec::numeric(h.clone()))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let training = RawTable::new(
        headers,
        x.iter()
            .map(|r| r.iter().map(f64::to_string).collect())
            .collect(),
    )
    .unwrap();
    let rows = OPERATORS
        .iter()
        .filter(|r| r.dialect == "legacy_recipe" && r.targets.contains(&"rust_recipe"))
        .collect::<Vec<_>>();
    let mut executed = BTreeSet::new();
    let mut refused = BTreeSet::new();
    for row in &rows {
        let stage = match row.name {
            "target" => "target kind=zscore".into(),
            "map" => "map expr=\"sin(x)\"".into(),
            "derive" => "derive expr=\"col(0)+row_mean\"".into(),
            name => name.to_string(),
        };
        let source = if row.facts().role == "head" {
            format!("(pipe X ({stage}))")
        } else {
            format!("(pipe X ({stage}) (ridge))")
        };
        let genome = GenomeIr::parse(&source).unwrap();
        let legacy = fit_legacy_numeric_bound(&genome, &x, &y).unwrap();
        assert!(feat_math::advanced_v3::emit_bound(&legacy, &EmitOptions::default()).is_err());
        assert!(emit_bound(&legacy, &EmitOptions::default()).is_err());
        // Rust bound emission accepts Pipeline state, not the raw Legacy
        // wrapper. Feature-position lag/roll remain in that legacy wrapper;
        // table-aware ordered versions have their own positive fitted tests.
        let model = if matches!(row.name, "lag" | "roll") {
            legacy
        } else {
            fit_table(
                &genome,
                &TaskSpec::regression(Objective::R2).unwrap(),
                &schema,
                &training,
                &y,
            )
            .unwrap_or_else(|e| panic!("{} fitted pipeline: {e}", row.name))
        };
        let rust = feat_math::advanced_v3::emit_bound(&model, &EmitOptions::default());
        let ts = emit_bound(&model, &EmitOptions::default());
        assert_eq!(
            rust.is_ok(),
            ts.is_ok(),
            "legacy bound admission {}",
            row.name
        );
        if let Err(error) = ts {
            assert!(!error.to_string().is_empty());
            refused.insert(row.name);
            continue;
        }
        let bundle = ts.unwrap();
        let table = RawTable::new(
            model
                .schema
                .columns
                .iter()
                .map(|c| c.name.clone())
                .collect(),
            probe
                .iter()
                .map(|r| r.iter().map(f64::to_string).collect())
                .collect(),
        )
        .unwrap();
        support::run(
            &root.join(format!("legacy-{}", row.name)),
            &bundle,
            &serde_json::json!({"rows":table.rows}),
            &model.predict_table(&table).unwrap(),
            true,
        );
        executed.insert(row.name);
    }
    assert!(!executed.is_empty());
    assert_eq!(executed.len() + refused.len(), rows.len());
    eprintln!(
        "CUDA C++ census: {}",
        serde_json::json!({"kind":"legacy-bound","total":rows.len(),"executed":executed,"refused":refused,"unaccounted":0})
    );
}
