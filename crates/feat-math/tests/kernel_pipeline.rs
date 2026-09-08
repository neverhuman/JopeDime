use feat_math::{dsl::parse_dsl, kernel::pipeline::PipelineSlots, GenomeIr};

#[test]
fn one_hundred_thousand_pipeline_children_preserve_target_and_modern_learner_slots() {
    let heads = [
        "ridge",
        "logistic",
        "logit_ridge",
        "tab_boost",
        "extra_trees",
        "gam_boost",
        "pls",
    ];
    let mut changed = 0;
    for seed in 0..100_000 {
        let head = heads[seed as usize % heads.len()];
        let source = format!("(pipe X (target kind=log1p) (standardize) ({head}))");
        let parent = PipelineSlots::from_pipeline(&parse_dsl(&source).unwrap()).unwrap();
        let columns = [1, 2, 32, 512][seed as usize % 4];
        let child = if seed % 2 == 0 {
            parent.mutate_v1(seed, 32, columns, 12).unwrap()
        } else {
            parent.crossover_v1(&parent, seed, 32, columns, 12).unwrap()
        };
        child.validate_legacy_widths(32, columns, 12).unwrap();
        let pipe = child.to_pipeline().unwrap();
        assert_eq!(pipe.stages[0].op, "target");
        assert_eq!(pipe.stages.iter().filter(|s| s.op == "target").count(), 1);
        assert_eq!(pipe.stages.last().unwrap().op, head);
        assert_eq!(pipe.stages.iter().filter(|s| s.op == head).count(), 1);
        changed += usize::from(pipe != parent.to_pipeline().unwrap());
        if seed < 1000 && head == "ridge" {
            feat_math::transpile::supports(&GenomeIr::Pipeline(pipe)).unwrap();
        }
    }
    assert!(changed > 40_000);
}

#[test]
fn downstream_columns_are_checked_after_width_changes() {
    for source in [
        "(pipe X (derive expr=\"c0\") (map expr=\"c1+1\") (ridge))",
        "(pipe X (standardize) (target kind=log1p) (ridge))",
        "(pipe X (logit_ridge) (ridge))",
    ] {
        let admitted = PipelineSlots::from_pipeline(&parse_dsl(source).unwrap())
            .and_then(|slots| slots.validate_legacy_widths(32, 2, 12));
        assert!(admitted.is_err(), "{source}");
    }
    let source = "(pipe X (derive expr=\"c0\" append=true) (map expr=\"c2+1\") (ridge))";
    PipelineSlots::from_pipeline(&parse_dsl(source).unwrap())
        .unwrap()
        .validate_legacy_widths(32, 2, 12)
        .unwrap();
}

#[test]
fn structurally_valid_modern_children_execute_on_their_actual_task() {
    use feat_math::{
        fit_table, ColumnSpec, DataSchema, Objective, RawTable, RowSemantics, TaskSpec,
    };
    let schema = DataSchema {
        columns: vec![ColumnSpec::numeric("c0"), ColumnSpec::numeric("c1")],
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::new(
        vec!["c0".into(), "c1".into()],
        (0..32)
            .map(|i| vec![(i as f64 / 32.).to_string(), ((i % 5) as f64).to_string()])
            .collect(),
    )
    .unwrap();
    for head in [
        "logit_ridge",
        "tab_boost",
        "extra_trees",
        "gam_boost",
        "pls",
    ] {
        let task = if head == "logit_ridge" {
            TaskSpec::binary(Objective::LogLoss).unwrap()
        } else {
            TaskSpec::regression(Objective::Rmse).unwrap()
        };
        let y: Vec<_> = (0..32)
            .map(|i| {
                if head == "logit_ridge" {
                    (i % 2) as f64
                } else {
                    i as f64 / 10.
                }
            })
            .collect();
        let parent = PipelineSlots::from_pipeline(
            &parse_dsl(&format!("(pipe X (standardize) ({head}))")).unwrap(),
        )
        .unwrap();
        for seed in 0..4 {
            let child = parent.mutate_v1(seed, 32, 2, 12).unwrap();
            let model = fit_table(
                &GenomeIr::Pipeline(child.to_pipeline().unwrap()),
                &task,
                &schema,
                &table,
                &y,
            )
            .unwrap();
            assert!(model
                .predict_table(&table)
                .unwrap()
                .iter()
                .all(|p| p.is_finite()));
        }
    }
}
