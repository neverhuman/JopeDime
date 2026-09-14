//! V-10 public-boundary conformance for schema, runtime CSV, feature layout,
//! and fitted input state.
//!
//! Exact honest training encodings and table-aware row-operation states are
//! crate-private. Their V-10 assertions live in the kernel unit-test lane and
//! are cross-referenced by the V-10 evidence receipt.

use feat_math::{
    fit_table, ColumnSpec, DataSchema, FeatureBlock, FeatureLineage, FeatureOrigin, FeatureRole,
    GenomeIr, Objective, RawTable, RowSemantics, TaskKind, TaskSpec, MAX_GENERATED_DENSE_WIDTH,
};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct CsvCorpus {
    policy: String,
    accepted: Vec<AcceptedCsv>,
    rejected: Vec<RejectedCsv>,
}

#[derive(Deserialize)]
struct AcceptedCsv {
    name: String,
    csv: String,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct RejectedCsv {
    name: String,
    csv: String,
}

fn raw_block(width: usize) -> FeatureBlock {
    let mut masks = vec![vec![false; width]; 2];
    if width > 0 {
        masks[0][0] = true;
    }
    FeatureBlock::new(
        vec![vec![1.0; width], vec![2.0; width]],
        vec![FeatureRole::Numeric; width],
        masks,
        Vec::new(),
        (0..width)
            .map(|source| FeatureLineage::raw(source, "input:numeric"))
            .collect(),
        width,
    )
    .unwrap()
}

fn fitted_input_json(schema: &DataSchema, table: &RawTable, target: &[f64], input: &str) -> Value {
    let task = TaskSpec::regression(Objective::R2).unwrap();
    let genome = GenomeIr::parse(&format!(
        "(pipe X {input} (extra_trees trees=1 depth=1 min_leaf=1))"
    ))
    .unwrap();
    let model = fit_table(&genome, &task, schema, table, target).unwrap();
    serde_json::to_value(model).unwrap()["input"].clone()
}

#[test]
fn strict_runtime_csv_corpus_is_canonical() {
    let corpus: CsvCorpus =
        serde_json::from_str(include_str!("fixtures/advanced_v3/csv-runtime-v1.json")).unwrap();
    assert_eq!(corpus.policy, "advanced-v3-csv-runtime-v1");

    for case in corpus.accepted {
        let table = RawTable::from_csv(&case.csv)
            .unwrap_or_else(|error| panic!("accepted CSV case {}: {error}", case.name));
        assert_eq!(table.headers, case.headers, "case {} headers", case.name);
        assert_eq!(table.rows, case.rows, "case {} rows", case.name);
    }
    for case in corpus.rejected {
        assert!(
            RawTable::from_csv(&case.csv).is_err(),
            "rejected CSV case {} was accepted",
            case.name
        );
    }
}

#[test]
fn schema_task_and_header_contracts_are_strict_and_hash_bound() {
    let valid = DataSchema {
        columns: vec![
            ColumnSpec::categorical("group"),
            ColumnSpec::numeric("time"),
            ColumnSpec::numeric("value"),
        ],
        row_semantics: RowSemantics::Ordered {
            group_by: vec!["group".into()],
            order_by: vec!["time".into()],
        },
    };
    valid.validate().unwrap();
    assert_eq!(valid.sha256().unwrap(), valid.sha256().unwrap());

    let mut changed = valid.clone();
    changed.columns.swap(1, 2);
    assert_ne!(valid.sha256().unwrap(), changed.sha256().unwrap());
    changed = valid.clone();
    changed.columns[2].kind = feat_math::ColumnKind::Categorical;
    assert_ne!(valid.sha256().unwrap(), changed.sha256().unwrap());
    changed = valid.clone();
    changed.row_semantics = RowSemantics::Iid;
    assert_ne!(valid.sha256().unwrap(), changed.sha256().unwrap());

    for schema in [
        DataSchema {
            columns: Vec::new(),
            row_semantics: RowSemantics::Iid,
        },
        DataSchema {
            columns: vec![ColumnSpec::numeric(" ")],
            row_semantics: RowSemantics::Iid,
        },
        DataSchema {
            columns: vec![ColumnSpec::numeric("x"), ColumnSpec::numeric("x")],
            row_semantics: RowSemantics::Iid,
        },
        DataSchema {
            columns: valid.columns.clone(),
            row_semantics: RowSemantics::Ordered {
                group_by: vec!["missing".into()],
                order_by: vec!["time".into()],
            },
        },
        DataSchema {
            columns: valid.columns.clone(),
            row_semantics: RowSemantics::Ordered {
                group_by: vec!["group".into(), "group".into()],
                order_by: vec!["time".into()],
            },
        },
        DataSchema {
            columns: valid.columns.clone(),
            row_semantics: RowSemantics::Ordered {
                group_by: vec!["group".into()],
                order_by: vec!["group".into()],
            },
        },
        DataSchema {
            columns: valid.columns.clone(),
            row_semantics: RowSemantics::Ordered {
                group_by: vec!["group".into()],
                order_by: Vec::new(),
            },
        },
    ] {
        assert!(schema.validate().is_err());
        assert!(schema.sha256().is_err());
    }

    let ordered = RawTable::new(
        vec!["group".into(), "time".into(), "value".into()],
        vec![vec!["A".into(), "1".into(), "2".into()]],
    )
    .unwrap();
    ordered.validate_against(&valid).unwrap();
    for headers in [
        vec!["time".into(), "group".into(), "value".into()],
        vec!["group".into(), "time".into()],
        vec![
            "group".into(),
            "time".into(),
            "value".into(),
            "extra".into(),
        ],
    ] {
        let table = RawTable::new(headers.clone(), vec![vec!["0".into(); headers.len()]]).unwrap();
        assert!(table.validate_against(&valid).is_err());
    }

    assert!(TaskSpec::regression(Objective::LogLoss).is_err());
    assert!(TaskSpec::binary(Objective::R2).is_err());
    assert!(TaskSpec {
        kind: TaskKind::Binary,
        objective: Objective::R2,
    }
    .validate()
    .is_err());
}

#[test]
fn wide_raw_and_one_to_one_layout_preserve_metadata() {
    for width in [1_024, 1_776] {
        let block = raw_block(width);
        assert_eq!(block.values[0].len(), width);
        assert_eq!(block.widths().raw_derived, width);
        assert_eq!(block.widths().generated_dense, 0);
        assert_eq!(block.widths().generated_sparse, 0);

        let transformed = block
            .with_one_to_one_values(vec![vec![3.0; width], vec![4.0; width]], "maxabs")
            .unwrap();
        assert_eq!(transformed.values[0].len(), width);
        assert_eq!(transformed.roles, block.roles);
        assert_eq!(transformed.missing_masks, block.missing_masks);
        assert_eq!(transformed.sparse_columns, block.sparse_columns);
        assert_eq!(transformed.lineage[0].source_columns, [0]);
        assert_eq!(
            transformed.lineage[0].operations,
            ["input:numeric", "maxabs"]
        );
    }
}

#[test]
fn generated_dense_sparse_role_mask_and_lineage_accounting_is_exact() {
    assert_eq!(MAX_GENERATED_DENSE_WIDTH, 512);
    let base = raw_block(2);
    let appended = base
        .with_generated_columns(
            vec![vec![7.0, 1.0], vec![8.0, 0.0]],
            vec![FeatureRole::Generated, FeatureRole::MissingIndicator],
            vec![
                FeatureLineage::generated_dense(vec![0, 1], "derived:dense"),
                FeatureLineage::generated_sparse(vec![1], "derived:missing"),
            ],
            true,
        )
        .unwrap();
    assert_eq!(appended.widths().raw_derived, 2);
    assert_eq!(appended.widths().generated_dense, 1);
    assert_eq!(appended.widths().generated_sparse, 1);
    assert_eq!(appended.sparse_columns, [3]);
    assert_eq!(appended.roles[2], FeatureRole::Generated);
    assert_eq!(appended.roles[3], FeatureRole::MissingIndicator);
    assert_eq!(appended.lineage[2].source_columns, [0, 1]);
    assert_eq!(appended.lineage[2].origin, FeatureOrigin::GeneratedDense);
    assert_eq!(appended.lineage[3].origin, FeatureOrigin::GeneratedSparse);
    assert!(appended.missing_masks[0][0]);
    assert!(!appended.missing_masks[0][2]);
    assert!(!appended.missing_masks[0][3]);

    for width in [511, 512] {
        let accepted = base.with_generated_columns(
            vec![vec![0.0; width]; 2],
            vec![FeatureRole::Generated; width],
            (0..width)
                .map(|_| FeatureLineage::generated_dense(vec![0], "dense-boundary"))
                .collect(),
            false,
        );
        assert!(accepted.is_ok(), "generated dense width {width}");
    }
    let rejected = base.with_generated_columns(
        vec![vec![0.0; 513]; 2],
        vec![FeatureRole::Generated; 513],
        (0..513)
            .map(|_| FeatureLineage::generated_dense(vec![0], "dense-overflow"))
            .collect(),
        false,
    );
    assert!(rejected.is_err());

    let sparse_width = 600;
    let sparse = base
        .with_generated_columns(
            vec![vec![0.0; sparse_width]; 2],
            vec![FeatureRole::CategoricalOneHot; sparse_width],
            (0..sparse_width)
                .map(|_| FeatureLineage::generated_sparse(vec![1], "onehot"))
                .collect(),
            false,
        )
        .unwrap();
    assert_eq!(sparse.widths().generated_dense, 0);
    assert_eq!(sparse.widths().generated_sparse, sparse_width);
    assert_eq!(sparse.sparse_columns, (0..sparse_width).collect::<Vec<_>>());
    assert!(sparse
        .missing_masks
        .iter()
        .flatten()
        .all(|missing| !missing));
}

#[test]
fn public_fitted_input_retains_all_1024_to_1776_raw_columns() {
    for width in [1_024, 1_776] {
        let schema = DataSchema {
            columns: (0..width)
                .map(|column| ColumnSpec::numeric(format!("c{column}")))
                .collect(),
            row_semantics: RowSemantics::Iid,
        };
        let headers = schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let rows = (0..8)
            .map(|row| {
                (0..width)
                    .map(|column| ((row * 17 + column * 11) % 31).to_string())
                    .collect()
            })
            .collect();
        let table = RawTable::new(headers, rows).unwrap();
        let target: Vec<f64> = (0..8).map(|row| row as f64 * 0.25).collect();
        let input = fitted_input_json(
            &schema,
            &table,
            &target,
            "(input numeric=zero missing=none categorical=frequency)",
        );
        assert_eq!(input["columns"].as_array().unwrap().len(), width);
    }
}

#[test]
fn numeric_and_categorical_missing_contract_is_visible_in_fitted_state() {
    let numeric_schema = DataSchema {
        columns: vec![ColumnSpec::numeric("x")],
        row_semantics: RowSemantics::Iid,
    };
    let numeric_table = RawTable::new(
        vec!["x".into()],
        ["", " na ", "NaN", "null", "inf", "-inf", "1", "3"]
            .into_iter()
            .map(|value| vec![value.to_string()])
            .collect(),
    )
    .unwrap();
    let target: Vec<f64> = (0..8).map(|row| row as f64).collect();
    for (mode, fill) in [("median", 2.0_f64), ("mean", 2.0_f64), ("zero", 0.0_f64)] {
        let input = fitted_input_json(
            &numeric_schema,
            &numeric_table,
            &target,
            &format!("(input numeric={mode} missing=indicator categorical=native)"),
        );
        let numeric = &input["columns"][0]["Numeric"];
        assert_eq!(numeric["fill"].as_f64().unwrap().to_bits(), fill.to_bits());
        assert_eq!(numeric["indicator"], true);
    }

    let categorical_schema = DataSchema {
        columns: vec![ColumnSpec::categorical("x")],
        row_semantics: RowSemantics::Iid,
    };
    let categorical_table = RawTable::new(
        vec!["x".into()],
        ["z", "", "nan", "a", "NA", "null", "nan", "z"]
            .into_iter()
            .map(|value| vec![value.to_string()])
            .collect(),
    )
    .unwrap();
    let input = fitted_input_json(
        &categorical_schema,
        &categorical_table,
        &target,
        "(input numeric=zero missing=none categorical=native)",
    );
    let categorical = &input["columns"][0]["Categorical"];
    assert_eq!(
        categorical["vocabulary"],
        serde_json::json!(["\u{0}MISSING", "a", "nan", "z"])
    );
    assert_eq!(categorical["counts"], serde_json::json!([3, 1, 2, 2]));
    assert!(categorical["vocabulary"]
        .as_array()
        .unwrap()
        .iter()
        .all(|value| value != "unseen"));
}

#[test]
fn binary_positive_class_is_exactly_one() {
    let schema = DataSchema {
        columns: vec![ColumnSpec::numeric("x")],
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::new(
        vec!["x".into()],
        (0..6).map(|row| vec![row.to_string()]).collect(),
    )
    .unwrap();
    let task = TaskSpec::binary(Objective::LogLoss).unwrap();
    let genome = GenomeIr::parse("(pipe X (input numeric=zero) (logit_ridge l2=0.1))").unwrap();
    assert!(fit_table(
        &genome,
        &task,
        &schema,
        &table,
        &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    )
    .is_ok());
    for invalid in [
        [0.0, 1.0, 0.5, 1.0, 0.0, 1.0],
        [0.0, 2.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, -1.0, 0.0, 1.0, 0.0, 1.0],
    ] {
        assert!(fit_table(&genome, &task, &schema, &table, &invalid).is_err());
    }
}
