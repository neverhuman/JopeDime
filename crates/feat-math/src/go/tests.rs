use super::*;
use crate::kernel::registry_v2::OPERATORS;
use crate::{fit_predict, GenomeIr};
use std::collections::BTreeSet;
use std::path::Path;

mod support {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/go_support/mod.rs"
    ));
}

fn data() -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let x = (0..32)
        .map(|i| {
            let v = i as f64 / 8.0 - 2.0;
            vec![v, (v * 0.7).sin(), (i % 5) as f64 - 2.0, v * v, 1.0]
        })
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|r| r[0] * 0.8 - r[1] * 0.3 + r[2] * 0.1)
        .collect();
    let probe = vec![x[3].clone(), x[17].clone(), vec![0.0; 5]];
    (x, y, probe)
}

fn recipe(root: &Path, source: &str) {
    let (x, y, probe) = data();
    let genome = GenomeIr::parse(source).unwrap();
    let expected = fit_predict(&genome, &x, &y, &probe).unwrap();
    let bundle = emit(&genome, &EmitOptions::default()).unwrap();
    support::run(
        root,
        &bundle.source,
        &bundle.kernel,
        &serde_json::json!({"train":x,"targets":y,"test":probe}),
        &expected,
        false,
    );
}

/// This denominator comes from the same registry used to admit publication.
/// Every Rust recipe spelling either executes through both TS module modes or
/// is explicitly refused by both emitters. New registry rows enter automatically.
#[test]
fn complete_recipe_registry_executes_or_refuses_identically() {
    if !support::enabled() {
        return;
    }
    let root = std::env::current_dir()
        .unwrap()
        .join("../../target/go-census/recipe");
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
        eprintln!("Go registry recipe: {} executed", row.name);
    }
    assert!(!executed.is_empty());
    assert_eq!(executed.len() + refused.len(), rows.len());
    eprintln!(
        "Go census: {}",
        serde_json::json!({"kind":"recipe","total":rows.len(),"executed":executed,"refused":refused,"unaccounted":0})
    );
}

#[test]
fn complete_bound_registry_and_codec_modes_execute() {
    use crate::advanced_v3::{
        fit_table, ColumnSpec, DataSchema, Objective, RawTable, RowSemantics, TaskSpec,
    };
    if !support::enabled() {
        return;
    }
    let root = std::env::current_dir()
        .unwrap()
        .join("../../target/go-census/bound");
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
        let rust = crate::advanced_v3::emit_bound(&model, &EmitOptions::default());
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
            &bundle.source,
            &bundle.kernel,
            &serde_json::json!({"rows":probe.rows}),
            &expected,
            true,
        );
        completed.insert(row.name);
        eprintln!("Go registry bound: {} executed", row.name);
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
            &bundle.source,
            &bundle.kernel,
            &serde_json::json!({"rows":probe.rows}),
            &model.predict_table(&probe).unwrap(),
            true,
        );
    }
    eprintln!(
        "Go census: {}",
        serde_json::json!({"kind":"bound","total":rows.len(),"executed":completed,"refused":refused,"unaccounted":0})
    );
}

#[test]
fn scalar_v2_functions_policies_and_carriers_and_row_vector_stages_execute() {
    use crate::kernel::numerical_policy::NumericalPolicy;
    use crate::kernel::ordered_axis::Axis;
    use crate::kernel::row_vector::{Program, RowStage};
    use crate::kernel::scalar_program::{
        ScalarCarrier, ScalarExpressionProgram, ScalarFunction, ScalarNode,
    };
    use crate::math_expr::Var;
    if !support::enabled() {
        return;
    }
    let root = std::env::current_dir()
        .unwrap()
        .join("../../target/go-census/v2");
    let (x, y, probe) = data();
    let mut count = 0;
    let functions = (0..=u16::MAX)
        .filter_map(|id| {
            ScalarFunction::from_versioned_id(2, id)
                .ok()
                .map(|f| (id, f))
        })
        .collect::<Vec<_>>();
    assert!(!functions.is_empty());
    for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
        for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
            for &(id, function) in &functions {
                let mut args = vec![ScalarNode::Var(if carrier == ScalarCarrier::Row {
                    Var::Col(0)
                } else {
                    Var::X
                })];
                if matches!(id, 7..=10) {
                    args.push(ScalarNode::Const(-0.5));
                }
                if id == 9 {
                    args.push(ScalarNode::Const(0.5));
                }
                let program = ScalarExpressionProgram::new(ScalarNode::Call(function, args))
                    .unwrap()
                    .with_numerical_policy(policy);
                let model = program.fit_training(&x, &y, carrier, 0.02).unwrap();
                let expected = model.predict(&probe).unwrap().0;
                let bundle = emit_scalar(&program, carrier, 5, 0.02).unwrap();
                support::run(
                    &root.join(format!("scalar-{policy:?}-{carrier:?}-{id}")),
                    &bundle.source,
                    &bundle.kernel,
                    &serde_json::json!({"train":x,"targets":y,"test":probe}),
                    &expected,
                    false,
                );
                count += 1;
            }
        }
    }
    for (index, stages) in [
        vec![],
        vec![RowStage::Savgol(
            crate::kernel::savgol::Config::new(2, 58, Axis::FeatureWithinRow, 5, 2, 1, "shrink")
                .unwrap(),
        )],
        vec![RowStage::Goertzel(
            crate::kernel::goertzel::Config::new(2, 61, Axis::FeatureWithinRow, &[0.0, 0.1, 0.5])
                .unwrap(),
        )],
    ]
    .into_iter()
    .enumerate()
    {
        let program = Program::new(2, 5, stages, 0.02).unwrap();
        let expected = program.fit(&x, &y).unwrap().predict(&probe).unwrap().0;
        let bundle = emit_row_vector(&program).unwrap();
        support::run(
            &root.join(format!("row-{index}")),
            &bundle.source,
            &bundle.kernel,
            &serde_json::json!({"train":x,"targets":y,"test":probe}),
            &expected,
            false,
        );
        count += 1;
    }
    assert_eq!(count, functions.len() * 2 * 2 + 3);
    eprintln!(
        "Go census: {}",
        serde_json::json!({"kind":"v2","total":count,"executed":count,"unaccounted":0})
    );
}

#[test]
fn fitted_legacy_registry_states_execute_or_match_rust_refusal() {
    use crate::advanced_v3::{
        fit_legacy_numeric_bound, fit_table, ColumnSpec, DataSchema, Objective, RawTable,
        RowSemantics, TaskSpec,
    };
    if !support::enabled() {
        return;
    }
    let root = std::env::current_dir()
        .unwrap()
        .join("../../target/go-census/bound");
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
        assert!(crate::advanced_v3::emit_bound(&legacy, &EmitOptions::default()).is_err());
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
        let rust = crate::advanced_v3::emit_bound(&model, &EmitOptions::default());
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
            &bundle.source,
            &bundle.kernel,
            &serde_json::json!({"rows":table.rows}),
            &model.predict_table(&table).unwrap(),
            true,
        );
        executed.insert(row.name);
    }
    assert!(!executed.is_empty());
    assert_eq!(executed.len() + refused.len(), rows.len());
    eprintln!(
        "Go census: {}",
        serde_json::json!({"kind":"legacy-bound","total":rows.len(),"executed":executed,"refused":refused,"unaccounted":0})
    );
}

#[test]
fn multiline_metadata_remains_documentation() {
    if !support::enabled() {
        return;
    }
    let (x, y, probe) = data();
    let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let options = EmitOptions {
        title: Some("Title\r\nSecond line\u{2028}third line".into()),
        ..EmitOptions::default()
    };
    let bundle = emit(&genome, &options).unwrap();
    let root = std::env::current_dir()
        .unwrap()
        .join("../../target/go-metadata");
    support::run(
        &root,
        &bundle.source,
        &bundle.kernel,
        &serde_json::json!({"train":x,"targets":y,"test":probe}),
        &fit_predict(&genome, &x, &y, &probe).unwrap(),
        false,
    );
    eprintln!("Go metadata: strict heading qualified");
}
