use super::super::{
    fit_table, ColumnSpec, DataSchema, Objective, RawTable, RowSemantics, TaskSpec,
};
use super::*;
use crate::GenomeIr;
use std::collections::BTreeMap;
#[test]
fn fitted_portable_refuses_unimplemented_effective_state() {
    let schema = DataSchema {
        columns: vec![ColumnSpec::numeric("amount")],
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::new(
        vec!["amount".into()],
        (0..12).map(|i| vec![i.to_string()]).collect(),
    )
    .unwrap();
    let targets = (0..12).map(|i| i as f64).collect::<Vec<_>>();
    for source in [
        "(pipe X (input numeric=median categorical=onehot) (standardize) (ridge))",
        "(pipe X (input numeric=median categorical=onehot) (histgbm))",
    ] {
        let model = fit_table(
            &GenomeIr::parse(source).unwrap(),
            &TaskSpec::regression(Objective::Rmse).unwrap(),
            &schema,
            &table,
            &targets,
        )
        .unwrap();
        assert!(emit_fitted_portable(&model).is_err(), "{source}");
    }
    let binary = (0..12).map(|i| (i % 2) as f64).collect::<Vec<_>>();
    let mut model = fit_table(
        &GenomeIr::parse("(pipe X (input numeric=median categorical=onehot) (logit_ridge))")
            .unwrap(),
        &TaskSpec::binary(Objective::RocAuc).unwrap(),
        &schema,
        &table,
        &binary,
    )
    .unwrap();
    if let BoundPredictor::Pipeline { calibration, .. } = &mut model.predictor {
        *calibration = CalibrationModel::Platt {
            slope: 1.0,
            intercept: 0.0,
        };
    }
    assert!(emit_fitted_portable(&model).is_err());
}
#[test]
fn public_onehot_state_excludes_unused_categories_and_label_statistics() {
    let schema = DataSchema {
        columns: vec![ColumnSpec::categorical("island")],
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::new(
        vec!["island".into()],
        (0..12)
            .map(|i| {
                vec![if i == 0 {
                    "PRIVATE_CATEGORY_CANARY".into()
                } else {
                    "tree".into()
                }]
            })
            .collect(),
    )
    .unwrap();
    let model = fit_table(
        &GenomeIr::parse("(pipe X (input numeric=median categorical=onehot max_onehot=1) (ridge))")
            .unwrap(),
        &TaskSpec::regression(Objective::Rmse).unwrap(),
        &schema,
        &table,
        &(0..12).map(|i| i as f64 + 1234.56789).collect::<Vec<_>>(),
    )
    .unwrap();
    let portable = emit_fitted_portable(&model).unwrap();
    let cpp = emit_fitted_public_cpp17(&model).unwrap();
    for text in [
        &portable.python,
        &portable.rust,
        portable.javascript.as_ref().unwrap(),
        &cpp.source,
        &cpp.codec,
    ] {
        assert!(!text.contains("PRIVATE_CATEGORY_CANARY"));
        for key in [
            "\"sums\"",
            "\"counts\"",
            "\"global\"",
            "schema_sha256",
            "vocabulary_sha256",
        ] {
            assert!(!text.contains(key), "{key}");
        }
        assert!(lint_fitted_public_text(&model, &[], text).is_empty());
    }
}
#[test]
fn branch_heads_preserve_missing_routes_in_python_and_rust() {
    use super::super::heads::{BoostStump, ExtraNode};
    use std::process::Command;
    let schema = DataSchema {
        columns: vec![ColumnSpec::numeric("amount")],
        row_semantics: RowSemantics::Iid,
    };
    let train = RawTable::new(
        vec!["amount".into()],
        (0..8).map(|i| vec![i.to_string()]).collect(),
    )
    .unwrap();
    let predict = RawTable::new(
        vec!["amount".into()],
        vec![vec![String::new()], vec!["0".into()]],
    )
    .unwrap();
    let scratch = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/fitted-inference-tests");
    std::fs::create_dir_all(&scratch).unwrap();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = scratch.join(format!("missing-routes-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    for name in ["tab_boost", "extra_trees"] {
        let mut model = fit_table(
            &GenomeIr::parse(&format!(
                "(pipe X (input numeric=median missing=indicator categorical=onehot) ({name}))"
            ))
            .unwrap(),
            &TaskSpec::regression(Objective::Rmse).unwrap(),
            &schema,
            &train,
            &(0..8).map(|v| v as f64).collect::<Vec<_>>(),
        )
        .unwrap();
        let BoundPredictor::Pipeline {
            head,
            missing_aware,
            ..
        } = &mut model.predictor
        else {
            panic!("pipeline")
        };
        assert!(*missing_aware);
        // Explicit synthetic coefficients make missing routing observable even
        // though the finite median falls to the opposite side of the split.
        match head {
            V3Head::TabBoost(m) => {
                m.base = 0.0;
                m.learning_rate = 1.0;
                m.stumps = vec![BoostStump {
                    feature: 0,
                    threshold: 100.0,
                    missing_left: false,
                    left: -2.0,
                    right: 2.0,
                }];
            }
            V3Head::ExtraTrees(m) => {
                m.trees = vec![ExtraNode::Split {
                    feature: 0,
                    threshold: 100.0,
                    missing_left: false,
                    left: Box::new(ExtraNode::Leaf(-2.0)),
                    right: Box::new(ExtraNode::Leaf(2.0)),
                }];
            }
            _ => panic!("unexpected head"),
        }
        let expected = model.predict_table(&predict).unwrap();
        assert_eq!(expected, vec![2.0, -2.0]);
        assert_eq!(fitted_portable_unsupported(&model).unwrap(), None);
        assert_eq!(
            fitted_javascript_unsupported(&model).unwrap(),
            Some(PortableUnsupported::Head)
        );
        assert_eq!(
            fitted_public_cpp_unsupported(&model).unwrap(),
            Some(PortableUnsupported::Head)
        );
        let emitted = emit_fitted_portable(&model).unwrap();
        let folder = root.join(name);
        std::fs::create_dir(&folder).unwrap();
        std::fs::write(folder.join("model.py"), &emitted.python).unwrap();
        let run=Command::new("python3").arg("-c").arg("import model;print(','.join(str(r[0]) for r in model.predict_encoded([[''],['0']])))").current_dir(&folder).output().unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let actual = String::from_utf8(run.stdout)
            .unwrap()
            .trim()
            .split(',')
            .map(|s| s.parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        std::fs::write(folder.join("model.rs"), &emitted.rust).unwrap();
        std::fs::write(folder.join("main.rs"),"mod model;fn main(){let v=model::predict_encoded(&[vec![String::new()],vec![\"0\".into()]]).unwrap();println!(\"{},{}\",v[0][0].to_bits(),v[1][0].to_bits());}").unwrap();
        let build = Command::new("rustc")
            .args(["--edition=2021", "-O", "main.rs", "-o", "predict"])
            .current_dir(&folder)
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
        let run = Command::new(folder.join("predict")).output().unwrap();
        assert!(run.status.success());
        let actual = String::from_utf8(run.stdout)
            .unwrap()
            .trim()
            .split(',')
            .map(|s| s.parse::<u64>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // The ordinary filled-value path remains faithful if this stored flag
        // is false; no category or missing-indicator mask is ever restored.
        if let BoundPredictor::Pipeline { missing_aware, .. } = &mut model.predictor {
            *missing_aware = false;
        }
        assert_eq!(model.predict_table(&predict).unwrap(), vec![-2.0, -2.0]);
    }
}
#[test]
fn fitted_portable_template_bytes_are_pinned_for_version_three() {
    assert_eq!(FITTED_PORTABLE_EMITTER, "fitted-portable/3");
    let expected = [
        (
            "matrix.py".to_owned(),
            "a09b906608e6e25648f28d93b1a4f31b09cbc4b5a8cd395888734de23d657072".to_owned(),
        ),
        (
            "matrix.js".to_owned(),
            "519e5228f6611e7e351a96b9f7f4d4750038b7c86762c719467c2a2468eaa7c8".to_owned(),
        ),
        (
            "matrix.rs".to_owned(),
            "c7bce8cdc4f5ac3293814d4d8b2afe76a75ba3772926a0ceaa3593f993bdba54".to_owned(),
        ),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    assert_eq!(fitted_portable_templates(), expected);
}
