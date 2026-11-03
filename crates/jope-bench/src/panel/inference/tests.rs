use super::super::data::raw_table;
use super::super::methods::{fit_rust, rust_methods, RustFitRequest};
use super::*;
use feat_math::advanced_v3::{ColumnSpec, DataSchema, RowSemantics};
use jope_pipeline::experiment::{Column, ColumnKind as Kind};
use std::process::Command;
pub(crate) fn fixture(task: Task, family: &str) -> (FittedRustArtifact, Vec<Vec<Option<String>>>) {
    let (request, rows) = fixture_request(task, family);
    (fit_rust(&request).unwrap().artifact, rows)
}
pub(crate) fn fixture_request(
    task: Task,
    family: &str,
) -> (RustFitRequest, Vec<Vec<Option<String>>>) {
    let columns = vec![
        Column {
            name: "amount".into(),
            kind: Kind::Numeric,
        },
        Column {
            name: "country".into(),
            kind: Kind::Categorical,
        },
    ];
    let categories = [
        None,
        Some("NA"),
        Some("null"),
        Some(""),
        Some("line\n\"quoted\"\u{2028}"),
        Some("__PREDICT____REGRESSION____CLASSES_JSON__"),
    ];
    let rows = (0..18)
        .map(|i| {
            vec![
                if i % 7 == 0 {
                    None
                } else {
                    Some(i.to_string())
                },
                categories[i % categories.len()].map(str::to_owned),
            ]
        })
        .collect::<Vec<_>>();
    let prediction = vec![
        vec![None, Some("NA".into())],
        vec![Some("1000000".into()), Some("unseen".into())],
        vec![Some("-0.0".into()), None],
        vec![Some("3.25".into()), Some("line\n\"quoted\"\u{2028}".into())],
    ];
    let classes = match task {
        Task::Regression => vec![],
        Task::Binary => vec!["tree\nquoted\"".into(), "search".into()],
        _ => vec!["zebra".into(), "alpha".into(), "middle".into()],
    };
    let targets = (0..18)
        .map(|i| {
            if task == Task::Regression {
                (i as f64) * 1.25 - 3.0
            } else {
                (i % classes.len()) as f64
            }
        })
        .collect();
    let request = RustFitRequest {
        task,
        schema: DataSchema {
            columns: vec![
                ColumnSpec::numeric("amount"),
                ColumnSpec::categorical("country"),
            ],
            row_semantics: RowSemantics::Iid,
        },
        classes,
        train: raw_table(&columns, &rows).unwrap(),
        targets,
        predict: raw_table(&columns, &prediction).unwrap(),
        method: rust_methods(task)
            .unwrap()
            .into_iter()
            .find(|method| method.family == family)
            .unwrap(),
    };
    (request, prediction)
}
pub(crate) fn emitted(artifact: &FittedRustArtifact, backend: Backend) -> PublicBundle {
    match export_fitted(artifact, backend).unwrap() {
        ExportOutcome::Emitted { bundle } => *bundle,
        ExportOutcome::Unsupported { reason, .. } => panic!("{backend:?}: {reason:?}"),
    }
}

#[path = "tests/advanced.rs"]
mod advanced;
#[path = "tests/basic.rs"]
mod basic;
