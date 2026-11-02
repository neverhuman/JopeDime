use super::*;
use crate::advanced_v3::{
    fit_table, predict_table, ColumnSpec, DataSchema, Objective, RawTable, RowSemantics, TaskSpec,
};
#[test]
fn bound_python_control_unicode_json_is_valid_and_preserves_fitted_bits() {
    let table = RawTable::new(
        vec!["label\u{2028}\u{301}\u{7f}".into()],
        (0..12)
            .map(|i| {
                vec![format!(
                    "category-{}#![forbid(unsafe_code)]\u{2028}\u{301}\u{7f}",
                    i % 3
                )]
            })
            .collect(),
    )
    .unwrap();
    let schema = DataSchema {
        columns: vec![ColumnSpec::categorical(table.headers[0].clone())],
        row_semantics: RowSemantics::Iid,
    };
    let model = fit_table(
        &crate::GenomeIr::parse("(pipe X (input categorical=onehot) (ridge))").unwrap(),
        &TaskSpec::regression(Objective::Rmse).unwrap(),
        &schema,
        &table,
        &(0..12).map(|i| i as f64).collect::<Vec<_>>(),
    )
    .unwrap();
    if super::typescript_support::enabled() {
        let bundle =
            crate::typescript::emit_bound(&model, &crate::transpile::EmitOptions::default())
                .unwrap();
        let scratch = tempfile::tempdir().unwrap();
        super::typescript_support::run(
            scratch.path(),
            &bundle.source,
            &bundle.kernel,
            &serde_json::json!({"rows":table.rows}),
            &model.predict_table(&table).unwrap(),
            true,
        );
    }
    if super::go_support::enabled() {
        let bundle =
            crate::go::emit_bound(&model, &crate::transpile::EmitOptions::default()).unwrap();
        let scratch = tempfile::tempdir().unwrap();
        super::go_support::run(
            scratch.path(),
            &bundle.source,
            &bundle.kernel,
            &serde_json::json!({"rows":table.rows}),
            &model.predict_table(&table).unwrap(),
            true,
        );
    }
    let source = emit_bound(&model, &crate::transpile::EmitOptions::default())
        .unwrap()
        .python;
    let rows = serde_json::to_string(&table.rows).unwrap();
    let script=format!("{source}\nimport struct\nrows=json.loads({})\nfor row in rows:\n    print(struct.pack('>d',predict_numeric(encode(row)[0])).hex())\n",serde_json::to_string(&rows).unwrap());
    let mut child = std::process::Command::new("python3")
            .args([
                "-c",
                "import sys;exec(compile(sys.stdin.read(),'<fitted>','exec'), {'__name__':'fitted_test'})",
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
    use std::io::Write;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(script.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| u64::from_str_radix(line, 16).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        predict_table(&model, &table)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}
