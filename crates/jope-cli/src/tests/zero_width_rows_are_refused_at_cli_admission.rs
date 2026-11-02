use super::*;

#[test]
fn zero_width_rows_are_refused_at_cli_admission() {
    let mut data = named(Task::Regression, "linear");
    data.columns.clear();
    for row in &mut data.rows {
        row.features.clear();
    }
    let error = validate_dataset(&data).unwrap_err();
    assert!(error.to_string().contains("unsupported feature width"));
}
