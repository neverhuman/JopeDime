//! Per-column descriptive statistics and non-selecting target suggestions.

use jope_runtime::table::{ColumnKind, CsvTable};
use std::collections::BTreeSet;

/// Maximum number of distinct nonmissing values retained for each column.
pub const DISTINCT_CAP: usize = 10_000;

/// A bounded profile of a parsed dataset.
#[derive(Clone, Debug, PartialEq)]
pub struct DatasetProfile {
    pub rows: usize,
    pub columns: Vec<ColumnProfile>,
    pub suggested_targets: Vec<String>,
}

/// Descriptive statistics for one parsed dataset column.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnProfile {
    pub name: String,
    pub kind: ColumnKind,
    pub missing_rate: f64,
    pub distinct_count: usize,
    pub distinct_capped: bool,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

/// Profile the already parsed table used by dataset registration.
///
/// Missing cells and numeric types follow `jope_runtime::table::parse_csv`: only `None` cells are
/// missing, and a column is numeric exactly when every present value parses as a finite `f64`.
/// This function does not parse or normalize cell text.
pub fn profile(table: &CsvTable) -> DatasetProfile {
    let rows = table.rows.len();
    let columns = table
        .headers
        .iter()
        .enumerate()
        .map(|(index, name)| profile_column(name, index, table))
        .collect();

    DatasetProfile {
        rows,
        columns,
        suggested_targets: suggest_targets(&table.headers),
    }
}

fn profile_column(name: &str, index: usize, table: &CsvTable) -> ColumnProfile {
    let values = table.rows.iter().filter_map(|row| row[index].as_deref());
    let kind = if values.clone().all(|value| numeric(value).is_some()) {
        ColumnKind::Numeric
    } else {
        ColumnKind::Categorical
    };

    let present = table.rows.iter().filter(|row| row[index].is_some()).count();
    let missing_rate = if table.rows.is_empty() {
        0.0
    } else {
        (table.rows.len() - present) as f64 / table.rows.len() as f64
    };

    let (distinct_count, distinct_capped) =
        distinct_count(table.rows.iter().filter_map(|row| row[index].as_deref()));

    let (min, max) = if kind == ColumnKind::Numeric {
        let mut min: Option<f64> = None;
        let mut max: Option<f64> = None;
        for value in table.rows.iter().filter_map(|row| row[index].as_deref()) {
            if let Some(value) = numeric(value) {
                min = Some(min.map_or(value, |current| current.min(value)));
                max = Some(max.map_or(value, |current| current.max(value)));
            }
        }
        (min, max)
    } else {
        (None, None)
    };

    ColumnProfile {
        name: name.to_owned(),
        kind,
        missing_rate,
        distinct_count,
        distinct_capped,
        min,
        max,
    }
}

fn numeric(value: &str) -> Option<f64> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
}

fn distinct_count<'a>(values: impl Iterator<Item = &'a str>) -> (usize, bool) {
    let mut distinct = BTreeSet::new();
    for value in values {
        if distinct.contains(value) {
            continue;
        }
        if distinct.len() == DISTINCT_CAP {
            return (DISTINCT_CAP, true);
        }
        distinct.insert(value);
    }
    (distinct.len(), false)
}

fn suggest_targets(headers: &[String]) -> Vec<String> {
    const TARGET_NAMES: &[&str] = &["target", "label", "y", "class", "outcome", "price"];

    let mut suggestions = Vec::new();
    for expected in TARGET_NAMES {
        for header in headers
            .iter()
            .filter(|header| header.eq_ignore_ascii_case(expected))
        {
            suggestions.push(header.clone());
        }
    }
    if let Some(last) = headers.last() {
        if !suggestions
            .iter()
            .any(|suggestion| suggestion.eq_ignore_ascii_case(last))
        {
            suggestions.push(last.clone());
        }
    }
    suggestions
}

#[cfg(test)]
mod tests {
    use super::{profile, ColumnProfile, DatasetProfile, DISTINCT_CAP};
    use jope_runtime::{limits::Limits, table};
    use std::path::PathBuf;

    fn fixture(relative_path: &str) -> DatasetProfile {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("test-support/datasets/v1")
            .join(relative_path);
        let mut bytes = std::fs::read(&path).expect("QLT-05 fixture is available");
        if let Some(metadata_path) = path.parent().map(|parent| parent.join("meta.json")) {
            if metadata_path.exists() {
                let metadata: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(metadata_path).expect("QLT-05 fixture metadata is available"),
                )
                .expect("QLT-05 fixture metadata parses");
                let headers = metadata["output_headers"]
                    .as_array()
                    .expect("fixture metadata contains output headers");
                let mut header = String::new();
                for (index, value) in headers.iter().enumerate() {
                    if index > 0 {
                        header.push(',');
                    }
                    let name = value.as_str().expect("column name is a string");
                    header.push('"');
                    header.push_str(&name.replace('"', "\"\""));
                    header.push('"');
                }
                header.push('\n');
                header.push_str(std::str::from_utf8(&bytes).expect("fixture is UTF-8"));
                bytes = header.into_bytes();
            }
        }
        let parsed = table::parse_csv(&bytes, &Limits::default()).expect("fixture parses");
        profile(&parsed)
    }

    fn column<'a>(dataset: &'a DatasetProfile, name: &str) -> &'a ColumnProfile {
        dataset
            .columns
            .iter()
            .find(|column| column.name == name)
            .expect("expected profile column")
    }

    #[test]
    fn qlt05_fixture_shapes_and_target_suggestions_match_parsed_files() {
        let cases = [
            (
                "binary-loanapp/train.csv",
                1591,
                54,
                &["price", "approve"][..],
                "approve",
                2,
                0.0,
                1.0,
            ),
            (
                "d2-regression/train.csv",
                1000,
                22,
                &["target"][..],
                "target",
                674,
                0.0,
                0.996683896,
            ),
            (
                "grouped-apartment-rent/train.csv",
                7998,
                12,
                &["price"][..],
                "price",
                1558,
                0.0,
                1.0,
            ),
            (
                "time-ordered-prisonlf/train.csv",
                1290,
                7,
                &["count"][..],
                "count",
                722,
                0.0,
                1.0,
            ),
            (
                "tiny-30-row-table/train.csv",
                25,
                13,
                &["target"][..],
                "target",
                25,
                0.0,
                1.0,
            ),
            (
                "messy-csv-like/raw-source-subset.csv",
                1000,
                35,
                &["target"][..],
                "target",
                664,
                1.0,
                300_000_000.0,
            ),
        ];

        for (path, rows, columns, suggestions, target_name, distinct, min, max) in cases {
            let result = fixture(path);
            assert_eq!(result.rows, rows, "{path}");
            assert_eq!(result.columns.len(), columns, "{path}");
            assert_eq!(
                result
                    .suggested_targets
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                suggestions,
                "{path}"
            );
            let target = column(&result, target_name);
            assert_eq!(
                target.kind,
                jope_runtime::table::ColumnKind::Numeric,
                "{path}"
            );
            assert_eq!(target.missing_rate, 0.0, "{path}");
            assert_eq!(target.distinct_count, distinct, "{path}");
            assert!(!target.distinct_capped, "{path}");
            assert_eq!(target.min, Some(min), "{path}");
            assert_eq!(target.max, Some(max), "{path}");
        }
    }

    #[test]
    fn tiny_qlt05_fixture_matches_hand_checked_profile() {
        let result = fixture("tiny-30-row-table/train.csv");
        let first = column(&result, "pH_target");
        assert_eq!(first.kind, jope_runtime::table::ColumnKind::Numeric);
        assert_eq!(first.missing_rate, 0.0);
        assert_eq!(first.distinct_count, 25);
        assert!(!first.distinct_capped);
        assert_eq!(first.min, Some(0.102763385));
        assert_eq!(first.max, Some(1.0));
        assert_eq!(result.suggested_targets, ["target"]);
    }

    #[test]
    fn messy_qlt05_fixture_reports_missing_values_and_named_target() {
        let result = fixture("messy-csv-like/raw-source-subset.csv");
        let subtype = column(&result, "Disaster Subtype");
        assert_eq!(subtype.kind, jope_runtime::table::ColumnKind::Numeric);
        assert_eq!(subtype.missing_rate, 182.0 / 1000.0);
        assert_eq!(subtype.distinct_count, 22);
        assert_eq!(subtype.min, Some(0.0));
        assert_eq!(subtype.max, Some(24.0));

        let target = column(&result, "target");
        assert_eq!(target.missing_rate, 0.0);
        assert_eq!(target.distinct_count, 664);
        assert_eq!(target.min, Some(1.0));
        assert_eq!(target.max, Some(300_000_000.0));
        assert_eq!(result.suggested_targets, ["target"]);
    }

    #[test]
    fn target_name_priority_is_case_insensitive_and_never_selects() {
        let parsed = table::parse_csv(
            b"Price,label,TARGET,target,y,outcome,class,other\n1,a,2,3,4,5,6,x\n",
            &Limits::default(),
        )
        .expect("small table parses");
        let result = profile(&parsed);
        assert_eq!(
            result.suggested_targets,
            ["TARGET", "target", "label", "y", "class", "outcome", "Price", "other"]
        );
    }

    #[test]
    fn distinct_count_stops_at_cap_and_marks_truncation() {
        let rows = (0..=DISTINCT_CAP)
            .map(|value| vec![Some(value.to_string())])
            .collect();
        let parsed = jope_runtime::table::CsvTable {
            headers: vec!["value".into()],
            rows,
            sha256: String::new(),
            bytes: 0,
        };
        let result = profile(&parsed);
        let value = column(&result, "value");
        assert_eq!(value.distinct_count, DISTINCT_CAP);
        assert!(value.distinct_capped);
    }

    #[test]
    fn empty_columns_have_zero_missing_rate_and_no_numeric_bounds() {
        let parsed = jope_runtime::table::CsvTable {
            headers: vec!["empty".into()],
            rows: Vec::new(),
            sha256: String::new(),
            bytes: 0,
        };
        let result = profile(&parsed);
        let empty = column(&result, "empty");
        assert_eq!(empty.missing_rate, 0.0);
        assert_eq!(empty.min, None);
        assert_eq!(empty.max, None);
    }
}
