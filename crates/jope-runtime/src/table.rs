//! Local table ingress (SPECIFICATION.md §13.1).
//!
//! The raw-cell encoding ([`encode_cell`], [`raw_table`]) and the fitting-input digest moved here
//! unchanged from `jope_pipeline::prepared_data`, which re-exports them: the appliance hands
//! `feat_math` exactly the cells the frozen panel does.
//!
//! The CSV contract, stated once:
//!
//! * The first record is the header. Names must be non-empty and unique; a UTF-8 byte-order mark
//!   is ignored. Every record has the header's width.
//! * An empty cell (after trimming ASCII whitespace) is missing; it is the only missing marker.
//!   Literal `NA`, `null` or `?` are values, which keeps categories injective.
//! * The target column is named explicitly and never used as a feature. A missing target fails.
//! * A feature column is numeric when every present cell parses as a finite number, otherwise
//!   categorical. Columns keep their header order. Constant and all-missing columns are kept; the
//!   input stage's median/indicator policy handles them.
//! * `auto` task: exactly two distinct target labels is binary; otherwise all-numeric targets are
//!   regression and anything else is multiclass. Integer-coded classes with more than two labels
//!   need an explicit `--task multiclass`.
//! * Class order is numeric when every label is a number, otherwise byte-lexicographic; for binary
//!   tasks the second class is the positive one.
//! * Prediction input matches feature columns by name in any order. A column the model does not
//!   know is refused, except the training target's name, which is ignored and reported.
//! * *Auxiliary* columns named by a regression objective (prices, unit costs, exposures) must be
//!   numeric and present in every training row. They score held-out predictions only: they are
//!   never features, and prediction input may carry them (they are ignored and reported).
use crate::limits::Limits;
use crate::task::{TaskChoice, TaskKind};
use crate::Failure;
use anyhow::{ensure, Context, Result};
use feat_math::advanced_v3::{ColumnSpec, DataSchema, RawTable, RowSemantics};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub use feat_math::advanced_v3::ColumnKind;

/// Fitted envelopes must retain this policy and replay `raw_table` at inference.
pub const RAW_CELL_ENCODING: &str = "jope.raw-cell-encoding/v1";
pub const CATEGORY_PREFIX: &str = "jope.category.v1:";

/// Encode one nullable cell. Every present category is namespaced, including literal
/// null/NA/empty, so RawTable's missing-token vocabulary cannot collapse it.
pub fn encode_cell(kind: ColumnKind, value: Option<&str>) -> Result<String> {
    let Some(value) = value else {
        return Ok(String::new());
    };
    match kind {
        ColumnKind::Numeric => {
            let number: f64 = value.trim().parse().context("invalid numeric feature")?;
            ensure!(
                number.is_finite(),
                "numeric features must be finite or null"
            );
            Ok(value.to_owned())
        }
        ColumnKind::Categorical => Ok(format!(
            "{CATEGORY_PREFIX}{}",
            serde_json::to_string(value)?
        )),
    }
}

/// Reusable label-free preprocessing for fitting, workers and fitted inference.
/// `None` is the only missing cell; original categorical strings are injective.
pub fn raw_table(schema: &DataSchema, values: &[Vec<Option<String>>]) -> Result<RawTable> {
    let rows = values
        .iter()
        .map(|row| {
            ensure!(row.len() == schema.columns.len(), "ragged raw feature row");
            row.iter()
                .zip(&schema.columns)
                .map(|(value, column)| encode_cell(column.kind, value.as_deref()))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let table = RawTable::new(
        schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect(),
        rows,
    )?;
    table.validate_against(schema)?;
    Ok(table)
}

/// The schema of independent rows over `columns`.
pub fn iid_schema(columns: Vec<ColumnSpec>) -> Result<DataSchema> {
    let schema = DataSchema {
        columns,
        row_semantics: RowSemantics::Iid,
    };
    schema.validate()?;
    Ok(schema)
}

/// W3-T04: the schema of [`fitting_input_sha256`].
pub const FITTING_INPUT: &str = "jope.fitting-input/v1";

/// W3-T04: the digest a fitted artifact binds its fitting input by: the exact training table and
/// targets, in row order, that a fit worker receives. A struct rather than a JSON map, so the
/// byte order never depends on serde_json features.
pub fn fitting_input_sha256(table: &RawTable, targets: &[f64]) -> Result<String> {
    #[derive(Serialize)]
    struct FittingInput<'a> {
        schema: &'static str,
        table: &'a RawTable,
        targets: &'a [f64],
    }
    ensure!(
        table.rows.len() == targets.len() && targets.iter().all(|t| t.is_finite()),
        "fitting input targets differ from its training rows"
    );
    Ok(crate::sha256_hex(&serde_json::to_vec(&FittingInput {
        schema: FITTING_INPUT,
        table,
        targets,
    })?))
}

/// A parsed CSV file: header and nullable cells.
#[derive(Clone, Debug)]
pub struct CsvTable {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<Option<String>>>,
    pub sha256: String,
    pub bytes: u64,
}

/// Read and bound a CSV file before any per-row allocation beyond the limits.
pub fn read_csv(path: &Path, limits: &Limits) -> Result<CsvTable> {
    let metadata =
        std::fs::metadata(path).map_err(|e| Failure::input(format!("{}: {e}", path.display())))?;
    if !metadata.is_file() {
        return Err(Failure::input(format!("{} is not a file", path.display())));
    }
    if metadata.len() > limits.max_input_bytes {
        return Err(Failure::input(format!(
            "{} is {} bytes; the input limit is {}",
            path.display(),
            metadata.len(),
            limits.max_input_bytes
        )));
    }
    let bytes =
        std::fs::read(path).map_err(|e| Failure::input(format!("{}: {e}", path.display())))?;
    parse_csv(&bytes, limits).with_context(|| path.display().to_string())
}

/// Parse CSV bytes under the supplied input, row, and column limits.
///
/// A UTF-8 BOM is accepted and removed before parsing. The header is required and must contain
/// non-empty column names; malformed CSV and inputs exceeding `limits` return an input error.
pub fn parse_csv(bytes: &[u8], limits: &Limits) -> Result<CsvTable> {
    let body = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_reader(body);
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| Failure::input(format!("CSV header: {e}")))?
        .iter()
        .map(|h| h.trim().to_owned())
        .collect();
    if headers.is_empty() || headers.iter().any(String::is_empty) {
        return Err(Failure::input("CSV header has an empty column name"));
    }
    if headers.len() > limits.max_columns {
        return Err(Failure::input(format!(
            "{} columns; the limit is {}",
            headers.len(),
            limits.max_columns
        )));
    }
    let mut seen = BTreeSet::new();
    if let Some(duplicate) = headers.iter().find(|h| !seen.insert(h.as_str())) {
        return Err(Failure::input(format!("duplicate column {duplicate:?}")));
    }
    let mut rows = Vec::new();
    for (index, record) in reader.records().enumerate() {
        let record =
            record.map_err(|e| Failure::input(format!("CSV record {}: {e}", index + 2)))?;
        if rows.len() == limits.max_rows {
            return Err(Failure::input(format!(
                "more than {} rows; raise --max-rows to accept more",
                limits.max_rows
            )));
        }
        rows.push(
            record
                .iter()
                .map(|cell| {
                    let trimmed = cell.trim_matches(|c: char| c.is_ascii_whitespace());
                    (!trimmed.is_empty()).then(|| cell.to_owned())
                })
                .collect(),
        );
    }
    Ok(CsvTable {
        headers,
        rows,
        sha256: crate::sha256_hex(bytes),
        bytes: bytes.len() as u64,
    })
}

fn is_number(text: &str) -> bool {
    text.trim().parse::<f64>().is_ok_and(f64::is_finite)
}

/// The training table, its schema, the task and the encoded targets.
#[derive(Clone, Debug)]
pub struct Training {
    pub kind: TaskKind,
    /// How the task was chosen, for the receipt.
    pub task_reason: String,
    pub target: String,
    pub schema: DataSchema,
    pub classes: Vec<String>,
    pub table: RawTable,
    pub targets: Vec<f64>,
    /// Numeric columns read only by the selection objective, aligned with `targets`.
    pub auxiliary: crate::fitness::Columns,
}

/// Split the target from a CSV table, infer feature kinds and the task, and encode both.
pub fn training(
    csv: &CsvTable,
    target: &str,
    task: TaskChoice,
    limits: &Limits,
) -> Result<Training> {
    training_with_auxiliary(csv, target, task, limits, &BTreeSet::new())
}

/// [`training`], with `auxiliary` columns held out of the features for the objective.
pub fn training_with_auxiliary(
    csv: &CsvTable,
    target: &str,
    task: TaskChoice,
    limits: &Limits,
    auxiliary: &BTreeSet<String>,
) -> Result<Training> {
    let target_index = csv
        .headers
        .iter()
        .position(|h| h == target)
        .ok_or_else(|| Failure::input(format!("no target column {target:?} in the header")))?;
    if auxiliary.contains(target) {
        return Err(Failure::input(format!(
            "the target {target:?} cannot also be an objective column; use y in the expression"
        )));
    }
    let mut auxiliary_values = crate::fitness::Columns::new();
    for name in auxiliary {
        let index =
            csv.headers.iter().position(|h| h == name).ok_or_else(|| {
                Failure::input(format!("no objective column {name:?} in the header"))
            })?;
        let values = csv
            .rows
            .iter()
            .enumerate()
            .map(|(row, cells)| {
                cells[index]
                    .as_deref()
                    .and_then(|cell| cell.trim().parse::<f64>().ok())
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| {
                        Failure::input(format!(
                            "objective column {name:?} needs a finite number in data row {}",
                            row + 1
                        ))
                    })
            })
            .collect::<Result<Vec<f64>>>()?;
        auxiliary_values.insert(name.clone(), values);
    }
    if csv.headers.len() < 2 + auxiliary.len() {
        return Err(Failure::input("the table has no feature columns"));
    }
    if csv.rows.is_empty() {
        return Err(Failure::input("the training table has no rows"));
    }
    let mut labels = Vec::with_capacity(csv.rows.len());
    for (index, row) in csv.rows.iter().enumerate() {
        match &row[target_index] {
            Some(label) => labels.push(label.trim().to_owned()),
            None => {
                return Err(Failure::input(format!(
                    "target {target:?} is missing in data row {}",
                    index + 1
                )))
            }
        }
    }
    let distinct: BTreeSet<&str> = labels.iter().map(String::as_str).collect();
    let numeric = labels.iter().all(|l| is_number(l));
    let (kind, task_reason) = match task {
        TaskChoice::Fixed(kind) => (kind, "requested".to_owned()),
        TaskChoice::Auto if distinct.len() == 2 => (
            TaskKind::Binary,
            "auto: exactly two distinct target labels".to_owned(),
        ),
        TaskChoice::Auto if numeric => (
            TaskKind::Regression,
            format!(
                "auto: numeric target with {} distinct values",
                distinct.len()
            ),
        ),
        TaskChoice::Auto => (
            TaskKind::Multiclass,
            format!(
                "auto: non-numeric target with {} distinct labels",
                distinct.len()
            ),
        ),
    };
    let (classes, targets) = match kind {
        TaskKind::Regression => {
            let targets = labels
                .iter()
                .enumerate()
                .map(|(i, l)| {
                    l.parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite())
                        .ok_or_else(|| {
                            Failure::input(format!(
                                "regression target {l:?} in data row {} is not a finite number",
                                i + 1
                            ))
                        })
                })
                .collect::<Result<Vec<f64>>>()?;
            (Vec::new(), targets)
        }
        TaskKind::Binary | TaskKind::Multiclass => {
            let mut classes: Vec<String> = distinct.iter().map(|s| (*s).to_owned()).collect();
            if numeric {
                classes.sort_by(|a, b| {
                    a.parse::<f64>()
                        .unwrap_or(0.0)
                        .total_cmp(&b.parse::<f64>().unwrap_or(0.0))
                        .then_with(|| a.cmp(b))
                });
            }
            let wanted = if kind == TaskKind::Binary {
                "exactly 2"
            } else {
                "at least 3"
            };
            let ok = if kind == TaskKind::Binary {
                classes.len() == 2
            } else {
                classes.len() >= 3
            };
            if !ok {
                return Err(Failure::input(format!(
                    "{} needs {wanted} classes; the target has {}",
                    crate::task::name(kind),
                    classes.len()
                )));
            }
            if classes.len() > limits.max_classes {
                return Err(Failure::input(format!(
                    "{} classes; the limit is {}",
                    classes.len(),
                    limits.max_classes
                )));
            }
            let index: BTreeMap<&str, usize> = classes
                .iter()
                .enumerate()
                .map(|(i, c)| (c.as_str(), i))
                .collect();
            let targets = labels.iter().map(|l| index[l.as_str()] as f64).collect();
            (classes, targets)
        }
        TaskKind::Timeseries => {
            return Err(Failure::input("timeseries tasks are not supported"));
        }
    };
    let mut columns = Vec::new();
    let mut values: Vec<Vec<Option<String>>> = vec![Vec::new(); csv.rows.len()];
    for (index, name) in csv.headers.iter().enumerate() {
        if index == target_index || auxiliary.contains(name) {
            continue;
        }
        let numeric = csv
            .rows
            .iter()
            .filter_map(|row| row[index].as_deref())
            .all(is_number);
        columns.push(if numeric {
            ColumnSpec::numeric(name)
        } else {
            ColumnSpec::categorical(name)
        });
        for (row, out) in csv.rows.iter().zip(values.iter_mut()) {
            out.push(row[index].clone());
        }
    }
    let schema = iid_schema(columns).map_err(|e| Failure::input(format!("schema: {e:#}")))?;
    let table = raw_table(&schema, &values).map_err(|e| Failure::input(format!("{e:#}")))?;
    Ok(Training {
        kind,
        task_reason,
        target: target.to_owned(),
        schema,
        classes,
        table,
        targets,
        auxiliary: auxiliary_values,
    })
}

/// Prediction rows in the fitted schema's column order, and the columns that were ignored.
pub fn prediction(
    csv: &CsvTable,
    schema: &DataSchema,
    target: &str,
) -> Result<(RawTable, Vec<String>)> {
    prediction_ignoring(csv, schema, target, &[])
}

/// [`prediction`], also ignoring the model's auxiliary objective columns when present.
pub fn prediction_ignoring(
    csv: &CsvTable,
    schema: &DataSchema,
    target: &str,
    auxiliary: &[String],
) -> Result<(RawTable, Vec<String>)> {
    let mut positions = Vec::with_capacity(schema.columns.len());
    for column in &schema.columns {
        let position = csv
            .headers
            .iter()
            .position(|h| *h == column.name)
            .ok_or_else(|| {
                Failure::input(format!("input lacks feature column {:?}", column.name))
            })?;
        positions.push(position);
    }
    let known: BTreeSet<&str> = schema.columns.iter().map(|c| c.name.as_str()).collect();
    let mut ignored = Vec::new();
    for header in &csv.headers {
        if known.contains(header.as_str()) {
            continue;
        }
        if header == target || auxiliary.contains(header) {
            ignored.push(header.clone());
        } else {
            return Err(Failure::input(format!(
                "input column {header:?} is not a feature of this model"
            )));
        }
    }
    let values: Vec<Vec<Option<String>>> = csv
        .rows
        .iter()
        .map(|row| positions.iter().map(|p| row[*p].clone()).collect())
        .collect();
    let table = raw_table(schema, &values).map_err(|e| Failure::input(format!("{e:#}")))?;
    Ok((table, ignored))
}

/// A column's name and kind, as a fitted file records it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnRecord {
    pub name: String,
    pub kind: ColumnKind,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> Limits {
        Limits::default()
    }

    #[test]
    fn missing_cells_categories_and_kinds_follow_the_contract() {
        let csv = parse_csv(
            "\u{feff}x, color ,y\n1.5,NA,3\n,red,4\n2, ,5\n".as_bytes(),
            &limits(),
        )
        .unwrap();
        assert_eq!(csv.headers, ["x", "color", "y"]);
        assert_eq!(csv.rows[1][0], None);
        assert_eq!(csv.rows[2][1], None);
        let training = training(&csv, "y", TaskChoice::Auto, &limits()).unwrap();
        assert_eq!(training.kind, TaskKind::Regression);
        assert_eq!(training.schema.columns[0].kind, ColumnKind::Numeric);
        assert_eq!(training.schema.columns[1].kind, ColumnKind::Categorical);
        assert_eq!(training.targets, [3.0, 4.0, 5.0]);
        assert_eq!(
            training.table.rows[0][1],
            format!("{CATEGORY_PREFIX}\"NA\"")
        );
        assert_eq!(training.table.rows[2][1], "");
    }

    #[test]
    fn tasks_and_class_order() {
        let binary = parse_csv(b"a,y\n1,yes\n2,no\n3,yes\n", &limits()).unwrap();
        let t = training(&binary, "y", TaskChoice::Auto, &limits()).unwrap();
        assert_eq!(
            (t.kind, t.classes.clone()),
            (TaskKind::Binary, vec!["no".into(), "yes".into()])
        );
        assert_eq!(t.targets, [1.0, 0.0, 1.0]);
        let numeric = parse_csv(b"a,y\n1,10\n2,2\n3,9\n4,2\n", &limits()).unwrap();
        let t = training(
            &numeric,
            "y",
            TaskChoice::Fixed(TaskKind::Multiclass),
            &limits(),
        )
        .unwrap();
        assert_eq!(t.classes, ["2", "9", "10"]);
        let t = training(&numeric, "y", TaskChoice::Auto, &limits()).unwrap();
        assert_eq!(t.kind, TaskKind::Regression);
        let labels = parse_csv(b"a,y\n1,cat\n2,dog\n3,eel\n", &limits()).unwrap();
        assert_eq!(
            training(&labels, "y", TaskChoice::Auto, &limits())
                .unwrap()
                .kind,
            TaskKind::Multiclass
        );
        assert!(training(&labels, "y", TaskChoice::Fixed(TaskKind::Binary), &limits()).is_err());
        assert!(training(
            &labels,
            "y",
            TaskChoice::Fixed(TaskKind::Regression),
            &limits()
        )
        .is_err());
    }

    #[test]
    fn malformed_input_is_refused_with_an_input_failure() {
        let refuse = |text: &str| {
            let error = parse_csv(text.as_bytes(), &limits())
                .and_then(|csv| training(&csv, "y", TaskChoice::Auto, &limits()).map(|_| ()))
                .unwrap_err();
            assert_eq!(
                crate::error::kind_of(&error),
                crate::FailureKind::Input,
                "{text}: {error:#}"
            );
        };
        refuse("a,a,y\n1,2,3\n");
        refuse("a,,y\n1,2,3\n");
        refuse("a,y\n1,2\n3\n");
        refuse("a,b\n1,2\n");
        refuse("a,y\n1,\n");
        refuse("y\n1\n2\n");
        refuse("a,y\n");
        let small = Limits {
            max_rows: 1,
            ..Limits::default()
        };
        assert!(parse_csv(b"a,y\n1,2\n3,4\n", &small).is_err());
    }

    #[test]
    fn prediction_columns_match_by_name_and_refuse_unknown_ones() {
        let csv = parse_csv(b"a,b,y\n1,x,3\n2,y,4\n", &limits()).unwrap();
        let t = training(&csv, "y", TaskChoice::Auto, &limits()).unwrap();
        let reordered = parse_csv(b"b,y,a\nx,0,1\n", &limits()).unwrap();
        let (table, ignored) = prediction(&reordered, &t.schema, "y").unwrap();
        assert_eq!(table.headers, ["a", "b"]);
        assert_eq!(ignored, ["y"]);
        let extra = parse_csv(b"a,b,id\n1,x,7\n", &limits()).unwrap();
        assert!(prediction(&extra, &t.schema, "y").is_err());
        let missing = parse_csv(b"a\n1\n", &limits()).unwrap();
        assert!(prediction(&missing, &t.schema, "y").is_err());
        let bad_number = parse_csv(b"a,b\nseven,x\n", &limits()).unwrap();
        assert!(prediction(&bad_number, &t.schema, "y").is_err());
    }

    #[test]
    fn objective_columns_are_numeric_never_features_and_ignored_at_prediction() {
        let regression = || TaskChoice::Fixed(TaskKind::Regression);
        let wanted = |name: &str| -> BTreeSet<String> { [name.to_owned()].into() };
        let csv = parse_csv(b"a,price,y\n1,10,3\n2,12.5,4\n", &limits()).unwrap();
        let t =
            training_with_auxiliary(&csv, "y", regression(), &limits(), &wanted("price")).unwrap();
        assert_eq!(t.table.headers, ["a"]);
        assert_eq!(t.auxiliary["price"], [10.0, 12.5]);
        let input = parse_csv(b"price,a\n9,1\n", &limits()).unwrap();
        let (table, ignored) =
            prediction_ignoring(&input, &t.schema, "y", &["price".to_owned()]).unwrap();
        assert_eq!(table.headers, ["a"]);
        assert_eq!(ignored, ["price"]);
        assert!(prediction(&input, &t.schema, "y").is_err());
        for (text, column) in [
            (&b"a,price,y\n1,,3\n2,1,4\n"[..], "price"),
            (b"a,price,y\n1,cheap,3\n2,1,4\n", "price"),
            (b"a,y\n1,3\n2,4\n", "price"),
            (b"a,y\n1,3\n2,4\n", "y"),
            (b"price,y\n1,3\n2,4\n", "price"),
        ] {
            let csv = parse_csv(text, &limits()).unwrap();
            assert!(
                training_with_auxiliary(&csv, "y", regression(), &limits(), &wanted(column))
                    .is_err(),
                "{column}"
            );
        }
    }
}
