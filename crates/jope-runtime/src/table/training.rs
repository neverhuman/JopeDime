use super::*;

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
