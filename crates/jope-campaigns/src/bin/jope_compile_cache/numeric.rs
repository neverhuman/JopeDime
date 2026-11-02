use super::*;

pub(super) fn read_numeric_csv(path: &Path) -> Result<NumericData> {
    let text =
        fs::read_to_string(path).with_context(|| format!("read numeric CSV {}", path.display()))?;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut features = None;
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut values = line
            .split(',')
            .map(|field| {
                field.trim().parse::<f64>().with_context(|| {
                    format!("{}:{} invalid numeric field", path.display(), line_no + 1)
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            values.len() >= 2,
            "{}:{} needs a feature and target",
            path.display(),
            line_no + 1
        );
        let width = values.len() - 1;
        if let Some(expected) = features {
            ensure!(
                width == expected,
                "{}:{} feature width drift",
                path.display(),
                line_no + 1
            );
        } else {
            features = Some(width);
        }
        let target = values.pop().expect("length checked");
        ensure!(
            target.is_finite(),
            "{}:{} non-finite target",
            path.display(),
            line_no + 1
        );
        for value in &mut values {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        x.push(values);
        y.push(target);
    }
    Ok(NumericData {
        x,
        y,
        features: features.context("empty numeric CSV")?,
    })
}

pub(super) fn validate_targets(task: &str, y: &[f64], split: &str) -> Result<()> {
    ensure!(!y.is_empty(), "{split} targets are empty");
    if task == "binary" {
        let positive = y.iter().filter(|value| **value == 1.0).count();
        ensure!(
            y.iter().all(|value| *value == 0.0 || *value == 1.0),
            "{split} binary targets are not exactly 0/1"
        );
        ensure!(
            positive > 0 && positive < y.len(),
            "{split} binary targets lack both classes"
        );
    }
    Ok(())
}

pub(super) fn numeric_schema(features: usize) -> DataSchema {
    DataSchema {
        columns: (0..features)
            .map(|index| ColumnSpec::numeric(format!("c{index}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    }
}

pub(super) fn raw_table(features: usize, rows: &[Vec<f64>]) -> Result<RawTable> {
    RawTable::new(
        (0..features).map(|index| format!("c{index}")).collect(),
        rows.iter()
            .map(|row| row.iter().map(|value| format!("{value:?}")).collect())
            .collect(),
    )
}

pub(super) fn score(task: &str, target: &[f64], prediction: &[f64]) -> Result<f64> {
    ensure!(
        target.len() == prediction.len() && prediction.iter().all(|value| value.is_finite()),
        "prediction output is missing or non-finite"
    );
    let value = if task == "binary" {
        ensure!(
            prediction.iter().all(|value| (0.0..=1.0).contains(value)),
            "binary compiled predictions are outside [0,1]"
        );
        auc(target, prediction)
    } else {
        r2(target, prediction)
    };
    ensure!(value.is_finite(), "compiled score is non-finite");
    Ok(value)
}

pub(super) fn prediction_sha256(values: &[f64]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sha256_bytes(&bytes)
}
