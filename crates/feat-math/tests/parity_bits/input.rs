use super::*;

pub(super) fn parse_bits(text: &str) -> Option<u64> {
    let digits = text.strip_prefix("0x")?;
    if digits.len() != 16 || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(digits, 16).ok()
}

pub(super) fn hex(bits: u64) -> String {
    format!("0x{bits:016x}")
}

/// `(name, value)` for every corpus entry, in file order.
pub(super) fn boundary_values() -> Vec<(String, f64)> {
    let corpus: Corpus = serde_json::from_str(BOUNDARY_V1).expect("boundary corpus parses");
    corpus
        .entries
        .into_iter()
        .map(|entry| {
            let bits = parse_bits(&entry.bits)
                .unwrap_or_else(|| panic!("{}: bad bits {}", entry.name, entry.bits));
            (entry.name, f64::from_bits(bits))
        })
        .collect()
}
pub(super) fn ordinary_rows() -> Vec<InputRow> {
    let mut rng = Pcg32::new(20_260_930, 11);
    (0..ORDINARY_ROWS)
        .map(|index| InputRow {
            id: format!("ordinary:{index:02}"),
            values: (0..WIDTH).map(|_| rng.next_f64() * 6.0 - 3.0).collect(),
        })
        .collect()
}

/// The value in every column (`all:`), or in one column with [`FILL`] in the others (`c0:` ..).
pub(super) fn shaped_rows(name: &str, value: f64) -> Vec<InputRow> {
    let mut rows = vec![InputRow {
        id: format!("all:{name}"),
        values: vec![value; WIDTH],
    }];
    for column in 0..WIDTH {
        let mut values = FILL.to_vec();
        values[column] = value;
        rows.push(InputRow {
            id: format!("c{column}:{name}"),
            values,
        });
    }
    rows
}

/// Rows whose entries are all zeros of mixed sign: which zero a sum, a minimum or a maximum
/// returns depends on the order its implementation visits them in.
pub(super) fn mixed_zero_rows() -> Vec<InputRow> {
    let (p, n) = (0.0_f64, -0.0_f64);
    [
        ("zeros:pos_pos_neg", [p, p, n]),
        ("zeros:neg_neg_pos", [n, n, p]),
        ("zeros:pos_neg_pos", [p, n, p]),
        ("zeros:neg_pos_neg", [n, p, n]),
    ]
    .into_iter()
    .map(|(id, values)| InputRow {
        id: id.to_owned(),
        values: values.to_vec(),
    })
    .collect()
}

pub(super) fn input_rows(inputs: Inputs) -> Vec<InputRow> {
    let corpus = boundary_values();
    let mut rows = ordinary_rows();
    match inputs {
        Inputs::Finite | Inputs::All => {
            for (name, value) in &corpus {
                if value.is_finite() || inputs == Inputs::All {
                    rows.extend(shaped_rows(name, *value));
                }
            }
            rows.extend(mixed_zero_rows());
        }
        Inputs::NonFiniteColumnZero => {
            for (name, value) in corpus.iter().filter(|(_, value)| !value.is_finite()) {
                rows.extend(
                    shaped_rows(name, *value)
                        .into_iter()
                        .filter(|row| row.id.starts_with("c0:")),
                );
            }
        }
    }
    let ids: BTreeSet<&str> = rows.iter().map(|row| row.id.as_str()).collect();
    assert_eq!(ids.len(), rows.len(), "duplicate input row id");
    rows
}

/// Finite training rows (every lane fits on them; the fitted C++ binding requires finite data).
pub(super) fn training_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Pcg32::new(20_260_929, 7);
    let mut x = Vec::with_capacity(TRAIN_ROWS);
    let mut y = Vec::with_capacity(TRAIN_ROWS);
    for _ in 0..TRAIN_ROWS {
        let row: Vec<f64> = (0..WIDTH).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        let target =
            1.3 * row[0] - 0.7 * row[1] + 0.4 * (2.0 * row[2]).sin() + 0.1 * rng.gaussian();
        x.push(row);
        y.push(target);
    }
    (x, y)
}

pub(super) fn csv_line(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:?}"))
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn write_train_csv(path: &Path, x: &[Vec<f64>], y: &[f64]) {
    let mut text = String::new();
    for (row, target) in x.iter().zip(y) {
        let mut cells = row.clone();
        cells.push(*target);
        text.push_str(&csv_line(&cells));
        text.push('\n');
    }
    fs::write(path, text).unwrap();
}

pub(super) fn write_features_csv(path: &Path, rows: &[InputRow]) {
    let mut text = String::new();
    for row in rows {
        text.push_str(&csv_line(&row.values));
        text.push('\n');
    }
    fs::write(path, text).unwrap();
}

pub(super) fn write_features_bits(path: &Path, rows: &[InputRow]) {
    let mut text = format!("{} {WIDTH}\n", rows.len());
    for row in rows {
        for value in &row.values {
            text.push_str(&format!("{:016x}\n", value.to_bits()));
        }
    }
    fs::write(path, text).unwrap();
}

// ------------------------------------------------------------------------------------------------
// Lanes
// ------------------------------------------------------------------------------------------------
