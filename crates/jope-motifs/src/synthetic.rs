//! Deterministic synthetic tables for context and determinism checks.
//!
//! Cells are strings built from `Pcg32::new(seed, stream)` with fixed seeds, stream 11 for
//! training rows and stream 12 for the probe, formatted with `f64::to_string`.

use crate::source::TaskLabel;
use anyhow::{Context, Result};
use feat_math::advanced_v3::{ColumnSpec, DataSchema, RawTable, RowSemantics};
use feat_math::Pcg32;

/// Stream of every training table.
pub const TRAIN_STREAM: u64 = 11;
/// Stream of every probe table.
pub const PROBE_STREAM: u64 = 12;
/// Rows of every probe table.
pub const PROBE_ROWS: usize = 11;
/// Index of the probe row whose cells are all empty.
pub const EMPTY_PROBE_ROW: usize = 5;
/// Levels of the categorical column of S4.
const LEVELS: [&str; 3] = ["north", "south", "east"];

/// The four training shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Shape {
    /// 18x3 numeric: the small-sample branches.
    S1,
    /// 64x6 numeric: the baseline.
    S2,
    /// 160x12 numeric: crosses the 80 and 150 row gates.
    S3,
    /// 64x6 numeric plus one three-level categorical column, about 5% empty cells, one
    /// constant column and 40% duplicated rows.
    S4,
}

impl Shape {
    /// Every synthetic shape in declaration order.
    pub const ALL: [Self; 4] = [Self::S1, Self::S2, Self::S3, Self::S4];

    /// Return the stable lowercase shape label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::S1 => "s1",
            Self::S2 => "s2",
            Self::S3 => "s3",
            Self::S4 => "s4",
        }
    }

    /// (seed, rows, numeric columns, categorical column).
    fn spec(self) -> (u64, usize, usize, bool) {
        match self {
            Self::S1 => (1801, 18, 3, false),
            Self::S2 => (6406, 64, 6, false),
            Self::S3 => (16012, 160, 12, false),
            Self::S4 => (6407, 64, 6, true),
        }
    }
}

/// One synthetic training table with its probe and the three task targets.
#[derive(Clone, Debug, PartialEq)]
pub struct Synthetic {
    /// Shape used to construct this synthetic dataset.
    pub shape: Shape,
    /// Column schema shared by the training and probe tables.
    pub schema: DataSchema,
    /// Deterministic synthetic training rows.
    pub train: RawTable,
    /// Deterministic synthetic rows used by probe checks.
    pub probe: RawTable,
    /// Regression target aligned to training rows.
    pub regression: Vec<f64>,
    /// Binary target aligned to training rows.
    pub binary: Vec<f64>,
    /// Multiclass target aligned to training rows.
    pub multiclass: Vec<f64>,
}

/// Rows of S4 that copy an earlier row, cells and targets alike.
const S4_DUPLICATED_FROM: usize = 38;
/// Numeric column of S4 that holds one constant value.
const S4_CONSTANT_COLUMN: usize = 3;
/// Share of S4 numeric cells left empty.
const S4_EMPTY_SHARE: f64 = 0.05;

impl Synthetic {
    /// Build one shape. The same shape always yields the same bytes.
    pub fn build(shape: Shape) -> Result<Self> {
        let (seed, rows, numeric, categorical) = shape.spec();
        let schema = schema(numeric, categorical);
        let headers: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();

        let mut rng = Pcg32::new(seed, TRAIN_STREAM);
        let mut cells: Vec<Vec<String>> = Vec::with_capacity(rows);
        let mut regression: Vec<f64> = Vec::with_capacity(rows);
        for _ in 0..rows {
            let latent: Vec<f64> = (0..numeric).map(|_| rng.range_f64(-2.0, 2.0)).collect();
            let noise = rng.gaussian();
            regression.push(target(&latent, noise));
            let mut row: Vec<String> = Vec::with_capacity(headers.len());
            for (column, value) in latent.iter().enumerate() {
                if categorical && column == S4_CONSTANT_COLUMN {
                    row.push(String::from("1"));
                } else if categorical && rng.next_f64() < S4_EMPTY_SHARE {
                    row.push(String::new());
                } else {
                    row.push(value.to_string());
                }
            }
            if categorical {
                row.push(LEVELS[usize::try_from(rng.next_u32() % 3)?].to_owned());
            }
            cells.push(row);
        }
        if categorical {
            // Rows from S4_DUPLICATED_FROM on repeat the first rows exactly.
            let copies = rows - S4_DUPLICATED_FROM;
            let (first, rest) = cells.split_at_mut(S4_DUPLICATED_FROM);
            rest.clone_from_slice(&first[..copies]);
            let (first, rest) = regression.split_at_mut(S4_DUPLICATED_FROM);
            rest.copy_from_slice(&first[..copies]);
        }
        let train = RawTable::new(headers.clone(), cells)?;
        train.validate_against(&schema)?;

        let mut rng = Pcg32::new(seed, PROBE_STREAM);
        let mut probe_cells: Vec<Vec<String>> = Vec::with_capacity(PROBE_ROWS);
        for index in 0..PROBE_ROWS {
            let mut row: Vec<String> = (0..numeric)
                .map(|column| {
                    let value = rng.range_f64(-2.0, 2.0);
                    if categorical && column == S4_CONSTANT_COLUMN {
                        String::from("1")
                    } else {
                        value.to_string()
                    }
                })
                .collect();
            if categorical {
                row.push(LEVELS[usize::try_from(rng.next_u32() % 3)?].to_owned());
            }
            if index == EMPTY_PROBE_ROW {
                row = vec![String::new(); row.len()];
            }
            probe_cells.push(row);
        }
        let probe = RawTable::new(headers, probe_cells)?;
        probe.validate_against(&schema)?;

        let binary = binary_targets(&regression)?;
        let multiclass = multiclass_targets(&regression)?;
        Ok(Self {
            shape,
            schema,
            train,
            probe,
            regression,
            binary,
            multiclass,
        })
    }

    /// Every shape, in order.
    pub fn all() -> Result<Vec<Self>> {
        Shape::ALL.iter().map(|shape| Self::build(*shape)).collect()
    }

    /// The targets of one task.
    pub fn targets(&self, task: TaskLabel) -> &[f64] {
        match task {
            TaskLabel::Regression => &self.regression,
            TaskLabel::Binary => &self.binary,
            TaskLabel::Multiclass => &self.multiclass,
        }
    }

    /// The training rows as numbers, when every cell is a number.
    pub fn numeric_rows(&self) -> Option<Vec<Vec<f64>>> {
        self.train
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| cell.parse::<f64>().ok())
                    .collect::<Option<Vec<f64>>>()
            })
            .collect()
    }
}

fn schema(numeric: usize, categorical: bool) -> DataSchema {
    let mut columns: Vec<ColumnSpec> = (0..numeric)
        .map(|index| ColumnSpec::numeric(format!("c{index}")))
        .collect();
    if categorical {
        columns.push(ColumnSpec::categorical("k"));
    }
    DataSchema {
        columns,
        row_semantics: RowSemantics::Iid,
    }
}

/// `sin(x0) + 0.5 x1^2 - x_{d-1} + 0.1 noise`.
fn target(latent: &[f64], noise: f64) -> f64 {
    let last = latent.last().copied().unwrap_or(0.0);
    latent[0].sin() + 0.5 * latent[1] * latent[1] - last + 0.1 * noise
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted
}

/// 1 above the regression median, else 0.
fn binary_targets(regression: &[f64]) -> Result<Vec<f64>> {
    let ordered = sorted(regression);
    let half = ordered.len() / 2;
    let median = if ordered.len().is_multiple_of(2) {
        (ordered.get(half.wrapping_sub(1)).context("empty targets")? + ordered[half]) / 2.0
    } else {
        *ordered.get(half).context("empty targets")?
    };
    Ok(regression
        .iter()
        .map(|value| if *value > median { 1.0 } else { 0.0 })
        .collect())
}

/// The regression tercile, coded 0, 1 or 2.
fn multiclass_targets(regression: &[f64]) -> Result<Vec<f64>> {
    let ordered = sorted(regression);
    let n = ordered.len();
    let first = *ordered.get(n / 3).context("empty targets")?;
    let second = *ordered.get(2 * n / 3).context("empty targets")?;
    Ok(regression
        .iter()
        .map(|value| {
            if *value < first {
                0.0
            } else if *value < second {
                1.0
            } else {
                2.0
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn tables_are_deterministic_and_rectangular() {
        let dimensions = [(18, 3), (64, 6), (160, 12), (64, 7)];
        for (shape, (rows, columns)) in Shape::ALL.into_iter().zip(dimensions) {
            let first = Synthetic::build(shape).expect("builds");
            let second = Synthetic::build(shape).expect("builds");
            assert_eq!(first, second, "{shape:?}");
            assert_eq!(first.train.rows.len(), rows);
            assert_eq!(first.train.headers.len(), columns);
            assert!(first.train.validate_against(&first.schema).is_ok());
            assert!(first.probe.validate_against(&first.schema).is_ok());
            assert_eq!(first.probe.rows.len(), PROBE_ROWS);
            assert!(first.probe.rows[EMPTY_PROBE_ROW]
                .iter()
                .all(String::is_empty));
            assert_eq!(first.regression.len(), rows);
            assert_ne!(first.train.rows, first.probe.rows);
        }

        let s2 = Synthetic::build(Shape::S2).expect("builds");
        assert!(s2.numeric_rows().is_some());
        let s4 = Synthetic::build(Shape::S4).expect("builds");
        assert!(s4.numeric_rows().is_none());
        let empty = s4
            .train
            .rows
            .iter()
            .flatten()
            .filter(|cell| cell.is_empty())
            .count();
        assert!(empty > 0 && empty < 64 * 6 / 5, "{empty} empty cells");
        assert!(s4
            .train
            .rows
            .iter()
            .all(|row| row[S4_CONSTANT_COLUMN] == "1"));
        let mut copies: BTreeMap<&Vec<String>, usize> = BTreeMap::new();
        for row in &s4.train.rows {
            *copies.entry(row).or_default() += 1;
        }
        let duplicated: usize = copies.values().filter(|count| **count > 1).sum();
        assert!(duplicated >= 2 * (64 - S4_DUPLICATED_FROM), "{duplicated}");
        let levels: std::collections::BTreeSet<&str> =
            s4.train.rows.iter().map(|row| row[6].as_str()).collect();
        assert_eq!(levels.len(), 3);
    }

    #[test]
    fn task_targets_satisfy_the_v3_target_contracts() {
        for data in Synthetic::all().expect("builds") {
            assert!(data.regression.iter().all(|value| value.is_finite()));

            let ones = data.binary.iter().filter(|value| **value == 1.0).count();
            let zeros = data.binary.iter().filter(|value| **value == 0.0).count();
            assert_eq!(ones + zeros, data.binary.len());
            assert!(ones >= 2 && zeros >= 2, "{:?}: {ones}/{zeros}", data.shape);

            let mut counts = [0usize; 3];
            for value in &data.multiclass {
                let class = [0.0, 1.0, 2.0]
                    .iter()
                    .position(|code| code == value)
                    .expect("a class code");
                counts[class] += 1;
            }
            assert!(counts.iter().all(|count| *count >= 1), "{:?}", data.shape);
            for task in TaskLabel::ALL {
                assert_eq!(data.targets(task).len(), data.train.rows.len());
            }
        }
    }
}
