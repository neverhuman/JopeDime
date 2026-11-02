#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// `ColumnType` is an enum for the column type contract in this module: A minimal Apache Parquet writer and reader for flat tables (the gpdistilled/v1 table).
pub enum ColumnType {
    /// The `Boolean` case of `ColumnType`.
    Boolean,
    /// The `Int64` case of `ColumnType`.
    Int64,
    /// The `Double` case of `ColumnType`.
    Double,
    /// The `Utf8` case of `ColumnType`.
    Utf8,
    /// INT64 milliseconds since the epoch (ConvertedType TIMESTAMP_MILLIS).
    TimestampMillis,
}

impl ColumnType {
    fn physical(self) -> i32 {
        match self {
            ColumnType::Boolean => 0,
            ColumnType::Int64 | ColumnType::TimestampMillis => 2,
            ColumnType::Double => 5,
            ColumnType::Utf8 => 6,
        }
    }

    fn from_physical(t: i64, converted: Option<i64>) -> Result<ColumnType> {
        Ok(match (t, converted) {
            (0, None) => ColumnType::Boolean,
            (2, None) => ColumnType::Int64,
            (2, Some(9)) => ColumnType::TimestampMillis,
            (5, None) => ColumnType::Double,
            (6, Some(0)) => ColumnType::Utf8,
            _ => bail!("unsupported parquet column type {t} (converted {converted:?})"),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// `ColumnSpec` is a record for the column spec contract in this module: A minimal Apache Parquet writer and reader for flat tables (the gpdistilled/v1 table).
pub struct ColumnSpec {
    /// Name recorded for this column spec.
    pub name: String,
    /// Ty recorded for this column spec.
    pub ty: ColumnType,
    /// Required recorded for this column spec.
    pub required: bool,
}

#[derive(Clone, Debug, PartialEq)]
/// `Cell` is an enum for the cell contract in this module: A minimal Apache Parquet writer and reader for flat tables (the gpdistilled/v1 table).
pub enum Cell {
    /// The `Null` case of `Cell`.
    Null,
    /// The `Bool` case of `Cell`.
    Bool(bool),
    /// The `I64` case of `Cell`.
    I64(i64),
    /// The `F64` case of `Cell`.
    F64(f64),
    /// The `Str` case of `Cell`.
    Str(String),
}

#[derive(Clone, Debug, PartialEq)]
/// `Table` is a record for the table contract in this module: A minimal Apache Parquet writer and reader for flat tables (the gpdistilled/v1 table).
pub struct Table {
    /// Collection of columns values carried by this table.
    pub columns: Vec<ColumnSpec>,
    /// Number of rows represented by this value.
    pub rows: Vec<Vec<Cell>>,
    /// Collection of metadata values carried by this table.
    pub metadata: Vec<(String, String)>,
}

/// Footer values use physical types; BYTE_ARRAY order is unsigned lexicographic bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum StatisticValue {
    /// The `Bool` case of `StatisticValue`.
    Bool(bool),
    /// The `I64` case of `StatisticValue`.
    I64(i64),
    /// The `F64` case of `StatisticValue`.
    F64(f64),
    /// The `Bytes` case of `StatisticValue`.
    Bytes(Vec<u8>),
}

impl StatisticValue {
    fn encode(&self) -> Vec<u8> {
        match self {
            Self::Bool(v) => vec![u8::from(*v)],
            Self::I64(v) => v.to_le_bytes().to_vec(),
            Self::F64(v) => v.to_le_bytes().to_vec(),
            Self::Bytes(v) => v.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
/// `ColumnStatistics` is a record for the column statistics contract in this module: A minimal Apache Parquet writer and reader for flat tables (the gpdistilled/v1 table).
pub struct ColumnStatistics {
    /// Number of occurrences, including nulls (ColumnMetaData.num_values).
    pub value_count: u64,
    /// Absent in a legacy/external footer means unknown, not zero.
    pub null_count: Option<u64>,
    /// Number of nan represented here.
    pub nan_count: Option<u64>,
    /// Optional min value; `None` means it is not available.
    pub min: Option<StatisticValue>,
    /// Optional max value; `None` means it is not available.
    pub max: Option<StatisticValue>,
    /// Lower bound on is exact.
    pub min_is_exact: Option<bool>,
    /// Upper bound on is exact.
    pub max_is_exact: Option<bool>,
}

fn column_statistics(ty: ColumnType, cells: &[&Cell]) -> Result<ColumnStatistics> {
    let mut present: Vec<&Cell> = cells
        .iter()
        .copied()
        .filter(|c| !matches!(c, Cell::Null))
        .collect();
    // Validate even all-null/type-mismatched chunks via the existing physical encoder.
    plain(ty, &present)?;
    let nulls = (cells.len() - present.len()) as u64;
    // Retain nonfinite values in the pages. Omit bounds for their chunk so pruning cannot
    // silently discard them. Null count counts actual nulls only.
    let has_nonfinite = present
        .iter()
        .any(|c| matches!(c, Cell::F64(v) if !v.is_finite()));
    present.sort_by(|a, b| match (a, b) {
        (Cell::Bool(a), Cell::Bool(b)) => a.cmp(b),
        (Cell::I64(a), Cell::I64(b)) => a.cmp(b),
        (Cell::F64(a), Cell::F64(b)) => a.total_cmp(b),
        (Cell::Str(a), Cell::Str(b)) => a.as_bytes().cmp(b.as_bytes()),
        _ => std::cmp::Ordering::Equal,
    });
    let value = |c: &&Cell| match c {
        Cell::Bool(v) => StatisticValue::Bool(*v),
        Cell::I64(v) => StatisticValue::I64(*v),
        Cell::F64(v) => StatisticValue::F64(*v),
        Cell::Str(v) => StatisticValue::Bytes(v.as_bytes().to_vec()),
        Cell::Null => unreachable!("present cells"),
    };
    let actual_min = if has_nonfinite {
        None
    } else {
        present.first().map(value)
    };
    let actual_max = if has_nonfinite {
        None
    } else {
        present.last().map(value)
    };
    // TYPE_ORDER requires outward zero bounds. Values in pages retain their exact bits.
    let min = actual_min.as_ref().map(|v| match v {
        StatisticValue::F64(v) if *v == 0.0 => StatisticValue::F64(-0.0),
        other => other.clone(),
    });
    let max = actual_max.as_ref().map(|v| match v {
        StatisticValue::F64(v) if *v == 0.0 => StatisticValue::F64(0.0),
        other => other.clone(),
    });
    Ok(ColumnStatistics {
        value_count: cells.len() as u64,
        null_count: Some(nulls),
        nan_count: (ty == ColumnType::Double).then(|| {
            present
                .iter()
                .filter(|c| matches!(c, Cell::F64(v) if v.is_nan()))
                .count() as u64
        }),
        min_is_exact: min
            .as_ref()
            .zip(actual_min.as_ref())
            .map(|(a, b)| a.encode() == b.encode()),
        max_is_exact: max
            .as_ref()
            .zip(actual_max.as_ref())
            .map(|(a, b)| a.encode() == b.encode()),
        min,
        max,
    })
}

fn decode_statistics(
    m: &BTreeMap<i16, Tv>,
    physical: i64,
    rows: u64,
    ordered: bool,
) -> Result<ColumnStatistics> {
    let null_count = int(m, 3).map(u64::try_from).transpose()?;
    let nan_count = int(m, 9).map(u64::try_from).transpose()?;
    ensure!(
        null_count.is_none_or(|n| n <= rows),
        "parquet: null count exceeds rows"
    );
    let decode = |b: &[u8]| -> Result<StatisticValue> {
        Ok(match physical {
            0 => {
                ensure!(
                    b.len() == 1 && b[0] <= 1,
                    "parquet: invalid boolean statistic"
                );
                StatisticValue::Bool(b[0] == 1)
            }
            1 => StatisticValue::I64(i32::from_le_bytes(b.try_into()?) as i64),
            2 => StatisticValue::I64(i64::from_le_bytes(b.try_into()?)),
            4 => StatisticValue::F64(f32::from_le_bytes(b.try_into()?) as f64),
            5 => StatisticValue::F64(f64::from_le_bytes(b.try_into()?)),
            6 => StatisticValue::Bytes(b.to_vec()),
            _ => bail!("parquet: unsupported statistics physical type {physical}"),
        })
    };
    // Use modern min_value/max_value; deprecated signed BYTE_ARRAY bounds are ambiguous.
    Ok(ColumnStatistics {
        value_count: rows,
        null_count,
        nan_count,
        min: bin(m, 6).filter(|_| ordered).map(decode).transpose()?,
        max: bin(m, 5).filter(|_| ordered).map(decode).transpose()?,
        min_is_exact: match m.get(&8) {
            Some(Tv::Bool(v)) => Some(*v),
            _ => None,
        },
        max_is_exact: match m.get(&7) {
            Some(Tv::Bool(v)) => Some(*v),
            _ => None,
        },
    })
}
