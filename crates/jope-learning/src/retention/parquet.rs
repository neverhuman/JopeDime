//! A minimal Apache Parquet writer and reader for flat tables (the `gp_distilled/v1` table).
//!
//! No Parquet crate is in `Cargo.lock`, and adding one would change the ADR-0004 evaluator
//! identity, so this module writes the subset it needs by hand:
//! - one row group, one data page (v1) per column;
//! - physical types BOOLEAN, INT64, DOUBLE, BYTE_ARRAY (UTF8), each REQUIRED or OPTIONAL;
//! - PLAIN values; definition levels in the RLE/bit-packed hybrid (bit width 1);
//! - pages either UNCOMPRESSED or ZSTD (one zstd frame per page);
//! - the footer (`FileMetaData`) in the Thrift compact protocol, with key/value metadata.
//!
//! The reader decodes exactly that subset (plus RLE runs in definition levels) and refuses
//! anything else, so a table can be reconciled against its raw source (strategy §8.2 E3).

use anyhow::{bail, ensure, Context, Result};
use std::collections::BTreeMap;

const MAGIC: &[u8; 4] = b"PAR1";
/// Decoded page size cap for the reader (a hostile footer cannot request more).
const MAX_PAGE: usize = 1 << 30;

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

// ---- Thrift compact protocol: writer ----

#[derive(Default)]
struct Tw {
    buf: Vec<u8>,
    last: Vec<i16>,
}

fn zz32(v: i32) -> u64 {
    u64::from(((v << 1) ^ (v >> 31)) as u32)
}

fn zz64(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

impl Tw {
    fn new() -> Tw {
        Tw {
            buf: Vec::new(),
            last: vec![0],
        }
    }
    fn varint(&mut self, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.buf.push(b);
                return;
            }
            self.buf.push(b | 0x80);
        }
    }
    fn field(&mut self, id: i16, ty: u8) {
        let last = self.last.last_mut().expect("struct context");
        let delta = id - *last;
        if (1..=15).contains(&delta) {
            self.buf.push(((delta as u8) << 4) | ty);
        } else {
            self.buf.push(ty);
            let z = zz32(i32::from(id));
            self.varint(z);
        }
        *self.last.last_mut().expect("struct context") = id;
    }
    fn i32f(&mut self, id: i16, v: i32) {
        self.field(id, 5);
        self.varint(zz32(v));
    }
    fn i64f(&mut self, id: i16, v: i64) {
        self.field(id, 6);
        self.varint(zz64(v));
    }
    fn binf(&mut self, id: i16, b: &[u8]) {
        self.field(id, 8);
        self.varint(b.len() as u64);
        self.buf.extend_from_slice(b);
    }
    fn struct_begin(&mut self, id: i16) {
        self.field(id, 12);
        self.last.push(0);
    }
    fn struct_end(&mut self) {
        self.buf.push(0);
        self.last.pop();
    }
    fn list_begin(&mut self, id: i16, elem: u8, n: usize) {
        self.field(id, 9);
        if n < 15 {
            self.buf.push(((n as u8) << 4) | elem);
        } else {
            self.buf.push(0xf0 | elem);
            self.varint(n as u64);
        }
    }
    fn elem_struct_begin(&mut self) {
        self.last.push(0);
    }
    fn elem_i32(&mut self, v: i32) {
        self.varint(zz32(v));
    }
    fn elem_bin(&mut self, b: &[u8]) {
        self.varint(b.len() as u64);
        self.buf.extend_from_slice(b);
    }
    fn finish(mut self) -> Vec<u8> {
        self.buf.push(0);
        self.buf
    }
}

// ---- encoders ----

fn bitpack_bools(values: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; values.len().div_ceil(8)];
    for (i, v) in values.iter().enumerate() {
        if *v {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

/// Definition levels (bit width 1) as one bit-packed run, with the v1 4-byte length prefix.
fn def_levels(defined: &[bool]) -> Vec<u8> {
    let mut run = Tw::new();
    let groups = defined.len().div_ceil(8) as u64;
    run.varint((groups << 1) | 1);
    let mut body = run.buf;
    body.extend(bitpack_bools(defined));
    let mut out = (body.len() as u32).to_le_bytes().to_vec();
    out.extend(body);
    out
}

fn plain(ty: ColumnType, cells: &[&Cell]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    match ty {
        ColumnType::Boolean => {
            let vals: Vec<bool> = cells
                .iter()
                .map(|c| match c {
                    Cell::Bool(b) => Ok(*b),
                    other => bail!("boolean column holds {other:?}"),
                })
                .collect::<Result<_>>()?;
            out = bitpack_bools(&vals);
        }
        ColumnType::Int64 | ColumnType::TimestampMillis => {
            for c in cells {
                let Cell::I64(v) = c else {
                    bail!("int64 column holds {c:?}")
                };
                out.extend(v.to_le_bytes());
            }
        }
        ColumnType::Double => {
            for c in cells {
                let Cell::F64(v) = c else {
                    bail!("double column holds {c:?}")
                };
                out.extend(v.to_le_bytes());
            }
        }
        ColumnType::Utf8 => {
            for c in cells {
                let Cell::Str(s) = c else {
                    bail!("utf8 column holds {c:?}")
                };
                out.extend((u32::try_from(s.len())?).to_le_bytes());
                out.extend(s.as_bytes());
            }
        }
    }
    Ok(out)
}

/// Page compression for [`write_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    /// The `Uncompressed` case of `Codec`.
    Uncompressed,
    /// The `Zstd` case of `Codec`.
    Zstd,
    /// The `Snappy` case of `Codec`.
    Snappy,
}

impl Codec {
    fn id(self) -> i32 {
        match self {
            Codec::Uncompressed => 0,
            Codec::Snappy => 1,
            Codec::Zstd => 6,
        }
    }

    fn apply(self, page: &[u8]) -> Vec<u8> {
        match self {
            Codec::Uncompressed => page.to_vec(),
            Codec::Zstd => jope_records::zstd_frame::compress(page),
            Codec::Snappy => super::snappy::compress(page),
        }
    }
}

/// Layout knobs. The defaults (one row group, one PLAIN page per column) are what
/// `gp_distilled/v1` uses; the others exist so fixtures can look like S3 Inventory files.
#[derive(Clone, Copy, Debug)]
pub struct WriteOptions {
    /// Codec recorded for this write options.
    pub codec: Codec,
    /// Rows per row group (0: one row group).
    pub row_group_rows: usize,
    /// Rows per data page (0: one page per column chunk).
    pub page_rows: usize,
    /// Dictionary-encode UTF8, INT64 and timestamp columns (PLAIN_DICTIONARY, as parquet-mr).
    pub dictionary: bool,
}

/// `values` of `width` bits as one bit-packed hybrid run (padded to a multiple of 8 values).
fn bitpacked_run(values: &[u32], width: u32) -> Vec<u8> {
    let groups = values.len().div_ceil(8);
    let mut run = Tw::new();
    run.varint(((groups as u64) << 1) | 1);
    let mut out = run.buf;
    let mut packed = vec![0u8; groups * width as usize];
    let mut bit = 0usize;
    for i in 0..groups * 8 {
        let v = values.get(i).copied().unwrap_or(0);
        for b in 0..width {
            if v >> b & 1 == 1 {
                packed[bit / 8] |= 1 << (bit % 8);
            }
            bit += 1;
        }
    }
    out.extend(packed);
    out
}

/// Distinct values in first-seen order and each present cell's index into them.
fn dictionary(present: &[&Cell]) -> (Vec<Cell>, Vec<u32>) {
    let mut seen: BTreeMap<String, u32> = BTreeMap::new();
    let mut dict = Vec::new();
    let mut idx = Vec::with_capacity(present.len());
    for c in present {
        let k = format!("{c:?}");
        let i = *seen.entry(k).or_insert_with(|| {
            dict.push((*c).clone());
            (dict.len() - 1) as u32
        });
        idx.push(i);
    }
    (dict, idx)
}

struct ChunkOut {
    file_offset: i64,
    data_offset: i64,
    dict_offset: Option<i64>,
    uncompressed: i64,
    compressed: i64,
    dictionary: bool,
    statistics: Option<ColumnStatistics>,
}

/// Appends one page (header + body) and returns (uncompressed, compressed) sizes incl. header.
fn put_page(
    out: &mut Vec<u8>,
    codec: Codec,
    raw: &[u8],
    header: impl FnOnce(&mut Tw),
    page_type: i32,
) -> Result<(i64, i64)> {
    let body = codec.apply(raw);
    let mut h = Tw::new();
    h.i32f(1, page_type);
    h.i32f(2, i32::try_from(raw.len())?);
    h.i32f(3, i32::try_from(body.len())?);
    header(&mut h);
    let header = h.finish();
    out.extend(&header);
    out.extend(&body);
    Ok((
        (header.len() + raw.len()) as i64,
        (header.len() + body.len()) as i64,
    ))
}

/// Writes `table` as Parquet. `zstd` selects ZSTD page compression, else UNCOMPRESSED.
pub fn write(table: &Table, zstd: bool, created_by: &str) -> Result<Vec<u8>> {
    let codec = if zstd {
        Codec::Zstd
    } else {
        Codec::Uncompressed
    };
    write_with(
        table,
        &WriteOptions {
            codec,
            row_group_rows: 0,
            page_rows: 0,
            dictionary: false,
        },
        created_by,
    )
}

/// Writes with to the configured destination.
pub fn write_with(table: &Table, opts: &WriteOptions, created_by: &str) -> Result<Vec<u8>> {
    write_table(table, opts, created_by, false)
}

/// Additive statistics footer; legacy write/write_with bytes remain unchanged.
pub fn write_with_statistics(
    table: &Table,
    opts: &WriteOptions,
    created_by: &str,
) -> Result<Vec<u8>> {
    write_table(table, opts, created_by, true)
}

fn write_table(
    table: &Table,
    opts: &WriteOptions,
    created_by: &str,
    statistics: bool,
) -> Result<Vec<u8>> {
    let ncol = table.columns.len();
    ensure!(ncol > 0, "a table needs at least one column");
    for (r, row) in table.rows.iter().enumerate() {
        ensure!(
            row.len() == ncol,
            "row {r} has {} cells, the schema {ncol}",
            row.len()
        );
    }
    let nrows = table.rows.len();
    let group_rows = if opts.row_group_rows == 0 {
        nrows.max(1)
    } else {
        opts.row_group_rows
    };
    let mut out = MAGIC.to_vec();
    let mut groups: Vec<(usize, Vec<ChunkOut>)> = Vec::new();
    let mut start = 0usize;
    loop {
        let rows = &table.rows[start..(start + group_rows).min(nrows)];
        let mut chunks = Vec::with_capacity(ncol);
        for (ci, col) in table.columns.iter().enumerate() {
            let cells: Vec<&Cell> = rows.iter().map(|r| &r[ci]).collect();
            if col.required {
                ensure!(
                    cells.iter().all(|c| !matches!(c, Cell::Null)),
                    "required column {} holds a null",
                    col.name
                );
            }
            let file_offset = out.len() as i64;
            let use_dict =
                opts.dictionary && col.ty != ColumnType::Boolean && col.ty != ColumnType::Double;
            let all_present: Vec<&Cell> = cells
                .iter()
                .copied()
                .filter(|c| !matches!(c, Cell::Null))
                .collect();
            let (dict, all_idx) = if use_dict {
                dictionary(&all_present)
            } else {
                (Vec::new(), Vec::new())
            };
            let (mut unc, mut comp) = (0i64, 0i64);
            let dict_offset = if use_dict {
                let off = out.len() as i64;
                let refs: Vec<&Cell> = dict.iter().collect();
                let raw = plain(col.ty, &refs).with_context(|| format!("column {}", col.name))?;
                let n = i32::try_from(dict.len())?;
                let (u, c) = put_page(
                    &mut out,
                    opts.codec,
                    &raw,
                    |h| {
                        h.struct_begin(7);
                        h.i32f(1, n);
                        h.i32f(2, 0); // PLAIN
                        h.struct_end();
                    },
                    2,
                )?;
                unc += u;
                comp += c;
                Some(off)
            } else {
                None
            };
            let width = (32 - (dict.len().max(2) as u32 - 1).leading_zeros()).max(1);
            let data_offset = out.len() as i64;
            let page_rows = if opts.page_rows == 0 {
                cells.len().max(1)
            } else {
                opts.page_rows
            };
            let mut p0 = 0usize;
            let mut idx_pos = 0usize;
            loop {
                let slice = &cells[p0..(p0 + page_rows).min(cells.len())];
                let defined: Vec<bool> = slice.iter().map(|c| !matches!(c, Cell::Null)).collect();
                let present: Vec<&Cell> = slice
                    .iter()
                    .copied()
                    .filter(|c| !matches!(c, Cell::Null))
                    .collect();
                let mut raw = if col.required {
                    Vec::new()
                } else {
                    def_levels(&defined)
                };
                if use_dict {
                    raw.push(width as u8);
                    raw.extend(bitpacked_run(
                        &all_idx[idx_pos..idx_pos + present.len()],
                        width,
                    ));
                    idx_pos += present.len();
                } else {
                    raw.extend(
                        plain(col.ty, &present).with_context(|| format!("column {}", col.name))?,
                    );
                }
                let n = i32::try_from(slice.len())?;
                let (u, c) = put_page(
                    &mut out,
                    opts.codec,
                    &raw,
                    |h| {
                        h.struct_begin(5);
                        h.i32f(1, n);
                        h.i32f(2, if use_dict { 2 } else { 0 }); // PLAIN_DICTIONARY | PLAIN
                        h.i32f(3, 3); // RLE definition levels
                        h.i32f(4, 3); // RLE repetition levels
                        h.struct_end();
                    },
                    0,
                )?;
                unc += u;
                comp += c;
                p0 += slice.len();
                if p0 >= cells.len() {
                    break;
                }
            }
            chunks.push(ChunkOut {
                file_offset: if use_dict { file_offset } else { data_offset },
                data_offset,
                dict_offset,
                uncompressed: unc,
                compressed: comp,
                dictionary: use_dict,
                statistics: statistics
                    .then(|| column_statistics(col.ty, &cells))
                    .transpose()?,
            });
        }
        groups.push((rows.len(), chunks));
        start += rows.len();
        if start >= nrows {
            break;
        }
    }
    let mut f = Tw::new();
    f.i32f(1, 1);
    f.list_begin(2, 12, ncol + 1);
    f.elem_struct_begin();
    f.binf(4, b"schema");
    f.i32f(5, i32::try_from(ncol)?);
    f.struct_end();
    for col in &table.columns {
        f.elem_struct_begin();
        f.i32f(1, col.ty.physical());
        f.i32f(3, if col.required { 0 } else { 1 });
        f.binf(4, col.name.as_bytes());
        match col.ty {
            ColumnType::Utf8 => f.i32f(6, 0),            // ConvertedType UTF8
            ColumnType::TimestampMillis => f.i32f(6, 9), // ConvertedType TIMESTAMP_MILLIS
            _ => {}
        }
        f.struct_end();
    }
    f.i64f(3, nrows as i64);
    f.list_begin(4, 12, groups.len());
    for (rows, chunks) in &groups {
        f.elem_struct_begin();
        f.list_begin(1, 12, ncol);
        for (col, m) in table.columns.iter().zip(chunks) {
            f.elem_struct_begin();
            f.i64f(2, m.file_offset);
            f.struct_begin(3);
            f.i32f(1, col.ty.physical());
            if m.dictionary {
                f.list_begin(2, 5, 3);
                f.elem_i32(0);
                f.elem_i32(2);
                f.elem_i32(3);
            } else {
                f.list_begin(2, 5, 2);
                f.elem_i32(0);
                f.elem_i32(3);
            }
            f.list_begin(3, 8, 1);
            f.elem_bin(col.name.as_bytes());
            f.i32f(4, opts.codec.id());
            f.i64f(5, *rows as i64);
            f.i64f(6, m.uncompressed);
            f.i64f(7, m.compressed);
            f.i64f(9, m.data_offset);
            if let Some(d) = m.dict_offset {
                f.i64f(11, d);
            }
            if let Some(stats) = &m.statistics {
                f.struct_begin(12);
                f.i64f(
                    3,
                    i64::try_from(stats.null_count.context("statistics null count")?)?,
                );
                if let Some(value) = &stats.max {
                    f.binf(5, &value.encode());
                }
                if let Some(value) = &stats.min {
                    f.binf(6, &value.encode());
                }
                if let Some(exact) = stats.max_is_exact {
                    f.field(7, if exact { 1 } else { 2 });
                }
                if let Some(exact) = stats.min_is_exact {
                    f.field(8, if exact { 1 } else { 2 });
                }
                if let Some(count) = stats.nan_count {
                    f.i64f(9, i64::try_from(count)?);
                }
                f.struct_end();
            }
            f.struct_end();
            f.struct_end();
        }
        f.i64f(2, chunks.iter().map(|m| m.uncompressed).sum());
        f.i64f(3, *rows as i64);
        f.struct_end();
    }
    f.list_begin(5, 12, table.metadata.len());
    for (k, v) in &table.metadata {
        f.elem_struct_begin();
        f.binf(1, k.as_bytes());
        f.binf(2, v.as_bytes());
        f.struct_end();
    }
    f.binf(6, created_by.as_bytes());
    if statistics {
        // Modern bounds have meaning only under an explicit ColumnOrder. TYPE_ORDER uses
        // signed numeric/Boolean order and unsigned lexicographic UTF8 bytes.
        f.list_begin(7, 12, ncol);
        for _ in &table.columns {
            f.elem_struct_begin();
            f.struct_begin(1); // ColumnOrder.TYPE_ORDER: empty TypeDefinedOrder
            f.struct_end();
            f.struct_end();
        }
    }
    let footer = f.finish();
    out.extend(&footer);
    out.extend((u32::try_from(footer.len())?).to_le_bytes());
    out.extend(MAGIC);
    Ok(out)
}

// ---- Thrift compact protocol: reader ----

#[derive(Clone, Debug)]
enum Tv {
    /// A list boolean or a double the reader consumes but does not use.
    Other,
    /// A struct-field boolean (compact protocol: the value is the field type).
    Bool(bool),
    Int(i64),
    Bin(Vec<u8>),
    List(Vec<Tv>),
    Struct(BTreeMap<i16, Tv>),
}

struct Tr<'a> {
    b: &'a [u8],
    p: usize,
    depth: usize,
}

impl<'a> Tr<'a> {
    fn byte(&mut self) -> Result<u8> {
        let v = *self.b.get(self.p).context("thrift: unexpected end")?;
        self.p += 1;
        Ok(v)
    }
    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.byte()?;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        bail!("thrift: varint too long")
    }
    fn zigzag(&mut self) -> Result<i64> {
        let v = self.varint()?;
        Ok(((v >> 1) as i64) ^ -((v & 1) as i64))
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.p.checked_add(n).context("thrift: length overflow")?;
        ensure!(end <= self.b.len(), "thrift: binary runs past the end");
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn value(&mut self, ty: u8) -> Result<Tv> {
        Ok(match ty {
            1 | 2 => Tv::Bool(ty == 1),
            3 => Tv::Int(i64::from(self.byte()? as i8)),
            4..=6 => Tv::Int(self.zigzag()?),
            7 => {
                self.bytes(8)?;
                Tv::Other
            }
            8 => {
                let n = usize::try_from(self.varint()?)?;
                Tv::Bin(self.bytes(n)?.to_vec())
            }
            9 | 10 => {
                let h = self.byte()?;
                let et = h & 0x0f;
                let mut n = u64::from(h >> 4);
                if n == 15 {
                    n = self.varint()?;
                }
                ensure!(
                    n <= self.b.len() as u64,
                    "thrift: list longer than the buffer"
                );
                let mut items = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    items.push(match et {
                        1 | 2 => {
                            self.byte()?;
                            Tv::Other
                        }
                        _ => self.value(et)?,
                    });
                }
                Tv::List(items)
            }
            12 => Tv::Struct(self.strukt()?),
            _ => bail!("thrift: unsupported type {ty}"),
        })
    }
    fn strukt(&mut self) -> Result<BTreeMap<i16, Tv>> {
        self.depth += 1;
        ensure!(self.depth < 32, "thrift: nesting too deep");
        let mut out = BTreeMap::new();
        let mut last = 0i16;
        loop {
            let h = self.byte()?;
            if h == 0 {
                break;
            }
            let ty = h & 0x0f;
            let delta = i16::from(h >> 4);
            let id = if delta == 0 {
                i16::try_from(self.zigzag()?)?
            } else {
                last + delta
            };
            last = id;
            out.insert(id, self.value(ty)?);
        }
        self.depth -= 1;
        Ok(out)
    }
}

fn int(m: &BTreeMap<i16, Tv>, id: i16) -> Option<i64> {
    match m.get(&id) {
        Some(Tv::Int(v)) => Some(*v),
        _ => None,
    }
}

fn bin(m: &BTreeMap<i16, Tv>, id: i16) -> Option<&[u8]> {
    match m.get(&id) {
        Some(Tv::Bin(v)) => Some(v),
        _ => None,
    }
}

fn list(m: &BTreeMap<i16, Tv>, id: i16) -> &[Tv] {
    match m.get(&id) {
        Some(Tv::List(v)) => v,
        _ => &[],
    }
}

fn strukt(v: &Tv) -> Result<&BTreeMap<i16, Tv>> {
    match v {
        Tv::Struct(m) => Ok(m),
        _ => bail!("parquet: expected a struct"),
    }
}

fn read_levels(data: &[u8], n: usize) -> Result<(Vec<bool>, usize)> {
    let (len, body) = length_prefixed(data)?;
    let out = rle_hybrid(body, 1, n)?;
    Ok((out.into_iter().map(|v| v == 1).collect(), 4 + len))
}

/// A 4-byte little-endian length and the bytes it covers.
fn length_prefixed(data: &[u8]) -> Result<(usize, &[u8])> {
    ensure!(data.len() >= 4, "parquet: RLE section truncated");
    let len = u32::from_le_bytes(data[..4].try_into()?) as usize;
    ensure!(
        4 + len <= data.len(),
        "parquet: RLE section runs past the page"
    );
    Ok((len, &data[4..4 + len]))
}

/// The RLE/bit-packed hybrid encoding (levels, dictionary indices, RLE booleans): `n` values of
/// `width` bits. Values decoded past `n` (padding of the last bit-packed group) are dropped.
fn rle_hybrid(data: &[u8], width: u32, n: usize) -> Result<Vec<u32>> {
    ensure!(width <= 32, "parquet: RLE bit width {width} too large");
    let mut r = Tr {
        b: data,
        p: 0,
        depth: 0,
    };
    let mut out = Vec::with_capacity(n);
    let vbytes = width.div_ceil(8) as usize;
    while out.len() < n {
        let h = r.varint()?;
        if h & 1 == 1 {
            let groups = usize::try_from(h >> 1)?;
            let packed = r.bytes(
                groups
                    .checked_mul(width as usize)
                    .context("parquet: RLE overflow")?,
            )?;
            let (mut acc, mut bits) = (0u64, 0u32);
            let mask = if width == 32 {
                u64::from(u32::MAX)
            } else {
                (1u64 << width) - 1
            };
            let mut it = packed.iter();
            for _ in 0..groups * 8 {
                while bits < width {
                    acc |=
                        u64::from(*it.next().context("parquet: bit-packed run truncated")?) << bits;
                    bits += 8;
                }
                out.push((acc & mask) as u32);
                acc >>= width;
                bits -= width;
            }
        } else {
            let run = usize::try_from(h >> 1)?;
            ensure!(run > 0, "parquet: empty RLE run");
            ensure!(
                out.len() + run <= n + 8,
                "parquet: RLE run past the page's values"
            );
            let v = u32::try_from(le_value(r.bytes(vbytes)?))?;
            out.extend(std::iter::repeat_n(v, run));
        }
    }
    out.truncate(n);
    Ok(out)
}

fn le_value(b: &[u8]) -> u64 {
    b.iter()
        .rev()
        .fold(0u64, |acc, &x| (acc << 8) | u64::from(x))
}

fn take_bytes<'a>(data: &'a [u8], p: &mut usize, k: usize) -> Result<&'a [u8]> {
    let end = p
        .checked_add(k)
        .context("parquet: value length overflows")?;
    ensure!(end <= data.len(), "parquet: values run past the page");
    let s = &data[*p..end];
    *p = end;
    Ok(s)
}

fn read_plain(ty: ColumnType, data: &[u8], n: usize) -> Result<Vec<Cell>> {
    let mut out = Vec::with_capacity(n);
    let mut p = 0usize;
    let mut take = |k: usize| take_bytes(data, &mut p, k);
    match ty {
        ColumnType::Boolean => {
            let bits = take(n.div_ceil(8))?;
            for i in 0..n {
                out.push(Cell::Bool(bits[i / 8] >> (i % 8) & 1 == 1));
            }
        }
        ColumnType::Int64 | ColumnType::TimestampMillis => {
            for _ in 0..n {
                out.push(Cell::I64(i64::from_le_bytes(take(8)?.try_into()?)));
            }
        }
        ColumnType::Double => {
            for _ in 0..n {
                out.push(Cell::F64(f64::from_le_bytes(take(8)?.try_into()?)));
            }
        }
        ColumnType::Utf8 => {
            for _ in 0..n {
                let len = u32::from_le_bytes(take(4)?.try_into()?) as usize;
                out.push(Cell::Str(
                    String::from_utf8(take(len)?.to_vec()).context("parquet: utf8 value")?,
                ));
            }
        }
    }
    Ok(out)
}

/// Reads a file written by [`write()`] (one row group, PLAIN, UNCOMPRESSED or ZSTD pages).
pub fn read(bytes: &[u8]) -> Result<Table> {
    ensure!(
        bytes.len() >= 12 && &bytes[..4] == MAGIC && &bytes[bytes.len() - 4..] == MAGIC,
        "not a parquet file"
    );
    let flen = u32::from_le_bytes(bytes[bytes.len() - 8..bytes.len() - 4].try_into()?) as usize;
    ensure!(
        flen + 12 <= bytes.len(),
        "parquet footer length {flen} exceeds the file"
    );
    let footer = &bytes[bytes.len() - 8 - flen..bytes.len() - 8];
    let meta = Tr {
        b: footer,
        p: 0,
        depth: 0,
    }
    .strukt()?;
    let nrows = usize::try_from(int(&meta, 3).context("parquet: num_rows missing")?)?;
    let schema = list(&meta, 2);
    ensure!(!schema.is_empty(), "parquet: empty schema");
    let mut columns = Vec::new();
    for el in &schema[1..] {
        let m = strukt(el)?;
        let name = String::from_utf8(bin(m, 4).context("parquet: column name")?.to_vec())?;
        let ty = ColumnType::from_physical(int(m, 1).context("parquet: column type")?, int(m, 6))?;
        let required = match int(m, 3) {
            Some(0) => true,
            Some(1) => false,
            other => bail!("parquet: column {name} repetition {other:?} unsupported"),
        };
        columns.push(ColumnSpec { name, ty, required });
    }
    let groups = list(&meta, 4);
    ensure!(
        nrows == 0 || !groups.is_empty(),
        "parquet: {nrows} rows but no row group"
    );
    ensure!(
        groups.len() <= 1,
        "parquet: {} row groups; this reader takes one",
        groups.len()
    );
    let mut cols: Vec<Vec<Cell>> = Vec::new();
    if let Some(g) = groups.first() {
        let chunks = list(strukt(g)?, 1);
        ensure!(
            chunks.len() == columns.len(),
            "parquet: {} column chunks for {} columns",
            chunks.len(),
            columns.len()
        );
        for (col, chunk) in columns.iter().zip(chunks) {
            let cm = strukt(
                strukt(chunk)?
                    .get(&3)
                    .context("parquet: column metadata missing")?,
            )?;
            let codec = int(cm, 4).unwrap_or(-1);
            let off = usize::try_from(int(cm, 9).context("parquet: data_page_offset")?)?;
            ensure!(off < bytes.len(), "parquet: page offset outside the file");
            let mut r = Tr {
                b: &bytes[off..],
                p: 0,
                depth: 0,
            };
            let ph = r.strukt()?;
            ensure!(
                int(&ph, 1) == Some(0),
                "parquet: column {} first page is not a v1 data page",
                col.name
            );
            let usize_ = usize::try_from(int(&ph, 2).context("parquet: page size")?)?;
            let csize = usize::try_from(int(&ph, 3).context("parquet: page size")?)?;
            ensure!(
                usize_ <= MAX_PAGE,
                "parquet: page larger than the reader cap"
            );
            let dph = strukt(ph.get(&5).context("parquet: data page header")?)?;
            ensure!(
                int(dph, 2) == Some(0),
                "parquet: column {} is not PLAIN",
                col.name
            );
            let n = usize::try_from(int(dph, 1).context("parquet: num_values")?)?;
            ensure!(
                n == nrows,
                "parquet: column {} has {n} values for {nrows} rows",
                col.name
            );
            let body = r.bytes(csize)?;
            let page = match codec {
                0 => body.to_vec(),
                6 => jope_records::zstd_frame::decompress(body, usize_)
                    .map_err(anyhow::Error::msg)?,
                c => bail!("parquet: codec {c} unsupported"),
            };
            ensure!(
                page.len() == usize_,
                "parquet: page decodes to {} bytes, header says {usize_}",
                page.len()
            );
            let (defined, used) = if col.required {
                (vec![true; n], 0)
            } else {
                read_levels(&page, n)?
            };
            let present = defined.iter().filter(|d| **d).count();
            let mut values = read_plain(col.ty, &page[used..], present)?.into_iter();
            cols.push(
                defined
                    .iter()
                    .map(|d| {
                        if *d {
                            values.next().unwrap_or(Cell::Null)
                        } else {
                            Cell::Null
                        }
                    })
                    .collect(),
            );
        }
    }
    let rows = (0..nrows)
        .map(|r| cols.iter().map(|c| c[r].clone()).collect())
        .collect();
    let metadata = list(&meta, 5)
        .iter()
        .map(|kv| {
            let m = strukt(kv)?;
            Ok((
                String::from_utf8(bin(m, 1).unwrap_or_default().to_vec())?,
                String::from_utf8(bin(m, 2).unwrap_or_default().to_vec())?,
            ))
        })
        .collect::<Result<_>>()?;
    Ok(Table {
        columns,
        rows,
        metadata,
    })
}

// ---- Streaming reader: many row groups and pages, dictionaries, Snappy (S3 Inventory) ----

/// Snappy, as S3 Inventory writes it.
const CODEC_SNAPPY: i64 = 1;
const CODEC_UNCOMPRESSED: i64 = 0;
const CODEC_ZSTD: i64 = 6;
const ENC_PLAIN: i64 = 0;
const ENC_PLAIN_DICTIONARY: i64 = 2;
const ENC_RLE: i64 = 3;
const ENC_RLE_DICTIONARY: i64 = 8;

/// One leaf column of a flat schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnMeta {
    /// Name recorded for this column meta.
    pub name: String,
    /// Parquet physical type (0 BOOLEAN, 1 INT32, 2 INT64, 4 FLOAT, 5 DOUBLE, 6 BYTE_ARRAY).
    pub physical: i64,
    /// Required recorded for this column meta.
    pub required: bool,
    /// For INT64 timestamps: divide by this to get milliseconds (1 millis, 1000 micros, 10^6
    /// nanos); `None` when the column is not a timestamp.
    pub timestamp_ms_divisor: Option<i64>,
}

#[derive(Clone, Debug)]
struct ChunkMeta {
    codec: i64,
    num_values: u64,
    start: usize,
    end: usize,
    statistics: Option<ColumnStatistics>,
}

#[derive(Clone, Debug)]
/// `RowGroupMeta` is a record for the row group meta contract in this module: A minimal Apache Parquet writer and reader for flat tables (the gpdistilled/v1 table).
pub struct RowGroupMeta {
    /// Number of rows represented by this value.
    pub rows: u64,
    chunks: Vec<ChunkMeta>,
}

/// Decoded values of one column chunk.
#[derive(Clone, Debug, PartialEq)]
pub enum Values {
    /// The `Bool` case of `Values`.
    Bool(Vec<Option<bool>>),
    /// The `I64` case of `Values`.
    I64(Vec<Option<i64>>),
    /// The `F64` case of `Values`.
    F64(Vec<Option<f64>>),
    /// The `Bytes` case of `Values`.
    Bytes(Vec<Option<Vec<u8>>>),
}

impl Values {
    /// Performs the len operation and returns an error if it cannot be completed.
    pub fn len(&self) -> usize {
        match self {
            Values::Bool(v) => v.len(),
            Values::I64(v) => v.len(),
            Values::F64(v) => v.len(),
            Values::Bytes(v) => v.len(),
        }
    }

    /// Returns whether empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What a chunk's pages looked like (for receipts: which decoder paths real files used).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct PageCounts {
    /// Dictionary recorded for this page counts.
    pub dictionary: u64,
    /// Data v1 recorded for this page counts.
    pub data_v1: u64,
    /// Data v2 recorded for this page counts.
    pub data_v2: u64,
    /// Index recorded for this page counts.
    pub index: u64,
    /// Encodings recorded for this page counts.
    pub encodings: std::collections::BTreeSet<i64>,
    /// Codecs recorded for this page counts.
    pub codecs: std::collections::BTreeSet<i64>,
}

impl PageCounts {
    /// Performs the add operation and returns an error if it cannot be completed.
    pub fn add(&mut self, o: &PageCounts) {
        self.dictionary += o.dictionary;
        self.data_v1 += o.data_v1;
        self.data_v2 += o.data_v2;
        self.index += o.index;
        self.encodings.extend(&o.encodings);
        self.codecs.extend(&o.codecs);
    }
}

/// A Parquet file opened for column-chunk reads. Flat schemas only; anything this reader does
/// not implement (nested columns, INT96, DELTA encodings, other codecs) is an error.
pub struct FileReader<'a> {
    bytes: &'a [u8],
    /// Collection of columns values carried by this file reader.
    pub columns: Vec<ColumnMeta>,
    /// Number of num rows represented here.
    pub num_rows: u64,
    /// Collection of row groups values carried by this file reader.
    pub row_groups: Vec<RowGroupMeta>,
    /// Optional created by value; `None` means it is not available.
    pub created_by: Option<String>,
}

fn timestamp_divisor(m: &BTreeMap<i16, Tv>) -> Result<Option<i64>> {
    // converted_type 9 TIMESTAMP_MILLIS, 10 TIMESTAMP_MICROS.
    match int(m, 6) {
        Some(9) => return Ok(Some(1)),
        Some(10) => return Ok(Some(1000)),
        _ => {}
    }
    // logicalType (10) -> TIMESTAMP (8) -> unit (2) -> MILLIS (1) | MICROS (2) | NANOS (3).
    let Some(Tv::Struct(lt)) = m.get(&10) else {
        return Ok(None);
    };
    let Some(Tv::Struct(ts)) = lt.get(&8) else {
        return Ok(None);
    };
    let Some(Tv::Struct(unit)) = ts.get(&2) else {
        bail!("parquet: timestamp without a unit");
    };
    Ok(Some(if unit.contains_key(&1) {
        1
    } else if unit.contains_key(&2) {
        1000
    } else if unit.contains_key(&3) {
        1_000_000
    } else {
        bail!("parquet: unknown timestamp unit")
    }))
}

fn decompress_page(codec: i64, body: &[u8], size: usize) -> Result<Vec<u8>> {
    ensure!(size <= MAX_PAGE, "parquet: page larger than the reader cap");
    let page = match codec {
        CODEC_UNCOMPRESSED => body.to_vec(),
        CODEC_SNAPPY => super::snappy::decompress(body, size)?,
        CODEC_ZSTD => {
            jope_records::zstd_frame::decompress(body, size).map_err(anyhow::Error::msg)?
        }
        c => bail!("parquet: codec {c} unsupported"),
    };
    ensure!(
        page.len() == size,
        "parquet: page decodes to {} bytes, header says {size}",
        page.len()
    );
    Ok(page)
}

/// PLAIN-decodes `n` values of one physical type.
fn plain_values(physical: i64, data: &[u8], n: usize, p: &mut usize) -> Result<Values> {
    Ok(match physical {
        0 => {
            let bits = take_bytes(data, p, n.div_ceil(8))?;
            Values::Bool(
                (0..n)
                    .map(|i| Some(bits[i / 8] >> (i % 8) & 1 == 1))
                    .collect(),
            )
        }
        1 => Values::I64(
            (0..n)
                .map(|_| {
                    Ok(Some(i64::from(i32::from_le_bytes(
                        take_bytes(data, p, 4)?.try_into()?,
                    ))))
                })
                .collect::<Result<_>>()?,
        ),
        2 => Values::I64(
            (0..n)
                .map(|_| {
                    Ok(Some(i64::from_le_bytes(
                        take_bytes(data, p, 8)?.try_into()?,
                    )))
                })
                .collect::<Result<_>>()?,
        ),
        5 => Values::F64(
            (0..n)
                .map(|_| {
                    Ok(Some(f64::from_le_bytes(
                        take_bytes(data, p, 8)?.try_into()?,
                    )))
                })
                .collect::<Result<_>>()?,
        ),
        6 => Values::Bytes(
            (0..n)
                .map(|_| {
                    let len = u32::from_le_bytes(take_bytes(data, p, 4)?.try_into()?) as usize;
                    Ok(Some(take_bytes(data, p, len)?.to_vec()))
                })
                .collect::<Result<_>>()?,
        ),
        t => bail!("parquet: physical type {t} unsupported"),
    })
}

fn gather(dict: &Values, idx: &[u32]) -> Result<Values> {
    let get = |i: u32| -> Result<usize> {
        let i = i as usize;
        ensure!(i < dict.len(), "parquet: dictionary index {i} out of range");
        Ok(i)
    };
    Ok(match dict {
        Values::Bool(d) => {
            Values::Bool(idx.iter().map(|&i| Ok(d[get(i)?])).collect::<Result<_>>()?)
        }
        Values::I64(d) => Values::I64(idx.iter().map(|&i| Ok(d[get(i)?])).collect::<Result<_>>()?),
        Values::F64(d) => Values::F64(idx.iter().map(|&i| Ok(d[get(i)?])).collect::<Result<_>>()?),
        Values::Bytes(d) => Values::Bytes(
            idx.iter()
                .map(|&i| Ok(d[get(i)?].clone()))
                .collect::<Result<_>>()?,
        ),
    })
}

/// Spreads `present` non-null values over `defined` slots.
fn spread(present: Values, defined: &[bool]) -> Result<Values> {
    fn go<T: Clone>(v: Vec<Option<T>>, defined: &[bool]) -> Result<Vec<Option<T>>> {
        let mut it = v.into_iter();
        let out: Vec<Option<T>> = defined
            .iter()
            .map(|d| if *d { it.next().flatten() } else { None })
            .collect();
        ensure!(
            it.next().is_none(),
            "parquet: more values than defined slots"
        );
        Ok(out)
    }
    Ok(match present {
        Values::Bool(v) => Values::Bool(go(v, defined)?),
        Values::I64(v) => Values::I64(go(v, defined)?),
        Values::F64(v) => Values::F64(go(v, defined)?),
        Values::Bytes(v) => Values::Bytes(go(v, defined)?),
    })
}

fn append(acc: &mut Option<Values>, v: Values) -> Result<()> {
    match (acc.as_mut(), v) {
        (None, v) => *acc = Some(v),
        (Some(Values::Bool(a)), Values::Bool(b)) => a.extend(b),
        (Some(Values::I64(a)), Values::I64(b)) => a.extend(b),
        (Some(Values::F64(a)), Values::F64(b)) => a.extend(b),
        (Some(Values::Bytes(a)), Values::Bytes(b)) => a.extend(b),
        _ => bail!("parquet: pages of one column decode to different types"),
    }
    Ok(())
}

impl<'a> FileReader<'a> {
    /// Constructs or opens a open value from the supplied inputs.
    pub fn open(bytes: &'a [u8]) -> Result<FileReader<'a>> {
        ensure!(
            bytes.len() >= 12 && &bytes[..4] == MAGIC && &bytes[bytes.len() - 4..] == MAGIC,
            "not a parquet file"
        );
        let flen = u32::from_le_bytes(bytes[bytes.len() - 8..bytes.len() - 4].try_into()?) as usize;
        ensure!(
            flen + 12 <= bytes.len(),
            "parquet footer length {flen} exceeds the file"
        );
        let footer = &bytes[bytes.len() - 8 - flen..bytes.len() - 8];
        let meta = Tr {
            b: footer,
            p: 0,
            depth: 0,
        }
        .strukt()?;
        let num_rows = u64::try_from(int(&meta, 3).context("parquet: num_rows missing")?)?;
        let schema = list(&meta, 2);
        ensure!(!schema.is_empty(), "parquet: empty schema");
        let root = strukt(&schema[0])?;
        ensure!(
            int(root, 5) == Some(schema.len() as i64 - 1),
            "parquet: nested schema (root has {:?} children for {} elements)",
            int(root, 5),
            schema.len() - 1
        );
        let mut columns = Vec::new();
        for el in &schema[1..] {
            let m = strukt(el)?;
            let name = String::from_utf8(bin(m, 4).context("parquet: column name")?.to_vec())?;
            ensure!(
                int(m, 5).is_none(),
                "parquet: column {name} is a group (nested)"
            );
            let physical = int(m, 1).context("parquet: column type")?;
            let required = match int(m, 3) {
                Some(0) => true,
                Some(1) => false,
                other => bail!("parquet: column {name} repetition {other:?} unsupported"),
            };
            let timestamp_ms_divisor = timestamp_divisor(m)?;
            ensure!(
                timestamp_ms_divisor.is_none() || physical == 2,
                "parquet: column {name} is a non-INT64 timestamp (INT96 unsupported)"
            );
            columns.push(ColumnMeta {
                name,
                physical,
                required,
                timestamp_ms_divisor,
            });
        }
        let orders = list(&meta, 7);
        ensure!(
            orders.is_empty() || orders.len() == columns.len(),
            "parquet: column order count mismatch"
        );
        let mut row_groups = Vec::new();
        let mut total = 0u64;
        for g in list(&meta, 4) {
            let g = strukt(g)?;
            let rows = u64::try_from(int(g, 3).context("parquet: row group num_rows")?)?;
            total += rows;
            let chunks = list(g, 1);
            ensure!(
                chunks.len() == columns.len(),
                "parquet: {} column chunks for {} columns",
                chunks.len(),
                columns.len()
            );
            let mut metas = Vec::new();
            for (ci, (col, chunk)) in columns.iter().zip(chunks).enumerate() {
                let cm = strukt(
                    strukt(chunk)?
                        .get(&3)
                        .context("parquet: column metadata missing (external chunk?)")?,
                )?;
                let path: Vec<&[u8]> = list(cm, 3)
                    .iter()
                    .filter_map(|v| match v {
                        Tv::Bin(b) => Some(b.as_slice()),
                        _ => None,
                    })
                    .collect();
                ensure!(
                    path == [col.name.as_bytes()],
                    "parquet: chunk path does not match column {}",
                    col.name
                );
                let data_off = usize::try_from(int(cm, 9).context("parquet: data_page_offset")?)?;
                let start = match int(cm, 11) {
                    Some(d) if d > 0 && (d as usize) < data_off => d as usize,
                    _ => data_off,
                };
                let size = usize::try_from(int(cm, 7).context("parquet: total_compressed_size")?)?;
                let end = start.checked_add(size).context("parquet: chunk overflow")?;
                ensure!(
                    end <= bytes.len() - 8 - flen,
                    "parquet: column {} chunk runs into the footer",
                    col.name
                );
                let num_values = u64::try_from(int(cm, 5).context("parquet: chunk num_values")?)?;
                ensure!(
                    num_values == rows,
                    "parquet: column {} has {num_values} values for {rows} rows",
                    col.name
                );
                metas.push(ChunkMeta {
                    codec: int(cm, 4).context("parquet: codec")?,
                    num_values,
                    start,
                    end,
                    statistics: cm
                        .get(&12)
                        .map(|s| {
                            decode_statistics(
                                strukt(s)?,
                                col.physical,
                                rows,
                                orders
                                    .get(ci)
                                    .and_then(|o| strukt(o).ok())
                                    .is_some_and(|m| m.contains_key(&1)),
                            )
                        })
                        .transpose()?,
                });
            }
            row_groups.push(RowGroupMeta {
                rows,
                chunks: metas,
            });
        }
        ensure!(
            total == num_rows,
            "parquet: row groups hold {total} rows, footer says {num_rows}"
        );
        let created_by = bin(&meta, 6).map(|b| String::from_utf8_lossy(b).to_string());
        Ok(FileReader {
            bytes,
            columns,
            num_rows,
            row_groups,
            created_by,
        })
    }

    /// Performs the column index operation and returns an error if it cannot be completed.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Footer statistics as typed values. None means absent, never a fabricated zero.
    pub fn statistics(&self, group: usize, col: usize) -> Result<Option<&ColumnStatistics>> {
        let chunk = self
            .row_groups
            .get(group)
            .context("parquet: no such row group")?
            .chunks
            .get(col)
            .context("parquet: no such column")?;
        Ok(chunk.statistics.as_ref())
    }

    /// Decodes column `col` of row group `group`: every page, with its dictionary.
    pub fn read_column(&self, group: usize, col: usize) -> Result<(Values, PageCounts)> {
        let rg = self
            .row_groups
            .get(group)
            .context("parquet: no such row group")?;
        let c = self.columns.get(col).context("parquet: no such column")?;
        let ch = &rg.chunks[col];
        let chunk = &self.bytes[ch.start..ch.end];
        let mut counts = PageCounts::default();
        counts.codecs.insert(ch.codec);
        let mut dict: Option<Values> = None;
        let mut acc: Option<Values> = None;
        let mut done = 0u64;
        let mut p = 0usize;
        while done < ch.num_values {
            ensure!(
                p < chunk.len(),
                "parquet: column {} ends before its values",
                c.name
            );
            let mut r = Tr {
                b: &chunk[p..],
                p: 0,
                depth: 0,
            };
            let ph = r.strukt()?;
            let header_len = r.p;
            let ty = int(&ph, 1).context("parquet: page type")?;
            let usize_ = usize::try_from(int(&ph, 2).context("parquet: page size")?)?;
            let csize = usize::try_from(int(&ph, 3).context("parquet: page size")?)?;
            let body = take_bytes(chunk, &mut (p + header_len), csize)?;
            p += header_len + csize;
            match ty {
                2 => {
                    counts.dictionary += 1;
                    ensure!(
                        dict.is_none(),
                        "parquet: column {} has two dictionaries",
                        c.name
                    );
                    let dh = strukt(ph.get(&7).context("parquet: dictionary page header")?)?;
                    let n = usize::try_from(int(dh, 1).context("parquet: dictionary size")?)?;
                    let enc = int(dh, 2).unwrap_or(ENC_PLAIN);
                    ensure!(
                        enc == ENC_PLAIN || enc == ENC_PLAIN_DICTIONARY,
                        "parquet: dictionary encoding {enc} unsupported"
                    );
                    let page = decompress_page(ch.codec, body, usize_)?;
                    dict = Some(plain_values(c.physical, &page, n, &mut 0)?);
                }
                0 | 3 => {
                    let (n, enc, defined, values): (usize, i64, Vec<bool>, Vec<u8>) = if ty == 0 {
                        counts.data_v1 += 1;
                        let dh = strukt(ph.get(&5).context("parquet: data page header")?)?;
                        let n = usize::try_from(int(dh, 1).context("parquet: num_values")?)?;
                        let page = decompress_page(ch.codec, body, usize_)?;
                        let (defined, used) = if c.required {
                            (vec![true; n], 0)
                        } else {
                            ensure!(
                                int(dh, 3) == Some(ENC_RLE),
                                "parquet: column {} definition levels not RLE",
                                c.name
                            );
                            read_levels(&page, n)?
                        };
                        let enc = int(dh, 2).context("parquet: encoding")?;
                        (n, enc, defined, page[used..].to_vec())
                    } else {
                        counts.data_v2 += 1;
                        let dh = strukt(ph.get(&8).context("parquet: data page v2 header")?)?;
                        let n = usize::try_from(int(dh, 1).context("parquet: num_values")?)?;
                        let dlen = usize::try_from(int(dh, 5).unwrap_or(0))?;
                        let rlen = usize::try_from(int(dh, 6).unwrap_or(0))?;
                        ensure!(rlen == 0, "parquet: repetition levels in a flat column");
                        let compressed = !matches!(dh.get(&7), Some(Tv::Bool(false)));
                        ensure!(
                            dlen <= body.len() && dlen <= usize_,
                            "parquet: v2 levels past the page"
                        );
                        let defined = if c.required {
                            vec![true; n]
                        } else {
                            rle_hybrid(&body[..dlen], 1, n)?
                                .into_iter()
                                .map(|v| v == 1)
                                .collect()
                        };
                        let rest = &body[dlen..];
                        let values = if compressed {
                            decompress_page(ch.codec, rest, usize_ - dlen)?
                        } else {
                            rest.to_vec()
                        };
                        (n, int(dh, 4).context("parquet: encoding")?, defined, values)
                    };
                    counts.encodings.insert(enc);
                    let present = defined.iter().filter(|d| **d).count();
                    let vals = match enc {
                        ENC_PLAIN => plain_values(c.physical, &values, present, &mut 0)?,
                        ENC_PLAIN_DICTIONARY | ENC_RLE_DICTIONARY => {
                            let d = dict
                                .as_ref()
                                .context("parquet: dictionary-encoded page without a dictionary")?;
                            let width = u32::from(
                                *values.first().context("parquet: empty dictionary page")?,
                            );
                            gather(d, &rle_hybrid(&values[1..], width, present)?)?
                        }
                        ENC_RLE if c.physical == 0 => {
                            let (_, b) = length_prefixed(&values)?;
                            Values::Bool(
                                rle_hybrid(b, 1, present)?
                                    .into_iter()
                                    .map(|v| Some(v == 1))
                                    .collect(),
                            )
                        }
                        e => bail!("parquet: column {} encoding {e} unsupported", c.name),
                    };
                    ensure!(
                        vals.len() == present,
                        "parquet: column {} decoded {} of {present} values",
                        c.name,
                        vals.len()
                    );
                    append(&mut acc, spread(vals, &defined)?)?;
                    done += n as u64;
                }
                1 => counts.index += 1,
                t => bail!("parquet: page type {t} unsupported"),
            }
        }
        ensure!(
            done == ch.num_values,
            "parquet: column {} decoded {done} values, chunk says {}",
            c.name,
            ch.num_values
        );
        let vals = match acc {
            Some(v) => v,
            None => plain_values(c.physical, &[], 0, &mut 0)?,
        };
        Ok((vals, counts))
    }
}
