//! Version1 column-major matrices; see `docs/product/jmat-format.md`.
use jope_runtime::table::CsvTable;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const MAGIC: &[u8; 8] = b"JMAT\r\n\x1a\n";
const HEADER: usize = 64;
const CHECKSUM: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidMatrix;
impl std::fmt::Display for InvalidMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("invalid version1 matrix")
    }
}
impl std::error::Error for InvalidMatrix {}
type Result<T> = std::result::Result<T, InvalidMatrix>;

pub use jope_runtime::table::ColumnKind;
fn width(kind: ColumnKind) -> usize {
    match kind {
        ColumnKind::Numeric => 8,
        ColumnKind::Categorical => 4,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Column {
    pub name: String,
    pub kind: ColumnKind,
    pub dictionary: Vec<String>,
    values: u64,
    present: u64,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    columns: Vec<Column>,
}

/// Values retain f64 bits, rather than conflating signed zeros in round-trip tests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JmatCell<'a> {
    Missing,
    Numeric(f64),
    Categorical(&'a str),
}

/// Validated immutable matrix backed by either bytes or an OS mapping.
pub struct Matrix<B> {
    storage: B,
    metadata: Metadata,
    rows: usize,
    row_ids: usize,
}

fn aligned(value: usize) -> Result<usize> {
    value.checked_add(63).map(|n| n & !63).ok_or(InvalidMatrix)
}
fn integer(bytes: &[u8], offset: usize) -> Result<usize> {
    let end = offset.checked_add(8).ok_or(InvalidMatrix)?;
    let raw = bytes.get(offset..end).ok_or(InvalidMatrix)?;
    usize::try_from(u64::from_le_bytes(
        raw.try_into().map_err(|_| InvalidMatrix)?,
    ))
    .map_err(|_| InvalidMatrix)
}
fn zero_padding(bytes: &[u8], from: usize, to: usize) -> Result<()> {
    if bytes
        .get(from..to)
        .ok_or(InvalidMatrix)?
        .iter()
        .any(|b| *b != 0)
    {
        return Err(InvalidMatrix);
    }
    Ok(())
}
fn numeric(value: &str) -> Option<f64> {
    value.trim().parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Encode the runtime parser's cells without choosing targets or fitting transforms.
pub fn encode(table: &CsvTable) -> Result<Vec<u8>> {
    if table.headers.is_empty()
        || table.headers.iter().any(|name| name.trim().is_empty())
        || table.headers.iter().collect::<BTreeSet<_>>().len() != table.headers.len()
        || table.rows.iter().any(|r| {
            r.len() != table.headers.len()
                || r.iter()
                    .flatten()
                    .any(|v| v.trim_matches(|c: char| c.is_ascii_whitespace()).is_empty())
        })
    {
        return Err(InvalidMatrix);
    }
    let mut metadata = Metadata {
        columns: Vec::new(),
    };
    for (index, name) in table.headers.iter().enumerate() {
        let kind = if table
            .rows
            .iter()
            .filter_map(|r| r[index].as_deref())
            .all(|s| numeric(s).is_some())
        {
            ColumnKind::Numeric
        } else {
            ColumnKind::Categorical
        };
        let dictionary = if kind == ColumnKind::Categorical {
            table
                .rows
                .iter()
                .filter_map(|r| r[index].clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };
        metadata.columns.push(Column {
            name: name.clone(),
            kind,
            dictionary,
            values: 0,
            present: 0,
        });
    }
    let preliminary = serde_json::to_vec(&metadata).map_err(|_| InvalidMatrix)?;
    let reserve = metadata
        .columns
        .len()
        .checked_mul(1024)
        .ok_or(InvalidMatrix)?;
    let row_ids = aligned(
        HEADER
            .checked_add(preliminary.len())
            .and_then(|n| n.checked_add(reserve))
            .ok_or(InvalidMatrix)?,
    )?;
    let mut bytes = vec![0; row_ids];
    for row in 0..table.rows.len() {
        bytes.extend_from_slice(&(row as u64).to_le_bytes());
    }
    for (index, column) in metadata.columns.iter_mut().enumerate() {
        bytes.resize(aligned(bytes.len())?, 0);
        column.values = bytes.len() as u64;
        for row in &table.rows {
            match column.kind {
                ColumnKind::Numeric => {
                    let value = match row[index].as_deref() {
                        None => 0.0,
                        Some(text) => numeric(text).ok_or(InvalidMatrix)?,
                    };
                    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                }
                ColumnKind::Categorical => {
                    let code = match &row[index] {
                        None => 0,
                        Some(value) => u32::try_from(
                            column
                                .dictionary
                                .binary_search(value)
                                .map_err(|_| InvalidMatrix)?,
                        )
                        .map_err(|_| InvalidMatrix)?,
                    };
                    bytes.extend_from_slice(&code.to_le_bytes());
                }
            }
        }
        bytes.resize(aligned(bytes.len())?, 0);
        column.present = bytes.len() as u64;
        bytes.extend(table.rows.iter().map(|r| u8::from(r[index].is_some())));
    }
    let metadata_bytes = serde_json::to_vec(&metadata).map_err(|_| InvalidMatrix)?;
    if HEADER
        .checked_add(metadata_bytes.len())
        .ok_or(InvalidMatrix)?
        > row_ids
    {
        return Err(InvalidMatrix);
    }
    let total = bytes.len().checked_add(CHECKSUM).ok_or(InvalidMatrix)?;
    bytes[..8].copy_from_slice(MAGIC);
    for (offset, value) in [
        (8, 1),
        (16, total),
        (24, metadata_bytes.len()),
        (32, table.rows.len()),
        (40, table.headers.len()),
        (48, row_ids),
    ] {
        bytes[offset..offset + 8].copy_from_slice(&(value as u64).to_le_bytes());
    }
    bytes[HEADER..HEADER + metadata_bytes.len()].copy_from_slice(&metadata_bytes);
    let checksum = Sha256::digest(&bytes);
    bytes.extend_from_slice(&checksum);
    Ok(bytes)
}

impl<B: AsRef<[u8]>> Matrix<B> {
    pub fn open(storage: B) -> Result<Self> {
        let bytes = storage.as_ref();
        let end = bytes.len().checked_sub(CHECKSUM).ok_or(InvalidMatrix)?;
        if end < HEADER
            || bytes.get(..8) != Some(MAGIC.as_slice())
            || integer(bytes, 8)? != 1
            || integer(bytes, 16)? != bytes.len()
            || integer(bytes, 56)? != 0
            || Sha256::digest(&bytes[..end]).as_slice() != &bytes[end..]
        {
            return Err(InvalidMatrix);
        }
        let metadata_end = HEADER
            .checked_add(integer(bytes, 24)?)
            .ok_or(InvalidMatrix)?;
        let metadata: Metadata =
            serde_json::from_slice(bytes.get(HEADER..metadata_end).ok_or(InvalidMatrix)?)
                .map_err(|_| InvalidMatrix)?;
        let rows = integer(bytes, 32)?;
        let row_ids = integer(bytes, 48)?;
        if metadata.columns.is_empty()
            || integer(bytes, 40)? != metadata.columns.len()
            || metadata.columns.iter().any(|c| c.name.trim().is_empty())
            || metadata
                .columns
                .iter()
                .map(|c| &c.name)
                .collect::<BTreeSet<_>>()
                .len()
                != metadata.columns.len()
            || !row_ids.is_multiple_of(64)
        {
            return Err(InvalidMatrix);
        }
        zero_padding(bytes, metadata_end, row_ids)?;
        let mut cursor = row_ids
            .checked_add(rows.checked_mul(8).ok_or(InvalidMatrix)?)
            .ok_or(InvalidMatrix)?;
        if cursor > end {
            return Err(InvalidMatrix);
        }
        for row in 0..rows {
            if integer(bytes, row_ids + row * 8)? != row {
                return Err(InvalidMatrix);
            }
        }
        for column in &metadata.columns {
            let start = aligned(cursor)?;
            zero_padding(bytes, cursor, start)?;
            if usize::try_from(column.values).map_err(|_| InvalidMatrix)? != start
                || (column.kind == ColumnKind::Numeric && !column.dictionary.is_empty())
                || column.dictionary.windows(2).any(|w| w[0] >= w[1])
                || column
                    .dictionary
                    .iter()
                    .any(|v| v.trim_matches(|c: char| c.is_ascii_whitespace()).is_empty())
            {
                return Err(InvalidMatrix);
            }
            cursor = start
                .checked_add(rows.checked_mul(width(column.kind)).ok_or(InvalidMatrix)?)
                .ok_or(InvalidMatrix)?;
            let present = aligned(cursor)?;
            zero_padding(bytes, cursor, present)?;
            if usize::try_from(column.present).map_err(|_| InvalidMatrix)? != present {
                return Err(InvalidMatrix);
            }
            cursor = present.checked_add(rows).ok_or(InvalidMatrix)?;
            if cursor > end {
                return Err(InvalidMatrix);
            }
            for row in 0..rows {
                let flag = bytes[present + row];
                let at = start + row * width(column.kind);
                let value = &bytes[at..at + width(column.kind)];
                if flag > 1 || (flag == 0 && value.iter().any(|b| *b != 0)) {
                    return Err(InvalidMatrix);
                }
                if flag == 1 {
                    match column.kind {
                        ColumnKind::Numeric => {
                            if !f64::from_le_bytes(value.try_into().map_err(|_| InvalidMatrix)?)
                                .is_finite()
                            {
                                return Err(InvalidMatrix);
                            }
                        }
                        ColumnKind::Categorical => {
                            if u32::from_le_bytes(value.try_into().map_err(|_| InvalidMatrix)?)
                                as usize
                                >= column.dictionary.len()
                            {
                                return Err(InvalidMatrix);
                            }
                        }
                    }
                }
            }
        }
        if cursor != end {
            return Err(InvalidMatrix);
        }
        Ok(Self {
            storage,
            metadata,
            rows,
            row_ids,
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> &[Column] {
        &self.metadata.columns
    }
    pub fn bytes(&self) -> &[u8] {
        self.storage.as_ref()
    }
    pub fn row_id(&self, row: usize) -> Option<u64> {
        (row < self.rows).then(|| {
            integer(self.bytes(), self.row_ids + row * 8).expect("validated row ID") as u64
        })
    }
    pub fn cell(&self, row: usize, column: usize) -> Option<JmatCell<'_>> {
        if row >= self.rows {
            return None;
        }
        let column = self.metadata.columns.get(column)?;
        let bytes = self.bytes();
        if bytes[column.present as usize + row] == 0 {
            return Some(JmatCell::Missing);
        }
        let start = column.values as usize + row * width(column.kind);
        Some(match column.kind {
            ColumnKind::Numeric => JmatCell::Numeric(f64::from_le_bytes(
                bytes[start..start + 8].try_into().expect("validated value"),
            )),
            ColumnKind::Categorical => {
                let code =
                    u32::from_le_bytes(bytes[start..start + 4].try_into().expect("validated code"));
                JmatCell::Categorical(&column.dictionary[code as usize])
            }
        })
    }
}

/// Map an immutable regular matrix file with read-only OS permissions.
///
/// # Safety
/// The caller must ensure no process writes to or truncates the underlying file
/// for the lifetime of the returned matrix. Read-only permissions on this view
/// alone do not enforce that condition. Engine CAS ownership must enforce it.
#[allow(unsafe_code)]
pub unsafe fn map_read_only(file: &std::fs::File) -> std::io::Result<Matrix<memmap2::Mmap>> {
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.len() < (HEADER + CHECKSUM) as u64
        || metadata.len() > isize::MAX as u64
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            InvalidMatrix,
        ));
    }
    // SAFETY: forwarded immutable-file lifetime contract; bounds checked above.
    let mapping = unsafe { memmap2::MmapOptions::new().map(file)? };
    Matrix::open(mapping).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests;
