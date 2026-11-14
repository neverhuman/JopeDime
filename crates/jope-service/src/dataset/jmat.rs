//! Version1 column-major matrices; see `docs/product/jmat-format.md`.
//!
//! Backed by the canonical standalone JMAT v1 codec in [`jope_prep::jmat_lite`].
use jope_prep::jmat_lite::{encode_jmat_v1, PrepColumnKind, PrepMatrix, PrepTable};
use jope_runtime::table::CsvTable;
use serde::{Deserialize, Serialize};

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
    columns: Vec<Column>,
    rows: usize,
    row_ids: usize,
    sha256_hex: String,
}

fn integer(bytes: &[u8], offset: usize) -> Result<usize> {
    let end = offset.checked_add(8).ok_or(InvalidMatrix)?;
    let raw = bytes.get(offset..end).ok_or(InvalidMatrix)?;
    usize::try_from(u64::from_le_bytes(
        raw.try_into().map_err(|_| InvalidMatrix)?,
    ))
    .map_err(|_| InvalidMatrix)
}

/// Encode the runtime parser's cells without choosing targets or fitting transforms.
pub fn encode(table: &CsvTable) -> Result<Vec<u8>> {
    let prep = PrepTable {
        headers: table.headers.clone(),
        rows: table.rows.clone(),
    };
    encode_jmat_v1(&prep).map_err(|_| InvalidMatrix)
}

impl<B: AsRef<[u8]>> Matrix<B> {
    pub fn open(storage: B) -> Result<Self> {
        let validated = PrepMatrix::open(storage).map_err(|_| InvalidMatrix)?;
        let (storage, prep_cols, rows, row_ids, sha256_hex) = validated.into_parts();
        let columns = prep_cols
            .into_iter()
            .map(|c| {
                let values = c.values_offset();
                let present = c.present_offset();
                Column {
                    name: c.name,
                    kind: match c.kind {
                        PrepColumnKind::Numeric => ColumnKind::Numeric,
                        PrepColumnKind::Categorical => ColumnKind::Categorical,
                    },
                    dictionary: c.dictionary,
                    values,
                    present,
                }
            })
            .collect();
        Ok(Self {
            storage,
            columns,
            rows,
            row_ids,
            sha256_hex,
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }
    pub fn bytes(&self) -> &[u8] {
        self.storage.as_ref()
    }
    pub fn sha256_hex(&self) -> &str {
        &self.sha256_hex
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
        let column = self.columns.get(column)?;
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
