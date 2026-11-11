//! Standalone, WebAssembly-compatible JMAT v1 column-major matrix encoder and reader.
//!
//! Produces the exact binary layout consumed by `jope_service::dataset::jmat::Matrix::open`
//! (`JMAT\r\n\x1a\n` magic, 64-byte header, JSON column metadata, 64-byte aligned row IDs,
//! column values, presence flags, and trailing 32-byte SHA-256 checksum), plus split-bound
//! `fit.jmat` / `holdout.jmat` emission ensuring holdout labels never enter the FIT artifact.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{PrepError, PrepResult};
use crate::split::SplitPlan;

const MAGIC: &[u8; 8] = b"JMAT\r\n\x1a\n";
const HEADER: usize = 64;
const CHECKSUM: usize = 32;

/// `PrepColumnKind` is an enum for the JMAT v1 column storage type (`numeric` or `categorical`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrepColumnKind {
    /// 64-bit IEEE-754 little-endian finite float (`8` bytes per cell).
    Numeric,
    /// 32-bit little-endian dictionary code (`4` bytes per cell).
    Categorical,
}

fn width(kind: PrepColumnKind) -> usize {
    match kind {
        PrepColumnKind::Numeric => 8,
        PrepColumnKind::Categorical => 4,
    }
}

/// `PrepColumn` is a struct for the JMAT v1 column header metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrepColumn {
    /// Column name (unique, non-empty).
    pub name: String,
    /// Storage representation (`numeric` or `categorical`).
    pub kind: PrepColumnKind,
    /// Sorted unique category strings when `kind == Categorical`; empty when `Numeric`.
    pub dictionary: Vec<String>,
    values: u64,
    present: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JmatMetadata {
    columns: Vec<PrepColumn>,
}

/// `PrepCell` is an enum for a borrowed cell read back from a validated `PrepMatrix`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrepCell<'a> {
    /// Missing cell (`present == 0`).
    Missing,
    /// Finite `f64` value (`present == 1`).
    Numeric(f64),
    /// Interned category string (`present == 1`).
    Categorical(&'a str),
}

/// `PrepTable` is a struct for an unencoded rectangular string/missing-cell table in `jope-prep`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrepTable {
    /// Column names in order.
    pub headers: Vec<String>,
    /// Row-major cells (`None` = missing).
    pub rows: Vec<Vec<Option<String>>>,
}

/// `SplitJmatArtifacts` is a struct for the separated `fit.jmat` and `holdout.jmat` byte buffers
/// and their SHA-256 digests produced from a `PrepTable` and a `SplitPlan`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitJmatArtifacts {
    /// JMAT v1 bytes containing strictly `plan.fit_rows()`.
    pub fit_jmat: Vec<u8>,
    /// SHA-256 digest of `fit_jmat`.
    pub fit_sha256: [u8; 32],
    /// JMAT v1 bytes containing strictly `plan.holdout_rows()`.
    pub holdout_jmat: Vec<u8>,
    /// SHA-256 digest of `holdout_jmat`.
    pub holdout_sha256: [u8; 32],
}

impl PrepTable {
    /// Slice a subset of rows in the given index order without reading unselected rows.
    pub fn select_rows(&self, indices: &[usize]) -> PrepResult<Self> {
        let mut selected = Vec::with_capacity(indices.len());
        for &idx in indices {
            let row = self.rows.get(idx).ok_or(PrepError::InvalidSplitIndex {
                index: idx,
                total_rows: self.rows.len(),
            })?;
            selected.push(row.clone());
        }
        Ok(Self {
            headers: self.headers.clone(),
            rows: selected,
        })
    }
}

/// Partition `table` strictly according to `plan` and encode separate `fit.jmat` and `holdout.jmat` buffers.
pub fn split_and_encode_jmat_v1(
    table: &PrepTable,
    plan: &SplitPlan,
) -> PrepResult<SplitJmatArtifacts> {
    plan.validate_for_table(table.rows.len())?;
    let fit_table = table.select_rows(plan.fit_rows())?;
    let holdout_table = table.select_rows(plan.holdout_rows())?;
    let fit_jmat = encode_jmat_v1(&fit_table)?;
    let holdout_jmat = encode_jmat_v1(&holdout_table)?;
    let fit_sha256: [u8; 32] = Sha256::digest(&fit_jmat).into();
    let holdout_sha256: [u8; 32] = Sha256::digest(&holdout_jmat).into();
    Ok(SplitJmatArtifacts {
        fit_jmat,
        fit_sha256,
        holdout_jmat,
        holdout_sha256,
    })
}

fn aligned(value: usize) -> PrepResult<usize> {
    value
        .checked_add(63)
        .map(|n| n & !63)
        .ok_or(PrepError::ArithmeticOverflow)
}

fn parse_finite_numeric(value: &str) -> Option<f64> {
    value.trim().parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Encode `table` into canonical JMAT v1 bytes with trailing SHA-256 checksum.
pub fn encode_jmat_v1(table: &PrepTable) -> PrepResult<Vec<u8>> {
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
        return Err(PrepError::InvalidJmat {
            reason: "headers must be unique/non-empty and cells must match column count",
        });
    }

    let mut metadata = JmatMetadata {
        columns: Vec::with_capacity(table.headers.len()),
    };
    for (index, name) in table.headers.iter().enumerate() {
        let kind = if table
            .rows
            .iter()
            .filter_map(|r| r[index].as_deref())
            .all(|s| parse_finite_numeric(s).is_some())
        {
            PrepColumnKind::Numeric
        } else {
            PrepColumnKind::Categorical
        };
        let dictionary = if kind == PrepColumnKind::Categorical {
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
        metadata.columns.push(PrepColumn {
            name: name.clone(),
            kind,
            dictionary,
            values: 0,
            present: 0,
        });
    }

    let preliminary = serde_json::to_vec(&metadata).map_err(|_| PrepError::InvalidJmat {
        reason: "failed to serialize preliminary column metadata",
    })?;
    let reserve = metadata
        .columns
        .len()
        .checked_mul(1024)
        .ok_or(PrepError::ArithmeticOverflow)?;
    let row_ids = aligned(
        HEADER
            .checked_add(preliminary.len())
            .and_then(|n| n.checked_add(reserve))
            .ok_or(PrepError::ArithmeticOverflow)?,
    )?;

    let mut bytes = vec![0u8; row_ids];
    for row in 0..table.rows.len() {
        bytes.extend_from_slice(&(row as u64).to_le_bytes());
    }
    for (index, column) in metadata.columns.iter_mut().enumerate() {
        bytes.resize(aligned(bytes.len())?, 0);
        column.values = bytes.len() as u64;
        for row in &table.rows {
            match column.kind {
                PrepColumnKind::Numeric => {
                    let value = match row[index].as_deref() {
                        None => 0.0,
                        Some(text) => parse_finite_numeric(text).ok_or(PrepError::InvalidJmat {
                            reason: "numeric column cell is not finite",
                        })?,
                    };
                    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                }
                PrepColumnKind::Categorical => {
                    let code = match &row[index] {
                        None => 0u32,
                        Some(value) => {
                            u32::try_from(column.dictionary.binary_search(value).map_err(|_| {
                                PrepError::InvalidJmat {
                                    reason: "categorical value missing from dictionary",
                                }
                            })?)
                            .map_err(|_| PrepError::ArithmeticOverflow)?
                        }
                    };
                    bytes.extend_from_slice(&code.to_le_bytes());
                }
            }
        }
        bytes.resize(aligned(bytes.len())?, 0);
        column.present = bytes.len() as u64;
        bytes.extend(table.rows.iter().map(|r| u8::from(r[index].is_some())));
    }

    let metadata_bytes = serde_json::to_vec(&metadata).map_err(|_| PrepError::InvalidJmat {
        reason: "failed to serialize finalized column metadata",
    })?;
    if HEADER
        .checked_add(metadata_bytes.len())
        .ok_or(PrepError::ArithmeticOverflow)?
        > row_ids
    {
        return Err(PrepError::InvalidJmat {
            reason: "metadata exceeded reserved header region",
        });
    }
    let total = bytes
        .len()
        .checked_add(CHECKSUM)
        .ok_or(PrepError::ArithmeticOverflow)?;
    bytes[..8].copy_from_slice(MAGIC);
    for (offset, value) in [
        (8, 1usize),
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

/// `PrepMatrix` is a struct for a validated, immutable JMAT v1 byte view in `jope-prep`.
pub struct PrepMatrix<B> {
    storage: B,
    metadata: JmatMetadata,
    rows: usize,
    row_ids: usize,
}

impl<B: AsRef<[u8]>> PrepMatrix<B> {
    /// Validate header magic, version, 64-byte alignments, cell finiteness, and trailing SHA-256.
    pub fn open(storage: B) -> PrepResult<Self> {
        let bytes = storage.as_ref();
        let end = bytes
            .len()
            .checked_sub(CHECKSUM)
            .ok_or(PrepError::InvalidJmat {
                reason: "buffer shorter than checksum",
            })?;
        if end < HEADER
            || bytes.get(..8) != Some(MAGIC.as_slice())
            || read_int(bytes, 8)? != 1
            || read_int(bytes, 16)? != bytes.len()
            || read_int(bytes, 56)? != 0
            || Sha256::digest(&bytes[..end]).as_slice() != &bytes[end..]
        {
            return Err(PrepError::InvalidJmat {
                reason: "header magic, version, total length, or SHA-256 mismatch",
            });
        }
        let metadata_end = HEADER
            .checked_add(read_int(bytes, 24)?)
            .ok_or(PrepError::ArithmeticOverflow)?;
        let metadata: JmatMetadata =
            serde_json::from_slice(bytes.get(HEADER..metadata_end).ok_or(
                PrepError::InvalidJmat {
                    reason: "metadata slice out of bounds",
                },
            )?)
            .map_err(|_| PrepError::InvalidJmat {
                reason: "malformed metadata JSON",
            })?;
        let rows = read_int(bytes, 32)?;
        let row_ids = read_int(bytes, 48)?;
        if metadata.columns.is_empty()
            || read_int(bytes, 40)? != metadata.columns.len()
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
            return Err(PrepError::InvalidJmat {
                reason: "invalid column metadata or unaligned row_ids offset",
            });
        }
        check_zero_padding(bytes, metadata_end, row_ids)?;
        let mut cursor = row_ids
            .checked_add(rows.checked_mul(8).ok_or(PrepError::ArithmeticOverflow)?)
            .ok_or(PrepError::ArithmeticOverflow)?;
        if cursor > end {
            return Err(PrepError::InvalidJmat {
                reason: "row_ids region exceeds payload",
            });
        }
        for row in 0..rows {
            if read_int(bytes, row_ids + row * 8)? != row {
                return Err(PrepError::InvalidJmat {
                    reason: "non-canonical row_id sequence",
                });
            }
        }
        for column in &metadata.columns {
            let start = aligned(cursor)?;
            check_zero_padding(bytes, cursor, start)?;
            if usize::try_from(column.values).map_err(|_| PrepError::ArithmeticOverflow)? != start
                || (column.kind == PrepColumnKind::Numeric && !column.dictionary.is_empty())
                || column.dictionary.windows(2).any(|w| w[0] >= w[1])
            {
                return Err(PrepError::InvalidJmat {
                    reason: "invalid column value offset or unsorted dictionary",
                });
            }
            cursor = start
                .checked_add(
                    rows.checked_mul(width(column.kind))
                        .ok_or(PrepError::ArithmeticOverflow)?,
                )
                .ok_or(PrepError::ArithmeticOverflow)?;
            let present = aligned(cursor)?;
            check_zero_padding(bytes, cursor, present)?;
            if usize::try_from(column.present).map_err(|_| PrepError::ArithmeticOverflow)?
                != present
            {
                return Err(PrepError::InvalidJmat {
                    reason: "invalid column presence bitmap offset",
                });
            }
            cursor = present
                .checked_add(rows)
                .ok_or(PrepError::ArithmeticOverflow)?;
            if cursor > end {
                return Err(PrepError::InvalidJmat {
                    reason: "presence bitmap exceeds payload",
                });
            }
            for row in 0..rows {
                let flag = bytes[present + row];
                let at = start + row * width(column.kind);
                let val = &bytes[at..at + width(column.kind)];
                if flag > 1 || (flag == 0 && val.iter().any(|&b| b != 0)) {
                    return Err(PrepError::InvalidJmat {
                        reason: "invalid presence flag or non-zero missing cell",
                    });
                }
                if flag == 1 {
                    match column.kind {
                        PrepColumnKind::Numeric => {
                            let f = f64::from_le_bytes(val.try_into().expect("8-byte slice"));
                            if !f.is_finite() {
                                return Err(PrepError::InvalidJmat {
                                    reason: "non-finite f64 in numeric column",
                                });
                            }
                        }
                        PrepColumnKind::Categorical => {
                            let code =
                                u32::from_le_bytes(val.try_into().expect("4-byte slice")) as usize;
                            if code >= column.dictionary.len() {
                                return Err(PrepError::InvalidJmat {
                                    reason: "categorical code out of dictionary bounds",
                                });
                            }
                        }
                    }
                }
            }
        }
        if cursor != end {
            return Err(PrepError::InvalidJmat {
                reason: "trailing unreferenced bytes before checksum",
            });
        }
        Ok(Self {
            storage,
            metadata,
            rows,
            row_ids,
        })
    }

    /// Number of rows in the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Validated column descriptors.
    pub fn columns(&self) -> &[PrepColumn] {
        &self.metadata.columns
    }

    /// Underlying JMAT v1 byte buffer.
    pub fn bytes(&self) -> &[u8] {
        self.storage.as_ref()
    }

    /// Canonical row identifier for `row`.
    pub fn row_id(&self, row: usize) -> Option<u64> {
        (row < self.rows).then(|| {
            read_int(self.bytes(), self.row_ids + row * 8).expect("validated row_id") as u64
        })
    }

    /// Read cell `(row, column)`.
    pub fn cell(&self, row: usize, column: usize) -> Option<PrepCell<'_>> {
        if row >= self.rows {
            return None;
        }
        let col = self.metadata.columns.get(column)?;
        let bytes = self.bytes();
        if bytes[col.present as usize + row] == 0 {
            return Some(PrepCell::Missing);
        }
        let start = col.values as usize + row * width(col.kind);
        Some(match col.kind {
            PrepColumnKind::Numeric => PrepCell::Numeric(f64::from_le_bytes(
                bytes[start..start + 8].try_into().expect("validated f64"),
            )),
            PrepColumnKind::Categorical => {
                let code =
                    u32::from_le_bytes(bytes[start..start + 4].try_into().expect("validated u32"));
                PrepCell::Categorical(&col.dictionary[code as usize])
            }
        })
    }
}

fn read_int(bytes: &[u8], offset: usize) -> PrepResult<usize> {
    let end = offset.checked_add(8).ok_or(PrepError::ArithmeticOverflow)?;
    let raw = bytes.get(offset..end).ok_or(PrepError::InvalidJmat {
        reason: "integer read out of bounds",
    })?;
    let arr: [u8; 8] = raw.try_into().map_err(|_| PrepError::InvalidJmat {
        reason: "short integer slice",
    })?;
    usize::try_from(u64::from_le_bytes(arr)).map_err(|_| PrepError::ArithmeticOverflow)
}

fn check_zero_padding(bytes: &[u8], from: usize, to: usize) -> PrepResult<()> {
    let slice = bytes.get(from..to).ok_or(PrepError::InvalidJmat {
        reason: "padding range out of bounds",
    })?;
    if slice.iter().any(|&b| b != 0) {
        return Err(PrepError::InvalidJmat {
            reason: "alignment padding contains non-zero byte",
        });
    }
    Ok(())
}
