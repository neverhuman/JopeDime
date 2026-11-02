//! Bounded CSV registration using the runtime parser and SVC-01's store.
use super::jmat;
use crate::store::{BlobRef, Record, RecordKey, Store, StoreError, Write};
use jope_api::{
    error::{ErrorCode, Problem},
    ids::DatasetId,
};
use jope_runtime::{limits::Limits, table};
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::{collections::BTreeSet, fs::OpenOptions, io::Read, path::Path};

#[derive(Clone, Debug)]
pub struct RegistrationLimits {
    pub max_dataset_bytes: u64,
    pub max_rows: usize,
    pub max_columns: usize,
    pub max_cell_bytes: usize,
    pub max_categories_per_column: usize,
}
impl Default for RegistrationLimits {
    fn default() -> Self {
        Self {
            max_dataset_bytes: 4 << 30,
            max_rows: 2_000_000,
            max_columns: 10_000,
            max_cell_bytes: 1 << 20,
            max_categories_per_column: 65_536,
        }
    }
}

#[derive(Debug)]
pub enum RegistrationError {
    Invalid(Box<Problem>),
    Admission,
    Store(StoreError),
}
impl std::fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(p) => write!(f, "{}: {}", p.code, p.detail),
            Self::Admission => f.write_str("dataset registration refused by admission guard"),
            Self::Store(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for RegistrationError {}
impl From<StoreError> for RegistrationError {
    fn from(e: StoreError) -> Self {
        Self::Store(e)
    }
}
type Result<T> = std::result::Result<T, RegistrationError>;
fn invalid(field: &str) -> RegistrationError {
    RegistrationError::Invalid(Box::new(Problem {
        r#type: "about:blank".into(),
        title: "Dataset invalid".into(),
        status: 422,
        detail: format!("Dataset violates {field}."),
        instance: None,
        code: ErrorCode::DatasetInvalid,
        retriable: false,
        remedy: format!("Correct the CSV or explicitly adjust {field} before registering again."),
        field: Some(field.into()),
    }))
}

/// Internal store metadata, not a replacement public API request/response DTO.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredDataset {
    pub id: DatasetId,
    pub format: String,
    pub csv: BlobRef,
    pub matrix: BlobRef,
    pub rows: u64,
    pub columns: u64,
}

/// A validated dataset whose immutable blobs exist but whose metadata has not
/// yet been committed. The caller chooses the transaction that owns metadata.
#[derive(Clone, Debug)]
pub struct PreparedRegistration {
    pub dataset: RegisteredDataset,
    pub profile: crate::dataset::profile::DatasetProfile,
    record: Record,
}

impl PreparedRegistration {
    /// Commit the dataset row through the transaction supplied by the caller.
    pub fn commit(&self, write: &mut Write<'_>) -> crate::store::Result<()> {
        let key = RecordKey::Dataset(self.dataset.id.clone());
        if let Some(existing) = write.get(&key)? {
            if existing != self.record {
                return Err(StoreError::Integrity(
                    "dataset ID already binds different content",
                ));
            }
        } else {
            write.put(&key, &self.record)?;
        }
        Ok(())
    }
}

struct Capture<R> {
    reader: R,
    bytes: Vec<u8>,
    limit: u64,
    exceeded: bool,
}
impl<R: Read> Read for Capture<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let remaining = self
            .limit
            .saturating_sub(self.bytes.len() as u64)
            .saturating_add(1);
        let take = buffer
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));
        let count = self.reader.read(&mut buffer[..take])?;
        if (self.bytes.len() as u64).saturating_add(count as u64) > self.limit {
            self.exceeded = true;
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "max_dataset_bytes",
            ));
        }
        self.bytes.extend_from_slice(&buffer[..count]);
        Ok(count)
    }
}

/// Register an upload stream. Reads at most the byte limit plus one refusal byte.
/// The caller supplies effective config limits and serializes engine store ownership.
pub fn register(
    store: &Store,
    reader: impl Read,
    limits: &RegistrationLimits,
) -> Result<RegisteredDataset> {
    let prepared = prepare(store, reader, limits)?;
    store.transaction(|write| prepared.commit(write))?;
    Ok(prepared.dataset)
}

/// Prepare a dataset for an enclosing metadata transaction. CSV and matrix
/// blobs are immutable CAS objects; only the dataset row is deferred.
pub fn prepare(
    store: &Store,
    reader: impl Read,
    limits: &RegistrationLimits,
) -> Result<PreparedRegistration> {
    prepare_with_admission(store, reader, limits, |_| true)
}

/// Prepare with a caller supplied free-space admission check before each CAS
/// publication. `additional_bytes` is the next blob's exact encoded size.
pub fn prepare_with_admission(
    store: &Store,
    reader: impl Read,
    limits: &RegistrationLimits,
    mut admit: impl FnMut(u64) -> bool,
) -> Result<PreparedRegistration> {
    let mut input = Capture {
        reader,
        bytes: Vec::new(),
        limit: limits.max_dataset_bytes,
        exceeded: false,
    };
    let read_result = std::io::copy(&mut input, &mut std::io::sink());
    if input.exceeded {
        return Err(invalid("max_dataset_bytes"));
    }
    read_result.map_err(StoreError::from)?;
    if !admit(input.bytes.len() as u64) {
        return Err(RegistrationError::Admission);
    }
    let csv = store.publish_blob(input.bytes.as_slice())?;
    // Validate framing and allocation bounds before the sole table parse. This
    // pass does not decode values or construct another table.
    framing(&input.bytes, limits)?;
    let runtime_limits = Limits {
        max_input_bytes: limits.max_dataset_bytes,
        max_rows: limits.max_rows,
        max_columns: limits.max_columns,
        ..Limits::default()
    };
    let parsed = table::parse_csv(&input.bytes, &runtime_limits).map_err(|_| invalid("csv"))?;
    drop(input); // The parsed table owns its cells; release the captured CSV buffer.
    if parsed.rows.is_empty() {
        return Err(invalid("csv"));
    }
    for column in 0..parsed.headers.len() {
        let numeric = parsed
            .rows
            .iter()
            .filter_map(|r| r[column].as_deref())
            .all(|v| v.trim().parse::<f64>().is_ok_and(f64::is_finite));
        if !numeric {
            let mut distinct = BTreeSet::new();
            for value in parsed.rows.iter().filter_map(|r| r[column].as_deref()) {
                distinct.insert(value);
                if distinct.len() > limits.max_categories_per_column {
                    return Err(invalid("max_categories_per_column"));
                }
            }
        }
    }
    let profile = crate::dataset::profile::profile(&parsed);
    let encoded = jmat::encode(&parsed).map_err(|_| invalid("matrix"))?;
    if !admit(encoded.len() as u64) {
        return Err(RegistrationError::Admission);
    }
    let matrix = store.publish_blob(encoded.as_slice())?;
    let digest = csv
        .sha256
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let id: DatasetId = format!("ds_{}", &digest[..24])
        .parse()
        .expect("SHA256 prefix is a dataset ID");
    let dataset = RegisteredDataset {
        id: id.clone(),
        format: "jope.jmat/v1".into(),
        csv,
        matrix,
        rows: parsed.rows.len() as u64,
        columns: parsed.headers.len() as u64,
    };
    let record = Record::metadata(
        serde_json::to_value(&dataset).map_err(StoreError::from)?,
        vec![dataset.csv.clone(), dataset.matrix.clone()],
    );
    Ok(PreparedRegistration {
        dataset,
        profile,
        record,
    })
}

/// Local-path adapter; only reads the supplied path. All writes use Store's home.
pub fn register_file(
    store: &Store,
    path: &Path,
    limits: &RegistrationLimits,
) -> Result<RegisteredDataset> {
    let prepared = prepare_file(store, path, limits)?;
    store.transaction(|write| prepared.commit(write))?;
    Ok(prepared.dataset)
}

/// Prepare a local regular file without committing its dataset metadata.
pub fn prepare_file(
    store: &Store,
    path: &Path,
    limits: &RegistrationLimits,
) -> Result<PreparedRegistration> {
    prepare_file_with_admission(store, path, limits, |_| true)
}

pub fn prepare_file_with_admission(
    store: &Store,
    path: &Path,
    limits: &RegistrationLimits,
    admit: impl FnMut(u64) -> bool,
) -> Result<PreparedRegistration> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NONBLOCK);

    // A blocking read-open on a FIFO without a writer never returns. Open
    // nonblocking on Unix, then inspect the opened handle to avoid a path
    // check/open race. Other platforms retain their existing open behavior.
    let file = options.open(path).map_err(StoreError::from)?;
    let metadata = file.metadata().map_err(StoreError::from)?;
    if !metadata.file_type().is_file() {
        return Err(invalid("csv"));
    }
    if metadata.len() > limits.max_dataset_bytes {
        return Err(invalid("max_dataset_bytes"));
    }
    prepare_with_admission(store, file, limits, admit)
}

/// Strict UTF8/quote grammar and bounds without allocating decoded records.
fn framing(bytes: &[u8], limits: &RegistrationLimits) -> Result<()> {
    std::str::from_utf8(bytes).map_err(|_| invalid("csv"))?;
    let bytes = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);
    // 0 field start, 1 unquoted, 2 quoted, 3 closing quote.
    let (mut state, mut length, mut columns, mut records, mut nonblank) =
        (0, 0_usize, 1_usize, 0_usize, false);
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        let mut end_record = false;
        if state == 2 {
            if byte == b'"' {
                state = 3;
            } else {
                length += 1;
            }
        } else if state == 3 && byte == b'"' {
            length += 1;
            state = 2;
        } else if byte == b',' {
            columns = columns
                .checked_add(1)
                .ok_or_else(|| invalid("max_columns"))?;
            state = 0;
            length = 0;
            nonblank = true;
        } else if byte == b'\n' || byte == b'\r' {
            end_record = true;
            if byte == b'\r' && bytes.get(index + 1) == Some(&b'\n') {
                index += 1;
            }
        } else if byte == b'"' && state == 0 {
            state = 2;
            nonblank = true;
        } else if state == 3 || byte == b'"' {
            return Err(invalid("csv"));
        } else {
            state = 1;
            length += 1;
            nonblank = true;
        }
        if length > limits.max_cell_bytes {
            return Err(invalid("max_cell_bytes"));
        }
        if columns > limits.max_columns {
            return Err(invalid("max_columns"));
        }
        if end_record {
            if nonblank {
                records += 1;
                if records.saturating_sub(1) > limits.max_rows {
                    return Err(invalid("max_rows"));
                }
            }
            state = 0;
            length = 0;
            columns = 1;
            nonblank = false;
        }
        index += 1;
    }
    if state == 2 {
        return Err(invalid("csv"));
    }
    if nonblank {
        records += 1;
    }
    if records.saturating_sub(1) > limits.max_rows {
        return Err(invalid("max_rows"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
