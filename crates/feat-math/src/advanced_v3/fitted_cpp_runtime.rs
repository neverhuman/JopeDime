//! Length-checked fitted-C++ codec and safe dynamic execution boundary.

use super::csv::RawTable;
use super::fitted_cpp::{emit_fitted_cpp17, FittedCppArtifact, FITTED_CPP_REVISION};
use super::fitted_runtime_image::sealed_image;
use super::input::FittedColumn;
use super::pipeline::BoundModel;
use super::schema::ColumnKind;
use super::types::text_hash;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::ffi::{c_char, c_int, c_void, CString};
use std::fs;
use std::io::Write;
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: &[u8; 8] = b"JDFCPP3\0";
const VERSION: u16 = 2;
const HEADER_LEN: usize = 8 + 2 + 2 + 4 + 8 + 8 + 8 + 8 + 8;
const CHECKSUM_LEN: usize = 32;
const MAX_CODEC_LEN: usize = 512 * 1024 * 1024;
/// Compile flags of every fitted model image (`crate::cxx::fitted_flags`), resolved once per process
/// because the joined string is part of the codec identity (`runtime_flags_sha256`).
fn compile_flags() -> &'static [String] {
    static FLAGS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    FLAGS.get_or_init(|| crate::cxx::fitted_flags().unwrap_or_else(|error| panic!("{error}")))
}
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
// These identities are pinned in fitted containers. The source remains byte-for-byte
// under tests/fixtures as a sanitizer-tested parity oracle; neither it nor its shared
// object is included in the product runtime.
const RUNTIME_SOURCE_SHA256: &str =
    "efc1c702784aefb129450b53909846731a6cfa8911c80dec59f28007430f81d1";
const RUNTIME_LIBRARY_SHA256: &str =
    "91020c7ac0add6db78b37e90610767142da4a744d9ff767cc1a2c6ced5a94c74";
#[cfg(test)]
const RUNTIME_ORACLE_SOURCE: &str = include_str!("../../tests/fixtures/fitted_runtime_oracle.cpp");

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CodecMetadata {
    schema: String,
    #[serde(default = "legacy_emitter_revision")]
    emitter_revision: u32,
    artifact_schema: String,
    task_sha256: String,
    schema_sha256: String,
    vocabulary_sha256: String,
    genome_sha256: String,
    row_semantics: super::schema::RowSemantics,
    source_sha256: String,
    model_library_sha256: String,
    runtime_source_sha256: String,
    runtime_library_sha256: String,
    runtime_flags_sha256: String,
    compiler: String,
}

fn legacy_emitter_revision() -> u32 {
    1
}

#[derive(Clone, Debug)]
pub struct FittedModelCodec {
    pub bytes: Vec<u8>,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EncodedTable {
    pub values: Vec<f64>,
    pub missing: Vec<u8>,
    pub rows: usize,
    pub columns: usize,
}

impl EncodedTable {
    pub fn from_bound_model(model: &BoundModel, table: &RawTable) -> Result<Self> {
        table.validate_against(&model.schema)?;
        model.validate_binding()?;
        let rows = table.rows.len();
        let columns = model.schema.columns.len();
        let cells = rows
            .checked_mul(columns)
            .context("encoded table size overflow")?;
        let mut values = Vec::with_capacity(cells);
        let mut missing = Vec::with_capacity(cells);
        for row in &table.rows {
            for (index, (spec, fitted)) in model
                .schema
                .columns
                .iter()
                .zip(&model.input.columns)
                .enumerate()
            {
                let raw = &row[index];
                match (spec.kind, fitted) {
                    (ColumnKind::Numeric, FittedColumn::Numeric { .. }) => {
                        let parsed = parse_numeric(raw)?;
                        values.push(parsed.unwrap_or(0.0));
                        missing.push(u8::from(parsed.is_none()));
                    }
                    (ColumnKind::Categorical, FittedColumn::Categorical { vocabulary, .. }) => {
                        let absent = category_is_missing(raw);
                        let key = if absent { "\0MISSING" } else { raw.as_str() };
                        let encoded = vocabulary
                            .binary_search_by(|candidate| candidate.as_str().cmp(key))
                            .map_or(-1.0, |position| position as f64);
                        values.push(encoded);
                        missing.push(u8::from(absent));
                    }
                    _ => bail!("bound schema and fitted input column {index} disagree"),
                }
            }
        }
        Ok(Self {
            values,
            missing,
            rows,
            columns,
        })
    }

    pub fn from_numeric_rows(rows: &[Vec<f64>]) -> Result<Self> {
        let columns = rows.first().map_or(0, Vec::len);
        if columns == 0 || rows.iter().any(|row| row.len() != columns) {
            bail!("encoded numeric table is empty or ragged");
        }
        let mut values = Vec::with_capacity(rows.len().saturating_mul(columns));
        let mut missing = Vec::with_capacity(values.capacity());
        for value in rows.iter().flatten() {
            values.push(if value.is_finite() { *value } else { 0.0 });
            missing.push(u8::from(!value.is_finite()));
        }
        Ok(Self {
            values,
            missing,
            rows: rows.len(),
            columns,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FittedCppError {
    pub stage: String,
    pub code: i32,
    pub message: String,
}

#[derive(Clone)]
pub struct FittedCppRuntime {
    work_root: PathBuf,
}

pub struct FittedCppModel {
    #[cfg(unix)]
    library: *mut c_void,
    #[cfg(unix)]
    _image: fs::File,
    #[cfg(unix)]
    columns: ModelColumns,
    #[cfg(unix)]
    predict_fn: ModelPredict,
}

#[cfg(unix)]
type ModelColumns = unsafe extern "C" fn() -> usize;
#[cfg(unix)]
type ModelPredict = unsafe extern "C" fn(
    *const f64,
    usize,
    *const u8,
    usize,
    usize,
    usize,
    *mut f64,
    usize,
) -> c_int;

impl std::fmt::Debug for FittedCppRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FittedCppRuntime")
            .field("work_root", &self.work_root)
            .finish_non_exhaustive()
    }
}

#[cfg(unix)]
impl FittedCppRuntime {
    pub fn new(work_root: impl Into<PathBuf>) -> Result<Self> {
        let work_root = work_root.into();
        fs::create_dir_all(&work_root)
            .with_context(|| format!("create fitted C++ work root {}", work_root.display()))?;
        Ok(Self { work_root })
    }

    /// Load a trusted encoder-produced codec. Its executable model image must
    /// come from a trusted producer: codec checksums establish integrity, not
    /// authorization to execute an arbitrary native library.
    pub fn load(&self, codec: &[u8]) -> Result<FittedCppModel> {
        let mut decoded = decode(codec)?;
        // Decode historical containers, but never dlopen an artifact emitted
        // before the current revision: revision 1 predates the whole-expression
        // bounds/domain repair, and revision 2 the fitted index checks and the
        // runtime's empty-row, padding and tied-interval semantics (revision 3).
        // Regenerate from the embedded fitted model without refitting or
        // changing old bytes.
        if decoded.emitter_revision < FITTED_CPP_REVISION {
            let upgraded = encode_fitted_cpp17(&decoded.model)?;
            decoded = decode(&upgraded.bytes)?;
        }
        load_model_library(&decoded.model_library)
    }
}

#[cfg(unix)]
fn load_model_library(bytes: &[u8]) -> Result<FittedCppModel> {
    let image = sealed_image(bytes, &sha256_hex(bytes))?;
    let path = CString::new(format!("/proc/self/fd/{}", image.as_raw_fd()))?;
    // SAFETY: path names a fully sealed image and the descriptor remains live
    // in the returned model for at least as long as its function pointers.
    let library = unsafe { dlopen(path.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
    if library.is_null() {
        bail!("load fitted model image: {}", dl_error());
    }
    // SAFETY: `library` is a live handle for the sealed image and both fixed
    // names are NUL-terminated; their returned addresses remain live until close.
    let columns = unsafe { load_symbol(library, b"jope_fitted_columns\0") };
    // SAFETY: same live sealed image and fixed NUL-terminated export contract.
    let predict = unsafe { load_symbol(library, b"jope_fitted_predict\0") };
    let (columns, predict) = match (columns, predict) {
        (Ok(columns), Ok(predict)) => (columns, predict),
        (Err(error), _) | (_, Err(error)) => {
            // SAFETY: this is the unique open reference and no function pointer escaped.
            unsafe { dlclose(library) };
            return Err(error);
        }
    };
    Ok(FittedCppModel {
        library,
        _image: image,
        // SAFETY: both exported symbols have the ABI defined by emit_abi;
        // the sealed library handle is retained for this model's lifetime.
        columns: unsafe { std::mem::transmute::<*mut c_void, ModelColumns>(columns) },
        predict_fn: unsafe { std::mem::transmute::<*mut c_void, ModelPredict>(predict) },
    })
}

#[cfg(unix)]
impl FittedCppModel {
    pub fn predict(&self, table: &EncodedTable) -> Result<Vec<f64>> {
        let cells = table
            .rows
            .checked_mul(table.columns)
            .context("encoded table cell count overflow")?;
        if table.values.len() != cells || table.missing.len() != cells {
            bail!("encoded table buffers do not match its declared shape");
        }
        // SAFETY: this symbol was resolved from the retained sealed model image
        // and has the `jope_fitted_columns` ABI defined by the emitter.
        let model_columns = unsafe { (self.columns)() };
        if table.columns != model_columns {
            bail!("fitted model column count mismatch");
        }
        let mut output = vec![0.0; table.rows];
        // SAFETY: the codec validated the exported model ABI; rows*columns was
        // checked above and all buffers stay borrowed for the synchronous call.
        let status = unsafe {
            (self.predict_fn)(
                table.values.as_ptr(),
                table.values.len(),
                table.missing.as_ptr(),
                table.missing.len(),
                table.rows,
                table.columns,
                output.as_mut_ptr(),
                output.len(),
            )
        };
        if status != 0 {
            bail!("fitted model prediction rejected typed table with status {status}");
        }
        Ok(output)
    }
}

#[cfg(unix)]
impl Drop for FittedCppModel {
    fn drop(&mut self) {
        // SAFETY: this is the unique successful dlopen reference; function
        // pointers cannot outlive the model and image descriptor.
        unsafe {
            dlclose(self.library);
        }
    }
}

#[cfg(not(unix))]
impl FittedCppRuntime {
    pub fn new(work_root: impl Into<PathBuf>) -> Result<Self> {
        let _ = work_root.into();
        Err(unsupported_runtime())
    }

    pub fn load(&self, _codec: &[u8]) -> Result<FittedCppModel> {
        Err(unsupported_runtime())
    }
}

#[cfg(not(unix))]
impl FittedCppModel {
    pub fn predict(&self, _table: &EncodedTable) -> Result<Vec<f64>> {
        Err(unsupported_runtime())
    }
}

#[cfg(not(unix))]
fn unsupported_runtime() -> anyhow::Error {
    std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "fitted model loading is supported only on Unix platforms",
    )
    .into()
}

pub fn encode_fitted_cpp17(model: &BoundModel) -> Result<FittedModelCodec> {
    let artifact = emit_fitted_cpp17(model)?;
    encode_artifact(model, &artifact)
}

pub fn validate_fitted_cpp_codec(bytes: &[u8]) -> Result<()> {
    decode(bytes).map(|_| ())
}

pub(super) fn encode_artifact(
    model: &BoundModel,
    artifact: &FittedCppArtifact,
) -> Result<FittedModelCodec> {
    let model_json = serde_json::to_vec(model)?;
    let (model_library, compiler) = compile_model_library(artifact.source.as_bytes())?;
    let metadata = CodecMetadata {
        schema: "jope-dime.fitted-model-codec.v2".to_string(),
        emitter_revision: FITTED_CPP_REVISION,
        artifact_schema: model.artifact_schema.clone(),
        task_sha256: model.task_sha256.clone(),
        schema_sha256: model.schema_sha256.clone(),
        vocabulary_sha256: model.vocabulary_sha256.clone(),
        genome_sha256: model.genome_sha256.clone(),
        row_semantics: model.schema.row_semantics.clone(),
        source_sha256: artifact.source_sha256.clone(),
        model_library_sha256: sha256_hex(&model_library),
        runtime_source_sha256: RUNTIME_SOURCE_SHA256.to_string(),
        runtime_library_sha256: RUNTIME_LIBRARY_SHA256.to_string(),
        runtime_flags_sha256: text_hash(&compile_flags().join(" ")),
        compiler,
    };
    let metadata_json = serde_json::to_vec(&metadata)?;
    let source = artifact.source.as_bytes();
    let total = HEADER_LEN
        .checked_add(metadata_json.len())
        .and_then(|value| value.checked_add(model_json.len()))
        .and_then(|value| value.checked_add(source.len()))
        .and_then(|value| value.checked_add(model_library.len()))
        .and_then(|value| value.checked_add(CHECKSUM_LEN))
        .context("fitted codec length overflow")?;
    if total > MAX_CODEC_LEN {
        bail!("fitted codec length {total} exceeds {MAX_CODEC_LEN}");
    }
    let mut bytes = Vec::with_capacity(total);
    bytes.extend_from_slice(MAGIC);
    put_u16(&mut bytes, VERSION);
    put_u16(&mut bytes, HEADER_LEN as u16);
    put_u32(&mut bytes, 0);
    put_u64(&mut bytes, total as u64);
    put_u64(&mut bytes, metadata_json.len() as u64);
    put_u64(&mut bytes, model_json.len() as u64);
    put_u64(&mut bytes, source.len() as u64);
    put_u64(&mut bytes, model_library.len() as u64);
    bytes.extend_from_slice(&metadata_json);
    bytes.extend_from_slice(&model_json);
    bytes.extend_from_slice(source);
    bytes.extend_from_slice(&model_library);
    let checksum = Sha256::digest(&bytes);
    bytes.extend_from_slice(&checksum);
    Ok(FittedModelCodec {
        sha256: sha256_hex(&bytes),
        bytes,
    })
}

mod codec;
pub(super) use codec::*;

#[cfg(unix)]
const RTLD_NOW: c_int = 2;
#[cfg(unix)]
const RTLD_LOCAL: c_int = 0;

#[cfg(unix)]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

#[cfg(unix)]
unsafe fn load_symbol(handle: *mut c_void, name: &[u8]) -> Result<*mut c_void> {
    // SAFETY: the caller supplies a live dlopen handle and a fixed, live,
    // NUL-terminated symbol name; the returned address is used while it stays open.
    let pointer = unsafe { dlsym(handle, name.as_ptr().cast()) };
    if pointer.is_null() {
        bail!("load fitted C++ symbol: {}", dl_error());
    }
    Ok(pointer)
}

#[cfg(unix)]
fn dl_error() -> String {
    // SAFETY: dlerror takes no arguments and returns a thread-local pointer.
    let pointer = unsafe { dlerror() };
    if pointer.is_null() {
        "dynamic loader returned no detail".to_string()
    } else {
        // SAFETY: a non-null dlerror result is a live NUL-terminated string until
        // the next loader operation on this thread; copy it before any such call.
        unsafe { std::ffi::CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(all(test, unix))]
mod tests;
