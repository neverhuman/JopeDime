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
    let columns = unsafe { load_symbol(library, b"jope_fitted_columns\0") };
    let predict = unsafe { load_symbol(library, b"jope_fitted_predict\0") };
    let (columns, predict) = match (columns, predict) {
        (Ok(columns), Ok(predict)) => (columns, predict),
        (Err(error), _) | (_, Err(error)) => {
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
        if table.columns != unsafe { (self.columns)() } {
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

#[derive(Debug)]
struct DecodedCodec {
    model: BoundModel,
    emitter_revision: u32,
    model_library: Vec<u8>,
}

fn decode(bytes: &[u8]) -> Result<DecodedCodec> {
    if bytes.len() < HEADER_LEN + CHECKSUM_LEN || bytes.len() > MAX_CODEC_LEN {
        bail!("fitted codec length is outside the permitted range");
    }
    if &bytes[..8] != MAGIC {
        bail!("fitted codec magic mismatch");
    }
    let version = read_u16(bytes, 8)?;
    let header_len = read_u16(bytes, 10)? as usize;
    let flags = read_u32(bytes, 12)?;
    let total = usize::try_from(read_u64(bytes, 16)?).context("codec total length overflow")?;
    let metadata_len = usize::try_from(read_u64(bytes, 24)?).context("metadata length overflow")?;
    let model_len = usize::try_from(read_u64(bytes, 32)?).context("model length overflow")?;
    let source_len = usize::try_from(read_u64(bytes, 40)?).context("source length overflow")?;
    let model_library_len =
        usize::try_from(read_u64(bytes, 48)?).context("model library length overflow")?;
    if version != VERSION || header_len != HEADER_LEN || flags != 0 || total != bytes.len() {
        bail!("fitted codec header is unsupported or inconsistent");
    }
    let payload_len = metadata_len
        .checked_add(model_len)
        .and_then(|value| value.checked_add(source_len))
        .and_then(|value| value.checked_add(model_library_len))
        .context("fitted codec section length overflow")?;
    if HEADER_LEN
        .checked_add(payload_len)
        .and_then(|value| value.checked_add(CHECKSUM_LEN))
        != Some(bytes.len())
    {
        bail!("fitted codec sections do not match the declared total length");
    }
    let checksum_at = bytes.len() - CHECKSUM_LEN;
    if Sha256::digest(&bytes[..checksum_at]).as_slice() != &bytes[checksum_at..] {
        bail!("fitted codec checksum mismatch");
    }
    let metadata_at = HEADER_LEN;
    let model_at = metadata_at + metadata_len;
    let source_at = model_at + model_len;
    let library_at = source_at + source_len;
    let metadata: CodecMetadata = serde_json::from_slice(&bytes[metadata_at..model_at])?;
    if metadata.schema != "jope-dime.fitted-model-codec.v2"
        || !(1..=FITTED_CPP_REVISION).contains(&metadata.emitter_revision)
        || metadata.runtime_flags_sha256 != text_hash(&compile_flags().join(" "))
        || metadata.runtime_source_sha256 != RUNTIME_SOURCE_SHA256
        || metadata.runtime_library_sha256 != RUNTIME_LIBRARY_SHA256
        || metadata.compiler.is_empty()
    {
        bail!("fitted codec metadata or runtime identity mismatch");
    }
    let model: BoundModel = serde_json::from_slice(&bytes[model_at..source_at])?;
    model.validate_binding()?;
    if model.artifact_schema != metadata.artifact_schema
        || model.task_sha256 != metadata.task_sha256
        || model.schema_sha256 != metadata.schema_sha256
        || model.vocabulary_sha256 != metadata.vocabulary_sha256
        || model.genome_sha256 != metadata.genome_sha256
        || model.schema.row_semantics != metadata.row_semantics
    {
        bail!("fitted codec metadata does not bind the serialized model");
    }
    let source = bytes[source_at..library_at].to_vec();
    if text_hash(std::str::from_utf8(&source)?) != metadata.source_sha256 {
        bail!("fitted codec source hash mismatch");
    }
    let model_library = bytes[library_at..checksum_at].to_vec();
    if sha256_hex(&model_library) != metadata.model_library_sha256 {
        bail!("fitted codec model library hash mismatch");
    }
    // The source must correspond to the admitted model, not merely carry a
    // self-consistent hash. Only the current revision's source can be compared
    // with the emitter's; older revisions keep their self-consistent hash, so
    // stored containers still decode, and `load` re-emits them.
    if metadata.emitter_revision == FITTED_CPP_REVISION
        && emit_fitted_cpp17(&model)?.source_sha256 != metadata.source_sha256
    {
        bail!("fitted codec source does not match the current emitter");
    }
    Ok(DecodedCodec {
        model,
        emitter_revision: metadata.emitter_revision,
        model_library,
    })
}

fn compile_model_library(source: &[u8]) -> Result<(Vec<u8>, String)> {
    let compiler = crate::cxx::version().context("query fitted C++ compiler version")?;
    if !compiler.contains("13.3") {
        bail!("fitted model compilation requires g++ 13.3, found {compiler:?}");
    }
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "jope-fitted-model-compile-{}-{sequence}",
        std::process::id()
    ));
    fs::create_dir(&directory)
        .with_context(|| format!("create model compilation directory {}", directory.display()))?;
    let library = directory.join("model.so");
    let mut child = Command::new(crate::cxx::compiler())
        .args(compile_flags())
        .args([
            "-shared",
            "-fPIC",
            "-Wl,--build-id=none",
            "-x",
            "c++",
            "-",
            "-o",
        ])
        .arg(&library)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("launch g++ for fitted model image")?;
    child
        .stdin
        .take()
        .context("g++ stdin unavailable")?
        .write_all(source)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        let _ = fs::remove_dir_all(&directory);
        bail!(
            "fitted C++ model compilation failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let bytes = fs::read(&library)?;
    fs::remove_dir_all(&directory)?;
    if bytes.is_empty() {
        bail!("fitted C++ compiler produced an empty model image");
    }
    Ok((bytes, compiler))
}

fn parse_numeric(raw: &str) -> Result<Option<f64>> {
    let text = raw.trim();
    if text.is_empty()
        || text.eq_ignore_ascii_case("na")
        || text.eq_ignore_ascii_case("nan")
        || text.eq_ignore_ascii_case("null")
    {
        return Ok(None);
    }
    let value = text
        .parse::<f64>()
        .with_context(|| format!("invalid numeric value '{raw}'"))?;
    Ok(value.is_finite().then_some(value))
}

fn category_is_missing(raw: &str) -> bool {
    let text = raw.trim();
    text.is_empty() || text.eq_ignore_ascii_case("null") || text.eq_ignore_ascii_case("na")
}

fn put_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn put_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn read_u16(bytes: &[u8], at: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(at..at + 2)
            .context("truncated fitted codec u16")?
            .try_into()?,
    ))
}

fn read_u32(bytes: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(at..at + 4)
            .context("truncated fitted codec u32")?
            .try_into()?,
    ))
}

fn read_u64(bytes: &[u8], at: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes
            .get(at..at + 8)
            .context("truncated fitted codec u64")?
            .try_into()?,
    ))
}

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
mod tests {
    use super::*;
    use crate::advanced_v3::{fit_table, ColumnSpec, DataSchema, RowSemantics};
    use crate::{GenomeIr, Objective, TaskSpec};

    #[test]
    fn embedded_runtime_loads_with_good_pin_and_rejects_mismatch() {
        let image_bytes = b"pinned fitted model image";
        let expected = sha256_hex(image_bytes);
        let image = sealed_image(image_bytes, &expected).unwrap();
        assert_eq!(
            fs::read(format!("/proc/self/fd/{}", image.as_raw_fd())).unwrap(),
            image_bytes
        );
        let error = sealed_image(b"different model image", &expected).unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"));
        assert_eq!(
            sha256_hex(RUNTIME_ORACLE_SOURCE.as_bytes()),
            RUNTIME_SOURCE_SHA256
        );
    }

    fn fixture() -> (BoundModel, RawTable) {
        let schema = DataSchema {
            columns: vec![ColumnSpec::numeric("x"), ColumnSpec::numeric("z")],
            row_semantics: RowSemantics::Iid,
        };
        let train = RawTable::from_csv("x,z\n0,1\n1,2\n2,3\n3,4\n4,5\n").unwrap();
        let test = RawTable::from_csv("x,z\n0.5,1.5\n,2.5\n").unwrap();
        let genome = GenomeIr::parse("(pipe X (input numeric=median) (ridge))").unwrap();
        let task = TaskSpec::regression(Objective::Rmse).unwrap();
        let model = fit_table(&genome, &task, &schema, &train, &[0.0, 0.2, 0.4, 0.6, 0.8]).unwrap();
        (model, test)
    }

    #[test]
    fn codec_rejects_every_truncation_and_corruption() {
        let (model, _) = fixture();
        let codec = encode_fitted_cpp17(&model).unwrap();
        assert!(decode(&codec.bytes).is_ok());
        for length in 0..codec.bytes.len() {
            assert!(decode(&codec.bytes[..length]).is_err(), "length {length}");
        }
        for index in [0, 8, 16, HEADER_LEN, codec.bytes.len() - 1] {
            let mut corrupt = codec.bytes.clone();
            corrupt[index] ^= 0x80;
            assert!(decode(&corrupt).is_err(), "byte {index}");
        }
    }

    #[test]
    fn historical_unsafe_expression_library_is_regenerated_before_loading() {
        let genome = GenomeIr::parse("(sym raw \"c9+1\")").unwrap();
        let model =
            crate::fit_legacy_numeric_bound(&genome, &[vec![1., 2.], vec![2., 3.]], &[0., 1.])
                .unwrap();
        let mut artifact = emit_fitted_cpp17(&model).unwrap();
        let GenomeIr::Symbolic { expr, .. } = genome else {
            unreachable!()
        };
        artifact.source = artifact
            .source
            .replace(&expr.to_cpp17(false).unwrap(), "(row[9]+1.0)");
        artifact.source_sha256 = text_hash(&artifact.source);
        let encoded = encode_artifact(&model, &artifact).unwrap();
        let old_metadata_len = read_u64(&encoded.bytes, 24).unwrap() as usize;
        let old_payload_at = HEADER_LEN + old_metadata_len;
        let mut metadata: serde_json::Value =
            serde_json::from_slice(&encoded.bytes[HEADER_LEN..old_payload_at]).unwrap();
        let old_library_hash = metadata["model_library_sha256"]
            .as_str()
            .unwrap()
            .to_string();
        metadata.as_object_mut().unwrap().remove("emitter_revision");
        let metadata = serde_json::to_vec(&metadata).unwrap();
        let mut historical = encoded.bytes[..HEADER_LEN].to_vec();
        historical.extend(&metadata);
        historical.extend(&encoded.bytes[old_payload_at..encoded.bytes.len() - CHECKSUM_LEN]);
        let total = (historical.len() + CHECKSUM_LEN) as u64;
        historical[16..24].copy_from_slice(&total.to_le_bytes());
        historical[24..32].copy_from_slice(&(metadata.len() as u64).to_le_bytes());
        let checksum = Sha256::digest(&historical);
        historical.extend(checksum);
        assert_eq!(decode(&historical).unwrap().emitter_revision, 1);
        let original = historical.clone();
        let scratch = tempfile::Builder::new()
            .prefix("kernel-codec-upgrade-")
            .tempdir()
            .unwrap();
        let runtime = FittedCppRuntime::new(scratch.path()).unwrap();
        let loaded = runtime.load(&historical).unwrap();
        let upgraded = encode_fitted_cpp17(&decode(&historical).unwrap().model).unwrap();
        let upgraded_library = decode(&upgraded.bytes).unwrap().model_library;
        assert_ne!(sha256_hex(&upgraded_library), old_library_hash);
        assert_eq!(
            loaded
                .predict(&EncodedTable::from_numeric_rows(&[vec![1., 2.]]).unwrap())
                .unwrap(),
            vec![0.0]
        );
        assert_eq!(historical, original);
        drop(loaded);
        drop(runtime);
        scratch.close().unwrap();
    }

    #[test]
    fn safe_runtime_matches_bound_predictions() {
        let (model, test) = fixture();
        let codec = encode_fitted_cpp17(&model).unwrap();
        let repeated = encode_fitted_cpp17(&model).unwrap();
        assert_eq!(codec.sha256, repeated.sha256);
        assert_eq!(codec.bytes, repeated.bytes);
        let decoded = decode(&codec.bytes).unwrap();
        assert_eq!(decoded.emitter_revision, 3);
        assert_eq!(decoded.model.genome_sha256, model.genome_sha256);
        let encoded = EncodedTable::from_bound_model(&model, &test).unwrap();
        let scratch = tempfile::Builder::new()
            .prefix("fitted-cpp-runtime-")
            .tempdir()
            .unwrap();
        let runtime = FittedCppRuntime::new(scratch.path()).unwrap();
        let loaded = runtime.load(&codec.bytes).unwrap();
        let actual = loaded.predict(&encoded).unwrap();
        let expected = model.predict_table(&test).unwrap();
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        drop(loaded);
    }

    #[test]
    fn runtime_rejects_malformed_shapes_without_entering_model_code() {
        let (model, test) = fixture();
        let codec = encode_fitted_cpp17(&model).unwrap();
        let scratch = tempfile::Builder::new()
            .prefix("fitted-cpp-runtime-malformed-")
            .tempdir()
            .unwrap();
        let runtime = FittedCppRuntime::new(scratch.path()).unwrap();
        let loaded = runtime.load(&codec.bytes).unwrap();
        let mut wrong_columns = EncodedTable::from_bound_model(&model, &test).unwrap();
        wrong_columns.columns += 1;
        assert!(loaded.predict(&wrong_columns).is_err());
        let overflow = EncodedTable {
            values: Vec::new(),
            missing: Vec::new(),
            rows: usize::MAX,
            columns: 2,
        };
        assert!(loaded.predict(&overflow).is_err());
        let mut corrupt = codec.bytes.clone();
        corrupt[0] ^= 1;
        assert!(runtime.load(&corrupt).is_err());
        drop(loaded);
    }

    #[test]
    fn runtime_c_abi_is_sanitizer_clean_and_fail_closed() {
        let scratch = tempfile::Builder::new()
            .prefix("fitted-cpp-runtime-sanitizer-")
            .tempdir()
            .unwrap();
        let root = scratch.path();
        let runtime_source = root.join("runtime.cpp");
        let driver_source = root.join("driver.cpp");
        let binary = root.join("runtime-sanitizer");
        fs::write(&runtime_source, RUNTIME_ORACLE_SOURCE).unwrap();
        fs::write(
            &driver_source,
            r#"
#include <cstddef>
#include <cstdint>
struct Error { std::int32_t code; char stage[32]; char message[256]; };
extern "C" int jope_runtime_model_load(const std::uint8_t*,std::size_t,const char*,void**,Error*) noexcept;
extern "C" int jope_runtime_model_predict(void*,const double*,std::size_t,const std::uint8_t*,std::size_t,std::size_t,std::size_t,double*,std::size_t,Error*) noexcept;
extern "C" void jope_runtime_model_free(void*) noexcept;
int main() {
  Error error{}; void* model=nullptr; std::uint8_t short_codec[64]{};
  if (jope_runtime_model_load(nullptr,0,"/tmp/unused",&model,&error)==0) return 1;
  if (jope_runtime_model_load(short_codec,sizeof(short_codec),"/tmp/unused",&model,&error)==0) return 2;
  short_codec[0]='J'; short_codec[1]='D'; short_codec[2]='F'; short_codec[3]='C';
  short_codec[4]='P'; short_codec[5]='P'; short_codec[6]='3';
  short_codec[8]=2; short_codec[10]=56; short_codec[16]=64;
  for (int i=24;i<56;i++) short_codec[i]=0xff;
  if (jope_runtime_model_load(short_codec,sizeof(short_codec),"/tmp/unused",&model,&error)==0) return 3;
  if (jope_runtime_model_predict(nullptr,nullptr,0,nullptr,0,0,0,nullptr,0,&error)==0) return 4;
  jope_runtime_model_free(nullptr);
  return 0;
}
"#,
        )
        .unwrap();
        let compiled = crate::cxx::command(None)
            .args([
                "-O1",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-fsanitize=address,undefined",
                "-fno-sanitize-recover=undefined",
            ])
            .arg(&runtime_source)
            .arg(&driver_source)
            .args(["-ldl", "-o"])
            .arg(&binary)
            .output()
            .unwrap();
        assert!(
            compiled.status.success(),
            "sanitized runtime compilation failed: {}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let run = Command::new(&binary).output().unwrap();
        assert!(
            run.status.success(),
            "sanitized runtime rejected fail-closed probe: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        scratch.close().unwrap();
    }

    /// The source that emitter revision 2 wrote for `fixture()`'s model, written by
    /// `emit_fitted_cpp17` at `8676953c1`, the base of kernel2 PR 07. It is kept so that a stored
    /// revision-2 container can still be built, and loaded, once the emitter moves on.
    const REVISION_TWO_SOURCE: &str =
        include_str!("../../tests/fixtures/fitted_cpp/revision2-ridge.cpp");

    /// `REVISION_TWO_SOURCE` and the current source of the same model differ only in the lines
    /// that revision 3 changed: the marker, the width error and its index check, `piece`,
    /// `row_mean`, `row_std` and the catch of status 8. Every other line, the model's own code
    /// included, is the same, so the fixture stays revision 2's source for this very model.
    #[test]
    fn revision_two_source_differs_from_the_current_one_only_in_revision_three_lines() {
        /// The lines of `left` that `right` lacks, counting repeats, in `left`'s order.
        fn only_in<'a>(left: &'a str, right: &str) -> Vec<&'a str> {
            let mut counts = std::collections::BTreeMap::new();
            for line in right.lines() {
                *counts.entry(line).or_insert(0usize) += 1;
            }
            left.lines()
                .filter(|line| match counts.get_mut(line) {
                    Some(count) if *count > 0 => {
                        *count -= 1;
                        false
                    }
                    _ => true,
                })
                .collect()
        }
        fn starts_with_each(lines: &[&str], prefixes: &[&str]) -> bool {
            lines.len() == prefixes.len()
                && lines
                    .iter()
                    .zip(prefixes)
                    .all(|(line, prefix)| line.starts_with(prefix))
        }
        let (model, _) = fixture();
        let current = emit_fitted_cpp17(&model).unwrap().source;
        let removed = only_in(REVISION_TWO_SOURCE, &current);
        assert!(
            starts_with_each(
                &removed,
                &[
                    "// jope-dime fitted-cpp-typed-table-v1 emitter-revision=2",
                    "[[maybe_unused]] static double piece(",
                    "[[maybe_unused]] static double row_mean(",
                    "[[maybe_unused]] static double row_std(",
                    "  } catch(...) { return 6; }",
                ]
            ),
            "{removed:#?}"
        );
        let added = only_in(&current, REVISION_TWO_SOURCE);
        assert!(
            starts_with_each(
                &added,
                &[
                    "// jope-dime fitted-cpp-typed-table-v1 emitter-revision=3",
                    "struct FittedWidthError {};",
                    "[[maybe_unused]] static std::size_t fitted_index(",
                    "[[maybe_unused]] static double piece(",
                    "[[maybe_unused]] static double row_mean(",
                    "[[maybe_unused]] static double row_std(",
                    "  } catch(const FittedWidthError&) { return 8; } catch(...) { return 6; }",
                ]
            ),
            "{added:#?}"
        );
    }

    /// `bytes` with its metadata section replaced by `edit` of it, and the header's lengths and
    /// the checksum recomputed.
    fn with_metadata(bytes: &[u8], edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
        let payload_at = HEADER_LEN + read_u64(bytes, 24).unwrap() as usize;
        let mut metadata: serde_json::Value =
            serde_json::from_slice(&bytes[HEADER_LEN..payload_at]).unwrap();
        edit(&mut metadata);
        let metadata = serde_json::to_vec(&metadata).unwrap();
        let mut rebuilt = bytes[..HEADER_LEN].to_vec();
        rebuilt.extend(&metadata);
        rebuilt.extend(&bytes[payload_at..bytes.len() - CHECKSUM_LEN]);
        let total = (rebuilt.len() + CHECKSUM_LEN) as u64;
        rebuilt[16..24].copy_from_slice(&total.to_le_bytes());
        rebuilt[24..32].copy_from_slice(&(metadata.len() as u64).to_le_bytes());
        let checksum = Sha256::digest(&rebuilt);
        rebuilt.extend(checksum);
        rebuilt
    }

    /// The metadata section of a container.
    fn metadata_of(bytes: &[u8]) -> serde_json::Value {
        let payload_at = HEADER_LEN + read_u64(bytes, 24).unwrap() as usize;
        serde_json::from_slice(&bytes[HEADER_LEN..payload_at]).unwrap()
    }

    #[test]
    fn stored_revision_two_container_validates_and_load_regenerates_it() {
        let (model, test) = fixture();
        let encoded = EncodedTable::from_bound_model(&model, &test).unwrap();
        let bits = |values: &[f64]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };

        // A container as revision 2 stored it: that revision's source, the library compiled from
        // it, and metadata that records revision 2.
        let revision_two = FittedCppArtifact {
            source: REVISION_TWO_SOURCE.to_string(),
            source_sha256: text_hash(REVISION_TWO_SOURCE),
            codec: String::new(),
            codec_sha256: String::new(),
        };
        let stored = with_metadata(
            &encode_artifact(&model, &revision_two).unwrap().bytes,
            |metadata| metadata["emitter_revision"] = serde_json::json!(2),
        );
        let stored_library = metadata_of(&stored)["model_library_sha256"]
            .as_str()
            .unwrap()
            .to_string();

        // It still decodes and validates: revision 2's source is not compared with the emitter's.
        validate_fitted_cpp_codec(&stored).unwrap();
        let decoded = decode(&stored).unwrap();
        assert_eq!(decoded.emitter_revision, 2);
        assert_eq!(
            serde_json::to_vec(&decoded.model).unwrap(),
            serde_json::to_vec(&model).unwrap()
        );

        // Its own library, loaded as stored (without `load`'s regeneration), gives revision 2's
        // predictions.
        let scratch = tempfile::Builder::new()
            .prefix("fitted-cpp-revision-two-")
            .tempdir()
            .unwrap();
        let runtime = FittedCppRuntime::new(scratch.path()).unwrap();
        let as_stored = load_model_library(&decoded.model_library).unwrap();
        assert_eq!(sha256_hex(&decoded.model_library), stored_library);
        let revision_two_predictions = as_stored.predict(&encoded).unwrap();
        drop(as_stored);

        // `load` regenerates it from the embedded model, without refitting: the library it loads
        // is the one revision 3 compiles for the same model, not the stored one, and the
        // predictions are revision 2's, bit for bit, and the runtime's.
        let loaded = runtime.load(&stored).unwrap();
        let library = sha256_hex(
            &decode(&encode_fitted_cpp17(&model).unwrap().bytes)
                .unwrap()
                .model_library,
        );
        assert_ne!(library, stored_library);
        let current = encode_fitted_cpp17(&model).unwrap();
        assert_eq!(
            library,
            metadata_of(&current.bytes)["model_library_sha256"]
                .as_str()
                .unwrap()
        );
        assert_eq!(
            metadata_of(&current.bytes)["emitter_revision"],
            serde_json::json!(3)
        );
        let predictions = loaded.predict(&encoded).unwrap();
        assert_eq!(bits(&predictions), bits(&revision_two_predictions));
        assert_eq!(
            bits(&predictions),
            bits(&model.predict_table(&test).unwrap())
        );
        drop(loaded);

        // A revision-3 container is held to the current emitter's source, and a later revision
        // is refused.
        let relabelled = with_metadata(&stored, |metadata| {
            metadata["emitter_revision"] = serde_json::json!(3)
        });
        let error = decode(&relabelled).unwrap_err().to_string();
        assert!(
            error.contains("does not match the current emitter"),
            "{error}"
        );
        let later = with_metadata(&stored, |metadata| {
            metadata["emitter_revision"] = serde_json::json!(4)
        });
        let error = decode(&later).unwrap_err().to_string();
        assert!(
            error.contains("metadata or runtime identity mismatch"),
            "{error}"
        );
        drop(runtime);
        scratch.close().unwrap();
    }
}
