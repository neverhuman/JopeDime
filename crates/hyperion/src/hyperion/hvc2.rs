//! Hyperion Vector Cache V2 (`.hvc2`) framing and receipt contracts.
//!
//! Files are deliberately simple and deterministic: an `HVC2` preamble, a little-endian format
//! version and JSON-header length, the canonical compact JSON header, then exact row-major
//! little-endian `f32` payload bytes. The header binds the source CSV and frozen Orig V2 profile.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::orig_v2::{
    OrigV2PairSemantics, OrigV2Receipt, ORIG_V2_MANIFEST_SHA256, ORIG_V2_PROFILE_ID,
    ORIG_V2_WEIGHTS_SHA256,
};

pub const HVC2_MAGIC: [u8; 4] = *b"HVC2";
pub const HVC2_VERSION: u16 = 2;
pub const HVC2_SCHEMA: &str = "jain.hyperion-vector-cache/v2";
pub const HVC2_RECEIPT_SCHEMA: &str = "jain.hyperion-vector-cache-receipt/v2";
pub const HVC2_TRAIN_FILENAME: &str = "train.hvc2";
pub const HVC2_TEST_FILENAME: &str = "test.hvc2";
pub const HVC2_RECEIPT_FILENAME: &str = "hyperion_v2.receipt.json";

const PREAMBLE_BYTES: u64 = 4 + 2 + 8;
const MAX_HEADER_BYTES: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Hvc2Split {
    Train,
    Test,
}

impl Hvc2Split {
    pub fn filename(self) -> &'static str {
        match self {
            Self::Train => HVC2_TRAIN_FILENAME,
            Self::Test => HVC2_TEST_FILENAME,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Hvc2Disposition {
    Encoded,
    NotApplicableNoExogenousFeatures,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hvc2Metadata {
    pub dataset: String,
    pub task: String,
    pub split: Hvc2Split,
    pub source_sha256: String,
    pub source_bytes: u64,
    pub disposition: Hvc2Disposition,
}

impl Hvc2Metadata {
    pub fn encoded(
        dataset: impl Into<String>,
        task: impl Into<String>,
        split: Hvc2Split,
        source_sha256: impl Into<String>,
        source_bytes: u64,
    ) -> Self {
        Self {
            dataset: dataset.into(),
            task: task.into(),
            split,
            source_sha256: source_sha256.into(),
            source_bytes,
            disposition: Hvc2Disposition::Encoded,
        }
    }

    pub fn no_exogenous_features(
        dataset: impl Into<String>,
        task: impl Into<String>,
        split: Hvc2Split,
        source_sha256: impl Into<String>,
        source_bytes: u64,
    ) -> Self {
        Self {
            dataset: dataset.into(),
            task: task.into(),
            split,
            source_sha256: source_sha256.into(),
            source_bytes,
            disposition: Hvc2Disposition::NotApplicableNoExogenousFeatures,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Hvc2Header {
    pub schema: String,
    pub version: u16,
    pub dataset: String,
    pub task: String,
    pub split: Hvc2Split,
    pub source_sha256: String,
    pub source_bytes: u64,
    pub rows: u64,
    pub columns: u64,
    pub dtype: String,
    pub layout: String,
    pub payload_bytes: u64,
    pub payload_sha256: String,
    pub profile: String,
    pub pair_semantics: OrigV2PairSemantics,
    pub train_fitted: bool,
    pub weights_sha256: String,
    pub manifest_sha256: String,
    pub disposition: Hvc2Disposition,
}

impl Hvc2Header {
    pub fn new(
        metadata: Hvc2Metadata,
        rows: usize,
        columns: usize,
        values: &[f32],
    ) -> Result<Self> {
        let expected_values = rows
            .checked_mul(columns)
            .context("HVC2 shape overflows usize")?;
        if values.len() != expected_values {
            bail!(
                "HVC2 payload has {} values for shape {rows}x{columns}",
                values.len()
            );
        }
        if values.iter().any(|value| !value.is_finite()) {
            bail!("HVC2 payload contains a non-finite value");
        }
        let payload_bytes = u64::try_from(expected_values)
            .context("HVC2 payload length does not fit u64")?
            .checked_mul(4)
            .context("HVC2 payload byte length overflows u64")?;
        let header = Self {
            schema: HVC2_SCHEMA.into(),
            version: HVC2_VERSION,
            dataset: metadata.dataset,
            task: metadata.task,
            split: metadata.split,
            source_sha256: metadata.source_sha256,
            source_bytes: metadata.source_bytes,
            rows: u64::try_from(rows).context("HVC2 row count does not fit u64")?,
            columns: u64::try_from(columns).context("HVC2 column count does not fit u64")?,
            dtype: "f32_le".into(),
            layout: "row_major".into(),
            payload_bytes,
            payload_sha256: sha256_f32_le(values),
            profile: ORIG_V2_PROFILE_ID.into(),
            pair_semantics: OrigV2PairSemantics::TrainFitted,
            train_fitted: true,
            weights_sha256: ORIG_V2_WEIGHTS_SHA256.into(),
            manifest_sha256: ORIG_V2_MANIFEST_SHA256.into(),
            disposition: metadata.disposition,
        };
        header.validate()?;
        Ok(header)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != HVC2_SCHEMA || self.version != HVC2_VERSION {
            bail!("unsupported HVC2 header schema/version");
        }
        if self.dataset.trim().is_empty() {
            bail!("HVC2 dataset identity is empty");
        }
        if !matches!(
            self.task.as_str(),
            "binary" | "regression" | "multiclass" | "timeseries"
        ) {
            bail!("unsupported HVC2 task '{}'", self.task);
        }
        validate_sha256(&self.source_sha256, "source")?;
        if self.source_bytes == 0 || self.rows == 0 {
            bail!("HVC2 source byte count and row count must be non-zero");
        }
        if self.dtype != "f32_le" || self.layout != "row_major" {
            bail!("unsupported HVC2 dtype/layout");
        }
        let expected_payload = self
            .rows
            .checked_mul(self.columns)
            .and_then(|values| values.checked_mul(4))
            .context("HVC2 header shape overflows u64")?;
        if self.payload_bytes != expected_payload {
            bail!(
                "HVC2 payload byte count {} does not match shape {}x{}",
                self.payload_bytes,
                self.rows,
                self.columns
            );
        }
        validate_sha256(&self.payload_sha256, "payload")?;
        if self.profile != ORIG_V2_PROFILE_ID
            || self.pair_semantics != OrigV2PairSemantics::TrainFitted
            || !self.train_fitted
        {
            bail!("HVC2 requires the frozen train-fitted Orig V2 profile");
        }
        if self.weights_sha256 != ORIG_V2_WEIGHTS_SHA256
            || self.manifest_sha256 != ORIG_V2_MANIFEST_SHA256
        {
            bail!("HVC2 model authority hashes do not match the frozen profile");
        }
        match (self.columns, self.disposition) {
            (0, Hvc2Disposition::NotApplicableNoExogenousFeatures) => {}
            (0, Hvc2Disposition::Encoded) => {
                bail!("zero-column HVC2 payload must be typed not-applicable")
            }
            (_, Hvc2Disposition::Encoded) => {}
            (_, Hvc2Disposition::NotApplicableNoExogenousFeatures) => {
                bail!("not-applicable HVC2 payload must have zero columns")
            }
        }
        Ok(())
    }

    pub fn shape(&self) -> Result<(usize, usize)> {
        Ok((
            usize::try_from(self.rows).context("HVC2 rows do not fit usize")?,
            usize::try_from(self.columns).context("HVC2 columns do not fit usize")?,
        ))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hvc2Artifact {
    pub header: Hvc2Header,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Hvc2FileBinding {
    pub filename: String,
    pub sha256: String,
    pub bytes: u64,
    pub header: Hvc2Header,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Hvc2ImplementationIdentity {
    pub core_tag: String,
    pub core_commit: String,
    pub cli_tag: String,
    pub cli_commit: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Hvc2NodeIdentity {
    pub hostname: String,
    pub gpu: String,
    pub cuda: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Hvc2Timings {
    pub parse_and_hash_ms: u64,
    pub encode_ms: u64,
    pub write_ms: u64,
    pub total_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Hvc2PairReceipt {
    pub schema: String,
    pub version: u16,
    pub dataset: String,
    pub task: String,
    pub disposition: Hvc2Disposition,
    pub profile: String,
    pub pair_semantics: OrigV2PairSemantics,
    pub train_fitted: bool,
    pub train_source_sha256: String,
    pub test_source_sha256: String,
    pub train_cache: Hvc2FileBinding,
    pub test_cache: Hvc2FileBinding,
    pub orig_v2: Option<OrigV2Receipt>,
    pub implementation: Hvc2ImplementationIdentity,
    pub node: Hvc2NodeIdentity,
    pub timings: Hvc2Timings,
    pub invalid_tokens: u64,
    pub non_finite_tokens: u64,
    pub created_at_utc: String,
}

impl Hvc2PairReceipt {
    pub fn validate(&self) -> Result<()> {
        if self.schema != HVC2_RECEIPT_SCHEMA || self.version != HVC2_VERSION {
            bail!("unsupported HVC2 receipt schema/version");
        }
        if self.profile != ORIG_V2_PROFILE_ID
            || self.pair_semantics != OrigV2PairSemantics::TrainFitted
            || !self.train_fitted
        {
            bail!("HVC2 receipt does not bind the frozen train-fitted profile");
        }
        validate_sha256(&self.train_source_sha256, "train source")?;
        validate_sha256(&self.test_source_sha256, "test source")?;
        validate_binding(
            &self.train_cache,
            Hvc2Split::Train,
            &self.dataset,
            &self.task,
            &self.train_source_sha256,
            self.disposition,
        )?;
        validate_binding(
            &self.test_cache,
            Hvc2Split::Test,
            &self.dataset,
            &self.task,
            &self.test_source_sha256,
            self.disposition,
        )?;
        match (self.disposition, self.orig_v2.as_ref()) {
            (Hvc2Disposition::Encoded, Some(orig_v2)) => {
                orig_v2.validate_frozen(OrigV2PairSemantics::TrainFitted)?;
            }
            (Hvc2Disposition::Encoded, None) => {
                bail!("encoded HVC2 receipt is missing the complete Orig V2 receipt")
            }
            (Hvc2Disposition::NotApplicableNoExogenousFeatures, None) => {}
            (Hvc2Disposition::NotApplicableNoExogenousFeatures, Some(_)) => {
                bail!("not-applicable HVC2 receipt must not claim model execution")
            }
        }
        for (label, value) in [
            ("core tag", &self.implementation.core_tag),
            ("core commit", &self.implementation.core_commit),
            ("CLI tag", &self.implementation.cli_tag),
            ("CLI commit", &self.implementation.cli_commit),
            ("hostname", &self.node.hostname),
            ("GPU identity", &self.node.gpu),
            ("CUDA identity", &self.node.cuda),
            ("creation timestamp", &self.created_at_utc),
        ] {
            if value.trim().is_empty() {
                bail!("HVC2 receipt {label} is empty");
            }
        }
        if self.timings.total_ms
            < self
                .timings
                .parse_and_hash_ms
                .saturating_add(self.timings.encode_ms)
                .saturating_add(self.timings.write_ms)
        {
            bail!("HVC2 receipt total timing is smaller than its component timings");
        }
        Ok(())
    }
}

fn validate_binding(
    binding: &Hvc2FileBinding,
    split: Hvc2Split,
    dataset: &str,
    task: &str,
    source_sha256: &str,
    disposition: Hvc2Disposition,
) -> Result<()> {
    binding.header.validate()?;
    validate_sha256(&binding.sha256, "cache file")?;
    if binding.filename != split.filename()
        || binding.header.split != split
        || binding.header.dataset != dataset
        || binding.header.task != task
        || binding.header.source_sha256 != source_sha256
        || binding.header.disposition != disposition
    {
        bail!("HVC2 receipt cache binding does not match the dataset contract");
    }
    let header_bytes = canonical_header_bytes(&binding.header)?;
    let expected_bytes = PREAMBLE_BYTES
        .checked_add(u64::try_from(header_bytes.len()).context("header length does not fit u64")?)
        .and_then(|bytes| bytes.checked_add(binding.header.payload_bytes))
        .context("HVC2 bound file length overflows u64")?;
    if binding.bytes != expected_bytes {
        bail!("HVC2 receipt bound file byte count is inconsistent");
    }
    Ok(())
}

pub fn write_hvc2_atomic(
    path: &Path,
    header: &Hvc2Header,
    row_major_values: &[f32],
) -> Result<Hvc2FileBinding> {
    header.validate()?;
    let expected_filename = header.split.filename();
    if path.file_name().and_then(|value| value.to_str()) != Some(expected_filename) {
        bail!("HVC2 split must be written as {expected_filename}");
    }
    let (rows, columns) = header.shape()?;
    let expected_values = rows
        .checked_mul(columns)
        .context("HVC2 shape overflows usize")?;
    if row_major_values.len() != expected_values {
        bail!("HVC2 write payload shape mismatch");
    }
    if row_major_values.iter().any(|value| !value.is_finite()) {
        bail!("HVC2 write payload contains a non-finite value");
    }
    let observed_payload_sha256 = sha256_f32_le(row_major_values);
    if observed_payload_sha256 != header.payload_sha256 {
        bail!("HVC2 write payload hash does not match its header");
    }
    let header_bytes = canonical_header_bytes(header)?;
    let parent = path.parent().context("HVC2 output path has no parent")?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("creating HVC2 directory {}", parent.display()))?;
    if path.exists() {
        bail!(
            "refusing to overwrite existing HVC2 file {}",
            path.display()
        );
    }
    let staging = staging_path(path)?;
    let write_result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&staging)
            .with_context(|| format!("creating staged HVC2 {}", staging.display()))?;
        file.write_all(&HVC2_MAGIC)?;
        file.write_all(&HVC2_VERSION.to_le_bytes())?;
        file.write_all(
            &u64::try_from(header_bytes.len())
                .context("HVC2 header length does not fit u64")?
                .to_le_bytes(),
        )?;
        file.write_all(&header_bytes)?;
        for value in row_major_values {
            file.write_all(&value.to_le_bytes())?;
        }
        file.sync_all()?;
        drop(file);
        install_no_clobber(&staging, path)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&staging);
    write_result?;
    hvc2_file_binding(path)
}

pub fn read_hvc2(path: &Path) -> Result<Hvc2Artifact> {
    let mut file = File::open(path).with_context(|| format!("opening HVC2 {}", path.display()))?;
    let file_bytes = file.metadata()?.len();
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .context("truncated HVC2 magic")?;
    if magic != HVC2_MAGIC {
        bail!("invalid HVC2 magic");
    }
    let mut version = [0u8; 2];
    file.read_exact(&mut version)
        .context("truncated HVC2 version")?;
    if u16::from_le_bytes(version) != HVC2_VERSION {
        bail!("unsupported HVC2 file version");
    }
    let mut header_len = [0u8; 8];
    file.read_exact(&mut header_len)
        .context("truncated HVC2 header length")?;
    let header_len = u64::from_le_bytes(header_len);
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        bail!("invalid HVC2 header length {header_len}");
    }
    let header_len_usize = usize::try_from(header_len).context("HVC2 header is too large")?;
    let mut header_bytes = vec![0u8; header_len_usize];
    file.read_exact(&mut header_bytes)
        .context("truncated HVC2 JSON header")?;
    let header: Hvc2Header =
        serde_json::from_slice(&header_bytes).context("parsing HVC2 JSON header")?;
    header.validate()?;
    if canonical_header_bytes(&header)? != header_bytes {
        bail!("HVC2 JSON header is not canonical");
    }
    let expected_file_bytes = PREAMBLE_BYTES
        .checked_add(header_len)
        .and_then(|bytes| bytes.checked_add(header.payload_bytes))
        .context("HVC2 total length overflows u64")?;
    if file_bytes != expected_file_bytes {
        if file_bytes < expected_file_bytes {
            bail!("truncated HVC2 payload");
        }
        bail!("HVC2 file has trailing bytes");
    }
    let value_count =
        usize::try_from(header.payload_bytes / 4).context("HVC2 value count does not fit usize")?;
    let mut values = Vec::with_capacity(value_count);
    let mut digest = Sha256::new();
    for index in 0..value_count {
        let mut bytes = [0u8; 4];
        file.read_exact(&mut bytes)
            .with_context(|| format!("truncated HVC2 value {index}"))?;
        digest.update(bytes);
        let value = f32::from_le_bytes(bytes);
        if !value.is_finite() {
            bail!("HVC2 payload contains a non-finite value at index {index}");
        }
        values.push(value);
    }
    let observed_payload_sha256 = digest_hex(digest.finalize());
    if observed_payload_sha256 != header.payload_sha256 {
        bail!("HVC2 payload SHA-256 mismatch");
    }
    Ok(Hvc2Artifact { header, values })
}

pub fn hvc2_file_binding(path: &Path) -> Result<Hvc2FileBinding> {
    let artifact = read_hvc2(path)?;
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("HVC2 filename is not UTF-8")?
        .to_string();
    if filename != artifact.header.split.filename() {
        bail!("HVC2 filename does not match its split");
    }
    Ok(Hvc2FileBinding {
        filename,
        sha256: sha256_file(path)?,
        bytes: std::fs::metadata(path)?.len(),
        header: artifact.header,
    })
}

pub fn write_hvc2_receipt_atomic(path: &Path, receipt: &Hvc2PairReceipt) -> Result<()> {
    if path.file_name().and_then(|value| value.to_str()) != Some(HVC2_RECEIPT_FILENAME) {
        bail!("HVC2 pair receipt must be named {HVC2_RECEIPT_FILENAME}");
    }
    receipt.validate()?;
    let parent = path.parent().context("HVC2 receipt path has no parent")?;
    verify_hvc2_pair(parent, receipt)?;
    if path.exists() {
        bail!(
            "refusing to overwrite existing HVC2 receipt {}",
            path.display()
        );
    }
    let mut bytes = serde_json::to_vec_pretty(receipt)?;
    bytes.push(b'\n');
    std::fs::create_dir_all(parent)?;
    let staging = staging_path(path)?;
    let write_result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&staging)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        install_no_clobber(&staging, path)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&staging);
    write_result
}

pub fn read_hvc2_receipt(path: &Path) -> Result<Hvc2PairReceipt> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading HVC2 receipt {}", path.display()))?;
    let receipt: Hvc2PairReceipt =
        serde_json::from_slice(&bytes).context("parsing HVC2 pair receipt")?;
    receipt.validate()?;
    Ok(receipt)
}

pub fn verify_hvc2_pair(directory: &Path, receipt: &Hvc2PairReceipt) -> Result<()> {
    receipt.validate()?;
    for expected in [&receipt.train_cache, &receipt.test_cache] {
        let path = directory.join(&expected.filename);
        let actual = hvc2_file_binding(&path)?;
        if &actual != expected {
            bail!("HVC2 file {} does not match its receipt", path.display());
        }
    }
    Ok(())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hashing {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest_hex(digest.finalize()))
}

fn canonical_header_bytes(header: &Hvc2Header) -> Result<Vec<u8>> {
    let bytes = serde_json::to_vec(header)?;
    if bytes.is_empty() || u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_HEADER_BYTES {
        bail!("HVC2 canonical header is empty or too large");
    }
    Ok(bytes)
}

fn sha256_f32_le(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    digest_hex(digest.finalize())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        bail!("invalid lowercase SHA-256 for HVC2 {label}");
    }
    Ok(())
}

fn digest_hex(bytes: impl AsRef<[u8]>) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes.as_ref() {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn staging_path(path: &Path) -> Result<PathBuf> {
    static NEXT_STAGING: AtomicU64 = AtomicU64::new(0);
    let parent = path.parent().context("atomic output path has no parent")?;
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("atomic output filename is not UTF-8")?;
    Ok(parent.join(format!(
        ".{filename}.{}.{}.tmp",
        std::process::id(),
        NEXT_STAGING.fetch_add(1, Ordering::Relaxed)
    )))
}

fn install_no_clobber(staging: &Path, destination: &Path) -> Result<()> {
    std::fs::hard_link(staging, destination).with_context(|| {
        format!(
            "installing {} without overwriting {}",
            staging.display(),
            destination.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "feat-core-hvc2-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn metadata(split: Hvc2Split) -> Hvc2Metadata {
        Hvc2Metadata::encoded("dataset", "regression", split, "a".repeat(64), 123)
    }

    fn write_sample(root: &Path) -> (PathBuf, Hvc2Header, Vec<f32>) {
        std::fs::create_dir_all(root).unwrap();
        let values = vec![1.0, 2.0, 3.5, -4.25];
        let header = Hvc2Header::new(metadata(Hvc2Split::Train), 2, 2, &values).unwrap();
        let path = root.join(HVC2_TRAIN_FILENAME);
        write_hvc2_atomic(&path, &header, &values).unwrap();
        (path, header, values)
    }

    #[test]
    fn deterministic_round_trip_is_row_major_and_exact() {
        let left = temp_dir("det-left");
        let right = temp_dir("det-right");
        let (left_path, header, values) = write_sample(&left);
        let (right_path, _, _) = write_sample(&right);
        assert_eq!(
            std::fs::read(&left_path).unwrap(),
            std::fs::read(&right_path).unwrap()
        );
        let decoded = read_hvc2(&left_path).unwrap();
        assert_eq!(decoded.header, header);
        assert_eq!(decoded.values, values);
        let _ = std::fs::remove_dir_all(left);
        let _ = std::fs::remove_dir_all(right);
    }

    #[test]
    fn rejects_corruption_truncation_trailing_bytes_and_version_drift() {
        let root = temp_dir("corruption");
        let (path, _, _) = write_sample(&root);
        let original = std::fs::read(&path).unwrap();

        let corrupt = root.join("corrupt.hvc2");
        let mut bytes = original.clone();
        *bytes.last_mut().unwrap() ^= 1;
        std::fs::write(&corrupt, bytes).unwrap();
        assert!(read_hvc2(&corrupt)
            .unwrap_err()
            .to_string()
            .contains("SHA-256"));

        let truncated = root.join("truncated.hvc2");
        std::fs::write(&truncated, &original[..original.len() - 1]).unwrap();
        assert!(read_hvc2(&truncated)
            .unwrap_err()
            .to_string()
            .contains("truncated"));

        let trailing = root.join("trailing.hvc2");
        let mut bytes = original.clone();
        bytes.push(0);
        std::fs::write(&trailing, bytes).unwrap();
        assert!(read_hvc2(&trailing)
            .unwrap_err()
            .to_string()
            .contains("trailing"));

        let unsupported = root.join("unsupported.hvc2");
        let mut bytes = original;
        bytes[4..6].copy_from_slice(&3u16.to_le_bytes());
        std::fs::write(&unsupported, bytes).unwrap();
        assert!(read_hvc2(&unsupported)
            .unwrap_err()
            .to_string()
            .contains("unsupported"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn zero_column_not_applicable_cache_is_valid() {
        let root = temp_dir("empty");
        std::fs::create_dir_all(&root).unwrap();
        let metadata = Hvc2Metadata::no_exogenous_features(
            "target-only",
            "timeseries",
            Hvc2Split::Test,
            "b".repeat(64),
            44,
        );
        let header = Hvc2Header::new(metadata, 9, 0, &[]).unwrap();
        let path = root.join(HVC2_TEST_FILENAME);
        write_hvc2_atomic(&path, &header, &[]).unwrap();
        let decoded = read_hvc2(&path).unwrap();
        assert_eq!(decoded.header.columns, 0);
        assert_eq!(decoded.header.rows, 9);
        assert!(decoded.values.is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_non_finite_and_wrong_model_hashes() {
        assert!(Hvc2Header::new(metadata(Hvc2Split::Train), 1, 1, &[f32::NAN]).is_err());
        let mut header = Hvc2Header::new(metadata(Hvc2Split::Train), 1, 1, &[1.0]).unwrap();
        header.weights_sha256 = "c".repeat(64);
        assert!(header.validate().is_err());
    }

    #[test]
    fn atomic_write_refuses_to_replace_an_existing_cache() {
        let root = temp_dir("no-clobber");
        let (path, header, values) = write_sample(&root);
        let before = std::fs::read(&path).unwrap();
        assert!(write_hvc2_atomic(&path, &header, &values).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        let _ = std::fs::remove_dir_all(root);
    }
}
