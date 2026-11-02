use super::super::orig_v2::{
    OrigV2PairSemantics, OrigV2Receipt, ORIG_V2_MANIFEST_SHA256, ORIG_V2_PROFILE_ID,
    ORIG_V2_WEIGHTS_SHA256,
};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const HVC2_MAGIC: [u8; 4] = *b"HVC2";
pub const HVC2_VERSION: u16 = 2;
pub const HVC2_SCHEMA: &str = "jain.hyperion-vector-cache/v2";
pub const HVC2_RECEIPT_SCHEMA: &str = "jain.hyperion-vector-cache-receipt/v2";
pub const HVC2_TRAIN_FILENAME: &str = "train.hvc2";
pub const HVC2_TEST_FILENAME: &str = "test.hvc2";
pub const HVC2_RECEIPT_FILENAME: &str = "hyperion_v2.receipt.json";

pub(crate) const PREAMBLE_BYTES: u64 = 4 + 2 + 8;
pub(crate) const MAX_HEADER_BYTES: u64 = 1024 * 1024;

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

pub(crate) fn canonical_header_bytes(header: &Hvc2Header) -> Result<Vec<u8>> {
    let bytes = serde_json::to_vec(header)?;
    if bytes.is_empty() || u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_HEADER_BYTES {
        bail!("HVC2 canonical header is empty or too large");
    }
    Ok(bytes)
}

pub(crate) fn sha256_f32_le(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    digest_hex(digest.finalize())
}

pub(crate) fn validate_sha256(value: &str, label: &str) -> Result<()> {
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

pub(crate) fn digest_hex(bytes: impl AsRef<[u8]>) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes.as_ref() {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}
