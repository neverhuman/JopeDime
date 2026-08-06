//! Pure, fail-closed identity contract for Foundation Encoder V2.
//!
//! This module performs no filesystem or runtime work. It deliberately keeps
//! the encoder capsule separate from the train-only dataset input, then binds
//! both canonical digests into a domain-separated cache key. V1 readers and
//! writers do not depend on these types.

use crate::selfplay_data::canonical_json_sha256;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

pub const ENCODER_IDENTITY_SCHEMA: &str = "JopeFoundationEncoderIdentity.v2";
pub const DATASET_INPUT_SCHEMA: &str = "JopeFoundationDatasetInput.v2";
pub const CONTEXT_TRANSFORM_SCHEMA: &str = "JopeContextTransformIdentity.v2";
pub const CACHE_KEY_DOMAIN: &str = "jope-foundation-encoder-cache-v2";
pub const OUTPUT_DTYPE: &str = "float32";
pub const OUTPUT_FEATURES: u32 = 2_048;
pub const OUTPUT_BYTES: u64 = 8_192;
pub const OUTPUT_BYTE_ORDER: &str = "little-endian";
pub const OUTPUT_LAYOUT: &str = "mean,std,p10,p90 by hidden-index";

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str, field: &str) -> Result<()> {
    ensure!(!value.trim().is_empty(), "{field} is absent");
    Ok(())
}

fn sha256(value: &str, field: &str) -> Result<()> {
    ensure!(is_sha256(value), "{field} is not a lowercase SHA-256");
    Ok(())
}

fn validate_relative_path(value: &str, field: &str) -> Result<()> {
    nonempty(value, field)?;
    let path = Path::new(value);
    ensure!(!path.is_absolute(), "{field} must be relative");
    ensure!(
        path.components()
            .all(|part| matches!(part, Component::Normal(_))),
        "{field} is not a normalized relative path"
    );
    Ok(())
}

fn path_looks_like_test_input(value: &str) -> bool {
    Path::new(value).components().any(|component| {
        let Component::Normal(name) = component else {
            return false;
        };
        let name = name.to_string_lossy().to_ascii_lowercase();
        name == "test" || name.starts_with("test.") || name.starts_with("test_")
    })
}

fn validate_unique_artifacts(artifacts: &[ArtifactIdentity], field: &str) -> Result<()> {
    ensure!(!artifacts.is_empty(), "{field} is empty");
    let mut paths = BTreeSet::new();
    for artifact in artifacts {
        artifact.validate()?;
        ensure!(
            paths.insert(artifact.path.as_str()),
            "{field} repeats artifact path: {}",
            artifact.path
        );
    }
    ensure!(
        artifacts.windows(2).all(|pair| pair[0].path < pair[1].path),
        "{field} must be strictly sorted by artifact path"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactIdentity {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

impl ArtifactIdentity {
    pub fn validate(&self) -> Result<()> {
        validate_relative_path(&self.path, "artifact path")?;
        sha256(&self.sha256, "artifact SHA-256")?;
        ensure!(self.size_bytes > 0, "artifact is empty: {}", self.path);
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FoundationDocumentIdentity {
    pub inventory_sha256: String,
    pub projection_sha256: String,
    pub standardization_sha256: String,
}

impl FoundationDocumentIdentity {
    fn validate(&self) -> Result<()> {
        sha256(&self.inventory_sha256, "foundation inventory SHA-256")?;
        sha256(&self.projection_sha256, "foundation projection SHA-256")?;
        sha256(
            &self.standardization_sha256,
            "foundation standardization SHA-256",
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BuildIdentity {
    pub repository_commit: String,
    pub compiler: String,
    pub target_triple: String,
    pub profile: String,
    pub enabled_features: Vec<String>,
    pub lockfiles: Vec<ArtifactIdentity>,
}

impl BuildIdentity {
    fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.repository_commit.len(), 40 | 64)
                && self
                    .repository_commit
                    .bytes()
                    .all(|byte| { byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte) }),
            "repository commit is not a lowercase 40- or 64-character hexadecimal digest"
        );
        nonempty(&self.compiler, "compiler")?;
        nonempty(&self.target_triple, "target triple")?;
        nonempty(&self.profile, "build profile")?;
        ensure!(
            self.enabled_features
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "enabled features must be unique and sorted"
        );
        validate_unique_artifacts(&self.lockfiles, "build lockfiles")?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimePackageIdentity {
    pub version: String,
    pub package_tree_sha256: String,
}

impl RuntimePackageIdentity {
    fn validate(&self, name: &str) -> Result<()> {
        nonempty(name, "runtime package name")?;
        nonempty(&self.version, "runtime package version")?;
        sha256(&self.package_tree_sha256, "runtime package-tree SHA-256")?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalRuntimeIdentity {
    pub requested_backend: String,
    pub required_actual_backend: String,
    pub device_class: String,
    pub gpu_compute_capability: Option<String>,
    pub cuda_version: Option<String>,
    pub driver_version: Option<String>,
    pub deterministic: bool,
    pub deterministic_flags: BTreeMap<String, bool>,
    pub input_dtype: String,
    pub accumulation_dtype: String,
    pub batch_size: u32,
    pub thread_count: u32,
    pub packages: BTreeMap<String, RuntimePackageIdentity>,
    pub kernel_policy_sha256: String,
    pub memory_policy_sha256: String,
}

impl NumericalRuntimeIdentity {
    fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.requested_backend.as_str(), "cpu" | "cuda"),
            "requested backend is not cpu or cuda"
        );
        ensure!(
            self.required_actual_backend == self.requested_backend,
            "requested and required actual backends differ"
        );
        nonempty(&self.device_class, "device class")?;
        nonempty(&self.input_dtype, "input dtype")?;
        nonempty(&self.accumulation_dtype, "accumulation dtype")?;
        ensure!(self.batch_size > 0, "batch size is zero");
        ensure!(self.thread_count > 0, "thread count is zero");
        ensure!(self.deterministic, "deterministic runtime mode is disabled");
        ensure!(
            !self.deterministic_flags.is_empty()
                && self.deterministic_flags.values().all(|enabled| *enabled),
            "deterministic runtime flags are absent or disabled"
        );
        ensure!(!self.packages.is_empty(), "runtime packages are absent");
        for (name, package) in &self.packages {
            package.validate(name)?;
        }
        ensure!(
            self.packages.contains_key("pytorch"),
            "runtime package tree does not bind PyTorch"
        );
        match self.requested_backend.as_str() {
            "cuda" => {
                nonempty(
                    self.gpu_compute_capability.as_deref().unwrap_or_default(),
                    "GPU compute capability",
                )?;
                nonempty(
                    self.cuda_version.as_deref().unwrap_or_default(),
                    "CUDA version",
                )?;
                nonempty(
                    self.driver_version.as_deref().unwrap_or_default(),
                    "driver version",
                )?;
                ensure!(
                    self.packages.contains_key("cuda"),
                    "CUDA backend does not bind a CUDA package tree"
                );
            }
            "cpu" => ensure!(
                self.gpu_compute_capability.is_none()
                    && self.cuda_version.is_none()
                    && self.driver_version.is_none(),
                "CPU identity ambiguously declares CUDA/device fields"
            ),
            _ => unreachable!("backend validated above"),
        }
        sha256(&self.kernel_policy_sha256, "kernel policy SHA-256")?;
        sha256(&self.memory_policy_sha256, "memory policy SHA-256")?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OutputContractIdentity {
    pub dtype: String,
    pub vector_count: u32,
    pub feature_count: u32,
    pub byte_length: u64,
    pub byte_order: String,
    pub finite_only: bool,
    pub feature_layout: String,
    pub format_sha256: String,
}

impl OutputContractIdentity {
    fn validate(&self) -> Result<()> {
        ensure!(self.dtype == OUTPUT_DTYPE, "output dtype is not float32");
        ensure!(
            self.vector_count == 1,
            "output does not contain exactly one vector"
        );
        ensure!(
            self.feature_count == OUTPUT_FEATURES,
            "output feature count is not {OUTPUT_FEATURES}"
        );
        ensure!(
            self.byte_length == OUTPUT_BYTES,
            "output byte length is not {OUTPUT_BYTES}"
        );
        ensure!(
            self.byte_order == OUTPUT_BYTE_ORDER,
            "output byte order is not {OUTPUT_BYTE_ORDER}"
        );
        ensure!(self.finite_only, "output permits nonfinite values");
        ensure!(
            self.feature_layout == OUTPUT_LAYOUT,
            "output feature layout is not the declared Foundation layout"
        );
        sha256(&self.format_sha256, "output format SHA-256")?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FoundationEncoderIdentityV2 {
    pub schema: String,
    pub bundle_sha256: String,
    pub prime_source_bundle_sha256: String,
    pub prime_package_tree_sha256: String,
    pub protocol_name: String,
    pub protocol: ArtifactIdentity,
    pub python_implementation_tree_sha256: String,
    pub python_encoder: ArtifactIdentity,
    pub python_support: Vec<ArtifactIdentity>,
    pub wrapper: ArtifactIdentity,
    pub batch_driver: ArtifactIdentity,
    pub native_supervisor: ArtifactIdentity,
    pub build: BuildIdentity,
    pub model_config: ArtifactIdentity,
    pub model_manifests: Vec<ArtifactIdentity>,
    pub model_weights: Vec<ArtifactIdentity>,
    pub prime_support_tree_sha256: String,
    pub prime_support: Vec<ArtifactIdentity>,
    pub preprocessing_sha256: String,
    pub seed: u64,
    pub runtime: NumericalRuntimeIdentity,
    pub output: OutputContractIdentity,
}

impl FoundationEncoderIdentityV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == ENCODER_IDENTITY_SCHEMA,
            "encoder identity schema is not {ENCODER_IDENTITY_SCHEMA}"
        );
        sha256(&self.bundle_sha256, "bundle SHA-256")?;
        sha256(
            &self.prime_source_bundle_sha256,
            "Prime source-bundle SHA-256",
        )?;
        sha256(
            &self.prime_package_tree_sha256,
            "Prime package-tree SHA-256",
        )?;
        nonempty(&self.protocol_name, "encoder protocol name")?;
        self.protocol.validate()?;
        sha256(
            &self.python_implementation_tree_sha256,
            "Python implementation-tree SHA-256",
        )?;
        self.python_encoder.validate()?;
        validate_unique_artifacts(&self.python_support, "Python encoder support tree")?;
        self.wrapper.validate()?;
        self.batch_driver.validate()?;
        self.native_supervisor.validate()?;
        self.build.validate()?;
        self.model_config.validate()?;
        validate_unique_artifacts(&self.model_manifests, "model manifests")?;
        validate_unique_artifacts(&self.model_weights, "model weights")?;
        sha256(
            &self.prime_support_tree_sha256,
            "Prime support-tree SHA-256",
        )?;
        validate_unique_artifacts(&self.prime_support, "Prime support tree")?;
        sha256(&self.preprocessing_sha256, "preprocessing SHA-256")?;
        self.runtime.validate()?;
        self.output.validate()?;

        let mut all_paths = BTreeSet::new();
        for artifact in self
            .python_support
            .iter()
            .chain(std::iter::once(&self.protocol))
            .chain(std::iter::once(&self.python_encoder))
            .chain(std::iter::once(&self.wrapper))
            .chain(std::iter::once(&self.batch_driver))
            .chain(std::iter::once(&self.native_supervisor))
            .chain(std::iter::once(&self.model_config))
            .chain(self.model_manifests.iter())
            .chain(self.model_weights.iter())
            .chain(self.prime_support.iter())
            .chain(self.build.lockfiles.iter())
        {
            ensure!(
                all_paths.insert(artifact.path.as_str()),
                "encoder identity repeats artifact path: {}",
                artifact.path
            );
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextTransformIdentityV2 {
    pub schema: String,
    pub task: String,
    pub encoder_cache_key_sha256: String,
    pub encoder_output_sha256: String,
    pub foundation: FoundationDocumentIdentity,
    pub transform_protocol: ArtifactIdentity,
    pub task_transform_sha256: String,
    pub standardization_fit_cohort_sha256: String,
}

impl ContextTransformIdentityV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CONTEXT_TRANSFORM_SCHEMA,
            "context-transform schema is not {CONTEXT_TRANSFORM_SCHEMA}"
        );
        ensure!(
            matches!(
                self.task.as_str(),
                "regression" | "binary" | "multiclass" | "timeseries"
            ),
            "context-transform task is invalid: {}",
            self.task
        );
        sha256(&self.encoder_cache_key_sha256, "encoder cache-key SHA-256")?;
        sha256(&self.encoder_output_sha256, "encoder output SHA-256")?;
        self.foundation.validate()?;
        self.transform_protocol.validate()?;
        sha256(&self.task_transform_sha256, "task-transform SHA-256")?;
        sha256(
            &self.standardization_fit_cohort_sha256,
            "standardization fit-cohort SHA-256",
        )?;
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TimeSeriesReductionIdentity {
    pub code: ArtifactIdentity,
    pub metadata_sha256: String,
    pub horizon: u32,
    pub frequency: String,
    pub lag_policy_sha256: String,
    pub known_future_covariates: Vec<String>,
    pub past_only_covariates: Vec<String>,
}

impl TimeSeriesReductionIdentity {
    fn validate(&self) -> Result<()> {
        self.code.validate()?;
        sha256(&self.metadata_sha256, "time-series metadata SHA-256")?;
        ensure!(self.horizon > 0, "time-series horizon is zero");
        nonempty(&self.frequency, "time-series frequency")?;
        sha256(&self.lag_policy_sha256, "lag policy SHA-256")?;
        ensure!(
            self.known_future_covariates
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "known-future covariates must be unique and sorted"
        );
        ensure!(
            self.past_only_covariates
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "past-only covariates must be unique and sorted"
        );
        let known: BTreeSet<_> = self.known_future_covariates.iter().collect();
        ensure!(
            self.past_only_covariates
                .iter()
                .all(|name| !known.contains(name)),
            "a time-series covariate has conflicting causality roles"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetReadIdentity {
    pub purpose: String,
    pub artifact: ArtifactIdentity,
}

impl DatasetReadIdentity {
    fn validate(&self) -> Result<()> {
        ensure!(
            matches!(
                self.purpose.as_str(),
                "train_data" | "dataset_metadata" | "reduction_code"
            ),
            "dataset-read purpose is not train-only: {}",
            self.purpose
        );
        self.artifact.validate()?;
        ensure!(
            !path_looks_like_test_input(&self.artifact.path),
            "dataset read path appears to contain test bytes: {}",
            self.artifact.path
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FoundationDatasetInputV2 {
    pub schema: String,
    pub task: String,
    pub source_family: String,
    pub train_sha256: String,
    pub train_size_bytes: u64,
    pub feature_target_schema_sha256: String,
    pub row_count: u64,
    pub column_count: u32,
    pub train_only: bool,
    pub files_read: Vec<DatasetReadIdentity>,
    pub time_series_reduction: Option<TimeSeriesReductionIdentity>,
}

impl FoundationDatasetInputV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == DATASET_INPUT_SCHEMA,
            "dataset input schema is not {DATASET_INPUT_SCHEMA}"
        );
        ensure!(
            matches!(
                self.task.as_str(),
                "regression" | "binary" | "multiclass" | "timeseries"
            ),
            "dataset input task is invalid: {}",
            self.task
        );
        nonempty(&self.source_family, "source family")?;
        sha256(&self.train_sha256, "train SHA-256")?;
        ensure!(self.train_size_bytes > 0, "train input is empty");
        sha256(
            &self.feature_target_schema_sha256,
            "feature/target schema SHA-256",
        )?;
        ensure!(self.row_count > 0, "dataset row count is zero");
        ensure!(self.column_count > 0, "dataset column count is zero");
        ensure!(self.train_only, "dataset input is not train-only");
        ensure!(!self.files_read.is_empty(), "files read are absent");
        let mut read_paths = BTreeSet::new();
        for file in &self.files_read {
            file.validate()?;
            ensure!(
                read_paths.insert(file.artifact.path.as_str()),
                "files read repeat artifact path: {}",
                file.artifact.path
            );
        }
        ensure!(
            self.files_read.windows(2).all(|pair| {
                (&pair[0].purpose, &pair[0].artifact.path)
                    < (&pair[1].purpose, &pair[1].artifact.path)
            }),
            "files read must be strictly sorted by purpose and artifact path"
        );
        ensure!(
            self.files_read
                .iter()
                .filter(|file| file.purpose == "train_data")
                .count()
                == 1,
            "dataset input must declare exactly one train file"
        );
        ensure!(
            self.files_read
                .iter()
                .any(|file| file.purpose == "train_data"
                    && file.artifact.sha256 == self.train_sha256
                    && file.artifact.size_bytes == self.train_size_bytes),
            "files read do not contain the declared train input"
        );
        match (&*self.task, &self.time_series_reduction) {
            ("timeseries", Some(reduction)) => {
                reduction.validate()?;
                ensure!(
                    self.files_read.iter().any(|file| {
                        file.purpose == "reduction_code" && file.artifact == reduction.code
                    }),
                    "time-series reduction code is not declared in files read"
                );
                ensure!(
                    self.files_read.iter().any(|file| {
                        file.purpose == "dataset_metadata"
                            && file.artifact.sha256 == reduction.metadata_sha256
                    }),
                    "time-series metadata is not declared in files read"
                );
            }
            ("timeseries", None) => ensure!(false, "time-series reduction identity is absent"),
            (_, Some(_)) => ensure!(false, "non-time-series input has a reduction identity"),
            (_, None) => {}
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

pub fn cache_key(
    encoder: &FoundationEncoderIdentityV2,
    input: &FoundationDatasetInputV2,
) -> Result<String> {
    let encoder_identity_sha256 = encoder.digest()?;
    let input_identity_sha256 = input.digest()?;
    canonical_json_sha256(&json!({
        "domain": CACHE_KEY_DOMAIN,
        "encoder_identity_sha256": encoder_identity_sha256,
        "input_identity_sha256": input_identity_sha256,
    }))
}

pub fn decode_encoder_identity(bytes: &[u8]) -> Result<FoundationEncoderIdentityV2> {
    let identity: FoundationEncoderIdentityV2 =
        serde_json::from_slice(bytes).context("decode encoder identity")?;
    identity.validate()?;
    Ok(identity)
}

pub fn decode_dataset_input(bytes: &[u8]) -> Result<FoundationDatasetInputV2> {
    let input: FoundationDatasetInputV2 =
        serde_json::from_slice(bytes).context("decode dataset input")?;
    input.validate()?;
    Ok(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn artifact(path: &str, byte: u8) -> ArtifactIdentity {
        ArtifactIdentity {
            path: path.to_owned(),
            sha256: digest(byte),
            size_bytes: u64::from(byte) + 1,
        }
    }

    fn encoder() -> FoundationEncoderIdentityV2 {
        FoundationEncoderIdentityV2 {
            schema: ENCODER_IDENTITY_SCHEMA.to_owned(),
            bundle_sha256: digest(1),
            prime_source_bundle_sha256: digest(2),
            prime_package_tree_sha256: digest(3),
            protocol_name: "foundation-pooled-vector-v2".to_owned(),
            protocol: artifact("protocols/foundation-pooled-vector-v2.json", 4),
            python_implementation_tree_sha256: digest(5),
            python_encoder: artifact("python/jope/encoders/foundation.py", 6),
            python_support: vec![artifact("python/jope/encoders/support.py", 38)],
            wrapper: artifact("pytorch/encoding/foundation_backfill.py", 39),
            batch_driver: artifact("pytorch/encoding/encode_batch.py", 40),
            native_supervisor: artifact("bin/jope_encoder_supervisor", 7),
            build: BuildIdentity {
                repository_commit: "ab".repeat(20),
                compiler: "rustc-1.test".to_owned(),
                target_triple: "x86_64-unknown-linux-gnu".to_owned(),
                profile: "release".to_owned(),
                enabled_features: vec!["cuda".to_owned(), "receipt-v2".to_owned()],
                lockfiles: vec![artifact("Cargo.lock", 8), artifact("uv.lock", 9)],
            },
            model_config: artifact("prime/config.json", 10),
            model_manifests: vec![artifact("weights/manifest.json", 41)],
            model_weights: vec![artifact("weights/starlight.safetensors", 11)],
            prime_support_tree_sha256: digest(42),
            prime_support: vec![artifact("prime/support/encoder.py", 43)],
            preprocessing_sha256: digest(15),
            seed: 23,
            runtime: NumericalRuntimeIdentity {
                requested_backend: "cuda".to_owned(),
                required_actual_backend: "cuda".to_owned(),
                device_class: "sm_89".to_owned(),
                gpu_compute_capability: Some("8.9".to_owned()),
                cuda_version: Some("13.test".to_owned()),
                driver_version: Some("999.test".to_owned()),
                deterministic: true,
                deterministic_flags: BTreeMap::from([
                    ("cudnn_benchmark_disabled".to_owned(), true),
                    ("deterministic_algorithms".to_owned(), true),
                ]),
                input_dtype: "float32".to_owned(),
                accumulation_dtype: "float32".to_owned(),
                batch_size: 16,
                thread_count: 4,
                packages: BTreeMap::from([
                    (
                        "cuda".to_owned(),
                        RuntimePackageIdentity {
                            version: "13.test".to_owned(),
                            package_tree_sha256: digest(32),
                        },
                    ),
                    (
                        "pytorch".to_owned(),
                        RuntimePackageIdentity {
                            version: "2.test".to_owned(),
                            package_tree_sha256: digest(33),
                        },
                    ),
                ]),
                kernel_policy_sha256: digest(16),
                memory_policy_sha256: digest(44),
            },
            output: OutputContractIdentity {
                dtype: OUTPUT_DTYPE.to_owned(),
                vector_count: 1,
                feature_count: OUTPUT_FEATURES,
                byte_length: OUTPUT_BYTES,
                byte_order: OUTPUT_BYTE_ORDER.to_owned(),
                finite_only: true,
                feature_layout: OUTPUT_LAYOUT.to_owned(),
                format_sha256: digest(17),
            },
        }
    }

    fn context_transform(encoder_cache_key_sha256: String) -> ContextTransformIdentityV2 {
        ContextTransformIdentityV2 {
            schema: CONTEXT_TRANSFORM_SCHEMA.to_owned(),
            task: "multiclass".to_owned(),
            encoder_cache_key_sha256,
            encoder_output_sha256: digest(34),
            foundation: FoundationDocumentIdentity {
                inventory_sha256: digest(12),
                projection_sha256: digest(13),
                standardization_sha256: digest(14),
            },
            transform_protocol: artifact("protocols/context-transform-v2.json", 35),
            task_transform_sha256: digest(36),
            standardization_fit_cohort_sha256: digest(37),
        }
    }

    fn input() -> FoundationDatasetInputV2 {
        FoundationDatasetInputV2 {
            schema: DATASET_INPUT_SCHEMA.to_owned(),
            task: "multiclass".to_owned(),
            source_family: "openml:unit-test".to_owned(),
            train_sha256: digest(18),
            train_size_bytes: 19,
            feature_target_schema_sha256: digest(19),
            row_count: 128,
            column_count: 12,
            train_only: true,
            files_read: vec![DatasetReadIdentity {
                purpose: "train_data".to_owned(),
                artifact: ArtifactIdentity {
                    path: "dataset/train.csv".to_owned(),
                    sha256: digest(18),
                    size_bytes: 19,
                },
            }],
            time_series_reduction: None,
        }
    }

    fn set_pointer(value: &mut Value, pointer: &str, replacement: Value) {
        *value.pointer_mut(pointer).expect("test pointer exists") = replacement;
    }

    #[test]
    fn deterministic_cache_key_binds_encoder_and_train_input() {
        let encoder = encoder();
        let input = input();
        let stable = cache_key(&encoder, &input).unwrap();
        assert_eq!(stable, cache_key(&encoder, &input).unwrap());

        let encoder_mutations = [
            ("/model_weights/0/sha256", Value::String(digest(20))),
            ("/native_supervisor/sha256", Value::String(digest(21))),
            ("/protocol/sha256", Value::String(digest(22))),
            (
                "/runtime/packages/pytorch/package_tree_sha256",
                Value::String(digest(23)),
            ),
            ("/runtime/input_dtype", Value::String("float64".to_owned())),
            ("/output/format_sha256", Value::String(digest(26))),
        ];
        for (pointer, replacement) in encoder_mutations {
            let mut value = serde_json::to_value(&encoder).unwrap();
            set_pointer(&mut value, pointer, replacement);
            let mutated: FoundationEncoderIdentityV2 = serde_json::from_value(value).unwrap();
            assert_ne!(stable, cache_key(&mutated, &input).unwrap(), "{pointer}");
        }

        let mut cpu = encoder.clone();
        cpu.runtime.requested_backend = "cpu".to_owned();
        cpu.runtime.required_actual_backend = "cpu".to_owned();
        cpu.runtime.device_class = "x86_64-avx2".to_owned();
        cpu.runtime.gpu_compute_capability = None;
        cpu.runtime.cuda_version = None;
        cpu.runtime.driver_version = None;
        cpu.runtime.packages.remove("cuda");
        assert_ne!(stable, cache_key(&cpu, &input).unwrap(), "cpu vs cuda");

        let input_mutations = [
            ("/train_sha256", Value::String(digest(27))),
            ("/feature_target_schema_sha256", Value::String(digest(28))),
            ("/task", Value::String("binary".to_owned())),
        ];
        for (pointer, replacement) in input_mutations {
            let mut value = serde_json::to_value(&input).unwrap();
            set_pointer(&mut value, pointer, replacement);
            if pointer == "/train_sha256" {
                set_pointer(
                    &mut value,
                    "/files_read/0/artifact/sha256",
                    Value::String(digest(27)),
                );
            }
            let mutated: FoundationDatasetInputV2 = serde_json::from_value(value).unwrap();
            assert_ne!(stable, cache_key(&encoder, &mutated).unwrap(), "{pointer}");
        }
    }

    #[test]
    fn context_transform_separately_binds_foundation_and_fit_cohort() {
        let encoder_key = cache_key(&encoder(), &input()).unwrap();
        let transform = context_transform(encoder_key);
        let stable = transform.digest().unwrap();
        for (pointer, replacement) in [
            ("/foundation/inventory_sha256", Value::String(digest(22))),
            ("/foundation/projection_sha256", Value::String(digest(23))),
            (
                "/foundation/standardization_sha256",
                Value::String(digest(24)),
            ),
            (
                "/standardization_fit_cohort_sha256",
                Value::String(digest(25)),
            ),
        ] {
            let mut value = serde_json::to_value(&transform).unwrap();
            set_pointer(&mut value, pointer, replacement);
            let changed: ContextTransformIdentityV2 = serde_json::from_value(value).unwrap();
            assert_ne!(stable, changed.digest().unwrap(), "{pointer}");
        }
    }

    #[test]
    fn actual_backend_mismatch_and_non_train_input_fail_closed() {
        let mut wrong_backend = encoder();
        wrong_backend.runtime.required_actual_backend = "cpu".to_owned();
        assert!(wrong_backend.validate().is_err());

        let mut nondeterministic = encoder();
        nondeterministic.runtime.deterministic = false;
        assert!(nondeterministic.validate().is_err());

        let mut leaked = input();
        leaked.files_read[0].purpose = "test_data".to_owned();
        assert!(leaked.validate().is_err());
    }

    #[test]
    fn time_series_requires_a_causal_reduction_identity() {
        let mut time_series = input();
        time_series.task = "timeseries".to_owned();
        assert!(time_series.validate().is_err());

        time_series.time_series_reduction = Some(TimeSeriesReductionIdentity {
            code: artifact("reduction/direct.py", 29),
            metadata_sha256: digest(30),
            horizon: 12,
            frequency: "monthly".to_owned(),
            lag_policy_sha256: digest(31),
            known_future_covariates: vec!["month".to_owned()],
            past_only_covariates: vec!["target".to_owned()],
        });
        time_series.files_read.extend([
            DatasetReadIdentity {
                purpose: "reduction_code".to_owned(),
                artifact: artifact("reduction/direct.py", 29),
            },
            DatasetReadIdentity {
                purpose: "dataset_metadata".to_owned(),
                artifact: artifact("dataset/meta.json", 30),
            },
        ]);
        time_series.files_read.sort_by(|left, right| {
            (&left.purpose, &left.artifact.path).cmp(&(&right.purpose, &right.artifact.path))
        });
        assert!(time_series.validate().is_ok());
    }

    #[test]
    fn unknown_critical_fields_and_noncanonical_artifact_paths_are_rejected() {
        let mut value = serde_json::to_value(encoder()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("unbound_weights".to_owned(), Value::Bool(true));
        assert!(decode_encoder_identity(&serde_json::to_vec(&value).unwrap()).is_err());

        let mut invalid = encoder();
        invalid.python_encoder.path = "../mutable/foundation.py".to_owned();
        assert!(invalid.validate().is_err());

        let mut uppercase_commit = encoder();
        uppercase_commit.build.repository_commit = "AB".repeat(20);
        assert!(uppercase_commit.validate().is_err());

        let encoded = serde_json::to_string(&encoder()).unwrap();
        let duplicate = encoded.replacen(
            "{\"schema\":",
            "{\"schema\":\"JopeFoundationEncoderIdentity.v1\",\"schema\":",
            1,
        );
        assert!(decode_encoder_identity(duplicate.as_bytes()).is_err());
    }

    #[test]
    fn canonical_digest_is_independent_of_json_object_key_order() {
        let identity = encoder();
        let expected = identity.digest().unwrap();
        let text = serde_json::to_string_pretty(&identity).unwrap();
        let decoded = decode_encoder_identity(text.as_bytes()).unwrap();
        assert_eq!(expected, decoded.digest().unwrap());
    }
}
