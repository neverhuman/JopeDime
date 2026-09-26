//! Consume pinned native frozen contexts; keep experimental classification apart.
//! This verifies artifact identities, not a new CUDA execution or numerical port.
use crate::{execution::Artifact, neural_search::Comparison, sha256_hex};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const ASSEMBLY_SHA256: &str =
    "6582df23f5c1ee048d11040a5f2bd1447032932fc23f6c5a3aa9cb4f2d6896cb";
pub const TRANSFORMS_SHA256: &str =
    "1b779c6ff7a1a812e3001342d663353d3c7ff54eb93392a600f01813f4c66e22";
pub const NATIVE_BACKEND: &str = "Rust/LibTorch 2.7 CUDA FP32; original frozen transforms";
pub const CLASSIFICATION_LINEAGE: &str = "asm.experimental-class-partition-pool/v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrainingIdentity {
    pub prepared_sha256: String,
    pub training_sha256: String,
    pub selected_training_rows_sha256: String,
    pub preprocessing_sha256: String,
    pub cut: usize,
}
impl TrainingIdentity {
    fn validate(&self) -> Result<()> {
        for digest in [
            &self.prepared_sha256,
            &self.training_sha256,
            &self.selected_training_rows_sha256,
            &self.preprocessing_sha256,
        ] {
            ensure!(
                jope_policy::is_sha256_hex(digest),
                "invalid training context identity"
            );
        }
        Ok(())
    }
}

/// Native context files already produced through the existing engine authority
/// and GPU fences. Caller pins their hashes in the campaign manifest. No dataset
/// loader, panel root, scoring labels or alternate encoder is accepted here.
#[derive(Serialize, Deserialize)]
pub struct NativeContextFiles {
    pub vector: Artifact,
    pub receipt: Artifact,
    pub frozen_transforms: Artifact,
    pub training: TrainingIdentity,
}

/// Native receipt lineage and independently qualified CPU lineage remain distinct.
#[derive(Serialize, Deserialize)]
pub enum ContextFiles {
    Native(Box<NativeContextFiles>),
    #[cfg(feature = "qualified-context")]
    QualifiedCpu(Box<crate::cpu_context_bridge::CpuContextFiles>),
}
impl ContextFiles {
    pub fn receipt(&self) -> &Artifact {
        match self {
            Self::Native(files) => &files.receipt,
            #[cfg(feature = "qualified-context")]
            Self::QualifiedCpu(files) => &files.artifact,
        }
    }
    pub fn inputs(&self) -> Vec<&Artifact> {
        match self {
            Self::Native(files) => vec![&files.vector, &files.receipt, &files.frozen_transforms],
            #[cfg(feature = "qualified-context")]
            Self::QualifiedCpu(files) => vec![
                &files.vector,
                &files.artifact,
                &files.frozen_transforms,
                &files.bundle,
            ],
        }
    }
    pub fn load_for_comparison(
        &self,
        comparison: &Comparison,
    ) -> Result<VerifiedRegressionContext> {
        let value = match self {
            Self::Native(files) => {
                ensure!(
                    comparison.encoding.backend == NATIVE_BACKEND,
                    "native context backend differs from input plan"
                );
                VerifiedRegressionContext::load(files)?
            }
            #[cfg(feature = "qualified-context")]
            Self::QualifiedCpu(files) => {
                use crate::cpu_context_bridge::{load_qualified, AdmissionExpectation};
                ensure!(
                    comparison.encoding.backend == jope_policy::context_producer::BACKEND,
                    "CPU context backend differs from input plan"
                );
                let expected = AdmissionExpectation {
                    source_sha40: comparison.encoding.source_sha40.clone(),
                    producer_sha256: comparison.encoding.program_sha256.clone(),
                    input_sha256: comparison.encoding.input_sha256.clone(),
                    qualification_sha256: comparison
                        .encoding
                        .qualification_sha256
                        .clone()
                        .ok_or_else(|| anyhow::anyhow!("CPU qualification pin absent"))?,
                    reference_manifest_sha256: comparison
                        .encoding
                        .reference_manifest_sha256
                        .clone()
                        .ok_or_else(|| anyhow::anyhow!("CPU reference pin absent"))?,
                    dataset_sha256: comparison.selection.outer().dataset_sha256.clone(),
                    training: TrainingIdentity {
                        prepared_sha256: comparison.prepared_sha256.clone(),
                        training_sha256: comparison.training_sha256.clone(),
                        selected_training_rows_sha256: comparison
                            .selection
                            .outer()
                            .fitting_rows_sha256
                            .clone(),
                        preprocessing_sha256: comparison.preprocessing_sha256.clone(),
                        cut: comparison.cut,
                    },
                };
                let verified = load_qualified(files, &expected)?;
                let execution: &crate::cpu_context_bridge::ExecutionProof =
                    &files.context_execution;
                ensure!(
                    comparison.context_cost.production_receipt.path == execution.outcome.path
                        && comparison.context_cost.production_receipt.sha256
                            == execution.outcome.sha256,
                    "original CPU encoding cost receipt changed"
                );
                let outcome: crate::execution::WorkerOutcome =
                    serde_json::from_slice(&std::fs::read(&execution.outcome.path)?)?;
                ensure!(
                    comparison.context_cost.original_wall_seconds
                        == outcome.consumption.wall_seconds
                        && comparison.context_cost.original_cpu_seconds
                            == outcome.consumption.cpu_seconds
                        && comparison.context_cost.original_gpu_seconds == Some(0.),
                    "CPU context cost differs from actual execution"
                );
                if !comparison.context_cost.cached {
                    ensure!(
                        execution.job.arm.as_ref().is_some_and(|a| comparison
                            .arm(true)
                            .is_ok_and(|expected| a.id_sha256 == expected.id_sha256))
                            && execution.job.request.experiment_sha256
                                == comparison.selection.outer().experiment_sha256,
                        "fresh context was not charged to original neural arm"
                    );
                }
                VerifiedRegressionContext {
                    values: verified.context().to_vec(),
                    sha256: jope_policy::context_sha256(verified.context()),
                    training: expected.training,
                    admission_pins_sha256: Some(verified.admission_pins_sha256().into()),
                }
            }
        };
        value.matches(comparison)?;
        Ok(value)
    }
}

/// The native context receipt of `files`, validated against its training
/// closure and the pinned vector (W2-T04 G-10e): parsed from exactly the
/// bytes its pin verified, read once.
fn native_receipt(files: &NativeContextFiles) -> Result<serde_json::Value> {
    let meta: serde_json::Value = serde_json::from_slice(&files.receipt.read_verified()?)?;
    validate_native_receipt(&meta, &files.training, &files.vector.sha256)?;
    Ok(meta)
}

/// The native context vector bytes of `files` (W2-T04 G-10e): exactly the
/// bytes its pin verified, read once.
fn native_vector(files: &NativeContextFiles) -> Result<Vec<u8>> {
    files.vector.read_verified()
}

pub struct VerifiedRegressionContext {
    values: Vec<f32>,
    sha256: String,
    training: TrainingIdentity,
    /// Digest of the controller-verified CPU context admission pins; `None` for a native
    /// context. The policy request must carry exactly this value.
    admission_pins_sha256: Option<String>,
}
impl VerifiedRegressionContext {
    pub fn load(files: &NativeContextFiles) -> Result<Self> {
        files.training.validate()?;
        native_receipt(files)?;
        ensure!(
            files.frozen_transforms.sha256 == TRANSFORMS_SHA256,
            "wrong frozen transform lineage"
        );
        files.frozen_transforms.verify()?;
        let bytes = native_vector(files)?;
        ensure!(
            bytes.len() == 2048 * 4,
            "accepted context must contain exactly2048 float32 values"
        );
        let values = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
            .collect::<Vec<_>>();
        ensure!(
            values.iter().all(|v| v.is_finite() && v.abs() <= 32.),
            "native context is nonfinite or unclipped"
        );
        Ok(Self {
            values,
            // The digest of the bytes consumed (equal to the pin, which
            // `native_vector` required of exactly these bytes).
            sha256: sha256_hex(&bytes),
            training: files.training.clone(),
            admission_pins_sha256: None,
        })
    }
    pub fn training(&self) -> &TrainingIdentity {
        &self.training
    }
    /// Digest of the controller-verified CPU admission pins (`None` for a native context).
    pub fn admission_pins_sha256(&self) -> Option<&str> {
        self.admission_pins_sha256.as_deref()
    }
    pub fn validate_training(&self, expected: &TrainingIdentity) -> Result<()> {
        expected.validate()?;
        ensure!(
            self.training == *expected,
            "context differs from independently verified training closure"
        );
        Ok(())
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub(crate) fn matches(&self, comparison: &Comparison) -> Result<()> {
        ensure!(comparison.selection.outer().task==jope_datasets::catalog::Task::Regression,"accepted policy context is regression-only; classification requires its distinct experimental lineage");
        ensure!(
            self.training.prepared_sha256 == comparison.prepared_sha256
                && self.training.training_sha256 == comparison.training_sha256
                && self.training.cut == comparison.cut
                && self.training.selected_training_rows_sha256
                    == comparison.selection.outer().fitting_rows_sha256
                && self.training.preprocessing_sha256 == comparison.preprocessing_sha256,
            "context is from different training cut"
        );
        Ok(())
    }
}
fn validate_native_receipt(
    meta: &serde_json::Value,
    train: &TrainingIdentity,
    context_sha: &str,
) -> Result<()> {
    ensure!(
        meta["schema"] == "parallel.encoding.v1"
            && meta["dataset_dimensions"] == 1536
            && meta["target_dimensions"] == 512,
        "native context dimensions/schema mismatch"
    );
    ensure!(
        meta["prepared_sha256"] == train.prepared_sha256
            && meta["training_sha256"] == train.training_sha256
            && meta["cut"] == train.cut
            && meta["selected_training_rows_sha256"] == train.selected_training_rows_sha256
            && meta["preprocessing_sha256"] == train.preprocessing_sha256,
        "native context training closure changed"
    );
    ensure!(
        meta["encoder_spec_sha256"] == ASSEMBLY_SHA256
            && meta["numerical_backend"] == NATIVE_BACKEND
            && meta["output_sha256"] == context_sha,
        "native context lineage/output changed"
    );
    ensure!(
        meta["partition"]
            .as_str()
            .is_some_and(|p| ["meta_train", "meta_validation"].contains(&p)),
        "sealed/unknown partition refused"
    );
    ensure!(
        meta.get("task").is_none_or(|t| t == "regression")
            && meta
                .get("qualification")
                .is_none_or(|q| q.as_bool() == Some(false))
            && meta
                .get("final_test_targets_used")
                .and_then(serde_json::Value::as_bool)
                == Some(false),
        "classification/qualification/heldout context cannot claim accepted regression lineage"
    );
    Ok(())
}

/// Experimental, non-accepted context. The target block pools statistics of the
/// class partition over TRAIN row encodings. Labels are used only for equality;
/// groups are visited by first row, so even floating-point summation order is
/// invariant under a bijective renaming. No ordinal numeric class IDs are used.
/// Hyperion/foundation blocks must come from the same train-only input closure.
#[derive(Debug, Serialize)]
pub struct ExperimentalClassificationContext {
    lineage: &'static str,
    training: TrainingIdentity,
    values: Vec<f32>,
    context_sha256: String,
    classes: usize,
    policy_compatible: bool,
}
impl ExperimentalClassificationContext {
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn sha256(&self) -> &str {
        &self.context_sha256
    }
}
pub fn classification_context(
    training: TrainingIdentity,
    train_row_ids: &[String],
    hyperion: &[f32],
    projected_foundation: &[f32],
    train_row_encodings: &[Vec<f32>],
    train_labels: &[String],
) -> Result<ExperimentalClassificationContext> {
    training.validate()?;
    ensure!(
        sha256_hex(&serde_json::to_vec(train_row_ids)?) == training.selected_training_rows_sha256,
        "classification rows differ from declared train selection"
    );
    ensure!(
        train_row_ids
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == train_row_ids.len(),
        "duplicate train row identity"
    );
    ensure!(
        hyperion.len() == 512
            && projected_foundation.len() == 1024
            && hyperion
                .iter()
                .chain(projected_foundation)
                .all(|v| v.is_finite()),
        "experimental context requires train Hyperion512 + projected foundation1024"
    );
    ensure!(
        !train_labels.is_empty()
            && train_labels.len() == train_row_encodings.len()
            && train_labels.len() == train_row_ids.len(),
        "classification training rows/labels mismatch"
    );
    let mut indices = BTreeMap::new();
    let mut sums: Vec<([f64; 128], usize)> = vec![];
    for (label, row) in train_labels.iter().zip(train_row_encodings) {
        ensure!(
            row.len() == 128 && row.iter().all(|v| v.is_finite()),
            "classification row encoding must be finite128-vector"
        );
        let next = sums.len();
        let group = *indices.entry(label).or_insert(next);
        if group == sums.len() {
            sums.push(([0.; 128], 0));
        }
        sums[group].1 += 1;
        for (sum, value) in sums[group].0.iter_mut().zip(row) {
            *sum += *value as f64;
        }
    }
    ensure!(
        sums.len() >= 2,
        "classification needs at least two train classes"
    );
    let means = sums
        .iter()
        .map(|(sum, n)| sum.map(|v| v / *n as f64))
        .collect::<Vec<_>>();
    let mut target = vec![0f32; 512];
    for j in 0..128 {
        let average = means.iter().map(|m| m[j]).sum::<f64>() / means.len() as f64;
        target[j] = average as f32;
        target[128 + j] = (means.iter().map(|m| (m[j] - average).powi(2)).sum::<f64>()
            / means.len() as f64)
            .sqrt() as f32;
        target[256 + j] = means.iter().map(|m| m[j]).fold(f64::INFINITY, f64::min) as f32;
        target[384 + j] = means.iter().map(|m| m[j]).fold(f64::NEG_INFINITY, f64::max) as f32;
    }
    let values = hyperion
        .iter()
        .chain(projected_foundation)
        .chain(&target)
        .copied()
        .collect::<Vec<_>>();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "experimental pooling overflow"
    );
    Ok(ExperimentalClassificationContext {
        lineage: CLASSIFICATION_LINEAGE,
        context_sha256: jope_policy::context_sha256(&values),
        training,
        values,
        classes: sums.len(),
        policy_compatible: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    /// W2-T02: the controller's duplicated lineage constants are the accepted semantic
    /// manifest's values (the release decision).
    /// Replace `target` (new inode, via rename) with `bytes` the first time a
    /// receiver reports it verified.
    fn replace_on_verification(
        target: &std::path::Path,
        bytes: Vec<u8>,
    ) -> crate::execution::verification_hook::Guard {
        let target = std::fs::canonicalize(target).unwrap();
        let mut done = false;
        crate::execution::verification_hook::install(move |path| {
            if !done && std::fs::canonicalize(path).ok().as_deref() == Some(target.as_path()) {
                done = true;
                let staged = target.with_extension("replacement");
                std::fs::write(&staged, &bytes).unwrap();
                std::fs::rename(&staged, &target).unwrap();
            }
        })
    }

    /// W2-T04 G-10e: the native context receipt and vector consumed are the
    /// bytes their pins verified; a file replaced after its check is never
    /// consumed.
    #[test]
    fn a_native_context_replaced_after_its_check_is_never_consumed() {
        let dir = tempfile::tempdir().unwrap();
        let vector: Vec<u8> = (0..2048)
            .flat_map(|i| ((i % 7) as f32 * 0.1).to_le_bytes())
            .collect();
        let mut meta = native_meta();
        meta["output_sha256"] = serde_json::json!(sha256_hex(&vector));
        let meta_bytes = serde_json::to_vec(&meta).unwrap();
        let pin = |name: &str, bytes: &[u8]| {
            let path = dir.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            Artifact {
                path,
                sha256: sha256_hex(bytes),
            }
        };
        let files = NativeContextFiles {
            vector: pin("context.bin", &vector),
            receipt: pin("receipt.json", &meta_bytes),
            frozen_transforms: Artifact {
                path: dir.path().join("absent-transforms"),
                sha256: TRANSFORMS_SHA256.into(),
            },
            training: training(),
        };
        // Positive controls.
        assert_eq!(native_vector(&files).unwrap(), vector);
        assert_eq!(native_receipt(&files).unwrap(), meta);
        // A context vector replaced after its check (another finite, clipped context).
        let other: Vec<u8> = (0..2048).flat_map(|_| 0.5f32.to_le_bytes()).collect();
        {
            let _hook = replace_on_verification(&files.vector.path, other);
            assert_eq!(
                native_vector(&files).unwrap(),
                vector,
                "a context vector other than the pinned bytes was consumed"
            );
        }
        std::fs::write(&files.vector.path, &vector).unwrap();
        // A receipt replaced after its check (still a valid receipt).
        let mut replaced = meta.clone();
        replaced["note"] = serde_json::json!("replaced after its check");
        {
            let _hook = replace_on_verification(
                &files.receipt.path,
                serde_json::to_vec(&replaced).unwrap(),
            );
            assert_eq!(
                native_receipt(&files).unwrap(),
                meta,
                "a context receipt other than the pinned bytes was consumed"
            );
        }
        // Negative control: bytes other than the pin are refused outright.
        std::fs::write(&files.vector.path, [0u8; 8192]).unwrap();
        assert!(native_vector(&files).is_err());
    }

    #[test]
    fn duplicated_lineage_constants_are_the_accepted_semantic_manifests() {
        let release = jope_policy::semantic::accepted().unwrap();
        assert_eq!(ASSEMBLY_SHA256, release.encoders.assembly_sha256);
        assert_eq!(TRANSFORMS_SHA256, release.encoders.frozen_transforms_sha256);
    }
    fn row_ids() -> Vec<String> {
        [3usize, 5, 8, 12]
            .iter()
            .map(|n| sha256_hex(n.to_string().as_bytes()))
            .collect()
    }
    fn training() -> TrainingIdentity {
        TrainingIdentity {
            prepared_sha256: sha256_hex(b"synthetic prepared"),
            training_sha256: sha256_hex(b"synthetic train"),
            selected_training_rows_sha256: sha256_hex(&serde_json::to_vec(&row_ids()).unwrap()),
            preprocessing_sha256: sha256_hex(b"synthetic preprocessing"),
            cut: 2,
        }
    }
    fn native_meta() -> serde_json::Value {
        let t = training();
        serde_json::json!({"schema":"parallel.encoding.v1", "dataset_dimensions":1536,"target_dimensions":512,"prepared_sha256":t.prepared_sha256,"training_sha256":t.training_sha256,"selected_training_rows_sha256":t.selected_training_rows_sha256,"preprocessing_sha256":t.preprocessing_sha256,"cut":t.cut,"encoder_spec_sha256":ASSEMBLY_SHA256,"numerical_backend":NATIVE_BACKEND,"output_sha256":sha256_hex(b"synthetic context"),"partition":"meta_train","final_test_targets_used":false})
    }
    #[test]
    fn accepted_receipt_refuses_wrong_rows_old128_and_classification_aliases() -> Result<()> {
        let expected = sha256_hex(b"synthetic context");
        validate_native_receipt(&native_meta(), &training(), &expected)?;
        for (key, value) in [
            (
                "selected_training_rows_sha256",
                serde_json::json!(sha256_hex(b"heldout rows")),
            ),
            ("dataset_dimensions", serde_json::json!(128)),
            ("task", serde_json::json!("binary")),
            ("partition", serde_json::json!("meta_test")),
            ("numerical_backend", serde_json::json!("CPU replacement")),
            ("qualification", serde_json::json!(true)),
            ("final_test_targets_used", serde_json::Value::Null),
        ] {
            let mut m = native_meta();
            m[key] = value;
            assert!(
                validate_native_receipt(&m, &training(), &expected).is_err(),
                "accepted altered {key}"
            );
        }
        Ok(())
    }
    #[test]
    fn experimental_binary_and_multiclass_contexts_are_bitwise_label_rename_invariant() -> Result<()>
    {
        let rows = (0..4)
            .map(|i| (0..128).map(|j| (i * 128 + j) as f32 / 128.).collect())
            .collect::<Vec<Vec<f32>>>();
        for (labels, renamed) in [
            (["a", "b", "a", "b"], ["z", "-100", "z", "-100"]),
            (["a", "b", "c", "b"], ["third", "first", "second", "first"]),
        ] {
            let before = classification_context(
                training(),
                &row_ids(),
                &[0.; 512],
                &[1.; 1024],
                &rows,
                &labels.map(String::from),
            )?;
            let after = classification_context(
                training(),
                &row_ids(),
                &[0.; 512],
                &[1.; 1024],
                &rows,
                &renamed.map(String::from),
            )?;
            assert_eq!(before.values().len(), 2048);
            assert_eq!(before.sha256(), after.sha256());
            assert_eq!(
                before
                    .values()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                after
                    .values()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert!(!before.policy_compatible);
            assert_eq!(before.lineage, CLASSIFICATION_LINEAGE);
        }
        Ok(())
    }
    #[test]
    fn experimental_pool_rejects_changed_training_selection_and_missing_classes() {
        let labels = ["a", "b", "a", "b"].map(String::from);
        let rows = vec![vec![1.; 128]; 4];
        assert!(classification_context(
            training(),
            &[
                row_ids()[0].clone(),
                row_ids()[1].clone(),
                row_ids()[2].clone(),
                sha256_hex(b"other row")
            ],
            &[0.; 512],
            &[1.; 1024],
            &rows,
            &labels
        )
        .is_err());
        assert!(classification_context(
            training(),
            &row_ids(),
            &[0.; 512],
            &[1.; 1024],
            &rows,
            &["a", "a", "a", "a"].map(String::from)
        )
        .is_err());
        assert!(classification_context(
            training(),
            &row_ids(),
            &[0.; 128],
            &[1.; 1024],
            &rows,
            &labels
        )
        .is_err());
    }
}
