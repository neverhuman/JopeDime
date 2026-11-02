use super::*;

#[derive(Serialize)]
struct LegacyCandidate<'a> {
    semantic_sha256: &'a str,
    canonical_genome: &'a str,
}

#[derive(Serialize)]
struct LegacyShard<'a> {
    schema: &'static str,
    task: &'static str,
    dataset_hash: &'a str,
    train_csv: String,
    test_csv: String,
    train_sha256: String,
    test_sha256: String,
    candidates: Vec<LegacyCandidate<'a>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct LegacyReceipt {
    schema: String,
    task: String,
    dataset_hash: String,
    semantic_sha256: String,
    predictions_sha256: Option<String>,
    predictions_path: Option<String>,
    prediction_count: Option<usize>,
    valid: bool,
    failure: Option<String>,
    failure_message: Option<String>,
    runtime_seconds: f64,
    train_sha256: String,
    test_sha256: String,
    manifest_sha256: String,
}

#[derive(Debug, Deserialize)]
struct LegacyDone {
    manifest_sha256: String,
}

#[derive(Debug, Clone)]
pub struct CorpusEvalAdapter {
    evaluator: EvaluatorFreezeV1,
    work_root: PathBuf,
    timeout_seconds: u64,
}

impl CorpusEvalAdapter {
    pub fn new(
        evaluator: EvaluatorFreezeV1,
        work_root: impl Into<PathBuf>,
        timeout_seconds: u64,
    ) -> Result<Self, String> {
        evaluator.validate()?;
        if timeout_seconds == 0 {
            return Err("evaluator timeout must be positive".into());
        }
        Ok(Self {
            evaluator,
            work_root: work_root.into(),
            timeout_seconds,
        })
    }

    pub fn evaluate_search_split(
        &self,
        dataset: &DatasetTableV1,
        split: &RowSplitV1,
        candidates: &[GenomeV1],
    ) -> Result<Vec<EvaluatorOutcomeV1>, String> {
        dataset.validate()?;
        if candidates.is_empty() || candidates.len() > 128 {
            return Err("evaluator batches require 1 through 128 candidates".into());
        }
        let expected_rows: BTreeSet<_> = dataset
            .rows
            .iter()
            .map(|row| row.split.stable_row_id.as_str())
            .collect();
        let train: BTreeSet<_> = split.train_row_ids.iter().map(String::as_str).collect();
        let validation: BTreeSet<_> = split
            .validation_row_ids
            .iter()
            .map(String::as_str)
            .collect();
        if train.len() != split.train_row_ids.len()
            || validation.len() != split.validation_row_ids.len()
            || !train.is_disjoint(&validation)
            || train.union(&validation).copied().collect::<BTreeSet<_>>() != expected_rows
        {
            return Err("evaluator split is not a complete dataset partition".into());
        }
        let train_bytes = dataset.csv_for(&split.train_row_ids)?;
        let validation_bytes = dataset.csv_for(&split.validation_row_ids)?;
        self.evaluate_bytes(
            dataset.task,
            &dataset.dataset_sha256,
            &dataset.training_sha256,
            &canonical_sha256(split)?,
            split.split_index,
            &split.validation_row_ids,
            &train_bytes,
            &validation_bytes,
            candidates,
        )
    }

    /// The selection freeze is validated before test_csv is opened. There is
    /// no adapter API that evaluates sealed bytes without that freeze.
    pub fn evaluate_sealed_test(
        &self,
        training: &DatasetTableV1,
        sealed: &SealedTestTableV1,
        selection: &Cv7SelectionFreezeV1,
    ) -> Result<EvaluatorOutcomeV1, String> {
        training.validate()?;
        sealed.validate_metadata()?;
        let winner = selection.winner()?.candidate.clone();
        let selection_sha256 = ensure_selection_before_test(
            selection,
            &training.dataset_sha256,
            &winner.semantic_sha256,
        )?;
        if sealed.dataset_sha256 != training.dataset_sha256 || winner.task != training.task {
            return Err("sealed-test table does not match the selected training dataset".into());
        }
        // First access to test.csv occurs only after the selection checks above.
        let test_bytes = fs::read(&sealed.test_csv_path)
            .map_err(|error| format!("read sealed test CSV: {error}"))?;
        if raw_sha256(&test_bytes) != sealed.test_sha256 {
            return Err("sealed test CSV changed after its hash freeze".into());
        }
        let split_sha256 = raw_sha256(
            format!("sealed-test-v1|{selection_sha256}|{}", sealed.test_sha256).as_bytes(),
        );
        let mut outcomes = self.evaluate_bytes(
            training.task,
            &training.dataset_sha256,
            &training.training_sha256,
            &split_sha256,
            0,
            &sealed.test_row_ids,
            &training.csv_bytes(),
            &test_bytes,
            &[winner],
        )?;
        Ok(outcomes.remove(0))
    }

    /// Batch form used by protocol-specific sealed evaluators after they have
    /// atomically published their own selection-set claim. The caller supplies
    /// that immutable claim identity; test bytes are not opened until every
    /// metadata and candidate binding below has passed.
    pub fn evaluate_claimed_sealed_batch(
        &self,
        training: &DatasetTableV1,
        sealed: &SealedTestTableV1,
        claim_sha256: &str,
        candidates: &[GenomeV1],
    ) -> Result<Vec<EvaluatorOutcomeV1>, String> {
        training.validate()?;
        sealed.validate_metadata()?;
        crate::search_v3::validate_sha256("sealed access claim", claim_sha256)?;
        if sealed.dataset_sha256 != training.dataset_sha256
            || candidates.is_empty()
            || candidates.len() > 128
        {
            return Err("claimed sealed batch has invalid dataset or candidate coverage".into());
        }
        for candidate in candidates {
            candidate.validate()?;
            if candidate.task != training.task {
                return Err("claimed sealed batch contains a task-incompatible candidate".into());
            }
        }
        // First access occurs after all claim, dataset, and candidate checks.
        let test_bytes = fs::read(&sealed.test_csv_path)
            .map_err(|error| format!("read sealed test CSV: {error}"))?;
        if raw_sha256(&test_bytes) != sealed.test_sha256 {
            return Err("sealed test CSV changed after its hash freeze".into());
        }
        let split_sha256 = raw_sha256(
            format!("claimed-sealed-v1|{claim_sha256}|{}", sealed.test_sha256).as_bytes(),
        );
        self.evaluate_bytes(
            training.task,
            &training.dataset_sha256,
            &training.training_sha256,
            &split_sha256,
            0,
            &sealed.test_row_ids,
            &training.csv_bytes(),
            &test_bytes,
            candidates,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_bytes(
        &self,
        task: GpTaskV1,
        dataset_sha256: &str,
        training_sha256: &str,
        split_sha256: &str,
        fold_index: u32,
        validation_row_ids: &[String],
        train_bytes: &[u8],
        validation_bytes: &[u8],
        candidates: &[GenomeV1],
    ) -> Result<Vec<EvaluatorOutcomeV1>, String> {
        self.evaluator.verify_artifact()?;
        for candidate in candidates {
            candidate.validate()?;
            if candidate.task != task {
                return Err("evaluator batch contains a task-incompatible genome".into());
            }
        }
        let batch_sha256 = raw_sha256(
            candidates
                .iter()
                .map(|candidate| candidate.semantic_sha256.as_str())
                .collect::<Vec<_>>()
                .join("|")
                .as_bytes(),
        );
        let invocation = raw_sha256(
            format!(
                "corpus-eval-adapter-v1|{dataset_sha256}|{split_sha256}|{batch_sha256}|{}",
                self.evaluator.artifact.sha256
            )
            .as_bytes(),
        );
        let directory = self.work_root.join(&invocation[..2]).join(&invocation);
        fs::create_dir_all(&directory)
            .map_err(|error| format!("create evaluator work directory: {error}"))?;
        let train_path = directory.join("train.csv");
        let validation_path = directory.join("validation.csv");
        write_exact(&train_path, train_bytes)?;
        write_exact(&validation_path, validation_bytes)?;
        let legacy_dataset =
            raw_sha256(format!("{dataset_sha256}|{split_sha256}|{batch_sha256}").as_bytes());
        let shard = LegacyShard {
            schema: "JopeFlywheelCorpusEvalShard.v1",
            task: task.as_str(),
            dataset_hash: &legacy_dataset,
            train_csv: train_path.to_string_lossy().into_owned(),
            test_csv: validation_path.to_string_lossy().into_owned(),
            train_sha256: raw_sha256(train_bytes),
            test_sha256: raw_sha256(validation_bytes),
            candidates: candidates
                .iter()
                .map(|candidate| LegacyCandidate {
                    semantic_sha256: &candidate.semantic_sha256,
                    canonical_genome: &candidate.canonical_genome,
                })
                .collect(),
        };
        // serde_json's compact, struct-field-order encoding is the exact
        // legacy shard wire format consumed by corpus_eval.
        let manifest_bytes = serde_json::to_vec(&shard)
            .map_err(|error| format!("encode legacy evaluator shard: {error}"))?;
        let manifest_sha256 = raw_sha256(&manifest_bytes);
        let manifest_path = directory.join("manifest.json");
        write_exact(&manifest_path, &manifest_bytes)?;
        let output_dir = directory.join("output");
        let done_path = output_dir.join(format!("{legacy_dataset}.done"));
        if done_path.is_file() {
            let done: LegacyDone = serde_json::from_slice(
                &fs::read(&done_path)
                    .map_err(|error| format!("read evaluator done file: {error}"))?,
            )
            .map_err(|error| format!("decode evaluator done file: {error}"))?;
            if done.manifest_sha256 != manifest_sha256 {
                return Err("stale evaluator completion collides with this invocation".into());
            }
        } else {
            if output_dir.is_dir()
                && fs::read_dir(&output_dir)
                    .map_err(|error| format!("scan evaluator output: {error}"))?
                    .next()
                    .is_some()
            {
                return Err("incomplete evaluator output collision".into());
            }
            let timeout_seconds = self.timeout_seconds.to_string();
            let output = Command::new(&self.evaluator.artifact.path)
                .args([
                    "--manifest",
                    manifest_path
                        .to_str()
                        .ok_or_else(|| "manifest path is not UTF-8".to_string())?,
                    "--output-dir",
                    output_dir
                        .to_str()
                        .ok_or_else(|| "output path is not UTF-8".to_string())?,
                    "--threads",
                    "1",
                    "--timeout-seconds",
                    &timeout_seconds,
                ])
                .output()
                .map_err(|error| format!("launch immutable corpus evaluator: {error}"))?;
            if !output.status.success() {
                return Err(format!(
                    "immutable corpus evaluator failed: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                ));
            }
        }
        self.read_outcomes(
            task,
            dataset_sha256,
            training_sha256,
            split_sha256,
            fold_index,
            validation_row_ids,
            candidates,
            &legacy_dataset,
            &manifest_sha256,
            &shard.train_sha256,
            &shard.test_sha256,
            &output_dir,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn read_outcomes(
        &self,
        task: GpTaskV1,
        dataset_sha256: &str,
        training_sha256: &str,
        split_sha256: &str,
        fold_index: u32,
        validation_row_ids: &[String],
        candidates: &[GenomeV1],
        legacy_dataset: &str,
        manifest_sha256: &str,
        train_sha256: &str,
        test_sha256: &str,
        output_dir: &Path,
    ) -> Result<Vec<EvaluatorOutcomeV1>, String> {
        let receipt_path = output_dir.join(format!("{legacy_dataset}.receipts.jsonl"));
        let receipt_bytes = fs::read(&receipt_path)
            .map_err(|error| format!("read legacy evaluator receipts: {error}"))?;
        let receipts: Vec<LegacyReceipt> = String::from_utf8(receipt_bytes)
            .map_err(|error| format!("legacy receipt is not UTF-8: {error}"))?
            .lines()
            .map(|line| {
                serde_json::from_str(line)
                    .map_err(|error| format!("decode legacy evaluator receipt: {error}"))
            })
            .collect::<Result<_, _>>()?;
        if receipts.len() != candidates.len() {
            return Err("legacy evaluator returned incomplete candidate coverage".into());
        }
        let by_semantic: BTreeMap<_, _> = receipts
            .iter()
            .map(|receipt| (receipt.semantic_sha256.as_str(), receipt))
            .collect();
        if by_semantic.len() != receipts.len() {
            return Err("legacy evaluator emitted duplicate semantic receipts".into());
        }
        let mut outcomes = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let receipt = by_semantic
                .get(candidate.semantic_sha256.as_str())
                .ok_or_else(|| "legacy evaluator omitted a candidate".to_string())?;
            if receipt.schema != "JopeFlywheelCorpusEvalReceipt.v1"
                || receipt.task != task.as_str()
                || receipt.dataset_hash != legacy_dataset
                || receipt.manifest_sha256 != manifest_sha256
                || receipt.train_sha256 != train_sha256
                || receipt.test_sha256 != test_sha256
                || !receipt.runtime_seconds.is_finite()
                || receipt.runtime_seconds < 0.0
            {
                return Err("legacy evaluator returned stale or malformed evidence".into());
            }
            let (bits, valid, invalid_reason) = if receipt.valid {
                let relative = receipt
                    .predictions_path
                    .as_deref()
                    .ok_or_else(|| "valid receipt lacks predictions_path".to_string())?;
                let expected = receipt
                    .predictions_sha256
                    .as_deref()
                    .ok_or_else(|| "valid receipt lacks predictions_sha256".to_string())?;
                let bytes = fs::read(output_dir.join(relative))
                    .map_err(|error| format!("read legacy predictions: {error}"))?;
                if raw_sha256(&bytes) != expected
                    || bytes.len() % 8 != 0
                    || receipt.prediction_count != Some(validation_row_ids.len())
                    || bytes.len() / 8 != validation_row_ids.len()
                {
                    return Err("legacy prediction evidence is corrupt or row-misaligned".into());
                }
                let bits = bytes
                    .chunks_exact(8)
                    .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect::<Vec<_>>();
                if bits.iter().any(|bits| !f64::from_bits(*bits).is_finite()) {
                    return Err("valid legacy predictions contain a nonfinite value".into());
                }
                (bits, true, None)
            } else {
                let reason = receipt
                    .failure_message
                    .clone()
                    .or_else(|| receipt.failure.clone())
                    .filter(|reason| !reason.trim().is_empty())
                    .ok_or_else(|| "invalid legacy receipt lacks a failure reason".to_string())?;
                (
                    vec![0.0f64.to_bits(); validation_row_ids.len()],
                    false,
                    Some(reason),
                )
            };
            let fit_sha256 = raw_sha256(
                format!(
                    "fit-v1|{training_sha256}|{split_sha256}|{}|{}",
                    candidate.semantic_sha256, self.evaluator.artifact.sha256
                )
                .as_bytes(),
            );
            let prediction = PredictionObjectV1 {
                schema: PredictionObjectV1::SCHEMA.into(),
                training_view_sha256: training_sha256.into(),
                fold_plan_sha256: split_sha256.into(),
                fold_index,
                candidate_semantic_sha256: candidate.semantic_sha256.clone(),
                evaluator_sha256: self.evaluator.artifact.sha256.clone(),
                fit_sha256,
                math_policy_sha256: self.evaluator.math_policy_sha256.clone(),
                backend_policy_sha256: self.evaluator.backend_policy_sha256.clone(),
                validation_row_ids: validation_row_ids.to_vec(),
                prediction_shape: vec![validation_row_ids.len()],
                prediction_f64_bits: bits,
                valid,
                invalid_reason,
            };
            let key = PredictionCacheKeyV1 {
                schema: PredictionCacheKeyV1::SCHEMA.into(),
                dataset_sha256: dataset_sha256.into(),
                training_sha256: training_sha256.into(),
                split_sha256: split_sha256.into(),
                candidate_semantic_sha256: candidate.semantic_sha256.clone(),
                evaluator_sha256: self.evaluator.artifact.sha256.clone(),
                math_policy_sha256: self.evaluator.math_policy_sha256.clone(),
                backend_policy_sha256: self.evaluator.backend_policy_sha256.clone(),
            };
            let outcome = EvaluatorOutcomeV1 {
                key,
                prediction,
                runtime: RuntimeObservationV1 {
                    cpu_seconds: receipt.runtime_seconds,
                    winner_latency_seconds: receipt.runtime_seconds,
                    timed_out: receipt.failure.as_deref() == Some("timeout"),
                },
                legacy_receipt_sha256: canonical_sha256(*receipt)?,
            };
            outcome.validate()?;
            outcomes.push(outcome);
        }
        Ok(outcomes)
    }
}

fn write_exact(path: &Path, bytes: &[u8]) -> Result<(), String> {
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            file.write_all(bytes)
                .map_err(|error| format!("write {}: {error}", path.display()))?;
            file.sync_all()
                .map_err(|error| format!("sync {}: {error}", path.display()))
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let present = fs::read(path)
                .map_err(|read| format!("read existing {}: {read}", path.display()))?;
            if present == bytes {
                Ok(())
            } else {
                Err(format!(
                    "immutable evaluator input collision at {}",
                    path.display()
                ))
            }
        }
        Err(error) => Err(format!("create {}: {error}", path.display())),
    }
}
