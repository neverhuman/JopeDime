use super::export::{export_genome, observe_genome};
use super::publish::{publish_corpus, safe_relative_path, sha256_bytes, validate_sha256};
use super::*;

#[cfg(test)]
#[cfg(test)]
pub(super) fn read_and_materialize<F>(
    archive_root: &Path,
    output_root: &Path,
    after_open: F,
) -> Result<PathBuf>
where
    F: FnOnce(),
{
    read_and_materialize_filtered(archive_root, output_root, None, after_open)
}

pub(super) fn read_and_materialize_filtered<F>(
    archive_root: &Path,
    output_root: &Path,
    cohort: Option<&str>,
    after_open: F,
) -> Result<PathBuf>
where
    F: FnOnce(),
{
    let manifest_path = archive_root.join(ARCHIVE_MANIFEST);
    let manifest_before =
        fs::read(&manifest_path).with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest: ArchiveManifest =
        serde_json::from_slice(&manifest_before).context("parse archive manifest")?;
    if !manifest.schema.contains("regression-analysis-archive.v1") {
        bail!("unsupported archive schema '{}'", manifest.schema);
    }
    validate_sha256(
        &manifest.all_evaluations_jsonl_sha256,
        "archive JSONL SHA-256",
    )?;
    let relative = safe_relative_path(&manifest.all_evaluations_jsonl)?;
    let ledger_path = archive_root.join(relative);
    let ledger_file =
        File::open(&ledger_path).with_context(|| format!("open {}", ledger_path.display()))?;
    let mut ledger = BufReader::new(HashingReader::new(ledger_file));
    after_open();

    let aggregated = aggregate_records(&mut ledger);
    // Complete the digest even after a parse error: changed input remains
    // retryable, while malformed input with a matching pin remains fatal.
    io::copy(&mut ledger, &mut io::sink())
        .with_context(|| format!("finish reading {}", ledger_path.display()))?;
    let actual_ledger_sha256 = ledger.into_inner().into_sha256();
    if actual_ledger_sha256 != manifest.all_evaluations_jsonl_sha256 {
        return Err(ArchiveChanged("master JSONL digest differs from manifest").into());
    }
    let (mut aggregates, _, record_count) = aggregated?;
    if record_count != manifest.evaluation_records {
        return Err(ArchiveChanged("master JSONL record count differs from manifest").into());
    }
    let manifest_after =
        fs::read(&manifest_path).with_context(|| format!("re-read {}", manifest_path.display()))?;
    if manifest_after != manifest_before {
        return Err(ArchiveChanged("manifest bytes changed").into());
    }
    if let Some(cohort) = cohort {
        aggregates.retain(|key, _| key.cohort == cohort);
        if aggregates.is_empty() {
            bail!("archive has no score examples for cohort {cohort:?}");
        }
    }
    let mut observed = ObservedSchema::default();
    for aggregate in aggregates.values() {
        observe_genome(&aggregate.genome, &mut observed);
    }
    publish_corpus(
        output_root,
        &manifest_before,
        &manifest,
        aggregates,
        observed,
        cohort,
    )
}

pub(super) fn aggregate_records<R: BufRead>(
    reader: &mut R,
) -> Result<(BTreeMap<ExampleKey, Aggregate>, ObservedSchema, usize)> {
    let mut aggregates = BTreeMap::<ExampleKey, Aggregate>::new();
    let mut observed = ObservedSchema::default();
    let mut records = 0usize;
    for (index, raw) in reader.split(b'\n').enumerate() {
        let line = index + 1;
        let raw = raw.with_context(|| format!("read archive record at line {line}"))?;
        if raw.is_empty() {
            continue;
        }
        records += 1;
        let record: Value = serde_json::from_slice(&raw)
            .with_context(|| format!("malformed archive record at line {line}"))?;
        let object = record
            .as_object()
            .with_context(|| format!("archive line {line} is not an object"))?;
        let required_string = |name: &str| -> Result<&str> {
            object
                .get(name)
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .with_context(|| format!("archive line {line} lacks string field '{name}'"))
        };
        let cohort = required_string("cohort")?.to_string();
        let dataset = required_string("dataset")?.to_string();
        let semantic_sha256 = required_string("semantic_sha256")?.to_string();
        validate_sha256(&semantic_sha256, "semantic SHA-256")
            .with_context(|| format!("archive line {line}"))?;
        let canonical_genome = required_string("canonical_genome")?.to_string();
        if sha256_bytes(canonical_genome.as_bytes()) != semantic_sha256 {
            bail!("archive line {line} semantic SHA-256 does not bind canonical genome");
        }
        let parsed = GenomeIr::parse(&canonical_genome)
            .with_context(|| format!("archive line {line} genome does not parse"))?;
        if parsed.canonical() != canonical_genome {
            bail!("archive line {line} genome is not canonical under the current grammar");
        }
        let genome = export_genome(&parsed)
            .with_context(|| format!("archive line {line} genome export failed"))?;
        observe_genome(&genome, &mut observed);

        let valid = object
            .get("valid")
            .and_then(Value::as_bool)
            .with_context(|| format!("archive line {line} lacks boolean field 'valid'"))?;
        let rmse = if valid {
            let value = object
                .get("rmse")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite() && *value >= 0.0)
                .with_context(|| format!("archive line {line} has invalid valid RMSE"))?;
            Some(value)
        } else {
            None
        };
        let runtime = match object.get("runtime_seconds") {
            None | Some(Value::Null) => None,
            Some(value) => Some(
                value
                    .as_f64()
                    .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
                    .with_context(|| format!("archive line {line} has invalid runtime"))?,
            ),
        };
        let failure_class = if valid {
            None
        } else {
            object
                .get("failure")
                .and_then(Value::as_object)
                .and_then(|failure| failure.get("class"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| Some("unknown".to_string()))
        };
        let key = ExampleKey {
            cohort,
            dataset,
            semantic_sha256,
        };
        let provenance = provenance_for(object, line);
        match aggregates.get_mut(&key) {
            Some(aggregate) => {
                if aggregate.canonical_genome != canonical_genome {
                    bail!(
                        "semantic key ({}, {}, {}) has conflicting canonical genomes",
                        key.cohort,
                        key.dataset,
                        key.semantic_sha256
                    );
                }
                if let (Some(left), Some(right)) = (aggregate.valid_rmse, rmse) {
                    if left.to_bits() != right.to_bits() {
                        bail!(
                            "semantic key ({}, {}, {}) has conflicting valid RMSE values",
                            key.cohort,
                            key.dataset,
                            key.semantic_sha256
                        );
                    }
                }
                if aggregate.valid_rmse.is_none() {
                    aggregate.valid_rmse = rmse;
                }
                aggregate.valid_count += usize::from(valid);
                aggregate.invalid_count += usize::from(!valid);
                if let Some(seconds) = runtime {
                    aggregate.runtimes.push(seconds);
                }
                if let Some(class) = failure_class {
                    *aggregate.failure_classes.entry(class).or_default() += 1;
                }
                aggregate.provenance.push(provenance);
            }
            None => {
                let mut failure_classes = BTreeMap::new();
                if let Some(class) = failure_class {
                    failure_classes.insert(class, 1);
                }
                aggregates.insert(
                    key,
                    Aggregate {
                        canonical_genome,
                        genome,
                        valid_rmse: rmse,
                        valid_count: usize::from(valid),
                        invalid_count: usize::from(!valid),
                        runtimes: runtime.into_iter().collect(),
                        failure_classes,
                        provenance: vec![provenance],
                    },
                );
            }
        }
    }
    if records == 0 {
        bail!("archive master JSONL is empty");
    }
    Ok((aggregates, observed, records))
}

fn provenance_for(record: &Map<String, Value>, source_line: usize) -> Value {
    let mut provenance = Map::new();
    provenance.insert("source_line".into(), json!(source_line));
    for key in [
        "source_campaign_id",
        "source_campaign_root",
        "source_campaign_order",
        "source_ledger_line",
        "run_id",
        "generation",
        "slot",
        "logical_evaluation",
        "fresh_evaluation",
        "cached",
        "node",
        "scq_job_id",
        "attempt",
        "receipt_file",
        "receipt_sha256",
        "checkpoint_file",
        "checkpoint_sha256",
        "binary_sha256",
        "train_sha256",
        "test_sha256",
        "predictions_sha256",
    ] {
        if let Some(value) = record.get(key) {
            provenance.insert(key.to_string(), value.clone());
        }
    }
    Value::Object(provenance)
}
