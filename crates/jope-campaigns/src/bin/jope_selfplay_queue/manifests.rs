use super::*;

pub(crate) fn valid_micro_waves(
    root: &Path,
    checkpoint_sha256: &str,
    epoch: &str,
) -> Result<Vec<MicroWave>> {
    let directory = root.join("waves");
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut waves = Vec::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        let Some(wave) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !valid_wave_id(wave) {
            continue;
        }
        let summary_path = path.join("summary.json");
        if !summary_path.is_file() {
            continue;
        }
        let summary = match read_json(&summary_path) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let valid = summary.get("wave_kind").and_then(Value::as_str) == Some(MICRO_WAVE_KIND)
            && summary.get("wave").and_then(Value::as_str) == Some(wave)
            && summary.get("checkpoint_epoch").and_then(Value::as_str) == Some(epoch)
            && summary
                .get("checkpoint_sha256")
                .or_else(|| summary.get("model_checkpoint_sha256"))
                .and_then(Value::as_str)
                == Some(checkpoint_sha256)
            // Shape check driven by the summary's own task_datasets map: every
            // declared task carries exactly one 64-dataset lane, the mandatory
            // regression/binary backbone is present, and the dataset total is
            // the lane sum. The previous literal (exactly 128 = 64+64) silently
            // dropped valid 192/256 waves -- a whole wave of evaluation work
            // vanishing with no error is exactly the failure class epoch-06
            // lane 09 warned about.
            && summary
                .get("task_datasets")
                .and_then(Value::as_object)
                .is_some_and(|tasks| {
                    !tasks.is_empty()
                        && tasks.values().all(|count| count.as_u64() == Some(64))
                        && tasks.contains_key("regression")
                        && tasks.contains_key("binary")
                        && summary.get("datasets").and_then(Value::as_u64)
                            == Some(64 * tasks.len() as u64)
                })
            && summary
                .get("candidates_per_dataset")
                .and_then(Value::as_u64)
                == Some(256)
            && summary
                .get("max_candidates_per_manifest")
                .and_then(Value::as_u64)
                .is_some_and(|count| count <= 100)
            && wave_manifest_directory(root, wave).is_dir();
        if valid {
            waves.push(MicroWave {
                wave: wave.to_owned(),
                summary_sha256: sha256_file(&summary_path)?,
            });
        }
    }
    waves.sort_by(|left, right| left.wave.cmp(&right.wave));
    Ok(waves)
}

pub(crate) fn current_admission_waves(
    root: &Path,
    waves: &[MicroWave],
    claims: &HashMap<String, Claim>,
    rejected: &BTreeSet<PathBuf>,
) -> Result<Vec<AdmissionWave>> {
    let claimed = claims
        .values()
        .map(|claim| claim.manifest.as_path())
        .collect::<BTreeSet<_>>();
    let mut admission = Vec::new();
    for wave in waves {
        let directory = wave_manifest_directory(root, &wave.wave);
        let manifest_paths = fs::read_dir(&directory)?
            .filter_map(|entry| entry.ok().map(|value| value.path()))
            .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
            .collect::<Vec<_>>();
        let manifest_count = manifest_paths.len();
        let unclaimed_paths = manifest_paths
            .iter()
            .filter(|path| !claimed.contains(path.as_path()))
            .filter(|path| !rejected.contains(*path))
            .collect::<Vec<_>>();
        let unclaimed_manifests = unclaimed_paths.len();
        let candidate_evaluation_count = unclaimed_paths
            .into_iter()
            .map(|path| manifest_candidate_count(path))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .try_fold(0_u64, u64::checked_add)
            .context("wave candidate-evaluation count overflow")?;
        if unclaimed_manifests > 0 {
            admission.push(AdmissionWave {
                wave: wave.wave.clone(),
                summary_sha256: wave.summary_sha256.clone(),
                manifest_count,
                unclaimed_manifests,
                candidate_evaluation_count,
                complete: unclaimed_manifests == manifest_count,
            });
        }
    }
    Ok(admission)
}

pub(crate) fn manifest_candidate_count(path: &Path) -> Result<u64> {
    let value = read_json(path)?;
    let count = value
        .get("candidates")
        .and_then(Value::as_array)
        .map(Vec::len)
        .context("wave manifest lacks candidate-evaluation work")?;
    ensure!(count > 0, "wave manifest has no candidate-evaluation work");
    Ok(count as u64)
}

/// Keep the newest two byte-complete waves outside scheduler admission. Older
/// work remains oldest-first, so reserve capacity cannot be consumed before a
/// newer keeper acknowledgment replenishes it.
pub(crate) fn partition_protected_reserve(
    waves: Vec<AdmissionWave>,
) -> (Vec<AdmissionWave>, Vec<AdmissionWave>) {
    let protected = waves
        .iter()
        .rev()
        .filter(|wave| wave.complete)
        .take(2)
        .map(|wave| wave.wave.clone())
        .collect::<BTreeSet<_>>();
    let (admission, reserve): (Vec<_>, Vec<_>) = waves
        .into_iter()
        .partition(|wave| !protected.contains(&wave.wave));
    (admission, reserve)
}

pub(crate) fn wave_manifest_directory(root: &Path, wave: &str) -> PathBuf {
    let canonical = root.join("waves").join(wave).join("manifests");
    if canonical.is_dir() {
        canonical
    } else {
        // Read-only compatibility for waves created before the immutable
        // wave bundle owned its manifest directory.
        root.join("manifests").join(wave)
    }
}

pub(crate) fn valid_wave_id(value: &str) -> bool {
    value
        .strip_prefix("strict-wave-")
        .is_some_and(|suffix| suffix.len() == 6 && suffix.bytes().all(|byte| byte.is_ascii_digit()))
}

pub(crate) fn manifest_wave(path: &Path) -> Result<String> {
    let value = read_json(path)?;
    let wave = value
        .get("strict_wave")
        .and_then(Value::as_str)
        .context("manifest lacks strict_wave")?;
    if !valid_wave_id(wave) {
        bail!("manifest strict_wave is malformed");
    }
    Ok(wave.to_owned())
}

/// Per-task authority metric (R1/D2); must agree with
/// `jope_dime_regressor::task::TaskKind::primary_metric`.
pub(crate) fn expected_manifest_metric(task: &str) -> Option<&'static str> {
    Task::parse(task).map(Task::primary_metric)
}

pub(crate) fn validate_manifest(
    path: &Path,
    checkpoint_sha256: &str,
    epoch: &str,
    wave: &str,
) -> Result<ValidManifest> {
    let value = read_json(path)?;
    // Timeseries shards are published as ...Shard.v2 because the evaluator
    // demands a reduction pin that a v1 shard is forbidden to carry; every
    // other task stays v1. Admitting only v1 would refuse every timeseries
    // manifest at the queue even though the builder produced it correctly.
    if !matches!(
        value.get("schema").and_then(Value::as_str),
        Some(MANIFEST_SCHEMA) | Some(MANIFEST_SCHEMA_V2)
    ) {
        bail!("manifest schema is outside self-play");
    }
    if value
        .get("evaluation_lane")
        .and_then(Value::as_str)
        .is_some_and(|lane| {
            matches!(
                lane,
                "checkpoint_gate" | "sealed_audit" | "evaluator_canary"
            )
        })
    {
        bail!("protected evaluation lane is not adaptive admission");
    }
    if value.get("strict_wave").and_then(Value::as_str) != Some(wave)
        || value.get("checkpoint_epoch").and_then(Value::as_str) != Some(epoch)
        || value.get("checkpoint_sha256").and_then(Value::as_str) != Some(checkpoint_sha256)
    {
        bail!("manifest is not from the newest current-checkpoint wave");
    }
    for field in [
        "checkpoint_sha256",
        "catalog_sha256",
        "protocol_sha256",
        "train_sha256",
        "test_sha256",
    ] {
        if !value
            .get(field)
            .and_then(Value::as_str)
            .is_some_and(is_sha256)
        {
            bail!("manifest lacks immutable {field}");
        }
    }
    if protocol_for_manifest(&value)? != value["protocol_sha256"].as_str().unwrap_or_default() {
        bail!("manifest evaluator protocol hash drift");
    }
    let generated_at_ms = value
        .get("generated_at_ms")
        .and_then(Value::as_u64)
        .context("manifest lacks generation timestamp")? as u128;
    let task = value
        .get("task")
        .and_then(Value::as_str)
        .filter(|task| matches!(*task, "regression" | "binary" | "timeseries" | "multiclass"))
        .context("manifest task is invalid")?;
    let dataset_hash = value
        .get("dataset_hash")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .context("manifest dataset identity is absent")?;
    if !matches!(
        value.get("micro_lane").and_then(Value::as_str),
        Some("stable_anchor" | "least_observed" | "hard_uncertain")
    ) {
        bail!("manifest is not assigned to a deterministic micro-wave lane");
    }
    let authority = value
        .get("label_authority")
        .context("manifest authority is absent")?;
    let expected_metric = expected_manifest_metric(task).expect("task filtered above");
    if authority.get("score_tier").and_then(Value::as_str) != Some("test_exact")
        || authority.get("metric").and_then(Value::as_str) != Some(expected_metric)
        || authority.get("test_adaptive").and_then(Value::as_bool) != Some(true)
        || authority.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
        || authority.get("generation").and_then(Value::as_str) != Some(wave)
        || authority
            .get("model_checkpoint_sha256")
            .and_then(Value::as_str)
            != Some(checkpoint_sha256)
    {
        bail!("manifest authority is not adaptive test-exact");
    }
    let protocol = value["protocol_sha256"]
        .as_str()
        .context("protocol hash absent")?;
    let candidates = value
        .get("candidates")
        .and_then(Value::as_array)
        .filter(|rows| !rows.is_empty() && rows.len() <= 100)
        .context("manifest candidate count is outside 1..=100")?;
    let mut semantics = BTreeSet::new();
    let mut pairs = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let genome = candidate
            .get("canonical_genome")
            .and_then(Value::as_str)
            .context("candidate genome is absent")?;
        let semantic = candidate
            .get("semantic_sha256")
            .and_then(Value::as_str)
            .context("candidate semantic hash is absent")?;
        if sha256_bytes(genome.as_bytes()) != semantic {
            bail!("candidate semantic hash drift");
        }
        if !semantics.insert(semantic.to_owned()) {
            bail!("manifest duplicates a semantic candidate");
        }
        let is_dl_proposal = candidate.get("slot_group").and_then(Value::as_str)
            == Some("dl_proposal")
            || candidate.get("source").and_then(Value::as_str)
                == Some("current_checkpoint_dl_proposal");
        if is_dl_proposal
            && candidate
                .get("model_checkpoint_sha256")
                .and_then(Value::as_str)
                != Some(checkpoint_sha256)
        {
            bail!("DL proposal checkpoint provenance drift");
        }
        pairs.push(PairKey {
            task: task.to_owned(),
            dataset_hash: dataset_hash.to_owned(),
            semantic_sha256: semantic.to_owned(),
            protocol_sha256: protocol.to_owned(),
        });
    }
    let sha256 = sha256_file(path)?;
    Ok(ValidManifest {
        path: path.to_owned(),
        sha256,
        generated_at_ms,
        pairs,
        value,
    })
}

pub(crate) fn preflight_manifest(
    registry: &GeneratedOperatorRegistry,
    manifest: &ValidManifest,
    bounded_screening_budget_seconds: Option<u64>,
) -> Result<TaskPreflightOutcomeV1> {
    let task_name = manifest
        .value
        .get("task")
        .and_then(Value::as_str)
        .context("manifest task is absent during preflight")?;
    let task = Task::parse(task_name).context("manifest task is unknown during preflight")?;
    let genomes = manifest
        .value
        .get("candidates")
        .and_then(Value::as_array)
        .context("manifest candidates are absent during preflight")?
        .iter()
        .map(|candidate| {
            candidate
                .get("canonical_genome")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .context("candidate genome is absent during preflight")
        })
        .collect::<Result<Vec<_>>>()?;
    let width = |field: &str| -> Result<Option<u32>> {
        manifest
            .value
            .get(field)
            .map(|value| {
                value
                    .as_u64()
                    .context("manifest width field is not an unsigned integer")
                    .and_then(|value| u32::try_from(value).context("manifest width overflows u32"))
            })
            .transpose()
    };
    let multiclass_metadata_complete = task != Task::Multiclass
        || (manifest
            .value
            .get("meta_json")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
            && manifest
                .value
                .get("meta_sha256")
                .and_then(Value::as_str)
                .is_some_and(is_sha256));
    preflight_task_manifest(
        registry,
        &TaskPreflightRequest {
            task,
            canonical_genomes: &genomes,
            feature_width: width("feature_width")?,
            mask_width: width("mask_width")?,
            observed_minimum_row_width: width("observed_minimum_row_width")?,
            required_minimum_row_width: width("required_minimum_row_width")?,
            multiclass_metadata_complete,
            release_a_only: true,
            regression_screening_cost_threshold: REGRESSION_SCREENING_COST_THRESHOLD,
            bounded_screening_budget_seconds,
        },
    )
}
