use super::*;

pub(super) fn compute_scaling_statistic(parent: &Value) -> Result<ScalingStatistic> {
    let pointer_path = PathBuf::from(value_string(parent, "/catalog_pointer_path")?);
    let pointer: Value = serde_json::from_slice(&fs::read(&pointer_path)?)?;
    let catalog_path = PathBuf::from(value_string(&pointer, "/catalog")?);
    let data_path = PathBuf::from(value_string(&pointer, "/data")?);
    let catalog_sha256 = value_string(&pointer, "/catalog_sha256")?;
    if sha256_file(&catalog_path)? != catalog_sha256
        || sha256_file(&data_path)? != value_string(&pointer, "/data_sha256")?
    {
        bail!("parent context catalog pointer drift");
    }
    let catalog: Value = serde_json::from_slice(&fs::read(catalog_path)?)?;
    let entries = catalog
        .get("entries")
        .and_then(Value::as_array)
        .context("context catalog lacks entries")?;
    let dimensions = value_usize(&catalog, "/dataset_dimensions")?;
    let target_dimensions = value_usize(&catalog, "/target_dimensions")?;
    if dimensions != 1536 || target_dimensions != 512 {
        bail!("context catalog dimensions drift");
    }
    let stride_floats = dimensions + target_dimensions;
    let blob = fs::read(data_path)?;
    let regression = entries
        .iter()
        .filter(|entry| {
            entry.get("task").and_then(Value::as_str) == Some("regression")
                && entry.get("split").and_then(Value::as_str) == Some("train")
        })
        .collect::<Vec<_>>();
    if regression.len() < 2 {
        bail!("scaling statistic needs at least two regression contexts");
    }
    let mut sums = vec![0.0; dimensions];
    let mut squared_sums = vec![0.0; dimensions];
    for entry in &regression {
        let index = value_usize(entry, "/catalog_index")?;
        let offset = index
            .checked_mul(stride_floats * 4)
            .context("context offset overflow")?;
        if offset + stride_floats * 4 > blob.len() {
            bail!("context slice outside frozen blob");
        }
        for coordinate in 0..dimensions {
            let start = offset + coordinate * 4;
            let value = f32::from_le_bytes(blob[start..start + 4].try_into().unwrap()) as f64;
            if !value.is_finite() {
                bail!("non-finite training-context coordinate");
            }
            sums[coordinate] += value;
            squared_sums[coordinate] += value * value;
        }
    }
    let count = regression.len() as f64;
    let population_sd = sums
        .iter()
        .zip(squared_sums)
        .map(|(sum, squared_sum)| {
            let variance = (squared_sum / count - (sum / count).powi(2)).max(0.0);
            variance.sqrt()
        })
        .collect();
    Ok(ScalingStatistic {
        schema: "TrainOnlyEmbeddingScalingStatistic.v1".to_string(),
        source_catalog_sha256: catalog_sha256,
        contexts: regression.len(),
        dimensions,
        population_sd,
    })
}

pub(super) fn proposal_manifest_from_parent(
    parent: &Value,
    path: &Path,
    run_id: &str,
) -> Result<String> {
    let entries = parent
        .get("datasets")
        .and_then(Value::as_array)
        .context("parent manifest lacks datasets")?;
    let catalog_sha256 = value_string(parent, "/source_catalog_sha256")?;
    let receipt = json!({
        "schema": "TrainOnlyProposalDatasetManifest.v1",
        "run_id": run_id,
        "catalog_sha256": catalog_sha256,
        "entries": entries.iter().map(|entry| json!({
            "task": "regression",
            "dataset_hash": entry.get("dataset_hash").cloned().unwrap_or(Value::Null),
            "context_sha256": entry.pointer("/encoding/context_sha256").cloned().unwrap_or(Value::Null),
        })).collect::<Vec<_>>(),
    });
    write_receipt(path, &receipt)
}

pub(super) fn parse_proposal_bundle(
    path: &Path,
    config: &TrainOnlyRedteamConfig,
    scaling_sha256: &str,
    parent: &Value,
) -> Result<ProposalBundleLink> {
    let mut expected_contexts = HashMap::new();
    for entry in parent
        .get("datasets")
        .and_then(Value::as_array)
        .context("parent manifest lacks datasets")?
    {
        let dataset = value_string(entry, "/dataset_hash")?;
        if !config.dataset_ids.contains(&dataset) {
            continue;
        }
        let context_sha256 = value_string(entry, "/encoding/context_sha256")?;
        let blob_path = PathBuf::from(value_string(entry, "/encoding/blob_path")?);
        let offset = entry
            .pointer("/encoding/offset_bytes")
            .and_then(Value::as_u64)
            .context("parent context lacks offset_bytes")? as usize;
        let length = entry
            .pointer("/encoding/length_bytes")
            .and_then(Value::as_u64)
            .context("parent context lacks length_bytes")? as usize;
        if length != (1536 + 512) * std::mem::size_of::<f32>() {
            bail!("parent context has an invalid byte length");
        }
        let blob = fs::read(&blob_path)?;
        let context = blob
            .get(offset..offset + length)
            .context("parent context is outside its frozen blob")?;
        if sha256_bytes(context) != context_sha256 {
            bail!("parent context bytes drift");
        }
        let target_summary_sha256 = sha256_bytes(&context[1536 * std::mem::size_of::<f32>()..]);
        expected_contexts.insert(dataset, (context_sha256, target_summary_sha256));
    }
    if expected_contexts.len() != config.dataset_ids.len() {
        bail!("parent context provenance does not cover the sealed cohort");
    }
    let file = File::open(path)?;
    let mut exact: HashMap<String, HashMap<u64, usize>> = HashMap::new();
    let mut jittered: HashMap<String, HashMap<u64, usize>> = HashMap::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)?;
        let dataset = value_string(&row, "/dataset_hash")?;
        if !config.dataset_ids.contains(&dataset) {
            bail!("proposal bundle contains an unfrozen dataset");
        }
        let (expected_context_sha256, expected_target_summary_sha256) = expected_contexts
            .get(&dataset)
            .context("proposal dataset lacks frozen context provenance")?;
        if row.get("task").and_then(Value::as_str) != Some("regression")
            || row.get("target_summary_unchanged").and_then(Value::as_bool) != Some(true)
            || row.get("context_sha256") != row.get("exact_context_sha256")
            || row.get("exact_context_sha256").and_then(Value::as_str)
                != Some(expected_context_sha256.as_str())
            || row.get("target_summary_sha256").and_then(Value::as_str)
                != Some(expected_target_summary_sha256.as_str())
        {
            bail!("proposal row violates exact-context/target-summary provenance");
        }
        let origin = value_string(&row, "/proposal_origin")?;
        let decoder_seed = row
            .get("invocation_seed")
            .and_then(Value::as_u64)
            .context("proposal lacks invocation_seed")?;
        match origin.as_str() {
            "exact_context" => {
                if !config.proposal.exact_decoder_seeds.contains(&decoder_seed)
                    || row
                        .get("perturbation_seed")
                        .is_some_and(|value| !value.is_null())
                    || row.get("embedding_jitter_scale").and_then(Value::as_f64) != Some(0.0)
                    || row.get("proposal_context_sha256").and_then(Value::as_str)
                        != Some(expected_context_sha256.as_str())
                {
                    bail!("exact proposal seed or perturbation metadata drift");
                }
                *exact
                    .entry(dataset)
                    .or_default()
                    .entry(decoder_seed)
                    .or_default() += 1;
            }
            "jittered_context" => {
                let perturbation_seed = row
                    .get("perturbation_seed")
                    .and_then(Value::as_u64)
                    .context("jittered proposal lacks perturbation seed")?;
                let perturbation_index = config
                    .proposal
                    .perturbation_seeds
                    .iter()
                    .position(|seed| *seed == perturbation_seed);
                if perturbation_index.is_none()
                    || config.proposal.jitter_decoder_seeds[perturbation_index.unwrap()]
                        != decoder_seed
                    || row.get("embedding_jitter_scale").and_then(Value::as_f64) != Some(0.05)
                    || row
                        .get("embedding_scaling_statistic_sha256")
                        .and_then(Value::as_str)
                        != Some(scaling_sha256)
                    || row.get("proposal_context_sha256").and_then(Value::as_str)
                        == Some(expected_context_sha256.as_str())
                {
                    bail!("jittered proposal provenance drift");
                }
                *jittered
                    .entry(dataset)
                    .or_default()
                    .entry(perturbation_seed)
                    .or_default() += 1;
            }
            _ => bail!("unsupported proposal origin"),
        }
    }
    for dataset in &config.dataset_ids {
        let exact_seeds = exact
            .get(dataset)
            .with_context(|| format!("no exact proposals for {dataset}"))?;
        for seed in &config.proposal.exact_decoder_seeds {
            if exact_seeds.get(seed) != Some(&config.proposal.exact_per_seed) {
                bail!("exact proposal quota drift for {dataset} seed {seed}");
            }
        }
        let jitter_seeds = jittered
            .get(dataset)
            .with_context(|| format!("no jittered proposals for {dataset}"))?;
        for seed in &config.proposal.perturbation_seeds {
            if jitter_seeds.get(seed) != Some(&config.proposal.jitter_per_perturbation) {
                bail!("jittered proposal quota drift for {dataset} seed {seed}");
            }
        }
    }
    Ok(ProposalBundleLink {
        path: absolute(path)?,
        sha256: sha256_file(path)?,
        exact_rows_per_dataset: config.proposal.exact_decoder_seeds.len()
            * config.proposal.exact_per_seed,
        jittered_rows_per_dataset: config.proposal.perturbation_seeds.len()
            * config.proposal.jitter_per_perturbation,
        scaling_statistic_sha256: scaling_sha256.to_string(),
    })
}
