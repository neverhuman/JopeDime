use super::*;

pub(crate) fn load_checkpoints(root: &Path) -> Result<Vec<GenerationCheckpoint>> {
    let generations_root = root.join("generations");
    if !generations_root.exists() {
        return Ok(Vec::new());
    }
    let mut generation_numbers = Vec::new();
    for entry in fs::read_dir(&generations_root)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            bail!("Binary42 generation directory has a non-UTF8 name");
        };
        let Some(raw) = name.strip_prefix("generation-") else {
            bail!("unexpected entry under Binary42 generations");
        };
        generation_numbers.push(raw.parse::<usize>().context("parse generation number")?);
    }
    generation_numbers.sort_unstable();
    if generation_numbers
        .iter()
        .enumerate()
        .any(|(expected, actual)| expected != *actual)
    {
        bail!("Binary42 generation sequence contains a gap");
    }
    let mut checkpoints = Vec::new();
    for (position, generation) in generation_numbers.iter().copied().enumerate() {
        let path = generation_root(root, generation).join("checkpoint.v1.json");
        if !path.exists() {
            if position + 1 == generation_numbers.len() {
                break;
            }
            bail!("Binary42 generation {generation} lacks a completed checkpoint");
        }
        let bytes = fs::read(&path)?;
        let checkpoint: GenerationCheckpoint = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&checkpoint)?
            || checkpoint.schema != CHECKPOINT_SCHEMA
            || !checkpoint.test_adaptive
            || checkpoint.benchmark_valid
            || checkpoint.generation != generation
            || checkpoint.logical_slots != POPULATION
            || checkpoint.slots.len() != POPULATION
            || checkpoint
                .slots
                .iter()
                .enumerate()
                .any(|(slot, record)| record.generation != generation || record.slot != slot)
        {
            bail!("Binary42 generation {generation} checkpoint is invalid");
        }
        let expected_previous = checkpoints
            .last()
            .map(|prior| canonical_json(prior).map(|bytes| sha256_bytes(&bytes)))
            .transpose()?;
        if checkpoint.previous_checkpoint_sha256 != expected_previous {
            bail!("Binary42 checkpoint hash chain mismatch at generation {generation}");
        }
        checkpoints.push(checkpoint);
    }
    Ok(checkpoints)
}

pub(crate) fn atomic_rebuild(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
    if temporary.exists() {
        fs::remove_file(&temporary)?;
    }
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(&temporary, path)?;
    Ok(())
}

pub(crate) fn evaluation_records(
    checkpoints: &[GenerationCheckpoint],
) -> Vec<EvaluationIndexRecord> {
    checkpoints
        .iter()
        .flat_map(|checkpoint| &checkpoint.slots)
        .map(|slot| EvaluationIndexRecord {
            test_adaptive: true,
            benchmark_valid: false,
            generation: slot.generation,
            slot: slot.slot,
            semantic_sha256: slot.member.semantic_sha256.clone(),
            ancestry_operation: slot.member.ancestry.operation.clone(),
            ancestry_parents: slot.member.ancestry.parents.clone(),
            cache_disposition: slot.cache_disposition.clone(),
            assigned_node: slot.assigned_node.clone(),
            evaluation_node: slot.evaluation_node.clone(),
            scq_job_id: slot.scq_job_id.clone(),
            auc: slot.score.as_ref().map(|score| score.auc),
            logloss: slot.score.as_ref().map(|score| score.logloss),
            complexity: slot.member.complexity,
            receipt_sha256: slot.receipt_sha256.clone(),
            prediction_sha256: slot.prediction_sha256.clone(),
            failure: slot.failure.clone(),
            runtime_seconds: slot.runtime_seconds,
        })
        .collect()
}

pub(crate) fn csv_field(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

pub(crate) fn rebuild_indexes(
    root: &Path,
    checkpoints: &[GenerationCheckpoint],
    extended: bool,
) -> Result<()> {
    let records = evaluation_records(checkpoints);
    let mut jsonl = Vec::new();
    for record in &records {
        jsonl.extend_from_slice(&canonical_json(record)?);
    }
    atomic_rebuild(&root.join("evaluations.jsonl"), &jsonl)?;
    let mut csv = if extended {
        b"test_adaptive,benchmark_valid,generation,slot,semantic_sha256,ancestry_operation,ancestry_parents,cache_disposition,assigned_node,evaluation_node,scq_job_id,auc,logloss,complexity,runtime_seconds,receipt_sha256,prediction_sha256,failure\n".to_vec()
    } else {
        b"test_adaptive,benchmark_valid,generation,slot,semantic_sha256,ancestry_operation,ancestry_parents,cache_disposition,assigned_node,evaluation_node,scq_job_id,auc,logloss,complexity,receipt_sha256,prediction_sha256,failure\n".to_vec()
    };
    for record in &records {
        let row = if extended {
            format!(
                "true,false,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                record.generation,
                record.slot,
                csv_field(&record.semantic_sha256),
                csv_field(&record.ancestry_operation),
                csv_field(&record.ancestry_parents.join(";")),
                csv_field(&record.cache_disposition),
                csv_field(&record.assigned_node),
                csv_field(&record.evaluation_node),
                csv_field(&record.scq_job_id),
                record
                    .auc
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                record
                    .logloss
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                record.complexity,
                record
                    .runtime_seconds
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                csv_field(&record.receipt_sha256),
                csv_field(&record.prediction_sha256),
                csv_field(record.failure.as_deref().unwrap_or("")),
            )
        } else {
            format!(
                "true,false,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                record.generation,
                record.slot,
                csv_field(&record.semantic_sha256),
                csv_field(&record.ancestry_operation),
                csv_field(&record.ancestry_parents.join(";")),
                csv_field(&record.cache_disposition),
                csv_field(&record.assigned_node),
                csv_field(&record.evaluation_node),
                csv_field(&record.scq_job_id),
                record
                    .auc
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                record
                    .logloss
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                record.complexity,
                csv_field(&record.receipt_sha256),
                csv_field(&record.prediction_sha256),
                csv_field(record.failure.as_deref().unwrap_or("")),
            )
        };
        csv.extend_from_slice(row.as_bytes());
    }
    atomic_rebuild(&root.join("evaluations.csv"), &csv)?;
    if extended {
        let mut genomes = BTreeMap::<String, GenomeIndexRecord>::new();
        for slot in checkpoints.iter().flat_map(|checkpoint| &checkpoint.slots) {
            genomes
                .entry(slot.member.semantic_sha256.clone())
                .or_insert_with(|| GenomeIndexRecord {
                    semantic_sha256: slot.member.semantic_sha256.clone(),
                    canonical_genome: slot.member.canonical_genome.clone(),
                    complexity: slot.member.complexity,
                    ancestry_operation: slot.member.ancestry.operation.clone(),
                    ancestry_parents: slot.member.ancestry.parents.clone(),
                    first_generation: slot.generation,
                    first_slot: slot.slot,
                });
        }
        let mut genomes_jsonl = Vec::new();
        let mut genomes_csv = b"semantic_sha256,canonical_genome,complexity,ancestry_operation,ancestry_parents,first_generation,first_slot\n".to_vec();
        for genome in genomes.values() {
            genomes_jsonl.extend_from_slice(&canonical_json(genome)?);
            genomes_csv.extend_from_slice(
                format!(
                    "{},{},{},{},{},{},{}\n",
                    csv_field(&genome.semantic_sha256),
                    csv_field(&genome.canonical_genome),
                    genome.complexity,
                    csv_field(&genome.ancestry_operation),
                    csv_field(&genome.ancestry_parents.join(";")),
                    genome.first_generation,
                    genome.first_slot,
                )
                .as_bytes(),
            );
        }
        atomic_rebuild(&root.join("genomes.jsonl"), &genomes_jsonl)?;
        atomic_rebuild(&root.join("genomes.csv"), &genomes_csv)?;
    }
    let mut auc_sorted = records
        .iter()
        .filter(|record| record.auc.is_some())
        .collect::<Vec<_>>();
    auc_sorted.sort_by(|left, right| {
        right
            .auc
            .unwrap()
            .total_cmp(&left.auc.unwrap())
            .then_with(|| left.logloss.unwrap().total_cmp(&right.logloss.unwrap()))
            .then_with(|| left.complexity.cmp(&right.complexity))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
            .then_with(|| left.generation.cmp(&right.generation))
            .then_with(|| left.slot.cmp(&right.slot))
    });
    let index = json!({
        "schema": "JopeBinary42TestAdaptiveAucIndex.v1",
        "test_adaptive": true,
        "benchmark_valid": false,
        "records": &auc_sorted,
    });
    atomic_rebuild(
        &root.join("indexes").join("auc-sorted.v1.json"),
        &canonical_json(&index)?,
    )?;
    if extended {
        let mut auc_jsonl = Vec::new();
        let mut auc_csv = b"generation,slot,semantic_sha256,auc,logloss,complexity,receipt_sha256,prediction_sha256\n".to_vec();
        for record in auc_sorted {
            auc_jsonl.extend_from_slice(&canonical_json(record)?);
            auc_csv.extend_from_slice(
                format!(
                    "{},{},{},{},{},{},{},{}\n",
                    record.generation,
                    record.slot,
                    csv_field(&record.semantic_sha256),
                    record.auc.unwrap(),
                    record.logloss.unwrap(),
                    record.complexity,
                    csv_field(&record.receipt_sha256),
                    csv_field(&record.prediction_sha256),
                )
                .as_bytes(),
            );
        }
        atomic_rebuild(&root.join("indexes").join("auc-sorted.jsonl"), &auc_jsonl)?;
        atomic_rebuild(&root.join("indexes").join("auc-sorted.csv"), &auc_csv)?;
    }
    Ok(())
}
