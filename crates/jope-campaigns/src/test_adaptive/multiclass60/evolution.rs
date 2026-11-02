use super::*;

pub(super) fn genome_complexity(source: &str) -> Result<usize> {
    let GenomeIr::Pipeline(pipeline) = GenomeIr::parse(source)? else {
        bail!("multiclass member must be a pipeline");
    };
    Ok(pipeline
        .stages
        .iter()
        .map(|stage| 1 + stage.params.len())
        .sum())
}

pub(super) fn slots_from_receipts(
    dataset: &DatasetBinding,
    generation: usize,
    population: &[SeedMember],
    manifests: &[PathBuf],
    receipts: &[PathBuf],
    expected_protocol_sha256: &str,
) -> Result<Vec<LogicalSlot>> {
    let mut slots = Vec::with_capacity(POPULATION);
    for shard in 0..SHARD_SIZES.len() {
        let values = read_receipt_values(&receipts[shard])?;
        if values.len() != SHARD_SIZES[shard] {
            bail!(
                "{} generation {generation} shard {shard} emitted {} receipts, expected {}",
                dataset.dataset_hash,
                values.len(),
                SHARD_SIZES[shard]
            );
        }
        let manifest_sha256 = sha256_file(&manifests[shard])?;
        let receipt_sha256 = sha256_file(&receipts[shard])?;
        let start = SHARD_SIZES[..shard].iter().sum::<usize>();
        for (offset, value) in values.into_iter().enumerate() {
            let slot = start + offset;
            let member = &population[slot];
            if value.get("schema").and_then(Value::as_str)
                != Some("JopeFlywheelCorpusEvalReceipt.v1")
                || value.get("task").and_then(Value::as_str) != Some("multiclass")
                || value.get("dataset_hash").and_then(Value::as_str)
                    != Some(dataset.dataset_hash.as_str())
                || value.get("semantic_sha256").and_then(Value::as_str)
                    != Some(member.semantic_sha256.as_str())
                || value.get("canonical_genome").and_then(Value::as_str)
                    != Some(member.canonical_genome.as_str())
                || value.get("train_sha256").and_then(Value::as_str)
                    != Some(dataset.train_sha256.as_str())
                || value.get("test_sha256").and_then(Value::as_str)
                    != Some(dataset.test_sha256.as_str())
                || value.get("manifest_sha256").and_then(Value::as_str)
                    != Some(manifest_sha256.as_str())
                || value.get("protocol_sha256").and_then(Value::as_str)
                    != Some(expected_protocol_sha256)
                || value.get("test_adaptive").and_then(Value::as_bool) != Some(true)
                || value.get("selection_bias").and_then(Value::as_str) != Some("test-selected")
                || value.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
                || value.get("generation").and_then(Value::as_str)
                    != Some(format!("multiclass-generation-{generation:03}").as_str())
            {
                bail!(
                    "{} generation {generation} slot {slot} receipt binding mismatch",
                    dataset.dataset_hash
                );
            }
            let valid = value.get("valid").and_then(Value::as_bool) == Some(true);
            let score = if valid {
                let macro_f1 = value
                    .get("score")
                    .and_then(Value::as_f64)
                    .filter(|score| score.is_finite())
                    .context("valid multiclass receipt lacks finite macro-F1")?;
                let accuracy = value
                    .get("secondary_score")
                    .and_then(Value::as_f64)
                    .filter(|score| score.is_finite())
                    .context("valid multiclass receipt lacks finite accuracy")?;
                if value.get("metric").and_then(Value::as_str) != Some("macro_f1")
                    || value.get("secondary_metric").and_then(Value::as_str) != Some("accuracy")
                    || value
                        .get("predictions_sha256")
                        .and_then(Value::as_str)
                        .is_none()
                    || value
                        .get("probabilities_sha256")
                        .and_then(Value::as_str)
                        .is_none()
                {
                    bail!("valid multiclass receipt lacks metric or prediction evidence");
                }
                Some(RankedScore {
                    macro_f1,
                    accuracy,
                    complexity: genome_complexity(&member.canonical_genome)?,
                    semantic_sha256: member.semantic_sha256.clone(),
                })
            } else {
                if value.get("failure").and_then(Value::as_str).is_none() {
                    bail!("invalid multiclass candidate receipt lacks failure disposition");
                }
                None
            };
            slots.push(LogicalSlot {
                slot,
                member: member.clone(),
                score,
                valid,
                failure: value
                    .get("failure")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                receipt_sha256: receipt_sha256.clone(),
                predictions_sha256: value
                    .get("predictions_sha256")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                probabilities_sha256: value
                    .get("probabilities_sha256")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                node: value
                    .get("node")
                    .and_then(Value::as_str)
                    .context("candidate receipt lacks node")?
                    .to_string(),
            });
        }
    }
    if slots.len() != POPULATION {
        bail!("generation produced {} logical slots", slots.len());
    }
    Ok(slots)
}

pub(super) fn slot_rank_order(left: &LogicalSlot, right: &LogicalSlot) -> Ordering {
    match (&left.score, &right.score) {
        (Some(left), Some(right)) => rank_order(left, right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => left
            .member
            .semantic_sha256
            .cmp(&right.member.semantic_sha256),
    }
}

pub(super) fn evolution_seed(
    dataset_hash: &str,
    generation: usize,
    slot: usize,
    retry: usize,
) -> u64 {
    let digest =
        Sha256::digest(format!("{RUN_ID}:{dataset_hash}:{generation}:{slot}:{retry}").as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("eight digest bytes"))
}

pub(super) fn regression_surrogate(source: &str) -> Result<String> {
    let GenomeIr::Pipeline(mut pipeline) = GenomeIr::parse(source)? else {
        bail!("multiclass parent must be a pipeline");
    };
    let head = pipeline
        .stages
        .pop()
        .context("multiclass parent lacks head")?;
    if !MULTICLASS_HEADS.contains(&head.op.as_str()) {
        bail!("multiclass parent has incompatible head {}", head.op);
    }
    pipeline.stages.push(Stage::new("ridge"));
    Ok(GenomeIr::Pipeline(pipeline).canonical())
}

pub(super) fn next_population(
    dataset: &DatasetBinding,
    ranked: &[LogicalSlot],
    generation: usize,
) -> Result<Vec<SeedMember>> {
    let valid = ranked.iter().filter(|slot| slot.valid).collect::<Vec<_>>();
    // A dataset-wide candidate-invalid generation is a model-logic result, not
    // an infrastructure failure and not a reason to kill every other active
    // dataset. Deterministically mutate the structurally valid failed genomes
    // with no elites so the next generation can escape the failed family.
    if valid.is_empty() {
        // Candidate mutation cannot repair a dataset-level evaluator
        // incompatibility.  Preserve the complete, canonical population and
        // advance deterministically; the corrected evaluator is introduced
        // through an immutable successor campaign rather than by changing the
        // binary pinned by this run.
        return ranked
            .iter()
            .enumerate()
            .map(|(slot, logical)| {
                let mut member = logical.member.clone();
                member.slot = slot;
                member.source = format!(
                    "all-invalid-reseed:generation={generation}:{}",
                    logical.member.semantic_sha256
                );
                Ok(member)
            })
            .collect();
    }
    // Keep valid candidates first for tournament selection, but retain the
    // remaining structurally valid genomes as diversity.  Restricting a
    // sparse generation to one or two successful parents can make the family
    // cap impossible to fill and burns thousands of deterministic retries.
    let parents = ranked.iter().collect::<Vec<_>>();
    if parents.is_empty() {
        bail!("{} generation has no candidate slots", dataset.dataset_hash);
    }
    let parent_pool = parents.len().min(30);
    let elite_count = valid.len().min(7);
    let mut next = Vec::with_capacity(POPULATION);
    let mut seen = BTreeSet::new();
    let mut families = BTreeMap::<String, usize>::new();
    for elite in valid.iter().take(elite_count) {
        let mut member = elite.member.clone();
        member.slot = next.len();
        member.source = format!("elite:{}", elite.member.semantic_sha256);
        *families
            .entry(multiclass_head_family(&member.canonical_genome)?)
            .or_default() += 1;
        seen.insert(member.semantic_sha256.clone());
        next.push(member);
    }
    for slot in elite_count..POPULATION {
        let mut produced = None;
        for retry in 0..2_000 {
            let mut rng = Pcg32::new(
                evolution_seed(&dataset.dataset_hash, generation, slot, retry),
                0x6d63_3237_326e_6578,
            );
            let tournament = |rng: &mut Pcg32| -> usize {
                (0..3)
                    .map(|_| rng.next_u32() as usize % parent_pool)
                    .min()
                    .expect("nonempty tournament")
            };
            let first = tournament(&mut rng);
            let first_source = regression_surrogate(&parents[first].member.canonical_genome)?;
            let (candidate, operation, parents) = if rng.next_f64() < 0.25 {
                let second = tournament(&mut rng);
                let second_source = regression_surrogate(&parents[second].member.canonical_genome)?;
                let Ok(candidate) = crossover_pipelines(
                    &first_source,
                    &second_source,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                ) else {
                    continue;
                };
                (
                    candidate.canonical,
                    "crossover",
                    vec![
                        parents[first].member.semantic_sha256.clone(),
                        parents[second].member.semantic_sha256.clone(),
                    ],
                )
            } else {
                let Ok(candidate) = mutate_pipeline(
                    &first_source,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                    dataset.n_features.max(1),
                ) else {
                    continue;
                };
                (
                    candidate.canonical,
                    "mutation",
                    vec![parents[first].member.semantic_sha256.clone()],
                )
            };
            let variant = rng.next_u32() as usize;
            let Ok(mut member) = rewrite_seed(&candidate, variant) else {
                continue;
            };
            if !seen.insert(member.semantic_sha256.clone()) {
                continue;
            }
            let family = multiclass_head_family(&member.canonical_genome)?;
            if *families.get(&family).unwrap_or(&0) >= POPULATION / MULTICLASS_HEADS.len() {
                continue;
            }
            *families.entry(family).or_default() += 1;
            member.slot = slot;
            member.source = format!("{operation}:{}:retry={retry}", parents.join("+"));
            produced = Some(member);
            break;
        }
        next.push(produced.with_context(|| {
            format!(
                "could not generate canonical multiclass child for {} generation {generation} slot {slot}",
                dataset.dataset_hash
            )
        })?);
    }
    Ok(next)
}

pub(super) fn checkpoint_path(root: &Path, dataset_hash: &str, generation: usize) -> PathBuf {
    root.join("datasets")
        .join(dataset_hash)
        .join("checkpoints")
        .join(format!("generation-{generation:03}.v1.json"))
}

pub(super) fn load_checkpoint(
    root: &Path,
    dataset: &DatasetBinding,
    generation: usize,
) -> Result<GenerationCheckpoint> {
    let path = checkpoint_path(root, &dataset.dataset_hash, generation);
    let checkpoint: GenerationCheckpoint = serde_json::from_slice(&fs::read(&path)?)?;
    if checkpoint.schema != CHECKPOINT_SCHEMA
        || checkpoint.run_id != RUN_ID
        || checkpoint.dataset_hash != dataset.dataset_hash
        || checkpoint.comparator_sha256 != dataset.comparator_sha256
        || checkpoint.generation != generation
        || !checkpoint.test_adaptive
        || checkpoint.selection_bias != "test-selected"
        || checkpoint.benchmark_valid
        || checkpoint.slots.len() != POPULATION
    {
        bail!("invalid multiclass checkpoint {}", path.display());
    }
    Ok(checkpoint)
}
