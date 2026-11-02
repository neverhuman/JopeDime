#[derive(Clone, Debug)]
struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

fn legacy_offspring_seed(
    parent_run_id: &str,
    dataset_hash: &str,
    generation: usize,
    slot: usize,
    retry: usize,
) -> u64 {
    let material = format!(
        "train-only-offspring-v1\0{parent_run_id}\0{dataset_hash}\0{generation}\0{slot}\0{retry}"
    );
    let digest = Sha256::digest(material.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

fn legacy_refill(
    parent_run_id: &str,
    dataset_hash: &str,
    generation: usize,
    ranked: &[LegacyScoredCandidate],
    history: &mut HashSet<String>,
) -> Result<Vec<Candidate>> {
    if ranked
        .iter()
        .filter(|candidate| candidate.robust_fitness.is_some())
        .count()
        < 30
    {
        bail!("generation-20 control has fewer than 30 rankable parents");
    }
    let mut next = Vec::with_capacity(60);
    let mut generation_semantics = HashSet::new();
    for elite in ranked.iter().take(3) {
        let mut candidate = elite.candidate.clone();
        generation_semantics.insert(candidate.semantic_sha256.clone());
        candidate.ancestry = json!({
            "operation": "elite",
            "parents": [candidate.semantic_sha256],
            "generation": generation,
        });
        next.push(candidate);
    }
    for slot in 3..60 {
        let mut accepted = None;
        for retry in 0..100_000 {
            let mut rng = DeterministicRng::new(legacy_offspring_seed(
                parent_run_id,
                dataset_hash,
                generation,
                slot,
                retry,
            ));
            let tournament = |rng: &mut DeterministicRng| {
                let mut winner = (rng.next_u64() as usize) % 30;
                for _ in 1..3 {
                    winner = winner.min((rng.next_u64() as usize) % 30);
                }
                winner
            };
            let left = tournament(&mut rng);
            let crossover = rng.next_u64().is_multiple_of(4);
            let (genome, parents, operation, roots) = if crossover {
                let right = tournament(&mut rng);
                let Ok(genome) = crossover_canonical_pipelines(
                    &ranked[left].candidate.canonical_genome,
                    &ranked[right].candidate.canonical_genome,
                    rng.next_u64(),
                    12,
                ) else {
                    continue;
                };
                let mut roots = ranked[left].candidate.roots.clone();
                roots.extend(ranked[right].candidate.roots.iter().cloned());
                (
                    genome,
                    vec![
                        ranked[left].candidate.semantic_sha256.clone(),
                        ranked[right].candidate.semantic_sha256.clone(),
                    ],
                    "crossover",
                    roots,
                )
            } else {
                let Ok(genome) = mutate_canonical_pipeline_v2(
                    &ranked[left].candidate.canonical_genome,
                    rng.next_u64(),
                    12,
                    true,
                ) else {
                    continue;
                };
                (
                    genome,
                    vec![ranked[left].candidate.semantic_sha256.clone()],
                    "mutation",
                    ranked[left].candidate.roots.clone(),
                )
            };
            let Ok(candidate) = candidate_from_parts(
                genome,
                roots,
                BTreeSet::new(),
                BTreeSet::new(),
                json!({
                    "operation": operation,
                    "parents": parents,
                    "generation": generation,
                    "slot": slot,
                    "retry": retry,
                    "operator_version": "canonical-v2",
                }),
                12,
            ) else {
                continue;
            };
            if !generation_semantics.insert(candidate.semantic_sha256.clone())
                || !history.insert(candidate.semantic_sha256.clone())
            {
                generation_semantics.remove(&candidate.semantic_sha256);
                continue;
            }
            accepted = Some(candidate);
            break;
        }
        next.push(accepted.context("could not generate a unique v2 continuation child")?);
    }
    Ok(next)
}

fn load_parent_archive(dataset: &DatasetInput) -> Result<(Vec<Candidate>, Vec<Candidate>)> {
    let mut archive = Vec::new();
    let mut generation_nine = Vec::new();
    for (generation, path) in dataset.parent_generation_paths.iter().enumerate() {
        let (checkpoint, digest): (Value, _) = read_receipt(path)?;
        if digest != dataset.parent_generation_sha256[generation]
            || checkpoint.get("generation").and_then(Value::as_u64) != Some(generation as u64)
        {
            bail!("parent generation identity drift");
        }
        let population = checkpoint
            .get("population")
            .and_then(Value::as_array)
            .context("parent checkpoint lacks population")?;
        let candidates = population
            .iter()
            .map(candidate_from_parent)
            .collect::<Result<Vec<_>>>()?;
        if candidates.len() != 60 {
            bail!("parent checkpoint population drift");
        }
        archive.extend(candidates.clone());
        if generation == 9 {
            generation_nine = candidates;
        }
    }
    Ok((archive, generation_nine))
}

mod g20_dataset;
pub(super) use g20_dataset::run_g20_dataset;

fn merge_candidate(candidates: &mut HashMap<String, Candidate>, mut candidate: Candidate) {
    if let Some(existing) = candidates.get_mut(&candidate.semantic_sha256) {
        existing.roots.append(&mut candidate.roots);
        existing
            .proposal_origins
            .append(&mut candidate.proposal_origins);
        existing.decoder_seeds.append(&mut candidate.decoder_seeds);
    } else {
        candidates.insert(candidate.semantic_sha256.clone(), candidate);
    }
}

fn load_smart_candidates(
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
) -> Result<(Vec<Candidate>, Vec<Candidate>, Vec<Candidate>)> {
    let mut exact = HashMap::new();
    let mut jittered = HashMap::new();
    let mut skipped = 0usize;
    for line in BufReader::new(File::open(&manifest.proposal_bundle.path)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)?;
        if row.get("dataset_hash").and_then(Value::as_str) != Some(&dataset.dataset_hash) {
            continue;
        }
        let origin = value_string(&row, "/proposal_origin")?;
        let seed = row
            .get("invocation_seed")
            .and_then(Value::as_u64)
            .context("proposal lacks invocation_seed")?;
        let genome = value_string(&row, "/canonical_genome")?;
        if row.get("semantic_sha256").and_then(Value::as_str)
            != Some(semantic_identity(&genome).as_str())
        {
            bail!("proposal semantic identity drift");
        }
        let candidate = candidate_from_parts(
            genome,
            BTreeSet::from(["dl".to_string()]),
            BTreeSet::from([origin.clone()]),
            BTreeSet::from([seed]),
            json!({
                "operation": "dl_proposal",
                "source": origin,
                "decoder_seed": seed,
                "perturbation_seed": row.get("perturbation_seed").cloned().unwrap_or(Value::Null),
                "proposal_context_sha256": row.get("proposal_context_sha256").cloned().unwrap_or(Value::Null),
                "exact_context_sha256": row.get("exact_context_sha256").cloned().unwrap_or(Value::Null),
                "scaling_statistic_sha256": row.get("embedding_scaling_statistic_sha256").cloned().unwrap_or(Value::Null),
            }),
            SMART_MAX_TRANSFORMS,
        );
        let Ok(candidate) = candidate else {
            skipped += 1;
            continue;
        };
        match origin.as_str() {
            "exact_context" => merge_candidate(&mut exact, candidate),
            "jittered_context" => merge_candidate(&mut jittered, candidate),
            _ => bail!("unsupported smart proposal origin"),
        }
    }
    let (ranking, digest): (Value, _) = read_receipt(&dataset.transfer_ranking_path)?;
    if digest != dataset.transfer_ranking_sha256
        || ranking.get("target_dataset_id").and_then(Value::as_str) != Some(&dataset.dataset_hash)
        || ranking
            .get("direct_dl_sources_excluded")
            .and_then(Value::as_bool)
            != Some(true)
        || !ranking
            .get("excluded_dataset_ids")
            .and_then(Value::as_array)
            .is_some_and(|values| {
                values
                    .iter()
                    .any(|value| value.as_str() == Some(&dataset.dataset_hash))
            })
    {
        bail!("target-clean transfer ranking provenance drift");
    }
    let mut transfer = HashMap::new();
    for row in ranking
        .get("genomes")
        .and_then(Value::as_array)
        .context("transfer ranking lacks genomes")?
        .iter()
    {
        let genome = value_string(row, "/canonical_genome")?;
        let parsed = GenomeIr::parse(&genome)?;
        let GenomeIr::Pipeline(pipeline) = &parsed else {
            bail!("target-clean transfer candidate is not a pipeline genome");
        };
        if pipeline.stages.len().saturating_sub(1) > SMART_MAX_TRANSFORMS {
            continue;
        }
        let candidate = candidate_from_parts(
            genome,
            BTreeSet::from(["transfer".to_string()]),
            BTreeSet::from(["target_clean_transfer".to_string()]),
            BTreeSet::new(),
            json!({
                "operation": "target_clean_transfer",
                "source_ancestry": row,
                "ranking_sha256": dataset.transfer_ranking_sha256,
            }),
            SMART_MAX_TRANSFORMS,
        );
        let Ok(candidate) = candidate else {
            skipped += 1;
            continue;
        };
        merge_candidate(&mut transfer, candidate);
        if transfer.len() == 256 {
            break;
        }
    }
    let values = |map: HashMap<String, Candidate>| {
        let mut values = map.into_values().collect::<Vec<_>>();
        values.sort_by(|left, right| left.semantic_sha256.cmp(&right.semantic_sha256));
        values
    };
    let exact = values(exact);
    let jittered = values(jittered);
    let transfer = values(transfer);
    if exact.len() < 80 || jittered.len() < 60 || transfer.len() != 256 {
        bail!(
            "progressive screen cannot fill quotas: exact={}, jittered={}, transfer={}",
            exact.len(),
            jittered.len(),
            transfer.len()
        );
    }
    if skipped > 0 {
        eprintln!(
            "smart-v3 load_smart_candidates skipped {} malformed candidates for {}",
            skipped, dataset.dataset_hash
        );
    }
    Ok((exact, jittered, transfer))
}

fn select_with_head_cap(
    ranked: &[RankedCandidate],
    count: usize,
    head_counts: &mut HashMap<String, usize>,
    cap: usize,
) -> Result<Vec<Candidate>> {
    let mut selected = Vec::with_capacity(count);
    let mut semantics = HashSet::new();
    for row in ranked {
        if selected.len() == count {
            break;
        }
        let present = *head_counts.get(&row.candidate.head_family).unwrap_or(&0);
        if present >= cap || !semantics.insert(row.candidate.semantic_sha256.clone()) {
            continue;
        }
        *head_counts
            .entry(row.candidate.head_family.clone())
            .or_default() += 1;
        selected.push(row.candidate.clone());
    }
    if selected.len() != count {
        bail!("head-family cap prevents filling an initialization quota");
    }
    Ok(selected)
}

fn validate_initial_populations(
    populations: &BTreeMap<String, Vec<Candidate>>,
    island_size: usize,
    expected_islands: &[&str],
    head_family_cap: usize,
) -> Result<()> {
    if populations.len() != expected_islands.len()
        || expected_islands
            .iter()
            .any(|island| populations.get(*island).is_none())
        || populations
            .values()
            .any(|population| population.len() != island_size)
    {
        bail!("Smart-v3 initialization island size drift");
    }

    let mut semantics = HashSet::new();
    for island in expected_islands {
        let mut head_counts = HashMap::new();
        for candidate in &populations[*island] {
            if !semantics.insert(candidate.semantic_sha256.clone()) {
                bail!("initial Smart-v3 population is not canonically unique");
            }
            let count = head_counts
                .entry(candidate.head_family.clone())
                .or_insert(0usize);
            *count += 1;
            if *count > head_family_cap {
                bail!("Smart-v3 {island} initialization head-family cap exceeded");
            }
        }
    }
    Ok(())
}

fn mixed_seed(run_id: &str, dataset_hash: &str, slot: usize, retry: usize) -> u64 {
    let digest = Sha256::digest(
        format!("smart-v3-mixed-init-v1\0{run_id}\0{dataset_hash}\0{slot}\0{retry}").as_bytes(),
    );
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}
