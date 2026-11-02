use super::*;

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

pub(crate) fn legacy_refill(
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

pub(crate) fn load_parent_archive(
    dataset: &DatasetInput,
) -> Result<(Vec<Candidate>, Vec<Candidate>)> {
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
