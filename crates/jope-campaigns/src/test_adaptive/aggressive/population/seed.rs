use super::*;

pub(super) fn derive_corrected_generation_zero(
    parent_seed: &SeedProvenance,
    run_id: &str,
    spec: &DatasetSpec,
) -> Result<(Vec<Member>, Vec<SelectedGenerationZeroSeed>)> {
    let mut seen = HashSet::with_capacity(TOTAL_POPULATION);
    let mut candidates = Vec::with_capacity(TOTAL_POPULATION);
    let mut population = Vec::with_capacity(TOTAL_POPULATION);
    for slot in 0..TOTAL_POPULATION {
        let island = slot / ISLAND_POPULATION;
        let local_slot = slot % ISLAND_POPULATION;
        let mut produced = None;
        for retry in 0..10_000usize {
            let steps = 1 + (slot + retry) % 4;
            let mut source = parent_seed.canonical_genome.clone();
            let mut evolved = None;
            let mut valid_chain = true;
            for step in 0..steps {
                let digest = Sha256::digest(
                    format!(
                        "golden46-corrected-g0:{run_id}:{}:{island}:{local_slot}:{retry}:{step}",
                        spec.dataset_hash
                    )
                    .as_bytes(),
                );
                let seed = u64::from_le_bytes(digest[..8].try_into().unwrap());
                let Ok(next) =
                    feat_math::full_rust::mutate_pipeline_expanded(&source, seed, spec.features)
                else {
                    valid_chain = false;
                    break;
                };
                source = next.canonical.clone();
                evolved = Some(next);
            }
            if !valid_chain {
                continue;
            }
            let evolved = evolved.context("generation-zero mutation chain was empty")?;
            if evolved.semantic_sha256 == parent_seed.semantic_sha256
                || !seen.insert(evolved.semantic_sha256.clone())
            {
                continue;
            }
            produced = Some(evolved);
            break;
        }
        let evolved = produced.with_context(|| {
            format!("could not derive unique corrected generation-zero candidate {slot}")
        })?;
        let member = Member::evolved(
            evolved,
            Ancestry {
                operation: "generation_zero_mutation".to_string(),
                parents: vec![parent_seed.semantic_sha256.clone()],
            },
        );
        candidates.push(SelectedGenerationZeroSeed {
            island,
            local_slot,
            seed_class: "unique_full_genome_mutation".to_string(),
            provenance: SeedProvenance {
                slot,
                canonical_genome: member.canonical_genome.clone(),
                semantic_sha256: member.semantic_sha256.clone(),
                complexity: member.complexity,
                rust_source_len: member.rust_source_len,
                source_kind: "corrected_generation_zero_candidate".to_string(),
                source_file: parent_seed.source_file.clone(),
                source_file_sha256: parent_seed.source_file_sha256.clone(),
                source_record: slot,
                source_rank: slot + 1,
                source_score: None,
            },
        });
        population.push(member);
    }
    Ok((population, candidates))
}

pub(super) fn golden46_generation_zero_population(
    _registry: &Path,
    run_id: &str,
    spec: &DatasetSpec,
) -> Result<Vec<Member>> {
    let checkpoint_dir =
        dataset_root(GOLDEN46_CORRECTED_PARENT_RUN, &spec.dataset_hash).join("checkpoints");
    let mut paths = fs::read_dir(&checkpoint_dir)
        .with_context(|| {
            format!(
                "read corrected-parent checkpoints for {}",
                spec.dataset_hash
            )
        })?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    paths.sort();
    let mut parent = None::<(SeedProvenance, String)>;
    for path in paths {
        let bytes = fs::read(&path)?;
        let checkpoint: AggressiveCheckpoint = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&checkpoint)?
            || checkpoint.run_id != GOLDEN46_CORRECTED_PARENT_RUN
            || checkpoint.dataset_hash != spec.dataset_hash
        {
            bail!("invalid corrected-parent checkpoint {}", path.display());
        }
        for outcome in checkpoint.slots.iter().filter(|outcome| outcome.valid) {
            let rmse = outcome.rmse();
            if !rmse.is_finite()
                || parent
                    .as_ref()
                    .is_some_and(|(seed, _)| seed.source_score.is_some_and(|best| rmse >= best))
            {
                continue;
            }
            let digest = sha256_bytes(&bytes);
            parent = Some((
                SeedProvenance {
                    slot: outcome.slot,
                    canonical_genome: outcome.member.canonical_genome.clone(),
                    semantic_sha256: outcome.member.semantic_sha256.clone(),
                    complexity: outcome.member.complexity,
                    rust_source_len: outcome.member.rust_source_len,
                    source_kind: "corrected_test_adaptive_parent".to_string(),
                    source_file: path.to_string_lossy().into_owned(),
                    source_file_sha256: digest.clone(),
                    source_record: outcome.slot,
                    source_rank: 1,
                    source_score: Some(rmse),
                },
                digest,
            ));
        }
    }
    let (parent_seed, parent_checkpoint_sha256) =
        parent.context("corrected Golden46 parent run has no valid candidate")?;
    let (population, candidates) = derive_corrected_generation_zero(&parent_seed, run_id, spec)?;
    let population_sha256 = sha256_bytes(&canonical_json(&candidates)?);
    let snapshot = GenerationZeroSeedSnapshot {
        schema: "JopeGolden46GenerationZeroSeeds.v2".to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        run_id: run_id.to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        provenance_group: provenance_group(&spec.dataset_hash).to_string(),
        arm: "corrected-test-adaptive-continuation".to_string(),
        strategy: "single best preserved v2 parent; 36 deterministic unique full-genome mutations"
            .to_string(),
        parent_run_id: GOLDEN46_CORRECTED_PARENT_RUN.to_string(),
        parent_checkpoint_sha256,
        parent_seed,
        population_sha256,
        generation_zero_can_terminate: false,
        candidates,
    };
    let snapshot_path = dataset_root(run_id, &spec.dataset_hash).join(GENERATION_ZERO_SEEDS_FILE);
    atomic_immutable_write(&snapshot_path, &canonical_json(&snapshot)?)?;
    Ok(population)
}

pub(super) fn initial_population(
    registry: &Path,
    run_id: &str,
    spec: &DatasetSpec,
) -> Result<Vec<Member>> {
    if is_golden46_run(run_id) {
        return golden46_generation_zero_population(registry, run_id, spec);
    }
    let seed_path = dataset_root(run_id, &spec.dataset_hash).join("transfer-seeds.v1.json");
    let base = load_or_create_transfer_seed_manifest(&seed_path, registry, spec)?
        .seeds
        .iter()
        .take(ISLAND_POPULATION)
        .map(Member::from_seed)
        .collect::<Vec<_>>();
    expand_initial_population(&base, spec.features)
}

pub(super) fn expand_initial_population(base: &[Member], features: usize) -> Result<Vec<Member>> {
    if base.is_empty() || base.len() > ISLAND_POPULATION {
        bail!("aggressive seed baseline does not fit one island");
    }
    let mut population = Vec::with_capacity(TOTAL_POPULATION);
    let mut seen = HashSet::with_capacity(TOTAL_POPULATION);
    for island in 0..ISLANDS {
        for local_slot in 0..ISLAND_POPULATION {
            if island == 0 && local_slot < base.len() {
                let member = base[local_slot].clone();
                if !seen.insert(member.semantic_sha256.clone()) {
                    bail!("aggressive seed baseline contains a duplicate genome");
                }
                population.push(member);
                continue;
            }
            let parent = &base[local_slot % base.len()];
            let mut produced = None;
            for retry in 0..1000usize {
                let digest = Sha256::digest(
                    format!("aggressive:g0:{island}:{local_slot}:{retry}").as_bytes(),
                );
                let seed = u64::from_le_bytes(digest[..8].try_into().unwrap());
                if let Ok(genome) = mutate_pipeline(&parent.canonical_genome, seed, features) {
                    if seen.insert(genome.semantic_sha256.clone()) {
                        produced = Some(Member::evolved(
                            genome,
                            Ancestry {
                                operation: "aggressive_generation_zero_mutation".to_string(),
                                parents: vec![parent.semantic_sha256.clone()],
                            },
                        ));
                        break;
                    }
                }
            }
            population.push(
                produced.with_context(|| {
                    format!("could not derive island {island} slot {local_slot}")
                })?,
            );
        }
    }
    if population.len() != TOTAL_POPULATION {
        bail!("aggressive initial population is incomplete");
    }
    Ok(population)
}
