fn build_screening(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    data: &RegressionData,
    bank: &FoldBankReceipt,
    workers: usize,
) -> Result<(ScreeningReceipt, String, PathBuf)> {
    let path = root
        .join("datasets")
        .join(&dataset.dataset_hash)
        .join("smart")
        .join("screening.v1.json");
    if path.exists() {
        let (screening, digest): (ScreeningReceipt, _) = read_receipt(&path)?;
        return Ok((screening, digest, path));
    }
    let (exact, jittered, transfer) = load_smart_candidates(manifest, dataset)?;
    let first = &bank.search[..1];
    let mut exact_first = evaluate_candidates(
        root,
        manifest,
        dataset,
        data,
        "search-screen-repeat-a",
        first,
        &exact,
        workers,
    )?;
    let mut jittered_first = evaluate_candidates(
        root,
        manifest,
        dataset,
        data,
        "search-screen-repeat-a",
        first,
        &jittered,
        workers,
    )?;
    let mut transfer_first = evaluate_candidates(
        root,
        manifest,
        dataset,
        data,
        "search-screen-repeat-a",
        first,
        &transfer,
        workers,
    )?;
    exact_first.truncate(80);
    jittered_first.truncate(60);
    transfer_first.truncate(100);
    let advanced_exact = evaluate_candidates(
        root,
        manifest,
        dataset,
        data,
        "search",
        &bank.search,
        &exact_first
            .iter()
            .map(|row| row.candidate.clone())
            .collect::<Vec<_>>(),
        workers,
    )?;
    let advanced_jittered = evaluate_candidates(
        root,
        manifest,
        dataset,
        data,
        "search",
        &bank.search,
        &jittered_first
            .iter()
            .map(|row| row.candidate.clone())
            .collect::<Vec<_>>(),
        workers,
    )?;
    let advanced_transfer = evaluate_candidates(
        root,
        manifest,
        dataset,
        data,
        "search",
        &bank.search,
        &transfer_first
            .iter()
            .map(|row| row.candidate.clone())
            .collect::<Vec<_>>(),
        workers,
    )?;
    let mut dl_ranked = advanced_exact
        .iter()
        .chain(&advanced_jittered)
        .cloned()
        .collect::<Vec<_>>();
    dl_ranked.sort_by(rank_loss);
    let mut dl_head_counts = HashMap::new();
    let mut transfer_head_counts = HashMap::new();
    let mut mixed_head_counts = HashMap::new();
    let head_family_cap = manifest
        .config
        .smart_v3
        .progressive_screen
        .max_per_terminal_head_family;
    let dl_population = select_with_head_cap(
        &dl_ranked,
        ISLAND_SIZE,
        &mut dl_head_counts,
        head_family_cap,
    )?;
    let transfer_population = select_with_head_cap(
        &advanced_transfer,
        ISLAND_SIZE,
        &mut transfer_head_counts,
        head_family_cap,
    )?;

    let mut mixed_candidates = Vec::new();
    let mut mixed_semantics = HashSet::new();
    for slot in 0..240 {
        let left = &dl_ranked[slot % dl_ranked.len()].candidate;
        let right = &advanced_transfer[(slot * 17 + 3) % advanced_transfer.len()].candidate;
        for retry in 0..100 {
            let seed = mixed_seed(&manifest.run_id, &dataset.dataset_hash, slot, retry);
            let Ok(genome) = crossover_canonical_pipelines(
                &left.canonical_genome,
                &right.canonical_genome,
                seed,
                SMART_MAX_TRANSFORMS,
            ) else {
                continue;
            };
            let mut roots = left.roots.clone();
            roots.extend(right.roots.iter().cloned());
            let mut origins = left.proposal_origins.clone();
            origins.extend(right.proposal_origins.iter().cloned());
            let mut decoder_seeds = left.decoder_seeds.clone();
            decoder_seeds.extend(right.decoder_seeds.iter().copied());
            let Ok(candidate) = candidate_from_parts(
                genome,
                roots,
                origins,
                decoder_seeds,
                json!({
                    "operation": "deterministic_dl_transfer_crossover",
                    "parents": [left.semantic_sha256, right.semantic_sha256],
                    "seed": seed,
                    "slot": slot,
                    "retry": retry,
                }),
                SMART_MAX_TRANSFORMS,
            ) else {
                continue;
            };
            if mixed_semantics.insert(candidate.semantic_sha256.clone()) {
                mixed_candidates.push(candidate);
            }
            break;
        }
    }
    if mixed_candidates.len() < 30 {
        bail!("cannot generate 30 unique deterministic mixed candidates");
    }
    let mixed_ranked = evaluate_candidates(
        root,
        manifest,
        dataset,
        data,
        "search",
        &bank.search,
        &mixed_candidates,
        workers,
    )?;
    let mixed_population = select_with_head_cap(
        &mixed_ranked,
        ISLAND_SIZE,
        &mut mixed_head_counts,
        head_family_cap,
    )?;
    let initial_populations = BTreeMap::from([
        ("dl".to_string(), dl_population),
        ("transfer".to_string(), transfer_population),
        ("mixed".to_string(), mixed_population),
    ]);
    validate_initial_populations(
        &initial_populations,
        ISLAND_SIZE,
        &["dl", "transfer", "mixed"],
        head_family_cap,
    )?;
    let screening = ScreeningReceipt {
        schema: "TrainOnlyRedteamSmartScreening.v1".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        exact_candidates: exact.len(),
        jittered_candidates: jittered.len(),
        transfer_candidates: transfer.len(),
        advanced_exact,
        advanced_jittered,
        advanced_transfer,
        initial_populations,
        development_only: true,
        confirmatory_valid: false,
    };
    let digest = write_receipt(&path, &screening)?;
    Ok((screening, digest, path))
}

fn nsga_order(population: &[RankedCandidate]) -> Vec<usize> {
    let dominates = |left: &RankedCandidate, right: &RankedCandidate| {
        left.robust_loss <= right.robust_loss
            && left.candidate.complexity <= right.candidate.complexity
            && (left.robust_loss < right.robust_loss
                || left.candidate.complexity < right.candidate.complexity)
    };
    let mut dominated = vec![Vec::new(); population.len()];
    let mut domination_count = vec![0usize; population.len()];
    for left in 0..population.len() {
        for right in 0..population.len() {
            if left == right {
                continue;
            }
            if dominates(&population[left], &population[right]) {
                dominated[left].push(right);
            } else if dominates(&population[right], &population[left]) {
                domination_count[left] += 1;
            }
        }
    }
    let mut fronts = Vec::new();
    let mut current = domination_count
        .iter()
        .enumerate()
        .filter_map(|(index, count)| (*count == 0).then_some(index))
        .collect::<Vec<_>>();
    while !current.is_empty() {
        current.sort_by(|left, right| rank_loss(&population[*left], &population[*right]));
        fronts.push(current.clone());
        let mut next = Vec::new();
        for index in current {
            for dominated_index in &dominated[index] {
                domination_count[*dominated_index] -= 1;
                if domination_count[*dominated_index] == 0 {
                    next.push(*dominated_index);
                }
            }
        }
        current = next;
    }
    let mut ordered = Vec::with_capacity(population.len());
    for mut front in fronts {
        let mut crowding = HashMap::<usize, f64>::new();
        for index in &front {
            crowding.insert(*index, 0.0);
        }
        for objective in 0..2 {
            front.sort_by(|left, right| {
                let left_value = if objective == 0 {
                    population[*left].robust_loss
                } else {
                    population[*left].candidate.complexity as f64
                };
                let right_value = if objective == 0 {
                    population[*right].robust_loss
                } else {
                    population[*right].candidate.complexity as f64
                };
                left_value.total_cmp(&right_value)
            });
            if front.len() > 1 {
                crowding.insert(front[0], f64::INFINITY);
                crowding.insert(*front.last().unwrap(), f64::INFINITY);
            }
            if front.len() > 2 {
                let first = if objective == 0 {
                    population[front[0]].robust_loss
                } else {
                    population[front[0]].candidate.complexity as f64
                };
                let last = if objective == 0 {
                    population[*front.last().unwrap()].robust_loss
                } else {
                    population[*front.last().unwrap()].candidate.complexity as f64
                };
                let scale = (last - first).abs().max(1e-15);
                for position in 1..(front.len() - 1) {
                    let before = if objective == 0 {
                        population[front[position - 1]].robust_loss
                    } else {
                        population[front[position - 1]].candidate.complexity as f64
                    };
                    let after = if objective == 0 {
                        population[front[position + 1]].robust_loss
                    } else {
                        population[front[position + 1]].candidate.complexity as f64
                    };
                    *crowding.get_mut(&front[position]).unwrap() += (after - before) / scale;
                }
            }
        }
        front.sort_by(|left, right| {
            crowding[right]
                .total_cmp(&crowding[left])
                .then_with(|| rank_loss(&population[*left], &population[*right]))
        });
        ordered.extend(front);
    }
    ordered
}

fn is_nondominated(population: &[RankedCandidate], index: usize) -> bool {
    let candidate = &population[index];
    !population.iter().any(|other| {
        other.robust_loss <= candidate.robust_loss
            && other.candidate.complexity <= candidate.candidate.complexity
            && (other.robust_loss < candidate.robust_loss
                || other.candidate.complexity < candidate.candidate.complexity)
    })
}

fn structural_niche(candidate: &Candidate) -> Result<String> {
    let parsed = GenomeIr::parse(&candidate.canonical_genome)?;
    let GenomeIr::Pipeline(pipeline) = parsed else {
        bail!("structural niche expects a pipeline");
    };
    Ok(format!(
        "{}|{}|{}",
        candidate.head_family,
        candidate.complexity,
        pipeline
            .stages
            .iter()
            .take(candidate.complexity)
            .map(|stage| stage.op.as_str())
            .collect::<Vec<_>>()
            .join(",")
    ))
}

fn smart_seed(
    run_id: &str,
    dataset_hash: &str,
    island: &str,
    generation: usize,
    slot: usize,
    retry: usize,
) -> u64 {
    let digest = Sha256::digest(
        format!(
            "smart-v3-offspring-v1\0{run_id}\0{dataset_hash}\0{island}\0{generation}\0{slot}\0{retry}"
        )
        .as_bytes(),
    );
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

fn mutation_arm(draw: u64) -> (&'static str, CanonicalMutationArm) {
    match draw % 100 {
        0..=29 => ("parameter_jitter", CanonicalMutationArm::ParameterJitter),
        30..=39 => ("insertion", CanonicalMutationArm::Insert),
        40..=59 => ("deletion", CanonicalMutationArm::Delete),
        60..=69 => ("adjacent_swap", CanonicalMutationArm::AdjacentSwap),
        70..=84 => ("head_replacement", CanonicalMutationArm::HeadReplacement),
        _ => ("simplification", CanonicalMutationArm::Simplify),
    }
}

fn smart_refill(
    manifest: &TrainOnlyRedteamManifest,
    dataset_hash: &str,
    island: &str,
    generation: usize,
    ranked: &[RankedCandidate],
    mixed_crossover_sources: Option<(&[RankedCandidate], &[RankedCandidate])>,
    history: &mut HashSet<String>,
) -> Result<(Vec<Candidate>, BTreeMap<String, usize>)> {
    let order = nsga_order(ranked);
    if order.len() != ISLAND_SIZE {
        bail!("Smart-v3 island population drift");
    }
    let mut next = order[..2]
        .iter()
        .map(|index| {
            let mut candidate = ranked[*index].candidate.clone();
            candidate.ancestry = json!({
                "operation": "elite",
                "parent": candidate.semantic_sha256,
                "generation": generation,
                "island": island,
            });
            candidate
        })
        .collect::<Vec<_>>();
    let mut generation_semantics = next
        .iter()
        .map(|candidate| candidate.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    let mut counts = BTreeMap::new();
    for slot in 2..ISLAND_SIZE {
        let mut accepted = None;
        for retry in 0..100_000 {
            let mut rng = DeterministicRng::new(smart_seed(
                &manifest.run_id,
                dataset_hash,
                island,
                generation,
                slot,
                retry,
            ));
            let tournament = |rng: &mut DeterministicRng| {
                let mut best = (rng.next_u64() as usize) % 15;
                for _ in 1..3 {
                    best = best.min((rng.next_u64() as usize) % 15);
                }
                order[best]
            };
            let left_index = tournament(&mut rng);
            let left = &ranked[left_index].candidate;
            let crossover = rng.next_u64().is_multiple_of(4);
            let (genome, roots, origins, seeds, ancestry, operator) = if crossover {
                let (crossover_left, right) = if island == "mixed" {
                    let (dl, transfer) = mixed_crossover_sources
                        .context("mixed island lacks DL×transfer crossover sources")?;
                    let dl_order = nsga_order(dl);
                    let transfer_order = nsga_order(transfer);
                    let dl_index = dl_order[(rng.next_u64() as usize) % 15];
                    let transfer_index = transfer_order[(rng.next_u64() as usize) % 15];
                    (&dl[dl_index].candidate, &transfer[transfer_index].candidate)
                } else {
                    let right_index = tournament(&mut rng);
                    (left, &ranked[right_index].candidate)
                };
                let Ok(genome) = crossover_canonical_pipelines(
                    &crossover_left.canonical_genome,
                    &right.canonical_genome,
                    rng.next_u64(),
                    SMART_MAX_TRANSFORMS,
                ) else {
                    continue;
                };
                let mut roots = crossover_left.roots.clone();
                roots.extend(right.roots.iter().cloned());
                let mut origins = crossover_left.proposal_origins.clone();
                origins.extend(right.proposal_origins.iter().cloned());
                let mut seeds = crossover_left.decoder_seeds.clone();
                seeds.extend(right.decoder_seeds.iter().copied());
                (
                    genome,
                    roots,
                    origins,
                    seeds,
                    json!({
                        "operation": if island == "mixed" { "dl_transfer_crossover" } else { "crossover" },
                        "parents": [crossover_left.semantic_sha256, right.semantic_sha256],
                        "generation": generation,
                        "island": island,
                        "slot": slot,
                        "retry": retry,
                    }),
                    "crossover",
                )
            } else {
                let (name, arm) = mutation_arm(rng.next_u64());
                let Ok(genome) = mutate_canonical_pipeline_with_arm(
                    &left.canonical_genome,
                    rng.next_u64(),
                    SMART_MAX_TRANSFORMS,
                    true,
                    arm,
                ) else {
                    continue;
                };
                (
                    genome,
                    left.roots.clone(),
                    left.proposal_origins.clone(),
                    left.decoder_seeds.clone(),
                    json!({
                        "operation": name,
                        "parent": left.semantic_sha256,
                        "generation": generation,
                        "island": island,
                        "slot": slot,
                        "retry": retry,
                    }),
                    name,
                )
            };
            let Ok(candidate) = candidate_from_parts(
                genome,
                roots,
                origins,
                seeds,
                ancestry,
                SMART_MAX_TRANSFORMS,
            ) else {
                continue;
            };
            if island == "dl" && candidate.roots != BTreeSet::from(["dl".to_string()])
                || island == "transfer"
                    && candidate.roots != BTreeSet::from(["transfer".to_string()])
                || island == "mixed"
                    && !(candidate.roots.contains("dl") && candidate.roots.contains("transfer"))
            {
                bail!("Smart-v3 island ancestry boundary drift");
            }
            if !generation_semantics.insert(candidate.semantic_sha256.clone())
                || !history.insert(candidate.semantic_sha256.clone())
            {
                generation_semantics.remove(&candidate.semantic_sha256);
                continue;
            }
            *counts.entry(operator.to_string()).or_default() += 1;
            accepted = Some(candidate);
            break;
        }
        next.push(accepted.context("cannot generate a unique Smart-v3 offspring")?);
    }
    Ok((next, counts))
}
