fn build_seed_bank(spec: &DatasetSpec) -> Result<SeedBank> {
    let mut seen = BTreeSet::new();
    let mut seeds = Vec::new();
    if let Ok(mut sources) = confirmed_transfer_winner_sources(&spec.dataset_hash) {
        admit_seed_sources(spec, std::mem::take(&mut sources), &mut seen, &mut seeds);
    }
    if seeds.len() < POPULATION {
        if let Ok(sources) = cached_fast30_bank_sources() {
            admit_seed_sources(spec, sources, &mut seen, &mut seeds);
        }
    }
    if seeds.len() < POPULATION {
        if let Ok(sources) = cached_binary_rewrite_sources() {
            admit_seed_sources(spec, sources, &mut seen, &mut seeds);
        }
    }
    if seeds.len() < POPULATION {
        if let Ok(sources) = cached_sotarena_scan_sources() {
            admit_seed_sources(spec, sources, &mut seen, &mut seeds);
        }
    }
    if seeds.len() < POPULATION {
        if let Ok(sources) = cached_local_flywheel_elite_sources() {
            admit_seed_sources(spec, sources, &mut seen, &mut seeds);
        }
    }
    if seeds.len() < POPULATION {
        admit_seed_sources(spec, cached_hand_authored_sources()?, &mut seen, &mut seeds);
    }
    if seeds.len() != POPULATION {
        bail!(
            "only {} valid, deduplicated TimeSeries42 generation-zero seeds were found for {}",
            seeds.len(),
            spec.dataset_hash
        );
    }
    Ok(SeedBank {
        schema: SEED_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        dataset_hash: spec.dataset_hash.clone(),
        population: POPULATION,
        selection:
            "deterministic provenance-ordered canonicalization, task admission, semantic dedup"
                .to_string(),
        seeds,
    })
}

fn load_or_create_seed_bank(root: &Path, spec: &DatasetSpec) -> Result<(SeedBank, String)> {
    let path = dataset_root(root, &spec.dataset_hash)
        .join("seeds")
        .join("generation-zero.v1.json");
    if path.is_file() {
        return read_seed_bank(&path, spec);
    }
    let bank = build_seed_bank(spec)?;
    let sha = write_json(&path, &bank)?;
    Ok((bank, sha))
}

fn read_seed_bank(path: &Path, spec: &DatasetSpec) -> Result<(SeedBank, String)> {
    let bytes = fs::read(path)?;
    let bank: SeedBank = serde_json::from_slice(&bytes)?;
    if bytes != canonical_json(&bank)?
        || bank.schema != SEED_SCHEMA
        || !bank.test_adaptive
        || bank.benchmark_valid
        || bank.dataset_hash != spec.dataset_hash
        || bank.population != POPULATION
        || bank.seeds.len() != POPULATION
        || bank
            .seeds
            .iter()
            .enumerate()
            .any(|(slot, seed)| seed.slot != slot)
    {
        bail!("invalid TimeSeries42 seed bank for {}", spec.dataset_hash);
    }
    Ok((bank, sha256_bytes(&bytes)))
}

fn load_existing_seed_bank(root: &Path, spec: &DatasetSpec) -> Result<(SeedBank, String)> {
    let path = dataset_root(root, &spec.dataset_hash)
        .join("seeds")
        .join("generation-zero.v1.json");
    if !path.is_file() {
        bail!(
            "base TimeSeries42 seed bank is missing for {}",
            spec.dataset_hash
        );
    }
    read_seed_bank(&path, spec)
}

fn seed_population_bindings(
    root: &Path,
    specs: &[DatasetSpec],
    create: bool,
) -> Result<Vec<SeedPopulationBinding>> {
    specs
        .iter()
        .map(|spec| {
            let path = dataset_root(root, &spec.dataset_hash)
                .join("seeds")
                .join("generation-zero.v1.json");
            let (_, seed_bank_sha256) = if create {
                load_or_create_seed_bank(root, spec)?
            } else {
                load_existing_seed_bank(root, spec)?
            };
            Ok(SeedPopulationBinding {
                dataset_hash: spec.dataset_hash.clone(),
                seed_bank_path: path.display().to_string(),
                seed_bank_sha256,
            })
        })
        .collect()
}
