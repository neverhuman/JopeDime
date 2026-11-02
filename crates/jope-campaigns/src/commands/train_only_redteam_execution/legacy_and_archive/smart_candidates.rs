use super::*;

pub(super) fn merge_candidate(
    candidates: &mut HashMap<String, Candidate>,
    mut candidate: Candidate,
) {
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

pub(super) fn load_smart_candidates(
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

pub(super) fn select_with_head_cap(
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

pub(super) fn validate_initial_populations(
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

pub(super) fn mixed_seed(run_id: &str, dataset_hash: &str, slot: usize, retry: usize) -> u64 {
    let digest = Sha256::digest(
        format!("smart-v3-mixed-init-v1\0{run_id}\0{dataset_hash}\0{slot}\0{retry}").as_bytes(),
    );
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}
