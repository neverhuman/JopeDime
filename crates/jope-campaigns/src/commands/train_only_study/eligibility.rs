use super::*;

mod context_slices;
pub(crate) use context_slices::copy_context_slices;

pub(super) type EligibleDatasets = (
    Vec<EligibilityRecord>,
    Vec<(CatalogEntry, RegressionData, String, SotarenaReportHashes)>,
);

pub(super) fn build_eligibility(
    run_id: &str,
    sotarena_root: &Path,
    catalog: &Catalog,
    exclusions: &CorpusExclusions,
    generations: usize,
    selected_ids: &[&str],
    expected_eligible: Option<usize>,
) -> Result<EligibleDatasets> {
    let entries = catalog
        .entries
        .iter()
        .filter(|entry| entry.task == "regression")
        .map(|entry| (entry.dataset_hash.as_str(), entry))
        .collect::<HashMap<_, _>>();
    let overlapping_ids = exclusions
        .dataset_ids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let overlapping_families = exclusions
        .source_families
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let overlapping_targets = exclusions
        .target_identities
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let mut records = Vec::with_capacity(GOLDEN42.len());
    let mut valid = Vec::new();
    for dataset in GOLDEN42 {
        let mut reasons = Vec::new();
        let mut source_family = None;
        let mut identity = None;
        let mut rows = None;
        let mut features = None;
        let mut train_sha256 = None;
        let mut context_sha256 = None;
        let mut checkpoint_exact_overlap = false;
        let mut checkpoint_source_family_overlap = false;
        let mut checkpoint_target_identity_overlap = false;
        let mut baseline_reports = None;
        if let Some(entry) = entries.get(dataset) {
            source_family = Some(entry.source_family.clone());
            context_sha256 = Some(entry.context_sha256.clone());
            let target = target_identity(sotarena_root, dataset)?;
            identity = Some(target.clone());
            checkpoint_exact_overlap = overlapping_ids.contains(dataset);
            checkpoint_source_family_overlap =
                overlapping_families.contains(entry.source_family.as_str());
            checkpoint_target_identity_overlap = overlapping_targets.contains(target.as_str());
            let train_path = sotarena_root
                .join("data/tabular-regression-v1")
                .join(dataset)
                .join("train.csv");
            match sha256_file(&train_path) {
                Ok(actual) if actual == entry.train_sha256 => {
                    train_sha256 = Some(actual);
                    match read_regression_csv(&train_path) {
                        Ok(data) => {
                            rows = Some(data.y.len());
                            features = Some(data.features);
                            reasons.extend(validate_train_data(&data));
                            if build_splits(
                                run_id,
                                dataset,
                                &entry.train_sha256,
                                &data.y,
                                generations,
                            )
                            .is_err()
                            {
                                reasons.push("invalid_or_nondistinct_splits".to_string());
                            }
                            match sotarena_report_hashes(sotarena_root, dataset) {
                                Ok(hashes) => baseline_reports = Some(hashes),
                                Err(_) => reasons
                                    .push("completed_required_baseline_reports_absent".to_string()),
                            }
                            if reasons.is_empty() {
                                let mut frozen_entry = (*entry).clone();
                                frozen_entry.train_csv = train_path;
                                valid.push((
                                    frozen_entry,
                                    data,
                                    target,
                                    baseline_reports.clone().unwrap(),
                                ));
                            }
                        }
                        Err(_) => reasons.push("training_csv_parse_failure".to_string()),
                    }
                }
                Ok(_) => reasons.push("training_csv_hash_drift".to_string()),
                Err(_) => reasons.push("training_csv_missing".to_string()),
            }
        } else {
            reasons.push("verified_encoding_absent".to_string());
        }
        reasons.sort();
        reasons.dedup();
        records.push(EligibilityRecord {
            dataset_hash: (*dataset).to_string(),
            source_family,
            target_identity: identity,
            rows,
            features,
            train_sha256,
            context_sha256,
            checkpoint_exact_overlap,
            checkpoint_source_family_overlap,
            checkpoint_target_identity_overlap,
            baseline_reports,
            selection_key: selection_key(dataset),
            eligible: reasons.is_empty(),
            reasons,
            selected: false,
        });
    }
    if expected_eligible.is_some_and(|expected| valid.len() != expected) {
        bail!(
            "Golden42 eligibility drift: expected {}, found {}",
            expected_eligible.unwrap(),
            valid.len()
        );
    }
    let by_id = valid
        .into_iter()
        .map(|row| (row.0.dataset_hash.clone(), row))
        .collect::<BTreeMap<_, _>>();
    let selected = selected_ids
        .iter()
        .map(|dataset| {
            by_id
                .get(*dataset)
                .cloned()
                .with_context(|| format!("frozen dataset is no longer eligible: {dataset}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let selected_ids = selected
        .iter()
        .map(|(entry, _, _, _)| entry.dataset_hash.as_str())
        .collect::<HashSet<_>>();
    for record in &mut records {
        record.selected = selected_ids.contains(record.dataset_hash.as_str());
    }
    Ok((records, selected))
}

pub(crate) fn proposal_score(row: &ProposalRow) -> Result<f64> {
    let score = row
        .selection_score
        .or(row.proposal_score)
        .or(row.reward_lcb)
        .context("DL proposal lacks reward-LCB/stability ranking evidence")?;
    if !score.is_finite()
        || row
            .instability_probability
            .is_some_and(|value| !value.is_finite())
    {
        bail!("DL proposal has non-finite ranking evidence");
    }
    Ok(score)
}

pub(super) fn load_dl_proposals(
    path: &Path,
    dataset_hash: &str,
    checkpoint_sha256: &str,
    context_sha256: &str,
    features: usize,
) -> Result<Vec<ProposalRow>> {
    let reader = BufReader::new(File::open(path)?);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let row: ProposalRow = serde_json::from_str(&line?)?;
        if row.task != "regression" || row.dataset_hash != dataset_hash {
            continue;
        }
        if row.model_checkpoint_sha256 != checkpoint_sha256
            || row.context_sha256.as_deref() != Some(context_sha256)
            || semantic_identity(&row.canonical_genome) != row.semantic_sha256
            || !genome_compatible(&row.canonical_genome, features)
        {
            continue;
        }
        proposal_score(&row)?;
        rows.push(row);
    }
    rows.sort_by(|left, right| {
        proposal_score(right)
            .unwrap()
            .total_cmp(&proposal_score(left).unwrap())
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    let mut semantics = HashSet::new();
    rows.retain(|row| semantics.insert(row.semantic_sha256.clone()));
    if rows.len() < 64 {
        bail!(
            "{dataset_hash} has {} valid unique checkpoint proposals; needs a 64-proposal reservoir",
            rows.len()
        );
    }
    rows.truncate(64);
    Ok(rows)
}

mod transfer;
pub(super) use transfer::build_transfer_rankings;

// Keep initial population identity, folds and search budget explicit.
#[allow(clippy::too_many_arguments)]
pub(super) fn generation_zero(
    run_id: &str,
    entry: &CatalogEntry,
    data: &RegressionData,
    encoding: &EncodingSlice,
    checkpoint: &CheckpointProvenance,
    proposals_path: &Path,
    proposals_sha256: &str,
    ranking: &GlobalRankingReceipt,
    ranking_sha256: &str,
) -> Result<TrainOnlyGenerationZero> {
    let proposals = load_dl_proposals(
        proposals_path,
        &entry.dataset_hash,
        &checkpoint.checkpoint_sha256,
        &entry.context_sha256,
        data.features,
    )?;
    let mut members = Vec::with_capacity(POPULATION);
    let mut used = HashSet::new();
    for proposal in proposals.into_iter().take(DL_QUOTA) {
        if !used.insert(proposal.semantic_sha256.clone()) {
            continue;
        }
        let score = proposal_score(&proposal)?;
        members.push(GenomeMember {
            slot: members.len(),
            complexity: genome_complexity(&GenomeIr::parse(&proposal.canonical_genome)?),
            canonical_genome: proposal.canonical_genome,
            semantic_sha256: proposal.semantic_sha256,
            source: "latest_checkpoint_dl".to_string(),
            source_evidence: json!({
                "reservoir_slot": proposal.slot,
                "ranking_score": score,
                "reward_lcb": proposal.reward_lcb,
                "instability_probability": proposal.instability_probability,
            }),
            ancestry: Ancestry {
                operation: "generation_zero_dl".to_string(),
                parents: Vec::new(),
                roots: BTreeSet::from(["dl".to_string()]),
                retry_count: 0,
            },
        });
    }
    if members.len() != DL_QUOTA {
        bail!("{} cannot fill the exact DL quota", entry.dataset_hash);
    }
    for ranked in &ranking.genomes {
        if members.len() == POPULATION {
            break;
        }
        if used.contains(&ranked.semantic_sha256)
            || semantic_identity(&ranked.canonical_genome) != ranked.semantic_sha256
            || !genome_compatible(&ranked.canonical_genome, data.features)
        {
            continue;
        }
        used.insert(ranked.semantic_sha256.clone());
        members.push(GenomeMember {
            slot: members.len(),
            complexity: genome_complexity(&GenomeIr::parse(&ranked.canonical_genome)?),
            canonical_genome: ranked.canonical_genome.clone(),
            semantic_sha256: ranked.semantic_sha256.clone(),
            source: "target_clean_cross_dataset_transfer".to_string(),
            source_evidence: serde_json::to_value(ranked)?,
            ancestry: Ancestry {
                operation: "generation_zero_transfer".to_string(),
                parents: Vec::new(),
                roots: BTreeSet::from(["transfer".to_string()]),
                retry_count: 0,
            },
        });
    }
    if members.len() != POPULATION {
        bail!(
            "{} cannot fill the exact target-clean transfer quota after compatibility and deduplication",
            entry.dataset_hash
        );
    }
    Ok(TrainOnlyGenerationZero {
        schema: "TrainOnlyGenerationZero.v2".to_string(),
        run_id: run_id.to_string(),
        dataset_hash: entry.dataset_hash.clone(),
        train_sha256: entry.train_sha256.clone(),
        encoding_sha256: encoding.context_sha256.clone(),
        checkpoint_sha256: checkpoint.checkpoint_sha256.clone(),
        dl_proposals_sha256: proposals_sha256.to_string(),
        transfer_ranking_sha256: ranking_sha256.to_string(),
        members,
    })
}
