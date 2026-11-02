use super::*;

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

pub(super) fn copy_context_slices(
    source_blob: &Path,
    output_blob: &Path,
    selected: &[(CatalogEntry, RegressionData, String, SotarenaReportHashes)],
) -> Result<(String, BTreeMap<String, EncodingSlice>)> {
    let mut source = File::open(source_blob)?;
    let mut bytes = Vec::with_capacity(selected.len() * CONTEXT_BYTES);
    let mut slices = BTreeMap::new();
    for (slot, (entry, _, _, _)) in selected.iter().enumerate() {
        let source_offset = entry
            .catalog_index
            .checked_mul(CONTEXT_BYTES)
            .context("context offset overflow")?;
        source.seek(SeekFrom::Start(source_offset as u64))?;
        let mut slice = vec![0_u8; CONTEXT_BYTES];
        source.read_exact(&mut slice)?;
        if sha256_bytes(&slice) != entry.context_sha256 {
            bail!("encoding slice drift for {}", entry.dataset_hash);
        }
        let output_offset = slot * CONTEXT_BYTES;
        bytes.extend_from_slice(&slice);
        slices.insert(
            entry.dataset_hash.clone(),
            EncodingSlice {
                blob_path: absolute_path(output_blob)?,
                blob_sha256: String::new(),
                offset_bytes: output_offset,
                length_bytes: CONTEXT_BYTES,
                context_sha256: entry.context_sha256.clone(),
            },
        );
    }
    let digest = sha256_bytes(&bytes);
    immutable_write(output_blob, &bytes)?;
    for slice in slices.values_mut() {
        slice.blob_sha256 = digest.clone();
    }
    Ok((digest, slices))
}

pub(super) fn proposal_score(row: &ProposalRow) -> Result<f64> {
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

pub(super) fn build_transfer_rankings(
    run_id: &str,
    observation_db: &Path,
    observation_db_sha256: &str,
    catalog: &Catalog,
    catalog_sha256: &str,
    target_entries: &[CatalogEntry],
) -> Result<BTreeMap<String, GlobalRankingReceipt>> {
    let targets = target_entries.to_vec();
    let family_by_dataset = catalog
        .entries
        .iter()
        .map(|entry| (entry.dataset_hash.clone(), entry.source_family.clone()))
        .collect::<HashMap<_, _>>();
    let connection = Connection::open_with_flags(
        observation_db,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.execute_batch(
        "CREATE TEMP TABLE dataset_family(
            dataset_hash TEXT PRIMARY KEY,
            source_family TEXT NOT NULL
        ) WITHOUT ROWID;",
    )?;
    {
        let mut insert = connection
            .prepare("INSERT INTO dataset_family(dataset_hash, source_family) VALUES (?1, ?2)")?;
        for entry in &catalog.entries {
            insert.execute([&entry.dataset_hash, &entry.source_family])?;
        }
    }
    connection.execute_batch(
        "CREATE TEMP TABLE transfer_targets(
            dataset_hash TEXT PRIMARY KEY,
            source_family TEXT NOT NULL
        ) WITHOUT ROWID;",
    )?;
    {
        let mut insert = connection
            .prepare("INSERT INTO transfer_targets(dataset_hash, source_family) VALUES (?1, ?2)")?;
        for target in &targets {
            insert.execute([&target.dataset_hash, &target.source_family])?;
        }
    }

    let mut ranked = targets
        .iter()
        .map(|target| {
            (
                target.dataset_hash.clone(),
                Vec::<GlobalRankedGenome>::new(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut statement =
        connection.prepare(include_str!("../../sql/train_only_transfer_rankings.sql"))?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let target_dataset_hash: String = row.get(0)?;
        let semantic_sha256: String = row.get(1)?;
        let canonical_genome: String = row.get(2)?;
        let source_ids = row
            .get::<_, String>(3)?
            .split(',')
            .map(str::to_string)
            .collect::<Vec<_>>();
        let mean_percentile: Option<f64> = row.get(4)?;
        let worst_percentile: Option<f64> = row.get(5)?;
        let strict_win_count = row.get::<_, i64>(6)? as usize;
        let dataset_count = row.get::<_, i64>(7)? as usize;
        let (Some(mean_percentile), Some(worst_percentile)) = (mean_percentile, worst_percentile)
        else {
            continue;
        };
        let mut included_ids = source_ids;
        included_ids.sort();
        included_ids.dedup();
        let source_families = included_ids
            .iter()
            .map(|dataset| {
                family_by_dataset
                    .get(dataset)
                    .cloned()
                    .unwrap_or_else(|| format!("dataset:{dataset}"))
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let Some(target_rankings) = ranked.get_mut(&target_dataset_hash) else {
            bail!("transfer ranking query returned unexpected target {target_dataset_hash}");
        };
        target_rankings.push(GlobalRankedGenome {
            rank: 0,
            canonical_genome: canonical_genome.clone(),
            semantic_sha256: semantic_sha256.clone(),
            mean_percentile,
            worst_percentile,
            strict_win_count,
            dataset_count,
            source_dataset_ids: included_ids,
            source_families,
        });
    }

    let mut receipts = BTreeMap::new();
    for target in targets {
        let mut genomes = ranked.remove(&target.dataset_hash).unwrap();
        genomes.sort_by(|left, right| {
            right
                .mean_percentile
                .total_cmp(&left.mean_percentile)
                .then_with(|| right.worst_percentile.total_cmp(&left.worst_percentile))
                .then_with(|| right.strict_win_count.cmp(&left.strict_win_count))
                .then_with(|| right.dataset_count.cmp(&left.dataset_count))
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        });
        genomes.truncate(50_000);
        for (rank, genome) in genomes.iter_mut().enumerate() {
            genome.rank = rank;
        }
        if genomes.len() < GLOBAL_QUOTA {
            bail!(
                "{} has fewer than {GLOBAL_QUOTA} target-clean transfer genomes",
                target.dataset_hash
            );
        }
        let excluded_dataset_ids = catalog
            .entries
            .iter()
            .filter(|entry| {
                entry.dataset_hash == target.dataset_hash
                    || entry.source_family == target.source_family
            })
            .map(|entry| entry.dataset_hash.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        receipts.insert(
            target.dataset_hash.clone(),
            GlobalRankingReceipt {
                schema: "TrainOnlyTransferGenomeRanking.v2".to_string(),
                run_id: run_id.to_string(),
                target_dataset_id: target.dataset_hash,
                target_source_family: target.source_family.clone(),
                observation_db_sha256: observation_db_sha256.to_string(),
                source_catalog_sha256: catalog_sha256.to_string(),
                excluded_dataset_ids,
                excluded_source_family: target.source_family,
                direct_dl_sources_excluded: true,
                ranking_policy:
                    "target and matching source family excluded; mean within-dataset percentile desc; worst percentile desc; strict win count desc; dataset count desc; semantic SHA-256 asc"
                        .to_string(),
                genomes,
            },
        );
    }
    Ok(receipts)
}

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
