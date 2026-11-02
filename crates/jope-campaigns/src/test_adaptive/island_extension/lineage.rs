use super::*;

pub(super) fn load_island_cache(
    root: &Path,
    checkpoints: &[(IslandGenerationCheckpoint, String)],
) -> Result<HashMap<String, CachedEvaluation>> {
    let mut cache = HashMap::new();
    for (checkpoint, _) in checkpoints {
        for outcome in checkpoint.islands.iter().flat_map(|island| &island.slots) {
            if outcome.cache_source.is_some() {
                continue;
            }
            let bytes = fs::read(root.join(&outcome.receipt_file))?;
            let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
            cache.insert(
                outcome.member.semantic_sha256.clone(),
                CachedEvaluation {
                    source: CacheSource {
                        generation: outcome.generation,
                        slot: outcome.global_slot,
                    },
                    receipt_file: outcome.receipt_file.clone(),
                    receipt_sha256: outcome.receipt_sha256.clone(),
                    receipt,
                },
            );
        }
    }
    Ok(cache)
}

pub(super) fn reconstruct_island_lineage(
    combined: &[(GenerationCheckpoint, String)],
    checkpoints: &[(IslandGenerationCheckpoint, String)],
    selected: &IslandSlotOutcome,
) -> Result<Vec<IslandLineageRecord>> {
    fn visit_old(
        combined: &[(GenerationCheckpoint, String)],
        generation: usize,
        slot: usize,
        seen: &mut BTreeSet<IslandParentRef>,
        output: &mut Vec<IslandLineageRecord>,
    ) -> Result<()> {
        let outcome = combined
            .get(generation)
            .and_then(|checkpoint| checkpoint.0.slots.get(slot))
            .context("old island-lineage location is absent")?;
        let key = IslandParentRef {
            generation,
            island: None,
            local_slot: slot,
            global_slot: slot,
            semantic_sha256: outcome.member.semantic_sha256.clone(),
        };
        if !seen.insert(key) {
            return Ok(());
        }
        let mut parents = Vec::new();
        for hash in &outcome.member.ancestry.parents {
            let location = (0..generation).rev().find_map(|parent_generation| {
                combined[parent_generation]
                    .0
                    .slots
                    .iter()
                    .find(|candidate| candidate.member.semantic_sha256 == *hash)
                    .map(|candidate| (parent_generation, candidate.slot))
            });
            if let Some((parent_generation, parent_slot)) = location {
                visit_old(combined, parent_generation, parent_slot, seen, output)?;
                parents.push(IslandParentRef {
                    generation: parent_generation,
                    island: None,
                    local_slot: parent_slot,
                    global_slot: parent_slot,
                    semantic_sha256: hash.clone(),
                });
            } else if outcome.member.ancestry.operation != "seed" {
                bail!("old island-lineage parent is absent");
            }
        }
        output.push(IslandLineageRecord {
            generation,
            island: None,
            local_slot: slot,
            global_slot: slot,
            semantic_sha256: outcome.member.semantic_sha256.clone(),
            operation: outcome.member.ancestry.operation.clone(),
            parents,
        });
        Ok(())
    }

    fn visit_island(
        combined: &[(GenerationCheckpoint, String)],
        checkpoints: &[(IslandGenerationCheckpoint, String)],
        reference: &IslandParentRef,
        seen: &mut BTreeSet<IslandParentRef>,
        output: &mut Vec<IslandLineageRecord>,
    ) -> Result<()> {
        if reference.generation < START_GENERATION {
            return visit_old(
                combined,
                reference.generation,
                reference.local_slot,
                seen,
                output,
            );
        }
        if !seen.insert(reference.clone()) {
            return Ok(());
        }
        let island = reference
            .island
            .context("island lineage reference has no island")?;
        let outcome = checkpoints
            .get(reference.generation - START_GENERATION)
            .and_then(|checkpoint| checkpoint.0.islands.get(island))
            .and_then(|state| state.slots.get(reference.local_slot))
            .context("island lineage reference is absent")?;
        if outcome.global_slot != reference.global_slot
            || outcome.member.semantic_sha256 != reference.semantic_sha256
        {
            bail!("island lineage reference does not match its outcome");
        }
        for parent in &outcome.parents {
            visit_island(combined, checkpoints, parent, seen, output)?;
        }
        output.push(IslandLineageRecord {
            generation: outcome.generation,
            island: Some(outcome.island),
            local_slot: outcome.local_slot,
            global_slot: outcome.global_slot,
            semantic_sha256: outcome.member.semantic_sha256.clone(),
            operation: outcome.member.ancestry.operation.clone(),
            parents: outcome.parents.clone(),
        });
        Ok(())
    }

    let selected_ref = island_parent_ref(selected);
    let mut output = Vec::new();
    visit_island(
        combined,
        checkpoints,
        &selected_ref,
        &mut BTreeSet::new(),
        &mut output,
    )?;
    output.sort_by(|left, right| {
        left.generation
            .cmp(&right.generation)
            .then_with(|| left.island.cmp(&right.island))
            .then_with(|| left.local_slot.cmp(&right.local_slot))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    Ok(output)
}

pub(super) fn island_genome_record(
    root: &Path,
    combined: &[(GenerationCheckpoint, String)],
    checkpoints: &[(IslandGenerationCheckpoint, String)],
    outcome: &IslandSlotOutcome,
    g0_rmse: f64,
    leader_rmse: f64,
    binary_sha256: &str,
) -> Result<IslandGenomeRecord> {
    let receipt_bytes = fs::read(root.join(&outcome.receipt_file))?;
    if sha256_bytes(&receipt_bytes) != outcome.receipt_sha256 {
        bail!("island genome receipt hash mismatch");
    }
    let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
    let metrics = outcome
        .metrics
        .as_ref()
        .context("selected island genome is invalid")?;
    Ok(IslandGenomeRecord {
        canonical_genome: outcome.member.canonical_genome.clone(),
        semantic_sha256: outcome.member.semantic_sha256.clone(),
        generation: outcome.generation,
        island: outcome.island,
        local_slot: outcome.local_slot,
        global_slot: outcome.global_slot,
        rmse: metrics.rmse,
        mae: metrics.mae,
        r2: metrics.r2,
        margin_vs_g0: g0_rmse - metrics.rmse,
        margin_vs_leader: leader_rmse - metrics.rmse,
        complexity: outcome.member.complexity,
        rust_source_len: outcome.member.rust_source_len,
        runtime_seconds: receipt.runtime_seconds,
        lineage: reconstruct_island_lineage(combined, checkpoints, outcome)?,
        prediction_sha256: receipt.predictions_sha256,
        receipt_file: outcome.receipt_file.clone(),
        receipt_sha256: outcome.receipt_sha256.clone(),
        binary_sha256: binary_sha256.to_string(),
    })
}

pub(super) fn find_island_outcome<'a>(
    checkpoint: &'a IslandGenerationCheckpoint,
    location: &IslandGenomeLocation,
) -> Result<&'a IslandSlotOutcome> {
    checkpoint
        .islands
        .get(location.island)
        .and_then(|state| state.slots.get(location.local_slot))
        .filter(|outcome| {
            outcome.generation == location.generation
                && outcome.global_slot == location.global_slot
                && outcome.member.semantic_sha256 == location.semantic_sha256
        })
        .context("island genome location is absent")
}

pub(super) fn collect_qualifying_records(
    root: &Path,
    combined: &[(GenerationCheckpoint, String)],
    checkpoints: &[(IslandGenerationCheckpoint, String)],
    spec: &DatasetSpec,
    binary_sha256: &str,
) -> Result<Vec<IslandGenomeRecord>> {
    let g0_rmse = combined[0].0.g0_best_rmse;
    let mut records = Vec::new();
    for (checkpoint, _) in checkpoints {
        for location in &checkpoint.qualifying_genomes {
            records.push(island_genome_record(
                root,
                combined,
                checkpoints,
                find_island_outcome(checkpoint, location)?,
                g0_rmse,
                spec.leader_rmse,
                binary_sha256,
            )?);
        }
    }
    Ok(records)
}

pub(super) fn replay_island_champion(
    worker: &CampaignIdentity,
    registry: &Path,
    campaign_root: &Path,
    scq_binary_sha256: &str,
    generation: usize,
    champion: &Member,
    test: &RegressionData,
) -> Result<Vec<ReplayRecord>> {
    let evaluations = scq_batch::replay_across_nodes(
        campaign_root,
        worker,
        registry,
        &worker.binary_sha256,
        scq_binary_sha256,
        generation,
        champion,
        test,
    )?;
    let baseline = evaluations
        .first()
        .context("cross-node replay produced no receipts")?;
    let baseline_rmse = baseline
        .receipt
        .metrics
        .as_ref()
        .context("cross-node replay failed")?
        .rmse;
    let baseline_predictions = baseline.receipt.predictions.clone();
    evaluations
        .into_iter()
        .map(|evaluation| {
            let rmse = evaluation
                .receipt
                .metrics
                .as_ref()
                .context("cross-node replay failed")?
                .rmse;
            let maximum_prediction_difference = baseline_predictions
                .iter()
                .zip(&evaluation.receipt.predictions)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0, f64::max);
            if (rmse - baseline_rmse).abs() > 1e-12 || maximum_prediction_difference > 1e-10 {
                bail!("cross-node replay disagreement");
            }
            Ok(ReplayRecord {
                receipt_file: evaluation.receipt_file,
                receipt_sha256: evaluation.receipt_sha256,
                rmse,
                maximum_prediction_difference,
            })
        })
        .collect()
}
