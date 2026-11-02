use super::*;

pub(in crate::test_adaptive::binary42) fn lineage_for(
    checkpoints: &[GenerationCheckpoint],
    winner_semantic: &str,
) -> Result<Vec<LineageRecord>> {
    let mut members = BTreeMap::<String, MemberRecord>::new();
    for slot in checkpoints.iter().flat_map(|checkpoint| &checkpoint.slots) {
        members
            .entry(slot.member.semantic_sha256.clone())
            .or_insert_with(|| slot.member.clone());
    }
    fn visit(
        semantic: &str,
        members: &BTreeMap<String, MemberRecord>,
        visiting: &mut BTreeSet<String>,
        visited: &mut BTreeSet<String>,
        lineage: &mut Vec<LineageRecord>,
    ) -> Result<()> {
        if visited.contains(semantic) {
            return Ok(());
        }
        if !visiting.insert(semantic.to_string()) {
            bail!("Binary42 lineage contains a cycle");
        }
        let member = members
            .get(semantic)
            .with_context(|| format!("Binary42 lineage parent {semantic} is absent"))?;
        for parent in &member.ancestry.parents {
            if parent != semantic {
                visit(parent, members, visiting, visited, lineage)?;
            }
        }
        visiting.remove(semantic);
        visited.insert(semantic.to_string());
        lineage.push(LineageRecord {
            semantic_sha256: member.semantic_sha256.clone(),
            canonical_genome: member.canonical_genome.clone(),
            complexity: member.complexity,
            ancestry: member.ancestry.clone(),
        });
        Ok(())
    }
    let mut lineage = Vec::new();
    visit(
        winner_semantic,
        &members,
        &mut BTreeSet::new(),
        &mut BTreeSet::new(),
        &mut lineage,
    )?;
    Ok(lineage)
}

pub(in crate::test_adaptive::binary42) fn collect_artifacts(
    root: &Path,
) -> Result<Vec<WinnerArtifact>> {
    fn walk(root: &Path, current: &Path, artifacts: &mut Vec<WinnerArtifact>) -> Result<()> {
        let mut entries = fs::read_dir(current)?
            .map(|entry| Ok(entry?.path()))
            .collect::<Result<Vec<_>>>()?;
        entries.sort();
        for path in entries {
            if path.is_dir() {
                walk(root, &path, artifacts)?;
            } else {
                let relative = path.strip_prefix(root)?.to_string_lossy().into_owned();
                if relative == "RUN_COMPLETE" || relative == "winner/artifact-manifest.v1.json" {
                    continue;
                }
                artifacts.push(WinnerArtifact {
                    path: relative,
                    sha256: sha256_file(&path)?,
                });
            }
        }
        Ok(())
    }
    let mut artifacts = Vec::new();
    walk(root, root, &mut artifacts)?;
    artifacts.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(artifacts)
}

pub(in crate::test_adaptive::binary42) fn publish_winner(
    root: &Path,
    campaign: &CampaignManifest,
    checkpoints: &[GenerationCheckpoint],
    confirmation: &ConfirmationManifest,
) -> Result<()> {
    let checkpoint = checkpoints
        .last()
        .context("Binary42 winner publication has no checkpoint")?;
    let selected = checkpoint
        .qualifying_winner
        .as_ref()
        .context("Binary42 final checkpoint has no qualifying winner")?;
    let slot = checkpoint
        .slots
        .get(selected.slot)
        .context("Binary42 winner slot is absent")?;
    let winner_root = root.join("winner");
    let genome = GenomeObject {
        schema: GENOME_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        semantic_sha256: slot.member.semantic_sha256.clone(),
        canonical_genome: slot.member.canonical_genome.clone(),
        complexity: slot.member.complexity,
    };
    write_json(&winner_root.join("genome.v1.json"), &genome)?;
    let prediction_object = root
        .join("objects")
        .join("predictions")
        .join(format!("{}.csv", slot.prediction_sha256));
    immutable_or_equal(
        &winner_root.join("predictions.csv"),
        &fs::read(&prediction_object)?,
    )?;
    let lineage = lineage_for(checkpoints, &selected.semantic_sha256)?;
    let lineage_value = json!({
        "schema": "JopeBinary42TestAdaptiveLineage.v1",
        "test_adaptive": true,
        "benchmark_valid": false,
        "winner_semantic_sha256": selected.semantic_sha256,
        "records": lineage,
    });
    let lineage_sha256 = write_json(&winner_root.join("lineage.v1.json"), &lineage_value)?;
    let confirmation_attempt_root = winner_root
        .join("confirmations")
        .join(format!("attempt-{:03}", confirmation.attempt));
    let confirmation_sha256 = sha256_file(&confirmation_attempt_root.join("confirmation.v1.json"))?;
    let receipt = WinnerReceipt {
        schema: WINNER_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        generation: selected.generation,
        slot: selected.slot,
        semantic_sha256: selected.semantic_sha256.clone(),
        canonical_genome: slot.member.canonical_genome.clone(),
        complexity: selected.complexity,
        auc: selected.auc,
        logloss: selected.logloss,
        frozen_leader_auc: campaign.dataset_binding.leader_auc,
        generation_zero_best_auc: checkpoint.generation_zero_best_auc,
        margin_vs_frozen_leader: selected.auc - campaign.dataset_binding.leader_auc,
        margin_vs_generation_zero: selected.auc - checkpoint.generation_zero_best_auc,
        prediction_sha256: slot.prediction_sha256.clone(),
        source_receipt_sha256: slot.receipt_sha256.clone(),
        four_node_confirmation_sha256: confirmation_sha256,
        lineage_sha256,
        release_binary_sha256: campaign.release_binary_sha256.clone(),
    };
    let final_receipt_sha256 = write_json(&winner_root.join("final-receipt.v1.json"), &receipt)?;
    let artifact_manifest = WinnerArtifactManifest {
        schema: "JopeBinary42TestAdaptiveArtifactManifest.v1".to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        artifacts: collect_artifacts(root)?,
    };
    let artifact_manifest_sha256 = write_json(
        &winner_root.join("artifact-manifest.v1.json"),
        &artifact_manifest,
    )?;
    let complete = json!({
        "schema": COMPLETE_SCHEMA,
        "test_adaptive": true,
        "benchmark_valid": false,
        "status": "complete",
        "run_id": campaign.run_id,
        "dataset": campaign.dataset,
        "winner_semantic_sha256": selected.semantic_sha256,
        "winner_auc": selected.auc,
        "final_receipt_sha256": final_receipt_sha256,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "completed_at_unix_ms": unix_millis(),
    });
    write_json(&root.join("RUN_COMPLETE"), &complete)?;
    Ok(())
}

pub(in crate::test_adaptive::binary42) fn publish_exhausted(
    root: &Path,
    campaign: &CampaignManifest,
    checkpoints: &[GenerationCheckpoint],
) -> Result<()> {
    if checkpoints.len() != ALL_MAX_GENERATIONS
        || checkpoints
            .iter()
            .enumerate()
            .any(|(generation, checkpoint)| {
                checkpoint.generation != generation || checkpoint.qualifying_winner.is_some()
            })
    {
        bail!("Binary42 exhaustion requires exactly generations 0-9 with no qualifying winner");
    }
    if generation_root(root, ALL_MAX_GENERATIONS).exists() {
        bail!("Binary42 exhaustion found a forbidden generation 10 directory");
    }
    let mut ranked = checkpoints
        .iter()
        .flat_map(|checkpoint| &checkpoint.slots)
        .filter(|slot| slot.score.is_some())
        .cloned()
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        rank_slots(left, right)
            .then_with(|| left.generation.cmp(&right.generation))
            .then_with(|| left.slot.cmp(&right.slot))
    });
    let best = ranked
        .first()
        .context("Binary42 exhausted campaign has no successful candidate")?;
    let score = best.score.as_ref().unwrap();
    let best_effort = root.join("best-effort");
    let genome = GenomeObject {
        schema: GENOME_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        semantic_sha256: best.member.semantic_sha256.clone(),
        canonical_genome: best.member.canonical_genome.clone(),
        complexity: best.member.complexity,
    };
    write_json(&best_effort.join("genome.v1.json"), &genome)?;
    let prediction_object = root
        .join("objects")
        .join("predictions")
        .join(format!("{}.csv", best.prediction_sha256));
    immutable_or_equal(
        &best_effort.join("predictions.csv"),
        &fs::read(&prediction_object)?,
    )?;
    let lineage = lineage_for(checkpoints, &best.member.semantic_sha256)?;
    let lineage_value = json!({
        "schema": "JopeBinary42TestAdaptiveLineage.v1",
        "test_adaptive": true,
        "benchmark_valid": false,
        "winner_semantic_sha256": best.member.semantic_sha256,
        "records": lineage,
    });
    let lineage_sha256 = write_json(&best_effort.join("lineage.v1.json"), &lineage_value)?;
    let receipt = ExhaustedReceipt {
        schema: EXHAUSTED_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        status: "exhausted".to_string(),
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        completed_generations: checkpoints.len(),
        logical_slots: checkpoints.len() * POPULATION,
        best_generation: best.generation,
        best_slot: best.slot,
        best_semantic_sha256: best.member.semantic_sha256.clone(),
        best_auc: score.auc,
        best_logloss: score.logloss,
        best_prediction_sha256: best.prediction_sha256.clone(),
        best_receipt_sha256: best.receipt_sha256.clone(),
        lineage_sha256,
        release_binary_sha256: campaign.release_binary_sha256.clone(),
        completed_at_unix_ms: unix_millis(),
    };
    let receipt_sha256 = write_json(&best_effort.join("final-receipt.v1.json"), &receipt)?;
    let terminal = ExhaustedMarker {
        schema: EXHAUSTED_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        status: "exhausted".to_string(),
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        completed_generations: ALL_MAX_GENERATIONS,
        logical_slots: ALL_MAX_GENERATIONS * POPULATION,
        best_semantic_sha256: best.member.semantic_sha256.clone(),
        best_auc: score.auc,
        final_receipt_sha256: receipt_sha256,
        completed_at_unix_ms: unix_millis(),
    };
    write_json(&root.join("RUN_EXHAUSTED"), &terminal)?;
    Ok(())
}
