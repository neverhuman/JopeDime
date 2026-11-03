use super::*;

mod helpers;
use helpers::{
    baseline_metric, load_or_create_baselines, load_or_create_final_cv_reranks,
    optional_baseline_metric, prior_aspfm_metric, score_frozen_champion, stronger_tabfm,
};

pub(crate) fn score(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    // This is the only gate above any construction of a test or baseline path.
    let (root, manifest, _, validation, validation_digest) =
        require_validated_score_gate(study_root, run_id, sotarena_root, workers)?;
    // Freeze the independent train-only top-10 CV choice for every dataset
    // before baseline metrics or test paths are constructed.
    let final_cv_reranks = load_or_create_final_cv_reranks(
        &root,
        &manifest,
        &validation,
        &validation_digest,
        workers,
    )?;
    let (baselines, baseline_digest) = load_or_create_baselines(&root, run_id, &manifest.datasets)?;

    let mut posthoc_hashes = BTreeMap::new();
    for dataset in &manifest.datasets {
        let (cv_rerank, cv_rerank_digest) = final_cv_reranks
            .get(&dataset.dataset_hash)
            .context("frozen final CV rerank missing")?;
        let baseline = baselines
            .entries
            .iter()
            .find(|entry| entry.dataset_hash == dataset.dataset_hash)
            .context("pinned baseline entry missing")?;
        let (selection, selection_digest): (TrainOnlySelectionComplete, _) =
            read_receipt(&selection_path(&root, &dataset.dataset_hash))?;
        if validation
            .dataset_selection_sha256
            .get(&dataset.dataset_hash)
            != Some(&selection_digest)
        {
            bail!("selection is not covered by the validation receipt");
        }
        let train = read_regression_csv(&dataset.train_path)?;
        let test_path = sotarena_root
            .join("data/tabular-regression-v1")
            .join(&dataset.dataset_hash)
            .join("test.csv");
        let test_sha256 = sha256_file(&test_path)?;
        let test = read_regression_csv(&test_path)?;
        let output_path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("posthoc-score.v2.json");
        if output_path.exists() {
            let (existing, digest): (TrainOnlyPosthocScore, _) = read_receipt(&output_path)?;
            if existing.schema != "TrainOnlyPosthocScore.v2"
                || existing.run_id != run_id
                || existing.dataset_hash != dataset.dataset_hash
                || existing.validation_receipt_sha256 != validation_digest
                || existing.selection_complete_sha256 != selection_digest
                || existing.generation_checkpoint_sha256 != selection.generation_checkpoint_sha256
                || existing.baseline_manifest_sha256 != baseline_digest
                || existing.train_sha256 != dataset.train_sha256
                || existing.encoding_sha256 != dataset.encoding.context_sha256
                || existing.checkpoint_sha256 != manifest.checkpoint.checkpoint_sha256
                || existing.test_sha256 != test_sha256
                || existing.kernel_version != KERNEL_VERSION
                || existing.scores.len() != manifest.generations
                || existing.final_cv_rerank_sha256 != *cv_rerank_digest
                || existing.final_cv_rerank_score.semantic_sha256
                    != cv_rerank.selected_semantic_sha256
                || existing.prior_aspfm_rmse != baseline.prior_aspfm_rmse
                || existing.tabpfn_v3_rmse != baseline.tabpfn_v3_rmse
                || existing.tabfm_rmse != baseline.tabfm_rmse
                || existing.tabfm_ensemble_rmse != baseline.tabfm_ensemble_rmse
                || existing.stronger_tabfm_algorithm != baseline.stronger_tabfm_algorithm
                || existing.stronger_tabfm_rmse != baseline.stronger_tabfm_rmse
                || !existing.test_trajectory_post_selection
            {
                bail!("existing post-hoc score identity drift");
            }
            for (generation, recorded) in existing.scores.iter().enumerate() {
                let (checkpoint, checkpoint_digest): (TrainOnlyGenerationCheckpoint, _) =
                    read_receipt(&checkpoint_path(&root, &dataset.dataset_hash, generation))?;
                if checkpoint_digest != selection.generation_checkpoint_sha256[generation]
                    || recorded.generation != generation
                    || recorded.semantic_sha256 != selection.generation_champions[generation]
                {
                    bail!("existing post-hoc generation identity drift");
                }
                let member = checkpoint
                    .population
                    .iter()
                    .find(|member| member.member.semantic_sha256 == recorded.semantic_sha256)
                    .context("existing post-hoc champion is absent")?;
                let replay = score_frozen_champion(
                    &member.member,
                    &train,
                    &test,
                    dataset.features,
                    generation,
                )?;
                if recorded.canonical_genome != replay.canonical_genome
                    || recorded.rmse != replay.rmse
                    || recorded.r2 != replay.r2
                    || recorded.mae != replay.mae
                {
                    bail!("existing post-hoc metric replay drift");
                }
            }
            let final_rmse = existing.scores[manifest.generations - 1].rmse;
            let (final_checkpoint, _): (TrainOnlyGenerationCheckpoint, _) = read_receipt(
                &checkpoint_path(&root, &dataset.dataset_hash, manifest.generations - 1),
            )?;
            let cv_member = final_checkpoint
                .population
                .iter()
                .find(|candidate| {
                    candidate.member.semantic_sha256 == cv_rerank.selected_semantic_sha256
                })
                .context("existing final CV selection is absent from final population")?;
            let cv_replay = score_frozen_champion(
                &cv_member.member,
                &train,
                &test,
                dataset.features,
                manifest.generations - 1,
            )?;
            if existing.final_cv_rerank_score.canonical_genome != cv_replay.canonical_genome
                || existing.final_cv_rerank_score.rmse != cv_replay.rmse
                || existing.final_cv_rerank_score.r2 != cv_replay.r2
                || existing.final_cv_rerank_score.mae != cv_replay.mae
            {
                bail!("existing final CV test metric replay drift");
            }
            let cv_rmse = existing.final_cv_rerank_score.rmse;
            if existing.beats_tabpfn_v3 != (final_rmse < baseline.tabpfn_v3_rmse)
                || existing.beats_stronger_tabfm != (final_rmse < baseline.stronger_tabfm_rmse)
                || existing.beats_prior_aspfm != (final_rmse < baseline.prior_aspfm_rmse)
                || existing.beats_both
                    != (final_rmse < baseline.tabpfn_v3_rmse
                        && final_rmse < baseline.stronger_tabfm_rmse)
                || existing.cv_rerank_beats_tabpfn_v3 != (cv_rmse < baseline.tabpfn_v3_rmse)
                || existing.cv_rerank_beats_stronger_tabfm
                    != (cv_rmse < baseline.stronger_tabfm_rmse)
                || existing.cv_rerank_beats_prior_aspfm != (cv_rmse < baseline.prior_aspfm_rmse)
                || existing.cv_rerank_beats_both
                    != (cv_rmse < baseline.tabpfn_v3_rmse && cv_rmse < baseline.stronger_tabfm_rmse)
            {
                bail!("existing post-hoc strict-win comparison drift");
            }
            posthoc_hashes.insert(dataset.dataset_hash.clone(), digest);
            continue;
        }

        let mut scores = Vec::with_capacity(manifest.generations);
        for generation in 0..manifest.generations {
            let (checkpoint, checkpoint_digest): (TrainOnlyGenerationCheckpoint, _) =
                read_receipt(&checkpoint_path(&root, &dataset.dataset_hash, generation))?;
            if selection.generation_checkpoint_sha256[generation] != checkpoint_digest
                || selection.generation_champions[generation] != checkpoint.champion_semantic_sha256
            {
                bail!("frozen generation champion ledger drift");
            }
            let champion = checkpoint
                .population
                .iter()
                .find(|scored| scored.member.semantic_sha256 == checkpoint.champion_semantic_sha256)
                .context("frozen generation champion absent from checkpoint")?;
            scores.push(score_frozen_champion(
                &champion.member,
                &train,
                &test,
                dataset.features,
                generation,
            )?);
        }
        let final_score = scores.last().context("post-hoc score list is empty")?;
        let (final_checkpoint, _): (TrainOnlyGenerationCheckpoint, _) = read_receipt(
            &checkpoint_path(&root, &dataset.dataset_hash, manifest.generations - 1),
        )?;
        let cv_member = final_checkpoint
            .population
            .iter()
            .find(|candidate| {
                candidate.member.semantic_sha256 == cv_rerank.selected_semantic_sha256
            })
            .context("final CV selection is absent from final population")?;
        let final_cv_rerank_score = score_frozen_champion(
            &cv_member.member,
            &train,
            &test,
            dataset.features,
            manifest.generations - 1,
        )?;
        let receipt = TrainOnlyPosthocScore {
            schema: "TrainOnlyPosthocScore.v2".to_string(),
            run_id: run_id.to_string(),
            dataset_hash: dataset.dataset_hash.clone(),
            validation_receipt_sha256: validation_digest.clone(),
            selection_complete_sha256: selection_digest,
            generation_checkpoint_sha256: selection.generation_checkpoint_sha256,
            baseline_manifest_sha256: baseline_digest.clone(),
            train_sha256: dataset.train_sha256.clone(),
            encoding_sha256: dataset.encoding.context_sha256.clone(),
            checkpoint_sha256: manifest.checkpoint.checkpoint_sha256.clone(),
            test_sha256,
            kernel_version: KERNEL_VERSION.to_string(),
            final_cv_rerank_sha256: cv_rerank_digest.clone(),
            cv_rerank_beats_prior_aspfm: final_cv_rerank_score.rmse < baseline.prior_aspfm_rmse,
            cv_rerank_beats_tabpfn_v3: final_cv_rerank_score.rmse < baseline.tabpfn_v3_rmse,
            cv_rerank_beats_stronger_tabfm: final_cv_rerank_score.rmse
                < baseline.stronger_tabfm_rmse,
            cv_rerank_beats_both: final_cv_rerank_score.rmse < baseline.tabpfn_v3_rmse
                && final_cv_rerank_score.rmse < baseline.stronger_tabfm_rmse,
            final_cv_rerank_score,
            prior_aspfm_rmse: baseline.prior_aspfm_rmse,
            tabpfn_v3_rmse: baseline.tabpfn_v3_rmse,
            tabfm_rmse: baseline.tabfm_rmse,
            tabfm_ensemble_rmse: baseline.tabfm_ensemble_rmse,
            stronger_tabfm_algorithm: baseline.stronger_tabfm_algorithm.clone(),
            stronger_tabfm_rmse: baseline.stronger_tabfm_rmse,
            beats_prior_aspfm: final_score.rmse < baseline.prior_aspfm_rmse,
            beats_tabpfn_v3: final_score.rmse < baseline.tabpfn_v3_rmse,
            beats_stronger_tabfm: final_score.rmse < baseline.stronger_tabfm_rmse,
            beats_both: final_score.rmse < baseline.tabpfn_v3_rmse
                && final_score.rmse < baseline.stronger_tabfm_rmse,
            scores,
            test_trajectory_post_selection: true,
        };
        let digest = write_receipt(&output_path, &receipt)?;
        posthoc_hashes.insert(dataset.dataset_hash.clone(), digest);
    }
    let cumulative_sha256 = receipt_digest(&json!({
        "validation_receipt_sha256": validation_digest,
        "baseline_manifest_sha256": baseline_digest,
        "posthoc_score_sha256": posthoc_hashes,
    }))?;
    let index = TrainOnlyPosthocIndex {
        schema: "TrainOnlyPosthocIndex.v2".to_string(),
        run_id: run_id.to_string(),
        validation_receipt_sha256: validation_digest,
        baseline_manifest_sha256: baseline_digest,
        dataset_ids: manifest.selected_dataset_ids,
        posthoc_score_sha256: posthoc_hashes,
        cumulative_sha256,
    };
    let path = root.join("posthoc-scores.v2.json");
    write_receipt(&path, &index)?;
    Ok(path)
}
