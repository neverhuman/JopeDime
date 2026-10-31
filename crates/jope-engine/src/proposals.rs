//! Freeze a pure training-context prediction, then assemble the adaptive GP population.
use crate::{
    corpus::ContextRow,
    gp::{SeedCandidate, SeedPopulation},
    grammar::{State, Vocab},
    manifest::*,
    neural::{Bf16, Config, Policy},
    prepare::Prepared,
    proposal_population::{self, PURE_PROTOCOL},
    service::{GpJob, GpJobs},
    telemetry::now,
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use tch::{nn, Device, Kind, Tensor};

pub fn rank(
    root: &Path,
    corpus: &Path,
    checkpoint: &Path,
    output: &Path,
    rank: usize,
    world: usize,
    cuts: usize,
    cut_start: usize,
    shuffled: bool,
) -> Result<()> {
    crate::task_contract::require_original_gpu_grant("proposal-rank")?;
    let task = crate::gpu_worker::Task::Proposal {
        corpus: corpus.into(),
        checkpoint: checkpoint.into(),
        output: output.into(),
        world,
        cuts,
        cut_start,
        shuffled,
    };
    let (_lease, _contract) = crate::gpu_worker::verify_proposal_launch(root, &task)?;
    if crate::gpu_worker::enabled() {
        return crate::gpu_worker::submit(
            root,
            rank,
            crate::gpu_worker::Task::Proposal {
                corpus: corpus.into(),
                checkpoint: checkpoint.into(),
                output: output.into(),
                world,
                cuts,
                cut_start,
                shuffled,
            },
        );
    }
    crate::gpu_worker::require_accepted_task(root, &task)?;
    let champion_snapshot = PathBuf::from(
        std::env::var_os("JOPE_CHAMPION_SNAPSHOT")
            .context("proposal rank requires the shared champion snapshot")?,
    );
    rank_impl(
        root,
        corpus,
        checkpoint,
        output,
        &champion_snapshot,
        rank,
        world,
        cuts,
        cut_start,
        shuffled,
        None,
        None,
    )
}

pub(crate) fn smartcluster_rank(
    task: &crate::smartcluster_worker::VerifiedTask,
    receipt: &Path,
) -> Result<()> {
    let crate::smartcluster_worker::Operation::Proposal {
        corpus,
        checkpoint,
        output,
        cuts,
        cut_start,
        shuffled,
        champion_snapshot,
        preparation_overlay,
    } = task.operation()
    else {
        anyhow::bail!("proposal operation required")
    };
    ensure!(
        !task.stop().load(std::sync::atomic::Ordering::Acquire),
        "native control already lost"
    );
    let root = task.root();
    let output = resolve(root, output)?;
    let champion_snapshot = resolve(root, champion_snapshot)?;
    rank_impl(
        root,
        &resolve(root, corpus)?,
        &resolve(root, checkpoint)?,
        &output,
        &champion_snapshot,
        0,
        1,
        *cuts,
        *cut_start,
        *shuffled,
        preparation_overlay
            .as_ref()
            .map(|a| (a, task.manifest().run.as_str())),
        None,
    )?;
    let result: Value = read_json(&output.join("complete-0.json"))?;
    write_json(
        receipt,
        &serde_json::json!({"schema":"jope.smartcluster_proposal_completion.v1","numerical_receipt":result,"artifacts":crate::smartcluster_worker::hash_directory(&output)?}),
    )
}

fn rank_impl(
    root: &Path,
    corpus: &Path,
    checkpoint: &Path,
    output: &Path,
    champion_path: &Path,
    rank: usize,
    world: usize,
    cuts: usize,
    cut_start: usize,
    shuffled: bool,
    overlay: Option<(&crate::native_graph::Artifact, &str)>,
    assignment: Option<&crate::training::topology::Assignment>,
) -> Result<()> {
    ensure!(
        rank < world && world <= 8 && cuts > 0,
        "invalid proposal worker topology"
    );
    // W7-T03 follow-up: the communicator rank shards the jobs; the CUDA
    // ordinal comes from the binding. No caller delivers an assignment yet,
    // so the labelled legacy mapping applies and is recorded.
    let binding =
        crate::training::topology::resolve(assignment, rank, world, crate::neural::local_devices)?;
    crate::neural::configure_runtime();
    let index = crate::teaching_store::read_index(root, corpus)?;
    let run: Value = read_json(&output.join("run.json"))?;
    let proposal_seed = run["proposal_seed"].as_u64().unwrap_or(SEED);
    let state: Value = read_json(&checkpoint.join("state.json"))?;
    ensure!(
        file_hash(&checkpoint.join("model.ot"))? == state["model_sha256"]
            && file_hash(&checkpoint.join("config.json"))? == state["config_sha256"]
            && file_hash(&corpus.join("index.json"))? == state["corpus_sha256"]
            && state["autocast_cache_lifetime"] == "scoped_v2"
            && file_hash(&corpus.join("vocabulary.json"))? == state["vocabulary_sha256"],
        "proposal checkpoint identity mismatch"
    );
    // W1 route R8: the vocabulary consumed is the one the checkpoint binds,
    // read once (the check above hashed it by path).
    let vocab: Vocab = crate::corpus::read_bound_vocabulary(
        corpus,
        state["vocabulary_sha256"]
            .as_str()
            .context("proposal checkpoint vocabulary absent")?,
    )?;
    let config: Config = read_json(&checkpoint.join("config.json"))?;
    ensure!(
        config.vocab_size == vocab.tokens.len() as i64,
        "proposal vocabulary mismatch"
    );
    let device = binding.cuda_device();
    let mut vs = nn::VarStore::new(device);
    let model = Policy::new(vs.root(), config)?;
    // W2-T04 (G-10c): exactly the model bytes the checkpoint state binds.
    crate::neural::load_bound_model(
        &mut vs,
        &checkpoint.join("model.ot"),
        &state["model_sha256"],
        "proposal checkpoint identity mismatch",
    )?;
    let programs = crate::teaching_store::teacher_programs(root, corpus, &index, 10)?;
    let selection_path = output
        .parent()
        .context("proposal parent absent")?
        .join("coverage-selection.json");
    let selected = if let Some(expected) = run["coverage_selection_sha256"].as_str() {
        ensure!(
            file_hash(&selection_path)? == expected,
            "proposal coverage selection changed"
        );
        if overlay.is_some() {
            let selection: Value = read_json(&selection_path)?;
            ensure!(
                selection["request"]["preparation_overlay"]
                    == serde_json::to_value(overlay.map(|(a, _)| a))?,
                "proposal operation and selected overlay differ"
            );
        }
        crate::service::coverage::selected_indices(&selection_path, corpus, &index.contexts)?
    } else {
        ensure!(!selection_path.exists(), "unbound coverage selection");
        index
            .contexts
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.partition != "meta_test" && c.cut >= cut_start && c.cut - cut_start < cuts
            })
            .map(|(i, _)| i)
            .collect()
    };
    let allowed = overlay
        .map(|(a, run)| crate::preparation_overlay::load(root, a, run))
        .transpose()?;
    if let Some(v) = &allowed {
        for &i in &selected {
            let c = &index.contexts[i];
            let p: crate::prepare::Prepared =
                read_json(&resolve(root, &c.prepared)?.join("prepared.json"))?;
            v.verify_context(root, c, &p)?;
        }
    }
    let allowed_views = allowed.as_ref().map(|v| v.views());
    let jobs = selected
        .iter()
        .map(|&i| (i, &index.contexts[i]))
        .collect::<Vec<_>>();
    let donors = index
        .contexts
        .iter()
        .filter(|c| {
            c.partition == "meta_train"
                && allowed_views
                    .as_ref()
                    .is_none_or(|v| v.contains(&(c.dataset.clone(), c.cut)))
        })
        .collect::<Vec<_>>();
    if let Some(v) = &allowed {
        for c in &donors {
            let p: crate::prepare::Prepared =
                read_json(&resolve(root, &c.prepared)?.join("prepared.json"))?;
            v.verify_context(root, c, &p)?;
        }
    }
    let mut donor_order = (0..donors.len()).collect::<Vec<_>>();
    donor_order.sort_by_key(|&i| crate::training::mix(SEED ^ i as u64 ^ 0x61626c617465));
    ensure!(donors.len() >= jobs.len(), "shuffled donor pool too small");
    for (j, (_, c)) in jobs.iter().enumerate() {
        let k = (j..donor_order.len())
            .find(|&k| donors[donor_order[k]].family != c.family)
            .context("no distinct shuffled family")?;
        donor_order.swap(j, k);
    }
    let mut generated_jobs = Vec::new();
    let started = std::time::Instant::now();
    let checkpoint_sha = file_hash(&checkpoint.join("state.json"))?;
    let champion_sha = file_hash(champion_path)?;
    let champion_snapshot: crate::champions::Snapshot = read_json(&champion_path)?;
    ensure!(
        champion_snapshot.schema == "parallel.champion_snapshot.v1"
            && champion_snapshot.request["corpus_sha256"] == state["corpus_sha256"]
            && champion_snapshot.request["checkpoint_sha256"] == checkpoint_sha
            && champion_snapshot.request["cuts"] == cuts
            && champion_snapshot.request["cut_start"] == cut_start
            && champion_snapshot.request["coverage_selection_sha256"]
                == run["coverage_selection_sha256"]
            && champion_snapshot.vocabulary_sha256 == state["vocabulary_sha256"],
        "proposal champion snapshot changed"
    );
    for (j, (context_id, c)) in jobs.iter().enumerate().filter(|(j, _)| binding.owns(*j)) {
        let destination = output.join(&c.dataset).join(format!("cut-{}", c.cut));
        fs::create_dir_all(&destination)?;
        ensure!(
            !destination.join("ranking.json").exists()
                && !destination.join("best-predicted.json").exists(),
            "legacy filtered-pool proposal artifacts require a distinct output directory"
        );
        let actual: &ContextRow = if shuffled { donors[donor_order[j]] } else { c };
        let bytes = fs::read(resolve(root, &actual.path)?)?;
        ensure!(
            hash(&bytes) == actual.sha256 && bytes.len() == 8192,
            "conditioning context changed"
        );
        let values = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let p: Prepared = read_json(&resolve(root, &c.prepared)?.join("prepared.json"))?;
        let cut = p
            .cuts
            .iter()
            .find(|v| v.index == c.cut)
            .context("cut missing")?;
        ensure!(
            file_hash(&resolve(root, &c.prepared)?.join("prepared.json"))? == c.prepared_sha256,
            "proposal cut identity changed"
        );
        let seeds = destination.join("seeds.json");
        let generation = destination.join("generation.json");
        if seeds.exists() && generation.exists() {
            let g: Value = read_json(&generation)?;
            let s: SeedPopulation = read_json(&seeds)?;
            ensure!(g["schema"] == "parallel.proposal_generation.v2"
                && g["pure_prediction_protocol"] == PURE_PROTOCOL
                && g["gp_assembly_protocol"] == proposal_population::ASSEMBLY_PROTOCOL,
                "saved proposal uses an incompatible prediction protocol; use a new output directory");
            let pure: Value = read_json(&destination.join("pure-ranking.json"))?;
            let gp: Value = read_json(&destination.join("gp-ranking.json"))?;
            let best: Value = read_json(&destination.join("pure-best-predicted.json"))?;
            let assembly: Value = read_json(&destination.join("gp-population-assembly.json"))?;
            let pure_ranking: crate::proposal_ranking::Ranking =
                serde_json::from_value(pure["ranking"].clone())?;
            let gp_ranking: crate::proposal_ranking::Ranking =
                serde_json::from_value(gp["ranking"].clone())?;
            for ranking in [&pure_ranking, &gp_ranking] {
                ensure!(
                    ranking.selection_id == crate::proposal_ranking::SELECTION_ID
                        && ranking.bindings.conditioning_context_sha256 == actual.sha256
                        && ranking.bindings.checkpoint_sha256 == checkpoint_sha
                        && ranking.bindings.vocabulary_sha256
                            == champion_snapshot.vocabulary_sha256
                        && ranking.pool_size == 64
                        && ranking.chosen_count == 64
                        && ranking.ranked.len() == 64
                        && ranking.training_context_only
                        && !ranking.heldout_labels_read
                        && !ranking.gp_scores_read,
                    "saved fixed-pool ranking identity changed"
                );
            }
            let pure_sha = file_hash(&destination.join("pure-ranking.json"))?;
            let assembly_sha = file_hash(&destination.join("gp-population-assembly.json"))?;
            ensure!(
                s.schema == "parallel.seed_population.v1"
                    && s.candidates.len() == 80
                    && g["seeds_sha256"] == file_hash(&seeds)?
                    && g["champion_snapshot_sha256"] == champion_sha
                    && g["pure_ranking_sha256"] == pure_sha
                    && g["gp_ranking_sha256"] == file_hash(&destination.join("gp-ranking.json"))?
                    && g["pure_best_predicted_sha256"]
                        == file_hash(&destination.join("pure-best-predicted.json"))?
                    && g["gp_population_assembly_sha256"] == assembly_sha
                    && pure["schema"] == "parallel.pure_neural_ranking.v1"
                    && pure["protocol"] == PURE_PROTOCOL
                    && pure["teacher_population_consulted"] == false
                    && gp["schema"] == "parallel.gp_population_ranking.v1"
                    && gp["protocol"] == proposal_population::ASSEMBLY_PROTOCOL
                    && gp["teacher_population_consulted"] == true
                    && gp["training_context_only_end_to_end"] == false
                    && gp["pure_ranking_sha256"] == pure_sha
                    && gp["population_assembly_sha256"] == assembly_sha
                    && gp["champion_snapshot_sha256"] == champion_sha
                    && assembly["schema"] == "parallel.gp_population_assembly.v1"
                    && assembly["protocol"] == proposal_population::ASSEMBLY_PROTOCOL
                    && assembly["pure_ranking_sha256"] == pure_sha
                    && assembly["champion_snapshot_sha256"] == champion_sha
                    && assembly["checkpoint_sha256"] == checkpoint_sha
                    && assembly["conditioning_context_sha256"] == actual.sha256
                    && assembly["pure_programs_sha256"]
                        == hash(&serde_json::to_vec(
                            &pure_ranking
                                .selected()
                                .iter()
                                .map(|p| &p.program)
                                .collect::<Vec<_>>()
                        )?)
                    && assembly["retained_tail_sha256"]
                        == hash(&serde_json::to_vec(&s.candidates[64..])?)
                    && s.candidates
                        .iter()
                        .take(64)
                        .map(|c| c.genome.as_str())
                        .eq(gp_ranking.selected().iter().map(|p| p.program.as_str()))
                    && s.candidates
                        .iter()
                        .map(|c| &c.genome)
                        .collect::<BTreeSet<_>>()
                        .len()
                        == 80
                    && g["checkpoint_sha256"] == checkpoint_sha
                    && g["conditioning_context_sha256"] == actual.sha256
                    && s.prepared_sha256 == c.prepared_sha256
                    && s.cut == c.cut
                    && s.training_sha256 == cut.training_sha256
                    && s.context_sha256 == c.sha256
                    && s.checkpoint_sha256 == checkpoint_sha,
                "completed proposal identity changed"
            );
            ensure!(
                best["schema"] == "parallel.pure_best_predicted_program.v1"
                    && best["protocol"] == PURE_PROTOCOL
                    && best["pure_ranking_sha256"] == pure_sha
                    && best["dataset"] == c.dataset
                    && best["cut"] == c.cut
                    && best["prepared_sha256"] == c.prepared_sha256
                    && best["training_context_sha256"] == c.sha256
                    && best["conditioning_context_sha256"] == actual.sha256
                    && best["conditioning_dataset"] == actual.dataset
                    && best["checkpoint_sha256"] == checkpoint_sha
                    && best["vocabulary_sha256"] == champion_snapshot.vocabulary_sha256
                    && best["prediction"] == serde_json::to_value(&pure_ranking.selected()[0])?
                    && best["training_context_only_end_to_end"] == true
                    && best["teacher_population_consulted"] == false
                    && best["gp_scores_used_for_prediction"] == false
                    && best["heldout_targets_used_for_prediction"] == false
                    && best["final_tests_opened"] == false,
                "saved pure prediction identity changed"
            );
            generated_jobs.push(GpJob {
                prepared: c.prepared.clone(),
                prepared_sha256: c.prepared_sha256.clone(),
                cut: c.cut,
                output: destination.join("search").strip_prefix(root)?.to_path_buf(),
                seeds: Some(seeds.strip_prefix(root)?.to_path_buf()),
                seeds_sha256: Some(file_hash(&seeds)?),
                ..Default::default()
            });
            write_json(
                &output.join(format!("rank-{rank}.json")),
                &json!({"rank":rank,"world":world,"completed":generated_jobs.len(),"phase":"verified saved proposals","timestamp":now()}),
            )?;
            continue;
        }
        let pure_fallback_seed =
            crate::training::mix(proposal_seed ^ j as u64 ^ 0x707572655f66696c);
        let (candidates, attempted, valid, duplicate, neural) = pure_pool_impl(
            &model,
            &vocab,
            &values,
            device,
            crate::training::mix(proposal_seed ^ j as u64),
            pure_fallback_seed,
            "regression",
            Some((&destination.join("all-valid-neural.json"), shuffled)),
        )?;
        let bindings = crate::proposal_ranking::Bindings {
            checkpoint_sha256: checkpoint_sha.clone(),
            conditioning_context_sha256: actual.sha256.clone(),
            vocabulary_sha256: champion_snapshot.vocabulary_sha256.clone(),
        };
        // This pool has never consulted an incumbent or a retained program.
        let pure_ranking = crate::proposal_ranking::score_pool(
            &model,
            &vocab,
            &values,
            &candidates
                .iter()
                .map(|c| c.genome.clone())
                .collect::<Vec<_>>(),
            bindings.clone(),
            crate::proposal_ranking::Options {
                device,
                batch_size: 64,
                keep: 64,
            },
        )?;
        let mut by_genome = candidates
            .into_iter()
            .map(|c| (c.genome.clone(), c))
            .collect::<BTreeMap<_, _>>();
        let pure_candidates = pure_ranking
            .selected()
            .iter()
            .map(|p| {
                by_genome
                    .remove(&p.program)
                    .context("pure ranked program absent")
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            by_genome.is_empty() && pure_candidates.len() == 64,
            "pure pool identity changed"
        );
        write_json(
            &destination.join("pure-ranking.json"),
            &json!({
                "schema":"parallel.pure_neural_ranking.v1","protocol":PURE_PROTOCOL,
                "ranking":pure_ranking,"sampling_seed":crate::training::mix(proposal_seed ^ j as u64),
                "fallback_seed":pure_fallback_seed,"attempted":attempted,"valid":valid,
                "within_neural_duplicates":duplicate,"unique_neural":neural,
                "teacher_population_consulted":false,
            }),
        )?;
        let pure_ranking_sha = file_hash(&destination.join("pure-ranking.json"))?;
        write_json(
            &destination.join("pure-best-predicted.json"),
            &json!({
                "schema":"parallel.pure_best_predicted_program.v1","protocol":PURE_PROTOCOL,
                "dataset":c.dataset,"cut":c.cut,"prepared_sha256":c.prepared_sha256,
                "training_context_sha256":c.sha256,"conditioning_context_sha256":actual.sha256,
                "conditioning_dataset":actual.dataset,"checkpoint_sha256":checkpoint_sha,
                "vocabulary_sha256":bindings.vocabulary_sha256,
                "context_mode":if shuffled{"shuffled_between_source_families"}else{"matching_training_cut"},
                "selection_id":pure_ranking.selection_id,"pure_ranking_sha256":pure_ranking_sha,
                "prediction":pure_ranking.selected()[0],"quality_units":pure_ranking.quality_units,
                "predicted_runtime_is_estimate":true,"training_context_only_end_to_end":true,
                "input_scope":"saved training context, frozen checkpoint/vocabulary and independent RNG; excludes current teacher identities and scores",
                "teacher_population_consulted":false,"gp_scores_used_for_prediction":false,
                "heldout_targets_used_for_prediction":false,"final_tests_opened":false,
            }),
        )?;

        // Adaptive teachers enter only after the pure prediction is frozen.
        let mut seen = BTreeSet::new();
        let mut random_seed = proposal_seed.wrapping_add(j as u64 * 100003);
        let mut insert_random = |candidates: &mut Vec<SeedCandidate>,
                                 seen: &mut BTreeSet<String>,
                                 origin: &str|
         -> Result<()> {
            for _ in 0..10000 {
                random_seed = random_seed.wrapping_add(1);
                let raw = feat_math::evolve::random_canonical_pipeline(random_seed, 6);
                let Ok(genome) = vocab.project_proposal(&raw) else {
                    continue;
                };
                if seen.insert(genome.clone()) {
                    candidates.push(SeedCandidate {
                        genome,
                        origin: origin.into(),
                        policy_logprob: None,
                        proposal_record_sha256: None,
                    });
                    return Ok(());
                }
            }
            anyhow::bail!("unable to fill diverse proposal population")
        };
        // The two policy arms share these exact retained/exploration slots.
        let mut retained = Vec::new();
        let mut incumbent_evidence = Vec::new();
        for champion in &champion_snapshot.entries {
            if retained.len() == 10 {
                break;
            }
            if champion.identity.dataset != c.dataset || champion.identity.cut != c.cut {
                continue;
            }
            champion.validate()?;
            ensure!(
                champion.identity.prepared_sha256 == c.prepared_sha256
                    && champion.identity.training_sha256 == cut.training_sha256
                    && champion.identity.validation_sha256 == cut.validation_sha256
                    && champion.identity.vocabulary_sha256 == champion_snapshot.vocabulary_sha256,
                "incumbent belongs to a different cut or evaluator"
            );
            // Exact grammar encoding, without parameter projection or score reuse.
            vocab.encode(champion.genome())?;
            if seen.insert(champion.genome().into()) {
                retained.push(SeedCandidate {
                    genome: champion.genome().into(),
                    origin: "incumbent_cut_teacher".into(),
                    policy_logprob: None,
                    proposal_record_sha256: None,
                });
                incumbent_evidence.push(json!({"identity":champion.identity,"observation_sha256":champion.observation_sha256,
                    "source":champion.source,"evaluator_matches_current":champion.identity.evaluator_sha256 == champion_snapshot.evaluator_sha256,
                    "reevaluation_required":true,"score_transferred":false}));
            }
        }
        if let Some(leaders) = programs.get(context_id) {
            for genome in leaders {
                if retained.len() == 10 {
                    break;
                }
                if seen.insert(genome.clone()) {
                    retained.push(SeedCandidate {
                        genome: genome.clone(),
                        origin: "retained_cut_teacher".into(),
                        policy_logprob: None,
                        proposal_record_sha256: None,
                    });
                }
            }
        }
        for champion in &champion_snapshot.entries {
            if retained.len() >= 16 {
                break;
            }
            if champion.identity.family != c.family || champion.identity.cut == c.cut {
                continue;
            }
            if seen.insert(champion.genome().into()) {
                retained.push(SeedCandidate {
                    genome: champion.genome().into(),
                    origin: "same_family_cross_cut_teacher".into(),
                    policy_logprob: None,
                    proposal_record_sha256: None,
                });
            }
        }
        while retained.len() < 10 {
            insert_random(&mut retained, &mut seen, "retained_slot_fallback")?;
        }
        while retained.len() < 16 {
            insert_random(&mut retained, &mut seen, "random_exploration")?;
        }

        let assembly = proposal_population::assemble(
            &pure_candidates,
            &retained,
            &vocab,
            crate::training::mix(proposal_seed ^ j as u64 ^ 0x67705f6f7665726c),
        )?;
        let mut assembly_receipt = serde_json::to_value(&assembly)?;
        assembly_receipt["pure_ranking_sha256"] = pure_ranking_sha.clone().into();
        assembly_receipt["champion_snapshot_sha256"] = champion_sha.clone().into();
        assembly_receipt["checkpoint_sha256"] = checkpoint_sha.clone().into();
        assembly_receipt["conditioning_context_sha256"] = actual.sha256.clone().into();
        write_json(
            &destination.join("gp-population-assembly.json"),
            &assembly_receipt,
        )?;
        // A filtered/refilled GP pool has a distinct ranking and input claim.
        let gp_ranking = if assembly.overlaps.is_empty() {
            pure_ranking.clone()
        } else {
            crate::proposal_ranking::score_pool(
                &model,
                &vocab,
                &values,
                &assembly
                    .neural_slots
                    .iter()
                    .map(|c| c.genome.clone())
                    .collect::<Vec<_>>(),
                bindings,
                crate::proposal_ranking::Options {
                    device,
                    batch_size: 64,
                    keep: 64,
                },
            )?
        };
        write_json(
            &destination.join("gp-ranking.json"),
            &json!({
                "schema":"parallel.gp_population_ranking.v1","protocol":proposal_population::ASSEMBLY_PROTOCOL,
                "ranking":gp_ranking,"pure_ranking_sha256":pure_ranking_sha,
                "population_assembly_sha256":file_hash(&destination.join("gp-population-assembly.json"))?,
                "champion_snapshot_sha256":champion_sha,"teacher_population_consulted":true,
                "training_context_only_end_to_end":false,
            }),
        )?;
        let mut by_genome = assembly
            .neural_slots
            .into_iter()
            .map(|c| (c.genome.clone(), c))
            .collect::<BTreeMap<_, _>>();
        let mut candidates = gp_ranking
            .selected()
            .iter()
            .map(|p| {
                by_genome
                    .remove(&p.program)
                    .context("GP ranked program absent")
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            by_genome.is_empty() && candidates.len() == 64,
            "GP ranked pool identity changed"
        );
        let scores: Vec<f64> = gp_ranking
            .selected()
            .iter()
            .map(|p| p.predicted_quality)
            .collect();
        crate::gp::attach_policy_logprobs(&mut candidates, &scores)?;
        let gp_neural = candidates
            .iter()
            .filter(|c| c.origin == "neural_policy")
            .count();
        candidates.extend(retained);
        let population = SeedPopulation {
            schema: "parallel.seed_population.v1".into(),
            prepared_sha256: c.prepared_sha256.clone(),
            cut: c.cut,
            training_sha256: cut.training_sha256.clone(),
            context_sha256: c.sha256.clone(),
            checkpoint_sha256: checkpoint_sha.clone(),
            candidates,
        };
        let seeds = destination.join("seeds.json");
        write_json(&seeds, &population)?;
        write_json(
            &destination.join("generation.json"),
            &json!({"schema":"parallel.proposal_generation.v2","pure_prediction_protocol":PURE_PROTOCOL,
                "gp_assembly_protocol":proposal_population::ASSEMBLY_PROTOCOL,"attempted":attempted,"valid":valid,
                "within_neural_duplicates":duplicate,"pure_unique_neural":neural,"unique_neural":gp_neural,
                "filled_population":80,"context_mode":if shuffled{"shuffled_between_source_families"}else{"matching_training_cut"},
                "conditioning_context_sha256":actual.sha256,"checkpoint_sha256":checkpoint_sha,
                "seeds_sha256":file_hash(&seeds)?,"champion_snapshot_sha256":champion_sha,"incumbents":incumbent_evidence,
                "pure_ranking_sha256":pure_ranking_sha,"pure_best_predicted_sha256":file_hash(&destination.join("pure-best-predicted.json"))?,
                "gp_ranking_sha256":file_hash(&destination.join("gp-ranking.json"))?,
                "gp_population_assembly_sha256":file_hash(&destination.join("gp-population-assembly.json"))?,
                "teacher_overlaps":assembly.overlaps,"overlap_refills":assembly.refills,
                "selection_id":gp_ranking.selection_id,"seed":crate::training::mix(proposal_seed^j as u64),"final_tests_opened":false}),
        )?;
        generated_jobs.push(GpJob {
            prepared: c.prepared.clone(),
            prepared_sha256: c.prepared_sha256.clone(),
            cut: c.cut,
            output: destination.join("search").strip_prefix(root)?.to_path_buf(),
            seeds: Some(seeds.strip_prefix(root)?.to_path_buf()),
            seeds_sha256: Some(file_hash(&seeds)?),
            ..Default::default()
        });
        write_json(
            &output.join(format!("rank-{rank}.json")),
            &json!({"rank":rank,"world":world,"completed":generated_jobs.len(),"phase":"grammar-masked neural proposals","timestamp":now(),"seconds":started.elapsed().as_secs_f64(),"unique_neural_last_cut":gp_neural}),
        )?;
    }
    write_json(
        &output.join(format!("jobs-{rank}.json")),
        &GpJobs {
            schema: "parallel.gp_jobs.v1".into(),
            qualification: true,
            jobs: generated_jobs,
            vocabulary: Some(
                corpus
                    .join("vocabulary.json")
                    .strip_prefix(root)?
                    .to_path_buf(),
            ),
        },
    )?;
    write_json(
        &output.join(format!("complete-{rank}.json")),
        &json!({"rank":rank,"world":world,"seconds":started.elapsed().as_secs_f64(),
            "topology":binding.receipt()}),
    )?;
    Ok(())
}

pub fn proposal_prefix(vocab: &Vocab, task: &str) -> Result<Vec<usize>> {
    Ok(vec![1, vocab.engine_ids["featmath"], vocab.task_id(task)?])
}

/// Shared unchanged pure-v4 sampler for production and isolated qualification.
pub fn pure_pool(
    model: &Policy,
    vocab: &Vocab,
    values: &[f32],
    device: Device,
    sampling_seed: u64,
    pure_fallback_seed: u64,
) -> Result<(Vec<SeedCandidate>, usize, usize, usize, usize)> {
    pure_pool_for_task(
        model,
        vocab,
        values,
        device,
        sampling_seed,
        pure_fallback_seed,
        "regression",
    )
}

pub fn pure_pool_for_task(
    model: &Policy,
    vocab: &Vocab,
    values: &[f32],
    device: Device,
    sampling_seed: u64,
    pure_fallback_seed: u64,
    task: &str,
) -> Result<(Vec<SeedCandidate>, usize, usize, usize, usize)> {
    pure_pool_impl(
        model,
        vocab,
        values,
        device,
        sampling_seed,
        pure_fallback_seed,
        task,
        None,
    )
}

fn pure_pool_impl(
    model: &Policy,
    vocab: &Vocab,
    values: &[f32],
    device: Device,
    sampling_seed: u64,
    pure_fallback_seed: u64,
    task: &str,
    capture: Option<(&Path, bool)>,
) -> Result<(Vec<SeedCandidate>, usize, usize, usize, usize)> {
    tch::manual_seed(sampling_seed as i64);
    let batch = 128usize;
    let contexts = Tensor::from_slice(&values)
        .view([1, 2048])
        .repeat([batch as i64, 1])
        .to_device(device);
    let mut candidates = Vec::new();
    let mut pure_seen = BTreeSet::new();
    let mut attempted = 0usize;
    let mut valid = 0usize;
    let mut duplicate = 0usize;
    let mut all_valid = Vec::new();
    for round in 0..8 {
        let task_id = vocab.task_id(task)?;
        let mut sequences = vec![vec![1usize, vocab.engine_ids["featmath"], task_id]; batch];
        let mut states = (0..batch)
            .map(|_| {
                let mut s = State::default();
                s.advance(&vocab, vocab.engine_ids["featmath"]).unwrap();
                s.advance(&vocab, task_id).unwrap();
                s
            })
            .collect::<Vec<_>>();
        for _ in 3..128 {
            if states.iter().all(State::done) {
                break;
            }
            let length = sequences[0].len();
            let tokens = Tensor::from_slice(
                &sequences
                    .iter()
                    .flat_map(|s| s.iter().map(|&t| t as i64))
                    .collect::<Vec<_>>(),
            )
            .view([batch as i64, length as i64])
            .to_device(device);
            let pad = tokens.ne(0);
            let mut legal = vec![0u8; batch * vocab.tokens.len()];
            for (i, s) in states.iter().enumerate() {
                let ids = if s.done() { vec![0] } else { s.legal(&vocab) };
                for id in ids {
                    legal[i * vocab.tokens.len() + id] = 1;
                }
            }
            let legal = Tensor::from_slice(&legal)
                .view([batch as i64, vocab.tokens.len() as i64])
                .to_kind(Kind::Bool)
                .to_device(device);
            let next = tch::no_grad(|| -> Result<Vec<i64>> {
                let _guard = Bf16::enter();
                let (logits, _, _, _, _) = model.forward(
                    &contexts.narrow(1, 0, 1536),
                    &contexts.narrow(1, 1536, 512),
                    &tokens,
                    &pad,
                    false,
                )?;
                let logits = logits
                    .select(1, length as i64)
                    .to_kind(Kind::Float)
                    .masked_fill(&legal.logical_not(), f64::NEG_INFINITY);
                let next = (logits / 1.0)
                    .softmax(-1, Kind::Float)
                    .multinomial(1, false);
                Ok(Vec::<i64>::try_from(
                    next.to_device(Device::Cpu).flatten(0, -1),
                )?)
            })?;
            for i in 0..batch {
                if !states[i].done() {
                    states[i].advance(&vocab, next[i] as usize)?;
                }
                sequences[i].push(next[i] as usize);
            }
        }
        attempted += batch;
        for (slot, (mut sequence, s)) in sequences.into_iter().zip(states).enumerate() {
            if !s.done() {
                continue;
            }
            sequence.truncate(
                sequence
                    .iter()
                    .position(|&t| t == 2)
                    .context("EOS absent")?
                    + 1,
            );
            let Ok(genome) = vocab.decode(&sequence) else {
                continue;
            };
            if feat_math::GenomeIr::parse(&genome).is_err() {
                continue;
            }
            valid += 1;
            if capture.is_some() {
                all_valid.push(json!({"round":round,"slot":slot,"genome":genome,
                    "program_sha256":hash(genome.as_bytes())}));
            }
            if !proposal_population::push_pure(&mut candidates, &mut pure_seen, genome) {
                duplicate += 1;
            }
        }
        if candidates.len() == 64 {
            break;
        }
    }
    if let Some((path, shuffled)) = capture {
        crate::smartcluster_worker::write_immutable_record(
            path,
            &json!({"schema":"jope.all_valid_neural.v1","sampling_seed":sampling_seed,
                "attempted":attempted,"valid":valid,"within_neural_duplicates":duplicate,"arm":if shuffled {"shuffled"} else {"model"},
                "rows":all_valid,"final_tests_opened":false}),
        )?;
    }
    let neural = candidates.len();
    proposal_population::fill_pure(&mut candidates, &mut pure_seen, &vocab, pure_fallback_seed)?;
    Ok((candidates, attempted, valid, duplicate, neural))
}
