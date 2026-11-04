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

mod rank_jobs;

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
    rank_jobs::generate_rank_jobs(
        root,
        output,
        rank,
        world,
        &binding,
        shuffled,
        &vocab,
        &model,
        device,
        &programs,
        proposal_seed,
        &jobs,
        &donors,
        &donor_order,
        started,
        &checkpoint_sha,
        &champion_sha,
        &champion_snapshot,
        &mut generated_jobs,
    )?;
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
