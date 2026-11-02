use super::*;

#[macro_use]
#[path = "training_loop.rs"]
mod training_loop;

mod model_init;
mod preflight;
mod rank_context;
use model_init::initialize as initialize_rank;
use preflight::prepare;
use rank_context::{RankInitialized, RankInput, RankPreflight};

pub(crate) fn rank_impl(
    root: &Path,
    corpus: &Path,
    run: &Path,
    launch: &Path,
    rank: usize,
    world: usize,
    micro: usize,
    steps: usize,
    checkpoint_steps: usize,
    resume: bool,
    initialize: Option<&Path>,
    global_batch: usize,
    batch_qualification: bool,
    replay_policy: Option<&Path>,
    native_stop: Option<&std::sync::atomic::AtomicBool>,
    verified_native_continuation: bool,
    declared_mode: Option<continuation::Mode>,
    assignment: Option<&topology::Assignment>,
    admitted: Option<&crate::native_loop::bounded::launch::AdmittedBootstrap>,
    partition_version: partition::Version,
    bounded_training: Option<&crate::native_loop::bounded::training_run::BoundedTraining>,
) -> Result<()> {
    let request = RankInput {
        root,
        corpus,
        run,
        launch,
        rank,
        world,
        micro,
        steps,
        checkpoint_steps,
        resume,
        initialize,
        global_batch,
        batch_qualification,
        replay_policy,
        native_stop,
        verified_native_continuation,
        declared_mode,
        assignment,
        admitted,
        partition_version,
        bounded_training,
    };
    let preflight = prepare(request)?;
    let initialized = initialize_rank(request, preflight)?;
    let RankInitialized {
        preflight,
        device,
        mut vs,
        model,
        mut optimizer,
        initialization_path,
        mut initialization,
        sampling_offset,
        steps,
    } = initialized;
    let RankPreflight {
        communicator: _,
        execution,
        policy,
        policy_sha,
        replay,
        seed,
        corpus_hash,
        resident,
        vocabulary,
        frozen_inputs,
        frozen_inputs_sha,
        families,
        validation,
        teacher_family,
        fresh,
        fresh_by_teacher,
        fresh_records,
        balanced,
        epoch_steps,
        fixed_panel,
        panel_sha256,
        config,
        recovered_continuation,
        mode,
        declaration,
        objective_sha,
        continuity,
        selected_checkpoint,
        selected_state_sha,
        selected_state,
        exposure_baseline,
        family_baseline,
        family_baseline_known,
    } = preflight;
    let (index, records) = resident.as_ref();
    // W6-T01: cached device staging is keyed by the content it was built
    // from; a cold load verifies every byte through the same functions.
    // W2-T04 (G-11a/b): a staged tensor lives on one device; the device is part
    // of every resident key.
    let context_key = format!("{device:?}:{}", inputs::context_identity(index)?);
    let context_tensor = RESIDENT_CONTEXT.with(|cache| -> Result<Tensor> {
        if let Some((identity, tensor)) = cache.borrow().as_ref() {
            if *identity == context_key {
                return Ok(tensor.shallow_clone());
            }
        }
        // Release an obsolete epoch before allocating its replacement.
        cache.borrow_mut().take();
        let contexts = inputs::stage_contexts(root, index)?;
        // Small immutable artifacts are staged once on each GPU; no CSV loading in training steps.
        let tensor = Tensor::from_slice(&contexts)
            .view([-1, 2048])
            .to_device(device);
        // Device tensors own their copied storage; release host staging before training.
        drop(contexts);
        *cache.borrow_mut() = Some((context_key.clone(), tensor.shallow_clone()));
        Ok(tensor)
    })?;
    let tensor_key = format!("{device:?}:{corpus_hash}:{frozen_inputs_sha}");
    let (resident_cache_hit, tensors) =
        RESIDENT_TENSORS.with(|cache| -> Result<(bool, Vec<Tensor>)> {
            if let Some((identity, tensors)) = cache.borrow().as_ref() {
                if *identity == tensor_key {
                    return Ok((true, tensors.iter().map(Tensor::shallow_clone).collect()));
                }
            }
            cache.borrow_mut().take();
            let staged = inputs::stage_records(index, vocabulary.tokens.len(), |i| records.get(i))?;
            let token_tensor = Tensor::from_slice(&staged.tokens)
                .view([-1, SEQUENCE as i64])
                .to_device(device);
            let legal_id_tensor = Tensor::from_slice(&staged.legal)
                .view([-1, SEQUENCE as i64])
                .to_device(device);
            let context_id_tensor = Tensor::from_slice(&staged.context_ids).to_device(device);
            let mask_table = Tensor::from_slice(&staged.masks)
                .view([-1, vocabulary.tokens.len() as i64])
                .to_kind(Kind::Bool)
                .to_device(device);
            drop(staged);
            let tensors = vec![token_tensor, legal_id_tensor, context_id_tensor, mask_table];
            *cache.borrow_mut() = Some((
                tensor_key.clone(),
                tensors.iter().map(Tensor::shallow_clone).collect(),
            ));
            Ok((false, tensors))
        })?;
    let [token_tensor, legal_id_tensor, context_id_tensor, mask_table]: [Tensor; 4] =
        tensors.try_into().ok().unwrap();
    // Validate every sampled comparison against the frozen index before its
    // records are read or any loss is formed (W6-T03).
    let comparisons = objective::SampledComparisons::new(index)?;
    // `global` selects the objective: `None` is v1's microbatch-local
    // composite; partition v2 passes the step's global denominators.
    let loss_for = |ids: &[usize],
                    train: bool,
                    global: Option<&partition::Denominators>|
     -> Result<(Tensor, [f64; 5])> {
        let corpus_partition = if train {
            "meta_train"
        } else {
            "meta_validation"
        };
        comparisons.check_ids(index, ids, corpus_partition)?;
        let rows = ids.iter().map(|&i| records.get(i)).collect::<Vec<_>>();
        let labels = ids
            .iter()
            .zip(&rows)
            .map(|(&id, r)| objective::RowLabels::from_record(id, r))
            .collect::<Vec<_>>();
        comparisons.check_rows(index, &labels)?;
        let length = rows.iter().map(|r| r.length).max().unwrap().div_ceil(8) * 8;
        let length = length.min(SEQUENCE) as i64;
        let id = Tensor::from_slice(&ids.iter().map(|&i| i as i64).collect::<Vec<_>>())
            .to_device(device);
        let tokens = token_tensor
            .index_select(0, &id)
            .narrow(1, 0, length)
            .to_kind(Kind::Int64)
            .bitwise_and(0xffffi64);
        let pad = tokens.ne(0);
        let context = context_tensor.index_select(0, &context_id_tensor.index_select(0, &id));
        let (logits, rewards, runtime, stability, ablation) = model.forward(
            &context.narrow(1, 0, 1536),
            &context.narrow(1, 1536, 512),
            &tokens,
            &pad,
            train,
        )?;
        let legal = mask_table
            .index_select(
                0,
                &legal_id_tensor
                    .index_select(0, &id)
                    .narrow(1, 1, length - 1)
                    .reshape([-1])
                    .to_kind(Kind::Int64)
                    .bitwise_and(0xffffi64),
            )
            .view([ids.len() as i64, length - 1, vocabulary.tokens.len() as i64]);
        let outputs = objective::Outputs {
            logits: &logits,
            rewards: &rewards,
            runtime: &runtime,
            stability: &stability,
            ablation: &ablation,
        };
        match global {
            None => objective::composite(
                &outputs,
                model.loss_log_vars.values(),
                &tokens,
                &pad,
                &legal,
                &labels,
            ),
            Some(global) => partition::composite(
                &outputs,
                model.loss_log_vars.values(),
                &tokens,
                &pad,
                &legal,
                &labels,
                global,
            ),
        }
    };
    let accumulation = execution.accumulation(rank)?;
    let started = Instant::now();
    let initial_step = optimizer.step as usize;
    let mut last_live_write = Instant::now() - Duration::from_secs(1);
    let mut measured_step_seconds = 0.0;
    let mut telemetry_seconds = 0.0;
    let mut coordination_seconds = 0.0;
    let mut checkpoint_seconds = 0.0;
    let mut metrics = if rank == 0 {
        Some(BufWriter::new(
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(run.join("metrics.jsonl"))?,
        ))
    } else {
        None
    };
    #[cfg(feature = "telemetry")]
    let mut telemetry = if rank == 0 {
        std::env::var_os("JOPE_TELEMETRY_RUN_DIR").and_then(|run_dir| {
            let metadata = jope_telemetry::manifest::RunMetadata {
                metric_groups: std::collections::BTreeMap::from([(
                    jope_telemetry::train::TRAIN_LOSS_COMPONENT,
                    std::collections::BTreeMap::from([
                        (0, "sequence_nll".to_owned()),
                        (1, "ranking_bt".to_owned()),
                        (2, "ranking_listwise".to_owned()),
                        (3, "runtime_huber".to_owned()),
                        (4, "contextfree_bt".to_owned()),
                    ]),
                )]),
                metadata: [
                    ("trainer".to_owned(), json!("jope-engine")),
                    ("corpus_sha256".to_owned(), json!(corpus_hash)),
                    ("world_size".to_owned(), json!(world)),
                    ("global_batch".to_owned(), json!(global_batch)),
                    ("microbatch".to_owned(), json!(micro)),
                    ("seed".to_owned(), json!(seed)),
                    ("device".to_owned(), json!(format!("{:?}", device))),
                ]
                .into_iter()
                .collect(),
                ..Default::default()
            };
            let mut builder = jope_telemetry::core::RecorderBuilder::new(run_dir, metadata);
            if let Err(error) = builder.add_producer("jope-engine") {
                eprintln!("jope-telemetry producer setup failed: {error:#}");
                return None;
            }
            match builder.start() {
                Ok((recorder, mut producers)) => {
                    producers.pop().map(|producer| (recorder, producer))
                }
                Err(error) => {
                    eprintln!("jope-telemetry initialization failed: {error:#}");
                    None
                }
            }
        })
    } else {
        None
    };
    let mut last_checkpoint = Instant::now();
    let mut validation_metrics = Value::Null;
    let mut latest: Value = if run.join("latest.json").is_file() {
        read_json(&run.join("latest.json"))?
    } else {
        Value::Null
    };
    let mut observed_exposure = crate::training_exposure::Counts::default();
    let mut committed_exposure = crate::training_exposure::Counts::default();
    let mut observed_families = crate::training_exposure::FamilyGroups::new();
    write_json(
        &launch.join(format!("rank-{rank}.json")),
        &json!({"schema":"parallel.neural_training.v1",
        "rank":rank,"world":world,"status":"active","step":initial_step,"steps":steps,
        "initial_optimizer_step":initial_step,"corpus_sha256":corpus_hash,"timestamp":now(),
        "exposure":crate::training_exposure::attempt_snapshot(&observed_exposure,&committed_exposure,false)?}),
    )?;
    // W6-T04: every consumed microbatch, attempted update and durable commit
    // of this rank and attempt, bound into each checkpoint receipt.
    let mut journal = consumption::Journal::create_with_shard(
        &launch.join(format!("consumption-rank-{rank}.jsonl")),
        consumption::Header {
            schema: consumption::SCHEMA.into(),
            rank,
            world,
            microbatch: micro,
            global_batch,
            accumulation,
            seed,
            sampler_id: active_sampler_id(replay.as_ref()).into(),
            rng: consumption::RNG_ID.into(),
            sampling_offset: sampling_offset as u64,
            initial_optimizer_step: initial_step as u64,
            target_optimizer_step: steps as u64,
            corpus_sha256: corpus_hash.clone(),
            replay_policy_sha256: policy_sha.clone(),
            resume_state_sha256: if resume {
                Some(file_hash(
                    &selected_checkpoint
                        .as_ref()
                        .context("resume selection absent")?
                        .join("state.json"),
                )?)
            } else {
                None
            },
        },
        execution.shard(rank)?,
    )?;
    let journal_path = journal.path().strip_prefix(root)?.to_path_buf();
    // The optimizer step whose update is in flight; cleared once applied.
    let mut attempting: Option<usize> = None;
    let (
        outcome,
        optimizer,
        mut journal,
        attempting,
        observed_exposure,
        committed_exposure,
        observed_families,
        corpus_hash,
    ) = training_loop!((
        root,
        run,
        launch,
        rank,
        world,
        micro,
        steps,
        checkpoint_steps,
        resume,
        global_batch,
        batch_qualification,
        native_stop,
        execution,
        policy,
        policy_sha,
        replay,
        seed,
        corpus_hash,
        index,
        records,
        frozen_inputs_sha,
        families,
        validation,
        teacher_family,
        fresh,
        fresh_by_teacher,
        fresh_records,
        balanced,
        epoch_steps,
        fixed_panel,
        panel_sha256,
        config,
        recovered_continuation,
        mode,
        declaration,
        objective_sha,
        continuity,
        exposure_baseline,
        family_baseline,
        family_baseline_known,
        binding,
        comm,
        device,
        vs,
        optimizer,
        initialization,
        sampling_offset,
        resident_cache_hit,
        comparisons,
        loss_for,
        accumulation,
        started,
        initial_step,
        last_live_write,
        measured_step_seconds,
        telemetry_seconds,
        coordination_seconds,
        checkpoint_seconds,
        metrics,
        telemetry,
        last_checkpoint,
        validation_metrics,
        latest,
        observed_exposure,
        committed_exposure,
        observed_families,
        journal,
        journal_path,
        attempting,
    ));
    if let Err(error) = &outcome {
        // An update that was attempted but not applied is recorded as
        // rejected; the journal refuses anything after it. Best effort: the
        // original error is what this rank returns.
        if let Some(step) = attempting {
            let _ = journal.rejected(step as u64, &format!("{error:#}"));
        }
        let _ = journal.sync();
    }
    if outcome.is_err() {
        // Normal error returns preserve all observed completed forwards. A
        // hard process death can leave an unreported tail and must remain
        // unknown during reconciliation; never infer zero lost exposure.
        if let Ok(mut exposure) = crate::training_exposure::attempt_snapshot(
            &observed_exposure,
            &committed_exposure,
            false,
        ) {
            exposure["failure_returned"] = json!(true);
            exposure["aborted_exposure_requires_stopped_rank_checkpoint_reconciliation"] =
                json!(true);
            let _ = write_json(
                &launch.join(format!("rank-{rank}.json")),
                &json!({
                "schema":"parallel.neural_training.v1","rank":rank,"world":world,"status":"failed",
                "initial_optimizer_step":initial_step,"step":optimizer.step,"corpus_sha256":corpus_hash,
                "exposure":exposure,"timestamp":now()}),
            );
        }
    }
    outcome
}
