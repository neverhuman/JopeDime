use super::*;

pub fn rank(
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
) -> Result<()> {
    rank_with_policy(
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
        None,
        None,
    )
}

/// `continuation_mode` is the request's optional explicit lineage-mode
/// declaration; `None` keeps the implicit meaning of older requests.
pub fn rank_with_policy(
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
    policy: Option<&Path>,
    continuation_mode: Option<continuation::Mode>,
) -> Result<()> {
    crate::task_contract::require_original_gpu_grant("train-rank")?;
    let _deadline = crate::gpu_worker::deadline_watchdog()?;
    if std::env::var_os("JOPE_GPU_WORKER_ACTIVE").is_some() {
        crate::gpu_worker::require_accepted_task(
            root,
            &crate::gpu_worker::Task::Train {
                corpus: corpus.into(),
                run: run.into(),
                launch: launch.into(),
                world,
                micro,
                steps,
                checkpoint_steps,
                resume,
                initialize: initialize.map(Into::into),
                replay_policy: policy.map(Into::into),
                continuation_mode,
            },
        )?;
    }
    let (_lease, partition_version) = crate::gpu_worker::verify_training_launch(
        root, corpus, run, launch, world, micro, steps, resume, initialize, policy,
    )?;
    if crate::gpu_worker::enabled() {
        return crate::gpu_worker::submit(
            root,
            rank,
            crate::gpu_worker::Task::Train {
                corpus: corpus.into(),
                run: run.into(),
                launch: launch.into(),
                world,
                micro,
                steps,
                checkpoint_steps,
                resume,
                initialize: initialize.map(Into::into),
                replay_policy: policy.map(Into::into),
                continuation_mode,
            },
        );
    }
    rank_impl(
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
        2048,
        false,
        policy,
        None,
        false,
        continuation_mode,
        None,
        None,
        partition_version,
        None,
    )
}

/// Explicit policy warm-start experiment; ordinary production/resume stays unchanged.
pub fn rank_replay(
    root: &Path,
    corpus: &Path,
    run: &Path,
    launch: &Path,
    rank: usize,
    world: usize,
    steps: usize,
    resume: bool,
    initialize: Option<&Path>,
    policy: &Path,
) -> Result<()> {
    rank_with_policy(
        root,
        corpus,
        run,
        launch,
        rank,
        world,
        256,
        steps,
        1000,
        resume,
        initialize,
        Some(policy),
        None,
    )
}

/// Isolated real-corpus qualification; ordinary controllers keep their existing batch.
pub fn qualify_batch_rank(
    root: &Path,
    corpus: &Path,
    run: &Path,
    launch: &Path,
    rank: usize,
    micro: usize,
    global_batch: usize,
    steps: usize,
    initialize: &Path,
) -> Result<()> {
    crate::task_contract::require_qualified_gpu_experiment("qualify-batch-rank")?;
    ensure!(
        [256, 512, 1024, 2048].contains(&micro) && global_batch == micro * 8,
        "qualification requires an explicit eight-rank global batch"
    );
    let experiment = std::env::var_os("JOPE_EXPERIMENT_DIR")
        .context("batch qualification requires an isolated experiment directory")?;
    let experiment = Path::new(&experiment).canonicalize()?;
    for path in [root, run, launch] {
        ensure!(
            path.canonicalize()?.starts_with(&experiment),
            "batch qualification root, output and launch must stay inside its experiment"
        );
    }
    rank_impl(
        root,
        corpus,
        run,
        launch,
        rank,
        8,
        micro,
        steps,
        steps,
        false,
        Some(initialize),
        global_batch,
        true,
        None,
        None,
        false,
        None,
        None,
        None,
        partition::Version::V1,
        None,
    )
}

/// SmartCluster's opaque process-local admission is distinct from legacy funding.
pub(crate) fn smartcluster_rank(
    task: &crate::smartcluster_worker::VerifiedTask,
    receipt: &Path,
) -> Result<()> {
    use crate::smartcluster_worker::Operation;
    let Operation::Train {
        corpus,
        run,
        launch,
        micro,
        checkpoint_steps,
        initialize,
        replay_policy,
        initial_optimizer_step,
        requested_step_increment,
        target_optimizer_step,
        continuation_mode,
    } = task.operation()
    else {
        anyhow::bail!("training operation required")
    };
    ensure!(
        !task.stop().load(std::sync::atomic::Ordering::Acquire),
        "native control already lost"
    );
    ensure!(
        initial_optimizer_step.checked_add(*requested_step_increment)
            == Some(*target_optimizer_step),
        "original training target changed"
    );
    let root = task.root();
    let corpus = resolve(root, corpus)?;
    let run = resolve(root, run)?;
    let launch_root = resolve(root, launch)?;
    let initial = resolve(root, initialize)?;
    let policy_path = resolve(root, replay_policy)?;
    let parent_bytes = fs::read(initial.join("state.json"))?;
    let parent: Value = serde_json::from_slice(&parent_bytes)?;
    let policy: Value = read_json(&policy_path)?;
    // An optimizer carry continues the parent's optimizer step. A `jope.native_loop.v2` weight
    // warm start (OWNER-DECISIONS.md D12) restarts it at zero from the bound parent weights and
    // is admitted only as a declared warm start over a verified native continuation corpus.
    let warm_start =
        crate::native_loop::warm_start::policy_mode(&policy, &hash(&parent_bytes), &parent)?;
    ensure!(
        (if warm_start.is_some() {
            *initial_optimizer_step == 0
        } else {
            parent["step"] == *initial_optimizer_step
        }) && file_hash(&initial.join("model.ot"))? == parent["model_sha256"]
            && file_hash(&initial.join("optimizer.ot"))? == parent["optimizer_sha256"],
        "original checkpoint model/optimizer changed"
    );
    ensure!(
        warm_start.is_some() || policy["optimizer"] == "carry",
        "native continuation must preserve optimizer bytes"
    );
    let corpus_index: Value = read_json(&corpus.join("index.json"))?;
    let native_continuation = corpus_index["qualification"] == false;
    if warm_start.is_some() {
        ensure!(
            native_continuation && *continuation_mode == Some(continuation::Mode::WeightWarmStart),
            "a weight warm start is admitted only as a declared native continuation"
        );
    }
    if native_continuation {
        verify_native_training_eligibility(
            task,
            &policy,
            warm_start
                .as_ref()
                .map(|binding| binding["parent_step"].as_u64().context("parent step"))
                .transpose()?,
        )?;
    }
    let original = json!({"schema":"jope.smartcluster_optimizer_window.v1","initial_optimizer_step":initial_optimizer_step,"requested_step_increment":requested_step_increment,"target_optimizer_step":target_optimizer_step,"parent_state_sha256":file_hash(&initial.join("state.json"))?,"corpus_sha256":file_hash(&corpus.join("index.json"))?,"policy_sha256":file_hash(&policy_path)?,"microbatch":micro,"global_batch":2048});
    fs::create_dir_all(&run)?;
    fs::create_dir_all(&launch_root)?;
    let window = run.join("smartcluster-window.json");
    if window.exists() {
        ensure!(
            read_json::<Value>(&window)? == original,
            "original native optimizer window changed"
        );
    } else {
        crate::smartcluster_worker::write_immutable_record(&window, &original)?;
    }
    if task.recovering() {
        crate::checkpoint_recovery::reconcile(root, &run)?;
    }
    let launch = native_attempt_launch(&launch_root, task.attempt(), &original)?;
    let resume = task.recovering() && run.join("latest.json").exists();
    if resume {
        let latest: Value = read_json(&run.join("latest.json"))?;
        let checkpoint = resolve(
            root,
            Path::new(
                latest["checkpoint"]
                    .as_str()
                    .context("recovery checkpoint absent")?,
            ),
        )?;
        ensure!(
            file_hash(&checkpoint.join("state.json"))? == latest["state_sha256"],
            "recovery state changed"
        );
        let state: Value = read_json(&checkpoint.join("state.json"))?;
        let step = state["step"]
            .as_u64()
            .context("recovery optimizer step absent")?;
        ensure!(
            step >= *initial_optimizer_step && step <= *target_optimizer_step,
            "recovery outside original optimizer window"
        );
        write_json(&launch.join("resume-input.json"), &latest)?;
    }
    rank_impl(
        root,
        &corpus,
        &run,
        &launch,
        0,
        1,
        *micro,
        (*requested_step_increment).try_into()?,
        *checkpoint_steps,
        resume,
        if resume { None } else { Some(initialize) },
        2048,
        false,
        Some(&policy_path),
        Some(task.stop()),
        native_continuation,
        *continuation_mode,
        None,
        None,
        partition::Version::V1,
        None,
    )?;
    let complete: Value = read_json(&launch.join("complete-0.json"))?;
    ensure!(
        complete["steps"] == *target_optimizer_step,
        "training did not finish original target"
    );
    let mut completion = json!({"schema":"jope.smartcluster_training_completion.v1","window":original,"numerical_receipt":complete,"latest":read_json::<Value>(&run.join("latest.json"))?});
    // A weight warm start's receipt names the parent weights it started from (D12).
    if let Some(binding) = warm_start {
        completion["warm_start"] = binding;
    }
    write_json(receipt, &completion)
}

/// NCCL rendezvous and rank receipts belong to one actual native allocation.
/// Preserve the original logical optimizer window and every old launch verbatim.
pub(crate) fn native_attempt_launch(
    root: &Path,
    attempt: jain_smartcluster::FencedAttempt,
    window: &Value,
) -> Result<std::path::PathBuf> {
    let launch = root.join(format!(
        "native-{}-attempt-{}-fence-{}",
        attempt.job_id,
        attempt.number.get(),
        attempt.fence.get()
    ));
    fs::create_dir(&launch)
        .context("native training allocation launch already exists or cannot be created")?;
    fs::File::open(root)?.sync_all()?;
    crate::smartcluster_worker::write_immutable_record(
        &launch.join("native-attempt.json"),
        &json!({"schema":"jope.native_training_launch.v1","attempt":attempt,"original_window_sha256":hash(&serde_json::to_vec(window)?)}),
    )?;
    Ok(launch)
}

/// This path is reachable only through the authenticated native worker. Legacy
/// rank entry points retain their original explicitly-qualified-corpus gate.
fn verify_native_training_eligibility(
    task: &crate::smartcluster_worker::VerifiedTask,
    policy: &Value,
    warm_start_parent_step: Option<u64>,
) -> Result<()> {
    use crate::{
        native_graph::Artifact,
        native_loop::{Request, Role},
        smartcluster_worker::Operation,
    };
    let Operation::Train {
        corpus,
        initialize,
        initial_optimizer_step,
        requested_step_increment,
        target_optimizer_step,
        ..
    } = task.operation()
    else {
        anyhow::bail!("native Train required")
    };
    let root = task.root();
    let manifest = task.manifest();
    let corpus = Artifact {
        path: corpus.join("index.json"),
        sha256: file_hash(&resolve(root, corpus)?.join("index.json"))?,
    };
    let parent = Artifact {
        path: initialize.join("state.json"),
        sha256: file_hash(&resolve(root, initialize)?.join("state.json"))?,
    };
    ensure!(
        manifest.parent_checkpoint_sha256.as_deref() == Some(parent.sha256.as_str())
            && manifest.corpus_sha256.as_deref() == Some(corpus.sha256.as_str()),
        "native training parent/corpus manifest differs"
    );
    let index: Value = crate::native_genome::load_corpus_index(root, &corpus)?;
    ensure!(
        index["qualification"] == false
            && index["label_policy"] == "strict_continuation_midrank_v1",
        "unqualified corpus is not native verified continuation targets"
    );
    let evidence = &policy["eligibility_evidence"];
    ensure!(
        evidence["schema"] == "jope.native_continuation_eligibility.v1",
        "native eligibility evidence absent"
    );
    let mut parameters = json!({"checkpoint":parent,"corpus":corpus,"initial_step":initial_optimizer_step,
        "requested_increment":requested_step_increment,"target_step":target_optimizer_step,"global_batch":2048});
    if let Some(step) = warm_start_parent_step {
        parameters["optimizer"] = json!("reset");
        parameters["parent_step"] = json!(step);
    }
    ensure!(
        evidence["original_window"] == parameters,
        "native replay original window differs"
    );
    let scientific: Artifact = serde_json::from_value(evidence["scientific_policy"].clone())?;
    let request = Request {
        schema: "jope.native_loop_request.v1".into(),
        run: manifest.run.clone(),
        generation: manifest.generation,
        id: manifest.task_id.clone(),
        role: Role::Train,
        parameters,
        predecessors: serde_json::from_value(evidence["verified_corpus_dependencies"].clone())?,
    };
    let expected = crate::native_loop::materialize::training_eligibility(
        root,
        &corpus,
        &parent,
        &scientific,
        &request,
    )?;
    ensure!(
        *policy == expected,
        "native replay differs from complete verified target reconstruction"
    );
    Ok(())
}

/// W7-D2: a bounded rank trains from its admitted plan. `training` is what
/// `native_loop::bounded::training_run::prepare` derived from the plan, the launch record and
/// the admitted bootstrap (including its `run.json`); `assignment` is the plan's trainer
/// assignment, bound to this rank's device view. The communicator comes only from `admitted`:
/// [`communicator_source`] refuses a shared `nccl.id` before any input or device work. The
/// partition is the one the plan selects (v2 for three ranks). The single-host
/// `train_controlled` launcher is not involved. Returns this rank's completion receipt.
pub fn bounded_rank(
    training: &crate::native_loop::bounded::training_run::BoundedTraining,
    assignment: &topology::Assignment,
    admitted: &crate::native_loop::bounded::launch::AdmittedBootstrap,
    stop: &std::sync::atomic::AtomicBool,
) -> Result<Value> {
    let version = partition::Version::of_record(&training.spec["execution_partition"])?;
    ensure!(
        training.spec["plan_sha256"].as_str() == Some(assignment.plan_sha256.as_str())
            && admitted.plan_sha256() == assignment.plan_sha256,
        "the bounded run, its assignment and its admitted bootstrap name different plans"
    );
    rank_impl(
        &training.input_root,
        &training.input_root.join(&training.corpus),
        &training.run_dir,
        &training.launch_dir,
        training.global_rank,
        training.world,
        training.microbatch,
        training.steps,
        training.checkpoint_steps,
        false,
        training.initialize.as_deref(),
        training.global_batch,
        false,
        None,
        Some(stop),
        false,
        None,
        Some(assignment),
        Some(admitted),
        version,
        Some(training),
    )?;
    let completion = read_json(
        &training
            .launch_dir
            .join(format!("complete-{}.json", training.global_rank)),
    )?;
    training.validate_completion(&completion)?;
    Ok(completion)
}

/// Where a trainer rank obtains its NCCL unique id (W7-T04).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommunicatorSource {
    /// The historical same-allocation rendezvous file in this launch's directory. Only a
    /// rank without a frozen bounded-run assignment uses it.
    LaunchFile(std::path::PathBuf),
    /// The verified full-group admission of the rank's frozen bounded-run plan.
    Admitted,
}

/// Decide, before any input, device or communicator work, which rendezvous a rank may use:
/// - no frozen assignment and no admission: the historical `nccl.id` in `launch`;
/// - a frozen bounded-run assignment: only the admitted bootstrap of exactly that plan,
///   rank and world. A missing bootstrap, a bootstrap for another plan, rank or world, and a
///   shared `nccl.id` in the launch directory are refused;
/// - an admitted bootstrap without a frozen assignment is refused.
pub fn communicator_source(
    assignment: Option<&topology::Assignment>,
    admitted: Option<&crate::native_loop::bounded::launch::AdmittedBootstrap>,
    global_rank: usize,
    launch: &Path,
) -> Result<CommunicatorSource> {
    let shared = launch.join("nccl.id");
    match (assignment, admitted) {
        (None, None) => Ok(CommunicatorSource::LaunchFile(shared)),
        (None, Some(_)) => {
            anyhow::bail!("an admitted communicator bootstrap requires its plan's frozen assignment")
        }
        (Some(_), None) => anyhow::bail!(
            "a frozen bounded-run assignment takes its communicator bootstrap only from the verified admission; the shared nccl.id rendezvous is refused"
        ),
        (Some(assignment), Some(bootstrap)) => {
            ensure!(
                bootstrap.plan_sha256() == assignment.plan_sha256
                    && bootstrap.global_rank() as usize == global_rank
                    && bootstrap.world_size() as usize == assignment.world,
                "admitted bootstrap is for another plan, rank or world"
            );
            ensure!(
                !shared.try_exists()?,
                "a bounded launch directory holds a shared nccl.id; refused"
            );
            Ok(CommunicatorSource::Admitted)
        }
    }
}

/// An admitted bootstrap creates only the communicator of the binding its plan derived.
pub fn verify_admitted_bootstrap(
    binding: &topology::Binding,
    bootstrap: &crate::native_loop::bounded::launch::AdmittedBootstrap,
) -> Result<()> {
    ensure!(
        binding.plan_sha256() == Some(bootstrap.plan_sha256())
            && binding.global_rank() == bootstrap.global_rank() as usize
            && binding.world() == bootstrap.world_size() as usize,
        "admitted bootstrap is not for this binding's plan, rank and world"
    );
    Ok(())
}
