use super::*;

/// Run one participant to completion. Returns the process exit code: 0 after publishing a
/// signed result (or on an explicit stop), [`EXIT_LOST_CONTROL`] when control is lost. Every
/// refusal before admission is an error, and then no work was started.
///
/// `sources` is how this process observes its host (its device visibility probe and its
/// identity). The production `jope-native-loop participant` passes
/// [`HostSources::system`](collect::HostSources::system); its own binary is always the one
/// actually running (`/proc/self/exe`), and its job arguments are always its own.
pub fn run(args: &ParticipantArgs, sources: &HostSources) -> Result<i32> {
    let (contract, record) =
        launch::load_launch(&args.plan, &args.plan_sha256, &args.launch_record)?;
    // W7-D2: a plan that selects training runs only on a build that has the trainer.
    let training = contract.training.is_some();
    ensure!(!training || cfg!(feature = "neural"), "{NO_TRAINER}");
    // W7-D1: before any work, prove this process is its planned SmartCluster job's leader.
    let installation = Installation::load(&args.installation)?;
    let claim = JobClaim::from_env(|name| std::env::var_os(name))?;
    let job = job_identity::verify_running_job(
        &installation.smartcluster.daemon,
        &claim,
        &ExpectedJob {
            plan_sha256: args.plan_sha256.clone(),
            run: contract.run.clone(),
            global_rank: record.global_rank,
            dispatch_id: record.dispatch_id.clone(),
            tenant: job_identity::own_tenant(),
            service: installation.smartcluster.service.clone(),
            arguments: args.job_arguments.clone(),
        },
    )?;
    // W7-D3: SmartCluster's GPU index must be the planned UUID (a CPU rank sees no device).
    let visibility = launch::resolve_visibility(
        &record,
        std::env::var_os(launch::VISIBILITY_VARIABLE).as_deref(),
        sources.commands,
    )?;
    let rank = record.global_rank;
    let planned = contract.participant(rank).context("rank")?;
    ensure!(
        training || planned.device.is_none(),
        "rank {rank} plans a GPU but the plan selects no training; the CPU-phase participant runs CPU ranks only"
    );
    if training {
        // Pin the planned UUID before anything can initialize CUDA: CUDA then sees exactly
        // that device at ordinal 0, whatever its enumeration order. No other thread exists yet.
        let uuid = visibility
            .pinned()
            .context("a training rank needs its planned GPU")?;
        std::env::set_var(launch::VISIBILITY_VARIABLE, uuid);
    }
    let key = load_signing_key(&args.participant_key)?;
    ensure!(
        hex::encode(key.verifying_key().to_bytes()) == planned.public_key,
        "participant key is not rank {rank}'s planned key"
    );
    let identity = collect::observe_identity(
        &installation.collect_spec(Some(Path::new("/proc/self/exe"))),
        sources,
    )?;
    let closure =
        rendezvous::observed_closure_digest(&installation.input_root, &contract.artifact_closure)?;
    let credential =
        HostCredential::load(&installation.certificate, &installation.certificate_key)?;
    let coordinator = args
        .coordinator
        .to_socket_addrs()
        .with_context(|| format!("coordinator address `{}`", args.coordinator))?
        .next()
        .context("coordinator address resolves to nothing")?;
    let pin = transport::coordinator_certificate(&contract)?;
    let mut stream = transport::connect(
        coordinator,
        &credential,
        (&pin.0, &pin.1),
        Timeouts::from_plan(&contract),
    )?;
    // Rank 0 carries the group bootstrap: an NCCL unique id (exactly 128 bytes) for a training
    // plan; otherwise the CPU phase creates no communicator and it is a fresh 32-byte token.
    let bootstrap = match (rank, training) {
        (0, true) => Some(nccl_bootstrap_hex()?),
        (0, false) => {
            let mut token = rendezvous::fresh_nonce()?.to_vec();
            token.extend(rendezvous::fresh_nonce()?);
            Some(hex::encode(token))
        }
        _ => None,
    };
    let grant_sha256 = super::super::digest(&planned.grant)?;
    let member = rendezvous::join(
        &mut stream,
        &contract,
        &args.plan_sha256,
        &key,
        |challenge| HelloBody {
            schema: rendezvous::HELLO_SCHEMA.into(),
            run: contract.run.clone(),
            attempt: contract.attempt,
            plan_sha256: args.plan_sha256.clone(),
            rendezvous_id: challenge.body.rendezvous_id.clone(),
            global_rank: rank,
            delivery_id: 1,
            sent_unix_ms: now_ms(),
            host: identity.host.clone(),
            machine_id: identity.machine_id.clone(),
            boot_id: identity.boot_id.clone(),
            device_uuid: planned.device.as_ref().map(|d| d.uuid.clone()),
            source_commit: identity.source_commit.clone(),
            executable_sha256: identity.executable_sha256.clone(),
            runtime: identity.runtime,
            installed_policy_sha256: identity.installed_policy_sha256.clone(),
            artifact_closure_sha256: closure.clone(),
            grant_sha256: grant_sha256.clone(),
            communicator_bootstrap: bootstrap.clone(),
        },
        &now_ms,
    )?;
    publish_immutable(
        &args.outbox,
        Path::new("admitted"),
        member.admission_sha256().as_bytes(),
    )?;
    if training {
        #[cfg(feature = "neural")]
        {
            let Some((completion, spec)) =
                train(args, &contract, &record, &member, &installation, stream)?
            else {
                return Ok(EXIT_LOST_CONTROL);
            };
            let result = serde_json::to_vec(&json!({
                "schema": TRAINING_RESULT_SCHEMA,
                "run": contract.run,
                "attempt": contract.attempt,
                "global_rank": rank,
                "host": identity.host,
                "machine_id": identity.machine_id,
                "boot_id": identity.boot_id,
                "device_uuid": record.cuda_visible_devices,
                "admission_sha256": member.admission_sha256(),
                "input_closure_sha256": closure,
                "smartcluster_job": job,
                "run_json_sha256": hash(&serde_json::to_vec(&spec)?),
                "execution_partition": spec["execution_partition"],
                "completion_sha256": hash(&serde_json::to_vec(&completion)?),
                "parameter_sha256": completion["parameter_sha256"],
                "steps": completion["steps"],
                "updates": contract.counters.planned_updates,
                "numerical": true,
            }))?;
            publish_result(
                args,
                &contract,
                &key,
                &member,
                &result,
                contract.counters.planned_updates,
            )?;
            return Ok(0);
        }
        #[cfg(not(feature = "neural"))]
        anyhow::bail!("{NO_TRAINER}");
    }
    // Admitted: work may start. One update round per heartbeat.
    let heartbeat = Duration::from_millis(contract.rendezvous.heartbeat_timeout_ms);
    stream.set_timeouts(Some(heartbeat), Some(heartbeat))?;
    let limit = contract.rendezvous.max_message_bytes;
    let started = now_ms();
    let mut chain = [0u8; 32];
    let (mut last, mut rounds) = (0u64, 0u64);
    while rounds < contract.counters.planned_updates {
        // Each heartbeat arrives whole within the heartbeat timeout, or control is lost: a
        // coordinator that trickles bytes is as silent as one that sends none.
        let Ok(frame) = rendezvous::read_frame_by(&mut stream, limit, Instant::now() + heartbeat)
        else {
            return Ok(EXIT_LOST_CONTROL);
        };
        match serde_json::from_slice::<ControlFrame>(&frame) {
            Ok(ControlFrame::Heartbeat { delivery_id }) if delivery_id > last => {
                last = delivery_id;
                rounds += 1;
                chain = cpu_proof_round(&chain, rounds, &closure, member.admission_sha256());
            }
            Ok(ControlFrame::Stop { .. }) => return Ok(0),
            // A replayed or regressed heartbeat, or anything else, is not control.
            _ => return Ok(EXIT_LOST_CONTROL),
        }
    }
    let result = serde_json::to_vec(&json!({
        "schema": CPU_PROOF_SCHEMA,
        "run": contract.run,
        "attempt": contract.attempt,
        "global_rank": rank,
        "host": identity.host,
        "machine_id": identity.machine_id,
        "boot_id": identity.boot_id,
        "admission_sha256": member.admission_sha256(),
        "input_closure_sha256": closure,
        "smartcluster_job": job,
        "updates": rounds,
        "chain_sha256": hex::encode(chain),
        "started_unix_ms": started,
        "numerical": false,
    }))?;
    publish_result(args, &contract, &key, &member, &result, rounds)?;
    Ok(0)
}
