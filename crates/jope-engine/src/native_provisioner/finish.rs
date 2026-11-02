use super::*;

pub(super) fn finish_acknowledged(
    config: &Config,
    witness: &WitnessPolicy,
    intent: &Intent,
    proof: &NativeGrantProof,
) -> Result<Value> {
    let sequence = intent.request.body.sequence;
    let requested_sha = &intent.request.body.task_manifest.sha256;
    let trust: NativeWitnessTrust = witness::verify_native_grant_proof(witness, proof)?;
    ensure!(
        bytes(&proof.request)? == bytes(&intent.request)?,
        "remote proof differs from durable signed intent"
    );
    let proof_artifact = install(
        &config.state,
        &format!("{sequence:08}.proof.json"),
        &bytes(proof)?,
    )?;
    let decision = install(
        &config.state,
        &format!("{sequence:08}.eligibility.json"),
        &bytes(&intent.proof)?,
    )?;
    ensure!(
        decision.sha256
            == intent
                .request
                .body
                .signed_grant
                .body
                .decision_receipt_sha256,
        "original eligibility decision changed"
    );
    let grant_artifact = install(
        &config.state,
        &format!("{sequence:08}.grant.json"),
        &bytes(&intent.request.body.signed_grant)?,
    )?;
    ensure!(
        config.release.sha256 == intent.request.body.signed_grant.body.release_receipt_sha256
            && config.coverage.sha256
                == intent
                    .request
                    .body
                    .signed_grant
                    .body
                    .coverage_receipt_sha256,
        "installed release/coverage differs from original grant"
    );
    checked(&config.root, &config.release)?;
    checked(&config.root, &config.coverage)?;
    let policy = json!({"schema":"jope.smartcluster_fit_policy.v1","grant":grant_artifact,"trust":trust,
        "decision":decision,"release":config.release,"coverage":config.coverage,
        "native_grant_proof":proof_artifact,"witness_policy":config.witness_policy});
    let installed = install(
        &config.task_policy_directory,
        &format!("{requested_sha}.json"),
        &bytes(&policy)?,
    )?;
    let receipt = json!({"schema":"jope.native_grant_provisioned.v1","sequence":sequence,
        "task_manifest_sha256":requested_sha,"policy":installed,"numerical_completion":false,
        "original_update_sha256":hash(&bytes(&intent.request)?),"config_sha256":intent.config_sha256});
    // Only this deterministic original installation receipt closes the frontier.
    install(
        &config.state,
        &format!("{sequence:08}.installed.json"),
        &bytes(&receipt)?,
    )?;
    Ok(receipt)
}

/// One durable update at a time. An uncertain SSH response leaves the original
/// intent intact; a later invocation retries its exact bytes before any new work.
pub async fn provision(config_path: &Path, input: &TaskInput) -> Result<Value> {
    ensure!(
        crate::host_os::effective_uid() == 0,
        "provisioner requires root custody"
    );
    protected(config_path, false)?;
    let config_bytes = read(config_path)?;
    let config: Config = serde_json::from_slice(&config_bytes)?;
    ensure!(
        config.schema == "jope.native_grant_provisioner.v1"
            && crate::manifest::file_hash(Path::new("/proc/self/exe"))? == config.executable_sha256,
        "unreviewed provisioner executable/config"
    );
    protected(&config.state, true)?;
    protected(&config.task_policy_directory, true)?;
    protected(&config.signing_key, false)?;
    let key_meta = fs::metadata(&config.signing_key)?;
    ensure!(key_meta.mode() & 0o077 == 0, "provisioner key permissions");
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(config.state.join("lock"))?;
    ensure!(lock.try_lock().is_ok(), "provisioner busy");
    let loop_config: LoopConfig =
        serde_json::from_slice(&checked(&config.root, &config.loop_config)?)?;
    protected(&config.witness_policy.path, false)?;
    let witness: WitnessPolicy =
        serde_json::from_slice(&checked(&config.root, &config.witness_policy)?)?;
    protected(&config.transport.path, false)?;
    let transport =
        WitnessClientConfig::from_transport_file(&config.transport.path, &config.transport.sha256)?;
    ensure!(
        transport.policy.sha256()? == witness.sha256()?,
        "transport immutable base differs"
    );
    let effective = witness::verify_epoch_chain(&witness, &transport.epoch_proofs)?;
    let delegation = effective
        .native_grant_delegation
        .as_ref()
        .context("finite grant delegation absent")?;
    ensure!(
        loop_config.run == delegation.run
            && loop_config.initial_optimizer_step == delegation.initial_optimizer_step
            && loop_config.cycles == 2
            && delegation.cycles == 2
            && loop_config.scientific_policy.sha256 == delegation.scientific_policy_sha256
            && config.executable_sha256 == delegation.controller_sha256
            && config.root == delegation.grant_template.root
            && config.socket == delegation.grant_template.daemon_socket
            && config.tenant == delegation.grant_template.tenant
            && config.project == delegation.grant_template.project,
        "frozen scientific/window/runtime roots differ from delegation"
    );
    let raw: [u8; 32] = read(&config.signing_key)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("provisioner key width"))?;
    let key = SigningKey::from_bytes(&raw);
    ensure!(
        hex::encode(key.verifying_key().to_bytes()) == delegation.provisioner_public_key,
        "provisioner key differs from immutable delegation"
    );

    ensure!(
        config.release.sha256 == delegation.grant_template.release_receipt_sha256
            && config.coverage.sha256 == delegation.grant_template.coverage_receipt_sha256,
        "provisioner template release/coverage mismatch"
    );
    checked(&config.root, &config.release)?;
    checked(&config.root, &config.coverage)?;
    let (first_sequence, initial_previous, initial_spent) = match transport.epoch_proofs.last() {
        None => (1, "0".repeat(64), 0),
        Some(proof) => {
            let body = &proof.epoch_transition.body;
            (
                body.next_update_sequence,
                body.previous_update_sha256.clone(),
                body.original_reserved_evaluations,
            )
        }
    };
    match (&config.predecessor_update_proof, first_sequence) {
        (None, 1) => {}
        (Some(artifact), sequence) if sequence > 1 => {
            protected(&artifact.path, false)?;
            let prior: NativeGrantProof =
                serde_json::from_slice(&checked(&config.root, artifact)?)?;
            witness::verify_native_grant_proof(&witness, &prior)?;
            ensure!(
                prior.request.body.sequence.checked_add(1) == Some(sequence)
                    && hash(&bytes(&prior.request)?) == initial_previous,
                "original predecessor update differs from signed epoch cursor"
            );
        }
        _ => anyhow::bail!("original predecessor grant proof required for nonempty global ledger"),
    }
    let journal_count = journal_length_from(&config.state, first_sequence)?;
    let requested_sha = input.task_manifest.sha256.clone();
    ensure!(
        requested_sha.len() == 64
            && requested_sha
                .bytes()
                .all(|x| x.is_ascii_hexdigit() && !x.is_ascii_uppercase()),
        "task digest invalid"
    );
    let mut sequence = first_sequence;
    let mut previous = initial_previous;
    let mut spent = initial_spent;
    let mut existing = None;
    loop {
        ensure!(sequence <= 100_000, "finite provisioner journal exhausted");
        let name = format!("{sequence:08}.intent.json");
        let p = config.state.join(&name);
        if !p.exists() {
            break;
        }
        protected(&p, false)?;
        let intent: Intent = serde_json::from_slice(&read(&p)?)?;
        validate_intent(
            &config,
            &config_bytes,
            &witness,
            &transport.epoch_proofs,
            &intent,
            sequence,
            &previous,
        )?;
        let ackpath = config.state.join(format!("{sequence:08}.proof.json"));
        if !ackpath.exists() {
            ensure!(
                sequence == journal_count,
                "unsettled intent precedes later journal updates"
            );
            ensure!(
                intent.request.body.task_manifest.sha256 == requested_sha,
                "original pending grant must be settled first"
            );
            existing = Some(intent);
            break;
        }
        protected(&ackpath, false)?;
        let ack: NativeGrantProof = serde_json::from_slice(&read(&ackpath)?)?;
        witness::verify_native_grant_proof(&witness, &ack)?;
        ensure!(
            bytes(&ack.request)? == bytes(&intent.request)?,
            "journal ACK differs from original intent"
        );
        finish_acknowledged(&config, &witness, &intent, &ack)?;
        if intent.request.body.task_manifest.sha256 == requested_sha {
            existing = Some(intent);
            break;
        }
        spent = spent
            .checked_add(u64::from(intent.request.body.evaluations))
            .context("budget overflow")?;
        ensure!(
            spent <= delegation.total_evaluation_budget,
            "stored journal exceeds delegated evaluation budget"
        );
        previous = hash(&bytes(&intent.request)?);
        sequence += 1;
    }
    let intent = if let Some(v) = existing {
        v
    } else {
        let proof_inputs = GrantProofInputs {
            root: config.root.clone(),
            loop_state: config.loop_state.clone(),
            graph_state: config.graph_state.clone(),
            worker_state: config.worker_state.clone(),
            tenant: config.tenant.clone(),
            project: config.project.clone(),
            config: config.loop_config.clone(),
            request: input.request.clone(),
            admission: input.admission.clone(),
            task_manifest: input.task_manifest.clone(),
        };
        let proof = Loop::prove_native_grant_request(
            &proof_inputs,
            &jain_smartcluster::Client::new(config.socket.clone()),
        )
        .await?;
        let evaluations: u32 = proof.evaluations.try_into()?;
        ensure!(
            !proof.authorization_issued
                && proof.verifier_executable_sha256 == config.executable_sha256
                && proof.run == delegation.run
                && proof.generation < 2
                && proof.task_manifest_sha256 == requested_sha
                && proof.scientific_policy.sha256 == delegation.scientific_policy_sha256
                && spent
                    .checked_add(u64::from(evaluations))
                    .is_some_and(|x| x <= delegation.total_evaluation_budget),
            "proof exceeds immutable provisioner delegation"
        );
        let issued = now()?;
        ensure!(
            issued < delegation.deadline_unix_ms
                && issued < proof.task_deadline.as_unix_millis()
                && issued >= delegation.grant_template.period_start_unix_ms
                && issued < delegation.grant_template.period_end_unix_ms
                && evaluations > 0
                && evaluations <= delegation.grant_template.max_candidates_per_task,
            "original task/delegation expired"
        );
        // Identity is computed without exposing a partial transaction. The
        // complete signed intent is the FIRST durable final artifact.
        let decision = Artifact {
            path: config.state.join(format!("{sequence:08}.eligibility.json")),
            sha256: hash(&bytes(&proof)?),
        };
        let mut grant = delegation.grant_template.clone();
        grant.approved_task_sha256 = BTreeSet::from([requested_sha.clone()]);
        grant.max_candidates_per_task = evaluations;
        grant.decision_receipt_sha256 = decision.sha256;
        grant.period_start_unix_ms = issued.max(grant.period_start_unix_ms);
        grant.period_end_unix_ms = grant
            .period_end_unix_ms
            .min(proof.task_deadline.as_unix_millis());
        let grant_sha256 = hash(&bytes(&grant)?);
        let signed_grant = SignedNativeGrant {
            signature: sign(&key, b"jope.smartcluster_fit_grant.signature.v1\0", &grant)?,
            body: grant,
        };
        let body = AddNativeGrantBody {
            schema: "parallel.add_native_grant.v1".into(),
            base_policy_sha256: effective.sha256()?,
            delegation_sha256: delegation.sha256()?,
            sequence,
            previous_update_sha256: previous,
            signed_grant,
            grant_sha256,
            loop_request: native_artifact(&config.root, &input.request)?,
            graph_admission: native_artifact(&config.root, &input.admission)?,
            task_manifest: native_artifact(&config.root, &input.task_manifest)?,
            run: proof.run.clone(),
            generation: proof.generation,
            role: match proof.role {
                Role::Direct => NativeGrantRole::Direct,
                Role::Continuation => NativeGrantRole::Continuation,
                _ => anyhow::bail!("non-numerical grant role"),
            },
            evaluations,
            issued_unix_ms: issued,
        };
        let request = SignedAddNativeGrant {
            signature: sign(&key, b"parallel.add_native_grant.request.v1\0", &body)?,
            body,
        };
        // Reject every bound violation before creating a durable frontier that
        // the witness could never acknowledge.
        witness::validate_native_grant_request_for_epoch(
            &witness,
            &transport.epoch_proofs,
            &request,
        )?;
        let intent = Intent {
            schema: "jope.native_grant_intent.v1".into(),
            config_sha256: hash(&config_bytes),
            proof,
            request,
        };
        install(
            &config.state,
            &format!("{sequence:08}.intent.json"),
            &bytes(&intent)?,
        )?;
        intent
    };
    ensure!(
        intent.request.body.loop_request == native_artifact(&config.root, &input.request)?
            && intent.request.body.graph_admission
                == native_artifact(&config.root, &input.admission)?
            && intent.request.body.task_manifest
                == native_artifact(&config.root, &input.task_manifest)?,
        "retry differs from original exact artifacts"
    );
    let decision = install(
        &config.state,
        &format!("{sequence:08}.eligibility.json"),
        &bytes(&intent.proof)?,
    )?;
    ensure!(
        decision.sha256
            == intent
                .request
                .body
                .signed_grant
                .body
                .decision_receipt_sha256,
        "original eligibility decision differs"
    );
    let ackname = format!("{sequence:08}.proof.json");
    let ackpath = config.state.join(&ackname);
    let proof: NativeGrantProof = if ackpath.exists() {
        serde_json::from_slice(&read(&ackpath)?)?
    } else {
        let deadline = crate::host_os::monotonic_ns()
            .context("monotonic clock")?
            .checked_add(30_000_000_000)
            .context("transport deadline overflow")?;
        witness::add_native_grant_remote(&transport, &intent.request, deadline)?
    };
    finish_acknowledged(&config, &witness, &intent, &proof)
}
