fn connection(state: &Arc<State>, stream: &mut UnixStream, stopping: bool) -> Result<()> {
    let peer = authenticated_peer(stream)?;
    match receive_json::<Request>(stream, Duration::from_secs(5))? {
        Request::SubmitNative { request } => {
            native_scq::connection(state, stream, &peer, request, stopping)?
        }
        Request::Funding {
            descriptor_sha256,
            deadline_monotonic_ns,
        } => {
            ensure!(!stopping, "authority draining: funding unavailable");
            funding_ipc::serve_request(
                state,
                stream,
                &peer,
                &descriptor_sha256,
                deadline_monotonic_ns,
            )?;
        }
        Request::Submit {
            provenance,
            descriptor_sha256,
        } => {
            ensure!(!stopping, "authority draining: new requests unavailable");
            ensure!(
                state
                    .active
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| (n < state
                        .config
                        .max_workers)
                        .then_some(n + 1))
                    .is_ok(),
                "all service slots occupied; no dispatch occurred"
            );
            let _active = CountGuard {
                counter: &state.active,
            };
            state.requests.fetch_add(1, Ordering::SeqCst);
            let input = File::from(receive_fd(stream, Duration::from_secs(5))?);
            let bytes = read_sealed(&input, state.config.max_descriptor_bytes)?;
            ensure!(
                sha(&bytes) == descriptor_sha256,
                "sealed descriptor content changed"
            );
            let inputs: ServiceInputs = serde_json::from_slice(&bytes)?;
            let fit = inputs.verify(
                &state.identity.executable_sha256,
                &state.runtime,
                &state.config,
            )?;
            let mut provenance = provenance;
            ensure!(
                provenance.details.is_object(),
                "request provenance must be an object"
            );
            let request_deadline = provenance.details["deadline_monotonic_ns"]
                .as_u64()
                .context("request absolute monotonic deadline absent or malformed")?;
            remaining_request(request_deadline)?;
            provenance.details["requesting_process"] = serde_json::to_value(peer)?;
            let claim = {
                let _dispatch = state.dispatch_gate()?;
                remaining_request(request_deadline)?;
                lock(&state.authority)?.claim(&fit, &provenance)?
            };
            let result = match claim {
                ClaimOutcome::Claimed(claim) => {
                    supervise(state, &fit, &inputs, &input, &claim, request_deadline, None)?
                }
                ClaimOutcome::Existing {
                    state: status,
                    completion_sha256: Some(sha),
                    ..
                } if status == "completed" || status == "terminal_failure" => {
                    state.reused.fetch_add(1, Ordering::SeqCst);
                    result_from_original(state, &sha, true)?
                }
                other => bail!("original fit unavailable; never replay: {other:?}"),
            };
            // This is intentionally last: a lost client connection cannot stop
            // supervision, receipt publication or the permanent seen entry.
            send_json(
                stream,
                &json!({"ok":true,"result":result}),
                Duration::from_secs(5),
            )?;
        }
        Request::Consume { ticket, fit_key } => {
            let mut pending = lock(&state.pending)?;
            let slot = pending
                .get(&ticket)
                .context("unknown or already accepted executor ticket")?;
            ensure!(
                peer == slot.process,
                "executor ticket presented by another process"
            );
            let slot = pending
                .remove(&ticket)
                .context("executor ticket already transferred")?;
            slot.send
                .send((stream.try_clone()?, fit_key))
                .map_err(|_| anyhow::anyhow!("original supervisor unavailable"))?;
        }
        Request::Verify {
            completion_sha256,
            scope,
        } => {
            {
                lock(&state.authority)?.verify_record_in_scope(&completion_sha256, &scope)?;
            }
            let result = result_from_original(state, &completion_sha256, true)?;
            send_json(
                stream,
                &json!({"ok":true,"result":result}),
                Duration::from_secs(5),
            )?;
        }
        Request::QueryEvaluate { descriptor_sha256 } => {
            ensure!(
                !stopping,
                "authority draining: new query requests unavailable"
            );
            ensure!(
                state
                    .active
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| (n < state
                        .config
                        .max_workers)
                        .then_some(n + 1))
                    .is_ok(),
                "all service slots occupied; no query accepted"
            );
            let _active = CountGuard {
                counter: &state.active,
            };
            let descriptor = File::from(receive_fd(stream, Duration::from_secs(5))?);
            let bytes = read_sealed(&descriptor, state.config.max_descriptor_bytes)?;
            ensure!(
                sha(&bytes) == descriptor_sha256,
                "sealed query descriptor changed"
            );
            let inputs: query::QueryInputs = serde_json::from_slice(&bytes)?;
            remaining_request(inputs.deadline_monotonic_ns)?;
            ensure!(
                inputs.descriptor_sha256()? == descriptor_sha256,
                "query descriptor must use canonical serialization"
            );
            // Separate access permission is checked before opening outcome
            // blobs or creating an accepted query. A fit scope never grants it.
            #[cfg(test)]
            let access = if state
                .config
                .synthetic_query_access
                .contains(&descriptor_sha256)
            {
                query::synthetic_access_for_test(&inputs)?
            } else {
                query::authorize(&inputs)?
            };
            #[cfg(not(test))]
            let access = query::authorize(&inputs)?;
            remaining_request(inputs.deadline_monotonic_ns)?;
            let claim = lock(&state.authority)?.prepare_query(&inputs, &access)?;
            let (record, reused) = match claim {
                QueryClaim::Existing(record) => (record, true),
                QueryClaim::Accepted => {
                    #[cfg(test)]
                    ensure!(
                        !state.config.query_fail_after_accept,
                        "synthetic query failure after durable acceptance"
                    );
                    remaining_request(inputs.deadline_monotonic_ns)?;
                    let (completion, model) = {
                        let mut authority = lock(&state.authority)?;
                        authority
                            .verify_record_in_scope(&inputs.completion_sha256, &inputs.scope)?;
                        (
                            authority.read_blob(&inputs.completion_sha256)?,
                            authority.model_for_completion(&inputs.completion_sha256)?,
                        )
                    };
                    ensure!(
                        current_runtime_digest(&state.config.contract)? == state.runtime,
                        "query producer runtime changed before prediction"
                    );
                    let artifacts = query::score_verified(
                        &completion,
                        &model,
                        &inputs,
                        &access,
                        &state.runtime,
                        |model, rows, deadline| query_worker::predict(state, model, rows, deadline),
                    )?;
                    ensure!(
                        current_runtime_digest(&state.config.contract)? == state.runtime,
                        "query producer runtime changed during prediction"
                    );
                    remaining_request(inputs.deadline_monotonic_ns)?;
                    (
                        lock(&state.authority)?.commit_query(&inputs, &artifacts)?,
                        false,
                    )
                }
            };
            let result = result_from_query(state, record, reused)?;
            // Lost replies preserve the committed original query; requests are
            // never resubmitted automatically by this client.
            send_json(
                stream,
                &json!({"ok":true,"result":result}),
                remaining_request(inputs.deadline_monotonic_ns)?,
            )?;
        }
        Request::QueryRetrieve {
            descriptor_sha256,
            deadline_monotonic_ns,
        } => {
            remaining_request(deadline_monotonic_ns)?;
            let descriptor = File::from(receive_fd(
                stream,
                remaining_request(deadline_monotonic_ns)?,
            )?);
            let bytes = read_sealed(&descriptor, state.config.max_descriptor_bytes)?;
            ensure!(
                sha(&bytes) == descriptor_sha256,
                "sealed query retrieval descriptor changed"
            );
            let inputs: query::QueryInputs = serde_json::from_slice(&bytes)?;
            ensure!(
                inputs.descriptor_sha256()? == descriptor_sha256,
                "query retrieval descriptor must use canonical serialization"
            );
            #[cfg(test)]
            let access = if state
                .config
                .synthetic_query_access
                .contains(&descriptor_sha256)
            {
                query::synthetic_access_for_test(&inputs)?
            } else {
                query::authorize(&inputs)?
            };
            #[cfg(not(test))]
            let access = query::authorize(&inputs)?;
            remaining_request(deadline_monotonic_ns)?;
            let record = lock(&state.authority)?.retrieve_query(&inputs, &access)?;
            let result = result_from_query(state, record, true)?;
            send_json(
                stream,
                &json!({"ok":true,"result":result}),
                remaining_request(deadline_monotonic_ns)?,
            )?;
        }
        Request::Stats => send_json(
            stream,
            &json!({"ok":true,"stats":state.stats()}),
            Duration::from_secs(5),
        )?,
    }
    Ok(())
}

fn result_from_query(
    state: &State,
    record: QueryRecord,
    reused: bool,
) -> Result<ServiceQueryResult> {
    let object = |digest: &str| state.config.authority_root.join("objects").join(digest);
    let authority = lock(&state.authority)?;
    let inputs: query::QueryInputs =
        serde_json::from_slice(&authority.read_blob(&record.request_sha256)?)?;
    let receipt = query::verify_artifacts(
        &authority.read_blob(&record.completion_sha256)?,
        &authority.read_blob(&record.model_sha256)?,
        &inputs,
        &authority.read_blob(&record.receipt_sha256)?,
        &authority.read_blob(&record.predictions_sha256)?,
        &authority.read_blob(&record.metrics_sha256)?,
    )?;
    ensure!(
        receipt.query_key == record.query_key
            && receipt.authority_instance == state.identity.authority_instance,
        "committed query authority/key mismatch"
    );
    Ok(ServiceQueryResult {
        authority_instance: state.identity.authority_instance.clone(),
        request_path: object(&record.request_sha256),
        completion_path: object(&record.completion_sha256),
        model_path: object(&record.model_sha256),
        predictions_path: object(&record.predictions_sha256),
        metrics_path: object(&record.metrics_sha256),
        receipt_path: object(&record.receipt_sha256),
        record,
        receipt,
        reused,
    })
}
