use super::*;

pub(super) struct ScqHoldClient {
    writer: UnixStream,
    reader: BufReader<UnixStream>,
}

impl ScqHoldClient {
    pub(super) fn connect(socket: &Path) -> Result<Self> {
        let writer = UnixStream::connect(socket)
            .with_context(|| format!("connect SCQ socket {}", socket.display()))?;
        writer.set_read_timeout(Some(Duration::from_secs(30)))?;
        writer.set_write_timeout(Some(Duration::from_secs(30)))?;
        let reader = BufReader::new(writer.try_clone()?);
        Ok(Self { writer, reader })
    }

    pub(super) fn hold(&mut self, job_id: &str) -> Result<()> {
        self.request("Hold", job_id)
    }

    pub(super) fn cancel(&mut self, job_id: &str) -> Result<()> {
        self.request("Cancel", job_id)
    }

    pub(super) fn release(&mut self, job_id: &str) -> Result<()> {
        self.request("Release", job_id)
    }

    fn request(&mut self, operation: &str, job_id: &str) -> Result<()> {
        let request = match operation {
            "Hold" => json!({"Hold": {"id": job_id}}),
            "Cancel" => json!({"Cancel": {"id": job_id}}),
            "Release" => json!({"Release": {"id": job_id}}),
            _ => bail!("unsupported SCQ mutation"),
        };
        serde_json::to_writer(&mut self.writer, &request)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        let mut response = String::new();
        if self.reader.read_line(&mut response)? == 0 {
            bail!("SCQ socket closed before {operation} response");
        }
        let response: Value =
            serde_json::from_str(response.trim()).context("parse SCQ mutation response")?;
        if let Some(error) = response
            .get("Error")
            .or_else(|| response.get("Err"))
            .or_else(|| response.get("error"))
            .filter(|value| !value.is_null())
        {
            bail!("SCQ {operation} rejected {job_id}: {error}");
        }
        if response.get("ok").and_then(Value::as_bool) == Some(false) {
            bail!("SCQ {operation} rejected {job_id}: {response}");
        }
        Ok(())
    }
}

pub(super) fn claim_matches_candidate(claim: &Claim, candidate: &HoldCandidate) -> bool {
    claim.state == "submitted"
        && claim.manifest_sha256 == candidate.manifest_sha256
        && claim.manifest.to_string_lossy() == candidate.manifest
}

pub(super) fn hold_over_socket(
    candidates: &[HoldCandidate],
    before: &HashMap<String, (String, String)>,
    before_claims: &HashMap<String, Claim>,
    socket: &Path,
    workers: usize,
) -> Result<Vec<HoldResult>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_size = candidates.len().div_ceil(workers);
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers.min(candidates.len()))
        .thread_name(|index| format!("queue-hold-{index}"))
        .build()?;
    Ok(pool.install(|| {
        candidates
            .par_chunks(chunk_size)
            .flat_map_iter(|chunk| {
                let mut client = ScqHoldClient::connect(socket);
                chunk.iter().map(move |candidate| {
                    let live = before.get(&candidate.job_id);
                    let still_eligible = live.is_some_and(|(project, state)| {
                        project == &candidate.project && is_holdable_state(state)
                    }) && before_claims
                        .get(&candidate.job_id)
                        .is_some_and(|claim| claim_matches_candidate(claim, candidate));
                    let command_error = if still_eligible {
                        socket_mutation_with_reconnect(
                            &mut client,
                            socket,
                            "Hold",
                            &candidate.job_id,
                        )
                    } else {
                        None
                    };
                    HoldResult {
                        job_id: candidate.job_id.clone(),
                        state_before_hold: live.map(|(_, state)| state.clone()),
                        hold_requested: still_eligible,
                        command_error,
                        state_after_hold: None,
                    }
                })
            })
            .collect()
    }))
}

pub(super) fn release_over_socket(
    candidates: &[HoldCandidate],
    before: &HashMap<String, (String, String)>,
    before_claims: &HashMap<String, Claim>,
    socket: &Path,
    workers: usize,
    active_checkpoint: &str,
) -> Result<Vec<ReleaseResult>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_size = candidates.len().div_ceil(workers);
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers.min(candidates.len()))
        .thread_name(|index| format!("queue-release-{index}"))
        .build()?;
    Ok(pool.install(|| {
        candidates
            .par_chunks(chunk_size)
            .flat_map_iter(|chunk| {
                let mut client = ScqHoldClient::connect(socket);
                chunk.iter().map(move |candidate| {
                    let live = before.get(&candidate.job_id);
                    let claim = before_claims.get(&candidate.job_id);
                    let fresh_manifest_valid = claim
                        .filter(|claim| claim_matches_candidate(claim, candidate))
                        .and_then(|claim| validate_manifest(claim).ok())
                        .and_then(|manifest| manifest_checkpoint_binding(&manifest))
                        .as_deref()
                        == Some(active_checkpoint);
                    let still_eligible = live.is_some_and(|(project, state)| {
                        project == SELFPLAY_PROJECT && state == "held"
                    }) && fresh_manifest_valid;
                    let command_error = if still_eligible {
                        socket_mutation_with_reconnect(
                            &mut client,
                            socket,
                            "Release",
                            &candidate.job_id,
                        )
                    } else {
                        Some("state_claim_manifest_or_checkpoint_changed_before_release".to_owned())
                    };
                    ReleaseResult {
                        job_id: candidate.job_id.clone(),
                        state_before_release: live.map(|(_, state)| state.clone()),
                        release_requested: still_eligible,
                        command_error,
                        state_after_release: None,
                    }
                })
            })
            .collect()
    }))
}

pub(super) fn cancel_over_socket(
    candidates: &[HoldCandidate],
    before: &HashMap<String, (String, String)>,
    before_claims: &HashMap<String, Claim>,
    socket: &Path,
    workers: usize,
    recovery_ids: &BTreeSet<String>,
) -> Result<Vec<RetirementResult>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_size = candidates.len().div_ceil(workers);
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers.min(candidates.len()))
        .thread_name(|index| format!("queue-cancel-{index}"))
        .build()?;
    Ok(pool.install(|| {
        candidates
            .par_chunks(chunk_size)
            .flat_map_iter(|chunk| {
                let mut client = ScqHoldClient::connect(socket);
                chunk.iter().map(move |candidate| {
                    let live = before.get(&candidate.job_id);
                    let claim_matches = before_claims
                        .get(&candidate.job_id)
                        .is_some_and(|claim| claim_matches_candidate(claim, candidate));
                    let still_held = live.is_some_and(|(project, state)| {
                        project == &candidate.project && state == "held"
                    });
                    let recoverable_absence = recovery_ids.contains(&candidate.job_id)
                        && live.is_none_or(|(_, state)| state == "cancelled");
                    let cancel_requested = claim_matches && still_held;
                    let command_error = if cancel_requested {
                        socket_mutation_with_reconnect(
                            &mut client,
                            socket,
                            "Cancel",
                            &candidate.job_id,
                        )
                    } else if claim_matches && recoverable_absence {
                        None
                    } else {
                        Some("state_or_claim_changed_before_cancel".to_owned())
                    };
                    RetirementResult {
                        job_id: candidate.job_id.clone(),
                        state_before_cancel: live.map(|(_, state)| state.clone()),
                        cancel_requested,
                        command_error,
                        state_after_cancel: None,
                        pair_updates: 0,
                        claim_updated: false,
                        recovery_from_prior_preview: recoverable_absence,
                        registry_error: None,
                    }
                })
            })
            .collect()
    }))
}

pub(super) fn socket_mutation_with_reconnect(
    client: &mut Result<ScqHoldClient>,
    socket: &Path,
    operation: &str,
    job_id: &str,
) -> Option<String> {
    let mut last_error = None;
    for _ in 0..2 {
        if client.is_err() {
            *client = ScqHoldClient::connect(socket);
        }
        let result = match client {
            Ok(client) if operation == "Hold" => client.hold(job_id),
            Ok(client) if operation == "Cancel" => client.cancel(job_id),
            Ok(client) if operation == "Release" => client.release(job_id),
            Ok(_) => return Some("unsupported SCQ mutation".to_owned()),
            Err(error) => Err(anyhow::anyhow!("{error:#}")),
        };
        match result {
            Ok(()) => return None,
            Err(error) => {
                last_error = Some(error.to_string());
                *client = ScqHoldClient::connect(socket);
            }
        }
    }
    last_error.or_else(|| Some("SCQ mutation failed without an error".to_owned()))
}
