use super::*;

impl ProposalOutbox {
    /// Performs the lease operation and returns an error if it cannot be completed.
    pub fn lease(&mut self, key: &str, owner: &str, now_ms: u64, ttl_ms: u64) -> Result<Lease> {
        ensure!(
            !owner.is_empty() && ttl_ms > 0,
            "lease owner and duration are required"
        );
        let deadline = self.record(key)?.context.deadline_unix_ms;
        ensure!(
            now_ms < deadline,
            "original I16 execution deadline has elapsed"
        );
        let mut status = self.status(key)?.clone();
        ensure!(
            (status.attempts.len() as u64) < self.limits.max_attempts,
            "proposal physical attempt cap reached"
        );
        if let Some(lease) = &status.lease {
            let started = status
                .attempts
                .last()
                .is_some_and(|a| a.generation == lease.generation);
            ensure!(
                now_ms >= lease.expires_unix_ms && !started,
                "live or started lease cannot be superseded without terminal/reap proof"
            );
        }
        if let Some(last) = status.attempts.last() {
            ensure!(
                last.terminal.as_ref().is_some_and(|t| matches!(
                    t,
                    TerminalKind::Failed {
                        retryable: true,
                        ..
                    }
                )),
                "unresolved or non-retryable physical attempt cannot be dispatched again"
            );
        }
        status.generation = status
            .generation
            .checked_add(1)
            .context("lease generation overflow")?;
        let expires_unix_ms = now_ms
            .checked_add(ttl_ms)
            .context("lease expiry overflow")?
            .min(deadline);
        let lease = Lease {
            request_key: key.to_string(),
            generation: status.generation,
            owner: owner.to_string(),
            expires_unix_ms,
        };
        status.lease = Some(lease.clone());
        self.persist_status(status)?;
        Ok(lease)
    }

    /// Persist dispatch-start before handing control to the release adapter. An expired started
    /// attempt remains unresolved and blocks another start until terminal/reap evidence arrives.
    pub fn start(&mut self, lease: &Lease, now_ms: u64) -> Result<()> {
        let mut status = self.status(&lease.request_key)?.clone();
        check_fence(&status, lease, now_ms)?;
        ensure!(
            status.attempts.last().is_none_or(|a| a.terminal.is_some()),
            "physical request already started or unresolved"
        );
        ensure!(
            (status.attempts.len() as u64) < self.limits.max_attempts,
            "physical attempt cap reached"
        );
        status.attempts.push(PhysicalAttempt {
            generation: lease.generation,
            owner: lease.owner.clone(),
            started_unix_ms: now_ms,
            unknown: false,
            terminal: None,
            ack_sha256: None,
            physical_cost_units: 0,
        });
        self.persist_status(status)
    }

    /// Performs the finish operation and returns an error if it cannot be completed.
    pub fn finish(
        &mut self,
        lease: &Lease,
        terminal: TerminalKind,
        physical_cost_units: u64,
    ) -> Result<()> {
        let mut status = self.status(&lease.request_key)?.clone();
        let record_sha256 = self.record(&lease.request_key)?.record_sha256.clone();
        let attempt = status
            .attempts
            .last_mut()
            .context("dispatch has no durable start")?;
        ensure!(
            attempt.generation == lease.generation && attempt.owner == lease.owner,
            "stale lease cannot terminalize this attempt or charge exposure"
        );
        ensure!(
            attempt.terminal.is_none(),
            "attempt already has a terminal receipt"
        );
        if let TerminalKind::Failed { receipt_sha256, .. } = &terminal {
            ensure!(
                is_hex64(receipt_sha256),
                "terminal receipt digest is invalid"
            );
        }
        if let TerminalKind::Refused { reason_sha256 } = &terminal {
            ensure!(is_hex64(reason_sha256), "refusal reason digest is invalid");
        }
        if let TerminalKind::Released {
            physical_attempt_id,
        } = &terminal
        {
            ensure!(
                !physical_attempt_id.is_empty(),
                "physical attempt identity is required"
            );
        }
        attempt.terminal = Some(terminal);
        attempt.physical_cost_units = physical_cost_units;
        attempt.ack_sha256 = Some(ack_sha256(
            &lease.request_key,
            &record_sha256,
            attempt.generation,
            attempt
                .terminal
                .as_ref()
                .context("terminal receipt absent")?,
            physical_cost_units,
        )?);
        status.lease = None;
        self.persist_status(status)
    }

    /// Performs the mark unknown operation and returns an error if it cannot be completed.
    pub fn mark_unknown(&mut self, lease: &Lease) -> Result<()> {
        let mut status = self.status(&lease.request_key)?.clone();
        let attempt = status
            .attempts
            .last_mut()
            .context("dispatch has no durable start")?;
        ensure!(
            attempt.generation == lease.generation
                && attempt.owner == lease.owner
                && attempt.terminal.is_none(),
            "stale lease cannot mark another attempt unknown"
        );
        attempt.unknown = true;
        status.lease = None;
        self.persist_status(status)
    }

    /// Resolves an unknown start after the caller verifies an authenticated terminal/reap receipt.
    /// I16's production verifier must be wired by the integrator; this draft worker API only
    /// binds the verified receipt digest and records physical effort, never optimizer exposure.
    pub fn resolve_unknown(
        &mut self,
        key: &str,
        receipt_sha256: &str,
        physical_cost_units: u64,
        retryable: bool,
        verifier: &impl TerminalProofVerifier,
    ) -> Result<()> {
        ensure!(
            is_hex64(receipt_sha256),
            "reap/terminal proof digest is invalid"
        );
        let mut status = self.status(key)?.clone();
        let record_sha256 = self.record(key)?.record_sha256.clone();
        let attempt = status
            .attempts
            .last_mut()
            .context("no started attempt to resolve")?;
        ensure!(
            attempt.unknown && attempt.terminal.is_none(),
            "attempt is not awaiting unknown-start resolution"
        );
        verifier.verify_terminal_or_reap(key, attempt.generation, receipt_sha256)?;
        attempt.terminal = Some(TerminalKind::Failed {
            receipt_sha256: receipt_sha256.to_string(),
            retryable,
        });
        attempt.physical_cost_units = physical_cost_units;
        attempt.ack_sha256 = Some(ack_sha256(
            key,
            &record_sha256,
            attempt.generation,
            attempt
                .terminal
                .as_ref()
                .context("terminal receipt absent")?,
            physical_cost_units,
        )?);
        self.persist_status(status)
    }

    fn persist_status(&mut self, status: DeliveryStatus) -> Result<()> {
        let bytes = canonical_json(&status)?;
        ensure!(
            bytes.len() as u64 <= self.limits.max_status_bytes,
            "proposal terminal state exceeds reserved status bytes"
        );
        let current = self.durable_bytes;
        let old = canonical_json(self.status(&status.request_key)?)?.len() as u64;
        let mut headroom = 0u64;
        for (key, existing) in &self.statuses {
            let value = if key == &status.request_key {
                &status
            } else {
                existing
            };
            let size = canonical_json(value)?.len() as u64;
            headroom = headroom
                .checked_add(
                    self.limits
                        .max_status_bytes
                        .checked_sub(size)
                        .context("proposal status exceeds reserved headroom")?,
                )
                .context("proposal terminal headroom overflow")?;
        }
        ensure!(
            current
                .checked_sub(old)
                .and_then(|n| n.checked_add(bytes.len() as u64))
                .and_then(|n| n.checked_add(headroom))
                .is_some_and(|n| n <= self.limits.max_total_bytes),
            "proposal total-state byte cap reached"
        );
        atomic_write(
            &self.request_dir(&status.request_key).join("status.json"),
            &bytes,
        )?;
        self.durable_bytes = current
            .checked_sub(old)
            .and_then(|n| n.checked_add(bytes.len() as u64))
            .context("proposal durable-byte accounting overflow")?;
        self.statuses.insert(status.request_key.clone(), status);
        Ok(())
    }

    pub(super) fn request_dir(&self, key: &str) -> PathBuf {
        self.root.join("requests").join(key)
    }
}
