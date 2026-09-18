use super::*;

impl DaemonActor {
    pub(super) fn request_schedule(&self) {
        if self.schedule_pending.swap(true, Ordering::AcqRel) {
            return;
        }
        let sender = self.sender.clone();
        let pending = Arc::clone(&self.schedule_pending);
        tokio::spawn(async move {
            if sender.send(ActorCommand::Schedule).await.is_err() {
                pending.store(false, Ordering::Release);
            }
        });
    }

    pub(super) fn queue_freezer(&self, job_id: JobId, attempt: FencedAttempt, frozen: bool) {
        self.queue_freezer_attempt(job_id, attempt, frozen, 0, Duration::ZERO);
    }

    fn queue_freezer_attempt(
        &self,
        job_id: JobId,
        attempt: FencedAttempt,
        frozen: bool,
        retry: u8,
        delay: Duration,
    ) {
        let sender = self.sender.clone();
        tokio::spawn(async move {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let _ = sender
                .send(ActorCommand::FreezerEffect {
                    job_id,
                    attempt,
                    frozen,
                    retry,
                })
                .await;
        });
    }

    pub(super) fn apply_freezer(
        &mut self,
        job_id: JobId,
        attempt: FencedAttempt,
        frozen: bool,
    ) -> Result<(), ActorError> {
        let Some(active) = self.active.get(&job_id) else {
            return Ok(());
        };
        if active.attempt != attempt || (frozen && active.cancelling) {
            return Ok(());
        }
        if !active.allocation.gpu.is_empty() {
            return Err(ActorError::Recovery(
                "refusing to freeze a GPU allocation".to_owned(),
            ));
        }
        let cgroup = active.cgroup.clone().ok_or_else(|| {
            ActorError::Recovery("freezer is unavailable for this allocation".to_owned())
        })?;
        if frozen {
            self.cgroups.freeze(&cgroup)?;
        } else {
            self.cgroups.resume(&cgroup)?;
            if let Some(active) = self.active.get_mut(&job_id) {
                active.overload_paused = false;
            }
        }
        Ok(())
    }

    pub(super) fn retry_freezer(
        &self,
        job_id: JobId,
        attempt: FencedAttempt,
        frozen: bool,
        retry: u8,
    ) {
        if !self
            .active
            .get(&job_id)
            .is_some_and(|active| active.attempt == attempt)
        {
            return;
        }
        let next = retry.saturating_add(1);
        self.queue_freezer_attempt(
            job_id,
            attempt,
            frozen,
            next,
            super::mps::effect_retry_delay(next),
        );
    }

    pub(super) fn transition(
        &self,
        job: &JobRecord,
        target: JobState,
        reason: &str,
    ) -> Result<TransitionReceipt, ActorError> {
        Ok(self
            .store
            .transition_before_effect(TransitionMutation::nonterminal(FencedTransition {
                job_id: job.id,
                from: job.state,
                to: target,
                attempt: job.active_attempt,
                at: now_timestamp(),
                reason: Some(reason.to_owned()),
            }))?)
    }

    pub(super) fn record_transition(
        &mut self,
        receipt: &TransitionReceipt,
    ) -> Result<(), ActorError> {
        self.record_durable_transition(&receipt.transition);
        Ok(())
    }

    pub(super) fn record_durable_transition(&mut self, transition: &FencedTransition) {
        match (transition.from, transition.to) {
            (JobState::Queued, target) if target != JobState::Queued => {
                self.queued_jobs = self.queued_jobs.saturating_sub(1);
                if let Ok(Some(job)) = self.store.get_job(transition.job_id) {
                    if let Some(count) = self.queued_by_tenant.get_mut(&job.spec.tenant) {
                        *count = count.saturating_sub(1);
                    }
                }
            }
            (source, JobState::Queued) if source != JobState::Queued => {
                self.queued_jobs = self.queued_jobs.saturating_add(1);
                if let Ok(Some(job)) = self.store.get_job(transition.job_id) {
                    *self.queued_by_tenant.entry(job.spec.tenant).or_default() += 1;
                }
            }
            _ => {}
        }
        self.publish_counts();
        if let Err(error) = self.publish_transition(transition) {
            tracing::error!(job_id = %transition.job_id, %error, "transition event publication deferred to durable outbox retry");
            self.schedule_transition_event_retry();
        }
    }

    pub(super) fn retry_transition_events(&self) -> Result<(), ActorError> {
        for pending in self.store.pending_transition_events()? {
            self.publish_transition(&pending.transition)?;
        }
        Ok(())
    }

    pub(super) fn schedule_transition_event_retry(&self) {
        let sender = self.sender.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let _ = sender.send(ActorCommand::RetryTransitionEvents).await;
        });
    }

    pub(super) fn publish_transition(
        &self,
        transition: &FencedTransition,
    ) -> Result<(), ActorError> {
        self.journal.append(
            transition.job_id,
            transition.attempt,
            JobEvent::StateTransition(transition.clone()),
            transition.at,
        )?;
        self.store.acknowledge_transition_event(transition)?;
        Ok(())
    }

    pub(super) fn authorized_job(
        &self,
        peer: &AuthenticatedPeer,
        job_id: JobId,
    ) -> Result<JobRecord, ServerError> {
        let job = self
            .store
            .get_job(job_id)
            .map_err(store_error)?
            .ok_or_else(|| ServerError::new(ErrorCode::NotFound, "job not found"))?;
        self.authorizer
            .authorize_job_access(peer, &job.submitted_by)
            .map_err(auth_error)?;
        Ok(job)
    }

    pub(super) fn notify_waiters(&mut self, job: &JobRecord) {
        if !job.state.is_terminal() {
            return;
        }
        if let Some(waiters) = self.waiters.remove(&job.id) {
            for waiter in waiters {
                let _ = waiter.reply.send(Ok(Response::Waited(job.clone())));
            }
        }
    }

    pub(super) fn prune_waiters(&mut self) {
        self.waiters.retain(|_, waiters| {
            waiters.retain(|waiter| !waiter.reply.is_closed());
            !waiters.is_empty()
        });
    }

    pub(super) fn initialize_counts(&mut self) -> Result<(), ActorError> {
        let jobs = self.store.queued_jobs(usize::MAX)?;
        self.queued_jobs = jobs.len() as u64;
        self.queued_by_tenant.clear();
        for job in jobs {
            *self.queued_by_tenant.entry(job.spec.tenant).or_default() += 1;
        }
        self.publish_counts();
        Ok(())
    }

    pub(super) fn publish_counts(&self) {
        self.health
            .set_job_counts(self.queued_jobs, self.active.len() as u64);
    }

    pub(super) fn schedule_count_refresh(&self) {
        let sender = self.sender.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(1)).await;
            let _ = sender.send(ActorCommand::RefreshCounts).await;
        });
    }
}
