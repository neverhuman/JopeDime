use super::*;

impl Keeper {
    pub(super) fn adopt_scheduler_jobs(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<usize> {
        let mut rows = Vec::new();
        for claim in claims.values() {
            if !matches!(claim.state.as_str(), "checked" | "submitting") {
                continue;
            }
            let Some(job) = scheduler.get(claim.job_id.as_str()) else {
                continue;
            };
            if !scheduler_job_is_adoptable(job) {
                continue;
            }
            let Ok(manifest) = self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves)
            else {
                continue;
            };
            let owner = self.validate_pair_reservations(
                &manifest,
                claim.owner.as_deref(),
                Some(claim.job_id.as_str()),
            )?;
            rows.push((manifest, owner, claim.job_id.clone()));
        }
        if rows.is_empty() {
            return Ok(0);
        }
        {
            let tx = self
                .backlog
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            {
                let mut attempts = tx.prepare_cached(
                    "SELECT COUNT(*) FROM keeper_attempts WHERE manifest_sha256=? AND job_id=? \
                     AND state IN ('checked','submitting','submitted')",
                )?;
                for (manifest, _, job_id) in &rows {
                    let count: i64 =
                        attempts.query_row(params![manifest.sha256, job_id], |row| row.get(0))?;
                    if count != manifest.pairs.len() as i64 {
                        bail!("scheduler adoption has incomplete active attempt history");
                    }
                }
            }
            tx.commit()?;
        }
        let submissions = rows
            .iter()
            .map(|(manifest, owner, job_id)| (manifest, owner.as_str(), job_id.as_str()))
            .collect::<Vec<_>>();
        self.record_submissions(&submissions)?;
        Ok(rows.len())
    }

    pub(super) fn record_submissions(
        &mut self,
        submissions: &[(&ValidManifest, &str, &str)],
    ) -> Result<()> {
        if submissions.is_empty() {
            return Ok(());
        }
        {
            let tx = self
                .pairs
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            for (manifest, owner, job_id) in submissions {
                ensure_native_reserved_attempts(&tx, &self.pair_schema, manifest, owner, job_id)?;
                update_pairs_submitted(&tx, &self.pair_schema, manifest, owner, job_id)?;
                update_native_attempts_submitted(&tx, &self.pair_schema, manifest, owner, job_id)?;
            }
            tx.commit()?;
        }
        {
            let tx = self
                .backlog
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            {
                let mut update_claim = tx.prepare_cached(
                    "UPDATE claims SET state='submitted',attempts=attempts+1,\
                     reservation_owner=?,updated_at_ms=? WHERE manifest_sha256=?",
                )?;
                let mut update_attempt = tx.prepare_cached(
                    "UPDATE keeper_attempts SET state='submitted',job_id=? \
                     WHERE manifest_sha256=? AND state IN ('checked','submitting')",
                )?;
                for (manifest, owner, job_id) in submissions {
                    update_claim.execute(params![owner, now_ms() as i64, manifest.sha256])?;
                    update_attempt.execute(params![job_id, manifest.sha256])?;
                }
            }
            tx.commit()?;
        }
        Ok(())
    }

    pub(super) fn submit_claims(&mut self, claims: &[ReadyClaim]) -> Result<(usize, usize, u64)> {
        {
            let tx = self
                .backlog
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            for row in claims {
                let changed = tx.execute(
                    "UPDATE claims SET state='submitting',updated_at_ms=? \
                     WHERE manifest_sha256=? AND state IN ('checked','submitting')",
                    params![now_ms() as i64, row.manifest.sha256],
                )?;
                if changed != 1 {
                    bail!("launch-ready claim changed before submission");
                }
                let attempts_changed = tx.execute(
                    "UPDATE keeper_attempts SET state='submitting' \
                     WHERE manifest_sha256=? AND state IN ('checked','submitting')",
                    [row.manifest.sha256.as_str()],
                )?;
                if attempts_changed != row.manifest.pairs.len() {
                    bail!("launch-ready claim has incomplete active attempt history");
                }
            }
            tx.commit()?;
        }
        let batch_outcomes = submit_batch(&self.args.scq, claims)?;
        let outcomes = claims
            .iter()
            .cloned()
            .zip(batch_outcomes)
            .map(|(row, (accepted, transient, detail))| (row, accepted, transient, detail))
            .collect::<Vec<_>>();
        let mut submitted = 0;
        let mut failed = 0;
        let mut submitted_candidate_evaluations = 0_u64;
        let accepted_rows = outcomes
            .iter()
            .filter(|(_, accepted, _, _)| *accepted)
            .map(|(row, _, _, _)| (&row.manifest, row.owner.as_str(), row.claim.job_id.as_str()))
            .collect::<Vec<_>>();
        self.record_submissions(&accepted_rows)?;
        for (row, accepted, transient, detail) in outcomes {
            if accepted {
                submitted += 1;
                submitted_candidate_evaluations =
                    submitted_candidate_evaluations.saturating_add(row.manifest.pairs.len() as u64);
            } else {
                failed += 1;
                self.backlog.execute(
                    "UPDATE claims SET state=?,updated_at_ms=? WHERE manifest_sha256=?",
                    params![
                        if transient {
                            "checked".to_owned()
                        } else {
                            format!(
                                "submit_failed:{}",
                                detail.chars().take(500).collect::<String>()
                            )
                        },
                        now_ms() as i64,
                        row.manifest.sha256
                    ],
                )?;
                self.backlog.execute(
                    "UPDATE keeper_attempts SET state=? WHERE manifest_sha256=? \
                     AND state='submitting'",
                    params![
                        if transient {
                            "checked"
                        } else {
                            "submit_failed"
                        },
                        row.manifest.sha256
                    ],
                )?;
            }
        }
        Ok((submitted, failed, submitted_candidate_evaluations))
    }

    pub(super) fn prepare_infrastructure_retries(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<usize> {
        let mut recovered = 0;
        for claim in claims.values() {
            if claim.state != "submitted" || claim.attempts >= 5 {
                continue;
            }
            let Some(job) = scheduler.get(claim.job_id.as_str()) else {
                // Missing scheduler state is ambiguous. Never infer an
                // infrastructure failure and never retry it.
                continue;
            };
            if job.state != "failed"
                || !job
                    .failure_class
                    .as_deref()
                    .is_some_and(is_infrastructure_failure_class)
            {
                continue;
            }
            if authoritative_receipt_exists(&self.args.root, &claim.job_id) {
                continue;
            }
            let Ok(manifest) = self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves)
            else {
                continue;
            };
            let owner = self.validate_pair_reservations(
                &manifest,
                claim.owner.as_deref(),
                Some(claim.job_id.as_str()),
            )?;
            let retry_number = claim.attempts;
            let new_job_id = format!("{}-r{retry_number}", claim.job_id);
            let new_spec = claim.spec.with_file_name(format!("{new_job_id}.toml"));
            rewrite_spec_job_id(&claim.spec, &new_spec, &claim.job_id, &new_job_id)?;
            scq_check(&self.args.scq, &new_spec)?;
            let failure = job.failure_class.as_deref().unwrap_or("infrastructure");
            self.mark_infrastructure_retry(&manifest, failure, &new_job_id, &owner)?;
            self.backlog.execute(
                "UPDATE claims SET job_id=?,spec=?,state='checked',updated_at_ms=? \
                 WHERE manifest_sha256=? AND state='submitted' AND job_id=?",
                params![
                    new_job_id,
                    new_spec.to_string_lossy(),
                    now_ms() as i64,
                    manifest.sha256,
                    claim.job_id
                ],
            )?;
            recovered += 1;
        }
        Ok(recovered)
    }

    pub(super) fn mark_infrastructure_retry(
        &mut self,
        manifest: &ValidManifest,
        failure: &str,
        new_job_id: &str,
        owner: &str,
    ) -> Result<()> {
        let tx = self
            .pairs
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let retry_increment = if self.pair_schema.has_infrastructure_retries {
            ",infrastructure_retries=infrastructure_retries+1"
        } else {
            ""
        };
        let changed = tx.execute(
            &format!(
                "UPDATE pairs SET state='infrastructure_retry',failure_class=?{retry_increment},\
                 job_id=NULL,updated_at_ms=? WHERE manifest_sha256=? AND state='submitted'"
            ),
            params![failure, now_ms() as i64, manifest.sha256],
        )?;
        if changed != manifest.pairs.len() {
            let prior: i64 = tx.query_row(
                "SELECT COUNT(*) FROM pairs WHERE manifest_sha256=? \
                 AND state='infrastructure_retry' AND job_id IS NULL",
                [manifest.sha256.as_str()],
                |row| row.get(0),
            )?;
            if changed != 0 || prior != manifest.pairs.len() as i64 {
                bail!("infrastructure retry pair transition count mismatch");
            }
        }
        finalize_native_attempts_for_retry(&tx, &self.pair_schema, manifest, failure)?;
        append_native_retry_attempts(&tx, &self.pair_schema, manifest, owner, new_job_id)?;
        tx.commit()?;
        let tx = self
            .backlog
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "UPDATE keeper_attempts SET state='infrastructure_retry',failure_class=?,\
             finalized_at_ms=? WHERE manifest_sha256=? AND state='submitted'",
            params![failure, now_ms() as i64, manifest.sha256],
        )?;
        append_keeper_attempts(&tx, manifest, owner, new_job_id, "checked", 1)?;
        tx.commit()?;
        Ok(())
    }
}
