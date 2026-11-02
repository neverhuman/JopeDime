use super::*;

impl Keeper {
    pub(super) fn validate_claim_manifest(
        &mut self,
        claim: &Claim,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<ValidManifest> {
        let wave = manifest_wave(&claim.manifest)?;
        if !waves.contains(&wave) {
            bail!("claim manifest wave is not current-checkpoint immutable supply");
        }
        let cache_key = (claim.manifest.clone(), claim.manifest_sha256.clone());
        if let Some(manifest) = self.manifest_cache.get(&cache_key) {
            return Ok(manifest.clone());
        }
        let manifest = validate_manifest(&claim.manifest, checkpoint_sha256, epoch, &wave)?;
        if manifest.sha256 != claim.manifest_sha256 {
            bail!("claim manifest hash drift");
        }
        self.manifest_cache.insert(cache_key, manifest.clone());
        Ok(manifest)
    }

    pub(super) fn ready_claims(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<Vec<ReadyClaim>> {
        let mut ready = Vec::new();
        for claim in claims.values() {
            if !matches!(claim.state.as_str(), "checked" | "submitting")
                || scheduler.contains_key(claim.job_id.as_str())
            {
                continue;
            }
            let manifest =
                match self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves) {
                    Ok(value) => value,
                    Err(_) => continue,
                };
            let owner = self.validate_pair_reservations(
                &manifest,
                claim.owner.as_deref(),
                (claim.state == "submitting").then_some(claim.job_id.as_str()),
            )?;
            ready.push(ReadyClaim {
                claim: claim.clone(),
                manifest,
                owner,
            });
        }
        Ok(ready)
    }

    pub(super) fn rebalance_launch_ready(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
        limit: usize,
    ) -> Result<()> {
        let mut eligible = Vec::new();
        for claim in claims.values() {
            if !matches!(
                claim.state.as_str(),
                "checked" | "submitting" | "deferred_horizon"
            ) || scheduler.contains_key(claim.job_id.as_str())
            {
                continue;
            }
            let Ok(manifest) = self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves)
            else {
                continue;
            };
            self.validate_pair_reservations(
                &manifest,
                claim.owner.as_deref(),
                (claim.state == "submitting").then_some(claim.job_id.as_str()),
            )?;
            eligible.push((claim, manifest.generated_at_ms));
        }
        eligible.sort_by(|(left, left_time), (right, right_time)| {
            // A crash-window `submitting` claim keeps priority so its
            // deterministic SCQ identity is reconciled before fresh work.
            let left_submit = left.state == "submitting";
            let right_submit = right.state == "submitting";
            right_submit
                .cmp(&left_submit)
                .then_with(|| left_time.cmp(right_time))
                .then_with(|| left.job_id.cmp(&right.job_id))
        });
        let tx = self
            .backlog
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        for (index, (claim, _)) in eligible.into_iter().enumerate() {
            let state = if index < limit {
                if claim.state == "submitting" {
                    "submitting"
                } else {
                    "checked"
                }
            } else {
                "deferred_horizon"
            };
            tx.execute(
                "UPDATE claims SET state=?,updated_at_ms=? WHERE manifest_sha256=?",
                params![state, now_ms() as i64, claim.manifest_sha256],
            )?;
            tx.execute(
                "UPDATE keeper_attempts SET state=? WHERE manifest_sha256=? \
                 AND state IN ('checked','submitting','deferred_horizon')",
                params![state, claim.manifest_sha256],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub(super) fn count_deferred_current(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<usize> {
        let mut count = 0;
        for claim in claims.values() {
            if claim.state != "deferred_horizon" || scheduler.contains_key(claim.job_id.as_str()) {
                continue;
            }
            let Ok(manifest) = self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves)
            else {
                continue;
            };
            self.validate_pair_reservations(&manifest, claim.owner.as_deref(), None)?;
            count += 1;
        }
        Ok(count)
    }

    pub(super) fn validate_pair_reservations(
        &mut self,
        manifest: &ValidManifest,
        claimed_owner: Option<&str>,
        allow_submitted_job: Option<&str>,
    ) -> Result<String> {
        let cache_key = (
            manifest.sha256.clone(),
            claimed_owner.map(str::to_owned),
            allow_submitted_job.map(str::to_owned),
        );
        if let Some(owner) = self.reservation_cache.get(&cache_key) {
            return Ok(owner.clone());
        }
        let owner_col = self.pair_schema.owner_column;
        let query = format!(
            "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,state,{owner_col},\
             manifest_path,manifest_sha256,job_id FROM pairs WHERE manifest_sha256=?"
        );
        let expected_pairs = manifest
            .pairs
            .iter()
            .map(|pair| {
                (
                    pair.task.clone(),
                    pair.dataset_hash.clone(),
                    pair.semantic_sha256.clone(),
                    pair.protocol_sha256.clone(),
                )
            })
            .collect::<BTreeSet<_>>();
        let (owner, observed_pairs) = {
            let mut statement = self.pairs.prepare_cached(&query)?;
            let rows = statement.query_map([manifest.sha256.as_str()], |row| {
                Ok((
                    (
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ),
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, Option<String>>(7)?,
                    row.get::<_, Option<String>>(8)?,
                ))
            })?;
            let mut owner = None;
            let mut observed_pairs = BTreeSet::new();
            for row in rows {
                let (pair, state, pair_owner, path, digest, submitted_job) = row?;
                observed_pairs.insert(pair);
                let launch_ready = matches!(state.as_str(), "reserved" | "infrastructure_retry");
                let crash_adoptable = state == "submitted"
                    && allow_submitted_job.is_some()
                    && submitted_job.as_deref() == allow_submitted_job;
                if !launch_ready && !crash_adoptable {
                    bail!("manifest protocol pair is not launch-ready");
                }
                if path.as_deref() != Some(manifest.path.to_string_lossy().as_ref())
                    || digest.as_deref() != Some(manifest.sha256.as_str())
                {
                    bail!("pair/manifest binding drift");
                }
                if let Some(expected) = claimed_owner {
                    if pair_owner != expected {
                        bail!("claim/pair owner drift");
                    }
                }
                if owner
                    .replace(pair_owner.clone())
                    .is_some_and(|prior| prior != pair_owner)
                {
                    bail!("manifest spans reservation owners");
                }
            }
            (owner, observed_pairs)
        };
        if observed_pairs != expected_pairs {
            bail!("manifest reserved protocol-pair set mismatch");
        }
        let owner = owner.context("manifest has no reserved protocol pairs")?;
        self.reservation_cache.insert(cache_key, owner.clone());
        Ok(owner)
    }
}
