use super::*;

impl Finalizer {
    pub(crate) fn with_journal(store: Arc<Store>, journal: Arc<EventLog>) -> Self {
        Self { store, journal }
    }
    pub fn new(store: Arc<Store>) -> Self {
        Self {
            journal: Arc::new(EventLog::new(Arc::clone(&store))),
            store,
        }
    }
    pub fn journal(&self) -> &Arc<EventLog> {
        &self.journal
    }

    /// A failed/interrupted claim is never automatically retried: its protected
    /// read outcome may be unknown. A completed replay verifies immutable output.
    pub fn finalize(
        &self,
        request: &FinalizationRequest,
        executor: &impl Executor,
    ) -> Result<FinalizationReceipt> {
        self.finalize_inner(request, executor, false)
    }

    /// Only an explicit public Continue dispatch may retry an interrupted claim.
    /// Preserve its unknown prior outcome; automatic replay still refuses it.
    pub(crate) fn finalize_after_explicit_continue(
        &self,
        request: &FinalizationRequest,
        executor: &impl Executor,
    ) -> Result<FinalizationReceipt> {
        self.finalize_inner(request, executor, true)
    }

    fn finalize_inner(
        &self,
        request: &FinalizationRequest,
        executor: &impl Executor,
        explicit_continue: bool,
    ) -> Result<FinalizationReceipt> {
        let record = self
            .store
            .get(&RecordKey::Run(request.run_id.clone()))?
            .context("run missing")?;
        let identity = digest(&serde_json::to_vec(&(
            request,
            &record.data["checkpoint"]["plan"],
            &record.data["checkpoint"]["champion"],
        ))?);
        let key = RecordKey::Command {
            principal: "discover.finalization.v1".into(),
            idempotency_key: request.run_id.to_string(),
        };
        let prior = self.store.get(&key)?;
        if let Some(record) = &prior {
            ensure!(
                record.data["identity"] == identity,
                "finalization request changed"
            );
            if record.data["state"] == "completed" {
                let receipt: FinalizationReceipt =
                    serde_json::from_value(record.data["receipt"].clone())?;
                let object = ModelObject::read(&receipt.model_path, request.max_model_bytes)?;
                ensure!(
                    object.id() == receipt.model_digest,
                    "finalized object identity changed"
                );
                object.verify_executable()?;
                return Ok(receipt);
            }
            ensure!(
                explicit_continue && record.data["state"] == "started",
                "finalization outcome failed or unknown; automatic protected-row reread refused"
            );
        }
        let snapshot = self
            .journal
            .snapshot(&request.run_id)?
            .context("run missing")?;
        ensure!(
            snapshot.state == RunState::Running
                || (snapshot.state == RunState::Paused
                    && record.run.as_ref().is_some_and(|r| r.interrupted)),
            "run cannot finalize"
        );
        let checkpoint = record.data["checkpoint"].clone();
        let plan: Plan = serde_json::from_value(checkpoint["plan"].clone())?;
        let champion: Champion = serde_json::from_value(checkpoint["champion"].clone())
            .context("finite champion missing")?;
        let reason: CompletionReason = serde_json::from_value(checkpoint["completed"].clone())
            .context("search is not complete")?;
        ensure!(
            plan.run_id == request.run_id
                && champion.loss.is_finite()
                && champion.value.is_finite(),
            "invalid completed champion"
        );
        self.store.verify_blob(&champion.model)?;
        ensure!(
            snapshot
                .best
                .as_ref()
                .is_some_and(|best| best.algorithm_id.to_string()
                    == champion.candidate.algorithm_id
                    && best.cv_value.to_bits() == champion.value.to_bits()),
            "snapshot and selected champion differ"
        );
        ensure!(
            request.fit.path == plan.matrix_path && request.fit.sha256 == plan.matrix_sha256,
            "FIT differs from the search plan"
        );
        ensure!(
            request
                .fit
                .rows
                .iter()
                .chain(&request.holdout.rows)
                .all(|r| r.dataset == plan.dataset_id),
            "training roster dataset mismatch"
        );
        ensure!(
            request.fit.rows.len() >= 2
                && !request.holdout.rows.is_empty()
                && request.test.as_ref().is_none_or(|p| !p.rows.is_empty()),
            "empty finalization partition"
        );
        ensure!(
            request.max_input_bytes > 0
                && request.max_model_bytes > 0
                && request.model_root.is_absolute()
                && request.scratch_root.is_absolute(),
            "invalid finalization limits or roots"
        );
        for part in [&request.fit, &request.holdout]
            .into_iter()
            .chain(request.test.as_ref())
        {
            ensure!(
                part.path.is_absolute() && is_digest(&part.sha256),
                "invalid prepared partition"
            );
        }
        let fitness = jope_runtime::fitness::Fitness::parse(&plan.objective)?;
        ensure!(
            fitness.columns().is_empty() && !fitness.requires_order(),
            "M1 finalization requires a column-free, order-independent regression objective"
        );
        let mut audit = RowAudit::new(
            ids(&request.fit),
            ids(&request.holdout),
            request.test.as_ref().map(ids).unwrap_or_default(),
        )?;
        self.store.transaction(|tx| {
            if tx.get(&key)? != prior {
                return Err(StoreError::Invalid("finalization claim changed"));
            }
            let current = tx
                .get(&RecordKey::Run(request.run_id.clone()))?
                .ok_or(StoreError::Invalid("run disappeared"))?;
            if current != record
                || current
                    .run
                    .as_ref()
                    .is_none_or(|r| r.revision != snapshot.revision.0 || r.state != snapshot.state)
            {
                return Err(StoreError::Invalid("run changed before finalization"));
            }
            if let Some(prior) = &prior {
                let history = RecordKey::Command {
                    principal: "discover.finalization.interrupted.v1".into(),
                    idempotency_key: format!("{}:{}", request.run_id, snapshot.revision.0),
                };
                if tx.get(&history)?.is_some() {
                    return Err(StoreError::Integrity("interrupted finalization history already exists"));
                }
                tx.put(&history, &Record::metadata(json!({
                    "outcome":"unknown", "explicit_continue":true, "prior":prior
                }), prior.blobs.clone()))?;
            }
            tx.put(
                &key,
                &Record::metadata(json!({"identity":identity,"state":"started", "attempt":prior.as_ref().and_then(|r|r.data["attempt"].as_u64()).unwrap_or(0).checked_add(1).ok_or(StoreError::Invalid("finalization attempts exhausted"))?}), vec![]),
            )
        })?;
        audit.begin_finalization();
        let result = self.execute(
            request,
            &plan,
            &champion,
            &snapshot,
            checkpoint.clone(),
            reason,
            executor,
            &mut audit,
            &key,
            &identity,
        );
        if let Err(error) = &result {
            let failure = json!({"identity":identity,"state":"failed_or_unknown","error":error.to_string(),"audit":audit});
            let mut failed = snapshot.clone();
            if let Some(revision) = failed.revision.0.checked_add(1) {
                failed.revision.0 = revision;
                failed.state = RunState::Failed;
                failed.phase = RunPhase::Finalizing;
                let _ = self.journal.commit(
                    RunUpdate {
                        expected_revision: Some(snapshot.revision.0),
                        snapshot: failed,
                        checkpoint,
                        blobs: record.blobs,
                    },
                    vec![],
                    |tx| {
                        let old = tx
                            .get(&key)?
                            .ok_or(StoreError::Invalid("finalization claim missing"))?;
                        if old.data["state"] != "started" {
                            return Err(StoreError::Invalid("finalization already resolved"));
                        }
                        tx.put(&key, &Record::metadata(failure.clone(), old.blobs))
                    },
                );
            }
            // Stale/cancelled or unknown commit: preserve a completed receipt and
            // never overwrite a newer controller state, but fence further reads.
            let _ = self.store.transaction(|tx| {
                if let Some(old) = tx.get(&key)? {
                    if old.data["state"] == "started" {
                        tx.put(&key, &Record::metadata(failure, old.blobs))?;
                    }
                }
                Ok(())
            });
        }
        result
    }

    fn ensure_current(&self, expected: &RunSnapshot) -> Result<()> {
        let current = self
            .journal
            .snapshot(&expected.run_id)?
            .context("run disappeared")?;
        ensure!(
            current.revision == expected.revision
                && current.state == expected.state
                && current.event_watermark == expected.event_watermark,
            "controller changed during finalization"
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        request: &FinalizationRequest,
        plan: &Plan,
        champion: &Champion,
        snapshot: &RunSnapshot,
        checkpoint: Value,
        reason: CompletionReason,
        executor: &impl Executor,
        audit: &mut RowAudit,
        key: &RecordKey,
        identity: &str,
    ) -> Result<FinalizationReceipt> {
        fs::create_dir_all(&request.scratch_root)?;
        ensure!(
            fs::symlink_metadata(&request.scratch_root)?.is_dir(),
            "scratch must be a real directory"
        );
        let scratch = tempfile::Builder::new()
            .prefix("finalize-")
            .tempdir_in(&request.scratch_root)?;
        let objective = RegressionObjective::parse(&plan.objective)?;
        let (fit_table, fit_object) = audit.with_rows(
            RowAuditPhase::FinalizationFit,
            &ids(&request.fit),
            |_| -> Result<_> {
                let table = load(&request.fit, plan, request.max_input_bytes)?;
                let object = refit(
                    executor,
                    plan,
                    champion,
                    &table,
                    &request.fit.path,
                    &request.fit.sha256,
                    Refit::Fit,
                    scratch.path(),
                    0,
                    request.max_model_bytes,
                )?;
                Ok((table, object))
            },
        )??;
        self.ensure_current(snapshot)?;
        let (holdout_table, holdout) = audit.with_rows(
            RowAuditPhase::HoldoutScore,
            &ids(&request.holdout),
            |_| -> Result<_> {
                let table = load(&request.holdout, plan, request.max_input_bytes)?;
                let score = score(&fit_object, &table, &plan.target, &objective)?;
                Ok((table, score))
            },
        )??;
        let mut union_ids = ids(&request.fit);
        union_ids.extend(ids(&request.holdout));
        self.ensure_current(snapshot)?;
        let object = match request.refit {
            Refit::Fit => None,
            Refit::FitHoldout => Some(audit.with_rows(
                RowAuditPhase::FinalizationRefit,
                &union_ids,
                |_| -> Result<_> {
                    let mut table = fit_table.clone();
                    table.rows.extend(holdout_table.rows.clone());
                    let bytes = crate::dataset::jmat::encode(&table)?;
                    ensure!(
                        bytes.len() as u64 <= request.max_input_bytes,
                        "refit matrix exceeds input bound"
                    );
                    let path = scratch.path().join("fit-holdout.jmat");
                    fs::write(&path, &bytes)?;
                    refit(
                        executor,
                        plan,
                        champion,
                        &table,
                        &path,
                        &digest(&bytes),
                        Refit::FitHoldout,
                        scratch.path(),
                        1,
                        request.max_model_bytes,
                    )
                },
            )??),
        };
        let exported = object.as_ref().unwrap_or(&fit_object);
        self.ensure_current(snapshot)?;
        let test = request
            .test
            .as_ref()
            .map(|part| {
                audit.with_rows(RowAuditPhase::TestScore, &ids(part), |_| -> Result<_> {
                    let table = load(part, plan, request.max_input_bytes)?;
                    score(exported, &table, &plan.target, &objective)
                })
            })
            .transpose()?
            .transpose()?;
        self.ensure_current(snapshot)?;
        let fit_path = fit_object.save(&request.model_root, request.max_model_bytes)?;
        let model_path = if object.is_some() {
            exported.save(&request.model_root, request.max_model_bytes)?
        } else {
            fit_path
        };
        let artifact: ArtifactId = format!("art_{}", &exported.id()[7..31]).parse()?;
        let nondeterminism: jope_api::Nondeterminism = serde_json::from_value(
            json!({"count":checkpoint["nondeterminism"].as_array().context("limit inventory missing")?.len().to_string(),"experiment_ids":checkpoint["nondeterminism"]}),
        )?;
        let receipt = FinalizationReceipt {
            run_id: request.run_id.clone(),
            algorithm_id: champion.candidate.algorithm_id.parse()?,
            model_artifact: artifact.clone(),
            model_digest: exported.id().into(),
            model_path,
            refit: request.refit,
            completion_reason: reason,
            budget_bound_by: match reason {
                CompletionReason::BudgetTime => Some("time".into()),
                CompletionReason::BudgetEvaluations => Some("evaluations".into()),
                _ => None,
            },
            budget_used: snapshot.budget_used.clone(),
            nondeterminism: nondeterminism.clone(),
            scores: FinalScores {
                cv: Some(champion.value),
                holdout: Some(holdout),
                test,
            },
            holdout_model_digest: fit_object.id().into(),
            holdout_scope: "holdout_scored_with_fit_only_model".into(),
            test_scope: request.test.as_ref().map(|_| {
                match request.refit {
                    Refit::Fit => "test_scored_with_fit_model",
                    Refit::FitHoldout => "test_scored_with_fit_holdout_model",
                }
                .into()
            }),
            fit_rows: request.fit.rows.len(),
            holdout_rows: request.holdout.rows.len(),
            test_rows: request.test.as_ref().map_or(0, |p| p.rows.len()),
            audit: serde_json::to_value(&audit)?,
        };
        let mut blobs = vec![champion.model.clone()];
        let fit_refs = publish(&self.store, &fit_object)?;
        blobs.extend(fit_refs.clone());
        let object_refs = publish(&self.store, exported)?;
        blobs.extend(object_refs.clone());
        let mut finished = snapshot.clone();
        finished.revision.0 = finished
            .revision
            .0
            .checked_add(1)
            .context("revision exhausted")?;
        finished.state = RunState::Completed;
        finished.phase = RunPhase::Finalizing;
        finished.completion_reason = Some(reason);
        finished
            .best
            .as_mut()
            .context("snapshot champion missing")?
            .model_artifact = Some(artifact.clone());
        let event = EventBody::RunFinished(RunFinishedBody {
            completion_reason: reason,
            champion: finished.best.clone(),
            scores: receipt.scores.clone(),
            budget_used: finished.budget_used.clone(),
            nondeterminism,
            evaluation_receipt: finished.evaluation_receipt.clone().map(Box::new),
        });
        self.journal.commit(RunUpdate { expected_revision: Some(snapshot.revision.0), snapshot: finished, checkpoint, blobs }, vec![event], |tx| {
            let old = tx.get(key)?.ok_or(StoreError::Invalid("finalization claim missing"))?;
            if old.data["state"] != "started" || old.data["identity"] != identity { return Err(StoreError::Invalid("finalization claim changed")); }
            // The assessment score belongs to this genuine FIT-only model. Publish
            // it atomically with the final champion so an explicit FIT export never
            // refits or labels FIT+HOLDOUT weights as FIT.
            let fit_artifact: ArtifactId = format!("art_{}", &fit_object.id()[7..31]).parse().map_err(|_| StoreError::Invalid("invalid FIT model identity"))?;
            let fit_record = Record::metadata(json!({"kind":"finalized_model","model_digest":fit_object.id(),"refit":Refit::Fit,"manifest":fit_object.manifest()}), fit_refs.clone());
            put_frozen_model(tx, RecordKey::Artifact(fit_artifact), fit_record)?;
            let record = Record::metadata(json!({"kind":"finalized_model","model_digest":exported.id(),"refit":request.refit,"manifest":exported.manifest()}), object_refs.clone());
            put_frozen_model(tx, RecordKey::Artifact(artifact), record)?;
            tx.put(key, &Record::metadata(json!({"identity":identity,"state":"completed","receipt":receipt}), object_refs))
        })?;
        Ok(receipt)
    }
}
