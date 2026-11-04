use super::*;

impl RunService {
    pub(crate) fn start(
        self: &Arc<Self>,
        spec: RunSpec,
        identity: RequestIdentity,
        _pub06_values: Pub06Injection,
    ) -> Response {
        let _guard = match self.admission.lock() {
            Ok(guard) => guard,
            Err(_) => return response(&io("Run admission lock is poisoned.")),
        };
        match self.store.idempotent_receipt(&identity) {
            Ok(Some(receipt)) => return receipt.into_response(),
            Ok(None) => {}
            Err(error) => return error.into_response(),
        }
        if !spec.dry_run && self.activity.active_runs() != 0 {
            return response(&issue(
                ErrorCode::BackendUnavailable,
                "The bounded CPU run backend is busy; retry this key after the active run drains.",
            ));
        }
        let mut prepared = match self.prepare(&spec) {
            Ok(value) => value,
            Err(error) => return response(&error),
        };
        if spec.dry_run {
            return match self.store.execute_idempotent(&identity, |_| {
                CommandOutcome::json(200, serde_json::to_value(&prepared.public)?)
            }) {
                Ok(receipt) => receipt.into_response(),
                Err(error) => error.into_response(),
            };
        }
        let fit = match self.store.publish_blob(prepared.fit.as_slice()) {
            Ok(value) => value,
            Err(error) => return response(&io(error)),
        };
        let holdout = match self.store.publish_blob(prepared.holdout.as_slice()) {
            Ok(value) => value,
            Err(error) => return response(&io(error)),
        };
        prepared.search.matrix_path = match self.store.verified_blob_path(&fit) {
            Ok(value) => value,
            Err(error) => return response(&io(error)),
        };
        let holdout_path = match self.store.verified_blob_path(&holdout) {
            Ok(value) => value,
            Err(error) => return response(&io(error)),
        };
        let request = FinalizationRequest {
            run_id: prepared.search.run_id.clone(),
            fit: Partition {
                path: prepared.search.matrix_path.clone(),
                sha256: prepared.search.matrix_sha256.clone(),
                rows: prepared.fit_rows,
            },
            holdout: Partition {
                path: holdout_path,
                sha256: hash(&prepared.holdout),
                rows: prepared.holdout_rows,
            },
            test: None,
            // Publish both genuine models once; exports select their recorded fit mode.
            refit: Refit::FitHoldout,
            model_root: self.home.join("models"),
            scratch_root: self.home.join("run-scratch"),
            max_input_bytes: self.config.limits.max_dataset_bytes.value,
            max_model_bytes: MODEL_LIMIT,
        };
        // This context precedes the first run commit, so every visible run can
        // resume from its original immutable plan and partition provenance.
        let context = Continuation {
            search: prepared.search.clone(),
            finalization: request.clone(),
            backend: ContinuationBackend {
                spec,
                placement: prepared.backend.placement.clone(),
                landmarks: prepared.backend.full_fit_landmarks.clone(),
            },
        };
        let persisted = self.store.transaction(|tx| {
            tx.put(
                &continuation_key(&prepared.search.run_id),
                &Record::metadata(
                    serde_json::to_value(&context)?,
                    vec![fit.clone(), holdout.clone()],
                ),
            )
        });
        if let Err(error) = persisted {
            return response(&io(error));
        }
        let sender = match self.worker_sender(
            _pub06_values,
            false,
            prepared.backend,
            context.backend.spec.languages.clone(),
        ) {
            Ok(sender) => sender,
            Err(error) => return response(&error),
        };
        let discovery = match Discovery::start_command(
            Arc::clone(&self.store),
            prepared.search,
            Arc::clone(&self.journal),
            identity.clone(),
            vec![fit, holdout],
            prepared.public.warnings,
        ) {
            Ok(value) => value,
            Err(error) => return response(&io(error)),
        };
        if sender.send((discovery, request)).is_err() {
            return response(&io("Run worker stopped before dispatch."));
        }
        match self.store.idempotent_receipt(&identity) {
            Ok(Some(receipt)) => receipt.into_response(),
            Ok(None) => response(&io("Committed run receipt missing.")),
            Err(error) => error.into_response(),
        }
    }

    pub(crate) fn worker_sender(
        self: &Arc<Self>,
        _pub06_values: Pub06Injection,
        explicit_continue: bool,
        backend: BackendPlan,
        languages: Vec<Language>,
    ) -> Result<mpsc::SyncSender<(Discovery, FinalizationRequest)>> {
        let (sender, receiver) = mpsc::sync_channel::<(Discovery, FinalizationRequest)>(1);
        let owner = Arc::clone(self);
        let allowed = run_language_filter(&languages)?;
        #[cfg(all(feature = "failpoints", debug_assertions))]
        let run_space = CpuRunSpace {
            allowed,
            pub06_values: _pub06_values,
        };
        #[cfg(not(all(feature = "failpoints", debug_assertions)))]
        let run_space = CpuRunSpace { allowed };
        let learned = backend.learned();
        let admission = backend.admission;
        let run_space = BackendSpace {
            native: run_space,
            learned,
        };
        let activity = self.activity.begin_run();
        if let Err(error) = std::thread::Builder::new()
            .name("jope-run".into())
            .spawn(move || {
                let _activity = activity;
                if let Ok((mut discovery, request)) = receiver.recv() {
                    let result = (|| -> anyhow::Result<()> {
                        let process = ProcessExecutor::new(
                            &owner.executable,
                            1,
                            &owner.home.join("worker-scratch"),
                        )?;
                        let executor = BackendExecutor {
                            inner: &process,
                            learned,
                        };
                        while discovery.completed().is_none() {
                            discovery.advance(&run_space, &admission, &executor, 1, usize::MAX)?;
                        }
                        let finalizer = Finalizer::with_journal(
                            Arc::clone(&owner.store),
                            Arc::clone(&owner.journal),
                        );
                        if explicit_continue {
                            finalizer.finalize_after_explicit_continue(&request, &executor)?;
                        } else {
                            finalizer.finalize(&request, &executor)?;
                        }
                        Ok(())
                    })();
                    if let Err(error) = result {
                        owner.fail(&request.run_id, &error.to_string());
                    }
                }
            })
        {
            return Err(io(error));
        }
        Ok(sender)
    }

    pub(crate) fn continue_run(
        self: &Arc<Self>,
        run: jope_api::ids::RunId,
        expected_revision: Option<u64>,
        identity: RequestIdentity,
    ) -> Response {
        let _guard = match self.admission.lock() {
            Ok(guard) => guard,
            Err(_) => return response(&io("Run admission lock is poisoned.")),
        };
        match self.store.idempotent_receipt(&identity) {
            Ok(Some(receipt)) => return receipt.into_response(),
            Ok(None) => {}
            Err(error) => return error.into_response(),
        }
        let attempt = (|| -> Result<()> {
            let snapshot = self
                .journal
                .snapshot(&run)
                .map_err(io)?
                .ok_or_else(|| issue(ErrorCode::NotFound, "Run does not exist."))?;
            if expected_revision.is_some_and(|revision| revision != snapshot.revision.0) {
                return Err(issue(
                    ErrorCode::RevisionConflict,
                    "Re-read the snapshot and retry.",
                ));
            }
            if snapshot.state != RunState::Paused || self.activity.active_runs() != 0 {
                return Err(issue(
                    ErrorCode::RevisionConflict,
                    "Continue requires an idle backend and a paused run.",
                ));
            }
            let stored_context = self
                .store
                .get(&continuation_key(&run))
                .map_err(io)?
                .ok_or_else(|| {
                    unavailable("Original continuation context is unavailable for this run.")
                })?;
            let context: Continuation = serde_json::from_value(stored_context.data).map_err(io)?;
            if context.search.run_id != run
                || context.finalization.run_id != run
                || context.search.binary_version != self.binary
            {
                return Err(unavailable(
                    "Continuation requires the original binary and run identity.",
                ));
            }
            // Rebuild only the original backend, retaining the checkpoint plan.
            let backend = context.backend.restore(
                &self.store,
                &context.search,
                &stored_context.blobs,
                self.activity.admitted_ranker(),
            )?;
            #[cfg(all(feature = "failpoints", debug_assertions))]
            let injection = None;
            #[cfg(not(all(feature = "failpoints", debug_assertions)))]
            let injection = ();
            let sender = self.worker_sender(
                injection,
                true,
                backend,
                context.backend.spec.languages.clone(),
            )?;
            let discovery = Discovery::continue_command(
                Arc::clone(&self.store),
                context.search,
                Arc::clone(&self.journal),
                identity.clone(),
            )
            .map_err(io)?;
            sender.send((discovery, context.finalization)).map_err(io)?;
            Ok(())
        })();
        if let Err(error) = attempt {
            return response(&error);
        }
        match self.store.idempotent_receipt(&identity) {
            Ok(Some(receipt)) => receipt.into_response(),
            Ok(None) => response(&io("Committed continue receipt missing.")),
            Err(error) => error.into_response(),
        }
    }

    pub(crate) fn fail(&self, run: &jope_api::ids::RunId, detail: &str) {
        let result = (|| -> anyhow::Result<()> {
            let record = self
                .store
                .get(&RecordKey::Run(run.clone()))?
                .ok_or_else(|| anyhow::anyhow!("Run missing"))?;
            let mut snapshot = self
                .journal
                .snapshot(run)?
                .ok_or_else(|| anyhow::anyhow!("Snapshot missing"))?;
            if snapshot.state != RunState::Running {
                return Ok(());
            }
            let expected = snapshot.revision.0;
            snapshot.revision.0 += 1;
            snapshot.state = RunState::Failed;
            self.journal.commit(
                RunUpdate {
                    expected_revision: Some(expected),
                    snapshot: snapshot.clone(),
                    checkpoint: record.data["checkpoint"].clone(),
                    blobs: record.blobs,
                },
                vec![EventBody::RunState(RunStateBody {
                    state: RunState::Failed,
                    phase: snapshot.phase,
                    pending: None,
                    reason: Some("The bounded CPU run failed; inspect local diagnostics.".into()),
                })],
                |_| Ok(()),
            )?;
            Ok(())
        })();
        let diagnostic = format!("run {run} failed: {detail}; failure journal: {result:?}");
        #[cfg(all(feature = "failpoints", debug_assertions))]
        crate::pub06_test::record_log(diagnostic.clone());
        eprintln!("{diagnostic}");
    }
}
