use super::*;

impl Loop {
    pub(super) async fn result(
        &self,
        graph: &Graph,
        client: &Client,
        r: &Request,
    ) -> Result<Option<(Artifact, VerifiedResult)>> {
        if !self.read_only {
            if let Some(binding) = self.binding(r)? {
                // Unadmitted/waiting-authority bindings cannot execute. Admission
                // occurs only in offer after the independent exact task grant.
                if !matches!(
                    r.role,
                    NativeLoopRole::Direct | NativeLoopRole::Continuation
                ) || crate::eval_registry::native_scq::policy::InstalledPolicy::load(
                    &self.config.fit_policy,
                )?
                .task_policy(&binding.admission.manifest.sha256)?
                .is_some()
                {
                    graph.reconcile_eligible(client, &binding.admission).await?;
                }
            }
        }
        let key = identity(&r.run, r.generation, &r.id)?;
        let path = self
            .graph_state
            .join("tasks")
            .join(key)
            .join("verified.json");
        if !path.exists() {
            return Ok(None);
        };
        let binding = self
            .binding(r)?
            .context("original loop binding is required before consuming graph result")?;
        let a = artifact(&self.root, &path)?;
        let result = graph.verified(client, &a).await?;
        verify_result_binding(r, &binding.admission, &result)?;
        ensure!(
            result.run == r.run && result.generation == r.generation && result.task_id == r.id,
            "loop result identity changed"
        );
        Ok(Some((a, result)))
    }
    pub(super) async fn failed(
        &self,
        graph: &Graph,
        client: &Client,
        request: &Request,
    ) -> Result<Option<Artifact>> {
        let Some(binding) = self.binding(request)? else {
            return Ok(None);
        };
        graph.failed(client, &binding.admission).await
    }
    pub(super) async fn offer(
        &self,
        graph: &Graph,
        client: &Client,
        r: &Request,
        progress: &mut Progress,
    ) -> Result<()> {
        if self.binding(r)?.is_none() {
            self.materialize_request(graph, client, r).await?;
        }
        let Some(binding) = self.binding(r)? else {
            progress.waiting_materialization += 1;
            progress
                .idle_reasons
                .insert("awaiting_hash_bound_task_materialization".into());
            return Ok(());
        };
        if let Some(failure) = self.failed(graph, client, r).await? {
            progress.idle_reasons.insert(format!(
                "terminal_native_failure_requires_explicit_recovery:{}",
                r.id
            ));
            frozen(
                &self.root,
                &self
                    .state
                    .join("refusals")
                    .join(format!("{}.json", identity(&r.run, r.generation, &r.id)?)),
                &json!({"schema":"jope.native_task_refusal.v1","request":binding.request,"failure":failure,"numerical_completion_claimed":false}),
            )?;
            return Ok(());
        }
        if matches!(
            r.role,
            NativeLoopRole::Direct | NativeLoopRole::Continuation
        ) {
            let key = identity(&r.run, r.generation, &r.id)?;
            let admission = frozen(
                &self.root,
                &self
                    .state
                    .join("authority-requests")
                    .join(format!("{key}.admission.json")),
                &binding.admission,
            )?;
            frozen(
                &self.root,
                &self
                    .state
                    .join("authority-requests")
                    .join(format!("{key}.json")),
                &json!({"schema":"jope.native_grant_request_outbox.v1","run":r.run,"generation":r.generation,"role":r.role,
                    "frozen_config":artifact(&self.root,&self.state.join("config.json"))?,"request":binding.request,
                    "admission":admission,"task_manifest":binding.admission.manifest,"authorization_issued":false}),
            )?;
            let policy = crate::eval_registry::native_scq::policy::InstalledPolicy::load(
                &self.config.fit_policy,
            )?;
            if policy
                .task_policy(&binding.admission.manifest.sha256)?
                .is_none()
            {
                progress.waiting_authority += 1;
                progress
                    .idle_reasons
                    .insert("awaiting_independent_exact_task_authorization".into());
                return Ok(());
            }
        }
        graph.admit(client, &binding.admission).await?;
        graph.reconcile_eligible(client, &binding.admission).await?;
        progress.offered += 1;
        Ok(())
    }
    pub(super) fn freeze_genome(
        &self,
        generation: u64,
        candidate: &Artifact,
        direct: Option<Artifact>,
        predecessors: BTreeMap<String, Artifact>,
        vocabulary: Artifact,
    ) -> Result<Request> {
        let c: Candidate = native_genome::load(&self.root, candidate)?;
        let stage = if direct.is_some() {
            Stage::Continuation
        } else {
            Stage::Direct
        };
        let id = format!(
            "g{generation}-{}-{}",
            if stage == Stage::Direct {
                "direct"
            } else {
                "continuation"
            },
            candidate.sha256
        );
        let base = self.state.join("numerical").join(&id);
        let output = base.join("output").strip_prefix(&self.root)?.to_owned();
        let task = GenomeTask {
            schema: "jope.native_genome_task.v1".into(),
            stage,
            candidate: candidate.clone(),
            output,
            vocabulary,
            admission_policy: self.config.label_admission_policy.clone(),
            qualification: self.config.qualification,
            preparation_overlay: self.config.preparation_overlay.clone(),
            evaluations: if stage == Stage::Direct {
                1
            } else {
                self.config.continuation_evaluations
            },
            seconds: if stage == Stage::Direct {
                self.config.direct_seconds
            } else {
                self.config.continuation_seconds
            },
            rng_seed: generation_seed(
                &self.config.run,
                generation,
                &format!(
                    "{}:{}:{}",
                    c.context.prepared_sha256, c.context.cut, c.matched_slot
                ),
            )?,
            direct_parent: direct,
        };
        let reference = frozen(&self.root, &base.join("genome-task.json"), &task)?;
        let (request_ref, r) = self.request(
            generation,
            id.clone(),
            if stage == Stage::Direct {
                NativeLoopRole::Direct
            } else {
                NativeLoopRole::Continuation
            },
            json!({"genome_task":reference}),
            predecessors,
        )?;
        let mut dependencies = native_genome::required_inputs(&self.root, &reference)?;
        dependencies.insert(request_ref.path.clone(), request_ref.sha256.clone());
        let mut closure = Vec::new();
        for (path, sha256) in &dependencies {
            serde_json::to_writer(
                &mut closure,
                &Artifact {
                    path: path.clone(),
                    sha256: sha256.clone(),
                },
            )?;
            closure.push(b'\n');
        }
        let closure_ref = immutable(&self.root, &base.join("closure.jsonl"), &closure)?;
        let input = InputManifest {
            schema: "jope.smartcluster_input.v1".into(),
            root: self.root.clone(),
            operation: if stage == Stage::Direct {
                Operation::DirectEvaluation {
                    manifest: reference.path,
                    sha256: reference.sha256,
                }
            } else {
                Operation::SeededContinuation {
                    manifest: reference.path,
                    sha256: reference.sha256,
                }
            },
            input_artifacts: BTreeMap::new(),
            closure: Some(ClosureManifest {
                path: closure_ref.path.clone(),
                sha256: closure_ref.sha256.clone(),
            }),
            output_receipt: base
                .join("result.json")
                .strip_prefix(&self.root)?
                .to_owned(),
        };
        let input_ref = frozen(&self.root, &base.join("task.input.json"), &input)?;
        let manifest = TaskManifest {
            schema: "jope.smartcluster_task.v1".into(),
            run: self.config.run.clone(),
            generation,
            task_id: id.clone(),
            kind: if stage == Stage::Direct {
                TaskKind::GpSearch
            } else {
                TaskKind::Continuation
            },
            input_manifest_sha256: input_ref.sha256,
            dependencies: BTreeMap::from([(
                closure_ref.path.to_string_lossy().into_owned(),
                closure_ref.sha256,
            )]),
            parent_checkpoint_sha256: Some(c.parent_checkpoint.sha256.clone()),
            corpus_sha256: Some(
                native_genome::load::<Value>(&self.root, &c.parent_checkpoint)?["corpus_sha256"]
                    .as_str()
                    .context("candidate parent corpus identity absent")?
                    .to_owned(),
            ),
            idempotency_key: IdempotencyKey::new(format!("{}/{id}", self.config.run))?,
            deadline: self.config.task_deadline,
            retry_allowance: 1,
            resources: self.config.cpu_resources,
            expected_receipt_id: format!("{}/{id}/complete", self.config.run),
            checkpoint_allowance_ms: None,
        };
        let manifest_ref = frozen(&self.root, &base.join("task.json"), &manifest)?;
        self.bind(&Binding {
            schema: "jope.native_loop_binding.v1".into(),
            request: request_ref,
            admission: Admission {
                schema: "jope.native_graph_admission.v1".into(),
                manifest: manifest_ref,
                predecessors: r.predecessors.clone(),
                delivery: Delivery::NativeJob,
            },
        })?;
        Ok(r)
    }
}
