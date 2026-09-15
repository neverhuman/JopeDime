//! Materialization derives immutable scientific inputs; it never mints native fit grants.
use super::*;
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub resident_resources: ResourceRequest,
    pub packing_resources: ResourceRequest,
    pub checkpoint_allowance_ms: u64,
    pub checkpoint_steps: usize,
    pub dataset_target: Artifact,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetTarget {
    schema: String,
    store: PathBuf,
    mirror_root: PathBuf,
    namespace: String,
    publication: Value,
    receiver: Option<Value>,
    packer: Artifact,
}
impl Loop {
    pub(super) async fn materialize_request(
        &self,
        graph: &Graph,
        client: &Client,
        r: &Request,
    ) -> Result<bool> {
        let Some(settings) = &self.config.materialization else {
            return Ok(false);
        };
        if !matches!(
            r.role,
            Role::ProposalModel | Role::ProposalShuffled | Role::Train | Role::PackAndPublish
        ) {
            return Ok(false);
        }
        ensure!(
            settings.checkpoint_allowance_ms > 0 && settings.checkpoint_steps > 0,
            "unmeasured or invalid resident checkpoint allowance"
        );
        // Admission dependencies are independently proven before eligibility-derived inputs are frozen.
        for reference in r.predecessors.values() {
            graph.verified(client, reference).await?;
        }
        let request_ref = frozen(
            &self.root,
            &self
                .state
                .join("requests")
                .join(format!("{}.json", identity(&r.run, r.generation, &r.id)?)),
            r,
        )?;
        let base = self.state.join("materialized").join(&r.id);
        physical_directory(&self.root, &base)?;
        let output_receipt = base
            .join("result.json")
            .strip_prefix(&self.root)?
            .to_owned();
        let (operation, mut dependencies, kind, delivery, resources, parent, corpus) = match r.role
        {
            Role::ProposalModel | Role::ProposalShuffled => {
                let checkpoint: Artifact =
                    serde_json::from_value(r.parameters["checkpoint"].clone())?;
                let corpus: Artifact = serde_json::from_value(r.parameters["corpus"].clone())?;
                let corpus_dir = corpus
                    .path
                    .parent()
                    .context("proposal corpus parent")?
                    .to_owned();
                let checkpoint_dir = checkpoint
                    .path
                    .parent()
                    .context("proposal checkpoint parent")?
                    .to_owned();
                let generation_dir = self
                    .state
                    .join("proposals")
                    .join(format!("g{}", r.generation));
                let output = generation_dir.join(if r.role == Role::ProposalModel {
                    "model"
                } else {
                    "shuffled"
                });
                physical_directory(&self.root, &output)?;
                let index =
                    crate::teaching_store::read_index(&self.root, &self.root.join(&corpus_dir))?;
                let allowed = self
                    .config
                    .preparation_overlay
                    .as_ref()
                    .map(|a| crate::preparation_overlay::views(&self.root, a, &self.config.run))
                    .transpose()?;
                let contexts:Vec<_>=index.contexts.iter().enumerate().filter(|(_,c)|allowed.as_ref().is_none_or(|v|v.contains(&(c.dataset.clone(),c.cut))) && c.partition=="meta_train" && !self.policy.excluded_families.contains(&c.family) && c.cut>=self.config.cut_start && c.cut-self.config.cut_start<self.config.cuts).map(|(i,c)|json!({"context_index":i,"identity":crate::service::coverage::Context::from_row(c)})).collect();
                ensure!(
                    !contexts.is_empty(),
                    "no eligible unsealed proposal contexts"
                );
                let selection = frozen(
                    &self.root,
                    &generation_dir.join("coverage-selection.json"),
                    &json!({"schema":"jope.native_generation_selection.v1","request":{"root":self.root,"protocol":"all_eligible_training_contexts_v1","run":r.run,"generation":r.generation,"corpus_sha256":corpus.sha256,"scientific_policy":self.config.scientific_policy,"preparation_overlay":self.config.preparation_overlay,"cut_start":self.config.cut_start,"cuts":self.config.cuts},"contexts":contexts}),
                )?;
                let snapshot = crate::champions::freeze_native_snapshot(
                    &self.root,
                    &self.root.join(&corpus_dir),
                    &self.root.join(&checkpoint_dir),
                    &output,
                    self.config.cuts,
                    self.config.cut_start,
                    &self.config.fit_policy,
                )?;
                let snapshot_ref = artifact(&self.root, &snapshot)?;
                frozen(
                    &self.root,
                    &output.join("run.json"),
                    &json!({"schema":"jope.native_proposal_run.v1","run":r.run,"generation":r.generation,"proposal_seed":r.parameters["proposal_seed"],"coverage_selection_sha256":selection.sha256,"champion_snapshot_sha256":snapshot_ref.sha256,"scientific_policy":self.config.scientific_policy}),
                )?;
                let output = output.strip_prefix(&self.root)?.to_owned();
                let operation = Operation::Proposal {
                    preparation_overlay: self.config.preparation_overlay.clone(),
                    corpus: corpus_dir.clone(),
                    checkpoint: checkpoint_dir.clone(),
                    output: output.clone(),
                    champion_snapshot: snapshot_ref.path.clone(),
                    cuts: self.config.cuts,
                    cut_start: self.config.cut_start,
                    shuffled: r.role == Role::ProposalShuffled,
                };
                let mut inputs = crate::resident_protocol::request_artifacts(
                    &self.root,
                    &json!({"kind":"Proposal","corpus":corpus_dir,"checkpoint":checkpoint_dir,"output":output}),
                    &json!({"JOPE_CHAMPION_SNAPSHOT":snapshot_ref.path}),
                )?;
                if let Some(a) = &self.config.preparation_overlay {
                    inputs.extend(crate::preparation_overlay::dependencies(&self.root, a)?);
                }
                (
                    operation,
                    inputs,
                    TaskKind::ResidentGpu,
                    Delivery::Resident {
                        job_id: self.config.resident_job,
                    },
                    settings.resident_resources.clone(),
                    Some(checkpoint.sha256),
                    Some(corpus.sha256),
                )
            }
            Role::Train => {
                let checkpoint: Artifact =
                    serde_json::from_value(r.parameters["checkpoint"].clone())?;
                let corpus: Artifact = serde_json::from_value(r.parameters["corpus"].clone())?;
                let corpus_dir = corpus
                    .path
                    .parent()
                    .context("training corpus parent")?
                    .to_owned();
                let checkpoint_dir = checkpoint
                    .path
                    .parent()
                    .context("training checkpoint parent")?
                    .to_owned();
                let policy = training_eligibility(
                    &self.root,
                    &corpus,
                    &checkpoint,
                    &self.config.scientific_policy,
                    r,
                )?;
                let replay = frozen(&self.root, &base.join("replay-policy.json"), &policy)?;
                let run = base.join("run");
                let launch = base.join("launch");
                physical_directory(&self.root, &run)?;
                physical_directory(&self.root, &launch)?;
                frozen(
                    &self.root,
                    &run.join("run.json"),
                    &json!({"schema":"jope.native_training_run.v1","run":r.run,"generation":r.generation,"initial_optimizer_step":r.parameters["initial_step"],"target_optimizer_step":r.parameters["target_step"],"global_batch":2048,"microbatch":self.config.microbatch,"replay_policy":replay,"scientific_policy":self.config.scientific_policy}),
                )?;
                let run = run.strip_prefix(&self.root)?.to_owned();
                let launch = launch.strip_prefix(&self.root)?.to_owned();
                let operation = Operation::Train {
                    corpus: corpus_dir.clone(),
                    run: run.clone(),
                    launch: launch.clone(),
                    micro: self.config.microbatch,
                    checkpoint_steps: settings.checkpoint_steps,
                    initialize: checkpoint_dir.clone(),
                    replay_policy: replay.path.clone(),
                    initial_optimizer_step: self.config.initial_optimizer_step
                        + r.generation * 3000,
                    requested_step_increment: 3000,
                    target_optimizer_step: self.config.initial_optimizer_step
                        + (r.generation + 1) * 3000,
                };
                let inputs = crate::resident_protocol::input_artifacts(
                    &self.root,
                    &json!({"kind":"Train","corpus":corpus_dir,"run":run,"launch":launch,"initialize":checkpoint_dir,"replay_policy":replay.path,"resume":false}),
                )?;
                (
                    operation,
                    inputs,
                    TaskKind::ResidentGpu,
                    Delivery::Resident {
                        job_id: self.config.resident_job,
                    },
                    settings.resident_resources.clone(),
                    Some(checkpoint.sha256),
                    Some(corpus.sha256),
                )
            }
            Role::PackAndPublish => {
                let target: DatasetTarget =
                    native_genome::load(&self.root, &settings.dataset_target)?;
                ensure!(
                    target.schema == "jope.native_dataset_target.v1"
                        && target.packer.path.is_absolute(),
                    "fixed dataset target contract absent"
                );
                let closure: Artifact = serde_json::from_value(r.parameters["closure"].clone())?;
                let set: Artifact = serde_json::from_value(r.parameters["target_set"].clone())?;
                let output = base.join("corpus").strip_prefix(&self.root)?.to_owned();
                let manifest = json!({"schema":"jain.datasets_service_manifest.v1","root":self.root,"store":target.store,"mirror_root":target.mirror_root,"namespace":target.namespace,"publication":target.publication,"receiver":target.receiver,"task_id":r.id,"attempt":1,"expected_receipt_id":format!("{}/{}/complete",r.run,r.id),"output_receipt":output_receipt,
                    "operation":{"kind":"pack_continuation_and_publish","packer":target.packer,"target_set":set,"closure":closure,"output":output}});
                let reference = frozen(&self.root, &base.join("dataset-manifest.json"), &manifest)?;
                let bytes = native_genome::checked(&self.root, &closure, 256 * 1024 * 1024)?;
                let mut inputs = BTreeMap::new();
                for line in bytes.split(|b| *b == b'\n').filter(|b| !b.is_empty()) {
                    ensure!(
                        line.len() <= 8192 && inputs.len() < 1_000_000,
                        "dataset closure bounds"
                    );
                    let a: Artifact = serde_json::from_slice(line)?;
                    ensure!(
                        inputs.insert(a.path, a.sha256).is_none(),
                        "duplicate dataset source path"
                    );
                }
                inputs.insert(closure.path, closure.sha256);
                inputs.insert(reference.path.clone(), reference.sha256.clone());
                inputs.insert(
                    settings.dataset_target.path.clone(),
                    settings.dataset_target.sha256.clone(),
                );
                (
                    Operation::PackAndPublish {
                        manifest: reference.path,
                        sha256: reference.sha256,
                        logical_attempt: Some(1),
                    },
                    inputs,
                    TaskKind::Packing,
                    Delivery::NativeJob,
                    settings.packing_resources.clone(),
                    None,
                    None,
                )
            }
            _ => return Ok(false),
        };
        dependencies.insert(
            self.config.scientific_policy.path.clone(),
            self.config.scientific_policy.sha256.clone(),
        );
        dependencies.insert(
            self.policy.family_partition_receipt.path.clone(),
            self.policy.family_partition_receipt.sha256.clone(),
        );
        dependencies.insert(
            self.policy.exclusion_receipt.path.clone(),
            self.policy.exclusion_receipt.sha256.clone(),
        );
        let mut closure_bytes = Vec::new();
        for (path, sha256) in dependencies {
            serde_json::to_writer(&mut closure_bytes, &Artifact { path, sha256 })?;
            closure_bytes.push(b'\n');
        }
        let closure = frozen_bytes(
            &self.root,
            &base.join("input-closure.jsonl"),
            &closure_bytes,
        )?;
        let input = InputManifest {
            schema: "jope.smartcluster_input.v1".into(),
            root: self.root.clone(),
            operation,
            input_artifacts: BTreeMap::new(),
            closure: Some(ClosureManifest {
                path: closure.path.clone(),
                sha256: closure.sha256.clone(),
            }),
            output_receipt,
        };
        let input_ref = frozen(&self.root, &base.join("task.input.json"), &input)?;
        let task = TaskManifest {
            schema: "jope.smartcluster_task.v1".into(),
            run: r.run.clone(),
            generation: r.generation,
            task_id: r.id.clone(),
            kind,
            input_manifest_sha256: input_ref.sha256,
            dependencies: BTreeMap::from([(
                closure.path.to_string_lossy().into_owned(),
                closure.sha256,
            )]),
            parent_checkpoint_sha256: parent,
            corpus_sha256: corpus,
            idempotency_key: IdempotencyKey::new(format!("{}/{}", r.run, r.id))?,
            deadline: self.config.task_deadline,
            retry_allowance: 2,
            resources,
            expected_receipt_id: format!("{}/{}/complete", r.run, r.id),
            checkpoint_allowance_ms: if kind == TaskKind::ResidentGpu {
                Some(settings.checkpoint_allowance_ms)
            } else {
                None
            },
        };
        let manifest = frozen(&self.root, &base.join("task.json"), &task)?;
        self.bind(&Binding {
            schema: "jope.native_loop_binding.v1".into(),
            request: request_ref,
            admission: Admission {
                schema: "jope.native_graph_admission.v1".into(),
                manifest,
                predecessors: r.predecessors.clone(),
                delivery,
            },
        })?;
        Ok(true)
    }
}
fn frozen_bytes(root: &Path, path: &Path, bytes: &[u8]) -> Result<Artifact> {
    immutable(root, path, bytes)
}
pub(crate) fn training_eligibility(
    root: &Path,
    corpus: &Artifact,
    parent: &Artifact,
    scientific: &Artifact,
    request: &Request,
) -> Result<Value> {
    ensure!(
        request.role == Role::Train && !request.predecessors.is_empty(),
        "training requires verified changed-corpus dependency"
    );
    let directory = corpus.path.parent().context("new corpus directory")?;
    let index: Value = native_genome::load_corpus_index(root, corpus)?;
    let target = Artifact {
        path: directory.join("target-set.json"),
        sha256: index["target_set_sha256"]
            .as_str()
            .context("continuation target set identity")?
            .into(),
    };
    let set: crate::continuation_corpus::TargetSet = native_genome::load(root, &target)?;
    ensure!(
        set.generation == request.generation
            && set.run == request.run
            && set.frozen_scientific_policy == *scientific
            && !set.qualification,
        "training target scientific scope changed"
    );
    let rendered = crate::continuation_corpus::render(root, &target)?;
    for (name, bytes) in &rendered {
        ensure!(
            crate::manifest::file_hash(&root.join(directory).join(name))? == hash(bytes),
            "training corpus differs from verified target reconstruction"
        );
    }
    let previous: Value = native_genome::load_corpus_index(root, &set.parent_corpus)?;
    ensure!(
        previous["records_sha256"] != index["records_sha256"],
        "metadata-only corpus change does not justify a new training block"
    );
    let state: Value = native_genome::load(root, parent)?;
    ensure!(
        state["step"] == request.parameters["initial_step"]
            && request.parameters["initial_step"]
                .as_u64()
                .and_then(|s| s.checked_add(3000))
                == request.parameters["target_step"].as_u64()
            && request.parameters["requested_increment"] == 3000
            && request.parameters["global_batch"] == 2048
            && state["vocabulary_sha256"] == index["vocabulary_sha256"],
        "original optimizer parent/window or vocabulary differs"
    );
    let groups = index["groups"]
        .as_array()
        .context("training groups absent")?;
    for group in groups {
        ensure!(
            group["partition"] == "meta_train"
                && group["records"].as_array().is_some_and(|r| r.len() >= 8),
            "ineligible continuation training group"
        );
    }
    let fresh = fresh_record_groups(
        root,
        rendered
            .get("labels.jsonl")
            .context("rendered labels absent")?,
        groups,
        request.generation,
    )?;
    ensure!(
        !fresh.is_empty(),
        "new block has no new-generation training targets"
    );
    let fresh_groups = fresh
        .keys()
        .map(|key| key.parse::<usize>().map_err(anyhow::Error::from))
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"schema":"parallel.replay_policy.v2","eligible":true,"optimizer":"carry","replay":"broad","sampling_seed":generation_seed(&request.run,request.generation,&parent.sha256)?,"parent_checkpoint_sha256":parent.sha256,"corpus_sha256":corpus.sha256,"vocabulary_sha256":index["vocabulary_sha256"],"fresh_groups":fresh_groups,"fresh_records":fresh,"final_tests_opened":false,
        "eligibility_evidence":{"schema":"jope.native_continuation_eligibility.v1","target_set":target,"scientific_policy":scientific,"verified_corpus_dependencies":request.predecessors,"original_window":request.parameters,"target_kind":"continuation_only_v1","new_records_required":true}}),
    )
}

fn fresh_record_groups(
    root: &Path,
    labels: &[u8],
    groups: &[Value],
    generation: u64,
) -> Result<BTreeMap<String, Vec<usize>>> {
    let mut fresh = BTreeMap::<String, Vec<usize>>::new();
    for line in labels.split(|b| *b == b'\n').filter(|b| !b.is_empty()) {
        let label: Value = serde_json::from_slice(line)?;
        let target: Value =
            native_genome::load(root, &serde_json::from_value(label["target"].clone())?)?;
        let candidate: Value =
            native_genome::load(root, &serde_json::from_value(target["candidate"].clone())?)?;
        if candidate["generation"] != generation {
            continue;
        }
        let record = label["record"].as_u64().context("label record")? as usize;
        let group = groups
            .iter()
            .position(|g| {
                g["context"] == label["context"]
                    && g["records"]
                        .as_array()
                        .is_some_and(|r| r.iter().any(|v| v == record))
            })
            .context("fresh record absent from original group")?;
        fresh.entry(group.to_string()).or_default().push(record);
    }
    for records in fresh.values() {
        ensure!(
            records.iter().collect::<BTreeSet<_>>().len() == records.len(),
            "duplicate fresh record identity"
        );
    }
    Ok(fresh)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn previous_generation_targets_are_not_relabelled_as_fresh() {
        let t = tempfile::tempdir().unwrap();
        let mut labels = Vec::new();
        for (record, generation) in [(0, 0), (1, 1)] {
            let c = frozen(
                t.path(),
                &t.path().join(format!("fixtures/c{record}.json")),
                &json!({"generation":generation}),
            )
            .unwrap();
            let target = frozen(
                t.path(),
                &t.path().join(format!("fixtures/t{record}.json")),
                &json!({"candidate":c}),
            )
            .unwrap();
            serde_json::to_writer(
                &mut labels,
                &json!({"record":record,"context":0,"target":target}),
            )
            .unwrap();
            labels.push(b'\n');
        }
        let groups = vec![json!({"context":0,"records":[0,1]})];
        assert_eq!(
            fresh_record_groups(t.path(), &labels, &groups, 1).unwrap(),
            BTreeMap::from([("0".to_owned(), vec![1])])
        );
        assert!(fresh_record_groups(t.path(), &labels, &groups, 2)
            .unwrap()
            .is_empty());
    }
}
