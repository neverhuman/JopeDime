use super::*;

mod recipe;
pub(super) use recipe::trainer_recipe;

pub(super) fn production_task_v3(
    ledger: &CausalEvidenceLedgerV3,
    accepted: &BTreeMap<Task, AcceptedPointer>,
) -> Option<Task> {
    Task::ALL.into_iter().find(|task| {
        ledger.promotion_epoch.get(task).is_some_and(|row| {
            row.production_gate_open
                && accepted.get(task).is_some_and(|pointer| {
                    row.evidence.accepted_parent_checkpoint_sha256.as_ref() == Some(&pointer.digest)
                })
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn render_spec(
    args: &Args,
    job_id: &str,
    output_name: &str,
    task: Task,
    mode: TrainerMode,
    seed: u64,
    corpus: &CorpusPointer,
    warm_path: &Path,
    warm_digest: &Sha256Digest,
    accepted_parent: &Sha256Digest,
    recipe: &str,
) -> Result<String> {
    let packed: Value = read_json(&corpus.packed_manifest)?;
    let packed_root = corpus
        .packed_manifest
        .parent()
        .context("packed manifest has no parent")?;
    let catalog = PathBuf::from(
        packed
            .get("catalog_data")
            .and_then(Value::as_str)
            .context("packed manifest lacks catalog data")?,
    );
    let catalog_sha = packed
        .get("catalog_data_sha256")
        .and_then(Value::as_str)
        .context("packed manifest lacks catalog digest")?;
    ensure!(
        catalog.is_file() && sha256_file(&catalog)? == catalog_sha,
        "packed catalog content drift"
    );
    let mode_name = match mode {
        TrainerMode::Shadow => "shadow",
        TrainerMode::Production => "production",
    };
    let quoted = |value: &str| serde_json::to_string(value).expect("JSON string");
    let resource_block = TRAINER.render_resource_block();
    let placement_block = TRAINER.render_placement_block();
    let limits_block = TRAINER.render_limits_block();
    let mut text = format!(
        "[job]\nid = {}\nproject = \"{PROJECT}\"\ntier = \"normal\"\n\
         command = \"/usr/bin/python3\"\n\
         args = [\"{{input:script}}\", \"train\", \"--corpus-manifest\", \"{{input:manifest}}\", \
         \"--output\", \"{{output:training}}/{output_name}\", \"--device\", \"cuda\", \
         \"--warm\", \"{{input:checkpoint}}\", \"--recipe\", \"{recipe}\", \"--seed\", \"{seed}\", \
         \"--task\", \"{}\", \"--mode\", \"{mode_name}\", \"--packed-manifest\", \
         \"{{input:packed_manifest}}\", \"--packed-catalog-data\", \"{{input:packed_catalog}}\", \
         \"--accepted-parent-sha256\", \"{}\"]\n\n\
         {resource_block}\n{placement_block}\n\
         [env]\nPYTHONPATH = {}\nOMP_NUM_THREADS = \"{}\"\n\
         JOPE_TRAINER_INTENT_SCHEMA = \"TrainerIntent.v3\"\n\n\
         [preempt]\npreemptible = {}\ncheckpoint = \"none\"\nresume = \"rerun\"\n\n\
         {limits_block}\n",
        quoted(job_id),
        task.as_str(),
        accepted_parent.as_str(),
        quoted(&args.prime_source.to_string_lossy()),
        TRAINER.omp_threads,
        TRAINER.preemptible,
    );
    let mut inputs = vec![
        (
            "script",
            args.trainer_script.clone(),
            "scripts/test_exact_train.py".to_owned(),
            None,
        ),
        (
            "common",
            args.common_script.clone(),
            "scripts/test_exact_common.py".to_owned(),
            None,
        ),
        (
            "packed_source",
            args.packed_script.clone(),
            "scripts/test_exact_packed_corpus.py".to_owned(),
            None,
        ),
        (
            "manifest",
            corpus.manifest.clone(),
            "corpus/corpus-manifest.json".to_owned(),
            Some(corpus.manifest_sha256.clone()),
        ),
        (
            "checkpoint",
            warm_path.to_owned(),
            format!("checkpoint/{}.model.pt", warm_digest.as_str()),
            Some(warm_digest.as_str().to_owned()),
        ),
        (
            "packed_manifest",
            corpus.packed_manifest.clone(),
            "packed/manifest.json".to_owned(),
            Some(corpus.packed_manifest_sha256.clone()),
        ),
        (
            "packed_catalog",
            catalog,
            "packed/catalog-data.f32le".to_owned(),
            Some(catalog_sha.to_owned()),
        ),
        (
            "trainer_source",
            args.prime_source.join("jope/policy/train_v3.py"),
            "provenance/train_v3.py".to_owned(),
            None,
        ),
    ];
    let files = packed
        .get("files")
        .and_then(Value::as_object)
        .context("packed manifest lacks files")?;
    for (index, (name, metadata)) in files.iter().enumerate() {
        let path = packed_root.join(name);
        let digest = metadata
            .get("sha256")
            .and_then(Value::as_str)
            .context("packed file lacks digest")?;
        inputs.push((
            Box::leak(format!("packed_{index}").into_boxed_str()),
            path,
            format!("packed/{name}"),
            Some(digest.to_owned()),
        ));
    }
    for (name, source, target, pinned) in inputs {
        ensure!(
            source.is_file(),
            "trainer input is absent: {}",
            source.display()
        );
        let digest = pinned.unwrap_or(sha256_file(&source)?);
        ensure!(
            sha256_file(&source)? == digest,
            "trainer input content drift: {}",
            source.display()
        );
        text.push_str(&format!(
            "[[input]]\nname = {}\nsource = {}\ntarget = {}\nsha256 = {}\n\n",
            quoted(name),
            quoted(&source.to_string_lossy()),
            quoted(&target),
            quoted(&digest),
        ));
    }
    text.push_str("[[output]]\nname = \"training\"\nsource = \"training\"\ndirectory = true\n");
    Ok(text)
}

pub(super) fn submit_or_register(args: &Args, spec: &Path, job_id: &str) -> Result<&'static str> {
    let check = Command::new(&args.scq).arg("check").arg(spec).output()?;
    let check_message = command_message(&check);
    if !check.status.success() {
        if capacity_backoff(&check_message) {
            register_admission(args, spec, job_id)?;
            return Ok("gpu_admission_requested");
        }
        bail!("task trainer dependency verification failed: {check_message}");
    }
    let submit = Command::new(&args.scq).arg("submit").arg(spec).output()?;
    let message = command_message(&submit);
    if submit.status.success() || message.to_ascii_lowercase().contains("already exists") {
        return Ok("trainer_submitted");
    }
    if capacity_backoff(&message) {
        register_admission(args, spec, job_id)?;
        return Ok("gpu_admission_requested");
    }
    bail!("task trainer submission failed: {message}")
}

pub(super) fn register_admission(args: &Args, spec: &Path, job_id: &str) -> Result<()> {
    let executable = fs::canonicalize(std::env::current_exe()?)?;
    let controller = executable
        .parent()
        .context("task controller executable has no bin directory")?
        .join("jope_gpu_lane_admission");
    let output = Command::new(controller)
        .args([
            "--root",
            args.root.to_string_lossy().as_ref(),
            "--register-role",
            "alternating-trainer",
            "--spec",
            spec.to_string_lossy().as_ref(),
            "--job-id",
            job_id,
        ])
        .output()?;
    if !output.status.success() {
        bail!(
            "task trainer admission registration failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

pub(super) fn harvest_result(
    args: &Args,
    state: &mut ControllerState,
    intent: &ActiveIntent,
    result_path: &Path,
) -> Result<()> {
    let result = read_json(result_path)?;
    ensure!(
        result.get("task").and_then(Value::as_str) == Some(intent.trainer.task.as_str()),
        "trainer result task mismatch"
    );
    let expected_mode = match intent.trainer.mode {
        TrainerMode::Shadow => "shadow",
        TrainerMode::Production => "production",
    };
    ensure!(
        result.get("trainer_mode").and_then(Value::as_str) == Some(expected_mode),
        "trainer result mode mismatch"
    );
    ensure!(
        result
            .get("accepted_ancestor_sha256")
            .and_then(Value::as_str)
            == Some(intent.trainer.warm_start_checkpoint_sha256.as_str()),
        "trainer result warm-start lineage mismatch"
    );
    let digest = Sha256Digest::parse(
        result
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            .context("trainer result lacks checkpoint digest")?
            .to_owned(),
    )?;
    let source = result_path
        .parent()
        .context("trainer result has no parent")?
        .join(format!("{}.model.pt", digest.as_str()));
    ensure!(
        source.is_file() && sha256_file(&source)? == digest.as_str(),
        "trainer result checkpoint artifact drift"
    );
    let directory = match intent.trainer.mode {
        TrainerMode::Shadow => args
            .root
            .join("training/shadow-checkpoints")
            .join(intent.trainer.task.as_str()),
        TrainerMode::Production => args
            .root
            .join("training/production-candidates")
            .join(intent.trainer.task.as_str()),
    };
    fs::create_dir_all(&directory)?;
    let target = directory.join(format!("{}.model.pt", digest.as_str()));
    copy_immutable(&source, &target, digest.as_str())?;
    let model = ModelRef {
        accepted_parent_checkpoint_sha256: intent.trainer.accepted_parent_checkpoint_sha256.clone(),
        checkpoint_sha256: digest.clone(),
        checkpoint_path: target.clone(),
    };
    match intent.trainer.mode {
        TrainerMode::Shadow => {
            let checkpoint = ShadowCheckpointV1 {
                schema: SHADOW_CHECKPOINT_SCHEMA.to_owned(),
                task: intent.trainer.task,
                accepted_parent_checkpoint_sha256: intent
                    .trainer
                    .accepted_parent_checkpoint_sha256
                    .clone(),
                shadow_parent_checkpoint_sha256: intent
                    .trainer
                    .shadow_parent_checkpoint_sha256
                    .clone(),
                checkpoint_sha256: digest.clone(),
                corpus_identity_sha256: intent.trainer.corpus_identity_sha256.clone(),
                corpus_watermark: intent.trainer.corpus_watermark,
                model_hashes: BTreeMap::from([("policy".to_owned(), digest.clone())]),
                trainer_intent_sha256: Sha256Digest::of_bytes(&serde_json::to_vec(
                    &intent.trainer,
                )?),
                created_at_ms: now_ms(),
                promotion_eligible: false,
            };
            checkpoint.validate()?;
            atomic_json(
                &directory.join(format!("{}.v1.json", digest.as_str())),
                &serde_json::to_value(&checkpoint)?,
            )?;
            write_shadow_canary_intent(args, &checkpoint)?;
            state.shadow_models.insert(intent.trainer.task, model);
            state
                .alternation
                .last_shadow_by_task
                .insert(intent.trainer.task, digest);
            state
                .last_shadow_corpus_watermark
                .insert(intent.trainer.task, intent.trainer.corpus_watermark);
            if intent
                .trainer
                .recipe
                .as_ref()
                .and_then(|recipe| recipe.get("recipe_name"))
                .and_then(Value::as_str)
                == Some("e")
            {
                let descendant_receipts = intent
                    .trainer
                    .recipe
                    .as_ref()
                    .and_then(|recipe| recipe.get("completed_descendant_receipts"))
                    .and_then(Value::as_u64)
                    .context("recipe E intent lacks descendant receipt count")?;
                state
                    .last_recipe_e_corpus_watermark
                    .insert(intent.trainer.task, intent.trainer.corpus_watermark);
                state
                    .last_recipe_e_descendant_receipts
                    .insert(intent.trainer.task, descendant_receipts);
            }
            append_event(&args.root, "shadow_checkpoint_quarantined", &checkpoint)?;
        }
        TrainerMode::Production => {
            atomic_json(
                &directory.join(format!("{}.candidate.v1.json", digest.as_str())),
                &json!({
                    "schema": "ProductionCheckpointCandidate.v1",
                    "task": intent.trainer.task,
                    "accepted_parent_checkpoint_sha256":
                        intent.trainer.accepted_parent_checkpoint_sha256,
                    "checkpoint_sha256": digest,
                    "checkpoint": target,
                    "trainer_intent": intent.trainer,
                    "promotion_requires_independent_task_canary_acceptances": 3,
                    "created_at_ms": now_ms(),
                }),
            )?;
            state
                .production_candidates
                .insert(intent.trainer.task, model);
            write_production_canary_intents(args, intent, &digest)?;
        }
    }
    Ok(())
}
