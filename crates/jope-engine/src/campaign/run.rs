use super::*;

#[cfg(feature = "neural")]
pub fn run(root: &Path, spec_path: &Path) -> Result<()> {
    crate::task_contract::require_original_gpu_grant("learning-campaign")?;
    run_admitted(root, spec_path)
}

#[cfg(feature = "neural")]
pub(super) fn run_admitted(root: &Path, spec_path: &Path) -> Result<()> {
    let spec: Spec = read_json(spec_path)?;
    ensure!(
        spec.schema == "parallel.campaign.v1" && spec.qualification,
        "campaign requires an explicit qualification specification"
    );
    ensure!(
        (3..=24).contains(&spec.hours) && spec.steps > 0 && spec.cuts > 0,
        "campaign requires 3–24 hours and positive step/cut counts"
    );
    let contexts = path(root, &spec.contexts)?;
    let initial = path(root, &spec.initial_corpus)?;
    let output = path(root, &spec.output)?;
    let observations = spec
        .observations
        .iter()
        .map(|p| path(root, p))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        observations.iter().all(|p| p.is_dir()),
        "missing GP observation source"
    );
    fs::create_dir_all(&output)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(root.join("runs/campaign.lock"))?;
    ensure!(
        lock.try_lock().is_ok(),
        "another campaign controller owns this workspace"
    );
    let index: corpus::Index = read_json(&initial.join("index.json"))?;
    let contexts_sha = file_hash(&contexts)?;
    crate::pipeline::verify_corpus(&initial, &contexts_sha, &index.vocabulary_sha256)?;
    let expected_families = family_sets(&index)?;
    ensure!(
        expected_families.0.len() == spec.required_training_families
            && expected_families.1.len() == spec.required_validation_families
            && spec.required_training_families > 0
            && spec.required_validation_families > 0,
        "teaching corpus differs from frozen family coverage: {}/{} training and {}/{} validation",
        expected_families.0.len(),
        spec.required_training_families,
        expected_families.1.len(),
        spec.required_validation_families
    );
    let provenance: Value = read_json(&initial.join("provenance.json"))?;
    for source in provenance["sources"]
        .as_array()
        .context("initial corpus sources absent")?
    {
        let source = path(
            root,
            Path::new(source["path"].as_str().context("source path absent")?),
        )?;
        ensure!(
            observations
                .iter()
                .any(|directory| source.starts_with(directory)),
            "initial teaching source is not covered by campaign observation roots: {}",
            source.display()
        );
    }
    let spec_sha = file_hash(spec_path)?;
    let executable_sha = file_hash(&std::env::current_exe()?)?;
    let (gp_sha, predictor_sha) = evaluator_hashes(root)?;
    let schedule_path = output.join("schedule.json");
    let started = now();
    let proposed = Schedule {
        spec_sha256: spec_sha,
        contexts_sha256: contexts_sha,
        initial_corpus_sha256: file_hash(&initial.join("index.json"))?,
        vocabulary_sha256: index.vocabulary_sha256.clone(),
        executable_sha256: executable_sha,
        gp_executable_sha256: gp_sha,
        predictor_executable_sha256: predictor_sha,
        started_at: started,
        stop_admission_at: started + (spec.hours - 2) * 3600,
        deadline: started + spec.hours * 3600,
    };
    let schedule = if schedule_path.exists() {
        let old: Schedule = read_json(&schedule_path)?;
        ensure!(
            old.same_inputs(&proposed),
            "immutable campaign input or executable changed"
        );
        ensure!(
            old.stop_admission_at == old.started_at + (spec.hours - 2) * 3600
                && old.deadline == old.started_at + spec.hours * 3600,
            "saved campaign deadline is inconsistent with its frozen duration"
        );
        old
    } else {
        write_json(&schedule_path, &proposed)?;
        write_json(&output.join("spec.json"), &spec)?;
        event(
            root,
            "Started a bounded qualification campaign; original final tests remain sealed",
        )?;
        proposed
    };
    let _deadline = DeadlineGuard::new(schedule.deadline);
    let existing_complete = if output.join("complete.json").exists() {
        Some(read_json::<Value>(&output.join("complete.json"))?)
    } else {
        None
    };
    let starts = cut_starts(&index, spec.cuts);
    ensure!(!starts.is_empty(), "no non-test context cut ranges");
    let mut corpus_path = initial.clone();
    let mut initialize: Option<PathBuf> = None;
    let mut finished = Vec::<Value>::new();
    let mut prior = observations.clone();
    let mut expired_cycle: Option<PathBuf> = None;
    for sequence in 0..10000 {
        let cycle = output.join(format!("cycle-{sequence:04}"));
        let receipt = cycle.join("complete.json");
        if !receipt.exists() && existing_complete.is_some() {
            break;
        }
        if !receipt.exists() && !cycle.join("cycle.json").exists() && !schedule.may_start(now()) {
            break;
        }
        let cut_start = spec.cut_strategy.start(&starts, sequence);
        if receipt.exists() {
            crate::pipeline::verify_cycle_identity(
                root,
                &corpus_path,
                &cycle,
                spec.steps,
                spec.cuts,
                cut_start,
                Some(&contexts),
                initialize.as_deref(),
                &prior,
                true,
            )?;
        } else if check_deadline().is_err() {
            expired_cycle = Some(cycle);
            break;
        }
        let (gp_sha, predictor_sha) = evaluator_hashes(root)?;
        ensure!(
            gp_sha == schedule.gp_executable_sha256
                && predictor_sha == schedule.predictor_executable_sha256,
            "campaign GP/predictor executable changed"
        );
        let publish = |phase: &str| {
            write_json(
                &root.join("runs/campaign-progress.json"),
                &json!({"schema":"parallel.campaign_progress.v1","timestamp":now(),"phase":phase,"cycle":sequence+1,"completed_cycles":finished.len(),"current_cycle":cycle,"started_at":schedule.started_at,"stop_admission_at":schedule.stop_admission_at,"deadline":schedule.deadline,"qualification":true,"pilot_started":false,"final_tests_opened":false,"contexts_sha256":schedule.contexts_sha256,"cut_start":cut_start}),
            )
        };
        if existing_complete.is_none() {
            publish("Continuous DL → GP → DL qualification")?;
        }
        if !receipt.exists() {
            event(
                root,
                &format!(
                    "Campaign cycle {}: cut range {}..{}, all eight GPU ranks",
                    sequence + 1,
                    cut_start,
                    cut_start + spec.cuts
                ),
            )?;
            if let Err(error) = crate::pipeline::run(
                root,
                &corpus_path,
                &cycle,
                spec.steps,
                spec.cuts,
                cut_start,
                Some(&contexts),
                initialize.as_deref(),
                &prior,
                true,
            ) {
                let expired = check_deadline().is_err();
                write_json(
                    &output.join("failure.json"),
                    &json!({"timestamp":now(),"cycle":cycle,"failure":format!("{error:#}"),"resumable":!expired,"deadline":schedule.deadline}),
                )?;
                if expired {
                    expired_cycle = Some(cycle);
                    break;
                }
                publish("Campaign cycle needs recovery")?;
                return Err(error);
            }
        }
        let completed: Value = read_json(&receipt)?;
        ensure!(
            completed["dl_gp_dl_complete"] == true
                && completed["final_tests_opened"] == false
                && completed["qualification"] == true,
            "invalid cycle completion receipt"
        );
        ensure!(
            file_hash(&cycle.join("corpus-2/index.json"))? == completed["teaching_corpus_sha256"]
                && file_hash(&cycle.join("comparisons.json"))? == completed["comparisons_sha256"]
                && file_hash(&cycle.join("predictor-exports.json"))?
                    == completed["predictor_exports_sha256"],
            "completed cycle artifacts changed"
        );
        crate::pipeline::verify_corpus(
            &cycle.join("corpus-2"),
            &schedule.contexts_sha256,
            &schedule.vocabulary_sha256,
        )?;
        let next_index: corpus::Index = read_json(&cycle.join("corpus-2/index.json"))?;
        ensure!(
            family_sets(&next_index)? == expected_families,
            "repacked corpus changed frozen training/validation family coverage"
        );
        let next = path(
            root,
            Path::new(
                completed["second_checkpoint"]
                    .as_str()
                    .context("checkpoint missing")?,
            ),
        )?;
        let first = crate::pipeline::latest(root, &cycle.join("train-1"))?;
        let first_training: Value = read_json(&cycle.join("train-1/complete.json"))?;
        let first_state: Value = read_json(&first.join("state.json"))?;
        ensure!(
            completed["first_checkpoint"] == first.strip_prefix(root)?.to_string_lossy().as_ref()
                && first_training["corpus_sha256"] == file_hash(&corpus_path.join("index.json"))?
                && first_training["steps"] == spec.steps
                && first_training["world"] == 8
                && first_training["latest"]["state_sha256"]
                    == file_hash(&first.join("state.json"))?
                && file_hash(&first.join("model.ot"))? == first_state["model_sha256"],
            "completed first training block changed"
        );
        ensure!(
            next == crate::pipeline::latest(root, &cycle.join("train-2"))?,
            "completed cycle checkpoint differs from training receipt"
        );
        let selected = crate::auxiliary_training::load_selection(root, &cycle.join("auxiliary"))?;
        let auxiliary_spec: Value = read_json(&cycle.join("auxiliary/spec.json"))?;
        ensure!(
            auxiliary_spec["corpus_sha256"] == file_hash(&corpus_path.join("index.json"))?
                && auxiliary_spec["frozen_proposal_checkpoint"]["state_sha256"]
                    == file_hash(&first.join("state.json"))?
                && completed["second_initialization"] == serde_json::to_value(&selected)?
                && completed["auxiliary_receipt_sha256"]
                    == file_hash(&cycle.join("auxiliary/complete.json"))?,
            "completed auxiliary selection binding changed"
        );
        let training: Value = read_json(&cycle.join("train-2/complete.json"))?;
        let state: Value = read_json(&next.join("state.json"))?;
        ensure!(
            training["corpus_sha256"] == completed["teaching_corpus_sha256"]
                && training["steps"] == spec.steps
                && training["world"] == 8
                && training["latest"]["state_sha256"] == file_hash(&next.join("state.json"))?
                && state["corpus_sha256"] == completed["teaching_corpus_sha256"]
                && state["vocabulary_sha256"] == schedule.vocabulary_sha256
                && state["initialization"]["state_sha256"] == selected.state_sha256,
            "completed training identity changed"
        );
        ensure!(
            file_hash(&next.join("model.ot"))? == state["model_sha256"]
                && file_hash(&next.join("optimizer.ot"))? == state["optimizer_sha256"]
                && file_hash(&next.join("optimizer.json"))? == state["optimizer_metadata_sha256"]
                && file_hash(&next.join("config.json"))? == state["config_sha256"],
            "completed checkpoint payload changed"
        );
        finished.push(json!({"sequence":sequence,"cycle":cycle,"receipt_sha256":file_hash(&receipt)?,"checkpoint":next,"checkpoint_state_sha256":file_hash(&next.join("state.json"))?,"teaching_corpus_sha256":completed["teaching_corpus_sha256"],"predictors_passed":completed["predictor_exports_passed"],"predictors_failed":completed["predictor_exports_failed"]}));
        corpus_path = cycle.join("corpus-2");
        initialize = Some(next);
        prior.push(cycle.clone());
        if existing_complete.is_none() {
            write_json(
                &output.join("completed-cycles.json"),
                &json!({"schema":"parallel.completed_cycles.v1","cycles":finished,"qualification":true,"final_tests_opened":false}),
            )?;
        }
    }
    if let Some(old) = existing_complete {
        ensure!(
            old["schedule"] == serde_json::to_value(&schedule)? && old["cycles"] == json!(finished),
            "completed campaign receipt changed"
        );
        return Ok(());
    }
    let expired = expired_cycle.is_some() || now() >= schedule.deadline;
    ensure!(
        expired || !finished.is_empty(),
        "campaign completed no learning cycle"
    );
    let complete = json!({"schema":"parallel.campaign_complete.v1","timestamp":now(),"status":if expired{"complete_expired"}else{"completed"},"schedule":schedule,"cycles":finished,"incomplete_cycle":expired_cycle,"dl_gp_dl_complete":!finished.is_empty(),"qualification":true,"pilot_started":false,"final_tests_opened":false,"checkpoint_selection":"not frozen for independent final scoring","stopping_rule":"No new cycle starts in the final two-hour reserve; fixed original deadline stops owned training and fitting children. Independent background GP services retain their own budgets; unfinished cycle work is excluded from completed campaign results."});
    write_json(&output.join("complete.json"), &complete)?;
    if expired {
        write_json(&output.join("expired.json"), &complete)?;
    }
    write_json(
        &root.join("runs/campaign-progress.json"),
        &json!({"schema":"parallel.campaign_progress.v1","timestamp":now(),"phase":if expired{"Qualification deadline reached; completed work saved"}else{"Completed bounded qualification campaign"},"complete":true,"status":complete["status"],"completed_cycles":finished.len(),"qualification":true,"pilot_started":false,"final_tests_opened":false}),
    )?;
    event(
        root,
        "Qualification campaign stopped; completed cycle receipts and checkpoints saved",
    )?;
    Ok(())
}
