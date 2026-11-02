use super::*;

pub fn run(
    root: &Path,
    corpus_path: &Path,
    output: &Path,
    steps: usize,
    cuts: usize,
    cut_start: usize,
    contexts_path: Option<&Path>,
    initialize: Option<&Path>,
    extra_observations: &[PathBuf],
    qualification: bool,
) -> Result<()> {
    crate::task_contract::require_original_gpu_grant("learning-cycle")?;
    crate::campaign::check_deadline()?;
    ensure!(
        qualification,
        "this campaign requires explicit qualification until full corpus/family gates pass"
    );
    ensure!(
        root.is_absolute() && output.is_absolute() && corpus_path.is_absolute(),
        "cycle paths must be absolute"
    );
    fs::create_dir_all(output)?;
    let _lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(output.join("cycle.lock"))?;
    ensure!(
        _lock.try_lock().is_ok(),
        "another controller owns this cycle directory"
    );
    let (executable, executable_sha) = verify_cycle_identity(
        root,
        corpus_path,
        output,
        steps,
        cuts,
        cut_start,
        contexts_path,
        initialize,
        extra_observations,
        false,
    )?;
    let index: corpus::Index = read_json(&corpus_path.join("index.json"))?;
    let default_contexts = root.join("data/encodings/contexts-v1/index.json");
    let contexts_path = contexts_path.unwrap_or(&default_contexts);
    ensure!(
        file_hash(contexts_path)? == index.context_index_sha256,
        "cycle context cohort differs from training corpus"
    );
    let total = index
        .contexts
        .iter()
        .filter(|c| c.partition != "meta_test" && c.cut >= cut_start && c.cut - cut_start < cuts)
        .count();
    ensure!(total > 0, "no saved contexts in requested cut range");
    let all_total = total * 3;
    let train1 = output.join("train-1");
    progress(
        root,
        if initialize.is_some() {
            "Continue policy learning on the expanded corpus"
        } else {
            "Fresh policy learning"
        },
        (0, total * 2),
        (0, all_total),
        (0, steps),
        false,
        "All selected cut contexts are already saved. GP continues in its bounded CPU pool.",
    )?;
    if !train1.join("complete.json").exists() {
        controller::train(
            root,
            corpus_path,
            &train1,
            crate::controller::TrainingOptions {
                world: 8,
                micro: 256,
                steps,
                checkpoint_steps: 1000,
                resume: train1.exists(),
                fail_rank: None,
                initialize,
            },
        )?;
    }
    let first = latest(root, &train1)?;
    let proposal_dir = output.join("neural");
    progress(
        root,
        "Neural proposals on all eight GPUs",
        (0, total * 2),
        (0, all_total),
        (0, steps),
        false,
        "Generate grammar-valid candidates from the completed policy checkpoint.",
    )?;
    if !proposal_dir.join("complete.json").exists() {
        controller::propose(
            root,
            corpus_path,
            &first,
            &proposal_dir,
            cuts,
            cut_start,
            false,
        )?;
    }
    let neural: GpJobs = read_json(&proposal_dir.join("jobs.json"))?;
    let generic_dir = output.join("generic");
    // The vocabulary this index binds, read once (W1 route R8).
    let vocab: Vocab = corpus::read_bound_vocabulary(corpus_path, &index.vocabulary_sha256)?;
    let generic = if generic_dir.join("jobs.json").exists() {
        read_json(&generic_dir.join("jobs.json"))?
    } else {
        baseline(root, &neural, &vocab, &generic_dir)?
    };
    let shuffled_dir = output.join("shuffled");
    progress(root,"Shuffled-context proposal comparison",(total,total*2),(0,all_total),(0,steps),false,"Same checkpoint and search budget; conditioning contexts are permuted across training families.")?;
    if !shuffled_dir.join("complete.json").exists() {
        controller::propose(
            root,
            corpus_path,
            &first,
            &shuffled_dir,
            cuts,
            cut_start,
            true,
        )?;
    }
    let shuffled: GpJobs = read_json(&shuffled_dir.join("jobs.json"))?;
    let mut combined = GpJobs {
        schema: "parallel.gp_jobs.v1".into(),
        qualification: true,
        jobs: Vec::new(),
        vocabulary: neural.vocabulary.clone(),
    };
    // Interleave arms so changes in host load affect comparisons similarly.
    for i in 0..neural.jobs.len() {
        combined.jobs.extend([
            neural.jobs[i].clone(),
            generic.jobs[i].clone(),
            shuffled.jobs[i].clone(),
        ]);
    }
    crate::campaign::check_deadline()?;
    crate::queue::merge(root, &combined)?;
    write_json(&output.join("search-jobs.json"), &combined)?;
    let pending_work = combined
        .jobs
        .iter()
        .any(|j| !root.join(&j.output).join("search.json").exists());
    let auxiliary = crate::auxiliary_training::Auxiliary::start(
        root,
        corpus_path,
        &first,
        &output.join("auxiliary"),
        crate::auxiliary_training::Options::for_cycle(steps),
        pending_work,
    )?;
    let begun = Instant::now();
    loop {
        crate::campaign::check_deadline()?;
        let pending = combined
            .jobs
            .iter()
            .filter(|j| !root.join(&j.output).join("search.json").exists())
            .map(|j| j.output.clone())
            .collect::<Vec<_>>();
        let done = combined.jobs.len() - pending.len();
        progress(root,"GP optimizes cut targets for the next teaching corpus",(total*2,total*2),(done,combined.jobs.len()),(0,steps),false,"Neural, generic/retained and shuffled-context arms share 1000-evaluation / 600-second view limits.")?;
        if pending.is_empty() {
            break;
        }
        let service: Value = read_json(&root.join("runs/gp-service.json")).unwrap_or_default();
        let healthy = check_gp_health(&service,&pending,begun.elapsed(),now()).and_then(|_| {
            ensure!(begun.elapsed()<Duration::from_secs(7200),"priority searches did not finish within two hours; resume cycle after inspecting worker receipts"); Ok(())
        });
        if let Err(error) = healthy {
            write_json(
                &output.join("gp-wait-failure.json"),
                &json!({"timestamp":now(),"failure":format!("{error:#}"),"pending_jobs":pending,"service":service,"completed_jobs":done,"resumable":true}),
            )?;
            progress(
                root,
                "GP queue needs recovery",
                (total * 2, total * 2),
                (done, combined.jobs.len()),
                (0, steps),
                false,
                &format!("{error:#}"),
            )?;
            return Err(error);
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    let auxiliary_selection = auxiliary.finish()?;
    let second_initialization = resolve(root, &auxiliary_selection.checkpoint)?;
    let mut comparisons = Vec::new();
    let mut predictor_exports = Vec::new();
    let mut family_ratios = BTreeMap::<String, Vec<f64>>::new();
    progress(
        root,
        "Fit and save predictors",
        (total * 2, total * 2),
        (all_total, all_total),
        (0, steps),
        false,
        "CPU fitting and export; liveness is separate from committed predictor receipts.",
    )?;
    let export_heartbeat = StageHeartbeat::start(
        &root.join("runs/cycle-progress.json"),
        read_json(&root.join("runs/cycle-progress.json"))?,
    )?;
    export_heartbeat.advance("predictor receipts committed", 0, Some(neural.jobs.len()))?;
    for i in 0..neural.jobs.len() {
        crate::campaign::check_deadline()?;
        let n: Value = read_json(&root.join(&neural.jobs[i].output).join("search.json"))?;
        let g: Value = read_json(&root.join(&generic.jobs[i].output).join("search.json"))?;
        let s: Value = read_json(&root.join(&shuffled.jobs[i].output).join("search.json"))?;
        let prepared: crate::prepare::Prepared =
            read_json(&root.join(&neural.jobs[i].prepared).join("prepared.json"))?;
        if prepared.meta_partition == "meta_validation" {
            if let (Some(a), Some(b)) = (
                n["best_validation_rmse"].as_f64(),
                g["best_validation_rmse"].as_f64(),
            ) {
                family_ratios
                    .entry(prepared.family.clone())
                    .or_default()
                    .push(((a + 1e-12) / (b + 1e-12)).ln());
            }
        }
        comparisons.push(json!({"dataset":prepared.dataset,"family":prepared.family,"partition":prepared.meta_partition,"cut":neural.jobs[i].cut,"neural":n,"generic_retained":g,"shuffled_context":s}));
        let job = &neural.jobs[i];
        let predictor = output
            .join("predictors")
            .join(&prepared.dataset)
            .join(format!("cut-{}.json", job.cut));
        let result = (|| {
            let seeds = job.seeds.as_ref().map(|p| resolve(root, p)).transpose()?;
            let vocabulary = neural
                .vocabulary
                .as_ref()
                .map(|p| resolve(root, p))
                .transpose()?;
            crate::gp::handoff::export_selected(&crate::gp::handoff::SelectedExport {
                root,
                search_output: &root.join(&job.output),
                prepared: &root.join(&job.prepared),
                cut: job.cut,
                seeds: seeds.as_deref(),
                vocabulary: vocabulary.as_deref(),
                executable: &executable,
                executable_sha: &executable_sha,
                predictor: &predictor,
            })
        })();
        predictor_exports.push(match result {
            Ok(receipt) => receipt,
            Err(error) => json!({"passed":false,"dataset":prepared.dataset,"cut":job.cut,"predictor":predictor,"prepared_sha256":job.prepared_sha256,"executable_sha256":executable_sha,"failure":format!("{error:#}"),"receipt":predictor.with_extension("fit.json"),"receipt_sha256":file_hash(&predictor.with_extension("fit.json")).ok()}),
        });
        write_json(
            &output.join("predictor-exports.json"),
            &json!({"schema":"parallel.predictor_exports.v1","exports":predictor_exports,"completed":predictor_exports.len(),"total":neural.jobs.len(),"evaluator_sha256":executable_sha,"final_tests_opened":false,"qualification":true,"failures_block_retraining":false}),
        )?;
        export_heartbeat.advance(
            "predictor receipts committed",
            predictor_exports.len(),
            Some(neural.jobs.len()),
        )?;
    }
    export_heartbeat.finish()?;
    write_json(
        &output.join("comparisons.json"),
        &json!({"schema":"parallel.search_comparison.v1","comparisons":comparisons,"meta_validation_family_log_rmse_ratios":family_ratios,"evaluation_bias":"adaptive_cut_heldout_teacher","qualification":true,"independent_final_test":false,"final_tests_opened":false}),
    )?;
    let corpus2 = output.join("corpus-2");
    progress(
        root,
        "Pack newly scored programs into the next teaching corpus",
        (total * 2, total * 2),
        (all_total, all_total),
        (0, steps),
        false,
        "Exact program labels and saved contexts remain bound to their cut hashes.",
    )?;
    let mut observation_roots = [
        "runs/gp-qualification",
        "runs/gp-expanded",
        "runs/gp-extra20",
    ]
    .into_iter()
    .map(|p| root.join(p))
    .filter(|p| p.exists())
    .collect::<Vec<_>>();
    observation_roots.extend([proposal_dir, generic_dir, shuffled_dir]);
    observation_roots.extend(extra_observations.iter().cloned());
    let packing_heartbeat = StageHeartbeat::start(
        &root.join("runs/cycle-progress.json"),
        read_json(&root.join("runs/cycle-progress.json"))?,
    )?;
    build_corpus_atomic(
        root,
        contexts_path,
        &corpus_path.join("vocabulary.json"),
        &observation_roots,
        &corpus2,
        Some(&packing_heartbeat),
    )?;
    packing_heartbeat.advance("corpus atomically published", 1, Some(1))?;
    packing_heartbeat.finish()?;
    let rebuilt: corpus::Index = read_json(&corpus2.join("index.json"))?;
    let coverage = |i: &corpus::Index| {
        i.groups
            .iter()
            .map(|g| (g.family.clone(), g.partition.clone()))
            .collect::<BTreeSet<_>>()
    };
    ensure!(
        coverage(&rebuilt).is_superset(&coverage(&index)),
        "repacking lost a training/validation family before retraining"
    );
    let train2 = output.join("train-2");
    progress(root,"Learn again from GP results",(total*2,total*2),(all_total,all_total),(0,steps),false,"Initialize the next policy from the fixed-panel best auxiliary checkpoint; bind the new GP corpus and reset optimizer deliberately.")?;
    crate::campaign::check_deadline()?;
    if !train2.join("complete.json").exists() {
        controller::train(
            root,
            &corpus2,
            &train2,
            crate::controller::TrainingOptions {
                world: 8,
                micro: 256,
                steps,
                checkpoint_steps: 1000,
                resume: train2.exists(),
                fail_rank: None,
                initialize: Some(&second_initialization),
            },
        )?;
    }
    crate::campaign::check_deadline()?;
    let second = latest(root, &train2)?;
    write_json(
        &output.join("complete.json"),
        &json!({"schema":"parallel.cycle_complete.v1","completed_at":now(),"first_checkpoint":first.strip_prefix(root)?,"second_checkpoint":second.strip_prefix(root)?,"second_initialization":auxiliary_selection,"auxiliary_receipt_sha256":file_hash(&output.join("auxiliary/complete.json"))?,"teaching_corpus_sha256":file_hash(&corpus2.join("index.json"))?,"neural_searches":total,"comparison_searches":total*2,"comparisons_sha256":file_hash(&output.join("comparisons.json"))?,"predictor_exports_sha256":file_hash(&output.join("predictor-exports.json"))?,"predictor_exports_passed":predictor_exports.iter().filter(|p|p["passed"]==true).count(),"predictor_exports_failed":predictor_exports.iter().filter(|p|p["passed"]!=true).count(),"predictor_evaluator_sha256":executable_sha,"dl_gp_dl_complete":true,"qualification":true,"pilot_started":false,"final_tests_opened":false}),
    )?;
    progress(root,"DL → GP → DL integration completed",(total*2,total*2),(all_total,all_total),(steps,steps),true,"Measured comparisons saved. Full provenance/family admission and the 24-hour pilot remain separate gates.")?;
    Ok(())
}
