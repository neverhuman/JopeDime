//! Resumable real-data DL -> adaptive GP -> DL integration campaign.
use crate::{
    controller, corpus,
    grammar::Vocab,
    manifest::*,
    service::GpJobs,
    telemetry::{now, StageHeartbeat},
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

fn check_gp_health(
    service: &Value,
    pending: &[PathBuf],
    elapsed: Duration,
    timestamp: u64,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let observation = crate::telemetry::observed_status(service, timestamp);
    ensure!((service.is_null() && elapsed < Duration::from_secs(10)) || observation["current_activity"] == "current", "GP service activity is stale, terminal or unbound while {} cycle jobs remain; reconcile the service before resuming this cycle", pending.len());
    ensure!(
        service["phase"] != "failed",
        "GP service reports failure while cycle jobs remain: {}",
        service["failure"]
    );
    ensure!(elapsed < Duration::from_secs(3) || service["phase"] != "completed", "GP service finished its queue but {} cycle jobs have no complete search receipts; restart the GP service to ingest pending priority jobs", pending.len());
    for row in service["failed_rows"].as_array().into_iter().flatten() {
        if row["output"]
            .as_str()
            .is_some_and(|p| pending.iter().any(|j| j == Path::new(p)))
        {
            anyhow::bail!("cycle GP search failed: {row}; recover the job and resume the cycle");
        }
    }
    Ok(())
}

pub(crate) use crate::corpus_publication::verify as verify_corpus;

fn build_corpus_atomic(
    root: &Path,
    contexts: &Path,
    vocab: &Path,
    observations: &[PathBuf],
    output: &Path,
    heartbeat: Option<&StageHeartbeat>,
) -> Result<()> {
    crate::corpus_publication::build(
        root,
        contexts,
        vocab,
        observations,
        output,
        true,
        Some(&corpus_cache_path(root, output)?),
        heartbeat,
    )
}

/// Reuse across cycles in one campaign, with an isolated sibling for standalone
/// cycles. The corpus API verifies namespace separation and all artifact hashes.
fn corpus_cache_path(root: &Path, corpus_output: &Path) -> Result<PathBuf> {
    let cycle = corpus_output
        .parent()
        .context("corpus cycle directory absent")?;
    let parent = cycle.parent().context("corpus campaign directory absent")?;
    let namespace = if cycle
        .file_name()
        .is_some_and(|n| n.to_string_lossy().starts_with("cycle-"))
        && parent != root.join("runs")
    {
        parent
    } else {
        cycle
    };
    let relative = namespace
        .strip_prefix(root)
        .context("corpus cache outside workspace")?;
    ensure!(
        relative.starts_with("runs") && relative.components().count() >= 2,
        "corpus cache requires a private campaign namespace"
    );
    Ok(namespace.join("corpus-scan-cache"))
}

#[cfg(test)]
mod corpus_cache_tests {
    use super::*;

    #[test]
    fn cache_is_shared_by_campaign_cycles_and_isolated_for_other_runs() {
        let root = Path::new("/workspace");
        let path = |s: &str| corpus_cache_path(root, &root.join(s)).unwrap();
        assert_eq!(
            path("runs/campaign/cycle-0000/corpus-2"),
            path("runs/campaign/cycle-0001/corpus-2")
        );
        assert_ne!(
            path("runs/campaign/cycle-0000/corpus-2"),
            path("runs/campaign/expansion/cycle-0000/corpus-2")
        );
        assert_eq!(
            path("runs/standalone/corpus-2"),
            root.join("runs/standalone/corpus-scan-cache")
        );
        assert_eq!(
            path("runs/cycle-0000/corpus-2"),
            root.join("runs/cycle-0000/corpus-scan-cache")
        );
        assert!(corpus_cache_path(root, &root.join("data/corpus-2")).is_err());
    }

    #[test]
    fn pending_jobs_require_current_process_bound_activity() {
        let pending = vec![PathBuf::from("runs/pending")];
        let recent_unbound =
            json!({"timestamp":100,"expires_at":110,"last_progress_at":100,"phase":"running"});
        assert!(check_gp_health(&recent_unbound, &pending, Duration::ZERO, 101).is_err());
        assert!(check_gp_health(&Value::Null, &pending, Duration::from_secs(9), 100).is_ok());
        assert!(check_gp_health(&Value::Null, &pending, Duration::from_secs(10), 100).is_err());
        assert!(check_gp_health(&recent_unbound, &[], Duration::from_secs(60), 160).is_ok());
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("heartbeat.json");
        let heartbeat = StageHeartbeat::start(&path, json!({"phase":"running"})).unwrap();
        let saved: Value = read_json(&path).unwrap();
        let timestamp = saved["timestamp"].as_u64().unwrap();
        assert!(check_gp_health(&saved, &pending, Duration::from_secs(30), timestamp + 9).is_ok());
        assert!(
            check_gp_health(&saved, &pending, Duration::from_secs(30), timestamp + 10).is_err()
        );
        heartbeat.finish().unwrap();
    }

    #[test]
    fn atomic_campaign_pack_creates_private_cache_and_preserves_full_artifacts() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let vocab = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json");
        let mut contexts = Vec::new();
        let mut observations = Vec::new();
        for family in ["a", "b"] {
            let prepared = root.join(family);
            write_json(&prepared.join("prepared.json"),&json!({"schema":"parallel.prepared.v1","dataset":family,"family":family,
                "task":"regression","meta_partition":"meta_train","seed":17,"clean_evaluation":true,"clean_status":"fixture",
                "original_train_sha256":"original","original_test_sha256":"sealed","selected_train_indices":[0,1],
                "selected_test_indices":[],"target_last":true,"rows_total":2,"cuts":[{"index":0,"train_indices":[0],
                "validation_indices":[1],"training_sha256":"train","validation_sha256":"validation","preprocessing_sha256":"preprocessing"}]})).unwrap();
            let prepared_sha = file_hash(&prepared.join("prepared.json")).unwrap();
            let vector = prepared.join("context.bin");
            fs::write(&vector, b"saved fixture context").unwrap();
            let vector_sha = file_hash(&vector).unwrap();
            write_json(&vector.with_extension("json"),&json!({"schema":"parallel.encoding.v1","dataset":family,"family":family,
                "partition":"meta_train","cut":0,"prepared_sha256":prepared_sha,"training_sha256":"train",
                "selected_training_rows_sha256":"rows","preprocessing_sha256":"preprocessing","encoder_spec_sha256":"assembly",
                "numerical_backend":"fixture","dataset_dimensions":1536,"target_dimensions":512,"output_sha256":vector_sha})).unwrap();
            contexts.push(json!({"dataset":family,"family":family,"cut":0,"partition":"meta_train","path":vector.strip_prefix(root).unwrap(),
                "sha256":vector_sha,"prepared":family,"prepared_sha256":prepared_sha}));
            // Misleading directory names must not filter valid row identities.
            let source = root.join(format!("observations/not-{family}/cut-999"));
            let rows=["standardize","robust_scale","norm_row","row_stats","rank","blom","sigmoid","step"].iter().enumerate().map(|(n,t)|{
                let genome=format!("(pipe X ({t}) (ridge))");json!({"dataset":family,"family":family,"cut":0,"partition":"meta_train",
                    "training_sha256":"train","validation_sha256":"validation","genome":genome,"program_sha256":hash(genome.as_bytes()),
                    "metrics":{"rmse":n as f64+1.},"runtime_seconds":0.1,"evaluation_bias":"fixture","failure":null})}).collect::<Vec<_>>();
            write_json(&source.join("observations.json"), &rows).unwrap();
            write_json(&source.join("search.json"), &json!({})).unwrap();
            observations.push(source);
        }
        let index = root.join("contexts.json");
        write_json(&index,&json!({"schema":"parallel.context_index.v1","qualification":true,"assembly_sha256":"assembly","encodings":contexts})).unwrap();
        let full = root.join("full");
        corpus::build(root, &index, &vocab, &observations, &full, true).unwrap();
        let first = root.join("runs/campaign/cycle-0000/corpus-2");
        let second = root.join("runs/campaign/cycle-0001/corpus-2");
        fs::create_dir_all(first.parent().unwrap()).unwrap();
        fs::create_dir_all(second.parent().unwrap()).unwrap();
        build_corpus_atomic(root, &index, &vocab, &observations, &first, None).unwrap();
        let cache = corpus_cache_path(root, &first).unwrap();
        assert!(cache.is_dir());
        let inventory = || {
            walkdir::WalkDir::new(&cache)
                .into_iter()
                .map(|e| e.unwrap().into_path())
                .filter(|p| p.is_file())
                .map(|p| {
                    let sha = file_hash(&p).unwrap();
                    (p, sha)
                })
                .collect::<BTreeMap<_, _>>()
        };
        let before = inventory();
        assert!(!before.is_empty());
        build_corpus_atomic(root, &index, &vocab, &observations, &second, None).unwrap();
        assert_eq!(before, inventory());
        for name in [
            "records.bin",
            "labels.jsonl",
            "index.json",
            "provenance.json",
            "vocabulary.json",
        ] {
            assert_eq!(
                fs::read(full.join(name)).unwrap(),
                fs::read(first.join(name)).unwrap()
            );
            assert_eq!(
                fs::read(full.join(name)).unwrap(),
                fs::read(second.join(name)).unwrap()
            );
        }
        assert!(!root.join("runs/corpus-progress.json").exists());
        assert!(!second.with_file_name(".corpus-2.staging").exists());
        fs::write(second.join("records.bin"), b"corrupt committed output").unwrap();
        assert!(build_corpus_atomic(root, &index, &vocab, &observations, &second, None).is_err());
        assert_eq!(
            fs::read(second.join("records.bin")).unwrap(),
            b"corrupt committed output"
        );
    }
}

fn export_predictor(
    executable: &Path,
    executable_sha: &str,
    prepared: &Path,
    cut: usize,
    genome: &str,
    output: &Path,
) -> Result<Value> {
    ensure!(
        file_hash(executable)? == executable_sha,
        "pinned predictor evaluator changed"
    );
    let receipt_path = output.with_extension("fit.json");
    if !receipt_path.exists() {
        // The isolated worker writes a failure receipt on timeout or memory failure.
        // Pre-spawn errors are captured by the campaign's export summary below.
        crate::gp::fit_view_isolated(executable, prepared, cut, genome, output)?;
    }
    let receipt: Value = read_json(&receipt_path)?;
    ensure!(
        receipt["schema"] == "parallel.fitted_view_export.v1"
            && receipt["executable_sha256"] == executable_sha
            && receipt["prepared_sha256"] == file_hash(&prepared.join("prepared.json"))?
            && receipt["cut"] == cut
            && receipt["program_sha256"] == hash(genome.as_bytes()),
        "fitted predictor receipt identity mismatch"
    );
    ensure!(
        receipt["passed"] == true,
        "fitted predictor export failed: {}",
        receipt["failure"]
    );
    ensure!(
        file_hash(output)? == receipt["predictor_sha256"],
        "fitted predictor payload changed"
    );
    Ok(
        json!({"passed":true,"predictor":output,"predictor_sha256":receipt["predictor_sha256"],"receipt":receipt_path,"receipt_sha256":file_hash(&receipt_path)?,"executable_sha256":executable_sha}),
    )
}

pub(crate) fn latest(root: &Path, run: &Path) -> Result<PathBuf> {
    let v: Value = read_json(&run.join("latest.json"))?;
    let cp = resolve(
        root,
        Path::new(v["checkpoint"].as_str().context("checkpoint path")?),
    )?;
    ensure!(
        file_hash(&cp.join("state.json"))? == v["state_sha256"],
        "checkpoint receipt changed"
    );
    Ok(cp)
}
fn progress(
    root: &Path,
    phase: &str,
    proposals: (usize, usize),
    gp: (usize, usize),
    train: (usize, usize),
    complete: bool,
    reason: &str,
) -> Result<()> {
    write_json(
        &root.join("runs/cycle-progress.json"),
        &json!({"schema":"parallel.cycle_progress.v1","timestamp":now(),"phase":phase,"proposals":{"completed":proposals.0,"total":proposals.1},"neural_seeded_gp":{"completed":gp.0,"total":gp.1},"retraining":{"step":train.0,"steps":train.1},"complete":complete,"reason":reason,"pilot_started":false,"qualification":true}),
    )
}
use crate::tournament::baseline;

pub(crate) fn verify_cycle_identity(
    root: &Path,
    corpus_path: &Path,
    output: &Path,
    steps: usize,
    cuts: usize,
    cut_start: usize,
    contexts_path: Option<&Path>,
    initialize: Option<&Path>,
    extra_observations: &[PathBuf],
    require_existing: bool,
) -> Result<(PathBuf, String)> {
    for arm in ["neural", "shuffled"] {
        controller::verify_proposal_protocol(
            &output.join(arm),
            require_existing || output.join("complete.json").exists(),
        )?;
    }
    let configured_executable = PathBuf::from(
        std::env::var_os("JOPE_GP_EXECUTABLE")
            .context("JOPE_GP_EXECUTABLE must name the pinned CPU-only predictor evaluator")?,
    );
    ensure!(
        configured_executable.is_absolute(),
        "JOPE_GP_EXECUTABLE must be absolute"
    );
    let executable = resolve(
        root,
        configured_executable
            .strip_prefix(root)
            .context("predictor evaluator must be inside the workspace")?,
    )?;
    let executable_sha = file_hash(&executable)?;
    let evaluator_receipt = output.join("predictor-evaluator.json");
    let evaluator = json!({"schema":"parallel.predictor_evaluator.v1","executable":executable.strip_prefix(root)?,"sha256":executable_sha,"timeout_seconds":30,"memory_limit_gib":8,"threads":1});
    ensure!(
        !require_existing || evaluator_receipt.is_file(),
        "completed cycle evaluator receipt absent"
    );
    if evaluator_receipt.exists() {
        ensure!(
            read_json::<Value>(&evaluator_receipt)? == evaluator,
            "cycle predictor evaluator identity changed"
        );
    } else {
        write_json(&evaluator_receipt, &evaluator)?;
    }
    let run_spec = output.join("cycle.json");
    let mut spec = json!({"schema":"parallel.cycle.v1","corpus_sha256":file_hash(&corpus_path.join("index.json"))?,"steps_each_training_block":steps,"cuts_each_dataset":cuts,"qualification":true,"evaluation_bias":"adaptive_cut_heldout_teacher","final_tests_opened":false});
    controller::bind_proposal_protocol(&mut spec);
    spec["champion_snapshot_protocol"] = "cycle-shared-cut-evaluator-v1".into();
    spec["auxiliary_training_protocol"] = crate::auxiliary_training::PROTOCOL.into();
    spec["auxiliary_training_options"] =
        serde_json::to_value(crate::auxiliary_training::Options::for_cycle(steps))?;
    if cut_start != 0 {
        spec["cut_start"] = cut_start.into();
    }
    if let Some(path) = contexts_path {
        spec["contexts_sha256"] = file_hash(path)?.into();
    }
    if let Some(path) = initialize {
        spec["initialization_sha256"] = file_hash(&path.join("state.json"))?.into();
    }
    if !extra_observations.is_empty() {
        spec["extra_observations"] = json!(extra_observations);
    }
    ensure!(
        !require_existing || run_spec.is_file(),
        "completed cycle specification absent"
    );
    if run_spec.exists() {
        ensure!(
            read_json::<Value>(&run_spec)? == spec,
            "cycle resume specification changed"
        );
    } else {
        write_json(&run_spec, &spec)?;
    }
    if require_existing || output.join("champion-snapshot.json").exists() {
        crate::champions::verify_cycle_snapshot(
            output,
            spec["corpus_sha256"].as_str().unwrap(),
            cuts,
            cut_start,
        )?;
    }
    Ok((executable, executable_sha))
}

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
    use std::os::fd::AsRawFd;
    let _lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(output.join("cycle.lock"))?;
    ensure!(
        unsafe { libc::flock(_lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
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
            8,
            256,
            steps,
            1000,
            train1.exists(),
            None,
            initialize,
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
    let vocab: Vocab = read_json(&corpus_path.join("vocabulary.json"))?;
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
        if let Some(genome) = n["best_program"].as_str() {
            let predictor = output
                .join("predictors")
                .join(&prepared.dataset)
                .join(format!("cut-{}.json", neural.jobs[i].cut));
            let result = export_predictor(
                &executable,
                &executable_sha,
                &root.join(&neural.jobs[i].prepared),
                neural.jobs[i].cut,
                genome,
                &predictor,
            );
            predictor_exports.push(match result {
                Ok(receipt) => receipt,
                Err(error) => json!({"passed":false,"dataset":prepared.dataset,"cut":neural.jobs[i].cut,"predictor":predictor,"prepared_sha256":neural.jobs[i].prepared_sha256,"program_sha256":hash(genome.as_bytes()),"executable_sha256":executable_sha,"failure":format!("{error:#}"),"receipt":predictor.with_extension("fit.json"),"receipt_sha256":file_hash(&predictor.with_extension("fit.json")).ok()}),
            });
        } else {
            predictor_exports.push(json!({"passed":false,"dataset":prepared.dataset,"cut":neural.jobs[i].cut,"failure":"No successful GP program to fit","executable_sha256":executable_sha}));
        }
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
            8,
            256,
            steps,
            1000,
            train2.exists(),
            None,
            Some(&second_initialization),
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

#[cfg(test)]
mod campaign_identity_tests {
    use super::*;

    #[test]
    fn completed_cycle_rejects_changed_inputs_without_rewriting_receipts() {
        struct Environment(Option<std::ffi::OsString>);
        impl Drop for Environment {
            fn drop(&mut self) {
                match &self.0 {
                    Some(v) => std::env::set_var("JOPE_GP_EXECUTABLE", v),
                    None => std::env::remove_var("JOPE_GP_EXECUTABLE"),
                }
            }
        }
        let _environment = Environment(std::env::var_os("JOPE_GP_EXECUTABLE"));
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let evaluator = root.join("evaluator");
        let corpus = root.join("corpus");
        let cycle = root.join("cycle");
        let checkpoint = root.join("checkpoint");
        let contexts = root.join("contexts.json");
        for path in [&corpus, &cycle, &checkpoint] {
            fs::create_dir(path).unwrap();
        }
        for path in [
            &evaluator,
            &contexts,
            &corpus.join("index.json"),
            &checkpoint.join("state.json"),
        ] {
            fs::write(path, b"frozen input").unwrap();
        }
        std::env::set_var("JOPE_GP_EXECUTABLE", &evaluator);
        let snapshot = json!({"schema":"parallel.champion_snapshot.v1",
            "request":{"corpus_sha256":file_hash(&corpus.join("index.json")).unwrap(),"cuts":2,"cut_start":10},
            "evaluator_sha256":"e","vocabulary_sha256":"v","entries":[]});
        write_json(&cycle.join("champion-snapshot.json"), &snapshot).unwrap();
        write_json(
            &cycle.join("champion-snapshot-binding.json"),
            &json!({"sha256":file_hash(&cycle.join("champion-snapshot.json")).unwrap()}),
        )
        .unwrap();
        let observations = vec![root.join("observations")];
        let verify = |existing| {
            verify_cycle_identity(
                root,
                &corpus,
                &cycle,
                300,
                2,
                10,
                Some(&contexts),
                Some(&checkpoint),
                &observations,
                existing,
            )
        };
        verify(false).unwrap();
        let mut proposal = json!({"checkpoint_sha256":"frozen"});
        controller::bind_proposal_protocol(&mut proposal);
        for arm in ["neural", "shuffled"] {
            for name in ["run.json", "complete.json"] {
                write_json(&cycle.join(arm).join(name), &proposal).unwrap();
            }
        }
        fs::write(cycle.join("complete.json"), b"already complete").unwrap();
        let identities = ["cycle.json", "predictor-evaluator.json", "complete.json"]
            .map(|name| (name, fs::read(cycle.join(name)).unwrap()));
        verify(true).unwrap();
        for input in [
            &evaluator,
            &contexts,
            &corpus.join("index.json"),
            &checkpoint.join("state.json"),
        ] {
            fs::write(input, b"changed input").unwrap();
            assert!(verify(true).is_err());
            fs::write(input, b"frozen input").unwrap();
        }
        assert!(verify_cycle_identity(
            root,
            &corpus,
            &cycle,
            301,
            2,
            10,
            Some(&contexts),
            Some(&checkpoint),
            &observations,
            true
        )
        .is_err());
        assert!(verify_cycle_identity(
            root,
            &corpus,
            &cycle,
            300,
            2,
            12,
            Some(&contexts),
            Some(&checkpoint),
            &observations,
            true
        )
        .is_err());
        assert!(verify_cycle_identity(
            root,
            &corpus,
            &cycle,
            300,
            2,
            10,
            Some(&contexts),
            Some(&checkpoint),
            &[],
            true
        )
        .is_err());
        let current: Value = read_json(&cycle.join("cycle.json")).unwrap();
        for field in [
            "sampling_protocol",
            "pure_prediction_protocol",
            "gp_assembly_protocol",
            "proposal_generation_schema",
        ] {
            let mut legacy = current.clone();
            legacy.as_object_mut().unwrap().remove(field);
            write_json(&cycle.join("cycle.json"), &legacy).unwrap();
            let before = fs::read(cycle.join("cycle.json")).unwrap();
            assert!(verify(true)
                .unwrap_err()
                .to_string()
                .contains("cycle resume specification changed"));
            assert_eq!(fs::read(cycle.join("cycle.json")).unwrap(), before);
            write_json(&cycle.join("cycle.json"), &current).unwrap();
        }
        for (name, bytes) in identities {
            assert_eq!(fs::read(cycle.join(name)).unwrap(), bytes);
        }
    }

    #[test]
    fn completed_cycle_rejects_legacy_proposal_before_reusing_any_outputs() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let output = root.join("cycle");
        for arm in ["neural", "shuffled"] {
            write_json(
                &output.join(arm).join("run.json"),
                &json!({"sampling_protocol":"shared-incumbents-and-retained-v3"}),
            )
            .unwrap();
            write_json(&output.join(arm).join("complete.json"), &json!({"world":8})).unwrap();
        }
        fs::write(output.join("complete.json"), b"legacy completion").unwrap();
        for require_existing in [false, true] {
            let error = verify_cycle_identity(
                root,
                &root.join("absent-corpus"),
                &output,
                3000,
                2,
                0,
                None,
                None,
                &[],
                require_existing,
            )
            .unwrap_err();
            assert!(error
                .to_string()
                .contains("proposal prediction protocol changed"));
        }
        assert_eq!(
            fs::read(output.join("complete.json")).unwrap(),
            b"legacy completion"
        );
        assert!(!output.join("cycle.json").exists());
        assert!(!output.join("predictor-evaluator.json").exists());
    }
}
