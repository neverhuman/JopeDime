//! A bounded, resumable sequence of qualification cycles. Original final tests stay sealed.
#[cfg(any(feature = "neural", test))]
use crate::corpus;
#[cfg(feature = "neural")]
use crate::{manifest::*, telemetry::now};
#[cfg(feature = "neural")]
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
#[cfg(feature = "neural")]
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

// Process-scoped: rank children are supervised and reaped by the controller;
// unrelated GP/encoding services keep their own independent resource budgets.
static ACTIVE_DEADLINE: AtomicU64 = AtomicU64::new(0);

pub(crate) fn check_deadline() -> anyhow::Result<()> {
    let deadline = ACTIVE_DEADLINE.load(Ordering::Relaxed);
    anyhow::ensure!(
        deadline == 0 || crate::telemetry::now() < deadline,
        "fixed campaign deadline reached"
    );
    Ok(())
}

#[cfg(any(feature = "neural", test))]
pub(crate) struct DeadlineGuard(u64);
#[cfg(any(feature = "neural", test))]
impl DeadlineGuard {
    pub(crate) fn new(deadline: u64) -> Self {
        Self(ACTIVE_DEADLINE.swap(deadline, Ordering::Relaxed))
    }
}
#[cfg(any(feature = "neural", test))]
impl Drop for DeadlineGuard {
    fn drop(&mut self) {
        ACTIVE_DEADLINE.store(self.0, Ordering::Relaxed);
    }
}
#[cfg(any(feature = "neural", test))]
use std::collections::BTreeSet;
#[cfg(feature = "neural")]
use std::{
    fs::{self, OpenOptions},
    os::fd::AsRawFd,
    path::Path,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CutStrategy {
    #[default]
    Rotation,
    FixedAnchors,
}
#[cfg(any(feature = "neural", test))]
impl CutStrategy {
    fn start(&self, starts: &[usize], sequence: usize) -> usize {
        match self {
            Self::Rotation => starts[sequence % starts.len()],
            Self::FixedAnchors => starts[0],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub schema: String,
    pub contexts: PathBuf,
    pub initial_corpus: PathBuf,
    pub output: PathBuf,
    pub observations: Vec<PathBuf>,
    pub hours: u64,
    pub steps: usize,
    pub cuts: usize,
    #[serde(default)]
    pub cut_strategy: CutStrategy,
    pub qualification: bool,
    pub required_training_families: usize,
    pub required_validation_families: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Schedule {
    pub spec_sha256: String,
    pub contexts_sha256: String,
    pub initial_corpus_sha256: String,
    pub vocabulary_sha256: String,
    pub executable_sha256: String,
    pub gp_executable_sha256: String,
    pub predictor_executable_sha256: String,
    pub started_at: u64,
    pub stop_admission_at: u64,
    pub deadline: u64,
}
impl Schedule {
    pub fn may_start(&self, timestamp: u64) -> bool {
        timestamp < self.stop_admission_at
    }
    pub fn same_inputs(&self, proposed: &Self) -> bool {
        let mut identity = self.clone();
        identity.started_at = proposed.started_at;
        identity.stop_admission_at = proposed.stop_admission_at;
        identity.deadline = proposed.deadline;
        identity == *proposed
    }
}
#[cfg(any(feature = "neural", test))]
fn cut_starts(index: &corpus::Index, cuts: usize) -> Vec<usize> {
    index
        .contexts
        .iter()
        .filter(|c| c.partition != "meta_test")
        .map(|c| c.cut / cuts * cuts)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
#[cfg(feature = "neural")]
fn path(root: &Path, p: &Path) -> Result<PathBuf> {
    resolve(root, p)
}
#[cfg(feature = "neural")]
fn event(root: &Path, message: &str) -> Result<()> {
    use std::io::Write;
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("runs/events.log"))?;
    writeln!(f, "{} · {}", now(), message)?;
    f.sync_data()?;
    Ok(())
}
#[cfg(feature = "neural")]
fn family_sets(index: &corpus::Index) -> Result<(BTreeSet<String>, BTreeSet<String>)> {
    ensure!(
        index
            .groups
            .iter()
            .all(|g| matches!(g.partition.as_str(), "meta_train" | "meta_validation")),
        "non-training/validation supervision in campaign corpus"
    );
    Ok((
        index
            .groups
            .iter()
            .filter(|g| g.partition == "meta_train")
            .map(|g| g.family.clone())
            .collect(),
        index
            .groups
            .iter()
            .filter(|g| g.partition == "meta_validation")
            .map(|g| g.family.clone())
            .collect(),
    ))
}

#[cfg(feature = "neural")]
fn evaluator_hashes(root: &Path) -> Result<(String, String)> {
    let service: Value = read_json(&root.join("runs/gp-service.json"))?;
    let gp = Path::new(
        service["executable"]
            .as_str()
            .context("GP service executable absent")?,
    );
    let gp = path(
        root,
        gp.strip_prefix(root)
            .context("GP executable outside workspace")?,
    )?;
    let gp_sha = file_hash(&gp)?;
    ensure!(
        service["executable_sha256"] == gp_sha,
        "GP service executable receipt changed"
    );
    let predictor = PathBuf::from(
        std::env::var_os("JOPE_GP_EXECUTABLE").context("pinned predictor executable absent")?,
    );
    let predictor = path(
        root,
        predictor
            .strip_prefix(root)
            .context("predictor executable outside workspace")?,
    )?;
    Ok((gp_sha, file_hash(&predictor)?))
}

#[cfg(feature = "neural")]
pub fn run(root: &Path, spec_path: &Path) -> Result<()> {
    crate::task_contract::require_original_gpu_grant("learning-campaign")?;
    run_admitted(root, spec_path)
}

#[cfg(feature = "neural")]
fn run_admitted(root: &Path, spec_path: &Path) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn restart_does_not_extend_deadline_or_admission() {
        let s = Schedule {
            spec_sha256: "a".into(),
            contexts_sha256: "b".into(),
            initial_corpus_sha256: "c".into(),
            vocabulary_sha256: "d".into(),
            executable_sha256: "e".into(),
            gp_executable_sha256: "g".into(),
            predictor_executable_sha256: "p".into(),
            started_at: 100,
            stop_admission_at: 200,
            deadline: 300,
        };
        let restored: Schedule = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert!(restored.may_start(199));
        assert!(!restored.may_start(200));
        assert!(!restored.may_start(1000));
        assert_eq!(s, restored);
        let mut restart = s.clone();
        restart.started_at = 900;
        restart.stop_admission_at = 1000;
        restart.deadline = 1100;
        assert!(s.same_inputs(&restart));
        for field in 0..7 {
            let mut changed = restart.clone();
            match field {
                0 => changed.spec_sha256.push('x'),
                1 => changed.contexts_sha256.push('x'),
                2 => changed.initial_corpus_sha256.push('x'),
                3 => changed.vocabulary_sha256.push('x'),
                4 => changed.executable_sha256.push('x'),
                5 => changed.gp_executable_sha256.push('x'),
                _ => changed.predictor_executable_sha256.push('x'),
            }
            assert!(!s.same_inputs(&changed));
        }
    }
    #[cfg(feature = "neural")]
    #[test]
    fn expired_campaign_exits_successfully_without_work_or_restart_loop() {
        struct Environment(Option<std::ffi::OsString>);
        impl Drop for Environment {
            fn drop(&mut self) {
                match &self.0 {
                    Some(value) => std::env::set_var("JOPE_GP_EXECUTABLE", value),
                    None => std::env::remove_var("JOPE_GP_EXECUTABLE"),
                }
            }
        }
        let _environment = Environment(std::env::var_os("JOPE_GP_EXECUTABLE"));
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let initial = root.join("corpus");
        let output = root.join("runs/campaign");
        fs::create_dir_all(&initial).unwrap();
        fs::create_dir_all(&output).unwrap();
        let evaluator = root.join("evaluator");
        fs::write(&evaluator, b"pinned CPU evaluator fixture").unwrap();
        std::env::set_var("JOPE_GP_EXECUTABLE", &evaluator);
        write_json(
            &root.join("runs/gp-service.json"),
            &json!({"executable":evaluator,"executable_sha256":file_hash(&evaluator).unwrap()}),
        )
        .unwrap();
        let contexts = root.join("contexts.json");
        fs::write(&contexts, b"{}\n").unwrap();
        fs::write(initial.join("vocabulary.json"), b"{}\n").unwrap();
        fs::write(
            initial.join("records.bin"),
            vec![0u8; 2 * corpus::RECORD_BYTES],
        )
        .unwrap();
        fs::write(initial.join("labels.jsonl"), b"{}\n{}\n").unwrap();
        let context = |family: &str, partition: &str| corpus::ContextRow {
            dataset: family.into(),
            family: family.into(),
            cut: 0,
            partition: partition.into(),
            path: "unused".into(),
            sha256: "unused".into(),
            prepared: "unused".into(),
            prepared_sha256: "unused".into(),
        };
        let index = corpus::Index {
            schema: "parallel.teaching_corpus.v1".into(),
            qualification: true,
            vocabulary_sha256: file_hash(&initial.join("vocabulary.json")).unwrap(),
            context_index_sha256: file_hash(&contexts).unwrap(),
            records_sha256: file_hash(&initial.join("records.bin")).unwrap(),
            record_bytes: corpus::RECORD_BYTES,
            records: 2,
            legal_sets: vec![vec![0]],
            contexts: vec![
                context("train", "meta_train"),
                context("validation", "meta_validation"),
            ],
            labels_sha256: None,
            groups: vec![
                corpus::Group {
                    context: 0,
                    family: "train".into(),
                    partition: "meta_train".into(),
                    records: vec![0],
                    max_length: 1,
                },
                corpus::Group {
                    context: 1,
                    family: "validation".into(),
                    partition: "meta_validation".into(),
                    records: vec![1],
                    max_length: 1,
                },
            ],
        };
        write_json(&initial.join("index.json"), &index).unwrap();
        write_json(&initial.join("provenance.json"),&json!({"sources":[],"index_sha256":file_hash(&initial.join("index.json")).unwrap(),"labels_sha256":file_hash(&initial.join("labels.jsonl")).unwrap()})).unwrap();
        let spec = Spec {
            schema: "parallel.campaign.v1".into(),
            contexts: "contexts.json".into(),
            initial_corpus: "corpus".into(),
            output: "runs/campaign".into(),
            observations: vec![],
            hours: 3,
            steps: 1,
            cuts: 1,
            cut_strategy: CutStrategy::Rotation,
            qualification: true,
            required_training_families: 1,
            required_validation_families: 1,
        };
        let spec_path = root.join("spec.json");
        write_json(&spec_path, &spec).unwrap();
        let started = now() - 20000;
        let schedule = Schedule {
            spec_sha256: file_hash(&spec_path).unwrap(),
            contexts_sha256: file_hash(&contexts).unwrap(),
            initial_corpus_sha256: file_hash(&initial.join("index.json")).unwrap(),
            vocabulary_sha256: index.vocabulary_sha256.clone(),
            executable_sha256: file_hash(&std::env::current_exe().unwrap()).unwrap(),
            gp_executable_sha256: file_hash(&evaluator).unwrap(),
            predictor_executable_sha256: file_hash(&evaluator).unwrap(),
            started_at: started,
            stop_admission_at: started + 3600,
            deadline: started + 10800,
        };
        write_json(&output.join("schedule.json"), &schedule).unwrap();
        // The shipped entry must refuse even this expired, nonnumerical request.
        assert!(run(root, &spec_path)
            .unwrap_err()
            .to_string()
            .contains("original native GPU grant verifier unavailable"));
        assert!(!output.join("complete.json").exists());
        run_admitted(root, &spec_path).unwrap();
        let completed: Value = read_json(&output.join("complete.json")).unwrap();
        assert_eq!(completed["status"], "complete_expired");
        assert_eq!(completed["dl_gp_dl_complete"], false);
        assert!(!output.join("cycle-0000").exists());
        let receipt = fs::read(output.join("complete.json")).unwrap();
        run_admitted(root, &spec_path).unwrap();
        assert_eq!(fs::read(output.join("complete.json")).unwrap(), receipt);
        check_deadline().unwrap();
    }

    #[test]
    fn ranges_ignore_test_only_cuts_and_keep_sparse_grouped_indices() {
        let row = |cut, partition: &str| corpus::ContextRow {
            dataset: "d".into(),
            family: "f".into(),
            cut,
            partition: partition.into(),
            path: "p".into(),
            sha256: "s".into(),
            prepared: "p".into(),
            prepared_sha256: "s".into(),
        };
        let index = corpus::Index {
            schema: "s".into(),
            qualification: true,
            vocabulary_sha256: "s".into(),
            context_index_sha256: "s".into(),
            records_sha256: "s".into(),
            record_bytes: 0,
            records: 0,
            legal_sets: vec![],
            contexts: vec![
                row(0, "meta_train"),
                row(1, "meta_train"),
                row(7, "meta_validation"),
                row(8, "meta_test"),
            ],
            labels_sha256: None,
            groups: vec![],
        };
        assert_eq!(cut_starts(&index, 2), vec![0, 6]);
        let starts = cut_starts(&index, 2);
        assert_eq!(CutStrategy::Rotation.start(&starts, 1), 6);
        assert_eq!(CutStrategy::FixedAnchors.start(&starts, 1), 0);
        assert_eq!(CutStrategy::FixedAnchors.start(&starts, 999), 0);
    }
}
