#![cfg(feature = "neural")]
//! W6-T07: accepted ancestry and exposure reconciliation (CPU only).
//!
//! Real checkpoints are committed through `checkpoint_commit` with real
//! consumption journals and a tiny CPU model:
//! - lineage A: cold start, commits at 2 and 4, one applied but uncommitted
//!   update lost to an interruption, then an exact resume that commits 6;
//! - lineage B: optimizer-carry transition from A@4 on a new corpus,
//!   committing 6 and 8;
//! - lineage C: weight warm start (optimizer reset) from A@6, committing 2.
//!
//! This proves the reconstruction logic on retained objects, not any
//! historical run: no historical checkpoint is read here.

use jope_parallel::{
    manifest::{file_hash, hash, read_json, write_json},
    neural::{AdamW, Config},
    task_contract::training_partition,
    training::{
        ancestry::{self, Link},
        checkpoint_commit,
        consumption::{self, Header, Journal},
        objective::{self, RowLabels},
    },
    training_exposure::Counts,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};
use tch::{nn, Device, Kind, Tensor};

const GLOBAL: usize = 8;
const SEED: u64 = 5;
const CORPUS_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const CORPUS_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

struct Run {
    root: PathBuf,
    run: PathBuf,
    launch: PathBuf,
    vs: nn::VarStore,
    w: Tensor,
    optimizer: AdamW,
    journal: Journal,
    initialization: Value,
    corpus: &'static str,
    offset: u64,
}

fn open(
    root: &Path,
    name: &str,
    attempt: usize,
    corpus: &'static str,
    initialization: Value,
    weights: Option<&Path>,
    optimizer: Option<&Path>,
    resume: Option<String>,
) -> Run {
    tch::set_num_threads(1);
    let run = root.join("runs").join(name);
    let launch = run.join(format!("launch-{attempt}"));
    fs::create_dir_all(&launch).unwrap();
    let mut vs = nn::VarStore::new(Device::Cpu);
    let w = vs.root().var("w", &[4], nn::Init::Const(0.5));
    if let Some(dir) = weights {
        vs.load(dir.join("model.ot")).unwrap();
    }
    let optimizer = match optimizer {
        Some(dir) => AdamW::load(&dir.join("optimizer.ot"), Device::Cpu).unwrap(),
        None => AdamW::new(1e-2, 0.01),
    };
    let offset = initialization["optimizer_parent_step"]
        .as_u64()
        .unwrap_or(0);
    let journal = Journal::create(
        &launch.join("consumption-rank-0.jsonl"),
        Header {
            schema: consumption::SCHEMA.into(),
            rank: 0,
            world: 1,
            microbatch: GLOBAL,
            global_batch: GLOBAL,
            accumulation: 1,
            seed: SEED,
            sampler_id: "fixture".into(),
            rng: consumption::RNG_ID.into(),
            sampling_offset: offset,
            initial_optimizer_step: optimizer.step,
            target_optimizer_step: 100,
            corpus_sha256: corpus.into(),
            replay_policy_sha256: None,
            resume_state_sha256: resume,
        },
    )
    .unwrap();
    write_json(
        &launch.join("training-inputs.json"),
        &json!({"schema":"fixture","availability":{"run":name}}),
    )
    .unwrap();
    Run {
        root: root.to_path_buf(),
        run,
        launch,
        vs,
        w,
        optimizer,
        journal,
        initialization,
        corpus,
        offset,
    }
}

fn fresh(root: &Path, name: &str) -> Run {
    open(
        root,
        name,
        1,
        CORPUS_A,
        json!({"kind":"fresh","continuation_mode":"cold_initialization",
            "exact_same_experiment_resume":false}),
        None,
        None,
        None,
    )
}

fn resume(root: &Path, name: &str, attempt: usize, checkpoint: &Path) -> Run {
    let state: Value = read_json(&checkpoint.join("state.json")).unwrap();
    let corpus = if state["corpus_sha256"] == CORPUS_A {
        CORPUS_A
    } else {
        CORPUS_B
    };
    open(
        root,
        name,
        attempt,
        corpus,
        state["initialization"].clone(),
        Some(checkpoint),
        Some(checkpoint),
        Some(file_hash(&checkpoint.join("state.json")).unwrap()),
    )
}

fn warm(root: &Path, name: &str, parent: &Path, carry: bool, corpus: &'static str) -> Run {
    let state: Value = read_json(&parent.join("state.json")).unwrap();
    let step = state["step"].as_u64().unwrap();
    let mut init = json!({"kind":"warm_start","state_sha256":file_hash(&parent.join("state.json")).unwrap(),
        "parent_step":step,"exact_same_experiment_resume":false,
        "continuation_mode":if carry {"new_corpus_transition"} else {"weight_warm_start"}});
    if carry {
        init["optimizer_parent_step"] = json!(step);
    }
    open(
        root,
        name,
        1,
        corpus,
        init,
        Some(parent),
        carry.then_some(parent),
        None,
    )
}

impl Run {
    fn train_to(&mut self, step: u64) {
        for s in self.optimizer.step..step {
            let local = s - self.offset;
            let serial = consumption::sampler_serial(local, GLOBAL, 1, 0, 0, GLOBAL).unwrap();
            let group = ((serial / 8) % 8) as usize;
            let ids = (group * 8..group * 8 + 8).collect::<Vec<_>>();
            let (rows, lengths): (Vec<_>, Vec<_>) = ids
                .iter()
                .enumerate()
                .map(|(k, &record)| {
                    (
                        RowLabels {
                            record,
                            context: group,
                            value: 1.0 / (1.0 + k as f32),
                            percentile: 1.0 - k as f32 / 7.0,
                            runtime: 0.1,
                        },
                        4 + k % 3,
                    )
                })
                .unzip();
            let target = Tensor::from_slice(&[0.1f32, -0.2, 0.3, 0.0]);
            (&self.w - target).square().sum(Kind::Float).backward();
            let norm = self.w.grad().norm().double_value(&[]);
            self.journal
                .microbatch(
                    s,
                    0,
                    serial,
                    consumption::dropout_seed(SEED, local, 0, 0),
                    &ids,
                    objective::supervision(&rows, &lengths).unwrap(),
                )
                .unwrap();
            self.optimizer.update_reference(&self.vs).unwrap();
            self.journal.applied(s, norm).unwrap();
        }
    }

    /// Commit the current step exactly as `training::checkpoint` does.
    fn commit(&mut self) -> PathBuf {
        let step = self.optimizer.step;
        let binding = self.journal.sync().unwrap();
        let journal = self.journal.path().strip_prefix(&self.root).unwrap();
        let consumption = json!({"path":journal,"binding":binding});
        let values: Vec<f32> = Vec::try_from(self.w.detach()).unwrap();
        let parameter = hash(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let receipt = self.launch.join(format!("checkpoint-{step}-rank-0.json"));
        write_json(
            &receipt,
            &json!({"rank":0,"step":step,"parameter_sha256":parameter,
                "consumption_journal":consumption}),
        )
        .unwrap();
        let receipts = json!([{"rank":0,"path":receipt.strip_prefix(&self.root).unwrap(),
            "sha256":file_hash(&receipt).unwrap()}]);
        let inputs = file_hash(&self.launch.join("training-inputs.json")).unwrap();
        let presentations = (step - self.offset) * GLOBAL as u64;
        let counts = Counts {
            program_presentations: presentations,
            comparison_groups: presentations / 8,
            actual_fresh_program_presentations: 0,
            groups_with_actual_fresh_program: 0,
            designated_fresh_groups: 0,
            designated_broad_groups: presentations / 8,
        };
        let (vs, optimizer, init, corpus) = (
            &self.vs,
            &self.optimizer,
            self.initialization.clone(),
            self.corpus,
        );
        let latest = checkpoint_commit::commit(
            &self.root,
            &self.run,
            step,
            |staging| {
                vs.save(staging.join("model.ot"))?;
                optimizer.save(&staging.join("optimizer.ot"))?;
                write_json(&staging.join("config.json"), &Config::regression(8))
            },
            |staging| {
                let digest = |name: &str| file_hash(&staging.join(name));
                Ok(json!({"schema":"parallel.checkpoint.v1","step":step,"world":1,
                    "global_batch":GLOBAL,"model_sha256":digest("model.ot")?,
                    "optimizer_sha256":digest("optimizer.ot")?,
                    "optimizer_metadata_sha256":digest("optimizer.json")?,
                    "config_sha256":digest("config.json")?,"corpus_sha256":corpus,
                    "vocabulary_sha256":"c".repeat(64),"parameter_sha256":parameter,
                    "seed":SEED,"rng":"fixture","sampler":"fixture","sampler_id":"fixture",
                    "scheduler":{"type":"constant"},"initialization":init,
                    "autocast_cache_lifetime":"scoped_v2","ranking_backward":"dense_pairwise_v2",
                    "runtime_objective":"huber_log1p_seconds_v2",
                    "execution_partition":training_partition(1, GLOBAL, GLOBAL)?,
                    "committed_exposure_by_rank":[{"rank":0,"counts":counts,
                        "consumption_journal":consumption}],
                    "training_inputs_sha256":inputs,
                    "objective":objective::identity(),"objective_sha256":objective::identity_sha256(),
                    "checkpoint_format":checkpoint_commit::FORMAT,
                    "required_objects":checkpoint_commit::required_object_names(),
                    "rank_receipts":receipts}))
            },
            &mut checkpoint_commit::no_stop,
        )
        .unwrap();
        self.journal
            .commit(step, latest["state_sha256"].as_str().unwrap())
            .unwrap();
        self.run.join(format!("checkpoints/step-{step}"))
    }
}

struct Scenario {
    _dir: tempfile::TempDir,
    root: PathBuf,
    a2: PathBuf,
    a4: PathBuf,
    a6: PathBuf,
    b6: PathBuf,
    b8: PathBuf,
    c2: PathBuf,
    a_first: PathBuf,
    a_second: PathBuf,
    b_journal: PathBuf,
    c_journal: PathBuf,
}

fn scenario() -> Scenario {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let mut a = fresh(&root, "a");
    a.train_to(2);
    let a2 = a.commit();
    a.train_to(4);
    let a4 = a.commit();
    a.train_to(5); // applied, never committed: lost to the interruption
    let a_first = a.journal.path().to_path_buf();
    drop(a);
    let mut a = resume(&root, "a", 2, &a4);
    a.train_to(6);
    let a6 = a.commit();
    let a_second = a.journal.path().to_path_buf();
    let mut b = warm(&root, "b", &a4, true, CORPUS_B);
    b.train_to(6);
    let b6 = b.commit();
    b.train_to(8);
    let b8 = b.commit();
    let b_journal = b.journal.path().to_path_buf();
    let mut c = warm(&root, "c", &a6, false, CORPUS_A);
    c.train_to(2);
    let c2 = c.commit();
    let c_journal = c.journal.path().to_path_buf();
    Scenario {
        _dir: dir,
        root,
        a2,
        a4,
        a6,
        b6,
        b8,
        c2,
        a_first,
        a_second,
        b_journal,
        c_journal,
    }
}

fn digest(dir: &Path) -> String {
    file_hash(&dir.join("state.json")).unwrap()
}

impl Scenario {
    fn runs(&self) -> Vec<PathBuf> {
        ["a", "b", "c"]
            .iter()
            .map(|name| self.root.join("runs").join(name))
            .collect()
    }
    fn genealogy(&self) -> ancestry::Genealogy {
        ancestry::build(ancestry::load_runs(&self.runs()).unwrap()).unwrap()
    }
}

/// Independent reading of a journal: records of microbatches whose step is
/// below `below`, keyed by corpus.
fn journal_records(path: &Path, below: u64, corpus: &str) -> BTreeSet<(String, u64)> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|e| e["event"] == "microbatch" && e["step"].as_u64().unwrap() < below)
        .flat_map(|e| {
            e["records"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| (corpus.to_string(), r.as_u64().unwrap()))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn manual(root: &Path, name: &str, state: Value) -> PathBuf {
    let dir = root.join("extra").join(name);
    fs::create_dir_all(&dir).unwrap();
    write_json(&dir.join("state.json"), &state).unwrap();
    dir
}

#[path = "training_ancestry/scenario_tests.rs"]
mod scenario_tests;

#[test]
fn retained_numeric_steps_do_not_supply_versioned_committed_credit() {
    scenario_tests::retained_numeric_steps_do_not_supply_versioned_committed_credit();
}

#[test]
fn the_genealogy_resolves_retained_edges_and_checks_independently() {
    scenario_tests::the_genealogy_resolves_retained_edges_and_checks_independently();
}

#[test]
fn exposure_counts_each_committed_segment_once_and_excludes_siblings() {
    scenario_tests::exposure_counts_each_committed_segment_once_and_excludes_siblings();
}

#[test]
fn unretained_or_journal_less_history_stays_unknown() {
    scenario_tests::unretained_or_journal_less_history_stays_unknown();
}

#[test]
fn unsupported_continuity_claims_and_altered_evidence_are_refused() {
    scenario_tests::unsupported_continuity_claims_and_altered_evidence_are_refused();
}
