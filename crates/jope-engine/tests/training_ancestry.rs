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

#[test]
fn the_genealogy_resolves_retained_edges_and_checks_independently() {
    let s = scenario();
    let g = s.genealogy();
    let edge = |dir: &Path| &g.edges[&digest(dir)];
    let resolved = |dir: &Path| Link::Resolved {
        digest: digest(dir),
    };
    assert_eq!(g.nodes.len(), 6);
    let a2 = edge(&s.a2);
    assert_eq!((a2.kind, a2.from_step), ("lineage_start", Some(0)));
    assert_eq!(
        (a2.weights.clone(), a2.optimizer_state),
        (Link::None, "none")
    );
    assert_eq!(edge(&s.a4).weights, resolved(&s.a2));
    assert_eq!(edge(&s.a4).kind, "same_lineage");
    // The exact resume continues lineage A: same lineage record.
    assert_eq!(edge(&s.a6).weights, resolved(&s.a4));
    assert_eq!(edge(&s.a6).from_step, Some(4));
    let b6 = edge(&s.b6);
    assert_eq!(
        (b6.kind, b6.mode.as_str(), b6.from_step),
        ("lineage_start", "new_corpus_transition", Some(4))
    );
    assert_eq!(
        (b6.weights.clone(), b6.optimizer.clone(), b6.optimizer_state),
        (resolved(&s.a4), resolved(&s.a4), "retained")
    );
    assert_eq!(edge(&s.b8).weights, resolved(&s.b6));
    let c2 = edge(&s.c2);
    assert_eq!(
        (c2.weights.clone(), c2.optimizer.clone(), c2.optimizer_state),
        (resolved(&s.a6), Link::None, "reset")
    );
    assert_eq!(c2.from_step, Some(0));
    // Order-independent and idempotent: shuffled, duplicated ingestion
    // describes the same genealogy.
    let mut nodes = ancestry::load_runs(&s.runs()).unwrap();
    nodes.reverse();
    nodes.extend(ancestry::load_runs(&s.runs()).unwrap());
    assert_eq!(
        ancestry::describe(&ancestry::build(nodes).unwrap()),
        ancestry::describe(&g)
    );
    // Every node digest is the independent SHA-256 of its manifest bytes.
    for dir in [&s.a2, &s.a4, &s.a6, &s.b6, &s.b8, &s.c2] {
        let d = hash(&fs::read(dir.join("state.json")).unwrap());
        assert!(g.nodes[&d].weights_retained && g.nodes[&d].optimizer_retained);
    }
}

#[test]
fn exposure_counts_each_committed_segment_once_and_excludes_siblings() {
    let s = scenario();
    let g = s.genealogy();
    let b8 = digest(&s.b8);
    let c2 = digest(&s.c2);
    let branch = ancestry::exposure(&s.root, &g, &[&b8]).unwrap();
    // B8 ← B6 ← A4 ← A2: four segments of two committed updates each. The
    // lost update after A4 is not committed exposure.
    assert_eq!(branch.segments.len(), 4);
    assert_eq!(branch.possible_presentations, 4 * 2 * GLOBAL as u64);
    assert_eq!(branch.exact.presentations, branch.possible_presentations);
    assert!(branch.possible_complete && branch.exact_complete);
    assert!(branch.unresolved_ancestors.is_empty());
    let expected = journal_records(&s.a_first, 4, CORPUS_A)
        .union(&journal_records(&s.b_journal, 8, CORPUS_B))
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(branch.unique_committed_records, expected.len() as u64);
    // A union of branches counts the shared ancestors A2 and A4 once.
    let union = ancestry::exposure(&s.root, &g, &[&b8, &c2]).unwrap();
    assert_eq!(union.segments.len(), 6);
    assert_eq!(union.possible_presentations, 6 * 2 * GLOBAL as u64);
    assert_eq!(union.exact.presentations, union.possible_presentations);
    let expected_union = expected
        .union(&journal_records(&s.a_second, 6, CORPUS_A))
        .cloned()
        .collect::<BTreeSet<_>>()
        .union(&journal_records(&s.c_journal, 2, CORPUS_A))
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(union.unique_committed_records, expected_union.len() as u64);
    // The sibling branch's work never enters B8's ancestry.
    assert!(branch
        .segments
        .iter()
        .all(|seg| seg.child != digest(&s.a6) && seg.child != c2));
    // Availability comes from each segment's own frozen inputs manifest.
    let b6 = branch
        .segments
        .iter()
        .find(|seg| seg.child == digest(&s.b6))
        .unwrap();
    assert_eq!(b6.availability, json!({"run":"b"}));
    assert_eq!(b6.corpus_sha256, CORPUS_B);
}

#[test]
fn unretained_or_journal_less_history_stays_unknown() {
    // A removed middle checkpoint: B's claimed parent is unresolved, and A6
    // now spans a window its journal does not cover.
    let s = scenario();
    let a4 = digest(&s.a4);
    fs::remove_dir_all(&s.a4).unwrap();
    let g = s.genealogy();
    assert_eq!(
        g.edges[&digest(&s.b6)].weights,
        Link::Unresolved {
            digest: Some(a4.clone())
        }
    );
    let b = ancestry::exposure(&s.root, &g, &[&digest(&s.b8)]).unwrap();
    assert_eq!(b.unresolved_ancestors, vec![Some(a4)]);
    assert!(!b.possible_complete && !b.exact_complete);
    // The retained new-corpus segments stay exact; nothing older is assumed.
    assert_eq!(b.segments.len(), 2);
    assert_eq!(b.exact.presentations, 2 * 2 * GLOBAL as u64);
    let a6 = ancestry::exposure(&s.root, &g, &[&digest(&s.a6)]).unwrap();
    let segment = a6
        .segments
        .iter()
        .find(|seg| seg.child == digest(&s.a6))
        .unwrap();
    assert_eq!(segment.from_step, Some(2));
    assert_eq!(segment.possible_presentations, Some(4 * GLOBAL as u64));
    assert_eq!(segment.exact_status, "journal_starts_after_segment");
    assert!(segment.exact.is_none() && !a6.exact_complete && a6.possible_complete);

    // A bound journal that is no longer retained.
    let s = scenario();
    fs::remove_file(&s.a_second).unwrap();
    let g = s.genealogy();
    let a6 = ancestry::exposure(&s.root, &g, &[&digest(&s.a6)]).unwrap();
    let segment = a6
        .segments
        .iter()
        .find(|seg| seg.child == digest(&s.a6))
        .unwrap();
    assert_eq!(segment.exact_status, "journal_not_retained");
    assert!(!a6.exact_complete);

    // A historical checkpoint with no consumption journal: possible only.
    let s = scenario();
    let legacy = manual(
        &s.root,
        "legacy",
        json!({"schema":"parallel.checkpoint.v1","step":3000,"world":8,"global_batch":2048,
            "corpus_sha256":CORPUS_A,"initialization":{"kind":"fresh"}}),
    );
    let g = ancestry::build([ancestry::load_node(&legacy).unwrap()]).unwrap();
    let old = ancestry::exposure(&s.root, &g, &[&digest(&legacy)]).unwrap();
    assert_eq!(old.possible_presentations, 3000 * 2048);
    assert!(old.possible_complete && !old.exact_complete);
    assert_eq!(old.segments[0].exact_status, "no_consumption_journal");
    assert_eq!(old.unique_committed_records, 0);
    assert!(!g.nodes[&digest(&legacy)].weights_retained);

    // A successful new run from a parent that is not retained does not
    // manufacture the missing history.
    let s = scenario();
    let missing = "e".repeat(64);
    let parent = manual(
        &s.root,
        "parent",
        json!({"schema":"parallel.checkpoint.v1","step":60000,"world":1,"global_batch":GLOBAL,
            "corpus_sha256":CORPUS_A,"initialization":{"kind":"fresh"}}),
    );
    let mut child = open(
        &s.root,
        "d",
        1,
        CORPUS_A,
        json!({"kind":"warm_start","state_sha256":missing,"parent_step":60000,
            "exact_same_experiment_resume":false,"continuation_mode":"weight_warm_start"}),
        None,
        None,
        None,
    );
    child.train_to(2);
    let d2 = child.commit();
    let g = ancestry::build(vec![
        ancestry::load_node(&parent).unwrap(),
        ancestry::load_node(&d2).unwrap(),
    ])
    .unwrap();
    let d = ancestry::exposure(&s.root, &g, &[&digest(&d2)]).unwrap();
    assert_eq!(d.unresolved_ancestors, vec![Some(missing)]);
    assert!(!d.exact_complete && !d.possible_complete);
    assert_eq!(d.exact.presentations, 2 * GLOBAL as u64);
    // A manifest with no lineage record is its own orphan of unknown origin.
    let orphan = manual(
        &s.root,
        "orphan",
        json!({"schema":"parallel.checkpoint.v1","step":7,"world":1,"global_batch":GLOBAL,
            "corpus_sha256":CORPUS_A}),
    );
    let g = ancestry::build([ancestry::load_node(&orphan).unwrap()]).unwrap();
    let o = ancestry::exposure(&s.root, &g, &[&digest(&orphan)]).unwrap();
    assert_eq!(o.unresolved_ancestors, vec![None]);
    assert_eq!(o.segments[0].exact_status, "unknown_origin");
    assert!(o.segments[0].possible_presentations.is_none());
}

#[test]
fn unsupported_continuity_claims_and_altered_evidence_are_refused() {
    let s = scenario();
    let nodes = || ancestry::load_runs(&s.runs()).unwrap();
    let a2: Value = read_json(&s.a2.join("state.json")).unwrap();
    let a4 = digest(&s.a4);
    let with = |name: &str, init: Value| {
        let mut state = a2.clone();
        state["step"] = json!(5);
        state["initialization"] = init;
        ancestry::load_node(&manual(&s.root, name, state)).unwrap()
    };
    let cases = vec![
        ("contradictory step", {
            let mut state = a2.clone();
            state["validation"] = json!({"sequence_nll":1.0});
            ancestry::load_node(&manual(&s.root, "twin", state)).unwrap()
        }),
        (
            "wrong parent step",
            with(
                "wrong-step",
                json!({"kind":"warm_start","state_sha256":a4,"parent_step":3}),
            ),
        ),
        (
            "wrong optimizer step",
            with(
                "wrong-optimizer",
                json!({"kind":"warm_start","state_sha256":a4,"parent_step":4,
                    "optimizer_parent_step":5}),
            ),
        ),
        (
            "continuity contradicts parent",
            with(
                "wrong-continuity",
                json!({"kind":"warm_start","state_sha256":a4,"parent_step":4,
                    "continuity":{"parent_step":4,"objective":{"parent_sha256":"f".repeat(64)},
                        "corpus":{"parent":CORPUS_A}}}),
            ),
        ),
        (
            "lineage claims exact resume",
            with(
                "resume-claim",
                json!({"kind":"fresh","exact_same_experiment_resume":true}),
            ),
        ),
    ];
    for (case, node) in cases {
        let mut all = nodes();
        all.push(node);
        assert!(ancestry::build(all).is_err(), "{case}: accepted");
    }
    // Positive control: a consistent claim on the same retained parent.
    let mut all = nodes();
    all.push(with(
        "consistent",
        json!({"kind":"warm_start","state_sha256":a4,"parent_step":4,
            "continuity":{"parent_step":4,"objective":{"parent_sha256":objective::identity_sha256()},
                "corpus":{"parent":CORPUS_A}}}),
    ));
    ancestry::build(all).unwrap();

    // Altered evidence is an error, never an unknown.
    let s = scenario();
    let mut bytes = fs::read(&s.a_first).unwrap();
    bytes[60] ^= 1;
    fs::write(&s.a_first, bytes).unwrap();
    let g = s.genealogy();
    assert!(ancestry::exposure(&s.root, &g, &[&digest(&s.a4)]).is_err());
    // Journal exposure disagreeing with committed steps × batch.
    let s = scenario();
    let mut state: Value = read_json(&s.a2.join("state.json")).unwrap();
    state["global_batch"] = json!(16);
    let altered = manual(&s.root, "batch", state);
    let g = ancestry::build([ancestry::load_node(&altered).unwrap()]).unwrap();
    let error = ancestry::exposure(&s.root, &g, &[&digest(&altered)]).unwrap_err();
    assert!(error.to_string().contains("disagrees"), "{error}");
    // Unknown targets are refused.
    let g = s.genealogy();
    assert!(ancestry::exposure(&s.root, &g, &[&"0".repeat(64)]).is_err());
}
