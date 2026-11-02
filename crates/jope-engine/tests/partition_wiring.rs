#![cfg(feature = "neural")]
//! W7-T06 wiring: the trainer's pre-device guards and records accept an
//! explicitly selected partition v2 and keep v1 byte-for-byte (CPU only).
//!
//! The reproduction run (`w6-pre-w7t06wire-*`) asked each v1 assumption for a
//! three-rank attempt through the existing API and recorded five refusals:
//! `verify_dir` and `verify_continuation` ("invalid training partition"),
//! the journal cursor ("invalid sampler partition"), the exposure baseline
//! ("differs from committed step window") and the validation panel ("supported
//! world size"). The three-rank checkpoint below is built through the real
//! commit path; the training step itself needs a device and is not run.

use jope_parallel::{
    checkpoint_recovery,
    corpus::{Group, Index},
    manifest::{file_hash, read_json, write_json},
    neural::{AdamW, Config},
    task_contract::{self, training_partition},
    training::{
        checkpoint_commit::{self, commit, load_latest, Expectation, Format},
        consumption::{self, Header, Journal},
        objective::{self, RowLabels},
        partition::{self, denominators, execution, CommittedUpdate, Execution, Plan, Version},
        validation_panel, validation_panel_partitioned,
    },
    training_exposure::{self, Counts},
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tch::{nn, Device, Kind};

const CORPUS: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const VOCABULARY: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const SEED: u64 = 7;

fn header(rank: usize, world: usize, micro: usize, global: usize, accumulation: usize) -> Header {
    Header {
        schema: consumption::SCHEMA.into(),
        rank,
        world,
        microbatch: micro,
        global_batch: global,
        accumulation,
        seed: SEED,
        sampler_id: "s".into(),
        rng: consumption::RNG_ID.into(),
        sampling_offset: 0,
        initial_optimizer_step: 0,
        target_optimizer_step: 10,
        corpus_sha256: CORPUS.into(),
        replay_policy_sha256: None,
        resume_state_sha256: None,
    }
}

/// Comparison `g` is records `8g..8g+8` of context `g`, with distinct values.
fn labels(ids: &[usize]) -> Vec<RowLabels> {
    ids.iter()
        .map(|&id| RowLabels {
            record: id,
            context: id / 8,
            value: 1.0 / (1.0 + (id % 8) as f32),
            percentile: 1.0 - (id % 8) as f32 / 7.0,
            runtime: 0.1,
        })
        .collect()
}

fn supervision(ids: &[usize]) -> objective::Supervision {
    objective::supervision(&labels(ids), &vec![5; ids.len()]).unwrap()
}

/// Journal one microbatch of `slot` with `rows` records at `serial`.
fn record(
    journal: &mut Journal,
    step: u64,
    slot: &partition::Slot,
    serial: u64,
    rows: usize,
) -> anyhow::Result<()> {
    let first = serial as usize;
    let ids = (first..first + rows).collect::<Vec<_>>();
    journal.microbatch(
        step,
        slot.microbatch,
        serial,
        slot.dropout_seed,
        &ids,
        supervision(&ids),
    )
}

/// Record the step-`step` microbatches of `rank` exactly as the trainer
/// does: ids are the global batch's rows at each slot.
fn journal_step(journal: &mut Journal, plan: &Plan, rank: usize, step: u64) {
    for slot in plan.slots(rank, SEED, step).unwrap() {
        record(journal, step, &slot, slot.serial, slot.rows).unwrap();
    }
    journal.applied(step, 0.5).unwrap();
}

fn counts(rows: u64) -> Counts {
    Counts {
        program_presentations: rows,
        comparison_groups: rows / 8,
        actual_fresh_program_presentations: 0,
        groups_with_actual_fresh_program: 0,
        designated_fresh_groups: 0,
        designated_broad_groups: rows / 8,
    }
}

fn validation_index() -> Index {
    let groups = (0..6)
        .map(|i| Group {
            context: i,
            family: format!("family{}", i / 2),
            partition: "meta_validation".into(),
            records: (i * 12..i * 12 + 12).collect(),
            max_length: 8,
        })
        .collect();
    Index {
        schema: "parallel.teaching_corpus.v1".into(),
        qualification: true,
        vocabulary_sha256: "v".into(),
        context_index_sha256: "c".into(),
        records_sha256: "r".into(),
        record_bytes: 532,
        records: 72,
        legal_sets: vec![],
        contexts: vec![],
        labels_sha256: None,
        groups,
    }
}

/// A completed original run of `record`'s topology and a continuation policy
/// for it, as `checkpoint_recovery`'s own fixture builds them.
fn continuation_fixture(root: &Path, record: &Value) -> Value {
    let (world, micro) = (record["world"].clone(), record["microbatch"].clone());
    let initialization = json!({"state_sha256":"a".repeat(64),"kind":"warm_start"});
    let mut state = json!({"schema":"parallel.checkpoint.v1","step":3000,"world":world,
        "global_batch":2048,"initialization":initialization,"corpus_sha256":"b".repeat(64),
        "vocabulary_sha256":"c".repeat(64),"seed":20260907,
        "sampler_id":"family_uniform_champion_plus_seven_quantiles_v2",
        "rng":"counter reset before every microbatch: mix(seed ^ step*65537 ^ rank*257 ^ microbatch); LibTorch CPU/CUDA manual_seed",
        "sampler":"family-uniform best teacher plus seven quantile comparators; next global example = step*2048",
        "ranking_backward":"dense_pairwise_v2","runtime_objective":"huber_log1p_seconds_v2",
        "autocast_cache_lifetime":"scoped_v2"});
    if record["schema"] == partition::SCHEMA {
        state["execution_partition"] = record.clone();
    }
    write_json(&root.join("state.json"), &state).unwrap();
    let parent = file_hash(&root.join("state.json")).unwrap();
    write_json(&root.join("run.json"),&json!({"schema":"parallel.training_run.v1","world":world,
        "microbatch":micro,"global_batch":2048,"initialization_sha256":initialization["state_sha256"],
        "corpus_sha256":state["corpus_sha256"],"sampler_id":state["sampler_id"],
        "executable_sha256":"d".repeat(64)})).unwrap();
    write_json(
        &root.join("complete.json"),
        &json!({"schema":"parallel.training_complete.v1",
        "world":world,"microbatch":micro,"steps":3000,
        "latest":{"step":3000,"state_sha256":parent},"corpus_sha256":state["corpus_sha256"]}),
    )
    .unwrap();
    write_json(&root.join("initialization.json"), &initialization).unwrap();
    rebind_policy(
        root,
        &json!({"schema":checkpoint_recovery::CONTINUATION_SCHEMA,
        "mode":"optimizer_carry_new_window","optimizer":"carry","exact_resume":false,
        "sampling_seed":20260909,"parent_checkpoint_sha256":parent,"corpus_sha256":"e".repeat(64),
        "vocabulary_sha256":"c".repeat(64),"execution_partition":record,
        "requested_step_increment":1000}),
    )
}

fn rebind_policy(root: &Path, policy: &Value) -> Value {
    let mut policy = policy.clone();
    for (key, path) in [
        ("original_run", "run.json"),
        ("original_completion", "complete.json"),
        ("original_initialization", "initialization.json"),
        ("parent_state", "state.json"),
    ] {
        policy[key] = json!({"path":path,"sha256":file_hash(&root.join(path)).unwrap()});
    }
    policy["parent_checkpoint_sha256"] = json!(file_hash(&root.join("state.json")).unwrap());
    policy
}

fn continue_with(root: &Path, policy: &Value, requested: &Value) -> anyhow::Result<Value> {
    checkpoint_recovery::verify_continuation_partition(
        root,
        policy,
        &"e".repeat(64),
        &"c".repeat(64),
        requested,
        1000,
    )
}

struct ThreeRank {
    _dir: tempfile::TempDir,
    root: PathBuf,
    run: PathBuf,
    plan: Plan,
    journals: Vec<PathBuf>,
}

impl ThreeRank {
    fn load(&self) -> anyhow::Result<checkpoint_commit::Committed> {
        load_latest(
            &self.root,
            &self.run,
            &Expectation {
                corpus_sha256: Some(CORPUS),
                vocabulary_sha256: Some(VOCABULARY),
                world: Some(3),
                global_batch: Some(2048),
                require_complete: true,
                lineage_sha256: None,
                intent_sha256: None,
                attempt_id: None,
                window_id: None,
            },
        )
    }

    fn rewrite_state(&self, edit: impl FnOnce(&mut Value)) {
        let path = self.run.join("checkpoints/step-1/state.json");
        let mut state: Value = read_json(&path).unwrap();
        edit(&mut state);
        fs::write(&path, serde_json::to_vec_pretty(&state).unwrap()).unwrap();
        let mut latest: Value = read_json(&self.run.join("latest.json")).unwrap();
        latest["state_sha256"] = json!(file_hash(&path).unwrap());
        write_json(&self.run.join("latest.json"), &latest).unwrap();
    }
}

/// One three-rank partition-v2 update committed through
/// `checkpoint_commit::commit`, with each rank's real consumption journal,
/// receipt and exposure, and the persisted partition state. The journals run
/// `journal_micro`-row microbatches; `edit` changes the manifest before
/// publication.
fn three_rank_checkpoint(
    journal_micro: usize,
    edit: &dyn Fn(&mut Value),
) -> anyhow::Result<ThreeRank> {
    tch::set_num_threads(1);
    let dir = tempfile::tempdir()?;
    let root = dir.path().canonicalize()?;
    let run = root.join("run");
    let launch = run.join("launch-0");
    fs::create_dir_all(&launch)?;
    let plan = Plan::new(3, 2048, 256)?;
    let journal_plan = Plan::new(3, 2048, journal_micro)?;
    let vs = nn::VarStore::new(Device::Cpu);
    let w = vs.root().var("w", &[4], nn::Init::Const(0.5));
    let mut optimizer = AdamW::new(1e-2, 0.01);
    (&w * &w).sum(Kind::Float).backward();
    let gradient = w.grad().copy();
    let norm = gradient.norm().double_value(&[]);
    optimizer.update_reference(&vs)?;
    let parameter = {
        let values: Vec<f32> = Vec::try_from(w.detach())?;
        jope_parallel::manifest::hash(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    };
    let mut journals = Vec::new();
    let mut receipts = Vec::new();
    let mut exposure = Vec::new();
    for rank in 0..3 {
        let shard = journal_plan.shards()[rank].clone();
        let path = launch.join(format!("consumption-rank-{rank}.jsonl"));
        let mut journal = Journal::create_with_shard(
            &path,
            header(rank, 3, journal_micro, 2048, shard.microbatches.len()),
            Some(shard.clone()),
        )?;
        journal_step(&mut journal, &journal_plan, rank, 0);
        let binding = journal.sync()?;
        let consumption = json!({"path":path.strip_prefix(&root)?,"binding":binding});
        let receipt = launch.join(format!("checkpoint-1-rank-{rank}.json"));
        write_json(
            &receipt,
            &json!({"rank":rank,"step":1,"parameter_sha256":parameter,
            "consumption_journal":consumption}),
        )?;
        receipts.push(json!({"rank":rank,"path":receipt.strip_prefix(&root)?,
            "sha256":file_hash(&receipt)?}));
        exposure.push(
            json!({"rank":rank,"counts":counts((shard.groups * 8) as u64),
            "consumption_journal":consumption}),
        );
        journals.push((path, journal));
    }
    let global_labels = labels(&(0..2048).collect::<Vec<_>>());
    let update = CommittedUpdate::new(
        &plan,
        1,
        0,
        &denominators(&global_labels)?,
        [1.25, 0.5, 0.75, 0.125, 0.5],
        &gradient,
        norm,
        &parameter,
    )?;
    let partition_state = plan.state(SEED, 0, update)?;
    let latest = commit(
        &root,
        &run,
        1,
        |staging| {
            vs.save(staging.join("model.ot"))?;
            optimizer.save(&staging.join("optimizer.ot"))?;
            write_json(&staging.join("config.json"), &Config::regression(8))
        },
        |staging| {
            let mut state = json!({"schema":"parallel.checkpoint.v1","step":1,"world":3,
                "global_batch":2048,"model_sha256":file_hash(&staging.join("model.ot"))?,
                "optimizer_sha256":file_hash(&staging.join("optimizer.ot"))?,
                "optimizer_metadata_sha256":file_hash(&staging.join("optimizer.json"))?,
                "config_sha256":file_hash(&staging.join("config.json"))?,"corpus_sha256":CORPUS,
                "vocabulary_sha256":VOCABULARY,"parameter_sha256":parameter,"seed":SEED,
                "rng":"counter reset before every microbatch","sampler":"global comparison serial",
                "sampler_id":"s","scheduler":{"type":"constant","learning_rate":optimizer.lr},
                "initialization":{"kind":"fresh"},"autocast_cache_lifetime":"scoped_v2",
                "ranking_backward":"dense_pairwise_v2","runtime_objective":"huber_log1p_seconds_v2",
                "execution_partition":plan.record(),"committed_exposure_by_rank":exposure,
                "checkpoint_format":checkpoint_commit::FORMAT,
                "required_objects":checkpoint_commit::required_object_names(),
                "objective":partition::objective_identity(),
                "objective_sha256":partition::objective_sha256(),
                "rank_receipts":receipts,"partition_state":partition_state});
            edit(&mut state);
            Ok(state)
        },
        &mut checkpoint_commit::no_stop,
    )?;
    let mut paths = Vec::new();
    for (path, mut journal) in journals {
        journal.commit(1, latest["state_sha256"].as_str().unwrap())?;
        paths.push(path);
    }
    Ok(ThreeRank {
        _dir: dir,
        root,
        run,
        plan,
        journals: paths,
    })
}

#[path = "partition_wiring/checkpoint.rs"]
mod checkpoint;
#[path = "partition_wiring/continuation.rs"]
mod continuation;
#[path = "partition_wiring/partition_tests.rs"]
mod partition_tests;
#[path = "partition_wiring/validation.rs"]
mod validation;
