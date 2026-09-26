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

#[test]
fn the_trainer_gate_keeps_v1_and_admits_only_an_explicit_v2() {
    // v1 is unchanged: world 3 is still refused; the v1 record is the old one.
    let error = execution(Version::V1, 3, 256, 2048, false).unwrap_err();
    assert!(
        error.to_string().contains("invalid training topology"),
        "{error}"
    );
    let v1 = execution(Version::V1, 8, 256, 2048, false).unwrap();
    assert_eq!(*v1.record(), training_partition(8, 256, 2048).unwrap());
    assert_eq!(v1.rows_per_rank(), vec![256; 8]);
    assert_eq!(v1.accumulation(3).unwrap(), 1);
    assert_eq!(v1.shard(3).unwrap(), None);
    assert_eq!(v1.objective_sha256(), objective::identity_sha256());
    assert!(execution(Version::V1, 8, 1024, 8192, true).is_ok());
    // v2: three ranks of 86, 85 and 85 comparisons.
    let v2 = execution(Version::V2, 3, 256, 2048, false).unwrap();
    assert_eq!(*v2.record(), Plan::new(3, 2048, 256).unwrap().record());
    assert_eq!(v2.rows_per_rank(), [688, 680, 680]);
    assert_eq!(
        (0..3)
            .map(|r| v2.accumulation(r).unwrap())
            .collect::<Vec<_>>(),
        [3, 3, 3]
    );
    assert_eq!(v2.shard(2).unwrap().unwrap().microbatches, [32, 32, 21]);
    assert_eq!(v2.objective_sha256(), partition::objective_sha256());
    assert_eq!(v2.record()["rng"], consumption::RNG_ID);
    for (world, micro, global, qualification) in [
        (0, 256, 2048, false),
        (9, 256, 2048, false),
        (3, 512, 2048, false),
        (3, 256, 1000, false),
        (3, 1024, 8192, true),
    ] {
        assert!(
            execution(Version::V2, world, micro, global, qualification).is_err(),
            "v2 {world} {micro} {global} {qualification}"
        );
    }
    // Selection is by the frozen record, checked against the topology.
    let declared = Execution::declared(v2.record(), 3, 256, 2048, false).unwrap();
    assert_eq!(declared.version(), Version::V2);
    assert_eq!(
        Execution::declared(v1.record(), 8, 256, 2048, false)
            .unwrap()
            .version(),
        Version::V1
    );
    assert!(Execution::declared(v2.record(), 4, 256, 2048, false).is_err());
    assert!(Execution::declared(v2.record(), 3, 128, 2048, false).is_err());
    let mut forged = v2.record().clone();
    forged["shards"][0]["groups"] = json!(87);
    assert!(Execution::declared(&forged, 3, 256, 2048, false).is_err());
    assert!(Version::of_record(&json!({"schema":"parallel.training_partition.v3"})).is_err());
    assert!(Version::of_record(&json!({})).is_err());
    // The records are self-describing and recomputable outside the neural build.
    for record in [v1.record(), v2.record()] {
        assert_eq!(task_contract::recompute_partition(record).unwrap(), *record);
    }
    assert_eq!(
        task_contract::partition_rows_per_rank(v2.record()).unwrap(),
        [688, 680, 680]
    );
    assert!(task_contract::partition_rows_per_rank(&forged).is_err());
}

#[test]
fn a_three_rank_journal_records_each_uneven_shard() {
    let dir = tempfile::tempdir().unwrap();
    let plan = Plan::new(3, 2048, 256).unwrap();
    // Negative controls: the v1 equation still refuses an uneven header, and
    // a v2 header must carry exactly its own rank's shard.
    let error = Journal::create(&dir.path().join("v1.jsonl"), header(0, 3, 256, 2048, 3))
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("partition is inconsistent"),
        "{error}"
    );
    assert!(consumption::sampler_serial(0, 2048, 3, 1, 0, 256).is_err());
    for (case, h, shard) in [
        (
            "another rank's shard",
            header(0, 3, 256, 2048, 3),
            plan.shards()[1].clone(),
        ),
        (
            "wrong accumulation",
            header(0, 3, 256, 2048, 2),
            plan.shards()[0].clone(),
        ),
        (
            "another microbatch",
            header(0, 3, 128, 2048, 3),
            plan.shards()[0].clone(),
        ),
    ] {
        let path = dir.path().join(format!("{case}.jsonl"));
        assert!(
            Journal::create_with_shard(&path, h, Some(shard)).is_err(),
            "{case}"
        );
    }
    for rank in 0..3 {
        let shard = plan.shards()[rank].clone();
        let path = dir.path().join(format!("rank-{rank}.jsonl"));
        let mut journal = Journal::create_with_shard(
            &path,
            header(rank, 3, 256, 2048, shard.microbatches.len()),
            Some(shard.clone()),
        )
        .unwrap();
        journal_step(&mut journal, &plan, rank, 0);
        // Step 1: the last microbatch is short (22 or 21 comparisons), so a
        // full 256-row microbatch or another slot's cursor is refused; a
        // refused event leaves the journal usable.
        let slots = plan.slots(rank, SEED, 1).unwrap();
        record(&mut journal, 1, &slots[0], slots[0].serial, slots[0].rows).unwrap();
        record(&mut journal, 1, &slots[1], slots[1].serial, slots[1].rows).unwrap();
        assert!(slots[2].rows < 256);
        assert!(record(&mut journal, 1, &slots[2], slots[2].serial, 256).is_err());
        assert!(record(&mut journal, 1, &slots[2], slots[1].serial, slots[2].rows).is_err());
        record(&mut journal, 1, &slots[2], slots[2].serial, slots[2].rows).unwrap();
        journal.applied(1, 0.5).unwrap();
        journal.commit(2, &"c".repeat(64)).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert!(String::from_utf8_lossy(&bytes)
            .lines()
            .next()
            .unwrap()
            .contains("\"shard\""));
        let reconciled = consumption::reconcile(&bytes, Some(2)).unwrap();
        assert_eq!(reconciled.shard.as_ref(), Some(&shard));
        assert_eq!(
            reconciled.committed.presentations,
            (2 * shard.groups * 8) as u64
        );
        assert_eq!(
            reconciled.next.serial,
            plan.slots(rank, SEED, 2).unwrap()[0].serial,
            "the next cursor is the rank's first slot of the next step"
        );
    }
    // An empty rank: more ranks than comparisons, zero microbatches.
    let wide = Plan::new(12, 88, 16).unwrap();
    let path = dir.path().join("empty.jsonl");
    let mut journal = Journal::create_with_shard(
        &path,
        header(11, 12, 16, 88, 0),
        Some(wide.shards()[11].clone()),
    )
    .unwrap();
    journal.applied(0, 0.0).unwrap();
    journal.commit(1, &"c".repeat(64)).unwrap();
    let reconciled = consumption::reconcile(&fs::read(&path).unwrap(), Some(1)).unwrap();
    assert_eq!(reconciled.committed.presentations, 0);
    assert_eq!(reconciled.applied_updates, 1);
}

#[test]
fn v1_records_are_unchanged() {
    // The v1 journal attempt line is byte-identical to the pre-change bytes.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rank-1.jsonl");
    drop(Journal::create(&path, header(1, 8, 256, 2048, 1)).unwrap());
    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        format!(
            "{{\"event\":\"attempt\",\"header\":{{\"schema\":\"jope.training_consumption.v1\",\
             \"rank\":1,\"world\":8,\"microbatch\":256,\"global_batch\":2048,\"accumulation\":1,\
             \"seed\":7,\"sampler_id\":\"s\",\"rng\":\"mix_seed_local_step_rank_microbatch_v1\",\
             \"sampling_offset\":0,\"initial_optimizer_step\":0,\"target_optimizer_step\":10,\
             \"corpus_sha256\":\"{CORPUS}\",\"replay_policy_sha256\":null,\
             \"resume_state_sha256\":null}}}}\n"
        )
    );
    // The v1 execution record is the literal it always was.
    assert_eq!(
        training_partition(8, 256, 2048).unwrap(),
        json!({"schema":"parallel.training_partition.v1","world":8,"microbatch":256,
            "global_batch":2048,"accumulation":1,"rng":"mix_seed_local_step_rank_microbatch_v1",
            "normalization":"microbatch_weighted_means_rank_mean_v1",
            "autocast":"scoped_bf16_fp32_master_v2"})
    );
    assert_eq!(
        task_contract::partition_rows_per_rank(&training_partition(8, 256, 2048).unwrap()).unwrap(),
        vec![256; 8]
    );
    // The v1 exposure baseline keeps its equal-row rule.
    let state = json!({"step":2,"committed_exposure_by_rank":
        (0..2).map(|r| json!({"rank":r,"counts":counts(2 * 1024)})).collect::<Vec<_>>()});
    assert!(training_exposure::checkpoint_baseline(&state, 1, 2, 1024)
        .unwrap()
        .is_some());
    assert!(training_exposure::checkpoint_baseline(&state, 1, 2, 1000).is_err());
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

#[test]
fn three_rank_exposure_baselines_follow_each_shard() {
    let rows = [688usize, 680, 680];
    let state = json!({"step":2,"committed_exposure_by_rank":
        (0..3).map(|r| json!({"rank":r,"counts":counts(2 * rows[r] as u64)})).collect::<Vec<_>>()});
    for rank in 0..3 {
        assert_eq!(
            training_exposure::checkpoint_baseline_rows(&state, rank, &rows).unwrap(),
            Some(counts(2 * rows[rank] as u64))
        );
        // No single equal split describes these ranks (the reproduced defect).
        assert!(training_exposure::checkpoint_baseline(&state, rank, 3, rows[rank]).is_err());
    }
    assert!(training_exposure::checkpoint_baseline_rows(&state, 0, &[680, 688, 680]).is_err());
    assert!(training_exposure::checkpoint_baseline_rows(&state, 3, &rows).is_err());
    // An empty rank consumes nothing and must commit nothing.
    let mut state = json!({"step":2,"committed_exposure_by_rank":[
        {"rank":0,"counts":counts(16)},{"rank":1,"counts":counts(0)}]});
    assert!(
        training_exposure::checkpoint_baseline_rows(&state, 1, &[8, 0])
            .unwrap()
            .is_some()
    );
    state["committed_exposure_by_rank"][1]["counts"] = json!(counts(8));
    assert!(training_exposure::checkpoint_baseline_rows(&state, 1, &[8, 0]).is_err());
    assert!(training_exposure::checkpoint_baseline_rows(&state, 0, &[0, 0]).is_err());
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

#[test]
fn the_v2_validation_panel_is_the_v1_eight_rank_panel_split_by_the_plan() {
    let index = validation_index();
    let v1_eight = validation_panel(&index, 8, SEED).unwrap().concat();
    assert_eq!(v1_eight.len(), partition::VALIDATION_PANEL_ROWS);
    for (world, micro) in [(1, 256), (3, 256), (3, 64), (5, 128), (8, 256)] {
        let plan = Plan::new(world, partition::VALIDATION_PANEL_ROWS, micro).unwrap();
        let panel = validation_panel_partitioned(&index, &plan, SEED).unwrap();
        assert_eq!(panel.len(), world);
        for (rank, slots) in panel.iter().enumerate() {
            assert_eq!(
                slots.iter().map(Vec::len).collect::<Vec<_>>(),
                plan.slots(rank, SEED, 0)
                    .unwrap()
                    .iter()
                    .map(|s| s.rows)
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            panel.concat().concat(),
            v1_eight,
            "world {world}: the panel does not depend on the rank count"
        );
    }
    // v1 keeps its world set; a panel plan of another size is refused.
    assert!(validation_panel(&index, 3, SEED).is_err());
    assert!(validation_panel_partitioned(&index, &Plan::new(3, 4096, 256).unwrap(), SEED).is_err());
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

#[test]
fn a_three_rank_original_continues_with_its_own_partition() {
    let v2 = Plan::new(3, 2048, 256).unwrap().record();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let policy = continuation_fixture(root, &v2);
    let recovered = continue_with(root, &policy, &v2).unwrap();
    assert_eq!(recovered["execution_partition"], v2);
    assert_eq!(recovered["target_optimizer_step"], 4000);
    assert_eq!(recovered["original_next_global_example"], 3000 * 2048);
    // The v1 entry point cannot express it; any other request is refused.
    assert!(checkpoint_recovery::verify_continuation(
        root,
        &policy,
        &"e".repeat(64),
        &"c".repeat(64),
        3,
        256,
        2048,
        1000
    )
    .is_err());
    for other in [
        Plan::new(3, 2048, 128).unwrap().record(),
        Plan::new(4, 2048, 256).unwrap().record(),
        training_partition(4, 256, 2048).unwrap(),
    ] {
        assert!(continue_with(root, &policy, &other).is_err());
    }
    // A parent whose recorded v2 partition disagrees with its launch topology.
    let mut state: Value = read_json(&root.join("state.json")).unwrap();
    state["execution_partition"]["shards"][0]["groups"] = json!(87);
    write_json(&root.join("state.json"), &state).unwrap();
    let forged = rebind_policy(root, &policy);
    let mut complete: Value = read_json(&root.join("complete.json")).unwrap();
    complete["latest"]["state_sha256"] = forged["parent_checkpoint_sha256"].clone();
    write_json(&root.join("complete.json"), &complete).unwrap();
    let forged = rebind_policy(root, &forged);
    let error = continue_with(root, &forged, &v2).unwrap_err();
    assert!(error.to_string().contains("launch topology"), "{error}");
    // v1 originals recover identically through both entry points.
    let dir = tempfile::tempdir().unwrap();
    let v1 = training_partition(8, 256, 2048).unwrap();
    let policy = continuation_fixture(dir.path(), &v1);
    assert_eq!(
        checkpoint_recovery::verify_continuation(
            dir.path(),
            &policy,
            &"e".repeat(64),
            &"c".repeat(64),
            8,
            256,
            2048,
            1000
        )
        .unwrap(),
        continue_with(dir.path(), &policy, &v1).unwrap()
    );
    assert!(continue_with(dir.path(), &policy, &v2).is_err());
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

#[test]
fn a_three_rank_checkpoint_commits_and_reloads_through_the_real_path() {
    let committed = three_rank_checkpoint(256, &|_| {}).unwrap();
    let loaded = committed.load().unwrap();
    assert_eq!((loaded.step, loaded.format), (1, Format::Complete));
    assert_eq!(loaded.state["execution_partition"], committed.plan.record());
    // Exposure, journals and RNG/sampler state resume per rank.
    let rows =
        task_contract::partition_rows_per_rank(&loaded.state["execution_partition"]).unwrap();
    assert_eq!(rows, [688, 680, 680]);
    let state: partition::State =
        serde_json::from_value(loaded.state["partition_state"].clone()).unwrap();
    let next = state.resume(&committed.plan, SEED).unwrap();
    for rank in 0..3 {
        assert_eq!(
            training_exposure::checkpoint_baseline_rows(&loaded.state, rank, &rows).unwrap(),
            Some(counts(rows[rank] as u64))
        );
        let reconciled =
            consumption::reconcile(&fs::read(&committed.journals[rank]).unwrap(), Some(1)).unwrap();
        assert_eq!(reconciled.committed_through, 1);
        assert_eq!(reconciled.committed.presentations, rows[rank] as u64);
        assert_eq!(reconciled.next.serial, next[rank][0].serial);
        assert_eq!(reconciled.next.dropout_seed, next[rank][0].dropout_seed);
    }
    // The in-commit verification never publishes an incomplete v2 manifest.
    for (case, edit) in [
        (
            "partition state absent",
            Box::new(|s: &mut Value| drop(s.as_object_mut().unwrap().remove("partition_state")))
                as Box<dyn Fn(&mut Value)>,
        ),
        (
            "v1 objective",
            Box::new(|s: &mut Value| {
                s["objective"] = objective::identity();
                s["objective_sha256"] = json!(objective::identity_sha256());
            }),
        ),
        (
            "legacy format",
            Box::new(|s: &mut Value| drop(s.as_object_mut().unwrap().remove("checkpoint_format"))),
        ),
    ] {
        assert!(
            three_rank_checkpoint(256, &*edit).is_err(),
            "{case}: published"
        );
    }
    // Journals of another partition never bind a v2 checkpoint.
    let error = three_rank_checkpoint(128, &|_| {}).err().unwrap();
    assert!(error.to_string().contains("another partition"), "{error:#}");
    // Every published-state alteration is refused on load.
    type Edit = Box<dyn Fn(&mut Value)>;
    let cases: Vec<(&str, Edit)> = vec![
        (
            "partition state removed",
            Box::new(|s| drop(s.as_object_mut().unwrap().remove("partition_state"))),
        ),
        (
            "rank cursor",
            Box::new(|s| s["partition_state"]["ranks"][1]["sampler_cursor"] = json!(1)),
        ),
        (
            "committed parameters",
            Box::new(|s| {
                s["partition_state"]["committed_update"]["parameter_sha256"] = json!("d".repeat(64))
            }),
        ),
        ("partition seed", Box::new(|s| s["seed"] = json!(SEED + 1))),
        (
            "shard record",
            Box::new(|s| s["execution_partition"]["shards"][0]["groups"] = json!(87)),
        ),
        (
            "equal exposure split",
            Box::new(|s| {
                s["committed_exposure_by_rank"][0]["counts"] = json!(counts(680));
                s["committed_exposure_by_rank"][1]["counts"] = json!(counts(688));
            }),
        ),
        (
            "format removed",
            Box::new(|s| drop(s.as_object_mut().unwrap().remove("checkpoint_format"))),
        ),
        (
            "v1 record",
            Box::new(|s| s["execution_partition"] = training_partition(1, 256, 2048).unwrap()),
        ),
    ];
    for (case, edit) in cases {
        let committed = three_rank_checkpoint(256, &|_| {}).unwrap();
        committed.rewrite_state(edit);
        assert!(committed.load().is_err(), "{case}: accepted");
        // Not even a legacy-tolerant load accepts it.
        assert!(
            load_latest(&committed.root, &committed.run, &Expectation::default()).is_err(),
            "{case}: accepted without require_complete"
        );
    }
}
