//! W7-D2: bounded ranks train. CPU fixtures up to the device boundary (default features, no
//! GPU, NCCL or device): a rank whose frozen plan selects training writes `run.json` from the
//! plan alone (partition v2 for three ranks) after checking the plan, its launch record and
//! the admitted NCCL bootstrap against each other; a shared `nccl.id` (the `train_controlled`
//! rendezvous) is refused; a plan's training selection must name its corpus, parent and GPUs.
//! Synthetic identities; not the three-host proof and no training step.
#[path = "support/bounded_fixture.rs"]
mod fixture;
use fixture::*;
use jope_parallel::{
    native_loop::bounded::{
        contract::{BoundedRunContract, ClosureEntry, TrainingSelection},
        controller,
        launch::{self, AdmittedBootstrap},
        training_run::{self, TRAINING_RUN_SCHEMA},
    },
    task_contract::{partition_rows_per_rank, training_partition, training_partition_v2},
};
use std::{fs, path::Path, process::Command};

const CORPUS_INDEX: &[u8] = br#"{"fixture":"corpus index"}"#;
/// The fixture counters start at 60 000 committed updates: the parent is at that step.
const PARENT_STATE: &[u8] = br#"{"fixture":"parent","step":60000}"#;
const V1: &str = "parallel.training_partition.v1";
const V2: &str = "parallel.training_partition.v2";

fn with_training_inputs(c: &mut BoundedRunContract, parent_state: &[u8]) {
    for (path, bytes) in [
        ("corpus/index.json", CORPUS_INDEX),
        ("parent/state.json", parent_state),
    ] {
        c.artifact_closure.push(ClosureEntry {
            path: path.into(),
            sha256: sha(bytes),
            bytes: bytes.len() as u64,
        });
    }
    c.artifact_closure.sort_by(|a, b| a.path.cmp(&b.path));
}
fn selection(partition: &str) -> TrainingSelection {
    TrainingSelection {
        execution_partition: partition.into(),
        microbatch: 256,
        global_batch: 2048,
        corpus: "corpus".into(),
        checkpoint_steps: 1,
        initialize: Some("parent".into()),
    }
}
/// The `n`-rank fixture plan selecting training under `partition`.
fn training_plan(n: usize, partition: &str, k: &Keys) -> BoundedRunContract {
    let mut c = contract(now_ms(), n, k);
    with_training_inputs(&mut c, PARENT_STATE);
    c.training = Some(selection(partition));
    c.validate_structure().unwrap();
    c
}
fn write_inputs(root: &Path, parent_state: &[u8]) {
    write_closure(root);
    for (path, bytes) in [
        ("corpus/index.json", CORPUS_INDEX),
        ("parent/state.json", parent_state),
    ] {
        fs::create_dir_all(root.join(path).parent().unwrap()).unwrap();
        fs::write(root.join(path), bytes).unwrap();
    }
}
/// Every rank's admitted bootstrap from one full-group admission carrying a 128-byte NCCL id.
fn bootstraps(c: &BoundedRunContract, k: &Keys) -> Vec<AdmittedBootstrap> {
    let plan = sha(&bytes(c));
    let (_, _, _, members) = admit_group(c, k, Some(hex::encode([7u8; 128])));
    (0..c.participants.len() as u32)
        .map(|r| {
            let record = launch::launch_record(c, &plan, r).unwrap();
            launch::nccl_bootstrap(c, &plan, &record, &members[r as usize]).unwrap()
        })
        .collect()
}
fn record(c: &BoundedRunContract, rank: u32) -> launch::LaunchRecord {
    launch::launch_record(c, &sha(&bytes(c)), rank).unwrap()
}
fn run_dir(root: &Path, c: &BoundedRunContract) -> std::path::PathBuf {
    root.join(training_run::RUNS_DIRECTORY)
        .join(&c.run)
        .join(format!("attempt-{}", c.attempt))
}

#[test]
fn a_bounded_rank_writes_the_v2_run_json_from_its_plan() {
    let k = keys(3);
    let c = training_plan(3, V2, &k);
    let plan = sha(&bytes(&c));
    let boots = bootstraps(&c, &k);
    let (_t, root) = tempdir();
    write_inputs(&root, PARENT_STATE);
    let mut launches = std::collections::BTreeSet::new();
    for rank in 0..3u32 {
        let prepared =
            training_run::prepare(&c, &plan, &record(&c, rank), &boots[rank as usize], &root)
                .unwrap();
        assert_eq!(
            (
                prepared.global_rank,
                prepared.world,
                prepared.microbatch,
                prepared.global_batch,
                prepared.steps,
                prepared.checkpoint_steps
            ),
            (rank as usize, 3, 256, 2048, 3, 1)
        );
        assert_eq!(prepared.run_dir, run_dir(&root, &c));
        assert!(
            prepared.launch_dir.is_dir()
                && fs::read_dir(&prepared.launch_dir).unwrap().next().is_none()
        );
        assert!(launches.insert(prepared.launch_dir.clone()));
        assert_eq!(prepared.corpus, Path::new("corpus"));
        assert_eq!(prepared.initialize.as_deref(), Some(Path::new("parent")));
    }
    // One run.json per host run directory, written from the plan alone; every rank on the
    // host writes the same bytes.
    let spec = training_run::training_run_spec(&c, &plan).unwrap();
    let written = fs::read(run_dir(&root, &c).join("run.json")).unwrap();
    assert_eq!(written, serde_json::to_vec(&spec).unwrap());
    assert_eq!(spec["schema"], TRAINING_RUN_SCHEMA);
    assert_eq!(spec["plan_sha256"], plan);
    assert_eq!(
        (spec["run"].clone(), spec["attempt"].clone()),
        (c.run.clone().into(), 1.into())
    );
    assert_eq!(spec["world"], 3);
    assert_eq!(
        spec["execution_partition"],
        training_partition_v2(3, 256, 2048).unwrap()
    );
    assert_eq!(
        partition_rows_per_rank(&spec["execution_partition"]).unwrap(),
        vec![688, 680, 680]
    );
    assert_eq!(spec["corpus_sha256"], sha(CORPUS_INDEX));
    assert_eq!(spec["initialization_sha256"], sha(PARENT_STATE));
    assert_eq!(
        spec["step_window"],
        serde_json::json!({"initial_optimizer_step": 60_000, "requested_step_increment": 3, "target_optimizer_step": 60_003})
    );
    assert_eq!(spec["checkpoint_steps"], 1);
    assert_eq!(spec["deadline_unix_ms"], c.original_deadline_unix_ms);
    assert_eq!(spec["communicator"], "admitted_bootstrap");
    assert_eq!(spec["assignment"], "frozen_plan");
    assert_eq!(spec["hosts"].as_object().unwrap().len(), 3);
    // A two-rank plan may select v1; its run.json carries the v1 record.
    let k2 = keys(2);
    let two = training_plan(2, V1, &k2);
    assert_eq!(
        training_run::training_run_spec(&two, &sha(&bytes(&two))).unwrap()["execution_partition"],
        training_partition(2, 256, 2048).unwrap()
    );
    // A plan that selects no training writes nothing, even with its own admitted bootstrap.
    let plain = contract(now_ms(), 3, &k);
    let (_t2, other) = tempdir();
    let e = training_run::prepare(
        &plain,
        &sha(&bytes(&plain)),
        &record(&plain, 0),
        &bootstraps(&plain, &k)[0],
        &other,
    )
    .unwrap_err();
    assert!(!other.join(training_run::RUNS_DIRECTORY).exists());
    assert!(format!("{e:#}").contains("selects no training"), "{e:#}");
    let e = training_run::training_run_spec(&plain, &sha(&bytes(&plain))).unwrap_err();
    assert!(e.to_string().contains("selects no training"), "{e}");
}

#[test]
fn the_plan_and_the_admitted_bootstrap_must_agree() {
    let k = keys(3);
    let c = training_plan(3, V2, &k);
    let plan = sha(&bytes(&c));
    let boots = bootstraps(&c, &k);
    let (_t, root) = tempdir();
    write_inputs(&root, PARENT_STATE);
    // Another rank's bootstrap.
    let e = training_run::prepare(&c, &plan, &record(&c, 1), &boots[0], &root).unwrap_err();
    assert!(
        e.to_string()
            .contains("the admitted bootstrap and the plan disagree"),
        "{e}"
    );
    // Another plan's bootstrap for the same rank.
    let mut other = training_plan(3, V2, &k);
    other.run = "w7-another-run".into();
    resign(&mut other, &k);
    let foreign = bootstraps(&other, &k);
    let e = training_run::prepare(&c, &plan, &record(&c, 1), &foreign[1], &root).unwrap_err();
    assert!(
        e.to_string()
            .contains("the admitted bootstrap and the plan disagree"),
        "{e}"
    );
    // A launch record the plan does not derive.
    let mut forged = record(&c, 1);
    forged.world_size = 2;
    assert!(training_run::prepare(&c, &plan, &forged, &boots[1], &root).is_err());
    // Nothing was written.
    assert!(!root.join(training_run::RUNS_DIRECTORY).exists());
    // Positive control.
    training_run::prepare(&c, &plan, &record(&c, 1), &boots[1], &root).unwrap();
}

#[test]
fn a_train_controlled_style_shared_nccl_id_is_refused() {
    let k = keys(3);
    let c = training_plan(3, V2, &k);
    let plan = sha(&bytes(&c));
    let boots = bootstraps(&c, &k);
    let (_t, root) = tempdir();
    write_inputs(&root, PARENT_STATE);
    let run = run_dir(&root, &c);
    // A launch directory holding the shared rendezvous file of the single-host launcher.
    let launch_dir = run.join(format!("launch-{}", record(&c, 0).dispatch_id));
    fs::create_dir_all(&launch_dir).unwrap();
    fs::write(launch_dir.join("nccl.id"), [0u8; 128]).unwrap();
    let e = training_run::prepare(&c, &plan, &record(&c, 0), &boots[0], &root).unwrap_err();
    assert!(
        e.to_string()
            .contains("a shared nccl.id (the train_controlled rendezvous)"),
        "{e}"
    );
    // A shared nccl.id in the run directory itself.
    fs::write(run.join("nccl.id"), [0u8; 128]).unwrap();
    let e = training_run::prepare(&c, &plan, &record(&c, 1), &boots[1], &root).unwrap_err();
    assert!(
        e.to_string()
            .contains("a shared nccl.id in the bounded run directory"),
        "{e}"
    );
    fs::remove_file(run.join("nccl.id")).unwrap();
    // A launch directory is never reused, even without an nccl.id.
    training_run::prepare(&c, &plan, &record(&c, 1), &boots[1], &root).unwrap();
    let e = training_run::prepare(&c, &plan, &record(&c, 1), &boots[1], &root).unwrap_err();
    assert!(
        e.to_string().contains("never reuses a launch directory"),
        "{e}"
    );
    // Another plan's run.json in this run's directory.
    let (_t2, root2) = tempdir();
    write_inputs(&root2, PARENT_STATE);
    fs::create_dir_all(run_dir(&root2, &c)).unwrap();
    fs::write(
        run_dir(&root2, &c).join("run.json"),
        b"{\"schema\":\"parallel.training_run.v1\"}",
    )
    .unwrap();
    let e = training_run::prepare(&c, &plan, &record(&c, 2), &boots[2], &root2).unwrap_err();
    assert!(
        format!("{e:#}").contains("another plan's run.json"),
        "{e:#}"
    );
}

#[test]
fn a_rank_trains_only_from_the_planned_input_bytes() {
    let k = keys(3);
    let c = training_plan(3, V2, &k);
    let plan = sha(&bytes(&c));
    let boots = bootstraps(&c, &k);
    // A corpus index that is not the planned bytes, and a missing one.
    let (_t, root) = tempdir();
    write_inputs(&root, PARENT_STATE);
    // Same length, other bytes.
    fs::write(
        root.join("corpus/index.json"),
        br#"{"fixture":"corpus INDEX"}"#,
    )
    .unwrap();
    let e = training_run::prepare(&c, &plan, &record(&c, 0), &boots[0], &root).unwrap_err();
    assert!(
        format!("{e:#}").contains("is not the planned bytes"),
        "{e:#}"
    );
    fs::remove_file(root.join("corpus/index.json")).unwrap();
    assert!(training_run::prepare(&c, &plan, &record(&c, 0), &boots[0], &root).is_err());
    // A parent checkpoint that is not at the plan's committed start.
    let early = br#"{"fixture":"parent","step":5}"#;
    let mut c5 = contract(now_ms(), 3, &k);
    with_training_inputs(&mut c5, early);
    c5.training = Some(selection(V2));
    c5.validate_structure().unwrap();
    let (_t5, root5) = tempdir();
    write_inputs(&root5, early);
    let e = training_run::prepare(
        &c5,
        &sha(&bytes(&c5)),
        &record(&c5, 0),
        &bootstraps(&c5, &k)[0],
        &root5,
    )
    .unwrap_err();
    assert!(
        e.to_string()
            .contains("not the plan's committed start 60000"),
        "{e}"
    );
}

#[test]
fn a_training_selection_names_its_corpus_parent_gpus_and_cadence() {
    let k = keys(3);
    let refused = |edit: &dyn Fn(&mut BoundedRunContract), needle: &str| {
        let mut c = training_plan(3, V2, &k);
        edit(&mut c);
        let e = c.validate_structure().unwrap_err();
        assert!(e.to_string().contains(needle), "{e} (expected `{needle}`)");
    };
    refused(
        &|c| {
            c.artifact_closure
                .retain(|e| e.path != Path::new("corpus/index.json"))
        },
        "corpus index",
    );
    refused(
        &|c| {
            c.artifact_closure
                .retain(|e| e.path != Path::new("parent/state.json"))
        },
        "parent checkpoint state",
    );
    refused(
        &|c| c.training.as_mut().unwrap().checkpoint_steps = 0,
        "checkpoint cadence",
    );
    refused(
        &|c| c.training.as_mut().unwrap().checkpoint_steps = 4,
        "checkpoint cadence",
    );
    refused(
        &|c| c.training.as_mut().unwrap().corpus = "../corpus".into(),
        "corpus index",
    );
    refused(
        &|c| c.training.as_mut().unwrap().initialize = None,
        "a fresh training run starts at 0 committed updates",
    );
    // Every rank of a training plan has a GPU.
    let mut cpu = training_plan(3, V2, &k);
    cpu.participants[2].device = None;
    cpu.participants[2].ceilings.gpu_vram_bytes = None;
    for w in cpu.work_census.iter_mut().filter(|w| w.global_rank == 2) {
        w.gpu_ms = 0;
        w.peak_vram_bytes = 0;
    }
    resign(&mut cpu, &k);
    let e = cpu.validate_structure().unwrap_err();
    assert!(e.to_string().contains("trains on a GPU"), "{e}");
    // A fresh start (no parent) is admitted at 0 committed updates.
    let mut fresh = training_plan(3, V2, &k);
    fresh.training.as_mut().unwrap().initialize = None;
    fresh.counters.committed_updates_at_start = 0;
    fresh.validate_structure().unwrap();
}

/// A build without the trainer (default features) refuses; a `neural` build goes on to its
/// job proof instead, so this applies to default builds only.
#[cfg(not(feature = "neural"))]
#[test]
fn a_default_build_participant_refuses_a_training_plan_before_any_work() {
    let k = keys(3);
    let c = training_plan(3, V2, &k);
    let (_t, root) = tempdir();
    let dir = root.join("run");
    let plan = controller::plan(&dir, &bytes(&c), now_ms(), None)
        .unwrap()
        .plan_sha256;
    let launch_dir = root.join("launch");
    fs::create_dir_all(launch_dir.join("outbox")).unwrap();
    fs::write(
        launch_dir.join("launch.json"),
        serde_json::to_vec(&launch::launch_record(&c, &plan, 0).unwrap()).unwrap(),
    )
    .unwrap();
    let p = |p: &Path| p.display().to_string();
    let status = Command::new(env!("CARGO_BIN_EXE_jope-native-loop"))
        .arg("participant")
        .args(["--plan", &p(&dir.join("plan.json"))])
        .args(["--plan-sha256", &plan])
        .args(["--launch-record", &p(&launch_dir.join("launch.json"))])
        .args(["--outbox", &p(&launch_dir.join("outbox"))])
        .args(["--coordinator", "127.0.0.1:9"])
        .args(["--participant-key", &p(&root.join("absent.key"))])
        .args(["--installation", &p(&root.join("absent.json"))])
        .env_clear()
        .status()
        .unwrap();
    assert!(!status.success());
    let refusal = fs::read_to_string(launch_dir.join("outbox/refusal.txt")).unwrap();
    assert!(
        refusal.contains("built without the neural feature"),
        "{refusal}"
    );
}

#[test]
fn consumed_inputs_are_the_prepared_bytes_and_request_fields_cannot_be_substituted() {
    let k = keys(3);
    let c = training_plan(3, V2, &k);
    let plan = sha(&bytes(&c));
    let boots = bootstraps(&c, &k);
    let (_t, root) = tempdir();
    write_inputs(&root, PARENT_STATE);
    let prepared = training_run::prepare(&c, &plan, &record(&c, 0), &boots[0], &root).unwrap();
    fs::write(
        root.join("corpus/index.json"),
        br#"{"fixture":"substituted index"}"#,
    )
    .unwrap();
    fs::write(
        root.join("parent/state.json"),
        br#"{"fixture":"other parent","step":60000}"#,
    )
    .unwrap();
    let (index, state) = prepared.verified_input_bytes().unwrap();
    assert_eq!(index, CORPUS_INDEX);
    assert_eq!(state.unwrap(), PARENT_STATE);
    let mut changed = prepared.clone();
    changed.steps += 1;
    assert!(changed.verified_input_bytes().is_err());
    let mut changed = prepared.clone();
    changed.spec["corpus_sha256"] = sha(b"other").into();
    assert!(changed.verified_input_bytes().is_err());
    let mut changed = prepared.clone();
    changed.input_root = root.join("other");
    assert!(changed.verified_input_bytes().is_err());
    // Fresh preparation against altered bytes refuses; a cached preparation consumes
    // only its originally verified inputs, never the replacement.
    assert!(training_run::prepare(&c, &plan, &record(&c, 1), &boots[1], &root).is_err());
}

#[test]
fn planned_updates_cannot_replace_the_actual_optimizer_completion() {
    let k = keys(3);
    let c = training_plan(3, V2, &k);
    let (_t, root) = tempdir();
    write_inputs(&root, PARENT_STATE);
    let prepared = training_run::prepare(
        &c,
        &sha(&bytes(&c)),
        &record(&c, 0),
        &bootstraps(&c, &k)[0],
        &root,
    )
    .unwrap();
    let actual = serde_json::json!({
        "rank": 0, "world": 3,
        "initial_optimizer_step": c.counters.committed_updates_at_start,
        "executed_steps": c.counters.planned_updates,
        "steps": c.counters.committed_updates_at_start + c.counters.planned_updates,
        "corpus_sha256": sha(CORPUS_INDEX), "parameter_sha256": sha(b"actual parameters")
    });
    prepared.validate_completion(&actual).unwrap();
    for field in [
        "rank",
        "world",
        "initial_optimizer_step",
        "executed_steps",
        "steps",
        "corpus_sha256",
        "parameter_sha256",
    ] {
        let mut missing = actual.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(prepared.validate_completion(&missing).is_err(), "{field}");
        let mut wrong = actual.clone();
        wrong[field] = serde_json::json!(999);
        assert!(prepared.validate_completion(&wrong).is_err(), "{field}");
    }
    let mut fresh_optimizer = actual.clone();
    fresh_optimizer["initial_optimizer_step"] = serde_json::json!(0);
    fresh_optimizer["steps"] = serde_json::json!(c.counters.planned_updates);
    assert!(prepared.validate_completion(&fresh_optimizer).is_err());
}
