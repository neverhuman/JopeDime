//! The operator recovery verbs (W3-T06, W8-T04): each is read-only by default and changes
//! anything only with an explicit `--apply`.
//!
//! * `hwm recover` rolls a lagging HWM ledger head and lagging `current.json` pointers forward to
//!   their ledger tips (`hwm::recover`). The plan reports what it would do and writes nothing.
//! * `progress reconcile` reconciles a run publication that lost a race: it adopts the winner's
//!   committed closure, then commits this host's own remaining work on top of it. The plan
//!   reports the winner, the losing commits, the work to re-commit and whether that commit would
//!   pass the receiving checks, and writes nothing.
//! * `revocations import` (W12-T05) imports a verified claims record's revocation closure into this
//!   host's append-only revocation ledger. The plan reports what it would append and writes
//!   nothing.

mod common;

use common::{jl, Run};
use jope_learning::canon::sha256_hex;
use jope_learning::hwm::{self, Candidate};
use jope_learning::ledger::Ledger;
use jope_learning::metric::Metric;
use jope_learning::record::*;
use jope_learning::state::{PointerKey, State};
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const BOUND: &str = "b5ccee8e33af34313c51896f52466f87f2f8544dcb01672e6d6b4832d1ac4393";

/// One candidate for a fixed cut of `dataset`, scored `value` (rmse: lower is better).
fn candidate(dataset: &str, value: f64, ts: u64) -> Candidate {
    let identity = HwmIdentity {
        comparison: None,
        task: "regression".into(),
        dataset_hash: dataset.into(),
        cut_id: sha256_hex(b"cut-0"),
        evaluator_sha256: BOUND.into(),
        vocabulary_sha256: UNBOUND.into(),
        protocol: "test".into(),
    };
    let dsl = format!("(pipe X (ridge alpha={value}))");
    let blob = format!("{dataset}/cut-0/{value}").into_bytes();
    let record = HwmRecord {
        schema: HWM_SCHEMA.into(),
        identity_sha256: identity.sha256().unwrap(),
        identity,
        legacy_identity_sha256: None,
        genome: Genome {
            program_sha256: sha256_hex(dsl.as_bytes()),
            dsl,
            semantic_sha256: None,
        },
        score: Score {
            metric: Metric::Rmse,
            value,
            direction: Metric::Rmse.direction(),
            units: "original_target_units".into(),
        },
        cost: Cost::default(),
        evaluator: Evaluator {
            sha256: BOUND.into(),
            version: "test".into(),
        },
        benchmark_valid: false,
        sealed: false,
        replayable: true,
        provenance: Provenance {
            source: "test".into(),
            run_id: "run".into(),
            git_sha: UNBOUND.into(),
            bundle_id: UNBOUND.into(),
            record_sha256: sha256_hex(&blob),
        },
        prev_event_sha256: None,
        improvement_number: 0,
        ts_unix: ts,
    };
    Candidate {
        record,
        blob,
        chain_from_source: false,
    }
}

fn ingest(state: &State, candidates: Vec<Candidate>) {
    hwm::ingest(state, None, candidates, json!({"test": "recovery-cli"})).unwrap();
}

/// Every file under `root`, relative path to sha256.
fn tree(root: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.insert(
                    path.strip_prefix(root).unwrap().display().to_string(),
                    sha256_hex(&fs::read(&path).unwrap()),
                );
            }
        }
    }
    out
}

#[test]
fn hwm_recover_is_read_only_by_default_and_rolls_forward_only_with_apply() {
    let dir = tempfile::tempdir().unwrap();
    let state = State::at(dir.path());
    let key = PointerKey::of(&candidate("d0", 0.5, 1).record);
    let pointer = state.pointer_path(&key).unwrap();
    let head = Ledger::open(state.root(), "hwm").root().join("HEAD.json");

    // A crash after the batch accepting 0.3 became durable, before its HEAD.json and pointer
    // moved: both lag the ledger's sealed tip.
    ingest(&state, vec![candidate("d0", 0.5, 1)]);
    let (stale_pointer, stale_head) = (fs::read(&pointer).unwrap(), fs::read(&head).unwrap());
    ingest(&state, vec![candidate("d0", 0.3, 2)]);
    fs::write(&pointer, &stale_pointer).unwrap();
    fs::write(&head, &stale_head).unwrap();
    let before = tree(dir.path());

    // The plan: exit 0, the pointer and the ledger head it would roll forward, and nothing written
    // (not even a lock file).
    let plan = jl(dir.path(), &["hwm", "recover"]).ok().json();
    assert_eq!(plan["applied"], false);
    assert_eq!(plan["recovered"], json!([key.rel()]));
    assert!(plan["ledger_head"].is_string(), "{plan}");
    assert_eq!(tree(dir.path()), before, "the plan wrote something");

    // Apply: both roll forward, and the ledger verifies.
    let applied = jl(dir.path(), &["hwm", "recover", "--apply"]).ok().json();
    assert_eq!(applied["applied"], true);
    assert_eq!(applied["recovered"], json!([key.rel()]));
    assert_eq!(applied["ledger_head"], plan["ledger_head"]);
    jl(dir.path(), &["hwm", "verify"]).ok();
    let current: HwmRecord = serde_json::from_slice(&fs::read(&pointer).unwrap()).unwrap();
    assert_eq!(current.score.value, 0.3);
    // Nothing left to do.
    let again = jl(dir.path(), &["hwm", "recover"]).ok().json();
    assert_eq!(again["recovered"], json!([]));
    assert!(again["ledger_head"].is_null());

    // A pointer holding a record the ledger never accepted is reported, exit 1, and left alone
    // by the plan and by apply.
    state
        .write_pointer(
            &key,
            &candidate("d0", 0.1, 3).record,
            Some(&current.provenance.record_sha256),
        )
        .unwrap();
    let forged = fs::read(&pointer).unwrap();
    for args in [&["hwm", "recover"][..], &["hwm", "recover", "--apply"][..]] {
        let run: Run = jl(dir.path(), args);
        assert_eq!(run.status, 1, "{}", run.stderr);
        assert_eq!(run.json()["refused"][0][0], key.rel());
        assert_eq!(fs::read(&pointer).unwrap(), forged);
    }
}

fn store_args<'a>(store: &'a str, args: &[&'a str]) -> Vec<&'a str> {
    let mut out = vec!["--store", store];
    out.extend(args);
    out
}

fn init(state: &Path, store: &str, units: &[&str]) -> Run {
    let mut args = vec![
        "progress",
        "init",
        "--run",
        "run-race",
        "--publisher",
        "xbabe2",
        "--source-commit",
        "c4f60b4c7a2e27a31fc41f875650c89900dc32ed",
    ];
    for unit in units {
        args.extend(["--unit", unit]);
    }
    jl(state, &store_args(store, &args))
}

const RUN: [&str; 4] = ["--run", "run-race", "--publisher", "xbabe2"];

fn verb(state: &Path, store: &str, verb: &str, extra: &[&str]) -> Run {
    let mut args = vec!["progress", verb];
    args.extend(RUN);
    args.extend(extra);
    jl(state, &store_args(store, &args))
}

#[test]
fn progress_reconcile_is_read_only_by_default_and_rebases_only_with_apply() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = (dir.path().join("host-a"), dir.path().join("host-b"));
    let store_dir = dir.path().join("store");
    fs::create_dir_all(&store_dir).unwrap();
    let store = store_dir.to_str().unwrap();

    // Two hosts freeze the same census; A publishes first. B's publisher then finds the run
    // committed at a commit it never recorded, and every flush refuses, writing nothing.
    let units = ["train:unit-a", "train:unit-b"];
    let out = init(&a, store, &units).ok().json();
    init(&b, store, &units).ok();
    let (unit_a, unit_b) = (
        out["units"][0]["work_id"].as_str().unwrap().to_string(),
        out["units"][1]["work_id"].as_str().unwrap().to_string(),
    );
    verb(&a, store, "schedule", &["--unit", &unit_a]).ok();
    let won = verb(&a, store, "flush", &[]).ok().json();
    assert_eq!(won["step"]["publication"]["outcome"], "advanced");
    verb(&b, store, "schedule", &["--unit", &unit_b]).ok();
    for _ in 0..2 {
        let stuck = verb(&b, store, "flush", &[]);
        assert_eq!(stuck.status, 2);
        assert!(
            stuck.stderr.contains("progress reconcile"),
            "{}",
            stuck.stderr
        );
    }

    // The plan is read-only: the store and B's state are byte-identical afterwards.
    let (store_before, b_before) = (tree(&store_dir), tree(&b));
    let plan = verb(&b, store, "reconcile", &[]).ok().json();
    assert_eq!(plan["admissible"], true, "{plan}");
    assert_eq!(plan["head"]["height"], 0);
    assert_eq!(plan["divergent"], json!([]));
    assert_eq!(plan["recommit"].as_array().unwrap().len(), 1);
    assert!(!plan["adopt"].as_array().unwrap().is_empty());
    assert!(plan["applied"].is_null());
    assert_eq!(
        tree(&store_dir),
        store_before,
        "the plan wrote to the store"
    );
    assert_eq!(tree(&b), b_before, "the plan wrote to local state");

    // Apply: B adopts A's commit and commits its own work on top of it.
    let applied = verb(&b, store, "reconcile", &["--apply"]).ok().json();
    assert_eq!(applied["applied"]["publication"]["outcome"], "advanced");
    assert_eq!(applied["applied"]["publication"]["height"], 1);
    assert_eq!(
        applied["applied"]["publication"]["from"], won["step"]["publication"]["commit"],
        "the winner is displaced"
    );
    let status = jl(
        &b,
        &store_args(store, &["progress", "status", "--run", "run-race"]),
    )
    .ok()
    .json();
    assert_eq!(status["head"]["height"], 1);
    assert_eq!(status["summary"]["counts"]["in_flight"], 2, "{status}");
    // Nothing is left to reconcile.
    let done = verb(&b, store, "reconcile", &[]).ok().json();
    assert_eq!(done["admissible"], false);
    assert_eq!(done["recommit"], json!([]));

    // A host that froze another census for the same run cannot rebase onto it: the plan says
    // why (exit 1), and apply writes nothing.
    let c = dir.path().join("host-c");
    let out = init(&c, store, &["train:unit-a", "train:unit-c"])
        .ok()
        .json();
    let unit_c = out["units"]
        .as_array()
        .unwrap()
        .iter()
        .find(|u| u["subject"] == "unit-c")
        .unwrap()["work_id"]
        .as_str()
        .unwrap()
        .to_string();
    verb(&c, store, "schedule", &["--unit", &unit_c]).ok();
    assert_eq!(verb(&c, store, "flush", &[]).status, 2);
    let (store_before, c_before) = (tree(&store_dir), tree(&c));
    for extra in [&[][..], &["--apply"][..]] {
        let run = verb(&c, store, "reconcile", extra);
        assert_eq!(run.status, 1, "{}", run.stderr);
        let plan = run.json();
        assert_eq!(plan["admissible"], false);
        assert!(
            plan["refusal"].as_str().unwrap().contains("graph rules"),
            "{plan}"
        );
        assert!(plan["applied"].is_null());
        assert_eq!(tree(&store_dir), store_before);
        assert_eq!(tree(&c), c_before);
    }
}

/// A repository-local claims record under `root` pinning `bytes` as node `name`, revoked when
/// `revoked`.
fn claims_record(root: &Path, name: &str, bytes: &[u8], revoked: bool) -> std::path::PathBuf {
    fs::create_dir_all(root.join("evidence")).unwrap();
    fs::write(root.join(format!("evidence/{name}")), bytes).unwrap();
    let revocations = if revoked {
        json!([{"node": name, "issuer": "reviewer", "cause": "corrupt input",
            "effective_utc": "2026-09-26T12:00:00Z", "sequence": 1}])
    } else {
        json!([])
    };
    let path = root.join("docs/internal/core-ip/claims/claims.json");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({"schema": "jope.core-ip-claims/v1",
            "evidence": {name: {"kind": "synthetic-fixture", "location": "repository",
                "path": format!("evidence/{name}"), "sha256": sha256_hex(bytes),
                "source_commit": "a".repeat(40), "subject": "fixture", "scope": "fixture",
                "depends_on": []}},
            "revocations": revocations, "claims": []}))
        .unwrap(),
    )
    .unwrap();
    path
}

#[test]
fn revocations_import_is_read_only_by_default_and_refuses_the_revoked_record_once_applied() {
    let dir = tempfile::tempdir().unwrap();
    let repo = tempfile::tempdir().unwrap();
    let revoked = candidate("d0", 0.5, 1);
    let path = claims_record(repo.path(), "record", &revoked.blob, true);
    let (path, root) = (path.to_str().unwrap(), repo.path().to_str().unwrap());
    let import = ["revocations", "import", "--claims", path, "--root", root];

    // The plan names what would be appended and writes nothing.
    let plan = jl(dir.path(), &import).ok().json();
    assert_eq!(plan["applied"], false);
    assert_eq!(plan["new"][0]["node"], "record");
    assert_eq!(plan["new"][0]["sha256"], sha256_hex(&revoked.blob));
    assert!(tree(dir.path()).is_empty(), "the plan wrote something");
    let status = jl(dir.path(), &["revocations", "status"]).ok().json();
    assert_eq!(status["status"]["revoked"], 0);

    // Apply, then the revoked record is refused by the HWM ledger.
    let applied = jl(dir.path(), &[&import[..], &["--apply"]].concat())
        .ok()
        .json();
    assert_eq!(applied["applied"], true);
    assert_eq!(
        jl(dir.path(), &["revocations", "status"]).ok().json()["status"]["revoked"],
        1
    );
    let state = State::at(dir.path());
    let report = hwm::ingest(&state, None, vec![revoked], json!({"k": 1})).unwrap();
    assert_eq!(report.rejects[0].reason, "revoked");

    // A record the claims verifier refuses imports nothing: exit 2, the ledger unchanged.
    let before = tree(dir.path());
    fs::write(repo.path().join("evidence/record"), b"tampered").unwrap();
    let out = jl(dir.path(), &[&import[..], &["--apply"]].concat());
    assert_eq!(out.status, 2, "{}", out.stderr);
    assert!(
        out.stderr.contains("evidence digest mismatch"),
        "{}",
        out.stderr
    );
    assert_eq!(tree(dir.path()), before);
}
