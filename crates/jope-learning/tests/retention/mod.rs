//! S3 retention brain (S3-RET): fixture tests for the read-only dry-run classifier.
//!
//! One synthetic ledger exercises every action path, the hard-T0 guard, reference roles,
//! bucketless keys, inventory-file skipping, prose mentions (D1), lifecycle hazards and the
//! fail-closed input checks.

use jope_learning::canon::sha256_hex;
use jope_learning::retention::classify::{expire_allowed, hard_t0, Tv};
use jope_learning::retention::groups::flatten;
use jope_learning::retention::inputs::InputSpec;
use jope_learning::retention::inventory::load_report;
use jope_learning::retention::md5::md5_hex;
use jope_learning::retention::parquet::{
    read, write as write_parquet, write_with, Cell, Codec, ColumnSpec, ColumnType, FileReader,
    PageCounts, Table, Values, WriteOptions,
};
use jope_learning::retention::plan::{build, write, Built, Config, PlanRow};
use jope_learning::retention::policy::{Policy, DEFAULT_POLICY_JSON};
use jope_learning::retention::refs::Role;
use jope_learning::retention::Action;
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

const LEARN: &str = "veox-jopedime-learning-use2";
const DATA: &str = "veox-jopedime-datasets-use2";
const SEALED: &str = "jopedime-714044927654-us-east-2-sealed";
const LEGACY: &str = "veox-legacy-thing";
const C1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const C2: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const P1: &str = "aaaa111111111111111111111111111111111111111111111111111111111111";
const P2: &str = "aaaa222222222222222222222222222222222222222222222222222222222222";

fn leaf(objects: u64, bytes: u64, max: &str) -> Value {
    json!({"objects": objects, "bytes": bytes, "min_last_modified": "2026-09-20T00:00:00Z", "max_last_modified": max})
}

fn dir(children: Vec<(&str, Value)>, direct_objects: u64, direct_bytes: u64) -> Value {
    let o: u64 = children
        .iter()
        .map(|(_, c)| c["objects"].as_u64().unwrap())
        .sum::<u64>()
        + direct_objects;
    let b: u64 = children
        .iter()
        .map(|(_, c)| c["bytes"].as_u64().unwrap())
        .sum::<u64>()
        + direct_bytes;
    let max = children
        .iter()
        .map(|(_, c)| c["max_last_modified"].as_str().unwrap().to_string())
        .max()
        .unwrap();
    let ch: serde_json::Map<String, Value> = children
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    json!({"objects": o, "bytes": b, "min_last_modified": "2026-09-20T00:00:00Z", "max_last_modified": max, "children": ch})
}

fn bucket(name: &str, prefixes: Vec<(&str, Value)>, root_o: u64, root_b: u64) -> Value {
    let o: u64 = prefixes
        .iter()
        .map(|(_, c)| c["objects"].as_u64().unwrap())
        .sum::<u64>()
        + root_o;
    let b: u64 = prefixes
        .iter()
        .map(|(_, c)| c["bytes"].as_u64().unwrap())
        .sum::<u64>()
        + root_b;
    let p: serde_json::Map<String, Value> = prefixes
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    json!({
        "name": name, "region": "us-east-2", "complete": true, "errors": [],
        "objects": o, "size": {"bytes": b}, "root_objects": root_o, "root_bytes": root_b,
        "min_last_modified": "2026-09-20T00:00:00Z", "max_last_modified": "2026-10-13T07:00:00Z",
        "by_storage_class": {"STANDARD": {"objects": o, "bytes": b}}, "prefixes": p
    })
}

fn ledger() -> Value {
    let buckets = vec![
        bucket(
            LEARN,
            vec![
                (
                    "experiments/",
                    dir(
                        vec![
                            (
                                "gp-teacher-v6/",
                                dir(
                                    vec![(
                                        "contexts/",
                                        dir(
                                            vec![
                                                (
                                                    "aaaa/",
                                                    leaf(1000, 50_000_000, "2026-10-01T00:00:00Z"),
                                                ),
                                                (
                                                    "bbbb/",
                                                    leaf(10, 1_000_000, "2026-10-13T07:00:00Z"),
                                                ),
                                                (
                                                    "cccc/",
                                                    leaf(2000, 60_000_000, "2026-10-01T00:00:00Z"),
                                                ),
                                            ],
                                            0,
                                            0,
                                        ),
                                    )],
                                    0,
                                    0,
                                ),
                            ),
                            (
                                "gp-policy-dl/",
                                dir(
                                    vec![(
                                        "v2-trainonly/",
                                        dir(
                                            vec![
                                                (
                                                    "program-results/",
                                                    leaf(100, 10_000_000, "2026-10-09T00:00:00Z"),
                                                ),
                                                (
                                                    "gp-hwm/",
                                                    leaf(50, 100_000, "2026-10-07T00:00:00Z"),
                                                ),
                                                (
                                                    "checkpoints/",
                                                    leaf(30, 3_000_000_000, "2026-10-08T00:00:00Z"),
                                                ),
                                            ],
                                            0,
                                            0,
                                        ),
                                    )],
                                    0,
                                    0,
                                ),
                            ),
                            (
                                "regression-evolution/",
                                dir(
                                    vec![(
                                        "v1/",
                                        dir(
                                            vec![
                                                (
                                                    "attempts/",
                                                    leaf(20, 200_000_000, "2026-09-29T00:00:00Z"),
                                                ),
                                                (
                                                    "contexts/",
                                                    leaf(5, 2_000, "2026-09-29T00:00:00Z"),
                                                ),
                                            ],
                                            0,
                                            0,
                                        ),
                                    )],
                                    0,
                                    0,
                                ),
                            ),
                            ("mystery/", leaf(5, 5_000, "2026-09-25T00:00:00Z")),
                        ],
                        2,
                        200,
                    ),
                ),
                (
                    "staging/",
                    dir(
                        vec![(
                            "lineage-a/",
                            dir(
                                vec![(
                                    "preserved-archives-20260908-F-D-E/",
                                    leaf(2, 20_000_000_000, "2026-09-18T00:00:00Z"),
                                )],
                                0,
                                0,
                            ),
                        )],
                        0,
                        0,
                    ),
                ),
            ],
            1,
            10,
        ),
        bucket(
            DATA,
            vec![("splits/", leaf(40, 400_000, "2026-09-30T00:00:00Z"))],
            0,
            0,
        ),
        bucket(
            SEALED,
            vec![("objects/", leaf(8, 8_000_000_000, "2026-09-21T00:00:00Z"))],
            0,
            0,
        ),
        bucket(
            LEGACY,
            vec![("data/", leaf(3, 3_000, "2025-07-01T00:00:00Z"))],
            0,
            0,
        ),
    ];
    let o: u64 = buckets.iter().map(|b| b["objects"].as_u64().unwrap()).sum();
    let b: u64 = buckets
        .iter()
        .map(|b| b["size"]["bytes"].as_u64().unwrap())
        .sum();
    json!({
        "schema": "jopedime.s3-object-ledger/v1", "generated_at": "2026-10-13T08:00:00Z",
        "complete": true, "capped": false, "bucket_count": buckets.len(),
        "totals": {"objects": o, "size": {"bytes": b}}, "buckets": buckets
    })
}

fn index() -> Value {
    let hwm = |prog: &str, score: f64| {
        json!({
            "bucket": LEARN, "key": format!("experiments/gp-policy-dl/v2-trainonly/gp-hwm/ds/f0/{C1}/{prog}.json"),
            "grain": "context-hwm", "score": score, "score_name": "cv_geomean_rmse_ratio",
            "qualified": false, "family": "gp-policy-dl", "consumer": "roster-conditional",
            "context_sha256": C1, "program_sha": prog
        })
    };
    json!({
        "schema": "jopedime.gp-results-index/v1",
        "counts": {"runs": 4, "qualified_true": 0},
        "derived_packets": [],
        "runs": [
            hwm(P1, 0.5),
            hwm(P2, 0.6),
            {"bucket": LEARN, "key": "experiments/regression-evolution/v1/contexts/x/HEAD.json",
             "grain": "context-head-pointer", "score": null, "score_name": null, "qualified": false,
             "family": "regression-evolution", "consumer": "none recorded", "context_sha256": C2, "program_sha": null},
            {"path": "xbabe3:/scratch", "grain": "local-evolution", "score": 0.06, "score_name": "best_rust_rmse",
             "qualified": false, "family": "xbabe3-scratch", "program_sha": P1}
        ]
    })
}

struct Fixture {
    _tmp: tempfile::TempDir,
    root: PathBuf,
}

impl Fixture {
    fn path(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }
}

fn put(path: &Path, bytes: &[u8]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let ledger_bytes = serde_json::to_vec(&ledger()).unwrap();
    put(&root.join("in/ledger.json"), &ledger_bytes);
    put(
        &root.join("in/ledger.sha256"),
        format!("{}  ledger.json\n", sha256_hex(&ledger_bytes)).as_bytes(),
    );
    put(
        &root.join("in/index.json"),
        &serde_json::to_vec(&index()).unwrap(),
    );
    let val = format!(
        "{}\n",
        json!({"context_id": C2, "artifacts": {"evaluation": {"bucket": LEARN, "key": "experiments/gp-policy-dl/v2-trainonly/program-results/obj1"}}})
    );
    let train = format!(
        "{}\n",
        json!({"context_id": C1, "frozen_hwm_teacher": {"program_sha256": P1},
               "ref": {"bucket": LEARN, "key": format!("experiments/gp-policy-dl/v2-trainonly/gp-hwm/ds/f0/{C1}/{P1}.json")}})
    );
    put(&root.join("roster/validation.jsonl"), val.as_bytes());
    put(&root.join("roster/train.jsonl"), train.as_bytes());
    let manifest = json!({"members": {
        "validation.jsonl": {"sha256": sha256_hex(val.as_bytes())},
        "train.jsonl": {"sha256": sha256_hex(train.as_bytes())}
    }});
    put(
        &root.join("roster/manifest.json"),
        &serde_json::to_vec(&manifest).unwrap(),
    );
    // Bucketless key + sha256 pin, as in the frozen24 roster's gp_hwm_ref.
    let f24 = json!({"contexts": [{"context_sha256": C1, "gp_hwm_ref": {"key": "experiments/gp-teacher-v6/contexts/aaaa/obj.json", "sha256": P2}}]});
    put(
        &root.join("f24/frozen24.json"),
        &serde_json::to_vec(&f24).unwrap(),
    );
    // Scan root: a doc pin and an inventory file that must not pin anything.
    put(
        &root.join("repo/docs/notes.md"),
        format!(
            "See s3://{LEARN}/experiments/mystery/file.json and the bucket s3://{LEARN}/ itself; globs s3://{LEARN}/progress/* and s3://{LEARN}/experiments/${{RUN}}/x.json and the elided s3://{LEARN}/experiments/regression-evolution/v1/attempts/a7f3… pin nothing.\n"
        )
        .as_bytes(),
    );
    let inventory = json!({"schema": "jopedime.gp-results-index/v1", "runs": [{"bucket": LEARN, "key": "experiments/regression-evolution/v1/attempts/a1"}]});
    put(
        &root.join("repo/index-copy.json"),
        &serde_json::to_vec(&inventory).unwrap(),
    );
    put(
        &root.join("repo/skipme/pin.json"),
        &serde_json::to_vec(
            &json!({"bucket": LEARN, "key": "experiments/gp-teacher-v6/contexts/cccc/x"}),
        )
        .unwrap(),
    );
    // Lifecycle: the applied P1 Deep Archive rule, and a bad current-version expiry over splits/.
    let learn_lc = json!({"Rules": [
        {"ID": "abort-mpu", "Status": "Enabled", "Filter": {"Prefix": ""}, "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}},
        {"ID": "p0-noncurrent-expire-90d", "Status": "Enabled", "Filter": {"Prefix": ""}, "NoncurrentVersionExpiration": {"NoncurrentDays": 90}},
        {"ID": "p1-preserved-archives-deep-archive", "Status": "Enabled", "Filter": {"Prefix": "staging/lineage-a/preserved-archives-20260908-F-D-E/"}, "Transitions": [{"Days": 0, "StorageClass": "DEEP_ARCHIVE"}]},
        {"ID": "checkpoint-blobs-gir", "Status": "Enabled", "Filter": {"And": {"Prefix": "experiments/", "Tags": [{"Key": "kind", "Value": "checkpoint"}]}}, "Transitions": [{"Days": 30, "StorageClass": "GLACIER_IR"}]}
    ]});
    let data_lc = json!({"Rules": [
        {"ID": "bad-expire", "Status": "Enabled", "Filter": {"Prefix": "splits/"}, "Expiration": {"Days": 30}}
    ]});
    put(
        &root.join("lc/veox-jopedime-learning-use2.json"),
        &serde_json::to_vec(&learn_lc).unwrap(),
    );
    put(
        &root.join("lc/veox-jopedime-datasets-use2.json"),
        &serde_json::to_vec(&data_lc).unwrap(),
    );
    Fixture { _tmp: tmp, root }
}

fn spec(role: &str, path: PathBuf) -> InputSpec {
    InputSpec {
        role: role.into(),
        path,
        expect_prefix: None,
    }
}

fn config(f: &Fixture) -> Config {
    Config {
        as_of: "2026-10-13T08:00:00Z".into(),
        ledger: spec("ledger", f.path("in/ledger.json")),
        gp_index: spec("gp-index", f.path("in/index.json")),
        extra_inputs: vec![],
        refs: vec![
            spec("val263", f.path("roster/validation.jsonl")),
            spec("train", f.path("roster/train.jsonl")),
            spec("frozen24", f.path("f24/frozen24.json")),
        ],
        roster_manifest: Some(spec("roster-manifest", f.path("roster/manifest.json"))),
        scans: vec![(Role::MainManifest, f.path("repo"))],
        scan_max_bytes: 1 << 20,
        scan_skip_dirs: vec!["skipme".into()],
        lifecycle_dir: Some(f.path("lc")),
        inventories: vec![],
        policy: None,
        previous_plan: None,
        source_commit: Some("test".into()),
    }
}

fn row<'a>(b: &'a Built, id: &str) -> &'a PlanRow {
    b.rows
        .iter()
        .find(|r| r.group_id == id)
        .unwrap_or_else(|| panic!("no row {id}"))
}

fn action(b: &Built, id: &str) -> Action {
    row(b, id).action
}

mod action_tests;
mod inventory_tests;
mod policy_tests;
mod source_guard_tests;
