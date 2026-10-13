//! S3 retention brain (S3-RET): fixture tests for the read-only dry-run classifier.
//!
//! One synthetic ledger exercises every action path, the hard-T0 guard, reference roles,
//! bucketless keys, inventory-file skipping, prose mentions (D1), lifecycle hazards and the
//! fail-closed input checks.

use jope_learning::canon::sha256_hex;
use jope_learning::retention::classify::{expire_allowed, hard_t0, Tv};
use jope_learning::retention::groups::flatten;
use jope_learning::retention::inputs::InputSpec;
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

#[test]
fn every_action_path_matches_the_binding_rules() {
    let f = fixture();
    let b = build(&config(&f)).unwrap();
    let l = |p: &str| format!("s3://{LEARN}/{p}");
    assert_eq!(
        action(&b, &l("#direct")),
        Action::Hold,
        "unmatched root objects default to HOLD"
    );
    assert_eq!(action(&b, &l("experiments/#direct")), Action::Hold);
    assert_eq!(
        action(&b, &l("experiments/gp-teacher-v6/contexts/aaaa/")),
        Action::KeepHot,
        "bucketless frozen24 key ref keeps it in place"
    );
    assert_eq!(
        action(&b, &l("experiments/gp-teacher-v6/contexts/bbbb/")),
        Action::KeepHot,
        "live writer"
    );
    assert_eq!(
        action(&b, &l("experiments/gp-teacher-v6/contexts/cccc/")),
        Action::Compact,
        "closed, small objects; skipped dir pins nothing"
    );
    assert_eq!(
        action(
            &b,
            &l("experiments/gp-policy-dl/v2-trainonly/program-results/")
        ),
        Action::KeepT0,
        "val263 reference"
    );
    assert_eq!(
        action(&b, &l("experiments/gp-policy-dl/v2-trainonly/gp-hwm/")),
        Action::KeepHot,
        "train reference by key"
    );
    assert_eq!(
        action(&b, &l("experiments/gp-policy-dl/v2-trainonly/checkpoints/")),
        Action::Hold,
        "no checkpoint->L6 linkage"
    );
    assert_eq!(
        action(&b, &l("experiments/regression-evolution/v1/attempts/")),
        Action::DistillCold,
        "inventory copy and elided key do not pin"
    );
    assert_eq!(
        action(&b, &l("experiments/regression-evolution/v1/contexts/")),
        Action::KeepT0,
        "holds GP rows of a val263 context"
    );
    assert_eq!(
        action(&b, &l("experiments/mystery/")),
        Action::KeepT0,
        "main-tree object reference pins"
    );
    assert_eq!(
        action(
            &b,
            &l("staging/lineage-a/preserved-archives-20260908-F-D-E/")
        ),
        Action::Cold
    );
    assert_eq!(action(&b, &format!("s3://{DATA}/splits/")), Action::KeepT0);
    assert_eq!(
        action(&b, &format!("s3://{SEALED}/objects/")),
        Action::KeepT0
    );
    assert_eq!(
        action(&b, &format!("s3://{LEGACY}/data/")),
        Action::Hold,
        "out of scope"
    );
    assert!(
        b.rows.iter().all(|r| r.action != Action::Expire),
        "no EXPIRE at prefix level"
    );
    assert!(b.rows.iter().all(|r| r.tv.value), "TV fails closed to true");
    assert_eq!(
        b.summary.totals.objects,
        ledger()["totals"]["objects"].as_u64().unwrap()
    );
    assert_eq!(b.summary.refs_bucketless_resolved, 1);
    assert_eq!(
        b.summary.refs_unmatched_in_scope, 0,
        "globs and placeholders are prefix mentions"
    );
    assert!(b.summary.prefix_mentions >= 3);
    for g in ["#direct", "experiments/#direct"] {
        assert!(
            row(&b, &l(g)).refs_by_role.is_empty(),
            "{g}: a glob/placeholder must not pin a group"
        );
    }
}

#[test]
fn tv_index_and_cost_fields_are_derived_from_the_inputs() {
    let f = fixture();
    let b = build(&config(&f)).unwrap();
    let hwm = row(
        &b,
        &format!("s3://{LEARN}/experiments/gp-policy-dl/v2-trainonly/gp-hwm/"),
    );
    let ix = hwm.index.as_ref().unwrap();
    assert_eq!(
        (ix.rows, ix.scored_rows, ix.contexts_with_alternatives),
        (2, 2, 1)
    );
    for u in ["ce-target", "negative-example", "knn-teacher", "reservoir"] {
        assert!(
            hwm.tv.uses.contains(&u),
            "gp-hwm TV lacks {u}: {:?}",
            hwm.tv.uses
        );
    }
    assert_eq!(hwm.pinned_programs.get("train"), Some(&1));
    let cold = row(
        &b,
        &format!("s3://{LEARN}/staging/lineage-a/preserved-archives-20260908-F-D-E/"),
    );
    assert_eq!(cold.cost.class_after, "DEEP_ARCHIVE");
    assert!(
        cold.cost.already_applied_by_lifecycle,
        "p1 rule already transitions it"
    );
    assert!(cold.cost.after_usd_mo < cold.cost.now_usd_mo);
    let dc = row(
        &b,
        &format!("s3://{LEARN}/experiments/regression-evolution/v1/attempts/"),
    );
    assert!(dc.cost.after_needs_e1_of_originals);
    assert!(
        dc.cost.one_time_off_aws_usd > dc.cost.one_time_in_region_usd,
        "reading from xbabe adds egress"
    );
    let unknown = row(
        &b,
        &format!("s3://{LEARN}/experiments/gp-teacher-v6/contexts/cccc/"),
    );
    assert_eq!(unknown.tv.uses, vec!["unknown-fail-closed"]);
    let mystery = row(&b, &format!("s3://{LEARN}/experiments/mystery/"));
    assert_eq!(
        mystery.ref_sources_sample,
        vec!["main-manifest:docs/notes.md".to_string()]
    );
    let ckpt = row(
        &b,
        &format!("s3://{LEARN}/experiments/gp-policy-dl/v2-trainonly/checkpoints/"),
    );
    assert!(
        ckpt.lifecycle
            .tagged
            .iter()
            .any(|t| t.starts_with("checkpoint-blobs-gir")),
        "tag rules are listed, not applied"
    );
    assert!(ckpt.lifecycle.cold_class.is_none());
}

#[test]
fn lifecycle_hazards_and_notes_are_reported() {
    let f = fixture();
    let b = build(&config(&f)).unwrap();
    assert!(
        b.summary
            .hazards
            .iter()
            .any(|h| h.kind == "expire-current-over-protected" && h.detail.contains("bad-expire")),
        "{:?}",
        b.summary.hazards
    );
    assert!(b
        .summary
        .notes
        .iter()
        .any(|n| n.kind == "lifecycle-not-supplied" && n.detail.contains(SEALED)));
    // D2 (owner, 2026-10-13T12:42:52Z): 90 d is the T0 noncurrent window too; nothing asks 365 d.
    let all = |b: &Built| -> Vec<String> {
        b.summary
            .hazards
            .iter()
            .chain(&b.summary.notes)
            .map(|n| n.kind.clone())
            .collect()
    };
    assert!(!all(&b)
        .iter()
        .any(|k| k.contains("365") || k.contains("noncurrent")));
    let lc = json!({"Rules": [{"ID": "short", "Status": "Enabled", "Filter": {"Prefix": ""}, "NoncurrentVersionExpiration": {"NoncurrentDays": 60}}]});
    put(
        &f.path("lc/veox-jopedime-learning-use2.json"),
        &serde_json::to_vec(&lc).unwrap(),
    );
    let b = build(&config(&f)).unwrap();
    assert!(
        b.summary
            .hazards
            .iter()
            .any(|h| h.kind == "t0-noncurrent-below-policy" && h.detail.contains("60d")),
        "{:?}",
        b.summary.hazards
    );
}

#[test]
fn plan_is_deterministic_and_never_overwritten() {
    let f = fixture();
    let a = build(&config(&f)).unwrap();
    let b = build(&config(&f)).unwrap();
    assert_eq!(a.plan_jsonl, b.plan_jsonl);
    assert_eq!(a.summary.plan_sha256, sha256_hex(&a.plan_jsonl));
    let out = f.path("out/plan1");
    write(&a, &out).unwrap();
    let side = fs::read_to_string(out.join("retention-plan.jsonl.sha256")).unwrap();
    assert!(side.starts_with(&a.summary.plan_sha256));
    let err = write(&b, &out).unwrap_err().to_string();
    assert!(err.contains("not empty"), "{err}");
    // A second plan diffs against the first.
    let mut cfg = config(&f);
    cfg.previous_plan = Some(out.join("retention-plan.jsonl"));
    let c = build(&cfg).unwrap();
    let d = c.summary.diff_vs_previous.unwrap();
    assert_eq!(d.get("unchanged").copied(), Some(c.rows.len() as u64));
}

#[test]
fn inputs_fail_closed() {
    let f = fixture();
    // Sidecar disagreement.
    put(
        &f.path("in/ledger.sha256"),
        b"0000000000000000000000000000000000000000000000000000000000000000  ledger.json\n",
    );
    let err = format!("{:#}", build(&config(&f)).err().unwrap());
    assert!(err.contains("disagrees with sidecar"), "{err}");

    // Unknown score identity.
    let f = fixture();
    let mut ix = index();
    ix["runs"][0]["score_name"] = json!("l7_mystery");
    put(&f.path("in/index.json"), &serde_json::to_vec(&ix).unwrap());
    let err = format!("{:#}", build(&config(&f)).err().unwrap());
    assert!(err.contains("unknown score identity"), "{err}");

    // Roster member hash mismatch.
    let f = fixture();
    put(
        &f.path("roster/train.jsonl"),
        format!("{}\n", json!({"context_id": C1})).as_bytes(),
    );
    let err = format!("{:#}", build(&config(&f)).err().unwrap());
    assert!(err.contains("roster manifest"), "{err}");

    // Expected sha256 prefix.
    let f = fixture();
    let mut cfg = config(&f);
    cfg.gp_index.expect_prefix = Some("deadbeef".into());
    let err = format!("{:#}", build(&cfg).err().unwrap());
    assert!(err.contains("does not start with the expected"), "{err}");

    // Ledger that does not reconcile.
    let f = fixture();
    let mut l = ledger();
    l["totals"]["objects"] = json!(1);
    put(&f.path("in/ledger.json"), &serde_json::to_vec(&l).unwrap());
    let lb = fs::read(f.path("in/ledger.json")).unwrap();
    put(
        &f.path("in/ledger.sha256"),
        format!("{}  ledger.json\n", sha256_hex(&lb)).as_bytes(),
    );
    let err = format!("{:#}", build(&config(&f)).err().unwrap());
    assert!(err.contains("totals say"), "{err}");

    // Incomplete bucket.
    let mut l = ledger();
    l["buckets"][0]["complete"] = json!(false);
    assert!(flatten(&l).is_err());
    let mut l = ledger();
    l["capped"] = json!(true);
    assert!(flatten(&l).is_err());
}

#[test]
fn hard_t0_cannot_be_removed_by_policy() {
    // A policy that calls splits/ and the sealed bucket expendable still yields KEEP_T0.
    let f = fixture();
    let mut p: Value = serde_json::from_str(DEFAULT_POLICY_JSON).unwrap();
    p["rules"].as_array_mut().unwrap().push(json!({"bucket": DATA, "prefix": "splits/", "kind": "gp-result-unconsumed", "family": "x", "why": "hostile"}));
    p["rules"].as_array_mut().unwrap().push(
        json!({"bucket": SEALED, "prefix": "", "kind": "archive", "family": "x", "why": "hostile"}),
    );
    put(
        &f.path("hostile-policy.json"),
        &serde_json::to_vec(&p).unwrap(),
    );
    let mut cfg = config(&f);
    cfg.policy = Some(f.path("hostile-policy.json"));
    let b = build(&cfg).unwrap();
    assert_eq!(action(&b, &format!("s3://{DATA}/splits/")), Action::KeepT0);
    assert_eq!(
        action(&b, &format!("s3://{SEALED}/objects/")),
        Action::KeepT0
    );
    assert!(hard_t0("veox-jopedime-archive-use2", "evidence/x/").is_some());
    assert!(hard_t0(
        "veox-jopedime-archive-use2",
        "staging/node-high-value/sealed-meta-test-winners/"
    )
    .is_some());
    assert!(hard_t0(
        "veox-jopedime-learning-use2",
        "experiments/gp-teacher-v6/contexts/"
    )
    .is_none());
}

#[test]
fn policy_validation_rejects_unsafe_settings() {
    let base: Value = serde_json::from_str(DEFAULT_POLICY_JSON).unwrap();
    Policy::parse(DEFAULT_POLICY_JSON.as_bytes()).unwrap();
    for (field, value) in [
        ("l6_direction", json!("higher_is_better")),
        ("noncurrent_days", json!(7)),
        ("pack_bytes", json!(1024)),
        ("reservoir_rate", json!(0.0)),
    ] {
        let mut p = base.clone();
        p[field] = value;
        assert!(
            Policy::parse(&serde_json::to_vec(&p).unwrap()).is_err(),
            "{field} accepted"
        );
    }
    // D1: structured refs and non-scan roles can never be demoted to prose mentions.
    for (field, value) in [
        ("prose_extensions", json!(["md", "json"])),
        ("prose_extensions", json!(["jsonl"])),
        ("prose_extensions", json!(["MD"])),
        ("prose_extensions", json!(["parquet"])),
        ("prose_roles", json!(["val263"])),
        ("prose_roles", json!(["evidence", "prereg"])),
        ("prose_roles", json!(["prose-mention"])),
    ] {
        let mut p = base.clone();
        p["reference_formats"][field] = value.clone();
        assert!(
            Policy::parse(&serde_json::to_vec(&p).unwrap()).is_err(),
            "reference_formats.{field} = {value} accepted"
        );
    }
}

#[test]
fn prose_evidence_mentions_are_tv_not_pins() {
    // Owner decision D1 (2026-10-13T12:36Z): JSON/JSONL evidence pins T0; an md/txt/log mention
    // is training value only (blocks EXPIRE, allows COMPACT/COLD); unruled formats still pin.
    let f = fixture();
    let l = |p: &str| format!("s3://{LEARN}/{p}");
    put(
        &f.path("evid/chat/AGENT_CHAT.md"),
        format!("ran s3://{LEARN}/experiments/gp-teacher-v6/contexts/cccc/x.json today\n")
            .as_bytes(),
    );
    put(
        &f.path("evid/logs/run.log"),
        format!("GET s3://{LEARN}/experiments/regression-evolution/v1/attempts/a2 200\n")
            .as_bytes(),
    );
    put(
        &f.path("evid/receipts/r.json"),
        &serde_json::to_vec(&json!({"bucket": LEARN, "key": "staging/lineage-a/preserved-archives-20260908-F-D-E/a.tar"}))
            .unwrap(),
    );
    put(
        &f.path("evid/notes.yaml"),
        format!("ref: s3://{LEARN}/experiments/gp-teacher-v6/contexts/bbbb/y\n").as_bytes(),
    );
    let mut cfg = config(&f);
    cfg.scans.push((Role::Evidence, f.path("evid")));
    let b = build(&cfg).unwrap();

    let cccc = row(&b, &l("experiments/gp-teacher-v6/contexts/cccc/"));
    assert_eq!(cccc.action, Action::Compact, "{}", cccc.reason);
    assert_eq!(cccc.t0, None);
    assert_eq!(cccc.refs_by_role.get("prose-mention"), Some(&1));
    assert!(cccc.tv.value && cccc.tv.uses.contains(&"prose-mention"));
    assert!(
        !expire_allowed(&cccc.tv, true, true),
        "a prose mention blocks EXPIRE"
    );
    assert!(cccc
        .ref_sources_sample
        .contains(&"prose-mention:chat/AGENT_CHAT.md".to_string()));
    let attempts = row(&b, &l("experiments/regression-evolution/v1/attempts/"));
    assert_eq!(attempts.action, Action::DistillCold, "{}", attempts.reason);
    assert!(attempts.tv.uses.contains(&"prose-mention"));
    assert_eq!(
        action(
            &b,
            &l("staging/lineage-a/preserved-archives-20260908-F-D-E/")
        ),
        Action::KeepT0,
        "structured JSON evidence pins"
    );
    let bbbb = row(&b, &l("experiments/gp-teacher-v6/contexts/bbbb/"));
    assert_eq!(bbbb.action, Action::KeepT0, "unruled yaml still pins");
    assert_eq!(bbbb.refs_by_role.get("evidence"), Some(&1));
    assert_eq!(
        action(&b, &l("experiments/mystery/")),
        Action::KeepT0,
        "main-tree prose is outside the ruling and still pins"
    );
    let ev = b.lock.scans.iter().find(|s| s.role == "evidence").unwrap();
    assert_eq!((ev.object_refs, ev.prose_refs), (4, 2));
    assert!(b.report_md.contains("(2 prose mentions, D1)"));

    // Without the policy entry (the pre-D1 behaviour) the same mentions pin.
    let mut p: Value = serde_json::from_str(DEFAULT_POLICY_JSON).unwrap();
    p["reference_formats"]["prose_roles"] = json!([]);
    put(&f.path("pre-d1.json"), &serde_json::to_vec(&p).unwrap());
    cfg.policy = Some(f.path("pre-d1.json"));
    let pre = build(&cfg).unwrap();
    assert_eq!(
        action(&pre, &l("experiments/gp-teacher-v6/contexts/cccc/")),
        Action::KeepT0
    );
    assert_eq!(
        action(&pre, &l("experiments/regression-evolution/v1/attempts/")),
        Action::KeepT0
    );
}

#[test]
fn expire_needs_all_three_conditions() {
    let no_tv = Tv {
        value: false,
        uses: vec![],
    };
    let tv = Tv {
        value: true,
        uses: vec!["unknown-fail-closed"],
    };
    assert!(expire_allowed(&no_tv, true, true));
    assert!(!expire_allowed(&tv, true, true));
    assert!(!expire_allowed(&no_tv, false, true));
    assert!(!expire_allowed(&no_tv, true, false));
}

#[test]
fn retention_sources_have_no_s3_write_or_delete_path() {
    // The brain and the write-side library are read-only/offline by construction: no S3 client,
    // no process spawning, no filesystem deletion. An apply path must be a separate reviewed tool.
    let sources = [
        ("mod.rs", include_str!("../src/retention/mod.rs")),
        ("classify.rs", include_str!("../src/retention/classify.rs")),
        (
            "distilled.rs",
            include_str!("../src/retention/distilled.rs"),
        ),
        ("groups.rs", include_str!("../src/retention/groups.rs")),
        ("index.rs", include_str!("../src/retention/index.rs")),
        ("inputs.rs", include_str!("../src/retention/inputs.rs")),
        (
            "lifecycle.rs",
            include_str!("../src/retention/lifecycle.rs"),
        ),
        ("pack.rs", include_str!("../src/retention/pack.rs")),
        ("parquet.rs", include_str!("../src/retention/parquet.rs")),
        ("plan.rs", include_str!("../src/retention/plan.rs")),
        ("policy.rs", include_str!("../src/retention/policy.rs")),
        ("refs.rs", include_str!("../src/retention/refs.rs")),
        ("tags.rs", include_str!("../src/retention/tags.rs")),
        (
            "bin/jope_retention.rs",
            include_str!("../src/bin/jope_retention.rs"),
        ),
    ];
    let forbidden = [
        "object_store",
        "jope_datasets",
        "remove_file",
        "remove_dir",
        "process::Command",
        "DeleteObject",
        "delete_object",
        "PutObject",
        "put_object",
        "PutBucketLifecycle",
        "CopyObject",
    ];
    for (name, text) in sources {
        for f in forbidden {
            assert!(!text.contains(f), "{name} contains forbidden {f:?}");
        }
    }
}
