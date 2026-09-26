//! W1-T07: the historical GPU-study staging is recorded without rewriting it;
//! the corrected fitting-only protocol has a new identity; the replay verifier
//! rejects the historical input closure; historical and corrected documents
//! cannot be substituted for one another.

use jope_datasets::hash::sha256_hex;
use jope_datasets::staging_disposition::*;
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

const DISPOSITION: &[u8] =
    include_bytes!("../releases/staging-disposition/historical-gpu-staging-20260925.json");
const PROTOCOL: &[u8] =
    include_bytes!("../releases/staging-disposition/fitting-only-staging-protocol-v1.json");
const HISTORICAL_SOURCE: &str = "802fb464c8a30eb7aeb05296313300e737a96044";

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn committed() -> (HistoricalDisposition, CorrectedProtocol) {
    let d = HistoricalDisposition::open(DISPOSITION).unwrap();
    let p = CorrectedProtocol::open(PROTOCOL, &d).unwrap();
    (d, p)
}

fn digest(seed: &str) -> String {
    sha256_hex(seed.as_bytes())
}

/// A fitting-input manifest naming the given source and object digests.
fn manifest(source: &str, raw: &[String], folds: &[String]) -> Vec<u8> {
    let inputs: Vec<Value> = raw
        .iter()
        .enumerate()
        .map(|(i, sha)| {
            let gp = if i == 0 {
                json!({"folds":folds.iter().map(|f| json!([{"path":"fold.json","sha256":f},[]])).collect::<Vec<_>>()})
            } else {
                Value::Null
            };
            json!({"id":format!("dataset-{i}"),"raw":{"path":"raw.json","sha256":sha},"gp":gp})
        })
        .collect();
    serde_json::to_vec(
        &json!({"schema":FITTING_INPUTS_SCHEMA,"panel_sha256":digest("panel"),
        "configuration_sha256":digest("configuration"),"projection_source_sha":source,"seed":0,
        "assignments":{},"inputs":inputs}),
    )
    .unwrap()
}

fn fresh(n: usize, tag: &str) -> Vec<String> {
    (0..n).map(|i| digest(&format!("{tag}-{i}"))).collect()
}

fn attested(manifest: &[u8], source: &str) -> RunClosure {
    RunClosure {
        manifest_sha256: sha256_hex(manifest),
        projection_source_sha: source.into(),
        projection_protocol: PROJECTION_PROTOCOL.into(),
        attestation: ProjectionAttestation {
            inputs_pre_projected: true,
            outer_scoring_labels_parsed: false,
            development_targets_parsed_outside_fitting_rows: false,
            sealed_tables_opened: false,
        },
    }
}

fn rejected(result: anyhow::Result<Value>, needle: &str) {
    let error = format!("{:#}", result.expect_err("accepted"));
    assert!(error.contains(needle), "expected {needle:?}: {error}");
}

#[test]
fn committed_disposition_records_the_retained_evidence_without_upgrading_it() {
    let (d, _) = committed();
    assert_eq!(d.encode().unwrap(), DISPOSITION);
    let o = &d.observed;
    assert!(o.outer_scoring_labels_parsed);
    assert!(!o.sealed_target_tables_opened && !o.outer_scoring_executed);
    assert!(!o.campaign_held_looks_consumed && !o.worker_payload_contains_outer_targets);
    assert!(!o.gpu_qualification_executed);
    assert_eq!(
        (o.direct_requests, o.full_gp_arms, o.prediction_batches),
        (0, 0, 0)
    );
    assert_eq!(o.controller_exit_code, 1);
    assert_eq!(o.status, "controller_failed_before_cuda_qualification");
    let k = &d.classification;
    assert!(!k.contamination_inferred && !k.protocol_compliant && !k.qualified_measurement);
    assert!(k.excluded_from_headline && !k.original_result_reclassified);
    assert!(!d.unknowns.is_empty());
    let c = &d.input_closure;
    assert_eq!(c.projection_source_sha, HISTORICAL_SOURCE);
    assert_eq!(d.study.source_sha, HISTORICAL_SOURCE);
    assert_eq!(c.members.len(), 10);
    assert_eq!(
        c.members
            .iter()
            .map(|m| m.gp_fold_sha256.len())
            .sum::<usize>(),
        15
    );
    assert_eq!(d.study.plan_schema, HISTORICAL_PLAN_SCHEMA);
    // The published evidence records are the bytes the disposition bound.
    let repository: Vec<_> = d
        .evidence
        .iter()
        .filter(|e| e.location == "repository")
        .collect();
    assert_eq!(repository.len(), REPOSITORY_FILES.len());
    for e in repository {
        assert_eq!(
            sha256_hex(&fs::read(repository_root().join(&e.path)).unwrap()),
            e.sha256,
            "published evidence {} changed after the disposition was derived",
            e.path
        );
    }
    // No field can be upgraded, with or without a recomputed identity.
    for (field, value) in [
        ("/classification/contamination_inferred", json!(true)),
        ("/classification/protocol_compliant", json!(true)),
        ("/classification/qualified_measurement", json!(true)),
        ("/classification/excluded_from_headline", json!(false)),
        ("/unknowns", json!([])),
    ] {
        let mut value_doc: Value = serde_json::from_slice(DISPOSITION).unwrap();
        *value_doc.pointer_mut(field).unwrap() = value;
        let mut forged: HistoricalDisposition = serde_json::from_value(value_doc).unwrap();
        assert!(forged.validate().is_err(), "{field} kept its identity");
        forged.disposition_id = forged.compute_id().unwrap();
        assert!(forged.validate().is_err(), "{field} upgraded the history");
    }
}

#[test]
fn the_committed_protocol_is_a_new_identity_derived_from_the_disposition() {
    let (d, p) = committed();
    assert_eq!(p, CorrectedProtocol::derive(&d).unwrap());
    assert_eq!(p.encode().unwrap(), PROTOCOL);
    assert_ne!(p.protocol_id, d.disposition_id);
    assert_eq!(p.historical_disposition_id, d.disposition_id);
    assert_eq!(p.forbidden_projection_sources, [HISTORICAL_SOURCE]);
    assert_eq!(
        p.forbidden_windows,
        [d.study.window_experiment_sha256.clone()]
    );
    // Manifest + ten projected inputs + fifteen inner-fold inputs.
    assert_eq!(p.forbidden_closure_sha256.len(), 26);
    assert!(p
        .forbidden_closure_sha256
        .contains(&d.input_closure.manifest_sha256));
    // Frozen label-free split memberships are panel inputs, not products.
    for m in &d.input_closure.members {
        assert!(!p.forbidden_closure_sha256.contains(&m.assignment_sha256));
    }
}

#[test]
fn the_replay_verifier_rejects_the_historical_input_closure() {
    let (d, p) = committed();
    let historical_raw: Vec<String> = d
        .input_closure
        .members
        .iter()
        .map(|m| m.raw_sha256.clone())
        .collect();
    let historical_folds: Vec<String> = d.input_closure.members[0].gp_fold_sha256.clone();
    let source = "a".repeat(40);
    // Projected by the historical staging source.
    rejected(
        verify_closure(
            &p,
            &manifest(HISTORICAL_SOURCE, &fresh(10, "raw"), &fresh(5, "fold")),
        ),
        "historical staging source",
    );
    // Reusing any projected object of the historical closure.
    let mut raw = fresh(10, "raw");
    raw[4] = historical_raw[4].clone();
    rejected(
        verify_closure(&p, &manifest(&source, &raw, &fresh(5, "fold"))),
        "historical staging path",
    );
    let mut folds = fresh(5, "fold");
    folds[2] = historical_folds[2].clone();
    rejected(
        verify_closure(&p, &manifest(&source, &fresh(10, "raw"), &folds)),
        "historical staging path",
    );
    // Positive control: an unrelated fitting-only closure passes this check.
    let clean = manifest(&source, &fresh(10, "raw"), &fresh(5, "fold"));
    let checked = verify_closure(&p, &clean).unwrap();
    assert_eq!(checked["objects_checked"], 15);
}

#[test]
fn a_corrected_fitting_only_run_gets_a_new_identity_and_its_own_window() {
    let (d, p) = committed();
    let source = "b".repeat(40);
    let bytes = manifest(
        &source,
        &fresh(10, "corrected"),
        &fresh(5, "corrected-fold"),
    );
    for window in [None, Some(digest("corrected-window"))] {
        let run = CorrectedRun::new(&p, attested(&bytes, &source), window.clone()).unwrap();
        let encoded = run.encode().unwrap();
        let accepted = verify_corrected_run(&d, &p, &encoded, &bytes).unwrap();
        assert_eq!(accepted["accepted"], true);
        assert_eq!(accepted["run_id"], run.run_id.as_str());
        assert_eq!(accepted["objects_checked"], 15);
        for other in [
            &d.disposition_id,
            &p.protocol_id,
            &d.study.window_experiment_sha256,
            &d.input_closure.manifest_sha256,
        ] {
            assert_ne!(&run.run_id, other);
        }
        assert_eq!(CorrectedRun::open(&encoded).unwrap(), run);
    }
}

#[test]
fn historical_and_corrected_documents_cannot_be_substituted() {
    let (d, p) = committed();
    // One schema per role; each document opens only as itself.
    assert!(HistoricalDisposition::open(PROTOCOL).is_err());
    assert!(CorrectedProtocol::open(DISPOSITION, &d).is_err());
    assert!(CorrectedRun::open(DISPOSITION).is_err());
    assert!(CorrectedRun::open(PROTOCOL).is_err());
    let source = "c".repeat(40);
    let bytes = manifest(&source, &fresh(10, "sub"), &fresh(5, "sub-fold"));
    let good = CorrectedRun::new(&p, attested(&bytes, &source), None).unwrap();
    // The historical window cannot be claimed.
    let run = CorrectedRun::new(
        &p,
        attested(&bytes, &source),
        Some(d.study.window_experiment_sha256.clone()),
    )
    .unwrap();
    rejected(
        verify_corrected_run(&d, &p, &run.encode().unwrap(), &bytes),
        "historical execution window",
    );
    // A run bound to the historical identity instead of the protocol.
    let mut historical = good.clone();
    historical.protocol_id = d.disposition_id.clone();
    let mut body = serde_json::to_value(&historical).unwrap();
    body.as_object_mut().unwrap().remove("run_id");
    historical.run_id =
        sha256_hex(&jope_datasets::regression_release::canonical_json(&body).unwrap());
    rejected(
        verify_corrected_run(&d, &p, &historical.encode().unwrap(), &bytes),
        "not under the corrected protocol",
    );
    // Other closure bytes than the run pins.
    let other = manifest(&source, &fresh(10, "other"), &fresh(5, "other-fold"));
    rejected(
        verify_corrected_run(&d, &p, &good.encode().unwrap(), &other),
        "pinned manifest",
    );
    // A closure that parsed outer labels is not fitting-only.
    let mut closure = attested(&bytes, &source);
    closure.attestation.outer_scoring_labels_parsed = true;
    let run = CorrectedRun::new(&p, closure, None).unwrap();
    rejected(
        verify_corrected_run(&d, &p, &run.encode().unwrap(), &bytes),
        "fitting-only projection",
    );
    // An edited run keeps no identity.
    let mut edited: Value = serde_json::from_slice(&good.encode().unwrap()).unwrap();
    edited["window_experiment_sha256"] = json!(digest("late window"));
    let edited = jope_datasets::regression_release::canonical_json(&edited).unwrap();
    assert!(CorrectedRun::open(&edited).is_err());
    // A protocol derived from another history is not this protocol.
    let mut other_history = d.clone();
    other_history.unknowns.push("another history".into());
    other_history.disposition_id = other_history.compute_id().unwrap();
    let foreign = CorrectedProtocol::derive(&other_history).unwrap();
    assert!(CorrectedProtocol::open(&foreign.encode().unwrap(), &d).is_err());
    rejected(
        verify_corrected_run(&d, &foreign, &good.encode().unwrap(), &bytes),
        "not the one derived",
    );
    verify_corrected_run(&d, &p, &good.encode().unwrap(), &bytes).unwrap();
}

// ---------------------------------------------------------------------------
// Derivation from a synthetic custody tree
// ---------------------------------------------------------------------------

struct Custody {
    _temp: tempfile::TempDir,
    study: PathBuf,
    repository: PathBuf,
}

fn write(path: &Path, bytes: &[u8]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

fn write_json(path: &Path, value: &Value) {
    write(path, &serde_json::to_vec_pretty(value).unwrap());
}

fn custody() -> Custody {
    let temp = tempfile::tempdir().unwrap();
    let study = temp.path().join("study-final");
    let repository = temp.path().join("repository");
    let source = "d".repeat(40);
    let object = |name: &str, content: &str| -> Value {
        let path = study.join(name);
        write(&path, content.as_bytes());
        json!({"path":path,"sha256":sha256_hex(content.as_bytes())})
    };
    let raw_a = object("raw-a.json", "{\"rows\":[1]}");
    let raw_b = object("raw-b.json", "{\"rows\":[2]}");
    let folds: Vec<Value> = (0..2)
        .map(|k| {
            json!([
                object(&format!("gp-a-fold-{k}.json"), &format!("fold {k}")),
                ["1"]
            ])
        })
        .collect();
    write_json(
        &study.join("retained-fitting-inputs.json"),
        &json!({"schema":FITTING_INPUTS_SCHEMA,"panel_sha256":digest("panel"),
            "configuration_sha256":digest("configuration"),"projection_source_sha":source,"seed":0,
            "assignments":{"a":{"path":"/splits/a.json","sha256":digest("split a")},
                "b":{"path":"/splits/b.json","sha256":digest("split b")}},
            "inputs":[{"id":"a","raw":raw_a,"gp":{"folds":folds}},{"id":"b","raw":raw_b,"gp":null}]}),
    );
    write_json(
        &study.join("plan.json"),
        &json!({"schema":HISTORICAL_PLAN_SCHEMA}),
    );
    write_json(&study.join("plan-command.json"), &json!(["jope", "plan"]));
    let deviation = json!({"schema":"jope.core-ip-protocol-deviation/v1","stage":"pre-window input staging",
        "source_sha":source,"finding":"outer labels parsed before projection",
        "outer_scoring_labels_parsed":true,"sealed_target_tables_opened":false,
        "outer_scoring_executed":false,"campaign_held_looks_consumed":false,
        "worker_payload_contains_outer_targets":false,"original_result_reclassified":false});
    write_json(&study.join("protocol-deviation.json"), &deviation);
    write(&study.join("run.stderr"), b"exit-code 1\n");
    let window = json!({"experiment_sha256":digest("window"),"started_unix_ms":1000,"deadline_unix_ms":2000});
    write_json(&study.join("journal/window.json"), &window);
    write_json(
        &study.join("journal/launch.json"),
        &json!({"exit_code":1,"original_window":window}),
    );
    write(&study.join("jope"), b"binary");
    let closeout = json!({"source_sha":source,"window":window,
        "status":"controller_failed_before_cuda_qualification","failure":"collision",
        "gpu_qualification_executed":false,"direct_requests":0,"full_gp_arms":0,"prediction_batches":0,
        "files":{study.join("plan.json").display().to_string():sha256_hex(&fs::read(study.join("plan.json")).unwrap()),
            study.join("run.stderr").display().to_string():sha256_hex(b"exit-code 1\n")}});
    write_json(&study.join("closeout.json"), &closeout);
    let data = repository.join("docs/internal/core-ip/data");
    write_json(&data.join("gpu-protocol-deviation.json"), &deviation);
    write_json(&data.join("gpu-closeout.json"), &closeout);
    write_json(
        &data.join("gpu-study-manifest.json"),
        &json!({"source_sha":source,"binary":{"sha256":sha256_hex(b"binary")},
            "plan":{"sha256":sha256_hex(&fs::read(study.join("plan.json")).unwrap())}}),
    );
    write_json(&data.join("gpu-plan-command.json"), &json!(["jope"]));
    write_json(
        &data.join("planner-local-proof.json"),
        &json!({"positive_plan_exit":0,"execution_started":false}),
    );
    Custody {
        _temp: temp,
        study,
        repository,
    }
}

#[test]
fn derivation_cross_checks_the_retained_evidence_and_is_deterministic() {
    let c = custody();
    let d = derive_disposition(&c.study, &c.repository).unwrap();
    let bytes = d.encode().unwrap();
    assert_eq!(
        derive_disposition(&c.study, &c.repository)
            .unwrap()
            .encode()
            .unwrap(),
        bytes
    );
    assert_eq!(HistoricalDisposition::open(&bytes).unwrap(), d);
    assert_eq!(d.input_closure.members.len(), 2);
    assert_eq!(d.input_closure.members[0].gp_fold_sha256.len(), 2);
    // manifest, 2 raw inputs, 2 folds, v2 plan, v3 local proof.
    assert_eq!(d.affected_descendants.len(), 7);
    // The corrected protocol rejects exactly this retained closure by replay.
    let p = CorrectedProtocol::derive(&d).unwrap();
    let retained = fs::read(c.study.join("retained-fitting-inputs.json")).unwrap();
    rejected(verify_closure(&p, &retained), "historical input closure");

    type Break = fn(&Custody);
    let breaks: [(&str, Break, &str); 6] = [
        (
            "missing controller stderr",
            |c| fs::remove_file(c.study.join("run.stderr")).unwrap(),
            "absent",
        ),
        (
            "changed retained projected input",
            |c| fs::write(c.study.join("raw-a.json"), b"{\"rows\":[9]}").unwrap(),
            "differs from the manifest",
        ),
        (
            "second staging source",
            |c| {
                let path = c.study.join("closeout.json");
                let mut v: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                v["source_sha"] = json!("e".repeat(40));
                write_json(&path, &v);
            },
            "more than one staging source",
        ),
        (
            "second window",
            |c| {
                let path = c.study.join("journal/launch.json");
                let mut v: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                v["original_window"]["started_unix_ms"] = json!(999);
                write_json(&path, &v);
            },
            "more than one execution window",
        ),
        (
            "published deviation differs",
            |c| {
                let path = c
                    .repository
                    .join("docs/internal/core-ip/data/gpu-protocol-deviation.json");
                let mut v: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                v["outer_scoring_labels_parsed"] = json!(false);
                write_json(&path, &v);
            },
            "differ",
        ),
        (
            "symlinked plan",
            |c| {
                let plan = c.study.join("plan.json");
                let moved = c.study.parent().unwrap().join("plan-elsewhere.json");
                fs::rename(&plan, &moved).unwrap();
                std::os::unix::fs::symlink(&moved, &plan).unwrap();
            },
            "symlink",
        ),
    ];
    for (name, damage, needle) in breaks {
        let c = custody();
        damage(&c);
        let error = format!(
            "{:#}",
            derive_disposition(&c.study, &c.repository).expect_err(name)
        );
        assert!(
            error.contains(needle),
            "{name}: expected {needle:?}: {error}"
        );
    }
}
