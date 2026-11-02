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
    include_bytes!("../../releases/staging-disposition/historical-gpu-staging-20260925.json");
const PROTOCOL: &[u8] =
    include_bytes!("../../releases/staging-disposition/fitting-only-staging-protocol-v1.json");
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
        std::slice::from_ref(&d.study.window_experiment_sha256)
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

/// W1-T07: the producer emits the corrected-run attestation for exactly the
/// manifest bytes it projected. It cannot attest a historical closure, and a
/// declaration of parsed outer labels yields a run that never verifies.
#[test]
fn the_producer_attests_exactly_its_fitting_only_closure() {
    let (d, p) = committed();
    let source = "d".repeat(40);
    let bytes = manifest(&source, &fresh(10, "attest"), &fresh(5, "attest-fold"));
    let fitting_only = attested(&bytes, &source).attestation;
    let run = attest_corrected_run(&p, &bytes, fitting_only.clone(), None).unwrap();
    assert_eq!(run.closure.manifest_sha256, sha256_hex(&bytes));
    assert_eq!(run.closure.projection_source_sha, source);
    assert_eq!(run.protocol_id, p.protocol_id);
    let encoded = run.encode().unwrap();
    assert_eq!(
        verify_corrected_run(&d, &p, &encoded, &bytes).unwrap()["accepted"],
        true
    );
    // Another manifest is not the attested one.
    let other = manifest(&source, &fresh(10, "other"), &fresh(5, "other-fold"));
    rejected(
        verify_corrected_run(&d, &p, &encoded, &other),
        "not the run's pinned manifest",
    );
    // A historical closure cannot be attested.
    let historical = manifest(
        HISTORICAL_SOURCE,
        &fresh(10, "hist"),
        &fresh(5, "hist-fold"),
    );
    rejected(
        attest_corrected_run(&p, &historical, fitting_only.clone(), None).map(|_| Value::Null),
        "historical staging source",
    );
    // A producer that declares parsed outer labels gets a run that never verifies.
    let mut parsed = fitting_only;
    parsed.outer_scoring_labels_parsed = true;
    let declared = attest_corrected_run(&p, &bytes, parsed, None).unwrap();
    rejected(
        verify_corrected_run(&d, &p, &declared.encode().unwrap(), &bytes),
        "fitting-only projection",
    );
}

/// The producer-side CLI (`staging-disposition attest`) writes the run once;
/// `staging-disposition verify --run` accepts it for exactly that closure.
#[test]
fn the_attest_verb_emits_a_run_the_verifier_accepts() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    let source = "e".repeat(40);
    let closure = dir.join("fitting-inputs.json");
    fs::write(
        &closure,
        manifest(&source, &fresh(10, "cli"), &fresh(5, "cli-fold")),
    )
    .unwrap();
    let run = dir.join("run.json");
    let cli = |args: &[&std::ffi::OsStr]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_jope-datasets"))
            .arg("staging-disposition")
            .args(args)
            .output()
            .unwrap()
    };
    let attest = |output: &Path| {
        cli(&[
            "attest".as_ref(),
            "--closure".as_ref(),
            closure.as_os_str(),
            "--output".as_ref(),
            output.as_os_str(),
        ])
    };
    let emitted = attest(&run);
    assert!(
        emitted.status.success(),
        "{}",
        String::from_utf8_lossy(&emitted.stderr)
    );
    let receipt: Value = serde_json::from_slice(&emitted.stdout).unwrap();
    let written = fs::read(&run).unwrap();
    assert_eq!(receipt["run_sha256"], sha256_hex(&written));
    assert_eq!(
        CorrectedRun::open(&written)
            .unwrap()
            .closure
            .manifest_sha256,
        sha256_hex(&fs::read(&closure).unwrap())
    );
    // The run is immutable: a second emission is refused.
    assert!(!attest(&run).status.success());
    let root = repository_root();
    let disposition = root.join(
        "crates/jope-datasets/releases/staging-disposition/historical-gpu-staging-20260925.json",
    );
    let protocol = root.join(
        "crates/jope-datasets/releases/staging-disposition/fitting-only-staging-protocol-v1.json",
    );
    let verified = cli(&[
        "verify".as_ref(),
        "--disposition".as_ref(),
        disposition.as_os_str(),
        "--protocol".as_ref(),
        protocol.as_os_str(),
        "--closure".as_ref(),
        closure.as_os_str(),
        "--run".as_ref(),
        run.as_os_str(),
    ]);
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stdout)
    );
    // A historical closure is refused before anything is written.
    let historical = dir.join("historical.json");
    fs::write(
        &historical,
        manifest(HISTORICAL_SOURCE, &fresh(10, "h"), &fresh(5, "h-fold")),
    )
    .unwrap();
    let refused_run = dir.join("refused.json");
    let refused = cli(&[
        "attest".as_ref(),
        "--closure".as_ref(),
        historical.as_os_str(),
        "--output".as_ref(),
        refused_run.as_os_str(),
    ]);
    assert!(!refused.status.success());
    assert!(!refused_run.exists());
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

#[path = "custody.rs"]
mod custody;
