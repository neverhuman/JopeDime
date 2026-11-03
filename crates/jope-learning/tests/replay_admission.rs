//! Tiny retained proof chains through the production admission and ingress APIs. Policy pins
//! are synthetic controller configuration; no fixture installs a product policy.
use jope_learning::canon::{canonical_json, canonical_sha256};
use jope_learning::ledger::Ledger;
use jope_learning::replay::admission::*;
use jope_learning::{bank, pools::source, state::State};
use jope_records::wire_contract::CanonicalPayload;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const NOW: u64 = 500;

#[path = "support/optional_dope.rs"]
mod optional_dope;

fn purposes() -> BTreeSet<Purpose> {
    [
        Purpose::GlobalPolicyTraining,
        Purpose::RepresentationLearning,
        Purpose::RetrievalIndexUpdate,
        Purpose::Distillation,
        Purpose::EpisodeFit,
        Purpose::Audit,
    ]
    .into_iter()
    .collect()
}
fn retain<T: serde::Serialize>(state: &State, value: &T) -> String {
    let payload =
        CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(value).unwrap()).unwrap();
    state.put_blob(payload.as_bytes()).unwrap()
}
fn scratch() -> PathBuf {
    let target =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/task-evidence/I04/scratch");
    std::fs::create_dir_all(&target).unwrap();
    tempfile::Builder::new()
        .prefix("replay-proof-")
        .tempdir_in(target)
        .unwrap()
        .keep()
}
fn quality() -> QualityLabel {
    QualityLabel {
        metric: "rmse".into(),
        value: ExactValue {
            f64_bits: format!("{:016x}", 0.5_f64.to_bits()),
        },
        unit: "target".into(),
        denominator: 3,
    }
}
fn receipt(state: &State, subject: &str) -> ExposureReceipt {
    ExposureReceipt {
        schema: EXPOSURE_SCHEMA.into(),
        subject_sha256: subject.into(),
        family: "fixture-family".into(),
        meta_partition: "train_search".into(),
        original_role: "train".into(),
        within_task_view: "training".into(),
        fidelity: "measured_cpu_cv".into(),
        producer: "fixture-controller-verifier".into(),
        policy_epoch: 1,
        observed_at_ms: 100,
        episode: Some("episode-a".into()),
        purposes: purposes(),
        prior_exposure: Some(vec![]),
        parents: vec![],
        observation: Observation::Completed {
            quality: Some(quality()),
            elapsed_ms: Some(13),
            policy_target_sha256: Some(state.put_blob(b"measured policy target").unwrap()),
        },
    }
}
fn policy(state: &State, receipts: &[String], epoch: u64, from: u64) -> PolicyPin {
    let policy = AdmissionPolicy {
        schema: POLICY_SCHEMA.into(),
        issuer: "fixture-controller".into(),
        epoch,
        effective_from_ms: from,
        effective_until_ms: 1000,
        evaluation_boundary_sha256: state
            .put_blob(format!("future-boundary-{epoch}-{from}").as_bytes())
            .unwrap(),
        purposes: purposes(),
        receipts: receipts.iter().cloned().collect(),
        supersedes: None,
    };
    PolicyPin {
        schema: ADOPTION_SCHEMA.into(),
        policy_sha256: retain(state, &policy),
        issuer: policy.issuer,
        epoch,
    }
}
fn bind(state: &State, pin: PolicyPin) -> anyhow::Result<AdmissionVerifier<'_>> {
    // Test controller installs the trusted configuration. Production exposes no installer and
    // no candidate-facing constructor taking a PolicyPin.
    let payload =
        CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(&pin).unwrap()).unwrap();
    let path = state.root().join(ADOPTION_PATH);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, payload.as_bytes()).unwrap();
    AdmissionVerifier::from_adopted_policy(state)
}
fn verifier<'a>(state: &'a State, receipts: &[String]) -> AdmissionVerifier<'a> {
    bind(state, policy(state, receipts, 1, 0)).unwrap()
}
fn refused(result: anyhow::Result<ReplayAdmission>, expected: &str) {
    let message = format!("{:#}", result.unwrap_err());
    println!("observed refusal: {message}");
    assert!(
        message.contains(expected),
        "expected {expected:?}: {message}"
    );
}

#[path = "replay_admission/ancestry.rs"]
mod ancestry;
#[path = "replay_admission/revision.rs"]
mod revision;
#[path = "replay_admission/train.rs"]
mod train;
