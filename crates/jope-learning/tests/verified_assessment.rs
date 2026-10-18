#[path = "../src/qualify/evidence.rs"]
mod evidence;
#[path = "../src/qualify/exposure.rs"]
mod exposure;

use evidence::{
    AssessmentBody, AssessmentPermission, FamilyRoster, LookAllocation, LookLedger, QualTestScore,
    RosterIssuer, SignedAssessment, ASSESSMENT_SCHEMA,
};
use exposure::{ExposureManifest, EXPOSURE_SCHEMA};
use serde::Serialize;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

const ISSUER: &str = "fixture-independent-roster-issuer";
const CAPABILITY: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

struct FixtureIssuer;

impl FixtureIssuer {
    fn sign(message: &[u8]) -> String {
        let digest = sha256(message);
        format!("{digest}{digest}")
    }
}

impl RosterIssuer for FixtureIssuer {
    fn verify_capability_signature(
        &self,
        issuer_id: &str,
        capability_sha256: &str,
        signed_message: &[u8],
        signature_hex: &str,
    ) -> Result<(), String> {
        if issuer_id != ISSUER
            || capability_sha256 != CAPABILITY
            || signature_hex != Self::sign(signed_message)
        {
            return Err("fixture issuer rejected signature or capability".into());
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
struct SharedLedger(Arc<Mutex<BTreeSet<(String, u32)>>>);

impl LookLedger for SharedLedger {
    fn reserve_once(
        &self,
        roster_decision_sha256: &str,
        selected_look: u32,
        _allocation_sha256: &str,
    ) -> Result<(), String> {
        let mut reserved = self.0.lock().map_err(|_| "fixture ledger poisoned")?;
        if reserved.insert((roster_decision_sha256.to_owned(), selected_look)) {
            Ok(())
        } else {
            Err("look was already reserved".into())
        }
    }
}

fn canonical<T: Serialize>(value: &T) -> Vec<u8> {
    serde_json::to_vec(&serde_json::to_value(value).unwrap()).unwrap()
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

fn seal(mut assessment: SignedAssessment) -> SignedAssessment {
    let mut message = b"jope.qualify-verified-assessment.signature.v1\0".to_vec();
    message.extend(canonical(&(
        assessment.issuer_id.as_str(),
        assessment.capability_sha256.as_str(),
        &assessment.body,
    )));
    assessment.signature_hex = FixtureIssuer::sign(&message);
    assessment
}

fn fixture() -> SignedAssessment {
    let roster = FamilyRoster {
        schema: "jope.qualify-adopted-family-roster.v1".into(),
        decision_sha256: "b".repeat(64),
        families: vec!["family-a".into(), "family-b".into()],
        min_families: 2,
    };
    let rows = vec![
        QualTestScore {
            family_id: "family-a".into(),
            context_id: "context-a".into(),
            incumbent_id: "incumbent-1".into(),
            incumbent_loss: 1.0,
            candidate_loss: 0.5,
        },
        QualTestScore {
            family_id: "family-b".into(),
            context_id: "context-b".into(),
            incumbent_id: "incumbent-1".into(),
            incumbent_loss: 2.0,
            candidate_loss: 1.5,
        },
    ];
    let roster_sha256 = sha256(&canonical(&roster));
    let raw_closure_sha256 = sha256(&canonical(&rows));
    let body = AssessmentBody {
        roster,
        exposure: ExposureManifest {
            schema: EXPOSURE_SCHEMA.into(),
            roster_sha256,
            evidence_sha256: raw_closure_sha256.clone(),
            intended_use: "report".into(),
            exposed_families: vec!["family-a".into(), "family-b".into()],
            held_out_families: vec![],
        },
        incumbent_id: "incumbent-1".into(),
        raw_scores: rows,
        raw_closure_sha256,
        alpha_total: 0.05,
        look_allocations: vec![LookAllocation {
            index: 1,
            alpha: 0.025,
        }],
        selected_look: 1,
        intended_use: "report".into(),
    };
    seal(SignedAssessment {
        schema: ASSESSMENT_SCHEMA.into(),
        issuer_id: ISSUER.into(),
        capability_sha256: CAPABILITY.into(),
        body,
        signature_hex: String::new(),
    })
}

#[test]
fn complete_authenticated_fixture_yields_only_report_permission_and_recomputed_scores() {
    let result = fixture()
        .verify(&FixtureIssuer, &SharedLedger::default())
        .unwrap();
    assert_eq!(result.permission, AssessmentPermission::ReportOnly);
    assert_eq!(result.incumbent_mean_loss, 1.5);
    assert_eq!(result.candidate_mean_loss, 1.0);
    assert_eq!(result.family_mean_deltas["family-a"], -0.5);
    assert_eq!(result.family_mean_deltas["family-b"], -0.5);
    assert_eq!(result.alpha, 0.025);
}

#[test]
fn forged_issuer_is_refused_even_with_nonempty_identity_and_signature() {
    let mut bad = fixture();
    bad.issuer_id = "forged-reviewer".into();
    let bad = seal(bad);
    assert!(bad
        .verify(&FixtureIssuer, &SharedLedger::default())
        .is_err());
}

#[test]
fn altered_raw_closure_roster_alpha_and_use_are_refused() {
    let issuer = FixtureIssuer;

    let mut raw = fixture();
    raw.body.raw_scores[0].candidate_loss = 0.1;
    assert!(seal(raw).verify(&issuer, &SharedLedger::default()).is_err());

    let mut roster = fixture();
    roster.body.roster.families.swap(0, 1);
    assert!(seal(roster)
        .verify(&issuer, &SharedLedger::default())
        .is_err());

    let mut alpha = fixture();
    alpha.body.alpha_total = 0.01;
    assert!(seal(alpha)
        .verify(&issuer, &SharedLedger::default())
        .is_err());

    let mut use_change = fixture();
    use_change.body.intended_use = "research_adoption".into();
    assert!(seal(use_change)
        .verify(&issuer, &SharedLedger::default())
        .is_err());
}

#[test]
fn unallocated_and_replayed_look_are_refused_across_host_clients() {
    let mut unallocated = fixture();
    unallocated.body.selected_look = 2;
    assert!(seal(unallocated)
        .verify(&FixtureIssuer, &SharedLedger::default())
        .is_err());

    let shared = SharedLedger::default();
    let host_one = shared.clone();
    let host_two = shared.clone();
    let signed = fixture();
    assert!(signed.verify(&FixtureIssuer, &host_one).is_ok());
    assert!(signed.verify(&FixtureIssuer, &host_two).is_err());
}
