//! Draft v1 authenticated input adapter for sealed-family qualification.
//!
//! A successful verification emits a report-only assessment. Concrete issuer and shared
//! cross-host ledger adapters are integration seams for accepted I04/I41 authority; this module
//! neither creates a seal nor changes the legacy diagnostic or serving gates.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use super::exposure::{distinct_nonempty, is_hex64, sha256_hex, ExposureManifest, EXPOSURE_SCHEMA};

pub const ASSESSMENT_SCHEMA: &str = "jope.qualify-verified-assessment.v1";
const SIGNING_DOMAIN: &[u8] = b"jope.qualify-verified-assessment.signature.v1\0";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyRoster {
    pub schema: String,
    pub decision_sha256: String,
    pub families: Vec<String>,
    pub min_families: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualTestScore {
    pub family_id: String,
    pub context_id: String,
    pub incumbent_id: String,
    pub incumbent_loss: f64,
    pub candidate_loss: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LookAllocation {
    pub index: u32,
    pub alpha: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AssessmentBody {
    pub roster: FamilyRoster,
    pub exposure: ExposureManifest,
    pub incumbent_id: String,
    pub raw_scores: Vec<QualTestScore>,
    pub raw_closure_sha256: String,
    pub alpha_total: f64,
    pub look_allocations: Vec<LookAllocation>,
    pub selected_look: u32,
    pub intended_use: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedAssessment {
    pub schema: String,
    pub issuer_id: String,
    pub capability_sha256: String,
    pub body: AssessmentBody,
    pub signature_hex: String,
}

/// The accepted I04 issuer adapter must verify a real signed capability and its exact message.
/// Implementations supplied by tests or callers are not production authority by themselves.
pub trait RosterIssuer {
    fn verify_capability_signature(
        &self,
        issuer_id: &str,
        capability_sha256: &str,
        signed_message: &[u8],
        signature_hex: &str,
    ) -> Result<(), String>;
}

/// The accepted shared authority must atomically fence duplicate reservations across hosts.
/// Implementations backed only by process-local memory do not satisfy that contract.
pub trait LookLedger {
    fn reserve_once(
        &self,
        roster_decision_sha256: &str,
        selected_look: u32,
        allocation_sha256: &str,
    ) -> Result<(), String>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedAssessment {
    pub assessment_sha256: String,
    pub roster_sha256: String,
    pub exposure_sha256: String,
    pub raw_closure_sha256: String,
    pub selected_look: u32,
    pub alpha: f64,
    pub incumbent_mean_loss: f64,
    pub candidate_mean_loss: f64,
    pub family_mean_deltas: BTreeMap<String, f64>,
    /// This adapter deliberately never returns research-adoption or serving permission.
    pub permission: AssessmentPermission,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssessmentPermission {
    ReportOnly,
}

impl SignedAssessment {
    /// Validate all content, authenticate the issuer, and atomically reserve this efficacy look.
    pub fn verify<I: RosterIssuer, L: LookLedger>(
        &self,
        issuer: &I,
        ledger: &L,
    ) -> Result<VerifiedAssessment, String> {
        if self.schema != ASSESSMENT_SCHEMA {
            return Err("unsupported assessment schema".into());
        }
        if self.issuer_id.trim().is_empty()
            || !is_hex64(&self.capability_sha256)
            || !is_hex128(&self.signature_hex)
        {
            return Err("issuer, capability digest or signature is malformed".into());
        }
        let payload_bytes = canonical_json(&(
            self.issuer_id.as_str(),
            self.capability_sha256.as_str(),
            &self.body,
        ))?;
        let mut signed_message = SIGNING_DOMAIN.to_vec();
        signed_message.extend_from_slice(&payload_bytes);
        issuer.verify_capability_signature(
            &self.issuer_id,
            &self.capability_sha256,
            &signed_message,
            &self.signature_hex,
        )?;

        let body = &self.body;
        if body.intended_use != "report" {
            return Err("only report permission is available; adoption remains held".into());
        }
        if body.incumbent_id.trim().is_empty() {
            return Err("incumbent ID is empty".into());
        }
        if body.roster.schema != "jope.qualify-adopted-family-roster.v1"
            || !is_hex64(&body.roster.decision_sha256)
            || body.roster.min_families == 0
        {
            return Err("roster lacks a valid governing decision or minimum family rule".into());
        }
        let roster_families = distinct_nonempty("roster families", &body.roster.families)?;
        if roster_families.len() < body.roster.min_families {
            return Err("roster has fewer families than its declared minimum".into());
        }
        let roster_sha256 = sha256_hex(&canonical_json(&body.roster)?);
        if body.exposure.schema != EXPOSURE_SCHEMA
            || body.exposure.roster_sha256 != roster_sha256
            || body.exposure.intended_use != body.intended_use
        {
            return Err("exposure is not bound to this roster and intended use".into());
        }
        let exposure_sha256 = body.exposure.validate()?;
        let exposed = distinct_nonempty("exposed families", &body.exposure.exposed_families)?;
        if exposed != roster_families {
            return Err("exposure family set differs from the adopted roster".into());
        }

        let alpha = selected_alpha(body)?;
        if !is_hex64(&body.raw_closure_sha256) {
            return Err("raw closure digest is malformed".into());
        }
        let raw_bytes = canonical_json(&body.raw_scores)?;
        let raw_closure_sha256 = sha256_hex(&raw_bytes);
        if raw_closure_sha256 != body.raw_closure_sha256
            || body.exposure.evidence_sha256 != raw_closure_sha256
        {
            return Err("raw score closure digest does not match the signed rows".into());
        }
        let (incumbent_mean_loss, candidate_mean_loss, family_mean_deltas) = recompute(
            &body.raw_scores,
            &roster_families,
            body.roster.min_families,
            &body.incumbent_id,
        )?;

        let allocation = body
            .look_allocations
            .iter()
            .find(|look| look.index == body.selected_look)
            .ok_or_else(|| "selected look has no declared allocation".to_string())?;
        let allocation_sha256 = sha256_hex(&canonical_json(allocation)?);
        ledger.reserve_once(
            &body.roster.decision_sha256,
            body.selected_look,
            &allocation_sha256,
        )?;

        Ok(VerifiedAssessment {
            assessment_sha256: sha256_hex(&canonical_json(self)?),
            roster_sha256,
            exposure_sha256,
            raw_closure_sha256,
            selected_look: body.selected_look,
            alpha,
            incumbent_mean_loss,
            candidate_mean_loss,
            family_mean_deltas,
            permission: AssessmentPermission::ReportOnly,
        })
    }
}

fn selected_alpha(body: &AssessmentBody) -> Result<f64, String> {
    if !body.alpha_total.is_finite() || body.alpha_total <= 0.0 || body.alpha_total > 1.0 {
        return Err("alpha_total must be finite and in (0, 1]".into());
    }
    let mut seen = BTreeSet::new();
    let mut sum = 0.0;
    let mut selected = None;
    for look in &body.look_allocations {
        if !seen.insert(look.index) || !look.alpha.is_finite() || look.alpha <= 0.0 {
            return Err(
                "look allocations must have unique indices and positive finite alpha".into(),
            );
        }
        sum += look.alpha;
        if look.index == body.selected_look {
            selected = Some(look.alpha);
        }
    }
    if sum > body.alpha_total + 1e-12 {
        return Err("look allocations exceed alpha_total".into());
    }
    selected.ok_or_else(|| "selected look has no declared allocation".into())
}

fn recompute(
    rows: &[QualTestScore],
    roster: &BTreeSet<&str>,
    min_families: usize,
    incumbent_id: &str,
) -> Result<(f64, f64, BTreeMap<String, f64>), String> {
    let mut contexts = BTreeSet::new();
    let mut family_losses: BTreeMap<&str, (f64, f64, usize)> = BTreeMap::new();
    for row in rows {
        if !roster.contains(row.family_id.as_str())
            || row.context_id.trim().is_empty()
            || row.incumbent_id != incumbent_id
            || !row.incumbent_loss.is_finite()
            || !row.candidate_loss.is_finite()
            || row.incumbent_loss < 0.0
            || row.candidate_loss < 0.0
        {
            return Err(
                "raw row is outside the selected incumbent/roster or has invalid metrics".into(),
            );
        }
        if !contexts.insert((row.family_id.as_str(), row.context_id.as_str())) {
            return Err("raw closure contains a duplicate family/context row".into());
        }
        let entry = family_losses.entry(&row.family_id).or_default();
        entry.0 += row.incumbent_loss;
        entry.1 += row.candidate_loss;
        entry.2 += 1;
    }
    if family_losses.len() < min_families || family_losses.len() != roster.len() {
        return Err("raw closure is incomplete or below the adopted minimum family count".into());
    }
    let mut deltas = BTreeMap::new();
    for (family, (incumbent, candidate, n)) in &family_losses {
        deltas.insert(
            (*family).to_string(),
            candidate / *n as f64 - incumbent / *n as f64,
        );
    }
    let n = family_losses.len() as f64;
    let incumbent_mean = family_losses
        .values()
        .map(|v| v.0 / v.2 as f64)
        .sum::<f64>()
        / n;
    let candidate_mean = family_losses
        .values()
        .map(|v| v.1 / v.2 as f64)
        .sum::<f64>()
        / n;
    Ok((incumbent_mean, candidate_mean, deltas))
}

fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    let value = serde_json::to_value(value).map_err(|e| e.to_string())?;
    serde_json::to_vec(&value).map_err(|e| e.to_string())
}

fn is_hex128(value: &str) -> bool {
    value.len() == 128
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
