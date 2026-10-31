//! Development-only exact calculations for finite discrete outcome laws.
//!
//! Exactness is conditional on an authenticated frozen law and a fully measured I29 table.
//! This module has no promotion capability and deliberately refuses unsupported sampling modes.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// `SamplingMode` is an enum for the sampling mode contract in this module: Development-only exact calculations for finite discrete outcome laws.
pub enum SamplingMode {
    /// Draws are independent, and an action may be selected more than once.
    IidWithReplacement,
    /// Each action can be selected at most once in the draw sequence.
    WithoutReplacement,
    /// Later selections may depend on earlier observations.
    Adaptive,
    /// Draws are correlated rather than independent.
    Correlated,
    /// Repeated action identities are collapsed before evaluation.
    Deduplicated,
    /// The draw set was selected through an explicit handoff.
    SelectedHandoff,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Atom` is a record for the atom contract in this module: Development-only exact calculations for finite discrete outcome laws.
pub struct Atom {
    /// Canonical measured action identity. Equal identities have their masses aggregated.
    pub action_sha256: String,
    /// Rational probability numerator over `DiscreteLaw::denominator`.
    pub mass: u64,
    /// `None` is an explicit failed draw, never a score of zero.
    pub score: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `DiscreteLaw` is a record for the discrete law contract in this module: Development-only exact calculations for finite discrete outcome laws.
pub struct DiscreteLaw {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Shared denominator used to interpret probability numerators in this law.
    pub denominator: u64,
    /// SHA-256 digest of the frozen action universe summarized by this law.
    pub universe_sha256: String,
    /// SHA-256 digest of the checkpoint bound to this evidence.
    pub checkpoint_sha256: String,
    /// SHA-256 digest identifying the evaluator used for this evidence.
    pub evaluator_sha256: String,
    /// SHA-256 digest of the receipt that authenticates this result.
    pub receipt_sha256: String,
    /// Sampling design used to produce the draws summarized here.
    pub sampling: SamplingMode,
    /// Measured outcomes and their probability masses.
    pub atoms: Vec<Atom>,
}

/// Interface for verifying law evidence before it is used.
pub trait LawVerifier {
    /// Verifies the bound evidence and returns an error when verification fails.
    fn verify(&self, law: &DiscreteLaw) -> Result<()>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// `ExactStatus` is an enum for the exact status contract in this module: Development-only exact calculations for finite discrete outcome laws.
pub enum ExactStatus {
    /// All supported outcome mass is evaluated exactly.
    Exact,
    /// Only part of the outcome mass is supported by measured outcomes.
    Partial,
    /// The input design is outside the exact evaluator’s supported cases.
    Unsupported,
    /// Required evidence is unavailable, so no exact result can be reported.
    Unavailable,
    /// Every requested draw failed to produce a measured score.
    AllDrawsFailed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// `BestAtom` is a record for the best atom contract in this module: Development-only exact calculations for finite discrete outcome laws.
pub struct BestAtom {
    /// Measured score for this outcome; absence denotes a failed draw.
    pub score: f64,
    /// Probability mass assigned to this outcome.
    pub probability: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// `ExactResult` is a record for the exact result contract in this module: Development-only exact calculations for finite discrete outcome laws.
pub struct ExactResult {
    /// Outcome or lifecycle status recorded for this value.
    pub status: ExactStatus,
    /// Number of draws included in the calculation.
    pub draws: u64,
    /// Probability mass represented by measured outcomes.
    pub covered_mass: f64,
    /// Probability mass not represented by measured outcomes.
    pub missing_mass: f64,
    /// Probability mass assigned to failed outcomes.
    pub failure_mass: f64,
    /// Probability every draw fails, retained separately from score atoms.
    pub all_draws_fail_probability: f64,
    /// Existing comparison score, when an incumbent is available.
    pub incumbent: Option<f64>,
    /// Measured outcomes and their probability masses.
    pub atoms: Vec<BestAtom>,
    /// Number of positive-probability outcomes whose computed mass underflowed.
    pub underflowed_atoms: usize,
    /// Scope label describing the population covered by this result.
    pub scope: &'static str,
}

/// Compute the distribution of the best score from B independent draws with replacement.
/// `measured_actions` is the exact action inventory authenticated by the I29 table adapter.
pub fn exact_best_of_b(
    law: &DiscreteLaw,
    measured_actions: &[String],
    draws: u64,
    incumbent: Option<f64>,
    verifier: &dyn LawVerifier,
) -> Result<ExactResult> {
    verifier.verify(law)?;
    ensure!(
        law.schema == "jope.frozen-discrete-law/v1",
        "unsupported law schema"
    );
    ensure!(law.denominator > 0, "law denominator must be positive");
    for id in [
        &law.universe_sha256,
        &law.checkpoint_sha256,
        &law.evaluator_sha256,
        &law.receipt_sha256,
    ] {
        ensure!(is_sha256(id), "law identities must be lowercase sha256");
    }
    if law.sampling != SamplingMode::IidWithReplacement {
        return Ok(empty_result(ExactStatus::Unsupported, draws, incumbent));
    }
    ensure!(
        incumbent.is_none_or(f64::is_finite),
        "incumbent must be finite"
    );
    let measured: std::collections::BTreeSet<_> = measured_actions.iter().collect();
    let mut scores = BTreeMap::<String, Option<f64>>::new();
    let mut mass_total = 0u64;
    let mut covered = 0u64;
    for atom in &law.atoms {
        ensure!(
            is_sha256(&atom.action_sha256),
            "action identity must be lowercase sha256"
        );
        ensure!(
            atom.score.is_none_or(f64::is_finite),
            "scores must be finite"
        );
        mass_total = mass_total
            .checked_add(atom.mass)
            .ok_or_else(|| anyhow::anyhow!("mass overflow"))?;
        let slot = scores
            .entry(atom.action_sha256.clone())
            .or_insert(atom.score);
        ensure!(
            *slot == atom.score,
            "duplicate action has conflicting outcomes"
        );
        if measured.contains(&atom.action_sha256) {
            covered = covered
                .checked_add(atom.mass)
                .ok_or_else(|| anyhow::anyhow!("mass overflow"))?;
        }
    }
    ensure!(
        mass_total == law.denominator,
        "declared law mass must sum exactly to denominator"
    );
    if draws == 0 {
        return Ok(empty_result(
            if incumbent.is_some() {
                ExactStatus::Exact
            } else {
                ExactStatus::Unavailable
            },
            0,
            incumbent,
        ));
    }
    let mut grouped = BTreeMap::<u64, u64>::new();
    let mut failure = 0u64;
    for atom in &law.atoms {
        if !measured.contains(&atom.action_sha256) {
            continue;
        }
        if let Some(score) = atom.score {
            // Preserve deterministic f64 ordering while grouping equal score atoms.
            *grouped.entry(score.to_bits()).or_default() += atom.mass;
        } else {
            failure += atom.mass;
        }
    }
    let mut ordered: Vec<_> = grouped
        .into_iter()
        .map(|(bits, mass)| (f64::from_bits(bits), mass))
        .collect();
    ordered.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut atoms = Vec::new();
    let mut underflowed = 0;
    let mut cumulative = failure as f64 / law.denominator as f64;
    let mut no_improvement_mass = cumulative;
    let b = draws as f64;
    for (score, mass) in ordered {
        if incumbent.is_some_and(|i| score <= i) {
            cumulative += mass as f64 / law.denominator as f64;
            no_improvement_mass = cumulative;
            continue;
        }
        let previous = cumulative;
        cumulative += mass as f64 / law.denominator as f64;
        let p = power_difference(cumulative, previous, b);
        if p == 0.0 && mass > 0 {
            underflowed += 1;
        }
        if p > 0.0 {
            atoms.push(BestAtom {
                score,
                probability: p,
            });
        }
    }
    if let Some(i) = incumbent {
        let non_improve = no_improvement_mass.powf(b);
        if non_improve > 0.0 {
            atoms.push(BestAtom {
                score: i,
                probability: non_improve,
            });
        } else {
            underflowed += 1;
        }
    }
    let full = covered == law.denominator;
    let status = if failure == law.denominator {
        ExactStatus::AllDrawsFailed
    } else if full {
        ExactStatus::Exact
    } else {
        ExactStatus::Partial
    };
    let all_draws_fail_probability = (failure as f64 / law.denominator as f64).powf(b);
    if !full {
        // No conditional-on-covered-mass distribution is computed by this API.
        atoms.clear();
    }
    Ok(ExactResult {
        status,
        draws,
        covered_mass: covered as f64 / law.denominator as f64,
        missing_mass: (law.denominator - covered) as f64 / law.denominator as f64,
        failure_mass: failure as f64 / law.denominator as f64,
        all_draws_fail_probability,
        incumbent,
        atoms,
        underflowed_atoms: underflowed,
        scope: "development_only_no_promotion_authority",
    })
}

fn power_difference(upper: f64, lower: f64, exponent: f64) -> f64 {
    if upper <= 0.0 {
        return 0.0;
    }
    let high = exponent * upper.ln();
    if lower <= 0.0 {
        return high.exp();
    }
    let ratio_log = exponent * ((lower - upper) / upper).ln_1p();
    high.exp() * -ratio_log.exp_m1()
}
fn empty_result(status: ExactStatus, draws: u64, incumbent: Option<f64>) -> ExactResult {
    ExactResult {
        status,
        draws,
        covered_mass: 0.0,
        missing_mass: 1.0,
        failure_mass: 0.0,
        all_draws_fail_probability: 0.0,
        incumbent,
        atoms: vec![],
        underflowed_atoms: 0,
        scope: "development_only_no_promotion_authority",
    }
}
fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// `CostClock` is an enum for the cost clock contract in this module: Development-only exact calculations for finite discrete outcome laws.
pub enum CostClock {
    /// Cost is used without a transform.
    Linear,
    /// Cost is transformed with `ln(1 + cost)`.
    Log1p,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// `DiscoveryAuc` is a record for the discovery AUC contract in this module: Development-only exact calculations for finite discrete outcome laws.
pub struct DiscoveryAuc {
    /// Cost transformation used to place observations on the discovery curve.
    pub clock: CostClock,
    /// Unit used for the cost values on the discovery curve.
    pub cost_unit: String,
    /// Maximum cost value included in the discovery-curve calculation.
    pub horizon: f64,
    /// Area under the discovery curve through the configured horizon.
    pub area: f64,
    /// Existing comparison score, when an incumbent is available.
    pub incumbent: f64,
    /// Ordered cost and incumbent-score observations used to form the curve.
    pub points: Vec<(f64, Option<f64>)>,
    /// Number of attempts with a measured failure disposition.
    pub failures: u64,
    /// Number of attempts whose outcomes were censored before completion.
    pub censored: u64,
    /// Interpolation rule used between observed discovery points.
    pub interpolation: &'static str,
    /// Adoption rule associated with the reported discovery result.
    pub adoption: &'static str,
}
impl DiscoveryAuc {
    /// Step-interpolated best-so-far area on a declared common horizon; diagnostic only.
    pub fn compute(
        clock: CostClock,
        cost_unit: &str,
        horizon: f64,
        incumbent: f64,
        points: &[(f64, Option<f64>)],
        failures: u64,
        censored: u64,
    ) -> Result<Self> {
        ensure!(
            !cost_unit.trim().is_empty()
                && horizon.is_finite()
                && horizon > 0.0
                && incumbent.is_finite(),
            "invalid clock declaration"
        );
        let mut sorted = points.to_vec();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        ensure!(
            sorted
                .iter()
                .all(|(x, y)| x.is_finite() && *x >= 0.0 && y.is_none_or(f64::is_finite)),
            "nonfinite cost or score"
        );
        let transform = |x: f64| match clock {
            CostClock::Linear => x,
            CostClock::Log1p => x.ln_1p(),
        };
        let end = transform(horizon);
        let mut last = 0.0;
        let mut best = incumbent;
        let mut area = 0.0;
        for (cost, score) in &sorted {
            let x = transform(*cost).min(end);
            if x > last {
                area += best * (x - last);
                last = x;
            }
            if let Some(s) = score {
                best = best.max(*s);
            }
        }
        area += best * (end - last);
        Ok(Self {
            clock,
            cost_unit: cost_unit.into(),
            horizon,
            area,
            incumbent,
            points: sorted,
            failures,
            censored,
            interpolation: "right_continuous_step_best_so_far",
            adoption: "diagnostic_only_d26_unresolved",
        })
    }
}
