//! Development-only exact calculations for finite discrete outcome laws.
//!
//! Exactness is conditional on an authenticated frozen law and a fully measured I29 table.
//! This module has no promotion capability and deliberately refuses unsupported sampling modes.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingMode {
    IidWithReplacement,
    WithoutReplacement,
    Adaptive,
    Correlated,
    Deduplicated,
    SelectedHandoff,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
pub struct DiscreteLaw {
    pub schema: String,
    pub denominator: u64,
    pub universe_sha256: String,
    pub checkpoint_sha256: String,
    pub evaluator_sha256: String,
    pub receipt_sha256: String,
    pub sampling: SamplingMode,
    pub atoms: Vec<Atom>,
}

pub trait LawVerifier {
    fn verify(&self, law: &DiscreteLaw) -> Result<()>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExactStatus {
    Exact,
    Partial,
    Unsupported,
    Unavailable,
    AllDrawsFailed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BestAtom {
    pub score: f64,
    pub probability: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExactResult {
    pub status: ExactStatus,
    pub draws: u64,
    pub covered_mass: f64,
    pub missing_mass: f64,
    pub failure_mass: f64,
    /// Probability every draw fails, retained separately from score atoms.
    pub all_draws_fail_probability: f64,
    pub incumbent: Option<f64>,
    pub atoms: Vec<BestAtom>,
    pub underflowed_atoms: usize,
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
pub enum CostClock {
    Linear,
    Log1p,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiscoveryAuc {
    pub clock: CostClock,
    pub cost_unit: String,
    pub horizon: f64,
    pub area: f64,
    pub incumbent: f64,
    pub points: Vec<(f64, Option<f64>)>,
    pub failures: u64,
    pub censored: u64,
    pub interpolation: &'static str,
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
