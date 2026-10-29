//! Flywheel and North Star records: TELEMETRY_SPEC rows F1–F8.
//!
//! **Elo comparability.** A Bradley–Terry fit rates one method universe. Each round's fit adds
//! that round as a contender, so the fit changes every round, and previously published ratings
//! move. The trend over rounds is therefore read **inside one snapshot**: the latest fit's
//! ratings for `ROUND_0..ROUND_k`. This is also how `ns/elo_tracker.py::check_phase_d_elo_gate`
//! is called. Snapshots with different [`EloFitId`]s are never joined into one series.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const ROUND_SCHEMA: &str = "jope.telemetry.flywheel-round/v1";
pub const ELO_SCHEMA: &str = "jope.telemetry.elo-snapshot/v1";
/// Row F3's null reason. `ns/H_BASELINE.md` names a qualified H38 candidate but states that a
/// global best of H0–H40 is unproven, so the file existing is not a validated baseline.
pub const NO_VALIDATED_BEST_H_BASELINE: &str = "no_validated_best_h_baseline";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EloSolver {
    /// `ns/elo_tracker.py::compute_family_grouped_bradley_terry_elo`: MM solver, 1/|family|
    /// weights, Laplace prior 0.01, anchor contender fixed at 1500, absolute tie margin.
    NsFamilyBt,
    /// `jope-arena::elo::fit_elo`: diagonal Newton, mean-centred at 1500, relative tie 1e-12.
    ArenaFitElo,
    /// `jope-records::elo_history` replay.
    RecordsHistory,
}

/// Everything that makes two Elo numbers comparable. Digests are hex SHA-256.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EloFitId {
    pub solver: EloSolver,
    pub contender_set_sha256: String,
    pub cohort_sha256: String,
    pub family_roster_sha256: Option<String>,
    pub fold_map_sha256: Option<String>,
    pub anchor: Option<String>,
    pub tie_rule: String,
}

impl EloFitId {
    /// Order-independent digest over the contender names.
    pub fn contender_set_digest<S: AsRef<str>>(contenders: &[S]) -> String {
        let mut v: Vec<&str> = contenders.iter().map(|s| s.as_ref()).collect();
        v.sort_unstable();
        v.dedup();
        let mut h = Sha256::new();
        for c in v {
            h.update(c.as_bytes());
            h.update(b"\n");
        }
        hex(&h.finalize())
    }

    /// One digest for the whole identity, used as the series key.
    pub fn digest(&self) -> String {
        let canon = serde_json::to_vec(self).expect("EloFitId serializes");
        hex(&Sha256::digest(canon))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Overall,
    Regression,
    BinaryClassification,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rating {
    pub elo: f64,
    pub ci95: Option<(f64, f64)>,
    pub wins: Option<u32>,
    pub ties: Option<u32>,
    pub losses: Option<u32>,
}

/// Row F2: one fit's ratings for every contender and task type.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EloSnapshot {
    pub schema: String,
    pub round: u32,
    pub fit: EloFitId,
    pub fit_digest: String,
    pub n_datasets: u32,
    pub n_families: u32,
    /// Solver build or source digest. `ns/elo_tracker.py` failed its real-data spot-check, so
    /// summaries flag snapshots from an unqualified solver.
    pub solver_version: Option<String>,
    /// Every contender has a score on every dataset of the panel.
    pub panel_complete: Option<bool>,
    /// Where family ids came from: an authoritative roster, or something weaker.
    pub family_source: Option<String>,
    pub ratings: BTreeMap<TaskType, BTreeMap<String, Rating>>,
    pub availability: Option<String>,
}

impl EloSnapshot {
    pub fn new(round: u32, fit: EloFitId, n_datasets: u32, n_families: u32) -> Self {
        let fit_digest = fit.digest();
        let availability = fit
            .family_roster_sha256
            .is_none()
            .then(|| "no_frozen_family_roster".to_owned());
        Self {
            schema: ELO_SCHEMA.into(),
            round,
            fit,
            fit_digest,
            n_datasets,
            n_families,
            solver_version: None,
            panel_complete: None,
            family_source: None,
            ratings: BTreeMap::new(),
            availability,
        }
    }
}

/// Row F4: the Phase D gate, with the same predicate as `check_phase_d_elo_gate`: at least 3
/// rounds, strictly rising on regression **and** binary, and the final round at or above the H
/// baseline on overall, regression and binary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateStatus {
    pub fit_digest: String,
    pub num_rounds: u32,
    pub regression_trending_up: bool,
    pub binary_clf_trending_up: bool,
    pub beats_h_baseline_regression: bool,
    pub beats_h_baseline_binary_clf: bool,
    pub beats_h_baseline_overall: bool,
    pub gate_pass: bool,
    pub availability: Option<String>,
}

pub fn phase_d_gate(
    snap: &EloSnapshot,
    round_names: &[&str],
    h_baseline: Option<&str>,
) -> GateStatus {
    let seq = |t: TaskType| -> Option<Vec<f64>> {
        let m = snap.ratings.get(&t)?;
        round_names
            .iter()
            .map(|r| m.get(*r).map(|x| x.elo))
            .collect()
    };
    let base = |t: TaskType| -> Option<f64> { Some(snap.ratings.get(&t)?.get(h_baseline?)?.elo) };
    let rising = |v: &Option<Vec<f64>>| {
        v.as_ref()
            .is_some_and(|v| v.len() >= 3 && v.windows(2).all(|w| w[1] > w[0]))
    };
    let beats = |v: &Option<Vec<f64>>, t| match (v.as_ref().and_then(|v| v.last()), base(t)) {
        (Some(last), Some(b)) => *last >= b,
        _ => false,
    };
    let (reg, bin, ovr) = (
        seq(TaskType::Regression),
        seq(TaskType::BinaryClassification),
        seq(TaskType::Overall),
    );
    let availability = if h_baseline.is_none() {
        Some(NO_VALIDATED_BEST_H_BASELINE.to_owned())
    } else if reg.is_none() || bin.is_none() || ovr.is_none() {
        Some("missing_round_ratings".to_owned())
    } else {
        snap.availability.clone()
    };
    let g = GateStatus {
        fit_digest: snap.fit_digest.clone(),
        num_rounds: round_names.len() as u32,
        regression_trending_up: rising(&reg),
        binary_clf_trending_up: rising(&bin),
        beats_h_baseline_regression: beats(&reg, TaskType::Regression),
        beats_h_baseline_binary_clf: beats(&bin, TaskType::BinaryClassification),
        beats_h_baseline_overall: beats(&ovr, TaskType::Overall),
        gate_pass: false,
        availability,
    };
    GateStatus {
        gate_pass: g.regression_trending_up
            && g.binary_clf_trending_up
            && g.beats_h_baseline_regression
            && g.beats_h_baseline_binary_clf
            && g.beats_h_baseline_overall,
        ..g
    }
}

/// Rows F1, F5–F8 at round close.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RoundRecord {
    pub schema: String,
    pub round: u32,
    pub generation: u32,
    pub model_ckpt_sha256: Option<String>,
    pub corpus_sha256: Option<String>,
    pub targets_sha256: Option<String>,
    /// F5: G0 seeds vs flywheel picks over the outer contexts (`flywheel_win_rate`).
    pub selfplay_win_rate: Option<f64>,
    pub selfplay_comparisons: Option<u32>,
    /// F6: `{arm}_minus_{ref}` mean and central CI, as reported by
    /// `compute_component_bootstrap_summary`.
    pub arm_contrasts: BTreeMap<String, Contrast>,
    /// E4: target-improvement deltas (lower fitness is better, so positive means improved).
    pub target_deltas: BTreeMap<String, f64>,
    pub wall_s: Option<f64>,
    pub gpu_h: Option<f64>,
    pub cpu_h: Option<f64>,
    pub toks_per_s: Option<f64>,
    pub source: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Contrast {
    pub mean: f64,
    pub ci_low: Option<f64>,
    pub ci_high: Option<f64>,
    pub ci_level: Option<f64>,
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fit() -> EloFitId {
        EloFitId {
            solver: EloSolver::NsFamilyBt,
            contender_set_sha256: EloFitId::contender_set_digest(&[
                "H", "ROUND_0", "ROUND_1", "ROUND_2",
            ]),
            cohort_sha256: "c".into(),
            family_roster_sha256: Some("r".into()),
            fold_map_sha256: Some("f".into()),
            anchor: Some("H".into()),
            tie_rule: "abs:1e-4".into(),
        }
    }

    fn rating(elo: f64) -> Rating {
        Rating {
            elo,
            ci95: None,
            wins: None,
            ties: None,
            losses: None,
        }
    }

    #[test]
    fn contender_digest_is_order_free() {
        assert_eq!(
            EloFitId::contender_set_digest(&["b", "a"]),
            EloFitId::contender_set_digest(&["a", "b", "a"])
        );
        assert_ne!(
            fit().digest(),
            EloFitId {
                tie_rule: "rel:1e-12".into(),
                ..fit()
            }
            .digest()
        );
    }

    #[test]
    fn gate_matches_ns_predicate() {
        let mut s = EloSnapshot::new(2, fit(), 10, 5);
        for (t, vals) in [
            (TaskType::Regression, [1490.0, 1500.0, 1510.0, 1500.0]),
            (
                TaskType::BinaryClassification,
                [1495.0, 1499.0, 1503.0, 1500.0],
            ),
            (TaskType::Overall, [1492.0, 1499.0, 1506.0, 1500.0]),
        ] {
            let m = s.ratings.entry(t).or_default();
            for (name, v) in ["ROUND_0", "ROUND_1", "ROUND_2", "H"].iter().zip(vals) {
                m.insert((*name).into(), rating(v));
            }
        }
        let rounds = ["ROUND_0", "ROUND_1", "ROUND_2"];
        let g = phase_d_gate(&s, &rounds, Some("H"));
        assert!(g.gate_pass, "{g:?}");
        assert!(g.availability.is_none());
        let g = phase_d_gate(&s, &rounds[..2], Some("H"));
        assert!(!g.gate_pass && !g.regression_trending_up);
        let g = phase_d_gate(&s, &rounds, None);
        assert!(!g.gate_pass);
        assert_eq!(
            g.availability.as_deref(),
            Some(NO_VALIDATED_BEST_H_BASELINE)
        );
        s.ratings
            .get_mut(&TaskType::BinaryClassification)
            .unwrap()
            .insert("ROUND_2".into(), rating(1499.0));
        assert!(
            !phase_d_gate(&s, &rounds, Some("H")).gate_pass,
            "plateau is not rising"
        );
    }

    #[test]
    fn missing_roster_is_flagged() {
        let s = EloSnapshot::new(
            0,
            EloFitId {
                family_roster_sha256: None,
                ..fit()
            },
            1,
            1,
        );
        assert_eq!(s.availability.as_deref(), Some("no_frozen_family_roster"));
    }
}
