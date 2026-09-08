//! Portability basket: per-dataset metrics + a no-regression gate. Every change must keep every dataset
//! in the basket (STS + HIGGS + OpenML-CC18 + synthetic) at or above its prior metric — so improvements
//! provably generalize beyond a single dataset rather than overfitting one benchmark.
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub enum MetricKind {
    Auc,
    R2,
    MacroAuc,
}

#[derive(Clone, Debug)]
pub struct BasketEntry {
    pub name: String,
    pub metric: f64, // task headline (AUC / R² / macro-AUC)
    pub kind: MetricKind,
    pub adv_auc: f64,      // adversarial train/holdout-shift diagnostic
    pub dataset_hash: u64, // identity: a changed hash means the dataset changed, not a regression
    pub secs: f64,
}

#[derive(Clone, Debug, Default)]
pub struct BasketReport {
    pub entries: Vec<BasketEntry>,
}
impl BasketReport {
    pub fn get(&self, name: &str) -> Option<&BasketEntry> {
        self.entries.iter().find(|e| e.name == name)
    }
}

#[derive(Debug, PartialEq)]
pub enum Cmp {
    Win,
    Tie,
    Loss,
}

/// Win/Tie/Loss of each dataset vs reference metrics, within an `eps` tie band.
pub fn win_loss(cur: &BasketReport, refs: &HashMap<String, f64>, eps: f64) -> Vec<(String, Cmp)> {
    cur.entries
        .iter()
        .map(|e| {
            let c = match refs.get(&e.name) {
                Some(&r) if e.metric > r + eps => Cmp::Win,
                Some(&r) if e.metric < r - eps => Cmp::Loss,
                _ => Cmp::Tie,
            };
            (e.name.clone(), c)
        })
        .collect()
}

#[derive(Debug, PartialEq)]
pub enum GateOutcome {
    Pass,
    /// (name, prev_metric, cur_metric) for each dataset that regressed.
    Regressed(Vec<(String, f64, f64)>),
}

/// FAIL if any dataset's metric drops more than `eps` vs the prior baseline AND its `dataset_hash` is
/// unchanged (a changed hash means the dataset itself changed — not a regression to attribute to code).
pub fn no_regression_gate(prev: &BasketReport, cur: &BasketReport, eps: f64) -> GateOutcome {
    let mut bad = Vec::new();
    for pe in &prev.entries {
        if let Some(ce) = cur.get(&pe.name) {
            if ce.dataset_hash == pe.dataset_hash && ce.metric < pe.metric - eps {
                bad.push((pe.name.clone(), pe.metric, ce.metric));
            }
        }
    }
    if bad.is_empty() {
        GateOutcome::Pass
    } else {
        GateOutcome::Regressed(bad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn e(name: &str, metric: f64, hash: u64) -> BasketEntry {
        BasketEntry {
            name: name.into(),
            metric,
            kind: MetricKind::Auc,
            adv_auc: 0.5,
            dataset_hash: hash,
            secs: 0.0,
        }
    }
    #[test]
    fn gate_catches_real_regression_only() {
        let prev = BasketReport {
            entries: vec![e("sts", 0.70, 1), e("higgs", 0.69, 2)],
        };
        // a real drop on sts (same hash) → fail
        let bad = BasketReport {
            entries: vec![e("sts", 0.68, 1), e("higgs", 0.69, 2)],
        };
        assert!(
            matches!(no_regression_gate(&prev, &bad, 0.002), GateOutcome::Regressed(v) if v.len() == 1)
        );
        // improvement / tie → pass
        let ok = BasketReport {
            entries: vec![e("sts", 0.705, 1), e("higgs", 0.691, 2)],
        };
        assert_eq!(no_regression_gate(&prev, &ok, 0.002), GateOutcome::Pass);
        // a big drop but the dataset CHANGED (different hash) → not a regression
        let changed = BasketReport {
            entries: vec![e("sts", 0.50, 99), e("higgs", 0.69, 2)],
        };
        assert_eq!(
            no_regression_gate(&prev, &changed, 0.002),
            GateOutcome::Pass
        );
    }
    #[test]
    fn win_loss_bands() {
        let cur = BasketReport {
            entries: vec![e("a", 0.71, 1), e("b", 0.60, 2), e("c", 0.65, 3)],
        };
        let refs: HashMap<String, f64> =
            [("a".into(), 0.70), ("b".into(), 0.62), ("c".into(), 0.6501)].into();
        let wl = win_loss(&cur, &refs, 0.002);
        assert_eq!(wl[0].1, Cmp::Win);
        assert_eq!(wl[1].1, Cmp::Loss);
        assert_eq!(wl[2].1, Cmp::Tie);
    }
}
