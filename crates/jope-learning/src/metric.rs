//! The metric-direction table (`jope_task_elo.rs:31-36`,
//! `8a55c4fef:python/src/jope/tools/promotion_gate.py:22-27`, archived): r2, auc,
//! macro_f1 and negative_mase improve upwards; rmse, logloss and mase improve downwards.
//! `oof_r2` (genebank's cross-validated r2) is the one addition, upward like r2.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(
    doc,
    doc = "`Metric` is an enum for the metric contract in this module: The metric-direction table (jopetaskelo.rs:31-36, 8a55c4fef:python/src/jope/tools/promotiongate.py:22-27, archived): r2, auc, macrof1 and negativemase improve upwards; rmse, logloss and mase improve downwards. oofr2."
)]
pub enum Metric {
    #[cfg_attr(doc, doc = "The `R2` case of `Metric`.")]
    R2,
    #[cfg_attr(doc, doc = "The `OofR2` case of `Metric`.")]
    OofR2,
    #[cfg_attr(doc, doc = "The `Auc` case of `Metric`.")]
    Auc,
    #[cfg_attr(doc, doc = "The `MacroF1` case of `Metric`.")]
    MacroF1,
    #[cfg_attr(doc, doc = "The `NegativeMase` case of `Metric`.")]
    NegativeMase,
    #[cfg_attr(doc, doc = "The `Rmse` case of `Metric`.")]
    Rmse,
    #[cfg_attr(doc, doc = "The `Logloss` case of `Metric`.")]
    Logloss,
    #[cfg_attr(doc, doc = "The `Mase` case of `Metric`.")]
    Mase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(
    doc,
    doc = "`Direction` is an enum for the direction contract in this module: The metric-direction table (jopetaskelo.rs:31-36, 8a55c4fef:python/src/jope/tools/promotiongate.py:22-27, archived): r2, auc, macrof1 and negativemase improve upwards; rmse, logloss and mase improve downwards. oofr2."
)]
pub enum Direction {
    #[cfg_attr(doc, doc = "The `Higher` case of `Direction`.")]
    Higher,
    #[cfg_attr(doc, doc = "The `Lower` case of `Direction`.")]
    Lower,
}

impl Metric {
    #[cfg_attr(doc, doc = "Constant identifying the all.")]
    pub const ALL: [Metric; 8] = [
        Metric::R2,
        Metric::OofR2,
        Metric::Auc,
        Metric::MacroF1,
        Metric::NegativeMase,
        Metric::Rmse,
        Metric::Logloss,
        Metric::Mase,
    ];

    #[cfg_attr(doc, doc = "Performs the direction operation.")]
    pub fn direction(self) -> Direction {
        match self {
            Metric::R2 | Metric::OofR2 | Metric::Auc | Metric::MacroF1 | Metric::NegativeMase => {
                Direction::Higher
            }
            Metric::Rmse | Metric::Logloss | Metric::Mase => Direction::Lower,
        }
    }

    #[cfg_attr(doc, doc = "Returns the stable string representation of this value.")]
    pub fn as_str(self) -> &'static str {
        match self {
            Metric::R2 => "r2",
            Metric::OofR2 => "oof_r2",
            Metric::Auc => "auc",
            Metric::MacroF1 => "macro_f1",
            Metric::NegativeMase => "negative_mase",
            Metric::Rmse => "rmse",
            Metric::Logloss => "logloss",
            Metric::Mase => "mase",
        }
    }

    #[cfg_attr(
        doc,
        doc = "Parses the supplied representation, returning an error when the input is invalid."
    )]
    pub fn parse(s: &str) -> Option<Metric> {
        Metric::ALL.into_iter().find(|m| m.as_str() == s)
    }
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Direction {
    /// Strict improvement only: an equal score is never an improvement (`champions.rs:452`).
    pub fn strictly_improves(self, candidate: f64, incumbent: f64) -> bool {
        if !candidate.is_finite() {
            return false;
        }
        match self {
            Direction::Higher => candidate > incumbent,
            Direction::Lower => candidate < incumbent,
        }
    }

    #[cfg_attr(doc, doc = "Returns the stable string representation of this value.")]
    pub fn as_str(self) -> &'static str {
        match self {
            Direction::Higher => "higher",
            Direction::Lower => "lower",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_table() {
        for (name, dir) in [
            ("r2", Direction::Higher),
            ("oof_r2", Direction::Higher),
            ("auc", Direction::Higher),
            ("macro_f1", Direction::Higher),
            ("negative_mase", Direction::Higher),
            ("rmse", Direction::Lower),
            ("logloss", Direction::Lower),
            ("mase", Direction::Lower),
        ] {
            let m = Metric::parse(name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(m.direction(), dir, "{name}");
            assert_eq!(m.as_str(), name);
            assert_eq!(serde_json::to_value(m).unwrap(), name);
            assert_eq!(serde_json::from_value::<Metric>(name.into()).unwrap(), m);
        }
        assert_eq!(Metric::parse("accuracy"), None);
        assert_eq!(Metric::ALL.len(), 8);
    }

    #[test]
    fn strictness() {
        assert!(Direction::Lower.strictly_improves(0.5, 0.6));
        assert!(!Direction::Lower.strictly_improves(0.6, 0.6));
        assert!(!Direction::Lower.strictly_improves(0.7, 0.6));
        assert!(Direction::Higher.strictly_improves(0.7, 0.6));
        assert!(!Direction::Higher.strictly_improves(0.6, 0.6));
        assert!(!Direction::Higher.strictly_improves(f64::NAN, 0.0));
        assert!(!Direction::Lower.strictly_improves(f64::NEG_INFINITY, 0.0));
    }
}
