//! Lossless version-four measurement records for fault-aware re-derivation.

use anyhow::{ensure, Context, Result};
use feat_math::{
    evolve::FaultPolicy,
    kernel::admission::FaultTotals,
    math_expr::{FaultClass, FaultSeverity},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{search::PolicyRecord, Kernel2Observation};
use crate::search::{self, Observation, ObservationRecord};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Faults {
    pub evaluated_cells: u64,
    pub domain_cells: u64,
    pub saturation_cells: u64,
    pub constant_passes: u64,
    pub hard_rows: u64,
    pub hard_classes: Vec<String>,
}

impl From<FaultTotals> for Faults {
    fn from(totals: FaultTotals) -> Self {
        Self {
            evaluated_cells: totals.evaluated_cells,
            domain_cells: totals.domain_cells,
            saturation_cells: totals.saturation_cells,
            constant_passes: totals.constant_passes,
            hard_rows: totals.hard_rows,
            hard_classes: totals
                .hard_classes
                .iter()
                .map(|c| c.name().into())
                .collect(),
        }
    }
}

impl Faults {
    fn totals(&self) -> Result<FaultTotals> {
        ensure!(
            self.domain_cells <= self.evaluated_cells
                && self.saturation_cells <= self.evaluated_cells
                && (self.constant_passes == 0 || self.domain_cells > 0),
            "invalid fault counts"
        );
        let mut totals = FaultTotals::default();
        totals.evaluated_cells = self.evaluated_cells;
        totals.domain_cells = self.domain_cells;
        totals.saturation_cells = self.saturation_cells;
        totals.constant_passes = self.constant_passes;
        totals.hard_rows = self.hard_rows;
        for name in &self.hard_classes {
            let class = FaultClass::ALL
                .iter()
                .copied()
                .find(|c| c.name() == name)
                .context("unknown hard fault class")?;
            ensure!(
                class.severity() == FaultSeverity::Hard
                    && !totals.hard_classes.contains(class.set()),
                "invalid hard fault class"
            );
            totals.hard_classes = totals.hard_classes.union(class.set());
        }
        ensure!(
            (self.hard_rows > 0) == totals.is_hard(),
            "hard rows/classes disagree"
        );
        Ok(totals)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    pub observation: ObservationRecord,
    pub faults: Faults,
    pub domain_fault_rate: f64,
    pub saturation_fault_rate: f64,
    pub constant_output: bool,
    pub faulted_cells: u64,
    pub selection_score: Option<f64>,
}

impl Measurement {
    pub fn record(value: &Kernel2Observation) -> Result<Self> {
        Ok(Self {
            observation: serde_json::from_value(serde_json::to_value(&value.observation)?)?,
            faults: value.faults.into(),
            domain_fault_rate: value.domain_fault_rate,
            saturation_fault_rate: value.saturation_fault_rate,
            constant_output: value.constant_output,
            faulted_cells: value.faulted_cells,
            selection_score: value.selection_score,
        })
    }

    pub fn derive(&self, policy: FaultPolicy) -> Result<Kernel2Observation> {
        let r = &self.observation;
        let origin = [
            search::BOOTSTRAP,
            search::INCUMBENT,
            "random",
            "mutate",
            "crossover",
            super::REFINE,
            // laneD:B20-rewrite-simplify
            super::REWRITE,
            // laneD:B19-minimax-linearize
            super::LINEARIZE,
        ]
        .into_iter()
        .find(|s| *s == r.origin)
        .context("unknown kernel2 origin")?;
        let status = search::STATUSES
            .into_iter()
            .chain([super::INADMISSIBLE])
            .find(|s| *s == r.status)
            .context("unknown kernel2 status")?;
        ensure!(
            r.program_sha256 == crate::sha256_hex(r.genome.as_bytes())
                && r.island.is_none()
                && r.seconds.is_finite()
                && r.seconds >= 0.0,
            "invalid observation identity or timing"
        );
        let faults = self.faults.totals()?;
        ensure!(
            self.domain_fault_rate.to_bits() == faults.domain_rate().to_bits()
                && self.saturation_fault_rate.to_bits() == faults.saturation_rate().to_bits()
                && self.constant_output == (faults.constant_passes > 0)
                && self.faulted_cells == faults.faulted_cells(),
            "fault telemetry does not re-derive"
        );
        if status == search::MEASURED {
            ensure!(
                r.failure.is_none()
                    && !faults.is_hard()
                    && [r.train_r2, r.hold_r2, r.rmse]
                        .into_iter()
                        .all(|s| s.is_some_and(f64::is_finite))
                    && r.rmse.is_some_and(|e| e >= 0.0),
                "invalid measured observation"
            );
            let hold = r.hold_r2.unwrap();
            let score = match policy {
                FaultPolicy::RejectHard => Some(hold),
                FaultPolicy::Penalize {
                    lambda_domain,
                    lambda_saturation,
                    ..
                } => faults.penalize(hold, lambda_domain, lambda_saturation),
                FaultPolicy::Legacy => None,
            };
            ensure!(
                score.is_some()
                    && self.selection_score.map(f64::to_bits) == score.map(f64::to_bits),
                "selection score does not re-derive"
            );
        } else {
            ensure!(
                r.failure.as_ref().is_some_and(|s| !s.is_empty())
                    && [r.train_r2, r.hold_r2, r.rmse, self.selection_score]
                        .into_iter()
                        .all(|s| s.is_none()),
                "refusal carries scores or lacks a reason"
            );
        }
        Ok(Kernel2Observation {
            observation: Observation {
                genome: r.genome.clone(),
                program_sha256: r.program_sha256.clone(),
                cut: r.cut,
                generation: r.generation,
                origin,
                status,
                train_r2: r.train_r2,
                hold_r2: r.hold_r2,
                rmse: r.rmse,
                seconds: r.seconds,
                failure: r.failure.clone(),
                island: None,
            },
            faults,
            domain_fault_rate: self.domain_fault_rate,
            saturation_fault_rate: self.saturation_fault_rate,
            constant_output: self.constant_output,
            faulted_cells: self.faulted_cells,
            selection_score: self.selection_score,
        })
    }
}

impl PolicyRecord {
    pub fn policy(&self) -> Result<FaultPolicy> {
        let policy = match *self {
            Self::RejectHard => FaultPolicy::RejectHard,
            Self::Penalize {
                lambda_domain,
                lambda_saturation,
                tie_break,
            } => FaultPolicy::Penalize {
                lambda_domain,
                lambda_saturation,
                tie_break,
            },
        };
        policy.validate().map_err(anyhow::Error::msg)?;
        Ok(policy)
    }
}
