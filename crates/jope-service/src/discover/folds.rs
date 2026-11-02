//! Constraint inheritance for every CV phase (spec §5.4–5.7 and §5.10).
//! Roster selection and actual engine/worker access instrumentation are separate
//! integration work: these helpers never load labels, fit transforms or models.
use super::audit::{AuditRowId, RowAuditPhase};
use super::splits::{k_fold, rung0_subsample};
use std::collections::{BTreeMap, BTreeSet};

type IndexFold = (Vec<usize>, Vec<usize>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CvPhase {
    Rung0,
    Rung1,
    Calibration,
}
impl CvPhase {
    pub fn audit_phase(self) -> RowAuditPhase {
        match self {
            Self::Rung0 => RowAuditPhase::Rung0,
            Self::Rung1 => RowAuditPhase::Rung1,
            Self::Calibration => RowAuditPhase::Calibration,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Rung0 => "rung0",
            Self::Rung1 => "rung1",
            Self::Calibration => "calibration",
        }
    }
}

/// Aligned metadata for the FIT roster only. Group codes are caller-interned
/// identities, not targets. Gap units are rows, never unspecified time units.
#[derive(Clone, Debug, Default)]
pub struct FoldConstraints {
    pub groups: Option<Vec<u64>>,
    pub times: Option<Vec<f64>>,
    pub gap_rows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldError {
    InvalidFitRoster,
    InvalidConstraintMetadata,
    InvalidFoldCount,
    InsufficientIndependentBlocks,
    GapLeavesNoTraining,
}
impl std::fmt::Display for FoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidFitRoster=>"FIT rows must be nonempty, unique and from one dataset",
            Self::InvalidConstraintMetadata=>"constraint lengths must match FIT; timestamps must be finite; a gap needs time order",
            Self::InvalidFoldCount=>"CV needs a positive row cap and at least two folds; rung0 uses exactly two",
            Self::InsufficientIndependentBlocks=>"too few independent groups or chronological blocks for the requested folds",
            Self::GapLeavesNoTraining=>"the requested gap leaves no training rows",
        })
    }
}
impl std::error::Error for FoldError {}

/// Private vectors prevent accidental mutation of a builder's validated result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CvFold {
    train: Vec<AuditRowId>,
    validation: Vec<AuditRowId>,
}
impl CvFold {
    pub fn train(&self) -> &[AuditRowId] {
        &self.train
    }
    pub fn validation(&self) -> &[AuditRowId] {
        &self.validation
    }
}

/// All phases inherit the same immutable rows and constraints. Caps select FIT
/// positions first; every resulting subset is still split under this policy.
pub struct FoldPlan {
    rows: Vec<AuditRowId>,
    constraints: FoldConstraints,
    seed: u64,
}
impl FoldPlan {
    pub fn new(
        rows: Vec<AuditRowId>,
        constraints: FoldConstraints,
        seed: u64,
    ) -> Result<Self, FoldError> {
        if rows.is_empty()
            || rows.iter().collect::<BTreeSet<_>>().len() != rows.len()
            || rows.iter().any(|r| r.dataset != rows[0].dataset)
        {
            return Err(FoldError::InvalidFitRoster);
        }
        if constraints
            .groups
            .as_ref()
            .is_some_and(|g| g.len() != rows.len())
            || constraints
                .times
                .as_ref()
                .is_some_and(|t| t.len() != rows.len() || t.iter().any(|x| !x.is_finite()))
            || (constraints.gap_rows > 0 && constraints.times.is_none())
        {
            return Err(FoldError::InvalidConstraintMetadata);
        }
        Ok(Self {
            rows,
            constraints,
            seed,
        })
    }

    pub fn folds(
        &self,
        phase: CvPhase,
        folds: usize,
        cap: usize,
    ) -> Result<Vec<CvFold>, FoldError> {
        if folds < 2
            || (phase == CvPhase::Rung0 && folds != 2)
            || cap == 0
            || folds > self.rows.len()
        {
            return Err(FoldError::InvalidFoldCount);
        }
        let label = format!("dsc03/{}/rows", phase.label());
        let selected = rung0_subsample(
            &(0..self.rows.len()).collect::<Vec<_>>(),
            cap,
            self.seed,
            &label,
        );
        let partitions = if let Some(times) = &self.constraints.times {
            self.time_folds(selected, times, folds)?
        } else {
            self.group_folds(&selected, folds, phase)?
        };
        Ok(partitions
            .into_iter()
            .map(|(train, validation)| CvFold {
                train: train.into_iter().map(|i| self.rows[i].clone()).collect(),
                validation: validation
                    .into_iter()
                    .map(|i| self.rows[i].clone())
                    .collect(),
            })
            .collect())
    }

    /// Cross-fit calibration inside a rung fold's training roster, never its
    /// validation rows. Group/time/gap policy is inherited without asking the
    /// executor to reconstruct aligned constraint vectors. Insufficient inner
    /// data is an explicit refusal, not a fallback to unconstrained calibration.
    pub fn calibration_folds(
        &self,
        outer: &CvFold,
        folds: usize,
        cap: usize,
    ) -> Result<Vec<CvFold>, FoldError> {
        let positions: BTreeMap<_, _> = self.rows.iter().enumerate().map(|(i, r)| (r, i)).collect();
        let selected: Vec<_> = outer
            .train()
            .iter()
            .map(|row| {
                positions
                    .get(row)
                    .copied()
                    .ok_or(FoldError::InvalidFitRoster)
            })
            .collect::<Result<_, _>>()?;
        let constraints = FoldConstraints {
            groups: self
                .constraints
                .groups
                .as_ref()
                .map(|g| selected.iter().map(|&i| g[i]).collect()),
            times: self
                .constraints
                .times
                .as_ref()
                .map(|t| selected.iter().map(|&i| t[i]).collect()),
            gap_rows: self.constraints.gap_rows,
        };
        Self::new(outer.train().to_vec(), constraints, self.seed)?.folds(
            CvPhase::Calibration,
            folds,
            cap,
        )
    }

    fn group_folds(
        &self,
        selected: &[usize],
        folds: usize,
        phase: CvPhase,
    ) -> Result<Vec<IndexFold>, FoldError> {
        let mut groups: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for &row in selected {
            let group = self
                .constraints
                .groups
                .as_ref()
                .map_or(row as u64, |g| g[row]);
            groups.entry(group).or_default().push(row);
        }
        if groups.len() < folds {
            return Err(FoldError::InsufficientIndependentBlocks);
        }
        let groups: Vec<_> = groups.into_values().collect();
        let group_folds = k_fold(
            groups.len(),
            folds,
            self.seed,
            &format!("dsc03/{}/groups", phase.label()),
        )
        .map_err(|_| FoldError::InvalidFoldCount)?;
        Ok(group_folds
            .into_iter()
            .map(|partition| {
                let val: BTreeSet<_> = partition
                    .iter()
                    .flat_map(|&g| groups[g].iter().copied())
                    .collect();
                let train = selected
                    .iter()
                    .filter(|r| !val.contains(r))
                    .copied()
                    .collect();
                let validation = selected
                    .iter()
                    .filter(|r| val.contains(r))
                    .copied()
                    .collect();
                (train, validation)
            })
            .collect())
    }

    fn time_folds(
        &self,
        mut order: Vec<usize>,
        times: &[f64],
        folds: usize,
    ) -> Result<Vec<IndexFold>, FoldError> {
        order.sort_by(|&a, &b| {
            times[a]
                .partial_cmp(&times[b])
                .expect("finite timestamps")
                .then_with(|| self.rows[a].cmp(&self.rows[b]))
        });
        let mut last = BTreeMap::new();
        if let Some(groups) = &self.constraints.groups {
            for (position, &row) in order.iter().enumerate() {
                last.insert(groups[row], position);
            }
        }
        // Every boundary separates distinct times and complete groups. Overlapping
        // group-time intervals coalesce transitively rather than weakening either
        // constraint. Boundary 0 also denotes an empty training prefix.
        let mut boundaries = vec![0];
        let mut group_end = 0;
        for (position, &row) in order.iter().enumerate() {
            if let Some(groups) = &self.constraints.groups {
                group_end = group_end.max(last[&groups[row]]);
            }
            if position >= group_end
                && (position + 1 == order.len() || times[row] < times[order[position + 1]])
            {
                boundaries.push(position + 1);
            }
        }
        let blocks = boundaries.len() - 1;
        if blocks <= folds {
            return Err(FoldError::InsufficientIndependentBlocks);
        }
        let mut result = Vec::with_capacity(folds);
        for fold in 0..folds {
            let start_block = ((fold + 1) as u128 * blocks as u128 / (folds + 1) as u128) as usize;
            let end_block = ((fold + 2) as u128 * blocks as u128 / (folds + 1) as u128) as usize;
            let start = boundaries[start_block];
            let end = boundaries[end_block];
            // Only the validation boundary must separate entire tied/grouped
            // blocks. Removing earlier training rows cannot add a validation
            // group or timestamp to training, so a row gap need not discard an
            // otherwise valid warmup block in full.
            let train_end = start.saturating_sub(self.constraints.gap_rows);
            if train_end == 0 {
                return Err(FoldError::GapLeavesNoTraining);
            }
            result.push((order[..train_end].to_vec(), order[start..end].to_vec()));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
