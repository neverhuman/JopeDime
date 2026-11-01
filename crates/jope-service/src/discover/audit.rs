//! Run-local row-access hooks for spec §5.4/gate S3.
//! Executors must call a hook at each actual data-access boundary. Constructing
//! folds alone records no access and does not qualify an uninstrumented engine.
use jope_api::ids::DatasetId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Matrix ordinals are dataset-local; TEST row 0 is not training row 0.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AuditRowId {
    pub dataset: DatasetId,
    pub row: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RowAuditPhase {
    Landmarks,
    Rung0,
    Rung1,
    Preprocessing,
    Calibration,
    FinalizationFit,
    HoldoutScore,
    FinalizationRefit,
    TestScore,
}
impl RowAuditPhase {
    fn search(self) -> bool {
        matches!(
            self,
            Self::Landmarks | Self::Rung0 | Self::Rung1 | Self::Preprocessing | Self::Calibration
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowAuditError {
    InvalidRosters,
    EmptyAccess,
    PhaseNotAllowed,
    ForbiddenRow,
}
impl std::fmt::Display for RowAuditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidRosters => "FIT must be nonempty; row rosters must be unique and disjoint",
            Self::EmptyAccess => "an empty access is not row-touch evidence",
            Self::PhaseNotAllowed => "phase is not permitted at this finalization boundary",
            Self::ForbiddenRow => "row is outside the phase's authorized roster",
        })
    }
}
impl std::error::Error for RowAuditError {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RowTouch {
    pub phase: RowAuditPhase,
    /// Preserve caller order and repeated touches, rather than inventing counts.
    pub rows: Vec<AuditRowId>,
}

/// Append-only evidence and a one-way boundary; never deserialize as authority.
#[derive(Debug, Serialize)]
pub struct RowAudit {
    fit: BTreeSet<AuditRowId>,
    holdout: BTreeSet<AuditRowId>,
    test: BTreeSet<AuditRowId>,
    finalizing: bool,
    entries: Vec<RowTouch>,
}
impl RowAudit {
    pub fn new(
        fit: Vec<AuditRowId>,
        holdout: Vec<AuditRowId>,
        test: Vec<AuditRowId>,
    ) -> Result<Self, RowAuditError> {
        let count = fit
            .len()
            .checked_add(holdout.len())
            .and_then(|n| n.checked_add(test.len()))
            .ok_or(RowAuditError::InvalidRosters)?;
        let unique: BTreeSet<_> = fit.iter().chain(&holdout).chain(&test).collect();
        if fit.is_empty() || unique.len() != count {
            return Err(RowAuditError::InvalidRosters);
        }
        Ok(Self {
            fit: fit.into_iter().collect(),
            holdout: holdout.into_iter().collect(),
            test: test.into_iter().collect(),
            finalizing: false,
            entries: Vec::new(),
        })
    }

    pub fn begin_finalization(&mut self) {
        self.finalizing = true;
    }

    pub fn entries(&self) -> &[RowTouch] {
        &self.entries
    }

    /// Record the requested rows immediately before access. Validation is atomic:
    /// an invalid row anywhere in the batch leaves the log unchanged.
    pub fn record(
        &mut self,
        phase: RowAuditPhase,
        rows: &[AuditRowId],
    ) -> Result<(), RowAuditError> {
        if rows.is_empty() {
            return Err(RowAuditError::EmptyAccess);
        }
        if phase.search() == self.finalizing {
            return Err(RowAuditError::PhaseNotAllowed);
        }
        let allowed = |row: &AuditRowId| match phase {
            RowAuditPhase::Landmarks
            | RowAuditPhase::Rung0
            | RowAuditPhase::Rung1
            | RowAuditPhase::Preprocessing
            | RowAuditPhase::Calibration
            | RowAuditPhase::FinalizationFit => self.fit.contains(row),
            RowAuditPhase::HoldoutScore => self.holdout.contains(row),
            RowAuditPhase::FinalizationRefit => {
                self.fit.contains(row) || self.holdout.contains(row)
            }
            RowAuditPhase::TestScore => self.test.contains(row),
        };
        if !rows.iter().all(allowed) {
            return Err(RowAuditError::ForbiddenRow);
        }
        self.entries.push(RowTouch {
            phase,
            rows: rows.to_vec(),
        });
        Ok(())
    }

    /// Refused requests never call the reader. Reader errors/panics do not erase
    /// attempted-access evidence; callers retain their actual outcome separately.
    pub fn with_rows<T>(
        &mut self,
        phase: RowAuditPhase,
        rows: &[AuditRowId],
        read: impl FnOnce(&[AuditRowId]) -> T,
    ) -> Result<T, RowAuditError> {
        self.record(phase, rows)?;
        Ok(read(rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    pub(super) fn row(dataset: u8, ordinal: u64) -> AuditRowId {
        AuditRowId {
            dataset: format!("ds_{dataset:024x}").parse().unwrap(),
            row: ordinal,
        }
    }

    #[test]
    fn dataset_namespace_and_every_phase_permission_are_enforced() {
        let fit = row(1, 0);
        let holdout = row(1, 1);
        let test = row(2, 0);
        let mut log =
            RowAudit::new(vec![fit.clone()], vec![holdout.clone()], vec![test.clone()]).unwrap();
        for phase in [
            RowAuditPhase::Landmarks,
            RowAuditPhase::Rung0,
            RowAuditPhase::Rung1,
            RowAuditPhase::Preprocessing,
            RowAuditPhase::Calibration,
        ] {
            assert_eq!(
                log.with_rows(phase, std::slice::from_ref(&fit), |rows| rows.len())
                    .unwrap(),
                1
            );
            let before = log.entries().to_vec();
            for forbidden in [&holdout, &test, &row(3, 0)] {
                assert_eq!(
                    log.with_rows(phase, &[fit.clone(), forbidden.clone()], |_| panic!(
                        "refused reader ran"
                    )),
                    Err(RowAuditError::ForbiddenRow)
                );
                assert_eq!(log.entries(), before);
            }
        }
        assert!(log.entries().iter().all(|e| e.rows == [fit.clone()]));
        assert_eq!(
            log.record(RowAuditPhase::HoldoutScore, std::slice::from_ref(&holdout)),
            Err(RowAuditError::PhaseNotAllowed)
        );
        log.begin_finalization();
        log.begin_finalization();
        for (phase, allowed) in [
            (RowAuditPhase::FinalizationFit, vec![fit.clone()]),
            (RowAuditPhase::HoldoutScore, vec![holdout.clone()]),
            (RowAuditPhase::FinalizationRefit, vec![fit.clone(), holdout]),
            (RowAuditPhase::TestScore, vec![test.clone()]),
        ] {
            log.record(phase, &allowed).unwrap();
        }
        assert_eq!(
            log.record(RowAuditPhase::FinalizationRefit, &[test]),
            Err(RowAuditError::ForbiddenRow)
        );
        assert_eq!(
            log.record(RowAuditPhase::Rung0, &[fit]),
            Err(RowAuditError::PhaseNotAllowed)
        );
        assert!(serde_json::to_value(&log).unwrap()["finalizing"]
            .as_bool()
            .unwrap());
    }

    #[test]
    fn invalid_rosters_empty_access_and_reader_failures_are_not_passing_evidence() {
        assert!(RowAudit::new(vec![], vec![], vec![]).is_err());
        assert!(RowAudit::new(vec![row(1, 0), row(1, 0)], vec![], vec![]).is_err());
        assert!(RowAudit::new(vec![row(1, 0)], vec![row(1, 0)], vec![]).is_err());
        let mut log = RowAudit::new(vec![row(1, 0)], vec![], vec![]).unwrap();
        assert_eq!(
            log.record(RowAuditPhase::Landmarks, &[]),
            Err(RowAuditError::EmptyAccess)
        );
        assert!(log.entries().is_empty());
        let result = log
            .with_rows(RowAuditPhase::Landmarks, &[row(1, 0)], |_| {
                Err::<(), _>("read failed")
            })
            .unwrap();
        assert!(result.is_err());
        assert_eq!(log.entries().len(), 1);
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 128, failure_persistence: None, ..ProptestConfig::default() })]
        #[test]
        fn search_log_contains_only_fit_rows_and_refusal_is_atomic(fit_count in 1_u64..100, bad_at in 0_usize..25, phase in 0_usize..5) {
            let fit:Vec<_>=(0..fit_count).map(|r|row(1,r)).collect();
            let mut log=RowAudit::new(fit.clone(),vec![row(1,fit_count)],vec![row(2,0)]).unwrap();
            let phases=[RowAuditPhase::Landmarks,RowAuditPhase::Rung0,RowAuditPhase::Rung1,RowAuditPhase::Preprocessing,RowAuditPhase::Calibration];
            log.record(phases[phase],&fit).unwrap();
            let mut request=fit.clone();let index=bad_at.min(request.len());request.insert(index,row(2,0));
            prop_assert_eq!(log.record(phases[phase],&request),Err(RowAuditError::ForbiddenRow));
            prop_assert_eq!(log.entries().len(),1);
            prop_assert!(log.entries().iter().flat_map(|e|&e.rows).all(|r|fit.contains(r)));
        }
    }
}
