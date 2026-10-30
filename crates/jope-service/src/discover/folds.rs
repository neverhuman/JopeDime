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
mod tests {
    use super::super::audit::RowAudit;
    use super::*;
    use proptest::prelude::*;

    fn ids(count: usize) -> Vec<AuditRowId> {
        let dataset: jope_api::ids::DatasetId = "ds_000000000000000000000001".parse().unwrap();
        (0..count)
            .map(|row| AuditRowId {
                dataset: dataset.clone(),
                row: row as u64,
            })
            .collect()
    }
    fn phases() -> [CvPhase; 3] {
        [CvPhase::Rung0, CvPhase::Rung1, CvPhase::Calibration]
    }

    fn check_groups(folds: &[CvFold], groups: &[u64], selected: usize) {
        let mut owners = BTreeMap::new();
        let mut validated = BTreeSet::new();
        for (index, fold) in folds.iter().enumerate() {
            assert!(!fold.train().is_empty() && !fold.validation().is_empty());
            let train: BTreeSet<_> = fold
                .train()
                .iter()
                .map(|r| groups[r.row as usize])
                .collect();
            let val: BTreeSet<_> = fold
                .validation()
                .iter()
                .map(|r| groups[r.row as usize])
                .collect();
            assert!(train.is_disjoint(&val));
            for group in val {
                assert!(
                    owners.insert(group, index).is_none(),
                    "a group spans validation folds"
                );
            }
            for row in fold.validation() {
                assert!(validated.insert(row.row));
            }
        }
        assert_eq!(validated.len(), selected);
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 128, failure_persistence: None, ..ProptestConfig::default() })]
        #[test]
        fn grouped_constraints_hold_in_every_phase(groups in prop::collection::vec(0_u64..12,12..120), seed in any::<u64>()) {
            let unique=groups.iter().collect::<BTreeSet<_>>().len();
            prop_assume!(unique>=3);
            let plan=FoldPlan::new(ids(groups.len()),FoldConstraints { groups:Some(groups.clone()), ..Default::default() },seed).unwrap();
            for phase in phases() {
                let k=if phase==CvPhase::Rung0 {2}else{3};
                let folds=plan.folds(phase,k,groups.len()).unwrap();
                prop_assert_eq!(&folds,&plan.folds(phase,k,groups.len()).unwrap());
                check_groups(&folds,&groups,groups.len());
                let mut audit=RowAudit::new(ids(groups.len()),vec![],vec![]).unwrap();
                for fold in folds {
                    for rows in [fold.train(),fold.validation()] { audit.record(phase.audit_phase(),rows).unwrap(); }
                }
                prop_assert!(audit.entries().iter().flat_map(|e|&e.rows).all(|r|r.dataset==ids(1)[0].dataset && r.row<groups.len() as u64));
            }
        }

        #[test]
        fn time_and_gap_are_strict_for_all_phases(n in 24_usize..150, gap in 0_usize..4, reverse in any::<bool>(), seed in any::<u64>()) {
            let times:Vec<_>=(0..n).map(|i|if reverse {(n-i) as f64}else{i as f64}).collect();
            let plan=FoldPlan::new(ids(n),FoldConstraints { times:Some(times.clone()),gap_rows:gap,..Default::default() },seed).unwrap();
            for phase in phases() {
                let k=if phase==CvPhase::Rung0 {2}else{3};
                let folds=plan.folds(phase,k,n).unwrap();let mut previous=f64::NEG_INFINITY;
                for fold in folds {
                    let train_end=times[fold.train().last().unwrap().row as usize];
                    let val_start=times[fold.validation()[0].row as usize];
                    prop_assert!(train_end<val_start);
                    prop_assert!(val_start-train_end>=gap as f64+1.0);
                    prop_assert!(val_start>previous);previous=times[fold.validation().last().unwrap().row as usize];
                    prop_assert!(fold.train().iter().all(|r|r.row<n as u64));
                }
            }
        }

        #[test]
        fn combined_time_ties_groups_and_gap_are_never_relaxed(group_count in 12_usize..40, group_size in 1_usize..5, gap in 0_usize..3, ties in any::<bool>()) {
            let n=group_count*group_size;
            let groups:Vec<_>=(0..n).map(|i|(i/group_size) as u64).collect();
            let times:Vec<_>=groups.iter().enumerate().map(|(i,&g)|if ties {g as f64}else{i as f64}).collect();
            let plan=FoldPlan::new(ids(n),FoldConstraints { groups:Some(groups.clone()),times:Some(times.clone()),gap_rows:gap },19).unwrap();
            for phase in phases() {
                let k=if phase==CvPhase::Rung0 {2}else{3};let folds=plan.folds(phase,k,n).unwrap();
                let mut val_groups=BTreeSet::new();
                for fold in folds {
                    let tg:BTreeSet<_>=fold.train().iter().map(|r|groups[r.row as usize]).collect();
                    let vg:BTreeSet<_>=fold.validation().iter().map(|r|groups[r.row as usize]).collect();
                    prop_assert!(tg.is_disjoint(&vg));
                    for g in vg { prop_assert!(val_groups.insert(g)); }
                    prop_assert!(times[fold.train().last().unwrap().row as usize]<times[fold.validation()[0].row as usize]);
                    let end=fold.train().last().unwrap().row as usize;let start=fold.validation()[0].row as usize;
                    prop_assert!(start-end>gap);
                }
            }
        }
    }

    #[test]
    fn overlapping_group_intervals_remain_whole_in_a_feasible_time_split() {
        let groups = vec![0, 1, 2, 3, 4, 3, 5, 6, 7];
        let plan = FoldPlan::new(
            ids(9),
            FoldConstraints {
                groups: Some(groups.clone()),
                times: Some((0..9).map(|i| i as f64).collect()),
                gap_rows: 0,
            },
            1,
        )
        .unwrap();
        let folds = plan.folds(CvPhase::Rung0, 2, 9).unwrap();
        assert_eq!(
            folds[0].train().iter().map(|r| r.row).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            folds[0]
                .validation()
                .iter()
                .map(|r| r.row)
                .collect::<Vec<_>>(),
            vec![2, 3, 4, 5]
        );
        for fold in folds {
            let train: BTreeSet<_> = fold
                .train()
                .iter()
                .map(|r| groups[r.row as usize])
                .collect();
            let validation: BTreeSet<_> = fold
                .validation()
                .iter()
                .map(|r| groups[r.row as usize])
                .collect();
            assert!(train.is_disjoint(&validation));
        }
    }

    #[test]
    fn nested_calibration_inherits_constraints_and_excludes_outer_validation() {
        let groups: Vec<_> = (0..120).map(|i| i / 2).collect();
        let times: Vec<_> = (0..120).map(|i| i as f64).collect();
        let plan = FoldPlan::new(
            ids(120),
            FoldConstraints {
                groups: Some(groups.clone()),
                times: Some(times.clone()),
                gap_rows: 2,
            },
            17,
        )
        .unwrap();
        for phase in [CvPhase::Rung0, CvPhase::Rung1] {
            for outer in plan.folds(phase, 2, 120).unwrap() {
                for inner in plan.calibration_folds(&outer, 2, 120).unwrap() {
                    assert!(inner
                        .train()
                        .iter()
                        .chain(inner.validation())
                        .all(|r| outer.train().contains(r)));
                    assert!(inner
                        .train()
                        .iter()
                        .chain(inner.validation())
                        .all(|r| !outer.validation().contains(r)));
                    let train: BTreeSet<_> = inner
                        .train()
                        .iter()
                        .map(|r| groups[r.row as usize])
                        .collect();
                    let validation: BTreeSet<_> = inner
                        .validation()
                        .iter()
                        .map(|r| groups[r.row as usize])
                        .collect();
                    assert!(train.is_disjoint(&validation));
                    assert!(
                        times[inner.train().last().unwrap().row as usize]
                            < times[inner.validation()[0].row as usize]
                    );
                    assert!(inner.validation()[0].row - inner.train().last().unwrap().row > 2);
                }
            }
        }
    }

    #[test]
    fn ties_noncontiguous_ids_and_overlapping_group_intervals_have_explicit_behavior() {
        let times = vec![0.0, -0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0];
        let mut rows = ids(times.len());
        for r in &mut rows {
            r.row = 100 + r.row * 7;
        }
        let plan = FoldPlan::new(
            rows.clone(),
            FoldConstraints {
                times: Some(times),
                gap_rows: 1,
                ..Default::default()
            },
            1,
        )
        .unwrap();
        for fold in plan.folds(CvPhase::Rung0, 2, rows.len()).unwrap() {
            assert!(fold
                .train()
                .iter()
                .chain(fold.validation())
                .all(|r| rows.contains(r)));
            assert!(fold.train().last().unwrap().row < fold.validation()[0].row);
        }
        // Interleaved group intervals coalesce to one block: refusing is safer
        // than silently ignoring either the group or time constraint.
        let impossible = FoldPlan::new(
            ids(6),
            FoldConstraints {
                groups: Some(vec![0, 1, 0, 1, 0, 1]),
                times: Some((0..6).map(|x| x as f64).collect()),
                gap_rows: 0,
            },
            1,
        )
        .unwrap();
        assert_eq!(
            impossible.folds(CvPhase::Rung0, 2, 6),
            Err(FoldError::InsufficientIndependentBlocks)
        );
    }

    #[test]
    fn caps_cannot_introduce_outside_rows_and_bad_inputs_do_not_fallback() {
        let mut rows = ids(30);
        for r in &mut rows {
            r.row += 1000;
        }
        let plan = FoldPlan::new(rows.clone(), Default::default(), 7).unwrap();
        for phase in phases() {
            let folds = plan.folds(phase, 2, 10).unwrap();
            check_disjoint(&folds);
            let selected: BTreeSet<_> = folds.iter().flat_map(|f| f.validation()).collect();
            assert_eq!(selected.len(), 10);
            assert!(selected.iter().all(|r| rows.contains(r)));
        }
        assert!(FoldPlan::new(vec![], Default::default(), 0).is_err());
        assert!(FoldPlan::new(
            vec![rows[0].clone(), rows[0].clone()],
            Default::default(),
            0
        )
        .is_err());
        assert!(FoldPlan::new(
            ids(2),
            FoldConstraints {
                groups: Some(vec![0]),
                ..Default::default()
            },
            0
        )
        .is_err());
        assert!(FoldPlan::new(
            ids(2),
            FoldConstraints {
                times: Some(vec![0.0, f64::NAN]),
                ..Default::default()
            },
            0
        )
        .is_err());
        assert!(FoldPlan::new(
            ids(2),
            FoldConstraints {
                gap_rows: 1,
                ..Default::default()
            },
            0
        )
        .is_err());
        assert_eq!(
            plan.folds(CvPhase::Rung0, 3, 30),
            Err(FoldError::InvalidFoldCount)
        );
        assert_eq!(
            plan.folds(CvPhase::Rung1, 2, 0),
            Err(FoldError::InvalidFoldCount)
        );
        let gap = FoldPlan::new(
            ids(6),
            FoldConstraints {
                times: Some((0..6).map(|x| x as f64).collect()),
                gap_rows: usize::MAX,
                ..Default::default()
            },
            0,
        )
        .unwrap();
        assert_eq!(
            gap.folds(CvPhase::Rung0, 2, 6),
            Err(FoldError::GapLeavesNoTraining)
        );
    }

    fn check_disjoint(folds: &[CvFold]) {
        for f in folds {
            assert!(f.train().iter().all(|r| !f.validation().contains(r)));
        }
    }

    #[test]
    fn real_catalog_train_rosters_follow_group_and_time_constraints() {
        // QLT05 source files are headerless. Read numeric constraint metadata
        // directly from the unchanged train blobs; do not touch test.csv or fit
        // a transform. SHA pins below bind these real catalog inputs.
        const GROUPED: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test-support/datasets/v1/grouped-apartment-rent/train.csv"
        ));
        const TIME: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test-support/datasets/v1/time-ordered-prisonlf/train.csv"
        ));
        assert_eq!(
            jope_runtime::sha256_hex(GROUPED.as_bytes()),
            "1599896e8259d16c56afb4a300cbd22846745657687ae8a041cc4f8e73a9c641"
        );
        assert_eq!(
            jope_runtime::sha256_hex(TIME.as_bytes()),
            "539dec765cd7ba40cb85b81776572f991624f2748e7b29822206c4366cba07ab"
        );
        let groups: Vec<_> = GROUPED
            .lines()
            .map(|line| {
                line.split(',')
                    .nth(7)
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
                    .to_bits()
            })
            .collect();
        // The fixture retains month-of-year (four seasonal values), not an
        // absolute timestamp. It exercises real tied numeric metadata; this is
        // not a claim of qualified calendar chronology or a live learning loop.
        let times: Vec<_> = TIME
            .lines()
            .map(|line| line.split(',').nth(1).unwrap().parse::<f64>().unwrap())
            .collect();
        let catalog_ids = |count: usize, source: &str| {
            let dataset: jope_api::ids::DatasetId =
                format!("ds_{}", &jope_runtime::sha256_hex(source.as_bytes())[..24])
                    .parse()
                    .unwrap();
            (0..count)
                .map(|row| AuditRowId {
                    dataset: dataset.clone(),
                    row: row as u64,
                })
                .collect::<Vec<_>>()
        };
        let grouped_rows = catalog_ids(groups.len(), GROUPED);
        let time_rows = catalog_ids(times.len(), TIME);
        let gp = FoldPlan::new(
            grouped_rows,
            FoldConstraints {
                groups: Some(groups.clone()),
                ..Default::default()
            },
            41,
        )
        .unwrap();
        let tp = FoldPlan::new(
            time_rows.clone(),
            FoldConstraints {
                times: Some(times.clone()),
                gap_rows: 2,
                ..Default::default()
            },
            41,
        )
        .unwrap();
        for phase in phases() {
            let k = if phase == CvPhase::Rung0 { 2 } else { 3 };
            check_groups(
                &gp.folds(phase, k, groups.len()).unwrap(),
                &groups,
                groups.len(),
            );
            let mut audit = RowAudit::new(time_rows.clone(), vec![], vec![]).unwrap();
            for fold in tp.folds(phase, k, times.len()).unwrap() {
                assert!(
                    times[fold.train().last().unwrap().row as usize]
                        < times[fold.validation()[0].row as usize]
                );
                for rows in [fold.train(), fold.validation()] {
                    audit.record(phase.audit_phase(), rows).unwrap();
                }
            }
            assert!(!audit.entries().is_empty());
        }
    }
}
