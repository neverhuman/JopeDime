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
