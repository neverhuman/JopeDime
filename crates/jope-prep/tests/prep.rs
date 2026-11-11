//! Contract, refusal, and holdout-isolation tests for `jope-prep` (Carbon Phase C1a).

use jope_prep::landmarks::{fit_transform_and_extract_bundle_high, RecipeError};
use jope_prep::{
    encode_jmat_v1, split_and_encode_jmat_v1, FitLandmarkBundle, PrepCell, PrepColumnKind,
    PrepError, PrepMatrix, PrepTable, SplitPlan, DEFAULT_HOLDOUT_FRACTION, FIT_LANDMARK_WIRE_BYTES,
    MAX_LANDMARK_FIT_ROWS, PREP_DOCS_URL,
};

fn synthetic_regression_table(rows: usize, features: usize) -> (Vec<f32>, Vec<f32>) {
    let mut x = Vec::with_capacity(rows * features);
    let mut y = Vec::with_capacity(rows);
    for r in 0..rows {
        let rf = r as f32;
        let mut target = 1.25_f32;
        for c in 0..features {
            let cf = (c + 1) as f32;
            let val = ((rf * 0.13 + cf * 0.37).sin() * cf) + (rf * 0.01 * cf);
            x.push(val);
            if c < 3 {
                target += val * (0.5 / cf);
            }
        }
        y.push(target);
    }
    (x, y)
}

#[test]
fn split_plan_is_deterministic_disjoint_and_caps_sampled_fit_rows() {
    let plan_a = SplitPlan::deterministic(2_500, 42, DEFAULT_HOLDOUT_FRACTION).unwrap();
    let plan_b = SplitPlan::deterministic(2_500, 42, DEFAULT_HOLDOUT_FRACTION).unwrap();
    assert_eq!(plan_a, plan_b);
    assert_eq!(plan_a.total_rows(), 2_500);
    assert_eq!(plan_a.holdout_rows().len(), 500);
    assert_eq!(plan_a.fit_rows().len(), 2_000);
    assert_eq!(plan_a.sampled_fit_rows().len(), MAX_LANDMARK_FIT_ROWS);
    plan_a.validate_for_table(2_500).unwrap();

    // Every row in 0..2_500 belongs to exactly one of fit_rows or holdout_rows.
    let mut seen = vec![0u8; 2_500];
    for &idx in plan_a.fit_rows() {
        seen[idx] += 1;
    }
    for &idx in plan_a.holdout_rows() {
        seen[idx] += 1;
    }
    assert!(seen.iter().all(|&count| count == 1));

    // Every sampled_fit_row is strictly inside fit_rows and never in holdout_rows.
    for &idx in plan_a.sampled_fit_rows() {
        assert!(plan_a.fit_rows().binary_search(&idx).is_ok());
        assert!(plan_a.holdout_rows().binary_search(&idx).is_err());
    }
}

#[test]
fn holdout_poison_invariance_proves_zero_holdout_leakage() {
    let rows = 320;
    let features = 12;
    let (clean_x, clean_y) = synthetic_regression_table(rows, features);
    let plan = SplitPlan::deterministic(rows, 20261110, 0.25).unwrap();

    let clean_bundle =
        FitLandmarkBundle::extract_fit_only(&clean_x, rows, features, &clean_y, &plan).unwrap();
    let clean_wire = clean_bundle.to_wire_bytes();
    assert_eq!(clean_wire.len(), FIT_LANDMARK_WIRE_BYTES);

    // Poison every HOLDOUT row in X and y with NaN, +Infinity, -Infinity, and extreme outliers.
    let mut poisoned_x = clean_x.clone();
    let mut poisoned_y = clean_y.clone();
    for (pos, &holdout_row) in plan.holdout_rows().iter().enumerate() {
        poisoned_y[holdout_row] = match pos % 3 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            _ => -1.0e30,
        };
        for col in 0..features {
            poisoned_x[holdout_row * features + col] = if (pos + col) % 2 == 0 {
                f32::NAN
            } else {
                f32::NEG_INFINITY
            };
        }
    }

    let poisoned_bundle =
        FitLandmarkBundle::extract_fit_only(&poisoned_x, rows, features, &poisoned_y, &plan)
            .unwrap();
    let poisoned_wire = poisoned_bundle.to_wire_bytes();

    // Bit-for-bit equality across all 6,368 wire bytes proves zero HOLDOUT dependence.
    assert_eq!(clean_wire, poisoned_wire);
    assert_eq!(clean_bundle, poisoned_bundle);

    // Direct call on the sampled FIT slice matches FitLandmarkBundle outputs bit-for-bit.
    let mut direct_x = Vec::new();
    let mut direct_y = Vec::new();
    for &r in plan.sampled_fit_rows() {
        direct_x.extend_from_slice(&clean_x[r * features..(r + 1) * features]);
        direct_y.push(clean_y[r]);
    }
    let direct =
        fit_transform_and_extract_bundle_high(&direct_x, direct_y.len(), features, &direct_y)
            .unwrap();
    assert_eq!(clean_bundle.vector256(), direct.vector());
    assert_eq!(clean_bundle.ranker_columns(), direct.columns());
}

#[test]
fn wire_roundtrip_and_tamper_refusals() {
    let (x, y) = synthetic_regression_table(96, 8);
    let plan = SplitPlan::deterministic(96, 7, 0.20).unwrap();
    let bundle = FitLandmarkBundle::extract_fit_only(&x, 96, 8, &y, &plan).unwrap();
    let wire = bundle.to_wire_bytes();

    let decoded = FitLandmarkBundle::from_wire_bytes(&wire).unwrap();
    assert_eq!(decoded, bundle);

    // Truncated buffer refused.
    assert!(matches!(
        FitLandmarkBundle::from_wire_bytes(&wire[..FIT_LANDMARK_WIRE_BYTES - 1]),
        Err(PrepError::InvalidLandmarkWire { .. })
    ));

    // Corrupted payload byte caught by trailing SHA-256.
    let mut flipped = wire;
    flipped[128] ^= 0x01;
    assert!(matches!(
        FitLandmarkBundle::from_wire_bytes(&flipped),
        Err(PrepError::InvalidLandmarkWire { .. })
    ));

    // Bad magic refused.
    let mut bad_magic = wire;
    bad_magic[0] = b'X';
    assert!(matches!(
        FitLandmarkBundle::from_wire_bytes(&bad_magic),
        Err(PrepError::InvalidLandmarkWire { .. })
    ));
}

#[test]
fn jmat_roundtrip_and_refusals() {
    let table = PrepTable {
        headers: vec!["x1".into(), "region".into(), "target".into()],
        rows: vec![
            vec![Some("1.5".into()), Some("east".into()), Some("10.0".into())],
            vec![None, Some("west".into()), Some("20.5".into())],
            vec![
                Some("-3.25".into()),
                Some("east".into()),
                Some("30.0".into()),
            ],
            vec![Some("0.0".into()), None, Some("40.0".into())],
            vec![
                Some("8.5".into()),
                Some("north".into()),
                Some("50.0".into()),
            ],
        ],
    };

    let plan = SplitPlan::deterministic(5, 99, 0.40).unwrap();
    let split_artifacts = split_and_encode_jmat_v1(&table, &plan).unwrap();
    let fit_mat = PrepMatrix::open(&split_artifacts.fit_jmat).unwrap();
    let holdout_mat = PrepMatrix::open(&split_artifacts.holdout_jmat).unwrap();
    assert_eq!(fit_mat.rows(), 3);
    assert_eq!(holdout_mat.rows(), 2);
    assert_eq!(fit_mat.columns()[0].kind, PrepColumnKind::Numeric);
    assert_eq!(fit_mat.columns()[1].kind, PrepColumnKind::Categorical);

    let full_bytes = encode_jmat_v1(&table).unwrap();
    let full_mat = PrepMatrix::open(&full_bytes).unwrap();
    assert_eq!(full_mat.rows(), 5);
    assert_eq!(full_mat.cell(0, 0), Some(PrepCell::Numeric(1.5)));
    assert_eq!(full_mat.cell(1, 0), Some(PrepCell::Missing));
    assert_eq!(full_mat.cell(0, 1), Some(PrepCell::Categorical("east")));
    assert_eq!(full_mat.cell(3, 1), Some(PrepCell::Missing));

    // Corrupting any byte in JMAT v1 fails checksum validation.
    let mut tampered = full_bytes.clone();
    let mid = tampered.len() / 2;
    tampered[mid] ^= 0xFF;
    assert!(matches!(
        PrepMatrix::open(&tampered),
        Err(PrepError::InvalidJmat { .. })
    ));

    // Duplicate column headers refused.
    let dup = PrepTable {
        headers: vec!["a".into(), "a".into()],
        rows: vec![vec![Some("1".into()), Some("2".into())]],
    };
    assert!(matches!(
        encode_jmat_v1(&dup),
        Err(PrepError::InvalidJmat { .. })
    ));
}

#[test]
fn split_and_conditioning_refusals_and_guidance_cover_all_variants() {
    assert_eq!(
        SplitPlan::deterministic(1, 0, 0.2),
        Err(PrepError::InsufficientRows { rows: 1 })
    );
    assert_eq!(
        SplitPlan::deterministic(10, 0, 0.0),
        Err(PrepError::InvalidHoldoutFraction { fraction: 0.0 })
    );
    assert_eq!(
        SplitPlan::deterministic(10, 0, 1.0),
        Err(PrepError::InvalidHoldoutFraction { fraction: 1.0 })
    );

    let plan = SplitPlan::deterministic(10, 1, 0.2).unwrap();
    assert_eq!(
        FitLandmarkBundle::extract_fit_only(&[], 10, 0, &[0.0; 10], &plan),
        Err(PrepError::ZeroFeatures)
    );
    assert_eq!(
        FitLandmarkBundle::extract_fit_only(&[1.0; 9], 10, 1, &[0.0; 10], &plan),
        Err(PrepError::ShapeMismatch {
            expected: 10,
            actual: 9
        })
    );
    let mut bad_y = [1.0_f32; 10];
    bad_y[plan.fit_rows()[0]] = f32::NAN;
    assert_eq!(
        FitLandmarkBundle::extract_fit_only(&[1.0; 10], 10, 1, &bad_y, &plan),
        Err(PrepError::NonFiniteFitTarget {
            fit_row: plan.fit_rows()[0]
        })
    );

    let all_variants = [
        PrepError::InsufficientRows { rows: 1 },
        PrepError::InvalidHoldoutFraction { fraction: 1.5 },
        PrepError::ZeroFeatures,
        PrepError::ShapeMismatch {
            expected: 20,
            actual: 19,
        },
        PrepError::ArithmeticOverflow,
        PrepError::InvalidSplitIndex {
            index: 9,
            total_rows: 5,
        },
        PrepError::NonFiniteFitTarget { fit_row: 0 },
        PrepError::LandmarkRecipe(RecipeError::EmptyTable),
        PrepError::InvalidLandmarkWire { reason: "bad wire" },
        PrepError::InvalidJmat { reason: "bad jmat" },
    ];
    for err in all_variants {
        let g = err.guidance();
        assert!(!g.purpose.is_empty());
        assert_eq!(g.reason, err.to_string());
        assert!(!g.common_fixes.is_empty());
        assert_eq!(g.docs_url, PREP_DOCS_URL);
        assert!(!g.repair_hint.is_empty());
    }
}
