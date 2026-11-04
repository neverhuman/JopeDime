use super::width_contract::{Side, REFUSALS};
use super::*;

#[test]
fn inputs_predict_panics_on_or_misreads_are_refused_before_any_stage() {
    for (index, &(source, fitted, side, predict_panics)) in REFUSALS.iter().enumerate() {
        let seed = 10 * index as u64;
        let model = fit_program(source, &data(48, fitted, seed));
        let required = required_input_width(&fitted_ops(&model)[0]);
        let valid = data(16, fitted, seed + 1);
        assert_eq!(
            bits(&model.try_predict(&valid).unwrap()),
            bits(&model.predict(&valid).unwrap()),
            "{source}: the fitted width"
        );
        let refused = match (side, required) {
            (Side::Narrower, WidthReq::Exactly(n) | WidthReq::AtLeast(n)) => n - 1,
            (Side::Wider, WidthReq::Exactly(n)) => n + 1,
            _ => panic!("{source}: {side:?} of {required:?}"),
        };
        let bad = data(16, refused, seed + 2);
        assert_eq!(
            model.try_predict(&bad),
            Err(WidthMismatch {
                stage: 0,
                expected: required,
                got: refused,
                kind: MismatchKind::StageInput,
            }),
            "{source} at width {refused}"
        );
        assert_eq!(
            panics(|| model.predict(&bad).unwrap()),
            predict_panics,
            "{source}: predict at width {refused}"
        );
    }

    // Roll's empty row is a typed refusal on both public prediction paths.
    let model = fit_program("(pipe X (roll k=2) (ridge))", &data(24, 3, 1));
    let empty_rows = vec![Vec::new(); 4];
    assert_eq!(
        model.try_predict(&empty_rows),
        Err(WidthMismatch {
            stage: 0,
            expected: WidthReq::AtLeast(1),
            got: 0,
            kind: MismatchKind::StageInput,
        })
    );
    assert_eq!(
        model.predict(&empty_rows),
        Err(WidthMismatch {
            stage: 0,
            expected: WidthReq::AtLeast(1),
            got: 0,
            kind: MismatchKind::StageInput,
        })
    );
    assert!(!panics(|| model.predict(&empty_rows)));
}

// ---------------------------------------------------------------------------------------------
// Heads, portfolios, symbolic models, ragged rows and empty inputs
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum HeadClass {
    Exact,
    LowerBound,
    Any,
}

#[test]
fn heads_require_the_width_their_fitted_state_reveals() {
    // (program, training rows, class of the head's requirement at 3 columns)
    let cases: &[(&str, usize, HeadClass)] = &[
        ("(pipe X (ridge))", 48, HeadClass::Exact),
        ("(pipe X (elasticnet))", 48, HeadClass::Exact),
        ("(pipe X (ridge_gcv))", 48, HeadClass::LowerBound),
        (
            "(pipe X (histgbm rounds=3 depth=2))",
            48,
            HeadClass::LowerBound,
        ),
        ("(pipe X (ensemble rounds=3 depth=2))", 48, HeadClass::Exact),
        (
            "(pipe X (histgbm2 rounds=3 leaves=4 bags=1))",
            48,
            HeadClass::LowerBound,
        ),
        ("(pipe X (krr m=50))", 48, HeadClass::Exact),
        ("(pipe X (gpr m=50 steps=2))", 48, HeadClass::Exact),
        (
            "(pipe X (gpr kernel=rq+linear m=50 steps=2))",
            48,
            HeadClass::Exact,
        ),
        ("(pipe X (ksmooth k=8))", 48, HeadClass::Exact),
        ("(pipe X (hurdle2))", 48, HeadClass::Exact),
        ("(pipe X (stack))", 48, HeadClass::Exact),
        // Constant fallbacks on too few rows keep no column.
        ("(pipe X (histgbm2 rounds=3))", 8, HeadClass::Any),
        ("(pipe X (histgbm rounds=3))", 3, HeadClass::Any),
        ("(pipe X (hurdle2))", 8, HeadClass::Any),
    ];
    for (index, &(source, rows, class)) in cases.iter().enumerate() {
        let seed = 7 * index as u64;
        let model = fit_program(source, &data(rows, 3, seed));
        let Model::Pipeline { head, .. } = &model else {
            panic!("{source}: a pipeline model")
        };
        let required = required_head_width(head);
        let expected = match class {
            HeadClass::Exact => WidthReq::Exactly(3),
            HeadClass::LowerBound => WidthReq::AtLeast(3),
            HeadClass::Any => WidthReq::Any,
        };
        assert_eq!(required, expected, "{source}");
        for width in [2, 3, 4] {
            let x = data(6, width, seed + 1);
            // Heads read columns defensively: predict never panics on width.
            let predicted = catch_unwind(AssertUnwindSafe(|| model.predict(&x).unwrap()))
                .unwrap_or_else(|_| panic!("{source}: predict at width {width} panicked"));
            match model.try_predict(&x) {
                Ok(tried) => {
                    assert!(required.admits(width), "{source} at width {width}");
                    assert_eq!(bits(&tried), bits(&predicted), "{source} at width {width}");
                }
                Err(mismatch) => {
                    // A refusal is only ever one that the requirement makes.
                    assert!(!required.admits(width), "{source} refused at width {width}");
                    assert_eq!(
                        mismatch,
                        WidthMismatch {
                            stage: 0,
                            expected: required,
                            got: width,
                            kind: MismatchKind::HeadInput,
                        },
                        "{source} at width {width}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_tree_head_on_wide_rows_requires_only_its_feature_count() {
    // histgbm2 keeps at most 256 features; 300 columns stay admitted.
    let model = fit_program(
        "(pipe X (histgbm2 rounds=3 leaves=4 bags=1))",
        &data(40, 300, 12),
    );
    let Model::Pipeline { head, .. } = &model else {
        panic!("a pipeline model")
    };
    assert_eq!(required_head_width(head), WidthReq::AtLeast(256));
    let wide = data(8, 300, 13);
    assert_eq!(
        bits(&model.try_predict(&wide).unwrap()),
        bits(&model.predict(&wide).unwrap())
    );
    assert_eq!(
        model.try_predict(&data(8, 255, 14)),
        Err(WidthMismatch {
            stage: 0,
            expected: WidthReq::AtLeast(256),
            got: 255,
            kind: MismatchKind::HeadInput,
        })
    );
}

#[test]
fn portfolios_require_their_fitted_width_unless_they_predict_the_mean() {
    // No training rows give a mean-only portfolio; the fields are public, so a padded one is
    // built from it.
    let mean_only = feat_math::portfolio::fit_portfolio(&[], &Vec::new(), &[], 5, false).unwrap();
    assert!(mean_only.mean_only);
    let mut padded = mean_only.clone();
    padded.mean_only = false;
    padded.width = 3;
    padded.keep = vec![0, 1, 2];

    let explicit = Model::Portfolio(mean_only.clone());
    for width in [0, 3, 7] {
        let x = data(4, width, 1);
        if width == 0 {
            assert_eq!(
                explicit.predict(&x),
                Err(WidthMismatch {
                    stage: 0,
                    expected: WidthReq::AtLeast(1),
                    got: 0,
                    kind: MismatchKind::StageInput,
                })
            );
            assert_eq!(explicit.try_predict(&x), explicit.predict(&x));
            continue;
        }
        assert_eq!(
            bits(&explicit.try_predict(&x).unwrap()),
            bits(&explicit.predict(&x).unwrap())
        );
    }

    for model in [
        Model::Portfolio(padded.clone()),
        Model::Pipeline {
            target: None,
            ops: Vec::new(),
            head: Box::new(FittedHead::Portfolio(Box::new(padded))),
        },
    ] {
        let x = data(4, 3, 2);
        assert_eq!(
            bits(&model.try_predict(&x).unwrap()),
            bits(&model.predict(&x).unwrap())
        );
        let wide = data(4, 5, 3);
        assert_eq!(
            model.try_predict(&wide),
            Err(WidthMismatch {
                stage: 0,
                expected: WidthReq::Exactly(3),
                got: 5,
                kind: MismatchKind::HeadInput,
            })
        );
        // The portfolio pads or truncates rows itself, so predict does not panic.
        assert!(!panics(|| model.predict(&wide).unwrap()));
    }
}

#[test]
fn symbolic_models_accept_any_width() {
    // An expression that reads a column beyond the row evaluates to 0.0: no width to require.
    let model = fit_program(
        "(sym affine \"sigmoid(c0) + 0.5*c2 - row_mean\")",
        &data(24, 3, 3),
    );
    for width in [0, 1, 3, 9] {
        let x = data(5, width, 4);
        assert_eq!(check_input_width(&model, width), Ok(()));
        if width == 0 {
            assert_eq!(model.try_predict(&x), model.predict(&x));
            assert!(model.predict(&x).is_err());
            continue;
        }
        assert_eq!(
            bits(&model.try_predict(&x).unwrap()),
            bits(&model.predict(&x).unwrap())
        );
    }
}

#[test]
fn ragged_rows_are_refused_and_empty_inputs_pass() {
    let pipeline = fit_program("(pipe X (standardize) (ridge))", &data(24, 3, 1));
    let symbolic = fit_program("(sym raw \"c0 + c1\")", &data(24, 3, 1));
    let mut ragged = data(4, 3, 2);
    ragged[2].pop();
    for model in [&pipeline, &symbolic] {
        assert_eq!(
            model.try_predict(&ragged),
            Err(WidthMismatch {
                stage: 0,
                expected: WidthReq::Exactly(3),
                got: 2,
                kind: MismatchKind::RaggedInput { row: 2 },
            })
        );
        let empty: Rows = Vec::new();
        assert_eq!(
            model.try_predict(&empty),
            Ok(model.predict(&empty).unwrap())
        );
        assert_eq!(model.try_predict(&empty), Ok(Vec::new()));
    }
}

// ---------------------------------------------------------------------------------------------
// Inconsistent fitted state and overflowing output widths
// ---------------------------------------------------------------------------------------------

type Edit = fn(&mut FittedOp);

/// One edit per rule of `check_fitted_state`, on a stage fitted on 4 columns: (stage, edit,
/// whether `predict` panics at the width the edited stage claims). No fit produces these.
#[rustfmt::skip]
const INCONSISTENT: &[(&str, Edit, bool)] = &[
    ("(standardize)", |op| if let FittedOp::Standardize { scales, .. } = op { scales.pop(); }, true),
    ("(robust_scale)", |op| if let FittedOp::RobustScale { scales, .. } = op { scales.pop(); }, true),
    ("(bin_onehot bins=4)", |op| if let FittedOp::BinOnehot { bins, .. } = op { *bins = 0; }, true),
    ("(hinge knots=3)", |op| if let FittedOp::Hinge { knots, .. } = op { knots.clear(); }, true),
    ("(winsor_clip q=0.1)", |op| if let FittedOp::WinsorClip { hi, .. } = op { hi.pop(); }, true),
    ("(winsor_clip q=0.1)", |op| if let FittedOp::WinsorClip { lo, hi, .. } = op { lo[0] = hi[0] + 1.0; }, true),
    ("(winsor_clip q=0.1)", |op| if let FittedOp::WinsorClip { lo, .. } = op { lo[1] = f64::NAN; }, true),
    ("(rff_cos n=5 seed=11)", |op| if let FittedOp::RffCos { b, .. } = op { b.pop(); }, true),
    ("(rff_sin n=5 seed=11)", |op| if let FittedOp::RffSin { w, .. } = op { w[2].pop(); }, false),
    ("(rand_proj k=5 seed=11)", |op| if let FittedOp::RandProj { w, .. } = op { w[1].push(0.5); }, false),
    ("(oblique_hinge n=5 seed=11)", |op| if let FittedOp::ObliqueHinge { t, .. } = op { t.pop(); }, true),
    ("(landmark_dist k=5 seed=11)", |op| if let FittedOp::LandmarkDist { landmarks, .. } = op { landmarks[3].pop(); }, false),
    ("(nystrom k=5 seed=11)", |op| if let FittedOp::Nystrom { landmarks, .. } = op { landmarks[2].push(0.0); }, false),
    ("(svd_project k=3 iters=8 seed=11)", |op| if let FittedOp::SvdProject { components, .. } = op { components[1].pop(); }, true),
    ("(target_bin_encode bins=5)", |op| if let FittedOp::TargetBinEncode { enc, .. } = op { enc.pop(); }, true),
    ("(target_bin_encode bins=5)", |op| if let FittedOp::TargetBinEncode { enc, .. } = op { enc[2].clear(); }, true),
    ("(impute add_indicator=true)", |op| if let FittedOp::Impute { indicator_cols, .. } = op { indicator_cols.push(4); }, true),
    ("(pair_ratio topk=4)", |op| if let FittedOp::PairRatio { eps, .. } = op { eps.pop(); }, false),
    ("(spline knots=4)", |op| if let FittedOp::Spline { knots, .. } = op { knots[1] = vec![f64::NAN]; }, true),
    ("(spline knots=4)", |op| if let FittedOp::Spline { knots, .. } = op { knots[3].clear(); }, true),
];

#[test]
fn inconsistent_fitted_state_is_refused_where_predict_panics() {
    let x = data(24, 4, 6);
    let y = target(&x);
    for (index, &(source, edit, predict_panics)) in INCONSISTENT.iter().enumerate() {
        let mut op = fit_op(&stage(source), &x, &y).unwrap();
        let before = format!("{op:?}");
        edit(&mut op);
        assert_ne!(
            format!("{op:?}"),
            before,
            "case {index} ({source}) edits the state"
        );
        let reason = check_fitted_state(&op).expect_err(source);
        let width = claimed_width(&op);
        assert!(
            matches!(output_width(&op, width), Err(Error::Shape(_))),
            "case {index} ({source})"
        );
        let model = one_stage_model(op, 3);
        let rows = data(3, width, 7);
        assert_eq!(
            model.try_predict(&rows),
            Err(WidthMismatch {
                stage: 0,
                expected: required_input_width(&fitted_ops(&model)[0]),
                got: width,
                kind: MismatchKind::InconsistentState { reason },
            }),
            "case {index} ({source})"
        );
        assert_eq!(
            panics(|| model.predict(&rows).unwrap()),
            predict_panics,
            "case {index} ({source}): predict"
        );
    }
}

#[test]
fn output_widths_that_overflow_are_refused() {
    let x = data(24, 3, 8);
    let y = target(&x);
    let largest_row = isize::MAX as usize / std::mem::size_of::<f64>();

    let mut cheb = fit_op(&stage("(cheb degree=3)"), &x, &y).unwrap();
    if let FittedOp::Cheb { degree, .. } = &mut cheb {
        *degree = usize::MAX;
    }
    assert!(matches!(output_width(&cheb, 2), Err(Error::Resource(_))));
    // Nothing to multiply: an empty row stays empty and is padded.
    assert_eq!(output_width(&cheb, 0), Ok(1));
    let model = one_stage_model(cheb, 3);
    assert_eq!(
        check_input_width(&model, 2),
        Err(WidthMismatch {
            stage: 0,
            expected: WidthReq::Any,
            got: 2,
            kind: MismatchKind::OutputOverflow,
        })
    );
    assert_eq!(
        model.try_predict(&data(2, 2, 9)).unwrap_err().kind,
        MismatchKind::OutputOverflow
    );

    let mut onehot = fit_op(&stage("(bin_onehot bins=4)"), &x, &y).unwrap();
    if let FittedOp::BinOnehot { bins, .. } = &mut onehot {
        *bins = usize::MAX / 2;
    }
    assert!(matches!(output_width(&onehot, 3), Err(Error::Resource(_))));

    // `fft_mag` writes k columns: up to the largest allocatable row they are only capped (the
    // stage is never applied here), one more is refused, and so is appending to the largest.
    for (k, append, expected) in [
        (1usize << 40, false, Some(512)),
        (largest_row, false, Some(512)),
        (largest_row + 1, false, None),
        (largest_row, true, None),
        (usize::MAX, false, None),
    ] {
        let op = FittedOp::FftMag { k, append };
        match expected {
            Some(width) => assert_eq!(output_width(&op, 1), Ok(width), "k={k} append={append}"),
            None => assert!(
                matches!(output_width(&op, 1), Err(Error::Resource(_))),
                "k={k} append={append}"
            ),
        }
    }
}
