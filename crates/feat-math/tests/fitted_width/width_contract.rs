use super::*;

// Every stage: its width class, and `output_width` against `apply_op`
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum Class {
    /// `Exactly(fitted width)`.
    Exact,
    /// `AtLeast(largest fitted column index + 1)`.
    Indices,
    /// `AtLeast(1)`.
    NonEmpty,
    Any,
}

/// One program per `FittedOp` variant (36), fitted on 8 columns.
const VARIANTS: &[(&str, Class)] = &[
    ("(standardize)", Class::Exact),
    ("(robust_scale)", Class::Exact),
    ("(rank)", Class::Exact),
    ("(blom)", Class::Exact),
    ("(quantile_bins bins=5)", Class::Exact),
    ("(bin_onehot bins=4)", Class::Exact),
    ("(hinge knots=3)", Class::Exact),
    ("(winsor_clip q=0.1)", Class::Exact),
    ("(cheb degree=3)", Class::Any),
    ("(poly_interact)", Class::Indices),
    ("(rff_cos n=5 gamma=0.7 seed=11)", Class::Exact),
    ("(rff_sin n=5 gamma=0.7 seed=11)", Class::Exact),
    ("(rand_proj k=5 seed=11)", Class::Exact),
    ("(landmark_dist k=5 gamma=0.7 seed=11)", Class::Exact),
    ("(oblique_hinge n=5 seed=11)", Class::Exact),
    ("(row_stats)", Class::Any),
    ("(pair_prod topk=4)", Class::Indices),
    ("(target_bin_encode bins=5 prior=3)", Class::Exact),
    ("(sigmoid)", Class::Any),
    ("(step t=0.2)", Class::Any),
    ("(norm_row)", Class::Any),
    ("(polar)", Class::Any),
    ("(lag k=2)", Class::Any),
    ("(roll k=3)", Class::NonEmpty),
    ("(fft_mag k=2)", Class::Any),
    ("(row_spectral)", Class::Any),
    ("(row_diff_stats)", Class::Any),
    ("(svd_project k=3 iters=8 seed=11)", Class::Exact),
    ("(nystrom k=5 gamma=0.7 seed=11)", Class::Exact),
    ("(map expr=\"asinh(x)+row_mean\")", Class::Any),
    ("(derive expr=\"row_l2+col(0)\")", Class::Any),
    ("(symlog c=0.7)", Class::Any),
    ("(select_corr k=3)", Class::Indices),
    ("(impute strategy=median add_indicator=true)", Class::Exact),
    ("(pair_ratio topk=5)", Class::Indices),
    ("(spline knots=4)", Class::Exact),
];

/// Widths each fitted stage is applied at, where its requirement admits them.
const WIDTHS: &[usize] = &[0, 1, 2, 3, 5, 7, 8, 9, 16, 600];

fn largest_index(op: &FittedOp) -> Option<usize> {
    match op {
        FittedOp::SelectCorr { indices, .. } => indices.iter().copied().max(),
        FittedOp::PolyInteract { pairs, .. }
        | FittedOp::PairProd { pairs, .. }
        | FittedOp::PairRatio { pairs, .. } => pairs.iter().map(|&(a, b)| a.max(b)).max(),
        _ => None,
    }
}

#[test]
fn every_stage_has_a_width_class_and_its_output_width_matches_apply_op() {
    let x = data(40, 8, 1);
    let y = target(&x);
    let mut variants = BTreeSet::new();
    let mut checked = 0;
    for &(source, class) in VARIANTS {
        for append in [false, true] {
            let source = if append {
                with_append(source)
            } else {
                source.to_string()
            };
            let op = fit_op(&stage(&source), &x, &y).unwrap_or_else(|e| panic!("{source}: {e}"));
            variants.insert(variant(&op));
            assert_eq!(check_fitted_state(&op), Ok(()), "{source}: fitted state");
            let required = required_input_width(&op);
            let expected = match class {
                Class::Exact => WidthReq::Exactly(8),
                Class::Indices => {
                    let largest = largest_index(&op).expect("fitted column indices");
                    WidthReq::AtLeast(largest + 1)
                }
                Class::NonEmpty => WidthReq::AtLeast(1),
                Class::Any => WidthReq::Any,
            };
            assert_eq!(required, expected, "{source}: width class");
            for (index, &width) in WIDTHS.iter().enumerate() {
                if !required.admits(width) {
                    assert!(
                        matches!(output_width(&op, width), Err(Error::Shape(_))),
                        "{source}: width {width} is refused"
                    );
                    continue;
                }
                let rows = data(3, width, 100 + index as u64);
                let produced = apply_op(&op, &rows).unwrap();
                let width_out = output_width(&op, width)
                    .unwrap_or_else(|e| panic!("{source} at width {width}: {e}"));
                assert!(
                    produced.iter().all(|row| row.len() == width_out),
                    "{source} at width {width}: apply_op wrote {:?} columns, output_width says {width_out}",
                    produced.iter().map(Vec::len).collect::<Vec<_>>()
                );
                checked += 1;
            }
        }
    }
    assert_eq!(variants.len(), 36, "every FittedOp variant: {variants:?}");
    assert!(checked > 300, "{checked} (stage, append, width) cases");
    println!(
        "STAGES {} variants, {checked} (stage, append, width) cases",
        variants.len()
    );
}

#[test]
fn output_width_caps_at_max_width_and_pads_empty_rows() {
    // (stage, fitted width, applied width, expected output width)
    let cases: &[(&str, usize, usize, usize)] = &[
        // 100 + 800 columns, capped.
        ("(cheb degree=8 append=true)", 8, 100, 512),
        // 100 columns of 8 one-hot bins, capped.
        ("(bin_onehot bins=8)", 100, 100, 512),
        // 40 columns of 16 knots, capped.
        ("(hinge knots=16)", 40, 40, 512),
        // One column leaves no pair: the empty row is padded to [0.0].
        ("(pair_ratio topk=4)", 1, 1, 1),
        ("(pair_ratio topk=4)", 1, 7, 1),
        // Appending an empty output keeps the input.
        ("(pair_ratio topk=4 append=true)", 1, 7, 7),
        // An empty input row gives an empty output row, padded.
        ("(sigmoid)", 3, 0, 1),
        ("(cheb degree=4 append=true)", 3, 0, 1),
        ("(map expr=\"x+1\")", 3, 0, 1),
        // Fixed widths do not depend on the input width.
        ("(row_stats)", 3, 0, 10),
        ("(row_spectral append=true)", 3, 1, 6),
        ("(derive expr=\"col(0)\")", 3, 0, 1),
    ];
    for &(source, fitted, applied, expected) in cases {
        let x = data(40, fitted, 2);
        let op =
            fit_op(&stage(source), &x, &target(&x)).unwrap_or_else(|e| panic!("{source}: {e}"));
        assert_eq!(
            output_width(&op, applied),
            Ok(expected),
            "{source} at {applied}"
        );
        let produced = apply_op(&op, &data(3, applied, 3)).unwrap();
        assert!(
            produced.iter().all(|row| row.len() == expected),
            "{source} at {applied}"
        );
    }

    // A constant column gives its raw value alone; the others their raw value and k - 2 terms.
    let mut x = data(40, 3, 4);
    for row in &mut x {
        row[0] = 1.0;
    }
    let op = fit_op(&stage("(spline knots=5)"), &x, &target(&x)).unwrap();
    assert_eq!(output_width(&op, 3), Ok(1 + 4 + 4));
    assert!(apply_op(&op, &x).unwrap().iter().all(|row| row.len() == 9));
}

// ---------------------------------------------------------------------------------------------
// The workplan cases and their siblings
// ---------------------------------------------------------------------------------------------

#[test]
fn select_corr_fitted_on_eight_columns_refuses_six() {
    // The target reads columns 1, 5 and 7 only, so k=3 keeps exactly those.
    let x = data(64, 8, 3);
    let y: Vec<f64> = x.iter().map(|row| row[1] - row[5] + row[7]).collect();
    let model = fit(&program("(pipe X (select_corr k=3) (ridge))"), &x, &y).unwrap();
    let ops = fitted_ops(&model);
    assert!(
        matches!(&ops[0], FittedOp::SelectCorr { indices, .. } if indices == &[1, 5, 7]),
        "{:?}",
        ops[0]
    );
    assert_eq!(required_input_width(&ops[0]), WidthReq::AtLeast(8));

    let valid = data(16, 8, 4);
    assert_eq!(
        bits(&model.try_predict(&valid).unwrap()),
        bits(&model.predict(&valid).unwrap())
    );

    let narrow = data(16, 6, 5);
    assert_eq!(
        model.try_predict(&narrow),
        Err(WidthMismatch {
            stage: 0,
            expected: WidthReq::AtLeast(8),
            got: 6,
            kind: MismatchKind::StageInput,
        })
    );
    assert!(
        panics(|| model.predict(&narrow).unwrap()),
        "predict still panics"
    );

    // Wider rows are read at the same indices, which the contract admits.
    let wide = data(16, 9, 6);
    assert_eq!(
        bits(&model.try_predict(&wide).unwrap()),
        bits(&model.predict(&wide).unwrap())
    );
}

#[test]
fn impute_fitted_on_four_columns_refuses_five() {
    let model = fit_program("(pipe X (impute) (ridge))", &data(64, 4, 7));
    assert_eq!(
        required_input_width(&fitted_ops(&model)[0]),
        WidthReq::Exactly(4)
    );
    let valid = data(16, 4, 8);
    assert_eq!(
        bits(&model.try_predict(&valid).unwrap()),
        bits(&model.predict(&valid).unwrap())
    );
    let refusal = Err(WidthMismatch {
        stage: 0,
        expected: WidthReq::Exactly(4),
        got: 5,
        kind: MismatchKind::StageInput,
    });
    // A zero in column 4 makes predict read a fill value that was never fitted.
    let wide = data(16, 5, 9);
    assert!(wide.iter().any(|row| row[4] == 0.0));
    assert_eq!(model.try_predict(&wide), refusal);
    assert!(
        panics(|| model.predict(&wide).unwrap()),
        "predict still panics"
    );
    // Without one, predict passes column 4 through unfitted and does not panic.
    let unfilled: Rows = wide
        .iter()
        .map(|row| row.iter().map(|value| value + 0.125).collect())
        .collect();
    assert!(unfilled
        .iter()
        .all(|row| row.iter().all(|value| *value != 0.0)));
    assert_eq!(model.try_predict(&unfilled), refusal);
    assert!(!panics(|| model.predict(&unfilled).unwrap()));
}

#[derive(Clone, Copy, Debug)]
enum Side {
    /// One column fewer than the stage requires.
    Narrower,
    /// One column more than an `Exactly` stage requires.
    Wider,
}

/// (program, fitted width, the side of the refused width, whether `predict` panics there)
const REFUSALS: &[(&str, usize, Side, bool)] = &[
    (
        "(pipe X (pair_ratio topk=4) (ridge))",
        6,
        Side::Narrower,
        true,
    ),
    (
        "(pipe X (pair_prod topk=4) (ridge))",
        6,
        Side::Narrower,
        true,
    ),
    ("(pipe X (poly_interact) (ridge))", 4, Side::Narrower, true),
    (
        "(pipe X (select_corr k=3) (ridge))",
        8,
        Side::Narrower,
        true,
    ),
    ("(pipe X (spline knots=4) (ridge))", 4, Side::Wider, true),
    ("(pipe X (standardize) (ridge))", 4, Side::Wider, true),
    ("(pipe X (standardize) (ridge))", 4, Side::Narrower, false),
    // Every column has a missing cell, so the fitted indicators include the last one.
    (
        "(pipe X (impute add_indicator=true) (ridge))",
        5,
        Side::Narrower,
        true,
    ),
    ("(pipe X (impute) (ridge))", 4, Side::Wider, true),
    ("(pipe X (winsor_clip) (ridge))", 4, Side::Wider, true),
    ("(pipe X (rank) (ridge))", 4, Side::Wider, true),
    ("(pipe X (bin_onehot bins=3) (ridge))", 4, Side::Wider, true),
    ("(pipe X (target_bin_encode) (ridge))", 4, Side::Wider, true),
    ("(pipe X (hinge) (ridge))", 4, Side::Wider, true),
    // Dot products truncate: predict misreads without panicking.
    ("(pipe X (rff_cos n=6) (ridge))", 4, Side::Wider, false),
    ("(pipe X (svd_project k=2) (ridge))", 4, Side::Wider, false),
    ("(pipe X (nystrom k=6) (ridge))", 4, Side::Narrower, false),
];
