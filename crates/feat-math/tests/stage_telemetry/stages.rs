use super::*;

/// compat_freeze's 5x4 stage matrix: ordinary values, signed zeros and guard neighbours,
/// non-finite values, magnitudes beyond the clamp, and rounding cases.
fn stage_matrix() -> Rows {
    vec![
        vec![1.0, -2.0, 0.5, 3.0],
        vec![0.0, -0.0, 1e-12, -1e-12],
        vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 2.0],
        vec![1e155, -parse_f64("1.3407807929942596e154"), 27.0, -30.0],
        vec![toward_zero(0.5), 2.5, -1.0, 4_503_599_627_370_497.0],
    ]
}

/// The next representable value toward zero (finite, nonzero input).
fn toward_zero(value: f64) -> f64 {
    f64::from_bits(value.to_bits() - 1)
}

fn parse_f64(text: &str) -> f64 {
    text.parse().expect("decimal literal")
}

/// compat_freeze's `STAGES`, verbatim.
const STAGES: &[&str] = &[
    "(map expr=\"sigmoid(x) + 0.1*col(0)\" append=true)",
    "(map expr=\"x^2 - min(x, c2) + tanh(c3)\")",
    "(map expr=\"step(col(0), 0.0) + 0.5*step(row_mean, c1)\" append=true)",
    "(map expr=\"exp(clip(col(0),-2,2)) + log(abs(col(1)) + 1.0) + asinh(col(2)) + atan(col(3))\")",
    "(map expr=\"clip(x, -1, 1)\")",
    "(map expr=\"x - row_mean + row_std\" append=true)",
    "(map expr=\"c7 + x\")",
    "(map expr=\"clip(1,(-1)^0.5,(-1)^0.5)\")",
    "(map expr=\"x/c0\")",
    "(derive expr=\"row_mean + row_std - c1/(c0 + 3)\" append=true)",
    "(derive expr=\"row_l1 + row_l2 + row_min + row_max\")",
    "(derive expr=\"erf(col(0)) + sign(col(1)) * floor(col(2)) + round(col(3))\" append=true)",
    "(derive expr=\"1/c0 + log(c1) + sqrt(c2) + log1p(c3)\" append=true)",
    "(derive expr=\"asinh(c0*c0) + exp(c3)\")",
    "(derive expr=\"c7\")",
    "(derive expr=\"x\")",
];

#[test]
fn compat_freeze_stages_equal_apply_op_on_the_raw_stage_matrix() {
    let lines: Vec<String> = STAGES
        .iter()
        .map(|stage| format!("    {stage:?},"))
        .collect();
    assert_eq!(lines, compat_freeze_entries("STAGES"));
    let (train, target) = toy_data(40, 5, 42);
    let train: Rows = train.iter().map(|row| row[..4].to_vec()).collect();
    let matrix = stage_matrix();
    for text in STAGES {
        // As compat_freeze's stage_case: fitted on the raw matrix, applied to it.
        let op = fit_op(&parse_stage(text), &matrix, &target).expect("fits");
        check_op(text, &op, &matrix);
        // Fitted as a program on the training rows, then predicting the matrix.
        let program = parse_program(&format!("(pipe X {text} (ridge))"));
        let model = fit(&program, &train, &target).expect("fits");
        let (traced, telemetry) = fit_traced(&program, &train, &target).expect("fits");
        assert_eq!(serialized(&traced), serialized(&model), "{text}");
        assert_eq!(telemetry.stages.len(), 1);
        let (predictions, pass) = predict_traced(&model, &matrix).expect("no zero-width row");
        assert_eq!(
            bits(&predictions),
            bits(&model.predict(&matrix).unwrap()),
            "{text}"
        );
        // NaN, inf and -inf: the entry sanitize replaces them before the stage.
        assert_eq!(pass.input_sanitized, 3, "{text}");
        assert_eq!(pass.stages, replay_ops(text, &model, &matrix), "{text}");
    }
}

/// Raw rows of width 5 with what the entry sanitize would remove and what overflows in the ops.
fn boundary_rows() -> Rows {
    vec![
        vec![0.5, -1.0, 1.5, -0.25, 2.0],
        vec![0.0, -0.0, 1e-300, -1e-300, 5e-324],
        vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0, -1.0],
        vec![1e200, -1e200, 1e155, -1e155, 3.0],
        vec![-2.0, 2.0, 0.0, 1e12, -1e12],
    ]
}

/// One stage for every `FittedOp` variant, as the DSL names them, with the variant it fits.
const EVERY_OP: [(&str, &str); 36] = [
    ("standardize", "Standardize"),
    ("robust_scale", "RobustScale"),
    ("rank", "Rank"),
    ("blom", "Blom"),
    ("quantile_bins", "QuantileBins"),
    ("bin_onehot bins=3", "BinOnehot"),
    ("hinge", "Hinge"),
    ("winsor_clip", "WinsorClip"),
    ("cheb degree=3", "Cheb"),
    ("poly_interact", "PolyInteract"),
    ("rff_cos n=8", "RffCos"),
    ("rff_sin n=8", "RffSin"),
    ("rand_proj k=4", "RandProj"),
    ("landmark_dist k=4", "LandmarkDist"),
    ("oblique_hinge n=4", "ObliqueHinge"),
    ("row_stats", "RowStats"),
    ("pair_prod", "PairProd"),
    ("target_bin_encode", "TargetBinEncode"),
    ("sigmoid", "Sigmoid"),
    ("step t=0.5", "Step"),
    ("norm_row", "NormRow"),
    ("polar", "Polar"),
    ("lag k=1", "Lag"),
    ("roll k=2", "Roll"),
    ("fft_mag k=2", "FftMag"),
    ("row_spectral", "RowSpectral"),
    ("row_diff_stats", "RowDiffStats"),
    ("svd_project k=2 iters=5", "SvdProject"),
    ("nystrom k=4", "Nystrom"),
    ("map expr=\"x*x - c1 + row_mean\"", "MapExpr"),
    ("derive expr=\"c0/c1 + log(c2)\"", "DeriveExpr"),
    ("symlog c=0.7", "Symlog"),
    ("select_corr k=3", "SelectCorr"),
    ("impute strategy=median add_indicator=true", "Impute"),
    ("pair_ratio topk=4", "PairRatio"),
    ("spline knots=4", "Spline"),
];

#[test]
fn every_op_equals_apply_op_with_append_on_and_off() {
    let (mut train, target) = toy_data(64, 5, 42);
    // Exact zeros, so impute fits a fill and an indicator column.
    train[3][2] = 0.0;
    train[11][4] = 0.0;
    let raw = boundary_rows();
    let mut variants = BTreeSet::new();
    for (text, variant) in EVERY_OP {
        for append in [false, true] {
            let stage = parse_stage(&format!("({text} append={append})"));
            let op = fit_op(&stage, &train, &target).unwrap_or_else(|e| panic!("{text}: {e}"));
            let json = serde_json::to_value(&op).expect("serialize the op");
            let tag = json
                .as_object()
                .and_then(|object| object.keys().next())
                .expect("an externally tagged variant");
            assert_eq!(tag, variant, "{text}");
            variants.insert(tag.clone());
            let appended = json[variant]["append"].as_bool();
            assert_eq!(appended, Some(append), "{text}: the fitted append flag");
            check_op(&format!("{text} append={append} (train)"), &op, &train);
            check_op(&format!("{text} append={append} (raw)"), &op, &raw);
        }
    }
    assert_eq!(variants.len(), 36, "one stage per FittedOp variant");
}

#[test]
fn the_width_cap_comes_before_the_epilogue_count() {
    // 300 columns appended to their 300 transforms make 600, cut to MAX_WIDTH = 512.
    let mut row = vec![1.0; 300];
    row[0] = 1e200;
    row[1] = 1e200;
    row[298] = 1e200;
    row[299] = 1e200;
    let x = vec![row];
    // polar's first radius, at merged column 300, overflows and is kept; its last, at merged
    // column 598, overflows and is cut.
    let polar = fit_op(&parse_stage("(polar append=true)"), &x, &[0.0]).expect("fits");
    let counts = check_op("polar append=true, 300 columns", &polar, &x);
    assert_eq!(counts.epilogue_sanitized, 1);
    let (rows, _) = apply_op_traced(&polar, &x);
    assert_eq!(rows[0].len(), MAX_WIDTH);
    // An expression's counts cover every cell it generated, the cut ones included.
    let map = fit_op(&parse_stage("(map expr=\"x\" append=true)"), &x, &[0.0]).expect("fits");
    let counts = check_op("map append=true, 300 columns", &map, &x);
    assert_eq!(counts.evaluated_cells, 300);
    assert_eq!(counts.epilogue_sanitized, 0);
}
