use super::*;

// ---------------------------------------------------------------------------------------------
// The schema-bound (advanced-v3) boundary
// ---------------------------------------------------------------------------------------------

fn numeric_schema() -> DataSchema {
    DataSchema {
        columns: vec![ColumnSpec::numeric("a"), ColumnSpec::numeric("b")],
        row_semantics: RowSemantics::Iid,
    }
}

fn training_table() -> RawTable {
    RawTable::from_csv(
        "a,b\n0,-2\n0.25,-1\n0.5,0\n0.75,1\n1,2\n1.25,3\n1.5,4\n1.75,5\n2,6\n2.25,7\n",
    )
    .unwrap()
}

const TRAINING_TARGET: [f64; 10] = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];

fn bound(source: &str) -> BoundModel {
    fit_table(
        &program(source),
        &TaskSpec::regression(Objective::Rmse).unwrap(),
        &numeric_schema(),
        &training_table(),
        &TRAINING_TARGET,
    )
    .unwrap_or_else(|error| panic!("{source}: {error}"))
}

#[test]
fn schema_bound_stages_return_errors_instead_of_panicking() {
    let table = RawTable::from_csv("a,b\n0.125,-1.5\n0.5,2.5\n2.125,6.5\n").unwrap();
    // (program, path to the first stage's fitted state, edit, message fragment)
    type Tamper = fn(&mut serde_json::Value);
    let cases: &[(&str, &[&str], Tamper, &str)] = &[
        (
            "(pipe X (select_corr k=2) (ridge))",
            &["Legacy", "SelectCorr", "indices"],
            |indices| *indices = serde_json::json!([0, 99]),
            "legacy stage select_corr needs at least 100 input columns, but the feature block has 2",
        ),
        (
            "(pipe X (impute add_indicator=true) (ridge))",
            &["Legacy", "Impute", "indicator_cols"],
            |indicators| indicators.as_array_mut().unwrap().push(serde_json::json!(5)),
            "legacy stage impute has inconsistent fitted state: an indicator column is not a fitted column",
        ),
        (
            "(pipe X (standardize) (ridge))",
            &["Legacy", "Standardize", "scales"],
            |scales| *scales = serde_json::json!([1.0]),
            "legacy stage standardize has inconsistent fitted state: means and scales differ in length",
        ),
        (
            "(pipe X (corr_screen k=32) (ridge))",
            &["CorrScreen", "indices"],
            |indices| *indices = serde_json::json!([0, 7]),
            "corr_screen reads column 7, but the feature block has 2 columns",
        ),
        (
            "(pipe X (pls_project k=2) (ridge))",
            &["PlsProject", "weights"],
            |weights| weights[0].as_array_mut().unwrap().extend([serde_json::json!(0.5), serde_json::json!(0.5)]),
            "pls_project weight spans 4 columns, but the feature block has 2 and the fitted means 2",
        ),
        // Consistent state whose output row no Vec can hold: `Vec` panicked with a capacity
        // overflow.
        (
            "(pipe X (cheb degree=3) (ridge))",
            &["Legacy", "Cheb", "degree"],
            |degree| *degree = serde_json::json!(1u64 << 61),
            "legacy stage cheb output width overflows at 2 input columns",
        ),
        (
            "(pipe X (bin_onehot bins=4) (ridge))",
            &["Legacy", "BinOnehot", "bins"],
            |bins| *bins = serde_json::json!(1u64 << 61),
            "legacy stage bin_onehot output width overflows at 2 input columns",
        ),
        (
            "(pipe X (fft_mag k=2) (ridge))",
            &["Legacy", "FftMag", "k"],
            |k| *k = serde_json::json!(1u64 << 61),
            "legacy stage fft_mag output width overflows at 2 input columns",
        ),
    ];
    for &(source, path, tamper, message) in cases {
        let model = bound(source);
        assert!(
            model.predict_table(&table).is_ok(),
            "{source}: the fitted model"
        );
        let mut json = serde_json::to_value(&model).unwrap();
        let mut state = &mut json["predictor"]["Pipeline"]["ops"][0];
        for key in path {
            state = &mut state[*key];
        }
        assert!(!state.is_null(), "{source}: {path:?} is the fitted state");
        tamper(state);
        let tampered: BoundModel = serde_json::from_value(json).unwrap();
        let outcome = catch_unwind(AssertUnwindSafe(|| tampered.predict_table(&table)))
            .unwrap_or_else(|_| panic!("{source}: predict_table panicked"));
        let error = outcome.expect_err(source).to_string();
        assert_eq!(error, message, "{source}");
    }
}

#[test]
fn schema_bound_tables_with_no_rows_compare_no_width() {
    // With no rows nothing is read, and a stage that generates columns takes its width from the
    // first row, so the block after it is narrower than at fit time: no column, or only the
    // appended input. No width is compared then, and each of these keeps its empty result.
    let empty = RawTable::new(vec!["a".to_string(), "b".to_string()], Vec::new()).unwrap();
    for source in [
        "(pipe X (rank) (standardize) (ridge))",
        "(pipe X (quantile_bins) (winsor_clip) (ridge))",
        "(pipe X (rank append=true) (standardize) (ridge))",
        "(pipe X (cheb degree=3) (standardize) (ridge))",
        "(pipe X (pair_ratio topk=4) (standardize) (ridge))",
        "(pipe X (hinge knots=3) (impute) (ridge))",
        "(pipe X (rank append=true) (pls_project k=2) (ridge))",
        "(pipe X (select_corr k=2) (impute add_indicator=true) (ridge))",
    ] {
        let model = bound(source);
        let predicted = catch_unwind(AssertUnwindSafe(|| model.predict_table(&empty)))
            .unwrap_or_else(|_| panic!("{source}: predict_table panicked"))
            .unwrap_or_else(|error| panic!("{source}: {error}"));
        assert_eq!(predicted, Vec::<f64>::new(), "{source}");
    }

    // The multiclass entry point walks the same stages.
    let mut csv = String::from("a,b\n");
    let mut labels = Vec::new();
    for class in 0..3 {
        for repeat in 0..4 {
            let (class, repeat) = (f64::from(class), f64::from(repeat));
            csv.push_str(&format!(
                "{},{}\n",
                class + 0.1 * repeat,
                0.3 * repeat - class
            ));
            labels.push(class);
        }
    }
    let classes = RawTable::from_csv(&csv).unwrap();
    let task = TaskSpec::multiclass(Objective::MacroF1).unwrap();
    for source in [
        "(pipe X (rank) (standardize) (multiclass_ridge))",
        "(pipe X (rank append=true) (standardize) (multiclass_ridge))",
    ] {
        let model = fit_table(
            &program(source),
            &task,
            &numeric_schema(),
            &classes,
            &labels,
        )
        .unwrap_or_else(|error| panic!("{source}: {error}"));
        let predicted = catch_unwind(AssertUnwindSafe(|| model.predict_multiclass_table(&empty)))
            .unwrap_or_else(|_| panic!("{source}: predict_multiclass_table panicked"))
            .unwrap_or_else(|error| panic!("{source}: {error}"));
        assert!(
            predicted.labels.is_empty() && predicted.probabilities.is_empty(),
            "{source}"
        );
    }

    // Finishing a stage looks the block up at a fitted column that the narrower block lacks:
    // this panicked, and is an error now.
    for (source, stage) in [
        (
            "(pipe X (rank) (select_corr k=2) (ridge))",
            "legacy stage select_corr",
        ),
        (
            "(pipe X (hinge knots=3) (impute add_indicator=true) (ridge))",
            "legacy stage impute",
        ),
        ("(pipe X (hinge) (corr_screen k=32) (ridge))", "corr_screen"),
    ] {
        let model = bound(source);
        let error = catch_unwind(AssertUnwindSafe(|| model.predict_table(&empty)))
            .unwrap_or_else(|_| panic!("{source}: predict_table panicked"))
            .expect_err(source)
            .to_string();
        assert!(
            error.starts_with(&format!("{stage} reads column "))
                && error.ends_with(", but the feature block has 0 columns"),
            "{source}: {error}"
        );
    }
}

#[test]
fn schema_bound_piecewise_linear_with_extra_edge_columns_returns_an_error() {
    // The `piecewise_linear` arm gathers the missing mask at every fitted edge column with an
    // interval, so edges fitted for more columns than the block has panicked there: the gap the
    // WP-1.7 receipt recorded, which this test pinned until kernel2 PR 07 (X3) closed it. The arm
    // refuses them with an error now. A table with no rows keeps its result: nothing is read or
    // gathered, and finishing still refuses the generated column whose source is not in the block.
    let table = RawTable::from_csv("a,b\n0.125,-1.5\n0.5,2.5\n2.125,6.5\n").unwrap();
    let model = bound("(pipe X (piecewise_linear bins=8) (ridge))");
    assert!(model.predict_table(&table).is_ok(), "the fitted model");
    let mut json = serde_json::to_value(&model).unwrap();
    let edges = json["predictor"]["Pipeline"]["ops"][0]["PiecewiseLinear"]["edges"]
        .as_array_mut()
        .expect("the fitted edges");
    assert_eq!(edges.len(), 2, "one edge column per input column");
    let first = edges[0].clone();
    edges.push(first);
    let tampered: BoundModel = serde_json::from_value(json).unwrap();
    let error = catch_unwind(AssertUnwindSafe(|| tampered.predict_table(&table)))
        .unwrap_or_else(|_| panic!("predict_table panicked"))
        .expect_err("edges for a third column")
        .to_string();
    assert_eq!(
        error,
        "piecewise_linear reads column 2, but the feature block has 2 columns"
    );
    let empty = RawTable::new(vec!["a".to_string(), "b".to_string()], Vec::new()).unwrap();
    let error = catch_unwind(AssertUnwindSafe(|| tampered.predict_table(&empty)))
        .unwrap_or_else(|_| panic!("predict_table panicked without rows"))
        .expect_err("a generated column whose source is not in the block")
        .to_string();
    assert_eq!(error, "generated feature source 2 is out of bounds");
}

// ---------------------------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------------------------

#[test]
fn mismatch_messages_name_the_stage_and_the_widths() {
    let cases = [
        (
            MismatchKind::StageInput,
            WidthReq::AtLeast(8),
            "stage 2 needs at least 8 input columns, got 6",
        ),
        (
            MismatchKind::HeadInput,
            WidthReq::Exactly(8),
            "the head needs exactly 8 input columns, got 6",
        ),
        (
            MismatchKind::RaggedInput { row: 4 },
            WidthReq::Exactly(8),
            "ragged input: row 4 has 6 columns, every row needs exactly 8",
        ),
        (
            MismatchKind::InconsistentState {
                reason: "means and scales differ in length",
            },
            WidthReq::Exactly(8),
            "stage 2 has inconsistent fitted state: means and scales differ in length",
        ),
        (
            MismatchKind::OutputOverflow,
            WidthReq::Any,
            "stage 2 output width overflows at input width 6",
        ),
    ];
    for (kind, expected, message) in cases {
        let mismatch = WidthMismatch {
            stage: 2,
            expected,
            got: 6,
            kind,
        };
        assert_eq!(mismatch.to_string(), message);
    }
    assert_eq!(WidthReq::Any.to_string(), "any number of");
}
