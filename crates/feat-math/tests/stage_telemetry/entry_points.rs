use super::*;

#[test]
fn the_traced_entry_points_refuse_a_zero_width_input() {
    let ragged: Rows = vec![vec![0.5, 1.0], Vec::new(), vec![2.0, -1.0]];
    let empty_rows: Rows = vec![Vec::new(); 3];
    let y = vec![1.0, 2.0, 3.0];
    let programs = [
        "(pipe X (ridge))",
        "(pipe X (standardize) (ridge))",
        "(sym raw \"row_sum\")",
        "(portfolio (member (pipe X (ridge))) (member (pipe X (standardize) (ridge))) \
         (blend kind=nnls folds=5 repeats=1 guard=clip))",
    ];
    for source in programs {
        let program = parse_program(source);
        for (x, row) in [(&ragged, 1), (&empty_rows, 0)] {
            let error = fit_traced(&program, x, &y).expect_err("refused");
            assert_eq!(
                error.downcast_ref::<ZeroWidthInput>(),
                Some(&ZeroWidthInput { row }),
                "{source}"
            );
            assert_eq!(
                error.to_string(),
                format!("zero-width input: row {row} has no columns")
            );
            assert!(error.to_string().starts_with(ZeroWidthInput::REASON));
        }
        // fit's own check comes first, with fit's message.
        let error = fit_traced(&program, &Vec::new(), &[]).expect_err("refused");
        let legacy = fit(&program, &Vec::new(), &[]).expect_err("refused");
        assert_eq!(error.to_string(), legacy.to_string());
        let error = fit_traced(&program, &empty_rows, &y[..2]).expect_err("refused");
        assert_eq!(
            error.to_string(),
            "training data is empty or target is short"
        );
    }
    // Public fit now rejects every zero-width row before a transform or head can index it.
    for source in programs {
        let program = parse_program(source);
        for (x, row) in [(&empty_rows, 0), (&ragged, 1)] {
            let error = fit(&program, x, &y).expect_err("zero-width rows are refused");
            assert_eq!(
                error.downcast_ref::<WidthMismatch>(),
                Some(&WidthMismatch {
                    stage: 0,
                    expected: WidthReq::AtLeast(1),
                    got: 0,
                    kind: MismatchKind::StageInput,
                }),
                "{source}, row {row}"
            );
        }
    }

    let (train, target, _) = fitted_surface_data();
    for source in ["(pipe X (standardize) (ridge))", "(sym raw \"c0 + c1\")"] {
        let model = fit(&parse_program(source), &train, &target).expect("fits");
        let error = predict_traced(&model, &ragged).expect_err("refused");
        assert_eq!(
            error.downcast_ref::<ZeroWidthInput>(),
            Some(&ZeroWidthInput { row: 1 })
        );
        // A set with no rows is not zero-width.
        let (predictions, pass) = predict_traced(&model, &Vec::new()).expect("accepted");
        assert!(predictions.is_empty() && model.predict(&Vec::new()).unwrap().is_empty());
        assert_eq!(pass.stages.len(), 1);
        assert_eq!(pass.stages[0].evaluated_cells, 0);
        assert!(!pass.stages[0].constant_output);
    }
    assert_eq!(
        ZeroWidthInput::find(&ragged),
        Some(ZeroWidthInput { row: 1 })
    );
    assert_eq!(ZeroWidthInput::find(&train), None);
    assert_eq!(ZeroWidthInput::find(&Vec::new()), None);
}

#[test]
fn the_telemetry_types_stay_shareable_across_threads() {
    fn shareable<T: Send + Sync + 'static>() {}
    shareable::<StageFaultCounts>();
    shareable::<PipelineTelemetry>();
    shareable::<ZeroWidthInput>();
}
