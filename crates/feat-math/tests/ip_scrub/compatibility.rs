use super::*;

#[test]
fn the_closed_form_docstring_no_longer_cites_the_method_literature() {
    let genome = GenomeIr::parse(SYMBOLIC[0]).unwrap();
    let artifact = emit(&genome, &EmitOptions::default()).unwrap();
    for text in [&artifact.python, &artifact.rust] {
        assert!(!text.to_lowercase().contains("keijzer"));
        // Docstrings are re-wrapped and comment-prefixed by the emitter, so look for the one
        // token that survives any wrapping.
        assert!(
            text.contains("least-squares"),
            "the calibration is still explained"
        );
    }
}

#[test]
fn fitted_portable_and_public_scalar_cpp_are_clean_on_every_supported_head() {
    use feat_math::advanced_v3::{
        emit_fitted_portable, emit_fitted_public_cpp17, lint_fitted_public_text,
    };
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::numeric("amount"),
            ColumnSpec::categorical("kind"),
        ],
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::new(
        vec!["amount".into(), "kind".into()],
        (0..18)
            .map(|i| vec![i.to_string(), format!("group-{}", i % 3)])
            .collect(),
    )
    .unwrap();
    for (head, task, classes) in [
        (
            "ridge",
            TaskSpec::regression(Objective::Rmse).unwrap(),
            0usize,
        ),
        (
            "logit_ridge",
            TaskSpec::binary(Objective::RocAuc).unwrap(),
            2,
        ),
        (
            "multiclass_ridge",
            TaskSpec::multiclass(Objective::Accuracy).unwrap(),
            3,
        ),
        (
            "multiclass_ovr_logistic",
            TaskSpec::multiclass(Objective::Accuracy).unwrap(),
            3,
        ),
        (
            "multiclass_diagonal_lda",
            TaskSpec::multiclass(Objective::Accuracy).unwrap(),
            3,
        ),
        (
            "multiclass_nearest_centroid",
            TaskSpec::multiclass(Objective::Accuracy).unwrap(),
            3,
        ),
    ] {
        let targets = (0..18)
            .map(|i| {
                if classes == 0 {
                    i as f64
                } else {
                    (i % classes) as f64
                }
            })
            .collect::<Vec<_>>();
        let model = fit_table(
            &GenomeIr::parse(&format!(
                "(pipe X (input numeric=median categorical=onehot) ({head}))"
            ))
            .unwrap(),
            &task,
            &schema,
            &table,
            &targets,
        )
        .unwrap();
        let output = emit_fitted_portable(&model).unwrap();
        for text in [&output.python, &output.rust]
            .into_iter()
            .chain(output.javascript.iter())
        {
            assert!(
                lint_fitted_public_text(&model, &[], text).is_empty(),
                "{head}"
            );
        }
        if classes < 3 {
            let cpp = emit_fitted_public_cpp17(&model).unwrap();
            for text in [&cpp.source, &cpp.codec] {
                assert!(
                    lint_fitted_public_text(&model, &[], text).is_empty(),
                    "{head}"
                );
            }
        }
    }
}
