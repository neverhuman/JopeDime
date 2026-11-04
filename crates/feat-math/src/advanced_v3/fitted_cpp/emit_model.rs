pub fn emit_fitted_cpp17(model: &BoundModel) -> Result<FittedCppArtifact> {
    supports_fitted_cpp17(model)?;
    let codec = serde_json::to_string(&Codec {
        schema: "jope-dime.fitted-cpp-typed-table.v1",
        emitter_revision: FITTED_CPP_REVISION,
        schema_sha256: &model.schema_sha256,
        vocabulary_sha256: &model.vocabulary_sha256,
        columns: model
            .schema
            .columns
            .iter()
            .zip(&model.input.columns)
            .map(|(schema, fitted)| match (schema.kind, fitted) {
                (ColumnKind::Numeric, FittedColumn::Numeric { .. }) => CodecColumn::Numeric,
                (ColumnKind::Categorical, FittedColumn::Categorical { vocabulary, .. }) => {
                    CodecColumn::Categorical { vocabulary }
                }
                _ => unreachable!("validated fitted input/schema correspondence"),
            })
            .collect(),
        categorical_unknown: -1,
        missing_bitmap: "one byte per raw cell; nonzero means missing",
    })?;

    let (head_functions, transform, predict, missing_aware, table_aware) = match &model.predictor {
        BoundPredictor::Pipeline {
            target,
            ops,
            head,
            calibration,
            missing_aware,
        } => {
            let mut functions = String::new();
            emit_head_function("head_0", head, &mut functions)?;
            let table_aware = ops
                .iter()
                .any(|op| matches!(op, V3Op::Lag { .. } | V3Op::Roll { .. }));
            if table_aware {
                (
                    functions,
                    emit_table_transforms(model, ops)?,
                    emit_predict_transformed(target, calibration, model.task.kind),
                    *missing_aware,
                    true,
                )
            } else {
                (
                    functions,
                    emit_transforms(ops)?,
                    emit_predict(target, calibration, model.task.kind),
                    *missing_aware,
                    false,
                )
            }
        }
        BoundPredictor::Legacy(crate::model::Model::Pipeline { target, ops, head }) => {
            let mut functions = String::new();
            emit_legacy_head("head_0", head, &mut functions)?;
            (
                functions,
                emit_legacy_transforms(ops)?,
                emit_predict(target, &CalibrationModel::None, model.task.kind),
                false,
                false,
            )
        }
        BoundPredictor::Legacy(crate::model::Model::Symbolic { expr, a, b }) => {
            let expression = expr.to_cpp17(false)?;
            let functions = format!(
                "static double head_0(const V& values){{const double* row=values.data();const std::size_t row_len=values.size();(void)row;(void)row_len;double value=math_safe({expression});double result={a}*value+{b};return std::isfinite(result)?clampv(result,-60.0,60.0):0.0;}}\n",
                a = f64_lit(*a),
                b = f64_lit(*b),
            );
            (
                functions,
                "static V transform(V x,V){return x;}\n".to_string(),
                emit_predict(&None, &CalibrationModel::None, model.task.kind),
                false,
                false,
            )
        }
        BoundPredictor::Legacy(crate::model::Model::Portfolio(portfolio)) => {
            let mut functions = String::new();
            emit_portfolio("head_0", portfolio, &mut functions)?;
            (
                functions,
                "static V transform(V x,V){return x;}\n".to_string(),
                emit_predict(&None, &CalibrationModel::None, model.task.kind),
                false,
                false,
            )
        }
    };
    let abi = if table_aware {
        emit_table_abi(model.schema.columns.len())
    } else {
        emit_abi(model.schema.columns.len())
    };
    let source = format!(
        "{PRELUDE}\n{head_functions}\n{encode}\n{transform}\n{predict}\n{abi}",
        encode = emit_input(model, missing_aware)?,
    );
    // The caller's column names and categorical vocabularies are masked before the lint runs
    // (`emit::user_text`), so the codec's vocabulary arrays cannot trip it on an innocent name.
    let user = super::emit::user_text(model);
    let portfolio = matches!(
        &model.predictor,
        BoundPredictor::Pipeline {
            head: V3Head::ClassificationPortfolio(_),
            ..
        }
    );
    for (what, text) in [
        ("fitted C++17 artifact", &source),
        ("fitted C++17 codec", &codec),
    ] {
        let hits = super::emit::lint_shipped_masked(text, &user, portfolio);
        if !hits.is_empty() {
            bail!("banned vocabulary in {what}: {}", hits.join(", "));
        }
    }
    Ok(FittedCppArtifact {
        source_sha256: text_hash(&source),
        codec_sha256: text_hash(&codec),
        source,
        codec,
    })
}
