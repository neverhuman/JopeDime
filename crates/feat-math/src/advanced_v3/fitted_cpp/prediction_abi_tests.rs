fn emit_predict(
    target: &Option<FittedTarget>,
    calibration: &CalibrationModel,
    task: TaskKind,
) -> String {
    let steps = emit_prediction_steps(target, calibration, task);
    format!(
        "static double predict_one(V x,V mask){{x=transform(std::move(x),std::move(mask));double value=head_0(x);\n{steps}  return value;\n}}\n"
    )
}

fn emit_predict_transformed(
    target: &Option<FittedTarget>,
    calibration: &CalibrationModel,
    task: TaskKind,
) -> String {
    let steps = emit_prediction_steps(target, calibration, task);
    format!(
        "static double predict_transformed(const V& x){{double value=head_0(x);\n{steps}  return value;\n}}\n"
    )
}

fn emit_prediction_steps(
    target: &Option<FittedTarget>,
    calibration: &CalibrationModel,
    task: TaskKind,
) -> String {
    let calibrated = match calibration {
        CalibrationModel::None => String::new(),
        CalibrationModel::Platt { slope, intercept } => format!(
            "  {{double p=clampv(value,1e-12,1.0-1e-12);value=sigmoid({}*std::log(p/(1.0-p))+{});}}\n",
            f64_lit(*slope),
            f64_lit(*intercept)
        ),
        CalibrationModel::Isotonic { upper, values } => format!(
            "  {{const V upper={};const V levels={};double p=clampv(value,1e-12,1.0-1e-12);if(!levels.empty())value=levels[std::min(lower(upper,p),levels.size()-1)];value=clampv(value,0.0,1.0);}}\n",
            f64_vec(upper),
            f64_vec(values)
        ),
    };
    let inverse = match target {
        None => String::new(),
        Some(FittedTarget::RankGauss {
            table_values,
            table_scores,
            ..
        }) => format!(
            "  value=clampv(rank_inverse(value,{},{}),-60.0,60.0);\n",
            f64_vec(table_scores),
            f64_vec(table_values)
        ),
        Some(FittedTarget::Log1p) => {
            "  value=clampv(std::exp(clampv(value,-745.0,700.0))-1.0,-60.0,60.0);\n".to_string()
        }
        Some(FittedTarget::Winsor { .. }) => "  value=clampv(value,-60.0,60.0);\n".to_string(),
        Some(FittedTarget::ZScore {
            mu,
            sigma,
            clamp_lo,
            clamp_hi,
        }) => format!(
            "  value=clampv(value*{}+{}, {}, {});\n",
            f64_lit(*sigma),
            f64_lit(*mu),
            f64_lit(*clamp_lo),
            f64_lit(*clamp_hi)
        ),
    };
    let final_task = match task {
        TaskKind::Regression | TaskKind::Timeseries => "  value=finitev(value);\n",
        TaskKind::Binary => "  value=std::isfinite(value)?clampv(value,0.0,1.0):0.5;\n",
        TaskKind::Multiclass => "  value=finitev(value);\n",
    };
    format!("{calibrated}{inverse}{final_task}")
}

/// The C ABI of a fitted model. `jope_fitted_predict` returns one status:
///
/// | Status | Meaning |
/// |---|---|
/// | 0 | every row is predicted |
/// | 1 | `columns` is not the fitted column count |
/// | 2 | `rows * columns` overflows |
/// | 3 | a buffer length does not fit `rows` and `columns` |
/// | 4 | a buffer is null while there are rows to predict |
/// | 5 | a categorical cell is neither -1 nor a fitted vocabulary index |
/// | 6 | any other C++ exception: the closed-form math's domain error, an allocation failure |
/// | 7 | a row of a table-aware model does not follow its group's fitted history (`emit_table_abi` only) |
/// | 8 | a stage's fitted state does not fit the row it reads (revision 3): a fitted column index, or the index of a fitted table entry, at or past the end of what it indexes, or `roll` on an empty row |
///
/// Status 8 comes from `FittedWidthError`, which `fitted_index` throws before any out-of-range
/// read; it is caught ahead of status 6 in both ABIs. The host Rust runtime reports model
/// statuses directly while preserving the same fail-closed behavior.
fn emit_abi(columns: usize) -> String {
    format!(
        r#"extern "C" std::size_t jope_fitted_columns() noexcept {{ return {columns}; }}
extern "C" int jope_fitted_predict(
    const double* values,std::size_t values_len,
    const std::uint8_t* missing,std::size_t missing_len,
    std::size_t rows,std::size_t columns,
    double* output,std::size_t output_len) noexcept {{
  if(columns!={columns})return 1;
  if(rows>std::numeric_limits<std::size_t>::max()/columns)return 2;
  const std::size_t cells=rows*columns;
  if(values_len!=cells||missing_len!=cells||output_len<rows)return 3;
  if(rows!=0&&(!values||!missing||!output))return 4;
  try {{
    V x,mask;
    for(std::size_t row=0;row<rows;++row){{
      const std::size_t offset=row*columns;
      if(!encode_row(values+offset,missing+offset,x,mask))return 5;
      output[row]=predict_one(x,mask);
    }}
    return 0;
  }} catch(const FittedWidthError&) {{ return 8; }} catch(...) {{ return 6; }}
}}
"#
    )
}

/// The C ABI of a table-aware fitted model (lag, roll): the statuses of `emit_abi`, with 7.
fn emit_table_abi(columns: usize) -> String {
    format!(
        r#"extern "C" std::size_t jope_fitted_columns() noexcept {{ return {columns}; }}
extern "C" int jope_fitted_predict(
    const double* values,std::size_t values_len,
    const std::uint8_t* missing,std::size_t missing_len,
    std::size_t rows,std::size_t columns,
    double* output,std::size_t output_len) noexcept {{
  if(columns!={columns})return 1;
  if(rows>std::numeric_limits<std::size_t>::max()/columns)return 2;
  const std::size_t cells=rows*columns;
  if(values_len!=cells||missing_len!=cells||output_len<rows)return 3;
  if(rows!=0&&(!values||!missing||!output))return 4;
  try {{
    M table(rows),masks(rows);
    for(std::size_t row=0;row<rows;++row){{
      const std::size_t offset=row*columns;
      if(!encode_row(values+offset,missing+offset,table[row],masks[row]))return 5;
    }}
    if(!transform_table(table,masks,values,missing,rows,columns))return 7;
    for(std::size_t row=0;row<rows;++row)output[row]=predict_transformed(table[row]);
  }} catch(const FittedWidthError&) {{ return 8; }} catch(...) {{ return 6; }}
  return 0;
}}
"#
    )
}

/// B12: export the explicit v2 FeatureWithinRow model through its fitted route.
/// Historical BoundModel artifacts retain their frozen emission bytes.
pub fn emit_row_vector_cpp17(
    model: &crate::kernel::row_vector::Model,
) -> crate::kernel::Result<String> {
    crate::kernel::row_vector::emit::fitted_cpp17(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prelude_marker_names_the_emitter_revision() {
        let marker = format!(
            "// jope-dime fitted-cpp-typed-table-v1 emitter-revision={FITTED_CPP_REVISION}"
        );
        assert_eq!(PRELUDE.lines().next(), Some(marker.as_str()));
    }

    #[test]
    fn only_the_legacy_predictor_paths_pad_an_empty_row() {
        let pad = "  if(x.empty()){x.assign(1,0.0);mask.assign(1,0.0);}\n";
        let ops = [
            FittedOp::Sigmoid { append: false },
            FittedOp::Sigmoid { append: true },
        ];
        // The legacy predictor's transform pads after every stage, as `ops::finish` does.
        let legacy = emit_legacy_transforms(&ops).unwrap();
        assert_eq!(legacy.matches(pad).count(), ops.len(), "{legacy}");
        let mut stage = String::new();
        emit_legacy_op_padded(&mut stage, &ops[0]).unwrap();
        assert!(stage.ends_with(pad), "{stage}");
        // The V3 pipeline path emits the same stages without padding, as its runtime does not pad.
        let v3 = emit_transforms(&ops.map(V3Op::Legacy)).unwrap();
        assert!(!v3.contains("x.assign("), "{v3}");
    }
}
