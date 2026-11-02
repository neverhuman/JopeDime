use super::*;

/// Assemble the `Model` struct with its `fit`/`predict` methods from the plan.
pub(super) fn model_cpp(stages: &[PlannedStage], head: &PlannedHead, expressions: bool) -> String {
    let (head_fit, head_predict, head_ty) = head_calls_cpp(head);

    let mut fields: Vec<String> = Vec::new();
    let mut fit_body: Vec<String> = Vec::new();
    let mut assigns: Vec<String> = Vec::new();
    let mut predict_body: Vec<String> = Vec::new();

    for (i, stage) in stages.iter().enumerate() {
        let n = i + 1;
        let append = if stage.append { "true" } else { "false" };
        let (fit_call, apply_template) = match &stage.op {
            PlannedOp::ScalarV2 { .. } if expressions => {
                (None, format!("scalar_stage{n}_apply(x, {{append}})"))
            }
            PlannedOp::MapExpr { .. } | PlannedOp::DeriveExpr { .. } if expressions => {
                (None, format!("expression_stage{n}(x, {{append}})"))
            }
            _ => op_calls_cpp(&stage.op),
        };
        let apply = apply_template
            .replace("{state}", &format!("stage{n}"))
            .replace("{append}", append);
        if let Some(ty) = op_struct_cpp(&stage.op) {
            fields.push(format!("    {ty} stage{n};"));
        }
        if let Some(fit_call) = fit_call {
            fit_body.push(format!("    auto stage{n} = {fit_call};"));
            assigns.push(format!("    m.stage{n} = std::move(stage{n});"));
        }
        fit_body.push(format!("    x = {apply};"));
        predict_body.push(format!("    x = {apply};"));
    }

    let mut out = String::new();
    out.push_str("// Feature stages applied in order, then the ridge regression head.\n");
    out.push_str("struct Model {\n");
    if expressions {
        out.push_str("    std::size_t input_width=0;\n");
    }
    for field in &fields {
        out.push_str(field);
        out.push('\n');
    }
    out.push_str(&format!("    {head_ty} head;\n\n"));
    out.push_str(
        "    static Model fit(const std::vector<std::vector<double>>& x_in, const std::vector<double>& y);\n",
    );
    out.push_str(
        "    std::vector<double> predict(const std::vector<std::vector<double>>& x_in) const;\n",
    );
    // B05–B07: public prediction faults on the typed one-stage recipe.
    if stages.len() == 1
        && matches!(
            stages.first().map(|s| &s.op),
            Some(PlannedOp::ScalarV2 { .. })
        )
    {
        out.push_str("    std::pair<std::vector<double>,std::vector<std::uint32_t>> predict_with_faults(const std::vector<std::vector<double>>& rows) const {auto features=scalar_stage1_values(rows);return {predict(rows),std::move(features.second)};}\n");
    }
    out.push_str("};\n\n");

    out.push_str(
        "Model Model::fit(const std::vector<std::vector<double>>& x_in, const std::vector<double>& y) {\n",
    );
    out.push_str("    if (x_in.empty() || y.size() < x_in.size()) {\n");
    out.push_str(
        "        throw std::invalid_argument(\"training data is empty or target is short\");\n",
    );
    out.push_str("    }\n");
    if expressions {
        out.push_str("    const std::size_t input_width = expression_width(x_in);\n");
    }
    let raw_scalar = matches!(
        stages.first().map(|s| &s.op),
        Some(PlannedOp::ScalarV2 { .. })
    );
    if raw_scalar {
        out.push_str("    if(x_in.empty()||y.size()!=x_in.size()){throw std::invalid_argument(\"scalar training shape\");}\n    for(double v:y){if(!std::isfinite(v)||std::abs(v)>1e12){throw std::invalid_argument(\"scalar training labels\");}}\n");
    }
    // B12: preserve the explicit v2 fit contract across native recipes.
    if stages
        .iter()
        .any(|s| matches!(s.op, PlannedOp::RowVectorInputV2 { .. }))
    {
        out.push_str("    if(x_in.empty() || y.size()!=x_in.size())throw std::invalid_argument(\"row-vector training shape\");for(double v:y)if(!std::isfinite(v)||std::abs(v)>1e12)throw std::invalid_argument(\"row-vector training labels\");\n");
    }
    out.push_str(if raw_scalar {
        "    std::vector<std::vector<double>> x = x_in;\n"
    } else {
        "    std::vector<std::vector<double>> x = sanitize(x_in);\n"
    });
    for line in &fit_body {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&format!("    {head_ty} head = {head_fit};\n"));
    if raw_scalar && matches!(head, PlannedHead::Ridge { .. }) {
        out.push_str("    if(std::all_of(y.begin(),y.end(),[](double v){return v==0.0&&std::signbit(v);}))head.w[0]=-0.0;\n");
    }

    out.push_str("    Model m;\n");
    if expressions {
        out.push_str("    m.input_width = input_width;\n");
    }
    for line in &assigns {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("    m.head = std::move(head);\n");
    out.push_str("    return m;\n");
    out.push_str("}\n\n");

    out.push_str(
        "std::vector<double> Model::predict(const std::vector<std::vector<double>>& x_in) const {\n",
    );
    if expressions {
        out.push_str("    if (!x_in.empty() && expression_width(x_in) != input_width) throw std::invalid_argument(\"prediction width\");\n");
    }
    out.push_str(if raw_scalar {
        "    std::vector<std::vector<double>> x = x_in;\n"
    } else {
        "    std::vector<std::vector<double>> x = sanitize(x_in);\n"
    });
    for line in &predict_body {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&format!("    return {head_predict};\n"));
    out.push('}');
    out
}

pub(super) fn expression_stage_cpp(
    n: usize,
    expr: &crate::mathexpr::MathAst,
    map: bool,
) -> Result<String, Vec<String>> {
    let expression = crate::math_expr::MathExpr::from_ast(expr.clone())
        .and_then(|expr| expr.to_cpp17(map))
        .map_err(|e| vec![e.to_string()])?;
    let values = if map {
        format!("for(double x:values) {{ (void)x; output.push_back(math_safe({expression})); }}")
    } else {
        format!("output.push_back(math_safe({expression}));")
    };
    Ok(format!(
        r#"static std::vector<std::vector<double>> expression_stage{n}(const std::vector<std::vector<double>>& input, bool append) {{
    std::vector<std::vector<double>> out;
    out.reserve(input.size());
    for(const auto& values:input) {{
        const double* row=values.data(); const std::size_t row_len=values.size();
        (void)row; (void)row_len;
        std::vector<double> output;
        {values}
        out.push_back(std::move(output));
    }}
    return finish_stage(input, out, append);
}}"#
    ))
}

pub(super) fn symbolic_cpp(
    expr: &crate::mathexpr::MathAst,
    affine: bool,
) -> Result<String, Vec<String>> {
    let expression = crate::math_expr::MathExpr::from_ast(expr.clone())
        .and_then(|expr| expr.to_cpp17(false))
        .map_err(|e| vec![e.to_string()])?;
    let fit = if affine {
        "double fm=0.0,ym=0.0; for(std::size_t i=0;i<x.size();++i){fm+=expression_value(x[i]);ym+=y[i];} fm/=static_cast<double>(x.size());ym/=static_cast<double>(x.size());double cov=0.0,var=0.0;for(std::size_t i=0;i<x.size();++i){double d=expression_value(x[i])-fm;cov+=d*(y[i]-ym);var+=d*d;}if(var>1e-24){m.a=cov/var;m.b=ym-m.a*fm;}else{m.a=0.0;m.b=ym;}"
    } else {
        "(void)y;"
    };
    Ok(format!(
        r#"static double expression_value(const std::vector<double>& values) {{
    const double* row=values.data(); const std::size_t row_len=values.size();
    (void)row; (void)row_len;
    return math_safe({expression});
}}
struct Model {{
    double a=1.0,b=0.0;
    std::size_t input_width=0;
    static Model fit(const std::vector<std::vector<double>>& input,const std::vector<double>& y) {{
        Model m; m.input_width=expression_width(input);
        if(y.size()<input.size()) throw std::invalid_argument("short target");
        const auto x=sanitize(input);
        {fit}
        return m;
    }}
    std::vector<double> predict(const std::vector<std::vector<double>>& input) const {{
        if(!input.empty() && expression_width(input)!=input_width) throw std::invalid_argument("prediction width");
        std::vector<double> out;
        for(const auto& row:sanitize(input)) {{double value=a*expression_value(row)+b;out.push_back(std::isfinite(value)?clamp_range(value,-60.0,60.0):0.0);}}
        return out;
    }}
}};"#
    ))
}
