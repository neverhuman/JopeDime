use super::*;

/// Emit version-two raw-cell prediction without changing the stored object or legacy semantics.
/// `predict_cells` reads contract order; `predict_table` matches headers by name and ignores extras.
pub fn emit_pipeline(pipeline: &MathPipeline) -> Result<String> {
    let input = &pipeline.input;
    ensure!(
        input.prediction_version == F64_PREDICTION_VERSION,
        "fitted Rust requires prediction_version=2"
    );
    // Reuse the public constructor's complete input validation, including mutable public fields.
    InputContract::new(
        input.columns.clone(),
        input.selected.clone(),
        input.target.clone(),
    )?;
    ensure!(
        input.task == "regression",
        "fitted Rust input task must be regression"
    );
    let genome = GenomeIr::parse(&pipeline.state.dsl)?;
    // Every exported math-pipeline carries 64 engine-derived numeric self-checks.
    let rows: Vec<Vec<f64>> = (0..64)
        .map(|i| {
            (0..input.selected.len())
                .map(|j| ((i + 17 * j) % 64) as f64 / 63.0)
                .collect()
        })
        .collect();
    let expected = pipeline.state.model.predict(&rows)?;
    let checks: Vec<CheckRow> = rows
        .into_iter()
        .zip(expected)
        .map(|(inputs, expected)| CheckRow { inputs, expected })
        .collect();
    let mut source = emit_with_checks(
        &genome,
        &pipeline.state.model,
        input.selected.len(),
        &checks,
    )?;
    let names: Vec<_> = input
        .columns
        .iter()
        .map(|c| format!("{:?}", c.name))
        .collect();
    let scalers: Vec<_> = input
        .columns
        .iter()
        .map(|c| format!("({:?}, {:?}, {})", c.min, c.max, c.constant))
        .collect();
    source.push_str(&format!(
        "\nconst MODEL_COLUMNS: &[&str] = &[{}];\n\
         const INPUT_SCALERS: &[(f64, f64, bool)] = &[{}];\n\
         const INPUT_SELECTED: &[usize] = &{:?};\n\
         const TARGET_MIN: f64 = {:?};\nconst TARGET_SCALE: f64 = {:?};\n{}",
        names.join(", "),
        scalers.join(", "),
        input.selected,
        input.target.min,
        input.target.scale,
        INPUT_ADAPTER
    ));
    // Input names are user data, not generated public vocabulary.
    Ok(source)
}

pub(super) fn lint(source: &str) -> Result<()> {
    let hits = feat_math::transpile::lint_shipped_text(source);
    ensure!(
        hits.is_empty(),
        "banned vocabulary in Rust artifact: {}",
        hits.join(", ")
    );
    Ok(())
}

pub(super) fn no_null(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Array(items) => items.iter().all(no_null),
        Value::Object(fields) => fields.values().all(no_null),
        _ => true,
    }
}

pub(super) fn is_identity_transform(transform: &OutputTransform) -> bool {
    transform.intercept == 0.0 && transform.slope == 1.0 && transform.lower.is_none()
}

pub(super) fn pipeline_fields(
    source: &str,
    stages: &[PlannedStage],
    ops: &[FittedOp],
    planned_head: &PlannedHead,
    head: &FittedHead,
) -> Result<Vec<String>> {
    ensure!(
        stages.len() == ops.len(),
        "fitted Rust stage count differs from recipe"
    );
    let mut fields = Vec::new();
    for (i, (stage, op)) in stages.iter().zip(ops).enumerate() {
        if let Some(value) = stage_literal(source, stage, op)? {
            fields.push(format!("stage{}: {value}", i + 1));
        }
    }
    fields.push(format!(
        "head: {}",
        head_literal(source, planned_head, head)?
    ));
    Ok(fields)
}

fn stage_literal(source: &str, stage: &PlannedStage, op: &FittedOp) -> Result<Option<String>> {
    use FittedOp as F;
    use PlannedOp as P;
    let value = serde_json::to_value(op)?;
    let state = value
        .as_object()
        .context("fitted operation variant")?
        .values()
        .next()
        .context("fitted operation")?;
    ensure!(
        state["append"].as_bool() == Some(stage.append),
        "fitted Rust append differs from recipe"
    );
    // Prediction-time literals must agree. Fit-only seeds, regularizers and knot counts need
    // not: their learned output is carried in the constants instead of being recomputed.
    let matches = match (&stage.op, op) {
        (P::Standardize, F::Standardize { .. })
        | (P::RobustScale, F::RobustScale { .. })
        | (P::Rank, F::Rank { .. })
        | (P::Blom, F::Blom { .. })
        | (P::Hinge { .. }, F::Hinge { .. })
        | (P::WinsorClip { .. }, F::WinsorClip { .. })
        | (P::PolyInteract, F::PolyInteract { .. })
        | (P::RffCos { .. }, F::RffCos { .. })
        | (P::RffSin { .. }, F::RffSin { .. })
        | (P::RandProj { .. }, F::RandProj { .. })
        | (P::ObliqueHinge { .. }, F::ObliqueHinge { .. })
        | (P::RowStats, F::RowStats { .. })
        | (P::PairProd { .. }, F::PairProd { .. })
        | (P::TargetBinEncode { .. }, F::TargetBinEncode { .. })
        | (P::Sigmoid, F::Sigmoid { .. })
        | (P::NormRow, F::NormRow { .. })
        | (P::Polar, F::Polar { .. })
        | (P::FftMag { .. }, F::FftMag { .. })
        | (P::RowSpectral, F::RowSpectral { .. })
        | (P::RowDiffStats, F::RowDiffStats { .. })
        | (P::SvdProject { .. }, F::SvdProject { .. })
        | (P::SelectCorr { .. }, F::SelectCorr { .. })
        | (P::Impute { .. }, F::Impute { .. })
        | (P::PairRatio { .. }, F::PairRatio { .. })
        | (P::Spline { .. }, F::Spline { .. }) => true,
        (P::QuantileBins { bins: a }, F::QuantileBins { bins: b, .. })
        | (P::BinOnehot { bins: a }, F::BinOnehot { bins: b, .. }) => a == b,
        (P::Cheb { degree: a }, F::Cheb { degree: b, .. }) => a == b,
        (P::LandmarkDist { gamma: a, .. }, F::LandmarkDist { gamma: b, .. })
        | (P::Nystrom { gamma: a, .. }, F::Nystrom { gamma: b, .. })
        | (P::Symlog { c: a }, F::Symlog { c: b, .. })
        | (P::Step { t: a }, F::Step { t: b, .. }) => a.to_bits() == b.to_bits(),
        (P::Lag { k: a }, F::Lag { k: b, .. }) | (P::Roll { k: a }, F::Roll { k: b, .. }) => a == b,
        (
            P::MapExpr { source: a, .. },
            F::MapExpr {
                source: b, expr, ..
            },
        )
        | (
            P::DeriveExpr { source: a, .. },
            F::DeriveExpr {
                source: b, expr, ..
            },
        ) => {
            a == b
                && serde_json::to_value(feat_math::math_expr::MathExpr::parse(a)?)?
                    == serde_json::to_value(expr)?
        }
        _ => false,
    };
    ensure!(
        matches,
        "fitted Rust operation or prediction parameter differs from recipe: {:?}",
        stage.op
    );
    let Some(ty) = spec::op_struct_name(&stage.op) else {
        return Ok(None);
    };
    let state = if let F::FftMag { k, .. } = op {
        json!(k)
    } else {
        state.clone()
    };
    Ok(Some(literal(source, ty, &state)?))
}

fn head_literal(source: &str, planned: &PlannedHead, head: &FittedHead) -> Result<String> {
    use FittedHead as F;
    use PlannedHead as P;
    let serialized = serde_json::to_value(head)?;
    let state = serialized
        .as_object()
        .context("fitted head variant")?
        .values()
        .next()
        .context("fitted head")?;
    let (ty, state) = match (planned, head) {
        (P::Ridge { .. }, F::Ridge(_)) => ("RidgeRegression", state.clone()),
        (P::RidgeGcv, F::RidgeGcv(_)) => ("GcvRidgeRegression", state.clone()),
        (P::ElasticNet { .. }, F::ElasticNet(_)) => ("ElasticNet", state.clone()),
        (P::Gbm(_), F::Gbm(_)) => ("GradientBoostedTrees", state.clone()),
        (P::Blend { .. }, F::Ensemble { ridge, gbm, w }) => (
            "BlendedRegression",
            json!({"linear":ridge, "boosted":gbm, "weight":w}),
        ),
        (P::HistGbm2 { .. }, F::HistGbm2(_)) => ("LeafwiseBoostedTrees", state.clone()),
        (P::Krr { .. }, F::Krr(_)) => ("KernelRidgeRegression", state.clone()),
        (P::Gpr { .. }, F::Gpr(_)) => ("GaussianProcessArdRegression", state.clone()),
        (P::GprKernel { spec, .. }, F::GprKernel(fitted))
            if serde_json::to_value(spec)? == serde_json::to_value(&fitted.spec)? =>
        {
            ("GaussianProcessCompositeKernelRegression", state.clone())
        }
        (P::Ksmooth { k, gamma }, F::Ksmooth(fitted))
            if *k == fitted.k && gamma.to_bits() == fitted.gamma.to_bits() =>
        {
            ("NadarayaWatsonRegression", state.clone())
        }
        (P::Stack, F::Stack(_)) => ("GreedyEnsembleSelection", state.clone()),
        (P::Portfolio(plan), F::Portfolio(fitted)) => {
            return portfolio_literal(source, plan, fitted)
        }
        _ => bail!("fitted Rust head or prediction parameter differs from recipe"),
    };
    literal(source, ty, &state)
}

fn portfolio_literal(
    source: &str,
    plan: &spec::PortfolioPlan,
    fitted: &feat_math::portfolio::PortfolioModel,
) -> Result<String> {
    use feat_math::portfolio::FittedMemberState as F;
    ensure!(
        fitted.members.len() == fitted.weights.len(),
        "fitted Rust ensemble weights differ from members"
    );
    let entries = fitted.members.iter().zip(&fitted.weights).map(|(member, weight)| -> Result<String> {
        let (index, planned) = plan.members.iter().enumerate().find(|(_, p)| p.name == member.name).context("fitted Rust ensemble member absent from recipe")?;
        ensure!(member.proj_view == planned.proj_view, "fitted Rust ensemble member view differs");
        let state = match (&planned.kind, &member.state) {
            (_, F::Mean | F::Skipped) => "None".to_owned(),
            (spec::PlannedMemberKind::Pipeline { stages, head }, F::Fitted(model)) => {
                let Model::Pipeline { target: None, ops, head: fitted_head } = model.as_ref() else { bail!("fitted Rust ensemble component is not a plain pipeline"); };
                let fields = pipeline_fields(source, stages, ops, head, fitted_head)?;
                format!("Some(ComponentState::C{}(Component{} {{ {} }}))", index + 1, index + 1, fields.join(", "))
            }
            (spec::PlannedMemberKind::GroupMean, F::GroupMean(model)) => {
                let value = serde_json::to_value(model)?;
                let table = value["table"].as_array().context("fitted Rust group table")?;
                let mapped = json!({"keys":table.iter().map(|v| v[0].clone()).collect::<Vec<_>>(), "values":table.iter().map(|v| v[1].clone()).collect::<Vec<_>>(), "fallback":value["g_mean"]});
                format!("Some(ComponentState::C{}({}))", index + 1, literal(source, "DuplicateRowAverage", &mapped)?)
            }
            _ => bail!("fitted Rust ensemble component differs from recipe"),
        };
        ensure!(weight.is_finite(), "non-finite fitted ensemble weight");
        Ok(format!("EnsembleEntry {{ projected: {}, weight: {weight:?}, state: {state} }}", member.proj_view))
    }).collect::<Result<Vec<_>>>()?;
    let proj = match &fitted.proj {
        None => "None".to_owned(),
        Some(op @ FittedOp::SvdProject { append: false, .. }) => {
            let v = serde_json::to_value(op)?;
            format!(
                "Some({})",
                literal(source, "PrincipalComponentProjection", &v["SvdProject"])?
            )
        }
        _ => bail!("fitted Rust ensemble projection differs"),
    };
    let state = serde_json::to_value(fitted)?;
    let fields = ["mean_only", "y_mean", "width", "keep", "clip_lo", "clip_hi"];
    let types = ["bool", "f64", "usize", "Vec<usize>", "f64", "f64"];
    let mut fields = fields
        .iter()
        .zip(types)
        .map(|(name, ty)| Ok(format!("{name}: {}", literal(source, ty, &state[name])?)))
        .collect::<Result<Vec<_>>>()?;
    fields.push(format!("proj: {proj}"));
    fields.push(format!("entries: vec![{}]", entries.join(", ")));
    Ok(format!(
        "InternallyValidatedEnsemble {{ {} }}",
        fields.join(", ")
    ))
}

/// Render the small field-type grammar declared by emit_rs, never a Debug dump of fitted state.
/// This binds constructors to the emitter's actual fields and refuses unrecognized template drift.
pub(super) fn literal(source: &str, ty: &str, value: &Value) -> Result<String> {
    if let Some(inner) = ty.strip_prefix("Vec<").and_then(|s| s.strip_suffix('>')) {
        let items = value
            .as_array()
            .with_context(|| format!("fitted Rust expected {ty}"))?;
        let values = items
            .iter()
            .map(|v| literal(source, inner, v))
            .collect::<Result<Vec<_>>>()?;
        return Ok(format!("vec![{}]", values.join(", ")));
    }
    if let Some(inner) = ty.strip_prefix("Option<").and_then(|s| s.strip_suffix('>')) {
        return if value.is_null() {
            Ok("None".into())
        } else {
            Ok(format!("Some({})", literal(source, inner, value)?))
        };
    }
    if ty == "(usize, usize)" {
        let pair = value
            .as_array()
            .filter(|v| v.len() == 2)
            .context("fitted Rust expected index pair")?;
        return Ok(format!(
            "({}, {})",
            literal(source, "usize", &pair[0])?,
            literal(source, "usize", &pair[1])?
        ));
    }
    match ty {
        "f64" => {
            return value
                .as_f64()
                .filter(|v| v.is_finite())
                .map(|v| format!("{v:?}"))
                .context("non-finite fitted Rust coefficient")
        }
        "usize" | "u64" => {
            return value
                .as_u64()
                .map(|v| v.to_string())
                .context("invalid fitted Rust unsigned integer")
        }
        "i64" => {
            return value
                .as_i64()
                .map(|v| v.to_string())
                .context("invalid fitted Rust signed integer")
        }
        "bool" => {
            return value
                .as_bool()
                .map(|v| v.to_string())
                .context("invalid fitted Rust boolean")
        }
        _ => {}
    }
    let mut value = value.clone();
    if matches!(
        ty,
        "RidgeRegression"
            | "GcvRidgeRegression"
            | "ElasticNet"
            | "KernelRidgeRegression"
            | "GaussianProcessArdRegression"
            | "GaussianProcessCompositeKernelRegression"
            | "NadarayaWatsonRegression"
    ) {
        let means = value["means"].as_array().context("missing fitted means")?;
        ensure!(
            value["scales"]
                .as_array()
                .is_some_and(|v| v.len() == means.len()),
            "fitted Rust scale width differs from means"
        );
        if matches!(ty, "RidgeRegression" | "GcvRidgeRegression" | "ElasticNet") {
            let bias = usize::from(ty != "ElasticNet");
            ensure!(
                value["w"]
                    .as_array()
                    .is_some_and(|v| v.len() == means.len() + bias),
                "fitted Rust coefficient width differs from means"
            );
        }
    }
    if matches!(ty, "TreeNode" | "BoostNode") {
        value = if let Some(leaf) = value.get("Leaf") {
            json!({"feature":-1,"threshold":0,"left":0,"right":0,"value":leaf})
        } else if let Some(split) = value.get("Split") {
            json!({"feature":split["feature"],"threshold":split.get("thr_bin").unwrap_or(&split["threshold"]),"left":split["left"],"right":split["right"],"value":0.0})
        } else {
            bail!("invalid fitted Rust split node");
        };
    }
    if ty == "GradientBoostedTrees" {
        value["fallback"] = value["constant_value"].clone();
    }
    let marker = format!("struct {ty} {{\n");
    let body = source
        .split_once(&marker)
        .with_context(|| format!("recipe has no fitted type {ty}"))?
        .1
        .split_once("\n}")
        .context("incomplete fitted type")?
        .0;
    let fields = body
        .lines()
        .map(|line| -> Result<String> {
            let (name, field_type) = line
                .trim()
                .strip_suffix(',')
                .and_then(|s| s.split_once(": "))
                .context("unsupported fitted Rust field declaration")?;
            let value = value
                .get(name)
                .with_context(|| format!("missing fitted Rust field {ty}.{name}"))?;
            Ok(format!("{name}: {}", literal(source, field_type, value)?))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(format!("{ty} {{ {} }}", fields.join(", ")))
}

const INPUT_ADAPTER: &str = r##"
fn encode_cell(cell: Option<&str>, column: usize) -> Result<f64, String> {
    let (min, max, constant) = INPUT_SCALERS[column];
    let value = match cell {
        None => return Ok(0.0),
        Some(cell) => {
            let text = cell.trim();
            match text.to_ascii_lowercase().as_str() {
                "" | "na" | "n/a" | "null" | "none" | "nan" | "?" | "#n/a" => return Ok(0.0),
                "true" | "yes" => 1.0,
                "false" | "no" => 0.0,
                _ => match text.parse::<f64>() {
                    Ok(v) if v.is_finite() => v,
                    _ => return Ok(0.0),
                },
            }
        }
    };
    if constant { return Ok(0.0); }
    let value = ((value - min) / (max - min)).clamp(0.0, 1.0);
    if !value.is_finite() { return Err("prediction failed: non-finite normalized input".into()); }
    Ok(value)
}

pub fn predict_cells(rows: &[Vec<Option<&str>>]) -> Result<Vec<f64>, String> {
    let encoded = rows.iter().map(|row| {
        if row.len() != MODEL_COLUMNS.len() { return Err("invalid raw row width".to_owned()); }
        INPUT_SELECTED.iter().map(|&j| encode_cell(row[j], j)).collect()
    }).collect::<Result<Vec<Vec<f64>>, String>>()?;
    predict_numeric(&encoded)?.into_iter().map(|v| {
        let value = if TARGET_SCALE <= 1e-12 { TARGET_MIN } else { TARGET_MIN + v * TARGET_SCALE };
        if !value.is_finite() { return Err("prediction failed: non-finite target prediction".into()); }
        Ok(value)
    }).collect()
}

pub fn predict_table(headers: &[&str], rows: &[Vec<Option<&str>>]) -> Result<Vec<f64>, String> {
    let at = MODEL_COLUMNS.iter().map(|name| headers.iter().position(|h| h == name)
        .ok_or_else(|| format!("input lacks column {:?}", name))).collect::<Result<Vec<_>, _>>()?;
    let ordered = rows.iter().map(|row| {
        if row.len() != headers.len() { return Err("invalid table row width".to_owned()); }
        Ok(at.iter().map(|&i| row[i]).collect())
    }).collect::<Result<Vec<_>, String>>()?;
    predict_cells(&ordered)
}
"##;
