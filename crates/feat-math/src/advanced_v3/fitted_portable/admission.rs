use super::*;
use std::collections::BTreeMap;

pub const FITTED_PORTABLE_EMITTER: &str = "fitted-portable/3";
pub const FITTED_PUBLIC_CPP_EMITTER: &str = "fitted-public-cpp/1:base-revision=2";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortableUnsupported {
    RowOrder,
    HistoricalState,
    ModelForm,
    Operations,
    TargetMapping,
    Calibration,
    MissingAware,
    CategoricalMode,
    Head,
}

/// Capability admission only. Invalid binding and coefficient errors remain failures.
pub fn fitted_portable_unsupported(model: &BoundModel) -> Result<Option<PortableUnsupported>> {
    model.validate_binding()?;
    ensure!(
        model.input.columns.len() == model.schema.columns.len()
            && model.input.columns.iter().zip(&model.schema.columns).all(
                |(fitted, column)| matches!(
                    (fitted, column.kind),
                    (FittedColumn::Numeric { .. }, ColumnKind::Numeric)
                        | (FittedColumn::Categorical { .. }, ColumnKind::Categorical)
                )
            ),
        "fitted input dimensions or column kinds differ"
    );
    let reason = if !matches!(model.schema.row_semantics, RowSemantics::Iid)
        || !matches!(model.input.row_order, FittedRowOrder::Iid)
    {
        Some(PortableUnsupported::RowOrder)
    } else if model.input.historical.iter().any(Option::is_some) {
        Some(PortableUnsupported::HistoricalState)
    } else if let BoundPredictor::Pipeline {
        target,
        ops,
        head,
        calibration,
        missing_aware,
    } = &model.predictor
    {
        if target.is_some() { Some(PortableUnsupported::TargetMapping) }
        else if !ops.is_empty() { Some(PortableUnsupported::Operations) }
        else if !matches!(calibration, CalibrationModel::None) { Some(PortableUnsupported::Calibration) }
        else if *missing_aware && !matches!(head,V3Head::TabBoost(_) | V3Head::ExtraTrees(_)) { Some(PortableUnsupported::MissingAware) }
        else if model.input.columns.iter().any(|column| matches!(column, FittedColumn::Categorical{mode,..} if !matches!(mode,CategoricalMode::Onehot))) { Some(PortableUnsupported::CategoricalMode) }
        else if !scalar::supports(head,model.task.kind) && !matches!((head,model.task.kind),
            (V3Head::Legacy(FittedHead::Ridge(_)),TaskKind::Regression)
            | (V3Head::LogitRidge(_),TaskKind::Binary)
            | (V3Head::MulticlassRidge(_),TaskKind::Multiclass)
            | (V3Head::MulticlassOvrLogistic(_),TaskKind::Multiclass)
            | (V3Head::MulticlassDiagonalLda(_),TaskKind::Multiclass)
            | (V3Head::MulticlassNearestCentroid(_),TaskKind::Multiclass)) { Some(PortableUnsupported::Head) }
        else { None }
    } else {
        Some(PortableUnsupported::ModelForm)
    };
    Ok(reason)
}

/// JavaScript remains at the original linear/multiclass coverage until separately qualified.
pub fn fitted_javascript_unsupported(model: &BoundModel) -> Result<Option<PortableUnsupported>> {
    if let Some(reason) = fitted_portable_unsupported(model)? {
        return Ok(Some(reason));
    }
    let BoundPredictor::Pipeline { head, .. } = &model.predictor else {
        unreachable!()
    };
    Ok(
        if matches!(
            head,
            V3Head::Legacy(FittedHead::Ridge(_))
                | V3Head::LogitRidge(_)
                | V3Head::MulticlassRidge(_)
                | V3Head::MulticlassOvrLogistic(_)
                | V3Head::MulticlassDiagonalLda(_)
                | V3Head::MulticlassNearestCentroid(_)
        ) {
            None
        } else {
            Some(PortableUnsupported::Head)
        },
    )
}
/// The public C++ host remains at the original ridge/logit subset.
pub fn fitted_public_cpp_unsupported(model: &BoundModel) -> Result<Option<PortableUnsupported>> {
    if let Some(reason) = fitted_portable_unsupported(model)? {
        return Ok(Some(reason));
    }
    let BoundPredictor::Pipeline { head, .. } = &model.predictor else {
        unreachable!()
    };
    Ok(
        if matches!(
            head,
            V3Head::Legacy(FittedHead::Ridge(_)) | V3Head::LogitRidge(_)
        ) {
            None
        } else {
            Some(PortableUnsupported::Head)
        },
    )
}

/// Versioned downstream artifact templates; Rust emits them but never executes them.
pub fn fitted_portable_templates() -> BTreeMap<String, String> {
    [
        (
            "matrix.py",
            include_str!("../../../contracts/fitted_portable_py.template"),
        ),
        (
            "matrix.js",
            include_str!("../../../contracts/fitted_portable.template"),
        ),
        ("matrix.rs", include_str!("../fitted_portable.rs.txt")),
    ]
    .into_iter()
    .map(|(name, text)| (name.to_owned(), super::super::types::text_hash(text)))
    .collect()
}

/// Lint assembled public files and metadata. Only actual required caller strings are masked.
pub fn lint_fitted_public_text(model: &BoundModel, classes: &[String], text: &str) -> Vec<String> {
    let mut raw = model
        .schema
        .columns
        .iter()
        .map(|c| c.name.clone())
        .collect::<Vec<_>>();
    raw.extend(classes.iter().cloned());
    for column in &model.input.columns {
        if let FittedColumn::Categorical {
            vocabulary, onehot, ..
        } = column
        {
            raw.extend(onehot.iter().filter_map(|i| vocabulary.get(*i)).cloned());
        }
    }
    let mut masks = Vec::new();
    for value in raw {
        let lower = value.to_lowercase();
        if value.len() < 3
            || (crate::transpile::lint_shipped_text(&value).is_empty()
                && !lower.contains("tree")
                && !lower.contains("search"))
        {
            continue;
        }
        let json = serde_json::to_string(&value).expect("string serialization");
        let inner = &json[1..json.len() - 1];
        let debug = format!("{value:?}");
        let nested = serde_json::to_string(inner).expect("string serialization");
        let debug_nested = format!("{inner:?}");
        masks.extend([
            value,
            inner.to_owned(),
            debug[1..debug.len() - 1].to_owned(),
            nested[1..nested.len() - 1].to_owned(),
            debug_nested[1..debug_nested.len() - 1].to_owned(),
        ]);
    }
    masks.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    masks.dedup();
    let mut masked = text.to_owned();
    for mask in masks {
        masked = masked.replace(&mask, &"_".repeat(mask.len()));
    }
    let mut hits = super::super::emit::lint_shipped_masked(&masked, &[], false);
    for word in ["tree", "search"] {
        if masked.to_lowercase().contains(word) {
            hits.push(word.into());
        }
    }
    hits
}

/// Public C++ is a versioned view of the existing emitter, restricted to the two
/// scalar heads qualified by the public Python-host replay. Original codec IDs and
/// nonselected category strings are excluded. Original private emitter bytes stay fixed.
pub fn emit_fitted_public_cpp17(
    model: &BoundModel,
) -> Result<super::super::fitted_cpp::FittedCppArtifact> {
    ensure!(
        fitted_public_cpp_unsupported(model)?.is_none(),
        "outside public C++ scalar subset"
    );
    // Validate all dimensions, indices and coefficients before invoking the native emitter.
    let _ = emit_fitted_portable(model)?;
    let original = super::super::fitted_cpp::emit_fitted_cpp17(model)?;
    let source = original
        .source
        .replace(
            "// Match Rust slice::partition_point's fixed-iteration binary-search path.",
            "// Match Rust slice::partition_point's fixed-iteration path.",
        )
        .replace(
            "// Fitted interpolated quantiles can cross by one ULP; preserving the search",
            "// Fitted interpolated quantiles can cross by one ULP; preserving the lookup",
        )
        .replace(
            "const std::vector<GNode>& tree,",
            "const std::vector<GNode>& nodes,",
        )
        .replace("const GNode& n=tree[at];", "const GNode& n=nodes[at];");
    let columns=model.input.columns.iter().map(|column|match column {
        FittedColumn::Numeric{..}=>serde_json::json!({"kind":"numeric"}),
        FittedColumn::Categorical{vocabulary,onehot,..}=>serde_json::json!({"kind":"categorical","entries":onehot.iter().map(|index|serde_json::json!({"value":vocabulary[*index],"index":index})).collect::<Vec<_>>()}),
    }).collect::<Vec<_>>();
    let codec = serde_json::to_string(
        &serde_json::json!({"schema":"jope.fitted-public-typed-table/v1","columns":columns,"categorical_unknown":-1,"missing_bitmap":"one byte per raw cell; nonzero means missing"}),
    )?;
    Ok(super::super::fitted_cpp::FittedCppArtifact {
        source_sha256: super::super::types::text_hash(&source),
        codec_sha256: super::super::types::text_hash(&codec),
        source,
        codec,
    })
}
