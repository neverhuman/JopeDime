//! Explicitly bounded fitted inference lowering. It carries only inference state.
use super::calibration::CalibrationModel;
use super::heads::V3Head;
use super::input::{CategoricalMode, FittedColumn, FittedRowOrder};
use super::pipeline::{BoundModel, BoundPredictor};
use super::schema::{ColumnKind, RowSemantics};
use super::types::TaskKind;
use crate::heads::FittedHead;
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
mod scalar;

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

pub fn fitted_portable_templates() -> BTreeMap<String, String> {
    [
        ("matrix.py", include_str!("fitted_portable.py")),
        ("matrix.js", include_str!("fitted_portable.js")),
        ("matrix.rs", include_str!("fitted_portable.rs.txt")),
    ]
    .into_iter()
    .map(|(name, text)| (name.to_owned(), super::types::text_hash(text)))
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
    let mut hits = super::emit::lint_shipped_masked(&masked, &[], false);
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
) -> Result<super::fitted_cpp::FittedCppArtifact> {
    ensure!(
        fitted_public_cpp_unsupported(model)?.is_none(),
        "outside public C++ scalar subset"
    );
    // Validate all dimensions, indices and coefficients before invoking the native emitter.
    let _ = emit_fitted_portable(model)?;
    let original = super::fitted_cpp::emit_fitted_cpp17(model)?;
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
    Ok(super::fitted_cpp::FittedCppArtifact {
        source_sha256: super::types::text_hash(&source),
        codec_sha256: super::types::text_hash(&codec),
        source,
        codec,
    })
}

#[derive(Clone, Debug)]
pub struct FittedPortableSources {
    pub python: String,
    pub rust: String,
    pub javascript: Option<String>,
    /// Matrix columns for multiclass, or [0, 1] for binary; empty for regression.
    pub class_codes: Vec<i64>,
}

#[derive(Serialize)]
enum Input {
    Numeric {
        fill: f64,
        indicator: bool,
        restore_missing: bool,
    },
    Onehot {
        vocabulary: Vec<String>,
        indices: Vec<usize>,
    },
}

#[derive(Serialize)]
struct Linear {
    means: Vec<f64>,
    scales: Vec<f64>,
    weights: Vec<f64>,
    bound: f64,
    finite_input: bool,
}

#[derive(Serialize)]
enum Head {
    Scalar {
        binary: bool,
        state: scalar::Scalar,
    },
    Regression(Linear),
    Binary(Linear),
    Softmax(Vec<Linear>),
    Ovr(Vec<Linear>),
    Lda {
        centers: Vec<Vec<f64>>,
        scales: Vec<f64>,
        priors: Vec<f64>,
    },
    Centroid {
        centers: Vec<Vec<f64>>,
        scales: Vec<f64>,
    },
}

#[derive(Serialize)]
struct State {
    input: Vec<Input>,
    head: Head,
}

fn linear(
    means: &[f64],
    scales: &[f64],
    weights: &[f64],
    bound: f64,
    finite_input: bool,
) -> Linear {
    Linear {
        means: means.to_vec(),
        scales: scales.to_vec(),
        weights: weights.to_vec(),
        bound,
        finite_input,
    }
}

/// A narrow, fail-closed emitter for a fitted IID input followed directly by a linear
/// regression, logistic, bounded scalar, or one of the four multiclass heads. No operations, calibration,
/// target mapping, historical state, or missing-aware routing are silently dropped.
/// The two admitted branch-based heads preserve native numeric missing masks.
/// Each function takes the existing encoded RawTable rows and returns a matrix: one
/// value per regression row, [1-p,p] for binary, and all fitted multiclass columns.
/// Python/Rust use their platform libm. JavaScript uses ECMAScript math; callers must
/// qualify it with an explicit numerical tolerance, never infer bit parity.
pub fn emit_fitted_portable(model: &BoundModel) -> Result<FittedPortableSources> {
    ensure!(
        fitted_portable_unsupported(model)?.is_none(),
        "outside portable fitted subset"
    );
    ensure!(
        matches!(model.schema.row_semantics, RowSemantics::Iid),
        "portable fitted input requires IID rows"
    );
    ensure!(
        matches!(model.input.row_order, FittedRowOrder::Iid)
            && model.input.historical.iter().all(Option::is_none),
        "portable fitted input does not support historical state"
    );
    let BoundPredictor::Pipeline {
        target,
        ops,
        head,
        calibration,
        missing_aware,
    } = &model.predictor
    else {
        bail!("portable fitted input requires a schema-bound pipeline");
    };
    ensure!(
        target.is_none() && ops.is_empty() && matches!(calibration, CalibrationModel::None),
        "portable fitted input does not support operations, target mapping, or calibration"
    );
    let input = model
        .input
        .columns
        .iter()
        .map(|column| match column {
            FittedColumn::Numeric { fill, indicator } => {
                ensure!(fill.is_finite(), "nonfinite fitted input fill");
                Ok(Input::Numeric {
                    fill: *fill,
                    indicator: *indicator,
                    restore_missing: *missing_aware,
                })
            }
            FittedColumn::Categorical {
                mode: CategoricalMode::Onehot,
                vocabulary,
                onehot,
                ..
            } => {
                ensure!(
                    onehot.iter().all(|index| *index < vocabulary.len()),
                    "invalid fitted onehot index"
                );
                Ok(Input::Onehot {
                    vocabulary: onehot
                        .iter()
                        .map(|index| vocabulary[*index].clone())
                        .collect(),
                    indices: (0..onehot.len()).collect(),
                })
            }
            FittedColumn::Categorical { .. } => {
                bail!("portable fitted input supports only onehot categories")
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let (head, class_codes) = match head {
        V3Head::Legacy(FittedHead::Ridge(value)) if model.task.kind == TaskKind::Regression => (
            Head::Regression(linear(&value.means, &value.scales, &value.w, 20.0, false)),
            vec![],
        ),
        V3Head::LogitRidge(value) if model.task.kind == TaskKind::Binary => (
            Head::Binary(linear(
                &value.means,
                &value.scales,
                &value.weights,
                30.0,
                false,
            )),
            vec![0, 1],
        ),
        V3Head::MulticlassRidge(value) if model.task.kind == TaskKind::Multiclass => {
            ensure!(
                value.feature_count == value.means.len(),
                "multiclass fitted width differs"
            );
            (
                Head::Softmax(
                    value
                        .weights
                        .iter()
                        .map(|weights| linear(&value.means, &value.scales, weights, 30.0, true))
                        .collect(),
                ),
                value.classes.clone(),
            )
        }
        V3Head::MulticlassOvrLogistic(value) if model.task.kind == TaskKind::Multiclass => (
            Head::Ovr(
                value
                    .models
                    .iter()
                    .map(|value| linear(&value.means, &value.scales, &value.weights, 30.0, false))
                    .collect(),
            ),
            value.classes.clone(),
        ),
        V3Head::MulticlassDiagonalLda(value) if model.task.kind == TaskKind::Multiclass => {
            ensure!(
                value.feature_count == value.variances.len(),
                "multiclass fitted width differs"
            );
            (
                Head::Lda {
                    centers: value.means.clone(),
                    scales: value.variances.clone(),
                    priors: value.log_priors.clone(),
                },
                value.classes.clone(),
            )
        }
        V3Head::MulticlassNearestCentroid(value) if model.task.kind == TaskKind::Multiclass => {
            ensure!(
                value.feature_count == value.scales.len(),
                "multiclass fitted width differs"
            );
            (
                Head::Centroid {
                    centers: value.centroids.clone(),
                    scales: value.scales.clone(),
                },
                value.classes.clone(),
            )
        }
        head if scalar::supports(head, model.task.kind) => (
            Head::Scalar {
                binary: model.task.kind == TaskKind::Binary,
                state: scalar::lower(head, model.task.kind)?,
            },
            if model.task.kind == TaskKind::Binary {
                vec![0, 1]
            } else {
                vec![]
            },
        ),
        _ => bail!("this fitted head/task is outside the portable fitted subset"),
    };
    let state = State { input, head };
    validate_state(&state)?;
    let serialized = serde_json::to_string(&state)?;
    let python = render(
        include_str!("fitted_portable.py"),
        &[("__STATE_JSON__", serde_json::to_string(&serialized)?)],
    );
    let javascript = if fitted_javascript_unsupported(model)?.is_none() {
        Some(render(
            include_str!("fitted_portable.js"),
            &[("__STATE_JSON__", serialized)],
        ))
    } else {
        None
    };
    let rust = rust_source(&state);
    for source in [&python, &rust].into_iter().chain(javascript.iter()) {
        let hits = lint_fitted_public_text(model, &[], source);
        ensure!(
            hits.is_empty(),
            "fitted source contains banned vocabulary: {}",
            hits.join(", ")
        );
    }
    Ok(FittedPortableSources {
        python,
        rust,
        javascript,
        class_codes,
    })
}

fn validate_state(state: &State) -> Result<()> {
    let width: usize = state
        .input
        .iter()
        .map(|input| match input {
            Input::Numeric { indicator, .. } => 1 + usize::from(*indicator),
            Input::Onehot { indices, .. } => indices.len(),
        })
        .sum();
    let check = |value: &Linear| -> Result<()> {
        ensure!(
            !value.means.is_empty()
                && value.means.len() <= width
                && value.scales.len() == value.means.len()
                && value.weights.len() == value.means.len() + 1,
            "fitted linear dimensions differ"
        );
        ensure!(
            value
                .means
                .iter()
                .chain(&value.weights)
                .all(|v| v.is_finite())
                && value.scales.iter().all(|v| v.is_finite() && *v > 0.0),
            "invalid fitted linear coefficients"
        );
        Ok(())
    };
    match &state.head {
        Head::Scalar { state, .. } => scalar::validate(state, width),
        Head::Regression(value) | Head::Binary(value) => check(value),
        Head::Softmax(values) | Head::Ovr(values) => {
            ensure!(
                values.len() >= 3,
                "multiclass fitted state has fewer than three classes"
            );
            values.iter().try_for_each(check)
        }
        Head::Lda {
            centers, scales, ..
        }
        | Head::Centroid { centers, scales } => {
            // The separate prior check below avoids treating absent priors as zeros.
            ensure!(
                centers.len() >= 3 && !scales.is_empty() && scales.len() <= width,
                "multiclass fitted dimensions differ"
            );
            ensure!(
                scales.iter().all(|v| v.is_finite() && *v > 0.0)
                    && centers
                        .iter()
                        .all(|row| row.len() == scales.len() && row.iter().all(|v| v.is_finite())),
                "invalid fitted distance coefficients"
            );
            if let Head::Lda { priors, .. } = &state.head {
                ensure!(
                    priors.len() == centers.len() && priors.iter().all(|v| v.is_finite()),
                    "invalid fitted priors"
                );
            }
            Ok(())
        }
    }
}

// Substitute only the template itself; user strings resembling markers remain literal.
fn render(mut template: &str, values: &[(&str, String)]) -> String {
    for (key, _) in values {
        assert!(template.contains(key), "missing template marker {key}");
    }
    let mut remaining = template.to_owned();
    for (key, _) in values {
        remaining = remaining.replace(key, "");
    }
    for fragment in remaining.split("__").skip(1).step_by(2) {
        assert!(
            !fragment
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b == b'_'),
            "unknown template marker"
        );
    }
    let mut output = String::new();
    while let Some((offset, key, value)) = values
        .iter()
        .filter_map(|(key, value)| template.find(key).map(|offset| (offset, *key, value)))
        .min_by_key(|item| item.0)
    {
        output.push_str(&template[..offset]);
        output.push_str(value);
        template = &template[offset + key.len()..];
    }
    output.push_str(template);
    output
}

fn floats(values: &[f64]) -> String {
    format!(
        "&[{}]",
        values
            .iter()
            .map(|value| format!("f64::from_bits(0x{:016x})", value.to_bits()))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn linear_call(value: &Linear) -> String {
    format!(
        "linear(x,{},{},{},{:.17e},{})",
        floats(&value.means),
        floats(&value.scales),
        floats(&value.weights),
        value.bound,
        value.finite_input
    )
}
fn rust_source(state: &State) -> String {
    let mut encode = String::from("let mut x = Vec::new();\n");
    for (index, input) in state.input.iter().enumerate() {
        match input {
            Input::Numeric {
                fill,
                indicator,
                restore_missing,
            } => {
                encode.push_str(&format!("let raw=&row[{index}]; let absent=missing(raw); let parsed=if absent {{ f64::NAN }} else {{ raw.trim().parse::<f64>().map_err(|_| \"invalid numeric input\".to_string())? }}; let absent=absent || !parsed.is_finite(); x.push(if absent && {restore_missing} {{ f64::NAN }} else if absent {{ f64::from_bits(0x{:016x}) }} else {{ parsed }});\n", fill.to_bits()));
                if *indicator {
                    encode.push_str("x.push(if absent { 1.0 } else { 0.0 });\n");
                }
            }
            Input::Onehot {
                vocabulary,
                indices,
            } => {
                let values = vocabulary
                    .iter()
                    .map(|value| format!("{value:?}"))
                    .collect::<Vec<_>>()
                    .join(",");
                encode.push_str(&format!("let key=category(&row[{index}]); let vocabulary: &[&str]=&[{values}]; let found=vocabulary.iter().position(|value| *value == key);\n"));
                for position in indices {
                    encode.push_str(&format!(
                        "x.push(if found==Some({position}) {{ 1.0 }} else {{ 0.0 }});\n"
                    ));
                }
            }
        }
    }
    encode.push_str("Ok(x)");
    let predict = match &state.head {
        Head::Scalar { binary, state } => {
            let value = scalar::rust(state, "x");
            if *binary {
                format!("{{let p={value};vec![1.0-p,p]}} ")
            } else {
                format!("vec![finite({value})]")
            }
        }
        Head::Regression(value) => {
            format!("vec![finite({}.clamp(-60.0,60.0))]", linear_call(value))
        }
        Head::Binary(value) => {
            format!("{{ let p=sigmoid({}); vec![1.0-p,p] }}", linear_call(value))
        }
        Head::Softmax(values) => format!(
            "softmax(vec![{}])",
            values
                .iter()
                .map(|value| format!("finite({})", linear_call(value)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Head::Ovr(values) => format!(
            "normalize(vec![{}])",
            values
                .iter()
                .map(|value| format!("sigmoid({})", linear_call(value)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Head::Lda {
            centers,
            scales,
            priors,
        } => format!(
            "softmax(vec![{}])",
            centers
                .iter()
                .zip(priors)
                .map(|(center, prior)| format!(
                    "f64::from_bits(0x{:016x})-0.5*distance(x,{},{},true)",
                    prior.to_bits(),
                    floats(center),
                    floats(scales)
                ))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Head::Centroid { centers, scales } => format!(
            "softmax(vec![{}])",
            centers
                .iter()
                .map(|center| format!("-distance(x,{},{},false)", floats(center), floats(scales)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    };
    render(
        include_str!("fitted_portable.rs.txt"),
        &[
            ("__WIDTH__", state.input.len().to_string()),
            ("__ENCODE__", encode),
            ("__PREDICT__", predict),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::super::{
        fit_table, ColumnSpec, DataSchema, Objective, RawTable, RowSemantics, TaskSpec,
    };
    use super::*;
    use crate::GenomeIr;
    #[test]
    fn fitted_portable_refuses_unimplemented_effective_state() {
        let schema = DataSchema {
            columns: vec![ColumnSpec::numeric("amount")],
            row_semantics: RowSemantics::Iid,
        };
        let table = RawTable::new(
            vec!["amount".into()],
            (0..12).map(|i| vec![i.to_string()]).collect(),
        )
        .unwrap();
        let targets = (0..12).map(|i| i as f64).collect::<Vec<_>>();
        for source in [
            "(pipe X (input numeric=median categorical=onehot) (standardize) (ridge))",
            "(pipe X (input numeric=median categorical=onehot) (histgbm))",
        ] {
            let model = fit_table(
                &GenomeIr::parse(source).unwrap(),
                &TaskSpec::regression(Objective::Rmse).unwrap(),
                &schema,
                &table,
                &targets,
            )
            .unwrap();
            assert!(emit_fitted_portable(&model).is_err(), "{source}");
        }
        let binary = (0..12).map(|i| (i % 2) as f64).collect::<Vec<_>>();
        let mut model = fit_table(
            &GenomeIr::parse("(pipe X (input numeric=median categorical=onehot) (logit_ridge))")
                .unwrap(),
            &TaskSpec::binary(Objective::RocAuc).unwrap(),
            &schema,
            &table,
            &binary,
        )
        .unwrap();
        if let BoundPredictor::Pipeline { calibration, .. } = &mut model.predictor {
            *calibration = CalibrationModel::Platt {
                slope: 1.0,
                intercept: 0.0,
            };
        }
        assert!(emit_fitted_portable(&model).is_err());
    }
    #[test]
    fn public_onehot_state_excludes_unused_categories_and_label_statistics() {
        let schema = DataSchema {
            columns: vec![ColumnSpec::categorical("island")],
            row_semantics: RowSemantics::Iid,
        };
        let table = RawTable::new(
            vec!["island".into()],
            (0..12)
                .map(|i| {
                    vec![if i == 0 {
                        "PRIVATE_CATEGORY_CANARY".into()
                    } else {
                        "tree".into()
                    }]
                })
                .collect(),
        )
        .unwrap();
        let model = fit_table(
            &GenomeIr::parse(
                "(pipe X (input numeric=median categorical=onehot max_onehot=1) (ridge))",
            )
            .unwrap(),
            &TaskSpec::regression(Objective::Rmse).unwrap(),
            &schema,
            &table,
            &(0..12).map(|i| i as f64 + 1234.56789).collect::<Vec<_>>(),
        )
        .unwrap();
        let portable = emit_fitted_portable(&model).unwrap();
        let cpp = emit_fitted_public_cpp17(&model).unwrap();
        for text in [
            &portable.python,
            &portable.rust,
            portable.javascript.as_ref().unwrap(),
            &cpp.source,
            &cpp.codec,
        ] {
            assert!(!text.contains("PRIVATE_CATEGORY_CANARY"));
            for key in [
                "\"sums\"",
                "\"counts\"",
                "\"global\"",
                "schema_sha256",
                "vocabulary_sha256",
            ] {
                assert!(!text.contains(key), "{key}");
            }
            assert!(lint_fitted_public_text(&model, &[], text).is_empty());
        }
    }
    #[test]
    fn branch_heads_preserve_missing_routes_in_python_and_rust() {
        use super::super::heads::{BoostStump, ExtraNode};
        use std::process::Command;
        let schema = DataSchema {
            columns: vec![ColumnSpec::numeric("amount")],
            row_semantics: RowSemantics::Iid,
        };
        let train = RawTable::new(
            vec!["amount".into()],
            (0..8).map(|i| vec![i.to_string()]).collect(),
        )
        .unwrap();
        let predict = RawTable::new(
            vec!["amount".into()],
            vec![vec![String::new()], vec!["0".into()]],
        )
        .unwrap();
        let scratch = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/fitted-inference-tests");
        std::fs::create_dir_all(&scratch).unwrap();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = scratch.join(format!("missing-routes-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        for name in ["tab_boost", "extra_trees"] {
            let mut model = fit_table(
                &GenomeIr::parse(&format!(
                    "(pipe X (input numeric=median missing=indicator categorical=onehot) ({name}))"
                ))
                .unwrap(),
                &TaskSpec::regression(Objective::Rmse).unwrap(),
                &schema,
                &train,
                &(0..8).map(|v| v as f64).collect::<Vec<_>>(),
            )
            .unwrap();
            let BoundPredictor::Pipeline {
                head,
                missing_aware,
                ..
            } = &mut model.predictor
            else {
                panic!("pipeline")
            };
            assert!(*missing_aware);
            // Explicit synthetic coefficients make missing routing observable even
            // though the finite median falls to the opposite side of the split.
            match head {
                V3Head::TabBoost(m) => {
                    m.base = 0.0;
                    m.learning_rate = 1.0;
                    m.stumps = vec![BoostStump {
                        feature: 0,
                        threshold: 100.0,
                        missing_left: false,
                        left: -2.0,
                        right: 2.0,
                    }];
                }
                V3Head::ExtraTrees(m) => {
                    m.trees = vec![ExtraNode::Split {
                        feature: 0,
                        threshold: 100.0,
                        missing_left: false,
                        left: Box::new(ExtraNode::Leaf(-2.0)),
                        right: Box::new(ExtraNode::Leaf(2.0)),
                    }];
                }
                _ => panic!("unexpected head"),
            }
            let expected = model.predict_table(&predict).unwrap();
            assert_eq!(expected, vec![2.0, -2.0]);
            assert_eq!(fitted_portable_unsupported(&model).unwrap(), None);
            assert_eq!(
                fitted_javascript_unsupported(&model).unwrap(),
                Some(PortableUnsupported::Head)
            );
            assert_eq!(
                fitted_public_cpp_unsupported(&model).unwrap(),
                Some(PortableUnsupported::Head)
            );
            let emitted = emit_fitted_portable(&model).unwrap();
            let folder = root.join(name);
            std::fs::create_dir(&folder).unwrap();
            std::fs::write(folder.join("model.py"), &emitted.python).unwrap();
            let run=Command::new("python3").arg("-c").arg("import model;print(','.join(str(r[0]) for r in model.predict_encoded([[''],['0']])))").current_dir(&folder).output().unwrap();
            assert!(
                run.status.success(),
                "{}",
                String::from_utf8_lossy(&run.stderr)
            );
            let actual = String::from_utf8(run.stdout)
                .unwrap()
                .trim()
                .split(',')
                .map(|s| s.parse::<f64>().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            std::fs::write(folder.join("model.rs"), &emitted.rust).unwrap();
            std::fs::write(folder.join("main.rs"),"mod model;fn main(){let v=model::predict_encoded(&[vec![String::new()],vec![\"0\".into()]]).unwrap();println!(\"{},{}\",v[0][0].to_bits(),v[1][0].to_bits());}").unwrap();
            let build = Command::new("rustc")
                .args(["--edition=2021", "-O", "main.rs", "-o", "predict"])
                .current_dir(&folder)
                .output()
                .unwrap();
            assert!(
                build.status.success(),
                "{}",
                String::from_utf8_lossy(&build.stderr)
            );
            let run = Command::new(folder.join("predict")).output().unwrap();
            assert!(run.status.success());
            let actual = String::from_utf8(run.stdout)
                .unwrap()
                .trim()
                .split(',')
                .map(|s| s.parse::<u64>().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                actual,
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            // The ordinary filled-value path remains faithful if this stored flag
            // is false; no category or missing-indicator mask is ever restored.
            if let BoundPredictor::Pipeline { missing_aware, .. } = &mut model.predictor {
                *missing_aware = false;
            }
            assert_eq!(model.predict_table(&predict).unwrap(), vec![-2.0, -2.0]);
        }
    }
    #[test]
    fn fitted_portable_template_bytes_are_pinned_for_version_three() {
        assert_eq!(FITTED_PORTABLE_EMITTER, "fitted-portable/3");
        let expected = [
            (
                "matrix.py".to_owned(),
                "a09b906608e6e25648f28d93b1a4f31b09cbc4b5a8cd395888734de23d657072".to_owned(),
            ),
            (
                "matrix.js".to_owned(),
                "519e5228f6611e7e351a96b9f7f4d4750038b7c86762c719467c2a2468eaa7c8".to_owned(),
            ),
            (
                "matrix.rs".to_owned(),
                "c7bce8cdc4f5ac3293814d4d8b2afe76a75ba3772926a0ceaa3593f993bdba54".to_owned(),
            ),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>();
        assert_eq!(fitted_portable_templates(), expected);
    }
}
