use super::*;
pub(super) fn resolve_input_layout(
    input: &Stage,
    schema: &DataSchema,
    supplied: Option<&CompileInputLayout>,
) -> Result<CompileInputLayout> {
    let numeric_columns = schema
        .columns
        .iter()
        .filter(|column| column.kind == ColumnKind::Numeric)
        .count();
    let categorical_columns = schema.columns.len() - numeric_columns;
    let missing_indicator = string_value(input, "missing")? == "indicator";
    let categorical = string_value(input, "categorical")?;
    let max_onehot = integer_value(input, "max_onehot")? as usize;

    if let Some(layout) = supplied {
        layout.validate()?;
        let numeric = count_role(layout, FeatureRole::Numeric);
        let missing = count_role(layout, FeatureRole::MissingIndicator);
        if numeric != numeric_columns {
            bail!(
                "input layout has {numeric} numeric features; schema/input require {numeric_columns}"
            );
        }
        let expected_missing = if missing_indicator {
            numeric_columns
        } else {
            0
        };
        if missing != expected_missing {
            bail!(
                "input layout has {missing} missing indicators; schema/input require {expected_missing}"
            );
        }
        validate_categorical_layout(layout, categorical, categorical_columns, max_onehot)?;
        return Ok(layout.clone());
    }

    let mut features = Vec::new();
    for column in &schema.columns {
        match column.kind {
            ColumnKind::Numeric => {
                features.push(CompileFeature {
                    role: FeatureRole::Numeric,
                    origin: FeatureOrigin::RawDerived,
                    missing_mask: true,
                });
                if missing_indicator {
                    features.push(CompileFeature {
                        role: FeatureRole::MissingIndicator,
                        origin: FeatureOrigin::GeneratedSparse,
                        missing_mask: false,
                    });
                }
            }
            ColumnKind::Categorical => {
                let role = match categorical {
                    "native" => FeatureRole::CategoricalNative,
                    "frequency" => FeatureRole::CategoricalFrequency,
                    "ordered" => FeatureRole::CategoricalOrdered,
                    "onehot" | "hybrid" => bail!(
                        "input categorical={categorical} requires an exact CompileInputLayout"
                    ),
                    _ => unreachable!("categorical mode was registry-validated"),
                };
                features.push(CompileFeature {
                    role,
                    origin: FeatureOrigin::RawDerived,
                    missing_mask: true,
                });
            }
        }
    }
    CompileInputLayout::new(features)
}

pub(super) fn validate_categorical_layout(
    layout: &CompileInputLayout,
    mode: &str,
    columns: usize,
    max_onehot: usize,
) -> Result<()> {
    let native = count_role(layout, FeatureRole::CategoricalNative);
    let frequency = count_role(layout, FeatureRole::CategoricalFrequency);
    let onehot = count_role(layout, FeatureRole::CategoricalOneHot);
    let ordered = count_role(layout, FeatureRole::CategoricalOrdered);
    let invalid = match mode {
        "native" => native != columns || frequency + onehot + ordered != 0,
        "frequency" => frequency != columns || native + onehot + ordered != 0,
        "ordered" => ordered != columns || native + frequency + onehot != 0,
        "onehot" => {
            native + frequency + ordered != 0
                || onehot
                    > columns
                        .checked_mul(max_onehot)
                        .context("one-hot grammar cost overflow")?
        }
        "hybrid" => {
            native != 0
                || frequency != ordered
                || frequency > columns
                || onehot
                    > (columns - frequency)
                        .checked_mul(max_onehot)
                        .context("hybrid grammar cost overflow")?
        }
        _ => unreachable!("categorical mode was registry-validated"),
    };
    if invalid {
        bail!("input layout roles are incompatible with categorical={mode} for {columns} columns");
    }
    Ok(())
}

pub(super) fn count_role(layout: &CompileInputLayout, role: FeatureRole) -> usize {
    layout
        .features
        .iter()
        .filter(|feature| feature.role == role)
        .count()
}

pub(super) fn apply_width_effect(
    effect: WidthEffect,
    stage: &Stage,
    shape: &mut FeatureShape,
) -> Result<()> {
    let append = bool_value(stage, "append").unwrap_or(false);
    match effect {
        WidthEffect::Registry(_) => {
            bail!("registry width rule is not valid in advanced-v3 grammar")
        }
        WidthEffect::None | WidthEffect::Input => {}
        WidthEffect::OneToOne => {
            if append {
                shape.append_one_to_one()?;
            }
        }
        WidthEffect::GeneratedPerFeature => {
            shape.replace_or_append_generated(shape.width(), append)?;
        }
        WidthEffect::ExpansionPerFeature { parameter } => {
            let factor = integer_value(stage, parameter)? as usize;
            let width = shape
                .width()
                .checked_mul(factor)
                .context("generated dense feature width overflow")?;
            shape.replace_or_append_generated(width, append)?;
        }
        WidthEffect::Select { parameter } | WidthEffect::Projection { parameter } => {
            let requested = integer_value(stage, parameter)? as usize;
            shape.replace_or_append_generated(requested.min(shape.width()), append)?;
        }
        WidthEffect::Impute => {
            let indicators = if bool_value(stage, "add_indicator")? {
                shape
                    .features
                    .iter()
                    .filter(|feature| feature.missing_mask)
                    .count()
            } else {
                0
            };
            if append {
                shape.append_one_to_one()?;
            } else {
                for feature in &mut shape.features {
                    feature.missing_mask = false;
                }
            }
            shape
                .features
                .extend(FeatureShape::generated(indicators, true)?);
            shape.validate()?;
        }
    }
    Ok(())
}

pub(super) fn widths_of(features: &[CompileFeature]) -> FeatureWidths {
    let mut widths = FeatureWidths::default();
    for feature in features {
        match feature.origin {
            FeatureOrigin::RawDerived => widths.raw_derived += 1,
            FeatureOrigin::GeneratedDense => widths.generated_dense += 1,
            FeatureOrigin::GeneratedSparse => widths.generated_sparse += 1,
        }
    }
    widths
}

pub(super) fn check_generated_dense(width: usize) -> Result<()> {
    if width > MAX_GENERATED_DENSE_WIDTH {
        bail!("generated dense width {width} exceeds {MAX_GENERATED_DENSE_WIDTH}");
    }
    Ok(())
}

pub(super) fn operation_is_target_aware(descriptor: &OperationDescriptor, stage: &Stage) -> bool {
    if descriptor.name == "select_corr" {
        return string_value(stage, "by").is_ok_and(|value| value == "corr");
    }
    descriptor.target_aware
}

pub(super) fn input_is_target_aware(stage: &Stage) -> bool {
    string_value(stage, "categorical").is_ok_and(|value| matches!(value, "ordered" | "hybrid"))
}

pub(super) fn integer_value(stage: &Stage, parameter: &str) -> Result<i64> {
    match stage.params.get(parameter) {
        Some(ParamValue::Int(value)) => Ok(*value),
        _ => bail!(
            "{}: materialized integer '{parameter}' is missing",
            stage.op
        ),
    }
}

pub(super) fn bool_value(stage: &Stage, parameter: &str) -> Result<bool> {
    match stage.params.get(parameter) {
        Some(ParamValue::Bool(value)) => Ok(*value),
        _ => bail!(
            "{}: materialized boolean '{parameter}' is missing",
            stage.op
        ),
    }
}

pub(super) fn string_value<'a>(stage: &'a Stage, parameter: &str) -> Result<&'a str> {
    match stage.params.get(parameter) {
        Some(ParamValue::Str(value)) => Ok(value),
        _ => bail!("{}: materialized string '{parameter}' is missing", stage.op),
    }
}

pub(super) fn semantic_hash(canonical: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(V3_GRAMMAR_VERSION.as_bytes());
    hasher.update(canonical.as_bytes());
    format!("{:x}", hasher.finalize())
}
