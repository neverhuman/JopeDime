use super::*;
pub(super) fn selected_domain(
    parameter: &ParamDescriptor,
    mode: CompileMode,
) -> Option<V3ParamDomain> {
    match mode {
        CompileMode::PrototypeCompatibility => parameter.prototype_domain,
        CompileMode::Production => parameter.production_domain,
    }
}

pub(super) fn static_value(value: StaticValue, task: &TaskSpec, operation: &str) -> ParamValue {
    match value {
        StaticValue::Integer(value) => ParamValue::Int(value),
        StaticValue::Float(value) => ParamValue::Float(value),
        StaticValue::Boolean(value) => ParamValue::Bool(value),
        StaticValue::String(value) => ParamValue::Str(value.to_string()),
        StaticValue::TaskLoss => ParamValue::Str(
            match (operation, task.kind) {
                ("tab_boost", TaskKind::Binary) => "log_loss",
                (
                    "tab_boost",
                    TaskKind::Regression | TaskKind::Timeseries | TaskKind::Multiclass,
                ) => "squared",
                _ => unreachable!("task-dependent loss is only used by tab_boost"),
            }
            .to_string(),
        ),
    }
}

pub(super) fn normalize_value(
    operation: &str,
    parameter: &ParamDescriptor,
    value: &ParamValue,
    domain: V3ParamDomain,
) -> Result<ParamValue> {
    let normalized = match parameter.kind {
        ParamKind::Integer => match value {
            ParamValue::Int(value) => ParamValue::Int(*value),
            ParamValue::Float(value) if value.is_finite() && value.fract() != 0.0 => {
                bail!(
                    "{operation}: parameter '{}' requires an integer; nonintegral float {value} is invalid",
                    parameter.name
                )
            }
            _ => bail!(
                "{operation}: parameter '{}' must have integer type",
                parameter.name
            ),
        },
        ParamKind::FiniteFloat => match value {
            ParamValue::Int(value) => ParamValue::Float(*value as f64),
            ParamValue::Float(value) if value.is_finite() => ParamValue::Float(*value),
            ParamValue::Float(_) => {
                bail!("{operation}: parameter '{}' must be finite", parameter.name)
            }
            _ => bail!(
                "{operation}: parameter '{}' must have numeric type",
                parameter.name
            ),
        },
        ParamKind::Boolean => match value {
            ParamValue::Bool(value) => ParamValue::Bool(*value),
            _ => bail!(
                "{operation}: parameter '{}' must have boolean type",
                parameter.name
            ),
        },
        ParamKind::String => match value {
            ParamValue::Str(value) => ParamValue::Str(value.clone()),
            _ => bail!(
                "{operation}: parameter '{}' must have string type",
                parameter.name
            ),
        },
        ParamKind::MaxFeatures => match value {
            ParamValue::Int(value) => ParamValue::Int(*value),
            ParamValue::Float(value) if value.is_finite() => ParamValue::Float(*value),
            ParamValue::Float(_) => {
                bail!("{operation}: parameter '{}' must be finite", parameter.name)
            }
            ParamValue::Str(value) => ParamValue::Str(value.clone()),
            ParamValue::Bool(_) => bail!(
                "{operation}: parameter '{}' must be an integer, fraction, sqrt, or log2",
                parameter.name
            ),
        },
        ParamKind::OofMode => match value {
            ParamValue::Int(value) => ParamValue::Int(*value),
            _ => bail!(
                "{operation}: parameter '{}' must be integer 5",
                parameter.name
            ),
        },
    };
    if !domain_accepts(domain, &normalized) {
        bail!(
            "{operation}: parameter '{}' value {} is outside the {:?} domain",
            parameter.name,
            normalized,
            domain
        );
    }
    Ok(normalized)
}

pub(super) fn domain_accepts(domain: V3ParamDomain, value: &ParamValue) -> bool {
    match (domain, value) {
        (V3ParamDomain::IntegerRange { min, max }, ParamValue::Int(value)) => {
            *value >= min && *value <= max
        }
        (V3ParamDomain::IntegerSet(values), ParamValue::Int(value)) => values.contains(value),
        (V3ParamDomain::FloatRange { min, max }, ParamValue::Float(value)) => {
            value.is_finite() && *value >= min && *value <= max
        }
        (V3ParamDomain::FloatSet(values), ParamValue::Float(value)) => {
            value.is_finite() && values.contains(value)
        }
        (V3ParamDomain::StringSet(values), ParamValue::Str(value)) => {
            values.contains(&value.as_str())
        }
        (V3ParamDomain::Boolean, ParamValue::Bool(_)) => true,
        (V3ParamDomain::MaxFeaturesPrototype, ParamValue::Int(value)) => *value >= 1,
        (V3ParamDomain::MaxFeaturesPrototype, ParamValue::Float(value)) => {
            value.is_finite() && *value > 0.0 && *value <= 1.0
        }
        (V3ParamDomain::MaxFeaturesPrototype, ParamValue::Str(value)) => {
            matches!(value.as_str(), "sqrt" | "log2")
        }
        (V3ParamDomain::MaxFeaturesProduction, ParamValue::Float(value)) => {
            [0.25, 0.5, 1.0].contains(value)
        }
        (V3ParamDomain::MaxFeaturesProduction, ParamValue::Str(value)) => {
            matches!(value.as_str(), "sqrt" | "log2")
        }
        (V3ParamDomain::OofFive, ParamValue::Int(value)) => *value == 5,
        _ => false,
    }
}

pub(super) fn is_active(
    condition: ActiveWhen,
    values: &BTreeMap<String, ParamValue>,
    task: TaskKind,
) -> bool {
    match condition {
        ActiveWhen::Always => true,
        ActiveWhen::Task(required) => task == required,
        ActiveWhen::StringEquals { parameter, value } => matches!(
            values.get(parameter),
            Some(ParamValue::Str(actual)) if actual == value
        ),
    }
}

pub(super) fn normalize_value_aliases(stage: &mut Stage) {
    if stage.params.get("threshold") == Some(&ParamValue::Str("fixed".to_string())) {
        stage
            .params
            .insert("threshold".to_string(), ParamValue::Str("0.5".to_string()));
    }
}

pub(super) fn validate_task_specific_values(stage: &Stage, task: TaskKind) -> Result<()> {
    if stage.op == "tab_boost" {
        let loss = string_value(stage, "loss")?;
        let accepted = match task {
            TaskKind::Regression | TaskKind::Timeseries => TAB_REGRESSION_LOSSES,
            TaskKind::Binary => TAB_BINARY_LOSSES,
            TaskKind::Multiclass => &[],
        };
        if !accepted.contains(&loss) {
            bail!("tab_boost: loss '{loss}' is incompatible with {task:?}");
        }
    }
    Ok(())
}

pub(super) fn check_operation(
    descriptor: &OperationDescriptor,
    task: &TaskSpec,
    schema: &DataSchema,
    options: &CompileV3Options<'_>,
    shape: &FeatureShape,
) -> Result<()> {
    if !descriptor.tasks.contains(task.kind) {
        bail!("{} is incompatible with {:?}", descriptor.name, task.kind);
    }
    if descriptor.row_requirement == RowRequirement::Ordered
        && !matches!(schema.row_semantics, RowSemantics::Ordered { .. })
    {
        bail!(
            "{} requires explicit ordered row semantics",
            descriptor.name
        );
    }
    let role_policy = match options.mode {
        CompileMode::PrototypeCompatibility => descriptor.prototype_roles,
        CompileMode::Production => descriptor.production_roles,
    };
    for feature in &shape.features {
        if !role_allowed(role_policy, feature.role) {
            bail!(
                "{} cannot consume feature role {:?} in {:?} mode",
                descriptor.name,
                feature.role,
                options.mode
            );
        }
    }
    let support = descriptor.backend_support.selected(options.mode);
    for &target in options.requested_emitters {
        if !support.supports(target) {
            bail!(
                "{} has no {:?} support in {:?} mode",
                descriptor.name,
                target,
                options.mode
            );
        }
    }
    if options.mode == CompileMode::Production {
        match descriptor.admission {
            ProductionAdmission::Baseline => {}
            ProductionAdmission::RequiresReceipt => {
                if !options.admitted_families.contains(&descriptor.family) {
                    bail!(
                        "{} family '{}' has no production admission receipt",
                        descriptor.name,
                        descriptor.family
                    );
                }
            }
            ProductionAdmission::Disabled => {
                bail!("{} is disabled in the production grammar", descriptor.name)
            }
        }
    }
    Ok(())
}

pub(super) fn role_allowed(policy: RolePolicy, role: FeatureRole) -> bool {
    match policy {
        RolePolicy::None | RolePolicy::Any | RolePolicy::NumericOrNative => true,
        RolePolicy::NumericLike => role != FeatureRole::CategoricalNative,
    }
}
