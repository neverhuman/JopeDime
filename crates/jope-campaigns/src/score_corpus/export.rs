use super::*;

pub(super) fn export_genome(genome: &GenomeIr) -> Result<GenomeExport> {
    let mut stages = Vec::new();
    let mut actions = Vec::new();
    match genome {
        GenomeIr::Pipeline(pipeline) => {
            actions.push(category_action("form", None, None, "pipeline"));
            export_pipeline(pipeline, "pipeline", &mut stages, &mut actions)?;
            Ok(GenomeExport {
                form: "pipeline".into(),
                stages,
                actions,
            })
        }
        GenomeIr::Symbolic {
            source,
            expr,
            affine,
        } => {
            actions.push(category_action("form", None, None, "symbolic"));
            actions.push(category_action(
                "parameter",
                Some("symbolic"),
                Some("mode"),
                if *affine { "affine" } else { "raw" },
            ));
            let ast = serde_json::to_value(expr)?;
            actions.push(category_action(
                "expression_begin",
                Some("symbolic"),
                Some("expr"),
                "begin",
            ));
            export_expression_actions(&ast, "symbolic", &mut actions)?;
            actions.push(category_action(
                "expression_end",
                Some("symbolic"),
                Some("expr"),
                "end",
            ));
            stages.push(TypedStage {
                path: "symbolic".into(),
                operation: "symbolic".into(),
                parameters: vec![
                    TypedParameter {
                        name: "mode".into(),
                        value_type: "categorical".into(),
                        value: json!(if *affine { "affine" } else { "raw" }),
                        expression_ast: None,
                    },
                    TypedParameter {
                        name: "expr".into(),
                        value_type: "expression".into(),
                        value: json!(source),
                        expression_ast: Some(ast),
                    },
                ],
            });
            Ok(GenomeExport {
                form: "symbolic".into(),
                stages,
                actions,
            })
        }
        GenomeIr::Portfolio(portfolio) => {
            actions.push(category_action("form", None, None, "portfolio"));
            for (member, pipeline) in portfolio.members.iter().enumerate() {
                actions.push(category_action(
                    "member_begin",
                    None,
                    None,
                    &member.to_string(),
                ));
                export_pipeline(
                    pipeline,
                    &format!("member[{member}]"),
                    &mut stages,
                    &mut actions,
                )?;
                actions.push(category_action(
                    "member_end",
                    None,
                    None,
                    &member.to_string(),
                ));
            }
            let blend = Stage {
                op: "blend".into(),
                params: BTreeMap::from([
                    ("folds".into(), ParamValue::Int(portfolio.folds as i64)),
                    ("guard".into(), ParamValue::Str("clip".into())),
                    ("kind".into(), ParamValue::Str("nnls".into())),
                    (
                        "repeats".into(),
                        if portfolio.repeats_auto {
                            ParamValue::Str("auto".into())
                        } else {
                            ParamValue::Int(1)
                        },
                    ),
                ]),
            };
            stages.push(export_stage(&blend, "blend", &mut actions)?);
            Ok(GenomeExport {
                form: "portfolio".into(),
                stages,
                actions,
            })
        }
    }
}

fn export_pipeline(
    pipeline: &DslExpr,
    prefix: &str,
    stages: &mut Vec<TypedStage>,
    actions: &mut Vec<GeneAction>,
) -> Result<()> {
    for (position, stage) in pipeline.stages.iter().enumerate() {
        stages.push(export_stage(
            stage,
            &format!("{prefix}.stage[{position}]"),
            actions,
        )?);
    }
    Ok(())
}

fn export_stage(stage: &Stage, path: &str, actions: &mut Vec<GeneAction>) -> Result<TypedStage> {
    actions.push(category_action("stage", Some(&stage.op), None, &stage.op));
    let mut parameters = Vec::new();
    for (name, value) in &stage.params {
        let (value_type, json_value, expression_ast) = match value {
            ParamValue::Int(value) => {
                actions.push(numeric_action(&stage.op, name, *value as f64)?);
                ("integer", json!(*value), None)
            }
            ParamValue::Float(value) => {
                actions.push(numeric_action(&stage.op, name, *value)?);
                ("float64", json!(*value), None)
            }
            ParamValue::Bool(value) => {
                actions.push(category_action(
                    "parameter",
                    Some(&stage.op),
                    Some(name),
                    if *value { "true" } else { "false" },
                ));
                ("boolean", json!(*value), None)
            }
            ParamValue::Str(value) if name == "expr" => {
                let expression = MathExpr::parse(value)
                    .with_context(|| format!("{}.{name} is not a valid expression", stage.op))?;
                let ast = serde_json::to_value(expression)?;
                actions.push(category_action(
                    "expression_begin",
                    Some(&stage.op),
                    Some(name),
                    "begin",
                ));
                export_expression_actions(&ast, &stage.op, actions)?;
                actions.push(category_action(
                    "expression_end",
                    Some(&stage.op),
                    Some(name),
                    "end",
                ));
                ("expression", json!(value), Some(ast))
            }
            ParamValue::Str(value) => {
                actions.push(category_action(
                    "parameter",
                    Some(&stage.op),
                    Some(name),
                    value,
                ));
                ("categorical", json!(value), None)
            }
        };
        parameters.push(TypedParameter {
            name: name.clone(),
            value_type: value_type.into(),
            value: json_value,
            expression_ast,
        });
    }
    Ok(TypedStage {
        path: path.into(),
        operation: stage.op.clone(),
        parameters,
    })
}

fn export_expression_actions(
    value: &Value,
    operation: &str,
    actions: &mut Vec<GeneAction>,
) -> Result<()> {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                actions.push(category_action(
                    "expression_node",
                    Some(operation),
                    Some("expr"),
                    key,
                ));
                export_expression_actions(child, operation, actions)?;
            }
        }
        Value::Array(values) => {
            actions.push(category_action(
                "expression_node",
                Some(operation),
                Some("expr"),
                "list_begin",
            ));
            for child in values {
                export_expression_actions(child, operation, actions)?;
            }
            actions.push(category_action(
                "expression_node",
                Some(operation),
                Some("expr"),
                "list_end",
            ));
        }
        Value::String(token) => actions.push(category_action(
            "expression_node",
            Some(operation),
            Some("expr"),
            token,
        )),
        Value::Number(number) => {
            let numeric = number
                .as_f64()
                .filter(|value| value.is_finite())
                .context("expression AST contains a non-finite number")?;
            actions.push(numeric_action(operation, "expr_constant", numeric)?);
        }
        Value::Bool(value) => actions.push(category_action(
            "expression_node",
            Some(operation),
            Some("expr"),
            if *value { "true" } else { "false" },
        )),
        Value::Null => actions.push(category_action(
            "expression_node",
            Some(operation),
            Some("expr"),
            "null",
        )),
    }
    Ok(())
}

fn category_action(
    kind: &str,
    operation: Option<&str>,
    parameter: Option<&str>,
    value: &str,
) -> GeneAction {
    GeneAction {
        kind: kind.into(),
        operation: operation.map(str::to_string),
        parameter: parameter.map(str::to_string),
        categorical_value: Some(value.into()),
        numeric_value: None,
    }
}

fn numeric_action(operation: &str, parameter: &str, value: f64) -> Result<GeneAction> {
    if !value.is_finite() {
        bail!("non-finite numeric parameter {operation}.{parameter}");
    }
    Ok(GeneAction {
        kind: "parameter".into(),
        operation: Some(operation.into()),
        parameter: Some(parameter.into()),
        categorical_value: None,
        numeric_value: Some(value),
    })
}

pub(super) fn observe_genome(genome: &GenomeExport, observed: &mut ObservedSchema) {
    for action in &genome.actions {
        let Some(operation) = action.operation.as_ref() else {
            continue;
        };
        observed.operations.insert(operation.clone());
        if let Some(parameter) = action.parameter.as_ref() {
            let key = (operation.clone(), parameter.clone());
            let value_type = if action.numeric_value.is_some() {
                "numeric"
            } else if action.kind.starts_with("expression") {
                "expression"
            } else {
                "categorical"
            };
            observed
                .parameters
                .entry(key.clone())
                .or_default()
                .insert(value_type.into());
            if let Some(value) = action.categorical_value.as_ref() {
                observed
                    .categorical_values
                    .entry(key)
                    .or_default()
                    .insert(value.clone());
            }
        }
    }
}
