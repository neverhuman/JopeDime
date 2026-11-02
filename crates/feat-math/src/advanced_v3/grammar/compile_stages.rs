use super::*;
pub(crate) fn compile_v3_genome(
    genome: &GenomeIr,
    task: &TaskSpec,
    schema: &DataSchema,
    options: &CompileV3Options<'_>,
) -> Result<CompiledV3Genome> {
    task.validate()?;
    schema.validate()?;
    let GenomeIr::Pipeline(dsl) = genome else {
        bail!("schema-bound advanced-v3 programs must be pipelines");
    };
    let classified = classify_stages(&dsl.stages, task.kind)?;
    let mut canonical_stages = Vec::with_capacity(dsl.stages.len() + 2);
    let mut cursor = 0usize;

    if classified
        .first()
        .is_some_and(|(_, descriptor)| descriptor.stage_class == StageClass::Target)
    {
        let (stage, descriptor) = classified[cursor];
        canonical_stages.push(materialize_stage(stage, descriptor, task, options.mode)?);
        cursor += 1;
    }

    let input_descriptor = descriptor_for("input", task.kind)
        .expect("the registry contains an input descriptor for both tasks");
    let input = if classified
        .get(cursor)
        .is_some_and(|(_, descriptor)| descriptor.stage_class == StageClass::Input)
    {
        let (stage, descriptor) = classified[cursor];
        cursor += 1;
        materialize_stage(stage, descriptor, task, options.mode)?
    } else {
        materialize_stage(
            &compatibility_input_stage(),
            input_descriptor,
            task,
            options.mode,
        )?
    };
    canonical_stages.push(input.clone());

    let mut shape = FeatureShape::new(resolve_input_layout(&input, schema, options.input_layout)?);
    let mut requires_ordered_rows = false;
    let mut target_aware = input_is_target_aware(&input);
    let mut families = BTreeSet::new();
    families.insert(input_descriptor.family.to_string());
    check_operation(input_descriptor, task, schema, options, &shape)?;

    while let Some((stage, descriptor)) = classified.get(cursor).copied() {
        if descriptor.stage_class == StageClass::Learner {
            break;
        }
        let canonical = materialize_stage(stage, descriptor, task, options.mode)?;
        check_operation(descriptor, task, schema, options, &shape)?;
        apply_width_effect(descriptor.width_effect, &canonical, &mut shape)?;
        requires_ordered_rows |= descriptor.row_requirement == RowRequirement::Ordered;
        target_aware |= operation_is_target_aware(descriptor, &canonical);
        families.insert(descriptor.family.to_string());
        canonical_stages.push(canonical);
        cursor += 1;
    }

    let head = if let Some((stage, descriptor)) = classified.get(cursor).copied() {
        debug_assert_eq!(descriptor.stage_class, StageClass::Learner);
        cursor += 1;
        (
            materialize_stage(stage, descriptor, task, options.mode)?,
            descriptor,
        )
    } else {
        let name = match task.kind {
            TaskKind::Binary => "logit_ridge",
            TaskKind::Multiclass => "multiclass_ridge",
            TaskKind::Regression | TaskKind::Timeseries => "ridge",
        };
        let descriptor = descriptor_for(name, task.kind).expect("default head is registered");
        (
            materialize_stage(&Stage::new(name), descriptor, task, options.mode)?,
            descriptor,
        )
    };
    if cursor != classified.len() {
        bail!("model heads are only valid as the final stage");
    }
    check_operation(head.1, task, schema, options, &shape)?;
    target_aware |= head.1.target_aware;
    families.insert(head.1.family.to_string());
    canonical_stages.push(head.0);

    // Include a present target descriptor in support/admission/family checks.
    if let Some((_, descriptor)) = classified
        .first()
        .filter(|(_, descriptor)| descriptor.stage_class == StageClass::Target)
    {
        check_operation(descriptor, task, schema, options, &shape)?;
        families.insert(descriptor.family.to_string());
        target_aware |= descriptor.target_aware;
    }

    let canonical = DslExpr {
        stages: canonical_stages.clone(),
    }
    .to_string();
    let semantic_hash = semantic_hash(&canonical);
    Ok(CompiledV3Genome::new(
        canonical,
        semantic_hash,
        canonical_stages,
        shape.widths(),
        requires_ordered_rows,
        target_aware,
        families.into_iter().collect(),
    ))
}

pub(super) fn classify_stages(
    stages: &[Stage],
    task: TaskKind,
) -> Result<Vec<(&Stage, &'static OperationDescriptor)>> {
    let mut classified = Vec::with_capacity(stages.len());
    let mut target_count = 0usize;
    let mut input_count = 0usize;
    let mut head_count = 0usize;
    for (index, stage) in stages.iter().enumerate() {
        let descriptor = match descriptor_for(&stage.op, task) {
            Some(descriptor) => descriptor,
            None if descriptor_for_any_task(&stage.op).is_some() => {
                bail!("{} is incompatible with {:?} tasks", stage.op, task)
            }
            None => bail!("unknown advanced-v3 operation '{}'", stage.op),
        };
        match descriptor.stage_class {
            StageClass::Target => {
                target_count += 1;
                if index != 0 || target_count != 1 {
                    bail!("target stage must appear once as the first stage");
                }
            }
            StageClass::Input => {
                input_count += 1;
                let required_index = usize::from(target_count == 1);
                if index != required_index || input_count != 1 {
                    bail!("input stage must appear once after the optional target stage");
                }
            }
            StageClass::Transform => {
                if head_count != 0 {
                    bail!("transform '{}' cannot follow a model head", stage.op);
                }
            }
            StageClass::Learner => {
                head_count += 1;
                if head_count != 1 || index + 1 != stages.len() {
                    bail!("model heads are only valid as the final stage");
                }
            }
        }
        classified.push((stage, descriptor));
    }
    Ok(classified)
}

pub(super) fn descriptor_for(op: &str, task: TaskKind) -> Option<&'static OperationDescriptor> {
    REGISTRY.iter().find(|descriptor| {
        descriptor.tasks.contains(task)
            && (descriptor.name == op || descriptor.aliases.contains(&op))
    })
}

pub(super) fn descriptor_for_any_task(op: &str) -> Option<&'static OperationDescriptor> {
    REGISTRY
        .iter()
        .find(|descriptor| descriptor.name == op || descriptor.aliases.contains(&op))
}

pub(super) fn compatibility_input_stage() -> Stage {
    let mut stage = Stage::new("input");
    stage
        .params
        .insert("numeric".to_string(), ParamValue::Str("zero".to_string()));
    stage
        .params
        .insert("missing".to_string(), ParamValue::Str("none".to_string()));
    stage.params.insert(
        "categorical".to_string(),
        ParamValue::Str("native".to_string()),
    );
    stage
}

pub(super) fn materialize_stage(
    source: &Stage,
    descriptor: &OperationDescriptor,
    task: &TaskSpec,
    mode: CompileMode,
) -> Result<Stage> {
    let mut supplied = BTreeMap::<&str, (&str, &ParamValue)>::new();
    for (key, value) in &source.params {
        let Some(parameter) = descriptor
            .parameters
            .iter()
            .find(|parameter| parameter.name == key || parameter.aliases.contains(&key.as_str()))
        else {
            if descriptor.accepts_append && key == "append" {
                if supplied.insert("append", (key, value)).is_some() {
                    bail!("{}: duplicate append parameter", source.op);
                }
                continue;
            }
            bail!("{}: unknown parameter '{key}'", source.op);
        };
        if let Some((previous, _)) = supplied.insert(parameter.name, (key, value)) {
            bail!(
                "{}: parameters '{previous}' and '{key}' both set '{}'",
                source.op,
                parameter.name
            );
        }
    }

    let mut out = Stage::new(descriptor.name);
    for parameter in descriptor.parameters {
        let domain = selected_domain(parameter, mode);
        let supplied_value = supplied.get(parameter.name).map(|(_, value)| *value);
        if domain.is_none() {
            if supplied_value.is_some() {
                bail!(
                    "{}: parameter '{}' is unavailable in {:?} mode",
                    descriptor.name,
                    parameter.name,
                    mode
                );
            }
            continue;
        }
        let value = match supplied_value {
            Some(value) => normalize_value(descriptor.name, parameter, value, domain.unwrap())?,
            None => match parameter.defaults.selected(mode) {
                Some(default) => static_value(default, task, descriptor.name),
                None => bail!(
                    "{}: missing required parameter '{}'",
                    descriptor.name,
                    parameter.name
                ),
            },
        };
        out.params.insert(parameter.name.to_string(), value);
    }
    if descriptor.accepts_append {
        let append = match supplied.get("append") {
            Some((_, ParamValue::Bool(value))) => *value,
            Some(_) => bail!("{}: append must be a boolean", descriptor.name),
            None => false,
        };
        out.params
            .insert("append".to_string(), ParamValue::Bool(append));
    }

    for parameter in descriptor.parameters {
        if !is_active(parameter.active_when, &out.params, task.kind) {
            if supplied.contains_key(parameter.name) {
                bail!(
                    "{}: parameter '{}' is inactive for the selected configuration",
                    descriptor.name,
                    parameter.name
                );
            }
            out.params.remove(parameter.name);
        }
    }
    normalize_value_aliases(&mut out);
    validate_task_specific_values(&out, task.kind)?;
    Ok(out)
}
