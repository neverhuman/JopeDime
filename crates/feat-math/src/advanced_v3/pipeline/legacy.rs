use super::*;

pub fn fit_legacy_numeric_bound(
    genome: &GenomeIr,
    x: &[Vec<f64>],
    y: &[f64],
) -> Result<BoundModel> {
    if x.is_empty() || x.len() != y.len() {
        bail!("training matrix is empty or target length differs from row count");
    }
    let width = x[0].len();
    if width == 0 || x.iter().any(|row| row.len() != width) {
        bail!("legacy numeric training matrix is empty or ragged");
    }
    if x.iter().flatten().any(|value| !value.is_finite())
        || y.iter().any(|value| !value.is_finite())
    {
        bail!("legacy numeric bound models require finite inputs and targets");
    }
    let schema = DataSchema {
        columns: (0..width)
            .map(|index| super::schema::ColumnSpec::numeric(format!("c{index}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable {
        headers: schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect(),
        rows: x
            .iter()
            .map(|row| row.iter().map(|value| value.to_string()).collect())
            .collect(),
    };
    let task = TaskSpec::regression(super::types::Objective::Rmse)?;
    let config = InputConfig::compatibility();
    let (input, block) = FittedInput::fit_transform(config, &schema, &table, y, task.kind)?;
    let predictor = BoundPredictor::Legacy(crate::model::fit(genome, &block.values, y)?);
    Ok(BoundModel {
        artifact_schema: ARTIFACT_SCHEMA.to_string(),
        task_sha256: stable_hash(&task)?,
        schema_sha256: schema.sha256()?,
        vocabulary_sha256: stable_hash(&input)?,
        genome_sha256: text_hash(&genome.canonical()),
        threshold: 0.5,
        task,
        schema,
        input,
        predictor,
    })
}

/// Bind an already-fitted legacy `crate::model::Model` directly into the V3
/// `BoundModel` envelope without re-running `crate::model::fit` or formatting
/// `x` to strings. Produces a `BoundModel` byte-for-byte identical to
/// [`fit_legacy_numeric_bound`].
pub fn bound_from_retained_legacy_model(
    genome: &GenomeIr,
    x: &[Vec<f64>],
    y: &[f64],
    retained_model: crate::model::Model,
) -> Result<BoundModel> {
    if x.is_empty() || x.len() != y.len() {
        bail!("training matrix is empty or target length differs from row count");
    }
    let width = x[0].len();
    if width == 0 || x.iter().any(|row| row.len() != width) {
        bail!("legacy numeric training matrix is empty or ragged");
    }
    if x.iter().flatten().any(|value| !value.is_finite())
        || y.iter().any(|value| !value.is_finite())
    {
        bail!("legacy numeric bound models require finite inputs and targets");
    }
    let schema = DataSchema {
        columns: (0..width)
            .map(|index| super::schema::ColumnSpec::numeric(format!("c{index}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let task = TaskSpec::regression(super::types::Objective::Rmse)?;
    let input = FittedInput::compatibility_numeric(width);
    let predictor = BoundPredictor::Legacy(retained_model);
    Ok(BoundModel {
        artifact_schema: ARTIFACT_SCHEMA.to_string(),
        task_sha256: stable_hash(&task)?,
        schema_sha256: schema.sha256()?,
        vocabulary_sha256: stable_hash(&input)?,
        genome_sha256: text_hash(&genome.canonical()),
        threshold: 0.5,
        task,
        schema,
        input,
        predictor,
    })
}

/// When a pipeline uses only compatibility numeric input on an all-finite
/// training matrix `x` and target vector `y` (with no missing or non-finite cells),
/// legacy non-OOF transforms within the 512-column dense cap, and a legacy
/// regression head, its `BoundPredictor::Pipeline` state is identical to the
/// already-fitted `crate::model::Model::Pipeline`.
///
/// If any training cell in `x` or target in `y` is non-finite (including `NaN`
/// used for missing cells before `model::fit`), or if `x` is empty or does not
/// match `schema` and `y`, returns `Ok(None)` so [`fit_table`] is executed with
/// its per-stage missingness mask semantics.
pub fn try_bound_pipeline_from_retained_legacy_model(
    genome: &GenomeIr,
    task: &TaskSpec,
    schema: &DataSchema,
    x: &[Vec<f64>],
    y: &[f64],
    retained_model: &crate::model::Model,
) -> Result<Option<BoundModel>> {
    if task.validate().is_err()
        || schema.validate().is_err()
        || validate_v3_genome(genome, task, schema).is_err()
    {
        return Ok(None);
    }
    if task.kind != TaskKind::Regression || schema.row_semantics != RowSemantics::Iid {
        return Ok(None);
    }
    if !schema
        .columns
        .iter()
        .all(|c| c.kind == super::schema::ColumnKind::Numeric)
    {
        return Ok(None);
    }
    let width = schema.columns.len();
    if x.is_empty()
        || x.len() != y.len()
        || width == 0
        || x.iter().any(|row| row.len() != width)
        || x.iter().flatten().any(|value| !value.is_finite())
        || y.iter().any(|value| !value.is_finite())
    {
        return Ok(None);
    }
    let GenomeIr::Pipeline(dsl) = genome else {
        return Ok(None);
    };
    let parsed = split_v3_pipeline(dsl)?;
    if parsed.input.is_some() {
        return Ok(None);
    }
    for stage in parsed.transforms {
        if matches!(
            stage.op.as_str(),
            "maxabs_scale"
                | "power_transform"
                | "piecewise_linear"
                | "corr_screen"
                | "pls_project"
                | "lag"
                | "roll"
                | "spline"
                | "impute"
        ) {
            return Ok(None);
        }
        if stage.op == "target_bin_encode" && stage.params.contains_key("oof") {
            return Ok(None);
        }
    }
    let head_op = parsed.head.map_or("ridge", |s| s.op.as_str());
    if !crate::heads::is_head_op(head_op) {
        return Ok(None);
    }
    let crate::model::Model::Pipeline { target, ops, head } = retained_model else {
        return Ok(None);
    };
    if ops.len() != parsed.transforms.len() {
        return Ok(None);
    }
    if crate::kernel::width::check_input_width(retained_model, width).is_err() {
        return Ok(None);
    }
    let input = FittedInput::compatibility_numeric(width);
    let v3_ops: Vec<V3Op> = ops.iter().cloned().map(V3Op::Legacy).collect();
    let v3_head = V3Head::Legacy((**head).clone());

    let probe_table = RawTable {
        headers: schema.columns.iter().map(|c| c.name.clone()).collect(),
        rows: vec![vec!["0.0".to_string(); width]],
    };
    let Ok(mut probe_block) = input.transform(&probe_table) else {
        return Ok(None);
    };
    let mut legacy_rows = vec![vec![0.0; width]];
    for ((stage, v3_op), leg_op) in parsed.transforms.iter().zip(v3_ops.iter()).zip(ops.iter()) {
        if crate::kernel::width::stage_name(leg_op) != stage.op.as_str() {
            return Ok(None);
        }
        if v3_op.validate_fitted_state(schema).is_err() {
            return Ok(None);
        }
        let Ok(next_block) = v3_op.apply_block(&probe_block, &probe_table) else {
            return Ok(None);
        };
        let Ok(next_legacy) = crate::ops::apply_op(leg_op, &legacy_rows) else {
            return Ok(None);
        };
        let v3_w = next_block.values.first().map_or(0, Vec::len);
        let leg_w = next_legacy.first().map_or(0, Vec::len);
        if v3_w == 0 || v3_w > crate::MAX_WIDTH || v3_w != leg_w {
            return Ok(None);
        }
        probe_block = next_block;
        legacy_rows = next_legacy;
    }
    let final_w = probe_block.values.first().map_or(0, Vec::len);
    if !crate::kernel::width::required_head_width(head).admits(final_w) {
        return Ok(None);
    }
    let Ok(v3_pred) = v3_head.predict(&probe_block.values) else {
        return Ok(None);
    };
    let Ok(leg_pred) = head.predict(&legacy_rows) else {
        return Ok(None);
    };
    if v3_pred.len() != 1 || leg_pred.len() != 1 || v3_pred[0].to_bits() != leg_pred[0].to_bits() {
        return Ok(None);
    }

    let bound = BoundModel {
        artifact_schema: ARTIFACT_SCHEMA.to_string(),
        task_sha256: stable_hash(task)?,
        schema_sha256: schema.sha256()?,
        vocabulary_sha256: stable_hash(&input)?,
        genome_sha256: text_hash(&genome.canonical()),
        threshold: 0.5,
        task: task.clone(),
        schema: schema.clone(),
        input,
        predictor: BoundPredictor::Pipeline {
            target: target.clone(),
            ops: v3_ops,
            head: v3_head,
            calibration: CalibrationModel::None,
            missing_aware: false,
        },
    };
    if bound.validate_binding().is_err() {
        return Ok(None);
    }
    Ok(Some(bound))
}
