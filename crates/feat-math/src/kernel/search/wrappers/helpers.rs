use super::*;

pub(crate) fn is_expression(stage: &Stage) -> bool {
    matches!(stage.op.as_str(), "map" | "derive")
}
pub(crate) fn expression_sites(pipe: &DslExpr) -> Vec<usize> {
    (0..pipe.stages.len() - 1)
        .filter(|&i| is_expression(&pipe.stages[i]))
        .collect()
}
pub(crate) fn expression(stage: &Stage) -> Result<MathExpr> {
    let Some(ParamValue::Str(source)) = stage.params.get("expr") else {
        return Err(Error::Syntax("expression carrier needs expr".into()));
    };
    MathExpr::parse(source).map_err(expression_error)
}
pub(crate) fn expression_error(error: anyhow::Error) -> Error {
    Error::Bounds(error.to_string())
}
pub(crate) fn seed(rng: &mut Pcg32) -> u64 {
    (u64::from(rng.next_u32()) << 32) | u64::from(rng.next_u32())
}
pub(crate) fn index_of(rng: &mut Pcg32, size: usize) -> usize {
    (seed(rng) % size as u64) as usize
}
pub(crate) fn check_ratio(n: u32, d: u32) -> Result<()> {
    if d == 0 || n > d {
        return Err(Error::Bounds("invalid profile probability".into()));
    }
    Ok(())
}
pub(crate) fn chance(rng: &mut Pcg32, n: u32, d: u32) -> bool {
    seed(rng) % u64::from(d) < u64::from(n)
}
pub(crate) fn check_weights(weights: &[PipelineArmWeight]) -> Result<()> {
    let mut seen = BTreeSet::new();
    if weights.iter().any(|row| !seen.insert(row.arm))
        || weights.iter().map(|row| u64::from(row.weight)).sum::<u64>() == 0
    {
        return Err(Error::Bounds(
            "pipeline weights must be unique with positive total mass".into(),
        ));
    }
    Ok(())
}
pub(crate) fn draw_arm(weights: &[PipelineArmWeight], rng: &mut Pcg32) -> PipelineArm {
    let total: u64 = weights.iter().map(|row| u64::from(row.weight)).sum();
    let mut draw = seed(rng) % total;
    for row in weights {
        if draw < u64::from(row.weight) {
            return row.arm;
        }
        draw -= u64::from(row.weight);
    }
    unreachable!("validated positive weights")
}

pub(crate) fn draw_domain(domain: &ParamDomain, rng: &mut Pcg32) -> Result<Option<ParamValue>> {
    let invalid = || Error::Bounds("invalid profile birth domain".into());
    let value = match domain {
        ParamDomain::IntRange { lo, hi } => {
            let span = i128::from(*hi) - i128::from(*lo) + 1;
            if span <= 0 || span > i128::from(u64::MAX) {
                return Err(invalid());
            }
            ParamValue::Int((i128::from(*lo) + i128::from(seed(rng) % span as u64)) as i64)
        }
        ParamDomain::FloatRange { lo, hi } => {
            if !lo.is_finite() || !hi.is_finite() || lo > hi || !(hi - lo).is_finite() {
                return Err(invalid());
            }
            ParamValue::Float(rng.range_f64(*lo, *hi))
        }
        ParamDomain::IntChoice(values) => {
            if values.is_empty() {
                return Err(invalid());
            }
            ParamValue::Int(values[index_of(rng, values.len())])
        }
        ParamDomain::FloatChoice(values) => {
            if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
                return Err(invalid());
            }
            ParamValue::Float(values[index_of(rng, values.len())])
        }
        ParamDomain::TextChoice(values) => {
            if values.is_empty() {
                return Err(invalid());
            }
            match &values[index_of(rng, values.len())] {
                TextChoice::Value(value) => ParamValue::Str((*value).into()),
                TextChoice::Omit => return Ok(None),
            }
        }
        ParamDomain::Flag {
            true_numerator,
            denominator,
        } => {
            check_ratio(*true_numerator, *denominator)?;
            ParamValue::Bool(chance(rng, *true_numerator, *denominator))
        }
    };
    Ok(Some(value))
}
