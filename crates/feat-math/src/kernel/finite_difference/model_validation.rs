//! The initial ordered fit closure is numeric identity input plus plain Ridge.
//! Opaque v1 programs remain lossless; unproved fitted shapes stay refused.
use super::*;
use crate::kernel::wire_v2::{VersionError, VersionResult};
use crate::spec::{Plan, PlannedHead};
use serde_json::Value;

pub(super) fn validate_fit_closure(genome: &crate::GenomeIr) -> Result<()> {
    let plan = crate::spec::build_plan(genome).map_err(|e| Error::Unsupported(e.join("; ")))?;
    if matches!(plan, Plan::Pipeline { target: None, stages, head: PlannedHead::Ridge { lambda_frac } }
        if stages.is_empty() && lambda_frac.is_finite() && lambda_frac >= 0.0)
    {
        Ok(())
    } else {
        Err(Error::Unsupported(
            "ordered difference fit requires an untransformed numeric Ridge pipeline".into(),
        ))
    }
}

pub(super) fn validate_compat_ridge(value: &Value, width: usize) -> VersionResult<()> {
    let input = serde_json::json!({
        "columns": (0..width).map(|_| serde_json::json!({"Numeric":{"fill":0.0,"indicator":false}})).collect::<Vec<_>>(),
        "config": {"numeric":"Zero","missing_indicator":false,"categorical":"Native",
            "max_onehot":16,"min_count":2,"prior":10.0,"folds":5},
        "historical":vec![Value::Null;width],"row_order":"Iid"
    });
    if value.get("input") != Some(&input) {
        return Err(VersionError::Malformed(
            "difference compatibility input layout".into(),
        ));
    }
    let pipeline = &value["predictor"]["Legacy"]["Pipeline"];
    let ridge = &pipeline["head"]["Ridge"];
    let means = ridge["means"].as_array();
    let scales = ridge["scales"].as_array();
    let weights = ridge["w"].as_array();
    let finite = |values: Option<&Vec<Value>>, n: usize, positive: bool| {
        values.is_some_and(|v| {
            v.len() == n
                && v.iter().all(|x| {
                    x.as_f64()
                        .is_some_and(|f| f.is_finite() && (!positive || f > 0.0))
                })
        })
    };
    if pipeline.get("target") != Some(&Value::Null)
        || pipeline["ops"].as_array().is_none_or(|ops| !ops.is_empty())
        || !finite(means, width, false)
        || !finite(scales, width, true)
        || !finite(weights, width + 1, false)
    {
        return Err(VersionError::Malformed(
            "difference fitted ridge state".into(),
        ));
    }
    Ok(())
}
