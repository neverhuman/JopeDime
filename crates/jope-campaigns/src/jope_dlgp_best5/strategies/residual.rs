use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualRowV1 {
    pub stable_row_id: String,
    pub target: f64,
    pub base_prediction: f64,
    pub residual: f64,
    pub fitted_without_row: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualTargetsV1 {
    pub oof_training: Vec<ResidualRowV1>,
    pub validation: Vec<ResidualRowV1>,
    pub inner_folds: u32,
}

/// Recreates five-fold base predictions inside the supplied outer training
/// partition. The callback receives fit row IDs and prediction row IDs; each
/// outer-validation prediction is made by a base fitted on outer training only.
pub fn cross_fitted_residual_targets<F>(
    rows: &[super::TargetRowV1],
    outer_train_row_ids: &[String],
    outer_validation_row_ids: &[String],
    seed: u64,
    mut predict_base: F,
) -> Result<ResidualTargetsV1, String>
where
    F: FnMut(&[String], &[String]) -> Result<Vec<f64>, String>,
{
    let by_id = rows
        .iter()
        .map(|row| {
            row.validate()?;
            Ok((row.stable_row_id.as_str(), row))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let outer_train = outer_train_row_ids.iter().collect::<BTreeSet<_>>();
    let outer_validation = outer_validation_row_ids.iter().collect::<BTreeSet<_>>();
    if outer_train.len() != outer_train_row_ids.len()
        || outer_validation.len() != outer_validation_row_ids.len()
        || !outer_train.is_disjoint(&outer_validation)
        || outer_train
            .iter()
            .chain(&outer_validation)
            .any(|row| !by_id.contains_key(row.as_str()))
    {
        return Err("outer residual partition is duplicate, overlapping, or unknown".into());
    }
    let mut groups = BTreeMap::<&str, Vec<&str>>::new();
    for row_id in outer_train_row_ids {
        let row = by_id[row_id.as_str()];
        groups
            .entry(row.atomic_group.as_str())
            .or_default()
            .push(row.stable_row_id.as_str());
    }
    if groups.len() < 5 {
        return Err("five-fold residual OOF requires at least five training groups".into());
    }
    let mut ordered_groups = groups.into_iter().collect::<Vec<_>>();
    ordered_groups.sort_by_key(|(group, _)| seed64(&["residual-oof", &seed.to_string(), group]));
    let mut folds = vec![Vec::<String>::new(); 5];
    for (index, (_, members)) in ordered_groups.into_iter().enumerate() {
        folds[index % 5].extend(members.into_iter().map(str::to_string));
    }
    let mut oof = Vec::with_capacity(outer_train.len());
    for fold in &folds {
        let held_out = fold.iter().collect::<BTreeSet<_>>();
        let fit = outer_train_row_ids
            .iter()
            .filter(|row| !held_out.contains(row))
            .cloned()
            .collect::<Vec<_>>();
        let predictions = predict_base(&fit, fold)?;
        if predictions.len() != fold.len() || predictions.iter().any(|value| !value.is_finite()) {
            return Err("base predictor returned invalid OOF predictions".into());
        }
        for (row_id, prediction) in fold.iter().zip(predictions) {
            let target = by_id[row_id.as_str()].target;
            oof.push(ResidualRowV1 {
                stable_row_id: row_id.clone(),
                target,
                base_prediction: prediction,
                residual: target - prediction,
                fitted_without_row: !fit.contains(row_id),
            });
        }
    }
    oof.sort_by(|left, right| left.stable_row_id.cmp(&right.stable_row_id));
    let validation_predictions = predict_base(outer_train_row_ids, outer_validation_row_ids)?;
    if validation_predictions.len() != outer_validation_row_ids.len()
        || validation_predictions
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err("base predictor returned invalid outer-validation predictions".into());
    }
    let mut validation = outer_validation_row_ids
        .iter()
        .zip(validation_predictions)
        .map(|(row_id, prediction)| {
            let target = by_id[row_id.as_str()].target;
            ResidualRowV1 {
                stable_row_id: row_id.clone(),
                target,
                base_prediction: prediction,
                residual: target - prediction,
                fitted_without_row: !outer_train.contains(row_id),
            }
        })
        .collect::<Vec<_>>();
    validation.sort_by(|left, right| left.stable_row_id.cmp(&right.stable_row_id));
    if oof.len() != outer_train.len()
        || oof.iter().any(|row| !row.fitted_without_row)
        || validation.iter().any(|row| !row.fitted_without_row)
    {
        return Err("residual target was predicted by a base fitted on itself".into());
    }
    Ok(ResidualTargetsV1 {
        oof_training: oof,
        validation,
        inner_folds: 5,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualGenomeV1 {
    pub residual_genome: GenomeV1,
    pub eta: f64,
}

impl ResidualGenomeV1 {
    pub fn validate(&self) -> Result<(), String> {
        self.residual_genome.validate()?;
        if ![0.25, 0.5, 0.75, 1.0].contains(&self.eta) {
            return Err("residual eta must be one of 0.25, 0.5, 0.75, 1.0".into());
        }
        Ok(())
    }

    pub fn semantic_sha256(&self, base_semantic_sha256: &str) -> Result<String, String> {
        self.validate()?;
        validate_sha256("residual base semantic", base_semantic_sha256)?;
        canonical_sha256(&(base_semantic_sha256, self))
    }
}

pub fn composite_residual_prediction(
    base: &[f64],
    residual: &[f64],
    eta: f64,
) -> Result<Vec<f64>, String> {
    if base.len() != residual.len()
        || ![0.25, 0.5, 0.75, 1.0].contains(&eta)
        || base.iter().chain(residual).any(|value| !value.is_finite())
    {
        return Err("invalid base/residual composite prediction".into());
    }
    Ok(base
        .iter()
        .zip(residual)
        .map(|(base, residual)| base + eta * residual)
        .collect())
}
