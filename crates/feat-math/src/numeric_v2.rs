//! Explicit successor numerics. Legacy model, operator and generator paths are untouched.
use crate::{dsl::Stage, heads::RidgeModel, GenomeIr, Rows, MAX_WIDTH};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};

pub const PROFILE_ID: &str = "numeric-profile-v2";
pub const OPERATOR_ID: &str = "numeric-operators-v2";
/// A semantic identity, not a registered evaluator implementation digest.
pub const SEMANTICS_ID: &str =
    "ridge-raw-train-range+missing-nonfinite+strict-width+quantile-partition-v2";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericProfileV2 {
    pub schema: Vec<String>,
    pub row_semantics: crate::RowSemantics,
    /// Stable names resolved against schema at fit, never a first-N feature truncation.
    pub selected_columns: Vec<String>,
}
impl NumericProfileV2 {
    pub fn new(schema: Vec<String>) -> Self {
        Self {
            schema,
            row_semantics: crate::RowSemantics::Iid,
            selected_columns: Vec::new(),
        }
    }
    pub fn validate(&self) -> Result<Vec<usize>> {
        ensure!(
            !self.schema.is_empty() && self.schema.len() <= MAX_WIDTH,
            "unsupported schema width"
        );
        let unique: std::collections::BTreeSet<_> = self.schema.iter().collect();
        ensure!(
            unique.len() == self.schema.len() && self.schema.iter().all(|s| !s.is_empty()),
            "schema names must be unique and nonempty"
        );
        let names = if self.selected_columns.is_empty() {
            &self.schema
        } else {
            &self.selected_columns
        };
        let mut indices = Vec::new();
        for name in names {
            let index = self
                .schema
                .iter()
                .position(|n| n == name)
                .context("unknown selected column")?;
            ensure!(!indices.contains(&index), "duplicate selected column");
            indices.push(index);
        }
        Ok(indices)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum TransformV2 {
    Poly {
        pairs: Vec<(usize, usize)>,
        append: bool,
    },
    Quantile {
        edges: Vec<Vec<f64>>,
        bins: usize,
        onehot: bool,
        append: bool,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericModelV2 {
    pub profile_id: String,
    pub operator_id: String,
    pub semantics_id: String,
    pub profile: NumericProfileV2,
    pub program: String,
    pub training_sha256: String,
    pub indices: Vec<usize>,
    pub fill: Vec<f64>,
    pub transforms: Vec<TransformV2>,
    pub ridge: RidgeModel,
    pub clamp_lo: f64,
    pub clamp_hi: f64,
}

fn allowed(stage: &Stage, names: &[&str]) -> Result<()> {
    ensure!(
        stage.params.keys().all(|k| names.contains(&k.as_str())),
        "unsupported parameter for {}",
        stage.op
    );
    Ok(())
}
fn integer(stage: &Stage, name: &str, default: usize) -> Result<usize> {
    match stage.params.get(name) {
        None => Ok(default),
        Some(crate::dsl::ParamValue::Int(n)) if *n > 0 => Ok(*n as usize),
        _ => bail!("{name} must be a positive integer"),
    }
}
fn append(stage: &Stage) -> Result<bool> {
    match stage.params.get("append") {
        None | Some(crate::dsl::ParamValue::Bool(false)) => Ok(false),
        Some(crate::dsl::ParamValue::Bool(true)) => Ok(true),
        _ => bail!("append must be boolean"),
    }
}
fn validate_rows(x: &Rows, width: usize) -> Result<()> {
    ensure!(
        x.iter().all(|r| r.len() == width),
        "row/schema width mismatch"
    );
    Ok(())
}
impl NumericModelV2 {
    /// Only TRAIN rows/targets enter this constructor. No holdout input is accepted.
    pub fn fit(profile: NumericProfileV2, genome: &GenomeIr, x: &Rows, y: &[f64]) -> Result<Self> {
        let indices = profile.validate()?;
        ensure!(
            !x.is_empty() && x.len() == y.len() && y.iter().all(|v| v.is_finite()),
            "invalid training rows/targets"
        );
        validate_rows(x, profile.schema.len())?;
        let GenomeIr::Pipeline(dsl) = genome else {
            bail!("v2 supports ridge pipelines only")
        };
        let (head, stages) = dsl.stages.split_last().context("ridge head required")?;
        ensure!(
            head.op == "ridge",
            "v2 supports ridge only; other heads are held"
        );
        allowed(head, &["lambda_frac"])?;
        let lambda = match head.params.get("lambda_frac") {
            None => 0.02,
            Some(crate::dsl::ParamValue::Float(v)) => *v,
            Some(crate::dsl::ParamValue::Int(v)) => *v as f64,
            _ => bail!("lambda_frac must be numeric"),
        };
        ensure!(lambda.is_finite() && lambda >= 0.0, "invalid lambda_frac");
        let mut fill = Vec::new();
        for &j in &indices {
            let values: Vec<_> = x.iter().map(|r| r[j]).filter(|v| v.is_finite()).collect();
            let value = values
                .iter()
                .fold(0.0, |sum, v| sum + v / values.len() as f64);
            ensure!(value.is_finite(), "imputation overflow");
            fill.push(value); // all-missing column has explicit zero fallback
        }
        let mut cur: Rows = x
            .iter()
            .map(|r| {
                indices
                    .iter()
                    .enumerate()
                    .map(|(k, &j)| if r[j].is_finite() { r[j] } else { fill[k] })
                    .collect()
            })
            .collect();
        let mut transforms = Vec::new();
        for stage in stages {
            let width = cur[0].len();
            let transform = match stage.op.as_str() {
                "impute" => {
                    allowed(stage, &[])?;
                    continue;
                }
                "poly_interact" => {
                    allowed(stage, &["topk", "append"])?;
                    let topk = integer(stage, "topk", width)?;
                    ensure!(topk <= MAX_WIDTH, "topk exceeds width budget");
                    let all = (0..width)
                        .map(|i| (i, i))
                        .chain((0..width).flat_map(|i| ((i + 1)..width).map(move |j| (i, j))));
                    let pairs: Vec<_> = all.take(topk).collect();
                    ensure!(pairs.len() == topk, "topk exceeds available interactions");
                    TransformV2::Poly {
                        pairs,
                        append: append(stage)?,
                    }
                }
                "quantile_bins" | "bin_onehot" => {
                    allowed(stage, &["bins", "append"])?;
                    let bins = integer(stage, "bins", 4)?;
                    ensure!((2..=64).contains(&bins), "bins outside v2 range");
                    let mut edges = Vec::new();
                    for j in 0..width {
                        let mut values: Vec<_> = cur.iter().map(|r| r[j]).collect();
                        values.sort_by(f64::total_cmp);
                        let mut e: Vec<_> = (1..bins)
                            .map(|b| crate::ops::quantile_sorted(&values, b as f64 / bins as f64))
                            .collect();
                        // Interpolation rounding can invert near-tie plateaus. V2 makes the
                        // fitted edge sequence monotone before lower-bound partitioning.
                        for k in 1..e.len() {
                            e[k] = e[k].max(e[k - 1]);
                        }
                        edges.push(e);
                    }
                    TransformV2::Quantile {
                        edges,
                        bins,
                        onehot: stage.op == "bin_onehot",
                        append: append(stage)?,
                    }
                }
                _ => bail!(
                    "unsupported v2 transform {} (including implicit target transforms)",
                    stage.op
                ),
            };
            cur = transform.apply(&cur)?;
            transforms.push(transform);
        }
        let lo = y.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let margin = ((hi - lo).max(1e-9)) * 0.05;
        let clamp_lo = lo - margin;
        let clamp_hi = hi + margin;
        ensure!(
            clamp_lo.is_finite() && clamp_hi.is_finite(),
            "target range overflow"
        );
        let ridge = RidgeModel::fit(&cur, y, lambda);
        ensure!(
            ridge
                .means
                .iter()
                .chain(&ridge.scales)
                .chain(&ridge.w)
                .all(|v| v.is_finite()),
            "ridge fitting overflow"
        );
        let mut training = Vec::new();
        for row in x {
            for value in row {
                training.extend(value.to_bits().to_le_bytes());
            }
        }
        for value in y {
            training.extend(value.to_bits().to_le_bytes());
        }
        let training_sha256 = crate::numeric_v2_emit::hash(&training);
        Ok(Self {
            training_sha256,
            profile_id: PROFILE_ID.into(),
            operator_id: OPERATOR_ID.into(),
            semantics_id: SEMANTICS_ID.into(),
            profile,
            program: genome.canonical(),
            indices,
            fill,
            transforms,
            ridge,
            clamp_lo,
            clamp_hi,
        })
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.profile_id == PROFILE_ID
                && self.operator_id == OPERATOR_ID
                && self.semantics_id == SEMANTICS_ID,
            "numeric identity mismatch"
        );
        ensure!(
            self.indices == self.profile.validate()?,
            "fitted schema selection mismatch"
        );
        ensure!(
            self.fill.len() == self.indices.len() && self.fill.iter().all(|v| v.is_finite()),
            "invalid fitted imputation"
        );
        ensure!(
            self.training_sha256.len() == 64
                && self.training_sha256.bytes().all(|c| c.is_ascii_hexdigit()),
            "invalid training identity"
        );
        ensure!(
            self.clamp_lo.is_finite()
                && self.clamp_hi.is_finite()
                && self.clamp_lo <= self.clamp_hi,
            "invalid prediction bounds"
        );
        let mut width = self.indices.len();
        for transform in &self.transforms {
            let (output, append) = match transform {
                TransformV2::Poly { pairs, append } => {
                    ensure!(
                        !pairs.is_empty() && pairs.iter().all(|&(a, b)| a < width && b < width),
                        "invalid fitted interactions"
                    );
                    (pairs.len(), *append)
                }
                TransformV2::Quantile {
                    edges,
                    bins,
                    onehot,
                    append,
                } => {
                    ensure!(
                        (2..=64).contains(bins)
                            && edges.len() == width
                            && edges.iter().all(|e| e.len() == bins - 1
                                && e.iter().all(|v| v.is_finite())
                                && e.windows(2).all(|w| w[0] <= w[1])),
                        "invalid fitted quantiles"
                    );
                    (width * if *onehot { *bins } else { 1 }, *append)
                }
            };
            width = output + if append { width } else { 0 };
            ensure!(width <= MAX_WIDTH, "invalid fitted width");
        }
        ensure!(
            self.ridge.means.len() == width
                && self.ridge.scales.len() == width
                && self.ridge.w.len() == width + 1,
            "invalid ridge dimensions"
        );
        ensure!(
            self.ridge
                .means
                .iter()
                .chain(&self.ridge.w)
                .all(|v| v.is_finite())
                && self.ridge.scales.iter().all(|v| v.is_finite() && *v > 0.0),
            "invalid ridge state"
        );
        Ok(())
    }
    pub fn predict(&self, schema: &[String], x: &Rows) -> Result<Vec<f64>> {
        self.validate()?;
        ensure!(
            self.profile_id == PROFILE_ID
                && self.operator_id == OPERATOR_ID
                && self.semantics_id == SEMANTICS_ID,
            "numeric identity mismatch"
        );
        ensure!(
            schema == self.profile.schema,
            "schema identity/order mismatch"
        );
        validate_rows(x, schema.len())?;
        let mut cur: Rows = x
            .iter()
            .map(|r| {
                self.indices
                    .iter()
                    .enumerate()
                    .map(|(k, &j)| if r[j].is_finite() { r[j] } else { self.fill[k] })
                    .collect()
            })
            .collect();
        for transform in &self.transforms {
            cur = transform.apply(&cur)?;
        }
        let mut result = self.ridge.predict_numeric_v2(&cur);
        for value in &mut result {
            ensure!(value.is_finite(), "prediction overflow");
            *value = value.clamp(self.clamp_lo, self.clamp_hi);
        }
        Ok(result)
    }
}
impl TransformV2 {
    pub fn apply(&self, x: &Rows) -> Result<Rows> {
        x.iter()
            .map(|row| {
                let (mut values, append) = match self {
                    Self::Poly { pairs, append } => (
                        pairs
                            .iter()
                            .map(|&(a, b)| row[a] * row[b])
                            .collect::<Vec<_>>(),
                        *append,
                    ),
                    Self::Quantile {
                        edges,
                        bins,
                        onehot,
                        append,
                    } => {
                        let mut values = Vec::new();
                        for (j, e) in edges.iter().enumerate() {
                            let k = e.partition_point(|edge| *edge < row[j]);
                            if *onehot {
                                values.extend((0..*bins).map(|b| f64::from(b == k)));
                            } else {
                                values.push(k as f64 / (*bins - 1) as f64);
                            }
                        }
                        (values, *append)
                    }
                };
                if append {
                    let mut full = row.clone();
                    full.append(&mut values);
                    values = full;
                }
                ensure!(
                    values.len() <= MAX_WIDTH,
                    "v2 width budget exceeded; no silent truncation"
                );
                ensure!(values.iter().all(|v| v.is_finite()), "transform overflow");
                Ok(values)
            })
            .collect()
    }
}

/// Split identity travels with predictions so a caller cannot mistake another
/// seed or row assignment for the same out-of-fold evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IidOofV2 {
    pub split_id: String,
    pub seed: u64,
    pub folds: usize,
    pub assignment: Vec<usize>,
    pub fold_identity_sha256: String,
    pub predictions: Vec<f64>,
}

/// Honest IID out-of-fold prediction. Ordered/grouped/temporal semantics are
/// deliberately unavailable here rather than silently shuffled.
pub fn predict_oof_iid(
    profile: NumericProfileV2,
    genome: &GenomeIr,
    x: &Rows,
    y: &[f64],
    folds: usize,
    seed: u64,
) -> Result<IidOofV2> {
    ensure!(x.len() == y.len(), "unaligned targets");
    ensure!(
        profile.row_semantics == crate::RowSemantics::Iid,
        "shuffled folds require declared IID semantics"
    );
    let assignment = crate::stack::numeric_v2_iid_folds(x.len(), folds, seed)?;
    let mut output = vec![0.0; x.len()];
    for fold in 0..folds {
        let train: Rows = x
            .iter()
            .enumerate()
            .filter(|(i, _)| assignment[*i] != fold)
            .map(|(_, r)| r.clone())
            .collect();
        let targets: Vec<f64> = y
            .iter()
            .enumerate()
            .filter(|(i, _)| assignment[*i] != fold)
            .map(|(_, v)| *v)
            .collect();
        let model = NumericModelV2::fit(profile.clone(), genome, &train, &targets)?;
        for (i, row) in x.iter().enumerate().filter(|(i, _)| assignment[*i] == fold) {
            output[i] = model.predict(&profile.schema, &vec![row.clone()])?[0];
        }
    }
    let split_id = "numeric-v2-iid-shuffle-v1";
    let fold_identity_sha256 = crate::numeric_v2_emit::hash(&serde_json::to_vec(&(
        split_id,
        x.len(),
        folds,
        seed,
        &assignment,
    ))?);
    Ok(IidOofV2 {
        split_id: split_id.into(),
        seed,
        folds,
        assignment,
        fold_identity_sha256,
        predictions: output,
    })
}
