//! Generated recipes use this private worker contract. Registered baseline gates
//! in methods.rs are unchanged. ExecutionPool owns process/resource admission.
use super::{
    data::RAW_CELL_ENCODING,
    methods::task_spec,
    score::{predict_bound, Predictions},
    worker::write_private,
};
use anyhow::{ensure, Context, Result};
use feat_math::{
    advanced_v3::{fit_table, validate_v3_genome, BoundModel, DataSchema, RawTable},
    GenomeIr,
};
use jope_datasets::catalog::Task;
use jope_pipeline::{
    sha256_hex,
    synthesized_recipe::{SynthesizedRecipe, FIT_RANDOMNESS_POLICY},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, io::Read, path::Path};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SynthesizedFitRequest {
    pub task: Task,
    pub schema: DataSchema,
    pub classes: Vec<String>,
    pub train: RawTable,
    pub targets: Vec<f64>,
    pub predict: RawTable,
    pub recipe: SynthesizedRecipe,
    /// Identity only; fit_table consumes DSL/internal engine seeds.
    pub scheduling_seed: u64,
}
fn classes_valid(task: Task, classes: &[String]) -> Result<()> {
    let count = classes.iter().collect::<BTreeSet<_>>().len();
    ensure!(
        count == classes.len()
            && classes.iter().all(|s| !s.is_empty())
            && match task {
                Task::Regression => count == 0,
                Task::Binary => count == 2,
                Task::Multiclass => count >= 3,
                _ => false,
            },
        "invalid generated fit class order/task"
    );
    Ok(())
}
impl SynthesizedFitRequest {
    pub fn validate(&self) -> Result<()> {
        self.recipe.validate()?;
        classes_valid(self.task, &self.classes)?;
        self.schema.validate()?;
        self.train.validate_against(&self.schema)?;
        self.predict.validate_against(&self.schema)?;
        ensure!(
            !self.train.rows.is_empty()
                && self.train.rows.len() == self.targets.len()
                && self.targets.iter().all(|v| v.is_finite()),
            "invalid generated fit training rows/targets"
        );
        if self.task != Task::Regression {
            let mut seen = BTreeSet::new();
            for value in &self.targets {
                ensure!(
                    *value >= 0. && value.fract() == 0. && *value < (self.classes.len() as f64),
                    "generated fit target outside declared class order"
                );
                seen.insert(*value as usize);
            }
            ensure!(
                seen.len() == self.classes.len(),
                "class absent from generated fit training rows"
            );
        }
        validate_v3_genome(
            &GenomeIr::parse(&self.recipe.executable_canonical)?,
            &task_spec(self.task)?,
            &self.schema,
        )
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedSynthesizedArtifact {
    pub schema: String,
    pub raw_cell_encoding: String,
    pub task: Task,
    pub classes: Vec<String>,
    pub recipe: SynthesizedRecipe,
    pub scheduling_seed: u64,
    pub model: BoundModel,
}
impl FittedSynthesizedArtifact {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "jope.panel-fitted-synthesized/v1"
                && self.raw_cell_encoding == RAW_CELL_ENCODING,
            "generated fitted artifact format changed"
        );
        self.recipe.validate()?;
        classes_valid(self.task, &self.classes)?;
        ensure!(
            self.model.task == task_spec(self.task)?
                && self.model.genome_sha256 == self.recipe.executable_sha256
                && self.recipe.randomness_policy == FIT_RANDOMNESS_POLICY,
            "generated fitted task/recipe/randomness changed"
        );
        validate_v3_genome(
            &GenomeIr::parse(&self.recipe.executable_canonical)?,
            &task_spec(self.task)?,
            &self.model.schema,
        )?;
        self.model.validate_binding()
    }
    pub fn predict(&self, table: &RawTable) -> Result<Predictions> {
        self.validate()?;
        predict_bound(&self.model, table, &self.classes)
    }
    pub fn bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        Ok(serde_json::to_vec(self)?)
    }
    pub fn from_bytes(bytes: &[u8], expected_sha256: &str) -> Result<Self> {
        ensure!(
            sha256_hex(bytes) == expected_sha256,
            "generated fitted artifact checksum differs"
        );
        let value: Self = serde_json::from_slice(bytes)?;
        value.validate()?;
        Ok(value)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SynthesizedWorkerOutput {
    pub schema: String,
    pub input_sha256: String,
    pub executable_recipe_sha256: String,
    pub physical_fit_starts: u64,
    pub artifact_sha256: String,
    pub predictions: Predictions,
    pub artifact: FittedSynthesizedArtifact,
}
/// One actual fit_table call, then lossless fitted reload and prediction replay.
/// Failure before publication supplies no fabricated physical-fit counter.
pub fn fit_synthesized(request: &SynthesizedFitRequest) -> Result<SynthesizedWorkerOutput> {
    request.validate()?;
    let model = fit_table(
        &GenomeIr::parse(&request.recipe.executable_canonical)?,
        &task_spec(request.task)?,
        &request.schema,
        &request.train,
        &request.targets,
    )?;
    let artifact = FittedSynthesizedArtifact {
        schema: "jope.panel-fitted-synthesized/v1".into(),
        raw_cell_encoding: RAW_CELL_ENCODING.into(),
        task: request.task,
        classes: request.classes.clone(),
        recipe: request.recipe.clone(),
        scheduling_seed: request.scheduling_seed,
        model,
    };
    let predictions = artifact.predict(&request.predict)?;
    let bytes = artifact.bytes()?;
    let artifact_sha256 = sha256_hex(&bytes);
    let restored = FittedSynthesizedArtifact::from_bytes(&bytes, &artifact_sha256)?;
    ensure!(
        serde_json::to_vec(&restored.predict(&request.predict)?)?
            == serde_json::to_vec(&predictions)?,
        "generated fitted reload changed predictions"
    );
    Ok(SynthesizedWorkerOutput {
        schema: "jope.panel-synthesized-worker-output/v1".into(),
        input_sha256: sha256_hex(&serde_json::to_vec(request)?),
        executable_recipe_sha256: request.recipe.executable_sha256.clone(),
        physical_fit_starts: 1,
        artifact_sha256,
        predictions,
        artifact,
    })
}
pub fn worker(input: &Path, output: &Path) -> Result<()> {
    ensure!(
        std::fs::metadata(input)?.len() <= 128 * 1024 * 1024,
        "generated fit input exceeds128MiB"
    );
    let mut bytes = vec![];
    std::fs::File::open(input)
        .context("read generated fit input")?
        .take(128 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 128 * 1024 * 1024,
        "generated fit input exceeds128MiB"
    );
    let request: SynthesizedFitRequest = serde_json::from_slice(&bytes)?;
    let mut result = fit_synthesized(&request)?;
    result.input_sha256 = sha256_hex(&bytes);
    write_private(output, &result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use feat_math::advanced_v3::{ColumnSpec, RowSemantics};
    #[test]
    fn generated_recipe_is_fitted_saved_reloaded_with_missing_and_unseen_categories() -> Result<()>
    {
        let schema = DataSchema {
            columns: vec![ColumnSpec::numeric("x"), ColumnSpec::categorical("kind")],
            row_semantics: RowSemantics::Iid,
        };
        let features = vec![
            jope_pipeline::experiment::Column {
                name: "x".into(),
                kind: jope_pipeline::experiment::ColumnKind::Numeric,
            },
            jope_pipeline::experiment::Column {
                name: "kind".into(),
                kind: jope_pipeline::experiment::ColumnKind::Categorical,
            },
        ];
        let values = (0..18)
            .map(|i| {
                vec![
                    Some((i as f64).to_string()),
                    if i % 5 == 0 {
                        None
                    } else {
                        Some(if i % 2 == 0 { "red" } else { "blue" }.into())
                    },
                ]
            })
            .collect::<Vec<_>>();
        let train = super::super::data::raw_table(&features, &values)?;
        let predict = super::super::data::raw_table(
            &features,
            &[
                vec![None, Some("new category".into())],
                vec![Some("3".into()), None],
            ],
        )?;
        let request = SynthesizedFitRequest {
            task: Task::Regression,
            schema,
            classes: vec![],
            train,
            targets: (0..18).map(|i| i as f64 * 2. + 1.).collect(),
            predict,
            recipe: SynthesizedRecipe::new(" (pipe X (standardize) (ridge lambda=0.5)) ".into())?,
            scheduling_seed: 21,
        };
        let result = fit_synthesized(&request)?;
        assert_eq!(result.physical_fit_starts, 1);
        assert!(result.artifact.recipe.inserted_input_stage);
        assert_eq!(
            result.artifact.recipe.original_proposal,
            request.recipe.original_proposal
        );
        assert_eq!(
            result.artifact.model.genome_sha256,
            result.executable_recipe_sha256
        );
        let bytes = result.artifact.bytes()?;
        let restored = FittedSynthesizedArtifact::from_bytes(&bytes, &result.artifact_sha256)?;
        assert_eq!(
            serde_json::to_value(restored.predict(&request.predict)?)?,
            serde_json::to_value(result.predictions)?
        );
        Ok(())
    }
    #[test]
    fn generated_classification_reuses_complete_probability_class_order() -> Result<()> {
        for (task, head, classes) in [
            (
                Task::Binary,
                "logit_ridge",
                vec!["yes".to_owned(), "no".to_owned()],
            ),
            (
                Task::Multiclass,
                "multiclass_ridge",
                vec!["z".to_owned(), "a".to_owned(), "middle".to_owned()],
            ),
        ] {
            let schema = DataSchema {
                columns: vec![ColumnSpec::numeric("x")],
                row_semantics: RowSemantics::Iid,
            };
            let table = RawTable::new(
                vec!["x".into()],
                (0..24).map(|i| vec![i.to_string()]).collect(),
            )?;
            let request = SynthesizedFitRequest {
                task,
                schema,
                classes: classes.clone(),
                train: table.clone(),
                targets: (0..24).map(|i| (i % classes.len()) as f64).collect(),
                predict: table,
                recipe: SynthesizedRecipe::new(format!("(pipe X ({head}))"))?,
                scheduling_seed: 11,
            };
            let result = fit_synthesized(&request)?;
            let Predictions::Probabilities {
                classes: actual,
                values,
            } = result.predictions
            else {
                anyhow::bail!("classification worker omitted probability matrix")
            };
            assert_eq!(actual, classes);
            assert_eq!(values.len(), 24);
            assert!(values.iter().all(|r| r.len() == classes.len()));
        }
        Ok(())
    }
}
