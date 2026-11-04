use clap::Args;
use jope_api::{ids::DatasetId, Language, RunSpec};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::{path::PathBuf, str::FromStr};

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Reuse the original key when retrying the same run request.
    #[arg(long)]
    pub idempotency_key: Option<String>,
    #[arg(
        value_name = "CSV",
        required_unless_present = "dataset",
        conflicts_with = "dataset"
    )]
    pub input: Option<PathBuf>,
    #[arg(long, required_unless_present = "input", conflicts_with = "input")]
    pub dataset: Option<String>,
    #[arg(long)]
    pub target: String,
    #[arg(long, default_value = "auto")]
    pub task: String,
    #[arg(long = "ignore-column")]
    pub ignore_columns: Vec<String>,
    #[arg(long)]
    pub test_dataset: Option<String>,
    #[arg(
        long,
        default_value = "{\"kind\":\"random\",\"holdout_fraction\":0.2,\"group_column\":null,\"time_column\":null}"
    )]
    pub split: String,
    #[arg(long, default_value = "rmse")]
    pub objective: String,
    #[arg(long, default_value = "medium")]
    pub effort: String,
    #[arg(long, default_value = "{\"evaluations\":null,\"wall_seconds\":null}")]
    pub budget: String,
    #[arg(long, default_value = "full")]
    pub landmarks: String,
    #[arg(long, default_value = "auto")]
    pub guidance: String,
    #[arg(long = "language", alias = "lang", value_delimiter = ',')]
    pub languages: Vec<String>,
    #[arg(long)]
    pub require_all_languages: bool,
    #[arg(long, default_value = "auto")]
    pub device: String,
    #[arg(long, default_value = "0")]
    pub seed: String,
    #[arg(long, conflicts_with_all = ["detach", "out", "events"])]
    pub dry_run: bool,
    #[arg(long)]
    pub plan_digest: Option<String>,
    #[arg(long, conflicts_with_all = ["dry_run", "events", "out"])]
    pub detach: bool,
    #[arg(long)]
    pub out: Option<PathBuf>,
    #[arg(long, default_value = "fit_holdout", value_parser = ["fit", "fit_holdout"])]
    pub refit: String,
    #[arg(long)]
    pub json: bool,
    #[arg(long, value_parser = ["ndjson"])]
    pub events: Option<String>,
    #[arg(long, default_value_t = 30)]
    pub wait: u64,
    #[arg(long)]
    pub after_event: Option<String>,
    /// Internal M1 probe using a prepared FIT-only service plan.
    #[arg(
        long,
        hide = true,
        requires = "json",
        conflicts_with_all = ["dry_run", "detach", "out", "events", "plan_digest", "after_event", "wait"]
    )]
    pub m1_plan: Option<PathBuf>,
}

impl RunArgs {
    pub fn run_spec(&self) -> anyhow::Result<RunSpec> {
        let dataset = DatasetId::from_str(self.dataset.as_deref().ok_or_else(|| {
            anyhow::anyhow!("CSV must be registered before constructing RunSpec")
        })?)
        .map_err(|_| anyhow::anyhow!("invalid dataset identifier"))?;
        let test_dataset = self
            .test_dataset
            .as_deref()
            .map(DatasetId::from_str)
            .transpose()
            .map_err(|_| anyhow::anyhow!("invalid test dataset identifier"))?;
        Ok(RunSpec {
            dataset,
            target: self.target.clone(),
            task: parse_value(&self.task)?,
            ignore_columns: self.ignore_columns.clone(),
            test_dataset,
            split: parse_json(&self.split)?,
            objective: parse_value(&self.objective)?,
            effort: parse_value(&self.effort)?,
            budget: parse_json(&self.budget)?,
            landmarks: parse_value(&self.landmarks)?,
            guidance: parse_value(&self.guidance)?,
            languages: self
                .languages
                .iter()
                .map(|value| parse_value(value))
                .collect::<anyhow::Result<Vec<Language>>>()?,
            require_all_languages: self.require_all_languages,
            device: self.device.clone(),
            seed: serde_json::from_value(Value::String(self.seed.clone()))?,
            dry_run: self.dry_run,
            plan_digest: self.plan_digest.clone(),
        })
    }
}

fn parse_json<T: DeserializeOwned>(raw: &str) -> anyhow::Result<T> {
    serde_json::from_str(raw).map_err(Into::into)
}

pub(super) fn parse_value<T: DeserializeOwned>(raw: &str) -> anyhow::Result<T> {
    let syntax = match serde_json::from_str::<Value>(raw) {
        Ok(value) => ValueSyntax::Json(value),
        Err(json_error) => ValueSyntax::BareText {
            value: raw.to_owned(),
            json_error,
        },
    };
    match syntax {
        ValueSyntax::Json(value) => serde_json::from_value(value).map_err(Into::into),
        ValueSyntax::BareText { value, json_error } => {
            serde_json::from_value(Value::String(value.clone()))
                .map_err(|source| anyhow::Error::new(BareTextValueError { json_error, source }))
        }
    }
}

enum ValueSyntax {
    Json(Value),
    BareText {
        value: String,
        json_error: serde_json::Error,
    },
}

#[derive(Debug)]
struct BareTextValueError {
    json_error: serde_json::Error,
    source: serde_json::Error,
}

impl std::fmt::Display for BareTextValueError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "input is neither valid JSON ({}) nor a valid plain-text value ({})",
            self.json_error, self.source
        )
    }
}

impl std::error::Error for BareTextValueError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}
