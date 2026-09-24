//! Private batch recipe validation, with no numerical fit or dataset access.
use anyhow::{ensure, Result};
use feat_math::{
    advanced_v3::{validate_v3_genome, DataSchema},
    GenomeIr,
};
use jope_datasets::catalog::Task;
use jope_pipeline::{sha256_hex, synthesized_recipe::SynthesizedRecipe};
use serde::{Deserialize, Serialize};
use std::{io::Read, path::Path};

const MAX_INPUT: usize = 8 * 1024 * 1024;
const MAX_OUTPUT: usize = 32 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    schema: String,
    task: Task,
    data_schema: DataSchema,
    programs: Vec<String>,
}
#[derive(Debug, Serialize)]
pub struct Validation {
    index: usize,
    original_bytes_sha256: String,
    recipe: Option<SynthesizedRecipe>,
    error: Option<String>,
}
#[derive(Debug, Serialize)]
pub struct Response {
    schema: &'static str,
    input_sha256: String,
    results: Vec<Validation>,
}

pub fn validate_bytes(bytes: &[u8]) -> Result<Response> {
    ensure!(bytes.len() <= MAX_INPUT, "validation input exceeds8MiB");
    let request: Request = serde_json::from_slice(bytes)?;
    ensure!(
        request.schema == "jope.experimental-canonicalization-request/v1",
        "unsupported private validation schema"
    );
    let task = super::methods::task_spec(request.task)?;
    request.data_schema.validate()?;
    ensure!(
        !request.programs.is_empty() && request.programs.len() <= 4096,
        "batch requires1..4096programs"
    );
    ensure!(
        request
            .programs
            .iter()
            .all(|p| !p.is_empty() && p.len() <= 16384),
        "proposal bytes outside bound"
    );
    let results = request
        .programs
        .into_iter()
        .enumerate()
        .map(|(index, original)| {
            let original_bytes_sha256 = sha256_hex(original.as_bytes());
            let checked = (|| {
                let recipe = SynthesizedRecipe::new(original)?;
                validate_v3_genome(
                    &GenomeIr::parse(&recipe.executable_canonical)?,
                    &task,
                    &request.data_schema,
                )?;
                Ok::<_, anyhow::Error>(recipe)
            })();
            match checked {
                Ok(recipe) => Validation {
                    index,
                    original_bytes_sha256,
                    recipe: Some(recipe),
                    error: None,
                },
                Err(error) => Validation {
                    index,
                    original_bytes_sha256,
                    recipe: None,
                    error: Some(format!("{error:#}")),
                },
            }
        })
        .collect();
    let response = Response {
        schema: "jope.experimental-canonicalization-response/v1",
        input_sha256: sha256_hex(bytes),
        results,
    };
    ensure!(
        serde_json::to_vec(&response)?.len() <= MAX_OUTPUT,
        "validation output exceeds32MiB"
    );
    Ok(response)
}

pub fn validate_files(input: &Path, output: &Path) -> Result<()> {
    let mut bytes = Vec::new();
    std::fs::File::open(input)?
        .take(MAX_INPUT as u64 + 1)
        .read_to_end(&mut bytes)?;
    super::worker::write_private(output, &validate_bytes(&bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use feat_math::advanced_v3::{ColumnSpec, RowSemantics};
    use serde_json::json;

    fn request(task: &str, programs: &[&str]) -> serde_json::Value {
        json!({"schema":"jope.experimental-canonicalization-request/v1", "task":task,
            "data_schema": DataSchema {columns:vec![ColumnSpec::numeric("x")],row_semantics:RowSemantics::Iid}, "programs":programs})
    }
    #[test]
    fn validates_all_three_tasks_without_fitting_or_replacing_heads() -> Result<()> {
        for (task, head) in [
            ("regression", "ridge"),
            ("binary", "logit_ridge"),
            ("multiclass", "multiclass_ridge"),
        ] {
            let original = format!("  (pipe X ({head}))  ");
            let bytes = serde_json::to_vec(&request(task, &[&original]))?;
            let response = validate_bytes(&bytes)?;
            let recipe = response.results[0]
                .recipe
                .as_ref()
                .expect("task-compatible head");
            assert_eq!(recipe.original_proposal, original);
            assert!(recipe.inserted_input_stage);
            assert_eq!(response.input_sha256, sha256_hex(&bytes));
            recipe.validate()?;
        }
        let response = validate_bytes(&serde_json::to_vec(&request(
            "multiclass",
            &["(pipe X (ridge))"],
        ))?)?;
        assert!(response.results[0].recipe.is_none());
        assert!(response.results[0].error.is_some());
        Ok(())
    }
    #[test]
    fn preserves_invalid_slots_and_rejects_unknown_label_fields() -> Result<()> {
        let mut value = request("binary", &["not a program", "(pipe X (logit_ridge))"]);
        let response = validate_bytes(&serde_json::to_vec(&value)?)?;
        assert_eq!(response.results.len(), 2);
        assert_eq!(response.results[0].index, 0);
        assert!(response.results[0].recipe.is_none());
        assert_eq!(response.results[1].index, 1);
        assert!(response.results[1].recipe.is_some());
        value["heldout_labels"] = json!([0, 1]);
        assert!(validate_bytes(&serde_json::to_vec(&value)?).is_err());
        assert!(validate_bytes(&vec![b' '; MAX_INPUT + 1]).is_err());
        Ok(())
    }
}
