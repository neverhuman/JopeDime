use crate::cli::DatasetCommand;
use jope_api::ids::DatasetId;
use jope_client::{files, Request, Transport};
use serde_json::Value;
use std::{fs::File, str::FromStr};

pub(crate) use super::client as resolve_client;

pub(crate) fn report(result: anyhow::Result<()>) -> std::process::ExitCode {
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("jopedime: {error:#}");
            let code = error
                .chain()
                .find_map(|cause| {
                    cause
                        .downcast_ref::<jope_client::Error>()
                        .and_then(|e| match e {
                            jope_client::Error::Problem(problem) => {
                                super::common::error_exit_code(problem.code)
                            }
                            jope_client::Error::Http(_) => Some(9),
                            _ => None,
                        })
                })
                .unwrap_or(7);
            std::process::ExitCode::from(code)
        }
    }
}

pub(crate) fn emit(value: &Value, json: bool) -> anyhow::Result<()> {
    if json {
        super::common::write_receipt(&mut std::io::stdout().lock(), value)?;
    } else {
        print!("{}", human_output(value)?);
    }
    Ok(())
}

fn human_output(value: &Value) -> anyhow::Result<String> {
    let mut output = String::new();
    if let Some(fields) = value.as_object() {
        for (key, value) in fields {
            if let Some(items) = value.as_array() {
                output.push_str(&format!("{key}: {} item(s)\n", items.len()));
                for item in items {
                    let id = item.get("dataset_id").or_else(|| item.get("id"));
                    let name = item.get("name");
                    if let Some(id) = id {
                        output.push_str(&format!("  {}", id.as_str().unwrap_or("?")));
                        if let Some(name) = name.and_then(Value::as_str) {
                            output.push_str(&format!("  {name}"));
                        }
                        output.push('\n');
                    }
                }
            } else if value.is_object() {
                output.push_str(&format!("{key}: {}\n", serde_json::to_string(value)?));
            } else {
                output.push_str(&format!(
                    "{key}: {}\n",
                    value
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| value.to_string())
                ));
            }
        }
    } else {
        output.push_str(&format!("{}\n", serde_json::to_string(value)?));
    }
    Ok(output)
}

pub(crate) fn run(command: DatasetCommand) -> anyhow::Result<()> {
    match command {
        DatasetCommand::Add {
            path,
            name,
            idempotency_key,
            json: json_out,
        } => {
            let client = resolve_client()?;
            let name = name
                .or_else(|| path.file_name().map(|v| v.to_string_lossy().into_owned()))
                .ok_or_else(|| anyhow::anyhow!("dataset name is required"))?;
            let key = super::common::resolve_idempotency_key(idempotency_key.as_deref());
            if idempotency_key.is_none() {
                super::common::write_idempotency_key(&mut std::io::stderr().lock(), &key)?;
            }
            let response = files::upload_csv(&client, &name, File::open(&path)?, &key)?;
            let mut value: Value = response.json()?;
            if let Some(object) = value.as_object_mut() {
                super::common::add_idempotency_key(object, &key);
            }
            emit(&value, json_out)
        }
        DatasetCommand::List { json: json_out } => {
            let value =
                jope_client::Transport::execute(&resolve_client()?, Request::new(3, 0, &[])?)?
                    .json()?;
            emit(&value, json_out)
        }
        DatasetCommand::Show { id, json: json_out } => {
            let id = DatasetId::from_str(&id).map_err(|_| anyhow::anyhow!("invalid dataset id"))?;
            let value = resolve_client()?.dataset_inspect(&id)?;
            emit(&serde_json::to_value(value)?, json_out)
        }
        DatasetCommand::Delete {
            id,
            idempotency_key,
            json: json_out,
        } => {
            let id = DatasetId::from_str(&id).map_err(|_| anyhow::anyhow!("invalid dataset id"))?;
            let client = resolve_client()?;
            let generated = idempotency_key.is_none();
            let key = super::common::resolve_idempotency_key(idempotency_key.as_deref());
            if generated {
                super::common::write_idempotency_key(&mut std::io::stderr().lock(), &key)?;
            }
            let mut value = serde_json::to_value(client.dataset_delete(&id, &key)?)?;
            if let Some(object) = value.as_object_mut() {
                super::common::add_idempotency_key(object, &key);
            }
            emit(&value, json_out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dataset_id_parser_rejects_arbitrary_strings() {
        assert!(DatasetId::from_str("not-a-dataset").is_err());
    }

    #[test]
    fn human_dataset_output_has_stable_golden_lines() {
        assert_eq!(
            super::human_output(&serde_json::json!({"dataset_id":"ds_01","name":"sample"}))
                .unwrap(),
            "dataset_id: ds_01\nname: sample\n"
        );
    }
}
