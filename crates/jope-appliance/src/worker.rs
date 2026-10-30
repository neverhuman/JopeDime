//! Hidden process-isolation entry. Concrete discovery evaluation is supplied by DSC-08.

pub fn run(protocol: u32) -> anyhow::Result<()> {
    use std::io::IsTerminal;
    if protocol != 1 || std::io::stdin().is_terminal() {
        return Err(jope_runtime::Failure::input(
            "__worker requires --protocol 1 and piped stdin",
        ));
    }
    // Preserve the appliance's authenticated-package boundary before processing any candidate.
    let _package = crate::bootstrap::open()?;
    jope_service::worker::serve(protocol, jope_service::discover::r#loop::evaluate_native)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    #[test]
    fn hidden_command_accepts_the_exact_worker_protocol_launch() {
        let parsed =
            crate::cli::Cli::try_parse_from(["jopedime", "__worker", "--protocol", "1"]).unwrap();
        assert!(matches!(
            parsed.command,
            crate::cli::Command::Worker { protocol: 1 }
        ));
        let help = crate::cli::Cli::command().render_help().to_string();
        assert!(!help.contains("__worker"));
    }

    #[test]
    fn unsupported_protocol_is_an_input_error_before_starting_threads() {
        let error = super::run(2).unwrap_err();
        assert_eq!(
            jope_runtime::error::kind_of(&error),
            jope_runtime::FailureKind::Input
        );
    }

    #[test]
    fn wired_native_callback_scores_the_real_train_fixture_without_a_placeholder() {
        let raw = include_bytes!("../../../test-support/datasets/v1/d2-regression/train.csv");
        let mut names: Vec<_> = (0..21).map(|j| format!("x{j}")).collect();
        names.push("target".into());
        let mut csv = names.join(",").into_bytes();
        csv.push(b'\n');
        csv.extend(raw);
        let mut table =
            jope_runtime::table::parse_csv(&csv, &jope_runtime::limits::Limits::default()).unwrap();
        table.rows.truncate(20);
        let bytes = jope_service::dataset::jmat::encode(&table).unwrap();
        let hash = jope_runtime::sha256_hex(&bytes);
        let matrix = jope_service::dataset::jmat::Matrix::open(bytes).unwrap();
        let request = jope_service::worker::Request {
            attempt: jope_service::worker::Attempt {
                run: jope_api::ids::RunId::new(),
                round: 0,
                slot: 0,
                attempt: 1,
            },
            matrix_path: "unused-callback-input".into(),
            fold_spec: serde_json::json!({"version":1,"matrix_sha256":hash,"target":"target","features":names[..21],"folds":[
                {"repeat":0,"train":(0..10).collect::<Vec<_>>(),"validation":(10..20).collect::<Vec<_>>()},
                {"repeat":0,"train":(10..20).collect::<Vec<_>>(),"validation":(0..10).collect::<Vec<_>>()}
            ]}),
            dsl: "(pipe X (ridge))".into(),
            objective: serde_json::json!("rmse"),
            seed: 73,
            limits: jope_service::worker::Limits {
                wall_ms: 2000,
                rss_bytes: 64 * 1024 * 1024,
            },
        };
        let scores = jope_service::discover::r#loop::evaluate_native(&request, &matrix).unwrap();
        assert_eq!(scores.values.len(), 3);
        assert!(scores.values.iter().all(|s| s.is_finite()));
        assert!(scores.values[2] > 0.0);
        assert_eq!(scores.predictions_digest.len(), 64);
    }
}
