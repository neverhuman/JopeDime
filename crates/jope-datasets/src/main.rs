//! `jope-datasets`: ingest, fetch, verify and link content-addressed datasets.

use clap::Parser;
use std::process::ExitCode;

fn main() -> ExitCode {
    match jope_datasets::cli::run(jope_datasets::cli::Cli::parse()) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datasets_cli_is_parsed_as_a_typed_command() {
        let cli = jope_datasets::cli::Cli::try_parse_from(["jope-datasets", "verify", "--quick"])
            .expect("valid datasets command should parse");
        assert!(matches!(
            cli.cmd,
            jope_datasets::cli::DatasetsCommand::Verify {
                quick: true,
                catalog: false,
                repo: None,
                registry: None,
            }
        ));
        assert!(jope_datasets::cli::Cli::try_parse_from(["jope-datasets", "unknown"]).is_err());
    }
}
