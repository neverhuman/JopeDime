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
    jope_service::worker::serve(protocol, |_request, _matrix| {
        // Never manufacture a score while the discovery/native evaluator adapter is absent.
        Err(jope_service::worker::WorkerFailure::Unavailable)
    })?;
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
}
