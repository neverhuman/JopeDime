//! Multi-role `carbon` CLI and AWS Lambda custom-runtime entrypoint (`version`, `self-verify`,
//! `doctor`, `quickstart`, `register`, `login`, `whoami`, `logout`, `ping`, `run`, `bench`,
//! `redteam`, `token-mint`, `plan`, `spot-check`, plus feature-gated operator/server subcommands).

mod carbon_cmd;

use std::env;
use std::io::IsTerminal as _;
#[cfg(feature = "engine")]
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser as _;
use jope_carbon::observability::{
    emit_cli_error, init_cli_observability, install_sigint_flag, write_stdout_line,
    CarbonColorMode, CarbonLogFormat,
};

#[cfg(feature = "eval")]
use carbon_cmd::run_eval_runtime;
use carbon_cmd::{
    args::{Cli, Command},
    auth_doctor::{dispatch_auth_doctor, is_auth_doctor_cmd},
    model_ops::{dispatch_model_ops, execute_positional_csv_shorthand, is_model_ops_cmd},
    ops_admin::dispatch_ops_admin,
    render_interactive_quickstart_banner,
};
#[cfg(feature = "engine")]
use carbon_cmd::{resolve_lambda_runtime_api, run_engine_runtime};

fn main() -> ExitCode {
    const CLI_MAIN_STACK_BYTES: usize = 16 * 1024 * 1024;
    match std::thread::Builder::new()
        .name("carbon-cli".to_owned())
        .stack_size(CLI_MAIN_STACK_BYTES)
        .spawn(run_cli_main)
    {
        Ok(handle) => match handle.join() {
            Ok(code) => code,
            Err(_) => ExitCode::from(101),
        },
        Err(_) => run_cli_main(),
    }
}

#[inline(never)]
fn run_cli_main() -> ExitCode {
    if let Ok(exe) = env::current_exe() {
        jope_carbon::cli::model_store::sweep_stale_windows_upgrade_backups(&exe);
    }
    let cli = Cli::parse();
    if cli.csv_file.is_some() && cli.command.is_some() {
        eprintln!("error: positional <CSV_FILE> cannot be combined with a subcommand");
        return ExitCode::from(2);
    }
    let log_format = match cli.log_format.trim().to_ascii_lowercase().as_str() {
        "text" | "" => CarbonLogFormat::Text,
        "json" => CarbonLogFormat::Json,
        other => {
            eprintln!("error: unsupported --log-format `{other}` (expected `text` or `json`)");
            return ExitCode::from(2);
        }
    };
    let color_mode = match cli.color.trim().to_ascii_lowercase().as_str() {
        "auto" | "" => CarbonColorMode::Auto,
        "always" => CarbonColorMode::Always,
        "never" => CarbonColorMode::Never,
        other => {
            eprintln!(
                "error: unsupported --color `{other}` (expected `auto`, `always`, or `never`)"
            );
            return ExitCode::from(2);
        }
    };
    init_cli_observability(cli.verbose, cli.quiet, log_format, color_mode);
    let skip_sigint_hook = {
        #[allow(unused_mut)]
        let mut skip = false;
        #[cfg(feature = "serve")]
        if matches!(&cli.command, Some(Command::Serve { .. })) {
            skip = true;
        }
        #[cfg(feature = "emu")]
        if matches!(
            &cli.command,
            Some(Command::Emu {
                self_test: false,
                listen: Some(_),
                ..
            })
        ) {
            skip = true;
        }
        skip
    };
    let _sigint_flag = (!skip_sigint_hook).then(install_sigint_flag);
    let global_quiet = cli.quiet;
    let global_ephemeral = cli.ephemeral;
    let keep_shorthand = cli.keep;

    #[cfg(feature = "engine")]
    if cli.csv_file.is_none() && cli.command.is_none() {
        let (rt_res, is_aws_prod) = resolve_lambda_runtime_api();
        if let Ok(runtime_api) = rt_res {
            let work_dir = match env::var("CARBON_WORK_DIR") {
                Ok(val) if !val.trim().is_empty() => PathBuf::from(val),
                _ => env::temp_dir().join("carbon-lambda"),
            };
            let role = env::var("CARBON_ROLE").unwrap_or_default();
            return match role.trim() {
                "engine" => run_engine_runtime(&runtime_api, work_dir, is_aws_prod),
                #[cfg(feature = "eval")]
                "eval" => run_eval_runtime(&runtime_api, work_dir),
                other => {
                    eprintln!(
                        "error: CARBON_ROLE='{other}' unsupported for `carbon` bootstrap (expected 'engine' or 'eval')"
                    );
                    ExitCode::from(2)
                }
            };
        }
    }

    if let Err(tls_err) = jope_carbon::tls::validate_tls_env_config() {
        return emit_cli_error(false, &tls_err);
    }

    if let Some(csv_path) = cli.csv_file {
        return execute_positional_csv_shorthand(csv_path, keep_shorthand, global_quiet);
    }

    if cli.command.is_none() && std::io::stdout().is_terminal() && std::io::stderr().is_terminal() {
        let _ = write_stdout_line(&render_interactive_quickstart_banner());
        return ExitCode::SUCCESS;
    }

    let cmd = cli.command.unwrap_or(Command::Version);
    if is_auth_doctor_cmd(&cmd) {
        return dispatch_auth_doctor(cmd);
    }
    if is_model_ops_cmd(&cmd) {
        return dispatch_model_ops(cmd, global_quiet, global_ephemeral);
    }
    dispatch_ops_admin(cmd, global_quiet)
}
