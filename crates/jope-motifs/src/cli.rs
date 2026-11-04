//! `jope motifs …`: the same `Cli`/`exec`/`run` shape as the other subcommand crates,
//! so the umbrella can delegate to it in process.
//!
//! `mine` reads the qualified records of a learning State and writes one new private
//! directory holding the fragment table and the list of every input it examined. It writes
//! nothing when no program was admitted. `validate` checks a table and, with `--state`,
//! confirms the table's recorded sources against a fresh read of the State.
//!
//! Standard output is operator-private and never committed, but it is still shipped text: it
//! passes the shipped-text lint and the owner ban, and it holds no name, id, expansion, host,
//! provenance value or family label. Errors print only a fixed code; full error chains go to
//! the optional private log.

use crate::export::{self, PublicChecks, PublicExport, StageCounts};
use crate::mine::{input_manifest, mine};
use crate::private_io::{create_private_dir, write_private_file, write_public_file, Policy};
use crate::source::{FamilyMap, QualifiedSet, QualifiedSource};
use crate::table::{Lifecycle, MineParameters, MotifTable};
use crate::teaching::{StateOpener, StateTeaching};
use crate::validate::{self, ValidateOptions, Validation};
use crate::{code_of, Code, Coded};
use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;
use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Schema label of the `validate --json` document.
pub const VALIDATE_SCHEMA: &str = "jope.motifs-validate/v1";
/// Schema label of the `mine --json` document.
pub const MINE_SCHEMA: &str = "jope.motifs-mine/v1";
/// The fragment table inside a `mine --out` directory.
pub const TABLE_FILE: &str = "table.json";
/// The list of every examined input inside a `mine --out` directory.
pub const INPUTS_FILE: &str = "inputs.json";

/// Every fixed error code. None carries program text, paths, names or digests.
pub const ERROR_CODES: [&str; 31] = [
    "state_unavailable",
    "input_set_changed",
    "input_join_failed",
    "record_unqualified",
    "record_unauthenticated",
    "label_burn_missing",
    "label_origin_refused",
    "no_admitted_programs",
    "noncanonical_program",
    "recipe_hash_mismatch",
    "invalid_parameters",
    "invalid_family_map",
    "output_path_refused",
    "table_unreadable",
    "table_schema",
    "table_digest_mismatch",
    "engine_drift",
    "fragment_id_mismatch",
    "fragment_name_mismatch",
    "expansion_not_canonical",
    "nested_reference",
    "reference_position",
    "unknown_reference",
    "signature_drift",
    "provenance_inconsistent",
    "support_mismatch",
    "table_order",
    "promotion_evidence_missing",
    "rewrite_mismatch",
    "public_export_leak",
    "io",
];

/// Every check failure code, by check class. Each class has a `..._not_exercised` code: a
/// check that examined nothing fails with it and is never reported as a pass. The `source`
/// class is the table-wide source check of `validate --state`; its failure lines name no
/// fragment.
pub const CHECK_CODES: [(&str, &str); 17] = [
    ("round_trip", "canonical_mismatch"),
    ("round_trip", "recipe_hash_mismatch"),
    ("round_trip", "unexpanded_accepted"),
    ("round_trip", "round_trip_not_exercised"),
    ("determinism", "fit_not_deterministic"),
    ("determinism", "determinism_not_exercised"),
    ("context_compatibility", "parse_failed"),
    ("context_compatibility", "not_idempotent"),
    ("context_compatibility", "slot_refused"),
    ("context_compatibility", "grammar_refused"),
    ("context_compatibility", "context_not_fitted"),
    ("context_compatibility", "context_not_independent"),
    ("context_compatibility", "context_not_exercised"),
    ("name_absence", "name_in_emission"),
    ("name_absence", "name_absence_not_exercised"),
    ("source", "source_not_confirmed"),
    ("source", "source_not_exercised"),
];

#[derive(Parser)]
#[command(
    name = "jope-motifs",
    version,
    about = "Reusable program fragments kept in a private table, with exact expansion checks."
)]
/// Arguments for the private `jope motifs` command group.
pub struct Cli {
    #[command(subcommand)]
    /// The requested mining or validation operation.
    pub cmd: MotifsCommand,
}

#[derive(Subcommand)]
/// Supported private motif commands.
pub enum MotifsCommand {
    /// Build a new private fragment table.
    Mine {
        /// State root; otherwise the resolved default.
        #[arg(long)]
        state: Option<PathBuf>,
        /// New private output directory: absolute, outside temporary and version-controlled paths.
        #[arg(long)]
        out: PathBuf,
        /// Keep a fragment only when it occurs in at least this many distinct datasets.
        #[arg(long, default_value_t = 2)]
        min_datasets: u32,
        /// With --families, keep a fragment only when it occurs in at least this many labelled families.
        #[arg(long, default_value_t = 1)]
        min_families: u32,
        /// Longest fragment, in stages (2 to 6).
        #[arg(long, default_value_t = 6)]
        max_stages: u32,
        /// Private JSON map from dataset digest to family label.
        #[arg(long)]
        families: Option<PathBuf>,
        /// New private file for full error chains.
        #[arg(long)]
        private_log: Option<PathBuf>,
        /// Print one JSON document.
        #[arg(long)]
        json: bool,
    },
    /// Check a private fragment table: exact rewrites, compatible contexts and absent reference names.
    Validate {
        /// Private fragment table to check.
        #[arg(long)]
        table: PathBuf,
        /// Recorded programs fitted per fragment, in provenance order.
        #[arg(long, default_value_t = 3)]
        hosts_per_fragment: u32,
        /// State root used to confirm the table's source records are still current.
        #[arg(long)]
        state: Option<PathBuf>,
        /// New private file for the per-check report.
        #[arg(long)]
        report: Option<PathBuf>,
        /// New file for the shareable summary, written only when every check passes.
        #[arg(long)]
        public_out: Option<PathBuf>,
        /// New private file for full error chains.
        #[arg(long)]
        private_log: Option<PathBuf>,
        /// Print one JSON document.
        #[arg(long)]
        json: bool,
    },
}

/// Compatibility name for the motif command enum.
pub type Cmd = MotifsCommand;

/// Where outputs may be written, and how a State is opened.
pub(crate) struct Env {
    pub(crate) policy: Policy,
    pub(crate) open_state: StateOpener,
}

impl Env {
    /// The standard policy, with the product's resolved state root
    /// (`jope_paths::jope_state`): no private or shareable output may be written under it, nor
    /// under a root named with `--state`. The State is opened by [`standard_opener`].
    pub(crate) fn standard() -> Self {
        Self {
            policy: Policy::standard(vec![jope_paths::jope_state()]),
            open_state: standard_opener(),
        }
    }

    /// The policy of one run: the standard state roots plus an explicit `--state`, made
    /// absolute so that a relative spelling is refused too.
    fn policy_for(&self, state: Option<&Path>) -> Policy {
        match state {
            Some(state) => {
                let root = std::path::absolute(state).unwrap_or_else(|_| state.to_path_buf());
                self.policy.clone().with_state_root(root)
            }
            None => self.policy.clone(),
        }
    }
}

/// The standard State opener: the learning crate's eligibility API (`learning::open`), which
/// refuses a root that holds no State (missing, or without the HWM chain tip) with
/// `state_unavailable`.
fn standard_opener() -> StateOpener {
    Box::new(crate::learning::open)
}

/// Run a command and map failures to the tool's fixed stderr message and exit status.
pub fn dispatch(cli: Cli) -> ExitCode {
    match run(cli) {
        Ok(code) => code,
        Err(_) => {
            eprintln!("error: io");
            ExitCode::from(2)
        }
    }
}

/// Run a command with the standard state opener and process streams.
pub fn run(cli: Cli) -> Result<ExitCode> {
    let stdout = std::io::stdout();
    let stderr = std::io::stderr();
    let status = run_with(
        cli,
        &Env::standard(),
        &mut stdout.lock(),
        &mut stderr.lock(),
    )?;
    Ok(ExitCode::from(status))
}

/// Run one verb with the given environment and streams. Returns the exit status: 0 when
/// everything passed, 1 when a check failed, 2 on error. An `Err` means a stream write failed.
pub(crate) fn run_with(
    cli: Cli,
    env: &Env,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<u8> {
    match cli.cmd {
        MotifsCommand::Mine {
            state,
            out: directory,
            min_datasets,
            min_families,
            max_stages,
            families,
            private_log,
            json,
        } => {
            let policy = env.policy_for(state.as_deref());
            let args = MineArgs {
                state,
                directory,
                min_datasets,
                min_families,
                max_stages,
                families,
                private_log,
            };
            match mine_verb(&args, env, &policy) {
                Ok(table) => {
                    render_mine(&table, json, out)?;
                    Ok(0)
                }
                Err(error) => {
                    let log = args.private_log.as_deref();
                    fail(&error, MINE_SCHEMA, json, &policy, log, out, err)
                }
            }
        }
        MotifsCommand::Validate {
            table,
            hosts_per_fragment,
            state,
            report,
            public_out,
            private_log,
            json,
        } => {
            let policy = env.policy_for(state.as_deref());
            let args = ValidateArgs {
                table,
                hosts_per_fragment,
                state,
                report,
                public_out,
                private_log,
            };
            match validate_verb(&args, env, &policy) {
                Ok(outcome) => {
                    render_validate(&outcome, json, out)?;
                    Ok(if outcome.validation.passed() { 0 } else { 1 })
                }
                Err(error) => {
                    let log = args.private_log.as_deref();
                    fail(&error, VALIDATE_SCHEMA, json, &policy, log, out, err)
                }
            }
        }
    }
}

#[derive(Serialize)]
struct FailureJson {
    schema: &'static str,
    ok: bool,
    error: &'static str,
}

/// Report an error: its fixed code on stderr, or the failure document on stdout with `--json`.
/// The full chain goes only to the private log. A refused or failing log never changes the
/// code, but it is reported. Returns exit status 2.
fn fail(
    error: &anyhow::Error,
    schema: &'static str,
    json: bool,
    policy: &Policy,
    private_log: Option<&Path>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<u8> {
    let code = code_of(error).unwrap_or("io");
    let log_refused = private_log.is_some_and(|log| {
        let chain = format!("{error:?}\n");
        write_private_file(policy, log, chain.as_bytes()).is_err()
    });
    if json {
        let failure = FailureJson {
            schema,
            ok: false,
            error: code,
        };
        writeln!(out, "{}", serde_json::to_string(&failure)?)?;
    } else {
        writeln!(err, "error: {code}")?;
    }
    if log_refused {
        writeln!(err, "warning: private log not written")?;
    }
    Ok(2)
}

/// Open the State and read its qualified programs through the eligible-State reader.
fn read_state(env: &Env, state: Option<&Path>) -> Result<QualifiedSet> {
    let reader = (env.open_state)(state).coded("state_unavailable")?;
    StateTeaching::new(reader.as_ref()).qualified()
}

mod mining_command;
mod validate_command;
use mining_command::{mine_verb, render_mine, MineArgs};
use validate_command::{render_validate, validate_verb, ValidateArgs};

#[cfg(test)]
mod tests;
