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

struct MineArgs {
    state: Option<PathBuf>,
    directory: PathBuf,
    min_datasets: u32,
    min_families: u32,
    max_stages: u32,
    families: Option<PathBuf>,
    private_log: Option<PathBuf>,
}

fn mine_verb(args: &MineArgs, env: &Env, policy: &Policy) -> Result<MotifTable> {
    // Every output path and every parameter is checked before the State is read, so a
    // refusal costs nothing.
    policy.check_private(&args.directory)?;
    if let Some(log) = &args.private_log {
        policy.check_private(log)?;
    }
    let families = match &args.families {
        Some(path) => Some(FamilyMap::parse(
            &std::fs::read(path).coded("invalid_family_map")?,
        )?),
        None => None,
    };
    let parameters = MineParameters {
        min_datasets: args.min_datasets,
        min_families: args.min_families,
        max_stages: args.max_stages,
        family_map_sha256: families.as_ref().map(|map| map.sha256().to_owned()),
    };
    parameters.check()?;
    let set = read_state(env, args.state.as_deref())?;
    // Nothing admitted means nothing was examined: no table is written.
    ensure!(!set.programs().is_empty(), Code("no_admitted_programs"));
    let table = mine(&set, families.as_ref(), parameters)?;
    let inputs = input_manifest(&set, &table)?;
    let table_bytes = table.to_json().coded("io")?;
    let input_bytes = serde_json::to_vec_pretty(&inputs).coded("io")?;
    write_private_bundle(policy, &args.directory, &table_bytes, &input_bytes)?;
    Ok(table)
}

/// Create the new 0700 directory and write the table and its input list into it, 0600 each.
/// If a write fails, the files of this run and then the directory are removed, so a failed
/// run never leaves a table without its inputs.
fn write_private_bundle(
    policy: &Policy,
    directory: &Path,
    table: &[u8],
    inputs: &[u8],
) -> Result<()> {
    let created = create_private_dir(policy, directory)?;
    let table_path = created.join(TABLE_FILE);
    let inputs_path = created.join(INPUTS_FILE);
    let written = write_private_file(policy, &table_path, table)
        .and_then(|()| write_private_file(policy, &inputs_path, inputs));
    if written.is_err() {
        // Best effort: a failed removal cannot be reported more usefully than the write error.
        let _ = std::fs::remove_file(&table_path);
        let _ = std::fs::remove_file(&inputs_path);
        let _ = std::fs::remove_dir(&created);
    }
    written
}

#[derive(Serialize)]
struct MineJson<'a> {
    schema: &'static str,
    ok: bool,
    table_sha256: &'a str,
    fragments: u32,
    quarantined: u32,
    stage_counts: StageCounts,
    parameters: ParametersJson,
    inputs: InputsJson,
}

#[derive(Serialize)]
struct ParametersJson {
    min_datasets: u32,
    min_families: u32,
    max_stages: u32,
    family_map: bool,
}

#[derive(Serialize)]
struct InputsJson {
    qualified: u32,
    used: u32,
    excluded_adaptive: u32,
    excluded_at_source: u32,
    skipped_form: u32,
    skipped_reserved: u32,
    skipped_slots: u32,
    skipped_unsupported: u32,
    dropped_untyped: u32,
}

fn render_mine(table: &MotifTable, json: bool, out: &mut dyn Write) -> Result<()> {
    let body = &table.body;
    let fragments = u32::try_from(body.motifs.len())?;
    let quarantined = u32::try_from(
        body.motifs
            .iter()
            .filter(|motif| motif.lifecycle == Lifecycle::Quarantined)
            .count(),
    )?;
    if json {
        let parameters = &body.parameters;
        let inputs = &body.inputs;
        let document = MineJson {
            schema: MINE_SCHEMA,
            ok: true,
            table_sha256: &table.table_sha256,
            fragments,
            quarantined,
            stage_counts: StageCounts::of(table),
            parameters: ParametersJson {
                min_datasets: parameters.min_datasets,
                min_families: parameters.min_families,
                max_stages: parameters.max_stages,
                family_map: parameters.family_map_sha256.is_some(),
            },
            inputs: InputsJson {
                qualified: inputs.qualified,
                used: inputs.used,
                excluded_adaptive: inputs.excluded_adaptive,
                excluded_at_source: body.source.excluded_at_source,
                skipped_form: inputs.skipped_form,
                skipped_reserved: inputs.skipped_reserved,
                skipped_slots: inputs.skipped_slots,
                skipped_unsupported: inputs.skipped_unsupported,
                dropped_untyped: inputs.dropped_untyped,
            },
        };
        writeln!(out, "{}", serde_json::to_string(&document)?)?;
    } else if fragments == 0 {
        // Nothing was kept, so nothing is quarantined either; the inputs are still listed.
        writeln!(
            out,
            "Wrote a private table with no fragments; every examined input is listed."
        )?;
    } else if quarantined == fragments {
        writeln!(
            out,
            "Wrote a private table with {fragments} fragments (all quarantined)."
        )?;
    } else {
        writeln!(
            out,
            "Wrote a private table with {fragments} fragments ({quarantined} quarantined)."
        )?;
    }
    Ok(())
}

struct ValidateArgs {
    table: PathBuf,
    hosts_per_fragment: u32,
    state: Option<PathBuf>,
    report: Option<PathBuf>,
    public_out: Option<PathBuf>,
    private_log: Option<PathBuf>,
}

struct Outcome {
    validation: Validation,
    /// The source check's failure code (`validate::verify_source`); `None` when the source was
    /// not read or was confirmed.
    source_failure: Option<&'static str>,
    public_written: bool,
}

fn validate_verb(args: &ValidateArgs, env: &Env, policy: &Policy) -> Result<Outcome> {
    // Every output path is checked before any work, so a refused path costs nothing.
    for path in [&args.private_log, &args.report].into_iter().flatten() {
        policy.check_private(path)?;
    }
    if let Some(path) = &args.public_out {
        policy.check_public(path)?;
    }
    ensure!(args.hosts_per_fragment >= 1, Code("invalid_parameters"));
    let bytes = std::fs::read(&args.table).coded("table_unreadable")?;
    let table = MotifTable::load(&bytes)?;
    // V0 first, so a table that fails integrity never costs a State read.
    table.verify()?;
    let fresh = args
        .state
        .as_deref()
        .map(|state| read_state(env, Some(state)))
        .transpose()?;
    let options = ValidateOptions {
        hosts_per_fragment: args.hosts_per_fragment,
    };
    let mut validation = validate::validate(&table, &options)?;
    let source_failure = fresh
        .as_ref()
        .and_then(|set| validate::verify_source(&mut validation, &table, set));
    if let Some(path) = &args.report {
        let report = serde_json::to_vec_pretty(&validation).coded("io")?;
        write_private_file(policy, path, &report)?;
    }
    let mut public_written = false;
    if let Some(path) = &args.public_out {
        if validation.passed() {
            let summary = PublicExport::from_validation(&validation, &table);
            let text = serde_json::to_string(&summary).coded("io")?;
            export::scan(&text, &table)?;
            write_public_file(policy, path, text.as_bytes())?;
            public_written = true;
        }
    }
    Ok(Outcome {
        validation,
        source_failure,
        public_written,
    })
}

#[derive(Serialize)]
struct FailureJson {
    schema: &'static str,
    ok: bool,
    error: &'static str,
}

#[derive(Serialize)]
struct FailureLine {
    /// The fragment's index; absent for the table-wide source check.
    #[serde(skip_serializing_if = "Option::is_none")]
    fragment: Option<u32>,
    check: &'static str,
    code: String,
}

#[derive(Serialize)]
struct ValidateJson<'a> {
    schema: &'static str,
    ok: bool,
    table_sha256: &'a str,
    fragments: u32,
    passed: u32,
    failed: u32,
    checks: PublicChecks,
    not_exercised: u32,
    source_verified: bool,
    failures: Vec<FailureLine>,
    promotion: &'static str,
    public_written: bool,
}

fn render_validate(outcome: &Outcome, json: bool, out: &mut dyn Write) -> Result<()> {
    let validation = &outcome.validation;
    let fragments = u32::try_from(validation.motifs.len())?;
    let passed = u32::try_from(validation.motifs.iter().filter(|m| m.passed()).count())?;
    let failed = fragments.saturating_sub(passed);
    if json {
        // Every failed run names why: a failed source check is a line of its own.
        let mut lines = BTreeSet::new();
        if let Some(code) = outcome.source_failure {
            lines.insert((None, "source", code.to_owned()));
        }
        for result in &validation.motifs {
            for (check, checked) in [
                ("round_trip", &result.round_trip),
                ("determinism", &result.determinism),
                ("context_compatibility", &result.context_compatibility),
                ("name_absence", &result.name_absence),
            ] {
                for failure in &checked.failures {
                    lines.insert((Some(result.index), check, failure.code.clone()));
                }
            }
        }
        let document = ValidateJson {
            schema: VALIDATE_SCHEMA,
            ok: validation.passed(),
            table_sha256: &validation.table_sha256,
            fragments,
            passed,
            failed,
            checks: PublicChecks::from_results(&validation.motifs),
            not_exercised: validation.motifs.iter().map(|m| m.not_exercised).sum(),
            source_verified: validation.source_verified,
            failures: lines
                .into_iter()
                .map(|(fragment, check, code)| FailureLine {
                    fragment,
                    check,
                    code,
                })
                .collect(),
            promotion: "required_not_run",
            public_written: outcome.public_written,
        };
        writeln!(out, "{}", serde_json::to_string(&document)?)?;
    } else {
        if fragments == 0 {
            writeln!(
                out,
                "Checked 0 fragments: nothing was examined, so the table does not pass. \
                 Promotion evidence: not run."
            )?;
        } else {
            writeln!(
                out,
                "Checked {fragments} fragments: {passed} passed, {failed} failed. \
                 Promotion evidence: not run."
            )?;
        }
        if validation.source_digest_equal.is_some() {
            let state = if validation.source_verified {
                "confirmed"
            } else {
                "not confirmed"
            };
            writeln!(out, "Source records: {state}.")?;
        }
        if outcome.public_written {
            writeln!(out, "Wrote the shareable summary.")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
