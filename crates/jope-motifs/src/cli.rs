//! `jope motifs …`: the same `Cli`/`Cmd`/`exec`/`run` shape as the other subcommand crates,
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
    pub cmd: Cmd,
}

#[derive(Subcommand)]
/// Supported private motif commands.
pub enum Cmd {
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
pub fn exec(cli: Cli) -> ExitCode {
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
        Cmd::Mine {
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
        Cmd::Validate {
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
mod tests {
    use super::*;
    use crate::export::OWNER_BANNED;
    use crate::fixture::{self, FakeState};
    use crate::mine::InputManifest;
    use crate::source::TaskLabel;
    use crate::teaching::{no_reader, EligibleState};
    use clap::CommandFactory;
    use feat_math::transpile::lint_shipped_text;
    use serde_json::Value;

    struct Run {
        status: u8,
        out: String,
        err: String,
    }

    fn env_with(policy: Policy, open_state: StateOpener) -> Env {
        Env { policy, open_state }
    }

    /// An opener that serves a copy of `state` for any root.
    fn opener(state: FakeState) -> StateOpener {
        Box::new(move |_: Option<&Path>| -> Result<Box<dyn EligibleState>> {
            Ok(Box::new(state.clone()))
        })
    }

    /// An opener that must never run: every refusal it guards happens before the State is read.
    fn unopened() -> StateOpener {
        Box::new(|_: Option<&Path>| -> Result<Box<dyn EligibleState>> {
            panic!("the State is read only after every path and parameter check")
        })
    }

    /// The test policy with a State in memory.
    fn state_env(state: FakeState) -> Env {
        env_with(Policy::for_test(Vec::new(), false), opener(state))
    }

    fn invoke(arguments: &[&str]) -> Run {
        let env = env_with(Policy::for_test(Vec::new(), false), Box::new(no_reader));
        invoke_in(&env, arguments)
    }

    fn invoke_in(env: &Env, arguments: &[&str]) -> Run {
        let cli = Cli::try_parse_from(arguments.iter().copied()).expect("arguments parse");
        let mut out = Vec::new();
        let mut err = Vec::new();
        let status = run_with(cli, env, &mut out, &mut err).expect("streams");
        Run {
            status,
            out: String::from_utf8(out).expect("utf-8"),
            err: String::from_utf8(err).expect("utf-8"),
        }
    }

    fn utf8(path: &Path) -> &str {
        path.to_str().expect("utf-8 path")
    }

    fn written(directory: &Path, name: &str, table: &MotifTable) -> String {
        let path = directory.join(name);
        std::fs::write(&path, table.to_json().expect("json")).expect("write table");
        path.to_str().expect("utf-8 path").to_owned()
    }

    fn read_table(directory: &Path) -> MotifTable {
        let bytes = std::fs::read(directory.join(TABLE_FILE)).expect("table written");
        MotifTable::load(&bytes).expect("table loads")
    }

    fn read_inputs(directory: &Path) -> InputManifest {
        let bytes = std::fs::read(directory.join(INPUTS_FILE)).expect("inputs written");
        serde_json::from_slice(&bytes).expect("inputs parse")
    }

    fn assert_clean(text: &str, table: &MotifTable) {
        assert!(lint_shipped_text(text).is_empty(), "{text}");
        let lower = text.to_lowercase();
        for word in OWNER_BANNED {
            assert!(!lower.contains(word), "{word}");
        }
        for sentinel in fixture::sentinels(table) {
            assert!(!lower.contains(&sentinel.to_lowercase()), "{sentinel}");
        }
        assert!(!text.contains("/home/") && !text.contains("/tmp/"));
    }

    #[test]
    fn help_for_every_subcommand_passes_shipped_text_lint_and_owner_ban() {
        let mut command = Cli::command();
        command.build();
        // `build` adds clap's own `help` subcommand; its text is still linted below.
        let verbs: Vec<&str> = command
            .get_subcommands()
            .map(|c| c.get_name())
            .filter(|name| *name != "help")
            .collect();
        assert_eq!(verbs, ["mine", "validate"]);
        let mut texts = vec![
            command.render_long_help().to_string(),
            command.render_help().to_string(),
        ];
        for subcommand in command.get_subcommands_mut() {
            texts.push(subcommand.render_long_help().to_string());
            texts.push(subcommand.render_help().to_string());
        }
        assert!(texts.len() >= 6);
        for flag in [
            "--hosts-per-fragment",
            "--state",
            "--out",
            "--min-datasets",
            "--min-families",
            "--max-stages",
            "--families",
        ] {
            assert!(texts.iter().any(|text| text.contains(flag)), "{flag}");
        }
        let table = fixture::labelled_table();
        fixture::assert_carried(&table);
        let sentinels = fixture::sentinels(&table);
        for text in &texts {
            assert!(lint_shipped_text(text).is_empty(), "{text}");
            let lower = text.to_lowercase();
            for word in OWNER_BANNED {
                assert!(!lower.contains(word), "{word} in {text}");
            }
            assert!(!text.contains(crate::NAME_PREFIX));
            for sentinel in &sentinels {
                assert!(!lower.contains(&sentinel.to_lowercase()), "{sentinel}");
            }
        }
    }

    #[test]
    fn json_stdout_passes_lint_and_holds_no_canary() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let table = fixture::labelled_small_table();
        fixture::assert_carried(&table);
        let path = written(directory.path(), "table.json", &table);
        let run = invoke(&["jope-motifs", "validate", "--table", &path, "--json"]);
        assert_eq!(run.status, 0, "{}{}", run.out, run.err);
        assert!(run.err.is_empty());
        assert_eq!(run.out.lines().count(), 1);
        let value: Value = serde_json::from_str(&run.out).expect("one JSON document");
        assert_eq!(value["schema"], VALIDATE_SCHEMA);
        assert_eq!(value["ok"], Value::Bool(true));
        assert_eq!(value["fragments"], 1);
        assert_eq!(value["promotion"], "required_not_run");
        assert_eq!(value["table_sha256"], table.table_sha256.as_str());
        assert_clean(&run.out, &table);
    }

    #[test]
    fn human_output_passes_lint_and_holds_no_canary() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let table = fixture::labelled_small_table();
        fixture::assert_carried(&table);
        let path = written(directory.path(), "table.json", &table);
        let run = invoke(&["jope-motifs", "validate", "--table", &path]);
        assert_eq!(run.status, 0, "{}{}", run.out, run.err);
        assert_eq!(
            run.out,
            "Checked 1 fragments: 1 passed, 0 failed. Promotion evidence: not run.\n"
        );
        assert_clean(&run.out, &table);
        assert!(!run.out.contains(&table.table_sha256));
    }

    #[test]
    fn error_codes_are_lint_clean_and_hold_no_reserved_prefix() {
        let codes = ERROR_CODES
            .iter()
            .chain(CHECK_CODES.iter().map(|(_, code)| code));
        for code in codes {
            assert!(lint_shipped_text(code).is_empty(), "{code}");
            assert!(!code.contains(crate::NAME_PREFIX), "{code}");
            assert!(!code.contains("teaching"), "{code}");
            assert!(!code.contains("motif"), "{code}");
            for word in OWNER_BANNED {
                assert!(!code.contains(word), "{code}");
            }
            assert!(
                code.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'),
                "{code}"
            );
        }
        let distinct: BTreeSet<&str> = ERROR_CODES.iter().copied().collect();
        assert_eq!(distinct.len(), ERROR_CODES.len());
        let mut pairs = CHECK_CODES.to_vec();
        pairs.sort_unstable();
        pairs.dedup();
        assert_eq!(pairs.len(), CHECK_CODES.len());
        for (check, _) in CHECK_CODES {
            assert!(lint_shipped_text(check).is_empty(), "{check}");
            let unexercised = if check == "context_compatibility" {
                String::from("context_not_exercised")
            } else {
                format!("{check}_not_exercised")
            };
            assert!(
                CHECK_CODES.contains(&(check, unexercised.as_str())),
                "{check}"
            );
        }
        let table = fixture::labelled_table();
        fixture::assert_carried(&table);
        let sentinels = fixture::sentinels(&table);
        let texts = ERROR_CODES
            .iter()
            .chain(CHECK_CODES.iter().map(|(check, _)| check))
            .chain(CHECK_CODES.iter().map(|(_, code)| code));
        for text in texts {
            for sentinel in &sentinels {
                assert!(!text.contains(sentinel.as_str()), "{sentinel} in {text}");
            }
        }
    }

    #[test]
    fn validate_exit_codes_are_0_1_2() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let passing = written(directory.path(), "passing.json", &fixture::small_table());
        let failing = written(directory.path(), "failing.json", &fixture::refused_table());
        let missing = directory.path().join("missing.json");
        let missing = missing.to_str().expect("utf-8 path");

        let run = invoke(&["jope-motifs", "validate", "--table", &passing]);
        assert_eq!(run.status, 0);
        let run = invoke(&["jope-motifs", "validate", "--table", &failing, "--json"]);
        assert_eq!(run.status, 1);
        let value: Value = serde_json::from_str(&run.out).expect("json");
        assert_eq!(value["ok"], Value::Bool(false));
        assert_eq!(value["failed"], 1);
        let failures = value["failures"].as_array().expect("failures");
        assert!(failures
            .iter()
            .any(|line| line["code"] == "context_not_fitted"
                && line["check"] == "context_compatibility"
                && line["fragment"] == 0));
        // Nothing fitted, so determinism is reported as not exercised, never as passed.
        assert!(failures
            .iter()
            .any(|line| line["code"] == "determinism_not_exercised"
                && line["check"] == "determinism"));
        assert_eq!(value["checks"]["determinism"]["passed"], 0);
        for line in failures {
            assert!(
                CHECK_CODES
                    .iter()
                    .any(|(check, code)| line["check"] == *check && line["code"] == *code),
                "{line}"
            );
        }
        let run = invoke(&["jope-motifs", "validate", "--table", missing]);
        assert_eq!(run.status, 2);
        assert_eq!(run.err, "error: table_unreadable\n");
        let run = invoke(&[
            "jope-motifs",
            "validate",
            "--table",
            &passing,
            "--hosts-per-fragment",
            "0",
        ]);
        assert_eq!(run.status, 2);
        assert_eq!(run.err, "error: invalid_parameters\n");
    }

    #[test]
    fn errors_print_fixed_codes_never_program_text() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let mut tampered = fixture::small_table();
        let key = tampered.body.hosts.keys().next().cloned().expect("a host");
        tampered
            .body
            .hosts
            .insert(key, fixture::canonical("(pipe X (standardize) (ridge))"));
        tampered.table_sha256 = tampered.body.digest().expect("digest");
        let path = written(directory.path(), "tampered.json", &tampered);
        let log = directory.path().join("log.txt");
        let log_path = log.to_str().expect("utf-8 path");

        let run = invoke(&[
            "jope-motifs",
            "validate",
            "--table",
            &path,
            "--private-log",
            log_path,
        ]);
        assert_eq!(run.status, 2);
        assert!(run.out.is_empty());
        assert_eq!(run.err, "error: provenance_inconsistent\n");
        assert_clean(&run.err, &tampered);
        let chain = std::fs::read_to_string(&log).expect("private log written");
        assert!(chain.contains("provenance_inconsistent"));

        // A log that cannot be written never changes the code, but the operator is told.
        let run = invoke(&[
            "jope-motifs",
            "validate",
            "--table",
            &path,
            "--private-log",
            log_path,
        ]);
        assert_eq!(run.status, 2);
        assert_eq!(
            run.err,
            "error: output_path_refused\nwarning: private log not written\n"
        );
        assert_clean(&run.err, &tampered);
        assert_eq!(std::fs::read_to_string(&log).expect("log kept"), chain);

        let garbage = directory.path().join("garbage.json");
        std::fs::write(&garbage, b"{\"schema\":\"(pipe X (ridge))\"}").expect("write");
        let run = invoke(&[
            "jope-motifs",
            "validate",
            "--table",
            garbage.to_str().expect("utf-8 path"),
            "--json",
        ]);
        assert_eq!(run.status, 2);
        let value: Value = serde_json::from_str(&run.out).expect("json");
        assert_eq!(value["ok"], Value::Bool(false));
        assert_eq!(value["error"], "table_schema");
        assert!(!run.out.contains("pipe"));
        assert!(ERROR_CODES.contains(&"table_schema"));
    }

    #[test]
    fn public_out_is_written_only_after_all_checks_pass() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let failing = written(directory.path(), "failing.json", &fixture::refused_table());
        let refused_summary = directory.path().join("refused-summary.json");
        let run = invoke(&[
            "jope-motifs",
            "validate",
            "--table",
            &failing,
            "--public-out",
            refused_summary.to_str().expect("utf-8 path"),
        ]);
        assert_eq!(run.status, 1);
        assert!(!refused_summary.exists());
        assert!(!run.out.contains("Wrote the shareable summary."));

        // No emitter produced an artifact for a multiclass-only fragment, so name absence was
        // not exercised: the run fails and no summary is written.
        let unexercised = written(
            directory.path(),
            "multiclass.json",
            &fixture::multiclass_table(),
        );
        let unexercised_summary = directory.path().join("multiclass-summary.json");
        let run = invoke(&[
            "jope-motifs",
            "validate",
            "--table",
            &unexercised,
            "--public-out",
            utf8(&unexercised_summary),
            "--json",
        ]);
        assert_eq!(run.status, 1, "{}{}", run.out, run.err);
        assert!(!unexercised_summary.exists());
        let value: Value = serde_json::from_str(&run.out).expect("json");
        assert_eq!(value["public_written"], Value::Bool(false));
        assert_eq!(value["checks"]["name_absence"]["passed"], 0);
        let failures = value["failures"].as_array().expect("failures");
        assert!(failures
            .iter()
            .any(|line| line["check"] == "name_absence"
                && line["code"] == "name_absence_not_exercised"));

        let table = fixture::labelled_small_table();
        fixture::assert_carried(&table);
        let passing = written(directory.path(), "passing.json", &table);
        let summary = directory.path().join("summary.json");
        let report = directory.path().join("report.json");
        let run = invoke(&[
            "jope-motifs",
            "validate",
            "--table",
            &passing,
            "--public-out",
            summary.to_str().expect("utf-8 path"),
            "--report",
            report.to_str().expect("utf-8 path"),
        ]);
        assert_eq!(run.status, 0, "{}{}", run.out, run.err);
        assert!(run.out.ends_with("Wrote the shareable summary.\n"));
        let text = std::fs::read_to_string(&summary).expect("summary written");
        assert_clean(&text, &table);
        assert!(!text.contains(&table.table_sha256[..16]));
        assert!(export::scan(&text, &table).is_ok());
        let private: Validation =
            serde_json::from_slice(&std::fs::read(&report).expect("report")).expect("json");
        assert!(private.passed());
        assert_eq!(private.table_sha256, table.table_sha256);
    }

    #[test]
    fn refused_output_paths_exit_2_before_any_work_and_write_nothing() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let table = written(directory.path(), "table.json", &fixture::small_table());
        let refused = |env: &Env, flag: &str, target: &Path| {
            let run = invoke_in(
                env,
                &[
                    "jope-motifs",
                    "validate",
                    "--table",
                    &table,
                    flag,
                    utf8(target),
                ],
            );
            assert_eq!(run.status, 2, "{flag}");
            assert!(run.out.is_empty(), "{flag}");
            // A refused log path is also the path the error chain would have gone to.
            let expected = if flag == "--private-log" {
                "error: output_path_refused\nwarning: private log not written\n"
            } else {
                "error: output_path_refused\n"
            };
            assert_eq!(run.err, expected, "{flag}");
            assert!(
                target.symlink_metadata().is_err(),
                "{flag} wrote {target:?}"
            );
        };

        // The standard policy: this scratch directory lies under the process temporary
        // directory (and, when that is inside a checkout, under a version-controlled path).
        // Whether it also refuses the resolved default state root is not shown here: that
        // would mean changing the process environment. `mine_refuses_output_under_state_root`
        // shows that every configured root is refused.
        let standard = Env::standard();
        refused(&standard, "--report", &directory.path().join("report.json"));
        refused(
            &standard,
            "--public-out",
            &directory.path().join("summary.json"),
        );

        // A version-controlled checkout, for private outputs.
        let checkout = directory.path().join("checkout");
        std::fs::create_dir_all(checkout.join(".git")).expect("checkout marker");
        let versioned = env_with(Policy::for_test(Vec::new(), true), unopened());
        refused(&versioned, "--report", &checkout.join("report.json"));
        refused(&versioned, "--private-log", &checkout.join("log.txt"));

        // The state root, for private and shareable outputs alike.
        let state = directory.path().join("state");
        std::fs::create_dir(&state).expect("state directory");
        let stateful = env_with(
            Policy::for_test(Vec::new(), false).with_state_root(state.clone()),
            unopened(),
        );
        refused(&stateful, "--report", &state.join("report.json"));
        refused(&stateful, "--public-out", &state.join("summary.json"));
    }

    #[test]
    fn mine_writes_the_private_table_and_every_input_identity() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let lines = fixture::lines();
        let state = FakeState::of(&lines);
        let out = directory.path().join("mined");
        let run = invoke_in(
            &state_env(state.clone()),
            &["jope-motifs", "mine", "--out", utf8(&out)],
        );
        assert_eq!(run.status, 0, "{}{}", run.out, run.err);
        assert!(run.err.is_empty());

        let table = read_table(&out);
        assert!(table.verify().is_ok());
        let set = StateTeaching::new(&state).qualified().expect("reads");
        assert_eq!(
            table,
            mine(&set, None, MineParameters::default()).expect("mines")
        );
        assert_eq!(table.body.motifs, fixture::table().body.motifs);
        let inputs = read_inputs(&out);
        assert_eq!(inputs, input_manifest(&set, &table).expect("inputs"));
        assert_eq!(inputs.records.len(), lines.len());

        assert_eq!(
            run.out,
            format!(
                "Wrote a private table with {} fragments (all quarantined).\n",
                table.body.motifs.len()
            )
        );
        assert_clean(&run.out, &table);
        let mut names: Vec<String> = std::fs::read_dir(&out)
            .expect("listing")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        assert_eq!(names, [INPUTS_FILE, TABLE_FILE]);
    }

    #[cfg(unix)]
    #[test]
    fn mine_output_is_a_new_0700_directory_of_0600_files() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().expect("temporary directory");
        let out = directory.path().join("mined");
        let env = state_env(FakeState::of(&fixture::small_lines()));
        let run = invoke_in(&env, &["jope-motifs", "mine", "--out", utf8(&out)]);
        assert_eq!(run.status, 0, "{}{}", run.out, run.err);
        let mode = |path: &Path| {
            std::fs::metadata(path)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777
        };
        assert_eq!(mode(&out), 0o700);
        assert_eq!(mode(&out.join(TABLE_FILE)), 0o600);
        assert_eq!(mode(&out.join(INPUTS_FILE)), 0o600);
        let before = std::fs::read(out.join(TABLE_FILE)).expect("table");

        // An existing directory is never reused or overwritten.
        let run = invoke_in(&env, &["jope-motifs", "mine", "--out", utf8(&out)]);
        assert_eq!(run.status, 2);
        assert_eq!(run.err, "error: output_path_refused\n");
        assert_eq!(std::fs::read(out.join(TABLE_FILE)).expect("table"), before);
    }

    #[test]
    fn mine_json_stdout_passes_lint_and_holds_no_canary() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let families = directory.path().join("families.json");
        std::fs::write(&families, fixture::family_map_bytes()).expect("write family map");
        let out = directory.path().join("mined");
        let mut state = FakeState::of(&fixture::labelled_small_lines());
        state.read.excluded = 2;
        let run = invoke_in(
            &state_env(state),
            &[
                "jope-motifs",
                "mine",
                "--out",
                utf8(&out),
                "--families",
                utf8(&families),
                "--min-families",
                "2",
                "--json",
            ],
        );
        assert_eq!(run.status, 0, "{}{}", run.out, run.err);
        assert!(run.err.is_empty());
        assert_eq!(run.out.lines().count(), 1);

        let table = read_table(&out);
        fixture::assert_carried(&table);
        assert_eq!(
            table.body.motifs,
            fixture::labelled_small_table().body.motifs
        );
        let value: Value = serde_json::from_str(&run.out).expect("one JSON document");
        assert_eq!(value["schema"], MINE_SCHEMA);
        assert_eq!(value["ok"], Value::Bool(true));
        assert_eq!(value["table_sha256"], table.table_sha256.as_str());
        assert_eq!(value["fragments"], 1);
        assert_eq!(value["quarantined"], 1);
        assert_eq!(value["stage_counts"]["s2"], 1);
        assert_eq!(value["parameters"]["min_datasets"], 2);
        assert_eq!(value["parameters"]["min_families"], 2);
        assert_eq!(value["parameters"]["max_stages"], 6);
        assert_eq!(value["parameters"]["family_map"], Value::Bool(true));
        assert_eq!(value["inputs"]["qualified"], 2);
        assert_eq!(value["inputs"]["used"], 2);
        assert_eq!(value["inputs"]["excluded_at_source"], 2);
        assert_eq!(value["inputs"]["excluded_adaptive"], 0);
        assert_clean(&run.out, &table);
    }

    #[test]
    fn mine_refuses_output_under_state_root() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let resolved = directory.path().join("resolved");
        let explicit = directory.path().join("explicit");
        std::fs::create_dir(&resolved).expect("resolved state directory");
        std::fs::create_dir(&explicit).expect("explicit state directory");
        // `resolved` stands for the product's resolved default root.
        let env = env_with(
            Policy::for_test(Vec::new(), false).with_state_root(resolved.clone()),
            unopened(),
        );
        for (state, target) in [
            (None, resolved.join("mined")),
            (Some(&explicit), explicit.join("mined")),
            (Some(&explicit), resolved.join("mined")),
        ] {
            let mut arguments = vec!["jope-motifs", "mine", "--out", utf8(&target)];
            if let Some(state) = state {
                arguments.extend(["--state", utf8(state)]);
            }
            let run = invoke_in(&env, &arguments);
            assert_eq!(run.status, 2, "{arguments:?}");
            assert!(run.out.is_empty(), "{arguments:?}");
            assert_eq!(run.err, "error: output_path_refused\n", "{arguments:?}");
            assert!(target.symlink_metadata().is_err(), "{arguments:?}");
        }

        // The private log is refused under an explicit state root as well.
        let log = explicit.join("log.txt");
        let out = directory.path().join("mined");
        let run = invoke_in(
            &env,
            &[
                "jope-motifs",
                "mine",
                "--state",
                utf8(&explicit),
                "--out",
                utf8(&out),
                "--private-log",
                utf8(&log),
            ],
        );
        assert_eq!(run.status, 2);
        assert_eq!(
            run.err,
            "error: output_path_refused\nwarning: private log not written\n"
        );
        assert!(log.symlink_metadata().is_err());
        assert!(out.symlink_metadata().is_err());
    }

    #[test]
    fn mine_with_zero_admitted_programs_writes_nothing() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let out = directory.path().join("mined");
        let mut empty = FakeState::of(&[]);
        empty.read.excluded = 4;
        let env = state_env(empty);
        let run = invoke_in(&env, &["jope-motifs", "mine", "--out", utf8(&out)]);
        assert_eq!(run.status, 2);
        assert!(run.out.is_empty());
        assert_eq!(run.err, "error: no_admitted_programs\n");
        assert!(out.symlink_metadata().is_err());

        let run = invoke_in(
            &env,
            &["jope-motifs", "mine", "--out", utf8(&out), "--json"],
        );
        assert_eq!(run.status, 2);
        let value: Value = serde_json::from_str(&run.out).expect("json");
        assert_eq!(value["schema"], MINE_SCHEMA);
        assert_eq!(value["ok"], Value::Bool(false));
        assert_eq!(value["error"], "no_admitted_programs");
        assert!(out.symlink_metadata().is_err());
    }

    #[test]
    fn mine_without_a_reader_exits_2_state_unavailable() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let out = directory.path().join("mined");
        // No reader at all.
        let run = invoke(&["jope-motifs", "mine", "--out", utf8(&out)]);
        assert_eq!(run.status, 2);
        assert_eq!(run.err, "error: state_unavailable\n");
        assert!(out.symlink_metadata().is_err());

        let mut broken = FakeState::of(&fixture::small_lines());
        broken.broken = true;
        let run = invoke_in(
            &state_env(broken),
            &["jope-motifs", "mine", "--out", utf8(&out)],
        );
        assert_eq!(run.status, 2);
        assert_eq!(run.err, "error: state_unavailable\n");
        assert!(out.symlink_metadata().is_err());
    }

    /// A root that holds no State (here an existing directory without the HWM chain tip, as a
    /// mistyped `--state` would be) and a missing root are both `state_unavailable` for both
    /// verbs, never an empty State, and nothing is written. The standard opener is the real
    /// learning adapter.
    #[test]
    fn verbs_over_a_root_without_a_state_write_nothing() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let root = directory.path().join("state");
        std::fs::create_dir(&root).expect("state root");
        let out = directory.path().join("mined");
        let env = env_with(
            Policy::for_test(Vec::new(), false),
            Env::standard().open_state,
        );
        let run = invoke_in(
            &env,
            &[
                "jope-motifs",
                "mine",
                "--state",
                utf8(&root),
                "--out",
                utf8(&out),
            ],
        );
        assert_eq!(run.status, 2, "{}{}", run.out, run.err);
        assert_eq!(run.err, "error: state_unavailable\n");
        assert!(out.symlink_metadata().is_err());

        let table = written(directory.path(), "table.json", &fixture::small_table());
        let run = invoke_in(
            &env,
            &[
                "jope-motifs",
                "validate",
                "--table",
                &table,
                "--state",
                utf8(&root),
            ],
        );
        assert_eq!(run.status, 2, "{}{}", run.out, run.err);
        assert!(run.out.is_empty());
        assert_eq!(run.err, "error: state_unavailable\n");

        let absent = directory.path().join("absent");
        let run = invoke_in(
            &env,
            &[
                "jope-motifs",
                "mine",
                "--state",
                utf8(&absent),
                "--out",
                utf8(&out),
            ],
        );
        assert_eq!(run.status, 2);
        assert_eq!(run.err, "error: state_unavailable\n");
        assert!(absent.symlink_metadata().is_err());
        assert!(out.symlink_metadata().is_err());
    }

    #[test]
    fn mine_refuses_unqualified_unburned_or_unauthenticated_records() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let out = directory.path().join("mined");
        let mut unburned = FakeState::of(&fixture::small_lines());
        unburned.read.entries[0].original_label_role = String::from("test");
        let mut unqualified = FakeState::of(&fixture::small_lines());
        unqualified.read.entries[1].improved_folds = 3;
        let mut unauthenticated = FakeState::of(&fixture::small_lines());
        unauthenticated.pointers[1].sealed = true;
        for (state, code) in [
            (unburned, "label_burn_missing"),
            (unqualified, "record_unqualified"),
            (unauthenticated, "record_unauthenticated"),
        ] {
            let run = invoke_in(
                &state_env(state),
                &["jope-motifs", "mine", "--out", utf8(&out)],
            );
            assert_eq!(run.status, 2, "{code}");
            assert_eq!(run.err, format!("error: {code}\n"));
            assert!(out.symlink_metadata().is_err(), "{code}");
        }

        // The full chain goes only to the private log.
        let log = directory.path().join("log.txt");
        let mut burned = FakeState::of(&fixture::small_lines());
        burned.read.entries[0].label_burn_sha256 = Some(fixture::digest("burn/unexpected"));
        let run = invoke_in(
            &state_env(burned),
            &[
                "jope-motifs",
                "mine",
                "--out",
                utf8(&out),
                "--private-log",
                utf8(&log),
            ],
        );
        assert_eq!(run.status, 2);
        assert_eq!(run.err, "error: label_origin_refused\n");
        let chain = std::fs::read_to_string(&log).expect("private log written");
        assert!(chain.contains("label_origin_refused"));
        assert!(out.symlink_metadata().is_err());
    }

    #[test]
    fn verbs_refuse_missing_required_arguments_and_bad_limits() {
        for arguments in [
            vec!["jope-motifs"],
            vec!["jope-motifs", "mine"],
            vec!["jope-motifs", "mine", "--state", "/state"],
            vec![
                "jope-motifs",
                "mine",
                "--out",
                "/mined",
                "--max-stages",
                "six",
            ],
            vec!["jope-motifs", "validate"],
            vec!["jope-motifs", "validate", "--state", "/state"],
        ] {
            assert!(
                Cli::try_parse_from(arguments.iter().copied()).is_err(),
                "{arguments:?}"
            );
        }

        let directory = tempfile::tempdir().expect("temporary directory");
        let out = directory.path().join("mined");
        let bad_map = directory.path().join("families.json");
        std::fs::write(&bad_map, b"{\"abc\":\"canaryfamily01\"}").expect("write");
        let missing_map = directory.path().join("absent.json");
        let env = env_with(Policy::for_test(Vec::new(), false), unopened());
        for (extra, code) in [
            (vec!["--max-stages", "7"], "invalid_parameters"),
            (vec!["--max-stages", "1"], "invalid_parameters"),
            (vec!["--min-datasets", "0"], "invalid_parameters"),
            (vec!["--min-families", "2"], "invalid_parameters"),
            (vec!["--families", utf8(&bad_map)], "invalid_family_map"),
            (vec!["--families", utf8(&missing_map)], "invalid_family_map"),
        ] {
            let mut arguments = vec!["jope-motifs", "mine", "--out", utf8(&out)];
            arguments.extend(extra);
            let run = invoke_in(&env, &arguments);
            assert_eq!(run.status, 2, "{arguments:?}");
            assert_eq!(run.err, format!("error: {code}\n"), "{arguments:?}");
            assert!(out.symlink_metadata().is_err(), "{arguments:?}");
        }
    }

    #[test]
    fn validate_with_state_reports_source_verification() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let lines = fixture::labelled_small_lines();
        let state = FakeState::of(&lines);
        let families = directory.path().join("families.json");
        std::fs::write(&families, fixture::family_map_bytes()).expect("write family map");
        let mined = directory.path().join("mined");
        let run = invoke_in(
            &state_env(state.clone()),
            &[
                "jope-motifs",
                "mine",
                "--out",
                utf8(&mined),
                "--families",
                utf8(&families),
            ],
        );
        assert_eq!(run.status, 0, "{}{}", run.out, run.err);
        let table = read_table(&mined);
        fixture::assert_carried(&table);
        let table_path = mined.join(TABLE_FILE);
        // The fake opener serves its State for any root; the root only has to be named.
        let root = directory.path().join("state");

        // Current: every occurrence is confirmed, and the shareable summary says so.
        let summary = directory.path().join("summary.json");
        let run = invoke_in(
            &state_env(state),
            &[
                "jope-motifs",
                "validate",
                "--table",
                utf8(&table_path),
                "--state",
                utf8(&root),
                "--public-out",
                utf8(&summary),
                "--json",
            ],
        );
        assert_eq!(run.status, 0, "{}{}", run.out, run.err);
        let value: Value = serde_json::from_str(&run.out).expect("json");
        assert_eq!(value["ok"], Value::Bool(true));
        assert_eq!(value["source_verified"], Value::Bool(true));
        assert_eq!(value["public_written"], Value::Bool(true));
        assert!(value["failures"].as_array().expect("failures").is_empty());
        assert_clean(&run.out, &table);
        let text = std::fs::read_to_string(&summary).expect("summary written");
        let shared: Value = serde_json::from_str(&text).expect("json");
        assert_eq!(shared["source_verified"], Value::Bool(true));
        assert_clean(&text, &table);
        assert!(export::scan(&text, &table).is_ok());

        // Stale: one recorded occurrence is no longer a current qualified record. The checks
        // themselves still pass, but the run fails and nothing is shared.
        let stale = FakeState::of(&[
            lines[0].clone(),
            fixture::qualified(lines[1].program(), TaskLabel::Regression, "d3", "s03-other"),
        ]);
        let stale_summary = directory.path().join("stale-summary.json");
        let run = invoke_in(
            &state_env(stale.clone()),
            &[
                "jope-motifs",
                "validate",
                "--table",
                utf8(&table_path),
                "--state",
                utf8(&root),
                "--public-out",
                utf8(&stale_summary),
                "--json",
            ],
        );
        assert_eq!(run.status, 1, "{}{}", run.out, run.err);
        let value: Value = serde_json::from_str(&run.out).expect("json");
        assert_eq!(value["ok"], Value::Bool(false));
        assert_eq!(value["source_verified"], Value::Bool(false));
        assert_eq!(value["failed"], 0);
        assert_eq!(value["public_written"], Value::Bool(false));
        assert!(!stale_summary.exists());
        // No fragment failed, so the failed run names its reason in a table-wide line.
        let failures = value["failures"].as_array().expect("failures");
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert_eq!(failures[0]["check"], "source");
        assert_eq!(failures[0]["code"], "source_not_confirmed");
        assert!(failures[0].get("fragment").is_none());
        assert!(CHECK_CODES.contains(&("source", "source_not_confirmed")));
        assert_clean(&run.out, &table);
        let report = directory.path().join("report.json");
        let run = invoke_in(
            &state_env(stale),
            &[
                "jope-motifs",
                "validate",
                "--table",
                utf8(&table_path),
                "--state",
                utf8(&root),
                "--report",
                utf8(&report),
            ],
        );
        assert_eq!(run.status, 1);
        assert!(
            run.out.ends_with("Source records: not confirmed.\n"),
            "{}",
            run.out
        );
        let private: Validation =
            serde_json::from_slice(&std::fs::read(&report).expect("report")).expect("json");
        assert!(!private.source_verified);
        // Equal digests prove nothing alone: the fake State keeps its digest.
        assert_eq!(private.source_digest_equal, Some(true));

        // A State that cannot be read is an error, never an unconfirmed pass.
        let run = invoke(&[
            "jope-motifs",
            "validate",
            "--table",
            utf8(&table_path),
            "--state",
            utf8(&root),
        ]);
        assert_eq!(run.status, 2);
        assert_eq!(run.err, "error: state_unavailable\n");
    }

    #[test]
    fn empty_table_fails_validation_and_shares_nothing() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let mined = directory.path().join("mined");
        let run = invoke_in(
            &state_env(FakeState::of(&fixture::small_lines())),
            &[
                "jope-motifs",
                "mine",
                "--out",
                utf8(&mined),
                "--min-datasets",
                "3",
                "--json",
            ],
        );
        assert_eq!(run.status, 0, "{}{}", run.out, run.err);
        let value: Value = serde_json::from_str(&run.out).expect("json");
        assert_eq!(value["fragments"], 0);
        assert_eq!(value["inputs"]["qualified"], 2);
        // No fragment was kept, but every examined input identity is preserved.
        assert_eq!(read_inputs(&mined).records.len(), 2);

        let table_path = mined.join(TABLE_FILE);
        let summary = directory.path().join("summary.json");
        let run = invoke(&[
            "jope-motifs",
            "validate",
            "--table",
            utf8(&table_path),
            "--public-out",
            utf8(&summary),
            "--json",
        ]);
        assert_eq!(run.status, 1, "{}{}", run.out, run.err);
        let value: Value = serde_json::from_str(&run.out).expect("json");
        assert_eq!(value["ok"], Value::Bool(false));
        assert_eq!(value["fragments"], 0);
        assert_eq!(value["public_written"], Value::Bool(false));
        assert!(!summary.exists());

        let run = invoke(&["jope-motifs", "validate", "--table", utf8(&table_path)]);
        assert_eq!(run.status, 1);
        assert_eq!(
            run.out,
            "Checked 0 fragments: nothing was examined, so the table does not pass. \
             Promotion evidence: not run.\n"
        );
        assert!(lint_shipped_text(&run.out).is_empty());

        // With --state, the source check examined no occurrence and says so.
        let root = directory.path().join("state");
        let run = invoke_in(
            &state_env(FakeState::of(&fixture::small_lines())),
            &[
                "jope-motifs",
                "validate",
                "--table",
                utf8(&table_path),
                "--state",
                utf8(&root),
                "--json",
            ],
        );
        assert_eq!(run.status, 1, "{}{}", run.out, run.err);
        let value: Value = serde_json::from_str(&run.out).expect("json");
        assert_eq!(value["source_verified"], Value::Bool(false));
        let failures = value["failures"].as_array().expect("failures");
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert_eq!(failures[0]["check"], "source");
        assert_eq!(failures[0]["code"], "source_not_exercised");
        assert!(failures[0].get("fragment").is_none());

        // The one-line summary of an empty table claims nothing about quarantine.
        let human = directory.path().join("mined-human");
        let run = invoke_in(
            &state_env(FakeState::of(&fixture::small_lines())),
            &[
                "jope-motifs",
                "mine",
                "--out",
                utf8(&human),
                "--min-datasets",
                "3",
            ],
        );
        assert_eq!(run.status, 0, "{}{}", run.out, run.err);
        assert_eq!(
            run.out,
            "Wrote a private table with no fragments; every examined input is listed.\n"
        );
        assert!(!run.out.contains("quarantined"));
        assert!(lint_shipped_text(&run.out).is_empty());
        let lower = run.out.to_lowercase();
        for word in OWNER_BANNED {
            assert!(!lower.contains(word), "{word}");
        }
        assert_eq!(read_inputs(&human).records.len(), 2);
    }
}
