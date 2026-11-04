use super::*;

pub(super) struct ValidateArgs {
    pub(super) table: PathBuf,
    pub(super) hosts_per_fragment: u32,
    pub(super) state: Option<PathBuf>,
    pub(super) report: Option<PathBuf>,
    pub(super) public_out: Option<PathBuf>,
    pub(super) private_log: Option<PathBuf>,
}

pub(super) struct Outcome {
    pub(super) validation: Validation,
    /// The source check's failure code (`validate::verify_source`); `None` when the source was
    /// not read or was confirmed.
    source_failure: Option<&'static str>,
    public_written: bool,
}

pub(super) fn validate_verb(args: &ValidateArgs, env: &Env, policy: &Policy) -> Result<Outcome> {
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

pub(super) fn render_validate(outcome: &Outcome, json: bool, out: &mut dyn Write) -> Result<()> {
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
