use super::evidence::is_one_of;
use super::*;

struct Attested {
    roles: HashMap<String, usize>,
}

fn role_count(roles: &HashMap<String, usize>, role: &str) -> usize {
    roles.get(role).copied().unwrap_or(0)
}

/// Validate one disposition record; returns its disposition and whether it closes the finding.
pub(crate) fn validate_record(
    root: &Path,
    record: &Value,
    register_entry: &Value,
    package_live: bool,
) -> Result<(String, bool)> {
    let finding = record
        .get("finding")
        .and_then(Value::as_str)
        .expect("the caller resolved the finding");
    let disposition = record.get("disposition");
    require!(
        is_one_of(disposition, &DISPOSITIONS),
        "{finding}: disposition"
    );
    let disposition = disposition.and_then(Value::as_str).expect("checked above");
    require!(
        record
            .get("scope")
            .and_then(Value::as_str)
            .is_some_and(|s| !py::strip(s).is_empty()),
        "{finding}: scope"
    );
    let empty = Value::Dict(Dict::new());
    let decided = record.get("decided_by").unwrap_or(&empty);
    require!(
        decided.as_dict().is_some()
            && decided.get("role").is_some_and(Value::truthy)
            && decided.get("identity").is_some_and(Value::truthy),
        "{finding}: decided_by"
    );
    let closed = match record.get("closed") {
        Some(Value::Bool(closed)) => *closed,
        _ => return Err(Error::Invalid(format!("{finding}: closed flag"))),
    };
    if disposition == "deferred-non-regression" {
        let subcases = record.get("deferred_subcases");
        let deferrable = subcases.and_then(Value::as_list).is_some_and(|list| {
            !list.is_empty() && list.iter().all(|s| is_one_of(Some(s), &DEFERRABLE))
        });
        require!(
            deferrable,
            "{finding}: deferral names only binary/multiclass/time-series subcases"
        );
        let scope = register_entry.get("scope").unwrap_or(&Value::Null);
        require!(
            py::contains(scope, &Value::from("deferred")) == Some(true),
            "{finding}: entry has no deferred scope"
        );
    }
    if disposition == "historical-unknown" || disposition == "missing-execution-evidence" {
        require!(
            record.get("reason").is_some_and(Value::truthy),
            "{finding}: open disposition needs its reason"
        );
    }

    let mut stages: HashMap<String, Attested> = HashMap::new();
    let attestations = match record.get("attestations") {
        None => Vec::new(),
        Some(value) => py::iterate(value)
            .ok_or_else(|| Error::Shape(format!("{finding}: attestations is not iterable")))?,
    };
    for attestation in &attestations {
        let stage = attestation.get("stage");
        require!(
            is_one_of(stage, &STAGES),
            "{finding}: stage {}",
            stage.map_or_else(|| "None".to_owned(), Value::repr)
        );
        let stage = stage.and_then(Value::as_str).expect("checked above");
        require!(
            !stages.contains_key(stage),
            "{finding}: duplicate {stage} attestation"
        );
        let commit = attestation.get("source_commit");
        require!(py::is_hex(commit, 40), "{finding}: {stage} source commit");
        let commit = commit.and_then(Value::as_str).expect("checked above");
        let references = attestation.get("evidence");
        require!(
            references
                .and_then(Value::as_list)
                .is_some_and(|r| !r.is_empty()),
            "{finding}: {stage} has no evidence"
        );
        let evidence = references
            .and_then(Value::as_list)
            .expect("checked above")
            .iter()
            .map(|reference| load_evidence(root, finding, stage, commit, reference))
            .collect::<Result<Vec<Value>>>()?;
        if stage == "hosted_ci" {
            require!(
                evidence.iter().all(|b| {
                    b.get("run_id").and_then(Value::as_int).is_some()
                        && b.get("attempt").and_then(Value::as_int).is_some()
                        && b.get("head_sha") == Some(&Value::from(commit))
                        && b.get("conclusion") == Some(&Value::from("success"))
                }),
                "{finding}: hosted CI evidence binding"
            );
        }
        if stage == "promotion" {
            require!(
                evidence.iter().all(|b| {
                    b.get("decision_owner_role") == Some(&Value::from("scientific-decision-owner"))
                }),
                "{finding}: promotion owner"
            );
        }
        if stage == "source_implementation" {
            require!(
                evidence
                    .iter()
                    .all(|b| b.get("paths").is_some_and(Value::truthy)),
                "{finding}: source trace paths"
            );
        }
        let roles = check_tests(finding, attestation, commit, &evidence)?;
        stages.insert(stage.to_owned(), Attested { roles });
    }

    if stages.contains_key("promotion") {
        require!(
            stages.contains_key("live_qualification"),
            "{finding}: promotion without live qualification"
        );
    }
    if stages.contains_key("live_qualification") {
        require!(
            stages.contains_key("hosted_ci"),
            "{finding}: live qualification without hosted CI"
        );
    }
    if !closed {
        return Ok((disposition.to_owned(), false));
    }
    require!(
        !NEVER_CLOSED.contains(&disposition),
        "{finding}: {disposition} cannot close"
    );
    let has = |stage: &str| stages.contains_key(stage);
    match disposition {
        "confirmed-defect" => {
            require!(
                has("source_implementation") && has("fixture_proof") && has("hosted_ci"),
                "{finding}: defect closure needs source, fixture and hosted CI"
            );
            let roles = &stages["fixture_proof"].roles;
            require!(
                role_count(roles, "pre-fix-failure") > 0
                    && role_count(roles, "negative") > 0
                    && role_count(roles, "positive") > 0,
                "{finding}: defect closure needs pre-fix failure, negative and positive tests"
            );
        }
        "verified-safeguard" => {
            require!(
                has("source_implementation") && has("fixture_proof") && has("hosted_ci"),
                "{finding}: safeguard closure needs trace, regression test and hosted CI"
            );
            require!(
                role_count(&stages["fixture_proof"].roles, "regression") > 0,
                "{finding}: safeguard closure needs regression protection"
            );
        }
        "research-result" => {
            require!(
                has("measurement_complete"),
                "{finding}: research needs complete measurement"
            );
        }
        _ => {}
    }
    if package_live {
        require!(
            has("live_qualification"),
            "{finding}: {} requires live qualification",
            register_entry
                .get("work_package")
                .map_or_else(String::new, Value::py_str)
        );
    }
    Ok((disposition.to_owned(), true))
}
