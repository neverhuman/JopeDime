use super::*;

fn is_one_of(value: Option<&Value>, set: &[&str]) -> bool {
    value
        .and_then(Value::as_str)
        .is_some_and(|s| set.contains(&s))
}

/// Resolve a finding-local evidence file without traversal or symlinks.
fn evidence_path(root: &Path, finding: &str, reference: &Value) -> Result<PathBuf> {
    let fields_ok = reference
        .as_dict()
        .is_some_and(|r| r.len() == 2 && r.contains_key("path") && r.contains_key("sha256"));
    require!(fields_ok, "{finding}: evidence reference fields");
    let text = reference
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Shape(format!("{finding}: evidence path is not a string")))?;
    let relative = PosixPath::parse(text);
    require!(
        !relative.is_absolute() && !relative.has_parent_reference(),
        "{finding}: evidence path traversal"
    );
    require!(
        relative.parts.len() >= 3
            && relative.parts[0] == "evidence"
            && relative.parts[1] == finding,
        "{finding}: evidence outside evidence/{finding}/"
    );
    let digest = reference.get("sha256");
    require!(py::is_hex(digest, 64), "{finding}: evidence digest format");
    let mut current = root.to_path_buf();
    for part in &relative.parts {
        current.push(part);
        require!(!py::is_symlink(&current), "{finding}: evidence symlink");
    }
    let path = relative.join_onto(root);
    let shown = relative.parts.join("/");
    require!(path.is_file(), "{finding}: evidence missing {shown}");
    require!(
        digest == Some(&Value::from(py::sha256_hex(&py::read_bytes(&path)?))),
        "{finding}: evidence digest mismatch {shown}"
    );
    Ok(path)
}

pub(super) fn load_evidence(
    root: &Path,
    finding: &str,
    stage: &str,
    commit: &str,
    reference: &Value,
) -> Result<Value> {
    let path = evidence_path(root, finding, reference)?;
    let body = py::load_json(&path)?;
    require!(body.as_dict().is_some(), "{finding}: evidence schema");
    require!(
        body.get("schema") == Some(&Value::from(EVIDENCE_SCHEMA)),
        "{finding}: evidence schema"
    );
    require!(
        body.get("finding") == Some(&Value::from(finding)),
        "{finding}: evidence names another finding"
    );
    require!(
        body.get("stage") == Some(&Value::from(stage)),
        "{finding}: evidence names another stage"
    );
    require!(
        body.get("source_commit") == Some(&Value::from(commit)),
        "{finding}: evidence names another commit"
    );
    let kind = body.get("evidence_kind");
    require!(
        is_one_of(kind, stage_kinds(stage)),
        "{finding}: {} cannot back {stage}",
        kind.map_or_else(|| "None".to_owned(), Value::repr)
    );
    if synthetic::marked(&body) {
        let kind = kind.and_then(Value::as_str).expect("kind checked above");
        require!(
            !NOT_SYNTHETIC_KINDS.contains(&kind),
            "{finding}: synthetic identity cannot be {kind} evidence"
        );
    }
    Ok(body)
}

/// A value's items as `for name in value` visited them (`None`: not iterable).
fn names(body: &Value, field: &str) -> Result<Vec<Value>> {
    match body.get(field) {
        None => Ok(Vec::new()),
        Some(value) => py::iterate(value)
            .ok_or_else(|| Error::Shape(format!("{field} is not iterable: {}", value.repr()))),
    }
}

/// Named tests must appear with the outcome their role requires.
pub(super) fn check_tests(
    finding: &str,
    attestation: &Value,
    commit: &str,
    evidence: &[Value],
) -> Result<HashMap<String, usize>> {
    let tests = match attestation.get("tests") {
        None => Vec::new(),
        Some(Value::List(tests)) => tests.clone(),
        Some(_) => return Err(Error::Invalid(format!("{finding}: tests list"))),
    };
    let mut outcomes: HashMap<String, HashSet<&'static str>> = HashMap::new();
    for body in evidence {
        for name in names(body, "passed_tests")? {
            if let Some(name) = name.as_str() {
                outcomes
                    .entry(name.to_owned())
                    .or_default()
                    .insert("passed");
            }
        }
        for name in names(body, "failed_tests")? {
            if let Some(name) = name.as_str() {
                outcomes
                    .entry(name.to_owned())
                    .or_default()
                    .insert("failed");
            }
        }
    }
    let mut roles: HashMap<String, usize> = HashMap::new();
    for test in &tests {
        let fields_ok = test.as_dict().is_some_and(|t| {
            t.contains_key("target") && t.contains_key("name") && t.contains_key("role")
        });
        require!(fields_ok, "{finding}: test fields");
        let role = test.get("role");
        require!(is_one_of(role, &TEST_ROLES), "{finding}: test role");
        let role = role.and_then(Value::as_str).expect("checked above");
        let qualified = format!(
            "{}::{}",
            test.get("target").expect("checked above").py_str(),
            test.get("name").expect("checked above").py_str()
        );
        *roles.entry(role.to_owned()).or_default() += 1;
        if role == "pre-fix-failure" {
            let pre = test.get("pre_fix_commit");
            require!(
                py::is_hex(pre, 40) && pre != Some(&Value::from(commit)),
                "{finding}: pre-fix failure needs a distinct pre-fix commit"
            );
            let pre = pre.expect("checked above");
            let qualified = Value::from(qualified.as_str());
            let observed = evidence.iter().any(|b| {
                b.get("pre_fix_commit") == Some(pre)
                    && py::contains(
                        b.get("failed_tests").unwrap_or(&Value::List(Vec::new())),
                        &qualified,
                    ) == Some(true)
            });
            require!(
                observed,
                "{finding}: pre-fix failure not observed at {}",
                pre.py_str()
            );
        } else {
            let seen = outcomes.get(&qualified);
            require!(
                seen.is_some_and(|s| s.contains("passed") && !s.contains("failed")),
                "{finding}: {qualified} not observed passing at the attested commit"
            );
        }
    }
    Ok(roles)
}
