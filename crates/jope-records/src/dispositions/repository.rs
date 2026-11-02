use super::*;

/// `git -C <dir> cat-file -e <commit>^{commit}`, failing as `subprocess.run(check=True)` did.
fn require_commit(dir: &Path, commit: &str) -> Result<()> {
    let object = format!("{commit}^{{commit}}");
    let status = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["cat-file", "-e", &object])
        .output()
        .map_err(|e| Error::io(Path::new("git"), e))?
        .status;
    if status.success() {
        return Ok(());
    }
    let dir = dir.display().to_string();
    let command = py::str_list_repr(&["git", "-C", &dir, "cat-file", "-e", &object]);
    Err(Error::Command(match status.code() {
        Some(code) => format!("Command '{command}' returned non-zero exit status {code}."),
        None => format!("Command '{command}' died with a signal."),
    }))
}

/// Verify `dispositions.json` under `root` (the findings directory). With `require_git`, every
/// cited commit must exist in `repository` (default: `root`'s checkout).
pub fn verify(root: &Path, require_git: bool, repository: Option<&Path>) -> Result<Summary> {
    findings::verify(root)?;
    let register_path = root.join("register.json");
    let register_bytes = py::read_bytes(&register_path)?;
    let register = py::load_json_bytes(&register_path, &register_bytes)?;
    let mut entries: HashMap<String, &Value> = HashMap::new();
    let register_entries = register
        .get("entries")
        .and_then(Value::as_list)
        .ok_or_else(|| Error::Shape("register: entries is not a list".into()))?;
    for entry in register_entries {
        if let Some(id) = entry.get("id").and_then(Value::as_str) {
            entries.insert(id.to_owned(), entry);
        }
    }
    let document = py::load_json(&root.join("dispositions.json"))?;
    require!(
        document.get("schema") == Some(&Value::from(SCHEMA)),
        "dispositions schema"
    );
    require!(
        document.get("register_sha256") == Some(&Value::from(py::sha256_hex(&register_bytes))),
        "dispositions bind another register"
    );
    let records = document.get("records").and_then(Value::as_list);
    require!(records.is_some(), "records list");
    let mut last_sequence: HashMap<String, i64> = HashMap::new();
    let mut effective: Vec<(String, (String, bool))> = Vec::new();
    for record in records.expect("checked above") {
        let finding = record.get("finding");
        let entry = finding.and_then(Value::as_str).and_then(|f| entries.get(f));
        require!(
            entry.is_some(),
            "unknown finding {}",
            finding.map_or_else(|| "None".to_owned(), Value::repr)
        );
        let finding = finding.and_then(Value::as_str).expect("checked above");
        let entry = *entry.expect("checked above");
        let sequence = record.get("sequence").and_then(Value::as_int);
        let last = last_sequence.get(finding).copied().unwrap_or(0);
        require!(
            sequence.is_some_and(|s| s > last),
            "{finding}: sequence must increase (append-only history)"
        );
        last_sequence.insert(finding.to_owned(), sequence.expect("checked above"));
        let live = is_one_of(entry.get("work_package"), &LIVE_PACKAGES);
        let outcome = validate_record(root, record, entry, live)?;
        match effective.iter_mut().find(|(f, _)| f == finding) {
            Some(slot) => slot.1 = outcome,
            None => effective.push((finding.to_owned(), outcome)),
        }
        if require_git {
            let dir = repository.unwrap_or(root);
            let attestations = record
                .get("attestations")
                .and_then(py::iterate)
                .unwrap_or_default();
            for attestation in &attestations {
                let mut commits = vec![attestation
                    .get("source_commit")
                    .map_or_else(String::new, Value::py_str)];
                if let Some(tests) = attestation.get("tests").and_then(Value::as_list) {
                    for test in tests {
                        if let Some(pre) = test.get("pre_fix_commit").filter(|p| p.truthy()) {
                            commits.push(pre.py_str());
                        }
                    }
                }
                for commit in &commits {
                    require_commit(dir, commit)?;
                }
            }
        }
    }
    let mut dispositions: BTreeMap<String, usize> = BTreeMap::new();
    for (_, (disposition, _)) in &effective {
        *dispositions.entry(disposition.clone()).or_default() += 1;
    }
    let closed = effective.iter().filter(|(_, (_, c))| *c).count();
    let mut closed_findings: Vec<String> = effective
        .iter()
        .filter(|(_, (_, c))| *c)
        .map(|(f, _)| f.clone())
        .collect();
    closed_findings.sort();
    Ok(Summary {
        entries: entries.len(),
        with_disposition: effective.len(),
        closed,
        open: entries.len() - closed,
        closed_findings,
        dispositions,
    })
}
