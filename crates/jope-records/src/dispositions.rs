//! `verify-dispositions`: scoped finding dispositions against hash-bound evidence (W12-T02).
//!
//! The v1 register stays an immutable open inventory. Dispositions live in a separate
//! append-only record whose closures are accepted only when stage-appropriate evidence exists,
//! is byte-identical to its recorded digest and names the same finding, stage and source
//! commit. A digest authenticates bytes, not the truth of their producer; independent review of
//! the evidence remains required. Evidence whose bytes carry the public conformance fixtures'
//! synthetic identity (W12-T07) may back only source-trace and synthetic-fixture evidence,
//! whatever kind it declares.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::py::{self, PosixPath};
use crate::pyjson::{Dict, Value};
use crate::{findings, require, synthetic, Error, Result};

pub const SCHEMA: &str = "jope.regression-finding-dispositions/v1";
pub const EVIDENCE_SCHEMA: &str = "jope.regression-finding-evidence/v1";
pub const DISPOSITIONS: [&str; 6] = [
    "confirmed-defect",
    "verified-safeguard",
    "missing-execution-evidence",
    "historical-unknown",
    "research-result",
    "deferred-non-regression",
];
/// Dispositions that record an honest open state; they can never close a finding.
pub const NEVER_CLOSED: [&str; 3] = [
    "missing-execution-evidence",
    "historical-unknown",
    "deferred-non-regression",
];
pub const STAGES: [&str; 6] = [
    "source_implementation",
    "fixture_proof",
    "hosted_ci",
    "live_qualification",
    "measurement_complete",
    "promotion",
];
/// Packages whose acceptance requires live operation on managed hosts or real storage.
pub const LIVE_PACKAGES: [&str; 4] = ["W7", "W8", "W10", "W11"];
pub const TEST_ROLES: [&str; 4] = ["pre-fix-failure", "negative", "positive", "regression"];
pub const DEFERRABLE: [&str; 3] = ["binary", "multiclass", "time-series"];
pub const SCOPE: &str =
    "evidence binding only; reviewed evidence, not this validator, carries truth";
/// Kinds that evidence carrying the conformance synthetic identity can never declare (W12-T07):
/// it may trace source or be a synthetic fixture, but never stand for real data, hosted CI,
/// live operation, measurement or a decision, so it can never close a finding on its own.
pub const NOT_SYNTHETIC_KINDS: [&str; 5] = [
    "real-corpus-fixture",
    "hosted-ci",
    "live-host",
    "measurement",
    "decision",
];

/// Evidence kinds each stage accepts. Synthetic fixtures never back live, measurement or
/// promotion stages, whatever their bytes or names say.
pub fn stage_kinds(stage: &str) -> &'static [&'static str] {
    match stage {
        "source_implementation" => &["source-trace"],
        "fixture_proof" => &["synthetic-fixture", "real-corpus-fixture"],
        "hosted_ci" => &["hosted-ci"],
        "live_qualification" => &["live-host"],
        "measurement_complete" => &["measurement"],
        "promotion" => &["decision"],
        _ => &[],
    }
}

/// The summary `verify-dispositions` prints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub entries: usize,
    pub with_disposition: usize,
    pub closed: usize,
    pub open: usize,
    pub closed_findings: Vec<String>,
    pub dispositions: BTreeMap<String, usize>,
}

impl Summary {
    pub fn to_json(&self) -> Value {
        let mut dict = Dict::new();
        dict.insert("entries", self.entries.into());
        dict.insert("with_disposition", self.with_disposition.into());
        dict.insert("closed", self.closed.into());
        dict.insert("open", self.open.into());
        dict.insert(
            "closed_findings",
            Value::List(
                self.closed_findings
                    .iter()
                    .map(|f| f.as_str().into())
                    .collect(),
            ),
        );
        dict.insert(
            "dispositions",
            Value::Dict(
                self.dispositions
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::from(*v)))
                    .collect(),
            ),
        );
        dict.insert("scope", SCOPE.into());
        Value::Dict(dict)
    }
}

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

fn load_evidence(
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
fn check_tests(
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

struct Attested {
    roles: HashMap<String, usize>,
}

fn role_count(roles: &HashMap<String, usize>, role: &str) -> usize {
    roles.get(role).copied().unwrap_or(0)
}

/// Validate one disposition record; returns its disposition and whether it closes the finding.
fn validate_record(
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
