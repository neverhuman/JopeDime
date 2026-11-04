//! Port of `test_verify_dispositions.py`, plus `--require-git` and the W12-T07 synthetic
//! identity rule.

mod mutations;
mod stages;
mod synthetic_identity;

use std::path::{Path, PathBuf};
use std::process::Command;

use jope_records::dispositions::{verify, Summary, EVIDENCE_SCHEMA, SCHEMA};
use jope_records::layout;
use jope_records::pyjson::{Dict, Value};

use crate::{assert_refused, at, digest, dump, dump_with, json, load, repo, scratch, set, strs};

const PRE: &str = "1111111111111111111111111111111111111111";
const FIXED: &str = "2222222222222222222222222222222222222222";
const TARGET: &str = "cargo:jope-parallel:gp_contracts";
const TEST: &str = "fitted_predictor_roundtrip";

/// Disposable copy of the real register with synthetic evidence only.
struct Fixture {
    _directory: tempfile::TempDir,
    root: PathBuf,
    register_sha256: String,
    fixed: String,
    pre: String,
}

impl Fixture {
    fn new() -> Self {
        let directory = crate::scratch();
        let root = directory.path().to_path_buf();
        let findings = layout::findings(&repo());
        crate::copy_tree(&findings.join("sources"), &root.join("sources"));
        std::fs::copy(findings.join("register.json"), root.join("register.json"))
            .expect("copy register");
        let register_sha256 = digest(&root.join("register.json"));
        Fixture {
            _directory: directory,
            root,
            register_sha256,
            fixed: FIXED.into(),
            pre: PRE.into(),
        }
    }

    fn evidence(&self, finding: &str, name: &str, body: &Value) -> Value {
        let path = self.root.join("evidence").join(finding).join(name);
        dump_with(&path, body, true, None);
        let mut reference = Dict::new();
        reference.insert("path", format!("evidence/{finding}/{name}").into());
        reference.insert("sha256", digest(&path).into());
        Value::Dict(reference)
    }

    fn body(&self, finding: &str, stage: &str, kind: &str, extra: &[(&str, Value)]) -> Value {
        let mut body = Dict::new();
        body.insert("schema", EVIDENCE_SCHEMA.into());
        body.insert("finding", finding.into());
        body.insert("stage", stage.into());
        body.insert("source_commit", self.fixed.as_str().into());
        body.insert("evidence_kind", kind.into());
        for (key, value) in extra {
            body.insert(*key, value.clone());
        }
        Value::Dict(body)
    }

    fn attestation(&self, stage: &str, evidence: Value) -> Value {
        let mut attestation = Dict::new();
        attestation.insert("stage", stage.into());
        attestation.insert("source_commit", self.fixed.as_str().into());
        attestation.insert("evidence", Value::List(vec![evidence]));
        Value::Dict(attestation)
    }

    /// A complete confirmed-defect closure for a non-live package.
    fn defect(&self, finding: &str, closed: bool) -> Value {
        let source = self.evidence(
            finding,
            "source.json",
            &self.body(
                finding,
                "source_implementation",
                "source-trace",
                &[("paths", strs(&["crates/jope-engine/src/gp.rs"]))],
            ),
        );
        let fixture = self.evidence(
            finding,
            "fixture.json",
            &self.body(
                finding,
                "fixture_proof",
                "synthetic-fixture",
                &[
                    ("pre_fix_commit", self.pre.as_str().into()),
                    ("failed_tests", strs(&[&format!("{TARGET}::{TEST}_pre")])),
                    (
                        "passed_tests",
                        strs(&[
                            &format!("{TARGET}::{TEST}"),
                            &format!("{TARGET}::{TEST}_negative"),
                        ]),
                    ),
                ],
            ),
        );
        let ci = self.evidence(
            finding,
            "ci.json",
            &self.body(
                finding,
                "hosted_ci",
                "hosted-ci",
                &[
                    ("run_id", 1i64.into()),
                    ("attempt", 1i64.into()),
                    ("head_sha", self.fixed.as_str().into()),
                    ("conclusion", "success".into()),
                ],
            ),
        );
        let mut fixture_stage = self.attestation("fixture_proof", fixture);
        set(
            &mut fixture_stage,
            "tests",
            json(&format!(
                r#"[{{"target": "{TARGET}", "name": "{TEST}_pre", "role": "pre-fix-failure", "pre_fix_commit": "{}"}},
                    {{"target": "{TARGET}", "name": "{TEST}_negative", "role": "negative"}},
                    {{"target": "{TARGET}", "name": "{TEST}", "role": "positive"}}]"#,
                self.pre
            )),
        );
        let mut record = json(&format!(
            r#"{{"finding": "{finding}", "sequence": 1, "disposition": "confirmed-defect",
                "scope": "regression/shared receiving boundary", "closed": {closed},
                "decided_by": {{"role": "independent-reviewer", "identity": "fixture"}}}}"#
        ));
        set(
            &mut record,
            "attestations",
            Value::List(vec![
                self.attestation("source_implementation", source),
                fixture_stage,
                self.attestation("hosted_ci", ci),
            ]),
        );
        record
    }

    fn write_bound(&self, records: Vec<Value>, register_sha256: &str) {
        let mut document = Dict::new();
        document.insert("schema", SCHEMA.into());
        document.insert("register_sha256", register_sha256.into());
        document.insert("records", Value::List(records));
        dump(&self.root.join("dispositions.json"), &Value::Dict(document));
    }

    fn write(&self, records: Vec<Value>) {
        self.write_bound(records, &self.register_sha256.clone());
    }

    fn verify(&self) -> jope_records::Result<Summary> {
        verify(&self.root, false, None)
    }
}

fn stage<'a>(record: &'a mut Value, name: &str) -> &'a mut Value {
    at(record, "attestations")
        .as_list_mut()
        .expect("attestations")
        .iter_mut()
        .find(|a| a.get("stage") == Some(&Value::from(name)))
        .unwrap_or_else(|| panic!("no {name} attestation"))
}

fn first_reference<'a>(record: &'a mut Value, name: &str) -> &'a mut Value {
    crate::nth(at(stage(record, name), "evidence"), 0)
}

#[test]
fn committed_dispositions_verify() {
    let result =
        verify(&layout::findings(&repo()), false, None).expect("committed record verifies");
    assert_eq!(result.entries, 195);
    assert_eq!(result.open + result.closed, 195);
}

#[test]
fn complete_defect_closure_is_accepted() {
    let fixture = Fixture::new();
    fixture.write(vec![fixture.defect("S03-G03", true)]);
    let result = fixture.verify().expect("complete closure");
    assert_eq!((result.closed, result.open), (1, 194));
    assert_eq!(result.closed_findings, ["S03-G03"]);
}

#[test]
fn open_honest_dispositions_are_accepted() {
    let fixture = Fixture::new();
    fixture.write(vec![
        json(
            r#"{"finding": "S08-G09", "sequence": 1, "disposition": "historical-unknown",
                "scope": "historical Elo matches", "closed": false,
                "reason": "original matches not retained",
                "decided_by": {"role": "implementation-owner", "identity": "fixture"}}"#,
        ),
        json(
            r#"{"finding": "S04-G10", "sequence": 1, "disposition": "deferred-non-regression",
                "scope": "classification subcase", "closed": false,
                "deferred_subcases": ["binary", "multiclass"],
                "decided_by": {"role": "implementation-owner", "identity": "fixture"}}"#,
        ),
    ]);
    assert_eq!(fixture.verify().expect("open dispositions").closed, 0);
}

type Mutate = Box<dyn Fn(&Fixture, &mut Value) -> bool>;

fn pop_stage(name: &'static str) -> Mutate {
    Box::new(move |_, record| {
        at(record, "attestations")
            .as_list_mut()
            .expect("attestations")
            .retain(|a| a.get("stage") != Some(&Value::from(name)));
        true
    })
}

fn edit_evidence(name: &'static str, changes: Vec<(&'static str, Value)>) -> Mutate {
    Box::new(move |fixture, record| {
        let reference = first_reference(record, name);
        let path = fixture
            .root
            .join(reference.get("path").and_then(Value::as_str).expect("path"));
        let mut body = load(&path);
        for (key, value) in &changes {
            set(&mut body, key, value.clone());
        }
        dump(&path, &body);
        set(reference, "sha256", digest(&path).into());
        true
    })
}

fn update_reference(name: &'static str, key: &'static str, value: Value) -> Mutate {
    Box::new(move |_, record| {
        set(first_reference(record, name), key, value.clone());
        true
    })
}

fn update_record(changes: Vec<(&'static str, Value)>) -> Mutate {
    Box::new(move |_, record| {
        for (key, value) in &changes {
            set(record, key, value.clone());
        }
        true
    })
}

fn symlink_evidence(fixture: &Fixture, record: &mut Value) -> bool {
    let reference = first_reference(record, "hosted_ci");
    let path = fixture
        .root
        .join(reference.get("path").and_then(Value::as_str).expect("path"));
    let target = path.with_file_name("real.json");
    std::fs::rename(&path, &target).expect("rename");
    match std::os::unix::fs::symlink("real.json", &path) {
        Ok(()) => true,
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::Unsupported
            ) =>
        {
            false
        }
        Err(error) => panic!("symlink: {error}"),
    }
}
