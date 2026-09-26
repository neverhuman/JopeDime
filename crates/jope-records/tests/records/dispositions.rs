//! Port of `test_verify_dispositions.py`, plus `--require-git` and the W12-T07 synthetic
//! identity rule.

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

type Mutate = Box<dyn Fn(&Fixture, &mut Value)>;

fn pop_stage(name: &'static str) -> Mutate {
    Box::new(move |_, record| {
        at(record, "attestations")
            .as_list_mut()
            .expect("attestations")
            .retain(|a| a.get("stage") != Some(&Value::from(name)));
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
    })
}

fn update_reference(name: &'static str, key: &'static str, value: Value) -> Mutate {
    Box::new(move |_, record| set(first_reference(record, name), key, value.clone()))
}

fn update_record(changes: Vec<(&'static str, Value)>) -> Mutate {
    Box::new(move |_, record| {
        for (key, value) in &changes {
            set(record, key, value.clone());
        }
    })
}

fn symlink_evidence(fixture: &Fixture, record: &mut Value) {
    let reference = first_reference(record, "hosted_ci");
    let path = fixture
        .root
        .join(reference.get("path").and_then(Value::as_str).expect("path"));
    let target = path.with_file_name("real.json");
    std::fs::rename(&path, &target).expect("rename");
    std::os::unix::fs::symlink("real.json", &path).expect("symlink");
}

#[test]
fn mutations_are_rejected() {
    let other_finding_ci = || Value::from("evidence/S03-G04/ci.json");
    let mutations: Vec<(&str, Mutate, &str)> = vec![
        (
            "unknown finding",
            update_record(vec![("finding", "S99-G1".into())]),
            "unknown finding",
        ),
        (
            "unsupported disposition",
            update_record(vec![("disposition", "closed".into())]),
            ": disposition$",
        ),
        (
            "digest mismatch",
            update_reference("hosted_ci", "sha256", "0".repeat(64).into()),
            "digest mismatch",
        ),
        (
            "path traversal",
            update_reference(
                "hosted_ci",
                "path",
                "evidence/S03-G03/../../register.json".into(),
            ),
            "path traversal",
        ),
        (
            "other finding directory",
            update_reference("hosted_ci", "path", other_finding_ci()),
            "outside evidence/S03-G03/",
        ),
        (
            "symlinked evidence",
            Box::new(symlink_evidence),
            "evidence symlink",
        ),
        (
            "evidence names another finding",
            edit_evidence("hosted_ci", vec![("finding", "S03-G04".into())]),
            "another finding",
        ),
        (
            "evidence names another stage",
            edit_evidence("hosted_ci", vec![("stage", "fixture_proof".into())]),
            "another stage",
        ),
        (
            "evidence names another commit",
            edit_evidence("hosted_ci", vec![("source_commit", PRE.into())]),
            "another commit",
        ),
        (
            "hosted CI failed",
            edit_evidence("hosted_ci", vec![("conclusion", "failure".into())]),
            "hosted CI evidence binding",
        ),
        (
            "hosted CI other head",
            edit_evidence("hosted_ci", vec![("head_sha", PRE.into())]),
            "hosted CI evidence binding",
        ),
        (
            "synthetic claims hosted CI",
            edit_evidence(
                "hosted_ci",
                vec![("evidence_kind", "synthetic-fixture".into())],
            ),
            "cannot back hosted_ci",
        ),
        (
            "defect without hosted CI",
            pop_stage("hosted_ci"),
            "needs source, fixture and hosted CI",
        ),
        (
            "defect without source",
            pop_stage("source_implementation"),
            "needs source, fixture and hosted CI",
        ),
        (
            "positive not observed",
            edit_evidence("fixture_proof", vec![("passed_tests", Value::List(vec![]))]),
            "not observed passing",
        ),
        (
            "positive also failed",
            edit_evidence(
                "fixture_proof",
                vec![(
                    "failed_tests",
                    strs(&[
                        &format!("{TARGET}::{TEST}_pre"),
                        &format!("{TARGET}::{TEST}"),
                    ]),
                )],
            ),
            "roundtrip not observed passing",
        ),
        (
            "pre-fix not observed",
            edit_evidence("fixture_proof", vec![("failed_tests", Value::List(vec![]))]),
            "pre-fix failure not observed",
        ),
        (
            "pre-fix commit equals fix",
            Box::new(|_, record| {
                let test = crate::nth(at(stage(record, "fixture_proof"), "tests"), 0);
                set(test, "pre_fix_commit", FIXED.into());
            }),
            "distinct pre-fix commit",
        ),
        (
            "no negative test",
            Box::new(|_, record| {
                at(stage(record, "fixture_proof"), "tests")
                    .as_list_mut()
                    .expect("tests")
                    .remove(1);
            }),
            "pre-fix failure, negative and positive",
        ),
        (
            "duplicate stage",
            Box::new(|_, record| {
                let copy = stage(record, "hosted_ci").clone();
                at(record, "attestations")
                    .as_list_mut()
                    .expect("attestations")
                    .push(copy);
            }),
            "duplicate hosted_ci",
        ),
        (
            "historical unknown closed",
            update_record(vec![
                ("disposition", "historical-unknown".into()),
                ("reason", "lost".into()),
            ]),
            "historical-unknown cannot close",
        ),
        (
            "missing evidence closed",
            update_record(vec![
                ("disposition", "missing-execution-evidence".into()),
                ("reason", "none".into()),
            ]),
            "missing-execution-evidence cannot close",
        ),
        (
            "deferral of regression",
            update_record(vec![
                ("disposition", "deferred-non-regression".into()),
                ("closed", false.into()),
                ("deferred_subcases", strs(&["regression"])),
            ]),
            "deferral names only",
        ),
        (
            "live package without live stage",
            Box::new(|fixture, record| *record = fixture.defect("S08-G03", true)),
            "W8 requires live qualification",
        ),
        (
            "no decision owner",
            update_record(vec![("decided_by", Value::Dict(Dict::new()))]),
            "decided_by",
        ),
        (
            "empty scope",
            update_record(vec![("scope", " ".into())]),
            ": scope$",
        ),
    ];
    // Each mutation must fail for its own reason, not an incidental earlier check.
    assert_eq!(mutations.len(), 26);
    for (label, mutate, reason) in mutations {
        let fixture = Fixture::new();
        let mut record = fixture.defect("S03-G03", true);
        mutate(&fixture, &mut record);
        fixture.write(vec![record]);
        let result = fixture.verify();
        assert!(result.is_err(), "{label}: accepted");
        assert_refused(result, reason);
    }
}

/// (label, stage, evidence kind, extra body fields, expected refusal)
type StageCase = (
    &'static str,
    &'static str,
    &'static str,
    Vec<(&'static str, Value)>,
    &'static str,
);

#[test]
fn live_and_promotion_stages_need_their_own_evidence() {
    let cases: [StageCase; 3] = [
        (
            "synthetic live",
            "live_qualification",
            "synthetic-fixture",
            vec![],
            "cannot back live_qualification",
        ),
        (
            "promotion without live",
            "promotion",
            "decision",
            vec![("decision_owner_role", "scientific-decision-owner".into())],
            "promotion without live qualification",
        ),
        (
            "promotion by implementer",
            "promotion",
            "decision",
            vec![("decision_owner_role", "implementation-owner".into())],
            "promotion owner",
        ),
    ];
    for (label, name, kind, extra, reason) in cases {
        let fixture = Fixture::new();
        let mut record = fixture.defect("S03-G03", true);
        let reference = fixture.evidence(
            "S03-G03",
            &format!("{name}.json"),
            &fixture.body("S03-G03", name, kind, &extra),
        );
        at(&mut record, "attestations")
            .as_list_mut()
            .expect("attestations")
            .push(fixture.attestation(name, reference));
        fixture.write(vec![record]);
        let result = fixture.verify();
        assert!(result.is_err(), "{label}: accepted");
        assert_refused(result, reason);
    }
}

#[test]
fn register_binding_and_append_only_history() {
    let fixture = Fixture::new();
    fixture.write_bound(vec![fixture.defect("S03-G03", true)], &"0".repeat(64));
    assert_refused(fixture.verify(), "bind another register");
    let first = fixture.defect("S03-G03", false);
    let mut later = fixture.defect("S03-G03", true);
    set(&mut later, "sequence", 1i64.into());
    fixture.write(vec![first.clone(), later.clone()]);
    assert_refused(fixture.verify(), "append-only");
    set(&mut later, "sequence", 2i64.into());
    fixture.write(vec![first, later]);
    assert_eq!(fixture.verify().expect("appended closure").closed, 1);
}

fn git(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args([
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@invalid",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .output()
        .expect("run git");
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("utf-8")
        .trim()
        .to_owned()
}

#[test]
fn require_git_needs_every_cited_commit() {
    // Addition (the Python suite had no `--require-git` case). A two-commit throwaway
    // repository stands in for the checkout, so the result does not depend on clone depth.
    let repository = scratch();
    git(repository.path(), &["init", "-q"]);
    git(
        repository.path(),
        &["commit", "-q", "--allow-empty", "-m", "pre-fix"],
    );
    let pre = git(repository.path(), &["rev-parse", "HEAD"]);
    git(
        repository.path(),
        &["commit", "-q", "--allow-empty", "-m", "fix"],
    );
    let fixed = git(repository.path(), &["rev-parse", "HEAD"]);

    let mut fixture = Fixture::new();
    fixture.fixed = fixed;
    fixture.pre = pre;
    fixture.write(vec![fixture.defect("S03-G03", true)]);
    let result = verify(&fixture.root, true, Some(repository.path())).expect("cited commits exist");
    assert_eq!(result.closed, 1);
    // Without --require-git, absent commits are not looked up.
    let absent = Fixture::new();
    absent.write(vec![absent.defect("S03-G03", true)]);
    assert_eq!(absent.verify().expect("not looked up").closed, 1);
    let error = verify(&absent.root, true, Some(repository.path())).expect_err("absent commits");
    assert!(
        !error.is_value_error(),
        "a failed git lookup is not a record rule: {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains(&format!("'cat-file', '-e', '{FIXED}^{{commit}}'"))
            && message.contains("returned non-zero exit status"),
        "{message}"
    );
}

/// Marks an evidence body with the public conformance fixtures' synthetic identity.
fn marked(body: &Value) -> Value {
    let mut body = body.clone();
    set(
        &mut body,
        "synthetic_identity",
        "jope.core-ip-conformance/v1".into(),
    );
    body
}

#[test]
fn synthetic_identity_cannot_close_through_hosted_or_live_evidence() {
    // W12-T07 (reproduced by repro.log): every evidence body of a W8 closure carries the
    // conformance synthetic identity while declaring hosted-ci and live-host kinds.
    let closure = |mark_ci: bool, mark_live: bool| {
        let fixture = Fixture::new();
        let finding = "S08-G03";
        let mut record = fixture.defect(finding, true);
        let live_body = fixture.body(finding, "live_qualification", "live-host", &[]);
        let live_body = if mark_live {
            marked(&live_body)
        } else {
            live_body
        };
        let live = fixture.evidence(finding, "live.json", &live_body);
        at(&mut record, "attestations")
            .as_list_mut()
            .expect("attestations")
            .push(fixture.attestation("live_qualification", live));
        for name in ["source_implementation", "fixture_proof", "hosted_ci"] {
            if name == "hosted_ci" && !mark_ci {
                continue;
            }
            let reference = first_reference(&mut record, name);
            let path = fixture
                .root
                .join(reference.get("path").and_then(Value::as_str).expect("path"));
            dump_with(&path, &marked(&load(&path)), true, None);
            set(reference, "sha256", digest(&path).into());
        }
        fixture.write(vec![record]);
        fixture
    };
    // Every case runs before the verdict, so a failure lists each gap rather than the first.
    let mut accepted = Vec::new();
    for (label, mark_ci, reason) in [
        (
            "marked hosted CI and live bodies",
            true,
            "^S08-G03: synthetic identity cannot be hosted-ci evidence$",
        ),
        (
            "marked live body",
            false,
            "^S08-G03: synthetic identity cannot be live-host evidence$",
        ),
    ] {
        match closure(mark_ci, true).verify() {
            Ok(summary) => accepted.push(format!("{label}: ACCEPTED {summary:?}")),
            Err(error) => assert_refused::<()>(Err(error), reason),
        }
    }
    assert!(
        accepted.is_empty(),
        "synthetic identity closed a finding:\n{}",
        accepted.join("\n")
    );
    // Positive control: marked source trace and synthetic fixture evidence keep their stages
    // when hosted CI and live evidence are not synthetic.
    let fixture = closure(false, false);
    assert_eq!(
        fixture
            .verify()
            .expect("real hosted and live evidence")
            .closed,
        1
    );
}

#[test]
fn synthetic_identity_cannot_be_declared_real_corpus_evidence() {
    // W12-T07 addition: a synthetic-marked fixture cannot pass as real-corpus fixture proof.
    let fixture = Fixture::new();
    let mut record = fixture.defect("S03-G03", false);
    let reference = first_reference(&mut record, "fixture_proof");
    let path = fixture
        .root
        .join(reference.get("path").and_then(Value::as_str).expect("path"));
    let mut body = marked(&load(&path));
    set(&mut body, "evidence_kind", "real-corpus-fixture".into());
    dump_with(&path, &body, true, None);
    set(reference, "sha256", digest(&path).into());
    fixture.write(vec![record]);
    assert_refused(
        fixture.verify(),
        "^S03-G03: synthetic identity cannot be real-corpus-fixture evidence$",
    );
}
