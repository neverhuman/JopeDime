use super::*;

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
