use super::*;

#[test]
fn a_given_source_commit_is_recorded_and_git_is_never_spawned() {
    let temp = tempfile::tempdir().unwrap();
    let marker = temp.path().join("git-invocations");
    let bin = fake_git(temp.path(), &marker);
    let commit = "0123456789abcdef0123456789abcdef01234567";

    let with = temp.path().join("with");
    let output = jope()
        .env("PATH", &bin)
        .args(run_args(&with, "evaluations=2,seconds=60"))
        .args(["--source-commit", commit])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!marker.exists(), "git was spawned despite --source-commit");
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(with.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["git_sha"], commit);

    // Positive control: without the flag the run does ask git (which fails here, so the
    // receipt simply carries no git_sha).
    let without = temp.path().join("without");
    let output = jope()
        .env("PATH", &bin)
        .args(run_args(&without, "evaluations=2,seconds=60"))
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(marker.exists(), "the control run never asked git");

    // A malformed commit is refused before anything is written.
    let bad = temp.path().join("bad");
    let output = jope()
        .args(run_args(&bad, "evaluations=2,seconds=60"))
        .args(["--source-commit", "HEAD"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(!bad.exists());
}
