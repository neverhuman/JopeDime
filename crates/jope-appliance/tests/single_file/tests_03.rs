use super::*;

#[test]
fn regression_output_collision_preserves_the_input_file() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.ridge"]);
    train_csv(dir.path());
    let expected = std::fs::read(dir.path().join("train.csv")).unwrap();
    // The shell becomes jopedime with exec, retaining its PID. Its input is deliberately
    // the atomic output's temporary name, so create_new must refuse it without deleting it.
    let child = Command::new("/bin/sh")
        .args([
            "-c",
            r#"
input=".model.jope.jopedime-$$.tmp"
while IFS= read -r line; do printf '%s\n' "$line"; done < train.csv > "$input"
exec "$1" fit --train "$input" --target y --output model.jope --method ridge
"#,
            "jopedime-regression",
        ])
        .arg(&exe)
        .current_dir(dir.path())
        .env_clear()
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let input = dir
        .path()
        .join(format!(".model.jope.jopedime-{}.tmp", child.id()));
    let output = child.wait_with_output().unwrap();
    assert_eq!(
        code(&output),
        7,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read(&input).expect("an output refusal must preserve its input"),
        expected
    );
    assert!(!dir.path().join("model.jope").exists());
}

#[test]
fn regression_receipt_failure_survives_a_closed_json_pipe() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.ridge"]);
    train_csv(dir.path());
    std::fs::create_dir(dir.path().join("receipt-dir")).unwrap();
    let output = closed_stdout(
        &exe,
        dir.path(),
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--output",
            "model.jope",
            "--method",
            "ridge",
            "--receipt",
            "receipt-dir",
            "--json",
        ],
    );
    assert!(
        dir.path().join("model.jope").is_file(),
        "the fit itself must succeed"
    );
    assert!(dir.path().join("receipt-dir").is_dir());
    assert_eq!(code(&output), 7, "receipt failure must survive EPIPE");
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
}

#[test]
fn regression_doctor_failure_survives_a_closed_json_pipe() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["neural.chimera.binary"]);
    let output = closed_stdout(&exe, dir.path(), &["doctor", "--json"]);
    assert_eq!(
        code(&output),
        3,
        "unexecutable package methods must still fail doctor"
    );
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
}
