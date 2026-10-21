//! The single file is the whole product (SPECIFICATION.md §14.2, §17.1): a packaged `jopedime`
//! copied alone into an empty directory, run with an empty environment, fits, saves, reloads and
//! predicts; every command authenticates its own package; each failure class has its exit status.
use jope_assets::verify::development_key;
use jope_assets::writer::{assemble_development, stage, write_new, EntryInput, StageOptions};
use jope_assets::{Codec, License, Scope};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_jopedime");

fn package(dir: &Path, methods: &[&str]) -> PathBuf {
    let code = std::fs::read(BIN).unwrap();
    let notice = EntryInput {
        role: "notice".into(),
        component: "jopedime".into(),
        path: "LICENSE".into(),
        origin: "repository:LICENSE".into(),
        license: License {
            spdx: "Apache-2.0".into(),
            redistribution: "permitted".into(),
        },
        backends: Vec::new(),
        codec: Codec::Zstd,
        bytes: b"Apache License, Version 2.0 (fixture notice)\n".repeat(20),
    };
    let staged = stage(
        &code,
        vec![notice],
        &StageOptions {
            product: "jopedime".into(),
            product_version: env!("CARGO_PKG_VERSION").into(),
            target: "x86_64-unknown-linux-gnu".into(),
            profile: "fixture".into(),
            scope: Scope::Development,
            source: Default::default(),
            methods: methods.iter().map(|m| (*m).to_owned()).collect(),
            key_id: development_key().id,
            frame_bytes: 1 << 20,
            level: 3,
        },
    )
    .unwrap();
    let path = dir.join("jopedime");
    write_new(&assemble_development(&code, &staged).unwrap(), &path).unwrap();
    path
}

fn train_csv(dir: &Path) {
    let mut text = String::from("x1,color,y\n");
    for i in 0..80 {
        let x = (i as f64 * 0.31).cos() * 2.0;
        let color = ["red", "blue", "NA"][i % 3];
        let y = 2.0 * x + if color == "red" { 1.0 } else { 0.0 } + (i % 5) as f64 * 0.01;
        text.push_str(&format!("{x:.5},{color},{y:.5}\n"));
    }
    std::fs::write(dir.join("train.csv"), text).unwrap();
    std::fs::write(
        dir.join("rows.csv"),
        "color,x1\nred,0.5\nblue,\npurple,-1.0\n",
    )
    .unwrap();
}

/// Run in `dir` with an empty environment: no PATH, HOME or JOPE_* variables.
fn run(exe: &Path, dir: &Path, args: &[&str]) -> Output {
    Command::new(exe)
        .args(args)
        .current_dir(dir)
        .env_clear()
        .output()
        .unwrap()
}

fn run_receipt_errors(receipt: &serde_json::Value) -> Vec<String> {
    let schema: serde_json::Value = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../schemas/jope.appliance-run.v1.schema.json"
        ))
        .unwrap(),
    )
    .unwrap();
    jope_datasets::schema::validate(&schema, receipt)
}

fn code(output: &Output) -> i32 {
    output.status.code().unwrap_or(-1)
}

fn ok(output: Output) -> Output {
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

#[test]
fn a_packaged_copy_fits_saves_reloads_and_predicts_alone() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["auto", "rust.ridge"]);
    train_csv(dir.path());
    let version: serde_json::Value =
        serde_json::from_slice(&ok(run(&exe, dir.path(), &["--version", "--json"])).stdout)
            .unwrap();
    assert_eq!(version["package"]["profile"], "fixture");
    assert_eq!(version["trusts_development_key"], true);
    let doctor: serde_json::Value =
        serde_json::from_slice(&ok(run(&exe, dir.path(), &["doctor", "--json"])).stdout).unwrap();
    assert_eq!(doctor["ok"], true);
    assert_eq!(doctor["package"]["status"], "authenticated");

    ok(run(
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
            "auto",
            "--candidates",
            "ridge,histgbm,pls",
            "--receipt",
            "fit.json",
        ],
    ));
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("fit.json")).unwrap()).unwrap();
    assert_eq!(run_receipt_errors(&receipt), Vec::<String>::new());
    assert_eq!(receipt["outcome"], "succeeded");
    assert_eq!(receipt["package"]["profile"], "fixture");
    assert_eq!(receipt["device"]["selected"], "cpu");
    assert_eq!(
        receipt["selection"]["candidates"].as_array().unwrap().len(),
        3
    );
    let selected = receipt["method"]["selected"].as_str().unwrap().to_owned();

    ok(run(
        &exe,
        dir.path(),
        &[
            "predict",
            "--model",
            "model.jope",
            "--input",
            "rows.csv",
            "--output",
            "a.csv",
        ],
    ));
    ok(run(
        &exe,
        dir.path(),
        &[
            "fit-predict",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--input",
            "rows.csv",
            "--output",
            "b.csv",
            "--method",
            &selected,
        ],
    ));
    let a = std::fs::read_to_string(dir.path().join("a.csv")).unwrap();
    assert_eq!(
        a,
        std::fs::read_to_string(dir.path().join("b.csv")).unwrap()
    );
    assert_eq!(a.lines().count(), 4);

    let verified: serde_json::Value =
        serde_json::from_slice(&ok(run(&exe, dir.path(), &["verify-self", "--json"])).stdout)
            .unwrap();
    assert_eq!(verified["ok"], true);
    let licenses = ok(run(&exe, dir.path(), &["licenses"]));
    assert!(String::from_utf8_lossy(&licenses.stdout).contains("fixture notice"));
    // No temporary files are left behind.
    let mut names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "a.csv",
            "b.csv",
            "fit.json",
            "jopedime",
            "model.jope",
            "rows.csv",
            "train.csv"
        ]
    );
}

#[test]
fn every_failure_class_has_its_exit_status() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["auto"]);
    train_csv(dir.path());
    let d = dir.path();
    // 2: input.
    std::fs::write(d.join("bad.csv"), "a,a,y\n1,2,3\n").unwrap();
    assert_eq!(
        code(&run(
            &exe,
            d,
            &["fit", "--train", "bad.csv", "--target", "y", "--output", "m"]
        )),
        2
    );
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "nope",
                "--output",
                "m"
            ]
        )),
        2
    );
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "train.csv"
            ]
        )),
        2
    );
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "m",
                "--method",
                "nope"
            ]
        )),
        2
    );
    assert_eq!(code(&run(&exe, d, &["frobnicate"])), 2);
    // 4: backend.
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "m",
                "--device",
                "cuda:0"
            ]
        )),
        4
    );
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "m",
                "--method",
                "proposal.step60000.regression"
            ]
        )),
        4
    );
    // 5: budget.
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "m",
                "--time-limit-seconds",
                "0"
            ]
        )),
        5
    );
    // A receipt may not overwrite an input or the output; nothing is written.
    let train_before = std::fs::read(d.join("train.csv")).unwrap();
    for receipt in ["train.csv", "m-alias.jope"] {
        assert_eq!(
            code(&run(
                &exe,
                d,
                &[
                    "fit",
                    "--train",
                    "train.csv",
                    "--target",
                    "y",
                    "--output",
                    "m-alias.jope",
                    "--method",
                    "ridge",
                    "--receipt",
                    receipt
                ],
            )),
            2,
            "{receipt}"
        );
    }
    assert_eq!(std::fs::read(d.join("train.csv")).unwrap(), train_before);
    assert!(!d.join("m-alias.jope").exists());
    // A time limit too large for a Duration is an input error, not a panic.
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "m",
                "--time-limit-seconds",
                "1e20"
            ],
        )),
        2
    );
    // 7: output.
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "missing/dir/m",
                "--method",
                "ridge"
            ]
        )),
        7
    );
    // 3: a damaged model file.
    ok(run(
        &exe,
        d,
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--output",
            "m.jope",
            "--method",
            "ridge",
        ],
    ));
    let mut model = std::fs::read(d.join("m.jope")).unwrap();
    let middle = model.len() / 2;
    model[middle] ^= 1;
    std::fs::write(d.join("m.jope"), model).unwrap();
    let out = run(
        &exe,
        d,
        &[
            "predict", "--model", "m.jope", "--input", "rows.csv", "--output", "p.csv",
        ],
    );
    assert_eq!(code(&out), 3, "{}", String::from_utf8_lossy(&out.stderr));
    assert!(!d.join("p.csv").exists(), "a failed run leaves no output");
    // A failed run still writes its receipt.
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "m",
                "--device",
                "cuda:1",
                "--receipt",
                "r.json"
            ]
        )),
        4
    );
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(d.join("r.json")).unwrap()).unwrap();
    assert_eq!(run_receipt_errors(&receipt), Vec::<String>::new());
    assert_eq!(receipt["outcome"], "failed");
    assert_eq!(receipt["error"]["exit_code"], 4);
}

#[test]
fn a_damaged_package_stops_every_command() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["auto"]);
    train_csv(dir.path());
    let good = std::fs::read(&exe).unwrap();
    let footer = good.len() - 128;
    let manifest_offset =
        u64::from_le_bytes(good[footer + 32..footer + 40].try_into().unwrap()) as usize;
    let manifest_len =
        u64::from_le_bytes(good[footer + 40..footer + 48].try_into().unwrap()) as usize;
    let tamper = |at: usize, name: &str| {
        use std::os::unix::fs::PermissionsExt;
        let mut bytes = good.clone();
        bytes[at] ^= 0x20;
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    };
    let manifest = tamper(manifest_offset + manifest_len / 2, "bad-manifest");
    for args in [
        &["doctor"][..],
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--output",
            "m",
            "--method",
            "ridge",
        ],
        &["licenses"],
        &["verify-self"],
    ] {
        assert_eq!(code(&run(&manifest, dir.path(), args)), 3, "{args:?}");
    }
    // A damaged asset frame: commands that need no asset run; reading or verifying it fails.
    let frame = tamper(footer - 64 - manifest_len - 10, "bad-frame");
    assert_eq!(
        code(&run(
            &frame,
            dir.path(),
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "m",
                "--method",
                "ridge"
            ]
        )),
        0
    );
    assert_eq!(code(&run(&frame, dir.path(), &["licenses"])), 3);
    assert_eq!(code(&run(&frame, dir.path(), &["verify-self"])), 3);
    // The executable region: only the full check sees it.
    let code_region = tamper(4096, "bad-code");
    let out = run(&code_region, dir.path(), &["verify-self"]);
    // The flipped byte may also break the program itself; either way it must not verify.
    assert_ne!(code(&out), 0);
}

#[test]
fn a_promised_method_this_build_cannot_run_fails_doctor() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["auto", "neural.chimera.binary"]);
    let out = run(&exe, dir.path(), &["doctor", "--json"]);
    assert_eq!(code(&out), 3);
    let doctor: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        doctor["unexecutable_promised_methods"][0]["id"],
        "neural.chimera.binary"
    );
}

#[test]
fn an_unpackaged_development_build_runs_native_methods() {
    let dir = tempfile::tempdir().unwrap();
    train_csv(dir.path());
    let exe = Path::new(BIN);
    let doctor: serde_json::Value =
        serde_json::from_slice(&ok(run(exe, dir.path(), &["doctor", "--json"])).stdout).unwrap();
    assert_eq!(doctor["package"]["status"], "unpackaged");
    ok(run(
        exe,
        dir.path(),
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--output",
            "m.jope",
            "--method",
            "ridge",
        ],
    ));
    assert_eq!(code(&run(exe, dir.path(), &["verify-self"])), 3);
}

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

fn closed_stdout(exe: &Path, dir: &Path, args: &[&str]) -> Output {
    use std::os::fd::OwnedFd;
    use std::os::unix::net::UnixStream;
    let (writer, reader) = UnixStream::pair().unwrap();
    drop(reader); // EPIPE on the first stdout write, independent of scheduling or buffer size.
    Command::new(exe)
        .args(args)
        .current_dir(dir)
        .env_clear()
        .stdout(std::process::Stdio::from(OwnedFd::from(writer)))
        .stderr(std::process::Stdio::piped())
        .output()
        .unwrap()
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
