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

/// Run `model mcp` over `models/` and return the response to one `tools/call`.
fn model_tool(
    exe: &Path,
    dir: &Path,
    name: &str,
    arguments: serde_json::Value,
) -> serde_json::Value {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new(exe)
        .args(["model", "mcp", "--root", "models"])
        .current_dir(dir)
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for message in [
        serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"fixture","version":"1"}}}),
        serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":name,"arguments":arguments}}),
    ] {
        writeln!(stdin, "{message}").unwrap();
    }
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    assert_eq!(
        code(&output),
        0,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    serde_json::from_str(text.lines().nth(1).unwrap()).unwrap()
}

#[path = "single_file/tests_01.rs"]
mod tests_01;
#[path = "single_file/tests_02.rs"]
mod tests_02;
#[path = "single_file/tests_03.rs"]
mod tests_03;
#[path = "single_file/tests_04.rs"]
mod tests_04;
#[path = "single_file/tests_05.rs"]
mod tests_05;
#[path = "single_file/tests_06.rs"]
mod tests_06;
#[path = "single_file/tests_07.rs"]
mod tests_07;

#[cfg(feature = "dope-encoding")]
#[path = "support/landmark_product.rs"]
mod landmark_product;
