use std::{fs, path::Path, process::Command};
pub fn enabled() -> bool {
    std::env::var("JOPE_GO_PARITY").as_deref() == Ok("1")
}
fn command(root: &Path, program: &str, args: &[&str]) {
    let out = Command::new(program)
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{program} {args:?}:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    eprint!("{}", String::from_utf8_lossy(&out.stdout));
}
pub fn run(
    root: &Path,
    source: &str,
    kernel: &str,
    input: &serde_json::Value,
    expected: &[f64],
    fitted: bool,
) {
    if !enabled() {
        return;
    }
    fs::create_dir_all(root).unwrap();
    for (p, s) in [
        ("model.go", source),
        ("model.go.kernel.rs", kernel),
        ("go.mod", include_str!("../../src/go/go.mod.txt")),
        ("go.sum", include_str!("../../src/go/go.sum.txt")),
    ] {
        fs::write(root.join(p), s).unwrap();
    }
    fs::write(root.join("input.json"), serde_json::to_vec(input).unwrap()).unwrap();
    fs::write(
        root.join("expected.json"),
        serde_json::to_vec(expected).unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("qualification_test.go"),
        include_str!("qualification.go.txt")
            .replace("__FITTED__", if fitted { "true" } else { "false" }),
    )
    .unwrap();
    command(
        root,
        "rustc",
        &[
            "--edition=2021",
            "--crate-name",
            "model_kernel",
            "--target",
            "wasm32-unknown-unknown",
            "--crate-type",
            "cdylib",
            "-C",
            "opt-level=3",
            "-C",
            "panic=abort",
            "model.go.kernel.rs",
            "-o",
            "model.wasm",
        ],
    );
    command(
        root,
        "go",
        &[
            "test",
            "-mod=readonly",
            "-count=1",
            "-timeout=30m",
            "-v",
            "./...",
        ],
    );
}
