//! Cross-language bit-parity oracle for the opt-in portable scalar kernel.

use std::ffi::OsString;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct PortableCommand {
    program: OsString,
    args: Vec<OsString>,
    env: Vec<(OsString, OsString)>,
}

impl PortableCommand {
    fn new(program: impl Into<OsString>, args: Vec<OsString>) -> Self {
        Self {
            program: program.into(),
            args,
            env: Vec::new(),
        }
    }

    fn with_env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }
}

struct PortableBackend {
    language: &'static str,
    source: PathBuf,
    build: PortableCommand,
    run: PortableCommand,
}

fn run_command(language: &str, phase: &str, spec: &PortableCommand) -> Option<Output> {
    let mut command = Command::new(&spec.program);
    command.args(&spec.args);
    command.envs(spec.env.iter().map(|(key, value)| (key, value)));
    match command.output() {
        Ok(output) => {
            assert!(
                output.status.success(),
                "{language} portable {phase} failed with {}:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
            Some(output)
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {
            // ADR-0002 §4: CI sets JOPE_PARITY_STRICT=1 so a missing toolchain is a failure, not a
            // silently skipped backend.
            if std::env::var_os("JOPE_PARITY_STRICT").is_some_and(|v| v == "1") {
                panic!(
                    "portable parity: {language} {phase} tool {:?} is unavailable and JOPE_PARITY_STRICT=1",
                    spec.program
                );
            }
            eprintln!(
                "portable parity SKIP: {language} {phase} tool {:?} is unavailable",
                spec.program
            );
            None
        }
        Err(error) => panic!(
            "could not start {language} portable {phase} tool {:?}: {error}",
            spec.program
        ),
    }
}

fn first_different_line(expected: &[u8], actual: &[u8]) -> usize {
    let first_byte = expected
        .iter()
        .zip(actual)
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| expected.len().min(actual.len()));
    expected[..first_byte.min(expected.len())]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        + 1
}

fn assert_stream_eq(language: &str, expected: &[u8], actual: &[u8]) {
    assert!(
        expected == actual,
        "{language} portable output differs from Rust at line {} ({} versus {} bytes)",
        first_different_line(expected, actual),
        expected.len(),
        actual.len()
    );
}

fn source(root: &Path, name: &str) -> PathBuf {
    root.join(name)
}

#[test]
fn portable_all_helper_backends_match_bits() {
    let source_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/portable");
    let work = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("portable_all_helpers");
    fs::create_dir_all(&work).unwrap();

    let rust_source = source(&source_root, "full_ref.rs");
    let cpp_source = source(&source_root, "full_ref.cpp");
    let js_source = source(&source_root, "full_ref.js");
    let py_source = source(&source_root, "full_ref.py");
    for fixture in [&rust_source, &cpp_source, &js_source, &py_source] {
        let text = fs::read_to_string(fixture).unwrap();
        for forbidden in ["train.csv", "test_features.csv", "read_csv", "dataset"] {
            assert!(
                !text.to_ascii_lowercase().contains(forbidden),
                "portable fixture {} contains forbidden data-access marker {forbidden}",
                fixture.display()
            );
        }
    }

    let rust_bin = work.join("portable_ref_rs");
    let cpp_bin = work.join("portable_ref_cpp");
    let cpp20_bin = work.join("portable_ref_cpp20");
    // JOPE_CXX_MARCH applies to every C++ backend; the standard is pinned per backend (the matrix).
    let march_flags: Vec<OsString> = feat_math::cxx::march_flag(None)
        .into_iter()
        .map(OsString::from)
        .collect();
    let pycache = work.join("pycache");
    let backends = vec![
        PortableBackend {
            language: "Rust",
            source: rust_source.clone(),
            build: PortableCommand::new(
                "rustc",
                vec![
                    "-O".into(),
                    "-o".into(),
                    rust_bin.as_os_str().to_owned(),
                    rust_source.as_os_str().to_owned(),
                ],
            ),
            run: PortableCommand::new(rust_bin.as_os_str().to_owned(), Vec::new()),
        },
        PortableBackend {
            language: "C++17",
            source: cpp_source.clone(),
            build: PortableCommand::new(
                feat_math::cxx::compiler(),
                march_flags
                    .iter()
                    .cloned()
                    .chain([
                        "-O3".into(),
                        "-ffp-contract=off".into(),
                        "-fno-fast-math".into(),
                        "-std=c++17".into(),
                        "-o".into(),
                        cpp_bin.as_os_str().to_owned(),
                        cpp_source.as_os_str().to_owned(),
                    ])
                    .collect(),
            ),
            run: PortableCommand::new(cpp_bin.as_os_str().to_owned(), Vec::new()),
        },
        // The same emitted C++ must also be a valid C++20 program with identical bits (plan: C++20 is a
        // compile target, not a separate emitter).
        PortableBackend {
            language: "C++20",
            source: cpp_source.clone(),
            build: PortableCommand::new(
                feat_math::cxx::compiler(),
                march_flags
                    .iter()
                    .cloned()
                    .chain([
                        "-O3".into(),
                        "-ffp-contract=off".into(),
                        "-fno-fast-math".into(),
                        "-std=c++20".into(),
                        "-o".into(),
                        cpp20_bin.as_os_str().to_owned(),
                        cpp_source.as_os_str().to_owned(),
                    ])
                    .collect(),
            ),
            run: PortableCommand::new(cpp20_bin.as_os_str().to_owned(), Vec::new()),
        },
        PortableBackend {
            language: "Python",
            source: py_source.clone(),
            build: PortableCommand::new(
                "python3",
                vec![
                    "-m".into(),
                    "py_compile".into(),
                    py_source.as_os_str().to_owned(),
                ],
            )
            .with_env("PYTHONPYCACHEPREFIX", pycache.as_os_str().to_owned()),
            run: PortableCommand::new("python3", vec![py_source.as_os_str().to_owned()]),
        },
        PortableBackend {
            language: "JavaScript",
            source: js_source.clone(),
            build: PortableCommand::new(
                "node",
                vec!["--check".into(), js_source.as_os_str().to_owned()],
            ),
            run: PortableCommand::new("node", vec![js_source.as_os_str().to_owned()]),
        },
    ];

    let mut rust_reference: Option<Vec<u8>> = None;
    let mut completed = Vec::new();
    for backend in &backends {
        assert!(
            backend.source.is_file(),
            "missing {} fixture",
            backend.language
        );
        if run_command(backend.language, "build", &backend.build).is_none() {
            continue;
        }
        let Some(first) = run_command(backend.language, "run", &backend.run) else {
            continue;
        };
        let second = run_command(backend.language, "repeat run", &backend.run)
            .expect("a tool available for the first run must remain available");
        assert_stream_eq(backend.language, &first.stdout, &second.stdout);
        if let Some(reference) = &rust_reference {
            assert_stream_eq(backend.language, reference, &first.stdout);
        } else {
            assert_eq!(
                backend.language, "Rust",
                "Rust must establish the reference"
            );
            rust_reference = Some(first.stdout);
        }
        completed.push(backend.language);
    }

    assert_eq!(completed.first(), Some(&"Rust"));
    eprintln!(
        "portable parity completed backends: {}",
        completed.join(", ")
    );
}
