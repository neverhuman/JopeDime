//! Print the jain-math executable op snapshot
//! (`parallel/assets/grammar/jain-math-executable-9fa594e2.json`) from the
//! feat-math registry at the git rev pinned in Cargo.toml. Registry line
//! numbers are read from the pinned checkout's `registry.rs`, located with
//! `cargo metadata`.
use feat_math::registry::{operator_registry, registry_hash, OperatorDescriptor, REGISTRY_VERSION};
use serde::Serialize;
use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;

const REPOSITORY: &str = "https://git.neverhuman.org/git/veox/jain-math.git";
const REGISTRY_PATH: &str = "crates/feat-math/src/registry.rs";
const ADMISSION_RULE: &str = "feat_math::registry::operator(name) is Some(d) && d.lifecycle.executable && !d.lifecycle.quarantined (jain-core crates/feat-core/src/jope_invention.rs admit() at a348ef8)";

#[derive(Serialize)]
struct Snapshot {
    schema: &'static str,
    engine: &'static str,
    admission_rule: &'static str,
    provenance: Provenance,
    executable: Vec<Entry>,
    not_executable: Vec<Entry>,
}

#[derive(Serialize)]
struct Provenance {
    repository: &'static str,
    rev: String,
    registry_path: &'static str,
    operators_table_line: usize,
    operator_lookup_line: usize,
    registry_version: &'static str,
    registry_hash: &'static str,
}

#[derive(Serialize)]
struct Entry {
    id: u64,
    name: &'static str,
    role: Value,
    lifecycle: &'static str,
    registry_line: usize,
}

/// Source directory and full git rev of the resolved feat-math package.
fn feat_math_checkout() -> (PathBuf, String) {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let out = Command::new(cargo)
        .args([
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--manifest-path",
        ])
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .output()
        .expect("run cargo metadata");
    assert!(out.status.success(), "cargo metadata failed");
    let meta: Value = serde_json::from_slice(&out.stdout).expect("parse cargo metadata");
    let package = meta["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "feat-math")
        .expect("feat-math package");
    let source = package["source"]
        .as_str()
        .expect("feat-math is a git dependency");
    assert!(
        source.starts_with(&format!("git+{REPOSITORY}?rev=")),
        "{source}"
    );
    let rev = source.rsplit('#').next().unwrap().to_string();
    assert_eq!(rev.len(), 40, "{source}");
    let manifest = PathBuf::from(package["manifest_path"].as_str().unwrap());
    (manifest.parent().unwrap().to_path_buf(), rev)
}

fn line_of(lines: &[&str], predicate: impl Fn(&str) -> bool, what: &str) -> usize {
    let hits: Vec<usize> = (0..lines.len()).filter(|&i| predicate(lines[i])).collect();
    assert_eq!(hits.len(), 1, "{what}: {hits:?}");
    hits[0] + 1
}

fn lifecycle(d: &OperatorDescriptor) -> &'static str {
    let l = d.lifecycle;
    match (l.executable, l.generatable, l.quarantined) {
        (true, true, false) => "generatable",
        (true, false, false) => "compatibility",
        (false, false, true) => "quarantined",
        other => panic!("{}: unexpected lifecycle {other:?}", d.name),
    }
}

fn main() {
    let (crate_dir, rev) = feat_math_checkout();
    let source = std::fs::read_to_string(crate_dir.join("src/registry.rs")).expect("registry.rs");
    let lines: Vec<&str> = source.lines().collect();
    let table = line_of(
        &lines,
        |l| l.starts_with("pub static OPERATORS:"),
        "OPERATORS",
    );
    let lookup = line_of(
        &lines,
        |l| l.starts_with("pub fn operator(name"),
        "operator()",
    );
    let mut snapshot = Snapshot {
        schema: "parallel.jain_math_executable_ops.v1",
        engine: "featmath",
        admission_rule: ADMISSION_RULE,
        provenance: Provenance {
            repository: REPOSITORY,
            rev,
            registry_path: REGISTRY_PATH,
            operators_table_line: table,
            operator_lookup_line: lookup,
            registry_version: REGISTRY_VERSION,
            registry_hash: registry_hash(),
        },
        executable: Vec::new(),
        not_executable: Vec::new(),
    };
    for d in operator_registry() {
        let id = u64::from(d.id.0);
        let id_line = format!("{id},");
        // 1-based line of the `op!(` whose next line is this descriptor's id.
        let registry_line: Vec<usize> = (table..lines.len())
            .filter(|&i| lines[i - 1].trim() == "op!(" && lines[i].trim() == id_line)
            .collect();
        assert_eq!(registry_line.len(), 1, "{}: {registry_line:?}", d.name);
        let life = lifecycle(d);
        let entry = Entry {
            id,
            name: d.name,
            role: serde_json::to_value(d.role).unwrap(),
            lifecycle: life,
            registry_line: registry_line[0],
        };
        if life == "quarantined" {
            snapshot.not_executable.push(entry);
        } else {
            snapshot.executable.push(entry);
        }
    }
    println!("{}", serde_json::to_string_pretty(&snapshot).unwrap());
}
