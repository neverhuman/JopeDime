//! Print the executable-op snapshot consumed by the engine
//! (`crates/jope-engine/assets/grammar/jain-math-executable-9fa594e2.json`) from THIS crate's
//! registry. Provenance (origin repository, rev, registry path) is copied from the committed
//! snapshot so the output stays byte-identical to the snapshot taken when the registry was
//! imported; registry line numbers are recomputed from `src/registry.rs`.
//!
//! Byte check (ADR-0002): `cargo run -p feat-math --example registry_dump | cmp - crates/jope-engine/assets/grammar/jain-math-executable-9fa594e2.json`
use feat_math::registry::{operator_registry, registry_hash, OperatorDescriptor, REGISTRY_VERSION};
use serde::Serialize;
use serde_json::Value;
use std::path::PathBuf;

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
    repository: String,
    rev: String,
    registry_path: String,
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

/// This crate's directory and the provenance recorded in the committed snapshot.
fn feat_math_checkout() -> (PathBuf, Provenance) {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let snapshot_path =
        crate_dir.join("../jope-engine/assets/grammar/jain-math-executable-9fa594e2.json");
    let snapshot: Value =
        serde_json::from_str(&std::fs::read_to_string(&snapshot_path).expect("committed snapshot"))
            .expect("snapshot parses");
    let p = &snapshot["provenance"];
    let provenance = Provenance {
        repository: p["repository"].as_str().expect("repository").to_owned(),
        rev: p["rev"].as_str().expect("rev").to_owned(),
        registry_path: p["registry_path"]
            .as_str()
            .expect("registry_path")
            .to_owned(),
        operators_table_line: 0,
        operator_lookup_line: 0,
        registry_version: REGISTRY_VERSION,
        registry_hash: registry_hash(),
    };
    (crate_dir, provenance)
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
    let (crate_dir, mut provenance) = feat_math_checkout();
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
        provenance: {
            provenance.operators_table_line = table;
            provenance.operator_lookup_line = lookup;
            provenance
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
