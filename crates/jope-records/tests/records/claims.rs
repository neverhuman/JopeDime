//! Port of `test_verify_claims.py`, plus the W12-T07 synthetic identity rule.

use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};

use jope_records::claims::{revoked_evidence, verify, Summary, SCHEMA};
use jope_records::layout;
use jope_records::pyjson::{Dict, Value};

use crate::{assert_refused, at, digest, dump, json, nth, repo, set, strs};

const COMMIT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

/// A repository-local evidence graph: root -> child -> grandchild, plus an unrelated node.
struct Fixture {
    _directory: tempfile::TempDir,
    root: PathBuf,
    path: PathBuf,
    nodes: Dict,
}

impl Fixture {
    fn new() -> Self {
        let directory = crate::scratch();
        let root = directory.path().to_path_buf();
        let path = root.join("docs/internal/core-ip/claims/claims.json");
        std::fs::create_dir_all(path.parent().expect("parent")).expect("claims directory");
        let mut nodes = Dict::new();
        for (name, kind, parents) in [
            ("E-root", "synthetic-fixture", vec![]),
            ("E-child", "hosted-ci", vec!["E-root"]),
            ("E-grandchild", "live-host", vec!["E-child"]),
            ("E-other", "measurement", vec![]),
        ] {
            let file = root.join("evidence").join(format!("{name}.json"));
            dump(&file, &json(&format!(r#"{{"node": "{name}"}}"#)));
            let mut node = Dict::new();
            node.insert("kind", kind.into());
            node.insert("location", "repository".into());
            node.insert("path", format!("evidence/{name}.json").into());
            node.insert("sha256", digest(&file).into());
            node.insert("source_commit", COMMIT.into());
            node.insert("subject", "subject".into());
            node.insert("scope", "scope".into());
            node.insert("depends_on", strs(&parents));
            nodes.insert(name, Value::Dict(node));
        }
        Fixture {
            _directory: directory,
            root,
            path,
            nodes,
        }
    }

    fn claim(&self, cid: &str, stage: &str, evidence: &[&str], status: &str) -> Value {
        let mut claim = json(&format!(
            r#"{{"id": "{cid}", "statement": "s", "subject": "subject", "metric": "m", "units": "u",
                "denominator": "d", "comparison": null, "baseline": null, "scope": "scope",
                "source_commit": "{COMMIT}", "observed_utc": "2026-09-26T00:00:00Z",
                "uncertainty": "exact", "limitations": ["fixture"], "stage": "{stage}",
                "evidence": [], "status": "{status}"}}"#
        ));
        set(&mut claim, "evidence", strs(evidence));
        claim
    }

    fn document(&self) -> Value {
        let mut document = Dict::new();
        document.insert("schema", SCHEMA.into());
        document.insert("evidence", Value::Dict(self.nodes.clone()));
        document.insert("revocations", Value::List(vec![]));
        document.insert(
            "claims",
            Value::List(vec![
                self.claim("C-live", "live", &["E-grandchild"], "active"),
                self.claim("C-measure", "measurement", &["E-other"], "active"),
            ]),
        );
        Value::Dict(document)
    }

    fn verify(&self, document: &Value) -> jope_records::Result<Summary> {
        dump(&self.path, document);
        verify(&self.path, Some(&self.root), false)
    }
}

fn node<'a>(document: &'a mut Value, name: &str) -> &'a mut Value {
    at(at(document, "evidence"), name)
}

fn claim_at(document: &mut Value, index: usize) -> &mut Value {
    nth(at(document, "claims"), index)
}

const MARK: &str = r#"{"synthetic_identity": "jope.core-ip-conformance/v1", "node": "x"}"#;

/// Write marked (or given) bytes for a node, optionally moving it to `path`, and re-pin it.
fn mark_with(
    fixture: &Fixture,
    document: &mut Value,
    name: &str,
    path: Option<&str>,
    bytes: &[u8],
) {
    let node = node(document, name);
    if let Some(path) = path {
        set(node, "path", path.into());
    }
    let target = fixture
        .root
        .join(node.get("path").and_then(Value::as_str).expect("path"));
    std::fs::create_dir_all(target.parent().expect("parent")).expect("evidence directory");
    std::fs::write(&target, bytes).expect("evidence bytes");
    set(node, "sha256", digest(&target).into());
}

fn mark(fixture: &Fixture, document: &mut Value, name: &str) {
    mark_with(fixture, document, name, None, MARK.as_bytes());
}

fn relabel(document: &mut Value, name: &str, kind: &str) {
    set(node(document, name), "kind", kind.into());
}

const CONFORMANCE: &str = "docs/internal/core-ip/conformance/claims/evidence/x.json";

mod synthetic;
mod verification;
