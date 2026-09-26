//! W12-T01: the capability/KPI sentence inventory is generated from the verified claims record.
//! Added (no Python predecessor).

use jope_records::claim_inventory::{check_or_write, render, target};
use jope_records::layout;
use jope_records::pyjson::Value;

use crate::{assert_refused, at, digest, dump, json, load, nth, repo, set};

const COMMIT: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

/// A one-claim record over one fixture evidence file, under a scratch repository.
struct Record {
    _directory: tempfile::TempDir,
    root: std::path::PathBuf,
    path: std::path::PathBuf,
    original: Value,
}

impl Record {
    fn new() -> Self {
        let directory = crate::scratch();
        let root = directory.path().to_path_buf();
        let path = root.join("docs/internal/core-ip/claims/claims.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let evidence = root.join("evidence/E-fixture.json");
        dump(&evidence, &json(r#"{"node": "E-fixture"}"#));
        let document = json(&format!(
            r#"{{"schema": "jope.core-ip-claims/v1", "revocations": [],
                "evidence": {{"E-fixture": {{"kind": "synthetic-fixture", "location": "repository",
                    "path": "evidence/E-fixture.json", "sha256": "{}", "source_commit": "{COMMIT}",
                    "subject": "the subject", "scope": "the scope", "depends_on": []}}}},
                "claims": [{{"id": "C-one", "statement": "The subject meets the contract | exactly.",
                    "subject": "the subject", "metric": "outcome", "units": "tests",
                    "denominator": "3 fixture cases", "comparison": {{"pre_fix": "{COMMIT}"}},
                    "baseline": null, "scope": "the scope", "source_commit": "{COMMIT}",
                    "observed_utc": "2026-09-26T00:00:00Z", "uncertainty": "exact",
                    "limitations": ["Fixture only.", "Second limit."], "stage": "fixture",
                    "evidence": ["E-fixture"], "status": "active"}}]}}"#,
            digest(&evidence)
        ));
        dump(&path, &document);
        Record {
            _directory: directory,
            root,
            original: load(&path),
            path,
        }
    }

    /// The original record (each render rewrites the file with the document it renders).
    fn document(&self) -> Value {
        self.original.clone()
    }

    fn render(&self, document: &Value) -> jope_records::Result<String> {
        dump(&self.path, document);
        render(&self.path, Some(&self.root)).map(|(text, _)| text)
    }
}

#[test]
fn the_committed_inventory_is_generated_from_the_verified_claims() {
    let claims = layout::claims(&repo());
    let (text, summary) = render(&claims, None).expect("committed claims render");
    check_or_write(&target(&claims), &text, false).expect("INVENTORY.md is current");
    let document = load(&claims);
    let ids: Vec<String> = document
        .get("claims")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|claim| claim.get("id").and_then(Value::as_str).unwrap().to_owned())
        .collect();
    assert!(!ids.is_empty());
    for id in &ids {
        assert!(text.contains(&format!("### {id}\n")), "{id} missing");
    }
    assert_eq!(summary.get("claims"), Some(&Value::from(ids.len())));
}

#[test]
fn every_binding_is_rendered_with_the_sentence() {
    let record = Record::new();
    let text = record.render(&record.document()).unwrap();
    for needle in [
        "## Publishable (active)",
        "### C-one",
        r"> The subject meets the contract \| exactly.",
        "| Subject | the subject |",
        "| Metric | outcome |",
        "| Units | tests |",
        "| Denominator | 3 fixture cases |",
        "| Baseline | none |",
        "| Input scope | the scope |",
        "| Observed (UTC) | 2026-09-26T00:00:00Z |",
        "| Evidence | E-fixture (synthetic-fixture) |",
        "- Fixture only.",
        "- Second limit.",
        "| fixture | active | 1 |",
    ] {
        assert!(text.contains(needle), "{needle:?} missing from\n{text}");
    }
    assert!(text.contains(&format!(r#"| Comparison | {{"pre_fix": "{COMMIT}"}} |"#)));
    // Deterministic: the same record renders the same bytes.
    assert_eq!(record.render(&record.document()).unwrap(), text);
}

/// A changed scope, subject, source or denominator is a different sentence; it cannot inherit
/// the old one's evidence, and a withdrawn claim is never listed as publishable.
#[test]
fn changed_bindings_do_not_inherit_and_withdrawn_claims_are_not_publishable() {
    let record = Record::new();
    let original = record.render(&record.document()).unwrap();
    // The claim alone widened: its evidence no longer covers it, so nothing renders.
    let mut widened = record.document();
    set(
        nth(at(&mut widened, "claims"), 0),
        "scope",
        "a wider scope".into(),
    );
    assert_refused(
        record.render(&widened),
        "subject, scope or source differs from evidence",
    );
    let mut other_source = record.document();
    set(
        nth(at(&mut other_source, "claims"), 0),
        "source_commit",
        "c".repeat(40).as_str().into(),
    );
    assert_refused(
        record.render(&other_source),
        "subject, scope or source differs from evidence",
    );
    // A new denominator renders as a different sentence, never as the old one.
    let mut denominator = record.document();
    set(
        nth(at(&mut denominator, "claims"), 0),
        "denominator",
        "4 fixture cases".into(),
    );
    let changed = record.render(&denominator).unwrap();
    assert_ne!(changed, original);
    assert!(changed.contains("| Denominator | 4 fixture cases |"));
    assert!(!changed.contains("| Denominator | 3 fixture cases |"));
    // Revoked: listed as withdrawn only.
    let mut revoked = record.document();
    set(
        &mut revoked,
        "revocations",
        json(
            r#"[{"node": "E-fixture", "issuer": "reviewer", "cause": "superseded fixture",
                 "effective_utc": "2026-09-26T01:00:00Z", "sequence": 1}]"#,
        ),
    );
    set(
        nth(at(&mut revoked, "claims"), 0),
        "status",
        "revoked".into(),
    );
    let withdrawn = record.render(&revoked).unwrap();
    let publishable = withdrawn
        .split("## Withdrawn")
        .next()
        .expect("publishable section");
    assert!(!publishable.contains("### C-one"), "{withdrawn}");
    assert!(withdrawn
        .split("## Withdrawn")
        .nth(1)
        .unwrap()
        .contains("### C-one"));
    assert!(withdrawn.contains("| fixture | revoked | 1 |"));
}

#[test]
fn a_stale_inventory_is_refused_and_rewritten_only_on_request() {
    let record = Record::new();
    let text = record.render(&record.document()).unwrap();
    let inventory = target(&record.path);
    std::fs::write(&inventory, "stale\n").unwrap();
    assert_refused(
        check_or_write(&inventory, &text, false),
        "INVENTORY.md is stale; regenerate with --write",
    );
    check_or_write(&inventory, &text, true).unwrap();
    check_or_write(&inventory, &text, false).unwrap();
}
