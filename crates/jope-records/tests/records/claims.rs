//! Port of `test_verify_claims.py`, plus the W12-T07 synthetic identity rule.

use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};

use jope_records::claims::{verify, Summary, SCHEMA};
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

#[test]
fn committed_claims_verify() {
    let result = verify(&layout::claims(&repo()), None, false).expect("committed claims verify");
    assert!(result.claims.get("active").copied().unwrap_or(0) >= 1);
}

#[test]
fn valid_graph_and_claims() {
    let fixture = Fixture::new();
    let result = fixture.verify(&fixture.document()).expect("valid graph");
    assert_eq!(
        (result.verified_bytes, result.revoked_evidence.len()),
        (4, 0)
    );
}

#[test]
fn revocation_invalidates_exactly_the_descendants() {
    let fixture = Fixture::new();
    let mut document = fixture.document();
    set(
        &mut document,
        "revocations",
        json(
            r#"[{"node": "E-root", "issuer": "reviewer", "cause": "leak",
                "effective_utc": "2026-09-26T01:00:00Z", "sequence": 1}]"#,
        ),
    );
    assert_refused(fixture.verify(&document), "C-live: cites revoked evidence");
    set(claim_at(&mut document, 0), "status", "revoked".into());
    let result = fixture.verify(&document).expect("revoked claim recorded");
    assert_eq!(
        result.revoked_evidence,
        ["E-child", "E-grandchild", "E-root"]
    );
    // The unrelated claim stays usable.
    let statuses: Vec<(&str, usize)> = result
        .claims
        .iter()
        .map(|(k, v)| (k.as_str(), *v))
        .collect();
    assert_eq!(statuses, [("active", 1), ("revoked", 1)]);
}

type Mutate = fn(&mut Value);

#[test]
fn mutations_are_rejected() {
    let mutations: [(&str, Mutate, &str); 18] = [
        (
            "altered scope",
            |d| set(claim_at(d, 0), "scope", "broader scope".into()),
            "subject, scope or source differs",
        ),
        (
            "altered subject",
            |d| set(claim_at(d, 0), "subject", "other model".into()),
            "subject, scope or source differs",
        ),
        (
            "other source",
            |d| set(claim_at(d, 0), "source_commit", "b".repeat(40).into()),
            "subject, scope or source differs",
        ),
        (
            "stage without evidence kind",
            |d| set(claim_at(d, 0), "stage", "promotion".into()),
            "no promotion evidence",
        ),
        (
            "synthetic backs live",
            |d| {
                set(
                    claim_at(d, 0),
                    "evidence",
                    strs(&["E-root", "E-grandchild"]),
                )
            },
            "synthetic evidence cannot support a live claim",
        ),
        (
            "unknown evidence",
            |d| set(claim_at(d, 0), "evidence", strs(&["E-missing"])),
            "evidence references",
        ),
        (
            "missing denominator",
            |d| set(claim_at(d, 0), "denominator", "".into()),
            "incomplete binding",
        ),
        (
            "no limitations",
            |d| set(claim_at(d, 0), "limitations", Value::List(vec![])),
            "limitations",
        ),
        (
            "unknown status",
            |d| set(claim_at(d, 0), "status", "passed".into()),
            "status",
        ),
        (
            "extra field",
            |d| set(claim_at(d, 0), "verified", true.into()),
            "claim fields",
        ),
        (
            "duplicate claim",
            |d| {
                let copy = claim_at(d, 0).clone();
                at(d, "claims").as_list_mut().expect("claims").push(copy);
            },
            "duplicate claim",
        ),
        (
            "digest mismatch",
            |d| set(node(d, "E-other"), "sha256", "0".repeat(64).into()),
            "evidence digest mismatch",
        ),
        (
            "path traversal",
            |d| {
                set(
                    node(d, "E-other"),
                    "path",
                    "../evidence/E-other.json".into(),
                )
            },
            "path traversal",
        ),
        (
            "cycle",
            |d| set(node(d, "E-root"), "depends_on", strs(&["E-grandchild"])),
            "dependency cycle",
        ),
        (
            "unknown parent",
            |d| set(node(d, "E-root"), "depends_on", strs(&["E-ghost"])),
            "unknown parent",
        ),
        (
            "unknown kind",
            |d| set(node(d, "E-other"), "kind", "vibes".into()),
            ": kind",
        ),
        (
            "revoke unknown",
            |d| {
                at(d, "revocations").as_list_mut().expect("revocations").push(json(
                    r#"{"node": "E-ghost", "issuer": "i", "cause": "c", "effective_utc": "t", "sequence": 1}"#,
                ));
            },
            "revocation of unknown",
        ),
        (
            "revocation not append-only",
            |d| {
                let revocations = at(d, "revocations").as_list_mut().expect("revocations");
                revocations.push(json(
                    r#"{"node": "E-other", "issuer": "i", "cause": "c", "effective_utc": "t", "sequence": 2}"#,
                ));
                revocations.push(json(
                    r#"{"node": "E-other", "issuer": "i", "cause": "c", "effective_utc": "t", "sequence": 1}"#,
                ));
            },
            "append-only",
        ),
    ];
    for (label, mutate, reason) in mutations {
        let fixture = Fixture::new();
        let mut document = fixture.document();
        mutate(&mut document);
        let result = fixture.verify(&document);
        assert!(result.is_err(), "{label}: accepted");
        assert_refused(result, reason);
    }
}

#[test]
fn missing_custody_is_reported_and_can_be_required() {
    let fixture = Fixture::new();
    let mut document = fixture.document();
    set(node(&mut document, "E-other"), "location", "custody".into());
    set(
        node(&mut document, "E-other"),
        "path",
        "/nonexistent/custody.json".into(),
    );
    assert_eq!(
        fixture
            .verify(&document)
            .expect("custody unavailable")
            .custody_unverified,
        1
    );
    dump(&fixture.path, &document);
    assert_refused(
        verify(&fixture.path, Some(&fixture.root), true),
        "custody evidence unavailable",
    );
}

/// Restores a directory's mode even when an assertion fails.
struct Sealed<'a>(&'a Path);

impl Drop for Sealed<'_> {
    fn drop(&mut self) {
        let _ = std::fs::set_permissions(self.0, std::fs::Permissions::from_mode(0o700));
    }
}

#[test]
fn inaccessible_custody_root_is_unavailable_not_a_crash() {
    // A CI runner is another user: looking up a path under another user's home fails with
    // EACCES rather than reporting absence. That is unavailable custody, never a verifier crash.
    let fixture = Fixture::new();
    let sealed = fixture.root.join("sealed");
    std::fs::create_dir(&sealed).expect("sealed directory");
    let evidence = sealed.join("custody.json");
    std::fs::write(&evidence, "{}").expect("custody evidence");
    if std::fs::metadata(&evidence).expect("metadata").uid() == 0 {
        eprintln!("skipped: root can traverse a mode-000 directory");
        return;
    }
    let mut document = fixture.document();
    set(node(&mut document, "E-other"), "location", "custody".into());
    set(
        node(&mut document, "E-other"),
        "path",
        evidence.display().to_string().into(),
    );
    set(
        node(&mut document, "E-other"),
        "sha256",
        digest(&evidence).into(),
    );
    assert_eq!(
        fixture
            .verify(&document)
            .expect("custody readable")
            .custody_unverified,
        0
    );
    std::fs::set_permissions(&sealed, std::fs::Permissions::from_mode(0o000)).expect("seal");
    let _guard = Sealed(&sealed);
    assert_eq!(
        fixture
            .verify(&document)
            .expect("custody sealed")
            .custody_unverified,
        1
    );
    assert_refused(
        verify(&fixture.path, Some(&fixture.root), true),
        "custody evidence unavailable",
    );
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

#[test]
fn synthetic_identity_cannot_acquire_real_or_live_attestation() {
    // W12-T07: identity comes from the evidence's location or bytes, never its declared kind.
    type Case = fn(&Fixture, &mut Value);
    let mutations: [(&str, Case, &str); 8] = [
        (
            "marked bytes declared live-host",
            |f, d| mark(f, d, "E-grandchild"),
            "E-grandchild: synthetic identity cannot be live-host",
        ),
        (
            "marked bytes declared measurement",
            |f, d| mark(f, d, "E-other"),
            "E-other: synthetic identity cannot be measurement",
        ),
        (
            "marked bytes declared real corpus",
            |f, d| {
                mark(f, d, "E-root");
                relabel(d, "E-root", "real-corpus-fixture");
            },
            "synthetic identity cannot be real-corpus-fixture",
        ),
        (
            "conformance path declared real corpus",
            |f, d| {
                mark_with(f, d, "E-root", Some(CONFORMANCE), MARK.as_bytes());
                relabel(d, "E-root", "real-corpus-fixture");
            },
            "synthetic identity cannot be real-corpus-fixture",
        ),
        (
            // Addition: the location alone identifies it, whatever the bytes say.
            "unmarked bytes on the conformance path declared real corpus",
            |f, d| {
                mark_with(f, d, "E-root", Some(CONFORMANCE), br#"{"node": "E-root"}"#);
                relabel(d, "E-root", "real-corpus-fixture");
            },
            "^E-root: synthetic identity cannot be real-corpus-fixture evidence$",
        ),
        (
            "marked hosted CI cited by a live claim",
            |f, d| {
                mark(f, d, "E-child");
                set(
                    claim_at(d, 0),
                    "evidence",
                    strs(&["E-grandchild", "E-child"]),
                );
            },
            "C-live: synthetic evidence cannot support a live claim",
        ),
        (
            "synthetic fixture cited by a historical claim",
            |_, d| {
                relabel(d, "E-other", "historical-record");
                set(claim_at(d, 1), "stage", "historical".into());
                set(claim_at(d, 1), "evidence", strs(&["E-other", "E-root"]));
            },
            "C-measure: synthetic evidence cannot support a historical claim",
        ),
        (
            // Addition: readable custody bytes carry their identity too.
            "marked custody bytes declared live-host",
            |f, d| {
                let custody = f.root.join("custody").join("live.json");
                std::fs::create_dir_all(custody.parent().expect("parent"))
                    .expect("custody directory");
                std::fs::write(&custody, MARK).expect("custody bytes");
                let grandchild = node(d, "E-grandchild");
                set(grandchild, "location", "custody".into());
                set(grandchild, "path", custody.display().to_string().into());
                set(grandchild, "sha256", digest(&custody).into());
            },
            "^E-grandchild: synthetic identity cannot be live-host evidence$",
        ),
    ];
    // Every case runs before the verdict, so a failure lists each gap rather than the first.
    let mut accepted = Vec::new();
    for (label, mutate, reason) in mutations {
        let fixture = Fixture::new();
        let mut document = fixture.document();
        mutate(&fixture, &mut document);
        match fixture.verify(&document) {
            Ok(summary) => accepted.push(format!("{label}: ACCEPTED {summary:?}")),
            Err(error) => assert_refused::<()>(Err(error), reason),
        }
    }
    assert!(
        accepted.is_empty(),
        "synthetic identity acquired an attestation:\n{}",
        accepted.join("\n")
    );
}

#[test]
fn synthetic_identity_through_a_symlinked_directory_is_still_located() {
    // W12-T07 addition: a repository path whose directory is a symlink into the conformance
    // tree is conformance evidence by location, even with unmarked bytes.
    let fixture = Fixture::new();
    let mut document = fixture.document();
    mark_with(
        &fixture,
        &mut document,
        "E-root",
        Some(CONFORMANCE),
        br#"{"node": "E-root"}"#,
    );
    let conformance = fixture
        .root
        .join(Path::new(CONFORMANCE).parent().expect("directory"));
    std::os::unix::fs::symlink(&conformance, fixture.root.join("evidence/linked"))
        .expect("symlink");
    set(
        node(&mut document, "E-root"),
        "path",
        "evidence/linked/x.json".into(),
    );
    relabel(&mut document, "E-root", "real-corpus-fixture");
    assert_refused(
        fixture.verify(&document),
        "^E-root: synthetic identity cannot be real-corpus-fixture evidence$",
    );
}

#[test]
fn synthetic_identity_keeps_fixture_hosted_and_implementation_stages() {
    // W12-T07 positive controls: marked bytes may back fixture, hosted-CI and implementation
    // stages, and unmarked or non-JSON bytes keep their declared kind.
    let fixture = Fixture::new();
    let mut document = fixture.document();
    mark(&fixture, &mut document, "E-root");
    mark(&fixture, &mut document, "E-child");
    set(
        &mut document,
        "claims",
        Value::List(vec![
            fixture.claim("C-fixture", "fixture", &["E-root"], "active"),
            fixture.claim("C-ci", "hosted-ci", &["E-child"], "active"),
            fixture.claim("C-measure", "measurement", &["E-other"], "active"),
        ]),
    );
    let active = |summary: Summary| summary.claims.into_iter().collect::<Vec<_>>();
    assert_eq!(
        active(fixture.verify(&document).expect("marked fixture evidence")),
        [("active".to_owned(), 3)]
    );
    mark_with(&fixture, &mut document, "E-other", None, b"\xff not json");
    assert_eq!(
        active(fixture.verify(&document).expect("non-JSON bytes")),
        [("active".to_owned(), 3)]
    );
    // Unavailable custody has an unknown identity here, not a synthetic one.
    set(node(&mut document, "E-other"), "location", "custody".into());
    set(
        node(&mut document, "E-other"),
        "path",
        "/nonexistent/custody.json".into(),
    );
    let summary = fixture.verify(&document).expect("unavailable custody");
    assert_eq!(summary.custody_unverified, 1);
}
