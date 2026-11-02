use super::*;

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
