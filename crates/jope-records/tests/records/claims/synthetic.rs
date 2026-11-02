use super::*;

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

#[test]
fn revoked_evidence_gives_consumers_digests_attributed_to_their_revocation() {
    let fixture = Fixture::new();
    let mut document = fixture.document();
    // Two revocations reach E-grandchild: E-root's (sequence 1) and its parent E-child's
    // (sequence 2). A node is attributed to the lowest sequence that reaches it.
    set(
        &mut document,
        "revocations",
        json(
            r#"[{"node": "E-root", "issuer": "reviewer", "cause": "leak",
                "effective_utc": "2026-09-26T01:00:00Z", "sequence": 1},
               {"node": "E-child", "issuer": "owner", "cause": "rerun",
                "effective_utc": "2026-09-26T02:00:00Z", "sequence": 2}]"#,
        ),
    );
    set(claim_at(&mut document, 0), "status", "revoked".into());
    dump(&fixture.path, &document);
    let closure =
        revoked_evidence(&fixture.path, Some(&fixture.root), false).expect("verified record");
    assert_eq!(closure.claims_sha256, digest(&fixture.path));
    let got: Vec<(&str, &str, i64, bool)> = closure
        .revoked
        .iter()
        .map(|r| {
            (
                r.node.as_str(),
                r.revoked_node.as_str(),
                r.sequence,
                r.sha256 == digest(&fixture.root.join(format!("evidence/{}.json", r.node))),
            )
        })
        .collect();
    assert_eq!(
        got,
        [
            ("E-child", "E-root", 1, true),
            ("E-grandchild", "E-root", 1, true),
            ("E-root", "E-root", 1, true),
        ]
    );
    // The unrelated node is not revoked.
    assert!(closure.revoked.iter().all(|r| r.node != "E-other"));
    // A record verify refuses yields no revocations at all.
    set(claim_at(&mut document, 0), "status", "active".into());
    dump(&fixture.path, &document);
    assert_refused(
        revoked_evidence(&fixture.path, Some(&fixture.root), false),
        "C-live: cites revoked evidence",
    );
    // No revocation: nothing revoked.
    dump(&fixture.path, &fixture.document());
    assert!(revoked_evidence(&fixture.path, Some(&fixture.root), false)
        .expect("verified record")
        .revoked
        .is_empty());
}
