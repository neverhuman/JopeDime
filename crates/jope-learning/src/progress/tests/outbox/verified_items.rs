use super::*;

#[test]
fn only_verified_content_addressed_items_enter() {
    let e = env();
    let r = run("run-w8-verified", 1, 3);
    let mut o = Outbox::open(&e.state, BIG).unwrap();
    let manifest = r.manifest.encoded().unwrap();
    o.append_object(&manifest).unwrap();
    let dir = Outbox::dir(&e.state);
    let before = snapshot(&dir);
    let blobs_before = snapshot(&e.state.root().join("blobs"));

    // A worker result that is not the bytes its completion will reference.
    let err = o
        .append_result(b"forged result", &BlobRef::of(&Run::result(0, 1)))
        .unwrap_err();
    assert!(err.to_string().contains("not the referenced"), "{err:#}");

    let good = r.scheduled(0).encoded().unwrap();
    let mut tampered = good.clone();
    let last = tampered.bytes.len() - 3;
    tampered.bytes[last] ^= 0x01;
    let mut other_kind = good.clone();
    other_kind.kind = Kind::CandidateEvidence;
    other_kind.digest = domain_digest(Kind::CandidateEvidence.schema(), &good.bytes);
    let mut no_deps = good.clone();
    no_deps.dependencies.clear();
    let pretty = serde_json::to_vec_pretty(r.scheduled(0).value()).unwrap();
    let noncanonical = Encoded {
        digest: domain_digest(Kind::Event.schema(), &pretty),
        bytes: pretty,
        ..good.clone()
    };
    // A run pointer offered as if it were an object: pointers never enter the outbox.
    let head = Head {
        schema: HEAD_SCHEMA.into(),
        run_id: r.id().into(),
        commit: good.digest.clone(),
        height: 0,
    }
    .encode()
    .unwrap();
    let pointer = Encoded {
        kind: Kind::Commit,
        run_id: r.id().into(),
        digest: domain_digest(Kind::Commit.schema(), &head),
        bytes: head,
        dependencies: Vec::new(),
    };
    for (label, forged) in [
        ("tampered", tampered),
        ("other kind", other_kind),
        ("dependencies hidden", no_deps),
        ("noncanonical", noncanonical),
        ("run pointer", pointer),
    ] {
        assert!(o.append_object(&forged).is_err(), "{label} was admitted");
    }
    assert_eq!(snapshot(&dir), before);
    assert_eq!(snapshot(&e.state.root().join("blobs")), blobs_before);
    assert_eq!(o.records(), 1);
    // Positive control.
    assert_eq!(o.append_object(&good).unwrap(), Appended::Added { seq: 2 });
}
