use super::*;

#[test]
fn hash_domains_separate_kinds_and_blobs() {
    let schemas: BTreeSet<&str> = Kind::ALL.iter().map(|k| k.schema()).collect();
    let dirs: BTreeSet<&str> = Kind::ALL.iter().map(|k| k.dir()).collect();
    assert_eq!(
        (schemas.len(), dirs.len()),
        (Kind::ALL.len(), Kind::ALL.len())
    );
    assert!(!schemas.contains(WORK_DOMAIN) && !schemas.contains(HEAD_SCHEMA));

    let r = run("run-w8-domain", 1, 3);
    let e = r.scheduled(0).encoded().unwrap();
    assert_eq!(domain_digest(Kind::Event.schema(), &e.bytes), e.digest);
    for other in Kind::ALL.into_iter().filter(|k| *k != Kind::Event) {
        assert_ne!(domain_digest(other.schema(), &e.bytes), e.digest);
        assert!(
            verify_object(other, &domain_digest(other.schema(), &e.bytes), &e.bytes).is_err(),
            "event bytes accepted as {other:?}"
        );
    }
    assert_ne!(
        sha256_hex(&e.bytes),
        e.digest,
        "an object digest is never a blob sha256"
    );
    assert!(verify_object(Kind::Event, &e.digest, &e.bytes).is_ok());
    assert!(verify_object(Kind::Event, &sha256_hex(&e.bytes), &e.bytes).is_err());
    assert!(decode_as::<Event>(&e.bytes, r.manifest.digest()).is_err());
    assert_eq!(e.key(), format!("progress/objects/event/{}.json", e.digest));
    assert_eq!(
        Dependency::Blob(BlobRef::of(b"x")).key(),
        State::blob_rel(&sha256_hex(b"x"))
    );
}

#[test]
fn every_kind_round_trips_and_names_its_dependencies() {
    let (r, s) = standard();
    let completion = s.event("u1-completed-2");
    let decision = s.decisions[1].1.clone();
    let corpus = CorpusManifest {
        schema: Kind::CorpusManifest.schema().into(),
        run_id: r.id().into(),
        corpus_id: "corpus-a".into(),
        parent: None,
        members: vec![BlobRef::of(b"member")],
        decisions: vec![decision.digest().into()],
    };
    let corpus_d = encode(&corpus).unwrap().digest;
    let exposure = Exposure {
        schema: Kind::Exposure.schema().into(),
        run_id: r.id().into(),
        update_index: 0,
        corpus: corpus_d.clone(),
        checkpoint_before: None,
        consumed_rows: 1,
    };
    let exposure_d = encode(&exposure).unwrap().digest;
    let checkpoint = Checkpoint {
        schema: Kind::Checkpoint.schema().into(),
        run_id: r.id().into(),
        update_index: 0,
        parent: None,
        exposure: exposure_d.clone(),
        corpus: corpus_d.clone(),
        state: vec![BlobRef::of(b"parameters"), BlobRef::of(b"optimizer")]
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
    };
    let evidence = CandidateEvidence {
        schema: Kind::CandidateEvidence.schema().into(),
        run_id: r.id().into(),
        work_id: r.work[1].clone(),
        attempt: 2,
        completion: completion.digest().into(),
        record: BlobRef::of(b"record"),
        retention: Retention::Full,
        artifacts: vec![BlobRef::of(b"fitted")],
    };
    let commit = Commit {
        schema: Kind::Commit.schema().into(),
        run_id: r.id().into(),
        run_manifest: r.manifest.digest().into(),
        parent: None,
        height: 0,
        objects: vec![
            ObjectRef {
                kind: Kind::Event,
                digest: completion.digest().into(),
            },
            ObjectRef {
                kind: Kind::Checkpoint,
                digest: encode(&checkpoint).unwrap().digest,
            },
        ],
    };
    let summary = Summary {
        schema: Kind::Summary.schema().into(),
        run_id: r.id().into(),
        derived_from: encode(&commit).unwrap().digest,
        height: 0,
        counts: Counts::default(),
        candidate_yield: Fraction::default(),
        exposures: 1,
        consumed_rows: 1,
        corpora: 1,
        unique_teachers: 1,
        checkpoints: 1,
        latest_checkpoint: Some(encode(&checkpoint).unwrap().digest),
    };
    let all = [
        r.manifest.encoded().unwrap(),
        completion.encoded().unwrap(),
        encode(&evidence).unwrap(),
        decision.encoded().unwrap(),
        encode(&corpus).unwrap(),
        encode(&exposure).unwrap(),
        encode(&checkpoint).unwrap(),
        encode(&commit).unwrap(),
        encode(&summary).unwrap(),
    ];
    assert_eq!(
        all.iter().map(|e| e.kind).collect::<Vec<_>>(),
        Kind::ALL.to_vec()
    );
    for e in &all {
        assert_eq!(
            verify_object(e.kind, &e.digest, &e.bytes).unwrap(),
            e.dependencies,
            "{:?}",
            e.kind
        );
        let mut corrupt = e.bytes.clone();
        let last = corrupt.len() - 2;
        corrupt[last] ^= 0x01;
        assert!(verify_object(e.kind, &e.digest, &corrupt).is_err());
    }
    let deps =
        |i: usize| -> BTreeSet<String> { all[i].dependencies.iter().map(|d| d.key()).collect() };
    let obj = |k: Kind, d: &str| object_key(k, d);
    assert!(deps(0).is_empty(), "the run manifest depends on nothing");
    assert_eq!(
        deps(1),
        BTreeSet::from([
            obj(Kind::RunManifest, r.manifest.digest()),
            blob_key(&BlobRef::of(&super::super::fixture::Run::result(1, 2)).sha256)
        ])
    );
    assert!(deps(3).contains(&obj(Kind::Event, completion.digest())));
    assert!(deps(4).contains(&obj(Kind::HwmDecision, decision.digest())));
    assert!(deps(6).contains(&obj(Kind::Exposure, &exposure_d)));
    assert!(deps(6).contains(&obj(Kind::CorpusManifest, &corpus_d)));
    assert_eq!(deps(6).len(), 4, "exposure, corpus and both state blobs");
    assert!(deps(7).contains(&obj(Kind::RunManifest, r.manifest.digest())));
    assert_eq!(all[8].kind.class(), Class::Derived);
    assert!(Kind::ALL
        .iter()
        .filter(|k| **k != Kind::Summary)
        .all(|k| k.class() != Class::Derived));
}
