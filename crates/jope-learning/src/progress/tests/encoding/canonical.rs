use super::*;

#[test]
fn canonical_encoding_is_unique_and_bounded() {
    let r = run("run-w8-canon", 2, 3);
    let e = r.completed(0, 1);
    let encoded = e.encoded().unwrap();
    let back = decode::<Event>(&encoded.bytes).unwrap();
    assert_eq!(back.digest(), e.digest());
    assert_eq!(back.value(), e.value());
    assert_eq!(canonical_json(back.value()).unwrap(), encoded.bytes);

    // Other spellings of the same value are refused, not normalised.
    let pretty = serde_json::to_vec_pretty(e.value()).unwrap();
    assert!(decode::<Event>(&pretty).is_err(), "whitespace");
    let text = String::from_utf8(encoded.bytes.clone()).unwrap();
    let schema_first = format!(
        "{{\"schema\":\"{}\",{}",
        Kind::Event.schema(),
        text[1..].replace(&format!(",\"schema\":\"{}\"", Kind::Event.schema()), "")
    );
    assert_ne!(schema_first.as_bytes(), encoded.bytes.as_slice());
    assert!(serde_json::from_str::<Value>(&schema_first).is_ok());
    assert!(
        decode::<Event>(schema_first.as_bytes()).is_err(),
        "key order"
    );
    let extra = with(e.value(), |v| v["extra"] = json!(1));
    assert!(decode::<Event>(&extra).is_err(), "unknown field");
    let extra_body = with(e.value(), |v| v["body"]["extra"] = json!(1));
    assert!(decode::<Event>(&extra_body).is_err(), "unknown body field");
    let exposure = Exposure {
        schema: Kind::Exposure.schema().into(),
        run_id: r.id().into(),
        update_index: 0,
        corpus: sha256_hex(b"corpus"),
        checkpoint_before: None,
        consumed_rows: 10,
    };
    let canonical = encode(&exposure).unwrap().bytes;
    assert!(decode::<Exposure>(&canonical).is_ok());
    let omitted = with(&exposure, |v| {
        v.as_object_mut().unwrap().remove("checkpoint_before");
    });
    assert!(
        decode::<Exposure>(&omitted).is_err(),
        "an omitted null is a second spelling"
    );
    let float = with(&exposure, |v| v["consumed_rows"] = json!(10.0));
    assert!(decode::<Exposure>(&float).is_err(), "no floating point");

    // Limits.
    assert!(decode::<Event>(&vec![b' '; MAX_OBJECT_BYTES + 1]).is_err());
    let bad_ids = [
        "",
        "..",
        "a/b",
        "../escape",
        "with space",
        "tab\t",
        &"x".repeat(MAX_ID_BYTES + 1),
    ];
    for bad in bad_ids {
        assert!(
            RunManifest::new(bad, PUBLISHER, SOURCE_COMMIT, 3, vec![spec(0)]).is_err()
                || encode(&RunManifest {
                    run_id: bad.into(),
                    ..r.manifest.value().clone()
                })
                .is_err(),
            "run_id {bad:?}"
        );
        assert!(
            Decoded::new(Event::new(
                &r.manifest,
                &r.work[0],
                1,
                EventBody::Started { worker: bad.into() }
            ))
            .is_err(),
            "worker {bad:?}"
        );
    }
    assert!(Decoded::new(Event::new(
        &r.manifest,
        &r.work[0].to_uppercase(),
        1,
        EventBody::Scheduled
    ))
    .is_err());
    for (attempt, body) in [
        (2, EventBody::Scheduled),
        (0, EventBody::Scheduled),
        (
            1,
            EventBody::Retry {
                failed_event: e.digest().into(),
            },
        ),
        (
            1,
            EventBody::Excluded {
                reason: "late".into(),
            },
        ),
        (
            0,
            EventBody::Completed {
                result: BlobRef::of(b"x"),
            },
        ),
        (MAX_ATTEMPTS + 1, EventBody::Started { worker: "w".into() }),
        (
            1,
            EventBody::Failed {
                class: FailureClass::Timeout,
                retryable: true,
                detail: "x".repeat(MAX_TEXT_BYTES + 1),
            },
        ),
        (
            1,
            EventBody::Failed {
                class: FailureClass::Timeout,
                retryable: true,
                detail: "line\nbreak".into(),
            },
        ),
        (
            1,
            EventBody::Failed {
                class: FailureClass::Timeout,
                retryable: true,
                detail: String::new(),
            },
        ),
    ] {
        assert!(
            Decoded::new(Event::new(&r.manifest, &r.work[0], attempt, body.clone())).is_err(),
            "attempt {attempt} {body:?}"
        );
    }
    for (commit, attempts) in [
        (&SOURCE_COMMIT.to_uppercase(), 3),
        (&SOURCE_COMMIT[..12].to_string(), 3),
        (&SOURCE_COMMIT.to_string(), 0),
        (&SOURCE_COMMIT.to_string(), MAX_ATTEMPTS + 1),
    ] {
        let m =
            RunManifest::new("run-w8-canon", PUBLISHER, commit, attempts, vec![spec(0)]).unwrap();
        assert!(encode(&m).is_err(), "{commit} {attempts}");
    }
    let mut unsorted = spec(0);
    unsorted.inputs = vec![sha256_hex(b"b"), sha256_hex(b"a")];
    unsorted.inputs.sort_by(|a, b| b.cmp(a));
    assert!(unsorted.work_id("run-w8-canon").is_err(), "unsorted set");

    let blobs: Vec<BlobRef> = (0..=MAX_REFS)
        .map(|i| BlobRef::of(i.to_string().as_bytes()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let corpus = CorpusManifest {
        schema: Kind::CorpusManifest.schema().into(),
        run_id: r.id().into(),
        corpus_id: "corpus-a".into(),
        parent: None,
        members: blobs.clone(),
        decisions: vec![],
    };
    assert!(encode(&corpus).is_err(), "over-long list");
    assert!(encode(&CorpusManifest {
        members: blobs[..MAX_REFS].to_vec(),
        ..corpus.clone()
    })
    .is_ok());
    assert!(encode(&CorpusManifest {
        members: vec![],
        ..corpus
    })
    .is_err());

    // Retention is explicit and consistent.
    let evidence = CandidateEvidence {
        schema: Kind::CandidateEvidence.schema().into(),
        run_id: r.id().into(),
        work_id: r.work[0].clone(),
        attempt: 1,
        completion: e.digest().into(),
        record: BlobRef::of(b"record"),
        retention: Retention::Full,
        artifacts: vec![],
    };
    assert!(
        encode(&evidence).is_err(),
        "full retention without artifacts"
    );
    assert!(encode(&CandidateEvidence {
        retention: Retention::Compact,
        artifacts: vec![BlobRef::of(b"fitted")],
        ..evidence.clone()
    })
    .is_err());
    assert!(encode(&CandidateEvidence {
        retention: Retention::Compact,
        ..evidence
    })
    .is_ok());

    // Commits: parent exactly above height 0, never adding commits, summaries or manifests.
    let commit = Commit {
        schema: Kind::Commit.schema().into(),
        run_id: r.id().into(),
        run_manifest: r.manifest.digest().into(),
        parent: None,
        height: 0,
        objects: vec![ObjectRef {
            kind: Kind::Event,
            digest: e.digest().into(),
        }],
    };
    assert!(encode(&commit).is_ok());
    for bad in [
        Commit {
            height: 1,
            ..commit.clone()
        },
        Commit {
            parent: Some(sha256_hex(b"p")),
            ..commit.clone()
        },
        Commit {
            objects: vec![],
            ..commit.clone()
        },
        Commit {
            objects: vec![ObjectRef {
                kind: Kind::Commit,
                digest: sha256_hex(b"c"),
            }],
            ..commit.clone()
        },
    ] {
        assert!(encode(&bad).is_err(), "{bad:?}");
    }

    // The mutable pointer has one encoding too.
    let head = Head {
        schema: HEAD_SCHEMA.into(),
        run_id: r.id().into(),
        commit: encode(&commit).unwrap().digest,
        height: 0,
    };
    let bytes = head.encode().unwrap();
    assert_eq!(Head::decode(&bytes).unwrap(), head);
    assert!(Head::decode(&serde_json::to_vec_pretty(&head).unwrap()).is_err());
    assert!(Head {
        schema: "jope.progress.head/v2".into(),
        ..head.clone()
    }
    .encode()
    .is_err());
    assert_eq!(
        head_key(r.id()).unwrap(),
        "progress/runs/run-w8-canon/HEAD.json"
    );
    assert!(head_key("../x").is_err());
}
