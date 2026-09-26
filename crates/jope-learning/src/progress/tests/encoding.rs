//! W8-T01: identities, canonical encodings, schema limits and hash domains of the inventory, and
//! the verified link from a progress HWM decision to the real W3 HWM ledger.

use super::fixture::{hwm_record, run, spec, standard, PUBLISHER, SOURCE_COMMIT};
use crate::canon::{canonical_json, sha256_hex};
use crate::hwm::{self, Candidate, Transition};
use crate::ledger::Ledger;
use crate::progress::census::Counts;
use crate::progress::inventory::*;
use crate::state::State;
use serde_json::{json, Value};
use std::collections::BTreeSet;

fn with(value: &impl serde::Serialize, edit: impl FnOnce(&mut Value)) -> Vec<u8> {
    let mut v = serde_json::to_value(value).unwrap();
    edit(&mut v);
    serde_json::to_vec(&v).unwrap()
}

#[test]
fn logical_work_identity_is_separate_from_attempt_and_delivery_identity() {
    let r = run("run-w8-ids", 3, 3);
    // The logical identity is a function of the run and the specification only.
    assert_eq!(spec(1).work_id("run-w8-ids").unwrap(), r.work[1]);
    assert_ne!(spec(1).work_id("run-w8-other").unwrap(), r.work[1]);
    let mut changed = spec(1);
    changed.inputs = vec![sha256_hex(b"another input")];
    assert_ne!(changed.work_id("run-w8-ids").unwrap(), r.work[1]);
    let mut changed = spec(1);
    changed.stage = "teacher-admission".into();
    assert_ne!(changed.work_id("run-w8-ids").unwrap(), r.work[1]);

    // Two attempts at one unit: same logical identity, different attempt and delivery identity.
    let f1 = r.failed(1, 1, FailureClass::Timeout, true);
    let a1 = r.completed(1, 1);
    let a2 = r.completed(1, 2);
    assert_eq!(a1.value().work_id, a2.value().work_id);
    assert_ne!(a1.value().attempt, a2.value().attempt);
    assert_ne!(a1.digest(), a2.digest());
    // The same bytes are the same delivery.
    assert_eq!(r.completed(1, 2).digest(), a2.digest());
    // A retry depends on the exact failure it retries, and on the manifest.
    let retry = r.retry(1, 2, &f1).encoded().unwrap();
    assert!(retry.dependencies.contains(&Dependency::Object(ObjectRef {
        kind: Kind::Event,
        digest: f1.digest().into()
    })));
    assert!(retry.dependencies.contains(&Dependency::Object(ObjectRef {
        kind: Kind::RunManifest,
        digest: r.manifest.digest().into()
    })));

    // The same logical work cannot be planned twice, and a unit must carry its own work id.
    let dup = RunManifest::new(
        "run-w8-ids",
        PUBLISHER,
        SOURCE_COMMIT,
        3,
        vec![spec(1), spec(1)],
    )
    .unwrap();
    assert!(encode(&dup).is_err());
    let mut forged = r.manifest.value().clone();
    forged.units[0].work_id = "0".repeat(64);
    assert!(
        encode(&forged).is_err(),
        "a work id its specification does not produce"
    );
    let mut unsorted = r.manifest.value().clone();
    unsorted.units.reverse();
    assert!(encode(&unsorted).is_err());
}

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
            blob_key(&BlobRef::of(&super::fixture::Run::result(1, 2)).sha256)
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

#[test]
fn hwm_decision_link_is_checked_against_the_real_hwm_ledger() {
    let dir = tempfile::tempdir().unwrap();
    let state = State::at(dir.path());
    let ledger = Ledger::open(state.root(), "hwm");
    let (r, s) = standard();
    let first = s.event("u0-completed-1");
    let second = s.event("u1-completed-2");

    let ingest = |value: f64| {
        let (record, blob) = hwm_record("cut-a", value);
        let report = hwm::ingest(
            &state,
            None,
            vec![Candidate {
                record: record.clone(),
                blob: blob.clone(),
                chain_from_source: false,
            }],
            json!({"test": "w8-t01"}),
        )
        .unwrap();
        (record, blob, report.batch.unwrap())
    };
    let link = |completion: &Decoded<Event>,
                record: &crate::record::HwmRecord,
                blob: &[u8],
                batch: &crate::ledger::BatchReceipt,
                transition: Transition| HwmDecision {
        schema: Kind::HwmDecision.schema().into(),
        run_id: r.id().into(),
        work_id: completion.value().work_id.clone(),
        attempt: completion.value().attempt,
        completion: completion.digest().into(),
        record: BlobRef::of(blob),
        identity_sha256: record.identity_sha256.clone(),
        ledger_batch: batch.batch.clone(),
        ledger_path: batch.path.clone(),
        ledger_batch_sha256: batch.sha256.clone(),
        seq: 0,
        transition,
        qualification_sha256: None,
    };

    let (rec1, blob1, batch1) = ingest(0.5);
    let init = link(first, &rec1, &blob1, &batch1, Transition::Initialized);
    encode(&init).unwrap();
    init.verify_against(&ledger).unwrap();
    let (rec2, blob2, batch2) = ingest(0.25);
    let improved = link(second, &rec2, &blob2, &batch2, Transition::Improved);
    improved.verify_against(&ledger).unwrap();

    // A progress record cannot claim what the ledger did not record.
    let claims = [
        (
            "initialization claimed as improvement",
            HwmDecision {
                transition: Transition::Improved,
                ..init.clone()
            },
        ),
        (
            "other record",
            HwmDecision {
                record: BlobRef::of(&blob2),
                ..init.clone()
            },
        ),
        (
            "other identity",
            HwmDecision {
                identity_sha256: sha256_hex(b"other"),
                ..init.clone()
            },
        ),
        (
            "missing event",
            HwmDecision {
                seq: 1,
                ..init.clone()
            },
        ),
        (
            "other batch bytes",
            HwmDecision {
                ledger_batch_sha256: batch2.sha256.clone(),
                ..init.clone()
            },
        ),
        (
            "qualification never relied on",
            HwmDecision {
                qualification_sha256: Some(sha256_hex(b"qualification")),
                ..init.clone()
            },
        ),
        (
            "batch outside this chain",
            HwmDecision {
                ledger_batch: "01K62ZZZZZZZZZZZZZZZZZZZZZ".into(),
                ledger_path: "2026/09/01K62ZZZZZZZZZZZZZZZZZZZZZ.jsonl".into(),
                ..init.clone()
            },
        ),
    ];
    for (label, claim) in claims {
        encode(&claim).unwrap();
        assert!(claim.verify_against(&ledger).is_err(), "{label} verified");
    }
    // Neither refusal wrote anything: the ledger still verifies and holds two batches.
    let report = ledger.require_intact().unwrap();
    assert_eq!(report.batches, 2);

    // A tampered ledger verifies no decision at all.
    let path = ledger.root().join(&batch1.path);
    let mut bytes = std::fs::read(&path).unwrap();
    let i = bytes.iter().position(|&b| b == b'5').unwrap();
    bytes[i] = b'6';
    std::fs::write(&path, &bytes).unwrap();
    assert!(init.verify_against(&ledger).is_err());
    assert!(improved.verify_against(&ledger).is_err());
}
