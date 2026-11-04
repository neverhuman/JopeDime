use super::*;

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
        let report = crate::hwm::ingest(
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
