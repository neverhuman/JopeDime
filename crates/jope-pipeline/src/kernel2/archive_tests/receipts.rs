use super::*;

#[test]
fn archive_runs_replay_receipt_and_all_sidecars_with_a_frozen_clock() {
    let temp = tempfile::tempdir().unwrap();
    for (label, refine, trials) in [
        ("coordinate", RefineMethod::Coordinate, 4),
        ("lm", lm(), 12),
    ] {
        let make = |name: &str| {
            opts(
                &temp.path().join(format!("{label}-{name}")),
                40,
                "kernel2-baseline",
                FaultPolicy::RejectHard,
                trials,
                selection(refine, true),
            )
        };
        let (a, b) = (make("a"), make("b"));
        let receipt = run::run(&a).unwrap();
        assert_eq!(receipt, run::run(&b).unwrap(), "{label}");
        for relative in SIDECARS {
            assert_eq!(read(&a, relative), read(&b, relative), "{label} {relative}");
        }
        assert_eq!(
            publication::verify_run(&a.run.output.join("receipt.json")).unwrap(),
            receipt
        );
    }
}

/// Rewrites a run's archive (when given) and its selection record, rehashing search.json and the
/// receipt, so that only the archive checks can refuse the change.
fn rewrite(
    output: &Path,
    receipt: &Receipt,
    archive: Option<&[u8]>,
    engine: Option<EngineRecord>,
) -> PathBuf {
    if let Some(bytes) = archive {
        std::fs::write(output.join(archive::PATH), bytes).unwrap();
    }
    let mut edited = receipt.clone();
    edited.search.engine_selection = engine;
    let summary = serde_json::to_vec_pretty(&edited.search).unwrap();
    std::fs::write(output.join("search/cut-0/search.json"), &summary).unwrap();
    edited.search_sha256 = crate::sha256_hex(&summary);
    let path = output.join("receipt.json");
    std::fs::write(&path, serde_json::to_vec(&edited).unwrap()).unwrap();
    path
}

/// The run's record with the archive digest set to that of `bytes`.
fn digested(engine: &EngineRecord, bytes: &[u8]) -> EngineRecord {
    let mut engine = engine.clone();
    engine.archive.as_mut().unwrap().sha256 = Some(crate::sha256_hex(bytes));
    engine
}

#[test]
fn the_archive_round_trips_and_verify_refuses_every_tampered_archive_or_record() {
    let temp = tempfile::tempdir().unwrap();
    let opt = archived_on(temp.path(), SUPERNOVAE);
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    let originals: Vec<(PathBuf, Vec<u8>)> =
        ["receipt.json", "search/cut-0/search.json", archive::PATH]
            .iter()
            .map(|r| (output.join(r), std::fs::read(output.join(r)).unwrap()))
            .collect();
    let restore = || {
        for (file, bytes) in &originals {
            std::fs::write(file, bytes).unwrap();
        }
    };
    // Round trip: the reader takes the published bytes back to themselves.
    let bytes = originals[2].1.clone();
    let text = String::from_utf8(bytes.clone()).unwrap();
    let back = Archive::from_json(&text).unwrap();
    assert_eq!(back.to_json().as_bytes(), bytes.as_slice());
    assert!(
        !back.members().is_empty(),
        "the fixture has a front to edit"
    );
    let engine = receipt.search.engine_selection.clone().unwrap();
    // The rewrite itself keeps a valid run.
    rewrite(output, &receipt, Some(&bytes), Some(engine.clone()));
    publication::verify_run(&path).unwrap();
    restore();

    let hex = |bits: u64| format!("{bits:016x}");
    let key_hex = |key: &[u8; 32]| key.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let members = back.members();
    let last = members.last().unwrap();
    let mut edits: Vec<(&str, String)> = Vec::new();
    // A lower loss on the last member keeps the front in order: only re-derivation refuses it.
    edits.push((
        "member loss",
        text.replacen(&hex(last.loss), &hex((last.loss() - 0.001).to_bits()), 1),
    ));
    let mut other = members[0].key;
    other[0] ^= 0x01;
    edits.push((
        "member key",
        text.replacen(&key_hex(&members[0].key), &key_hex(&other), 1),
    ));
    edits.push((
        "member complexity",
        text.replacen(
            &format!("\"complexity\":{}", last.complexity),
            &format!("\"complexity\":{}", last.complexity + 1),
            1,
        ),
    ));
    edits.push((
        "cost version",
        text.replace(pareto::PIPELINE_COST_VERSION, pareto::COST_VERSION),
    ));
    // The front without its simplest member is still a front (possibly empty): only re-derivation
    // refuses it.
    let kept: Vec<Candidate> = members[1..]
        .iter()
        .map(|m| Candidate {
            loss: m.loss(),
            complexity: m.complexity,
            age: m.age,
            faulted_cells: m.faulted_cells,
            hard_fault: false,
            key: m.key,
        })
        .collect();
    let mut fewer = Archive::with_cost_version(pareto::PIPELINE_COST_VERSION).unwrap();
    fewer.update(&kept);
    edits.push(("member removed", fewer.to_json()));
    // A dominated member appended: the reader refuses it.
    edits.push((
        "dominated member",
        text.replacen(
            "]}",
            &format!(
                ",{{\"loss\":\"{}\",\"complexity\":{},\"age\":null,\"faulted_cells\":0,\"key\":\"{}\"}}]}}",
                hex((last.loss() + 1.0).to_bits()),
                last.complexity + 1,
                "0".repeat(64)
            ),
            1,
        ),
    ));
    let mut errors = Vec::new();
    for (label, edited) in &edits {
        assert_ne!(
            edited.as_bytes(),
            bytes.as_slice(),
            "{label}: the edit was not applied"
        );
        let path = rewrite(
            output,
            &receipt,
            Some(edited.as_bytes()),
            Some(digested(&engine, edited.as_bytes())),
        );
        let error = publication::verify_run(&path).unwrap_err();
        errors.push(format!("{error:#}"));
        restore();
    }
    // Record edits, the archive bytes unchanged.
    let mut records: Vec<(&str, Option<EngineRecord>)> = Vec::new();
    let mut record = engine.clone();
    record.archive.as_mut().unwrap().sha256 = Some("0".repeat(64));
    records.push(("digest", Some(record)));
    let mut record = engine.clone();
    record.archive.as_mut().unwrap().sha256 = None;
    records.push(("digest dropped", Some(record)));
    let mut record = engine.clone();
    record.archive.as_mut().unwrap().cost_version = pareto::COST_VERSION.into();
    records.push(("record cost version", Some(record)));
    let mut record = engine.clone();
    record.archive.as_mut().unwrap().id = "feat-math.front-loss-cost.v2".into();
    records.push(("record id", Some(record)));
    records.push(("record dropped, file kept", None));
    for (label, record) in records {
        assert_ne!(
            record.as_ref(),
            Some(&engine),
            "{label}: the edit was not applied"
        );
        let path = rewrite(output, &receipt, None, record);
        let error = publication::verify_run(&path).unwrap_err();
        errors.push(format!("{error:#}"));
        restore();
    }
    // The file removed under its record.
    std::fs::remove_file(output.join(archive::PATH)).unwrap();
    let error = publication::verify_run(&path).unwrap_err();
    errors.push(format!("{error:#}"));
    restore();
    publication::verify_run(&path).unwrap();
    // Each edit is refused by the check named for it, in the order above.
    let rederive = "the archive does not re-derive from the observations";
    let named = [
        ("member loss", rederive),
        ("member key", rederive),
        ("member complexity", rederive),
        (
            "cost version",
            "the archive's cost version disagrees with its record",
        ),
        ("member removed", rederive),
        (
            "dominated member",
            "archived members are not their own front",
        ),
        ("digest", "evidence digest mismatch"),
        ("digest dropped", "a published archive needs its digest"),
        (
            "record cost version",
            "unknown archive method or cost version",
        ),
        ("record id", "unknown archive method or cost version"),
        (
            "record dropped, file kept",
            "an archive is published without its selection",
        ),
        (
            "file removed, record kept",
            "a selected archive is not published",
        ),
    ];
    assert_eq!(errors.len(), named.len());
    for (error, (label, check)) in errors.iter().zip(named) {
        assert!(error.contains(check), "{label}: {error}");
    }
    // The refusals pass the shipped-text lint, with the run's random temporary path left out.
    let root = output.display().to_string();
    for error in errors {
        let text = error.replace(&root, "<run>");
        let hits = feat_math::transpile::lint_shipped_text(&text);
        assert!(hits.is_empty(), "{text}: {hits:?}");
    }
}

#[test]
fn the_selection_record_round_trips_and_unknown_records_are_refused() {
    let archived = selection(RefineMethod::Coordinate, true);
    let record = archived.record().unwrap();
    assert_eq!(
        serde_json::to_value(&record).unwrap(),
        serde_json::json!({"archive": {"id": FRONT_LOSS_COST, "cost_version": pareto::PIPELINE_COST_VERSION}})
    );
    assert_eq!(record.selection().unwrap(), archived);
    // A published record carries the digest; the selection is the same.
    let mut published = record.clone();
    published.archive.as_mut().unwrap().sha256 = Some("a".repeat(64));
    assert_eq!(published.selection().unwrap(), archived);
    let both = selection(lm(), true);
    let record = both.record().unwrap();
    assert!(record.damped_ls().is_some() && record.archive.is_some());
    assert_eq!(record.selection().unwrap(), both);
    assert_eq!(selection(RefineMethod::Coordinate, false).record(), None);
    for bad in [
        serde_json::json!({"archive": {"id": "other", "cost_version": pareto::PIPELINE_COST_VERSION}}),
        serde_json::json!({"archive": {"id": FRONT_LOSS_COST, "cost_version": pareto::COST_VERSION}}),
        serde_json::json!({"archive": {"id": FRONT_LOSS_COST, "cost_version": "feat-math.cost.pipeline-nodes.v2"}}),
    ] {
        let parsed: EngineRecord = serde_json::from_value(bad.clone()).unwrap();
        assert!(parsed.selection().is_err(), "{bad}");
    }
    assert!(serde_json::from_value::<ArchiveRecord>(serde_json::json!(
        {"id": FRONT_LOSS_COST, "cost_version": pareto::PIPELINE_COST_VERSION, "members": 3}
    ))
    .is_err());
    // The selection keeps every other check: an archive alongside an invalid refinement method.
    let invalid = EngineSelection {
        refine: RefineMethod::Lm(LmOptions {
            tau: f64::NAN,
            ..LmOptions::default()
        }),
        archive: Some(ArchivePolicy::ArchiveOnly),
        simplify: false,
        linearize: false,
    };
    assert!(evaluator().with_engine(invalid).is_err());
    let e: Kernel2Evaluator = evaluator().with_engine(archived.clone()).unwrap();
    assert_eq!(e.engine(), &archived);
    assert_eq!(e.engine_record(), archived.record());
}
