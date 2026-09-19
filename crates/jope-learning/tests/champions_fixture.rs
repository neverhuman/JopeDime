//! Ten real `parallel.cut_champion.v1` files (8 training/validation + 2 sealed) with their engine
//! event chains: the engine canonicalisation is reproduced byte-for-byte, the ingest preserves the
//! original bytes, and `hwm verify` walks every chain.

mod common;

use common::{fixtures, jl};
use jope_learning::champion::Champion;
use jope_learning::hwm;
use jope_learning::sources::champions;
use jope_learning::state::State;
use std::fs;

#[test]
fn engine_canonicalisation_is_reproduced_on_real_files() {
    let dir = fixtures().join("champions");
    let events = fixtures().join("champion-events");
    let mut checked = 0;
    for (stem, path) in champions::champion_files(&dir).unwrap() {
        let bytes = fs::read(&path).unwrap();
        let ch = Champion::parse(&bytes).unwrap();
        ch.validate().unwrap();
        // The file name is Identity::key().
        assert_eq!(ch.identity.key().unwrap(), stem);
        // current.json re-serialises to the sha256 that names its own event file.
        let own = ch.event_sha256().unwrap();
        let own_path = events.join(&stem).join(format!("event-{own}.json"));
        assert!(own_path.exists(), "{}", own_path.display());
        for e in fs::read_dir(events.join(&stem)).unwrap() {
            let e = e.unwrap();
            let name = e.file_name().to_string_lossy().into_owned();
            let sha = name
                .strip_prefix("event-")
                .and_then(|s| s.strip_suffix(".json"))
                .unwrap();
            let ev = Champion::parse(&fs::read(e.path()).unwrap()).unwrap();
            assert_eq!(ev.event_sha256().unwrap(), sha, "{name}");
            assert_eq!(ev.identity, ch.identity);
            checked += 1;
        }
    }
    assert_eq!(checked, 19);
}

#[test]
fn ingest_preserves_bytes_and_verify_walks_every_chain() {
    let tmp = tempfile::tempdir().unwrap();
    let state_dir = tmp.path().join("state");
    let dir = fixtures().join("champions");
    let events = fixtures().join("champion-events");
    let rep = jl(
        &state_dir,
        &[
            "hwm",
            "ingest",
            "--source",
            "champions",
            dir.to_str().unwrap(),
            "--bundle-id",
            "fixture",
        ],
    )
    .ok()
    .json();
    assert_eq!(rep["candidates"], 10);
    assert_eq!(rep["accepted"], 10);
    assert_eq!(rep["rejected"], 0);
    assert_eq!(rep["pointers_written"], 10);
    assert_eq!(rep["blobs_written"], 10);
    assert_eq!(rep["scan"]["sealed"], 2);

    // `hwm show --raw` returns the identical bytes for every fixture.
    let state = State::at(&state_dir);
    let mut sealed = 0;
    for (stem, path) in champions::champion_files(&dir).unwrap() {
        let bytes = fs::read(&path).unwrap();
        let ch = Champion::parse(&bytes).unwrap();
        let cut = champions::cut_id(&ch.identity).unwrap();
        let shown = jl(
            &state_dir,
            &["hwm", "show", &ch.identity.dataset, "--cut", &cut, "--raw"],
        )
        .ok();
        assert_eq!(shown.stdout, bytes, "{stem}");
        let entry = jl(
            &state_dir,
            &["hwm", "show", &ch.identity.dataset, "--cut", &cut],
        )
        .ok()
        .json();
        let r = &entry[0]["record"];
        assert_eq!(r["legacy_identity_sha256"], stem);
        assert_eq!(r["score"]["metric"], "rmse");
        assert_eq!(r["score"]["value"], ch.rmse_original_units);
        assert_eq!(r["improvement_number"], ch.improvement_number);
        assert_eq!(
            r["prev_event_sha256"],
            serde_json::json!(ch.previous_event_sha256)
        );
        assert_eq!(r["ts_unix"], ch.timestamp);
        assert_eq!(r["sealed"], ch.identity.is_sealed());
        assert_eq!(r["evaluator"]["sha256"], ch.identity.evaluator_sha256);
        sealed += usize::from(ch.identity.is_sealed());
        let found = jl(&state_dir, &["hwm", "find", &stem]).ok().json();
        assert_eq!(found["record"]["legacy_identity_sha256"], stem);
        let blob = jl(
            &state_dir,
            &[
                "hwm",
                "blob",
                r["provenance"]["record_sha256"].as_str().unwrap(),
            ],
        )
        .ok();
        assert_eq!(blob.stdout, bytes);
    }
    assert_eq!(sealed, 2);

    // Verify with the engine event directories: every chain link resolves.
    let v = hwm::verify(&state, std::slice::from_ref(&events)).unwrap();
    assert!(v.ok(), "{:?}", v.breaks);
    assert_eq!(
        (v.events, v.accepted, v.identities, v.pointers_on_disk),
        (10, 10, 10, 10)
    );
    assert_eq!(v.champion_records, 10);
    assert_eq!(v.champion_chain_links_walked, 9);
    assert_eq!(v.champion_chain_links_unresolved, 0);
    // Without event directories the chain fields are still checked, links just cannot be walked.
    let v = hwm::verify(&state, &[]).unwrap();
    assert!(v.ok(), "{:?}", v.breaks);
    assert_eq!(v.champion_chain_links_unresolved, 5);
    let cli = jl(
        &state_dir,
        &["hwm", "verify", "--events", events.to_str().unwrap()],
    )
    .ok()
    .json();
    assert_eq!(cli["breaks"].as_array().unwrap().len(), 0);
    assert_eq!(cli["ledger"]["breaks"].as_array().unwrap().len(), 0);

    // Re-ingesting the same winners changes nothing: 10 ties, no pointer moves, chain intact.
    let rep = jl(
        &state_dir,
        &[
            "hwm",
            "ingest",
            "--source",
            "champions",
            dir.to_str().unwrap(),
            "--show-rejects",
        ],
    )
    .ok()
    .json();
    assert_eq!(rep["accepted"], 0);
    assert_eq!(rep["rejected"], 10);
    assert_eq!(rep["pointers_written"], 0);
    assert!(rep["rejects"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["reason"] == "tie"));
    let v = hwm::verify(&state, std::slice::from_ref(&events)).unwrap();
    assert!(v.ok(), "{:?}", v.breaks);
    assert_eq!((v.events, v.accepted, v.rejected), (20, 10, 10));

    // Freeze: sealed winners never leave the ledger.
    let out = tmp.path().join("freeze");
    let f = jl(
        &state_dir,
        &["hwm", "freeze", "--out", out.to_str().unwrap()],
    )
    .ok()
    .json();
    assert_eq!(
        (f["included"].as_u64(), f["excluded_sealed"].as_u64()),
        (Some(8), Some(2))
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["entries"].as_array().unwrap().len(), 8);
    for e in manifest["entries"].as_array().unwrap() {
        let blob = out.join(e["blob"].as_str().unwrap());
        let original = dir.join(format!(
            "{}.json",
            e["legacy_identity_sha256"].as_str().unwrap()
        ));
        assert_eq!(fs::read(blob).unwrap(), fs::read(original).unwrap());
    }

    // Tamper with the first batch: verify reports the break and exits 1.
    let ledger = jope_learning::ledger::Ledger::open(&state_dir, "hwm");
    let (_, first) = ledger.batches().unwrap().into_iter().next().unwrap();
    let mut bytes = fs::read(&first).unwrap();
    let i = bytes
        .windows(14)
        .position(|w| w == b"\"improvement_n")
        .unwrap();
    bytes[i + 1] = b'I';
    fs::write(&first, &bytes).unwrap();
    let r = jl(&state_dir, &["hwm", "verify"]);
    assert_eq!(r.status, 1);
    let kinds: Vec<String> = r.json()["ledger"]["breaks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["kind"].as_str().unwrap().to_string())
        .collect();
    assert!(kinds.contains(&"events-sha256".to_string()), "{kinds:?}");
    // And no further append is accepted on a broken chain.
    let r = jl(
        &state_dir,
        &[
            "hwm",
            "ingest",
            "--source",
            "champions",
            dir.to_str().unwrap(),
        ],
    );
    assert_eq!(r.status, 2);
    assert!(r.stderr.contains("chain broken"), "{}", r.stderr);
}

#[test]
fn mirror_receives_batches_blobs_and_pointers() {
    let tmp = tempfile::tempdir().unwrap();
    let state_dir = tmp.path().join("state");
    let mirror = tmp.path().join("mirror");
    fs::create_dir_all(&mirror).unwrap();
    let url = format!("file://{}", mirror.display());
    let dir = fixtures().join("champions");
    jl(
        &state_dir,
        &[
            "--store",
            &url,
            "hwm",
            "ingest",
            "--source",
            "champions",
            dir.to_str().unwrap(),
        ],
    )
    .ok();
    let ledger = jope_learning::ledger::Ledger::open(&state_dir, "hwm");
    let head = ledger.head().unwrap().unwrap();
    assert_eq!(
        fs::read(mirror.join("hwm/ledger").join(&head.path)).unwrap(),
        fs::read(ledger.root().join(&head.path)).unwrap()
    );
    assert_eq!(
        fs::read(mirror.join("hwm/ledger/HEAD.json")).unwrap(),
        fs::read(ledger.root().join("HEAD.json")).unwrap()
    );
    let state = State::at(&state_dir);
    for (key, record) in state.pointers().unwrap() {
        assert_eq!(
            fs::read(mirror.join(key.rel())).unwrap(),
            fs::read(state.pointer_path(&key).unwrap()).unwrap()
        );
        assert!(mirror
            .join(State::blob_rel(&record.provenance.record_sha256))
            .exists());
    }
}
