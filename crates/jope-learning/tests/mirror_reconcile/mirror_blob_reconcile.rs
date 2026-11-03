/// Every `record_sha256` the mirror's own ledger batches name whose blob is *not* on the mirror.
/// The mirror's copy is self-contained exactly when this is empty: `hwm verify` reads a blob per
/// event, so a name with no blob is a ledger that cannot be walked from the mirror alone.
use super::{mirror_cli::tree, mirror_head_reconcile::candidate, *};

fn named_blobs_missing_from(mirror: &Path) -> Vec<String> {
    let mut missing = Vec::new();
    for entry in walkdir::WalkDir::new(mirror.join("hwm/ledger"))
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("jsonl"))
    {
        let (events, _, _, _) = Ledger::read_batch(entry.path()).unwrap();
        for v in events {
            let sha = v["record"]["provenance"]["record_sha256"]
                .as_str()
                .expect("every event names a record sha256");
            if !mirror.join(State::blob_rel(sha)).exists() {
                missing.push(sha.to_string());
            }
        }
    }
    missing.sort();
    missing.dedup();
    missing
}

/// One failed blob PUT must not leave the mirror's `HEAD.json` naming batches whose blobs the
/// mirror does not hold.
///
/// The invariant: **while any blob a batch names is still missing from the mirror, the mirror's
/// `HEAD.json` has not moved past that batch.** It is worth exactly one thing, and that thing is
/// recovery: `sync::plan` derives `missing` from the mirror's tip alone, so a tip that has moved
/// past a batch puts that batch out of reach of every later push. Hold the invariant and the retry
/// sees the same gap and closes it; break it and the mirror's own ledger names records it will
/// never hold, while each later push reports success.
///
/// Fails on the round-two code at the first assertion: `sync::push` moved `HEAD.json` before
/// `push_mirror` uploaded the blobs, so the injected failure leaves the mirror's tip at the local
/// tip with the blobs of two batches missing, and steps 4 and 5 then backfill nothing.
#[test]
fn a_failed_blob_push_leaves_the_mirrors_head_where_it_was() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror_dir = tmp.path().join("mirror");
    fs::create_dir_all(&mirror_dir).unwrap();
    let m = Mirror::open(&format!("file://{}", mirror_dir.display())).unwrap();
    let state = State::at(&tmp.path().join("state"));
    let mirror_head = mirror_dir.join("hwm/ledger/HEAD.json");

    // 1. mirrored: the mirror holds batch 1 and the blob it names.
    hwm::ingest(&state, Some(&m), vec![candidate(0.5, 1)], json!({"k": 1})).unwrap();
    assert!(named_blobs_missing_from(&mirror_dir).is_empty());
    let head_1 = fs::read_to_string(&mirror_head).unwrap();
    let ledger_1 = tree(&mirror_dir.join("hwm/ledger"));

    // 2. local only: the mirror is behind by a batch whose blob it has never seen.
    hwm::ingest(&state, None, vec![candidate(0.4, 2)], json!({"k": 2})).unwrap();

    // 3. every blob PUT now fails: a regular file sits where the store must create the `sha256`
    //    directory, which no uid can write through. Stands in for one transient object-store
    //    error — a 503, a throttle, a revoked key.
    let blobs = mirror_dir.join("blobs/sha256");
    let stashed = tmp.path().join("blobs-sha256");
    fs::rename(&blobs, &stashed).unwrap();
    fs::write(&blobs, b"an object store that cannot take a blob\n").unwrap();
    let err = hwm::ingest(&state, Some(&m), vec![candidate(0.3, 3)], json!({"k": 3}))
        .expect_err("the blob PUT must fail the push");
    assert!(
        format!("{err:#}").contains("blobs/sha256/"),
        "expected the blob put to be what failed, got: {err:#}"
    );
    assert_eq!(
        fs::read_to_string(&mirror_head).unwrap(),
        head_1,
        "the mirror's tip moved over batches whose blobs never went up; no later push will \
         backfill them, because `plan` only ever returns batches after that tip"
    );
    assert_eq!(
        tree(&mirror_dir.join("hwm/ledger")),
        ledger_1,
        "a batch object went up before the blobs it names"
    );

    // 4. the object store recovers. The gap is still the gap, so the next push closes it — batches,
    //    the blobs they name, and the pointers they moved.
    fs::remove_file(&blobs).unwrap();
    fs::rename(&stashed, &blobs).unwrap();
    hwm::ingest(&state, Some(&m), vec![candidate(0.25, 4)], json!({"k": 4})).unwrap();

    let ledger = Ledger::open(state.root(), "hwm");
    assert_eq!(
        fs::read(&mirror_head).unwrap(),
        fs::read(ledger.root().join("HEAD.json")).unwrap(),
        "the mirror did not catch up after the store recovered"
    );
    assert_eq!(
        named_blobs_missing_from(&mirror_dir),
        Vec::<String>::new(),
        "the mirror's ledger still names blobs the mirror does not hold"
    );
    // Including the record from the batch the failed push was in the middle of, and the one it
    // never reached.
    for value in ["0.5", "0.4", "0.3", "0.25"] {
        let sha = jope_learning::canon::sha256_hex(format!("d0/cut-0/{value}").as_bytes());
        assert!(
            mirror_dir.join(State::blob_rel(&sha)).exists(),
            "blob of the {value} record is missing from the mirror"
        );
    }
    let key = pointer_key();
    assert_eq!(
        fs::read(mirror_dir.join(key.rel())).unwrap(),
        fs::read(state.pointer_path(&key).unwrap()).unwrap(),
        "the pointer did not catch up"
    );
}

/// A blob a batch names and the **local** store cannot supply must never be skipped.
///
/// The round-three ordering guarantees that while a blob is missing from the *mirror*, the
/// mirror's `HEAD.json` has not moved past the batch that names it. It says nothing about a blob
/// missing from the *local* store, and that hole was wide enough to falsify the whole guarantee:
/// `state.read_blob` returning `None` simply skipped the upload, the batch object went up anyway,
/// `HEAD.json` moved past it, and `sync::plan` — which derives the backfill window from that
/// `HEAD.json` and nothing else — never offered that batch again. The mirror was left naming a
/// record that neither host could supply, permanently, at exit 0.
///
/// Fails on the pre-fix code at the first assertion: step 4 exits `Ok`, the mirror's tip advances
/// to `batches: 3`, and the mirror's own batch events name ten record blobs while it holds nine.
#[test]
fn a_blob_the_local_store_cannot_supply_is_never_published_over() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror = tmp.path().join("mirror");
    fs::create_dir_all(&mirror).unwrap();
    let url = format!("file://{}", mirror.display());
    let state = tmp.path().join("state");
    let dir_a = champion_subset(&tmp.path().join("champs-a"), 0, 5);
    let dir_b = champion_subset(&tmp.path().join("champs-b"), 5, 5);
    let mirror_head = mirror.join("hwm/ledger/HEAD.json");

    // 1. mirrored: the mirror holds batch 1 and the five blobs it names.
    ingest(&state, Some(&url), &dir_a, "a").ok();
    let head_1 = fs::read(&mirror_head).unwrap();
    let ledger_1 = tree(&mirror.join("hwm/ledger"));

    // 2. local only: the mirror is a batch behind, and has never seen that batch's blobs.
    ingest(&state, None, &dir_b, "b").ok();

    // 3. one blob local batch 2 names is lost from the *local* store — a pruned cache, a bad
    //    disk, an operator `rm`. The ledger still names it; `hwm verify` calls it `blob-missing`.
    let ledger = Ledger::open(&state, "hwm");
    let batch_2 = ledger.batches().unwrap().into_iter().next_back().unwrap().1;
    let (events, _, _, _) = Ledger::read_batch(&batch_2).unwrap();
    let lost = events[0]["record"]["provenance"]["record_sha256"]
        .as_str()
        .unwrap()
        .to_string();
    let lost_path = State::at(&state).blob_path(&lost);
    let lost_bytes = fs::read(&lost_path).unwrap();
    fs::remove_file(&lost_path).unwrap();

    // 4. mirrored again. The push must refuse: publishing batch 2 would move the mirror's tip
    //    past a batch whose blob nobody holds, and no later push would ever come back for it.
    let r = ingest(&state, Some(&url), &dir_a, "a");
    assert_ne!(
        r.status,
        0,
        "a blob no one can supply was published over; stdout: {}",
        String::from_utf8_lossy(&r.stdout)
    );
    assert!(
        r.stderr.contains(&lost)
            && r.stderr
                .contains("Refusing to publish a batch the mirror could not back"),
        "the refusal must name the blob it could not supply, got: {}",
        r.stderr
    );
    assert_eq!(
        named_blobs_missing_from(&mirror),
        Vec::<String>::new(),
        "the mirror's own ledger names a blob the mirror does not hold"
    );
    assert_eq!(
        fs::read(&mirror_head).unwrap(),
        head_1,
        "the mirror's tip moved over a batch naming a blob nobody can supply"
    );
    assert_eq!(
        tree(&mirror.join("hwm/ledger")),
        ledger_1,
        "a batch object went up before the blobs it names could be supplied"
    );

    // 5. the blob is restored from a bundle. The gap is still the gap, so the next push closes it.
    fs::write(&lost_path, &lost_bytes).unwrap();
    ingest(&state, Some(&url), &dir_a, "a").ok();
    assert_eq!(
        fs::read(&mirror_head).unwrap(),
        fs::read(ledger.root().join("HEAD.json")).unwrap(),
        "the mirror did not catch up once the blob was restored"
    );
    assert_eq!(named_blobs_missing_from(&mirror), Vec::<String>::new());
    assert_eq!(
        tree(&mirror.join("blobs")),
        tree(&state.join("blobs")),
        "every blob the mirrored ledger names must be on the mirror"
    );
}

/// The negative control for the refusal above: a blob only the *mirror* still has is not a gap,
/// and must not wedge the push.
///
/// Failing closed on `read_blob == None` alone would turn every pruned local blob into a mirror
/// that can never be written again — the same one-way door the reconcile was built to remove. The
/// batch is publishable exactly when *someone* can back it, so the local store is asked first and
/// the mirror second.
#[test]
fn a_blob_only_the_mirror_still_has_does_not_wedge_the_push() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror_dir = tmp.path().join("mirror");
    fs::create_dir_all(&mirror_dir).unwrap();
    let m = Mirror::open(&format!("file://{}", mirror_dir.display())).unwrap();
    let state = State::at(&tmp.path().join("state"));

    // Both records are on the mirror, pushed while the local store still had their blobs.
    hwm::ingest(&state, Some(&m), vec![candidate(0.5, 1)], json!({"k": 1})).unwrap();
    hwm::ingest(&state, Some(&m), vec![candidate(0.4, 2)], json!({"k": 2})).unwrap();

    // A local-only batch of one tie: it names the 0.4 record again, and nothing else.
    hwm::ingest(&state, None, vec![candidate(0.4, 3)], json!({"k": 3})).unwrap();

    // The local blob for 0.4 is pruned. The mirror still has it, so the backfilled batch is
    // already backed and the push has nothing to upload for it.
    let sha = jope_learning::canon::sha256_hex(b"d0/cut-0/0.4");
    assert!(mirror_dir.join(State::blob_rel(&sha)).exists());
    fs::remove_file(state.blob_path(&sha)).unwrap();

    let rep = hwm::ingest(&state, Some(&m), vec![candidate(0.3, 4)], json!({"k": 4}))
        .expect("a blob the mirror already holds must not wedge the push");
    let ledger = Ledger::open(state.root(), "hwm");
    assert_eq!(
        fs::read(mirror_dir.join("hwm/ledger/HEAD.json")).unwrap(),
        fs::read(ledger.root().join("HEAD.json")).unwrap(),
        "the mirror did not catch up"
    );
    assert_eq!(named_blobs_missing_from(&mirror_dir), Vec::<String>::new());
    assert_eq!(rep.accepted, 1);
}

/// The mirror-only fallback must prove **identity**, not presence.
///
/// `read_blob` re-hashes what it reads and `put_immutable` compares bytes on an `Exists`, so every
/// other way a record enters this system is checked against the sha that names it. The fallback
/// was the one path that accepted an object this host never wrote, and it accepted it on the
/// strength of a successful GET alone. One wrong object under the immutable key — a truncated
/// upload, a key reused by an unrelated writer, a restore from the wrong bundle — would then
/// launder itself into a published, `HEAD`-reachable batch that no reader can verify, and
/// `hwm verify` against the mirror reports `blob-corrupt` for a record the push called backed.
///
/// Fails on the pre-fix code at the first assertion: the ingest returns `Ok` and the mirror's tip
/// advances over a batch whose record the mirror does not actually hold.
#[test]
fn a_wrong_object_under_the_immutable_key_cannot_stand_in_for_the_blob() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror_dir = tmp.path().join("mirror");
    fs::create_dir_all(&mirror_dir).unwrap();
    let m = Mirror::open(&format!("file://{}", mirror_dir.display())).unwrap();
    let state = State::at(&tmp.path().join("state"));
    let mirror_head = mirror_dir.join("hwm/ledger/HEAD.json");

    // 1. mirrored: the mirror holds batch 1 and the blob it names.
    hwm::ingest(&state, Some(&m), vec![candidate(0.5, 1)], json!({"k": 1})).unwrap();
    let head_1 = fs::read(&mirror_head).unwrap();
    let ledger_1 = tree(&mirror_dir.join("hwm/ledger"));

    // 2. local only: the mirror is a batch behind and has never seen that batch's blob.
    hwm::ingest(&state, None, vec![candidate(0.4, 2)], json!({"k": 2})).unwrap();

    // 3. the local blob is lost, and something else is sitting under its immutable key on the
    //    mirror. Nothing on this path re-reads it, so nothing has ever checked it.
    let sha = jope_learning::canon::sha256_hex(b"d0/cut-0/0.4");
    fs::remove_file(state.blob_path(&sha)).unwrap();
    let rel = State::blob_rel(&sha);
    let wrong = b"wrong bytes under the required immutable key".to_vec();
    fs::create_dir_all(mirror_dir.join(&rel).parent().unwrap()).unwrap();
    fs::write(mirror_dir.join(&rel), &wrong).unwrap();

    // 4. the push must refuse: a GET that returns *something* is not proof the mirror can back
    //    the record.
    let err = hwm::ingest(&state, Some(&m), vec![candidate(0.3, 3)], json!({"k": 3}))
        .expect_err("mirror blob presence was taken as proof of content identity");
    let msg = format!("{err:#}");
    assert!(
        msg.contains(&sha) && msg.contains("not this record"),
        "the refusal must name the record and say the mirror's object is not it, got: {msg}"
    );
    assert_eq!(
        fs::read(&mirror_head).unwrap(),
        head_1,
        "the mirror's tip moved over a batch the mirror cannot back"
    );
    assert_eq!(
        tree(&mirror_dir.join("hwm/ledger")),
        ledger_1,
        "a batch object went up although its record could not be supplied"
    );
    // The refusal does not "repair" the mirror by overwriting an immutable key: whatever is there
    // is another writer's business, and this host has nothing to replace it with anyway.
    assert_eq!(
        fs::read(mirror_dir.join(&rel)).unwrap(),
        wrong,
        "the push overwrote an immutable key to make itself succeed"
    );

    // 5. restore the real blob locally. The correct bytes now conflict with the wrong object, and
    //    `put_immutable` refuses that too rather than silently replacing it — the push still fails
    //    closed, and still leaves the tip and the foreign object alone.
    state.put_blob(b"d0/cut-0/0.4").unwrap();
    let err = hwm::ingest(&state, Some(&m), vec![candidate(0.25, 4)], json!({"k": 4}))
        .expect_err("an immutable key holding different bytes must not be overwritten");
    assert!(
        format!("{err:#}").contains("different bytes"),
        "expected the immutable-collision refusal, got: {err:#}"
    );
    assert_eq!(fs::read(&mirror_head).unwrap(), head_1);
    assert_eq!(fs::read(mirror_dir.join(&rel)).unwrap(), wrong);
}
