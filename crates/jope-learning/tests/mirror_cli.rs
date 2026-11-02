use super::*;

// CLI level: the reported three-step sequence, run through the built binary.
// --------------------------------------------------------------------------------------

/// `champions[skip..skip+take]` copied into a directory of its own.
fn champion_subset(dest: &Path, skip: usize, take: usize) -> PathBuf {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/champions");
    fs::create_dir_all(dest).unwrap();
    let mut names: Vec<_> = fs::read_dir(&src)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    names.sort();
    let chosen: Vec<_> = names.into_iter().skip(skip).take(take).collect();
    assert_eq!(chosen.len(), take, "fixture corpus too small");
    for name in chosen {
        fs::copy(src.join(&name), dest.join(&name)).unwrap();
    }
    dest.to_path_buf()
}

fn ingest(state: &Path, store: Option<&str>, dir: &Path, bundle: &str) -> Run {
    let mut args: Vec<&str> = Vec::new();
    if let Some(url) = store {
        args.extend(["--store", url]);
    }
    args.extend([
        "hwm",
        "ingest",
        "--source",
        "champions",
        dir.to_str().unwrap(),
        "--bundle-id",
        bundle,
    ]);
    jl(state, &args)
}

/// Every file under `root`, as paths relative to it.
fn tree(root: &Path) -> BTreeSet<String> {
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| {
            e.path()
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect()
}

#[test]
fn a_local_only_ingest_no_longer_wedges_the_mirror() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror = tmp.path().join("mirror");
    fs::create_dir_all(&mirror).unwrap();
    let url = format!("file://{}", mirror.display());
    let state = tmp.path().join("state");
    let dir_a = champion_subset(&tmp.path().join("champs-a"), 0, 5);
    let dir_b = champion_subset(&tmp.path().join("champs-b"), 5, 5);
    let mirror_head = mirror.join("hwm/ledger/HEAD.json");

    // 1. mirrored ingest: mirror HEAD = batch 1.
    ingest(&state, Some(&url), &dir_a, "a").ok();
    let head_1 = fs::read(&mirror_head).unwrap();

    // 2. the same host ingests again with NO --store: local moves to batch 2, mirror stays at 1.
    ingest(&state, None, &dir_b, "b").ok();
    assert_eq!(
        fs::read(&mirror_head).unwrap(),
        head_1,
        "an unmirrored ingest must not touch the mirror"
    );

    // 3. mirrored again. On the pre-fix code this is rc=2 `mirror pointer changed`, for ever.
    let r = ingest(&state, Some(&url), &dir_a, "a");
    assert_eq!(
        r.status, 0,
        "the mirror must catch up, not wedge; stderr: {}",
        r.stderr
    );

    // The mirror is now a complete copy: same batches, same tip, same blobs, same pointers, and
    // no object left unreachable from HEAD.
    let ledger = Ledger::open(&state, "hwm");
    assert_eq!(
        fs::read(&mirror_head).unwrap(),
        fs::read(ledger.root().join("HEAD.json")).unwrap()
    );
    let local_batches: BTreeSet<String> =
        ledger.batches().unwrap().into_iter().map(|b| b.0).collect();
    assert_eq!(
        local_batches.len(),
        3,
        "three local batches: {local_batches:?}"
    );
    let mirror_batches: BTreeSet<String> = tree(&mirror.join("hwm/ledger"))
        .into_iter()
        .filter(|p| p.ends_with(".jsonl"))
        .collect();
    assert_eq!(
        mirror_batches, local_batches,
        "the mirror must hold exactly this host's batches — no gap and no unreachable object"
    );
    assert_eq!(
        tree(&mirror.join("blobs")),
        tree(&state.join("blobs")),
        "every blob the mirrored ledger names must be on the mirror"
    );
    let st = State::at(&state);
    for (key, _) in st.pointers().unwrap() {
        assert_eq!(
            fs::read(mirror.join(key.rel())).unwrap(),
            fs::read(st.pointer_path(&key).unwrap()).unwrap(),
            "pointer {} was not caught up",
            key.rel()
        );
    }
    // And it stays unwedged.
    ingest(&state, Some(&url), &dir_a, "a").ok();
    assert_eq!(
        fs::read(&mirror_head).unwrap(),
        fs::read(ledger.root().join("HEAD.json")).unwrap()
    );
}

#[test]
fn a_diverged_mirror_is_refused_and_nothing_is_written() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror = tmp.path().join("mirror");
    fs::create_dir_all(&mirror).unwrap();
    let url = format!("file://{}", mirror.display());
    let dir_a = champion_subset(&tmp.path().join("champs-a"), 0, 5);
    let dir_b = champion_subset(&tmp.path().join("champs-b"), 5, 5);

    // Host A owns the mirror's chain.
    let state_a = tmp.path().join("state-a");
    ingest(&state_a, Some(&url), &dir_a, "host-a").ok();
    let before = tree(&mirror);
    let head_a = fs::read(mirror.join("hwm/ledger/HEAD.json")).unwrap();

    // Host B has a state tree that has never seen host A's batch: it cannot continue that chain,
    // and catching it up would mean orphaning A's tip.
    let state_b = tmp.path().join("state-b");
    let r = ingest(&state_b, Some(&url), &dir_b, "host-b");
    assert_ne!(
        r.status,
        0,
        "a diverged mirror must still fail closed; stdout: {}",
        String::from_utf8_lossy(&r.stdout)
    );
    assert!(
        r.stderr.contains("holds a chain this host cannot continue"),
        "expected the divergence refusal, got: {}",
        r.stderr
    );
    assert!(
        r.stderr.contains("re-run the same command without --store"),
        "the refusal must say what to run, got: {}",
        r.stderr
    );

    // Nothing on the mirror moved, and — because the check runs before the local append — host B
    // sealed no batch and moved no pointer either, so a retry has no wider gap to close.
    assert_eq!(tree(&mirror), before, "the mirror was written to");
    assert_eq!(
        fs::read(mirror.join("hwm/ledger/HEAD.json")).unwrap(),
        head_a
    );
    assert!(
        Ledger::open(&state_b, "hwm").batches().unwrap().is_empty(),
        "host B appended locally despite the refusal: {:?}",
        Ledger::open(&state_b, "hwm").batches().unwrap()
    );
    assert!(State::at(&state_b).pointers().unwrap().is_empty());
}

// --------------------------------------------------------------------------------------
