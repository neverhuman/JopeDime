use super::*;

/// The reviewer's PR0-R4 regression test (review 03), adopted with its name and intent. Its
/// scratch directory is in a fresh test directory instead of the review's evidence directory.
#[test]
pub(super) fn review03_legacy_followup_preserves_kernel2_preparation() {
    let root = scratch();
    let dir = root.path().join("rust-legacy-followup");
    let config = shared_scratch_config();
    let first = run(&config, &dir).unwrap();
    let record = &first.document.body.datasets[0];
    let path = dir
        .join("prepared")
        .join(&record.dataset_hash)
        .join("prepared.json");
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(sha256_hex(&bytes), record.prepared_sha256);
    let legacy = Config {
        profile_a: Profile::Legacy,
        profile_b: Profile::Legacy,
        seeds: vec![7],
        fold_seed: 20_260_908,
        refinement: None,
        ..config
    };
    let second = run(&legacy, &dir);
    println!("legacy followup successful: {}", second.is_ok());
    let after = std::fs::read(path).unwrap();
    println!(
        "before {} after {}",
        record.prepared_sha256,
        sha256_hex(&after)
    );
    assert_eq!(
        bytes, after,
        "later comparison invalidated first prepared_sha256"
    );
}

/// The reviewer's PR0-R6 regression test (review 03), adopted with its name and intent. Its
/// directories are in a fresh test directory instead of the review's evidence directory.
#[test]
pub(super) fn review03_concurrent_library_runs_preserve_both_preparations() {
    use std::sync::{Arc, Barrier};
    let temporary = scratch();
    let root = temporary.path().join("concurrent");
    let mut both_succeeded = 0;
    let mut broken = 0;
    let mut attempts = Vec::new();
    for attempt in 0..8 {
        let scratch = root.join(format!("attempt-{attempt}"));
        let barrier = Arc::new(Barrier::new(2));
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = [6, 7]
                .into_iter()
                .map(|seed| {
                    let scratch = scratch.clone();
                    let barrier = barrier.clone();
                    scope.spawn(move || {
                        let config = Config {
                            seeds: vec![seed],
                            fold_seed: 20_260_907 + (seed - 6),
                            ..shared_scratch_config()
                        };
                        barrier.wait();
                        run(&config, &scratch)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        let mut detail = Vec::new();
        for (index, result) in results.iter().enumerate() {
            match result {
                Ok(report) => {
                    write_document(
                        &root.join(format!("attempt-{attempt}-{index}.json")),
                        &report.document,
                    )
                    .unwrap();
                    let record = &report.document.body.datasets[0];
                    let actual = sha256_hex(
                        &std::fs::read(
                            scratch
                                .join("prepared")
                                .join(&record.dataset_hash)
                                .join("prepared.json"),
                        )
                        .unwrap(),
                    );
                    let valid = actual == record.prepared_sha256;
                    if !valid {
                        broken += 1;
                    }
                    detail.push(serde_json::json!({"success":true,"prepared_sha256":record.prepared_sha256,"actual_sha256":actual,"valid":valid}));
                }
                Err(e) => {
                    detail.push(serde_json::json!({"success":false,"error":format!("{e:#}")}))
                }
            }
        }
        if results.iter().all(Result::is_ok) {
            both_succeeded += 1;
        }
        attempts.push(serde_json::json!({"attempt":attempt,"runs":detail}));
        if broken > 0 {
            break;
        }
    }
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
            root.join("result.json"),
            serde_json::to_vec_pretty(&serde_json::json!({"attempts":attempts,"both_succeeded":both_succeeded,"broken":broken})).unwrap(),
        )
        .unwrap();
    println!("both succeeded {both_succeeded}, invalid documents {broken}");
    assert_eq!(
        broken, 0,
        "same-process comparisons broke each other's partition evidence"
    );
}

/// Review finding PR0-R4, reopened: after a comparison with a kernel2 profile, a legacy-only
/// comparison into the same scratch directory is refused before anything is written, with
/// a new seed and fold seed (the review's reproducer) and with the same ones. Every entry
/// keeps its kind, modification time and bytes, the kernel2 marker included, and the first
/// document's `prepared_sha256` still names `prepared.json`.
#[test]
pub(super) fn ablation_legacy_refuses_a_scratch_a_kernel2_comparison_used() {
    let dir = scratch();
    let first = run(&shared_scratch_config(), dir.path()).unwrap();
    let record = &first.document.body.datasets[0];
    let before = tree(dir.path());
    for legacy in [
        legacy_scratch_config(7, 20_260_908),
        legacy_scratch_config(6, 20_260_907),
    ] {
        scratch_refusal_saying(&legacy, dir.path(), SCRATCH_HOLDS_KERNEL2);
        let changed = changed_entries(&before, &tree(dir.path()));
        assert!(
            changed.is_empty(),
            "{legacy:?}: a refused comparison changed {changed:?}"
        );
    }
    assert!(before
        .get(Path::new(KERNEL2_MARKER))
        .is_some_and(|entry| entry.0 == "file"));
    let prepared = dir
        .path()
        .join("prepared")
        .join(&record.dataset_hash)
        .join("prepared.json");
    assert_eq!(
        sha256_hex(&std::fs::read(prepared).unwrap()),
        record.prepared_sha256
    );
}

/// After a legacy-only comparison, a comparison with a kernel2 profile into the same scratch
/// directory is refused because the scratch is not empty, with nothing changed and no marker
/// written.
#[test]
pub(super) fn ablation_kernel2_refuses_a_scratch_a_legacy_comparison_used() {
    let dir = scratch();
    run(&legacy_scratch_config(7, 20_260_908), dir.path()).unwrap();
    let before = tree(dir.path());
    scratch_refusal(&shared_scratch_config(), dir.path());
    let changed = changed_entries(&before, &tree(dir.path()));
    assert!(
        changed.is_empty(),
        "a refused comparison changed {changed:?}"
    );
    assert_eq!(names(dir.path()), set(&["prepared", "search"]));
}

/// A legacy-only comparison into a scratch directory another legacy-only comparison used is
/// admitted as before, here with a new seed and fold seed: its body bytes are those of the
/// same comparison in a fresh scratch, and neither scratch holds a lock file or a kernel2
/// marker afterwards.
#[test]
pub(super) fn ablation_legacy_reruns_into_a_legacy_scratch_as_before() {
    let dir = scratch();
    run(&legacy_scratch_config(6, 20_260_907), dir.path()).unwrap();
    let again = legacy_scratch_config(7, 20_260_908);
    let second = run(&again, dir.path()).unwrap();
    let fresh = scratch();
    let alone = run(&again, fresh.path()).unwrap();
    assert_eq!(
        String::from_utf8_lossy(&second.body_bytes),
        String::from_utf8_lossy(&alone.body_bytes)
    );
    for dir in [dir.path(), fresh.path()] {
        assert_eq!(names(dir), set(&["prepared", "search"]));
    }
}

/// The kernel2 marker stays after a comparison with a kernel2 profile, and a legacy-only
/// comparison writes none; neither leaves its lock file.
#[test]
pub(super) fn ablation_kernel2_marker_persists_and_legacy_writes_none() {
    let dir = scratch();
    run(&shared_scratch_config(), dir.path()).unwrap();
    assert_eq!(
        names(dir.path()),
        set(&[KERNEL2_MARKER, "prepared", "search"])
    );
    assert!(dir.path().join(KERNEL2_MARKER).is_file());
    let legacy = scratch();
    run(&legacy_scratch_config(6, 20_260_907), legacy.path()).unwrap();
    assert_eq!(names(legacy.path()), set(&["prepared", "search"]));
}

/// Review finding PR0-R6: a scratch directory whose lock file exists is refused by every
/// comparison, naming the lock and leaving it as it is, a stale one (no comparison running)
/// included; a symbolic link there counts too. A comparison removes its own lock after it
/// succeeds and after a refusal made under its reservation.
#[test]
pub(super) fn ablation_a_lock_file_refuses_and_a_comparison_removes_its_own() {
    let dir = scratch();
    let lock = dir.path().join(LOCK_FILE);
    std::fs::write(&lock, b"pid 1 started_unix 0\n").unwrap();
    let before = tree(dir.path());
    for config in [
        legacy_scratch_config(6, 20_260_907),
        shared_scratch_config(),
    ] {
        let error = scratch_refusal_saying(&config, dir.path(), SCRATCH_RESERVED);
        assert!(
            error.contains(&format!("({})", lock.display()))
                && error.contains("remove the lock file"),
            "{error}"
        );
        let changed = changed_entries(&before, &tree(dir.path()));
        assert!(
            changed.is_empty(),
            "{config:?}: a refused comparison changed {changed:?}"
        );
    }
    #[cfg(unix)]
    {
        let linked = scratch();
        std::os::unix::fs::symlink("missing", linked.path().join(LOCK_FILE)).unwrap();
        let before = tree(linked.path());
        scratch_refusal_saying(&shared_scratch_config(), linked.path(), SCRATCH_RESERVED);
        let changed = changed_entries(&before, &tree(linked.path()));
        assert!(
            changed.is_empty(),
            "a refused comparison changed {changed:?}"
        );
    }
    std::fs::remove_file(&lock).unwrap();
    run(&shared_scratch_config(), dir.path()).unwrap();
    assert!(
        std::fs::symlink_metadata(&lock).is_err(),
        "the lock outlived its comparison"
    );
    let before = tree(dir.path());
    scratch_refusal_saying(&shared_scratch_config(), dir.path(), SCRATCH_IN_USE);
    scratch_refusal_saying(
        &legacy_scratch_config(7, 20_260_908),
        dir.path(),
        SCRATCH_HOLDS_KERNEL2,
    );
    assert!(
        std::fs::symlink_metadata(&lock).is_err(),
        "a refused comparison left its lock"
    );
    let changed = changed_entries(&before, &tree(dir.path()));
    assert!(
        changed.is_empty(),
        "a refused comparison changed {changed:?}"
    );
}

/// The reservation removes its lock file, and nothing else, when it is dropped, also while a
/// panic unwinds.
#[test]
pub(super) fn ablation_reservation_is_released_while_unwinding() {
    let dir = scratch();
    let lock = dir.path().join(LOCK_FILE);
    std::fs::write(dir.path().join("kept"), b"kept").unwrap();
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _reservation = Reservation::take(dir.path()).unwrap();
        assert!(lock.is_file());
        panic!("unwinding with the reservation held");
    }));
    assert!(unwound.is_err());
    assert!(std::fs::symlink_metadata(&lock).is_err());
    drop(Reservation::take(dir.path()).unwrap());
    assert_eq!(names(dir.path()), set(&["kept"]));
}

/// Review finding PR0-R6: comparisons that start together on one absent scratch directory
/// (three threads behind a barrier: seeds 6, 7 and 6 with fold seeds 20260907, 20260908 and
/// 20260907; several attempts). Exactly one succeeds and every other one is refused. The
/// scratch then holds the winner's evidence alone: its kernel2 marker, its `prepared.json`
/// with the document's `prepared_sha256`, and in each slot only the search of its seed,
/// whose published best program and RMSE are the document's M1. No lock file is left.
#[test]
pub(super) fn ablation_concurrent_comparisons_on_one_scratch_admit_exactly_one() {
    use std::sync::{Arc, Barrier};
    let contenders = [(6, 20_260_907), (7, 20_260_908), (6, 20_260_907)];
    let refusals = [
        SCRATCH_RESERVED,
        SCRATCH_HOLDS_KERNEL2,
        SCRATCH_IN_USE,
        SCRATCH_NOT_UNUSED,
    ];
    for attempt in 0..4 {
        let root = scratch();
        let dir = root.path().join("scratch");
        let barrier = Arc::new(Barrier::new(contenders.len()));
        let results: Vec<(u64, Result<Report>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = contenders
                .iter()
                .map(|&(seed, fold_seed)| {
                    let (dir, barrier) = (dir.clone(), barrier.clone());
                    scope.spawn(move || {
                        let config = Config {
                            seeds: vec![seed],
                            fold_seed,
                            ..shared_scratch_config()
                        };
                        barrier.wait();
                        (seed, run(&config, &dir))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });
        let winners: Vec<(u64, &Report)> = results
            .iter()
            .filter_map(|(seed, result)| result.as_ref().ok().map(|report| (*seed, report)))
            .collect();
        let [(seed, report)] = winners.as_slice() else {
            panic!("attempt {attempt}: {} comparisons succeeded", winners.len());
        };
        for error in results
            .iter()
            .filter_map(|(_, result)| result.as_ref().err())
        {
            let error = format!("{error:#}");
            assert!(
                refusals.iter().any(|refusal| error.contains(refusal)),
                "attempt {attempt}: {error}"
            );
            let text = error.replace(&dir.display().to_string(), "<scratch>");
            let hits = feat_math::transpile::lint_shipped_text(&text);
            assert!(hits.is_empty(), "{text:?}: {hits:?}");
        }
        let record = &report.document.body.datasets[0];
        let hash = &record.dataset_hash;
        let prepared = dir.join("prepared").join(hash).join("prepared.json");
        assert_eq!(
            sha256_hex(&std::fs::read(prepared).unwrap()),
            record.prepared_sha256,
            "attempt {attempt}"
        );
        for slot in ["A", "B"] {
            let output = search_output(&dir, slot, hash, *seed);
            assert_eq!(
                names(output.parent().unwrap()),
                BTreeSet::from([format!("seed-{seed}")]),
                "attempt {attempt} slot {slot}"
            );
            let published: serde_json::Value = serde_json::from_slice(
                &std::fs::read(output.join("search/cut-0/search.json")).unwrap(),
            )
            .unwrap();
            let m1 = &report
                .document
                .body
                .runs
                .iter()
                .find(|searched| searched.slot == slot)
                .unwrap()
                .metrics
                .m1;
            let best = &published["best"]["observation"];
            assert_eq!(
                best["rmse"].as_f64().map(f64::to_bits),
                Some(m1.rmse.to_bits()),
                "attempt {attempt} slot {slot}"
            );
            assert_eq!(best["genome"].as_str(), Some(m1.program.as_str()));
            assert_eq!(published["rng_seed"].as_u64(), Some(*seed));
        }
        assert_eq!(
            names(&dir),
            set(&[KERNEL2_MARKER, "prepared", "search"]),
            "attempt {attempt}"
        );
    }
}

/// A symbolic link to an empty directory is refused as the scratch of a comparison with a
/// kernel2 profile also when the path ends in `/`, which would make `symlink_metadata`
/// follow the link, with nothing written through it.
#[cfg(unix)]
#[test]
pub(super) fn ablation_kernel2_refuses_a_symlinked_scratch_ending_in_a_slash() {
    let root = scratch();
    std::fs::create_dir(root.path().join("empty")).unwrap();
    std::os::unix::fs::symlink("empty", root.path().join("link")).unwrap();
    let slash = PathBuf::from(format!("{}/", root.path().join("link").display()));
    let before = tree(root.path());
    scratch_refusal(&shared_scratch_config(), &slash);
    let changed = changed_entries(&before, &tree(root.path()));
    assert!(
        changed.is_empty(),
        "a refused comparison changed {changed:?}"
    );
}

// Review 04: finding PR0-R8 (the --out document write).

/// [`shared_scratch_args`] for a legacy-only comparison: legacy A=A, seed 7, fold seed
/// 20260908, as [`legacy_scratch_config`]`(7, 20_260_908)`.
pub(super) fn legacy_out_args(scratch: PathBuf, out: PathBuf) -> AblationArgs {
    AblationArgs {
        profile_a: Profile::Legacy.id().to_owned(),
        profile_b: Profile::Legacy.id().to_owned(),
        refine_top_k: None,
        refine_trials: None,
        seeds: vec![7],
        fold_seed: 20_260_908,
        ..shared_scratch_args(scratch, out)
    }
}

/// `execute` refused for its `--out`: the message holds [`OUT_REFUSED`] and `reason` and,
/// with the test directory left out, passes the shipped-text lint; nothing under `root`
/// changed (no scratch created, no document written, no evidence touched).
pub(super) fn out_refusal(args: &AblationArgs, root: &Path, reason: &str) -> String {
    let before = tree(root);
    let error = format!("{:#}", execute(args).unwrap_err());
    assert!(
        error.contains(OUT_REFUSED) && error.contains(reason),
        "{reason:?} not in {error}"
    );
    let mut text = error.clone();
    for prefix in [std::fs::canonicalize(root).unwrap(), root.to_path_buf()] {
        text = text.replace(&prefix.display().to_string(), "<root>");
    }
    let hits = feat_math::transpile::lint_shipped_text(&text);
    assert!(hits.is_empty(), "{text:?}: {hits:?}");
    let changed = changed_entries(&before, &tree(root));
    assert!(changed.is_empty(), "a refused --out changed {changed:?}");
    error
}

/// The independent reviewer's PR0-R8 regression test (review 04), adopted with its name and
/// intent. Its directories are in a fresh test directory instead of the review's evidence
/// directory.
#[test]
pub(super) fn review04_execute_output_preserves_marked_kernel2_evidence() {
    let temporary = scratch();
    let root = temporary.path().join("out-regression");
    let first_dir = root.join("first");
    let first = run(&shared_scratch_config(), &first_dir).unwrap();
    write_document(&root.join("first.json"), &first.document).unwrap();
    let record = &first.document.body.datasets[0];
    let prepared = first_dir
        .join("prepared")
        .join(&record.dataset_hash)
        .join("prepared.json");
    let before = std::fs::read(&prepared).unwrap();
    let mut args = shared_scratch_args(root.join("second"), prepared.clone());
    args.profile_a = "legacy".to_owned();
    args.profile_b = "legacy".to_owned();
    args.refine_top_k = None;
    args.refine_trials = None;
    args.seeds = vec![7];
    args.fold_seed = 20_260_908;
    let result = execute(&args);
    let after = std::fs::read(&prepared).unwrap();
    println!(
        "second execute success: {}; before: {}; after: {}; marker exists: {}",
        result.is_ok(),
        sha256_hex(&before),
        sha256_hex(&after),
        first_dir.join(KERNEL2_MARKER).exists()
    );
    assert_eq!(
        sha256_hex(&after),
        record.prepared_sha256,
        "a successful execute --out bypassed the marker and invalidated the earlier document"
    );
}

/// Review finding PR0-R8, the self-collision: a comparison with a kernel2 profile whose
/// `--out` names its own `prepared.json`, or any other path inside its own `--scratch`, is
/// refused before the scratch is created or anything is computed, also when the scratch
/// already exists empty.
#[test]
pub(super) fn ablation_execute_refuses_an_out_inside_its_own_kernel2_scratch() {
    let root = scratch();
    let hash = load_dataset(&d5("pmlb-first-principles-supernovae-zg"))
        .unwrap()
        .row
        .dataset_hash;
    let own = root.path().join("scratch");
    let reason = "inside this comparison's kernel2 --scratch";
    for out in [
        own.join("prepared").join(&hash).join("prepared.json"),
        own.join("ablation.json"),
    ] {
        out_refusal(&shared_scratch_args(own.clone(), out), root.path(), reason);
    }
    std::fs::create_dir(&own).unwrap();
    out_refusal(
        &shared_scratch_args(own.clone(), own.join("ablation.json")),
        root.path(),
        reason,
    );
}

/// Review finding PR0-R8: a later legacy-only comparison into a fresh scratch whose `--out`
/// lies in a kernel2 comparison's scratch is refused, directly, through a symbolic link to
/// that scratch, and through a `..` (after an existing directory, or after one that does not
/// exist yet). Nothing is written anywhere, and the first document's `prepared_sha256` still
/// names `prepared.json`.
#[test]
pub(super) fn ablation_execute_refuses_an_out_in_a_kernel2_scratch() {
    let root = scratch();
    let first_dir = root.path().join("first");
    let first = run(&shared_scratch_config(), &first_dir).unwrap();
    let record = &first.document.body.datasets[0];
    let relative = Path::new("prepared")
        .join(&record.dataset_hash)
        .join("prepared.json");
    let second = root.path().join("second");
    let marked = "holds .ablation-kernel2";
    let mut outs = vec![
        (first_dir.join(&relative), marked),
        (first_dir.join("search").join("..").join(&relative), marked),
        (first_dir.join("ablation.json"), marked),
        (
            root.path().join("missing/../first").join(&relative),
            "a `..` after a directory that does not exist yet",
        ),
    ];
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink("first", root.path().join("link")).unwrap();
        outs.insert(0, (root.path().join("link").join(&relative), marked));
    }
    for (out, reason) in outs {
        out_refusal(&legacy_out_args(second.clone(), out), root.path(), reason);
    }
    assert_eq!(
        sha256_hex(&std::fs::read(first_dir.join(&relative)).unwrap()),
        record.prepared_sha256
    );
}

/// Review finding PR0-R8: a `--out` in, or anywhere under, a directory that holds
/// `.ablation-lock` (a comparison in progress) is refused, with nothing written.
#[test]
pub(super) fn ablation_execute_refuses_an_out_under_a_lock_file() {
    let root = scratch();
    let held = root.path().join("held");
    std::fs::create_dir(&held).unwrap();
    std::fs::write(held.join(LOCK_FILE), b"pid 1 started_unix 0\n").unwrap();
    let second = root.path().join("second");
    for out in [
        held.join("ablation.json"),
        held.join("sub/dir/ablation.json"),
    ] {
        out_refusal(
            &legacy_out_args(second.clone(), out),
            root.path(),
            "holds .ablation-lock",
        );
    }
}

/// A `--out` that is a symbolic link (to an ordinary file) or an existing directory is
/// refused, for a legacy-only comparison too, with nothing written.
#[test]
pub(super) fn ablation_execute_refuses_an_out_that_is_not_a_regular_file() {
    let root = scratch();
    let second = root.path().join("second");
    #[cfg(unix)]
    {
        std::fs::write(root.path().join("target.json"), b"kept\n").unwrap();
        std::os::unix::fs::symlink("target.json", root.path().join("link.json")).unwrap();
        out_refusal(
            &legacy_out_args(second.clone(), root.path().join("link.json")),
            root.path(),
            "is a symbolic link",
        );
    }
    std::fs::create_dir(root.path().join("dir.json")).unwrap();
    out_refusal(
        &legacy_out_args(second, root.path().join("dir.json")),
        root.path(),
        "not a regular file",
    );
}

/// A legacy-only comparison writes `--out` as before: in an ordinary directory, with the body
/// of the same comparison run through the library, and inside its own legacy scratch, which
/// then holds the document beside its evidence and no lock file or marker.
#[test]
pub(super) fn ablation_execute_writes_a_legacy_out_as_before() {
    let root = scratch();
    let out = root.path().join("out/ablation.json");
    execute(&legacy_out_args(root.path().join("scratch"), out.clone())).unwrap();
    let written: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    let alone = scratch();
    let report = run(&legacy_scratch_config(7, 20_260_908), alone.path()).unwrap();
    assert_eq!(
        written["body_sha256"].as_str(),
        Some(report.document.body_sha256.as_str())
    );
    let own = root.path().join("own");
    let inside = own.join("ablation.json");
    execute(&legacy_out_args(own.clone(), inside.clone())).unwrap();
    assert!(inside.is_file());
    assert_eq!(names(&own), set(&["ablation.json", "prepared", "search"]));
}

// fix-05: the --out document is written by rename, never by truncating an existing file.

/// The names directly under `dir` that look like [`replace_document`]'s temporary files.
pub(super) fn temporary_documents(dir: &Path) -> Vec<String> {
    names(dir)
        .into_iter()
        .filter(|name| name.contains(".tmp-"))
        .collect()
}

/// A `--out` that is a hard link to a kernel2 scratch's `prepared.json`, or to one of its
/// published searches, is admitted and replaced: the document lands at `--out` as a new
/// inode, the evidence keeps its bytes, modification time and its single link, the first
/// document's `prepared_sha256` still names `prepared.json`, and no temporary file is left.
#[cfg(unix)]
#[test]
pub(super) fn ablation_execute_replaces_an_out_hard_linked_to_kernel2_evidence() {
    use std::os::unix::fs::MetadataExt as _;
    let root = scratch();
    let first_dir = root.path().join("first");
    let first = run(&shared_scratch_config(), &first_dir).unwrap();
    let record = &first.document.body.datasets[0];
    let prepared = first_dir
        .join("prepared")
        .join(&record.dataset_hash)
        .join("prepared.json");
    let search =
        search_output(&first_dir, "A", &record.dataset_hash, 6).join("search/cut-0/search.json");
    for (name, evidence) in [
        ("prepared-link.json", &prepared),
        ("search-link.json", &search),
    ] {
        let out = root.path().join(name);
        std::fs::hard_link(evidence, &out).unwrap();
        let before = tree(&first_dir);
        execute(&legacy_out_args(
            root.path().join(format!("second-{name}")),
            out.clone(),
        ))
        .unwrap();
        let (out_meta, evidence_meta) = (
            std::fs::metadata(&out).unwrap(),
            std::fs::metadata(evidence).unwrap(),
        );
        assert_ne!(out_meta.ino(), evidence_meta.ino(), "{name}");
        assert_eq!(evidence_meta.nlink(), 1, "{name}");
        let changed = changed_entries(&before, &tree(&first_dir));
        assert!(
            changed.is_empty(),
            "{name}: the evidence changed {changed:?}"
        );
        let written: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
        assert_eq!(written["schema"].as_str(), Some(SCHEMA), "{name}");
    }
    assert_eq!(
        sha256_hex(&std::fs::read(&prepared).unwrap()),
        record.prepared_sha256
    );
    assert!(temporary_documents(root.path()).is_empty());
}

/// A `--out` that is a hard link to an ordinary file outside any scratch is replaced: the
/// other link keeps its old bytes, the document lands at `--out` as a new inode, and no
/// temporary file is left.
#[cfg(unix)]
#[test]
pub(super) fn ablation_execute_replaces_an_out_hard_linked_to_an_ordinary_file() {
    use std::os::unix::fs::MetadataExt as _;
    let root = scratch();
    let other = root.path().join("other.txt");
    std::fs::write(&other, b"kept\n").unwrap();
    let out = root.path().join("ablation.json");
    std::fs::hard_link(&other, &out).unwrap();
    execute(&legacy_out_args(root.path().join("scratch"), out.clone())).unwrap();
    assert_eq!(std::fs::read(&other).unwrap(), b"kept\n");
    assert_ne!(
        std::fs::metadata(&out).unwrap().ino(),
        std::fs::metadata(&other).unwrap().ino()
    );
    let written: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(written["schema"].as_str(), Some(SCHEMA));
    assert!(temporary_documents(root.path()).is_empty());
}

/// A legacy-only `--out` that is an existing regular file, longer than the document, is
/// overwritten as before: the file then holds exactly one document (nothing of the old
/// bytes is left after it), with the body of the same comparison run through the library,
/// and no temporary file is left.
#[test]
pub(super) fn ablation_execute_overwrites_an_existing_legacy_out() {
    let root = scratch();
    let out = root.path().join("ablation.json");
    std::fs::write(&out, vec![b'x'; 1 << 20]).unwrap();
    execute(&legacy_out_args(root.path().join("scratch"), out.clone())).unwrap();
    let bytes = std::fs::read(&out).unwrap();
    assert!(bytes.len() < 1 << 20 && bytes.ends_with(b"}\n"));
    let written: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let alone = scratch();
    let report = run(&legacy_scratch_config(7, 20_260_908), alone.path()).unwrap();
    assert_eq!(
        written["body_sha256"].as_str(),
        Some(report.document.body_sha256.as_str())
    );
    assert!(temporary_documents(root.path()).is_empty());
}

/// [`replace_document`] writes exactly [`write_document`]'s bytes, for a legacy and a kernel2
/// document, also over an existing longer file, and returns the same sha256.
#[test]
pub(super) fn ablation_document_replacement_writes_the_same_bytes() {
    let root = scratch();
    for (name, config) in [
        ("legacy", legacy_scratch_config(7, 20_260_908)),
        ("kernel2", shared_scratch_config()),
    ] {
        let report = run(&config, &root.path().join(name)).unwrap();
        let plain = root.path().join(format!("{name}-plain.json"));
        let replaced = root.path().join(format!("{name}-replaced.json"));
        std::fs::write(&replaced, vec![b'x'; 1 << 20]).unwrap();
        let plain_sha256 = write_document(&plain, &report.document).unwrap();
        let replaced_sha256 = replace_document(&replaced, &report.document).unwrap();
        let bytes = std::fs::read(&replaced).unwrap();
        assert_eq!(bytes, std::fs::read(&plain).unwrap(), "{name}");
        assert_eq!(
            (&replaced_sha256, &plain_sha256),
            (&sha256_hex(&bytes), &sha256_hex(&bytes))
        );
    }
    assert!(temporary_documents(root.path()).is_empty());
}

/// A write that fails after the temporary file was created and written (the rename onto a
/// directory fails) leaves no temporary file and changes nothing: the error names the rename.
#[test]
pub(super) fn ablation_document_replacement_removes_its_temporary_file_on_failure() {
    let root = scratch();
    let report = run(
        &legacy_scratch_config(6, 20_260_907),
        &root.path().join("s"),
    )
    .unwrap();
    let blocked = root.path().join("ablation.json");
    std::fs::create_dir(&blocked).unwrap();
    std::fs::write(blocked.join("kept"), b"kept\n").unwrap();
    let before = tree(root.path());
    let error = format!(
        "{:#}",
        replace_document(&blocked, &report.document).unwrap_err()
    );
    assert!(error.contains("rename "), "{error}");
    let changed = changed_entries(&before, &tree(root.path()));
    assert!(changed.is_empty(), "a failed write changed {changed:?}");
    assert!(temporary_documents(root.path()).is_empty());
}

// polish-01: review 05's notes on reserved --out names, the same winner, the sign test's
// underflow and the directory sync.

/// Review note PR0-O-LEGACY-CONTROL-NAMES: a `--out` whose file name is `.ablation-lock` or
/// `.ablation-kernel2` is refused for every comparison, before anything is created: a
/// legacy-only `--out` inside its own legacy scratch (absent, or an existing empty
/// directory), one in an ordinary directory (also spelled with a trailing `/`), and a
/// kernel2 comparison's, outside or inside its own scratch. A name that only begins with a
/// reserved name is not refused.
#[test]
pub(super) fn ablation_execute_refuses_a_reserved_out_name() {
    let root = scratch();
    let reason = "is reserved for a comparison's lock file or marker";
    for name in [LOCK_FILE, KERNEL2_MARKER] {
        let own = root.path().join(format!("own{name}"));
        let kernel2 = root.path().join(format!("kernel2{name}"));
        let ordinary = root.path().join(format!("ordinary{name}"));
        let slash = PathBuf::from(format!("{}/", ordinary.join(name).display()));
        for args in [
            legacy_out_args(own.clone(), own.join(name)),
            legacy_out_args(root.path().join("second"), ordinary.join(name)),
            legacy_out_args(root.path().join("second"), slash),
            shared_scratch_args(kernel2.clone(), root.path().join(name)),
            shared_scratch_args(kernel2.clone(), kernel2.join(name)),
        ] {
            let error = out_refusal(&args, root.path(), reason);
            assert!(error.contains(name), "{name} not in {error}");
        }
        std::fs::create_dir(&own).unwrap();
        let error = out_refusal(
            &legacy_out_args(own.clone(), own.join(name)),
            root.path(),
            reason,
        );
        assert!(error.contains(name), "{name} not in {error}");
        assert!(names(&own).is_empty());
    }
    let own = root.path().join("own");
    let near = own.join(format!("{LOCK_FILE}.json"));
    execute(&legacy_out_args(own.clone(), near.clone())).unwrap();
    assert!(near.is_file());
    assert_eq!(
        names(&own),
        set(&[".ablation-lock.json", "prepared", "search"])
    );
}
