use super::*;

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
