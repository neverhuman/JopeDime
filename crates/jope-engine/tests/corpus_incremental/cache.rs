use super::*;

#[test]
fn reuse_is_byte_identical_with_exact_counts_first_wins_and_no_source_writes() {
    let f = Fixture::new();
    let original = f
        .observations()
        .into_iter()
        .flat_map(|p| {
            walkdir::WalkDir::new(p)
                .into_iter()
                .map(|e| e.unwrap().into_path())
        })
        .filter(|p| p.is_file())
        .map(|p| {
            let sha = file_hash(&p).unwrap();
            (p, sha)
        })
        .collect::<Vec<_>>();
    f.full("full", true);
    let first = f.cached("first", true).unwrap();
    assert_eq!(
        first,
        ScanCacheStats {
            ledgers_hashed: 3,
            ledgers_parsed: 3,
            ledgers_reused: 0,
            entries_committed: 3
        }
    );
    let second = f.cached("reuse", true).unwrap();
    assert_eq!(
        second,
        ScanCacheStats {
            ledgers_hashed: 3,
            ledgers_parsed: 0,
            ledgers_reused: 3,
            entries_committed: 0
        }
    );
    f.equal("full", "first");
    f.equal("full", "reuse");
    let provenance: Value = read_json(&f.root().join("reuse/provenance.json")).unwrap();
    assert_eq!(provenance["counts"]["observations_seen"], 39);
    assert_eq!(provenance["counts"]["no_matching_context"], 3);
    assert_eq!(provenance["counts"]["failed_program"], 3);
    assert_eq!(provenance["counts"]["duplicate_program_cut"], 3);
    assert_eq!(provenance["selected_programs"], 24);
    let labels = fs::read_to_string(f.root().join("reuse/labels.jsonl")).unwrap();
    assert!(!labels.contains("duplicate must not win"));
    for (path, sha) in original {
        assert_eq!(file_hash(&path).unwrap(), sha);
    }
}

#[test]
fn changed_same_size_ledger_and_added_removed_sources_are_reconciled() {
    let f = Fixture::new();
    f.cached("initial", true).unwrap();
    let path = f.ledger("family-a");
    let before = fs::read(&path).unwrap();
    let mut rows: Value = serde_json::from_slice(&before).unwrap();
    rows[0]["metrics"]["rmse"] = json!(8.0);
    write_json(&path, &rows).unwrap();
    assert_eq!(before.len(), fs::metadata(&path).unwrap().len() as usize);
    let stats = f.cached("changed", true).unwrap();
    assert_eq!(stats.ledgers_parsed, 1);
    assert_eq!(stats.ledgers_reused, 2);
    f.full("changed-full", true);
    f.equal("changed-full", "changed");
    assert_ne!(
        fs::read(f.root().join("initial/records.bin")).unwrap(),
        fs::read(f.root().join("changed/records.bin")).unwrap()
    );
    fs::remove_dir_all(f.ledger("family-c").parent().unwrap()).unwrap();
    let stats = f.cached("removed", true).unwrap();
    assert_eq!(stats.ledgers_reused, 2);
    assert_eq!(stats.ledgers_parsed, 0);
    f.full("removed-full", true);
    f.equal("removed-full", "removed");
    let added = f.ledger("a-first");
    let mut rows: Value = read_json(&f.ledger("family-a")).unwrap();
    rows[0]["metrics"]["rmse"] = json!(0.5);
    rows[0]["evaluation_bias"] = json!("earlier ledger wins");
    write_json(&added, &rows).unwrap();
    write_json(&added.with_file_name("search.json"), &json!({})).unwrap();
    let stats = f.cached("added", true).unwrap();
    assert_eq!(stats.ledgers_reused, 2);
    assert_eq!(stats.ledgers_parsed, 1);
    f.full("added-full", true);
    f.equal("added-full", "added");
    assert!(fs::read_to_string(f.root().join("added/labels.jsonl"))
        .unwrap()
        .contains("earlier ledger wins"));
}

#[test]
fn context_vocab_qualification_search_and_source_path_bind_cache_identity() {
    let f = Fixture::new();
    f.cached("initial", true).unwrap();
    // Semantically identical files with different exact identities cannot hit.
    for (i, path) in [f.contexts(), f.vocab()].into_iter().enumerate() {
        use std::io::Write;
        fs::OpenOptions::new()
            .append(true)
            .open(path)
            .unwrap()
            .write_all(b"\n")
            .unwrap();
        let stats = f.cached(&format!("binding-{i}"), true).unwrap();
        assert_eq!(stats.ledgers_parsed, 3);
        assert_eq!(stats.ledgers_reused, 0);
        f.full(&format!("binding-full-{i}"), true);
        f.equal(&format!("binding-full-{i}"), &format!("binding-{i}"));
    }
    assert!(f
        .cached("clean", false)
        .unwrap_err()
        .to_string()
        .contains("production corpus requires"));
    assert!(!f.root().join("clean").exists());
    write_json(
        &f.ledger("family-a").with_file_name("search.json"),
        &json!({"changed":true}),
    )
    .unwrap();
    let stats = f.cached("search-changed", true).unwrap();
    assert_eq!(stats.ledgers_parsed, 1);
    assert_eq!(stats.ledgers_reused, 2);
    fs::rename(
        f.ledger("family-a").parent().unwrap(),
        f.ledger("renamed-family").parent().unwrap(),
    )
    .unwrap();
    let stats = f.cached("path-changed", true).unwrap();
    assert_eq!(stats.ledgers_parsed, 1);
    assert_eq!(stats.ledgers_reused, 2);
    f.full("path-changed-full", true);
    f.equal("path-changed-full", "path-changed");
}

#[test]
fn altered_context_or_prepared_artifacts_are_rejected_before_cache_reuse() {
    for artifact in ["context.bin", "prepared.json"] {
        let f = Fixture::new();
        f.cached("initial", true).unwrap();
        fs::write(f.root().join("family-a").join(artifact), b"changed").unwrap();
        assert!(f.cached("must-fail", true).is_err());
        assert!(!f.root().join("must-fail").exists());
    }
}

#[test]
fn cache_corruption_and_misbound_manifest_are_errors_not_silent_labels() {
    for mode in ["payload", "manifest", "missing"] {
        let f = Fixture::new();
        f.cached("initial", true).unwrap();
        let entry = &f.entries()[0];
        match mode {
            "payload" => {
                let p = entry.join("contribution.json");
                let mut rows: Value = read_json(&p).unwrap();
                rows["selected"]["0"] = json!({});
                write_json(&p, &rows).unwrap();
            }
            "manifest" => {
                let p = entry.join("manifest.json");
                let mut m: Value = read_json(&p).unwrap();
                m["binding"]["source_sha256"] = json!("wrong");
                write_json(&p, &m).unwrap();
            }
            _ => fs::remove_file(entry.join("contribution.json")).unwrap(),
        }
        assert!(f.cached("must-fail", true).is_err(), "accepted {mode}");
        assert!(!f.root().join("must-fail").exists());
    }
}

#[test]
fn duplicate_rows_still_validate_and_invalid_changed_scores_never_hit_cache() {
    let f = Fixture::new();
    f.cached("initial", true).unwrap();
    let path = f.ledger("family-a");
    let mut rows: Value = read_json(&path).unwrap();
    rows[8]["program_sha256"] = json!("invalid duplicate");
    write_json(&path, &rows).unwrap();
    assert!(f.cached("bad-duplicate", true).is_err());
    rows[8]["program_sha256"] = rows[0]["program_sha256"].clone();
    rows[0]["metrics"]["rmse"] = json!(-1.0);
    write_json(&path, &rows).unwrap();
    assert!(f.cached("bad-score", true).is_err());
}

#[cfg(unix)]
#[test]
fn cache_and_source_symlinks_and_overlapping_namespaces_are_rejected() {
    use std::os::unix::fs::symlink;
    let f = Fixture::new();
    fs::create_dir_all(f.root().join("elsewhere")).unwrap();
    fs::create_dir_all(f.root().join("runs/experiment")).unwrap();
    symlink(f.root().join("elsewhere"), f.cache()).unwrap();
    assert!(f.cached("linked-cache", true).is_err());
    fs::remove_file(f.cache()).unwrap();
    let source = f.ledger("family-a");
    fs::rename(&source, source.with_extension("original")).unwrap();
    symlink(source.with_extension("original"), &source).unwrap();
    assert!(f.cached("linked-source", true).is_err());
    fs::remove_file(&source).unwrap();
    fs::rename(source.with_extension("original"), source).unwrap();
    f.cached("initial", true).unwrap();
    let entry = f.entries()[0].clone();
    let payload = entry.join("contribution.json");
    fs::rename(&payload, payload.with_extension("original")).unwrap();
    symlink(payload.with_extension("original"), &payload).unwrap();
    assert!(f.cached("linked-payload", true).is_err());
    for cache in [
        f.root().join("runs/global"),
        f.root().join("runs/experiment/output/cache"),
    ] {
        assert!(corpus::build_observed_cached(
            f.root(),
            jope_parallel::corpus::CorpusSources {
                contexts_path: &f.contexts(),
                vocab_path: &f.vocab(),
                observations: &f.observations()
            },
            &f.root().join("runs/experiment/output"),
            true,
            &cache,
            |_, _, _| Ok(()),
        )
        .is_err());
    }
}
