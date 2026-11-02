use super::*;

#[test]
fn generation_population_is_frozen_training_only_and_distinct_within_population() {
    let (tmp, _, mut rows) = fixture();
    let root = tmp.path();
    let mut ledger = Ledger::open(root).unwrap();
    seed_fixture_counts(&ledger, &rows);
    let mut sealed = rows[0].clone();
    sealed.partition = "meta_test".into();
    rows.push(sealed);
    let corpus = root.join("corpus");
    fs::create_dir(&corpus).unwrap();
    fs::write(corpus.join("index.json"), b"{}").unwrap();
    let owner = root.join("runs/generation-a");
    let one = ledger
        .freeze_population(root, &corpus, &owner, &rows, 2, 100)
        .unwrap();
    assert_eq!(one["selected_contexts"], 2);
    let again = ledger
        .freeze_population(root, &corpus, &owner, &rows, 2, 999)
        .unwrap();
    assert_eq!(one, again);
    let two = ledger
        .freeze_population(
            root,
            &corpus,
            &root.join("runs/generation-b"),
            &rows,
            2,
            101,
        )
        .unwrap();
    for population in [&one, &two] {
        let rows = population["contexts"].as_array().unwrap();
        let unique = rows
            .iter()
            .map(|row| row["context_index"].as_u64().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique.len(), rows.len());
    }
    let path = root.join("selection.json");
    write_json(&path, &one).unwrap();
    assert_eq!(selected_indices(&path, &corpus, &rows).unwrap().len(), 2);
    let mut legacy = one.clone();
    legacy["request"]
        .as_object_mut()
        .unwrap()
        .remove("implementation");
    write_json(&path, &legacy).unwrap();
    assert!(
        selected_indices(&path, &corpus, &rows).is_err(),
        "old selection implementation was silently accepted"
    );
    write_json(&path, &one).unwrap();
    let i = one["contexts"][0]["context_index"].as_u64().unwrap() as usize;
    rows[i].prepared_sha256 = "c".repeat(64);
    assert!(selected_indices(&path, &corpus, &rows).is_err());
    assert!(ledger
        .freeze_population(root, &corpus, &owner, &rows, 3, 200)
        .is_err());
}

#[test]
fn earlier_or_unknown_ledger_versions_are_rejected_without_mutating_original_bytes() {
    for version in [0, 1, 99] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::create_dir(root.join("runs")).unwrap();
        let path = root.join("runs/gp-coverage.sqlite3");
        let db = Connection::open(&path).unwrap();
        db.execute_batch(
            "CREATE TABLE coverage_plans(context TEXT PRIMARY KEY,owner TEXT,timestamp INTEGER);
            INSERT INTO coverage_plans VALUES('context','original owner',100);",
        )
        .unwrap();
        db.pragma_update(None, "user_version", version).unwrap();
        drop(db);
        let before = file_hash(&path).unwrap();
        assert!(Ledger::open(root)
            .err()
            .unwrap()
            .to_string()
            .contains("explicit reconciliation"));
        assert_eq!(file_hash(&path).unwrap(), before);
    }
    let temp = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(temp.path()).unwrap();
    assert_eq!(
        ledger
            .db
            .query_row("PRAGMA user_version", [], |r| r.get::<_, u32>(0))
            .unwrap(),
        LEDGER_VERSION
    );
    assert_eq!(ledger.status().unwrap()["implementation"], IMPLEMENTATION);
}

#[test]
fn queued_plans_balance_families_and_datasets_without_inflating_dispatch_attempts() {
    for families in [["a", "b"], ["same", "same"]] {
        let (tmp, _, mut rows) = fixture();
        let root = tmp.path();
        rows.truncate(2);
        for (i, row) in rows.iter_mut().enumerate() {
            row.family = families[i].into();
            row.dataset = format!("dataset-{i}");
        }
        let mut ledger = Ledger::open(root).unwrap();
        seed_fixture_counts(&ledger, &rows);
        let corpus = root.join("corpus");
        fs::create_dir(&corpus).unwrap();
        fs::write(corpus.join("index.json"), b"{}").unwrap();
        ledger
            .freeze_population(
                root,
                &corpus,
                &root.join("runs/earlier"),
                &rows[..1],
                1,
                100,
            )
            .unwrap();
        let next = ledger
            .freeze_population(root, &corpus, &root.join("runs/later"), &rows, 1, 101)
            .unwrap();
        assert_eq!(
            next["contexts"][0]["identity"]["dataset"], "dataset-1",
            "prior queued plan must contribute to family/dataset selection exposure"
        );
        assert_eq!(
            ledger.status().unwrap()["attempted"],
            0,
            "generation planning must not be reported as attempted dispatch"
        );
    }
}
