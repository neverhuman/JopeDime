use super::*;

/// Route R8 at the teaching store's ledger ingestion (teaching_store.rs:551,
/// 572 before the fix): the search receipt an observation is validated
/// against is the receipt whose digest the ledger commits.
#[test]
fn a_search_receipt_is_validated_from_the_bytes_its_ledger_commits() {
    let opened = |valid: &[u8]| {
        let mut receipt: Value = serde_json::from_slice(valid).unwrap();
        receipt["final_tests_opened"] = json!(true);
        serde_json::to_vec(&receipt).unwrap()
    };
    // All three controls use the same immutable completions, but independent
    // output databases. Restore the input receipt before each replacement.
    let s = strict();
    // Positive control: nothing replaced.
    let search = s.f.ledger("family-a").with_file_name("search.json");
    let valid = fs::read(&search).unwrap();
    let (digest, labels) = family_a_state(&publish(&s, "control"));
    assert_eq!(digest, hash(&valid));
    assert!(labels > 0);

    // The committed receipt opened final tests; a valid receipt replaces it
    // after its digest was taken. No label may be admitted under it.
    let committed = opened(&valid);
    fs::write(&search, &committed).unwrap();
    let db = {
        let _hook = replace_on_verification(&search, valid.clone());
        publish(&s, "opened")
    };
    let (digest, labels) = family_a_state(&db);
    assert_eq!(digest, hash(&committed));
    assert_eq!(
        labels, 0,
        "labels were admitted under a search receipt other than the one the ledger commits"
    );

    // The other direction: a valid committed receipt replaced by one that
    // opened final tests. The committed receipt decides.
    fs::write(&search, &valid).unwrap();
    let db = {
        let _hook = replace_on_verification(&search, opened(&valid));
        publish(&s, "valid")
    };
    let (digest, labels) = family_a_state(&db);
    assert_eq!(digest, hash(&valid));
    assert!(
        labels > 0,
        "the committed valid receipt was not the one the admission validated"
    );
}

/// Route R8 at `teaching_store::verify_shard_metadata` (:100-106 before the
/// fix): the shard binding is checked on exactly the verified `shard.json`.
#[test]
fn an_honest_shard_manifest_replaced_after_its_check_is_still_accepted() {
    let (s, corpus) = published("shard-honest");
    let (shard, path, _, bad) = first_shard(&s, &corpus);
    // Positive control.
    super::verify_shard_metadata(s.f.root(), &shard).unwrap();
    let _hook = replace_on_verification(&path, bad);
    let checked = super::verify_shard_metadata(s.f.root(), &shard)
        .map(|_| ())
        .map_err(|e| format!("{e:#}"));
    assert_eq!(
        checked,
        Ok(()),
        "the binding was checked on a shard manifest other than the verified one"
    );
}

/// The same receiver, other direction: a bound `shard.json` that disagrees
/// with its binding is refused, even if a matching one replaces it after the
/// digest check.
#[test]
fn a_shard_manifest_disagreeing_with_its_binding_is_refused_even_if_replaced() {
    let (s, corpus) = published("shard-forged");
    let (shard, path, good, bad) = first_shard(&s, &corpus);
    fs::write(&path, &bad).unwrap();
    let mut forged = shard.clone();
    forged.sha256 = hash(&bad);
    // Negative control: refused without a replacement.
    let refused = failure(super::verify_shard_metadata(s.f.root(), &forged));
    assert!(refused.contains("shard binding changed"), "{refused}");
    let _hook = replace_on_verification(&path, good);
    let refused = failure(super::verify_shard_metadata(s.f.root(), &forged));
    assert!(
        refused.contains("shard binding changed"),
        "a shard manifest that disagrees with its binding was accepted: {refused}"
    );
}
