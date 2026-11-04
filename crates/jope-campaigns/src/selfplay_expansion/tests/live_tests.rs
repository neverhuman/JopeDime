use super::*;

#[test]
fn merge_live_without_tranches_is_a_readonly_noop() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    let live_root = temp.path().join("live");
    write_live_catalog(
        &live_root,
        &[("regression", "1111b9dad5725a1e", "fam-a", "train")],
    );
    let before = fs::read(live_root.join("current.json")).unwrap();
    let result = merge_live(&paths, &live_root, &[], true).unwrap();
    assert_eq!(result["nothing_to_merge"], true);
    assert_eq!(before, fs::read(live_root.join("current.json")).unwrap());
    assert!(!live_root.join("pointer-history.jsonl").exists());
    assert!(!live_root.join(".merge.lock").exists());
    let result = merge_live(&paths, &live_root, &[], false).unwrap();
    assert_eq!(result["nothing_to_merge"], true);
    assert_eq!(before, fs::read(live_root.join("current.json")).unwrap());
    assert!(!live_root.join("pointer-history.jsonl").exists());
}

#[test]
fn merge_live_appends_tranche_flips_pointer_and_is_idempotent() {
    let temp = TempDir::new().unwrap();
    let id = "feedb9dad5725a1e";
    let (paths, _entry, receipt) = activated_fixture(&temp, id, "tranche-merge");
    let live_root = temp.path().join("live");
    let old_pointer = write_live_catalog(
        &live_root,
        &[("regression", "1111b9dad5725a1e", "fam-a", "train")],
    );
    let old_blob = fs::read(Path::new(old_pointer["data"].as_str().unwrap())).unwrap();
    let summary = merge_live(&paths, &live_root, &[], false).unwrap();
    assert_eq!(summary["nothing_to_merge"], false);
    assert_eq!(summary["row_count"], 2);
    let pointer = read_json(&live_root.join("current.json")).unwrap();
    assert_ne!(pointer["catalog_sha256"], old_pointer["catalog_sha256"]);
    let merged = read_json(Path::new(pointer["catalog"].as_str().unwrap())).unwrap();
    let entries = merged["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 2);
    let original = read_json(Path::new(old_pointer["catalog"].as_str().unwrap())).unwrap();
    assert_eq!(entries[0], original["entries"].as_array().unwrap()[0]);
    assert_eq!(entries[1]["current_catalog"], false);
    assert_eq!(entries[1]["dataset_hash"], id);
    assert_eq!(entries[1]["catalog_index"], 1);
    assert_eq!(entries[1]["source_catalog_index"], Value::Null);
    assert!(entries[1]["train_csv"]
        .as_str()
        .unwrap()
        .contains("regression-expanded"));
    let new_blob = fs::read(Path::new(pointer["data"].as_str().unwrap())).unwrap();
    assert_eq!(new_blob.len(), 2 * CONTEXT_STRIDE);
    assert_eq!(&new_blob[..CONTEXT_STRIDE], &old_blob[..]);
    assert_eq!(
        sha256_bytes(&new_blob[CONTEXT_STRIDE..]),
        entries[1]["context_sha256"].as_str().unwrap()
    );
    let tranche_catalog = read_json(Path::new(receipt["catalog"].as_str().unwrap())).unwrap();
    assert_eq!(
        tranche_catalog["entries"][0]["context_sha256"],
        entries[1]["context_sha256"]
    );
    let history = fs::read_to_string(live_root.join("pointer-history.jsonl")).unwrap();
    assert_eq!(history.lines().count(), 1);
    assert!(history.contains(old_pointer["catalog_sha256"].as_str().unwrap()));
    // Re-running after success is a clean no-op (tranche already merged).
    let again = merge_live(&paths, &live_root, &[], false).unwrap();
    assert_eq!(again["nothing_to_merge"], true);
    assert_eq!(again["already_merged_tranches"][0], "tranche-merge");
    assert_eq!(
        fs::read_to_string(live_root.join("pointer-history.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    // Excluding the tranche also reports nothing to merge.
    let excluded = merge_live(&paths, &live_root, &["tranche-merge".to_owned()], true).unwrap();
    assert_eq!(excluded["nothing_to_merge"], true);
}

#[test]
fn merge_live_inherits_v1_family_split() {
    let temp = TempDir::new().unwrap();
    let id = "feedb9dad5725a1e";
    let (paths, _entry, receipt) = activated_fixture(&temp, id, "tranche-split");
    let tranche_catalog = read_json(Path::new(receipt["catalog"].as_str().unwrap())).unwrap();
    let family = tranche_catalog["entries"][0]["source_family"]
        .as_str()
        .unwrap()
        .to_owned();
    let tranche_split = tranche_catalog["entries"][0]["split"].as_str().unwrap();
    let opposite = if tranche_split == "train" {
        "validation"
    } else {
        "train"
    };
    let live_root = temp.path().join("live");
    write_live_catalog(
        &live_root,
        &[("regression", "1111b9dad5725a1e", &family, opposite)],
    );
    let summary = merge_live(&paths, &live_root, &[], false).unwrap();
    assert_eq!(summary["inherited_splits"], 1);
    let pointer = read_json(&live_root.join("current.json")).unwrap();
    let merged = read_json(Path::new(pointer["catalog"].as_str().unwrap())).unwrap();
    assert_eq!(merged["entries"][1]["split"], opposite);
}

#[test]
fn merge_live_rejects_conflicting_identity_and_v1_straddle() {
    let temp = TempDir::new().unwrap();
    let id = "feedb9dad5725a1e";
    let (paths, _entry, _receipt) = activated_fixture(&temp, id, "tranche-dup");
    let live_root = temp.path().join("live");
    write_live_catalog(&live_root, &[("regression", id, "fam-a", "train")]);
    let error = merge_live(&paths, &live_root, &[], true).unwrap_err();
    assert!(error.to_string().contains("different context"));
    let straddle_root = temp.path().join("straddle");
    write_live_catalog(
        &straddle_root,
        &[
            ("regression", "1111b9dad5725a1e", "fam-a", "train"),
            ("regression", "2222b9dad5725a1e", "fam-a", "validation"),
        ],
    );
    let error = merge_live(&paths, &straddle_root, &[], true).unwrap_err();
    assert!(error.to_string().contains("straddles"));
    let error = merge_live(&paths, &live_root, &["missing-id".to_owned()], true).unwrap_err();
    assert!(error.to_string().contains("not in the activation ledger"));
}

#[test]
fn install_sidecars_is_additive_and_fail_closed() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    let id = "0aa0b9dad5725a1e";
    let registry = paths.source_root.join("binary").join(id);
    fs::create_dir_all(&registry).unwrap();
    let from = temp.path().join("staged");
    let staged = from.join(id);
    fs::create_dir_all(&staged).unwrap();
    let target_row = (0..512).map(|_| "0.5").collect::<Vec<_>>().join(",");
    fs::write(staged.join("train_target.vs"), format!("{target_row}\n")).unwrap();
    fs::write(staged.join("train3.vs"), "0.25\n").unwrap();
    let receipts = temp.path().join("receipts/sidecars.jsonl");
    let summary = install_sidecars(&paths, &from, "binary", &receipts).unwrap();
    assert_eq!(summary["installed"], 2);
    assert!(registry.join("train_target.vs").is_file());
    let again = install_sidecars(&paths, &from, "binary", &receipts).unwrap();
    assert_eq!(again["installed"], 0);
    assert_eq!(again["skipped_identical"], 2);
    fs::write(staged.join("train3.vs"), "0.75\n").unwrap();
    let error = install_sidecars(&paths, &from, "binary", &receipts).unwrap_err();
    assert!(error.to_string().contains("refusing to overwrite"));
    let lines = fs::read_to_string(&receipts).unwrap();
    assert!(lines.lines().count() >= 5);
    assert!(lines.contains("\"conflict\""));
    fs::write(
        staged.join("train_target.vs"),
        format!("{target_row}\n{target_row}\n"),
    )
    .unwrap();
    fs::remove_file(staged.join("train3.vs")).unwrap();
    let error = install_sidecars(&paths, &from, "binary", &receipts).unwrap_err();
    assert!(error.to_string().contains("1x512"));
    let missing_from = temp.path().join("staged-missing");
    fs::create_dir_all(missing_from.join("9bb0b9dad5725a1e")).unwrap();
    fs::write(
        missing_from.join("9bb0b9dad5725a1e").join("train3.vs"),
        "0.1\n",
    )
    .unwrap();
    let summary = install_sidecars(&paths, &missing_from, "binary", &receipts).unwrap();
    assert_eq!(summary["missing_registry_dir"], 1);
}
