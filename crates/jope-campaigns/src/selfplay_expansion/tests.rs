use super::cache::verify_native_encoder;
use super::context::parse_npy;
use super::prelude::*;
use super::*;
use tempfile::TempDir;

use super::*;
use tempfile::TempDir;

fn paths(temp: &TempDir) -> ExpansionPaths {
    ExpansionPaths {
        source_root: temp.path().join("remote"),
        mirror_root: temp.path().join("mirror"),
        cache_root: temp.path().join("cache"),
        catalog_root: temp.path().join("catalog"),
        receipt_root: temp.path().join("receipts"),
        sealed_path: temp.path().join("private/sealed.json"),
        standardization: temp.path().join("standardization.json"),
    }
}

fn dataset(root: &Path, task: &str, id: &str, seed: &str) {
    let path = root.join(task).join(id);
    fs::create_dir_all(&path).unwrap();
    fs::write(path.join("train.csv"), format!("x,y\n1,{seed}\n")).unwrap();
    fs::write(path.join("test.csv"), format!("x,y\n2,{seed}\n")).unwrap();
    atomic_json(
        &path.join("meta.json"),
        &json!({"origin":"synthetic","generator_recipe":{"kind":task,"seed":seed}}),
    )
    .unwrap();
    fs::write(
        path.join("hyperion3train128dim.vs"),
        "0,".repeat(127) + "0\n",
    )
    .unwrap();
    fs::write(path.join("train3.vs"), "0\n").unwrap();
    fs::write(
        path.join("train_target.vs"),
        format!(
            "{}\n",
            (0..TARGET_DIM).map(|_| "0").collect::<Vec<_>>().join(",")
        ),
    )
    .unwrap();
}

#[test]
fn mirror_conflict_is_fail_closed_and_receipted() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    let id = "0014b9dad5725a1e";
    dataset(&paths.source_root, "regression", id, "1");
    dataset(&paths.mirror_root, "regression", id, "2");
    let error = discover(&paths, Some("regression"), None).unwrap_err();
    assert!(error.to_string().contains("mirror content conflict"));
    assert_eq!(fs::read_dir(&paths.receipt_root).unwrap().count(), 1);
}

#[test]
fn exhaustive_audit_receipts_every_conflict_and_writes_stable_summary() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    for (id, source_seed, mirror_seed) in [
        ("0014b9dad5725a1e", "1", "9"),
        ("006bca140f93e411", "2", "8"),
        ("00709d315efacf0c", "3", "3"),
    ] {
        dataset(&paths.source_root, "regression", id, source_seed);
        dataset(&paths.mirror_root, "regression", id, mirror_seed);
    }
    let first = audit_discover(&paths, Some("regression"), None).unwrap();
    assert_eq!(first.inspected, 3);
    assert_eq!(first.entries.len(), 1);
    assert_eq!(first.conflicts.len(), 2);
    assert_eq!(first.conflicts[0].dataset_id, "0014b9dad5725a1e");
    assert_eq!(first.conflicts[1].dataset_id, "006bca140f93e411");
    assert!(first
        .conflicts
        .iter()
        .all(|conflict| conflict.receipt.is_file()));
    let first_summary = write_conflict_summary(&paths, &first).unwrap();
    let first_bytes = fs::read(&first_summary).unwrap();

    let second = audit_discover(&paths, Some("regression"), None).unwrap();
    let second_summary = write_conflict_summary(&paths, &second).unwrap();
    assert_eq!(first_summary, second_summary);
    assert_eq!(first_bytes, fs::read(second_summary).unwrap());
    let pointer = read_json(&paths.receipt_root.join("latest-summary.json")).unwrap();
    assert_eq!(pointer["conflict_count"], 2);
    assert_eq!(pointer["inspected"], 3);
}

#[test]
fn cache_is_atomic_and_validates_all_hashes() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    let id = "0014b9dad5725a1e";
    dataset(&paths.source_root, "regression", id, "1");
    let entry = discover(&paths, Some("regression"), None)
        .unwrap()
        .remove(0);
    let manifest = materialize_cache(&paths, &entry, None).unwrap();
    assert!(manifest.train_only_encoding);
    let cache = cache_path(&paths, "regression", id);
    validate_cache(&cache, Some(&entry)).unwrap();
    assert_eq!(
        materialize_cache(&paths, &entry, None).unwrap(),
        manifest,
        "a complete cache must be resumed without rewriting"
    );
    fs::write(cache.join("train.csv"), "drift").unwrap();
    assert!(validate_cache(&cache, Some(&entry)).is_err());
}

#[test]
fn legacy_sidecar_cache_is_upgraded_with_verified_source_files() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    let id = "0014b9dad5725a1e";
    dataset(&paths.source_root, "timeseries_reduced_v2", id, "1");
    let entry = discover(&paths, Some("timeseries"), None)
        .unwrap()
        .remove(0);
    materialize_cache(&paths, &entry, None).unwrap();
    let cache = cache_path(&paths, "timeseries", id);
    let current = read_json(&cache.join("cache-manifest.v1.json")).unwrap();
    let old_files = current["files"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(name, _)| {
            !matches!(
                name.as_str(),
                "train.csv" | "test.csv" | "signature_manifest.json"
            )
        })
        .map(|(name, digest)| (name.clone(), digest.clone()))
        .collect::<serde_json::Map<_, _>>();
    fs::remove_file(cache.join("cache-manifest.v1.json")).unwrap();
    fs::remove_file(cache.join("signature_manifest.json")).unwrap();
    fs::remove_file(cache.join("train.csv")).unwrap();
    fs::remove_file(cache.join("test.csv")).unwrap();
    atomic_json(
        &cache.join("cache-manifest.v1.json"),
        &json!({
            "schema": "JopeExpandedEncodingCache.v1",
            "task": entry.task,
            "source": entry.source,
            "dataset_hash": entry.dataset_hash,
            "encoder_protocol": ENCODER_PROTOCOL,
            "train_only_encoding": true,
            "immutable": true,
            "staged_by": "legacy-fixture",
            "files": old_files,
        }),
    )
    .unwrap();

    let manifest = materialize_cache(&paths, &entry, None).unwrap();
    assert_eq!(manifest.dataset_id, id);
    assert!(cache.join("train.csv").is_file());
    assert!(cache.join("test.csv").is_file());
    assert!(cache.join("signature_manifest.json").is_file());
    validate_cache(&cache, Some(&entry)).unwrap();
}

#[test]
fn missing_encodings_fail_without_leaving_a_partial_cache() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    let id = "0014b9dad5725a1e";
    dataset(&paths.source_root, "binary", id, "1");
    fs::remove_file(
        paths
            .source_root
            .join("binary")
            .join(id)
            .join("train_target.vs"),
    )
    .unwrap();
    let entry = discover(&paths, Some("binary"), None).unwrap().remove(0);
    let error = materialize_cache(&paths, &entry, None).unwrap_err();
    assert!(error
        .to_string()
        .contains("no native Rust Hyperion encoder"));
    assert!(!cache_path(&paths, "binary", id).exists());
}

#[test]
fn native_encoder_rejects_scripts() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("encoder.py");
    fs::write(&path, "#!/usr/bin/env python3\n").unwrap();
    assert!(verify_native_encoder(&path).is_err());
}

#[test]
fn npy_parser_reads_f32_and_shape() {
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    let mut header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }".to_vec();
    while !(bytes.len() + 2 + header.len() + 1).is_multiple_of(64) {
        header.push(b' ');
    }
    header.push(b'\n');
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&1.5_f32.to_le_bytes());
    bytes.extend_from_slice(&(-2.0_f32).to_le_bytes());
    let parsed = parse_npy(bytes.as_slice()).unwrap();
    assert_eq!(parsed.shape, vec![2]);
    assert_eq!(parsed.values, vec![1.5, -2.0]);
}

#[test]
fn sealed_status_never_exposes_dataset_identities() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    let mut entries = Vec::new();
    for task in TASKS {
        for index in 0..sealed_target(task) {
            entries.push(InventoryEntry {
                task: task.to_owned(),
                dataset_id: format!("{index:016x}"),
                dataset_hash: format!("{index:016x}"),
                source: PathBuf::from("private"),
                source_family: format!("{task}-family-{index}"),
                train_sha256: "a".repeat(64),
                test_sha256: "b".repeat(64),
                meta_sha256: "c".repeat(64),
                mirror: None,
            });
        }
    }
    let redacted = freeze_sealed(&paths, &entries, 7).unwrap();
    let encoded = serde_json::to_string(&redacted).unwrap();
    assert!(!encoded.contains("dataset_id"));
    assert!(!encoded.contains("family-"));
    assert_eq!(
        fs::metadata(&paths.sealed_path)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    let adaptive = adaptive_inventory(&paths, &entries).unwrap();
    assert!(adaptive.is_empty());
}

#[test]
fn family_split_is_deterministic_and_disjoint() {
    let family = "synthetic:abc";
    assert_eq!(
        jope_datasets::family_contracts::source_family_campaign_split(family),
        jope_datasets::family_contracts::source_family_campaign_split(family)
    );
    let split = jope_datasets::family_contracts::source_family_campaign_split(family);
    assert!(matches!(split, "train" | "validation"));
}

fn npy_bytes(shape: &[usize], values: &[f32]) -> Vec<u8> {
    let mut output = b"\x93NUMPY\x01\x00".to_vec();
    let shape = if shape.len() == 1 {
        format!("{},", shape[0])
    } else {
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape}), }}").into_bytes();
    while !(output.len() + 2 + header.len() + 1).is_multiple_of(64) {
        header.push(b' ');
    }
    header.push(b'\n');
    output.extend_from_slice(&(header.len() as u16).to_le_bytes());
    output.extend_from_slice(&header);
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output
}

fn write_projection(path: &Path) {
    use zip::write::SimpleFileOptions;
    let file = File::create(path).unwrap();
    let mut archive = zip::ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, shape, values) in [
        (
            "components.npy",
            vec![1024, 2048],
            vec![0.0_f32; 1024 * 2048],
        ),
        ("pca_mean.npy", vec![2048], vec![0.0_f32; 2048]),
        ("z_mean.npy", vec![1024], vec![0.0_f32; 1024]),
        ("z_scale.npy", vec![1024], vec![1.0_f32; 1024]),
    ] {
        archive.start_file(name, options).unwrap();
        archive.write_all(&npy_bytes(&shape, &values)).unwrap();
    }
    archive.finish().unwrap();
}

#[test]
fn tranche_activation_is_hashed_validated_and_idempotent() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    let id = "feedb9dad5725a1e";
    dataset(&paths.source_root, "regression", id, "1");
    let entry = discover(&paths, Some("regression"), None)
        .unwrap()
        .remove(0);
    materialize_cache(&paths, &entry, None).unwrap();

    let foundation = temp.path().join("foundation");
    fs::create_dir_all(&foundation).unwrap();
    fs::write(
        foundation.join(format!("{id}.npy")),
        npy_bytes(&[2048], &vec![0.0; 2048]),
    )
    .unwrap();
    let projection = temp.path().join("projection.npz");
    write_projection(&projection);
    atomic_json(
        &paths.standardization,
        &json!({
            "dim": 512,
            "mean": vec![0.0_f32; 512],
            "scale": vec![1.0_f32; 512],
            "foundation": {
                "cache_dir": foundation,
                "projection_file": projection,
                "raw_dim": 2048,
                "proj_dim": 1024,
            }
        }),
    )
    .unwrap();
    let mut sealed = Vec::new();
    for task in TASKS {
        for index in 0..sealed_target(task) {
            sealed.push(InventoryEntry {
                task: task.to_owned(),
                dataset_id: format!("{index:016x}"),
                dataset_hash: format!("{index:016x}"),
                source: PathBuf::from("private"),
                source_family: format!("{task}-sealed-{index}"),
                train_sha256: "a".repeat(64),
                test_sha256: "b".repeat(64),
                meta_sha256: "c".repeat(64),
                mirror: None,
            });
        }
    }
    freeze_sealed(&paths, &sealed, 7).unwrap();
    let first = activate_tranche(&paths, "tranche-test", std::slice::from_ref(&entry)).unwrap();
    assert_eq!(
        first,
        activate_tranche(&paths, "tranche-test", std::slice::from_ref(&entry)).unwrap()
    );
    let status = validate_catalog_pointer(&paths).unwrap();
    assert_eq!(status["healthy"], true);
    assert_eq!(status["row_count"], 1);
    assert_eq!(
        fs::read_to_string(paths.catalog_root.join("activations.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
}

/// Full activation fixture: one regression dataset, materialized cache,
/// foundation/projection/standardization stubs, sealed cohort, activated
/// tranche. Returns the entry and the activation receipt.
fn activated_fixture(
    temp: &TempDir,
    id: &str,
    tranche_id: &str,
) -> (ExpansionPaths, InventoryEntry, Value) {
    let paths = paths(temp);
    dataset(&paths.source_root, "regression", id, "1");
    let entry = discover(&paths, Some("regression"), None)
        .unwrap()
        .remove(0);
    materialize_cache(&paths, &entry, None).unwrap();
    let foundation = temp.path().join("foundation");
    fs::create_dir_all(&foundation).unwrap();
    fs::write(
        foundation.join(format!("{id}.npy")),
        npy_bytes(&[2048], &vec![0.0; 2048]),
    )
    .unwrap();
    let projection = temp.path().join("projection.npz");
    write_projection(&projection);
    atomic_json(
        &paths.standardization,
        &json!({
            "dim": 512,
            "mean": vec![0.0_f32; 512],
            "scale": vec![1.0_f32; 512],
            "foundation": {
                "cache_dir": foundation,
                "projection_file": projection,
                "raw_dim": 2048,
                "proj_dim": 1024,
            }
        }),
    )
    .unwrap();
    let mut sealed = Vec::new();
    for task in TASKS {
        for index in 0..sealed_target(task) {
            sealed.push(InventoryEntry {
                task: task.to_owned(),
                dataset_id: format!("{index:016x}"),
                dataset_hash: format!("{index:016x}"),
                source: PathBuf::from("private"),
                source_family: format!("{task}-sealed-{index}"),
                train_sha256: "a".repeat(64),
                test_sha256: "b".repeat(64),
                meta_sha256: "c".repeat(64),
                mirror: None,
            });
        }
    }
    freeze_sealed(&paths, &sealed, 7).unwrap();
    let receipt = activate_tranche(&paths, tranche_id, std::slice::from_ref(&entry)).unwrap();
    (paths, entry, receipt)
}

fn write_live_catalog(live_root: &Path, rows: &[(&str, &str, &str, &str)]) -> Value {
    fs::create_dir_all(live_root).unwrap();
    let mut blob = Vec::new();
    let mut entries = Vec::new();
    for (index, (task, dataset, family, split)) in rows.iter().enumerate() {
        let slice = vec![(index as u8).wrapping_add(1); CONTEXT_STRIDE];
        let context_sha256 = sha256_bytes(&slice);
        blob.extend_from_slice(&slice);
        entries.push(json!({
            "catalog_index": index,
            "context_sha256": context_sha256,
            "dataset_hash": dataset,
            "metric": if *task == "binary" { "auc" } else { "r2" },
            "source_catalog_index": index,
            "source_family": family,
            "split": split,
            "task": task,
            "test_csv": format!("/live/{dataset}/test.csv"),
            "test_sha256": "t".repeat(64),
            "train_csv": format!("/live/{dataset}/train.csv"),
            "train_sha256": "u".repeat(64),
        }));
    }
    let data_sha256 = sha256_bytes(&blob);
    let data_path = live_root.join(format!("{data_sha256}.f32.bin"));
    fs::write(&data_path, &blob).unwrap();
    let catalog = json!({
        "schema": "JopeTestOnlyContextCatalog.v1",
        "immutable": true,
        "dataset_dimensions": DATASET_DIM,
        "target_dimensions": TARGET_DIM,
        "record_stride_bytes": CONTEXT_STRIDE,
        "row_count": rows.len(),
        "regression_count": rows.iter().filter(|row| row.0 == "regression").count(),
        "binary_count": rows.iter().filter(|row| row.0 == "binary").count(),
        "data_sha256": data_sha256,
        "standardization_sha256": "s".repeat(64),
        "regression_source_data_sha256": "r".repeat(64),
        "source_family_disjoint": true,
        "entries": entries,
    });
    let catalog_path = live_root.join("live.catalog.json");
    atomic_json(&catalog_path, &catalog).unwrap();
    let pointer = json!({
        "schema": "JopeTestOnlyContextPointer.v1",
        "catalog": catalog_path,
        "catalog_sha256": sha256_file(&catalog_path).unwrap(),
        "data": data_path,
        "data_sha256": data_sha256,
    });
    atomic_json(&live_root.join("current.json"), &pointer).unwrap();
    pointer
}

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
