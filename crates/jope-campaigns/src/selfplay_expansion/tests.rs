use super::cache::verify_native_encoder;
use super::context::parse_npy;
use super::prelude::*;
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

mod activation_tests;
mod cache_tests;
mod identity_tests;
mod live_tests;
