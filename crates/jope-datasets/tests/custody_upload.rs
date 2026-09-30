use jope_datasets::custody::upload;
use jope_datasets::hash::sha256_hex;
use jope_datasets::store::Store;
use serde_json::json;
use std::path::Path;

fn add_file(root: &Path, inventory: &mut String, path: &str, body: &[u8], class: &str) -> String {
    let source = root.join(path);
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    std::fs::write(source, body).unwrap();
    let sha = sha256_hex(body);
    let row = json!({
        "source_root":root.display().to_string(),
        "path":path,
        "path_hex":hex::encode(path.as_bytes()),
        "type":"file",
        "classification":class,
        "size":body.len(),
        "sha256":sha,
        "s3_object":null,
        "s3_version":null
    });
    inventory.push_str(&row.to_string());
    inventory.push('\n');
    sha
}

#[test]
fn custody_upload_preserves_real_synthetic_path_and_catalog_superset() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("remote_super");
    std::fs::create_dir_all(&root).unwrap();
    let store_dir = temp.path().join("store");
    std::fs::create_dir_all(&store_dir).unwrap();
    let store = Store::open(&format!("file://{}", store_dir.display())).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut first = String::new();
    let real_sha = add_file(
        &root,
        &mut first,
        "quant/synthetic/binary/0123456789abcdef/train.csv",
        b"1,2\n",
        "real",
    );
    add_file(
        &root,
        &mut first,
        "quant/synthetic/binary/0123456789abcdef/score.json",
        b"{\"score\":1}",
        "unresolved",
    );
    add_file(
        &root,
        &mut first,
        "quant/synthetic/binary/fedcba9876543210/train.csv",
        b"generated\n",
        "confirmed_synthetic",
    );
    let first_inventory = temp.path().join("first-inventory.jsonl");
    std::fs::write(&first_inventory, first).unwrap();
    let first_out = temp.path().join("first-catalog.jsonl");
    assert_eq!(
        rt.block_on(upload(&first_inventory, &root, &first_out, &store))
            .unwrap(),
        2
    );
    assert_eq!(
        rt.block_on(upload(&first_inventory, &root, &first_out, &store))
            .unwrap(),
        2
    );
    let pointer = rt
        .block_on(store.get_string("custody/quant/catalog/LATEST"))
        .unwrap()
        .unwrap();
    let first_catalog = rt
        .block_on(store.get_string(&format!(
            "custody/quant/catalog/sha256/{}.jsonl",
            pointer.trim()
        )))
        .unwrap()
        .unwrap();
    assert_eq!(first_catalog.lines().count(), 2);
    assert!(first_catalog.contains(&real_sha));

    let mut second = String::new();
    add_file(
        &root,
        &mut second,
        "quant/models/prediction.bin",
        b"prediction",
        "unresolved",
    );
    let second_inventory = temp.path().join("second-inventory.jsonl");
    std::fs::write(&second_inventory, second).unwrap();
    let second_out = temp.path().join("second-catalog.jsonl");
    assert_eq!(
        rt.block_on(upload(&second_inventory, &root, &second_out, &store))
            .unwrap(),
        3
    );
    assert_eq!(
        std::fs::read_to_string(second_out).unwrap().lines().count(),
        3
    );

    let old_blob = store_dir.join("custody/quant/blobs/sha256").join(real_sha);
    let mut corrupt = std::fs::read(&old_blob).unwrap();
    corrupt[0] ^= 1;
    std::fs::write(old_blob, corrupt).unwrap();
    let latest_before = rt
        .block_on(store.get_string("custody/quant/catalog/LATEST"))
        .unwrap();
    let third_out = temp.path().join("third-catalog.jsonl");
    assert!(rt
        .block_on(upload(&second_inventory, &root, &third_out, &store))
        .is_err());
    assert_eq!(
        rt.block_on(store.get_string("custody/quant/catalog/LATEST"))
            .unwrap(),
        latest_before
    );
}

#[test]
fn changed_source_is_not_published() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("source.csv");
    std::fs::write(&path, b"1,2\n").unwrap();
    let old = sha256_hex(b"1,2\n");
    std::fs::write(&path, b"9,2\n").unwrap();
    let store_dir = temp.path().join("store");
    std::fs::create_dir_all(&store_dir).unwrap();
    let store = Store::open(&format!("file://{}", store_dir.display())).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    assert!(rt
        .block_on(store.put_file_verified("blobs/sha256/old", &path, &old, 4))
        .is_err());
    assert!(rt
        .block_on(store.size("blobs/sha256/old"))
        .unwrap()
        .is_none());
}

#[test]
fn interrupted_custody_upload_resumes_verified_rows() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("remote_super");
    std::fs::create_dir_all(&root).unwrap();
    let store_dir = temp.path().join("store");
    std::fs::create_dir_all(&store_dir).unwrap();
    let store = Store::open(&format!("file://{}", store_dir.display())).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut inventory = String::new();
    add_file(&root, &mut inventory, "quant/a", b"first", "unresolved");
    add_file(&root, &mut inventory, "quant/b", b"second", "unresolved");
    let inventory_path = temp.path().join("inventory.jsonl");
    std::fs::write(&inventory_path, inventory).unwrap();
    std::fs::remove_file(root.join("quant/b")).unwrap();
    let output = temp.path().join("catalog.jsonl");
    assert!(rt
        .block_on(upload(&inventory_path, &root, &output, &store))
        .is_err());
    let partial = output.with_extension("jsonl.partial");
    assert_eq!(
        std::fs::read_to_string(&partial).unwrap().lines().count(),
        1
    );
    std::fs::write(root.join("quant/b"), b"second").unwrap();
    assert_eq!(
        rt.block_on(upload(&inventory_path, &root, &output, &store))
            .unwrap(),
        2
    );
    assert_eq!(std::fs::read_to_string(output).unwrap().lines().count(), 2);
}

#[test]
fn partial_census_cannot_publish() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("remote_super");
    std::fs::create_dir(&root).unwrap();
    let partial = temp.path().join("remote-super.jsonl.partial");
    std::fs::write(&partial, b"").unwrap();
    let store_dir = temp.path().join("store");
    std::fs::create_dir(&store_dir).unwrap();
    let store = Store::open(&format!("file://{}", store_dir.display())).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    assert!(rt
        .block_on(upload(
            &partial,
            &root,
            &temp.path().join("out.jsonl"),
            &store
        ))
        .is_err());
    assert!(rt
        .block_on(store.get_bytes("custody/quant/catalog/LATEST"))
        .unwrap()
        .is_none());
}
