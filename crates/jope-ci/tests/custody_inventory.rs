use jope_ci::custody_inventory::run;
use serde_json::Value;

#[test]
fn inventory_keeps_incomplete_real_data_under_synthetic_path() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path().join("remote_super");
    let dataset = root.join("quant/synthetic/binary/0123456789abcdef");
    std::fs::create_dir_all(&dataset).unwrap();
    std::fs::write(dataset.join("meta.json"), r#"{"origin":"real"}"#).unwrap();
    std::fs::write(dataset.join("train.csv"), b"1,2\n").unwrap();
    std::os::unix::fs::symlink("train.csv", dataset.join("link.csv")).unwrap();
    let out = fixture.path().join("inventory.jsonl");
    run(&root, &out).unwrap();
    let rows: Vec<Value> = std::fs::read_to_string(&out)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let dir = rows
        .iter()
        .find(|row| row["path"] == "quant/synthetic/binary/0123456789abcdef")
        .unwrap();
    assert_eq!(dir["classification"], "real");
    assert_eq!(
        dir["incomplete_dataset_missing"],
        serde_json::json!(["test.csv"])
    );
    let file = rows
        .iter()
        .find(|row| row["path"] == "quant/synthetic/binary/0123456789abcdef/train.csv")
        .unwrap();
    assert_eq!(file["classification"], "real");
    assert_eq!(
        file["sha256"],
        "52186c933993da4082b3cdc7c40bb4bf735b391ff54a2ef78c037dda6c38a680"
    );
    let link = rows.iter().find(|row| row["type"] == "symlink").unwrap();
    assert_eq!(link["link_target_hex"], "747261696e2e637376");
    assert!(
        run(&root, &out).is_err(),
        "completed inventory is immutable"
    );
}
