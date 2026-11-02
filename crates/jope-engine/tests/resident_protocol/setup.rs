fn request() -> Value {
    json!({"rank":0,"task":{"kind":"synthetic","output":"same"}})
}
fn worker() -> WorkerIdentity {
    WorkerIdentity {
        process: process_identity(std::process::id()).unwrap(),
        socket_device: 1,
        socket_inode: 2,
        executable_sha256: "a".repeat(64),
    }
}

fn manifest_request(root: &std::path::Path) -> Value {
    fs::create_dir_all(root.join("corpus")).unwrap();
    fs::create_dir_all(root.join("run")).unwrap();
    write_json(&root.join("run/run.json"), &json!({"schema":"test"})).unwrap();
    write_json(
        &root.join("corpus/vocabulary.json"),
        &json!({"tokens":["original"]}),
    )
    .unwrap();
    fs::write(root.join("corpus/records.bin"), b"original records").unwrap();
    write_json(&root.join("corpus/index.json"), &json!({"schema":"parallel.teaching_corpus.v1",
        "vocabulary_sha256":manifest::file_hash(&root.join("corpus/vocabulary.json")).unwrap(),
        "records_sha256":manifest::file_hash(&root.join("corpus/records.bin")).unwrap(),"contexts":[]})).unwrap();
    json!({"root":root,"rank":0,"task":{"kind":"Train","corpus":root.join("corpus"),
        "run":root.join("run"),"resume":false,"initialize":null},"env":{},"executable_sha256":"a".repeat(64)})
}

fn checkpoint_fixture(root: &std::path::Path, name: &str) -> String {
    let cp = root.join(name);
    fs::create_dir_all(&cp).unwrap();
    for filename in ["model.ot", "optimizer.ot", "optimizer.json", "config.json"] {
        fs::write(cp.join(filename), format!("{name}:{filename}")).unwrap();
    }
    write_json(
        &cp.join("state.json"),
        &json!({
        "model_sha256":manifest::file_hash(&cp.join("model.ot")).unwrap(),
        "optimizer_sha256":manifest::file_hash(&cp.join("optimizer.ot")).unwrap(),
        "optimizer_metadata_sha256":manifest::file_hash(&cp.join("optimizer.json")).unwrap(),
        "config_sha256":manifest::file_hash(&cp.join("config.json")).unwrap()}),
    )
    .unwrap();
    manifest::file_hash(&cp.join("state.json")).unwrap()
}
