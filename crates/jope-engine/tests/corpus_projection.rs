//! Synthetic sharded corpora only. Excluded context paths are absent, mode 000
//! or a dangling symlink: opening any of them fails (mode 000 only without root),
//! and the symlink also fails `manifest::resolve` and `fs::metadata`.
use jope_parallel::{
    corpus::{ContextRow, Group, Record, RECORD_BYTES, SEQUENCE},
    corpus_projection::{self, verify_admitted_context_closure, verify_panel_record_contexts},
    manifest::{file_hash, hash, read_json, write_json},
    teaching_store::{self, Manifest, Shard, SCHEMA},
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::Command,
};

const CONTEXTS: [(&str, &str); 5] = [
    ("a", "meta_train"),
    ("x", "meta_test"),
    ("b", "meta_validation"),
    ("y", "meta_test"),
    ("c", "meta_train"),
];

/// Writes `src/corpus` with one shard per listed context position. Only the
/// admitted context files exist; the two meta_test files are never written.
fn fixture(root: &Path, shard_contexts: &[usize], corrupt_record: bool) {
    fs::create_dir_all(root.join("src/contexts")).unwrap();
    let mut contexts = Vec::new();
    for (cut, (name, partition)) in CONTEXTS.into_iter().enumerate() {
        let path = Path::new("src/contexts").join(format!("{name}.bin"));
        let bytes = format!("context {name} for cut {cut}");
        if partition != "meta_test" {
            fs::write(root.join(&path), &bytes).unwrap();
        }
        contexts.push(ContextRow {
            dataset: format!("dataset-{name}"),
            family: format!("family-{}", cut % 2),
            cut,
            partition: partition.into(),
            path,
            sha256: hash(bytes.as_bytes()),
            prepared: Path::new("src/prepared").join(name),
            prepared_sha256: hash(name.as_bytes()),
        });
    }
    let mut shards = Vec::new();
    let mut offsets = Vec::new();
    let mut total = 0usize;
    for (n, &context) in shard_contexts.iter().enumerate() {
        let relative = Path::new("src/store/shards").join(format!("shard-{n}"));
        let directory = root.join(&relative);
        fs::create_dir_all(&directory).unwrap();
        let count = 8 + n;
        let mut records = Vec::new();
        let mut labels = Vec::new();
        for i in 0..count {
            let corrupt = corrupt_record && n + 1 == shard_contexts.len() && i == 3;
            let mut record = Record {
                tokens: [0; SEQUENCE],
                legal: [0; SEQUENCE],
                length: 3,
                context: if corrupt { context + 1 } else { context },
                value: i as f32 / 8.,
                percentile: 1. - i as f32 / 16.,
                runtime: 0.5,
            };
            record.tokens[..3].copy_from_slice(&[1, 2 + n as i64, 3 + i as i64]);
            records.extend(record.bytes());
            let label = json!({"record":i,"context":context,"program":format!("p-{n}-{i}"),
                "rmse_cut_normalized":0.25 + i as f64,"runtime_seconds":0.5});
            labels.extend(serde_json::to_vec(&label).unwrap());
            labels.push(b'\n');
        }
        fs::write(directory.join("records.bin"), &records).unwrap();
        fs::write(directory.join("labels.jsonl"), &labels).unwrap();
        let mut shard = Shard {
            context,
            path: relative,
            sha256: String::new(),
            records: count,
            max_length: 3,
            records_sha256: hash(&records),
            labels_sha256: hash(&labels),
            legal_sets: vec![vec![0]],
        };
        write_json(&directory.join("shard.json"), &shard).unwrap();
        shard.sha256 = file_hash(&directory.join("shard.json")).unwrap();
        offsets.extend((total as u64).to_le_bytes());
        offsets.extend((count as u64).to_le_bytes());
        total += count;
        shards.push(shard);
    }
    let corpus = root.join("src/corpus");
    fs::create_dir_all(&corpus).unwrap();
    fs::write(corpus.join("groups.bin"), &offsets).unwrap();
    fs::write(corpus.join("vocabulary.json"), b"fixture vocabulary").unwrap();
    let manifest = Manifest {
        schema: SCHEMA.into(),
        qualification: true,
        label_policy: "reference_v1".into(),
        semantic_dependencies_sha256: String::new(),
        admission_policy: None,
        vocabulary_sha256: file_hash(&corpus.join("vocabulary.json")).unwrap(),
        context_index_sha256: hash(b"original context index"),
        contexts,
        records: total,
        record_bytes: RECORD_BYTES,
        records_sha256: hash(&serde_json::to_vec(&shards).unwrap()),
        groups_sha256: hash(&offsets),
        shards,
        sources: vec![],
        counts: BTreeMap::from([("groups".to_string(), shard_contexts.len())]),
    };
    let mut index = serde_json::to_value(&manifest).unwrap();
    // Recorded source indexes predate this key; the projection must not add it.
    index
        .as_object_mut()
        .unwrap()
        .remove("semantic_dependencies_sha256");
    index["retained_metadata"] = json!({"kept":"verbatim"});
    write_json(&corpus.join("index.json"), &index).unwrap();
    let provenance = json!({"index_sha256":file_hash(&corpus.join("index.json")).unwrap()});
    write_json(&corpus.join("provenance.json"), &provenance).unwrap();
}

fn project(root: &Path, output: &str) -> anyhow::Result<Value> {
    corpus_projection::project(root, root, Path::new("src/corpus"), Path::new(output))
}

fn refusal<T: std::fmt::Debug>(result: anyhow::Result<T>) -> String {
    format!("{:#}", result.unwrap_err())
}

fn untouched(root: &Path, output: &str) -> bool {
    let staging = root.join(format!("{output}.partial"));
    !root.join(output).exists() && !staging.exists()
}

fn keys(value: &Value) -> BTreeSet<String> {
    value.as_object().unwrap().keys().cloned().collect()
}

#[test]
fn projection_remaps_admitted_contexts_and_preserves_every_other_byte() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path();
    fixture(root, &[0, 2, 4], false);
    let receipt = project(root, "projected").unwrap();
    let source = root.join("src/corpus");
    let output = root.join("projected");
    assert_eq!(receipt["schema"], corpus_projection::PROJECTION_SCHEMA);
    assert_eq!(receipt["contexts_in"], 5);
    assert_eq!(receipt["contexts_out"], 3);
    assert_eq!(receipt["first_removed_position"], 1);
    assert_eq!(receipt["remapped_shards"], 2);
    assert_eq!(receipt["records"], 27);
    assert_eq!(
        receipt["removed_contexts_by_partition"],
        json!({"meta_test":2})
    );
    let source_index: Value = read_json(&source.join("index.json")).unwrap();
    let index: Value = read_json(&output.join("index.json")).unwrap();
    let mut expected = keys(&source_index);
    expected.insert("projection".into());
    assert_eq!(keys(&index), expected);
    assert!(index.get("semantic_dependencies_sha256").is_none());
    assert_eq!(
        index["retained_metadata"],
        source_index["retained_metadata"]
    );
    let catalogue = teaching_store::read_index(root, &output).unwrap();
    let groups: Vec<usize> = catalogue.groups.iter().map(|g| g.context).collect();
    assert_eq!(groups, [0, 1, 2]);
    let (loaded, records) = teaching_store::load(root, &output).unwrap();
    for group in &loaded.groups {
        for &id in &group.records {
            assert_eq!(records.get(id).context, group.context);
        }
    }
    let retained = [
        ("a", "meta_train"),
        ("b", "meta_validation"),
        ("c", "meta_train"),
    ];
    assert_eq!(loaded.contexts.len(), retained.len());
    for (row, (name, partition)) in loaded.contexts.iter().zip(retained) {
        let copy = Path::new("projected/contexts/src/contexts").join(format!("{name}.bin"));
        assert_eq!(row.partition, partition);
        assert_eq!(row.path, copy);
        assert_eq!(file_hash(&root.join(&row.path)).unwrap(), row.sha256);
    }
    let sources = source_index["contexts"].as_array().unwrap();
    let projected = index["contexts"].as_array().unwrap();
    assert_eq!(projected.len(), retained.len());
    for (row, position) in projected.iter().zip([0usize, 2, 4]) {
        let mut row = row.clone();
        row["path"] = sources[position]["path"].clone();
        assert_eq!(row, sources[position]);
    }
    let path = |shard: &Shard, file: &str| root.join(&shard.path).join(file);
    let before: Vec<Shard> = serde_json::from_value(source_index["shards"].clone()).unwrap();
    let after: Vec<Shard> = serde_json::from_value(index["shards"].clone()).unwrap();
    for ((old, new), context) in before.iter().zip(&after).zip([0u32, 1, 2]) {
        assert_eq!(new.context, context as usize);
        assert!(new.path.starts_with("projected/shards"));
        let was = fs::read(path(old, "records.bin")).unwrap();
        let now = fs::read(path(new, "records.bin")).unwrap();
        assert_eq!(was.len(), now.len());
        for (a, b) in was.chunks(RECORD_BYTES).zip(now.chunks(RECORD_BYTES)) {
            assert_eq!(a[..516], b[..516]);
            assert_eq!(a[516..520], (old.context as u32).to_le_bytes());
            assert_eq!(b[516..520], context.to_le_bytes());
            assert_eq!(a[520..], b[520..]);
        }
        let was = fs::read_to_string(path(old, "labels.jsonl")).unwrap();
        let now = fs::read_to_string(path(new, "labels.jsonl")).unwrap();
        assert_eq!(now.lines().count(), new.records);
        for (i, (a, b)) in was.lines().zip(now.lines()).enumerate() {
            let mut row: Value = serde_json::from_str(b).unwrap();
            assert_eq!(row["record"], i);
            assert_eq!(row["context"], context);
            row["context"] = json!(old.context);
            assert_eq!(row, serde_json::from_str::<Value>(a).unwrap());
        }
    }
    assert_eq!(
        fs::read(source.join("groups.bin")).unwrap(),
        fs::read(output.join("groups.bin")).unwrap()
    );
    let contexts = fs::read(output.join("contexts.json")).unwrap();
    assert_eq!(index["context_index_sha256"], hash(&contexts));
    assert_ne!(
        index["context_index_sha256"],
        source_index["context_index_sha256"]
    );
    let provenance: Value = read_json(&output.join("provenance.json")).unwrap();
    assert_eq!(
        provenance["index_sha256"],
        file_hash(&output.join("index.json")).unwrap()
    );
    assert_eq!(provenance["projection"], index["projection"]);
    assert_eq!(receipt["records_sha256"], index["records_sha256"]);
    assert_eq!(
        receipt["output_index_sha256"],
        file_hash(&output.join("index.json")).unwrap()
    );
}

#[test]
fn excluded_context_files_are_never_opened() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path();
    fixture(root, &[0, 2, 4], false);
    let excluded = [
        root.join("src/contexts/x.bin"),
        root.join("src/contexts/y.bin"),
    ];
    assert!(excluded.iter().all(|p| fs::symlink_metadata(p).is_err()));
    project(root, "absent").unwrap();
    // Mode 000 x: open fails without root. Dangling y: resolve, stat and open fail.
    fs::write(&excluded[0], b"excluded").unwrap();
    fs::set_permissions(&excluded[0], fs::Permissions::from_mode(0o000)).unwrap();
    std::os::unix::fs::symlink("missing-target", &excluded[1]).unwrap();
    let receipt = project(root, "unreadable").unwrap();
    assert_eq!(receipt["contexts_out"], 3);
    let copied = root.join("unreadable/contexts/src/contexts");
    assert!(copied.join("a.bin").is_file());
    assert!(fs::symlink_metadata(copied.join("x.bin")).is_err());
    assert!(fs::symlink_metadata(copied.join("y.bin")).is_err());
}

#[test]
fn closure_guard_refuses_the_source_and_accepts_the_projection() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path();
    fixture(root, &[0, 2, 4], false);
    let source: Value = read_json(&root.join("src/corpus/index.json")).unwrap();
    let error = refusal(verify_admitted_context_closure(&source));
    assert!(
        error.contains("non-admitted context partition at position 1"),
        "{error}"
    );
    assert!(error.contains("meta_test"), "{error}");
    project(root, "projected").unwrap();
    let projected: Value = read_json(&root.join("projected/index.json")).unwrap();
    verify_admitted_context_closure(&projected).unwrap();
}

#[test]
fn shard_on_a_non_admitted_context_is_refused() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path();
    fixture(root, &[0, 1, 4], false);
    let error = refusal(project(root, "projected"));
    assert!(
        error.contains("shard references a non-admitted context"),
        "{error}"
    );
    assert!(untouched(root, "projected"));
}

#[test]
fn record_context_that_disagrees_with_its_shard_is_refused() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path();
    fixture(root, &[0, 2, 4], true);
    teaching_store::load(root, &root.join("src/corpus")).unwrap();
    let error = refusal(project(root, "projected"));
    assert!(
        error.contains("record context differs from its shard context"),
        "{error}"
    );
    assert!(untouched(root, "projected"));
}

#[test]
fn projection_bytes_are_deterministic_across_roots() {
    let d = tempfile::tempdir().unwrap();
    let source = d.path();
    fixture(source, &[0, 2, 4], false);
    let mut written = Vec::new();
    for name in ["first", "second"] {
        let root = source.join(name);
        fs::create_dir(&root).unwrap();
        let out = Path::new("projected");
        corpus_projection::project(&root, source, Path::new("src/corpus"), out).unwrap();
        teaching_store::load(&root, &root.join(out)).unwrap();
        for file in ["index.json", "provenance.json", "contexts.json"] {
            written.push(fs::read(root.join(out).join(file)).unwrap());
        }
    }
    assert_eq!(written[..3], written[3..]);
}

#[test]
fn existing_output_or_staging_is_never_replaced() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path();
    fixture(root, &[0, 2, 4], false);
    project(root, "projected").unwrap();
    let path = root.join("projected/index.json");
    let index = fs::read(&path).unwrap();
    let error = refusal(project(root, "projected"));
    assert!(error.contains("already exists"), "{error}");
    assert_eq!(fs::read(&path).unwrap(), index);
    let marker = root.join("other.partial/marker");
    fs::create_dir(root.join("other.partial")).unwrap();
    fs::write(&marker, b"kept").unwrap();
    assert!(project(root, "other").is_err());
    assert!(!root.join("other").exists());
    assert_eq!(fs::read(&marker).unwrap(), b"kept");
}

#[test]
fn cli_projects_a_corpus_and_prints_its_receipt() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path();
    fixture(root, &[0, 2, 4], false);
    let run = |root: &Path, source_root: Option<&Path>| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_jope-parallel"));
        command.arg("--root").arg(root).arg("project-corpus");
        if let Some(source_root) = source_root {
            command.arg("--source-root").arg(source_root);
        }
        command
            .args(["--corpus", "src/corpus", "--output", "projected"])
            .output()
            .unwrap()
    };
    let done = run(root, None);
    assert!(done.status.success(), "{done:?}");
    let receipt: Value = serde_json::from_slice(&done.stdout).unwrap();
    assert_eq!(receipt["schema"], "jope.corpus-projection/v1");
    assert_eq!(receipt["contexts_out"], 3);
    let index = fs::read(root.join("projected/index.json")).unwrap();
    assert_eq!(receipt["output_index_sha256"], hash(&index));
    assert!(!run(root, None).status.success());
    let other = root.join("other");
    fs::create_dir(&other).unwrap();
    let done = run(other.as_path(), Some(root));
    assert!(done.status.success(), "{done:?}");
    let again = fs::read(other.join("projected/index.json")).unwrap();
    assert_eq!(again, index);
}

#[test]
fn panel_records_must_carry_their_group_validation_context() {
    let row = |partition: &str| ContextRow {
        dataset: partition.into(),
        family: "family".into(),
        cut: 0,
        partition: partition.into(),
        path: "context.bin".into(),
        sha256: hash(partition.as_bytes()),
        prepared: "prepared".into(),
        prepared_sha256: hash(b"prepared"),
    };
    let contexts = [
        row("meta_train"),
        row("meta_validation"),
        row("meta_validation"),
    ];
    let groups = [0, 1, 2].map(|context| Group {
        context,
        family: "family".into(),
        partition: contexts[context].partition.clone(),
        records: (context * 8..context * 8 + 8).collect(),
        max_length: 3,
    });
    let recorded = |id: usize| -> anyhow::Result<usize> { Ok(id / 8) };
    verify_panel_record_contexts(&[8, 9, 15], recorded, &groups, &contexts).unwrap();
    let refused = verify_panel_record_contexts(&[8, 0], recorded, &groups, &contexts);
    let error = refusal(refused);
    assert!(error.contains("meta_validation"), "{error}");
    // Record 9 claims the other validation context: only the owner check refuses it.
    let swapped = |id: usize| -> anyhow::Result<usize> { Ok(if id == 9 { 2 } else { id / 8 }) };
    let error = refusal(verify_panel_record_contexts(
        &[8, 9],
        swapped,
        &groups,
        &contexts,
    ));
    assert!(error.contains("differs from its group context"), "{error}");
    let refused = verify_panel_record_contexts(&[24], recorded, &groups, &contexts);
    assert!(refused.is_err());
}

#[test]
fn closure_guard_is_an_allowlist_that_accepts_empty_closures() {
    verify_admitted_context_closure(&json!({"contexts":[]})).unwrap();
    verify_admitted_context_closure(&json!({"schema":"parallel.teaching_corpus.v1"})).unwrap();
    let admitted = json!({"contexts":[{"partition":"meta_train"},
        {"partition":"meta_validation"}]});
    verify_admitted_context_closure(&admitted).unwrap();
    let cases = [
        json!({"path":"context.bin"}),
        json!({"partition":"development_validation"}),
        json!({"partition":null}),
        json!({"partition":7}),
        json!({"partition":"meta_test"}),
    ];
    for case in cases {
        let index = json!({"contexts":[{"partition":"meta_train"},case]});
        let error = refusal(verify_admitted_context_closure(&index));
        assert!(
            error.contains("non-admitted context partition at position 1"),
            "{error}"
        );
    }
    let malformed = json!({"contexts":{"partition":"meta_train"}});
    assert!(verify_admitted_context_closure(&malformed).is_err());
    assert!(corpus_projection::admitted_partition("meta_validation"));
    assert!(!corpus_projection::admitted_partition("meta_test"));
    assert_eq!(
        corpus_projection::ADMITTED_PARTITIONS,
        ["meta_train", "meta_validation"]
    );
}
