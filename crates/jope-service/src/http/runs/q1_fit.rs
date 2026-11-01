//! Strict native FIT import under the existing exact Q1 binding.
use super::Q1DatasetBinding;
use anyhow::{ensure, Context, Result};
use jope_runtime::table::CsvTable;
use serde_json::{json, Value};
use std::{fs, io::Read, path::Path};

const MAX_METADATA_BYTES: u64 = 8 * 1024 * 1024;

fn shape(table: &CsvTable, binding: &Q1DatasetBinding) -> bool {
    table.rows.len() as u64 == binding.train_rows
        && table.headers.len() == binding.feature_columns.len() + 1
        && table.headers.contains(&binding.target)
        && binding
            .feature_columns
            .iter()
            .all(|name| table.headers.contains(name))
}

fn read_metadata(home: &Path, sha: &str, bytes: Option<u64>) -> Result<Vec<u8>> {
    ensure!(super::valid_hex(sha, 64), "invalid native metadata digest");
    let q1 = home.join("q1");
    let catalog = q1.join("catalog");
    for directory in [&q1, &catalog] {
        ensure!(
            fs::symlink_metadata(directory)?.file_type().is_dir(),
            "native metadata directory is not real"
        );
    }
    let path = catalog.join(format!("{sha}.json"));
    let metadata = fs::symlink_metadata(&path)?;
    ensure!(
        metadata.file_type().is_file() && metadata.len() <= MAX_METADATA_BYTES,
        "native metadata must be a bounded regular file"
    );
    ensure!(
        bytes.is_none_or(|size| size == metadata.len()),
        "native metadata size differs"
    );
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    let opened = file.metadata()?;
    ensure!(
        opened.file_type().is_file() && opened.len() <= MAX_METADATA_BYTES,
        "opened native metadata is not bounded regular data"
    );
    let mut value = Vec::new();
    file.take(MAX_METADATA_BYTES + 1).read_to_end(&mut value)?;
    ensure!(
        value.len() as u64 <= MAX_METADATA_BYTES && super::hash(&value) == sha,
        "native metadata digest differs"
    );
    Ok(value)
}

fn native_metadata(home: &Path, binding: &Q1DatasetBinding) -> Result<()> {
    let record: Value =
        serde_json::from_slice(&read_metadata(home, &binding.catalog_record_sha256, None)?)?;
    ensure!(
        record["id"] == binding.id
            && record["canonical_sha256"] == binding.dataset_sha256
            && record["origin"] == "real"
            && record["task"] == "regression",
        "native catalog dataset identity differs"
    );
    let train = &record["files"]["train"];
    ensure!(
        train["sha256"] == binding.train_blob_sha256
            && train["bytes"] == binding.train_blob_bytes
            && train["rows"] == binding.train_rows
            && train["columns"] == binding.features + 1
            && train["has_header"] == false
            && record["shape"]["features"] == binding.features
            && record["target"]["index"] == -1
            && record["target"]["name"] == binding.target,
        "native FIT format or role differs"
    );
    let meta = &record["files"]["meta"];
    let meta_sha = meta["sha256"]
        .as_str()
        .context("native meta digest missing")?;
    let meta_bytes = meta["bytes"].as_u64().context("native meta size missing")?;
    ensure!(
        meta_bytes <= MAX_METADATA_BYTES,
        "native meta exceeds limit"
    );
    let meta: Value = serde_json::from_slice(&read_metadata(home, meta_sha, Some(meta_bytes))?)?;
    ensure!(
        meta["dataset_hash"] == binding.id
            && meta["target_column"] == binding.target
            && meta["feature_columns"] == json!(binding.feature_columns)
            && meta["n_features"] == binding.features
            && meta["train_rows"] == binding.train_rows
            && meta["headers_removed_from_csv"] == true
            && meta["target_is_final_column"] == true,
        "native target, feature order or row identity differs"
    );
    // Some native records retain pre-ingest output_headers/counts. The current
    // feature_columns/n_features, manifest width and final-target flag agree.
    Ok(())
}

fn header(binding: &Q1DatasetBinding) -> Vec<u8> {
    let mut out = Vec::new();
    for (index, name) in binding
        .feature_columns
        .iter()
        .chain(std::iter::once(&binding.target))
        .enumerate()
    {
        if index > 0 {
            out.push(b',');
        }
        if name
            .bytes()
            .any(|byte| matches!(byte, b',' | b'"' | b'\n' | b'\r'))
        {
            out.push(b'"');
            for byte in name.bytes() {
                out.push(byte);
                if byte == b'"' {
                    out.push(byte);
                }
            }
            out.push(b'"');
        } else {
            out.extend_from_slice(name.as_bytes());
        }
    }
    out.push(b'\n');
    out
}

pub(super) fn parse(home: &Path, binding: &Q1DatasetBinding, raw: &[u8]) -> Result<CsvTable> {
    ensure!(
        raw.len() as u64 == binding.train_blob_bytes
            && super::hash(raw) == binding.train_blob_sha256,
        "FIT bytes differ from binding"
    );
    let limits = jope_runtime::limits::Limits {
        max_input_bytes: binding.train_blob_bytes,
        ..Default::default()
    };
    if let Ok(table) = jope_runtime::table::parse_csv(raw, &limits) {
        if shape(&table, binding) {
            return Ok(table);
        }
    }
    // No guessed header or caller flag: native metadata must prove the exact
    // bound content is headerless, with this ordered feature list and target.
    native_metadata(home, binding)?;
    let mut prepared = header(binding);
    let size = prepared
        .len()
        .checked_add(raw.len())
        .context("FIT import size overflow")?;
    prepared.extend_from_slice(raw);
    let table = jope_runtime::table::parse_csv(
        &prepared,
        &jope_runtime::limits::Limits {
            max_input_bytes: u64::try_from(size)?,
            ..Default::default()
        },
    )?;
    ensure!(
        shape(&table, binding),
        "imported FIT shape differs from binding"
    );
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn fixture(raw: &[u8]) -> (TempDir, Q1DatasetBinding, Value, Value) {
        let home = tempfile::tempdir().unwrap();
        fs::create_dir_all(home.path().join("q1/catalog")).unwrap();
        let meta = json!({"dataset_hash":"a".repeat(16),"target_column":"y", "feature_columns":["x,quoted"],
            "n_features":1,"train_rows":2,"headers_removed_from_csv":true,"target_is_final_column":true,
            "output_headers":["stale","x,quoted","y"],"counts":{"n_features":2}});
        let meta_raw = serde_json::to_vec(&meta).unwrap();
        let catalog = json!({"id":"a".repeat(16),"canonical_sha256":"a".repeat(64),"origin":"real","task":"regression",
            "shape":{"features":1},"target":{"index":-1,"name":"y"},
            "files":{"train":{"sha256":super::super::hash(raw),"bytes":raw.len(),"rows":2,"columns":2,"has_header":false},
                "meta":{"sha256":super::super::hash(&meta_raw),"bytes":meta_raw.len()}}});
        let binding: Q1DatasetBinding = serde_json::from_value(json!({"id":"a".repeat(16),"task":"regression",
            "dataset_sha256":"a".repeat(64),"family":"b".repeat(40),"family_authoritative":true,
            "rows":2,"features":1,"cut_sha256":"c".repeat(64),"holdout_group":"sealed_meta_test",
            "catalog_version":"v246","catalog_record_sha256":"d".repeat(64),"dataset_binding_sha256":"e".repeat(64),
            "target":"y","feature_columns":["x,quoted"],"train_rows":2,"train_blob_sha256":super::super::hash(raw),
            "train_blob_bytes":raw.len(),"folds":[{"repeat":0,"train":[0],"validation":[1]},{"repeat":0,"train":[1],"validation":[0]}]})).unwrap();
        (home, binding, catalog, meta)
    }
    fn install(home: &Path, binding: &mut Q1DatasetBinding, mut catalog: Value, meta: &Value) {
        let meta_raw = serde_json::to_vec(meta).unwrap();
        catalog["files"]["meta"] =
            json!({"sha256":super::super::hash(&meta_raw),"bytes":meta_raw.len()});
        let raw = serde_json::to_vec(&catalog).unwrap();
        binding.catalog_record_sha256 = super::super::hash(&raw);
        fs::write(
            home.join("q1/catalog")
                .join(format!("{}.json", binding.catalog_record_sha256)),
            raw,
        )
        .unwrap();
        fs::write(
            home.join("q1/catalog")
                .join(format!("{}.json", super::super::hash(&meta_raw))),
            meta_raw,
        )
        .unwrap();
    }
    #[test]
    fn pinned_headerless_import_retains_first_row_values_and_source_bytes() {
        let raw = b"1,0.25\n2,0.75\n";
        let (home, mut binding, catalog, meta) = fixture(raw);
        assert!(parse(home.path(), &binding, raw).is_err());
        install(home.path(), &mut binding, catalog, &meta);
        let table = parse(home.path(), &binding, raw).unwrap();
        assert_eq!(table.headers, ["x,quoted", "y"]);
        assert_eq!(
            table.rows,
            vec![
                vec![Some("1".into()), Some("0.25".into())],
                vec![Some("2".into()), Some("0.75".into())]
            ]
        );
        assert_eq!(raw, b"1,0.25\n2,0.75\n");
    }
    #[test]
    fn existing_headered_binding_needs_no_native_metadata() {
        let raw = b"\"x,quoted\",y\n1,0.25\n2,0.75\n";
        let (home, binding, _, _) = fixture(raw);
        fs::remove_dir_all(home.path().join("q1")).unwrap();
        assert_eq!(parse(home.path(), &binding, raw).unwrap().rows.len(), 2);
    }
    #[test]
    fn metadata_role_target_order_and_digest_drift_refuse() {
        let raw = b"1,0.25\n2,0.75\n";
        for field in [
            "has_header",
            "train_sha",
            "dataset",
            "feature_order",
            "target",
            "rows",
        ] {
            let (home, mut binding, mut catalog, mut meta) = fixture(raw);
            match field {
                "has_header" => catalog["files"]["train"]["has_header"] = json!(true),
                "train_sha" => catalog["files"]["train"]["sha256"] = json!("f".repeat(64)),
                "dataset" => catalog["canonical_sha256"] = json!("f".repeat(64)),
                "feature_order" => meta["feature_columns"] = json!(["other"]),
                "target" => meta["target_column"] = json!("other"),
                "rows" => meta["train_rows"] = json!(3),
                _ => unreachable!(),
            }
            install(home.path(), &mut binding, catalog, &meta);
            assert!(parse(home.path(), &binding, raw).is_err(), "{field}");
        }
        let (home, mut binding, catalog, meta) = fixture(raw);
        install(home.path(), &mut binding, catalog, &meta);
        let path = home
            .path()
            .join("q1/catalog")
            .join(format!("{}.json", binding.catalog_record_sha256));
        fs::write(path, b"{}").unwrap();
        assert!(parse(home.path(), &binding, raw).is_err());
        assert!(parse(home.path(), &binding, b"3,0.25\n2,0.75\n").is_err());
    }
    #[cfg(unix)]
    #[test]
    fn symlinked_native_metadata_is_refused() {
        let raw = b"1,0.25\n2,0.75\n";
        let (home, mut binding, catalog, meta) = fixture(raw);
        install(home.path(), &mut binding, catalog, &meta);
        let path = home
            .path()
            .join("q1/catalog")
            .join(format!("{}.json", binding.catalog_record_sha256));
        let saved = home.path().join("saved.json");
        fs::rename(&path, &saved).unwrap();
        std::os::unix::fs::symlink(saved, path).unwrap();
        assert!(parse(home.path(), &binding, raw).is_err());
    }
}
