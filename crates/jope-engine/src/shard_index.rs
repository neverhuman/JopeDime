//! Historical shard-index closure (MR06). Lives outside `label_admission.rs` so that file
//! keeps its frozen compat-freeze lineage identity. Read-only: it hashes and compares bytes.

use crate::label_admission::verify_shard_directory_payload_hashes;
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

/// Verify every shard directory named by a historical shard index against its recorded
/// digests. Refuses unless exactly `expected_shards` entries exist and each one matches. Each
/// entry's `path` is resolved under `shard_root`. Returns the number of verified shards.
pub fn verify_shard_index(
    index: &Value,
    shard_root: &Path,
    expected_shards: usize,
) -> Result<usize> {
    let shards = index["shards"]
        .as_array()
        .context("shard index has no shards array")?;
    ensure!(
        shards.len() == expected_shards,
        "shard count differs: index has {}, expected {expected_shards}",
        shards.len()
    );
    for (i, shard) in shards.iter().enumerate() {
        let path = shard["path"].as_str().context("shard path absent")?;
        let expected_sha = shard["sha256"].as_str().context("shard sha256 absent")?;
        let expected_records = shard["records_sha256"]
            .as_str()
            .context("records_sha256 absent")?;
        let expected_labels = shard["labels_sha256"]
            .as_str()
            .context("labels_sha256 absent")?;
        verify_shard_directory_payload_hashes(
            &shard_root.join(path),
            expected_sha,
            expected_records,
            expected_labels,
        )
        .with_context(|| format!("shard {i} ({path}) failed payload verification"))?;
    }
    Ok(shards.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::hash;
    use serde_json::json;

    #[test]
    fn shard_index_payload_closure_verified() {
        let root = tempfile::tempdir().unwrap();
        let mut shards = Vec::new();
        for (i, body) in [b"alpha".as_slice(), b"beta".as_slice()].iter().enumerate() {
            let dir = root.path().join(format!("store/shards/s{i}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("shard.json"), body).unwrap();
            std::fs::write(dir.join("records.bin"), [i as u8; 4]).unwrap();
            std::fs::write(dir.join("labels.jsonl"), b"{}\n").unwrap();
            shards.push(json!({
                "path": format!("store/shards/s{i}"),
                "sha256": hash(body),
                "records_sha256": hash(&[i as u8; 4]),
                "labels_sha256": hash(b"{}\n"),
            }));
        }
        let index = json!({ "shards": shards });
        assert_eq!(verify_shard_index(&index, root.path(), 2).unwrap(), 2);
        assert!(
            verify_shard_index(&index, root.path(), 7877).is_err(),
            "wrong shard count accepted"
        );
        std::fs::write(root.path().join("store/shards/s1/records.bin"), [9u8; 4]).unwrap();
        assert!(
            verify_shard_index(&index, root.path(), 2).is_err(),
            "tampered shard accepted"
        );
    }
}
