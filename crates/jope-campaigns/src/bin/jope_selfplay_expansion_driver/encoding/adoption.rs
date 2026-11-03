use super::super::super::*;
use super::*;

pub(crate) fn publish_encoding_receipt(receipt: &Value) -> Result<()> {
    let dataset_content = receipt
        .pointer("/dataset/content_sha256")
        .and_then(Value::as_str)
        .context("encoding receipt lacks dataset content identity")?;
    let attempt = receipt
        .get("attempt_number")
        .and_then(Value::as_u64)
        .context("encoding receipt lacks attempt number")?;
    let job = receipt
        .get("job_identity")
        .and_then(Value::as_str)
        .context("encoding receipt lacks job identity")?;
    let directory = Path::new(root_str()).join("ops/encoding-adoption-v2");
    std::fs::create_dir_all(&directory)?;
    let path = directory.join(format!(
        "{}-a{}-{}.json",
        dataset_content,
        attempt,
        &sha256_bytes(job.as_bytes())[..16]
    ));
    let mut bytes = serde_json::to_vec(receipt)?;
    bytes.push(b'\n');
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut file) => {
            use std::io::Write;
            file.write_all(&bytes)?;
            file.sync_data()?;
            let mut events = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(Path::new(root_str()).join("ops/encoding-adoption-v2.jsonl"))?;
            events.write_all(&bytes)?;
            events.sync_data()?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if std::fs::read(&path)? != bytes {
                bail!(
                    "immutable encoding adoption receipt drift: {}",
                    path.display()
                );
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

pub(crate) fn emit_chunk_adoption_receipts(
    job_id: &str,
    job_state: &str,
    task: &str,
    root: &Path,
) -> Result<usize> {
    let Some(listing) = encoding_chunk_listing(job_id)? else {
        return Ok(0);
    };
    let encoder = listing
        .get("encoder_identity_sha256")
        .and_then(Value::as_str)
        .context("EncodingChunk.v2 lacks encoder identity")?;
    let attempt = listing
        .get("attempt_number")
        .and_then(Value::as_u64)
        .context("EncodingChunk.v2 lacks attempt number")?;
    let rows = listing
        .get("ids")
        .and_then(Value::as_array)
        .context("EncodingChunk.v2 lacks dataset identities")?;
    let registry = Path::new(registry_str()).join(task);
    let mut adopted = 0;
    for row in rows {
        let dataset_id = row
            .get("id")
            .and_then(Value::as_str)
            .context("EncodingChunk.v2 dataset lacks id")?;
        let train_sha256 = row
            .get("train_sha256")
            .and_then(Value::as_str)
            .context("EncodingChunk.v2 dataset lacks train hash")?;
        let test_sha256 = row
            .get("test_sha256")
            .and_then(Value::as_str)
            .context("EncodingChunk.v2 dataset lacks test hash")?;
        let train = registry.join(dataset_id).join("train.csv");
        let test = registry.join(dataset_id).join("test.csv");
        if sha256_file(&train)? != train_sha256 || sha256_file(&test)? != test_sha256 {
            bail!("{job_id}: registry source content changed for {dataset_id}");
        }
        let dataset_content_sha256 =
            sha256_bytes(format!("{train_sha256}:{test_sha256}").as_bytes());
        let target = root.join("targets").join(dataset_id);
        let foundation = root.join("foundation").join(format!("{dataset_id}.npy"));
        let disposition = if target.is_dir() {
            adopted += 1;
            json!({
                "kind": "succeeded",
                "encoded_train_sha256": hash_directory(&target)?,
                "encoded_test_sha256": if foundation.is_file() {
                    sha256_file(&foundation)?
                } else {
                    hash_directory(&target)?
                },
            })
        } else {
            json!({
                "kind": "failed",
                "error_class": format!("{job_state}:missing_dataset_output"),
            })
        };
        publish_encoding_receipt(&json!({
            "schema": "EncodingAdoptionReceipt.v2",
            "dataset": {
                "dataset_id": dataset_id,
                "content_sha256": dataset_content_sha256,
            },
            "attempt_number": attempt,
            "job_identity": job_id,
            "encoder_identity_sha256": encoder,
            "disposition": disposition,
            "recorded_at_ms": now_ms(),
        }))?;
    }
    Ok(adopted)
}
