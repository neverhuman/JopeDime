use super::*;

pub(crate) fn validate_receipt_chain(path: &Path) -> Result<Vec<Value>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut rows = Vec::new();
    let mut previous: Option<String> = None;
    for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let mut value: Value = serde_json::from_str(&line)
            .with_context(|| format!("parse receipt line {}", index + 1))?;
        let object = value
            .as_object_mut()
            .context("receipt row must be a JSON object")?;
        ensure!(
            object.get("schema").and_then(Value::as_str) == Some(RECEIPT_SCHEMA),
            "unsupported receipt schema at line {}",
            index + 1
        );
        ensure!(
            object.get("sequence").and_then(Value::as_u64) == Some((rows.len() + 1) as u64),
            "receipt sequence drift at line {}",
            index + 1
        );
        let stored = object
            .remove("receipt_hash")
            .and_then(|value| value.as_str().map(str::to_owned))
            .context("receipt hash is absent")?;
        ensure!(
            object.get("previous_receipt_hash").and_then(Value::as_str) == previous.as_deref(),
            "receipt chain predecessor drift at line {}",
            index + 1
        );
        let actual = sha256_bytes(&canonical_json_bytes(&value)?);
        ensure!(
            stored == actual,
            "receipt content hash drift at line {}",
            index + 1
        );
        value
            .as_object_mut()
            .expect("object checked")
            .insert("receipt_hash".to_owned(), Value::String(stored.clone()));
        previous = Some(stored);
        rows.push(value);
    }
    Ok(rows)
}

pub(crate) fn durable_write(
    action: &str,
    request: Value,
    summary: Value,
    output: &Path,
    value: &Value,
    receipts: &Path,
    idempotency_key: &str,
) -> Result<Value> {
    ensure!(
        !idempotency_key.trim().is_empty(),
        "idempotency key must not be empty"
    );
    ensure!(
        output != receipts,
        "output and receipt log paths must differ"
    );
    let request_sha256 = sha256_bytes(&canonical_json_bytes(&request)?);
    let _lock = ReceiptLock::acquire(receipts)?;
    let existing = validate_receipt_chain(receipts)?;
    for receipt in &existing {
        if receipt.get("idempotency_key").and_then(Value::as_str) != Some(idempotency_key) {
            continue;
        }
        ensure!(
            receipt.get("action").and_then(Value::as_str) == Some(action)
                && receipt.get("request_sha256").and_then(Value::as_str) == Some(&request_sha256),
            "idempotency key was already used for different GP bridge inputs"
        );
        let expected = receipt
            .get("output_sha256")
            .and_then(Value::as_str)
            .context("idempotent receipt lacks output hash")?;
        ensure!(
            output.is_file() && sha256_file(output)? == expected,
            "idempotent output is absent or changed"
        );
        return Ok(receipt.clone());
    }

    let serialized = canonical_json_bytes(value)?;
    let output_sha256 = sha256_bytes(&serialized);
    if output.exists() {
        let existing_value = read_json(output)?;
        ensure!(
            canonical_json_bytes(&existing_value)? == serialized,
            "refusing to overwrite a nonidentical GP bridge output"
        );
    } else {
        atomic_json(output, value)?;
    }
    ensure!(
        sha256_file(output)? == sha256_bytes(&[serialized.as_slice(), b"\n"].concat()),
        "atomic output verification failed"
    );
    let output_file_sha256 = sha256_file(output)?;
    let previous = existing
        .last()
        .and_then(|row| row.get("receipt_hash"))
        .and_then(Value::as_str);
    let mut receipt = json!({
        "schema": RECEIPT_SCHEMA,
        "sequence": existing.len() + 1,
        "created_at_ms": now_ms(),
        "action": action,
        "idempotency_key": idempotency_key,
        "request_sha256": request_sha256,
        "output": output,
        "output_sha256": output_file_sha256,
        "content_sha256": output_sha256,
        "previous_receipt_hash": previous,
    });
    if !summary.is_null() {
        receipt
            .as_object_mut()
            .expect("JSON object")
            .insert("summary".to_owned(), summary);
    }
    let receipt_hash = sha256_bytes(&canonical_json_bytes(&receipt)?);
    receipt
        .as_object_mut()
        .expect("JSON object")
        .insert("receipt_hash".to_owned(), Value::String(receipt_hash));
    if let Some(parent) = receipts.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(receipts)
        .with_context(|| format!("open receipt log {}", receipts.display()))?;
    file.write_all(&canonical_json_bytes(&receipt)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(receipt)
}

pub(crate) fn proposal_request(
    action: &str,
    args: &ProposalInputs,
    verified: &VerifiedProposals,
) -> Value {
    json!({
        "action": action,
        "checkpoint_pointer": args.checkpoint_pointer,
        "checkpoint_pointer_sha256": verified.checkpoint_pointer_sha256,
        "proposals_pointer": args.proposals_pointer,
        "proposals_pointer_sha256": verified.proposal_pointer_sha256,
        "proposal_file_sha256": verified.proposal_file_sha256,
        "checkpoint_sha256": verified.checkpoint_sha256,
        "task": args.task.as_str(),
        "dataset_hash": args.dataset,
    })
}
