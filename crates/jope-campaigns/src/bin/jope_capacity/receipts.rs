use super::*;

pub(super) fn write_receipt(root: &Path, key: &str, decision: &Value) -> Result<Value> {
    let (receipts, idempotency) = receipt_dirs(root);
    fs::create_dir_all(&receipts)?;
    fs::create_dir_all(&idempotency)?;
    let key_sha256 = sha256_bytes(key.as_bytes());
    let receipt = json!({
        "schema": "JopeSelfplayCapacityReceipt.v1",
        "idempotency_key_sha256": key_sha256,
        "decision": decision,
    });
    // atomic_json terminates every document with exactly one newline. Hash the
    // bytes that are actually persisted so filename and on-disk content agree.
    let mut persisted = canonical_json_bytes(&receipt)?;
    persisted.push(b'\n');
    let digest = sha256_bytes(&persisted);
    let receipt_path = receipts.join(format!("{digest}.json"));
    if receipt_path.is_file() {
        if sha256_file(&receipt_path)? != digest {
            bail!("existing capacity receipt content drift");
        }
    } else {
        atomic_json(&receipt_path, &receipt)?;
    }
    atomic_json(
        &idempotency.join(format!("{key_sha256}.json")),
        &json!({
            "schema": "JopeSelfplayCapacityIdempotency.v1",
            "receipt_sha256": digest,
        }),
    )?;
    Ok(receipt)
}
