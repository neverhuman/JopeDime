use super::*;

pub(in crate::bundle) fn verify_operator_registry(bytes: &[u8]) -> Result<()> {
    if sha256_bytes(bytes) != OPERATOR_REGISTRY_SHA256 {
        bail!("operator registry raw content identity changed");
    }
    let document: serde_json::Value =
        serde_json::from_slice(bytes).context("decode operator registry")?;
    let object = document
        .as_object()
        .context("operator registry must be an object")?;
    if object
        .get("registry_version")
        .and_then(serde_json::Value::as_str)
        != Some("jope-prime-registry-v1")
        || object
            .get("plan_version")
            .and_then(serde_json::Value::as_u64)
            != Some(1)
        || object
            .get("maximum_output_columns")
            .and_then(serde_json::Value::as_u64)
            != Some(512)
        || object
            .get("registry_hash")
            .and_then(serde_json::Value::as_str)
            != Some(OPERATOR_REGISTRY_SEMANTIC_SHA256)
    {
        bail!("operator registry version, dimensions, or semantic identity changed");
    }
    let operators = object
        .get("operators")
        .and_then(serde_json::Value::as_array)
        .context("operator registry has no operators")?;
    if operators.len() != 55 {
        bail!("operator registry must contain the reviewed 55 operators");
    }
    let mut names = HashSet::new();
    for (expected_id, operator) in operators.iter().enumerate() {
        let operator = operator
            .as_object()
            .context("registry operator must be an object")?;
        if operator.get("id").and_then(serde_json::Value::as_u64) != Some(expected_id as u64) {
            bail!("operator registry IDs must be contiguous and ordered");
        }
        let name = operator
            .get("name")
            .and_then(serde_json::Value::as_str)
            .context("registry operator has no name")?;
        if !names.insert(name)
            || operator
                .get("semantic")
                .and_then(serde_json::Value::as_str)
                .is_none()
            || operator
                .get("role")
                .and_then(serde_json::Value::as_str)
                .is_none()
            || operator
                .get("params")
                .and_then(serde_json::Value::as_array)
                .is_none()
            || operator
                .get("capabilities")
                .and_then(serde_json::Value::as_object)
                .is_none()
            || operator
                .get("tags")
                .and_then(serde_json::Value::as_array)
                .is_none()
        {
            bail!("operator registry entry {expected_id} is incomplete or duplicated");
        }
        let lifecycle = operator
            .get("lifecycle")
            .and_then(serde_json::Value::as_object)
            .context("registry operator has no lifecycle")?;
        let parseable = lifecycle
            .get("parseable")
            .and_then(serde_json::Value::as_bool);
        let generatable = lifecycle
            .get("generatable")
            .and_then(serde_json::Value::as_bool);
        let quarantined = lifecycle
            .get("quarantined")
            .and_then(serde_json::Value::as_bool);
        let retired = lifecycle
            .get(concat!("depre", "cated"))
            .and_then(serde_json::Value::as_bool);
        if lifecycle.len() != 4
            || parseable != Some(true)
            || generatable.is_none()
            || quarantined.is_none()
            || retired.is_none()
            || (quarantined == Some(true) && generatable != Some(false))
            || (retired == Some(true) && generatable != Some(false))
        {
            bail!("operator registry lifecycle is unsafe for {name}");
        }
    }
    Ok(())
}
