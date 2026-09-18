use super::*;

pub(super) fn verify_file_bytes(
    file: &ModelFileV1,
    bytes: &[u8],
) -> Result<Option<Vec<TensorSpec>>> {
    if bytes.len() as u64 != file.bytes {
        bail!(
            "{} has {} bytes, expected {}",
            file.path,
            bytes.len(),
            file.bytes
        );
    }
    if sha256_bytes(bytes) != file.sha256 {
        bail!("{} sha256 mismatch", file.path);
    }
    if Path::new(&file.path)
        .extension()
        .and_then(|value| value.to_str())
        != Some("safetensors")
    {
        return Ok(None);
    }
    let (inventory, inventory_sha256) = safetensors_inventory(bytes, &file.path)?;
    if inventory.len() != file.tensor_count
        || Some(inventory_sha256.as_str()) != file.tensor_inventory_sha256.as_deref()
    {
        bail!("{} tensor inventory mismatch", file.path);
    }
    Ok(Some(inventory))
}

pub(super) fn canonical_json_bytes(value: &serde_json::Value) -> Result<Vec<u8>> {
    fn write(value: &serde_json::Value, output: &mut Vec<u8>) -> Result<()> {
        match value {
            serde_json::Value::Null => output.extend_from_slice(b"null"),
            serde_json::Value::Bool(value) => {
                output.extend_from_slice(if *value { b"true" } else { b"false" })
            }
            serde_json::Value::Number(value) => {
                output.extend_from_slice(value.to_string().as_bytes())
            }
            serde_json::Value::String(value) => {
                output.extend_from_slice(serde_json::to_string(value)?.as_bytes())
            }
            serde_json::Value::Array(values) => {
                output.push(b'[');
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        output.push(b',');
                    }
                    write(value, output)?;
                }
                output.push(b']');
            }
            serde_json::Value::Object(values) => {
                output.push(b'{');
                let mut keys = values.keys().collect::<Vec<_>>();
                keys.sort_unstable();
                for (index, key) in keys.into_iter().enumerate() {
                    if index > 0 {
                        output.push(b',');
                    }
                    output.extend_from_slice(serde_json::to_string(key)?.as_bytes());
                    output.push(b':');
                    write(&values[key], output)?;
                }
                output.push(b'}');
            }
        }
        Ok(())
    }
    let mut output = Vec::new();
    write(value, &mut output)?;
    Ok(output)
}

pub(super) fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(super) fn valid_source_revision(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(valid_sha256)
}

pub(super) fn valid_source_uri(value: &str) -> bool {
    value
        .strip_prefix("urn:jain:model-source:sha256:")
        .is_some_and(valid_sha256)
}

pub(super) fn validate_license(license: &ModelLicenseV1, file: &ModelFileV1) -> Result<()> {
    let path = &file.path;
    if license.schema != "jain.model-license/v1"
        || license.component_id.is_empty()
        || license.component_id.len() > 256
        || license.component_kind
            != if file.kind == "jope_operator_registry" {
                "operator_registry"
            } else {
                "model"
            }
        || license.sha256 != file.sha256
        || license.spdx_expression.is_empty()
        || license.spdx_expression.len() > 256
        || !matches!(
            license.license_class.as_str(),
            "open" | "eligible_non_open" | "restricted"
        )
        || !valid_source_revision(&license.source_revision)
        || !valid_source_uri(&license.source_uri)
        || !matches!(
            license.redistribution.as_str(),
            "permitted" | "restricted" | "prohibited"
        )
        || license.model_lineage.is_empty()
        || license.model_lineage.len() > 2048
        || license.notice_paths.is_empty()
    {
        bail!("bundle file {path} has incomplete license metadata");
    }
    let mut notices = HashSet::with_capacity(license.notice_paths.len());
    for notice in &license.notice_paths {
        validate_relative_path(notice)?;
        if !notices.insert(notice) {
            bail!("bundle file {path} has duplicate notice path {notice}");
        }
    }

    let policy = &license.policy_eligibility;
    match license.license_class.as_str() {
        "open" => {
            if !policy.eligible
                || !policy.open_only
                || license.redistribution != "permitted"
                || license.spdx_expression.starts_with("LicenseRef-")
            {
                bail!("bundle file {path} has inconsistent open license metadata");
            }
        }
        "eligible_non_open" => {
            if !policy.eligible || policy.open_only || license.redistribution == "prohibited" {
                bail!("bundle file {path} has inconsistent eligible license metadata");
            }
        }
        "restricted" => {
            if policy.eligible || policy.open_only {
                bail!("bundle file {path} has inconsistent restricted license metadata");
            }
        }
        _ => unreachable!("license class checked above"),
    }
    if license.redistribution == "prohibited" && policy.eligible {
        bail!("bundle file {path} cannot be eligible when redistribution is prohibited");
    }
    Ok(())
}

pub(super) fn safetensors_inventory(
    bytes: &[u8],
    label: &str,
) -> Result<(Vec<TensorSpec>, String)> {
    if bytes.len() < 8 {
        bail!("invalid safetensors header for {label}");
    }
    let mut length_bytes = [0u8; 8];
    length_bytes.copy_from_slice(&bytes[..8]);
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > 64 * 1024 * 1024 || header_len as usize + 8 > bytes.len() {
        bail!("invalid safetensors header length for {label}");
    }
    let header_end = 8 + header_len as usize;
    let values =
        serde_json::from_slice::<serde_json::Map<String, serde_json::Value>>(&bytes[8..header_end])
            .with_context(|| format!("decode safetensors header from {label}"))?;
    let payload_len = bytes.len() - header_end;
    let mut names = values
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .collect::<Vec<_>>();
    names.sort_unstable();
    let mut digest = Sha256::new();
    let mut inventory = Vec::with_capacity(names.len());
    let mut intervals = Vec::with_capacity(names.len());
    for name in &names {
        let entry = values
            .get(*name)
            .expect("tensor name came from this safetensors header")
            .as_object()
            .with_context(|| format!("invalid tensor {name} in {label}"))?;
        let dtype = entry
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("tensor {name} has no dtype"))?;
        let shape_values = entry
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .with_context(|| format!("tensor {name} has no shape"))?;
        let shape = shape_values
            .iter()
            .map(|dimension| {
                dimension
                    .as_u64()
                    .and_then(|dimension| usize::try_from(dimension).ok())
                    .with_context(|| format!("tensor {name} has an invalid dimension"))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = entry
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .with_context(|| format!("tensor {name} has invalid offsets"))?;
        let start = offsets[0]
            .as_u64()
            .with_context(|| format!("tensor {name} has invalid start offset"))?;
        let end = offsets[1]
            .as_u64()
            .with_context(|| format!("tensor {name} has invalid end offset"))?;
        if start > end || end > payload_len as u64 {
            bail!("tensor {name} escapes the safetensors payload");
        }
        let bytes_per_element = match dtype {
            "BOOL" | "I8" | "U8" => 1usize,
            "F16" | "BF16" | "I16" | "U16" => 2,
            "F32" | "I32" | "U32" => 4,
            "F64" | "I64" | "U64" => 8,
            _ => bail!("tensor {name} has unsupported dtype {dtype}"),
        };
        let expected_bytes = shape
            .iter()
            .try_fold(bytes_per_element, |size, dimension| {
                size.checked_mul(*dimension)
                    .context("safetensors tensor byte size overflow")
            })?;
        if end - start != expected_bytes as u64 {
            bail!("tensor {name} byte range does not match its dtype and shape");
        }
        intervals.push((start, end, name.as_str()));
        digest.update(name.as_bytes());
        digest.update([0]);
        digest.update(dtype.as_bytes());
        digest.update([0]);
        for (index, dimension) in shape.iter().enumerate() {
            if index > 0 {
                digest.update(b",");
            }
            digest.update(dimension.to_string().as_bytes());
        }
        digest.update(b"\n");
        inventory.push(TensorSpec {
            name: (*name).clone(),
            dtype: dtype.to_owned(),
            shape,
        });
    }
    intervals.sort_unstable_by_key(|interval| interval.0);
    let mut cursor = 0u64;
    for (start, end, name) in intervals {
        if start != cursor {
            bail!("tensor {name} leaves a gap or overlap in the safetensors payload");
        }
        cursor = end;
    }
    if cursor != payload_len as u64 {
        bail!("safetensors payload for {label} contains unbound trailing bytes");
    }
    Ok((inventory, format!("{:x}", digest.finalize())))
}

pub(super) fn validate_relative_path(path: &str) -> Result<()> {
    let path = Path::new(path);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::CurDir
                    | Component::ParentDir
                    | Component::RootDir
                    | Component::Prefix(_)
            )
        })
        || path.to_string_lossy().contains('\\')
    {
        bail!("model bundle paths must be relative and must not escape the bundle: {path:?}");
    }
    if path.as_os_str().is_empty() {
        bail!("empty model bundle path");
    }
    Ok(())
}

pub(super) fn sha256_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
