#[derive(Deserialize)]
struct ApplianceCatalog {
    schema: String,
    format_version: u16,
    product: String,
    entries: Vec<ApplianceEntry>,
}
#[derive(Deserialize)]
struct ApplianceEntry {
    id: String,
    sha256: String,
    bytes: u64,
}

fn verify_asset_context(
    context: SignedAssetContext<'_>,
    limits: BundleLimits,
    keys: &[VerifyingKey],
) -> Result<BTreeMap<String, AssetIdentity>> {
    let footer = context.footer;
    let u16_at = |n| u16::from_le_bytes(footer[n..n + 2].try_into().expect("fixed footer"));
    let u32_at = |n| u32::from_le_bytes(footer[n..n + 4].try_into().expect("fixed footer"));
    let u64_at = |n| u64::from_le_bytes(footer[n..n + 8].try_into().expect("fixed footer"));
    if &footer[..8] != b"JOPEEXE1"
        || u16_at(8) != 1
        || u16_at(10) != 0
        || u32_at(12) != 128
        || u32_at(56) != 64
        || u32_at(60) != 1
    {
        return Err(invalid("invalid signed appliance footer"));
    }
    let base = u64_at(24);
    let offset = u64_at(32);
    let len = u64_at(40);
    if base == 0
        || base
            .checked_next_multiple_of(4096)
            .is_none_or(|aligned| offset < aligned)
        || len == 0
        || len > limits.manifest_bytes
        || len != context.manifest.len() as u64
    {
        return Err(invalid(
            "signed catalog exceeds its bound or has invalid layout",
        ));
    }
    let end = offset
        .checked_add(len)
        .filter(|n| n.checked_add(192).is_some())
        .ok_or_else(|| invalid("signed catalog layout overflows"))?;
    if u64_at(16) != end || u64_at(48) != end {
        return Err(invalid("signed catalog layout disagrees with footer"));
    }
    if Sha256::digest(context.manifest).as_slice() != &footer[96..128] {
        return Err(invalid("signed catalog SHA-256 mismatch"));
    }
    let key = keys
        .iter()
        .find(|key| {
            key.to_public_key_der()
                .is_ok_and(|der| Sha256::digest(der.as_bytes()).as_slice() == &footer[64..96])
        })
        .ok_or_else(|| invalid("untrusted package signer"))?;
    let signature = Signature::from_slice(context.signature)
        .map_err(|_| invalid("invalid package signature encoding"))?;
    if signature.normalize_s() != signature {
        return Err(invalid("noncanonical package signature"));
    }
    let mut message = Vec::with_capacity(SIGNING_DOMAIN.len() + 128);
    message.extend_from_slice(SIGNING_DOMAIN);
    message.extend_from_slice(footer);
    key.verify(&message, &signature)
        .map_err(|_| invalid("package signature verification failed"))?;
    let catalog: ApplianceCatalog = serde_json::from_slice(context.manifest)
        .map_err(|_| invalid("invalid signed appliance catalog"))?;
    if catalog.schema != "jope.appliance-manifest/v1"
        || catalog.format_version != 1
        || catalog.product != "jopedime"
        || catalog.entries.len() > 4096
    {
        return Err(invalid("unsupported signed appliance catalog"));
    }
    let mut assets = BTreeMap::new();
    let mut previous = None;
    for entry in catalog.entries {
        if previous.as_ref().is_some_and(|p| p >= &entry.id) || !relative_name(&entry.id, true) {
            return Err(invalid("signed assets are not sorted unique names"));
        }
        previous = Some(entry.id.clone());
        assets.insert(
            entry.id,
            AssetIdentity {
                bytes: entry.bytes,
                sha256: parse_sha256(&entry.sha256)?,
            },
        );
    }
    Ok(assets)
}

fn safe_path(path: &str) -> bool {
    relative_name(path, false)
}

fn relative_name(path: &str, signed_id: bool) -> bool {
    !path.is_empty()
        && path.len() <= 256
        && path.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(b, b'/' | b'.' | b'_' | b'-')
                || (signed_id && matches!(b, b'+' | b':'))
        })
        && path
            .split('/')
            .all(|p| !p.is_empty() && p != "." && p != "..")
}

fn validate_tensors(
    bytes: &[u8],
    expected: &BTreeMap<String, TensorSpec>,
    limits: BundleLimits,
) -> Result<()> {
    check_tensor_header(bytes, limits)?;
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("checked prefix")) as usize;
    serde_json::from_slice::<UniqueTensorNames>(&bytes[8..8 + header_len])
        .map_err(|_| invalid("invalid or duplicate tensor header names"))?;
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|_| invalid("invalid SafeTensors header or offsets"))?;
    if tensors.len() != expected.len() {
        return Err(invalid("tensor inventory length mismatch"));
    }
    for (name, tensor) in tensors.iter() {
        let spec = expected
            .get(name)
            .ok_or_else(|| invalid("unexpected tensor name"))?;
        if tensor.shape() != spec.shape || tensor.dtype() != spec.dtype {
            return Err(invalid("tensor shape or dtype mismatch"));
        }
        check_finite(tensor.data(), spec.dtype)?;
    }
    Ok(())
}

// SafeTensors' map deserializer collapses repeated top-level tensor names.
// Refuse ambiguity before its validated offset/dtype parser sees the header.
struct UniqueTensorNames;
impl<'de> Deserialize<'de> for UniqueTensorNames {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct Names;
        impl<'de> serde::de::Visitor<'de> for Names {
            type Value = UniqueTensorNames;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a tensor header with unique names")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut names = BTreeSet::new();
                while let Some(name) = map.next_key::<String>()? {
                    if !names.insert(name) {
                        return Err(serde::de::Error::custom("duplicate tensor name"));
                    }
                    map.next_value::<serde::de::IgnoredAny>()?;
                }
                Ok(UniqueTensorNames)
            }
        }
        deserializer.deserialize_map(Names)
    }
}

fn check_finite(bytes: &[u8], dtype: Dtype) -> Result<()> {
    let (width, exponent) = match dtype {
        Dtype::F64 => (8, 0x7ff0_0000_0000_0000u64),
        Dtype::F32 => (4, 0x7f80_0000),
        Dtype::F16 => (2, 0x7c00),
        Dtype::BF16 => (2, 0x7f80),
        Dtype::BOOL
        | Dtype::U8
        | Dtype::I8
        | Dtype::I16
        | Dtype::U16
        | Dtype::I32
        | Dtype::U32
        | Dtype::I64
        | Dtype::U64 => return Ok(()),
        _ => return Err(invalid("unsupported tensor dtype")),
    };
    for value in bytes.chunks_exact(width) {
        let mut word = [0u8; 8];
        word[..width].copy_from_slice(value);
        if u64::from_le_bytes(word) & exponent == exponent {
            return Err(invalid("nonfinite tensor parameter or buffer"));
        }
    }
    Ok(())
}
