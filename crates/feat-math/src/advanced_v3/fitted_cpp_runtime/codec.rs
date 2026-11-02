use super::*;

#[derive(Debug)]
pub(super) struct DecodedCodec {
    pub(super) model: BoundModel,
    pub(super) emitter_revision: u32,
    pub(super) model_library: Vec<u8>,
}

pub(super) fn decode(bytes: &[u8]) -> Result<DecodedCodec> {
    if bytes.len() < HEADER_LEN + CHECKSUM_LEN || bytes.len() > MAX_CODEC_LEN {
        bail!("fitted codec length is outside the permitted range");
    }
    if &bytes[..8] != MAGIC {
        bail!("fitted codec magic mismatch");
    }
    let version = read_u16(bytes, 8)?;
    let header_len = read_u16(bytes, 10)? as usize;
    let flags = read_u32(bytes, 12)?;
    let total = usize::try_from(read_u64(bytes, 16)?).context("codec total length overflow")?;
    let metadata_len = usize::try_from(read_u64(bytes, 24)?).context("metadata length overflow")?;
    let model_len = usize::try_from(read_u64(bytes, 32)?).context("model length overflow")?;
    let source_len = usize::try_from(read_u64(bytes, 40)?).context("source length overflow")?;
    let model_library_len =
        usize::try_from(read_u64(bytes, 48)?).context("model library length overflow")?;
    if version != VERSION || header_len != HEADER_LEN || flags != 0 || total != bytes.len() {
        bail!("fitted codec header is unsupported or inconsistent");
    }
    let payload_len = metadata_len
        .checked_add(model_len)
        .and_then(|value| value.checked_add(source_len))
        .and_then(|value| value.checked_add(model_library_len))
        .context("fitted codec section length overflow")?;
    if HEADER_LEN
        .checked_add(payload_len)
        .and_then(|value| value.checked_add(CHECKSUM_LEN))
        != Some(bytes.len())
    {
        bail!("fitted codec sections do not match the declared total length");
    }
    let checksum_at = bytes.len() - CHECKSUM_LEN;
    if Sha256::digest(&bytes[..checksum_at]).as_slice() != &bytes[checksum_at..] {
        bail!("fitted codec checksum mismatch");
    }
    let metadata_at = HEADER_LEN;
    let model_at = metadata_at + metadata_len;
    let source_at = model_at + model_len;
    let library_at = source_at + source_len;
    let metadata: CodecMetadata = serde_json::from_slice(&bytes[metadata_at..model_at])?;
    if metadata.schema != "jope-dime.fitted-model-codec.v2"
        || !(1..=FITTED_CPP_REVISION).contains(&metadata.emitter_revision)
        || metadata.runtime_flags_sha256 != text_hash(&compile_flags().join(" "))
        || metadata.runtime_source_sha256 != RUNTIME_SOURCE_SHA256
        || metadata.runtime_library_sha256 != RUNTIME_LIBRARY_SHA256
        || metadata.compiler.is_empty()
    {
        bail!("fitted codec metadata or runtime identity mismatch");
    }
    let model: BoundModel = serde_json::from_slice(&bytes[model_at..source_at])?;
    model.validate_binding()?;
    if model.artifact_schema != metadata.artifact_schema
        || model.task_sha256 != metadata.task_sha256
        || model.schema_sha256 != metadata.schema_sha256
        || model.vocabulary_sha256 != metadata.vocabulary_sha256
        || model.genome_sha256 != metadata.genome_sha256
        || model.schema.row_semantics != metadata.row_semantics
    {
        bail!("fitted codec metadata does not bind the serialized model");
    }
    let source = bytes[source_at..library_at].to_vec();
    if text_hash(std::str::from_utf8(&source)?) != metadata.source_sha256 {
        bail!("fitted codec source hash mismatch");
    }
    let model_library = bytes[library_at..checksum_at].to_vec();
    if sha256_hex(&model_library) != metadata.model_library_sha256 {
        bail!("fitted codec model library hash mismatch");
    }
    // The source must correspond to the admitted model, not merely carry a
    // self-consistent hash. Only the current revision's source can be compared
    // with the emitter's; older revisions keep their self-consistent hash, so
    // stored containers still decode, and `load` re-emits them.
    if metadata.emitter_revision == FITTED_CPP_REVISION
        && emit_fitted_cpp17(&model)?.source_sha256 != metadata.source_sha256
    {
        bail!("fitted codec source does not match the current emitter");
    }
    Ok(DecodedCodec {
        model,
        emitter_revision: metadata.emitter_revision,
        model_library,
    })
}

pub(super) fn compile_model_library(source: &[u8]) -> Result<(Vec<u8>, String)> {
    let compiler = crate::cxx::version().context("query fitted C++ compiler version")?;
    if !compiler.contains("13.3") {
        bail!("fitted model compilation requires g++ 13.3, found {compiler:?}");
    }
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "jope-fitted-model-compile-{}-{sequence}",
        std::process::id()
    ));
    fs::create_dir(&directory)
        .with_context(|| format!("create model compilation directory {}", directory.display()))?;
    let library = directory.join("model.so");
    let mut child = Command::new(crate::cxx::compiler())
        .args(compile_flags())
        .args([
            "-shared",
            "-fPIC",
            "-Wl,--build-id=none",
            "-x",
            "c++",
            "-",
            "-o",
        ])
        .arg(&library)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("launch g++ for fitted model image")?;
    child
        .stdin
        .take()
        .context("g++ stdin unavailable")?
        .write_all(source)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        let _ = fs::remove_dir_all(&directory);
        bail!(
            "fitted C++ model compilation failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let bytes = fs::read(&library)?;
    fs::remove_dir_all(&directory)?;
    if bytes.is_empty() {
        bail!("fitted C++ compiler produced an empty model image");
    }
    Ok((bytes, compiler))
}

pub(super) fn parse_numeric(raw: &str) -> Result<Option<f64>> {
    let text = raw.trim();
    if text.is_empty()
        || text.eq_ignore_ascii_case("na")
        || text.eq_ignore_ascii_case("nan")
        || text.eq_ignore_ascii_case("null")
    {
        return Ok(None);
    }
    let value = text
        .parse::<f64>()
        .with_context(|| format!("invalid numeric value '{raw}'"))?;
    Ok(value.is_finite().then_some(value))
}

pub(super) fn category_is_missing(raw: &str) -> bool {
    let text = raw.trim();
    text.is_empty() || text.eq_ignore_ascii_case("null") || text.eq_ignore_ascii_case("na")
}

pub(super) fn put_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(super) fn put_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

pub(super) fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

pub(super) fn read_u16(bytes: &[u8], at: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(at..at + 2)
            .context("truncated fitted codec u16")?
            .try_into()?,
    ))
}

pub(super) fn read_u32(bytes: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(at..at + 4)
            .context("truncated fitted codec u32")?
            .try_into()?,
    ))
}

pub(super) fn read_u64(bytes: &[u8], at: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes
            .get(at..at + 8)
            .context("truncated fitted codec u64")?
            .try_into()?,
    ))
}
