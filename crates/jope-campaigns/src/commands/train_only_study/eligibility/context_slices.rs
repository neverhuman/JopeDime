use super::*;

pub(crate) fn copy_context_slices(
    source_blob: &Path,
    output_blob: &Path,
    selected: &[(CatalogEntry, RegressionData, String, SotarenaReportHashes)],
) -> Result<(String, BTreeMap<String, EncodingSlice>)> {
    let mut source = File::open(source_blob)?;
    let mut bytes = Vec::with_capacity(selected.len() * CONTEXT_BYTES);
    let mut slices = BTreeMap::new();
    for (slot, (entry, _, _, _)) in selected.iter().enumerate() {
        let source_offset = entry
            .catalog_index
            .checked_mul(CONTEXT_BYTES)
            .context("context offset overflow")?;
        source.seek(SeekFrom::Start(source_offset as u64))?;
        let mut slice = vec![0_u8; CONTEXT_BYTES];
        source.read_exact(&mut slice)?;
        if sha256_bytes(&slice) != entry.context_sha256 {
            bail!("encoding slice drift for {}", entry.dataset_hash);
        }
        let output_offset = slot * CONTEXT_BYTES;
        bytes.extend_from_slice(&slice);
        slices.insert(
            entry.dataset_hash.clone(),
            EncodingSlice {
                blob_path: absolute_path(output_blob)?,
                blob_sha256: String::new(),
                offset_bytes: output_offset,
                length_bytes: CONTEXT_BYTES,
                context_sha256: entry.context_sha256.clone(),
            },
        );
    }
    let digest = sha256_bytes(&bytes);
    immutable_write(output_blob, &bytes)?;
    for slice in slices.values_mut() {
        slice.blob_sha256 = digest.clone();
    }
    Ok((digest, slices))
}
