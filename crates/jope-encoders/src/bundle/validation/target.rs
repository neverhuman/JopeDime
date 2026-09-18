use super::jope::{tensor, typed_tensor};
use super::*;

pub(in crate::bundle) fn verify_target_metadata(
    bytes: &[u8],
    manifest: &ModelBundleV1,
    inventories: BTreeMap<String, Vec<TensorSpec>>,
) -> Result<TargetEncoderMetadataV1> {
    let metadata: TargetEncoderMetadataV1 =
        serde_json::from_slice(bytes).context("decode target encoder metadata")?;
    let identity = manifest
        .encoders
        .get("target")
        .context("model bundle has no target encoder")?;
    let artifact = manifest
        .files
        .iter()
        .find(|file| file.path == identity.artifact)
        .context("target artifact is not listed")?;
    let metadata_file = manifest
        .files
        .iter()
        .find(|file| file.path == TARGET_METADATA)
        .context("target metadata file is not listed")?;
    if metadata.schema != "jain.target-encoder/v1"
        || Path::new(&identity.artifact)
            .file_name()
            .and_then(|name| name.to_str())
            != Some(metadata.artifact.as_str())
        || metadata.artifact_bytes != artifact.bytes
        || metadata.artifact_sha256 != artifact.sha256
        || metadata.source_revision != identity.source_revision
        || metadata.source_uri != identity.source_uri
        || metadata.tensor_inventory_algorithm != manifest.tensor_inventory_algorithm
        || metadata.tensor_inventory_sha256
            != artifact
                .tensor_inventory_sha256
                .as_deref()
                .context("target artifact has no inventory identity")?
        || metadata.tensor_count != artifact.tensor_count
    {
        bail!("target metadata does not bind its artifact, source, and inventory");
    }
    if metadata_file.license.source_revision != metadata.source_revision
        || metadata_file.license.source_uri != metadata.source_uri
        || metadata.license.sha256 != artifact.sha256
        || metadata.license.source_revision != artifact.license.source_revision
        || metadata.license.source_uri != artifact.license.source_uri
        || metadata.license.license_class != artifact.license.license_class
        || metadata.license.redistribution != artifact.license.redistribution
        || metadata.license.policy_eligibility != artifact.license.policy_eligibility
        || metadata.license.notice_paths != artifact.license.notice_paths
    {
        bail!("target metadata license and artifact license are not semantically bound");
    }
    if metadata.seed != 17
        || metadata.input_preprocessing.binary != "threshold_ge_0.5"
        || metadata.input_preprocessing.regression != "clip_closed_interval_0_1"
        || metadata.input_preprocessing.non_finite != "drop"
        || metadata.input_preprocessing.empty_after_filter != "reject"
        || metadata.input_preprocessing.dtype != "float32"
        || metadata.input_preprocessing.signature_views != 2
        || metadata.input_preprocessing.window_size != 32
        || metadata.input_preprocessing.neural_seed
            != "blake2b-64-low63('encode|seed|length|window_size|stride') then numpy PCG64"
        || metadata.input_preprocessing.signature_seed
            != "blake2b-64-low63('fusion-signature-inference|seed|task_id|original_length|view_index|window_size') then numpy PCG64"
        || metadata.task_support != ["binary", "regression"]
    {
        bail!("target seed, clipping, finite filtering, or sampling contract changed");
    }
    if metadata.vector_dim != 512
        || metadata.signature_dim != 2344
        || metadata.classical_component_dim != 384
        || metadata.actual_svd_dim != 8
        || metadata.feature_names.len() != 586
        || metadata.signature_dim != metadata.feature_names.len() * 4
        || metadata.model_config.attention_heads != 8
        || metadata.model_config.attention_layers != 4
        || metadata.model_config.embedding_dim != 64
        || metadata.model_config.fourier_frequencies != 4
        || metadata.model_config.hidden_dim != 512
        || metadata.model_config.histogram_bins != 16
        || metadata.model_config.rbf_bins != 16
        || !metadata.model_config.rich_stats
        || metadata.model_config.token_dim != 512
    {
        bail!("target declared dimensions do not match the frozen runtime");
    }
    let feature_names = metadata.feature_names.iter().collect::<HashSet<_>>();
    if feature_names.len() != metadata.feature_names.len() {
        bail!("target feature names are not unique");
    }
    let mut feature_digest = Sha256::new();
    for name in &metadata.feature_names {
        feature_digest.update(name.as_bytes());
        feature_digest.update(b"\n");
    }
    if format!("{:x}", feature_digest.finalize())
        != "ba969d2c5c9a50691187aa9191c8b99a0259f0fddecd9a612cd1406f9920d036"
    {
        bail!("target feature-name identity changed");
    }
    if metadata.normalization.blocks != "each block nan_to_num_then_l2; zero norm maps to all-zero"
        || metadata.normalization.final_value
            != "concatenate_then_truncate_or_zero_pad_to_512_then_l2"
        || metadata.normalization.neural_window_pool != "arithmetic_mean_then_l2"
        || (metadata.normalization.epsilon - 1e-12).abs() > f64::EPSILON
    {
        bail!("target nan_to_num and normalization contract changed");
    }
    let expected_blocks = [
        ("neural_best", 0, 64, 64, None),
        ("neural_final", 64, 128, 64, None),
        ("classical_svd", 128, 512, 8, Some(384)),
    ];
    if metadata.block_layout.len() != expected_blocks.len()
        || metadata.block_layout.iter().zip(expected_blocks).any(
            |(block, (name, start, end, source_dim, padded_dim))| {
                block.name != name
                    || block.start != start
                    || block.end != end
                    || block.source_dim != source_dim
                    || block.padded_dim != padded_dim
                    || block.weight != 1.0
            },
        )
    {
        bail!("target block layout does not fill the declared 512 dimensions");
    }
    let actual_tensors = inventories
        .get(&identity.artifact)
        .context("target artifact inventory was not parsed")?;
    if actual_tensors != &metadata.tensors {
        bail!("target metadata tensor declarations do not match actual tensor shapes");
    }
    typed_tensor(
        actual_tensors,
        "vectorizer.scaler.mean",
        "F64",
        &[metadata.signature_dim],
    )?;
    typed_tensor(
        actual_tensors,
        "vectorizer.scaler.scale",
        "F64",
        &[metadata.signature_dim],
    )?;
    tensor(
        actual_tensors,
        "vectorizer.reducer.components",
        &[metadata.actual_svd_dim, metadata.signature_dim],
    )?;
    for prefix in ["neural_best", "neural_final"] {
        tensor(
            actual_tensors,
            &format!("{prefix}.fusion.5.weight"),
            &[
                metadata.model_config.embedding_dim,
                metadata.model_config.hidden_dim,
            ],
        )?;
        tensor(
            actual_tensors,
            &format!("{prefix}.set_blocks.0.attn.in_proj_weight"),
            &[
                3 * metadata.model_config.token_dim,
                metadata.model_config.token_dim,
            ],
        )?;
    }
    if metadata.source_bundle.format != "target_maker_fused_bundle"
        || metadata.source_bundle.version != 1
        || metadata.source_bundle.components.len() != 7
        || metadata.source_bundle.oracle_files.len() != 5
    {
        bail!("target source bundle declaration changed");
    }
    let fit_records = metadata.fit_records.iter().collect::<HashSet<_>>();
    if metadata.source_epochs.best != 17
        || metadata.source_epochs.final_value != 19
        || metadata.fit_records.len() != 8
        || fit_records.len() != metadata.fit_records.len()
        || metadata.fit_records.iter().any(|record| {
            record.len() != 32
                || !record
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
    {
        bail!("target fit-record or source-epoch provenance changed");
    }
    let mut source_paths = HashSet::new();
    for source in metadata
        .source_bundle
        .components
        .iter()
        .chain(&metadata.source_bundle.oracle_files)
    {
        validate_relative_path(&source.path)?;
        if !valid_sha256(&source.sha256) || !source_paths.insert(source.path.as_str()) {
            bail!("target source bundle has an invalid or duplicate source identity");
        }
    }
    let oracle = metadata
        .source_bundle
        .oracle_files
        .iter()
        .find(|file| file.path == "target_maker/fused.py")
        .context("target source bundle does not identify the fused Python oracle")?;
    if oracle.sha256 != identity.recipe_sha256
        || identity.recipe_sha256
            != "fee6643c70e475374576b74cc54ce1e94b614de35a4fad53f5a7b1c170bfa040"
    {
        bail!("target oracle implementation is not bound to the encoder recipe");
    }
    Ok(metadata)
}
