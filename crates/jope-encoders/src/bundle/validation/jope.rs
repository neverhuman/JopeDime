use super::*;

pub(in crate::bundle) fn verify_jope_metadata(
    bytes: &[u8],
    manifest: &ModelBundleV1,
    policy_tensors: &[TensorSpec],
    preprocessing_tensors: &[TensorSpec],
) -> Result<()> {
    let metadata: JopePolicyMetadata =
        serde_json::from_slice(bytes).context("decode JOPE policy metadata")?;
    if metadata.model_version != manifest.policy_version
        || metadata.source_checkpoint != manifest.source_checkpoint
        || metadata.source_checkpoint_sha256 != manifest.source_checkpoint_sha256
    {
        bail!("JOPE policy metadata does not match the bound source checkpoint identity");
    }
    let vocabulary_sha256 = sha256_bytes(&canonical_json_bytes(&metadata.vocab)?);
    let grammar_sha256 = sha256_bytes(&canonical_json_bytes(&serde_json::Value::Array(
        metadata.grammar_specs.clone(),
    ))?);
    if vocabulary_sha256 != manifest.vocabulary_canonical_sha256
        || manifest.grammar_sha256 != JOPE_GRAMMAR_SHA256
        || grammar_sha256 != JOPE_GRAMMAR_CONTENT_SHA256
    {
        bail!(
            "JOPE vocabulary or grammar canonical content identity changed: vocab={vocabulary_sha256}, grammar={grammar_sha256}"
        );
    }

    let config = &metadata.config;
    if (
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.ffn_dim,
        config.dataset_dim,
        config.target_dim,
        config.reward_ensemble_k,
        config.embedding_rows,
        config.routing_classes,
        config.vocab_size,
        config.max_seq_len,
        config.padding_idx,
    ) != (512, 6, 4, 2048, 1536, 512, 4, 1359, 14, 1103, 128, 0)
        || config.gelu != "erf"
        || !config.norm_first
        || (config.layernorm_eps - 1e-5).abs() > f64::EPSILON
        || (config.attn_scale - (1.0 / 128.0f64.sqrt())).abs() > 1e-15
        || !config.d_model.is_multiple_of(config.n_heads)
    {
        bail!("JOPE policy model dimensions or numerical grammar changed");
    }
    verify_jope_vocab_and_grammar(&metadata.vocab, &metadata.grammar_specs, config)?;
    verify_jope_policy_tensors(policy_tensors, preprocessing_tensors, config)?;
    let names = policy_tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<HashSet<_>>();
    let dropped = metadata
        .dropped_training_tensors
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let expected_dropped = JOPE_DROPPED_TRAINING_TENSORS
        .into_iter()
        .collect::<HashSet<_>>();
    if dropped.len() != metadata.dropped_training_tensors.len()
        || dropped != expected_dropped
        || dropped.iter().any(|name| names.contains(name))
    {
        bail!("JOPE dropped-training tensor declaration changed or overlaps runtime tensors");
    }
    Ok(())
}

fn verify_jope_vocab_and_grammar(
    vocab: &serde_json::Value,
    grammar_specs: &[serde_json::Value],
    config: &JopePolicyConfig,
) -> Result<()> {
    let vocab = vocab
        .as_object()
        .context("JOPE vocabulary must be an object")?;
    let tokens = vocab
        .get("tokens")
        .and_then(serde_json::Value::as_array)
        .context("JOPE vocabulary has no tokens")?;
    if tokens.len() != config.vocab_size
        || tokens.first().and_then(serde_json::Value::as_str) != Some("PAD")
        || tokens.get(1).and_then(serde_json::Value::as_str) != Some("BOS")
        || tokens.get(2).and_then(serde_json::Value::as_str) != Some("EOS")
        || tokens.iter().any(|token| token.as_str().is_none())
    {
        bail!("JOPE vocabulary token surface does not match model dimensions");
    }
    let token_names = tokens
        .iter()
        .map(|token| token.as_str().expect("checked above"))
        .collect::<HashSet<_>>();
    if token_names.len() != tokens.len() {
        bail!("JOPE vocabulary contains duplicate tokens");
    }
    let engines = vocab
        .get("engines")
        .and_then(serde_json::Value::as_array)
        .context("JOPE vocabulary has no engines")?
        .iter()
        .map(|engine| engine.as_str().context("JOPE engine must be a string"))
        .collect::<Result<BTreeSet<_>>>()?;
    let tasks = vocab
        .get("tasks")
        .and_then(serde_json::Value::as_array)
        .context("JOPE vocabulary has no tasks")?
        .iter()
        .map(|task| task.as_str().context("JOPE task must be a string"))
        .collect::<Result<BTreeSet<_>>>()?;
    if engines != BTreeSet::from(["featmath", "newmath"])
        || tasks != BTreeSet::from(["binary", "regression"])
    {
        bail!("JOPE engine or task vocabulary changed");
    }
    for key in ["engine_ids", "task_ids"] {
        let ids = vocab
            .get(key)
            .and_then(serde_json::Value::as_object)
            .with_context(|| format!("JOPE vocabulary has no {key}"))?;
        if ids.values().any(|id| {
            id.as_u64()
                .is_none_or(|id| id as usize >= config.vocab_size)
        }) {
            bail!("JOPE vocabulary {key} contains an out-of-range token");
        }
    }
    let operations = vocab
        .get("ops")
        .and_then(serde_json::Value::as_array)
        .context("JOPE vocabulary has no operations")?;
    if operations.len() != 91 {
        bail!("JOPE operation vocabulary changed size");
    }
    let mut operation_ids = HashSet::new();
    for operation in operations {
        let operation = operation
            .as_object()
            .context("JOPE operation must be an object")?;
        let id = operation
            .get("token_id")
            .and_then(serde_json::Value::as_u64)
            .context("JOPE operation has no token ID")? as usize;
        let engine = operation
            .get("engine")
            .and_then(serde_json::Value::as_str)
            .context("JOPE operation has no engine")?;
        if id >= config.vocab_size
            || !operation_ids.insert(id)
            || !engines.contains(engine)
            || operation
                .get("name")
                .and_then(serde_json::Value::as_str)
                .is_none()
        {
            bail!("JOPE operation vocabulary is inconsistent");
        }
    }
    if grammar_specs.len() != engines.len() {
        bail!("JOPE grammar registry must contain one grammar per engine");
    }
    let mut grammar_engines = BTreeSet::new();
    for grammar in grammar_specs {
        let grammar = grammar
            .as_object()
            .context("JOPE grammar must be an object")?;
        let engine = grammar
            .get("engine")
            .and_then(serde_json::Value::as_str)
            .context("JOPE grammar has no engine")?;
        let max_transforms = grammar
            .get("max_transforms")
            .and_then(serde_json::Value::as_u64)
            .context("JOPE grammar has no transform limit")?;
        let max_expr_stages = grammar
            .get("max_expr_stages")
            .and_then(serde_json::Value::as_u64)
            .context("JOPE grammar has no expression limit")?;
        let expected_transforms = vocab["max_transforms"]
            .as_object()
            .and_then(|values| values.get(engine))
            .and_then(serde_json::Value::as_u64);
        let expected_expr = vocab["max_expr_stages"]
            .as_object()
            .and_then(|values| values.get(engine))
            .and_then(serde_json::Value::as_u64);
        if !grammar_engines.insert(engine)
            || !engines.contains(engine)
            || expected_transforms != Some(max_transforms)
            || expected_expr != Some(max_expr_stages)
            || grammar
                .get("version")
                .and_then(serde_json::Value::as_str)
                .is_none()
            || grammar
                .get("evaluator_version")
                .and_then(serde_json::Value::as_str)
                .is_none()
            || grammar
                .get("transforms")
                .and_then(serde_json::Value::as_object)
                .is_none()
            || grammar
                .get("heads")
                .and_then(serde_json::Value::as_object)
                .is_none()
            || grammar
                .get("expr")
                .and_then(serde_json::Value::as_object)
                .is_none()
        {
            bail!("JOPE grammar does not match its vocabulary declaration");
        }
    }
    Ok(())
}

pub(super) fn tensor<'a>(
    tensors: &'a [TensorSpec],
    name: &str,
    shape: &[usize],
) -> Result<&'a TensorSpec> {
    typed_tensor(tensors, name, "F32", shape)
}

pub(super) fn typed_tensor<'a>(
    tensors: &'a [TensorSpec],
    name: &str,
    dtype: &str,
    shape: &[usize],
) -> Result<&'a TensorSpec> {
    let tensor = tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .with_context(|| format!("missing required tensor {name}"))?;
    if tensor.dtype != dtype || tensor.shape != shape {
        bail!(
            "tensor {name} has {:?}/{}, expected {shape:?}/{dtype}",
            tensor.shape,
            tensor.dtype
        );
    }
    Ok(tensor)
}

fn verify_jope_policy_tensors(
    tensors: &[TensorSpec],
    preprocessing: &[TensorSpec],
    config: &JopePolicyConfig,
) -> Result<()> {
    if tensors.len() != 123 || tensors.iter().any(|tensor| tensor.dtype != "F32") {
        bail!("JOPE policy tensor count or dtype changed");
    }
    tensor(
        tensors,
        "token_emb.weight",
        &[config.embedding_rows, config.d_model],
    )?;
    tensor(
        tensors,
        "reward_token_emb.weight",
        &[config.embedding_rows, config.d_model],
    )?;
    tensor(
        tensors,
        "pos_emb.weight",
        &[config.max_seq_len + 1, config.d_model],
    )?;
    tensor(tensors, "encoder.0.weight", &[config.d_model, 2048])?;
    tensor(
        tensors,
        "z_proj.weight",
        &[config.d_model, config.target_dim],
    )?;
    tensor(
        tensors,
        "ctx_proj.weight",
        &[config.d_model, config.d_model],
    )?;
    tensor(
        tensors,
        "routing_head.weight",
        &[config.routing_classes, config.d_model],
    )?;
    tensor(tensors, "routing_head.bias", &[config.routing_classes])?;
    tensor(
        tensors,
        "reward_head_ablation.0.weight",
        &[config.d_model, config.d_model],
    )?;
    tensor(tensors, "reward_head_ablation.0.bias", &[config.d_model])?;
    tensor(
        tensors,
        "reward_head_ablation.3.weight",
        &[1, config.d_model],
    )?;
    tensor(tensors, "reward_head_ablation.3.bias", &[1])?;
    tensor(
        tensors,
        "runtime_head.0.weight",
        &[config.d_model, 2 * config.d_model],
    )?;
    tensor(tensors, "runtime_head.0.bias", &[config.d_model])?;
    tensor(tensors, "runtime_head.2.weight", &[1, config.d_model])?;
    tensor(tensors, "runtime_head.2.bias", &[1])?;
    for layer in 0..config.n_layers {
        tensor(
            tensors,
            &format!("decoder.layers.{layer}.self_attn.in_proj_weight"),
            &[3 * config.d_model, config.d_model],
        )?;
        tensor(
            tensors,
            &format!("decoder.layers.{layer}.linear1.weight"),
            &[config.ffn_dim, config.d_model],
        )?;
        tensor(
            tensors,
            &format!("decoder.layers.{layer}.linear2.weight"),
            &[config.d_model, config.ffn_dim],
        )?;
    }
    for head in 0..config.reward_ensemble_k {
        tensor(
            tensors,
            &format!("reward_heads.{head}.0.weight"),
            &[config.d_model, config.dataset_dim],
        )?;
        tensor(
            tensors,
            &format!("reward_heads.{head}.3.weight"),
            &[1, config.d_model],
        )?;
    }
    if preprocessing.len() != 6 {
        bail!("JOPE preprocessing tensor set changed");
    }
    tensor(preprocessing, "foundation_components", &[1024, 2048])?;
    tensor(preprocessing, "foundation_mean", &[1024])?;
    tensor(preprocessing, "foundation_pca_mean", &[2048])?;
    tensor(preprocessing, "foundation_scale", &[1024])?;
    tensor(preprocessing, "hyperion_mean", &[512])?;
    tensor(preprocessing, "hyperion_scale", &[512])?;
    if 1024 + 512 != config.dataset_dim {
        bail!("JOPE preprocessing dimensions do not compose to dataset_dim");
    }
    Ok(())
}
