//! Rust counterpart of the checkpoint compatibility rules in
//! `jope.policy.model_v3_checkpoint`.
//!
//! This module parses the checkpoint's architecture metadata and determines which saved
//! tensors the v3 loader may retain. The PyTorch `JopePolicyModelV3` itself remains owned by
//! Python; this crate's Candle `PolicyModel` has a separate tensor format and architecture.
use anyhow::{bail, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const DATASET_BLOCK_RMS: f64 = 1.228194;
pub const TARGET_BLOCK_RMS: f64 = 0.085749;

/// Defaults and recorded architecture values used to reconstruct a policy_v3 checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct Architecture {
    pub dataset_dim: usize,
    pub target_dim: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub reward_heads: usize,
    pub embedding_rows: Option<usize>,
    pub routing_classes: usize,
    pub fused_task_heads: bool,
    pub target_keep_dim: Option<usize>,
    pub split_reward_context: bool,
    pub input_block_norm: bool,
    pub dataset_block_rms: f64,
    pub target_block_rms: f64,
    pub task_head_schema: String,
}

fn field<'a>(state: &'a Value, name: &str) -> Option<&'a Value> {
    state.as_object()?.get(name)
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn integer_value(value: &Value) -> Option<usize> {
    match value {
        Value::Bool(value) => Some(usize::from(*value)),
        Value::Number(value) => value
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .or_else(|| {
                let number = value.as_f64()?;
                (number.is_finite() && number >= 0.0 && number < usize::MAX as f64)
                    .then_some(number.trunc() as usize)
            }),
        Value::String(value) => value.trim().parse::<usize>().ok(),
        _ => None,
    }
}

fn integer(state: &Value, name: &str, default: usize) -> Result<usize> {
    match field(state, name) {
        None => Ok(default),
        Some(value) => integer_value(value)
            .ok_or_else(|| anyhow::anyhow!("checkpoint field {name} is not an integer: {value}")),
    }
}

fn optional_integer(state: &Value, name: &str) -> Result<Option<usize>> {
    match field(state, name) {
        None => Ok(None),
        Some(Value::Null) => Ok(None),
        Some(value) => integer_value(value)
            .map(Some)
            .ok_or_else(|| anyhow::anyhow!("checkpoint field {name} is not an integer: {value}")),
    }
}

fn float(state: &Value, name: &str, default: f64) -> Result<f64> {
    match field(state, name) {
        None => Ok(default),
        Some(Value::Bool(value)) => Ok(if *value { 1.0 } else { 0.0 }),
        Some(Value::String(value)) => value
            .trim()
            .parse::<f64>()
            .map_err(|_| anyhow::anyhow!("checkpoint field {name} is not numeric: {value}")),
        Some(value) => value
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("checkpoint field {name} is not numeric: {value}")),
    }
}

fn python_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        _ => value.to_string(),
    }
}

fn python_repr(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// Validate a saved action-token table against the current append-only vocabulary.
pub fn validate_action_vocab_compatibility(
    saved_tokens: Option<&[String]>,
    current_tokens: &[String],
    embedding_rows: usize,
) -> Result<()> {
    let Some(saved_tokens) = saved_tokens.filter(|tokens| !tokens.is_empty()) else {
        return Ok(());
    };
    if saved_tokens == current_tokens {
        return Ok(());
    }
    if current_tokens.get(..saved_tokens.len()) != Some(saved_tokens) {
        bail!("action vocab drift: checkpoint tokens are not an exact current prefix");
    }
    if embedding_rows < current_tokens.len() {
        bail!("action vocab extension exceeds the checkpoint embedding table");
    }
    Ok(())
}

/// Confirm that a checkpoint has grammar metadata from which its action vocabulary can be
/// reconstructed. Frozen vocabulary view equality is checked separately by
/// [`validate_frozen_action_vocab_views`].
pub fn require_grammar_specs(state: &Value) -> Result<()> {
    if field(state, "frozen_action_vocab").is_some_and(|value| !value.is_null()) {
        return Ok(());
    }
    if field(state, "grammar_specs").is_some_and(truthy) {
        return Ok(());
    }
    bail!("policy_v3 checkpoint missing grammar_specs")
}

/// Preserve the Python loader's exact comparison between frozen grammar parsing views and the
/// checkpoint's recorded grammar/vocabulary JSON.
pub fn validate_frozen_action_vocab_views(
    checkpoint_specs: Option<&Value>,
    reconstructed_specs: &[Value],
    checkpoint_vocab: Option<&Value>,
    reconstructed_vocab: &Value,
) -> Result<()> {
    if checkpoint_specs != Some(&Value::Array(reconstructed_specs.to_vec()))
        || checkpoint_vocab != Some(reconstructed_vocab)
    {
        bail!("frozen action table parsing views changed");
    }
    Ok(())
}

/// Read model dimensions and v1/v2 task-head compatibility exactly as the Python checkpoint
/// loader does, including the v3.3 defaults for metadata added in v3.4.
pub fn architecture_from_checkpoint(state: &Value) -> Result<Architecture> {
    require_grammar_specs(state)?;
    let rows = field(state, "embedding_rows")
        .filter(|value| truthy(value))
        .map(|_| integer(state, "embedding_rows", 0))
        .transpose()?
        .filter(|rows| *rows > 0);
    let task_schema = field(state, "task_head_schema").filter(|value| truthy(value));
    let schema_value = task_schema
        .and_then(|value| value.get("schema"))
        .filter(|value| truthy(value))
        .cloned()
        .unwrap_or_else(|| Value::String("PrimeTaskHeads.v1".into()));
    let schema_name = python_string(&schema_value);
    if !matches!(
        schema_name.as_str(),
        "PrimeTaskHeads.v1" | "PrimeTaskHeads.v2"
    ) {
        bail!(
            "unsupported task_head_schema {}; this build understands PrimeTaskHeads.v1 (identity lanes) and PrimeTaskHeads.v2 (timeseries/multiclass lane aliases)",
            python_repr(&schema_name)
        );
    }
    Ok(Architecture {
        dataset_dim: integer(state, "dataset_dim", 512)?,
        target_dim: integer(state, "target_dim", 512)?,
        d_model: integer(state, "d_model", 256)?,
        n_layers: integer(state, "n_layers", 2)?,
        n_heads: integer(state, "n_heads", 4)?,
        ffn_dim: integer(state, "ffn_dim", 1024)?,
        reward_heads: integer(state, "reward_ensemble_k", 4)?,
        embedding_rows: rows,
        routing_classes: integer(state, "routing_classes", 0)?,
        fused_task_heads: field(state, "fused_task_heads").is_some_and(truthy),
        target_keep_dim: optional_integer(state, "target_keep_dim")?,
        split_reward_context: field(state, "split_reward_context").is_some_and(truthy),
        input_block_norm: field(state, "input_block_norm").is_some_and(truthy),
        dataset_block_rms: float(state, "dataset_block_rms", DATASET_BLOCK_RMS)?,
        target_block_rms: float(state, "target_block_rms", TARGET_BLOCK_RMS)?,
        task_head_schema: schema_name,
    })
}

/// Keep only checkpoint tensors with names and shapes present in the rebuilt architecture.
/// Context encoders cannot be skipped: doing so would silently randomize dataset conditioning.
pub fn compatible_tensor_names(
    current: &BTreeMap<String, Vec<usize>>,
    saved: &BTreeMap<String, Vec<usize>>,
) -> Result<BTreeSet<String>> {
    let compatible: BTreeSet<String> = saved
        .iter()
        .filter_map(|(name, shape)| (current.get(name) == Some(shape)).then(|| name.clone()))
        .collect();
    let context_missing: Vec<&String> = current
        .keys()
        .filter(|name| {
            matches!(
                name.split('.').next(),
                Some("encoder" | "reward_encoder" | "ctx_proj")
            ) && !compatible.contains(*name)
        })
        .collect();
    if !context_missing.is_empty() {
        let missing = context_missing
            .iter()
            .map(|name| python_repr(name))
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "policy_v3 checkpoint cannot supply the context encoder(s) at the rebuilt shape; would re-randomise [{missing}]. Migrate the checkpoint instead of loading it (see train_v3.migrate_context_state)."
        );
    }
    Ok(compatible)
}

/// Validated reconstruction inputs for a v3 model implementation. The caller supplies the
/// vocabulary and shape inventories produced by its model/grammar implementation; this
/// function applies the checkpoint loader's compatibility decisions to them.
#[derive(Clone, Debug, PartialEq)]
pub struct RebuildPlan {
    pub architecture: Architecture,
    pub grammar_specs: Vec<Value>,
    pub frozen_action_vocab: Option<Value>,
    pub compatible_tensor_names: BTreeSet<String>,
}

/// Apply the Python loader's checkpoint decisions before a model backend instantiates weights.
/// Frozen-view reconstruction is supplied by the caller because grammar parsing is owned by its
/// respective engine crate.
pub fn prepare_policy_v3_load(
    state: &Value,
    current_tokens: &[String],
    current_shapes: &BTreeMap<String, Vec<usize>>,
    saved_shapes: &BTreeMap<String, Vec<usize>>,
    frozen_views: Option<(&[Value], &Value)>,
) -> Result<RebuildPlan> {
    if field(state, "model").is_none() {
        bail!("'model'");
    }
    let architecture = architecture_from_checkpoint(state)?;
    let frozen = field(state, "frozen_action_vocab").filter(|value| !value.is_null());
    let grammar_specs = field(state, "grammar_specs")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let action_vocab = field(state, "action_vocab");
    let saved_tokens = action_vocab
        .and_then(|value| value.get("tokens"))
        .and_then(Value::as_array)
        .map(|tokens| {
            tokens
                .iter()
                .map(|token| {
                    token.as_str().map(str::to_owned).ok_or_else(|| {
                        anyhow::anyhow!("checkpoint action_vocab.tokens contains a non-string")
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    validate_action_vocab_compatibility(
        saved_tokens.as_deref(),
        current_tokens,
        architecture.embedding_rows.unwrap_or(0),
    )?;
    if frozen.is_some() {
        let (specs, vocab) = frozen_views
            .ok_or_else(|| anyhow::anyhow!("frozen action table parsing views changed"))?;
        validate_frozen_action_vocab_views(
            field(state, "grammar_specs"),
            specs,
            action_vocab,
            vocab,
        )?;
    }
    let compatible_tensor_names = compatible_tensor_names(current_shapes, saved_shapes)?;
    Ok(RebuildPlan {
        architecture,
        grammar_specs,
        frozen_action_vocab: frozen.cloned(),
        compatible_tensor_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).into()).collect()
    }

    #[test]
    fn action_vocab_accepts_missing_equal_and_sufficient_append_only_tables() {
        let current = strings(&["a", "b", "c"]);
        assert!(validate_action_vocab_compatibility(None, &current, 0).is_ok());
        assert!(validate_action_vocab_compatibility(Some(&current), &current, 0).is_ok());
        assert!(
            validate_action_vocab_compatibility(Some(&strings(&["a", "b"])), &current, 3).is_ok()
        );
    }

    #[test]
    fn action_vocab_rejects_reordering_and_insufficient_embedding_rows() {
        let current = strings(&["a", "c", "b"]);
        let error = validate_action_vocab_compatibility(Some(&strings(&["a", "b"])), &current, 3)
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "action vocab drift: checkpoint tokens are not an exact current prefix"
        );
        let current = strings(&["a", "b", "c"]);
        let error = validate_action_vocab_compatibility(Some(&strings(&["a", "b"])), &current, 2)
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "action vocab extension exceeds the checkpoint embedding table"
        );
    }

    #[test]
    fn frozen_vocabulary_requires_both_serialized_views_to_match() {
        let specs = json!([{"engine":"prime"}]);
        let vocab = json!({"tokens":["a"]});
        assert!(validate_frozen_action_vocab_views(
            Some(&specs),
            specs.as_array().unwrap(),
            Some(&vocab),
            &vocab
        )
        .is_ok());
        assert_eq!(
            validate_frozen_action_vocab_views(
                Some(&json!([])),
                specs.as_array().unwrap(),
                Some(&vocab),
                &vocab
            )
            .unwrap_err()
            .to_string(),
            "frozen action table parsing views changed"
        );
    }

    #[test]
    fn architecture_defaults_keep_historical_v33_dimensions_and_task_heads() {
        let plan = architecture_from_checkpoint(&json!({"grammar_specs":[{}]})).unwrap();
        assert_eq!(plan.dataset_dim, 512);
        assert_eq!(plan.target_dim, 512);
        assert_eq!(
            (plan.d_model, plan.n_layers, plan.n_heads, plan.ffn_dim),
            (256, 2, 4, 1024)
        );
        assert_eq!(plan.reward_heads, 4);
        assert_eq!(plan.embedding_rows, None);
        assert_eq!(plan.dataset_block_rms, DATASET_BLOCK_RMS);
        assert_eq!(plan.target_block_rms, TARGET_BLOCK_RMS);
        assert_eq!(plan.task_head_schema, "PrimeTaskHeads.v1");
    }

    #[test]
    fn task_head_v2_is_accepted_and_unknown_schema_is_refused() {
        let plan = architecture_from_checkpoint(&json!({"grammar_specs":[{}], "task_head_schema":{"schema":"PrimeTaskHeads.v2"}, "embedding_rows": 8, "target_keep_dim": 16, "fused_task_heads": true})).unwrap();
        assert_eq!(plan.task_head_schema, "PrimeTaskHeads.v2");
        assert_eq!(plan.embedding_rows, Some(8));
        assert_eq!(plan.target_keep_dim, Some(16));
        assert!(plan.fused_task_heads);
        let error = architecture_from_checkpoint(
            &json!({"grammar_specs":[{}], "task_head_schema":{"schema":"PrimeTaskHeads.v3"}}),
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "unsupported task_head_schema 'PrimeTaskHeads.v3'; this build understands PrimeTaskHeads.v1 (identity lanes) and PrimeTaskHeads.v2 (timeseries/multiclass lane aliases)");
    }

    #[test]
    fn checkpoint_number_coercions_follow_python_int_float_and_bool_rules() {
        let plan = architecture_from_checkpoint(&json!({
            "grammar_specs":[{}],
            "dataset_dim":16.9,
            "target_dim":"640",
            "dataset_block_rms":true,
            "target_block_rms":"0.25",
            "fused_task_heads":"false"
        }))
        .unwrap();
        assert_eq!(plan.dataset_dim, 16);
        assert_eq!(plan.target_dim, 640);
        assert_eq!(plan.dataset_block_rms, 1.0);
        assert_eq!(plan.target_block_rms, 0.25);
        assert!(plan.fused_task_heads);
    }

    #[test]
    fn shape_mismatch_is_skipped_for_heads_but_never_for_context_encoders() {
        let current = BTreeMap::from([
            ("encoder.0.weight".into(), vec![2, 3]),
            ("reward_heads.0.weight".into(), vec![4, 2]),
        ]);
        let saved = BTreeMap::from([
            ("encoder.0.weight".into(), vec![2, 3]),
            ("reward_heads.0.weight".into(), vec![5, 2]),
            ("future.weight".into(), vec![1]),
        ]);
        let retained = compatible_tensor_names(&current, &saved).unwrap();
        assert_eq!(retained, BTreeSet::from(["encoder.0.weight".to_string()]));
        let saved = BTreeMap::from([("encoder.0.weight".into(), vec![1, 3])]);
        assert_eq!(compatible_tensor_names(&current, &saved).unwrap_err().to_string(), "policy_v3 checkpoint cannot supply the context encoder(s) at the rebuilt shape; would re-randomise ['encoder.0.weight']. Migrate the checkpoint instead of loading it (see train_v3.migrate_context_state).");
    }

    #[test]
    fn prepare_plan_binds_vocab_architecture_and_retained_shapes() {
        let state = json!({
            "grammar_specs": [{"engine":"prime"}],
            "action_vocab": {"tokens":["a","b"]},
            "embedding_rows": 2,
            "model": {}
        });
        let shapes = BTreeMap::from([
            ("encoder.0.weight".into(), vec![2, 3]),
            ("reward_heads.0.weight".into(), vec![4, 2]),
        ]);
        let plan =
            prepare_policy_v3_load(&state, &strings(&["a", "b"]), &shapes, &shapes, None).unwrap();
        assert_eq!(plan.grammar_specs, vec![json!({"engine":"prime"})]);
        assert_eq!(plan.architecture.embedding_rows, Some(2));
        assert_eq!(
            plan.compatible_tensor_names,
            shapes.keys().cloned().collect()
        );
        assert!(plan.frozen_action_vocab.is_none());
    }
}
