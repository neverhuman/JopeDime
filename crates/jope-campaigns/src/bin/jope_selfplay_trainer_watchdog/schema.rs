/// Parameter groups that must be present in a policy_v3.3 warm-start checkpoint.
/// These are exactly the 22 tensors the original 130-tensor anchor lacks.
pub(crate) const REQUIRED_CHECKPOINT_GROUPS: &[&str] = &[
    "binary_reward_heads.",
    "binary_routing_head.",
    "binary_stability_head.",
];
/// policy_v3.4 adds the reward branch's own context encoder (8 tensors: two Linear
/// weight/bias pairs and two LayerNorm weight/bias pairs), taking the full model to
/// 160 tensors / 32,240,687 parameters.
pub(crate) const REQUIRED_CHECKPOINT_GROUPS_V34: &[&str] = &[
    "binary_reward_heads.",
    "binary_routing_head.",
    "binary_stability_head.",
    "reward_encoder.",
];

/// One complete-model shape. A checkpoint may only be pinned into a successor spec
/// when it satisfies exactly one of these, because a short checkpoint warm-starts a
/// lobotomised network with no error from PyTorch.
///
/// `marker` makes the choice unambiguous rather than "whichever bar it clears":
/// a checkpoint that CONTAINS `reward_encoder.` is judged against the v3.4 bar
/// (160 tensors), so a truncated v3.4 archive can never squeak through on the v3.3
/// count. Schemas are ordered most-specific first; the last entry is the fallback.
pub(crate) struct CheckpointSchema {
    pub(crate) name: &'static str,
    pub(crate) marker: Option<&'static str>,
    pub(crate) tensors: usize,
    pub(crate) groups: &'static [&'static str],
}

pub(crate) const CHECKPOINT_SCHEMAS: &[CheckpointSchema] = &[
    CheckpointSchema {
        name: "policy_v3.4-split-context",
        marker: Some("reward_encoder."),
        tensors: 160,
        groups: REQUIRED_CHECKPOINT_GROUPS_V34,
    },
    CheckpointSchema {
        name: "policy_v3.3-fused-heads",
        marker: None,
        tensors: 152,
        groups: REQUIRED_CHECKPOINT_GROUPS,
    },
];

/// Fallback schema for archives that carry no schema marker (every checkpoint
/// written before the v3.4 un-sharing).
pub(crate) fn legacy_checkpoint_schema() -> &'static CheckpointSchema {
    CHECKPOINT_SCHEMAS
        .last()
        .expect("checkpoint schema table is never empty")
}

/// Pick the schema a checkpoint is claiming by the marker groups it carries.
pub(crate) fn resolve_checkpoint_schema(present: &[String]) -> &'static CheckpointSchema {
    CHECKPOINT_SCHEMAS
        .iter()
        .find(|schema| {
            schema
                .marker
                .is_some_and(|marker| present.iter().any(|group| group == marker))
        })
        .unwrap_or_else(|| legacy_checkpoint_schema())
}
