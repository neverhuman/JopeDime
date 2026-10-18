//! Native-loop protocol v2 (`jope.native_loop.v2`): a legacy parent continues as a weight
//! warm start (owner decision D12 of 2026-09-26,
//! `docs/internal/core-ip/work-orders/OWNER-DECISIONS.md`).
//!
//! The trainer's pre-CUDA guard refuses an optimizer carry from a parent without an objective
//! identity, such as the accepted step-60000 checkpoint. A `jope.native_loop.v1` run's first
//! block is exactly such a carry, so it is refused. A v2 run instead starts its first block as
//! a weight warm start:
//! - **Fresh optimizer.** Nothing of the parent's optimizer is carried. Its step counter
//!   restarts at zero, so a v2 run's optimizer windows are `0 → 3000` (the warm start) and
//!   `3000 → 6000` (an ordinary optimizer carry from the warm start's own checkpoint, which has
//!   an objective identity). The parent's weight position is kept explicitly as `parent_step`.
//! - **Parent weights bound.** The block's request declares `optimizer: reset` and its
//!   `parent_step`. Its replay policy carries a [`WARM_START_SCHEMA`] binding naming the
//!   parent checkpoint, the parent weights digest (`model_sha256`) and the parent step. The
//!   trained checkpoint and the completion receipt must name that same binding.
//! - **v1 is unchanged.** Its requests, replay policies, inputs and results keep their bytes and
//!   meaning, and a v1 block from a legacy parent is still an optimizer carry, still refused.
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};

/// The fixed two-block optimizer-carry protocol.
pub const LOOP_V1: &str = "jope.native_loop.v1";
/// The legacy-parent weight-warm-start protocol.
pub const LOOP_V2: &str = "jope.native_loop.v2";
/// An explicitly verified descendant carries its own optimizer epoch.
pub const LOOP_V3: &str = "jope.native_loop.v3";
pub const COMPLETION_V3: &str = "jope.native_loop_completion.v3";
/// The binding a warm start's replay policy, trained checkpoint and completion receipt name.
pub const WARM_START_SCHEMA: &str = "jope.native_weight_warm_start.v1";
/// A v2 run's completion: its optimizer window is the fresh optimizer's.
pub const COMPLETION_V2: &str = "jope.native_loop_completion.v2";
/// What the trainer records for an optimizer that starts from nothing
/// (`training::rank_impl` warm-start initialization).
pub const FRESH_OPTIMIZER: &str = "fresh FP32 AdamW";
/// Optimizer updates per training block.
pub const BLOCK_UPDATES: u64 = 3000;

/// The optimizer window of one training block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub initial: u64,
    pub target: u64,
    /// The parent's step when this block is a weight warm start.
    pub warm_start_parent_step: Option<u64>,
}

/// The optimizer window of `generation`'s training block. `base` is the run's configured
/// `initial_optimizer_step`, the accepted parent's step.
pub fn window(schema: &str, base: u64, generation: u64) -> Result<Window> {
    let offset = generation
        .checked_mul(BLOCK_UPDATES)
        .context("optimizer window overflow")?;
    let (initial, warm_start_parent_step) = match schema {
        LOOP_V1 | LOOP_V3 => (
            base.checked_add(offset)
                .context("optimizer window overflow")?,
            None,
        ),
        LOOP_V2 => (offset, (generation == 0).then_some(base)),
        other => bail!("unknown native loop schema {other}"),
    };
    Ok(Window {
        initial,
        target: initial
            .checked_add(BLOCK_UPDATES)
            .context("optimizer window overflow")?,
        warm_start_parent_step,
    })
}

/// A training request's parameters. An optimizer-carry block has exactly the historical v1
/// fields; a warm start adds `optimizer: reset` and its `parent_step`.
pub fn train_parameters(
    checkpoint: &impl Serialize,
    corpus: &impl Serialize,
    window: &Window,
) -> Result<Value> {
    let mut parameters = json!({"checkpoint":serde_json::to_value(checkpoint)?,"corpus":serde_json::to_value(corpus)?,"initial_step":window.initial,"requested_increment":3000,"target_step":window.target,"global_batch":2048});
    if let Some(step) = window.warm_start_parent_step {
        parameters["optimizer"] = json!("reset");
        parameters["parent_step"] = json!(step);
    }
    Ok(parameters)
}

const CARRY_FIELDS: [&str; 6] = [
    "checkpoint",
    "corpus",
    "initial_step",
    "requested_increment",
    "target_step",
    "global_batch",
];

/// A training request's optimizer mode against its parent `state.json`: `None` for an
/// optimizer carry (the v1 fields, unchanged), or the parent step of a weight warm start.
/// Unknown fields, an explicit or unknown optimizer mode, and a warm start that does not
/// restart the optimizer at step zero are refused.
pub fn request_mode(parameters: &Value, parent_state: &Value) -> Result<Option<u64>> {
    let fields = parameters
        .as_object()
        .context("training parameters must be an object")?;
    let warm = fields.contains_key("optimizer") || fields.contains_key("parent_step");
    ensure!(
        fields.keys().all(|k| CARRY_FIELDS.contains(&k.as_str())
            || (warm && matches!(k.as_str(), "optimizer" | "parent_step"))),
        "training request carries a field outside its optimizer window"
    );
    if !warm {
        return Ok(None);
    }
    ensure!(
        parameters["optimizer"] == "reset",
        "a training request's optimizer mode is either implicit (carry) or reset"
    );
    let step = parameters["parent_step"]
        .as_u64()
        .context("a weight warm start names its parent step")?;
    ensure!(
        parent_state["step"].as_u64() == Some(step),
        "weight warm start parent step differs from the parent checkpoint"
    );
    ensure!(
        parameters["initial_step"] == 0 && parameters["target_step"] == BLOCK_UPDATES,
        "a weight warm start restarts the optimizer at step zero; it continues no optimizer step"
    );
    Ok(Some(step))
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The binding a warm start from this parent carries: parent checkpoint, parent weights
/// digest and parent step, with the optimizer reset and its step counter at zero.
pub fn binding(parent_checkpoint_sha256: &str, parent_state: &Value) -> Result<Value> {
    let model = parent_state["model_sha256"]
        .as_str()
        .filter(|s| is_digest(s))
        .context("parent weights digest absent")?;
    let step = parent_state["step"]
        .as_u64()
        .context("parent optimizer step absent")?;
    ensure!(
        is_digest(parent_checkpoint_sha256),
        "parent checkpoint digest malformed"
    );
    Ok(
        json!({"schema":WARM_START_SCHEMA,"parent_checkpoint_sha256":parent_checkpoint_sha256,
        "parent_model_sha256":model,"parent_step":step,"optimizer":"reset","optimizer_step_base":0}),
    )
}

/// The warm-start replay policy: the optimizer-carry policy the loop derives for the same
/// request, with the optimizer reset and the parent weights bound.
pub fn warm_start_policy(mut carry: Value, binding: Value) -> Value {
    carry["optimizer"] = json!("reset");
    carry["warm_start"] = binding;
    carry
}

/// A replay policy's optimizer mode against its parent: `None` for an optimizer carry, which
/// carries no warm-start binding, or the binding of a weight warm start, which must be exactly
/// this parent's and must have been frozen for a warm-start window. Any mixture is refused.
pub fn policy_mode(
    policy: &Value,
    parent_checkpoint_sha256: &str,
    parent_state: &Value,
) -> Result<Option<Value>> {
    match policy["optimizer"].as_str() {
        Some("carry") => {
            ensure!(
                policy.get("warm_start").is_none(),
                "an optimizer-carry replay policy carries no weight-warm-start binding"
            );
            Ok(None)
        }
        Some("reset") => {
            let bound = policy
                .get("warm_start")
                .context("a weight-warm-start replay policy must bind its parent weights")?;
            ensure!(
                *bound == binding(parent_checkpoint_sha256, parent_state)?,
                "weight warm start does not bind exactly this parent's weights"
            );
            let original = &policy["eligibility_evidence"]["original_window"];
            ensure!(
                original["optimizer"] == "reset"
                    && original["parent_step"] == bound["parent_step"]
                    && original["initial_step"] == 0,
                "weight-warm-start replay policy was not frozen for a warm-start window"
            );
            Ok(Some(bound.clone()))
        }
        _ => bail!("unknown replay/optimizer mode"),
    }
}

/// A trained checkpoint of a weight warm start: it names the bound parent weights and step and
/// its replay policy, started a fresh optimizer, and continues no optimizer step.
pub fn check_trained(
    state: &Value,
    binding: &Value,
    parent_checkpoint_sha256: &str,
    policy_sha256: &str,
) -> Result<()> {
    let init = &state["initialization"];
    ensure!(
        init["kind"] == "warm_start"
            && init["state_sha256"] == parent_checkpoint_sha256
            && init["model_sha256"] == binding["parent_model_sha256"]
            && init["parent_step"] == binding["parent_step"]
            && init["replay_policy_sha256"] == policy_sha256,
        "weight warm start checkpoint does not name its bound parent weights"
    );
    ensure!(
        init["optimizer"] == FRESH_OPTIMIZER
            && init.get("optimizer_parent_step").is_none_or(Value::is_null)
            && init["continuation_mode"] == "weight_warm_start"
            && init["continuity"]["mode"] == "weight_warm_start"
            && init["continuity"]["optimizer"] == "reset",
        "a weight warm start carries no optimizer state and continues no optimizer step"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACCEPTED_STATE: &[u8] = include_bytes!(
        "../../../../docs/internal/asm-manual/data/training-history/accepted60000-state.json"
    );
    const ACCEPTED_STATE_SHA256: &str =
        "8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6";
    const ACCEPTED_MODEL_SHA256: &str =
        "053df4f53b84d2ebe7c14fbf907a149760af3d11de4a57b5ff89be8c2d3e8129";

    fn accepted() -> Value {
        assert_eq!(crate::manifest::hash(ACCEPTED_STATE), ACCEPTED_STATE_SHA256);
        serde_json::from_slice(ACCEPTED_STATE).unwrap()
    }

    fn artifact(name: &str) -> Value {
        json!({"path":format!("{name}/state.json"),"sha256":"a".repeat(64)})
    }

    #[test]
    fn v1_windows_and_request_bytes_are_unchanged() {
        for (generation, initial) in [(0, 60000), (1, 63000)] {
            let w = window(LOOP_V1, 60000, generation).unwrap();
            assert_eq!(
                w,
                Window {
                    initial,
                    target: initial + 3000,
                    warm_start_parent_step: None
                }
            );
            // The exact literal `Loop::tick` froze before v2 existed.
            let (checkpoint, corpus) = (artifact("parent"), artifact("corpus"));
            let historical = json!({"checkpoint":checkpoint,"corpus":corpus,"initial_step":60000+generation*3000,"requested_increment":3000,"target_step":60000+(generation+1)*3000,"global_batch":2048});
            let now = train_parameters(&checkpoint, &corpus, &w).unwrap();
            assert_eq!(
                serde_json::to_vec(&now).unwrap(),
                serde_json::to_vec(&historical).unwrap()
            );
            assert_eq!(request_mode(&now, &json!({"step":initial})).unwrap(), None);
        }
        assert!(window("jope.native_loop.v4", 60000, 0).is_err());
    }

    #[test]
    fn v2_starts_a_fresh_optimizer_from_the_accepted_parent_then_carries_its_own() {
        let parent = accepted();
        let first = window(LOOP_V2, 60000, 0).unwrap();
        assert_eq!(
            first,
            Window {
                initial: 0,
                target: 3000,
                warm_start_parent_step: Some(60000)
            }
        );
        let params = train_parameters(&artifact("parent"), &artifact("corpus"), &first).unwrap();
        assert_eq!(params["optimizer"], "reset");
        assert_eq!(params["parent_step"], 60000);
        assert_eq!(request_mode(&params, &parent).unwrap(), Some(60000));
        // The second block carries the warm start's own optimizer, 3000 → 6000.
        let second = window(LOOP_V2, 60000, 1).unwrap();
        assert_eq!(
            second,
            Window {
                initial: 3000,
                target: 6000,
                warm_start_parent_step: None
            }
        );
        let params = train_parameters(&artifact("g0"), &artifact("corpus"), &second).unwrap();
        assert_eq!(request_mode(&params, &json!({"step":3000})).unwrap(), None);
        // The binding names the parent weights digest and resets the optimizer.
        let bound = binding(ACCEPTED_STATE_SHA256, &parent).unwrap();
        assert_eq!(
            bound,
            json!({"schema":WARM_START_SCHEMA,"parent_checkpoint_sha256":ACCEPTED_STATE_SHA256,
                "parent_model_sha256":ACCEPTED_MODEL_SHA256,"parent_step":60000,
                "optimizer":"reset","optimizer_step_base":0})
        );
        let carry = json!({"schema":"parallel.replay_policy.v2","optimizer":"carry",
            "eligibility_evidence":{"original_window":train_parameters(&artifact("parent"), &artifact("corpus"), &first).unwrap()}});
        let policy = warm_start_policy(carry, bound.clone());
        assert_eq!(
            policy_mode(&policy, ACCEPTED_STATE_SHA256, &parent).unwrap(),
            Some(bound)
        );
        assert_eq!(
            crate::task_contract::classify_continuation(false, true, Some(&policy)).unwrap(),
            crate::task_contract::ContinuationMode::WeightWarmStart
        );
    }

    #[test]
    fn mixed_forged_or_continuing_warm_start_requests_are_refused() {
        let parent = accepted();
        let first = window(LOOP_V2, 60000, 0).unwrap();
        let good = train_parameters(&artifact("parent"), &artifact("corpus"), &first).unwrap();
        type Edit = fn(&mut Value);
        let cases: [(&str, Edit); 8] = [
            ("explicit carry", |p| p["optimizer"] = json!("carry")),
            ("unknown mode", |p| p["optimizer"] = json!("partial")),
            ("parent step without a mode", |p| {
                p.as_object_mut().unwrap().remove("optimizer");
            }),
            ("mode without a parent step", |p| {
                p.as_object_mut().unwrap().remove("parent_step");
            }),
            ("another parent step", |p| p["parent_step"] = json!(57000)),
            ("continues the parent's optimizer step", |p| {
                p["initial_step"] = json!(60000);
                p["target_step"] = json!(63000);
            }),
            ("claims a later window", |p| {
                p["initial_step"] = json!(3000);
                p["target_step"] = json!(6000);
            }),
            ("unknown field", |p| {
                p["optimizer_parent_step"] = json!(60000)
            }),
        ];
        for (case, edit) in cases {
            let mut forged = good.clone();
            edit(&mut forged);
            assert!(request_mode(&forged, &parent).is_err(), "{case} admitted");
        }
        let mut carry = train_parameters(
            &artifact("parent"),
            &artifact("corpus"),
            &window(LOOP_V1, 60000, 0).unwrap(),
        )
        .unwrap();
        carry["replay"] = json!("broad");
        assert!(request_mode(&carry, &parent).is_err());
    }

    #[test]
    fn mixed_or_forged_warm_start_policies_are_refused() {
        let parent = accepted();
        let first = window(LOOP_V2, 60000, 0).unwrap();
        let bound = binding(ACCEPTED_STATE_SHA256, &parent).unwrap();
        let carry = json!({"schema":"parallel.replay_policy.v2","optimizer":"carry",
            "eligibility_evidence":{"original_window":train_parameters(&artifact("parent"), &artifact("corpus"), &first).unwrap()}});
        let policy = warm_start_policy(carry.clone(), bound);
        // An optimizer carry without a binding stays valid (every v1 policy).
        assert_eq!(
            policy_mode(&carry, ACCEPTED_STATE_SHA256, &parent).unwrap(),
            None
        );
        type Edit = fn(&mut Value);
        let cases: [(&str, Edit); 9] = [
            ("carry naming a warm start", |p| {
                p["optimizer"] = json!("carry")
            }),
            ("reset without a binding", |p| {
                p.as_object_mut().unwrap().remove("warm_start");
            }),
            ("unknown optimizer", |p| p["optimizer"] = json!("partial")),
            ("another parent weights digest", |p| {
                p["warm_start"]["parent_model_sha256"] = json!("b".repeat(64))
            }),
            ("another parent step", |p| {
                p["warm_start"]["parent_step"] = json!(57000)
            }),
            ("another parent checkpoint", |p| {
                p["warm_start"]["parent_checkpoint_sha256"] = json!("c".repeat(64))
            }),
            ("a carried optimizer step base", |p| {
                p["warm_start"]["optimizer_step_base"] = json!(60000)
            }),
            ("an extra binding field", |p| {
                p["warm_start"]["optimizer_sha256"] = json!("d".repeat(64))
            }),
            ("frozen for a carry window", |p| {
                let w = &mut p["eligibility_evidence"]["original_window"];
                w.as_object_mut().unwrap().remove("optimizer");
            }),
        ];
        for (case, edit) in cases {
            let mut forged = policy.clone();
            edit(&mut forged);
            assert!(
                policy_mode(&forged, ACCEPTED_STATE_SHA256, &parent).is_err(),
                "{case} admitted"
            );
        }
    }
}
