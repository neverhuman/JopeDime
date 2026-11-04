#![cfg(feature = "neural")]
//! W6-T06: the four continuation modes and the full objective identity (CPU,
//! no model). These are the pre-CUDA guards `training::rank_impl` applies; the
//! GPU path itself is not executed here.

use jope_parallel::{
    manifest::hash,
    neural::{self, Config},
    task_contract::{training_partition, verify_resume_metadata},
    training::{
        continuation::{check_parent, classify, Current, Mode},
        objective,
    },
};
use serde_json::{json, Value};

const CORPUS: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const NEW_CORPUS: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const VOCABULARY: &str = "3333333333333333333333333333333333333333333333333333333333333333";
const SAMPLER: &str = "family_uniform_champion_plus_seven_quantiles_v2";
/// The refusal a parent without an objective identity gets for exact resume and optimizer carry.
const LEGACY_REFUSAL: &str = "continues only as a weight warm start";
/// The accepted step-60000 `state.json`, byte for byte, as retained in the ASM manual's training
/// history. Its SHA-256 is the accepted state that `models/bundles/policy-step-60000.json` names.
const ACCEPTED_STEP_60000_STATE: &[u8] = include_bytes!(
    "../../../docs/internal/asm-manual/data/training-history/accepted60000-state.json"
);
const ACCEPTED_STEP_60000_STATE_SHA256: &str =
    "8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6";

fn config() -> Value {
    serde_json::to_value(Config::regression(401)).unwrap()
}

fn parent() -> Value {
    json!({"schema":"parallel.checkpoint.v1","step":3000,"world":8,"global_batch":2048,
        "corpus_sha256":CORPUS,"vocabulary_sha256":VOCABULARY,"seed":7,"sampler_id":SAMPLER,
        "ranking_backward":"dense_pairwise_v2","runtime_objective":"huber_log1p_seconds_v2",
        "execution_partition":training_partition(8, 256, 2048).unwrap(),
        "objective":objective::identity(),"objective_sha256":objective::identity_sha256(),
        "initialization":{"kind":"fresh","continuation_mode":"cold_initialization"},
        "optimizer_sha256":"a".repeat(64)})
}

/// The same parent trained under a different objective (one coefficient),
/// with its own internally consistent identity and the same legacy IDs.
fn parent_with_changed_objective() -> Value {
    let mut state = parent();
    state["objective"]["coefficients"]["ranking_bt"] = json!(0.5);
    state["objective_sha256"] = json!(objective::identity_digest(&state["objective"]));
    state
}

fn legacy(state: &mut Value) {
    let fields = state.as_object_mut().unwrap();
    fields.remove("objective");
    fields.remove("objective_sha256");
}

struct Request {
    corpus: &'static str,
    vocabulary: &'static str,
    world: usize,
    micro: usize,
    seed: u64,
    sampler: &'static str,
    config: Value,
    objective: String,
}

impl Request {
    fn same() -> Self {
        Self {
            corpus: CORPUS,
            vocabulary: VOCABULARY,
            world: 8,
            micro: 256,
            seed: 7,
            sampler: SAMPLER,
            config: config(),
            objective: objective::identity_sha256(),
        }
    }
    fn current(&self) -> Current<'_> {
        Current {
            corpus_sha256: self.corpus,
            vocabulary_sha256: self.vocabulary,
            world: self.world,
            microbatch: self.micro,
            global_batch: 2048,
            seed: self.seed,
            sampler_id: self.sampler,
            config: &self.config,
            objective_sha256: &self.objective,
        }
    }
    fn check(&self, mode: Mode, state: &Value, policy: Option<&Value>) -> anyhow::Result<Value> {
        check_parent(mode, state, &config(), &self.current(), policy)
    }
}

/// Owner decision 2026-09-26 (`docs/internal/core-ip/work-orders/OWNER-DECISIONS.md`, D12): a
/// parent without an objective identity cannot prove its objective, so it is refused for exact
/// resume and optimizer carry even when both legacy identifiers match. It continues only as a
/// weight warm start (optimizer reset, nothing carried), recorded as `legacy_ids_only`.

/// The accepted step-60000 checkpoint is a legacy parent. The native loop's replay policy
/// (`optimizer: carry`) classifies to an optimizer-carry transition, which the pre-CUDA guard now
/// refuses; the same parent with the optimizer reset is a weight warm start and continues.

/// D12, native loop: the first training block from the accepted step-60000 checkpoint, as each
/// native-loop protocol freezes it. Under `jope.native_loop.v1` it is an optimizer carry, and the
/// pre-CUDA guard still refuses it. Under `jope.native_loop.v2` it is a weight warm start that
/// binds the parent weights digest and that the guard admits with the optimizer reset.
mod identity;
mod transitions;
