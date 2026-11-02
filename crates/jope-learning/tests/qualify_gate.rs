//! The seal and champion gates are off by default and refuse to seal or promote, whatever the
//! flag says. The evaluators behind them are exercised on synthetic inputs and on the
//! committed G-JUDGE roster proposal (read, never modified).

use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;

use jope_learning::canon::sha256_hex;
use jope_learning::qualify::gate::{Gate, FLAG};
use jope_learning::qualify::promotion::{self, Verdict};
use jope_learning::qualify::roster;
use jope_learning::qualify::{request_promotion, request_seal, Action, Refusal};
use serde_json::{json, Value};

const PROPOSAL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../deploy/dl-flywheel/pipelines/rlfw/judge/SEALED_ROSTER_PROPOSAL.json"
);

/// A named edit of a valid fixture document.
type Mutation = Box<dyn Fn(&mut Value)>;

fn bytes(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}

fn committed_proposal() -> Vec<u8> {
    std::fs::read(PROPOSAL).unwrap()
}

/// A sound proposal: two training, one development, two confirmation and three sealed families.
fn proposal(alpha_budget: Value) -> Value {
    let record = |n: u32| {
        json!({
            "context_ids": [],
            "dataset_hashes": [format!("d{n}")],
            "member_roles": ["sealed_final"],
            "sources": ["sealed_catalog"],
            "view_ids": [format!("d{n}__r0__f0"), format!("d{n}__r0__f1")],
        })
    };
    json!({
        "adopted": false,
        "alpha_budget": alpha_budget,
        "encoder_pretraining_closure": "unmeasured",
        "partition": {
            "confirmation": ["c1", "c2"],
            "development": ["v1"],
            "meta_training": ["t1", "t2"],
            "sealed_final": ["s1", "s2", "s3"],
        },
        "proof": {"adopted": false, "failures": [], "families": 3, "passed": true,
                  "seal_created": false, "seal_spent": false},
        "qualifying": false,
        "roles": ["meta_training", "development", "confirmation", "sealed_final"],
        "schema": "jope.rlfw-sealed-roster-proposal/v1",
        "seal_created": false,
        "seal_spent": false,
        "sealed_family_records": {"s1": record(1), "s2": record(2), "s3": record(3)},
        "sources": {"closure": {"sealed_context_ids_in_closure": 0, "sealed_views_in_closure": 0}},
    })
}

fn sha(n: u8) -> String {
    format!("{n:02x}").repeat(32)
}

const FAMILIES: usize = 12;

fn manifest() -> Value {
    let families: Vec<String> = (1..=FAMILIES).map(|n| format!("fam-{n:02}")).collect();
    json!({
        "schema": "jope.qualify-arena-manifest/v1",
        "arena_id": "fixture-arena",
        "metric": {"name": "outer_rmse_ratio", "lower_is_better": true},
        "incumbent": {"id": "frozen-v1", "sha256": sha(1)},
        "challenger": {"id": "candidate", "sha256": sha(2)},
        "roster_role": "confirmation",
        "families": families,
        "family_weighting": "equal",
        "min_families": 8,
        "practical_margin": 0.005,
        "alpha_total": 0.05,
        "looks": [{"index": 1, "alpha": 0.01}, {"index": 2, "alpha": 0.04}],
        "contenders": 1,
        "bootstrap": {"resamples": 2000, "seed": 20_261_013},
        "max_failure_rate": 0.05,
        "safety": {"panel": "l6_full_val263", "reference": "frozen-v1", "max_regression": 0.0},
    })
}

/// Two contexts per family; `effect(family, context)` is the incumbent minus the challenger.
fn rows(effect: impl Fn(usize, usize) -> f64) -> Vec<Value> {
    let mut rows = Vec::new();
    for family in 1..=FAMILIES {
        for context in 0..2 {
            rows.push(json!({
                "family_id": format!("fam-{family:02}"),
                "context_id": format!("ctx-{family:02}-{context}"),
                "selection": "selected",
                "challenger": 1.0 - effect(family, context),
                "incumbent": 1.0,
                "attempts": 10,
                "failures": 0,
            }));
        }
    }
    rows
}

fn results(manifest: &[u8], rows: Vec<Value>) -> Value {
    json!({
        "schema": "jope.qualify-arena-results/v1",
        "manifest_sha256": sha256_hex(manifest),
        "look": 1,
        "prior_looks": [],
        "rows": rows,
        "holds": [],
        "cost": {"cpu_seconds": 3600.0, "wall_seconds": 600.0},
        "safety": {"challenger": 0.8970, "reference": 0.8973},
    })
}

/// A clear lift: every family gains about 0.05.
#[allow(clippy::cast_precision_loss)]
fn clear_lift(family: usize, context: usize) -> f64 {
    0.05 + 0.001 * ((family + context) % 3) as f64
}

fn eligible() -> (Vec<u8>, Vec<u8>) {
    let manifest = bytes(&manifest());
    let results = bytes(&results(&manifest, rows(clear_lift)));
    (manifest, results)
}

// ---- the flag ----------------------------------------------------------------------------------

mod confirmation;
mod evidence_bounds;
mod flags_and_admission;
mod promotion_decisions;
mod proposal_validation;
