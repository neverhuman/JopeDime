//! Factored Grammar-Masked Typed-Edit Action Space, Trace Accounting & Deterministic Replay (MR33 / G10 / SPEC_A G7 / SPEC_B G12 & G13).
//!
//! Verifies:
//! 1. `spec_a_g7_zero_silently_truncated_or_misbound_accepted_children_over_10k_attempts`:
//!    Over >= 10,000 free-running factored grammar-masked typed-edit attempts, zero silently
//!    truncated or misbound accepted children occur; invalid/over-length/unselected/time-lag
//!    candidates are explicitly refused before admission.
//! 2. `spec_a_g7_legacy_mode_legal_sets_and_decode_traces_exact_equality`:
//!    In legacy 401-token grammar mode, legal token sets (`State::legal`), decode traces
//!    (`Vocab::replay`), and conditional token laws (`decode::score_legal` / `BehaviorLaw::native`)
//!    are byte-and-bit-identical to the canonical grammar route and deterministic replay receipts.
//! 3. `spec_b_g12_off_bin_descendants_never_inherit_projected_scores_and_unsupported_counted`:
//!    Off-bin / adapted descendants never inherit projected reward scores from another program
//!    (`ProposalLedger` and `EditPolicy` reject mismatched `scored_program_sha256` / `ResidualClosure`),
//!    and unsupported / refused / shadow records are explicitly counted with `fitting_authority == false`.
//! 4. `spec_b_g13_enumerated_action_laws_normalize_no_minted_propensity`:
//!    Enumerated discrete and mixed continuous action laws normalize to 1.0 over every legal mask,
//!    and `Propensity` cannot be minted or forged from `Unknown` selection laws.
//! 5. `g10_all_attempt_trajectory_logging_and_trace_vs_lower_bound_accounting`:
//!    100% all-attempt trajectory logging and deterministic replay receipt verification with
//!    exact trace-vs-lower-bound step and diagnostics accounting.

#[path = "../src/edit_policy.rs"]
mod edit_policy;

use edit_policy::{
    ActionMask, ColumnDescriptor, ColumnPointer, ContextAbiV2, EditAction, EditKey, EditKind,
    EditOperator, EditPolicy, EditSite, Fidelity, NormalDensity, PolicyComponent,
    ProbabilityMeasure, MAX_PROGRAM_TOKENS,
};
use jope_policy::{
    decode::{self, DrawOutcome, UniformOutcome},
    grammar::{bind_vocabulary, BoundVocabulary, State},
    proposal_ledger::{
        AdaptationEdge, Denominators, NeuralDraw, ProposalLedger, SamplingRecord, SlotPrediction,
        SlotRecord, SlotSource, UniformDraw, EDGE_DECODE, EDGE_SCORE_ENCODING, LEDGER_SCHEMA,
    },
    seeds::ORIGIN_NEURAL,
    sha256_hex,
    support_transition::{BehaviorLaw, PropensityValue},
    CHECKPOINT_VOCABULARY_SHA256,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

fn digest(byte: u8) -> String {
    format!("{byte:02x}").repeat(32)
}

fn bound_vocab() -> BoundVocabulary {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../jope-engine/assets/grammar/featmath.json");
    bind_vocabulary(&std::fs::read(path).unwrap(), CHECKPOINT_VOCABULARY_SHA256).unwrap()
}

fn make_columns(count: usize) -> Vec<ColumnDescriptor> {
    (0..count)
        .map(|i| ColumnDescriptor {
            key: format!("col_{i:03}"),
            kind: if i % 4 == 0 {
                "categorical".into()
            } else {
                "continuous".into()
            },
        })
        .collect()
}

fn make_context(
    col_count: usize,
    selected_indices: &[usize],
    program_tokens: Vec<u32>,
    ordered_time: Option<String>,
    charged_micros: u64,
) -> ContextAbiV2 {
    let cols = make_columns(col_count);
    let selected: Vec<String> = selected_indices
        .iter()
        .map(|&idx| cols[idx].key.clone())
        .collect();
    ContextAbiV2::new(
        cols,
        selected,
        digest(0x11),
        digest(0x22),
        ordered_time,
        digest(0x33),
        program_tokens,
        401,
        digest(0x44),
        digest(0x55),
        charged_micros,
    )
    .expect("valid ContextAbiV2")
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct TrajectoryAttemptRecord {
    attempt_index: usize,
    mixture_draw_bits: u64,
    normal_draw_bits: u64,
    accepted: bool,
    disposition: String,
    action_digest: Option<String>,
    measure_kind: Option<String>,
    measure_value_bits: Option<u64>,
    charged_micros: u64,
    lower_bound_micros: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ReplayTrajectoryReceipt {
    schema: String,
    total_attempts: usize,
    accepted_count: usize,
    refused_count: usize,
    truncated_accepted: usize,
    misbound_accepted: usize,
    total_charged_micros: u64,
    total_lower_bound_micros: u64,
    records_sha256: String,
}

mod spec_a;
mod spec_b;
mod trajectory;

pub use spec_a::{verify_spec_a_g7_10k_attempts, verify_spec_a_g7_legacy_mode_parity};
pub use spec_b::{
    verify_spec_b_g12_off_bin_and_unsupported,
    verify_spec_b_g13_action_laws_normalize_and_no_minted_propensity,
};
pub use trajectory::verify_g10_all_attempt_trajectory_and_replay_receipt;

#[test]
fn spec_a_g7_zero_silently_truncated_or_misbound_accepted_children_over_10k_attempts() {
    verify_spec_a_g7_10k_attempts();
}

#[test]
fn spec_a_g7_legacy_mode_legal_sets_and_decode_traces_exact_equality() {
    verify_spec_a_g7_legacy_mode_parity();
}

#[test]
fn spec_b_g12_off_bin_descendants_never_inherit_projected_scores_and_unsupported_counted() {
    verify_spec_b_g12_off_bin_and_unsupported();
}

#[test]
fn spec_b_g13_enumerated_action_laws_normalize_no_minted_propensity() {
    verify_spec_b_g13_action_laws_normalize_and_no_minted_propensity();
}

#[test]
fn g10_all_attempt_trajectory_logging_and_trace_vs_lower_bound_accounting() {
    verify_g10_all_attempt_trajectory_and_replay_receipt();
}

#[test]
fn zero_truncated_misbound_over_10k_legacy_trace_parity_and_g12_g13() {
    verify_spec_a_g7_10k_attempts();
    verify_spec_a_g7_legacy_mode_parity();
    verify_spec_b_g12_off_bin_and_unsupported();
    verify_spec_b_g13_action_laws_normalize_and_no_minted_propensity();
    verify_g10_all_attempt_trajectory_and_replay_receipt();
}
