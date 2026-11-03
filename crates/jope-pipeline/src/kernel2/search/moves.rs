//! Deterministic moves at the end of each search round.
//!
//! A move proposes children from the round's ranked parents without drawing from any random
//! stream. Each new child the profile admits is measured like every other program: one
//! evaluation from the shared allowance, under the start guard refinement uses, with its pass's
//! origin. Proposal work is not charged. Every proposal is recorded in [`PATH`]. With no move
//! selected the search never calls this module and nothing is written.
//!
//! The first pass is the exact rewrite pass ([`REWRITE_SATURATE`], origin [`REWRITE`]): every
//! `map` or `derive` stage's expression of each of the round's top `refinement.top_k` ranked
//! parents with such a stage is simplified by `MathExpr::simplify_v2`, from the expression text
//! alone. A smaller tree the profile refuses is discarded and its reason recorded, and so is a
//! stage whose expression cannot be rewritten at all ([`NOT_RUN`]); the pass goes on either way.
//!
//! The second pass, run after the first when both are selected, is the bounded minimax
//! linearization pass ([`MINIMAX_LINEARIZE`], origin [`LINEARIZE`]). For each of the round's top
//! `refinement.top_k` ranked parents with a `derive` stage that has a subtree it can consider,
//! `MathExpr::linearize_v2` fits each one-variable subtree near the root of each `derive` stage's
//! expression by its degree-1 discrete minimax line, on that stage's training input alone, and
//! offers the line in its place when the line is smaller, meets the fit's minimax bound on those
//! rows, and the profile admits the result. Each offer is one child program, measured once; the
//! measured score ranks it, never the line's error. Every considered subtree is recorded, offered
//! or refused, with its fit, bound and reason.
//!
//! `check` re-derives the records against the published measurements. For each round and pass
//! it ranks the measurements made before the pass, as the pass ranked them, and requires the
//! records to follow that list of parents in order: all of it, or a prefix only in a run that
//! ended in that round on its evaluations or seconds bound (exactly where the allowance ran out
//! under an evaluations stop; under a seconds stop, where the start guard refused, which is not
//! published). A rewrite record re-derives from its parent's text alone, and a measured rewritten
//! program must score bit for bit as its parent. A linearization record re-derives from its
//! parent's text and the fit it records: the training rows are not published, so the fit, its
//! range and the line's measured error are checked for consistency with each other and with the
//! other records of the same stage expression, never measured again.

// laneD:B20-rewrite-simplify (the module); laneD:B19-minimax-linearize (the second pass).

use anyhow::{anyhow, bail, ensure, Context, Result};
use feat_math::{
    dsl::ParamValue,
    kernel::search::admit_child,
    math_expr::{ExpressionScope, MathExpr},
    GenomeIr,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::Path, time::Duration};

use super::{generation::BARREN_ROUND_LIMIT, CutSearch, Options, State};
use crate::{
    clock::Stamp,
    kernel2::{
        receipt::Search,
        records::{Faults, Measurement},
        refinement::parse_untrusted_pipeline,
        EngineRecord, Kernel2Evaluator, Kernel2Observation, LINEARIZE, MINIMAX_LINEARIZE, REWRITE,
        REWRITE_SATURATE,
    },
    prepare::Cut,
    search::{STOPPED_BY_EVALUATIONS, STOPPED_BY_SECONDS},
    sha256_hex,
};

// laneD:B19-minimax-linearize begin
mod linearize;
pub use linearize::{FitRecord, LinearizationRecord, PREFIX_FAULTS};
// laneD:B19-minimax-linearize end

/// Where a run publishes its move records, relative to its output directory.
pub const PATH: &str = "search/cut-0/moves.json";

/// Every stage kept its expression, and the profile refused nothing.
pub const UNCHANGED: &str = "unchanged";
/// No stage change survived the profile, or the assembled program failed pipeline admission.
pub const REFUSED: &str = "refused";
/// The admitted program was already measured.
pub const DUPLICATE: &str = "duplicate";
/// The start guard refused the measurement; the run stopped on its seconds bound.
pub const NOT_STARTED: &str = "not_started";
/// The admitted program was measured at `measurement`, whatever that measurement's status.
pub const MEASURED: &str = "measured";
// laneD:B19-minimax-linearize begin
/// The shared allowance was spent before the admitted program could be measured; the pass ended
/// and the run stopped on its evaluations bound. Only the linearization pass, which can offer
/// several programs per parent, records it.
pub const NO_ALLOWANCE: &str = "no_allowance";
// laneD:B19-minimax-linearize end

/// The stop of a stage whose expression could not be rewritten at all, because it does not parse
/// or the profile refuses it: saturation never ran, and the stage's refusal says why.
pub const NOT_RUN: &str = "not_run";
/// The start of the refusal of a stage whose expression could not be rewritten at all.
pub const REWRITE_FAILED: &str = "the stage's expression could not be rewritten";

/// One proposal of one move for one parent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoveRecord {
    /// The move's method id.
    pub id: String,
    /// The search round after which the move ran: the measured child's round index.
    pub round: usize,
    pub parent_sha256: String,
    /// The assembled program, when a stage changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub program: Option<String>,
    /// One of [`UNCHANGED`], [`REFUSED`], [`DUPLICATE`], [`NOT_STARTED`], [`NO_ALLOWANCE`] or
    /// [`MEASURED`].
    pub outcome: String,
    /// Why pipeline admission refused the assembled program.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The child's index in `observations.json`, for [`MEASURED`] only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurement: Option<usize>,
    /// The rewrite pass's record of each expression stage, in stage order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rewrites: Vec<StageRewrite>,
    // laneD:B19-minimax-linearize begin
    /// The linearization pass's record of its one subtree; absent from the rewrite pass's
    /// records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linearization: Option<LinearizationRecord>,
    // laneD:B19-minimax-linearize end
}

/// The rewrite pass on one expression stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageRewrite {
    /// The stage's index in the parent.
    pub stage: usize,
    /// Why saturation stopped: `saturated`, `iteration_limit`, `node_limit` or `malformed`; or
    /// [`NOT_RUN`] when it never ran, with every counter 0 and the reason in `refusal`.
    pub stop: String,
    pub iterations: usize,
    pub enodes: usize,
    pub applications: usize,
    pub peak: usize,
    /// The smaller expression the profile admitted, which replaced the stage's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
    /// Why the profile refused the smaller expression, which was discarded; or, with the stop
    /// [`NOT_RUN`], why the stage's own expression could not be rewritten.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl StageRewrite {
    /// Whether the profile's admission (the pool, the caps or the negation rule) refused a smaller
    /// expression that saturation found, so the stage kept its own.
    pub fn refused_by_pool(&self) -> bool {
        self.refusal.is_some() && self.stop != NOT_RUN
    }

    /// Whether the stage's own expression could not be rewritten at all ([`NOT_RUN`]).
    pub fn not_run(&self) -> bool {
        self.stop == NOT_RUN
    }
}

/// How many times the rewrite pass dropped a smaller expression because the profile's admission
/// refused it, over every record of `moves`: the count `MovesRecord::refused_by_pool` publishes,
/// refusal events per round a parent is read; a parent ranked top for N rounds adds N. It does
/// not count distinct expressions.
pub fn refused_by_pool(moves: &[MoveRecord]) -> usize {
    rewrite_stages(moves)
        .filter(|r| r.refused_by_pool())
        .count()
}

/// How many stage entries of the rewrite pass's records have the stop [`NOT_RUN`]: the count
/// `MovesRecord::not_run` publishes. A genuine run is expected to have none.
pub fn not_run(moves: &[MoveRecord]) -> usize {
    rewrite_stages(moves).filter(|r| r.not_run()).count()
}

fn rewrite_stages(moves: &[MoveRecord]) -> impl Iterator<Item = &StageRewrite> {
    moves
        .iter()
        .filter(|m| m.id == REWRITE_SATURATE)
        .flat_map(|m| &m.rewrites)
}

/// The rewrite pass's proposal for one parent, before any search state is read.
#[derive(Debug)]
pub(crate) struct Proposal {
    rewrites: Vec<StageRewrite>,
    program: Option<String>,
    reason: Option<String>,
    /// A stage changed and the assembled program passed pipeline admission.
    admitted: bool,
}

impl Proposal {
    /// The record's outcome when the proposal was not admitted.
    fn outcome(&self) -> &'static str {
        if self.program.is_some() || self.rewrites.iter().any(|r| r.refusal.is_some()) {
            REFUSED
        } else {
            UNCHANGED
        }
    }
}

mod accounting;
mod rewrite;

#[cfg(test)]
use accounting::scored;
pub(crate) use accounting::{check, publish};
#[cfg(test)]
use accounting::{not_run_count, pool_count};
#[cfg(test)]
use rewrite::owned;
pub(crate) use rewrite::propose;
pub(super) use rewrite::{finish_round, ranked_parents, selected};
#[cfg(test)]
use rewrite::{has_expression_stage, linearize_round, simplify_round};

#[cfg(test)]
mod tests;

// laneD:B19-minimax-linearize
#[cfg(test)]
mod linearize_tests;

#[cfg(test)]
mod gp_source_move_capture_tests {
    #[test]
    fn complete_rewrite_and_linearization_proposals_are_captured() {
        if std::env::var_os("JOPE_GP_MOVE_CAPTURE_CHILD").is_some() {
            let profile = feat_math::kernel::profile::shipped_kernel2_profiles()
                .into_iter()
                .find(|p| p.id == "kernel2-baseline")
                .unwrap();
            let evaluator = crate::kernel2::Kernel2Evaluator::new(profile).unwrap();
            let program = "(pipe X (derive expr=\"c0*1.0\" append=false) (ridge))";
            assert!(super::propose(&evaluator, program, 1)
                .unwrap()
                .program
                .is_some());
            let x = (0..16).map(|i| vec![i as f64 / 10.0]).collect::<Vec<_>>();
            let cut = crate::prepare::Cut {
                index: 0,
                train_y: x.iter().map(|r| r[0] * 2.0 + 1.0).collect(),
                train_x: x,
                hold_x: vec![],
                hold_y: vec![],
                hold_indices: vec![],
            };
            assert!(super::linearize::propose(
                &evaluator,
                "(pipe X (derive expr=\"sin((0.1*c0))*3.0\" append=false) (ridge))",
                &cut,
                1
            )
            .unwrap()
            .iter()
            .any(|p| p.program.is_some()));
            return;
        }
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/gp-source-tests");
        std::fs::create_dir_all(&root).unwrap();
        let directory = tempfile::tempdir_in(root).unwrap();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "kernel2::search::moves::gp_source_move_capture_tests::complete_rewrite_and_linearization_proposals_are_captured", "--nocapture"])
            .env("JOPE_GP_MOVE_CAPTURE_CHILD", "1").env("JOPE_GP_SOURCE_DIR", directory.path()).output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let mut rewrite = 0;
        let mut linearize = 0;
        for entry in std::fs::read_dir(directory.path()).unwrap() {
            for line in std::fs::read_to_string(entry.unwrap().path())
                .unwrap()
                .lines()
            {
                let row: serde_json::Value = serde_json::from_str(line).unwrap();
                feat_math::GenomeIr::parse(row["program"].as_str().unwrap()).unwrap();
                if row["engine"] == "kernel2-move" {
                    let coordinate = row["coordinate"].as_str().unwrap();
                    rewrite += usize::from(coordinate == "rewrite");
                    linearize += usize::from(coordinate.starts_with("linearize/stage/"));
                }
            }
        }
        assert_eq!(rewrite, 1);
        assert!(linearize > 0);
    }
}
