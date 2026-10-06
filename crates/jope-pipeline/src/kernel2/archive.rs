//! The loss × cost front of a run's measured programs, published opt-in as evidence only.
//!
//! The front is built from the run's observations after the search has ended and never feeds it
//! back: ranking, parents and the selected program are decided without it. Its inputs are the
//! observations ranking admits (measured, with a finite selection score and no Hard fault), each
//! offered to [`feat_math::kernel::pareto::Archive`] as:
//! - loss: the negated selection score, so a lower loss is a better-ranked score. The selection
//!   score is the held-out R² on the cut's validation rows (a fold of the training rows) after the
//!   configured fault penalty, as ranking uses it;
//! - complexity: [`program_cost`] under [`COST_VERSION`];
//! - faulted cells and the Hard flag: from the observation's fault totals;
//! - key: the sha256 of the program's exact codec bytes (`Program::import_legacy`, then
//!   `to_bytes`). The codec keeps each expression's text as written, so two spellings of one
//!   expression (`1*c0` and `1 * c0`) are two keys; it keeps a number parameter as its value, so
//!   two spellings of one number share one;
//! - no age.
//!
//! The archive depends only on the set of admitted observations, never on their order, and
//! `publication::verify_run` rebuilds it from the published observations and compares bytes. The
//! search never builds it: `search::run_cut` only records the selection, and publication builds
//! and writes the front.

// laneD:B18-pareto-archive

use anyhow::{bail, Context, Result};
use feat_math::{
    dsl::ParamValue,
    kernel::{
        pareto::{self, Archive, Candidate},
        Program,
    },
    math_expr::{Expr, MathExpr},
    GenomeIr,
};
use sha2::{Digest, Sha256};
use std::path::Path;

use super::{search::CutSearch, EngineRecord, Kernel2Observation};
use crate::sha256_hex;

/// The cost version of every archived complexity.
pub const COST_VERSION: &str = pareto::PIPELINE_COST_VERSION;

/// Where a run publishes its archive, relative to its output directory.
pub const PATH: &str = "search/cut-0/archive.json";

/// Whether ranking admits `o`: the filter of `Kernel2Evaluator::rank`.
pub(super) fn ranked(o: &Kernel2Observation) -> bool {
    o.observation.measured() && o.selection_score.is_some_and(f64::is_finite) && !o.faults.is_hard()
}

/// The archive of the observations ranking admits.
pub fn build(observations: &[Kernel2Observation]) -> Result<Archive> {
    let entries = observations
        .iter()
        .filter(|o| ranked(o))
        .map(entry)
        .collect::<Result<Vec<_>>>()?;
    let mut archive = Archive::with_cost_version(COST_VERSION).map_err(anyhow::Error::msg)?;
    archive.update(&entries);
    Ok(archive)
}

/// One admitted observation as the archive takes it.
fn entry(o: &Kernel2Observation) -> Result<Candidate> {
    let genome = GenomeIr::parse(&o.observation.genome)?;
    let bytes = Program::import_legacy(&genome).to_bytes()?;
    Ok(Candidate {
        loss: -o
            .selection_score
            .context("ranked observation without a score")?,
        complexity: program_cost(&genome)?,
        age: None,
        faulted_cells: o.faults.faulted_cells(),
        hard_fault: o.faults.is_hard(),
        key: Sha256::digest(&bytes).into(),
    })
}

/// A pipeline program's cost under [`COST_VERSION`]: its stages, the terminal head included, plus
/// the expression nodes (each literal, variable, operator and call) of every `map` or `derive`
/// stage's `expr`.
pub fn program_cost(genome: &GenomeIr) -> Result<u32> {
    let GenomeIr::Pipeline(pipe) = genome else {
        bail!("the pipeline cost needs a pipeline program");
    };
    let mut cost = pipe.stages.len();
    for stage in &pipe.stages {
        if matches!(stage.op.as_str(), "map" | "derive") {
            let Some(ParamValue::Str(source)) = stage.params.get("expr") else {
                bail!("expression stage without an expression");
            };
            cost += nodes(MathExpr::parse(source)?.ast());
        }
    }
    u32::try_from(cost).context("program cost overflow")
}

/// Every node of `expr`.
fn nodes(expr: &Expr) -> usize {
    let mut count = 0;
    let mut pending = vec![expr];
    while let Some(node) = pending.pop() {
        count += 1;
        match node {
            Expr::Const(_) | Expr::Var(_) => {}
            Expr::Unary(_, child) => pending.push(child),
            Expr::Binary(_, a, b) => pending.extend([a.as_ref(), b.as_ref()]),
            Expr::Call(_, args) => pending.extend(args),
        }
    }
    count
}

/// Writes the archive under `root` when the run selected it, and returns the run's engine record
/// with the archive's digest. Without the selection nothing is written and the record is the
/// search's own.
pub(super) fn publish(searched: &CutSearch, root: &Path) -> Result<Option<EngineRecord>> {
    let mut engine = searched.engine.clone();
    if let Some(record) = engine.as_mut().and_then(|e| e.archive.as_mut()) {
        let json = build(&searched.observations)?.to_json();
        std::fs::write(root.join(PATH), json.as_bytes())?;
        record.sha256 = Some(sha256_hex(json.as_bytes()));
    }
    Ok(engine)
}
