//! Conservative pre-allocation feasibility classification (W5-T04).
//!
//! `classify` propagates, stage by stage, an interval of generated
//! (pre-epilogue) and output widths through a pipeline program using each
//! operator's own parameter clamps and checked arithmetic, and bounds the f64
//! cells that are simultaneously live while the stage is fitted and applied:
//! the caller's input, the current design matrix, the generated/merged output
//! and the operator's fitted state. The covered resource is materialised f64
//! cells (memory); compute time is not covered and stays under the caller's
//! runtime containment.
//!
//! * `ProvenFeasible` — every stage and the head are covered and every upper
//!   bound is within the limit.
//! * `ProvenOutsideLimits` — a lower bound provably exceeds the limit (checked
//!   arithmetic that overflows proves it too), or the fitter refuses the program
//!   or shape outright. Nothing proportional to the declared shape is allocated.
//! * `Unknown` — an operator, head or program form is not covered, or an upper
//!   bound exceeds the limit while no lower bound does. Unknown is not a
//!   soundness proof and never authorises unbounded execution.
//!
//! The rules are one pure function of (program, shape, limits): every
//! comparison arm receives identical rules.
use crate::dsl::Stage;
use crate::genome::GenomeIr;
use serde::Serialize;

pub const PROTOCOL: &str = "feat_math.feasibility.v1";

/// Declared input: `rows` is the largest row count fitted or applied, `cols`
/// the raw feature width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Shape {
    pub rows: u64,
    pub cols: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Limits {
    /// Largest number of simultaneously live f64 cells a fit may materialise.
    pub max_live_cells: u64,
}

impl Default for Limits {
    /// 2 GiB of f64 cells: the evaluator's sealed input-view budget.
    fn default() -> Self {
        Self {
            max_live_cells: 256 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FeasibilityVerdict {
    ProvenFeasible,
    ProvenOutsideLimits,
    Unknown,
}

/// Closed interval; `hi: None` means no finite upper bound is proven. A
/// saturated `lo` of `u64::MAX` means "at least u64::MAX".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Bound {
    pub lo: u64,
    pub hi: Option<u64>,
}

impl Bound {
    fn exact(value: u64) -> Self {
        Self {
            lo: value,
            hi: Some(value),
        }
    }
    fn unbounded_from(lo: u64) -> Self {
        Self { lo, hi: None }
    }
    fn add(self, other: Self) -> Self {
        Self {
            lo: self.lo.saturating_add(other.lo),
            hi: self.hi.zip(other.hi).and_then(|(a, b)| a.checked_add(b)),
        }
    }
    fn mul(self, other: Self) -> Self {
        Self {
            lo: self.lo.saturating_mul(other.lo),
            hi: self.hi.zip(other.hi).and_then(|(a, b)| a.checked_mul(b)),
        }
    }
    /// Image under a monotone non-decreasing `f`; `None` from `f` is overflow.
    fn map(self, f: impl Fn(u64) -> Option<u64>) -> Self {
        Self {
            lo: f(self.lo).unwrap_or(u64::MAX),
            hi: self.hi.and_then(&f),
        }
    }
    /// The shared epilogue: truncation to `MAX_WIDTH`, and an empty row gets
    /// one zero column.
    fn epilogue(self) -> Self {
        let cap = crate::MAX_WIDTH as u64;
        Self {
            lo: self.lo.min(cap).max(1),
            hi: Some(self.hi.unwrap_or(u64::MAX).min(cap).max(1)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StageBound {
    pub index: usize,
    pub op: String,
    pub input_width: Bound,
    /// `None` when the operator is not covered.
    pub generated_width: Option<Bound>,
    pub output_width: Bound,
    pub live_cells: Option<Bound>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Classification {
    pub protocol: &'static str,
    pub verdict: Verdict,
    pub reasons: Vec<String>,
    pub shape: Shape,
    pub limits: Limits,
    pub stages: Vec<StageBound>,
    pub head: Option<String>,
    pub head_live_cells: Option<Bound>,
}

/// Generated width and fitted-state cells of one covered transform stage, from
/// the same parameter accessors and clamps as `ops::fit_op`.
fn stage_bounds(stage: &Stage, width: Bound, rows: u64) -> Option<(Bound, Bound)> {
    let w = width;
    let k = |value: usize| Bound::exact(value as u64);
    let times = |factor: u64| w.map(|v| v.checked_mul(factor));
    let same = w;
    Some(match stage.op.as_str() {
        "standardize" | "norm_features" | "robust_scale" | "winsor_clip" | "winsor_moments" => {
            (same, times(2))
        }
        "rank" | "blom" => (same, times(rows)),
        "quantile_bins" => {
            let bins = stage.get_usize("bins", 8).clamp(2, 64) as u64;
            (same, times(bins + 1))
        }
        "target_bin_encode" => {
            let bins = stage.get_usize("bins", 8).clamp(2, 32) as u64;
            (same, times(2 * bins + 1))
        }
        "sigmoid" | "step" | "norm_row" | "polar" | "lag" | "roll" | "symlog" | "map" => {
            (same, Bound::exact(0))
        }
        "bin_onehot" => {
            let bins = stage.get_usize("bins", 6).clamp(2, 32) as u64;
            (times(bins), times(bins + 1))
        }
        "hinge" => {
            let knots = stage.get_usize("knots", 4).clamp(1, 16) as u64;
            (times(knots), times(knots))
        }
        "cheb" => {
            let degree = stage.get_usize("degree", 4).clamp(1, 8) as u64;
            (times(degree), Bound::exact(0))
        }
        "poly_interact" => {
            // Every diagonal pair, then off-diagonal pairs while fewer than
            // MAX_WIDTH pairs exist: max(w, min(MAX_WIDTH, w + C(w, 2))).
            let cap = crate::MAX_WIDTH as u64;
            let pairs = w.map(|v| {
                Some(if v >= cap {
                    v
                } else {
                    cap.min(v + v * (v - 1) / 2)
                })
            });
            (pairs, pairs.map(|p| p.checked_mul(2)))
        }
        "rff_cos" | "rff_sin" | "oblique_hinge" => {
            let n = stage.get_usize("n", 32).clamp(1, 256) as u64;
            let state = w.map(|v| v.checked_mul(n)?.checked_add(n)?.checked_add(rows));
            (Bound::exact(n), state)
        }
        "rand_proj" => {
            let n = stage.get_usize("k", 16).clamp(1, 256) as u64;
            (Bound::exact(n), times(n))
        }
        "landmark_dist" => {
            let n = stage.get_usize("k", 16).clamp(1, 128) as u64;
            (Bound::exact(n), times(n))
        }
        "row_stats" => (k(10), w),
        "row_spectral" => (k(5), w),
        "row_diff_stats" => (k(8), w),
        "derive" => (k(1), Bound::exact(0)),
        "fft_mag" => {
            let requested = stage.get_usize("k", 8) as u64;
            let n = w.map(|v| Some(requested.clamp(1, (v / 2).max(1))));
            (n, Bound::exact(0))
        }
        "pair_prod" => {
            let topk = stage.get_usize("topk", 8).clamp(2, 24) as u64;
            let pairs = w.map(|v| {
                let t = topk.min(v);
                Some(t * t.saturating_sub(1) / 2)
            });
            (pairs, pairs.map(|p| p.checked_mul(2)?.checked_add(topk)))
        }
        "svd_project" => {
            let requested = stage.get_usize("k", 8).clamp(1, 64) as u64;
            let n = w.map(|v| Some(requested.min(v)));
            (n, w.map(|v| v.checked_mul(requested + 1)))
        }
        "nystrom" => {
            let n = (stage.get_usize("k", 16).clamp(1, 128) as u64).min(rows);
            (
                Bound::exact(n),
                w.map(|v| v.checked_mul(n)?.checked_add(rows)),
            )
        }
        "select_corr" => {
            let requested = stage.get_usize("k", 32).clamp(1, 512) as u64;
            (w.map(|v| Some(requested.min(v))), times(2))
        }
        "spline" => {
            // Near-constant columns emit one value, others knots - 1 values.
            let knots = stage.get_usize("knots", 5).clamp(3, 12) as u64;
            (
                Bound {
                    lo: w.lo,
                    hi: w.hi.and_then(|v| v.checked_mul(knots - 1)),
                },
                times(knots),
            )
        }
        "pair_ratio" => {
            let topk = stage.get_usize("topk", 8).clamp(1, 64) as u64;
            let n = w.map(|v| {
                let m = v.min(32);
                Some(topk.min(m * m.saturating_sub(1)))
            });
            (n, Bound::exact(32 * 32 * 4).add(Bound::exact(rows)))
        }
        "impute" => {
            // Indicators only for columns with a zero, while the total width
            // stays below MAX_WIDTH, and only when requested.
            let add = matches!(
                stage.params.get("add_indicator"),
                Some(crate::dsl::ParamValue::Bool(true))
            ) || stage.get_usize("add_indicator", 0) == 1;
            let cap = crate::MAX_WIDTH as u64;
            let generated = Bound {
                lo: w.lo,
                hi: w
                    .hi
                    .map(|v| v + if add { v.min(cap.saturating_sub(v)) } else { 0 }),
            };
            (generated, w.map(|v| v.checked_add(rows)))
        }
        _ => return None,
    })
}

fn get_append(stage: &Stage) -> bool {
    crate::ops::get_append(stage)
}

/// Accumulated verdict evidence.
struct Evidence {
    limit: u64,
    outside: bool,
    unknown: bool,
    reasons: Vec<String>,
}

impl Evidence {
    fn check(&mut self, cells: Option<Bound>, what: &str) {
        let limit = self.limit;
        match cells {
            Some(b) if b.lo > limit => {
                self.outside = true;
                self.reasons.push(format!(
                    "{what}: at least {} live cells exceed {limit}",
                    b.lo
                ));
            }
            Some(Bound { hi: Some(hi), .. }) if hi <= limit => {}
            Some(b) => {
                self.unknown = true;
                self.reasons.push(format!(
                    "{what}: live cells in [{}, {}] may exceed {limit}",
                    b.lo,
                    b.hi.map_or("unbounded".into(), |h| h.to_string())
                ));
            }
            None => self.not_covered(format!("{what}: not covered")),
        }
    }
    fn not_covered(&mut self, reason: String) {
        self.unknown = true;
        self.reasons.push(reason);
    }
}

/// Classify one program over a declared shape. Pure and allocation-bounded by
/// the program's length, never by the declared shape.
pub fn classify(genome: &GenomeIr, shape: Shape, limits: &Limits) -> Classification {
    let mut outcome = Classification {
        protocol: PROTOCOL,
        verdict: Verdict::ProvenFeasible,
        reasons: Vec::new(),
        shape,
        limits: *limits,
        stages: Vec::new(),
        head: None,
        head_live_cells: None,
    };
    let mut evidence = Evidence {
        limit: limits.max_live_cells,
        outside: false,
        unknown: false,
        reasons: Vec::new(),
    };
    if shape.rows == 0 || shape.cols == 0 {
        outcome.verdict = Verdict::ProvenOutsideLimits;
        outcome
            .reasons
            .push("empty input: the fitter refuses empty rows or columns".into());
        return outcome;
    }
    let GenomeIr::Pipeline(dsl) = genome else {
        outcome.verdict = Verdict::Unknown;
        outcome
            .reasons
            .push("program form not covered (symbolic or explicit portfolio)".into());
        return outcome;
    };
    let rest = match crate::model::split_target(&dsl.stages) {
        Ok((_, rest)) => rest,
        Err(error) => {
            outcome.verdict = Verdict::ProvenOutsideLimits;
            outcome
                .reasons
                .push(format!("program refused by the fitter: {error}"));
            return outcome;
        }
    };
    let (transforms, head) = crate::model::split_stages(rest);
    let rows = Bound::exact(shape.rows);
    let input = rows.mul(Bound::exact(shape.cols));
    // The caller's rows stay alive while the fitter clones them once.
    let caller = input;
    evidence.check(Some(caller.add(input)), "input copy");
    let mut width = Bound::exact(shape.cols);
    for (index, stage) in transforms.iter().enumerate() {
        let what = format!("stage {index} ({})", stage.op);
        let (generated, live, output) = match stage_bounds(stage, width, shape.rows) {
            Some((generated, state)) => {
                let merged = if get_append(stage) {
                    width.add(generated)
                } else {
                    generated
                };
                // Upper: caller + current + generated rows + merged rows (they
                // coexist while the epilogue merges) + fitted state.
                // Lower: caller + current + generated rows (all allocated).
                let upper = caller
                    .add(rows.mul(width))
                    .add(rows.mul(generated))
                    .add(rows.mul(merged))
                    .add(state);
                let lower = caller.add(rows.mul(width)).add(rows.mul(generated));
                (
                    Some(generated),
                    Some(Bound {
                        lo: lower.lo,
                        hi: upper.hi,
                    }),
                    merged.epilogue(),
                )
            }
            None => (None, None, Bound::unbounded_from(1).epilogue()),
        };
        if generated.is_none() {
            evidence.not_covered(format!("{what}: operator not covered"));
        } else {
            evidence.check(live, &what);
        }
        outcome.stages.push(StageBound {
            index,
            op: stage.op.clone(),
            input_width: width,
            generated_width: generated,
            output_width: output,
            live_cells: live,
        });
        width = output;
    }
    outcome.head = Some(head.op.clone());
    let head_cells = match head.op.as_str() {
        // Normal equations: a p×p Gram matrix beside the design matrix.
        "ridge" | "logistic" => Some(
            caller
                .add(rows.mul(width))
                .add(width.mul(width))
                .add(width.map(|p| p.checked_mul(3))),
        ),
        _ => None,
    };
    outcome.head_live_cells = head_cells;
    if head_cells.is_none() {
        evidence.not_covered(format!("head {} not covered", head.op));
    } else {
        evidence.check(head_cells, &format!("head ({})", head.op));
    }
    outcome.verdict = if evidence.outside {
        Verdict::ProvenOutsideLimits
    } else if evidence.unknown {
        Verdict::Unknown
    } else {
        Verdict::ProvenFeasible
    };
    outcome.reasons = evidence.reasons;
    outcome
}

/// Compatibility alias for the domain-specific `FeasibilityVerdict` type.
pub type Verdict = FeasibilityVerdict;

#[cfg(test)]
mod tests;
