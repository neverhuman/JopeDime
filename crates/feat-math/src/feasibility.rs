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
pub enum Verdict {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{apply_op, apply_op_generated, fit_op};
    use crate::Pcg32;
    use std::collections::BTreeSet;

    /// Per-thread live/peak heap accounting, so bounds are checked against
    /// actual allocations and not only against the classifier's own model.
    mod probe {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;
        thread_local! {
            static LIVE: Cell<usize> = const { Cell::new(0) };
            static PEAK: Cell<usize> = const { Cell::new(0) };
        }
        fn track(grow: usize, shrink: usize) {
            let _ = LIVE.try_with(|live| {
                let now = live.get().saturating_add(grow).saturating_sub(shrink);
                live.set(now);
                let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
            });
        }
        pub struct Counting;
        // SAFETY: every allocation and pointer/layout pair is forwarded unchanged
        // to System. Accounting uses const-initialized Cells, saturating arithmetic
        // and try_with, so it neither allocates nor unwinds during TLS teardown.
        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                // SAFETY: GlobalAlloc's caller supplies a valid nonzero layout;
                // System receives that layout and its pointer is returned unchanged.
                let pointer = unsafe { System.alloc(layout) };
                if !pointer.is_null() {
                    track(layout.size(), 0);
                }
                pointer
            }
            unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
                // SAFETY: GlobalAlloc's caller supplies a valid nonzero layout;
                // System owns the allocation and initializes all requested bytes.
                let pointer = unsafe { System.alloc_zeroed(layout) };
                if !pointer.is_null() {
                    track(layout.size(), 0);
                }
                pointer
            }
            unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
                // SAFETY: allocations come from System. GlobalAlloc's caller
                // supplies the live pointer and its current allocation layout.
                unsafe { System.dealloc(pointer, layout) };
                track(0, layout.size());
            }
            unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
                // SAFETY: GlobalAlloc's caller supplies a live System allocation,
                // its current layout and a nonzero size within the isize::MAX limit.
                // System's returned pointer (or null on failure) is forwarded unchanged.
                let moved = unsafe { System.realloc(pointer, layout, size) };
                if !moved.is_null() {
                    // Old and new buffers may coexist while the data is copied.
                    track(size, 0);
                    track(0, layout.size());
                }
                moved
            }
        }
        /// Open a measurement window at the current live bytes.
        pub fn start() -> usize {
            let live = LIVE.with(Cell::get);
            PEAK.with(|peak| peak.set(live));
            live
        }
        pub fn peak() -> usize {
            PEAK.with(Cell::get)
        }
    }

    #[global_allocator]
    static COUNTING: probe::Counting = probe::Counting;

    fn program(source: &str) -> GenomeIr {
        GenomeIr::parse(source).unwrap()
    }

    fn data(rows: usize, cols: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Pcg32::new(seed, 11);
        let x: Vec<Vec<f64>> = (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| {
                        // Include exact zeros (impute's missing marker) and ties.
                        let v = rng.range_f64(-2.0, 2.0);
                        if v.abs() < 0.3 {
                            0.0
                        } else {
                            (v * 4.0).round() / 4.0
                        }
                    })
                    .collect()
            })
            .collect();
        let y = x.iter().map(|r| r.iter().sum::<f64>() + 1.0).collect();
        (x, y)
    }

    /// Covered transforms over a range of widths, including MAX_WIDTH edges.
    const STAGES: &[&str] = &[
        "standardize",
        "standardize append=true",
        "robust_scale",
        "rank append=true",
        "blom",
        "quantile_bins bins=70",
        "target_bin_encode bins=3 append=true",
        "sigmoid",
        "step t=0.1",
        "norm_row",
        "polar",
        "lag k=3",
        "roll k=2 append=true",
        "symlog c=2.0",
        "winsor_clip q=0.1",
        "bin_onehot bins=4",
        "bin_onehot bins=40 append=true",
        "hinge knots=3",
        "cheb degree=9",
        "poly_interact",
        "poly_interact append=true",
        "rff_cos n=300 seed=3",
        "rff_sin n=5 append=true",
        "oblique_hinge n=7",
        "rand_proj k=4",
        "landmark_dist k=6",
        "row_stats",
        "row_spectral append=true",
        "row_diff_stats",
        "fft_mag k=5",
        "pair_prod topk=30",
        "svd_project k=100 iters=2",
        "nystrom k=500",
        "select_corr k=3",
        "spline knots=4",
        "pair_ratio topk=100",
        "impute add_indicator=true",
        "impute append=true",
        "derive expr=\"row_l1 + row_max\"",
        "map expr=\"clip(x, -1, 1)\" append=true",
    ];

    #[test]
    fn covered_operator_bounds_contain_measured_widths_and_cells() {
        let mut measured_cases = BTreeSet::new();
        for (case, source) in STAGES.iter().enumerate() {
            for (rows, cols) in [
                (1, 1),
                (3, 2),
                (9, 5),
                (17, 33),
                (6, 300),
                (4, 512),
                (3, 600),
            ] {
                let genome = program(&format!("(pipe X ({source}) (ridge))"));
                let GenomeIr::Pipeline(dsl) = &genome else {
                    unreachable!()
                };
                let stage = &dsl.stages[0];
                let (x, y) = data(rows, cols, case as u64 * 97 + cols as u64);
                let shape = Shape {
                    rows: rows as u64,
                    cols: cols as u64,
                };
                let limits = Limits {
                    max_live_cells: u64::MAX,
                };
                let classified = classify(&genome, shape, &limits);
                let bound = &classified.stages[0];
                // Measure the stage exactly as the fitter runs it: fit, then
                // apply (generated kernel, then the full epilogue).
                let base = probe::start();
                let Ok(op) = fit_op(stage, &x, &y) else {
                    // A deterministic fit refusal allocates nothing further.
                    continue;
                };
                let generated_rows = apply_op_generated(&op, &x);
                let generated = generated_rows[0].len() as u64;
                drop(generated_rows);
                let output_rows = apply_op(&op, &x).unwrap();
                let output = output_rows[0].len() as u64;
                let peak = (probe::peak() - base) as u64;
                drop(output_rows);
                let g = bound.generated_width.expect("covered operator");
                assert!(
                    g.lo <= generated && g.hi.is_some_and(|h| generated <= h),
                    "{source} {rows}x{cols}: generated {generated} outside {g:?}"
                );
                let o = bound.output_width;
                assert!(
                    o.lo <= output && o.hi.is_some_and(|h| output <= h),
                    "{source} {rows}x{cols}: output {output} outside {o:?}"
                );
                // Allocation soundness. The stage bound counts the caller's
                // rows (allocated before the window) and logical f64 cells;
                // bytes allow 2x vector growth, row headers and small scratch.
                let live = bound.live_cells.unwrap();
                let caller = rows as u64 * cols as u64;
                let upper = 16 * (live.hi.unwrap() - caller) + 64 * rows as u64 + 65_536;
                assert!(
                    peak <= upper,
                    "{source} {rows}x{cols}: measured {peak} bytes above bound {upper}"
                );
                // The generated rows are definitely allocated.
                assert!(
                    peak >= 8 * rows as u64 * g.lo,
                    "{source} {rows}x{cols}: measured {peak} bytes below the lower bound"
                );
                assert_eq!(classified.verdict, Verdict::ProvenFeasible, "{source}");
                measured_cases.insert(*source);
            }
        }
        // Every covered stage was actually fitted and measured at least once.
        assert_eq!(
            measured_cases.len(),
            STAGES.len(),
            "unmeasured: {:?}",
            STAGES
                .iter()
                .filter(|s| !measured_cases.contains(*s))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn adversarial_dimensions_and_overflow_refuse_without_allocating() {
        let limits = Limits::default();
        for (source, rows, cols) in [
            ("(pipe X (standardize) (ridge))", u64::MAX, u64::MAX),
            ("(pipe X (bin_onehot bins=32) (ridge))", 1, u64::MAX / 4),
            (
                "(pipe X (cheb degree=8 append=true) (ridge))",
                u64::MAX / 2,
                3,
            ),
            ("(pipe X (ridge))", 1 << 40, 1 << 30),
        ] {
            let classified = classify(&program(source), Shape { rows, cols }, &limits);
            assert_eq!(
                classified.verdict,
                Verdict::ProvenOutsideLimits,
                "{source}: {classified:?}"
            );
        }
        // One-hot expansion before truncation: 50k rows × 512 columns × 32 bins.
        let onehot = classify(
            &program("(pipe X (bin_onehot bins=32) (ridge))"),
            Shape {
                rows: 50_000,
                cols: 512,
            },
            &limits,
        );
        assert_eq!(onehot.verdict, Verdict::ProvenOutsideLimits);
        assert!(onehot.reasons[0].contains("stage 0 (bin_onehot)"));
        assert_eq!(
            onehot.stages[0].generated_width,
            Some(Bound::exact(512 * 32))
        );
        // Truncation happens after allocation; the output is still MAX_WIDTH.
        assert_eq!(onehot.stages[0].output_width, Bound::exact(512));
        // Degenerate shapes and fitter-refused programs are outside the contract.
        for shape in [Shape { rows: 0, cols: 4 }, Shape { rows: 4, cols: 0 }] {
            assert_eq!(
                classify(&program("(pipe X (ridge))"), shape, &limits).verdict,
                Verdict::ProvenOutsideLimits
            );
        }
        let misplaced = classify(
            &program("(pipe X (standardize) (target kind=rank_gauss) (ridge))"),
            Shape { rows: 4, cols: 4 },
            &limits,
        );
        assert_eq!(misplaced.verdict, Verdict::ProvenOutsideLimits);
    }

    #[test]
    fn valid_edge_width_programs_are_feasible_and_fit() {
        let limits = Limits::default();
        for (source, rows, cols) in [
            ("(pipe X (standardize) (ridge))", 64, 512),
            ("(pipe X (standardize append=true) (ridge))", 64, 512),
            ("(pipe X (pair_prod) (ridge))", 8, 1),
            ("(pipe X (poly_interact) (ridge))", 8, 1),
            ("(pipe X (pair_ratio) (ridge))", 8, 1),
            ("(pipe X (fft_mag k=64) (ridge))", 8, 1),
            ("(pipe X (bin_onehot bins=32) (ridge))", 16, 16),
        ] {
            let genome = program(source);
            let classified = classify(
                &genome,
                Shape {
                    rows: rows as u64,
                    cols: cols as u64,
                },
                &limits,
            );
            assert_eq!(
                classified.verdict,
                Verdict::ProvenFeasible,
                "{source}: {classified:?}"
            );
            let (x, y) = data(rows, cols, 5);
            let model = crate::fit(&genome, &x, &y).unwrap();
            assert_eq!(model.predict(&x).unwrap().len(), rows);
        }
        // MAX_WIDTH is the documented edge: fifty thousand rows stay feasible.
        let wide = classify(
            &program("(pipe X (standardize) (ridge))"),
            Shape {
                rows: 50_000,
                cols: crate::MAX_WIDTH as u64,
            },
            &limits,
        );
        assert_eq!(wide.verdict, Verdict::ProvenFeasible);
    }

    #[test]
    fn uncovered_forms_are_unknown_and_classification_is_arm_independent() {
        let limits = Limits::default();
        let shape = Shape { rows: 10, cols: 3 };
        for source in [
            "(pipe X (standardize) (gpr m=400))",
            "(pipe X (not_an_operator) (ridge))",
            "(sym affine \"col(0)\")",
            "(portfolio (member (pipe X (ridge))) (blend))",
        ] {
            let classified = classify(&program(source), shape, &limits);
            assert_eq!(classified.verdict, Verdict::Unknown, "{source}");
            assert!(!classified.reasons.is_empty());
        }
        // An uncovered stage still yields a proof when a later bound is violated.
        let later = classify(
            &program("(pipe X (not_an_operator) (bin_onehot bins=32) (ridge))"),
            Shape {
                rows: 1 << 23,
                cols: 1,
            },
            &limits,
        );
        assert_eq!(later.verdict, Verdict::ProvenOutsideLimits);
        // A pure function of (program, shape, limits): no arm or origin input.
        let genome = program("(pipe X (hinge knots=16) (ridge))");
        assert_eq!(
            classify(&genome, shape, &limits),
            classify(&genome, shape, &limits)
        );
        // Tightening the limit only moves verdicts toward refusal.
        let tight = Limits { max_live_cells: 10 };
        assert_ne!(
            classify(&genome, shape, &tight).verdict,
            Verdict::ProvenFeasible
        );
    }
}
