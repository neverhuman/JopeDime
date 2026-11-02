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
