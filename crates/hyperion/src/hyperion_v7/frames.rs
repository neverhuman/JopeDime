//! Chunk → [`SweepFrame`] downsampling for the live UI. Small matrices stream literal original
//! positions (grid == matrix); large matrices are honestly binned to a bounded UI grid, each bin
//! carrying the mean of the running-mean confidences its touched cells hold right now.

use std::collections::BTreeMap;

use crate::progress::{SweepCell, SweepFrame};

use super::sweep::ChunkReport;

pub const GRID_MAX_ROWS: usize = 128;
pub const GRID_MAX_COLS: usize = 64;
pub const FRAME_MAX_CELLS: usize = 4096;

#[inline]
fn round3(v: f32) -> f32 {
    (v * 1000.0).round() / 1000.0
}

#[inline]
fn round4(v: f32) -> f32 {
    (v * 10_000.0).round() / 10_000.0
}

/// Build the streamed frame for one chunk. `done_frac` is the OVERALL encode progress (the caller
/// rescales for the train/hold share); `t` is seconds since pipeline start.
pub fn build_frame(report: &ChunkReport<'_>, done_frac: f64, t: f64) -> SweepFrame {
    let acc = report.accum;
    let (h, w) = (acc.h, acc.w);
    let identity = h <= GRID_MAX_ROWS && w <= GRID_MAX_COLS;
    let (grid_rows, grid_cols) = if identity {
        (h, w)
    } else {
        (h.min(GRID_MAX_ROWS), w.min(GRID_MAX_COLS))
    };

    let cells: Vec<SweepCell> = if identity {
        let step = report.touched.len().div_ceil(FRAME_MAX_CELLS).max(1);
        report
            .touched
            .iter()
            .step_by(step)
            .map(|&(r, c)| SweepCell(r, c, round3(acc.cell_confidence(r, c))))
            .collect()
    } else {
        // Bin touched cells to the UI grid; each bin = mean of its touched cells' CURRENT running
        // means (deterministic order via BTreeMap).
        let mut bins: BTreeMap<(u32, u32), (f64, u32)> = BTreeMap::new();
        for &(r, c) in report.touched {
            let gr = (r as usize * grid_rows / h) as u32;
            let gc = (c as usize * grid_cols / w) as u32;
            let e = bins.entry((gr, gc)).or_insert((0.0, 0));
            e.0 += acc.cell_confidence(r, c) as f64;
            e.1 += 1;
        }
        let step = bins.len().div_ceil(FRAME_MAX_CELLS).max(1);
        bins.into_iter()
            .step_by(step)
            .map(|((gr, gc), (sum, n))| SweepCell(gr, gc, round3((sum / n as f64) as f32)))
            .collect()
    };

    SweepFrame {
        sweep: report.sweep,
        sweeps: report.sweeps,
        frac: report.frac,
        done_frac: done_frac.clamp(0.0, 1.0),
        rows: h as u64,
        cols: w as u64,
        grid_rows: grid_rows as u32,
        grid_cols: grid_cols as u32,
        cells,
        vector: acc.dataset_vector().into_iter().map(round4).collect(),
        t,
    }
}
