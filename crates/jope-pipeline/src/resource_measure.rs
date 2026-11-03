//! One candidate's resource readings for resource-aware objectives
//! ([`jope_runtime::fitness::resources`]): what it cost to train and to serve.
//!
//! When the search's objective reads any resource, every candidate records a fixed panel so the
//! accuracy/cost trade-off is visible in `observations.json`, whatever the objective weighs:
//! fit seconds, process CPU seconds, held-out batch prediction time (median of three passes),
//! per-row and per-second batch rates, peak heap growth where the binary installs the meter,
//! serialized model bytes, program nodes, stage count and row counts. Single-row latency costs
//! up to [`LATENCY_ROWS`] extra predictions, so it is measured only when the objective reads it.
//!
//! The candidate's `seconds` keeps its meaning (fit plus the two scoring predictions); the extra
//! passes run after it is read. Under a test clock every machine reading is zero, so frozen runs
//! stay byte-identical; structural readings are always real.
use crate::clock::{Clock, Stamp};
use crate::prepare::Cut;
use anyhow::{ensure, Context, Result};
use feat_math::genome::GenomeIr;
use jope_runtime::fitness::{ResourceVar, Resources};
use jope_runtime::meter;
use std::collections::BTreeSet;

/// Held-out rows whose one-row predictions give the latency percentiles.
pub const LATENCY_ROWS: usize = 256;
const PREDICT_PASSES: usize = 3;

pub(crate) struct Measured {
    pub train: Vec<f64>,
    pub hold: Vec<f64>,
    pub resources: Resources,
    /// Seconds from `started` through the held-out scoring prediction.
    pub seconds: f64,
}

/// Fit `parsed` on the cut's training rows, predict both splits, and read the resource panel
/// plus everything in `wanted`.
pub(crate) fn fit_and_measure(
    parsed: &GenomeIr,
    cut: &Cut,
    wanted: &BTreeSet<ResourceVar>,
    clock: Clock,
    started: Stamp,
) -> Result<Measured> {
    let real = clock.measures();
    let heap = real && meter::heap_meter_installed();
    if real && !heap {
        if let Some(missing) = wanted.iter().find(|r| r.needs_heap_meter()) {
            anyhow::bail!(
                "{missing} needs the process heap meter, which this binary does not install \
                 (the `jope` command line does)"
            );
        }
    }
    let cpu = || -> Result<f64> {
        if !real {
            return Ok(0.0);
        }
        meter::process_cpu_seconds().context("fit_cpu_seconds needs /proc/self/task/*/schedstat")
    };
    let cpu_wanted = wanted.contains(&ResourceVar::FitCpuSeconds);
    let cpu_before = if cpu_wanted { Some(cpu()?) } else { cpu().ok() };

    let window = if heap { meter::HeapWindow::open() } else { None };
    let fit_clock = clock.start();
    let model = feat_math::fit(parsed, &cut.train_x, &cut.train_y)?;
    let fit_seconds = fit_clock.secs();
    let fit_peak = window.map(meter::HeapWindow::close);
    let cpu_after = if cpu_wanted { Some(cpu()?) } else { cpu().ok() };

    let train = model.predict(&cut.train_x)?;
    let window = if heap { meter::HeapWindow::open() } else { None };
    let pass = clock.start();
    let hold = model.predict(&cut.hold_x)?;
    let mut passes = vec![pass.secs()];
    let predict_peak = window.map(meter::HeapWindow::close);
    let seconds = started.secs();
    for _ in 1..PREDICT_PASSES {
        let pass = clock.start();
        std::hint::black_box(model.predict(&cut.hold_x)?);
        passes.push(pass.secs());
    }
    passes.sort_by(f64::total_cmp);
    let predict_seconds = passes[passes.len() / 2];
    let hold_rows = cut.hold_x.len();
    ensure!(hold_rows > 0, "resource readings need at least one held-out row");

    let mut resources = Resources::new();
    let mut put = |var: ResourceVar, value: f64| {
        resources.insert(var, if var.is_measured() && !real { 0.0 } else { value });
    };
    put(ResourceVar::FitSeconds, fit_seconds);
    if let (Some(before), Some(after)) = (cpu_before, cpu_after) {
        put(ResourceVar::FitCpuSeconds, (after - before).max(0.0));
    }
    put(ResourceVar::PredictSeconds, predict_seconds);
    put(
        ResourceVar::PredictUsPerRow,
        predict_seconds * 1e6 / hold_rows as f64,
    );
    put(
        ResourceVar::PredictRowsPerSecond,
        hold_rows as f64 / predict_seconds.max(1e-9),
    );
    if let Some(peak) = fit_peak {
        put(ResourceVar::FitPeakHeapBytes, peak as f64);
    }
    if let Some(peak) = predict_peak {
        put(ResourceVar::PredictPeakHeapBytes, peak as f64);
    }
    if !real {
        for var in wanted.iter().filter(|r| r.needs_heap_meter()) {
            put(*var, 0.0);
        }
    }
    if wanted.iter().any(|r| r.needs_row_latency()) {
        let mut latencies = Vec::with_capacity(LATENCY_ROWS.min(hold_rows));
        for row in cut.hold_x.iter().take(LATENCY_ROWS) {
            let one = vec![row.clone()];
            let call = clock.start();
            std::hint::black_box(model.predict(&one)?);
            latencies.push(call.secs() * 1e6);
        }
        latencies.sort_by(f64::total_cmp);
        put(ResourceVar::RowLatencyP50Us, nearest_rank(&latencies, 0.50));
        put(ResourceVar::RowLatencyP99Us, nearest_rank(&latencies, 0.99));
    }
    put(
        ResourceVar::ModelBytes,
        serde_json::to_vec(&model)
            .context("serialize the fitted model for model_bytes")?
            .len() as f64,
    );
    match crate::kernel2::archive::program_cost(parsed) {
        Ok(cost) => put(ResourceVar::ProgramNodes, f64::from(cost)),
        Err(error) if wanted.contains(&ResourceVar::ProgramNodes) => {
            return Err(error.context("program_nodes"))
        }
        Err(_) => (),
    }
    put(
        ResourceVar::StageCount,
        match parsed {
            GenomeIr::Pipeline(pipe) => pipe.stages.len() as f64,
            _ => 1.0,
        },
    );
    put(ResourceVar::TrainRows, cut.train_x.len() as f64);
    put(ResourceVar::HoldRows, hold_rows as f64);
    for var in wanted {
        ensure!(
            resources.contains_key(var),
            "resource {var} could not be measured for this candidate"
        );
    }
    Ok(Measured {
        train,
        hold,
        resources,
        seconds,
    })
}

/// The nearest-rank `q` quantile of sorted, nonempty values.
fn nearest_rank(sorted: &[f64], q: f64) -> f64 {
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_matches_the_textbook_definition() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(nearest_rank(&v, 0.50), 50.0);
        assert_eq!(nearest_rank(&v, 0.99), 99.0);
        assert_eq!(nearest_rank(&[7.0], 0.99), 7.0);
        assert_eq!(nearest_rank(&[1.0, 2.0, 3.0], 0.5), 2.0);
    }
}
