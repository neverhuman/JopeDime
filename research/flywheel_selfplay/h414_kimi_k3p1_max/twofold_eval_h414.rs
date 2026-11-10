//! Direct 2-Fold `JTT1` Benchmark Evaluator for Frozen Rust Frontier Regressors on `DL-Val91`.
//!
//! Compiles the four frozen challenge entrants byte-for-byte from their canonical repository paths:
//!   1. `Opus5p5_Extra` (`crates/jope-bench/regressors/Opus5p5_Extra.rs`)
//!   2. `Gemini4_argon_high` (`crates/jope-automl/src/Gemini4_argon_high.rs`)
//!   3. `GPT6p1_Sol_Max` (`crates/jope-automl/src/GPT6p1_Sol_Max.rs`)
//!   4. `GPT6_Astra_Max` (`crates/jope-bench/submissions/GPT6_Astra_Max.rs`)
//!
//! Evaluates each entrant on the exact 2-fold `JTT1` splits (`fold 0` vs `fold 1`) for every
//! context in the supplied context list, computing exact fold RMSEs, baseline mean-predictor RMSEs,
//! fold RMSE ratios, `cv_geomean_rmse_ratio`, `logratio`, S1 `fitness`, and `utility`.
//!
//! H414 (2026-11-07): H257's `eval_frontier_dl_val91.rs` (origin/main, sha256 5a7bee26…) with exactly three
//! additions and nothing else changed: (1) `extern crate kimik3p1_max;`, (2) its match arm, (3) an optional
//! fifth argument, a comma-separated entrant filter (default: the four H257 entrants plus `kimik3p1_max`), named in
//! the usage text.
//! Scoring, record format and threading are byte-for-byte H257.

#![allow(dead_code, unused_imports, unused_assignments, unused_variables, unused_mut)]

extern crate gemini4_argon_high;
extern crate gpt6_astra_max;
extern crate gpt6p1_sol_max;
extern crate kimik3p1_max;
extern crate opus5p5_extra;

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const SEED: u64 = 42;

fn fit_predict_entrant(
    entrant: &str,
    seed: u64,
    train: &[f64],
    n_rows: usize,
    n_cols: usize,
    features: &[f64],
    n_predict: usize,
) -> Result<Vec<f64>, String> {
    let n_feat = n_cols.saturating_sub(1);
    match entrant {
        "Opus5p5_Extra" => {
            use opus5p5_extra::RegressorApi;
            let mut m = opus5p5_extra::Regressor::new(seed);
            m.fit(train, n_rows, n_cols)?;
            m.predict(features, n_predict, n_feat)
        }
        "Gemini4_argon_high" => {
            use gemini4_argon_high::RegressorApi;
            let mut m = gemini4_argon_high::Regressor::new(seed);
            m.fit(train, n_rows, n_cols)?;
            m.predict(features, n_predict, n_feat)
        }
        "GPT6p1_Sol_Max" => {
            use gpt6p1_sol_max::RegressorApi;
            let mut m = gpt6p1_sol_max::Regressor::new(seed);
            m.fit(train, n_rows, n_cols)?;
            m.predict(features, n_predict, n_feat)
        }
        "GPT6_Astra_Max" => {
            use gpt6_astra_max::RegressorApi;
            let mut m = gpt6_astra_max::Regressor::new(seed);
            m.fit(train, n_rows, n_cols)?;
            m.predict(features, n_predict, n_feat)
        }
        "kimik3p1_max" => {
            use kimik3p1_max::RegressorApi;
            let mut m = kimik3p1_max::Regressor::new(seed);
            m.fit(train, n_rows, n_cols)?;
            m.predict(features, n_predict, n_feat)
        }
        _ => Err(format!("unknown entrant: {}", entrant)),
    }
}

struct Jtt1Table {
    nt: usize,
    nq: usize,
    width: usize,
    folds: Vec<u8>,
    data: Vec<f64>, // row-major nt x (width + 1)
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(b)
}

fn read_f64_le(bytes: &[u8], offset: usize) -> f64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[offset..offset + 8]);
    f64::from_le_bytes(b)
}

fn parse_jtt1(path: &Path) -> Result<Jtt1Table, String> {
    let raw = fs::read(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    if raw.len() < 17 || &raw[0..4] != b"JTT1" || raw[16] != 2 {
        return Err(format!("invalid 2-fold JTT1 header in {}", path.display()));
    }
    let nt = read_u32_le(&raw, 4) as usize;
    let nq = read_u32_le(&raw, 8) as usize;
    let width = read_u32_le(&raw, 12) as usize;
    let cols = width + 1;
    let expected = 17 + nt + (nt + nq) * cols * 8;
    if raw.len() != expected {
        return Err(format!(
            "length mismatch in {}: got {}, expected {}",
            path.display(),
            raw.len(),
            expected
        ));
    }
    let folds = raw[17..17 + nt].to_vec();
    let mut data = Vec::with_capacity(nt * cols);
    let base = 17 + nt;
    for i in 0..(nt * cols) {
        data.push(read_f64_le(&raw, base + i * 8));
    }
    Ok(Jtt1Table {
        nt,
        nq,
        width,
        folds,
        data,
    })
}

#[derive(Clone, Debug)]
struct EntrantEvalResult {
    cid: String,
    entrant: String,
    status: String,
    rmse_fold0: f64,
    rmse_fold1: f64,
    base_rmse_fold0: f64,
    base_rmse_fold1: f64,
    ratio_fold0: f64,
    ratio_fold1: f64,
    cv_geomean_rmse_ratio: f64,
    logratio: f64,
    tail_logratio: f64,
    sd_logratio: f64,
    cv_fold_wins: usize,
    cv_gate_passed: bool,
    s1_fitness_zero_comp: f64,
    s1_utility_zero_comp: f64,
    unconstrained_utility: f64,
    wall_ms: f64,
    fold0_preds: Vec<f64>,
    fold1_preds: Vec<f64>,
}

fn evaluate_context_entrant(cid: &str, tbl: &Jtt1Table, entrant: &str) -> EntrantEvalResult {
    let t0 = Instant::now();
    let cols = tbl.width + 1;
    let mut rmse_folds = [0.0f64; 2];
    let mut base_folds = [0.0f64; 2];
    let mut ratio_folds = [1.0f64; 2];
    let mut preds_folds: [Vec<f64>; 2] = [Vec::new(), Vec::new()];

    for f in 0..2u8 {
        let mut train_mat = Vec::new();
        let mut n_train = 0usize;
        let mut y_train_sum = 0.0f64;

        let mut val_feat = Vec::new();
        let mut y_val = Vec::new();

        for r in 0..tbl.nt {
            let row_slice = &tbl.data[r * cols..(r + 1) * cols];
            if tbl.folds[r] != f {
                train_mat.extend_from_slice(row_slice);
                y_train_sum += row_slice[tbl.width];
                n_train += 1;
            } else {
                val_feat.extend_from_slice(&row_slice[0..tbl.width]);
                y_val.push(row_slice[tbl.width]);
            }
        }
        let n_val = y_val.len();
        let y_train_mean = y_train_sum / (n_train.max(1) as f64);
        let mut base_sse = 0.0f64;
        for &yv in &y_val {
            let d = yv - y_train_mean;
            base_sse += d * d;
        }
        let base_rmse = (base_sse / (n_val.max(1) as f64)).sqrt();
        base_folds[f as usize] = base_rmse;

        let preds_res = fit_predict_entrant(
            entrant,
            SEED,
            &train_mat,
            n_train,
            cols,
            &val_feat,
            n_val,
        );
        match preds_res {
            Ok(preds) if preds.len() == n_val && preds.iter().all(|v| v.is_finite()) => {
                let mut sse = 0.0f64;
                for (p, &yv) in preds.iter().zip(y_val.iter()) {
                    let d = *p - yv;
                    sse += d * d;
                }
                let rmse = (sse / (n_val.max(1) as f64)).sqrt();
                rmse_folds[f as usize] = rmse;
                ratio_folds[f as usize] = (rmse / base_rmse.max(1e-12)).max(1e-12);
                preds_folds[f as usize] = preds;
            }
            _ => {
                let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
                return EntrantEvalResult {
                    cid: cid.to_string(),
                    entrant: entrant.to_string(),
                    status: "failed".to_string(),
                    rmse_fold0: f64::NAN,
                    rmse_fold1: f64::NAN,
                    base_rmse_fold0: base_folds[0],
                    base_rmse_fold1: base_folds[1],
                    ratio_fold0: 1e9,
                    ratio_fold1: 1e9,
                    cv_geomean_rmse_ratio: 1e9,
                    logratio: 0.0,
                    tail_logratio: 0.0,
                    sd_logratio: 0.0,
                    cv_fold_wins: 0,
                    cv_gate_passed: false,
                    s1_fitness_zero_comp: 100.0,
                    s1_utility_zero_comp: -2.0,
                    unconstrained_utility: -2.0,
                    wall_ms,
                    fold0_preds: Vec::new(),
                    fold1_preds: Vec::new(),
                };
            }
        }
    }

    let lr0 = ratio_folds[0].ln();
    let lr1 = ratio_folds[1].ln();
    let mean_lr = 0.5 * (lr0 + lr1);
    let tail_lr = lr0.max(lr1);
    let sd_lr = 0.5 * (lr0 - lr1).abs();
    let geomean_ratio = mean_lr.exp();
    let wins = (if ratio_folds[0] <= 1.0 { 1 } else { 0 })
        + (if ratio_folds[1] <= 1.0 { 1 } else { 0 });
    let gate = (wins == 2) && (geomean_ratio <= 0.90);

    let raw_obj = 0.8 * mean_lr + 0.1 * tail_lr + 0.1 * sd_lr;
    let s1_fitness = if gate {
        raw_obj
    } else {
        100.0 + mean_lr.max(0.0) + tail_lr.max(0.0) + sd_lr.max(0.0)
    };
    let s1_utility = if gate { -s1_fitness } else { -2.0 };
    let unconstrained_utility = -raw_obj;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    EntrantEvalResult {
        cid: cid.to_string(),
        entrant: entrant.to_string(),
        status: "completed".to_string(),
        rmse_fold0: rmse_folds[0],
        rmse_fold1: rmse_folds[1],
        base_rmse_fold0: base_folds[0],
        base_rmse_fold1: base_folds[1],
        ratio_fold0: ratio_folds[0],
        ratio_fold1: ratio_folds[1],
        cv_geomean_rmse_ratio: geomean_ratio,
        logratio: mean_lr,
        tail_logratio: tail_lr,
        sd_logratio: sd_lr,
        cv_fold_wins: wins,
        cv_gate_passed: gate,
        s1_fitness_zero_comp: s1_fitness,
        s1_utility_zero_comp: s1_utility,
        unconstrained_utility,
        wall_ms,
        fold0_preds: preds_folds[0].clone(),
        fold1_preds: preds_folds[1].clone(),
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        return Err(format!(
            "Usage: {} <tables_dir> <cids_file> <out_jsonl> [num_threads] [entrant,entrant,...]",
            args[0]
        ));
    }
    let tables_dir = PathBuf::from(&args[1]);
    let cids_file = PathBuf::from(&args[2]);
    let out_jsonl = PathBuf::from(&args[3]);
    let num_threads: usize = args
        .get(4)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(6)
        .clamp(1, 12);

    let cids_text =
        fs::read_to_string(&cids_file).map_err(|e| format!("read cids_file: {}", e))?;
    let cids: Vec<String> = cids_text
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let all_entrants = [
        "Opus5p5_Extra",
        "Gemini4_argon_high",
        "GPT6p1_Sol_Max",
        "GPT6_Astra_Max",
        "kimik3p1_max",
    ];
    let entrants: Vec<&str> = match args.get(5) {
        None => all_entrants.to_vec(),
        Some(list) => {
            let mut chosen = Vec::new();
            for name in list.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                let known = all_entrants
                    .iter()
                    .find(|e| **e == name)
                    .ok_or_else(|| format!("unknown entrant in filter: {}", name))?;
                if !chosen.contains(known) {
                    chosen.push(*known);
                }
            }
            if chosen.is_empty() {
                return Err(String::from("empty entrant filter"));
            }
            chosen
        }
    };

    let mut tables = Vec::with_capacity(cids.len());
    for cid in &cids {
        let p = tables_dir.join(format!("{}.jtt1", cid));
        let tbl = parse_jtt1(&p)?;
        tables.push((cid.clone(), tbl));
    }

    let mut tasks = Vec::with_capacity(tables.len() * entrants.len());
    for (idx, _) in tables.iter().enumerate() {
        for &entrant in &entrants {
            tasks.push((idx, entrant));
        }
    }

    let next_task = AtomicUsize::new(0);
    let results: Mutex<Vec<EntrantEvalResult>> = Mutex::new(Vec::with_capacity(tasks.len()));
    let t_start = Instant::now();

    std::thread::scope(|s| {
        for _ in 0..num_threads {
            s.spawn(|| loop {
                let k = next_task.fetch_add(1, Ordering::Relaxed);
                if k >= tasks.len() {
                    break;
                }
                let (tbl_idx, entrant) = tasks[k];
                let (ref cid, ref tbl) = tables[tbl_idx];
                let res = evaluate_context_entrant(cid, tbl, entrant);
                results.lock().unwrap().push(res);
            });
        }
    });

    let mut out_vec = results.into_inner().unwrap();
    out_vec.sort_by(|a, b| a.cid.cmp(&b.cid).then_with(|| a.entrant.cmp(&b.entrant)));

    let mut f = fs::File::create(&out_jsonl).map_err(|e| format!("create out: {}", e))?;
    for r in &out_vec {
        let preds0_str = r
            .fold0_preds
            .iter()
            .map(|v| format!("{:.8}", v))
            .collect::<Vec<_>>()
            .join(",");
        let preds1_str = r
            .fold1_preds
            .iter()
            .map(|v| format!("{:.8}", v))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(
            f,
            "{{\"cid\":\"{}\",\"entrant\":\"{}\",\"status\":\"{}\",\"rmse_fold0\":{:.10},\"rmse_fold1\":{:.10},\"base_rmse_fold0\":{:.10},\"base_rmse_fold1\":{:.10},\"ratio_fold0\":{:.10},\"ratio_fold1\":{:.10},\"cv_geomean_rmse_ratio\":{:.10},\"logratio\":{:.10},\"tail_logratio\":{:.10},\"sd_logratio\":{:.10},\"cv_fold_wins\":{},\"cv_gate_passed\":{},\"s1_fitness_zero_comp\":{:.10},\"s1_utility_zero_comp\":{:.10},\"unconstrained_utility\":{:.10},\"wall_ms\":{:.3},\"fold0_preds\":[{}],\"fold1_preds\":[{}]}}",
            r.cid,
            r.entrant,
            r.status,
            r.rmse_fold0,
            r.rmse_fold1,
            r.base_rmse_fold0,
            r.base_rmse_fold1,
            r.ratio_fold0,
            r.ratio_fold1,
            r.cv_geomean_rmse_ratio,
            r.logratio,
            r.tail_logratio,
            r.sd_logratio,
            r.cv_fold_wins,
            if r.cv_gate_passed { "true" } else { "false" },
            r.s1_fitness_zero_comp,
            r.s1_utility_zero_comp,
            r.unconstrained_utility,
            r.wall_ms,
            preds0_str,
            preds1_str
        )
        .map_err(|e| format!("write line: {}", e))?;
    }
    eprintln!(
        "Evaluated {} context-entrant pairs across {} contexts in {:.2}s",
        out_vec.len(),
        cids.len(),
        t_start.elapsed().as_secs_f64()
    );
    Ok(())
}
