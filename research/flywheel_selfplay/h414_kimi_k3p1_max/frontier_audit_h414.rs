//! H414 AUDIT-protocol evaluator: the H412 harness (frontier_audit_h412.rs) with a sixth frozen frontier regressor,
//! kimik3p1_max (crates/jope-automl/src/kimik3p1_max.rs, owner file sha256
//! 0c211694ec66605b994595fd8f4cf2b8894df6f156caa3b80f1fcc75f5d6ceb3). Nothing else differs: the five H412
//! entrants, the seed, the parser, the fold assembly and the arithmetic are unchanged.
//!
//! H412 header:
//! H412 AUDIT-protocol evaluator: the H335 harness (frontier_audit_h335.rs, sha256 a2ecafd1a5ef3363412d6cff11de9a4e898595d9760beaa8d6f1ac7ba3a1f2bd)
//! with a fifth
//! frozen frontier regressor, grok4p7_xhigh (crates/jope-automl/src/grok4p7_xhigh.rs, sha256
//! 8e648cef93186aca9d10868ed1ed2a62407950169082aa1f4fab87af379664f8, commit 1177c4f9c1). Nothing else differs:
//! the four H335 entrants, the seed, the parser, the fold assembly and the arithmetic are unchanged.
//!
//! Original H335 header:
//! H335 AUDIT-protocol evaluator for the four frozen Rust frontier regressors.
//!
//! Derived from the H257 harness
//! `research/flywheel_selfplay/h257_phase2_subsample_gp_curriculum/eval_frontier_dl_val91.rs`
//! (sha256 5a7bee26248481e87877ea0fb91f555c03431398c6681a60514bfa3f75219c4a, checkout 46d4e5bd0).
//! The entrant dispatch (`fit_predict_entrant`), the seed (42), the JTT1 parser checks, the
//! per-fold matrix assembly and the RMSE / mean-baseline arithmetic are kept from H257.
//!
//! Differences from H257 (H335 AUDIT protocol):
//!   * one envelope x one entrant per process (the caller imposes RLIMIT_CPU / wall limits);
//!   * only the direction fit-on-fold-0 (SEARCH) -> predict-fold-1 (AUDIT) is computed; fold-1
//!     rows are never passed to `fit`;
//!   * envelopes with `nq != 0` or fold bytes outside {0, 1} are refused (exit code 3);
//!   * a degenerate baseline (empty fold, non-finite or <= 1e-12 mean-predictor RMSE) is
//!     reported as `status: "failed"`, `utility: 0.0` (H257 clamped the baseline to 1e-12);
//!   * a failed or non-finite outcome reports `utility: 0.0` (H257 reported -2.0 / 1e9 ratios);
//!   * an entrant panic is caught (the evaluation runs on a spawned thread, as in H257's
//!     worker threads) and reported as `status: "failed"`.
//!
//! Utility: u = -ln(RMSE_audit / RMSE_base), RMSE_base = RMSE of mean(y_SEARCH) on AUDIT rows,
//! with H257's ratio floor `ratio.max(1e-12)`.
//!
//! Usage: frontier_audit_h335 <envelope.jtt1> <entrant> <out.json|-> [--dump-preds <file>]
//! Exit codes: 0 = record written (completed or failed); 2 = usage; 3 = envelope refused;
//! 4 = output I/O error.

#![allow(dead_code, unused_imports, unused_assignments, unused_variables, unused_mut)]

extern crate gemini4_argon_high;
extern crate gpt6_astra_max;
extern crate gpt6p1_sol_max;
extern crate grok4p7_xhigh;
extern crate kimik3p1_max;
extern crate opus5p5_extra;

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const SEED: u64 = 42;
const HARNESS_VERSION: &str = "h414-frontier-audit-v1";
const ENTRANTS: [&str; 6] = [
    "Opus5p5_Extra",
    "Gemini4_argon_high",
    "GPT6p1_Sol_Max",
    "GPT6_Astra_Max",
    "grok4p7_xhigh",
    "kimik3p1_max",
];
const FOLD_SEARCH: u8 = 0;
const FOLD_AUDIT: u8 = 1;

// Kept verbatim from H257.
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
        "grok4p7_xhigh" => {
            use grok4p7_xhigh::RegressorApi;
            let mut m = grok4p7_xhigh::Regressor::new(seed);
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
    data: Vec<f64>, // row-major nt x (width + 1); column `width` is the target
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

// H257 parse_jtt1 plus the H335 refusals (nq != 0, fold bytes outside {0, 1}).
fn parse_jtt1(path: &Path) -> Result<Jtt1Table, String> {
    let raw = fs::read(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    if raw.len() < 17 || &raw[0..4] != b"JTT1" || raw[16] != 2 {
        return Err(format!("invalid 2-fold JTT1 header in {}", path.display()));
    }
    let nt = read_u32_le(&raw, 4) as usize;
    let nq = read_u32_le(&raw, 8) as usize;
    let width = read_u32_le(&raw, 12) as usize;
    if nq != 0 {
        return Err(format!(
            "refused {}: nq = {} (AUDIT envelopes must carry nq = 0)",
            path.display(),
            nq
        ));
    }
    let cols = width
        .checked_add(1)
        .ok_or_else(|| format!("width overflow in {}", path.display()))?;
    let expected = (nt + nq)
        .checked_mul(cols)
        .and_then(|v| v.checked_mul(8))
        .and_then(|v| v.checked_add(17 + nt))
        .ok_or_else(|| format!("size overflow in {}", path.display()))?;
    if raw.len() != expected {
        return Err(format!(
            "length mismatch in {}: got {}, expected {}",
            path.display(),
            raw.len(),
            expected
        ));
    }
    let folds = raw[17..17 + nt].to_vec();
    if let Some(bad) = folds.iter().find(|&&f| f != FOLD_SEARCH && f != FOLD_AUDIT) {
        return Err(format!(
            "refused {}: fold byte {} not in {{0, 1}}",
            path.display(),
            bad
        ));
    }
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

struct Outcome {
    status: &'static str,
    n_search: usize,
    n_audit: usize,
    rmse_audit: f64,
    base_rmse_audit: f64,
    ratio: f64,
    utility: f64,
    wall_ms: f64,
    error: Option<String>,
    preds: Option<Vec<f64>>,
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn evaluate_audit(tbl: &Jtt1Table, entrant: &str) -> Outcome {
    let t0 = Instant::now();
    let cols = tbl.width + 1;

    // SEARCH rows (fold 0) form the fit matrix, AUDIT rows (fold 1) are predicted.
    let mut train_mat = Vec::new();
    let mut n_train = 0usize;
    let mut y_train_sum = 0.0f64;
    let mut val_feat = Vec::new();
    let mut y_val = Vec::new();
    for r in 0..tbl.nt {
        let row_slice = &tbl.data[r * cols..(r + 1) * cols];
        if tbl.folds[r] == FOLD_SEARCH {
            train_mat.extend_from_slice(row_slice);
            y_train_sum += row_slice[tbl.width];
            n_train += 1;
        } else {
            val_feat.extend_from_slice(&row_slice[0..tbl.width]);
            y_val.push(row_slice[tbl.width]);
        }
    }
    let n_val = y_val.len();

    let mut out = Outcome {
        status: "failed",
        n_search: n_train,
        n_audit: n_val,
        rmse_audit: f64::NAN,
        base_rmse_audit: f64::NAN,
        ratio: f64::NAN,
        utility: 0.0,
        wall_ms: 0.0,
        error: None,
        preds: None,
    };
    if n_train == 0 || n_val == 0 {
        out.error = Some(format!(
            "degenerate split: n_search = {}, n_audit = {}",
            n_train, n_val
        ));
        out.wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        return out;
    }

    let y_train_mean = y_train_sum / (n_train.max(1) as f64);
    let mut base_sse = 0.0f64;
    for &yv in &y_val {
        let d = yv - y_train_mean;
        base_sse += d * d;
    }
    let base_rmse = (base_sse / (n_val.max(1) as f64)).sqrt();
    out.base_rmse_audit = base_rmse;
    if !base_rmse.is_finite() || base_rmse <= 1e-12 {
        out.error = Some(format!(
            "degenerate baseline: base_rmse_audit = {:?}",
            base_rmse
        ));
        out.wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        return out;
    }

    // Run fit/predict on a spawned worker thread (H257 evaluated on scoped worker threads);
    // a panic is reported as a failure instead of aborting without a record.
    let entrant_owned = entrant.to_string();
    let joined = std::thread::Builder::new()
        .name("h335-eval".to_string())
        .spawn(move || {
            fit_predict_entrant(
                &entrant_owned,
                SEED,
                &train_mat,
                n_train,
                cols,
                &val_feat,
                n_val,
            )
        })
        .map_err(|e| format!("spawn worker: {}", e))
        .and_then(|h| h.join().map_err(|p| format!("panic: {}", panic_text(&*p))));

    match joined {
        Ok(Ok(preds)) => {
            if preds.len() != n_val {
                out.error = Some(format!(
                    "prediction length {} != n_audit {}",
                    preds.len(),
                    n_val
                ));
            } else if !preds.iter().all(|v| v.is_finite()) {
                out.error = Some("non-finite prediction".to_string());
            } else {
                let mut sse = 0.0f64;
                for (p, &yv) in preds.iter().zip(y_val.iter()) {
                    let d = *p - yv;
                    sse += d * d;
                }
                let rmse = (sse / (n_val.max(1) as f64)).sqrt();
                let ratio = (rmse / base_rmse.max(1e-12)).max(1e-12);
                let utility = -ratio.ln();
                out.rmse_audit = rmse;
                out.ratio = ratio;
                if rmse.is_finite() && ratio.is_finite() && utility.is_finite() {
                    out.status = "completed";
                    out.utility = utility;
                } else {
                    out.error = Some("non-finite rmse/ratio/utility".to_string());
                }
            }
            out.preds = Some(preds);
        }
        Ok(Err(e)) => out.error = Some(format!("entrant error: {}", e)),
        Err(e) => out.error = Some(e),
    }
    out.wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    out
}

// ---------------------------------------------------------------------------------------------
// Minimal SHA-256 (FIPS 180-4) so the harness stays dependency-free.

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_hex(msg: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (msg.len() as u64).wrapping_mul(8);
    let mut padded = msg.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    let mut w = [0u32; 64];
    for chunk in padded.chunks(64) {
        for t in 0..16 {
            w[t] = u32::from_be_bytes([
                chunk[4 * t],
                chunk[4 * t + 1],
                chunk[4 * t + 2],
                chunk[4 * t + 3],
            ]);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16]
                .wrapping_add(s0)
                .wrapping_add(w[t - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for t in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[t])
                .wrapping_add(w[t]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|v| format!("{:08x}", v)).collect()
}

fn preds_le_bytes(preds: &[f64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(preds.len() * 8);
    for v in preds {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

// ---------------------------------------------------------------------------------------------
// JSON output (fixed key order; floats use Rust's shortest round-trip representation).

fn json_f64(v: f64) -> String {
    if v.is_finite() {
        format!("{:?}", v)
    } else {
        "null".to_string()
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_opt_str(s: &Option<String>) -> String {
    match s {
        Some(v) => json_str(v),
        None => "null".to_string(),
    }
}

fn record_json(cid: &str, entrant: &str, o: &Outcome, preds_sha: &Option<String>) -> String {
    format!(
        "{{\"cid\":{},\"entrant\":{},\"seed\":{},\"status\":{},\"n_search\":{},\"n_audit\":{},\"rmse_audit\":{},\"base_rmse_audit\":{},\"ratio\":{},\"utility\":{},\"wall_ms\":{},\"error\":{},\"audit_preds_sha256\":{},\"harness_version\":{}}}\n",
        json_str(cid),
        json_str(entrant),
        SEED,
        json_str(o.status),
        o.n_search,
        o.n_audit,
        json_f64(o.rmse_audit),
        json_f64(o.base_rmse_audit),
        json_f64(o.ratio),
        json_f64(o.utility),
        json_f64(o.wall_ms),
        json_opt_str(&o.error),
        json_opt_str(preds_sha),
        json_str(HARNESS_VERSION),
    )
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".tmp.{}", std::process::id()));
    let tmp = PathBuf::from(tmp);
    {
        let mut f =
            fs::File::create(&tmp).map_err(|e| format!("create {}: {}", tmp.display(), e))?;
        f.write_all(bytes)
            .map_err(|e| format!("write {}: {}", tmp.display(), e))?;
        f.sync_all()
            .map_err(|e| format!("sync {}: {}", tmp.display(), e))?;
    }
    fs::rename(&tmp, path).map_err(|e| format!("rename to {}: {}", path.display(), e))
}

fn usage(prog: &str) -> String {
    format!(
        "Usage: {} <envelope.jtt1> <entrant> <out.json|-> [--dump-preds <file>]\n  entrants: {}",
        prog,
        ENTRANTS.join(", ")
    )
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let prog = args.first().map(String::as_str).unwrap_or("frontier_audit_h335");
    if args.len() != 4 && !(args.len() == 6 && args[4] == "--dump-preds") {
        eprintln!("{}", usage(prog));
        return ExitCode::from(2);
    }
    let envelope = PathBuf::from(&args[1]);
    let entrant = args[2].as_str();
    let out_path = args[3].as_str();
    let dump_preds = if args.len() == 6 {
        Some(PathBuf::from(&args[5]))
    } else {
        None
    };
    if !ENTRANTS.contains(&entrant) {
        eprintln!("unknown entrant {:?}\n{}", entrant, usage(prog));
        return ExitCode::from(2);
    }
    let cid = match envelope.file_stem().and_then(|s| s.to_str()) {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => {
            eprintln!("cannot derive cid from {}", envelope.display());
            return ExitCode::from(2);
        }
    };

    let tbl = match parse_jtt1(&envelope) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("envelope refused: {}", e);
            return ExitCode::from(3);
        }
    };

    let outcome = evaluate_audit(&tbl, entrant);
    let preds_sha = outcome
        .preds
        .as_ref()
        .map(|p| sha256_hex(&preds_le_bytes(p)));

    if let (Some(dp), Some(p)) = (dump_preds.as_ref(), outcome.preds.as_ref()) {
        if let Err(e) = write_atomic(dp, &preds_le_bytes(p)) {
            eprintln!("{}", e);
            return ExitCode::from(4);
        }
    }

    let line = record_json(&cid, entrant, &outcome, &preds_sha);
    if out_path == "-" {
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        if let Err(e) = lock.write_all(line.as_bytes()).and_then(|_| lock.flush()) {
            eprintln!("write stdout: {}", e);
            return ExitCode::from(4);
        }
    } else if let Err(e) = write_atomic(Path::new(out_path), line.as_bytes()) {
        eprintln!("{}", e);
        return ExitCode::from(4);
    }
    ExitCode::SUCCESS
}
