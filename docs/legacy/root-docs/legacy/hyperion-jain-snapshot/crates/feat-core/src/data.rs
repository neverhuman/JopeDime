//! Data layer: a column-major float matrix, dataset loading from CSV, and stratified splits.
//!
//! COLUMN-MAJOR is deliberate: every hot operation (GP terminal eval, importance blocks, mask
//! columns, correlation) slices a *feature*, which is then a contiguous `&[f32]`. This also matches
//! the CatBoost shim's ingest, so the matrix is handed to the FFI with zero transpose.
use crate::rng::seeded;
use anyhow::{bail, Context, Result};
use rand::seq::SliceRandom;
use std::path::Path;
use std::sync::Arc;

/// Column-major matrix: column `j` occupies `data[j*n .. j*n + n]`.
#[derive(Clone)]
pub struct Matrix {
    pub n: usize,
    pub c: usize,
    pub data: Arc<Vec<f32>>,
    /// Lazily-materialized NaN column, returned by `col()` for an out-of-range index so a stale/wider
    /// column reference degrades to a missing feature instead of panicking with a slice overrun.
    nan_pad: std::sync::OnceLock<Vec<f32>>,
}

impl Matrix {
    pub fn new(data: Vec<f32>, n: usize, c: usize) -> Self {
        assert_eq!(data.len(), n * c, "matrix data length mismatch");
        Matrix {
            n,
            c,
            data: Arc::new(data),
            nan_pad: std::sync::OnceLock::new(),
        }
    }
    /// Column `j` over all rows. An out-of-range `j` (e.g. a stale pre-leak-drop / wider-frame index) is
    /// logged with a one-time backtrace and returned as a NaN "missing" column — NEVER a panic.
    #[inline]
    pub fn col(&self, j: usize) -> &[f32] {
        if j < self.c {
            &self.data[j * self.n..(j + 1) * self.n]
        } else {
            warn_matrix_oob("col", self.n, self.c, j, 0);
            &self.nan_pad.get_or_init(|| vec![f32::NAN; self.n])[..self.n]
        }
    }
    #[inline]
    pub fn at(&self, row: usize, col: usize) -> f32 {
        if col < self.c && row < self.n {
            self.data[col * self.n + row]
        } else {
            f32::NAN
        }
    }
    /// Select columns over ALL rows (column-major). Used for full-train / holdout fits.
    /// Out-of-range column indices are treated as a missing feature (NaN) rather than panicking, and are
    /// logged with full context so the caller can be diagnosed. Never crashes on a bad index.
    pub fn select_cols(&self, cols: &[u32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n * cols.len());
        for &c in cols {
            let c = c as usize;
            if c < self.c {
                out.extend_from_slice(self.col(c));
            } else {
                warn_matrix_oob("select_cols", self.n, self.c, c, 0);
                out.extend(std::iter::repeat_n(f32::NAN, self.n));
            }
        }
        out
    }

    /// Materialize a dense COLUMN-MAJOR submatrix of (rows x cols) for the FFI (only copy point).
    /// Out-of-range row/col indices become NaN (missing) with a logged warning instead of a panic.
    pub fn submatrix(&self, rows: &[usize], cols: &[u32]) -> Vec<f32> {
        let (m, p) = (rows.len(), cols.len());
        let mut out = vec![f32::NAN; m * p];
        for (cj, &c) in cols.iter().enumerate() {
            let c = c as usize;
            if c >= self.c {
                warn_matrix_oob("submatrix.col", self.n, self.c, c, m);
                continue; // leave this output column as NaN
            }
            let src = self.col(c);
            let dst = &mut out[cj * m..(cj + 1) * m];
            for (di, &r) in rows.iter().enumerate() {
                if r < self.n {
                    dst[di] = src[r];
                } else {
                    warn_matrix_oob("submatrix.row", self.n, self.c, r, m);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod matrix_oob_tests {
    use super::*;
    // select_cols / submatrix must NEVER panic on an out-of-range column: they yield NaN (missing).
    // The reported crash shape: col(j) with j == column count on an n-row frame ("range end index out of
    // range"). col()/at() must degrade to NaN, never panic.
    #[test]
    fn col_and_at_out_of_range_are_nan_not_panic() {
        let m = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2); // 3 rows x 2 cols
        assert_eq!(m.col(1), &[4.0, 5.0, 6.0]); // in-range unchanged
        let c = m.col(9); // absent column
        assert_eq!(c.len(), 3);
        assert!(c.iter().all(|v| v.is_nan()));
        assert_eq!(m.at(2, 0), 3.0); // in-range
        assert!(m.at(0, 9).is_nan()); // col out of range
        assert!(m.at(9, 0).is_nan()); // row out of range
    }

    #[test]
    fn out_of_range_col_is_nan_not_panic() {
        let m = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2); // 3 rows x 2 cols
        let sc = m.select_cols(&[0, 5]); // col 5 absent
        assert_eq!(&sc[0..3], &[1.0, 2.0, 3.0]);
        assert!(sc[3..6].iter().all(|v| v.is_nan()));
        let sm = m.submatrix(&[0, 2, 9], &[1, 7]); // col 7 absent, row 9 absent
        assert_eq!(sm[0], 4.0); // (row0,col1)
        assert_eq!(sm[1], 6.0); // (row2,col1)
        assert!(sm[2].is_nan()); // row 9 out of range
        assert!(sm[3..6].iter().all(|v| v.is_nan())); // col 7 out of range
    }
}

/// Log a Matrix out-of-range access with full context, and a ONE-TIME backtrace so the offending caller
/// is captured the first time it happens (subsequent hits log just the dims/index to avoid spam).
fn warn_matrix_oob(what: &str, n: usize, c: usize, idx: usize, m: usize) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static BT_LOGGED: AtomicBool = AtomicBool::new(false);
    eprintln!(
        "[matrix-oob] {what}: requested index {idx} on matrix {n}x{c} (valid cols 0..{c}, rows 0..{n}; \
         request m={m}) -> filling NaN (missing) instead of panicking"
    );
    if !BT_LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[matrix-oob] first occurrence backtrace:\n{}",
            std::backtrace::Backtrace::force_capture()
        );
    }
}

/// Optional multi-target labels (for Phase 4 MIMO). Column-major `k*n`; `primary` indexes the
/// scored target (== the dataset's `y`).
#[derive(Clone)]
pub struct MultiTargets {
    pub data: Vec<f32>,
    pub k: usize,
    pub primary: usize,
    pub names: Vec<String>,
}

/// A loaded dataset: design matrix + primary binary target (+ optional multi-targets).
#[derive(Clone)]
pub struct Dataset {
    pub x: Matrix,
    pub y: Vec<f32>,
    pub y_multi: Option<MultiTargets>,
    pub names: Vec<String>,
}

impl Dataset {
    pub fn n(&self) -> usize {
        self.x.n
    }
    pub fn p(&self) -> usize {
        self.x.c
    }
}

pub(crate) fn parse_field(s: &str) -> f32 {
    let t = s.trim();
    if t.is_empty() {
        return f32::NAN;
    }
    t.parse::<f32>().unwrap_or(f32::NAN)
}

/// Load a CSV with a header row. `targets` names the label column(s); `primary` is the scored one.
/// All other columns become numeric features (non-numeric/empty -> NaN, treated as missing).
pub fn load_csv(path: &Path, targets: &[&str], primary: &str) -> Result<Dataset> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path:?}"))?;
    let mut lines = text.lines();
    let header = lines.next().context("empty CSV")?;
    let cols: Vec<&str> = header.split(',').map(|s| s.trim()).collect();
    let ncol = cols.len();

    let tgt_idx: Vec<usize> = targets
        .iter()
        .map(|t| {
            cols.iter()
                .position(|c| c == t)
                .with_context(|| format!("target column '{t}' not in header"))
        })
        .collect::<Result<_>>()?;
    let primary_pos = targets
        .iter()
        .position(|t| t == &primary)
        .context("primary not in targets")?;
    let is_target: Vec<bool> = (0..ncol).map(|i| tgt_idx.contains(&i)).collect();
    let feat_cols: Vec<usize> = (0..ncol).filter(|&i| !is_target[i]).collect();
    let names: Vec<String> = feat_cols.iter().map(|&i| cols[i].to_string()).collect();
    let p = feat_cols.len();
    let k = tgt_idx.len();

    // Parse rows into column-major buffers.
    let mut feat_data: Vec<Vec<f32>> = vec![Vec::new(); p];
    let mut tgt_data: Vec<Vec<f32>> = vec![Vec::new(); k];
    let mut n = 0usize;
    for (lineno, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != ncol {
            bail!(
                "row {} has {} fields, expected {ncol}",
                lineno + 2,
                fields.len()
            );
        }
        for (fi, &ci) in feat_cols.iter().enumerate() {
            feat_data[fi].push(parse_field(fields[ci]));
        }
        for (ti, &ci) in tgt_idx.iter().enumerate() {
            // labels: round to {0,1}
            let v = parse_field(fields[ci]);
            tgt_data[ti].push(if v > 0.5 { 1.0 } else { 0.0 });
        }
        n += 1;
    }
    if n == 0 {
        bail!("CSV has no data rows");
    }

    let mut flat = Vec::with_capacity(n * p);
    for col in &feat_data {
        flat.extend_from_slice(col);
    }
    let x = Matrix::new(flat, n, p);
    let y = tgt_data[primary_pos].clone();

    let y_multi = if k > 1 {
        let mut md = Vec::with_capacity(n * k);
        for col in &tgt_data {
            md.extend_from_slice(col);
        }
        Some(MultiTargets {
            data: md,
            k,
            primary: primary_pos,
            names: targets.iter().map(|s| s.to_string()).collect(),
        })
    } else {
        None
    };

    Ok(Dataset {
        x,
        y,
        y_multi,
        names,
    })
}

/// A typed train/holdout load: encoders fit on TRAIN, applied to both (holdout firewall preserved),
/// plus the inferred task and the indices of categorical feature columns.
pub struct LoadedPair {
    pub train: Dataset,
    pub hold: Dataset,
    pub task: crate::task::TaskSpec,
    pub cat_cols: Vec<usize>,
    pub encoders: crate::encode::Encoders,
    pub hyperion: Option<crate::hyperion::HyperionState>,
    pub preprocess_issues: Vec<String>,
}

/// Typed train+holdout loader: classifies columns, encodes categoricals fold-safely (train-fit), and
/// auto-detects the task. On an all-numeric, binary-{0,1} file this is byte-identical to two `load_csv`
/// calls; on a file with string columns it *uses* them (encoded) instead of silently dropping them.
pub fn load_train_hold(
    train_path: &Path,
    hold_path: &Path,
    targets: &[&str],
    primary: &str,
    enc: crate::encode::CatEncoding,
) -> Result<LoadedPair> {
    load_train_hold_with(train_path, hold_path, targets, primary, enc, true)
}

/// Like `load_train_hold`, but `has_header=false` reads BOTH files as headerless (synthesized
/// `f0..`/`target` names). Train & hold must share the same schema/column count.
pub fn load_train_hold_with(
    train_path: &Path,
    hold_path: &Path,
    targets: &[&str],
    primary: &str,
    enc: crate::encode::CatEncoding,
    has_header: bool,
) -> Result<LoadedPair> {
    let hyperion = crate::hyperion::HyperionConfig::from_env()?;
    load_train_hold_with_hyperion(
        train_path,
        hold_path,
        targets,
        primary,
        enc,
        has_header,
        &hyperion,
        &crate::progress::NullSink,
    )
}

/// Like [`load_train_hold_with`], but streams phase-0 ("HYPERION" encoding) progress on `sink`.
pub fn load_train_hold_with_progress(
    train_path: &Path,
    hold_path: &Path,
    targets: &[&str],
    primary: &str,
    enc: crate::encode::CatEncoding,
    has_header: bool,
    sink: &dyn crate::progress::Sink,
) -> Result<LoadedPair> {
    let hyperion = crate::hyperion::HyperionConfig::from_env()?;
    load_train_hold_with_hyperion(
        train_path, hold_path, targets, primary, enc, has_header, &hyperion, sink,
    )
}

/// Configurable train/holdout loader used by CLI surfaces that expose Hyperion flags.
#[allow(clippy::too_many_arguments)]
pub fn load_train_hold_with_hyperion(
    train_path: &Path,
    hold_path: &Path,
    targets: &[&str],
    primary: &str,
    enc: crate::encode::CatEncoding,
    has_header: bool,
    hyperion: &crate::hyperion::HyperionConfig,
    sink: &dyn crate::progress::Sink,
) -> Result<LoadedPair> {
    use crate::encode::{build_targets, fit_preprocess_plan, transform};
    use crate::ingest::read_frame_with;
    use crate::task::{infer_task, TaskInferConfig};

    let t0 = std::time::Instant::now();
    let ftr = read_frame_with(train_path, targets, primary, has_header)?;
    let fho = read_frame_with(hold_path, targets, primary, has_header)?;

    // `encode::transform` matches columns BY NAME and silently zero-fills a missing name, so a
    // train/hold width mismatch would corrupt the holdout instead of erroring. Guard explicitly.
    if ftr.feats.len() != fho.feats.len() {
        bail!(
            "train has {} feature columns but holdout has {}; train and hold must share the same schema/column count",
            ftr.feats.len(),
            fho.feats.len()
        );
    }

    let tgt_parsed: Vec<Vec<f32>> = ftr
        .targets
        .iter()
        .map(|c| c.tokens.iter().map(|t| parse_field(t)).collect())
        .collect();
    let task = infer_task(&tgt_parsed, ftr.primary, &TaskInferConfig::default());

    let encoders = fit_preprocess_plan(&ftr, &task, enc);
    let (xtr, names, cat_cols) = transform(&ftr, &encoders);
    let (xho, _, _) = transform(&fho, &encoders);
    let hyp = crate::hyperion::fit_transform_pair(&xtr, &xho, &names, hyperion, sink, t0)?;
    let (ytr, ytr_multi) = build_targets(&ftr, &task, &encoders);
    let (yho, yho_multi) = build_targets(&fho, &task, &encoders);
    let preprocess_issues = encoders.issues.clone();

    Ok(LoadedPair {
        train: Dataset {
            x: hyp.train,
            y: ytr,
            y_multi: ytr_multi,
            names: hyp.names.clone(),
        },
        hold: Dataset {
            x: hyp.hold,
            y: yho,
            y_multi: yho_multi,
            names: hyp.names,
        },
        task,
        cat_cols,
        encoders,
        hyperion: hyp.state,
        preprocess_issues,
    })
}

/// Load one CSV and create a deterministic internal holdout. Encoders are fit on the internal train
/// rows only, then replayed onto the holdout rows.
pub fn load_train_split(
    path: &Path,
    targets: &[&str],
    primary: &str,
    enc: crate::encode::CatEncoding,
    holdout_frac: f64,
    seed: u64,
) -> Result<LoadedPair> {
    let hyperion = crate::hyperion::HyperionConfig::from_env()?;
    load_train_split_with_hyperion(
        path,
        targets,
        primary,
        enc,
        holdout_frac,
        seed,
        &hyperion,
        &crate::progress::NullSink,
    )
}

/// Like [`load_train_split`], but streams phase-0 ("HYPERION" encoding) progress on `sink`.
#[allow(clippy::too_many_arguments)]
pub fn load_train_split_with_progress(
    path: &Path,
    targets: &[&str],
    primary: &str,
    enc: crate::encode::CatEncoding,
    holdout_frac: f64,
    seed: u64,
    sink: &dyn crate::progress::Sink,
) -> Result<LoadedPair> {
    let hyperion = crate::hyperion::HyperionConfig::from_env()?;
    load_train_split_with_hyperion(
        path,
        targets,
        primary,
        enc,
        holdout_frac,
        seed,
        &hyperion,
        sink,
    )
}

/// Configurable train/split loader used by CLI surfaces that expose Hyperion flags.
#[allow(clippy::too_many_arguments)]
pub fn load_train_split_with_hyperion(
    path: &Path,
    targets: &[&str],
    primary: &str,
    enc: crate::encode::CatEncoding,
    holdout_frac: f64,
    seed: u64,
    hyperion: &crate::hyperion::HyperionConfig,
    sink: &dyn crate::progress::Sink,
) -> Result<LoadedPair> {
    use crate::encode::{build_targets, fit_preprocess_plan, transform};
    use crate::ingest::{read_frame, Frame, RawColumn};
    use crate::task::{infer_task, TaskInferConfig, TaskKind};

    let t0 = std::time::Instant::now();

    fn subset_col(c: &RawColumn, rows: &[usize]) -> RawColumn {
        RawColumn {
            name: c.name.clone(),
            kind: c.kind,
            tokens: rows.iter().map(|&i| c.tokens[i].clone()).collect(),
        }
    }
    fn subset_frame(f: &Frame, rows: &[usize]) -> Frame {
        Frame {
            feats: f.feats.iter().map(|c| subset_col(c, rows)).collect(),
            targets: f.targets.iter().map(|c| subset_col(c, rows)).collect(),
            primary: f.primary,
            n: rows.len(),
            issues: f.issues.clone(),
        }
    }
    fn random_split(n: usize, frac: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut seeded(seed));
        let mut h = (n as f64 * frac).round() as usize;
        if n > 1 {
            h = h.clamp(1, n - 1);
        }
        let hold = idx[..h].to_vec();
        let train = idx[h..].to_vec();
        (train, hold)
    }
    fn split_indices(y: &[f32], kind: TaskKind, frac: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
        if !matches!(kind, TaskKind::Binary | TaskKind::MultiLabel) {
            return random_split(y.len(), frac, seed);
        }
        let mut rng = seeded(seed);
        let mut pos: Vec<usize> = (0..y.len()).filter(|&i| y[i] > 0.5).collect();
        let mut neg: Vec<usize> = (0..y.len()).filter(|&i| y[i] <= 0.5).collect();
        pos.shuffle(&mut rng);
        neg.shuffle(&mut rng);
        let h_pos = (pos.len() as f64 * frac).round() as usize;
        let h_neg = (neg.len() as f64 * frac).round() as usize;
        let mut hold = Vec::new();
        hold.extend_from_slice(&pos[..h_pos.min(pos.len())]);
        hold.extend_from_slice(&neg[..h_neg.min(neg.len())]);
        let mut train = Vec::new();
        train.extend_from_slice(&pos[h_pos.min(pos.len())..]);
        train.extend_from_slice(&neg[h_neg.min(neg.len())..]);
        if hold.is_empty() || train.is_empty() {
            return random_split(y.len(), frac, seed);
        }
        hold.shuffle(&mut seeded(seed ^ 0xA1));
        train.shuffle(&mut seeded(seed ^ 0xB2));
        (train, hold)
    }

    let frame = read_frame(path, targets, primary)?;
    let tgt_parsed: Vec<Vec<f32>> = frame
        .targets
        .iter()
        .map(|c| c.tokens.iter().map(|t| parse_field(t)).collect())
        .collect();
    let task = infer_task(&tgt_parsed, frame.primary, &TaskInferConfig::default());
    let y_all: Vec<f32> = frame.targets[frame.primary]
        .tokens
        .iter()
        .map(|t| {
            let v = parse_field(t);
            if matches!(task.kind, TaskKind::Regression) {
                v
            } else {
                task.class_index(v)
            }
        })
        .collect();
    let (train_idx, hold_idx) = split_indices(&y_all, task.kind, holdout_frac, seed);
    let ftr = subset_frame(&frame, &train_idx);
    let fho = subset_frame(&frame, &hold_idx);
    let encoders = fit_preprocess_plan(&ftr, &task, enc);
    let (xtr, names, cat_cols) = transform(&ftr, &encoders);
    let (xho, _, _) = transform(&fho, &encoders);
    let hyp = crate::hyperion::fit_transform_pair(&xtr, &xho, &names, hyperion, sink, t0)?;
    let (ytr, ytr_multi) = build_targets(&ftr, &task, &encoders);
    let (yho, yho_multi) = build_targets(&fho, &task, &encoders);
    let preprocess_issues = encoders.issues.clone();

    Ok(LoadedPair {
        train: Dataset {
            x: hyp.train,
            y: ytr,
            y_multi: ytr_multi,
            names: hyp.names.clone(),
        },
        hold: Dataset {
            x: hyp.hold,
            y: yho,
            y_multi: yho_multi,
            names: hyp.names,
        },
        task,
        cat_cols,
        encoders,
        hyperion: hyp.state,
        preprocess_issues,
    })
}

/// Stratified train/holdout split by the primary target. `holdout_frac` in (0,1).
pub fn split_holdout(ds: &Dataset, holdout_frac: f64, seed: u64) -> (Dataset, Dataset) {
    let n = ds.n();
    let mut rng = seeded(seed);
    let mut pos: Vec<usize> = (0..n).filter(|&i| ds.y[i] > 0.5).collect();
    let mut neg: Vec<usize> = (0..n).filter(|&i| ds.y[i] <= 0.5).collect();
    pos.shuffle(&mut rng);
    neg.shuffle(&mut rng);
    let h_pos = (pos.len() as f64 * holdout_frac).round() as usize;
    let h_neg = (neg.len() as f64 * holdout_frac).round() as usize;
    let mut hold_idx: Vec<usize> = Vec::new();
    hold_idx.extend_from_slice(&pos[..h_pos]);
    hold_idx.extend_from_slice(&neg[..h_neg]);
    let mut train_idx: Vec<usize> = Vec::new();
    train_idx.extend_from_slice(&pos[h_pos..]);
    train_idx.extend_from_slice(&neg[h_neg..]);
    hold_idx.shuffle(&mut seeded(seed ^ 0xA1));
    train_idx.shuffle(&mut seeded(seed ^ 0xB2));
    (subset(ds, &train_idx), subset(ds, &hold_idx))
}

/// Build a new Dataset from a row subset (used by split_holdout).
fn subset(ds: &Dataset, rows: &[usize]) -> Dataset {
    let n = rows.len();
    let p = ds.p();
    let mut flat = Vec::with_capacity(n * p);
    for j in 0..p {
        let src = ds.x.col(j);
        for &r in rows {
            flat.push(src[r]);
        }
    }
    let x = Matrix::new(flat, n, p);
    let y: Vec<f32> = rows.iter().map(|&r| ds.y[r]).collect();
    let y_multi = ds.y_multi.as_ref().map(|m| {
        let mut md = Vec::with_capacity(n * m.k);
        for t in 0..m.k {
            let src = &m.data[t * ds.n()..(t + 1) * ds.n()];
            for &r in rows {
                md.push(src[r]);
            }
        }
        MultiTargets {
            data: md,
            k: m.k,
            primary: m.primary,
            names: m.names.clone(),
        }
    });
    Dataset {
        x,
        y,
        y_multi,
        names: ds.names.clone(),
    }
}
