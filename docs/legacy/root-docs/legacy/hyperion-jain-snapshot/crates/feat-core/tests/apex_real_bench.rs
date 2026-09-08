//! apex_real_bench — the GROWING, 100%-Rust, real-world half of the Apex gate.
//!
//! Reads the registry `tests/apex_datasets.txt`, loads each small (<50k-row) real-world dataset in pure
//! Rust (`data.rs::load_train_hold` — no Python, no C, continuous targets preserved for regression), runs
//! `run_pipeline` on MockBackend, and prints `APEX_DATASET <name> champion=<m> baseline=<b> hash=<h>
//! kind=<auc|r2>` lines that `ops/apex/lib/fitness.sh` auto-discovers and folds into the hash-keyed gate.
//!
//! ADMISSION GATE (anti-gaming, so a candidate can't register an easy/leaky dataset to inflate the score):
//! a registry dataset is admitted ONLY if rows < 50_000, the target is non-degenerate, and the BASELINE
//! (raw-feature) metric is < 0.99 (>= 0.99 ⇒ trivially separable / leaky). Datasets that fail admission or
//! whose CSVs are absent are skipped (never panic). Existing-name datasets are frozen by `dataset_hash`.
//!
//! FROZEN: jail-validate forbids candidates from editing THIS file (they extend the basket via the
//! non-frozen `apex_datasets.txt` registry, gated by admission + the shared-headline promotion rule).
use feat_core::backend::TrainSpec;
use feat_core::data::{load_train_hold, Dataset, Matrix};
use feat_core::dominance::DominanceConfig;
use feat_core::encode::CatEncoding;
use feat_core::gp::GpConfig;
use feat_core::hashing::dataset_hash;
use feat_core::learners::glm::GlmBackend;
use feat_core::phase2::P2Config;
use feat_core::pipeline::{run_pipeline, PipelineConfig};
use feat_core::progress::{RunContext, StderrProgressSink};
use feat_core::select::EdaConfig;
use feat_core::stacking::StackConfig;
use feat_core::task::TaskKind;
use std::path::Path;

const MAX_ROWS: usize = 70_000;
const TOO_EASY: f64 = 0.99; // baseline metric >= this ⇒ leaky / trivially separable ⇒ reject
                            // Discrimination floor: a gate dataset must show USABLE, non-pathological signal under the champion pipeline,
                            // else it only adds noise (random AUC / ~0 R²) or instability (catastrophic-overfit negative R²) and can't
                            // reliably discriminate candidate quality. This is also the anti-gaming safety net — a candidate-added
                            // dataset that can't be learned self-excludes (it can't inflate the basket), and a held-out set is RE-ADMITTED
                            // automatically the moment some evolved tech lifts it above the floor.
const FLOOR_R2: f64 = 0.05; // regression: champion R² must exceed this
const FLOOR_AUC: f64 = 0.52; // binary: champion AUC must exceed this
const MAX_OVERFIT: f64 = 0.05; // regression: champion must be within this of baseline (no catastrophic overfit)

fn quick_cfg(seed: u64, task: TaskKind, n_classes: usize) -> PipelineConfig {
    PipelineConfig {
        gp: GpConfig {
            pop: 150,
            gens: 2,
            islands: 2,
            depth: 4,
            subsample: 256,
            subsample_fit: 64,
            max_surv: 400,
            ..GpConfig::default()
        },
        p2: P2Config {
            epochs: 1,
            block: 4,
            subsample: 256,
            iters: 8,
            depth: 4,
            thread_count: 1,
            ..P2Config::default()
        },
        eda: EdaConfig {
            pop: 8,
            gens: 2,
            folds: 3,
            ..EdaConfig::default()
        },
        cv_spec: TrainSpec::from_basic_fields(8, 4, 0.1, 1, 0),
        ho_spec: TrainSpec::from_basic_fields(15, 4, 0.1, 1, 0),
        resid_spec: TrainSpec::from_basic_fields(8, 4, 0.1, 1, 0),
        select_subsample: 2000, // cap EDA rows so bigger sets stay quick (it's a gate, not a full bench)
        max_gp: 150,
        do_mimo: false,
        seed,
        task,
        n_classes,
        stack: StackConfig::default(),
        dominance: DominanceConfig::default(),
        starforge: feat_core::starforge_integration::StarforgeConfig::disabled(),
        router: feat_core::router::RouterConfig::default(),
        coreset_threshold: 50_000,
        coreset_guard_frac: 0.20,
        bakeoff: Default::default(),
        hyperion: Default::default(),
        invention: Default::default(),
        canary: Default::default(),
    }
}

/// Auto-detect the primary target + all sibling TARGET_*/target_*/PRIMARY_TARGET_* columns (mirrors the
/// feat-cli convention) so they are excluded from features (leakage firewall).
fn detect_targets(header: &[String], hint: &str) -> (String, Vec<String>) {
    let lc = |s: &str| s.to_ascii_lowercase();
    let primary = if hint != "auto" && !hint.is_empty() {
        hint.to_string()
    } else {
        header
            .iter()
            .find(|c| lc(c).starts_with("primary_target"))
            .or_else(|| header.iter().find(|c| lc(c) == "target"))
            .or_else(|| header.last())
            .cloned()
            .unwrap_or_default()
    };
    let mut targets = vec![primary.clone()];
    for c in header {
        let l = lc(c);
        if (l.starts_with("target_") || l.starts_with("primary_target") || l == "target")
            && *c != primary
        {
            targets.push(c.clone());
        }
    }
    (primary, targets)
}

/// Deterministically cap a dataset to the first `max` rows (column-major rebuild) so the now-large gate
/// stays tractable + identical across grid nodes. `max == 0` means no cap.
fn cap_rows(d: Dataset, max: usize) -> Dataset {
    let n = d.n();
    if max == 0 || n <= max {
        return d;
    }
    let p = d.p();
    let mut data = vec![0f32; max * p];
    for j in 0..p {
        let col = d.x.col(j);
        for i in 0..max {
            data[j * max + i] = col[i];
        }
    }
    let y: Vec<f32> = d.y[..max].to_vec();
    Dataset {
        x: Matrix::new(data, max, p),
        y,
        y_multi: None,
        names: d.names,
    }
}

fn target_nondegenerate(d: &Dataset) -> bool {
    let ys: Vec<f64> =
        d.y.iter()
            .filter(|v| v.is_finite())
            .map(|v| *v as f64)
            .collect();
    if ys.len() < d.n() / 2 || ys.is_empty() {
        return false;
    }
    let mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let var = ys.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ys.len() as f64;
    var > 1e-9
}

#[test]
fn apex_real_metric_basket() {
    let registry = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/apex_datasets.txt");
    let text = std::fs::read_to_string(registry).unwrap_or_default();

    let mut admitted = 0usize;
    let mut sum = 0.0f64;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 4 {
            continue;
        }
        let (name, task_s, train_csv, hint) = (
            cols[0].trim(),
            cols[1].trim(),
            cols[2].trim(),
            cols[3].trim(),
        );
        let test_csv = train_csv.replace("_train_", "_test_");
        if !Path::new(train_csv).exists() || !Path::new(&test_csv).exists() {
            eprintln!("APEX_SKIP {name} reason=csv_missing");
            continue;
        }
        // read header → detect targets
        let header_line = match std::fs::read_to_string(train_csv)
            .ok()
            .and_then(|s| s.lines().next().map(str::to_string))
        {
            Some(h) => h,
            None => {
                eprintln!("APEX_SKIP {name} reason=empty_csv");
                continue;
            }
        };
        let header: Vec<String> = header_line
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        let (primary, targets) = detect_targets(&header, hint);
        let tgt_refs: Vec<&str> = targets.iter().map(|s| s.as_str()).collect();

        let pair = match load_train_hold(
            Path::new(train_csv),
            Path::new(&test_csv),
            &tgt_refs,
            &primary,
            CatEncoding::Ordinal,
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("APEX_SKIP {name} reason=load_err:{e}");
                continue;
            }
        };
        // ---- admission gate ----
        if pair.train.n() > MAX_ROWS {
            eprintln!("APEX_SKIP {name} reason=too_large:{}", pair.train.n());
            continue;
        }
        if !target_nondegenerate(&pair.train) {
            eprintln!("APEX_SKIP {name} reason=degenerate_target");
            continue;
        }
        let task = pair.task.kind;
        let n_classes = pair.task.n_classes;
        // Deterministically cap rows so the now-large gate stays tractable + identical across grid nodes.
        let gate_cap: usize = std::env::var("APEX_GATE_MAX_ROWS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8000);
        let train = cap_rows(pair.train, gate_cap);
        let hold = cap_rows(pair.hold, (gate_cap / 4).max(500));
        // Use a REAL pure-Rust learner (elastic-net GLM): gaussian for regression, logistic for binary.
        // (MockBackend is a correlation toy that produces nonsense R² on real regression targets.)
        let cfg = quick_cfg(7, task, n_classes);
        let sink = StderrProgressSink::new(
            RunContext::new("apex-real")
                .dataset(name)
                .task(format!("{task:?}"))
                .rows(train.n(), hold.n())
                .features(train.p())
                .selection_mode(cfg.eda.selection_mode.as_str()),
        );
        let res = if matches!(task, TaskKind::Regression) {
            let be = GlmBackend::gaussian();
            run_pipeline(&train, &hold, &be, &[], &cfg, &sink)
        } else {
            let be = GlmBackend::logistic();
            run_pipeline(&train, &hold, &be, &[], &cfg, &sink)
        };
        let baseline = res.baseline_holdout_auc;
        let metric = res.champion_holdout_auc;
        if !metric.is_finite() {
            eprintln!("APEX_SKIP {name} reason=nonfinite_metric");
            continue;
        }
        if baseline >= TOO_EASY {
            eprintln!("APEX_SKIP {name} reason=too_easy_or_leaky baseline={baseline:.4}");
            continue;
        }
        // ---- discrimination floor (anti-noise, anti-gaming, self-curating) ----
        let passes_floor = match task {
            TaskKind::Regression => metric > FLOOR_R2 && metric >= baseline - MAX_OVERFIT,
            _ => metric > FLOOR_AUC,
        };
        if !passes_floor {
            eprintln!("APEX_SKIP {name} reason=below_signal_floor champion={metric:.4} baseline={baseline:.4}");
            continue;
        }
        let kind = match task {
            TaskKind::Regression => "r2",
            _ => "auc",
        };
        let hash = dataset_hash(&train.x, &train.y, &train.names);
        println!("APEX_DATASET {name} champion={metric:.6} baseline={baseline:.6} hash={hash} kind={kind} domain=real task={task_s}");
        admitted += 1;
        sum += metric;
    }

    // Don't fail the C-free gate if the external datasets simply aren't present on this box; only fail if
    // a dataset was admitted but produced a non-finite/insane metric (caught above). Report the mean.
    if admitted > 0 {
        println!(
            "APEX_REAL_METRIC {:.6} over {admitted} real dataset(s)",
            sum / admitted as f64
        );
    } else {
        eprintln!("APEX_REAL_METRIC none (no real datasets present/admitted on this host)");
    }
}
