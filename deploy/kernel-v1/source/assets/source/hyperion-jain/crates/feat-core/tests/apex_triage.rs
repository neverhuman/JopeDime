//! apex_triage — the CHEAP (<60s) ranker for Apex Drive sandbox experiments.
//!
//! This is NOT a promotion gate. It runs the pure-Rust pipeline at a deliberately tiny config (small GP/EDA,
//! 1000-row cap) over a 4-dataset subset that spans the champion's failure modes (a weak-signal regression
//! `parkinsons`, a mid regression `bike_sharing`, and two binary `sts_afib`/`support2`). The sandbox engine
//! runs this in a throwaway git-worktree to score whether a hypothesis MOVES THE NEEDLE before spending an
//! expensive jailgun patch on it. It is deterministic (seed-fixed, hashes stable) so a worktree's triage
//! metric is directly comparable to the champion's triage baseline. The full 16-dataset `apex_real_bench`
//! remains the ONLY promotion authority.
//!
//! Prints `APEX_TRIAGE <name> champion=<m> baseline=<b> hash=<h> kind=<auc|r2>` per dataset and a single
//! `APEX_TRIAGE_METRIC <mean>` line. Reads the shared registry `tests/apex_datasets.txt` (so paths stay in
//! one place) but only runs the TRIAGE_SET subset.
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

/// The triage subset — a representative slice of the FULL gate's real datasets (2 regression + 2 binary,
/// incl. the headline `sts_mimo_afib`). It is a strict subset of `apex_real_bench`'s registry so a triage
/// gain PREDICTS a gate gain. (Was `sts_afib`, which no longer matches the renamed registry entry → silently
/// skipped; fixed to `sts_mimo_afib`.)
const TRIAGE_SET: &[&str] = &[
    // 8-dataset proxy for the 16-gate (4 regression + 4 binary). Widened from 4 because a 4-dataset triage
    // overfit: changes that helped the 4 regressed the other 12 of the gate (4/6 graduates regressed). A
    // broader subset vets graduates against more of the gate so a triage gain better predicts a gate gain.
    "parkinsons",
    "bike_sharing",
    "gas_turbine",
    "scm20d",
    "sts_mimo_afib",
    "support2",
    "ai4i",
    "mi_complications",
];
/// Deterministic row cap for the cheaper measure (vs 8000 in the full gate). Overridable for tuning.
const TRIAGE_ROW_CAP: usize = 3000;

/// Pipeline config that MIRRORS the full gate's `apex_real_bench::quick_cfg` EXACTLY, so the cheap triage
/// behaves like the gate (just on 4 datasets at a 3000-row cap instead of 16 at 8000). This is the key
/// triage↔gate ALIGNMENT: previously triage ran a 3–4× smaller config (pop 50/gens 1) whose deltas did not
/// transfer to the gate, so good ideas were filtered out and bad ones graduated. Keep these IN SYNC with
/// `apex_real_bench::quick_cfg`. The full 16-dataset `apex_real_bench` remains the ONLY promotion authority.
fn quick_triage_cfg(seed: u64, task: TaskKind, n_classes: usize) -> PipelineConfig {
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
        select_subsample: 2000,
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

/// Auto-detect the primary target + sibling TARGET_* columns (mirrors apex_real_bench's leakage firewall).
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

/// Deterministically cap a dataset to the first `max` rows (column-major rebuild) so triage is fast + stable.
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

#[test]
fn apex_triage_basket() {
    let registry = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/apex_datasets.txt");
    let text = std::fs::read_to_string(registry).unwrap_or_default();
    let cap: usize = std::env::var("APEX_TRIAGE_ROW_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(TRIAGE_ROW_CAP);

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
        let (name, _task_s, train_csv, hint) = (
            cols[0].trim(),
            cols[1].trim(),
            cols[2].trim(),
            cols[3].trim(),
        );
        if !TRIAGE_SET.contains(&name) {
            continue;
        }
        let test_csv = train_csv.replace("_train_", "_test_");
        if !Path::new(train_csv).exists() || !Path::new(&test_csv).exists() {
            eprintln!("APEX_TRIAGE_SKIP {name} reason=csv_missing");
            continue;
        }
        let header_line = match std::fs::read_to_string(train_csv)
            .ok()
            .and_then(|s| s.lines().next().map(str::to_string))
        {
            Some(h) => h,
            None => {
                eprintln!("APEX_TRIAGE_SKIP {name} reason=empty_csv");
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
                eprintln!("APEX_TRIAGE_SKIP {name} reason=load_err:{e}");
                continue;
            }
        };
        let task = pair.task.kind;
        let n_classes = pair.task.n_classes;
        let train = cap_rows(pair.train, cap);
        let hold = cap_rows(pair.hold, (cap / 4).max(250));
        let cfg = quick_triage_cfg(7, task, n_classes);
        let sink = StderrProgressSink::new(
            RunContext::new("apex-triage")
                .dataset(name)
                .task(format!("{task:?}"))
                .rows(train.n(), hold.n())
                .features(train.p())
                .selection_mode(cfg.eda.selection_mode.as_str()),
        );
        let res = if matches!(task, TaskKind::Regression) {
            run_pipeline(&train, &hold, &GlmBackend::gaussian(), &[], &cfg, &sink)
        } else {
            run_pipeline(&train, &hold, &GlmBackend::logistic(), &[], &cfg, &sink)
        };
        let metric = res.champion_holdout_auc;
        if !metric.is_finite() {
            eprintln!("APEX_TRIAGE_SKIP {name} reason=nonfinite");
            continue;
        }
        let kind = match task {
            TaskKind::Regression => "r2",
            _ => "auc",
        };
        let hash = dataset_hash(&train.x, &train.y, &train.names);
        println!(
            "APEX_TRIAGE {name} champion={metric:.6} baseline={:.6} hash={hash} kind={kind}",
            res.baseline_holdout_auc
        );
        admitted += 1;
        sum += metric;
    }

    if admitted > 0 {
        println!(
            "APEX_TRIAGE_METRIC {:.6} over {admitted} triage dataset(s)",
            sum / admitted as f64
        );
    } else {
        eprintln!("APEX_TRIAGE_METRIC none (no triage datasets present on this host)");
    }
}
