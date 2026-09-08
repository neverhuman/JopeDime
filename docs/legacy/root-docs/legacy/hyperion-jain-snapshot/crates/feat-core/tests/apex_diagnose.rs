//! apex_diagnose — the DIAGNOSE phase of Apex Drive. Runs the champion pipeline at the gate config over the
//! real basket and prints a rich per-dataset weakness signal that `diagnose.sh` turns into a ranked
//! WEAKNESS.json (which datasets/stages to target this generation). NOT a gate — it never gates promotion;
//! it just tells the funnel where to dig.
//!
//! Per dataset prints:
//!   DATASET_DIAGNOSTIC <name> kind=<r2|auc> baseline=<b> champion=<c> lift=<c-b> cv=<cv>
//!     cv_holdout_gap=<cv-champion> survivors=<n> used=<size> aug=<feats> leak_dropped=<n>
//!     ensemble=<fuser|none> oof_lcb=<x> secs=<s>
//!
//! Reads the shared registry `tests/apex_datasets.txt`. Uses the SAME admission + cap as the real gate so the
//! diagnosed numbers match what the champion actually scores when gated.
use feat_core::backend::TrainSpec;
use feat_core::data::{load_train_hold, Dataset, Matrix};
use feat_core::dominance::DominanceConfig;
use feat_core::encode::CatEncoding;
use feat_core::gp::GpConfig;
use feat_core::learners::glm::GlmBackend;
use feat_core::phase2::P2Config;
use feat_core::pipeline::{run_pipeline, PipelineConfig};
use feat_core::progress::{RunContext, StderrProgressSink};
use feat_core::select::EdaConfig;
use feat_core::stacking::StackConfig;
use feat_core::task::TaskKind;
use std::path::Path;

/// Same gate config as apex_real_bench's quick_cfg (so diagnostics match gated scores).
fn gate_cfg(seed: u64, task: TaskKind, n_classes: usize) -> PipelineConfig {
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
fn apex_diagnose_basket() {
    let registry = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/apex_datasets.txt");
    let text = std::fs::read_to_string(registry).unwrap_or_default();
    let cap: usize = std::env::var("APEX_GATE_MAX_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000);

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
        let test_csv = train_csv.replace("_train_", "_test_");
        if !Path::new(train_csv).exists() || !Path::new(&test_csv).exists() {
            continue;
        }
        let header_line = match std::fs::read_to_string(train_csv)
            .ok()
            .and_then(|s| s.lines().next().map(str::to_string))
        {
            Some(h) => h,
            None => continue,
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
            Err(_) => continue,
        };
        let task = pair.task.kind;
        let n_classes = pair.task.n_classes;
        let train = cap_rows(pair.train, cap);
        let hold = cap_rows(pair.hold, (cap / 4).max(500));
        let cfg = gate_cfg(7, task, n_classes);
        let sink = StderrProgressSink::new(
            RunContext::new("apex-diagnose")
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
        let kind = match task {
            TaskKind::Regression => "r2",
            _ => "auc",
        };
        let gap = res.champion_cv_auc - res.champion_holdout_auc;
        let (ens, oof) = match &res.ensemble {
            Some(e) => (e.selected_fuser.clone(), e.oof_lcb),
            None => ("none".to_string(), f64::NAN),
        };
        println!(
            "DATASET_DIAGNOSTIC {name} kind={kind} baseline={:.6} champion={:.6} lift={:.6} cv={:.6} cv_holdout_gap={:.6} survivors={} used={} aug={} leak_dropped={} ensemble={} oof_lcb={:.4} secs={:.1}",
            res.baseline_holdout_auc,
            res.champion_holdout_auc,
            res.lift,
            res.champion_cv_auc,
            gap,
            res.n_survivors,
            res.champion_size,
            res.n_features,
            res.trust.leak_features_dropped,
            ens,
            oof,
            res.secs,
        );
    }
    println!("DATASET_DIAGNOSTIC_DONE");
}
