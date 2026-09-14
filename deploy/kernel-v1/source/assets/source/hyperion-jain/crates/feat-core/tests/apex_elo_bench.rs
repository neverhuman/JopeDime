//! apex_elo_bench — score jain by task-local **chess-Elo + rank** against the competing algorithm families on a
//! curated discriminative coreset (from the ~/quant benchmark study), instead of the old mean-holdout apex_metric.
//!
//! For each coreset dataset (registry `tests/apex_coreset.txt`, sep(z)-sorted): run jain's pipeline on the
//! dataset's OWN train.csv→test.csv (full test, no row cap) to get jain's utility (r² for regression, roc_auc for
//! binary — both are higher-better and present for every competitor too), then read the pre-computed competitor
//! scores from `<dir>/reports/<algo>.json`. Per dataset, rank {jain}∪{competitors} by utility; pool pairwise
//! outcomes across the task's datasets and run `compute_elo` (init 1000, 60 rounds, K=64, 400-logistic) — IDENTICAL
//! to `~/quant/model_score_tui.py`. Report per task: `APEX_ELO task=<t> jain=<elo> rank=<r>/<n> meanpct=<m>` where
//! meanpct = mean over the task's datasets of `100*(n-rank)/(n-1)` (monotone [0,100], the promotion key). Higher is
//! better. Competitor reports + datasets are part of the frozen scoring surface (anti-tamper via fitness.sh).
use feat_core::backend::TrainSpec;
use feat_core::data::load_train_hold;
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
use std::collections::HashMap;
use std::path::Path;

/// Pipeline config mirroring the gate's `quick_cfg` (apex_real_bench). Coreset datasets are tiny (<1000 rows) so
/// no row cap is needed; jain scores on the FULL test set the competitors used.
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

/// Minimal flat-JSON number extractor: find `"key"` then the number after the next `:`. Robust for the machine-
/// generated competitor reports (no nesting on these keys).
fn json_num(text: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\"");
    let i = text.find(&pat)?;
    let after = &text[i + pat.len()..];
    let colon = after.find(':')?;
    let s: String = after[colon + 1..]
        .trim_start()
        .chars()
        .take_while(|c| {
            c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+' || *c == 'e' || *c == 'E'
        })
        .collect();
    s.parse::<f64>().ok()
}

/// Group `*_classifier_*`/`*_regressor_*` family variants like model_score_tui's canonical_model_name.
fn canon(algo: &str) -> String {
    for m in ["_classifier_", "_regressor_"] {
        if let Some(p) = algo.find(m) {
            return format!("{}_{}", &algo[..p], &algo[p + m.len()..]);
        }
    }
    for s in ["_classifier", "_regressor"] {
        if let Some(stripped) = algo.strip_suffix(s) {
            return stripped.to_string();
        }
    }
    algo.to_string()
}

/// Elo: init 1000, 60 synchronous rounds, K=64, expected = 1/(1+10^((rB-rA)/400)). Verbatim from model_score_tui.
fn compute_elo(n: usize, matches: &[(usize, usize, f64)]) -> Vec<f64> {
    let mut r = vec![1000.0f64; n];
    for _ in 0..60 {
        let mut d = vec![0.0f64; n];
        let mut cnt = vec![0u32; n];
        for &(a, b, o) in matches {
            let ea = 1.0 / (1.0 + 10f64.powf((r[b] - r[a]) / 400.0));
            d[a] += o - ea;
            d[b] += (1.0 - o) - (1.0 - ea);
            cnt[a] += 1;
            cnt[b] += 1;
        }
        for i in 0..n {
            if cnt[i] > 0 {
                r[i] += 64.0 * d[i] / cnt[i] as f64;
            }
        }
    }
    r
}

/// Average-tie 1-based ranks for utilities sorted best-first.
fn avg_tie_ranks(sorted_desc: &[f64]) -> Vec<f64> {
    let n = sorted_desc.len();
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut end = i + 1;
        while end < n && (sorted_desc[end] - sorted_desc[i]).abs() <= 1e-12 {
            end += 1;
        }
        let avg = (i + 1 + end) as f64 / 2.0;
        for r in ranks.iter_mut().take(end).skip(i) {
            *r = avg;
        }
        i = end;
    }
    ranks
}

fn run_task(task_name: &str, rows: &[(String, String, f64)]) {
    let is_reg = task_name == "regression";
    let metric_key = if is_reg { "r2" } else { "roc_auc" };
    let mut idx: HashMap<String, usize> = HashMap::new();
    let mut model_of: Vec<String> = Vec::new();
    let id = |name: &str, model_of: &mut Vec<String>, idx: &mut HashMap<String, usize>| -> usize {
        if let Some(&i) = idx.get(name) {
            i
        } else {
            let i = model_of.len();
            idx.insert(name.to_string(), i);
            model_of.push(name.to_string());
            i
        }
    };
    let mut matches: Vec<(usize, usize, f64)> = Vec::new();
    let mut jain_percentiles: Vec<f64> = Vec::new();
    let mut n_datasets = 0usize;

    for (name, dir, sep) in rows {
        let train = format!("{dir}/train.csv");
        let test = format!("{dir}/test.csv");
        if !Path::new(&train).exists() || !Path::new(&test).exists() {
            eprintln!("APEX_ELO_SKIP {name} reason=csv_missing");
            continue;
        }
        let pair = match load_train_hold(
            Path::new(&train),
            Path::new(&test),
            &["target"],
            "target",
            CatEncoding::Ordinal,
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("APEX_ELO_SKIP {name} reason=load_err:{e}");
                continue;
            }
        };
        let task = pair.task.kind;
        let n_classes = pair.task.n_classes;
        let cfg = quick_cfg(7, task, n_classes);
        let be = if is_reg {
            GlmBackend::gaussian()
        } else {
            GlmBackend::logistic()
        };
        let sink = StderrProgressSink::new(
            RunContext::new("apex-elo")
                .dataset(name)
                .task(format!("{task:?}"))
                .rows(pair.train.n(), pair.hold.n())
                .features(pair.train.p())
                .selection_mode(cfg.eda.selection_mode.as_str()),
        );
        let res = run_pipeline(&pair.train, &pair.hold, &be, &[], &cfg, &sink);
        let jain_util = res.champion_holdout_auc;
        if !jain_util.is_finite() {
            eprintln!("APEX_ELO_SKIP {name} reason=jain_nonfinite");
            continue;
        }

        // competitor utilities from <dir>/reports/<algo>.json (status=completed, primary metric present)
        let mut util: HashMap<String, f64> = HashMap::new();
        if let Ok(rd) = std::fs::read_dir(format!("{dir}/reports")) {
            for ent in rd.flatten() {
                let p = ent.path();
                if p.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                let txt = match std::fs::read_to_string(&p) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                if !txt.contains("\"completed\"") {
                    continue;
                }
                let algo = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                if let Some(m) = json_num(&txt, metric_key) {
                    if m.is_finite() {
                        // keep best variant per canonical family
                        let c = canon(algo);
                        util.entry(c)
                            .and_modify(|v| {
                                if m > *v {
                                    *v = m
                                }
                            })
                            .or_insert(m);
                    }
                }
            }
        }
        util.insert("jain".to_string(), jain_util);
        if util.len() < 2 {
            eprintln!("APEX_ELO_SKIP {name} reason=no_competitors");
            continue;
        }

        // per-dataset standing: sort by utility desc, average-tie ranks
        let mut entries: Vec<(String, f64)> = util.into_iter().collect();
        entries.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let utils_desc: Vec<f64> = entries.iter().map(|e| e.1).collect();
        let ranks = avg_tie_ranks(&utils_desc);
        let n = entries.len();
        // jain rank/percentile
        let jpos = entries.iter().position(|e| e.0 == "jain").unwrap();
        let jrank = ranks[jpos];
        let jpct = if n > 1 {
            100.0 * (n as f64 - jrank) / (n as f64 - 1.0)
        } else {
            100.0
        };
        jain_percentiles.push(jpct);
        n_datasets += 1;
        // Emit a fitness-compatible APEX_DATASET line so the existing fitness.sh lock/compare/anti-tamper
        // machinery treats jain's per-dataset PERCENTILE (0-1) as the metric -> APEX_METRIC = Mean% (the
        // promotion key). hash = content hash of the (frozen, external) coreset train data for anti-tamper.
        let hash =
            feat_core::hashing::dataset_hash(&pair.train.x, &pair.train.y, &pair.train.names);
        println!(
            "APEX_DATASET {name} champion={:.6} baseline=0.000000 hash={hash} kind=elo domain=coreset task={task_name}",
            jpct / 100.0
        );

        // pairwise matches for this dataset (indices into the task-global model list)
        let ids: Vec<usize> = entries
            .iter()
            .map(|e| id(&e.0, &mut model_of, &mut idx))
            .collect();
        for a in 0..n {
            for b in (a + 1)..n {
                let (ua, ub) = (entries[a].1, entries[b].1);
                let outcome = if (ua - ub).abs() <= 1e-12 {
                    0.5
                } else if ua > ub {
                    1.0
                } else {
                    0.0
                };
                matches.push((ids[a], ids[b], outcome));
            }
        }
        println!("APEX_ELO_DATASET {name} task={task_name} jain_util={jain_util:.6} jain_rank={jrank:.1}/{n} sep={sep:.4}");
    }

    if n_datasets == 0 {
        eprintln!("APEX_ELO task={task_name} none (no datasets scored)");
        return;
    }
    let elos = compute_elo(model_of.len(), &matches);
    let jidx = idx["jain"];
    let jain_elo = elos[jidx];
    // leaderboard rank of jain by elo desc
    let rank = 1 + elos.iter().filter(|&&e| e > jain_elo + 1e-9).count();
    let meanpct = jain_percentiles.iter().sum::<f64>() / n_datasets as f64;
    // best competitor elo (frontier)
    let best = elos.iter().cloned().fold(f64::MIN, f64::max);
    println!(
        "APEX_ELO task={task_name} jain={jain_elo:.1} rank={rank}/{n} meanpct={meanpct:.3} frontier={best:.1} datasets={nd}",
        n = model_of.len(),
        nd = n_datasets
    );
}

#[test]
fn apex_elo_basket() {
    let registry = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/apex_coreset.txt");
    let text = std::fs::read_to_string(registry).unwrap_or_default();
    let mut reg: Vec<(String, String, f64)> = Vec::new();
    let mut bin: Vec<(String, String, f64)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 4 {
            continue;
        }
        let (name, task, dir) = (
            cols[0].trim().to_string(),
            cols[1].trim(),
            cols[2].trim().to_string(),
        );
        let sep: f64 = cols
            .get(3)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);
        match task {
            "regression" => reg.push((name, dir, sep)),
            "binary" => bin.push((name, dir, sep)),
            _ => {}
        }
    }
    // already sep-sorted in the registry; run each task independently
    run_task("regression", &reg);
    run_task("binary", &bin);
}
