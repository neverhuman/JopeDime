//! Multi-panel scientific & systems benchmark harness comparing zero-knowledge Carbon cloud runs
//! against `Ridge` and all 10 pure-Rust `jope-comparators` learners (`AutoGluon-style`, `FLAML-style`,
//! `H2O-style`, `MLJAR-style`, `XGBoost-style`, `LightGBM-style`, `CatBoost-style`, `RandomForest`,
//! `ExtraTrees`, `HistGradientBoosting`) on identical `(FIT, HOLDOUT)` splits with Wilcoxon signed-rank
//! significance, scale-free normalized RMSE, latency p50/p95, and USD/1,000-runs Pareto frontiers.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use jope_api::{
    carbon::{CarbonExecutionModeWire, CarbonRunStateWire, CarbonTierWire},
    ids::{DatasetId, RunId},
    Effort,
};
use jope_comparators::{
    fit_comparator, mae as comp_mae, r_squared as comp_r2, rmse as comp_rmse, ComparatorId,
    ComparatorTier, Deadline, FeatureMatrix, RidgeModel, RidgeParams,
};
use serde::{Deserialize, Serialize};

use crate::{
    cells::engine_invoker::now_epoch_ms,
    cli::{
        client::{CarbonProbedEndpoint, CarbonThinClient},
        tabular::{
            load_delimited_table, parse_delimited_str, prepare_encoded_training_set, CarbonRawTable,
        },
    },
    modes::planner::{compute_live_run_cost, CarbonIngressCostMode, CarbonLiveCostInputs},
    storage::cas::sha256_lower_hex,
};

/// `CarbonComparatorScore` is a struct recording one baseline learner's same-split holdout metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonComparatorScore {
    pub key: String,
    pub label: String,
    pub reference: String,
    pub known_gaps: String,
    pub rmse: f64,
    pub mae: f64,
    pub r2: Option<f64>,
    pub fit_ms: f64,
}

/// `CarbonTaskBenchRecord` is a struct recording Carbon + all comparator scores on one benchmark task.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonTaskBenchRecord {
    pub task_id: String,
    pub panel: String,
    pub description: String,
    pub total_rows: usize,
    pub fit_rows: usize,
    pub holdout_rows: usize,
    pub encoded_features: usize,
    pub carbon_tier: CarbonTierWire,
    pub carbon_effort: Effort,
    pub carbon_execution_mode: CarbonExecutionModeWire,
    pub carbon_eval_shards: u32,
    pub carbon_waves: u32,
    pub carbon_candidates: u64,
    pub carbon_wall_ms: f64,
    pub carbon_search_ms: f64,
    pub carbon_predict_ms: f64,
    pub carbon_estimated_usd: f64,
    pub carbon_encrypted_cuen: bool,
    pub carbon_shredded: bool,
    pub target_std: f64,
    pub carbon_rmse: f64,
    pub carbon_norm_rmse: f64,
    pub carbon_mae: f64,
    pub carbon_r2: Option<f64>,
    pub carbon_rank: usize,
    pub total_methods: usize,
    pub comparators: Vec<CarbonComparatorScore>,
}

/// `CarbonMethodAggregateSummary` is a struct summarizing one method's rank, accuracy, and significance across all tasks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonMethodAggregateSummary {
    pub key: String,
    pub label: String,
    pub tasks_evaluated: usize,
    pub mean_rank: f64,
    pub geomean_norm_rmse: f64,
    pub mean_r2: f64,
    pub median_r2: f64,
    pub carbon_wins_vs_method: usize,
    pub carbon_ties_vs_method: usize,
    pub carbon_losses_vs_method: usize,
    pub paired_wilcoxon_p_value: Option<f64>,
}

/// `CarbonScientificBenchmarkOptions` is a struct configuring the multi-dataset head-to-head benchmark suite.
#[derive(Clone, Debug)]
pub struct CarbonScientificBenchmarkOptions {
    pub endpoints: Vec<String>,
    pub pinned_keks: BTreeMap<String, String>,
    pub tier: CarbonTierWire,
    pub effort: Effort,
    pub tenant_id: String,
    pub auth_token: Option<String>,
    pub suite: String,
    pub datasets_dir: Option<PathBuf>,
    pub csv_dir: Option<PathBuf>,
    pub seed: u64,
}

/// `CarbonScientificBenchmarkReport` is a struct containing the complete multi-panel benchmark leaderboard and telemetry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonScientificBenchmarkReport {
    pub schema: String,
    pub timestamp_epoch_ms: u64,
    pub suite: String,
    pub endpoint: String,
    pub region: String,
    pub carbon_tier: CarbonTierWire,
    pub carbon_effort: Effort,
    pub zero_knowledge_cuen_all: bool,
    pub all_runs_shredded: bool,
    pub task_count: usize,
    pub comparator_count: usize,
    pub p50_wall_ms: f64,
    pub p95_wall_ms: f64,
    pub mean_wall_ms: f64,
    pub total_estimated_usd: f64,
    pub mean_usd_per_run: f64,
    pub usd_per_1000_runs: f64,
    pub runs_per_1_usd: Option<f64>,
    pub provenance_notes: Vec<String>,
    pub leaderboard: Vec<CarbonMethodAggregateSummary>,
    pub tasks: Vec<CarbonTaskBenchRecord>,
}

struct Splitmix64 {
    state: u64,
}

impl Splitmix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().clamp(1e-12, 1.0 - 1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

struct BenchmarkTaskSource {
    task_id: String,
    panel: String,
    description: String,
    kind: TaskDataKind,
}

enum TaskDataKind {
    SingleTable(CarbonRawTable, Option<String>),
    PreSplitTables {
        train: CarbonRawTable,
        test: CarbonRawTable,
        target: Option<String>,
    },
}

fn build_numeric_raw_table(headers: Vec<String>, rows: Vec<Vec<f64>>) -> CarbonRawTable {
    let str_rows = rows
        .into_iter()
        .map(|r| r.into_iter().map(|v| Some(format!("{v:.8}"))).collect())
        .collect();
    CarbonRawTable {
        headers,
        rows: str_rows,
        delimiter: ',',
        had_header: true,
    }
}

fn generate_panel_a_and_b_tasks(
    suite: &str,
    base_seed: u64,
) -> Result<Vec<BenchmarkTaskSource>, String> {
    let mut tasks = Vec::new();
    let is_smoke = suite.eq_ignore_ascii_case("smoke");
    let n_std = if is_smoke { 220 } else { 520 };

    // A1: Friedman #1 (1991)
    {
        let mut rng = Splitmix64::new(base_seed ^ 0xA001);
        let d = 10usize;
        let mut headers: Vec<String> = (0..d).map(|j| format!("x{j}")).collect();
        headers.push("target".to_owned());
        let mut rows = Vec::with_capacity(n_std);
        for _ in 0..n_std {
            let x: Vec<f64> = (0..d).map(|_| rng.next_f64()).collect();
            let y = 10.0 * (std::f64::consts::PI * x[0] * x[1]).sin()
                + 20.0 * (x[2] - 0.5).powi(2)
                + 10.0 * x[3]
                + 5.0 * x[4]
                + 0.5 * rng.normal();
            let mut r = x;
            r.push(y);
            rows.push(r);
        }
        tasks.push(BenchmarkTaskSource {
            task_id: "friedman_1".to_owned(),
            panel: "A_literature".to_owned(),
            description:
                "Friedman #1 (1991): nonlinear sin(pi*x0*x1) + quadratic + linear + 5 noise cols"
                    .to_owned(),
            kind: TaskDataKind::SingleTable(
                build_numeric_raw_table(headers, rows),
                Some("target".to_owned()),
            ),
        });
    }

    // A2: Friedman #2 (AC impedance)
    {
        let mut rng = Splitmix64::new(base_seed ^ 0xA002);
        let headers = vec![
            "R".to_owned(),
            "omega".to_owned(),
            "L".to_owned(),
            "C".to_owned(),
            "target".to_owned(),
        ];
        let mut rows = Vec::with_capacity(n_std);
        for _ in 0..n_std {
            let r_ohm = rng.uniform(0.0, 100.0);
            let omega = rng.uniform(40.0 * std::f64::consts::PI, 560.0 * std::f64::consts::PI);
            let l_hen = rng.uniform(0.0, 1.0);
            let c_uf = rng.uniform(1.0, 11.0);
            let react = omega * l_hen - 1.0 / (omega * c_uf * 1e-6);
            let z = (r_ohm * r_ohm + react * react).sqrt() + 8.0 * rng.normal();
            rows.push(vec![r_ohm, omega, l_hen, c_uf, z]);
        }
        tasks.push(BenchmarkTaskSource {
            task_id: "friedman_2_impedance".to_owned(),
            panel: "A_literature".to_owned(),
            description:
                "Friedman #2 (1991): AC series circuit impedance sqrt(R^2 + (wL - 1/wC)^2)"
                    .to_owned(),
            kind: TaskDataKind::SingleTable(
                build_numeric_raw_table(headers, rows),
                Some("target".to_owned()),
            ),
        });
    }

    // A3: Piecewise Axis-Aligned Step Regimes (Tree-Favorable Control!)
    {
        let mut rng = Splitmix64::new(base_seed ^ 0xA006);
        let d = 8usize;
        let mut headers: Vec<String> = (0..d).map(|j| format!("s{j}")).collect();
        headers.push("target".to_owned());
        let mut rows = Vec::with_capacity(n_std);
        for _ in 0..n_std {
            let x: Vec<f64> = (0..d).map(|_| rng.uniform(-1.0, 1.0)).collect();
            let mut y = 0.35 * x[4];
            if x[0] > 0.2 {
                y += 4.5;
            }
            if x[1] < -0.25 && x[2] > 0.0 {
                y -= 3.8;
            }
            if x[3] > 0.45 {
                y += 2.7;
            }
            if x[0] <= 0.2 && x[5] < -0.4 {
                y -= 2.1;
            }
            y += 0.15 * rng.normal();
            let mut r = x;
            r.push(y);
            rows.push(r);
        }
        tasks.push(BenchmarkTaskSource {
            task_id: "piecewise_axis_step".to_owned(),
            panel: "A_literature".to_owned(),
            description: "Tree-favorable control: axis-aligned discontinuous step thresholds"
                .to_owned(),
            kind: TaskDataKind::SingleTable(
                build_numeric_raw_table(headers, rows),
                Some("target".to_owned()),
            ),
        });
    }

    if is_smoke {
        return Ok(tasks);
    }

    // A4: Friedman #3 (AC phase angle)
    {
        let mut rng = Splitmix64::new(base_seed ^ 0xA003);
        let headers = vec![
            "R".to_owned(),
            "omega".to_owned(),
            "L".to_owned(),
            "C".to_owned(),
            "target".to_owned(),
        ];
        let mut rows = Vec::with_capacity(n_std);
        for _ in 0..n_std {
            let r_ohm = rng.uniform(5.0, 100.0);
            let omega = rng.uniform(40.0 * std::f64::consts::PI, 560.0 * std::f64::consts::PI);
            let l_hen = rng.uniform(0.0, 1.0);
            let c_uf = rng.uniform(1.0, 11.0);
            let react = omega * l_hen - 1.0 / (omega * c_uf * 1e-6);
            let phi = (react / r_ohm).atan() + 0.03 * rng.normal();
            rows.push(vec![r_ohm, omega, l_hen, c_uf, phi]);
        }
        tasks.push(BenchmarkTaskSource {
            task_id: "friedman_3_phase".to_owned(),
            panel: "A_literature".to_owned(),
            description: "Friedman #3 (1991): AC circuit phase angle atan((wL - 1/wC)/R)"
                .to_owned(),
            kind: TaskDataKind::SingleTable(
                build_numeric_raw_table(headers, rows),
                Some("target".to_owned()),
            ),
        });
    }

    // A5: Ishigami 6D non-monotonic global sensitivity
    {
        let mut rng = Splitmix64::new(base_seed ^ 0xA004);
        let d = 6usize;
        let mut headers: Vec<String> = (0..d).map(|j| format!("z{j}")).collect();
        headers.push("target".to_owned());
        let mut rows = Vec::with_capacity(n_std);
        for _ in 0..n_std {
            let x: Vec<f64> = (0..d)
                .map(|_| rng.uniform(-std::f64::consts::PI, std::f64::consts::PI))
                .collect();
            let y = x[0].sin()
                + 7.0 * x[1].sin().powi(2)
                + 0.1 * x[2].powi(4) * x[0].sin()
                + 0.1 * rng.normal();
            let mut r = x;
            r.push(y);
            rows.push(r);
        }
        tasks.push(BenchmarkTaskSource {
            task_id: "ishigami_6d".to_owned(),
            panel: "A_literature".to_owned(),
            description: "Ishigami non-monotonic oscillatory benchmark sin(z0) + 7*sin^2(z1) + 0.1*z2^4*sin(z0)".to_owned(),
            kind: TaskDataKind::SingleTable(build_numeric_raw_table(headers, rows), Some("target".to_owned())),
        });
    }

    // A6: Kepler / Feynman rational physical law
    {
        let mut rng = Splitmix64::new(base_seed ^ 0xA005);
        let d = 8usize;
        let mut headers: Vec<String> = (0..d).map(|j| format!("p{j}")).collect();
        headers.push("target".to_owned());
        let mut rows = Vec::with_capacity(n_std);
        for _ in 0..n_std {
            let x: Vec<f64> = (0..d).map(|_| rng.uniform(-2.0, 2.0)).collect();
            let y = (3.2 * x[0] * x[1]) / (1.0 + x[2] * x[2]) + 1.5 * (x[3].abs() + 0.25).sqrt()
                - 0.9 * x[4]
                + 0.05 * rng.normal();
            let mut r = x;
            r.push(y);
            rows.push(r);
        }
        tasks.push(BenchmarkTaskSource {
            task_id: "kepler_rational_8d".to_owned(),
            panel: "A_literature".to_owned(),
            description:
                "Physical rational + root law: 3.2*p0*p1/(1+p2^2) + 1.5*sqrt(|p3|+0.25) - 0.9*p4"
                    .to_owned(),
            kind: TaskDataKind::SingleTable(
                build_numeric_raw_table(headers, rows),
                Some("target".to_owned()),
            ),
        });
    }

    // B1: Missing-value 15% messy CSV
    {
        let mut rng = Splitmix64::new(base_seed ^ 0xB001);
        let mut csv_text =
            String::from("sensor_a,sensor_b,sensor_c,sensor_d,sensor_e,sensor_f,target\n");
        for _ in 0..480 {
            let x: Vec<f64> = (0..6).map(|_| rng.uniform(-2.0, 2.0)).collect();
            let y = 2.1 * x[0] - 1.4 * x[1]
                + 0.9 * (x[0] * x[2])
                + 1.2 * x[3].sin()
                + 0.08 * rng.normal();
            for &val in &x {
                if rng.next_f64() < 0.15 {
                    csv_text.push_str("NA,");
                } else {
                    csv_text.push_str(&format!("{val:.6},"));
                }
            }
            csv_text.push_str(&format!("{y:.6}\n"));
        }
        let table = parse_delimited_str(&csv_text, "missing_15pct_tabular", Some(','), true)?;
        tasks.push(BenchmarkTaskSource {
            task_id: "missing_15pct_tabular".to_owned(),
            panel: "B_tabular_stress".to_owned(),
            description:
                "Messy sensor table with 15% MCAR missing cells ('NA') across all features"
                    .to_owned(),
            kind: TaskDataKind::SingleTable(table, Some("target".to_owned())),
        });
    }

    // B2: Mixed Categorical + Boolean + Quoted Strings + ID column
    {
        let mut rng = Splitmix64::new(base_seed ^ 0xB002);
        let zones = [
            "\"Bay Area, North\"",
            "\"Central, Metro\"",
            "Coastal",
            "Inland",
        ];
        let zone_eff = [2.8, 1.1, 1.9, -1.5];
        let mut csv_text =
            String::from("record_id,zone,waterfront,sqft_k,age_yr,rooms,noise_x,price\n");
        for idx in 0..480 {
            let z_i = (rng.next_u64() as usize) % zones.len();
            let wf = rng.next_f64() < 0.25;
            let sqft = rng.uniform(0.6, 4.2);
            let age = rng.uniform(0.0, 65.0);
            let rooms = rng.uniform(2.0, 9.0);
            let nx = rng.uniform(-1.0, 1.0);
            let price = 3.5 * sqft - 0.04 * age
                + 0.22 * sqft * rooms
                + zone_eff[z_i]
                + if wf { 1.75 } else { 0.0 }
                + 0.12 * rng.normal();
            csv_text.push_str(&format!(
                "house_{idx:04},{},{},{sqft:.5},{age:.4},{rooms:.3},{nx:.4},{price:.6}\n",
                zones[z_i],
                if wf { "true" } else { "false" }
            ));
        }
        let table = parse_delimited_str(&csv_text, "mixed_categorical_housing", Some(','), true)?;
        tasks.push(BenchmarkTaskSource {
            task_id: "mixed_categorical_housing".to_owned(),
            panel: "B_tabular_stress".to_owned(),
            description: "Quoted strings, categoricals, booleans, and unique string ID column"
                .to_owned(),
            kind: TaskDataKind::SingleTable(table, Some("price".to_owned())),
        });
    }

    // B3: Collinear 16D latent factors
    {
        let mut rng = Splitmix64::new(base_seed ^ 0xB003);
        let d = 16usize;
        let mut headers: Vec<String> = (0..d).map(|j| format!("c{j}")).collect();
        headers.push("target".to_owned());
        let mut rows = Vec::with_capacity(480);
        for _ in 0..480 {
            let u0 = rng.normal();
            let u1 = rng.normal();
            let u2 = rng.normal();
            let mut x = Vec::with_capacity(d);
            for j in 0..d {
                let base = match j % 3 {
                    0 => u0,
                    1 => u1,
                    _ => u2,
                };
                x.push(base + 0.18 * rng.normal());
            }
            let y = 2.0 * u0 - 1.5 * u1 + 0.85 * (u0 * u1) + 1.1 * u2.cos() + 0.1 * rng.normal();
            x.push(y);
            rows.push(x);
        }
        tasks.push(BenchmarkTaskSource {
            task_id: "collinear_sensors_16d".to_owned(),
            panel: "B_tabular_stress".to_owned(),
            description: "High multicollinearity (r ~ 0.94 across 16 sensor channels)".to_owned(),
            kind: TaskDataKind::SingleTable(
                build_numeric_raw_table(headers, rows),
                Some("target".to_owned()),
            ),
        });
    }

    if suite.eq_ignore_ascii_case("comprehensive") {
        // B4: Heavy-tailed heteroscedastic shocks
        {
            let mut rng = Splitmix64::new(base_seed ^ 0xB004);
            let d = 6usize;
            let mut headers: Vec<String> = (0..d).map(|j| format!("h{j}")).collect();
            headers.push("target".to_owned());
            let mut rows = Vec::with_capacity(500);
            for _ in 0..500 {
                let x: Vec<f64> = (0..d).map(|_| rng.uniform(-2.0, 2.0)).collect();
                // Student-t(3) via normal / sqrt(chi2_3 / 3)
                let g1 = rng.normal();
                let g2 = rng.normal();
                let g3 = rng.normal();
                let chi3 = (g1 * g1 + g2 * g2 + g3 * g3) / 3.0;
                let t3 = rng.normal() / chi3.max(0.05).sqrt();
                let scale = 0.06 + 0.08 * x[0].abs();
                let y =
                    1.9 * x[0] + 1.3 * (x[1] * x[2]) - 0.95 * (1.0 + x[3] * x[3]).ln() + scale * t3;
                let mut r = x;
                r.push(y);
                rows.push(r);
            }
            tasks.push(BenchmarkTaskSource {
                task_id: "heavy_tailed_heteroscedastic".to_owned(),
                panel: "B_tabular_stress".to_owned(),
                description: "Heteroscedastic Student-t(3) heavy-tailed noise regime".to_owned(),
                kind: TaskDataKind::SingleTable(
                    build_numeric_raw_table(headers, rows),
                    Some("target".to_owned()),
                ),
            });
        }

        // B5: Scale 1,500 x 10 HybridFanOut throughput & $/1k runs
        {
            let mut rng = Splitmix64::new(base_seed ^ 0xB005);
            let d = 10usize;
            let mut headers: Vec<String> = (0..d).map(|j| format!("f{j}")).collect();
            headers.push("target".to_owned());
            let mut rows = Vec::with_capacity(1_500);
            for _ in 0..1_500 {
                let x: Vec<f64> = (0..d).map(|_| rng.uniform(-2.5, 2.5)).collect();
                let y = 2.2 * x[0] - 1.3 * x[1]
                    + 0.65 * x[0] * x[1]
                    + 0.9 * x[2].sin()
                    + 0.4 * (x[3] * x[3] - 1.0)
                    + 0.05 * rng.normal();
                let mut r = x;
                r.push(y);
                rows.push(r);
            }
            tasks.push(BenchmarkTaskSource {
                task_id: "scale_1500x10_fanout".to_owned(),
                panel: "B_tabular_stress".to_owned(),
                description:
                    "1,500-row x 10-feature scale benchmark exercising calibrated HybridFanOut"
                        .to_owned(),
                kind: TaskDataKind::SingleTable(
                    build_numeric_raw_table(headers, rows),
                    Some("target".to_owned()),
                ),
            });
        }
    }

    Ok(tasks)
}

fn load_datasets_dir_tasks(dir: &Path) -> Result<Vec<BenchmarkTaskSource>, String> {
    let mut subdirs: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("read datasets dir {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|x| x.path()))
        .filter(|p| p.is_dir() && p.join("train.csv").is_file() && p.join("test.csv").is_file())
        .collect();
    subdirs.sort();
    let mut out = Vec::new();
    for sd in subdirs {
        let name = sd
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("dataset")
            .to_owned();
        let train = load_delimited_table(&sd.join("train.csv"), Some(','), false)?;
        let test = load_delimited_table(&sd.join("test.csv"), Some(','), false)?;
        out.push(BenchmarkTaskSource {
            task_id: name.clone(),
            panel: "C_real_tabular".to_owned(),
            description: format!("Real tabular regression dataset '{name}' (pre-split train/test)"),
            kind: TaskDataKind::PreSplitTables {
                train,
                test,
                target: Some("-1".to_owned()),
            },
        });
    }
    Ok(out)
}

fn load_csv_dir_tasks(dir: &Path) -> Result<Vec<BenchmarkTaskSource>, String> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("read csv dir {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|x| x.path()))
        .filter(|p| {
            p.is_file() && matches!(p.extension().and_then(|e| e.to_str()), Some("csv" | "tsv"))
        })
        .collect();
    files.sort();
    let mut out = Vec::new();
    for f in files {
        let stem = f
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("custom")
            .to_owned();
        let table = load_delimited_table(&f, None, true)?;
        out.push(BenchmarkTaskSource {
            task_id: stem.clone(),
            panel: "D_user_csv".to_owned(),
            description: format!("User CSV dataset '{}'", f.display()),
            kind: TaskDataKind::SingleTable(table, Some("last".to_owned())),
        });
    }
    Ok(out)
}

/// Fit fast local pure-Rust comparators (`Ridge`, `RandomForest`, `LightGBM-style`, `AutoGluon-style`)
/// on the exact same `(fit_x, fit_y)` -> `(holdout_x, holdout_y)` split for `carbon run` / `carbon fit` scorecards.
pub fn compute_quick_run_comparators(
    fit_x: &[Vec<f64>],
    fit_y: &[f64],
    holdout_x: &[Vec<f64>],
    holdout_y: &[f64],
    seed: u64,
) -> Vec<CarbonComparatorScore> {
    if fit_x.len() < 12
        || holdout_x.len() < 3
        || fit_x.len() != fit_y.len()
        || holdout_x.len() != holdout_y.len()
    {
        return Vec::new();
    }
    let cols = fit_x[0].len();
    if cols == 0 || fit_x.len() * cols > 500_000 {
        return Vec::new();
    }
    evaluate_comparators_subset(
        fit_x,
        fit_y,
        holdout_x,
        holdout_y,
        seed,
        &[
            ComparatorId::RandomForest,
            ComparatorId::LightgbmStyle,
            ComparatorId::AutogluonStyle,
        ],
    )
}

/// Fit `Ridge` + all 10 `ComparatorId::ALL` pure-Rust comparators on identical `(fit_x, fit_y)` -> `(holdout_x, holdout_y)`.
pub fn evaluate_all_comparators_same_split(
    fit_x: &[Vec<f64>],
    fit_y: &[f64],
    holdout_x: &[Vec<f64>],
    holdout_y: &[f64],
    seed: u64,
) -> Vec<CarbonComparatorScore> {
    evaluate_comparators_subset(fit_x, fit_y, holdout_x, holdout_y, seed, &ComparatorId::ALL)
}

fn evaluate_comparators_subset(
    fit_x: &[Vec<f64>],
    fit_y: &[f64],
    holdout_x: &[Vec<f64>],
    holdout_y: &[f64],
    seed: u64,
    ids: &[ComparatorId],
) -> Vec<CarbonComparatorScore> {
    let n_fit = fit_x.len();
    let n_hold = holdout_x.len();
    let cols = fit_x.first().map(Vec::len).unwrap_or(0);
    if n_fit < 4 || n_hold == 0 || cols == 0 {
        return Vec::new();
    }
    let flat_fit: Vec<f64> = fit_x.iter().flat_map(|r| r.iter().copied()).collect();
    let flat_hold: Vec<f64> = holdout_x.iter().flat_map(|r| r.iter().copied()).collect();
    let Ok(train_mat) = FeatureMatrix::new(&flat_fit, n_fit, cols) else {
        return Vec::new();
    };
    let Ok(hold_mat) = FeatureMatrix::new(&flat_hold, n_hold, cols) else {
        return Vec::new();
    };

    let mut scores = Vec::with_capacity(1 + ids.len());

    // 1. Standardized Ridge (alpha = 1.0)
    let t_ridge = Instant::now();
    if let Ok(ridge) = RidgeModel::fit(train_mat, fit_y, &RidgeParams { alpha: 1.0 }) {
        if let Ok(preds) = ridge.predict(hold_mat) {
            if let (Ok(r), Ok(m)) = (comp_rmse(holdout_y, &preds), comp_mae(holdout_y, &preds)) {
                let r2 = comp_r2(holdout_y, &preds).ok();
                scores.push(CarbonComparatorScore {
                    key: "ridge_l2".to_owned(),
                    label: "Standardized Ridge (alpha=1.0, Rust)".to_owned(),
                    reference: "Closed-form Cholesky Ridge".to_owned(),
                    known_gaps: "fixed alpha=1.0".to_owned(),
                    rmse: r,
                    mae: m,
                    r2,
                    fit_ms: t_ridge.elapsed().as_secs_f64() * 1_000.0,
                });
            }
        }
    }

    // 2. Selected `jope-comparators` roster learners
    for &cid in ids {
        let t0 = Instant::now();
        let desc = cid.descriptor();
        if let Ok(fitted) = fit_comparator(
            cid,
            ComparatorTier::Medium,
            train_mat,
            fit_y,
            seed,
            &Deadline::no_limit(),
        ) {
            if let Ok(preds) = fitted.predict(hold_mat) {
                if let (Ok(r), Ok(m)) = (comp_rmse(holdout_y, &preds), comp_mae(holdout_y, &preds))
                {
                    let r2 = comp_r2(holdout_y, &preds).ok();
                    scores.push(CarbonComparatorScore {
                        key: desc.key.to_owned(),
                        label: desc.label.to_owned(),
                        reference: desc.reference.to_owned(),
                        known_gaps: desc.known_gaps.to_owned(),
                        rmse: r,
                        mae: m,
                        r2,
                        fit_ms: t0.elapsed().as_secs_f64() * 1_000.0,
                    });
                }
            }
        }
    }
    scores
}

fn erfc_approx(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 (max error 1.5e-7)
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let r = poly * (-z * z).exp();
    if x >= 0.0 {
        r
    } else {
        2.0 - r
    }
}

/// Two-sided paired Wilcoxon signed-rank p-value (normal approximation with tie & continuity correction).
pub fn paired_wilcoxon_two_sided_p(diffs: &[f64]) -> Option<f64> {
    let mut non_zero: Vec<f64> = diffs
        .iter()
        .copied()
        .filter(|d| d.is_finite() && d.abs() > 1e-12)
        .collect();
    let n = non_zero.len();
    if n < 5 {
        return None;
    }
    non_zero.sort_by(|a, b| {
        a.abs()
            .partial_cmp(&b.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranks = vec![0.0; n];
    let mut tie_term = 0.0;
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && (non_zero[j].abs() - non_zero[i].abs()).abs() <= 1e-12 {
            j += 1;
        }
        let avg_rank = 0.5 * ((i + 1 + j) as f64);
        for slot in &mut ranks[i..j] {
            *slot = avg_rank;
        }
        let t = (j - i) as f64;
        if t > 1.0 {
            tie_term += t * t * t - t;
        }
        i = j;
    }

    let w_pos: f64 = non_zero
        .iter()
        .zip(ranks.iter())
        .filter_map(|(&d, &rk)| (d > 0.0).then_some(rk))
        .sum();
    let nf = n as f64;
    let mean_w = nf * (nf + 1.0) / 4.0;
    let var_w = (nf * (nf + 1.0) * (2.0 * nf + 1.0) - 0.5 * tie_term) / 24.0;
    if var_w <= 1e-12 {
        return Some(1.0);
    }
    let z = ((w_pos - mean_w).abs() - 0.5).max(0.0) / var_w.sqrt();
    Some(erfc_approx(z / std::f64::consts::SQRT_2).clamp(0.0, 1.0))
}

fn mint_ids() -> Result<(RunId, DatasetId), String> {
    let u = ulid::Ulid::new().to_string();
    let run_id: RunId = format!("run_{u}")
        .parse()
        .map_err(|_| "construct RunId failed".to_owned())?;
    let ds_hex = &sha256_lower_hex(u.as_bytes())[..24];
    let ds_id: DatasetId = format!("ds_{ds_hex}")
        .parse()
        .map_err(|_| "construct DatasetId failed".to_owned())?;
    Ok((run_id, ds_id))
}

/// Execute the multi-dataset head-to-head scientific & systems benchmark suite against live Carbon cells.
pub fn execute_scientific_benchmark_suite(
    opts: &CarbonScientificBenchmarkOptions,
    show_stderr_progress: bool,
) -> Result<CarbonScientificBenchmarkReport, String> {
    let (mut base_client, probed) =
        CarbonThinClient::connect_fastest_pinned(opts.endpoints.clone(), &opts.pinned_keks)?;
    let first_probe: CarbonProbedEndpoint = probed
        .into_iter()
        .next()
        .ok_or_else(|| "no reachable Carbon endpoints".to_owned())?;

    let mut effective_tenant = opts.tenant_id.clone();
    let mut minted_anon = false;
    if let Some(tok) = &opts.auth_token {
        base_client = base_client.with_capability_token(tok.clone());
    } else if !base_client.enterprise_self_hosted()
        && opts.tier == CarbonTierWire::Free
        && std::env::var("CARBON_AUTH_TOKEN").is_err()
        && std::env::var("CARBON_AUTH_SECRET_HEX").is_err()
    {
        let anon = base_client.authenticate_anonymous()?;
        effective_tenant = anon.tenant_id;
        minted_anon = true;
    }

    let mut task_sources = generate_panel_a_and_b_tasks(&opts.suite, opts.seed)?;
    let mut provenance_notes = vec![
        "All methods are fitted strictly on identical FIT rows and scored on identical HOLDOUT rows.".to_owned(),
        "Panel A includes 'piecewise_axis_step' (discontinuous axis-aligned thresholds) as an explicit tree-favorable control.".to_owned(),
        "Pure-Rust comparators in `jope-comparators` carry explicit reference/gap descriptors and run with deterministic seeds.".to_owned(),
    ];

    if let Some(ds_dir) = &opts.datasets_dir {
        let real_tasks = load_datasets_dir_tasks(ds_dir)?;
        if !real_tasks.is_empty() {
            provenance_notes.push(format!(
                "Included {} pre-split table(s) from '{}'. Note: `datasets/test` ('test-10') is a historically favorable subset documented in `docs/same-cut-bench.md`; interpret alongside unbiased Panels A & B.",
                real_tasks.len(),
                ds_dir.display()
            ));
            task_sources.extend(real_tasks);
        }
    }
    if let Some(c_dir) = &opts.csv_dir {
        let user_tasks = load_csv_dir_tasks(c_dir)?;
        task_sources.extend(user_tasks);
    }

    if task_sources.is_empty() {
        return Err("benchmark suite resolved zero tasks".to_owned());
    }

    let max_feats = if opts.tier == CarbonTierWire::Free {
        64
    } else {
        256
    };
    let mut task_records = Vec::with_capacity(task_sources.len());

    for (idx, src) in task_sources.into_iter().enumerate() {
        let task_seed = opts.seed.wrapping_add(idx as u64 * 17);
        let (prepared, fit_x, fit_y, holdout_x, holdout_y, total_rows) = match src.kind {
            TaskDataKind::SingleTable(tbl, tgt) => {
                let enc =
                    prepare_encoded_training_set(&tbl, tgt.as_deref(), task_seed, 0.20, max_feats)?;
                (
                    enc.prepared,
                    enc.fit_x,
                    enc.fit_y,
                    enc.holdout_x,
                    enc.holdout_y,
                    enc.total_valid_rows,
                )
            }
            TaskDataKind::PreSplitTables {
                train,
                test,
                target,
            } => {
                // Encode `train` with an 85/15 internal split for Carbon upload while scoring ALL of `test`
                // via zero-knowledge `/predict` AND fitting all comparators on the exact same `train` FIT rows!
                let enc = prepare_encoded_training_set(
                    &train,
                    target.as_deref(),
                    task_seed,
                    0.15,
                    max_feats,
                )?;
                let test_enc_opt = enc.encoder.encode_prediction_table(&test)?;
                let t_col_idx = test.headers.len().saturating_sub(1);
                let mut ext_x = Vec::with_capacity(test.rows.len());
                let mut ext_y = Vec::with_capacity(test.rows.len());
                for (r_raw, r_enc) in test.rows.iter().zip(test_enc_opt) {
                    if let Some(yv) = r_raw
                        .get(t_col_idx)
                        .and_then(|c| c.as_deref())
                        .and_then(|s| s.trim().parse::<f64>().ok())
                        .filter(|v| v.is_finite())
                    {
                        ext_x.push(r_enc.into_iter().map(|v| v.unwrap_or(0.0)).collect());
                        ext_y.push(yv);
                    }
                }
                let tot = enc.fit_x.len() + ext_x.len();
                (enc.prepared, enc.fit_x, enc.fit_y, ext_x, ext_y, tot)
            }
        };

        base_client.reset_telemetry_counters();
        let t_wall = Instant::now();
        base_client.upload_prepared(&effective_tenant, opts.tier, &prepared)?;
        let (run_id, ds_id) = mint_ids()?;
        let run_str = run_id.to_string();

        let t_search = Instant::now();
        let mut prog = base_client.start_run(
            run_id,
            ds_id,
            &effective_tenant,
            opts.tier,
            opts.effort,
            "rmse",
            task_seed,
            &prepared,
        )?;
        let mut search_legs = 1u32;
        while prog.state == CarbonRunStateWire::Paused
            && prog.can_resume
            && prog.wave_index < prog.waves_target
            && search_legs < 4
        {
            search_legs += 1;
            let rem = prog.waves_target.saturating_sub(prog.wave_index).max(1);
            prog = base_client.resume_run(
                &effective_tenant,
                &run_str,
                Some(opts.effort),
                Some(rem),
            )?;
        }
        let search_ms = t_search.elapsed().as_secs_f64() * 1_000.0;

        // Score full holdout split via zero-knowledge encrypted `/predict` (chunked if > 1,000 rows)!
        let t_pred = Instant::now();
        let mut carbon_preds = Vec::with_capacity(holdout_x.len());
        for chunk in holdout_x.chunks(1_000) {
            let wire_rows: Vec<Vec<Option<f64>>> = chunk
                .iter()
                .map(|r| r.iter().copied().map(Some).collect())
                .collect();
            let resp = base_client.predict(
                &effective_tenant,
                &run_str,
                prepared.feature_names.clone(),
                wire_rows,
            )?;
            carbon_preds.extend_from_slice(&resp.predictions);
        }
        let predict_ms = t_pred.elapsed().as_secs_f64() * 1_000.0;
        let shred = base_client.shred_run(&effective_tenant, &run_str)?;
        let wall_ms = t_wall.elapsed().as_secs_f64() * 1_000.0;

        let (polls, req_b, resp_b) = base_client.telemetry_counters();
        let ingress = if base_client.enterprise_self_hosted() {
            CarbonIngressCostMode::SelfHostedVpc
        } else if base_client.active_endpoint().contains(".lambda-url.") {
            CarbonIngressCostMode::LambdaFunctionUrl
        } else {
            CarbonIngressCostMode::ApiGatewayHttpV2
        };
        let cost = compute_live_run_cost(&CarbonLiveCostInputs {
            ingress_mode: ingress,
            minted_ephemeral_token: minted_anon && idx == 0,
            engine_search_invocations: search_legs,
            poll_requests: polls,
            signaled_early_stop: false,
            shredded_after_run: shred.shredded,
            waves_completed: prog.wave_index,
            eval_shards: prog.eval_shards,
            eval_memory_mb: prog.eval_memory_mb,
            eval_invocations: prog.eval_invocations.0,
            eval_total_wall_ms: prog.eval_total_wall_ms.0,
            search_wall_ms: search_ms,
            predict_wall_ms: predict_ms,
            uploaded_cas_bytes: prepared.artifacts.fit_jmat.len()
                + prepared.artifacts.holdout_jmat.len(),
            http_request_bytes: req_b,
            http_response_bytes: resp_b,
        });

        let c_rmse = comp_rmse(&holdout_y, &carbon_preds).map_err(|e| e.to_string())?;
        let c_mae = comp_mae(&holdout_y, &carbon_preds).map_err(|e| e.to_string())?;
        let c_r2 = comp_r2(&holdout_y, &carbon_preds).ok();

        let y_mean = holdout_y.iter().sum::<f64>() / (holdout_y.len().max(1) as f64);
        let y_std = (holdout_y.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>()
            / (holdout_y.len().max(1) as f64))
            .sqrt()
            .max(1e-9);
        let c_norm_rmse = c_rmse / y_std;

        let comparators =
            evaluate_all_comparators_same_split(&fit_x, &fit_y, &holdout_x, &holdout_y, task_seed);
        let mut better_count = 0usize;
        for comp in &comparators {
            if comp.rmse + 1e-12 < c_rmse {
                better_count += 1;
            }
        }
        let carbon_rank = 1 + better_count;
        let total_methods = 1 + comparators.len();

        if show_stderr_progress {
            let best_comp = comparators.iter().min_by(|a, b| {
                a.rmse
                    .partial_cmp(&b.rmse)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            eprintln!(
                "[bench {:>2}] {:<28} | Carbon R^2={:>7.4} RMSE={:>8.4} (rank {}/{}) | best_baseline={} ({:.4}) | {:.2}s | ${:.5}",
                idx + 1,
                src.task_id,
                c_r2.unwrap_or(0.0),
                c_rmse,
                carbon_rank,
                total_methods,
                best_comp.map(|b| b.key.as_str()).unwrap_or("-"),
                best_comp.map(|b| b.rmse).unwrap_or(0.0),
                wall_ms / 1_000.0,
                cost.total_estimated_usd,
            );
        }

        task_records.push(CarbonTaskBenchRecord {
            task_id: src.task_id,
            panel: src.panel,
            description: src.description,
            total_rows,
            fit_rows: fit_x.len(),
            holdout_rows: holdout_y.len(),
            encoded_features: prepared.feature_names.len(),
            carbon_tier: opts.tier,
            carbon_effort: opts.effort,
            carbon_execution_mode: prog.execution_mode,
            carbon_eval_shards: prog.eval_shards,
            carbon_waves: prog.wave_index,
            carbon_candidates: prog.candidates_evaluated.0,
            carbon_wall_ms: wall_ms,
            carbon_search_ms: search_ms,
            carbon_predict_ms: predict_ms,
            carbon_estimated_usd: cost.total_estimated_usd,
            carbon_encrypted_cuen: prog.encrypted_cuen,
            carbon_shredded: shred.shredded,
            target_std: y_std,
            carbon_rmse: c_rmse,
            carbon_norm_rmse: c_norm_rmse,
            carbon_mae: c_mae,
            carbon_r2: c_r2,
            carbon_rank,
            total_methods,
            comparators,
        });
    }

    let leaderboard = build_aggregate_leaderboard(&task_records, opts.tier, opts.effort);
    let mut walls: Vec<f64> = task_records.iter().map(|t| t.carbon_wall_ms).collect();
    walls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50_wall_ms = walls[walls.len() / 2];
    let p95_idx = (((walls.len() - 1) as f64) * 0.95).round() as usize;
    let p95_wall_ms = walls[p95_idx.min(walls.len() - 1)];
    let mean_wall_ms = walls.iter().sum::<f64>() / (walls.len() as f64);
    let total_estimated_usd: f64 = task_records.iter().map(|t| t.carbon_estimated_usd).sum();
    let mean_usd_per_run = total_estimated_usd / (task_records.len() as f64);
    let usd_per_1000_runs = mean_usd_per_run * 1_000.0;
    let runs_per_1_usd = (mean_usd_per_run > 0.0).then_some(1.0 / mean_usd_per_run);
    let comparator_count = task_records
        .first()
        .map(|t| t.comparators.len())
        .unwrap_or(0);

    Ok(CarbonScientificBenchmarkReport {
        schema: "jopedime.carbon.scientific-benchmark.v1".to_owned(),
        timestamp_epoch_ms: now_epoch_ms(),
        suite: opts.suite.clone(),
        endpoint: base_client.active_endpoint().to_owned(),
        region: first_probe.region,
        carbon_tier: opts.tier,
        carbon_effort: opts.effort,
        zero_knowledge_cuen_all: task_records.iter().all(|t| t.carbon_encrypted_cuen),
        all_runs_shredded: task_records.iter().all(|t| t.carbon_shredded),
        task_count: task_records.len(),
        comparator_count,
        p50_wall_ms,
        p95_wall_ms,
        mean_wall_ms,
        total_estimated_usd,
        mean_usd_per_run,
        usd_per_1000_runs,
        runs_per_1_usd,
        provenance_notes,
        leaderboard,
        tasks: task_records,
    })
}

fn build_aggregate_leaderboard(
    tasks: &[CarbonTaskBenchRecord],
    tier: CarbonTierWire,
    effort: Effort,
) -> Vec<CarbonMethodAggregateSummary> {
    let mut method_keys: Vec<(String, String)> = vec![(
        "carbon_r882".to_owned(),
        format!("Carbon ({tier:?} / {effort:?}, r882 zero-knowledge cloud)"),
    )];
    if let Some(first) = tasks.first() {
        for c in &first.comparators {
            method_keys.push((c.key.clone(), c.label.clone()));
        }
    }

    // Compute exact average-tie ranks across all methods per task
    let mut per_task_ranks: Vec<BTreeMap<String, f64>> = Vec::with_capacity(tasks.len());
    for t in tasks {
        let mut entries: Vec<(String, f64)> = Vec::with_capacity(1 + t.comparators.len());
        entries.push(("carbon_r882".to_owned(), t.carbon_rmse));
        for c in &t.comparators {
            entries.push((c.key.clone(), c.rmse));
        }
        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut rmap = BTreeMap::new();
        let mut i = 0usize;
        while i < entries.len() {
            let mut j = i + 1;
            while j < entries.len() && (entries[j].1 - entries[i].1).abs() <= 1e-12 {
                j += 1;
            }
            let avg_rk = 0.5 * ((i + 1 + j) as f64);
            for (k, _) in &entries[i..j] {
                rmap.insert(k.clone(), avg_rk);
            }
            i = j;
        }
        per_task_ranks.push(rmap);
    }

    let mut summaries = Vec::with_capacity(method_keys.len());
    for (m_key, m_label) in method_keys {
        let mut ranks = Vec::new();
        let mut log_nrmse = Vec::new();
        let mut r2s = Vec::new();
        let mut wins = 0usize;
        let mut ties = 0usize;
        let mut losses = 0usize;
        let mut paired_diffs = Vec::new();

        for (t_idx, t) in tasks.iter().enumerate() {
            let (m_rmse, m_r2_opt) = if m_key == "carbon_r882" {
                (t.carbon_rmse, t.carbon_r2)
            } else if let Some(c) = t.comparators.iter().find(|c| c.key == m_key) {
                (c.rmse, c.r2)
            } else {
                continue;
            };
            if let Some(&rk) = per_task_ranks[t_idx].get(&m_key) {
                ranks.push(rk);
            }
            let nrmse = (m_rmse / t.target_std.max(1e-9)).max(1e-9);
            log_nrmse.push(nrmse.ln());
            if let Some(r2) = m_r2_opt {
                r2s.push(r2.clamp(-1.0, 1.0));
            }
            if m_key != "carbon_r882" {
                let scale = t.carbon_rmse.max(m_rmse).max(1e-9);
                let rel_diff = (m_rmse - t.carbon_rmse) / scale;
                if rel_diff.abs() <= 1e-3 {
                    ties += 1;
                } else if t.carbon_rmse < m_rmse {
                    wins += 1;
                } else {
                    losses += 1;
                }
                paired_diffs.push((m_rmse - t.carbon_rmse) / t.target_std.max(1e-9));
            }
        }
        if ranks.is_empty() {
            continue;
        }
        let mean_rank = ranks.iter().sum::<f64>() / (ranks.len() as f64);
        let geomean_norm_rmse = (log_nrmse.iter().sum::<f64>() / (log_nrmse.len() as f64)).exp();
        r2s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean_r2 = if r2s.is_empty() {
            0.0
        } else {
            r2s.iter().sum::<f64>() / (r2s.len() as f64)
        };
        let median_r2 = if r2s.is_empty() {
            0.0
        } else {
            r2s[r2s.len() / 2]
        };
        let p_val = if m_key == "carbon_r882" {
            None
        } else {
            paired_wilcoxon_two_sided_p(&paired_diffs)
        };

        summaries.push(CarbonMethodAggregateSummary {
            key: m_key,
            label: m_label,
            tasks_evaluated: ranks.len(),
            mean_rank,
            geomean_norm_rmse,
            mean_r2,
            median_r2,
            carbon_wins_vs_method: wins,
            carbon_ties_vs_method: ties,
            carbon_losses_vs_method: losses,
            paired_wilcoxon_p_value: p_val,
        });
    }

    summaries.sort_by(|a, b| {
        a.mean_rank
            .partial_cmp(&b.mean_rank)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.geomean_norm_rmse
                    .partial_cmp(&b.geomean_norm_rmse)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    summaries
}

/// Render a publication-ready GitHub-Flavored Markdown report for `CarbonScientificBenchmarkReport`.
pub fn render_scientific_benchmark_markdown(report: &CarbonScientificBenchmarkReport) -> String {
    let mut md = String::with_capacity(8_192);
    md.push_str("# Carbon Multi-Dataset Head-to-Head Tabular Benchmark Report\n\n");
    md.push_str(&format!(
        "- **Endpoint & Region**: `{}` (`{}`)\n- **Carbon Configuration**: `{:?}` tier, `{:?}` effort (`CUEN\\x01` zero-knowledge = `{}`, remote S3 shredded = `{}`)\n- **Suite & Scale**: `{}` (`{}` tasks x `{}` baselines + Carbon = `{}` total methods)\n- **Latency & Cost Frontier**: p50 `{:.2}s`, p95 `{:.2}s`, mean `${:.6}`/run (`${:.2}` per `1,000` runs, `{:.0}` runs / `$1`)\n\n",
        report.endpoint,
        report.region,
        report.carbon_tier,
        report.carbon_effort,
        report.zero_knowledge_cuen_all,
        report.all_runs_shredded,
        report.suite,
        report.task_count,
        report.comparator_count,
        report.comparator_count + 1,
        report.p50_wall_ms / 1_000.0,
        report.p95_wall_ms / 1_000.0,
        report.mean_usd_per_run,
        report.usd_per_1000_runs,
        report.runs_per_1_usd.unwrap_or(0.0),
    ));

    md.push_str("## 1. Aggregate Leaderboard (Same-Split Outer Holdout)\n\n");
    md.push_str("| Rank | Method | Mean Rank | Geomean Norm-RMSE | Mean R² | Median R² | Carbon W / T / L | Wilcoxon p-val |\n");
    md.push_str("| ---: | :--- | ---: | ---: | ---: | ---: | :---: | ---: |\n");
    for (idx, row) in report.leaderboard.iter().enumerate() {
        let wtl = if row.key == "carbon_r882" {
            "—".to_owned()
        } else {
            format!(
                "{} / {} / {}",
                row.carbon_wins_vs_method, row.carbon_ties_vs_method, row.carbon_losses_vs_method
            )
        };
        let p_str = row
            .paired_wilcoxon_p_value
            .map(|p| format!("{p:.4}"))
            .unwrap_or_else(|| "—".to_owned());
        md.push_str(&format!(
            "| {} | **{}** | {:.2} | {:.4} | {:.4} | {:.4} | {} | {} |\n",
            idx + 1,
            row.label,
            row.mean_rank,
            row.geomean_norm_rmse,
            row.mean_r2,
            row.median_r2,
            wtl,
            p_str
        ));
    }

    md.push_str("\n## 2. Per-Task Breakdown (Carbon vs Strongest Same-Split Baseline)\n\n");
    md.push_str("| Task | Panel | Shape (N x D) | Mode | Carbon RMSE | Carbon R² | Rank | Best Baseline (RMSE) | Wall (s) | Est. USD |\n");
    md.push_str("| :--- | :--- | :---: | :--- | ---: | ---: | :---: | :--- | ---: | ---: |\n");
    for t in &report.tasks {
        let best_b = t.comparators.iter().min_by(|a, b| {
            a.rmse
                .partial_cmp(&b.rmse)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let b_str = best_b
            .map(|b| format!("`{}` ({:.4})", b.key, b.rmse))
            .unwrap_or_else(|| "—".to_owned());
        md.push_str(&format!(
            "| `{}` | `{}` | `{}x{}` | `{:?}` | {:.4} | {:.4} | `{}/{}` | {} | {:.2}s | `${:.5}` |\n",
            t.task_id,
            t.panel,
            t.total_rows,
            t.encoded_features,
            t.carbon_execution_mode,
            t.carbon_rmse,
            t.carbon_r2.unwrap_or(0.0),
            t.carbon_rank,
            t.total_methods,
            b_str,
            t.carbon_wall_ms / 1_000.0,
            t.carbon_estimated_usd,
        ));
    }

    md.push_str("\n## 3. Scientific Methodology & Comparator Provenance\n\n");
    for note in &report.provenance_notes {
        md.push_str(&format!("- {note}\n"));
    }
    if let Some(first) = report.tasks.first() {
        md.push_str("\n| Comparator Key | Reference | Known Gaps |\n| :--- | :--- | :--- |\n");
        for c in &first.comparators {
            md.push_str(&format!(
                "| `{}` | {} | {} |\n",
                c.key, c.reference, c.known_gaps
            ));
        }
    }
    md
}
