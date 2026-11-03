use super::*;

pub(crate) fn summary_with(m1_rmse: f64, m3: f64) -> Summary {
    Summary {
        m1_rmse,
        m1_r2: 0.5,
        m2_rmse: m1_rmse,
        m3,
        m6_rate: 0.0,
        m7_stages: 2.0,
        m7_expression_nodes: None,
        m8_tokens: Some(1000.0),
        m8_bytes: Some(9000.0),
        m11: 0.3,
    }
}

pub(crate) fn slot(name: &'static str, rmse: [f64; 2], sd: Option<f64>, m3: f64) -> SlotAggregate {
    SlotAggregate {
        slot: name,
        profile: "legacy",
        datasets: rmse
            .iter()
            .enumerate()
            .map(|(i, r)| DatasetAggregate {
                dataset: format!("d{i}"),
                seeds: 5,
                m1_rmse_seed_sd: sd,
                mean: summary_with(*r, m3),
            })
            .collect(),
        overall: summary_with((rmse[0] + rmse[1]) / 2.0, m3),
    }
}

/// Evaluations of the PR-tier kernel2 comparison, and its refinement allowance: one refined
/// parent per round with two objective trials.
pub(crate) const KERNEL2_EVALUATIONS: usize = 16;
pub(crate) const SMALL_REFINEMENT: Refinement = Refinement {
    top_k: 1,
    trials_per_program: 2,
};

/// The PR-tier kernel2 comparison: the legacy comparison's two datasets with kernel2-baseline
/// on both sides and seed 6 only. Seed 5 draws boosted-tree heads of 1000 rounds there, which
/// take 13 to 20 seconds a search in a debug build; seed 6's take about 3 and still spend
/// refinement trials.
pub(crate) fn kernel2_config() -> Config {
    Config {
        profile_a: Profile::Kernel2Baseline,
        profile_b: Profile::Kernel2Baseline,
        seeds: vec![6],
        evaluations: KERNEL2_EVALUATIONS,
        refinement: Some(SMALL_REFINEMENT),
        ..small_config()
    }
}

/// kernel2-baseline against itself, twice with different job counts: byte-identical bodies
/// (wall time is in `timing`, never in the body), no difference, the route's own winner as
/// M1, consistent budget accounting, and a paired comparison of nothing but ties.
pub(crate) fn sorted_keys(value: &serde_json::Value) -> Vec<String> {
    let mut keys: Vec<String> = value.as_object().unwrap().keys().cloned().collect();
    keys.sort();
    keys
}

/// The kernel2 arms leave the legacy document as it was: the legacy profile record and a
/// legacy cut record serialize to the bytes they had before, and a legacy-only run's body,
/// run records and timing carry exactly the keys they had before.
/// A run record carrying what the paired comparison reads: slot, dataset, seed and M1.
pub(crate) fn paired_run(
    slot: &'static str,
    dataset: &str,
    seed: u64,
    rmse: f64,
    program: &str,
) -> RunRecord {
    RunRecord {
        slot,
        profile: "kernel2-baseline",
        dataset: dataset.to_owned(),
        seed,
        metrics: Metrics {
            m1: Best {
                rmse,
                r2: 0.5,
                cuts_measured: 1,
                program_sha256: sha256_hex(program.as_bytes()),
                program: program.to_owned(),
            },
            m2: TopMedian {
                programs: 1,
                rmse,
                r2: 0.5,
            },
            m3: Speed {
                per_cut: vec![1],
                mean: 1.0,
            },
            m4: None,
            m5: None,
            m6: Invalid {
                rate: 0.0,
                invalid: 0,
                attempted: 1,
                by_status: BTreeMap::new(),
            },
            m7: Shape {
                programs: 1,
                mean_stages: 1.0,
                expression_stages: 0,
                mean_expression_nodes: None,
                mean_expression_depth: None,
                max_expression_nodes: None,
                max_expression_depth: None,
            },
            m8: Cost {
                python_bytes: None,
                rust_bytes: None,
                python_tokens: None,
                rust_tokens: None,
                unsupported: None,
            },
            m9: None,
            m11: Yield {
                per_evaluation: 0.0,
                valid_derived: 0,
                evaluations: 1,
            },
        },
        cuts: Vec::new(),
    }
}

/// The paired comparison on a synthetic input whose every value is exact in binary: wins,
/// ties and losses overall and per dataset, both medians, the differences, the largest gain
/// and the mean without it, the sign test, the same-winner count, an all-tie input, the
/// bit-equality rule for a tie, and the refusal of unpaired runs.
#[path = "tests_1/kernel2.rs"]
mod kernel2;
#[path = "tests_1/legacy.rs"]
mod legacy;
#[path = "tests_1/paired.rs"]
mod paired;

pub(crate) use kernel2::*;
pub(crate) use legacy::*;
pub(crate) use paired::*;
