//! Deterministic, fixture-sized planted regression tasks.
//!
//! The solver view is intentionally a separate type from the hidden answer.
//! Callers must opt in to `hidden_answer()`; serialization of `SolverTask`
//! cannot expose program text or weights.

use anyhow::{Context, Result, ensure};
use feat_math::{Rows, genome::GenomeIr, model};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Instant;

pub const GENERATOR_ID: &str = "jopedime.planted-linear.v1";
pub const TASK_SCHEMA: &str = "jope.dream-task/v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolverTask {
    pub schema: String,
    pub generator_id: String,
    pub seed: u64,
    pub x_train: Rows,
    pub y_train: Vec<f64>,
    pub noise: NoiseModel,
    /// Hash of the TRAIN-only matrix used to construct this fixture.
    pub source_x_sha256: String,
    /// Explicit role guard. This fixture generator has no non-train input path.
    pub source_role: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoiseModel {
    pub kind: String,
    pub sigma: f64,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Misspecification {
    InGrammar,
    UnmodeledSmoothTerm,
}

#[derive(Clone, Debug)]
struct HiddenAnswer {
    program: String,
    weights: Vec<f64>,
    intercept: f64,
    misspecification: Misspecification,
}

#[derive(Clone, Debug)]
pub struct PlantedTask {
    solver: SolverTask,
    hidden: HiddenAnswer,
}

impl PlantedTask {
    /// Generate from a caller-supplied TRAIN-role feature matrix. Validation and
    /// test rows are never accepted by this API.
    pub fn generate(
        x_train: Rows,
        seed: u64,
        sigma: f64,
        misspecification: Misspecification,
    ) -> Result<Self> {
        ensure!(x_train.len() >= 8, "at least eight TRAIN rows are required");
        ensure!(
            x_train
                .iter()
                .all(|row| row.len() >= 2 && row.iter().all(|x| x.is_finite())),
            "TRAIN matrix must have two finite columns"
        );
        ensure!(
            sigma.is_finite() && sigma >= 0.0,
            "noise sigma must be finite and nonnegative"
        );

        let weights = vec![0.75, 0.0];
        let intercept = 0.2;
        let program = match misspecification {
            Misspecification::InGrammar => "(sym affine \"c0\")".to_owned(),
            Misspecification::UnmodeledSmoothTerm => "(sym affine \"c0\")".to_owned(),
        };
        let noise_seed = seed ^ 0x4a4f_5045_4e4f_4953;
        let mut noise_rng = SplitMix64(noise_seed);
        let y_train = x_train
            .iter()
            .map(|row| {
                let smooth = if misspecification == Misspecification::UnmodeledSmoothTerm {
                    0.5 * row[1].sin()
                } else {
                    0.0
                };
                intercept
                    + weights[0] * row[0]
                    + weights[1] * row[1]
                    + smooth
                    + sigma * noise_rng.normal()
            })
            .collect();
        let source_x_sha256 = sha256(&serde_json::to_vec(&x_train)?);
        Ok(Self {
            solver: SolverTask {
                schema: TASK_SCHEMA.into(),
                generator_id: GENERATOR_ID.into(),
                seed,
                x_train,
                y_train,
                noise: NoiseModel {
                    kind: "gaussian_iid_box_muller_v1".into(),
                    sigma,
                    seed: noise_seed,
                },
                source_x_sha256,
                source_role: "train".into(),
            },
            hidden: HiddenAnswer {
                program,
                weights,
                intercept,
                misspecification,
            },
        })
    }

    pub fn solver_task(&self) -> &SolverTask {
        &self.solver
    }

    /// Hidden truth is available only through this explicit evaluation hook;
    /// never pass this value to context, retrieval, or training-row exporters.
    pub fn hidden_answer(&self) -> HiddenAnswerView<'_> {
        HiddenAnswerView {
            program: &self.hidden.program,
            weights: &self.hidden.weights,
            intercept: self.hidden.intercept,
            misspecification: self.hidden.misspecification,
        }
    }

    pub fn solver_context_json(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(&self.solver)?)
    }

    pub fn oracle_refit(&self, repeats: usize) -> Result<OracleRefit> {
        ensure!(repeats > 0 && repeats <= 32, "repeats must be in 1..=32");
        let genome = GenomeIr::parse(&self.hidden.program)
            .context("parse hidden construction for oracle refit")?;
        let start = Instant::now();
        let mut squared_error = 0.0;
        let mut fit_count = 0u32;
        let mut fit_elapsed_ns = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let fit_start = Instant::now();
            let fitted = model::fit(&genome, &self.solver.x_train, &self.solver.y_train)?;
            let predictions = fitted.predict(&self.solver.x_train)?;
            fit_elapsed_ns.push(fit_start.elapsed().as_nanos());
            squared_error = predictions
                .iter()
                .zip(&self.solver.y_train)
                .map(|(p, y)| (p - y).powi(2))
                .sum::<f64>()
                / predictions.len() as f64;
            fit_count += 1;
        }
        let elapsed_ns = start.elapsed().as_nanos();
        Ok(OracleRefit {
            rmse: squared_error.sqrt(),
            noise_floor_rmse: self.solver.noise.sigma,
            repeats: fit_count,
            fit_elapsed_ns,
            charged_elapsed_ns: elapsed_ns,
            claim: "attainable_reference_not_global_optimum".into(),
        })
    }
}

pub struct HiddenAnswerView<'a> {
    pub program: &'a str,
    pub weights: &'a [f64],
    pub intercept: f64,
    pub misspecification: Misspecification,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OracleRefit {
    pub rmse: f64,
    pub noise_floor_rmse: f64,
    pub repeats: u32,
    /// Wall time per fit, including prediction for its RMSE measurement.
    pub fit_elapsed_ns: Vec<u128>,
    pub charged_elapsed_ns: u128,
    pub claim: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolvabilityGuard {
    pub accepted: bool,
    pub reason: String,
    pub train_rows: usize,
    pub signal_variance: f64,
    pub noise_variance: f64,
}

impl SolvabilityGuard {
    pub fn evaluate(task: &SolverTask) -> Result<Self> {
        ensure!(task.schema == TASK_SCHEMA, "unsupported task schema");
        ensure!(
            task.generator_id == GENERATOR_ID,
            "unknown generator identity"
        );
        ensure!(task.source_role == "train", "source role must be train");
        ensure!(
            sha256(&serde_json::to_vec(&task.x_train)?) == task.source_x_sha256,
            "TRAIN matrix digest mismatch"
        );
        ensure!(
            task.x_train.len() == task.y_train.len(),
            "feature/target length mismatch"
        );
        ensure!(
            task.y_train.iter().all(|y| y.is_finite()),
            "target contains non-finite values"
        );
        let mean = task.y_train.iter().sum::<f64>() / task.y_train.len() as f64;
        let observed_variance = task.y_train.iter().map(|y| (y - mean).powi(2)).sum::<f64>()
            / task.y_train.len() as f64;
        let noise_variance = task.noise.sigma.powi(2);
        let signal_variance = (observed_variance - noise_variance).max(0.0);
        let accepted = task.y_train.len() >= 8 && signal_variance > noise_variance * 0.05;
        Ok(Self {
            accepted,
            reason: if accepted {
                "train_signal_exceeds_declared_noise"
            } else {
                "insufficient_signal_above_declared_noise"
            }
            .into(),
            train_rows: task.y_train.len(),
            signal_variance,
            noise_variance,
        })
    }

    /// A finite set of predictions can find collisions, but cannot certify
    /// semantic equivalence outside those probes.
    pub fn finite_probe_similarity(
        left: &[f64],
        right: &[f64],
        tolerance: f64,
    ) -> Result<ProbeComparison> {
        ensure!(
            left.len() == right.len(),
            "probe vectors must have equal length"
        );
        ensure!(
            tolerance.is_finite() && tolerance >= 0.0,
            "tolerance must be finite and nonnegative"
        );
        ensure!(
            left.iter().chain(right).all(|value| value.is_finite()),
            "probe values must be finite"
        );
        Ok(ProbeComparison {
            within_tolerance: left
                .iter()
                .zip(right)
                .all(|(a, b)| (a - b).abs() <= tolerance),
            semantic_equivalence_proven: false,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeComparison {
    pub within_tolerance: bool,
    pub semantic_equivalence_proven: bool,
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
