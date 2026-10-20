//! Product adapters for the pure feat-math strategy kernel.
//!
//! Driver code owns profile selection, input identity, cancellation and the actual evaluator.
//! The shared kernel owns proposal ordering, duplicate admission, bounded allocation and trace
//! ordering. Existing `search` and kernel2 routes remain the default until explicitly selected.

use anyhow::{ensure, Result};
use feat_math::kernel::{
    profile::SearchProfile,
    search::random_program,
    strategy::{
        execute, Allocator, Candidate, Cost, Evaluation, Evaluator, Identity, Proposal, Proposer,
        Reservation, Run, Strategy, StrategyConfig,
    },
};
use sha2::{Digest, Sha256};

use crate::{
    clock::Clock,
    kernel2::{Kernel2Evaluator, Kernel2Observation},
    prepare::Cut,
};

/// Result of a product cut executed by the shared deterministic kernel.
#[derive(Clone, Debug)]
pub struct ProductRun {
    pub run: Run,
    pub measurements: Vec<Kernel2Observation>,
}

struct ProductProposer<'a> {
    profile: &'a SearchProfile,
    width: usize,
    seed: u64,
}

impl Proposer for ProductProposer<'_> {
    fn propose(&mut self, attempt: usize) -> Option<Proposal> {
        let candidate = random_program(
            self.profile,
            self.width,
            self.seed.wrapping_add(attempt as u64),
        )
        .ok()?;
        Some(Proposal {
            candidate: Candidate {
                id: format!("proposal-{attempt}"),
                bytes: candidate.into_bytes(),
            },
            origin: "random".to_owned(),
        })
    }
}

struct UnitAllocator;
impl Allocator for UnitAllocator {
    fn reserve(&mut self, candidate: &Candidate, remaining: u64) -> Option<Reservation> {
        (remaining > 0).then(|| Reservation {
            id: format!("evaluation:{}", candidate.id),
            cost: Cost { units: 1 },
        })
    }
}

struct ProductEvaluator<'a> {
    evaluator: &'a Kernel2Evaluator,
    cut: &'a Cut,
    measurements: Vec<Kernel2Observation>,
}

impl Evaluator for ProductEvaluator<'_> {
    fn evaluate(&mut self, candidate: &Candidate, _: &Reservation) -> Evaluation {
        let genome = String::from_utf8_lossy(&candidate.bytes).into_owned();
        let generation = candidate
            .id
            .trim_start_matches("proposal-")
            .parse()
            .unwrap_or(0);
        let observation =
            self.evaluator
                .evaluate(&genome, self.cut, generation, "random", Clock::Wall);
        // The generic strategy ranks lower scores first; kernel2's score is higher-is-better.
        let score = observation.selection_score.map(|value| -value);
        let outcome = observation.observation.status.to_owned();
        self.measurements.push(observation);
        Evaluation { outcome, score }
    }
}

/// Run a bounded opt-in kernel2 route through the shared pure search state machine.
/// Each unique candidate receives one declared evaluation unit. The route is deterministic
/// because it uses a fixed seed stream. Wall-clock readings are excluded from its trace;
/// ordinary `run_cut` is unchanged.
pub fn run_kernel2(
    cut: &Cut,
    evaluator: &Kernel2Evaluator,
    seed: u64,
    source_commit: &str,
    max_attempts: usize,
) -> Result<ProductRun> {
    ensure!(
        max_attempts > 0,
        "shared strategy needs at least one attempt"
    );
    let profile = evaluator.profile();
    let identity = kernel2_identity(cut, profile, evaluator, seed, source_commit)?;
    let strategy = Strategy {
        identity,
        config: StrategyConfig {
            max_attempts,
            cost_budget: max_attempts as u64,
            // Match kernel::codec::MAX_BYTES (private to feat-math) until a public contract is allocated.
            max_candidate_bytes: 16 * 1024 * 1024,
        },
    };
    let mut proposer = ProductProposer {
        profile,
        width: cut.train_x.first().map_or(0, Vec::len),
        seed,
    };
    let mut allocator = UnitAllocator;
    let mut adapter = ProductEvaluator {
        evaluator,
        cut,
        measurements: Vec::new(),
    };
    let run = execute(&strategy, &mut proposer, &mut allocator, &mut adapter)?;
    Ok(ProductRun {
        run,
        measurements: adapter.measurements,
    })
}

/// Digest ordered training and held-out bytes to bind a trace to its exact cut.
pub fn cut_input_sha256(cut: &Cut) -> String {
    let bytes = serde_json::to_vec(&(
        &cut.train_x,
        &cut.train_y,
        &cut.hold_x,
        &cut.hold_y,
        &cut.hold_indices,
    ))
    .expect("cut serializes");
    format!("{:x}", Sha256::digest(bytes))
}

/// Build the deterministic kernel2 adapter identity with an explicit input digest.
pub fn kernel2_identity(
    cut: &Cut,
    profile: &SearchProfile,
    evaluator: &Kernel2Evaluator,
    seed: u64,
    source_commit: &str,
) -> Result<Identity> {
    ensure!(
        source_commit_is_full_sha(source_commit),
        "shared strategy requires a full source commit SHA"
    );
    let mut identity = Identity::new(
        profile.id,
        profile.revision,
        profile.sha256(),
        "kernel2.wp1.5",
        format!(
            "higher-selection-score-v1;fault-policy:{:?}",
            evaluator.fault_policy()
        ),
        "kernel2.random_program(seed+attempt)-v1",
        "one-evaluation-unit-v1",
    );
    identity.seed = seed;
    identity.input_sha256 = cut_input_sha256(cut);
    identity.context_id = format!("cut:{}", cut.index);
    identity.source_id = source_commit.to_ascii_lowercase();
    Ok(identity)
}

fn source_commit_is_full_sha(source_commit: &str) -> bool {
    source_commit.len() == 40 && source_commit.bytes().all(|byte| byte.is_ascii_hexdigit())
}
