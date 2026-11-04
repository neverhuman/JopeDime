//! The `parallel.seed_population.v1` file that jope-dime `search --seeds` accepts
//! (`gp.rs:551-584`, checks at `gp.rs:1359-1375`), and the end-to-end proposal that fills it.
//!
//! Layout written here: 64 pool programs in ranking order, each with its normalized
//! `rank_log_weight` (softmax log-weight of the predicted quality, `attach_policy_logprobs`),
//! followed by 16 grammar-uniform exploration programs. Origins are recorded as they happened:
//! `neural_policy` for sampled programs, `neural_rejection_fallback` for pool slots the policy
//! did not fill, `random_exploration` for the tail.
//!
//! Provenance: a [`SeedPopulation`] is only built inside [`propose`] and [`propose_challenger`]
//! (or the `testing` helpers, which refuse the accepted state), and [`write_population`] only
//! writes the population of a [`Proposal`], which has private fields. A file read back by
//! [`validate_for_search`] is a [`SeedFile`], which no writer accepts.
mod population;
mod proposal;

#[cfg(any(test, feature = "testing"))]
pub use population::assemble_for_testing;
pub use population::{
    logprob_from_scores, population_bytes, validate_for_search, write_population, SeedBinding,
    SeedCandidate, SeedFile, SeedPopulation, ORIGIN_NEURAL, ORIGIN_NEURAL_FALLBACK, ORIGIN_RANDOM,
    SEED_COUNT, SEED_SCHEMA,
};
#[cfg(any(test, feature = "testing"))]
pub use proposal::propose_unbound;
#[cfg(test)]
pub(crate) use proposal::propose_with_vocabulary_for_test;
pub use proposal::{propose, propose_challenger, Proposal, ProposalSeeds};
