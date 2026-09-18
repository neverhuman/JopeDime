//! Bit-exact Rust port of the numpy 1.26 random machinery that produced the
//! frozen jope-dime encoder sampling recipes: `SeedSequence`, `PCG64`, and the
//! `Generator` integer paths behind `permutation`, `shuffle`, `choice` and
//! `integers`. The committed `sampling.json` files are the oracle; no Python
//! or numpy runs at build or test time.
#![forbid(unsafe_code)]

pub mod blake2b;
pub mod document;
pub mod generator;
pub mod pcg64;
pub mod recipes;
pub mod seed_sequence;

pub use generator::{default_rng, Generator, SampleError};
pub use pcg64::Pcg64;
pub use seed_sequence::SeedSequence;
