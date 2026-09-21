//! Deterministic, seedable RNG used everywhere (reproducible across platforms via ChaCha8).
pub use rand::{Rng, SeedableRng};

/// The project RNG type. ChaCha8 is fast and platform-deterministic.
pub type Rng64 = rand_chacha::ChaCha8Rng;

/// Construct a seeded RNG.
pub fn seeded(seed: u64) -> Rng64 {
    rand_chacha::ChaCha8Rng::seed_from_u64(seed)
}
