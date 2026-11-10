//! SplitMix64: the explicit, seedable generator behind every random choice in this crate.
//!
//! No global or thread-local state is used, so a fit is a pure function of its inputs and seed.

/// `SplitMix64` is a struct for the deterministic random stream contract in this module: a
/// 64-bit SplitMix generator whose whole state is the explicit seed it was created from.
#[derive(Clone, Debug)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, bound)` by multiply-shift; `0` when `bound` is `0`.
    pub(crate) fn below(&mut self, bound: usize) -> usize {
        ((u128::from(self.next_u64()) * bound as u128) >> 64) as usize
    }

    /// Fisher–Yates shuffle.
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            let other = self.below(index + 1);
            items.swap(index, other);
        }
    }
}

/// An independent seed for sub-stream `stream` (for example one per tree) of `seed`.
pub(crate) fn stream_seed(seed: u64, stream: u64) -> u64 {
    SplitMix64::new(seed ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03)).next_u64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn below_stays_in_range_and_covers_it() {
        let mut rng = SplitMix64::new(7);
        let mut seen = [false; 5];
        for _ in 0..500 {
            let value = rng.below(5);
            assert!(value < 5);
            seen[value] = true;
        }
        assert!(seen.iter().all(|hit| *hit));
        assert_eq!(rng.below(0), 0);
    }

    #[test]
    fn streams_are_reproducible_and_distinct() {
        assert_eq!(stream_seed(11, 3), stream_seed(11, 3));
        assert_ne!(stream_seed(11, 3), stream_seed(11, 4));
        assert_ne!(stream_seed(11, 3), stream_seed(12, 3));
    }

    #[test]
    fn shuffle_is_a_seeded_permutation() {
        let mut first: Vec<u32> = (0..20).collect();
        let mut second = first.clone();
        SplitMix64::new(5).shuffle(&mut first);
        SplitMix64::new(5).shuffle(&mut second);
        assert_eq!(first, second);
        let mut sorted = first.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<u32>>());
        assert_ne!(first, sorted);
    }
}
