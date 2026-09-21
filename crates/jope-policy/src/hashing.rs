//! SplitMix64 finalizer, copied from feat-core `hashing.rs` (`seeds::ProposalSeeds::derive` is the
//! only caller; the rest of that module needs feat-core's data layer and is not ported).

/// SplitMix64 finalizer — strong avalanche, fully deterministic.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
