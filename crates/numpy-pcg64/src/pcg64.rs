//! `numpy.random.PCG64` (numpy 1.26, `_pcg64.pyx` / `pcg64.h`): PCG XSL-RR
//! 128/64 with numpy's seeding and 32-bit output buffering.

use crate::seed_sequence::SeedSequence;

/// PCG_DEFAULT_MULTIPLIER_128.
const MULTIPLIER: u128 = (0x2360_ed05_1fc6_5da4_u128 << 64) | 0x4385_df64_9fcc_f645_u128;

/// numpy's PCG64 bit generator state, including the buffered upper half of the
/// last 64-bit draw used for `next_uint32`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64 {
    /// `PCG64(SeedSequence(...))`: four 64-bit words give the 128-bit state
    /// seed (`val[0] << 64 | val[1]`) and stream (`val[2] << 64 | val[3]`).
    pub fn from_seed_sequence(seed: &SeedSequence) -> Self {
        let val = seed.generate_state_u64(4);
        let initstate = (u128::from(val[0]) << 64) | u128::from(val[1]);
        let initseq = (u128::from(val[2]) << 64) | u128::from(val[3]);
        Self::from_state_and_sequence(initstate, initseq)
    }

    /// `pcg_setseq_128_srandom_r(initstate, initseq)`.
    pub fn from_state_and_sequence(initstate: u128, initseq: u128) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (initseq << 1) | 1,
            has_uint32: false,
            uinteger: 0,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    /// Current `(state, inc)` pair, as numpy's `bit_generator.state` reports.
    pub fn state(&self) -> (u128, u128) {
        (self.state, self.inc)
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(MULTIPLIER).wrapping_add(self.inc);
    }

    /// `pcg64_next64`: advance, then XSL-RR output of the new state.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let xored = ((self.state >> 64) as u64) ^ (self.state as u64);
        xored.rotate_right((self.state >> 122) as u32)
    }

    /// `pcg64_next32`: the low half of a fresh 64-bit draw, then its buffered
    /// high half on the following call.
    pub fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        next as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_u32_splits_one_u64_low_then_high() {
        let seed = SeedSequence::new(20_260_707);
        let mut wide = Pcg64::from_seed_sequence(&seed);
        let mut narrow = wide.clone();
        let first = wide.next_u64();
        assert_eq!(narrow.next_u32(), first as u32);
        assert_eq!(narrow.next_u32(), (first >> 32) as u32);
        assert_eq!(narrow.next_u64(), wide.next_u64());
    }

    #[test]
    fn stream_increment_is_odd() {
        let rng = Pcg64::from_state_and_sequence(0, 0);
        assert_eq!(rng.state().1, 1);
    }
}
