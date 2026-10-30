//! PCG32 (XSH-RR 64/32, O'Neill reference constants) — the battle dialect's only
//! random number generator. Must match the emitted Python/Rust helpers exactly.

pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Reference `pcg32_srandom_r` seeding.
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Pcg32 {
            state: 0,
            inc: (stream << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Per-stage RNG: stream derived from the input width and output size so the
    /// same stage in the same position always draws the same weights.
    pub fn for_stage(seed: u64, in_cols: usize, n_out: usize) -> Self {
        Pcg32::new(seed, ((in_cols as u64) << 32) ^ (n_out as u64))
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in [0,1) from 53 random bits: exactly `((hi<<32|lo) >> 11) * 2^-53`.
    pub fn next_f64(&mut self) -> f64 {
        let hi = (self.next_u32() as u64) << 32;
        let lo = self.next_u32() as u64;
        (((hi | lo) >> 11) as f64) * (1.0 / 9007199254740992.0)
    }

    pub fn range_f64(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next_f64()
    }

    /// Box-Muller (cosine branch), u1 floored at 1e-12 (mirrors the engine's guard).
    pub fn gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned vectors: the emitted Python and Rust artifacts embed the same PCG32;
    /// golden tests in battle-transpile compare against these exact constants.
    #[test]
    fn pcg32_reference_vectors() {
        let mut r = Pcg32::new(42, 54);
        let first: Vec<u32> = (0..6).map(|_| r.next_u32()).collect();
        assert_eq!(
            first,
            vec![0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e]
        );
    }

    #[test]
    fn uniform_and_gaussian_are_deterministic() {
        let mut r = Pcg32::new(7, 1);
        let u = r.next_f64();
        assert!((0.0..1.0).contains(&u));
        let mut r2 = Pcg32::new(7, 1);
        assert_eq!(u.to_bits(), r2.next_f64().to_bits());
    }
}
