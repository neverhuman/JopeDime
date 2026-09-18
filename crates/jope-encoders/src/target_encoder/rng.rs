use super::*;
use std::collections::HashSet;

/// NumPy 1.26 `SeedSequence(integer)` plus the original PCG64 stream.
pub(crate) struct NumpyPcg64 {
    state: u128,
    increment: u128,
    buffered_u32: Option<u32>,
}

impl NumpyPcg64 {
    const MULTIPLIER: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

    pub(crate) fn new(seed: u64) -> Self {
        let words = seed_sequence(seed);
        // NumPy passes each pair to pcg64_set_seed as [high, low].
        let initial_state = ((words[0] as u128) << 64) | words[1] as u128;
        let sequence = ((words[2] as u128) << 64) | words[3] as u128;
        let increment = sequence.wrapping_shl(1) | 1;
        let mut rng = Self {
            state: 0,
            increment,
            buffered_u32: None,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initial_state);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(self.increment);
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.step();
        let rotation = (self.state >> 122) as u32;
        let xsl = ((self.state >> 64) as u64) ^ self.state as u64;
        xsl.rotate_right(rotation)
    }

    fn next_u32(&mut self) -> u32 {
        if let Some(value) = self.buffered_u32.take() {
            return value;
        }
        let value = self.next_u64();
        self.buffered_u32 = Some((value >> 32) as u32);
        value as u32
    }

    fn bounded_u32(&mut self, inclusive_max: u32) -> u32 {
        if inclusive_max == 0 {
            return 0;
        }
        let range = inclusive_max.wrapping_add(1);
        loop {
            let product = self.next_u32() as u64 * range as u64;
            let leftover = product as u32;
            let threshold = (u32::MAX - inclusive_max) % range;
            if leftover >= threshold {
                return (product >> 32) as u32;
            }
        }
    }

    fn interval(&mut self, inclusive_max: usize) -> usize {
        if inclusive_max == 0 {
            return 0;
        }
        let mut mask = inclusive_max as u64;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        loop {
            let value = if inclusive_max <= u32::MAX as usize {
                self.next_u32() as u64 & mask
            } else {
                self.next_u64() & mask
            };
            if value <= inclusive_max as u64 {
                return value as usize;
            }
        }
    }

    pub(crate) fn permutation(&mut self, length: usize) -> Vec<usize> {
        let mut values = (0..length).collect::<Vec<_>>();
        for index in (1..length).rev() {
            let swap = self.interval(index);
            values.swap(index, swap);
        }
        values
    }

    pub(crate) fn choice(
        &mut self,
        population: usize,
        size: usize,
        replace: bool,
    ) -> Result<Vec<usize>> {
        if population == 0 || (!replace && size > population) {
            bail!("invalid target sampling request");
        }
        if replace {
            return Ok((0..size)
                .map(|_| self.bounded_u32((population - 1) as u32) as usize)
                .collect());
        }
        let mut selected = HashSet::with_capacity(size);
        let mut values = Vec::with_capacity(size);
        for upper in population - size..population {
            let candidate = self.bounded_u32(upper as u32) as usize;
            let value = if selected.insert(candidate) {
                candidate
            } else {
                selected.insert(upper);
                upper
            };
            values.push(value);
        }
        for index in (1..size).rev() {
            let swap = self.bounded_u32(index as u32) as usize;
            values.swap(index, swap);
        }
        Ok(values)
    }
}

pub(super) fn seed_sequence(seed: u64) -> [u64; 4] {
    const INIT_A: u32 = 0x43b0_d7e5;
    const MULT_A: u32 = 0x931e_8875;
    const INIT_B: u32 = 0x8b51_f9dd;
    const MULT_B: u32 = 0x58f3_8ded;
    const MIX_L: u32 = 0xca01_f9dd;
    const MIX_R: u32 = 0x4973_f715;

    fn hashmix(value: u32, constant: &mut u32) -> u32 {
        let mut value = value ^ *constant;
        *constant = constant.wrapping_mul(MULT_A);
        value = value.wrapping_mul(*constant);
        value ^ (value >> 16)
    }
    fn mix(left: u32, right: u32) -> u32 {
        let value = MIX_L
            .wrapping_mul(left)
            .wrapping_sub(MIX_R.wrapping_mul(right));
        value ^ (value >> 16)
    }

    let entropy = [seed as u32, (seed >> 32) as u32];
    let mut pool = [0u32; 4];
    let mut hash_constant = INIT_A;
    for (index, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(entropy.get(index).copied().unwrap_or(0), &mut hash_constant);
    }
    for source in 0..pool.len() {
        for destination in 0..pool.len() {
            if source != destination {
                pool[destination] =
                    mix(pool[destination], hashmix(pool[source], &mut hash_constant));
            }
        }
    }
    let mut state32 = [0u32; 8];
    let mut output_constant = INIT_B;
    for (index, slot) in state32.iter_mut().enumerate() {
        let mut value = pool[index % pool.len()] ^ output_constant;
        output_constant = output_constant.wrapping_mul(MULT_B);
        value = value.wrapping_mul(output_constant);
        *slot = value ^ (value >> 16);
    }
    std::array::from_fn(|index| state32[index * 2] as u64 | ((state32[index * 2 + 1] as u64) << 32))
}
