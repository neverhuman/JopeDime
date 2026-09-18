//! `numpy.random.SeedSequence` (numpy 1.26, `bit_generator.pyx`).
//!
//! The entropy is converted to little-endian 32-bit words, hashed into a pool
//! of four words, and the pool is expanded by `generate_state`.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

/// numpy's default pool size.
pub const POOL_SIZE: usize = 4;

/// A numpy-compatible `SeedSequence` with the default pool size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeedSequence {
    pool: [u32; POOL_SIZE],
}

/// numpy `_int_to_uint32_array`: little-endian 32-bit words, `[0]` for zero.
pub fn int_to_u32_words(mut value: u128) -> Vec<u32> {
    if value == 0 {
        return vec![0];
    }
    let mut words = Vec::new();
    while value > 0 {
        words.push((value & 0xffff_ffff) as u32);
        value >>= 32;
    }
    words
}

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut value = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let result = x
        .wrapping_mul(MIX_MULT_L)
        .wrapping_sub(y.wrapping_mul(MIX_MULT_R));
    result ^ (result >> XSHIFT)
}

impl SeedSequence {
    /// `SeedSequence(entropy)` for a non-negative integer entropy.
    pub fn new(entropy: u128) -> Self {
        Self::with_spawn_key(entropy, &[])
    }

    /// `SeedSequence(entropy, spawn_key=spawn_key)`.
    pub fn with_spawn_key(entropy: u128, spawn_key: &[u128]) -> Self {
        let mut run = int_to_u32_words(entropy);
        let spawn: Vec<u32> = spawn_key
            .iter()
            .flat_map(|&k| int_to_u32_words(k))
            .collect();
        if !spawn.is_empty() && run.len() < POOL_SIZE {
            run.resize(POOL_SIZE, 0);
        }
        run.extend_from_slice(&spawn);
        Self::from_assembled_entropy(&run)
    }

    /// Build the pool from already-assembled 32-bit entropy words
    /// (`mix_entropy`).
    pub fn from_assembled_entropy(entropy: &[u32]) -> Self {
        let mut hash_const = INIT_A;
        let mut pool = [0u32; POOL_SIZE];
        for (i, slot) in pool.iter_mut().enumerate() {
            let word = entropy.get(i).copied().unwrap_or(0);
            *slot = hashmix(word, &mut hash_const);
        }
        for i_src in 0..POOL_SIZE {
            for i_dst in 0..POOL_SIZE {
                if i_src != i_dst {
                    let hashed = hashmix(pool[i_src], &mut hash_const);
                    pool[i_dst] = mix(pool[i_dst], hashed);
                }
            }
        }
        for &word in entropy.iter().skip(POOL_SIZE) {
            for slot in pool.iter_mut() {
                let hashed = hashmix(word, &mut hash_const);
                *slot = mix(*slot, hashed);
            }
        }
        Self { pool }
    }

    /// The mixed entropy pool.
    pub fn pool(&self) -> [u32; POOL_SIZE] {
        self.pool
    }

    /// `generate_state(n_words, dtype=np.uint32)`.
    pub fn generate_state_u32(&self, n_words: usize) -> Vec<u32> {
        let mut hash_const = INIT_B;
        (0..n_words)
            .map(|i| {
                let mut value = self.pool[i % POOL_SIZE] ^ hash_const;
                hash_const = hash_const.wrapping_mul(MULT_B);
                value = value.wrapping_mul(hash_const);
                value ^ (value >> XSHIFT)
            })
            .collect()
    }

    /// `generate_state(n_words, dtype=np.uint64)`: twice as many 32-bit words
    /// viewed as little-endian 64-bit words.
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        self.generate_state_u32(n_words * 2)
            .chunks_exact(2)
            .map(|pair| u64::from(pair[0]) | (u64::from(pair[1]) << 32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_entropy_words_are_little_endian() {
        assert_eq!(int_to_u32_words(0), vec![0]);
        assert_eq!(int_to_u32_words(20_260_707), vec![20_260_707]);
        assert_eq!(
            int_to_u32_words(0x0123_4567_89ab_cdef),
            vec![0x89ab_cdef, 0x0123_4567]
        );
    }

    #[test]
    fn empty_spawn_key_matches_plain_entropy() {
        assert_eq!(
            SeedSequence::new(20_260_707),
            SeedSequence::with_spawn_key(20_260_707, &[])
        );
        assert_ne!(
            SeedSequence::new(20_260_707),
            SeedSequence::with_spawn_key(20_260_707, &[0])
        );
    }

    #[test]
    fn u64_state_is_a_little_endian_view_of_u32_state() {
        let seq = SeedSequence::new(17);
        let words = seq.generate_state_u32(8);
        let wide = seq.generate_state_u64(4);
        for (i, value) in wide.iter().enumerate() {
            assert_eq!(*value as u32, words[2 * i]);
            assert_eq!((*value >> 32) as u32, words[2 * i + 1]);
        }
    }
}
