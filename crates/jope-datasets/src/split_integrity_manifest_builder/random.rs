//! CPython-compatible `random.Random(int).shuffle` for the family-order selection.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub(super) struct PythonRandom {
    state: [u32; N],
    index: usize,
}

impl PythonRandom {
    pub(super) fn from_seed(mut seed: u64) -> Self {
        let mut key = vec![seed as u32];
        seed >>= 32;
        if seed != 0 {
            key.push(seed as u32);
        }
        let mut rng = Self {
            state: [0; N],
            index: N,
        };
        rng.init_by_array(&key);
        rng
    }

    pub(super) fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let other = self.randbelow(index + 1);
            values.swap(index, other);
        }
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for index in 1..N {
            self.state[index] = 1_812_433_253_u32
                .wrapping_mul(self.state[index - 1] ^ (self.state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let mut index = 1;
        let mut key_index = 0;
        for _ in 0..N.max(key.len()) {
            self.state[index] = (self.state[index]
                ^ (self.state[index - 1] ^ (self.state[index - 1] >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[key_index])
            .wrapping_add(key_index as u32);
            index += 1;
            key_index += 1;
            if index >= N {
                self.state[0] = self.state[N - 1];
                index = 1;
            }
            if key_index >= key.len() {
                key_index = 0;
            }
        }
        for _ in 0..N - 1 {
            self.state[index] = (self.state[index]
                ^ (self.state[index - 1] ^ (self.state[index - 1] >> 30))
                    .wrapping_mul(1_566_083_941))
            .wrapping_sub(index as u32);
            index += 1;
            if index >= N {
                self.state[0] = self.state[N - 1];
                index = 1;
            }
        }
        self.state[0] = 0x8000_0000;
        self.index = N;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            for index in 0..N {
                let joined =
                    (self.state[index] & UPPER_MASK) | (self.state[(index + 1) % N] & LOWER_MASK);
                self.state[index] = self.state[(index + M) % N]
                    ^ (joined >> 1)
                    ^ if joined & 1 == 0 { 0 } else { MATRIX_A };
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    fn getrandbits(&mut self, bits: u32) -> u64 {
        let mut remaining = bits;
        let mut shift = 0;
        let mut value = 0_u64;
        while remaining > 0 {
            let take = remaining.min(32);
            let word = self.next_u32() >> (32 - take);
            value |= u64::from(word) << shift;
            remaining -= take;
            shift += take;
        }
        value
    }

    fn randbelow(&mut self, upper: usize) -> usize {
        let bits = usize::BITS - upper.leading_zeros();
        loop {
            let value = self.getrandbits(bits);
            if value < upper as u64 {
                return value as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PythonRandom;

    #[test]
    fn integer_seed_shuffle_matches_cpython() {
        let mut values: Vec<_> = (0..10).collect();
        PythonRandom::from_seed(1).shuffle(&mut values);
        assert_eq!(values, [6, 8, 9, 7, 5, 3, 0, 4, 1, 2]);
    }
}
