use sha2::{Digest, Sha256};

/// CPython-compatible integer-seeded MT19937 stream for placeholder embeddings.
/// This preserves `random.Random(seed).gauss(0, 1)` and `shuffle` semantics.
pub(super) struct PythonRandom {
    state: [u32; 624],
    index: usize,
    cached_gauss: Option<f64>,
}

impl PythonRandom {
    pub(super) fn from_seed(seed: u64) -> Self {
        let mut state = [0u32; 624];
        state[0] = 19_650_218;
        for i in 1..624 {
            state[i] = 1_812_433_253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        let key = [seed as u32, (seed >> 32) as u32];
        let mut i = 1usize;
        let mut j = 0usize;
        for _ in 0..624.max(key.len()) {
            state[i] = (state[i] ^ (state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                state[0] = state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            state[i] = (state[i]
                ^ (state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                state[0] = state[623];
                i = 1;
            }
        }
        state[0] = 0x8000_0000;
        Self {
            state,
            index: 624,
            cached_gauss: None,
        }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as u64;
        let b = (self.next_u32() >> 6) as u64;
        ((a << 26) + b) as f64 / 9_007_199_254_740_992.0
    }

    pub(super) fn gauss(&mut self) -> f64 {
        if let Some(value) = self.cached_gauss.take() {
            return value;
        }
        loop {
            let x = 2.0 * self.random() - 1.0;
            let y = 2.0 * self.random() - 1.0;
            let z = x * x + y * y;
            if z > 0.0 && z < 1.0 {
                let scale = (-2.0 * z.ln() / z).sqrt();
                self.cached_gauss = Some(y * scale);
                return x * scale;
            }
        }
    }

    pub(super) fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let bits = (usize::BITS - (i as usize).leading_zeros()) as u32;
            let j = loop {
                let r = if bits <= 32 {
                    (self.next_u32() >> (32 - bits)) as usize
                } else {
                    let low = self.next_u32() as u64;
                    let high_bits = bits - 32;
                    let high = (self.next_u32() >> (32 - high_bits)) as u64;
                    (low | (high << 32)) as usize
                };
                if r <= i {
                    break r;
                }
            };
            values.swap(i, j);
        }
    }
}

pub(super) fn placeholder(dataset_hash: &str) -> (Vec<f64>, Vec<f64>) {
    let digest = Sha256::digest(dataset_hash.as_bytes());
    let seed = digest[..8].iter().fold(0u64, |n, b| (n << 8) | *b as u64);
    let mut rng = PythonRandom::from_seed(seed);
    (
        (0..1536).map(|_| rng.gauss()).collect(),
        (0..512).map(|_| rng.gauss()).collect(),
    )
}
