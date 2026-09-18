//! The integer-sampling subset of `numpy.random.Generator` (numpy 1.26,
//! `_generator.pyx`, `_bounded_integers.pyx`, `distributions.c`).

use std::collections::HashSet;
use std::fmt;

use crate::pcg64::Pcg64;
use crate::seed_sequence::SeedSequence;

/// Argument errors numpy raises as `ValueError`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleError {
    /// `a must be a positive integer unless no samples are taken`.
    EmptyPopulation,
    /// `Cannot take a larger sample than population when replace is False`.
    SampleLargerThanPopulation { population: u64, size: u64 },
}

impl fmt::Display for SampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPopulation => {
                f.write_str("a must be a positive integer unless no samples are taken")
            }
            Self::SampleLargerThanPopulation { population, size } => write!(
                f,
                "cannot take a sample of {size} from a population of {population} without replacement"
            ),
        }
    }
}

impl std::error::Error for SampleError {}

/// `numpy.random.Generator(PCG64(...))`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generator {
    bitgen: Pcg64,
}

/// `numpy.random.default_rng(seed)` for a non-negative integer seed.
pub fn default_rng(seed: u128) -> Generator {
    Generator::new(Pcg64::from_seed_sequence(&SeedSequence::new(seed)))
}

/// Smallest all-ones mask covering `max` (`gen_mask`).
fn gen_mask(max: u64) -> u64 {
    if max == 0 {
        0
    } else {
        u64::MAX >> max.leading_zeros()
    }
}

impl Generator {
    pub fn new(bitgen: Pcg64) -> Self {
        Self { bitgen }
    }

    pub fn bit_generator(&self) -> &Pcg64 {
        &self.bitgen
    }

    pub fn next_u32(&mut self) -> u32 {
        self.bitgen.next_u32()
    }

    pub fn next_u64(&mut self) -> u64 {
        self.bitgen.next_u64()
    }

    /// `random_interval(bitgen, max)`: masked rejection sampling in `[0, max]`
    /// (used by `shuffle`/`permutation`).
    pub fn random_interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mask = gen_mask(max);
        if max <= u64::from(u32::MAX) {
            loop {
                let value = u64::from(self.bitgen.next_u32()) & mask;
                if value <= max {
                    return value;
                }
            }
        }
        loop {
            let value = self.bitgen.next_u64() & mask;
            if value <= max {
                return value;
            }
        }
    }

    /// `buffered_bounded_lemire_uint32` (the buffer is unused for 32 bits).
    fn bounded_lemire_u32(&mut self, rng: u32) -> u32 {
        debug_assert!(rng != u32::MAX);
        let rng_excl = rng + 1;
        let mut m = u64::from(self.bitgen.next_u32()) * u64::from(rng_excl);
        let mut leftover = m as u32;
        if leftover < rng_excl {
            let threshold = (u32::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = u64::from(self.bitgen.next_u32()) * u64::from(rng_excl);
                leftover = m as u32;
            }
        }
        (m >> 32) as u32
    }

    /// `bounded_lemire_uint64`.
    fn bounded_lemire_u64(&mut self, rng: u64) -> u64 {
        debug_assert!(rng != u64::MAX);
        let rng_excl = rng + 1;
        let mut m = u128::from(self.bitgen.next_u64()) * u128::from(rng_excl);
        let mut leftover = m as u64;
        if leftover < rng_excl {
            let threshold = (u64::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = u128::from(self.bitgen.next_u64()) * u128::from(rng_excl);
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// `random_bounded_uint64(bitgen, 0, rng, 0, use_masked=false)`: Lemire
    /// sampling in the inclusive range `[0, rng]`.
    pub fn random_bounded_u64(&mut self, rng: u64) -> u64 {
        if rng == 0 {
            0
        } else if rng < u64::from(u32::MAX) {
            u64::from(self.bounded_lemire_u32(rng as u32))
        } else if rng == u64::from(u32::MAX) {
            u64::from(self.bitgen.next_u32())
        } else if rng == u64::MAX {
            self.bitgen.next_u64()
        } else {
            self.bounded_lemire_u64(rng)
        }
    }

    /// `integers(0, high, size=size, dtype=np.int64)` (exclusive `high`,
    /// Lemire, `random_bounded_uint64_fill`).
    pub fn integers(&mut self, high: u64, size: usize) -> Result<Vec<u64>, SampleError> {
        if high == 0 {
            return Err(SampleError::EmptyPopulation);
        }
        Ok((0..size)
            .map(|_| self.random_bounded_u64(high - 1))
            .collect())
    }

    /// `_shuffle_raw(first=1)` as `shuffle` uses it for a 1-d array.
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        // The hosted jope-dime foundation sampling.json at 07304d6f is the
        // required-lane oracle. It regenerates with `random_interval` (masked
        // rejection), not Lemire `random_bounded_u64`. Keep this path aligned
        // with that file; `shuffle_int` stays Lemire as `choice` uses it.
        for i in (1..values.len()).rev() {
            let j = self.random_interval(i as u64) as usize;
            values.swap(i, j);
        }
    }

    /// `_shuffle_int(n, first, data)` as `choice` uses it.
    fn shuffle_int(&mut self, values: &mut [u64], first: usize) {
        for i in (first..values.len()).rev() {
            let j = self.random_bounded_u64(i as u64) as usize;
            values.swap(i, j);
        }
    }

    /// `permutation(n)`: `arange(n)` shuffled in place.
    pub fn permutation(&mut self, n: usize) -> Vec<u64> {
        let mut values: Vec<u64> = (0..n as u64).collect();
        self.shuffle(&mut values);
        values
    }

    /// `choice(population, size=size, replace=replace)` for an integer
    /// population, no `p`, and the default `shuffle=True`.
    pub fn choice(
        &mut self,
        population: u64,
        size: usize,
        replace: bool,
    ) -> Result<Vec<u64>, SampleError> {
        if size == 0 {
            return Ok(Vec::new());
        }
        if population == 0 {
            return Err(SampleError::EmptyPopulation);
        }
        if replace {
            return self.integers(population, size);
        }
        let size_u64 = size as u64;
        if size_u64 > population {
            return Err(SampleError::SampleLargerThanPopulation {
                population,
                size: size_u64,
            });
        }
        const SHUFFLE_CUTOFF: u64 = 50;
        if population > 10_000 && size_u64 > population / SHUFFLE_CUTOFF {
            // Tail shuffle of arange(population).
            let mut values: Vec<u64> = (0..population).collect();
            let first =
                usize::try_from((population - size_u64).max(1)).expect("population fits in memory");
            self.shuffle_int(&mut values, first);
            return Ok(values.split_off(values.len() - size));
        }
        // Floyd's algorithm; the open-addressing hash set in numpy only
        // decides membership, so any set gives identical output.
        let mut seen = HashSet::with_capacity(size);
        let mut out = Vec::with_capacity(size);
        for j in (population - size_u64)..population {
            let value = self.random_bounded_u64(j);
            if seen.insert(value) {
                out.push(value);
            } else {
                seen.insert(j);
                out.push(j);
            }
        }
        self.shuffle_int(&mut out, 1);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_cover_the_maximum() {
        assert_eq!(gen_mask(0), 0);
        assert_eq!(gen_mask(1), 1);
        assert_eq!(gen_mask(5), 7);
        assert_eq!(gen_mask(8), 15);
        assert_eq!(gen_mask(u64::MAX), u64::MAX);
    }

    #[test]
    fn permutation_is_a_permutation() {
        let mut rng = default_rng(20_260_707);
        let mut values = rng.permutation(97);
        values.sort_unstable();
        assert_eq!(values, (0..97).collect::<Vec<_>>());
    }

    #[test]
    fn choice_without_replacement_is_distinct() {
        let mut rng = default_rng(5);
        for population in [32, 33, 100, 20_000] {
            let mut values = rng.choice(population, 32, false).unwrap();
            assert!(values.iter().all(|&v| v < population));
            values.sort_unstable();
            values.dedup();
            assert_eq!(values.len(), 32);
        }
    }

    #[test]
    fn choice_validates_like_numpy() {
        let mut rng = default_rng(5);
        assert_eq!(rng.choice(0, 1, true), Err(SampleError::EmptyPopulation));
        assert_eq!(rng.choice(0, 0, false), Ok(Vec::new()));
        assert!(matches!(
            rng.choice(3, 4, false),
            Err(SampleError::SampleLargerThanPopulation { .. })
        ));
        assert_eq!(rng.choice(1, 5, true), Ok(vec![0; 5]));
    }
}
