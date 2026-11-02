use super::*;

pub(super) struct Random {
    state: u64,
}

impl Random {
    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(super) fn next(&mut self) -> u64 {
        // A fixed integer mixing construction, not a source of external entropy.
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub(super) fn index(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    pub(super) fn shuffle(&mut self, values: &mut [usize]) {
        for i in (1..values.len()).rev() {
            let j = self.index(i + 1);
            values.swap(i, j);
        }
    }
}

pub(super) fn sample_indices(total: usize, maximum: usize, rng: &mut Random) -> Vec<usize> {
    let count = total.min(maximum);
    let mut selected: Vec<usize> = (0..count).collect();
    for i in count..total {
        let j = rng.index(i + 1);
        if j < count {
            selected[j] = i;
        }
    }
    rng.shuffle(&mut selected);
    selected
}
