//! Deterministic fold assignment shared by every experiment and the runtime CV:
//! FNV-1a over (dataset key, row index) so fold membership never depends on which
//! variant is asking.

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

pub fn fold_of(dataset_key: &str, row: usize, k: usize) -> usize {
    let mut h = FNV_OFFSET;
    for b in dataset_key.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    for b in (row as u64).to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    (h % k.max(1) as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::fold_of;

    #[test]
    fn folds_are_stable_and_balanced() {
        let k = 5;
        let mut counts = vec![0usize; k];
        for row in 0..1000 {
            counts[fold_of("0014b9dad5725a1e", row, k)] += 1;
        }
        for c in &counts {
            assert!(*c > 120 && *c < 280, "unbalanced folds: {counts:?}");
        }
        assert_eq!(fold_of("abc", 17, 5), fold_of("abc", 17, 5));
        assert_ne!(
            (0..50).map(|r| fold_of("a", r, 5)).collect::<Vec<_>>(),
            (0..50).map(|r| fold_of("b", r, 5)).collect::<Vec<_>>()
        );
    }
}
