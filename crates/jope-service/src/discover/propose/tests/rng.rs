use super::support::*;
use super::*;

#[test]
fn delete_uses_one_unbiased_draw_from_the_shared_proposer_stream() {
    let source =
        parse_canonical_pipeline("(pipe X (robust_scale) (standardize) (sigmoid) (ridge))")
            .unwrap();
    let mut expected_rng = round_rng(101_635_475, 777);
    let expected_index = index(&mut expected_rng, source.stages.len());
    let expected_removed = source.stages[expected_index].op.clone();
    let next_shared_value = expected_rng.next_u64();

    let mut actual_rng = round_rng(101_635_475, 777);
    let mut actual_stages = source.stages;
    let actual_index = delete_stage(&mut actual_stages, &mut actual_rng).unwrap();

    assert_eq!(actual_index, expected_index);
    assert_eq!(actual_rng.next_u64(), next_shared_value);
    assert_eq!(actual_stages.len(), 3);
    assert!(!actual_stages
        .iter()
        .any(|stage| stage.op == expected_removed));
}

struct Bank {
    role: BankRole,
    records: Vec<Retrieved>,
    called: RefCell<Vec<([f64; 12], String, usize)>>,
}

impl Bank {
    fn new() -> Self {
        Self {
            role: BankRole::Proposer,
            records: vec![
                Retrieved {
                    program: Program::from_dsl("(pipe X (quantile_bins bins=16) (ridge))"),
                    family_id: "other-a".into(),
                    dl_train: true,
                    distance_squared: 1.0,
                },
                Retrieved {
                    program: Program::from_dsl("(pipe X (winsor_clip q=0.1) (ridge))"),
                    family_id: "other-b".into(),
                    dl_train: true,
                    distance_squared: 2.0,
                },
            ],
            called: RefCell::new(vec![]),
        }
    }
}

impl ChampionBank for Bank {
    fn contract(&self) -> Option<BankContract> {
        Some(BankContract {
            role: self.role,
            coordinate_space: RAW_COORDINATES.into(),
            snapshot_sha256: "1".repeat(64),
        })
    }
    fn nearest(&self, raw: &[f64; 12], family: &str, limit: usize) -> Result<Vec<Retrieved>> {
        self.called.borrow_mut().push((*raw, family.into(), limit));
        Ok(self.records.iter().take(limit).cloned().collect())
    }
}

#[test]
fn retry_head_order_preserves_uniform_weights_and_the_shared_rng() {
    let mut counts = [0usize; 3];
    for seed in 0..1024 {
        let mut rng = round_rng(seed, 0);
        let mut expected = rng.clone();
        let order = head_retry_order(0, 3, 1, &rng);
        assert_eq!(order[0], 0);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
        assert_eq!(order, head_retry_order(0, 3, 1, &rng));
        counts[*order.iter().find(|&&i| i != 0).unwrap()] += 1;
        assert_eq!(rng.next_u64(), expected.next_u64());
    }
    // Conditional on the first head being known, B/C retain equal weights.
    // A cyclic scan always chooses B and fails this data-free control.
    assert!(
        counts[1].abs_diff(counts[2]) < 256,
        "biased retry weights: {counts:?}"
    );
    let rng = round_rng(42, 0);
    assert_ne!(
        head_retry_order(0, 13, 1, &rng),
        head_retry_order(0, 13, 6, &rng)
    );
}

#[test]
fn retry_stream_starts_at_zero_independently_of_the_main_word_position() {
    let initial = round_rng(42, 0);
    let mut advanced = initial.clone();
    advanced.set_word_pos(197);
    for step in [1, 6, 11] {
        assert_eq!(
            head_retry_order(0, 13, step, &initial),
            head_retry_order(0, 13, step, &advanced)
        );
    }
    assert_eq!(initial.get_word_pos(), 0);
    assert_eq!(advanced.get_word_pos(), 197);
}
