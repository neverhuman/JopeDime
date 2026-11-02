use super::*;

use crate::hyperion_v7::sweep::PatchOutputs;

struct FakeModel;

impl PatchModel for FakeModel {
    fn forward_batch(&self, x: &[f32], _m: &[f32], batch: usize, p: usize) -> Result<PatchOutputs> {
        Ok(PatchOutputs {
            emb: vec![0.0; batch],
            logits: x
                .iter()
                .map(|value| value.mul_add(0.25, p as f32))
                .collect(),
        })
    }

    fn emb_dim(&self) -> usize {
        1
    }
}

fn fake_settings() -> HyperionSweepSettings {
    HyperionSweepSettings {
        version: "v7".into(),
        step: 0,
        dim: 128,
        layers: 1,
        heads: 1,
        emb_dim: 1,
        n_bands: 1,
        min_patch: 2,
        max_patch: 12,
        ws_list: vec![3, 6, 9, 12],
        q_low: 1.0,
        q_high: 99.0,
        pad_u8: 0,
        quant_scale: 1.0,
        seed: 42,
    }
}

fn fake_identity() -> HyperionModelIdentity {
    HyperionModelIdentity {
        weights_path: "weights".into(),
        manifest_path: "manifest".into(),
        weights_bytes: 1,
        weights_hash: ORIG_V2_WEIGHTS_SHA256.into(),
        manifest_hash: ORIG_V2_MANIFEST_SHA256.into(),
        tensor_count: 1,
    }
}

fn fake_encode(train: &Matrix, hold: &Matrix) -> OrigV2PairOutput {
    encode_pair_with_model(
        &FakeModel,
        &fake_settings(),
        &fake_identity(),
        ORIG_V2_WEIGHTS_SHA256,
        ORIG_V2_MANIFEST_SHA256,
        "test",
        train,
        hold,
        OrigV2PairSemantics::TrainFitted,
        &OrigV2Config::default(),
    )
    .unwrap()
}

#[test]
fn rank_fit_has_old_unique_rank_then_robust_u8_contract() {
    let x = Matrix::new(vec![9.0, 1.0, 1.0, 5.0], 4, 1);
    let (grid, sorted) = rank_fit_grid(&x, 1.0, 99.0);
    assert_eq!(sorted, vec![vec![1.0, 1.0, 5.0, 9.0]]);
    assert_eq!(grid.h, 4);
    assert_eq!(grid.w, 1);
    // Index breaks the tied 1.0 values deterministically, just as argsort(argsort) produced
    // unique ordinals rather than tie-averaged ranks.
    assert!(grid.data[1] < grid.data[2]);
    assert_eq!(grid.data[0], 255);
}

#[test]
fn train_fitted_rank_replay_uses_only_fit_distribution() {
    let train = Matrix::new(vec![0.0, 10.0, 20.0, 30.0], 4, 1);
    let hold = Matrix::new(vec![-5.0, 15.0, 35.0], 3, 1);
    let (_, sorted) = rank_fit_grid(&train, 1.0, 99.0);
    let replay = rank_replay_grid(&hold, &sorted, 1.0, 99.0).unwrap();
    assert_eq!(replay.data[0], 0);
    assert!(replay.data[1] > 0 && replay.data[1] < 255);
    assert_eq!(replay.data[2], 255);
}

#[test]
fn joint_rows_preserve_train_then_hold_order() {
    let train = Matrix::new(vec![1.0, 2.0, 10.0, 20.0], 2, 2);
    let hold = Matrix::new(vec![3.0, 30.0], 1, 2);
    let joint = join_rows(&train, &hold).unwrap();
    assert_eq!((joint.n, joint.c), (3, 2));
    assert_eq!(joint.data.as_ref(), &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
}

#[test]
fn defaults_pin_the_correct_all100_profile() {
    let config = OrigV2Config::default();
    assert_eq!(config.sweeps, 100);
    assert_eq!(config.windows, vec![3, 6, 9, 12]);
    assert_eq!(config.shifts, vec![(0, 0), (1, 1)]);
    assert_eq!(config.base_seed, 42);
    assert_eq!(config.seed_stride, 777);
    assert_eq!(config.batch, 4096);
    assert!(config.require_pinned_model);
}

#[test]
fn reusable_encoder_path_is_deterministic() {
    let train = Matrix::new(vec![1.0, 3.0, 2.0, 4.0], 2, 2);
    let hold = Matrix::new(vec![5.0, 6.0], 1, 2);
    let one_shot = fake_encode(&train, &hold);
    let reused = fake_encode(&train, &hold);
    assert_eq!(one_shot.train_cells.data, reused.train_cells.data);
    assert_eq!(one_shot.hold_cells.data, reused.hold_cells.data);
    assert_eq!(one_shot.receipt, reused.receipt);
}

#[test]
fn train_fitted_train_cache_is_invariant_to_test_values() {
    let train = Matrix::new(vec![1.0, 3.0, 2.0, 4.0], 2, 2);
    let first_hold = Matrix::new(vec![5.0, 6.0], 1, 2);
    let second_hold = Matrix::new(vec![-100.0, 100.0], 1, 2);
    let first = fake_encode(&train, &first_hold);
    let second = fake_encode(&train, &second_hold);
    assert_eq!(first.train_cells.data, second.train_cells.data);
}

#[test]
fn frozen_receipt_validation_rejects_profile_drift() {
    let train = Matrix::new(vec![1.0, 2.0], 2, 1);
    let hold = Matrix::new(vec![3.0], 1, 1);
    let mut receipt = fake_encode(&train, &hold).receipt;
    receipt
        .validate_frozen(OrigV2PairSemantics::TrainFitted)
        .unwrap();

    receipt.sweeps -= 1;
    assert!(receipt
        .validate_frozen(OrigV2PairSemantics::TrainFitted)
        .is_err());
}
