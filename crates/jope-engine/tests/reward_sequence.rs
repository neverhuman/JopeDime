#![cfg(feature = "neural")]

use jope_parallel::neural::OrderAwareRewardEncoder;
use tch::{nn, Device, Kind, Tensor};

#[test]
fn attention_distinguishes_order_and_ignores_trailing_padding_and_other_programs(
) -> anyhow::Result<()> {
    tch::set_num_threads(1);
    tch::manual_seed(821);
    let variables = nn::VarStore::new(Device::Cpu);
    let encoder = OrderAwareRewardEncoder::new(variables.root(), 32, 128)?;
    let input = Tensor::from_slice(&[1i64, 2, 3, 4, 1, 3, 2, 4]).view([2, 4]);
    let present = Tensor::ones([2, 4], (Kind::Bool, Device::Cpu));
    let encoded = encoder.forward(&input, &present, false)?;
    assert_eq!(encoded.size(), [2, 256]);
    assert!(
        (&encoded.get(0) - &encoded.get(1))
            .abs()
            .max()
            .double_value(&[])
            > 1e-5,
        "same token multiset must retain operator/parameter order information"
    );
    let first = encoder.forward(&input.narrow(0, 0, 1), &present.narrow(0, 0, 1), false)?;
    assert!(first.allclose(&encoded.narrow(0, 0, 1), 1e-5, 1e-6, false));
    let padded = Tensor::cat(
        &[input, Tensor::zeros([2, 3], (Kind::Int64, Device::Cpu))],
        1,
    );
    let padded_mask = Tensor::cat(
        &[present, Tensor::zeros([2, 3], (Kind::Bool, Device::Cpu))],
        1,
    );
    assert!(encoder
        .forward(&padded, &padded_mask, false)?
        .allclose(&encoded, 1e-5, 1e-6, false));
    encoded.square().mean(Kind::Float).backward();
    for parameter in variables.trainable_variables() {
        assert!(parameter.grad().defined());
        assert_ne!(parameter.grad().isfinite().all().int64_value(&[]), 0);
    }
    Ok(())
}

#[test]
fn experimental_encoder_rejects_missing_programs_and_changed_vocabulary() -> anyhow::Result<()> {
    tch::set_num_threads(1);
    let variables = nn::VarStore::new(Device::Cpu);
    let encoder = OrderAwareRewardEncoder::new(variables.root(), 8, 4)?;
    let mask = Tensor::zeros([1, 2], (Kind::Bool, Device::Cpu));
    let tokens = Tensor::from_slice(&[1i64, 2]).view([1, 2]);
    assert!(encoder.forward(&tokens, &mask, false).is_err());
    let changed = Tensor::from_slice(&[1i64, 8]).view([1, 2]);
    assert!(encoder
        .forward(&changed, &mask.logical_not(), false)
        .is_err());
    assert!(encoder
        .forward(&tokens.to_kind(Kind::Float), &mask, false)
        .is_err());
    Ok(())
}
