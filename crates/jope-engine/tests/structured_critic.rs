#![cfg(feature = "neural")]

use jope_parallel::neural::critic::{StructuralCritic, INPUT_DIM};
use tch::{nn, Device, Kind, Tensor};

#[test]
fn scores_each_candidate_without_an_uncertainty_channel() {
    let variables = nn::VarStore::new(Device::Cpu);
    let critic = StructuralCritic::new(&variables.root());
    let input = Tensor::zeros([3, INPUT_DIM], (Kind::Float, Device::Cpu));
    let scores = critic.forward(&input).unwrap();
    assert_eq!(scores.size(), [3]);
    assert_eq!(scores.kind(), Kind::Float);
    assert!(scores.isfinite().all().int64_value(&[]) != 0);
}

#[test]
fn rejects_wrong_feature_width_and_non_finite_values() {
    let variables = nn::VarStore::new(Device::Cpu);
    let critic = StructuralCritic::new(&variables.root());
    let wrong_width = Tensor::zeros([1, INPUT_DIM - 1], (Kind::Float, Device::Cpu));
    assert!(critic.forward(&wrong_width).is_err());
    let non_finite = Tensor::full([1, INPUT_DIM], f64::NAN, (Kind::Float, Device::Cpu));
    assert!(critic.forward(&non_finite).is_err());
}
