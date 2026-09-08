#![cfg(feature = "neural")]
use anyhow::{ensure, Result};
use jope_parallel::neural::{ranking, ranking_pairwise};
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};

// Frozen v44 reference. Keep this complete implementation independent of the
// optimized helper so shared refactors cannot make equivalence tests tautological.
fn frozen_v44_ranking(
    reward: &Tensor,
    values: &[f64],
    percentiles: &[f32],
    groups: &[String],
) -> Result<(Tensor, Tensor)> {
    ensure!(
        values.len() == groups.len() && percentiles.len() == groups.len(),
        "ranking labels mismatch"
    );
    let mut grouped: BTreeMap<&str, Vec<i64>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        grouped.entry(g).or_default().push(i as i64);
    }
    let mut winners = Vec::new();
    let mut losers = Vec::new();
    let mut weights = Vec::new();
    let mut listwise = Vec::new();
    for ids in grouped.values() {
        for (pos, &a) in ids.iter().enumerate() {
            for &b in &ids[pos + 1..] {
                let va = values[a as usize];
                let vb = values[b as usize];
                if !va.is_finite() || !vb.is_finite() || va == vb {
                    continue;
                }
                let (w, l) = if va > vb { (a, b) } else { (b, a) };
                winners.push(w);
                losers.push(l);
                weights.push(
                    (va - vb).abs()
                        * if percentiles[a as usize].max(percentiles[b as usize]) >= 0.9 {
                            2.0
                        } else {
                            1.0
                        },
                );
            }
        }
        if ids.len() >= 3 && ids.iter().all(|&i| percentiles[i as usize].is_finite()) {
            let ps = Tensor::from_slice(
                &ids.iter()
                    .map(|&i| percentiles[i as usize])
                    .collect::<Vec<_>>(),
            )
            .to_device(reward.device());
            let target = (&ps / 0.25).softmax(0, Kind::Float);
            let pred = reward
                .index_select(0, &Tensor::from_slice(ids).to_device(reward.device()))
                .to_kind(Kind::Float)
                .log_softmax(0, Kind::Float);
            listwise.push(-(target * pred).sum(Kind::Float));
        }
    }
    let zero = reward.sum(Kind::Float) * 0.0;
    let bt = if winners.is_empty() {
        zero.shallow_clone()
    } else {
        let scale = weights.iter().sum::<f64>() / weights.len() as f64;
        // Repeated index_select entries backpropagate through CUDA atomic adds.
        // Tiny, unordered cancellation errors then become visible AdamW updates
        // to the otherwise translation-invariant reward biases. Broadcast each
        // reward once and reduce a dense comparison matrix in a fixed layout.
        let n = groups.len();
        let mut dense = vec![0f32; n * n];
        for ((&w, &l), &weight) in winners.iter().zip(&losers).zip(&weights) {
            dense[w as usize * n + l as usize] = (weight / scale) as f32;
        }
        let weights = Tensor::from_slice(&dense)
            .view([n as i64, n as i64])
            .to_device(reward.device());
        let scores = reward.to_kind(Kind::Float);
        (((scores.unsqueeze(0) - scores.unsqueeze(1)).softplus()) * weights).sum(Kind::Float)
            / winners.len() as f64
    };
    Ok((
        bt,
        if listwise.is_empty() {
            zero
        } else {
            Tensor::stack(&listwise, 0).mean(Kind::Float)
        },
    ))
}

fn assert_same_bits(a: &Tensor, b: &Tensor) {
    let a: Vec<f32> = Vec::try_from(a.flatten(0, -1)).unwrap();
    let b: Vec<f32> = Vec::try_from(b.flatten(0, -1)).unwrap();
    assert_eq!(
        a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}
fn compare(values: &[f64], percentiles: &[f32], groups: &[String]) {
    let input =
        (Tensor::arange(values.len() as i64, (Kind::Float, Device::Cpu)).fmod(19.) - 9.) / 3.;
    let old = input.copy().set_requires_grad(true);
    let new = input.copy().set_requires_grad(true);
    let reference = frozen_v44_ranking(&old, values, percentiles, groups)
        .unwrap()
        .0;
    let optimized = ranking_pairwise(&new, values, percentiles, groups).unwrap();
    assert_same_bits(&reference, &optimized);
    reference.backward();
    optimized.backward();
    assert_same_bits(&old.grad(), &new.grad());
    // The public full-ranking path still supplies the unchanged listwise loss
    // for the actual reward objective.
    let old = input.copy().set_requires_grad(true);
    let new = input.copy().set_requires_grad(true);
    let (a, b) = frozen_v44_ranking(&old, values, percentiles, groups).unwrap();
    let (c, d) = ranking(&new, values, percentiles, groups).unwrap();
    assert_same_bits(&a, &c);
    assert_same_bits(&b, &d);
    (a + b * 0.1).backward();
    (c + d * 0.1).backward();
    assert_same_bits(&old.grad(), &new.grad());
}
#[test]
fn pairwise_ablation_preserves_values_and_gradients_for_full_microbatch() {
    tch::set_num_threads(1);
    let values = (0..256)
        .map(|i| ((i * 31) % 83) as f64 / 17.0)
        .collect::<Vec<_>>();
    let percentiles = (0..256).map(|i| (i % 8) as f32 / 7.0).collect::<Vec<_>>();
    let groups = (0..256)
        .map(|i| format!("context-{}:comparison-{}", (i / 8) % 7, i / 8))
        .collect::<Vec<_>>();
    compare(&values, &percentiles, &groups);
}
#[test]
fn pairwise_ablation_preserves_ties_missing_values_and_irregular_groups() {
    tch::set_num_threads(1);
    let groups = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "d"].map(str::to_owned);
    compare(
        &[1., 1., 2., 0., 3., f64::NAN, 2., 4., 2., 5.],
        &[0., 0.1, 1., 0., 1., 0., f32::NAN, 0.7, 1., 0.5],
        &groups,
    );
    compare(&[1.; 10], &[0.5; 10], &groups);
}
#[test]
fn pairwise_ablation_preserves_label_shape_validation() {
    let reward = Tensor::zeros([4], (Kind::Float, Device::Cpu));
    assert!(ranking_pairwise(&reward, &[1., 2.], &[0., 1.], &["a".into()]).is_err());
}
