#![cfg(feature = "neural")]

use jope_parallel::{
    neural::{AdamW, Config, Policy},
    training::{
        objective::{self, Outputs, RowLabels},
        replay_bridge::{
            Disposition, FreshOccurrence, HeadEligibility, ReplayBundle, ReplayIdentity,
        },
    },
};
use std::collections::BTreeMap;
use tch::{nn, Device, Kind, Tensor};

fn identity() -> ReplayIdentity {
    let digest = "a".repeat(64);
    ReplayIdentity {
        schema: "jope.training-intent.v1".into(),
        bundle_id: "fixture-bundle-1".into(),
        intent_sha256: digest.clone(),
        lineage_sha256: digest.clone(),
        corpus_sha256: digest.clone(),
        parent_state_sha256: digest.clone(),
        objective_sha256: digest.clone(),
        recipe_sha256: digest.clone(),
        partition_sha256: digest,
        window_id: "window-1".into(),
        rng_identity: "fixture-seed-41".into(),
        sampler_cursor: 8,
    }
}

fn group(
    comparison: &str,
    record_offset: usize,
    masked_term: Option<&str>,
) -> Vec<FreshOccurrence> {
    (0..8)
        .map(|i| FreshOccurrence {
            occurrence_id: format!("{comparison}-occurrence-{i}"),
            evidence_sha256: format!("{:064x}", i + 1),
            comparison_id: comparison.into(),
            record_id: record_offset + i,
            context_id: 7,
            disposition: if i == 0 {
                Disposition::Chosen
            } else {
                Disposition::Rejected
            },
            value: [0.95, 0.8, 0.7, 0.62, 0.5, 0.41, 0.3, 0.2][i],
            percentile: [1.0, 0.86, 0.71, 0.57, 0.43, 0.29, 0.14, 0.0][i],
            runtime: [0.4, 0.1, 2.0, 0.0, 5.0, 0.3, 1.2, 0.7][i],
            token_length: 5,
            eligibility: HeadEligibility {
                sequence_nll: !(masked_term == Some("sequence_nll") && i == 7),
                ranking_bt: !(masked_term == Some("ranking_bt") && i == 7),
                ranking_listwise: !(masked_term == Some("ranking_listwise") && i == 7),
                runtime_huber: !(masked_term == Some("runtime_huber") && i == 7),
                contextfree_bt: !(masked_term == Some("contextfree_bt") && i == 7),
            },
        })
        .collect()
}

#[test]
fn frozen_bundle_selects_complete_occurrences_and_rejects_label_edits() {
    let expected = identity();
    let bundle = ReplayBundle::seal(expected.clone(), group("comparison-a", 0, None)).unwrap();
    bundle.verify(&expected).unwrap();
    let selected = bundle.select(&expected).unwrap();
    assert_eq!(selected.occurrences.len(), 8);
    assert_eq!(selected.denominators.comparisons, 1);
    assert_eq!(selected.denominators.chosen_occurrences, 1);
    assert_eq!(selected.denominators.rejected_occurrences, 7);
    assert_eq!(selected.denominators.ranking_pairs, 28);
    assert!(selected
        .occurrences
        .iter()
        .all(|x| x.evidence_sha256.len() == 64));
    assert!(objective::supervision(&selected.labels, &selected.token_lengths).is_ok());

    let mut repeated = group("comparison-a", 0, None);
    repeated.extend(group("comparison-b", 0, None));
    let repeated = ReplayBundle::seal(identity(), repeated).unwrap();
    let repeated = repeated.select(&identity()).unwrap();
    assert_eq!(repeated.denominators.comparisons, 2);
    assert!(repeated
        .denominators
        .occurrence_multiplicity_by_record
        .values()
        .all(|count| *count == 2));

    let mut altered = bundle.clone();
    altered.occurrences[0].value -= 0.01;
    assert!(altered.verify(&expected).is_err());
    let mut wrong_intent = expected;
    wrong_intent.intent_sha256 = "b".repeat(64);
    assert!(bundle.select(&wrong_intent).is_err());
}

#[test]
fn masked_active_head_excludes_the_whole_comparison_without_filling_labels() {
    let expected = identity();
    for term in [
        "sequence_nll",
        "ranking_bt",
        "ranking_listwise",
        "runtime_huber",
        "contextfree_bt",
    ] {
        let mut occurrences = group("eligible", 0, None);
        occurrences.extend(group("masked", 8, Some(term)));
        let bundle = ReplayBundle::seal(expected.clone(), occurrences).unwrap();
        let selected = bundle.select(&expected).unwrap();
        assert_eq!(selected.occurrences.len(), 8, "masked {term}");
        assert_eq!(selected.excluded.len(), 1, "masked {term}");
        assert_eq!(selected.excluded[0].comparison_id, "masked");
        assert_eq!(
            selected.excluded[0].reason,
            format!("ineligible active term: {term}")
        );
        assert!(selected
            .occurrences
            .iter()
            .all(|x| x.comparison_id == "eligible"));
        assert_eq!(selected.denominators.runtime_rows, 8);
    }
}

fn fixture_batch() -> (Tensor, Tensor, Tensor, Tensor, Vec<RowLabels>) {
    let n = 8i64;
    let tokens = Tensor::from_slice(&[
        1i64, 3, 4, 2, 5, 1, 4, 5, 2, 6, 1, 5, 6, 2, 7, 1, 6, 7, 2, 3, 1, 7, 3, 2, 4, 1, 3, 4, 2,
        5, 1, 4, 5, 2, 6, 1, 5, 6, 2, 7,
    ])
    .reshape([n, 5]);
    let dataset = (Tensor::arange(n * 12, (Kind::Float, Device::Cpu)) * 0.037)
        .sin()
        .reshape([n, 12]);
    let target = (Tensor::arange(n * 4, (Kind::Float, Device::Cpu)) * 0.11)
        .cos()
        .reshape([n, 4]);
    let legal = Tensor::ones([n, 4, 8], (Kind::Bool, Device::Cpu));
    let values = [0.95, 0.8, 0.7, 0.62, 0.5, 0.41, 0.3, 0.2];
    let percentile = [1.0, 0.86, 0.71, 0.57, 0.43, 0.29, 0.14, 0.0];
    let labels = (0..8)
        .map(|i| RowLabels {
            record: i,
            context: 3,
            value: values[i],
            percentile: percentile[i],
            runtime: 0.5,
        })
        .collect();
    (dataset, target, tokens, legal, labels)
}

fn tiny_config() -> Config {
    let mut c = Config::regression(8);
    c.dataset_dim = 12;
    c.target_dim = 4;
    c.target_keep_dim = 4;
    c.width = 16;
    c.heads = 2;
    c.ffn = 32;
    c.layers = 2;
    c.dropout = 0.0;
    c
}

fn forward(policy: &Policy, batch: &(Tensor, Tensor, Tensor, Tensor, Vec<RowLabels>)) -> Tensor {
    let (dataset, target, tokens, legal, labels) = batch;
    let pad = tokens.ne(0);
    let (logits, rewards, runtime, stability, ablation) = policy
        .forward(dataset, target, tokens, &pad, false)
        .unwrap();
    objective::composite(
        &Outputs {
            logits: &logits,
            rewards: &rewards,
            runtime: &runtime,
            stability: &stability,
            ablation: &ablation,
        },
        policy.loss_log_vars.values(),
        tokens,
        &pad,
        legal,
        labels,
    )
    .unwrap()
    .0
}

fn gradient_sums(vs: &nn::VarStore) -> BTreeMap<String, f64> {
    vs.variables()
        .into_iter()
        .filter_map(|(name, parameter)| {
            let gradient = parameter.grad();
            gradient
                .defined()
                .then(|| (name, gradient.abs().sum(Kind::Float).double_value(&[])))
        })
        .collect()
}

#[test]
fn fresh_label_changes_native_gradient_and_zero_decay_update_survives_restart() {
    tch::set_num_threads(1);
    tch::manual_seed(20261018);
    let expected = identity();
    let mut measured = group("native-consumption", 0, None);
    measured[0].value = 0.97;
    let bundle = ReplayBundle::seal(expected.clone(), measured).unwrap();
    let selection = bundle.select(&expected).unwrap();
    assert_eq!(selection.occurrences.len(), 8);
    assert!(selection
        .occurrences
        .iter()
        .all(|occurrence| occurrence.occurrence_id.starts_with("native-consumption-")));
    let no_fresh_batch = fixture_batch();
    let mut fresh_batch = fixture_batch();
    fresh_batch.4 = selection.labels.clone();
    let config = tiny_config();
    let vs = nn::VarStore::new(Device::Cpu);
    let policy = Policy::new(vs.root(), config.clone()).unwrap();
    let initial_trace = {
        let pad = fresh_batch.2.ne(0);
        policy
            .forward(&fresh_batch.0, &fresh_batch.1, &fresh_batch.2, &pad, false)
            .unwrap()
            .1
    };
    tch::manual_seed(20261018);
    let control_vs = nn::VarStore::new(Device::Cpu);
    let control = Policy::new(control_vs.root(), config.clone()).unwrap();
    let control_initial_trace = {
        let pad = no_fresh_batch.2.ne(0);
        control
            .forward(
                &no_fresh_batch.0,
                &no_fresh_batch.1,
                &no_fresh_batch.2,
                &pad,
                false,
            )
            .unwrap()
            .1
    };
    assert_eq!(
        (&control_initial_trace - &initial_trace)
            .abs()
            .max()
            .double_value(&[]),
        0.0,
        "frozen and no-fresh controls must share the same seeded parent"
    );
    let baseline_loss = forward(&control, &no_fresh_batch);
    let baseline = baseline_loss.double_value(&[]);
    baseline_loss.backward();
    let baseline_gradients = gradient_sums(&control_vs);
    let mut control_optimizer = AdamW::new(1e-3, 0.0);
    control_optimizer.update_reference(&control_vs).unwrap();
    // Fresh measurement changes only the selected chosen row's label.
    let perturbed = fresh_batch;
    let loss = forward(&policy, &perturbed);
    assert!((loss.double_value(&[]) - baseline).abs() > 1e-7);
    loss.backward();
    let gradients = gradient_sums(&vs);
    let reward_gradient_delta = gradients
        .iter()
        .filter(|(name, _)| name.starts_with("reward_heads."))
        .map(|(name, value)| (value - baseline_gradients.get(name).copied().unwrap_or(0.0)).abs())
        .sum::<f64>();
    assert!(
        gradients.values().sum::<f64>().is_finite()
            && gradients.values().sum::<f64>() > 0.0
            && reward_gradient_delta > 0.0,
        "fresh objective had no active gradient"
    );
    let mut optimizer = AdamW::new(1e-3, 0.0);
    optimizer.update_reference(&vs).unwrap();
    assert_eq!(optimizer.step, 1);
    assert_eq!(control_optimizer.step, 1);
    let path = tempfile::tempdir().unwrap();
    let model = path.path().join("model.ot");
    vs.save(&model).unwrap();
    let mut restored_vs = nn::VarStore::new(Device::Cpu);
    let restored = Policy::new(restored_vs.root(), config).unwrap();
    restored_vs.load(&model).unwrap();
    let after_trace = {
        let pad = perturbed.2.ne(0);
        restored
            .forward(&perturbed.0, &perturbed.1, &perturbed.2, &pad, false)
            .unwrap()
            .1
    };
    assert!(
        (&after_trace - &initial_trace)
            .abs()
            .max()
            .double_value(&[])
            > 0.0
    );
    let live_trace = {
        let pad = perturbed.2.ne(0);
        policy
            .forward(&perturbed.0, &perturbed.1, &perturbed.2, &pad, false)
            .unwrap()
            .1
    };
    assert_eq!(
        (&after_trace - &live_trace).abs().max().double_value(&[]),
        0.0
    );
    let no_fresh_trace = {
        let pad = perturbed.2.ne(0);
        control
            .forward(&perturbed.0, &perturbed.1, &perturbed.2, &pad, false)
            .unwrap()
            .1
    };
    assert!(
        (&after_trace - &no_fresh_trace)
            .abs()
            .max()
            .double_value(&[])
            > 0.0,
        "fresh and no-fresh updates must remain distinguishable"
    );
}
