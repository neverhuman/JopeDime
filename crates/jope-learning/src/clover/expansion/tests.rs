mod closed_loop;
mod equal_attempt;
mod seed_validation;

use super::*;

fn pin(label: &str) -> String {
    jope_pipeline::sha256_hex(label.as_bytes())
}

fn roles() -> Vec<u8> {
    serde_json::to_vec(&json!({"schema":"clover.discovery-family-roles/v1",
        "families":{"synthetic-linear-parent":"train","forbidden-parent":"sealed"}}))
    .unwrap()
}

fn fixture() -> (TrainingRoster, Cut, PolicySeeds, RetrievalSeeds) {
    let cut = Cut {
        index: 0,
        train_x: (0..16).map(|i| vec![i as f64, (i * i) as f64]).collect(),
        train_y: (0..16).map(|i| 1. + i as f64 * 0.5).collect(),
        hold_x: (16..20).map(|i| vec![i as f64, (i * i) as f64]).collect(),
        hold_y: (16..20).map(|i| 1. + i as f64 * 0.5).collect(),
        hold_indices: (16..20).collect(),
    };
    let context = DiscoveryContext {
        context_sha256: pin("synthetic-context"),
        parent_family: "synthetic-linear-parent".into(),
        role: "train".into(),
        dataset_sha256: pin("synthetic-data"),
        cut: 0,
        training_sha256: rows_digest(&cut.train_x, &cut.train_y).unwrap(),
        inner_holdout_sha256: rows_digest(&cut.hold_x, &cut.hold_y).unwrap(),
        preprocessing_sha256: pin("fixture-preprocessing"),
        encoder_input_sha256: jope_pipeline::sha256_hex(&vec![0; 2051 * 8]),
    };
    let genome = canonical_genome("(pipe X (standardize append=false) (ridge))").unwrap();
    let policy = PolicySeeds {
        schema: "clover.policy-seeds/v1".into(),
        snapshot: PolicySnapshot {
            model_class: "PolicyLM".into(),
            context_width: 2051,
            parameter_sha256: pin("fixture-parameters-not-a-live-actor"),
            checkpoint_sha256: pin("fixture-checkpoint"),
            code_sha256: pin("fixture-code"),
            tokenizer_sha256: pin("fixture-tokenizer"),
            support_sha256: pin("fixture-support"),
            normalizer_sha256: pin("fixture-normalizer"),
        },
        context_sha256: context.context_sha256.clone(),
        encoder_input_sha256: context.encoder_input_sha256.clone(),
        sampling_protocol: "trie_children_temperature_v1".into(),
        batch_protocol_sha256: pin("fixture-batch-protocol"),
        temperature: 1.,
        proposals: vec![PolicyProposal {
            program_sha256: jope_pipeline::sha256_hex(genome.as_bytes()),
            genome,
            draw_seed: 101,
            behavior_logprob: -3.,
        }],
        surrogate_funnel: None,
    };
    let retrieval = RetrievalSeeds {
        seeds: vec![RetrievalSeed {
            genome: canonical_genome("(pipe X (row_stats append=true) (ridge))").unwrap(),
            source_context_sha256: context.context_sha256.clone(),
            measurement_receipt_sha256: pin("fixture-retrieval-receipt"),
        }],
    };
    let roster = TrainingRoster {
        schema: "clover.discovery-roster/v1".into(),
        role_manifest_sha256: jope_pipeline::sha256_hex(&roles()),
        contexts: vec![context],
    };
    (roster, cut, policy, retrieval)
}
