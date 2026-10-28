//! Frozen real Rust outputs exercise production admission, not live publication authority.
use super::*;

const FIXTURE: &[u8] = include_bytes!("../fixtures/dope_released_optional_v1.json");
const WEIGHTS: &[u8] =
    include_bytes!("../../../../deploy/appliance/assets/dope-mlc-v3/model.safetensors");
const WEIGHTS_SHA: &str = "d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c";

fn decode_hex(value: &str) -> Vec<u8> {
    assert_eq!(value.len() % 2, 0);
    (0..value.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&value[i..i + 2], 16).unwrap())
        .collect()
}

fn proposed(state: &State, subject: &str) -> ExposureReceipt {
    let mut r = receipt(state, subject);
    // An encoder output is not a measured predictive result, runtime, or policy target.
    r.observation = Observation::Proposed;
    r.fidelity = "frozen_rust_encoder_output".into();
    r.producer = "fixture-controller-rust-dope".into();
    r
}

#[test]
fn real_dope_masks_preserve_train_and_protected_ancestry_without_invented_labels() {
    let fixture: Value = serde_json::from_slice(FIXTURE).unwrap();
    assert_eq!(fixture["schema"], "jope.dope-flywheel-fixture/v1");
    assert_eq!(fixture["qualification"], "released-baseline-execution-only");
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 9);
    let mut coverage = BTreeSet::new();
    for case in cases {
        let task = case["task"].as_str().unwrap();
        let stage = case["stage"].as_str().unwrap();
        assert!(coverage.insert((task, stage)));
        let bytes = decode_hex(case["embedding_le_f32_hex"].as_str().unwrap());
        assert_eq!(bytes.len(), 512 * 4);
        assert!(bytes
            .chunks_exact(4)
            .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()));
        assert_eq!(
            jope_learning::canon::sha256_hex(&bytes),
            case["embedding_sha256"]
        );
        assert_eq!(case["receipt"]["weights_sha256"], WEIGHTS_SHA);
        let mask = case["receipt"]["mask"].as_array().unwrap();
        assert_eq!(mask.len(), 23);
        let observed = mask.iter().filter(|v| v.as_bool().unwrap()).count();
        match stage {
            "none" => assert_eq!(observed, 0),
            "macro" => {
                assert!(observed > 0);
                for (i, value) in mask.iter().enumerate() {
                    if value.as_bool().unwrap() {
                        assert!([1, 2, 3, 4, 5, 6, 14, 15, 22].contains(&(i + 1)));
                    }
                }
            }
            "terminal" => assert!(observed > 9),
            _ => panic!("unexpected mask stage"),
        }
        for role in ["train", "sealed", "judge", "meta_test", "test_adaptive"] {
            let state = State::at(&scratch());
            // Numeric table data is an artifact, not a canonical identity payload.
            let input = state
                .put_blob(&serde_json::to_vec(&fixture["inputs"][task]).unwrap())
                .unwrap();
            let mut parent = proposed(&state, &input);
            parent.original_role = role.into();
            let input_proof = retain(&state, &parent);
            let weights = state.put_blob(WEIGHTS).unwrap();
            assert_eq!(weights, WEIGHTS_SHA);
            let weights_proof = retain(&state, &proposed(&state, &weights));
            let scope = retain(&state, &case["receipt"]);
            let scope_proof = retain(&state, &proposed(&state, &scope));
            let subject = state.put_blob(&bytes).unwrap();
            let mut output = proposed(&state, &subject);
            output.parents = vec![
                input_proof.clone(),
                weights_proof.clone(),
                scope_proof.clone(),
            ];
            let proof = retain(&state, &output);
            let proofs = vec![input_proof, weights_proof, scope_proof, proof.clone()];
            let admission = verifier(&state, &proofs);
            for purpose in [
                Purpose::GlobalPolicyTraining,
                Purpose::RepresentationLearning,
                Purpose::RetrievalIndexUpdate,
                Purpose::Distillation,
            ] {
                let result = admission.admit(&subject, &proof, purpose, None, NOW);
                if role == "train" {
                    let result = result.unwrap();
                    assert_eq!(result.subject_sha256(), subject);
                    let labels = result.labels();
                    assert!(labels.quality.value().is_none());
                    assert!(labels.runtime.value().is_none());
                    assert!(labels.failure.value().is_none());
                    assert!(labels.policy.value().is_none());
                } else {
                    refused(result, "protected");
                }
            }
            if role == "train" {
                let mut altered = bytes.clone();
                altered[0] ^= 1;
                let altered = state.put_blob(&altered).unwrap();
                refused(
                    admission.admit(&altered, &proof, Purpose::RepresentationLearning, None, NOW),
                    "another subject",
                );
                let mut unadmitted = output.clone();
                unadmitted.producer = "unadmitted-copy".into();
                let unadmitted = retain(&state, &unadmitted);
                refused(
                    admission.admit(
                        &subject,
                        &unadmitted,
                        Purpose::RepresentationLearning,
                        None,
                        NOW,
                    ),
                    "issuer proof",
                );
                output.parents.push("f".repeat(64));
                let missing = retain(&state, &output);
                let mut with_missing = proofs.clone();
                with_missing.push(missing.clone());
                let admission = verifier(&state, &with_missing);
                assert!(admission
                    .admit(
                        &subject,
                        &missing,
                        Purpose::RepresentationLearning,
                        None,
                        NOW
                    )
                    .is_err());
            }
        }
    }
    assert_eq!(coverage.len(), 9);
}
