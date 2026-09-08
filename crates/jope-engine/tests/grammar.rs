use jope_parallel::{grammar::*, manifest::*};
use std::path::Path;
#[test]
fn masks_and_decoding_match_reference_fixtures() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let p = root.join("assets/grammar/featmath.json");
    let v: Vocab = read_json(&p).unwrap();
    let fixtures: serde_json::Value = read_json(&root.join("assets/grammar/parity.json")).unwrap();
    assert_eq!(fixtures["vocab_sha256"], file_hash(&p).unwrap());
    for case in fixtures["fixtures"].as_array().unwrap() {
        let tokens: Vec<usize> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let mut state = State::default();
        for (&token, expected) in tokens[1..].iter().zip(case["masks"].as_array().unwrap()) {
            assert_eq!(serde_json::to_value(state.legal(&v)).unwrap(), *expected);
            state.advance(&v, token).unwrap();
        }
        assert!(state.done());
        let genome = v.decode(&tokens).unwrap();
        assert!(feat_math::dsl::parse_dsl(&genome).is_ok());
    }
}
#[test]
fn exact_program_labels_never_snap_parameters() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let v: Vocab = read_json(&root.join("assets/grammar/featmath.json")).unwrap();
    for genome in [
        "(pipe X (standardize) (ridge))",
        "(pipe X (rff_cos n=16 gamma=0.5 seed=7 append=true) (ridge))",
        "(pipe X (map expr=\"tanh(x*x)\") (ridge))",
    ] {
        let t = v.encode(genome).unwrap();
        assert!(!v.decode(&t).unwrap().is_empty());
    }
    assert!(v
        .encode("(pipe X (rff_cos n=17 gamma=0.513 seed=7) (ridge))")
        .is_err());
}
#[test]
fn proposal_projection_happens_before_scoring() {
    let v: jope_parallel::grammar::Vocab =
        jope_parallel::manifest::read_json(std::path::Path::new("assets/grammar/featmath.json"))
            .unwrap();
    let original = "(pipe X (nystrom gamma=0.873 k=23 seed=912) (ridge))";
    assert!(v.encode(original).is_err());
    let projected = v.project_proposal(original).unwrap();
    assert_ne!(projected, original);
    assert!(v.encode(&projected).is_ok());
    assert_eq!(v.project_proposal(&projected).unwrap(), projected);
}
