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

#[test]
fn grammar_source_capture_retains_decoded_and_projected_programs() {
    if std::env::var_os("JOPE_GP_GRAMMAR_CAPTURE_CHILD").is_some() {
        let vocab: jope_parallel::grammar::Vocab = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("assets/grammar/featmath.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let original = "(pipe X (nystrom gamma=0.873 k=23 seed=912) (ridge))";
        let projected = vocab.project_proposal(original).unwrap();
        assert_ne!(projected, original);
        let tokens = vocab.encode(&projected).unwrap();
        assert_eq!(vocab.decode(&tokens).unwrap(), projected);
        return;
    }
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/gp-source-tests");
    std::fs::create_dir_all(&root).unwrap();
    let directory = root.join(format!(
        "engine-grammar-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).unwrap();
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "grammar_source_capture_retains_decoded_and_projected_programs",
            "--nocapture",
        ])
        .env("JOPE_GP_GRAMMAR_CAPTURE_CHILD", "1")
        .env("JOPE_GP_SOURCE_DIR", &directory)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let mut coordinates = std::collections::BTreeMap::<String, usize>::new();
    for entry in std::fs::read_dir(&directory).unwrap() {
        for line in std::fs::read_to_string(entry.unwrap().path())
            .unwrap()
            .lines()
        {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(record["engine"], "engine-grammar");
            let program = record["program"].as_str().unwrap();
            feat_math::GenomeIr::parse(program).unwrap();
            assert_ne!(
                program,
                "(pipe X (nystrom gamma=0.873 k=23 seed=912) (ridge))"
            );
            *coordinates
                .entry(record["coordinate"].as_str().unwrap().into())
                .or_default() += 1;
        }
    }
    assert!(coordinates["decode"] >= 2); // Encoding also replays decoded programs.
    assert_eq!(coordinates["projection"], 1);
    std::fs::remove_dir_all(directory).unwrap();
}
