#![cfg(feature = "neural")]

use crate::{
    corpus::{ContextRow, Group, Index},
    grammar::Vocab,
    manifest::{file_hash, hash, read_json, write_json},
    neural::{Config, Policy},
};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::Path, process::Command};
use tch::{nn, Device};

// This private unit-test child exercises the CPU post-admission algorithm without
// issuing a grant. No such entry exists in the shipped binary.
#[test]
#[ignore = "private post-admission CPU fixture child; not a qualified GPU grant"]
fn manifest_child() {
    let Some(root) = std::env::var_os("JOPE_RANKED_FIXTURE_ROOT") else {
        return;
    };
    let manifest = std::env::var_os("JOPE_RANKED_FIXTURE_MANIFEST").unwrap();
    let root = Path::new(&root);
    let manifest = Path::new(&manifest);
    let result = match std::env::var("JOPE_RANKED_FIXTURE_COMMAND")
        .unwrap()
        .as_str()
    {
        "propose-ranked" => super::run_admitted(root, manifest, 0, 1),
        "assemble-ranked" => super::assemble(root, manifest),
        other => panic!("unexpected fixture command {other}"),
    };
    if let Err(error) = result {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}

fn invoke(root: &Path, manifest: &Path, command: &str) -> std::process::Output {
    Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "ranked_proposals::manifest_run_tests::manifest_child",
            "--ignored",
            "--nocapture",
        ])
        .env("JOPE_RANKED_FIXTURE_ROOT", root)
        .env("JOPE_RANKED_FIXTURE_MANIFEST", manifest)
        .env("JOPE_RANKED_FIXTURE_COMMAND", command)
        .env("JOPE_RANKED_DEVICE", "cpu")
        .output()
        .unwrap()
}

#[test]
fn isolated_manifest_run_freezes_six_controls_resumes_and_assembles_shared_tail() {
    tch::set_num_threads(1);
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let output = root.join("experiment");
    let corpus = output.join("frozen/corpus");
    fs::create_dir_all(&corpus).unwrap();
    let vocabulary = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json");
    fs::copy(vocabulary, corpus.join("vocabulary.json")).unwrap();
    let vocab: Vocab = read_json(&corpus.join("vocabulary.json")).unwrap();
    let vocab_sha = file_hash(&corpus.join("vocabulary.json")).unwrap();
    let prepared = root.join("prepared");
    fs::create_dir_all(&prepared).unwrap();
    let prepared_path = prepared.join("prepared.json");
    write_json(&prepared_path,&json!({
        "schema":"parallel.prepared.v1","dataset":"validation","family":"validation_family","task":"regression",
        "meta_partition":"meta_validation","seed":20260907,"family_partition_frozen":true,
        "clean_evaluation":false,"clean_status":"qualification fixture","original_train_sha256":"a".repeat(64),
        "original_test_sha256":"b".repeat(64),"selected_train_indices":[0,1],"selected_test_indices":[0],
        "target_last":true,"rows_total":3,"cuts":[{"index":5,"train_indices":[0],"validation_indices":[1],
        "training_sha256":"c".repeat(64),"validation_sha256":"d".repeat(64),"preprocessing_sha256":"e".repeat(64)}]
    })).unwrap();
    let prepared_sha = file_hash(&prepared_path).unwrap();
    let context_bytes = |offset: usize| {
        (0..2048)
            .flat_map(|i| (((i + offset) % 11) as f32 / 11.).to_le_bytes())
            .collect::<Vec<_>>()
    };
    let validation_context = root.join("validation.f32");
    let train_context = root.join("training.f32");
    fs::write(&validation_context, context_bytes(0)).unwrap();
    fs::write(&train_context, context_bytes(3)).unwrap();
    let validation_sha = file_hash(&validation_context).unwrap();
    let train_sha = file_hash(&train_context).unwrap();
    let index = Index {
        schema: "parallel.corpus.v1".into(),
        qualification: true,
        vocabulary_sha256: vocab_sha.clone(),
        context_index_sha256: "f".repeat(64),
        records_sha256: "0".repeat(64),
        record_bytes: 532,
        records: 2,
        legal_sets: vec![],
        contexts: vec![
            ContextRow {
                dataset: "training".into(),
                family: "training_family".into(),
                cut: 0,
                partition: "meta_train".into(),
                path: train_context.clone(),
                sha256: train_sha,
                prepared: prepared.clone(),
                prepared_sha256: prepared_sha.clone(),
            },
            ContextRow {
                dataset: "validation".into(),
                family: "validation_family".into(),
                cut: 5,
                partition: "meta_validation".into(),
                path: validation_context.clone(),
                sha256: validation_sha.clone(),
                prepared: prepared.clone(),
                prepared_sha256: prepared_sha.clone(),
            },
        ],
        labels_sha256: None,
        groups: vec![
            Group {
                context: 0,
                family: "training_family".into(),
                partition: "meta_train".into(),
                records: vec![0],
                max_length: 5,
            },
            Group {
                context: 1,
                family: "validation_family".into(),
                partition: "meta_validation".into(),
                records: vec![1],
                max_length: 5,
            },
        ],
    };
    write_json(&corpus.join("index.json"), &index).unwrap();
    fs::write(corpus.join("labels.jsonl"),format!("{}\n{}\n",json!({"record":0,"context":0,"program":"(pipe X (ridge_gcv))","rmse_cut_normalized":0.3}),json!({"record":1,"context":1,"program":"(pipe X (ridge))","rmse_cut_normalized":0.8}))).unwrap();
    write_json(&corpus.join("provenance.json"),&json!({"index_sha256":file_hash(&corpus.join("index.json")).unwrap(),"labels_sha256":file_hash(&corpus.join("labels.jsonl")).unwrap()})).unwrap();
    let mut config = Config::regression(vocab.tokens.len() as i64);
    config.width = 16;
    config.heads = 4;
    config.layers = 1;
    config.ffn = 32;
    config.dropout = 0.;
    let vs = nn::VarStore::new(Device::Cpu);
    let _model = Policy::new(vs.root(), config.clone()).unwrap();
    let checkpoint = output.join("frozen/checkpoint");
    let earlier = output.join("frozen/earlier");
    let mut artifact_paths = vec![
        corpus.join("index.json"),
        corpus.join("vocabulary.json"),
        corpus.join("labels.jsonl"),
        corpus.join("provenance.json"),
        prepared_path,
        validation_context.clone(),
        train_context,
    ];
    for directory in [&checkpoint, &earlier] {
        fs::create_dir_all(directory).unwrap();
        vs.save(directory.join("model.ot")).unwrap();
        write_json(&directory.join("config.json"), &config).unwrap();
        write_json(&directory.join("state.json"),&json!({
            "model_sha256":file_hash(&directory.join("model.ot")).unwrap(),"config_sha256":file_hash(&directory.join("config.json")).unwrap(),
            "corpus_sha256":file_hash(&corpus.join("index.json")).unwrap(),"vocabulary_sha256":vocab_sha,"autocast_cache_lifetime":"scoped_v2"
        })).unwrap();
        artifact_paths.extend([
            directory.join("model.ot"),
            directory.join("config.json"),
            directory.join("state.json"),
        ]);
    }
    let original_snapshot = output.join("frozen/champion-snapshot.json");
    write_json(&original_snapshot, &json!({"schema":"parallel.champion_snapshot.v1","request":{},"evaluator_sha256":"a".repeat(64),"vocabulary_sha256":vocab_sha,"entries":[]})).unwrap();
    artifact_paths.push(original_snapshot.clone());
    let artifacts = artifact_paths
        .iter()
        .map(|p| json!({"path":p,"sha256":file_hash(p).unwrap()}))
        .collect::<Vec<_>>();
    let manifest = output.join("manifest.json");
    write_json(&manifest,&json!({"schema":"parallel.seed_experiment.v1","output":output,"corpus":corpus,"checkpoint":checkpoint,"earlier_checkpoint":earlier,"seed":20260908,"artifacts":artifacts,"champion_snapshot":original_snapshot,
        "views":[{"dataset":"validation","family":"validation_family","cut":5,"prepared":prepared,"prepared_sha256":prepared_sha,"context_path":validation_context,"context_sha256":validation_sha}]
    })).unwrap();
    let champion_sentinel = root.join("runs/champions/sentinel");
    fs::create_dir_all(champion_sentinel.parent().unwrap()).unwrap();
    fs::write(&champion_sentinel, b"production stays durable").unwrap();
    // Assembly cannot consult teachers until every pure ranking is frozen.
    assert!(!invoke(root, &manifest, "assemble-ranked").status.success());
    let first = invoke(root, &manifest, "propose-ranked");
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let ranking_path = output.join("proposals/validation/cut-5/rankings.json");
    let frozen = fs::read(&ranking_path).unwrap();
    let ranking: Value = serde_json::from_slice(&frozen).unwrap();
    assert_eq!(ranking["sampling"]["attempted"], 128);
    assert_eq!(ranking["rankings"]["current"].as_array().unwrap().len(), 64);
    assert_eq!(ranking["rankings"].as_object().unwrap().len(), 6);
    assert_eq!(ranking["rankings"]["generic"][0], "(pipe X (ridge))");
    assert_eq!(
        ranking["reward_runtime_parity"]["selected_order_identical"],
        true
    );
    assert_eq!(
        ranking["rankings"]["larger_pool"],
        ranking["rankings"]["earlier_checkpoint"]
    );
    let pool = ranking["candidate_pool"].as_array().unwrap();
    assert!(pool.len() > 64);
    assert!(pool.iter().any(|p| p["program"] == "(pipe X (ridge_gcv))"));
    for p in pool {
        let program = p["program"].as_str().unwrap();
        assert_eq!(hash(program.as_bytes()), p["program_sha256"]);
        vocab.encode(program).unwrap();
        assert!(p["origins"]
            .as_array()
            .unwrap()
            .iter()
            .all(|o| o["source_family"] != "validation_family"));
    }
    let resume = invoke(root, &manifest, "propose-ranked");
    assert!(
        resume.status.success(),
        "{}",
        String::from_utf8_lossy(&resume.stderr)
    );
    assert_eq!(frozen, fs::read(&ranking_path).unwrap());
    let assemble = invoke(root, &manifest, "assemble-ranked");
    assert!(
        assemble.status.success(),
        "{}",
        String::from_utf8_lossy(&assemble.stderr)
    );
    let mut tails = Vec::new();
    for arm in ["current", "larger_pool", "generic"] {
        let population: Value =
            read_json(&output.join(format!("downstream/validation/cut-5/{arm}/seeds.json")))
                .unwrap();
        let candidates = population["candidates"].as_array().unwrap();
        assert_eq!(candidates.len(), 80);
        assert_eq!(
            candidates
                .iter()
                .map(|p| p["genome"].as_str().unwrap())
                .collect::<BTreeSet<_>>()
                .len(),
            80
        );
        tails.push(candidates[64..].to_vec());
    }
    assert!(tails.windows(2).all(|pair| pair[0] == pair[1]));
    assert_eq!(frozen, fs::read(&ranking_path).unwrap());
    assert_eq!(
        fs::read(&champion_sentinel).unwrap(),
        b"production stays durable"
    );
    // Regenerate every pure control after changing the adaptive teacher store
    // and actual materialized held-out labels, keeping saved training inputs
    // and model weights fixed. No ranking or prediction may change.
    fs::write(&champion_sentinel, b"replacement retained teachers").unwrap();
    let heldout = prepared.join("cut-5/validation.csv");
    fs::create_dir_all(heldout.parent().unwrap()).unwrap();
    fs::write(&heldout, b"1,999999\n2,-999999\n").unwrap();
    let alternate_output = root.join("experiment-altered-teachers");
    let alternate_manifest = alternate_output.join("manifest.json");
    let alternate_snapshot = alternate_output.join("champion-snapshot.json");
    let genome = "(pipe X (standardize) (ridge))";
    let identity = json!({"dataset":"validation","family":"validation_family","partition":"meta_validation","cut":5,"prepared_sha256":prepared_sha,"training_sha256":"c".repeat(64),"validation_sha256":"d".repeat(64),"evaluator_sha256":"a".repeat(64),"vocabulary_sha256":vocab_sha});
    let mut observation = identity.clone();
    for (key, value) in json!({"genome":genome,"program_sha256":hash(genome.as_bytes()),"metrics":{"rmse":0.01},"target_scale":1.0,"failure":null,"evaluation_bias":"adaptive_cut_heldout_teacher","independent_evaluation":false}).as_object().unwrap() {
        observation[key] = value.clone();
    }
    let champion = json!({"schema":"parallel.cut_champion.v1","identity":identity,"observation":observation,"observation_sha256":hash(&serde_json::to_vec(&observation).unwrap()),"source":"fixture-observation.json","rmse_original_units":0.01,"previous_event_sha256":null,"previous_rmse_original_units":null,"improvement_number":0,"timestamp":0});
    write_json(&alternate_snapshot, &json!({"schema":"parallel.champion_snapshot.v1","request":{},"evaluator_sha256":"a".repeat(64),"vocabulary_sha256":vocab_sha,"entries":[champion]})).unwrap();
    let mut altered: Value = read_json(&manifest).unwrap();
    altered["output"] = json!(alternate_output);
    altered["champion_snapshot"] = json!(alternate_snapshot);
    altered["artifacts"]
        .as_array_mut()
        .unwrap()
        .push(json!({"path":alternate_snapshot,"sha256":file_hash(&alternate_snapshot).unwrap()}));
    write_json(&alternate_manifest, &altered).unwrap();
    let regeneration = invoke(root, &alternate_manifest, "propose-ranked");
    assert!(
        regeneration.status.success(),
        "{}",
        String::from_utf8_lossy(&regeneration.stderr)
    );
    let regenerated: Value =
        read_json(&alternate_output.join("proposals/validation/cut-5/rankings.json")).unwrap();
    for field in ["rankings", "predictions", "candidate_pool", "sampling"] {
        assert_eq!(
            ranking[field], regenerated[field],
            "teacher/heldout change affected {field}"
        );
    }
    let alternate_assembly = invoke(root, &alternate_manifest, "assemble-ranked");
    assert!(
        alternate_assembly.status.success(),
        "{}",
        String::from_utf8_lossy(&alternate_assembly.stderr)
    );
    let alternate_population: Value =
        read_json(&alternate_output.join("downstream/validation/cut-5/larger_pool/seeds.json"))
            .unwrap();
    assert_eq!(alternate_population["candidates"][64]["genome"], genome);
    assert_eq!(
        alternate_population["candidates"][64]["origin"],
        "incumbent_cut_teacher"
    );
    assert_ne!(
        alternate_population["candidates"].as_array().unwrap()[64..],
        tails[0]
    );
    assert_eq!(
        fs::read(&champion_sentinel).unwrap(),
        b"replacement retained teachers"
    );
    assert_eq!(fs::read(&heldout).unwrap(), b"1,999999\n2,-999999\n");
    // An input change is rejected before any new prediction can be admitted.
    fs::write(&validation_context, b"changed context").unwrap();
    assert!(!invoke(root, &manifest, "propose-ranked").status.success());
    assert_eq!(frozen, fs::read(&ranking_path).unwrap());
}
