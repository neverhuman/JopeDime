#![cfg(feature = "neural")]
use jope_parallel::{
    corpus::{Group, Index},
    training::{sample, sample_replay},
};
use std::collections::BTreeMap;

fn index() -> Index {
    Index {
        schema: "parallel.teaching_corpus.v1".into(),
        qualification: true,
        vocabulary_sha256: "v".into(),
        context_index_sha256: "c".into(),
        records_sha256: "r".into(),
        record_bytes: 532,
        records: 40,
        legal_sets: vec![],
        contexts: vec![],
        labels_sha256: None,
        groups: (0..5)
            .map(|i| Group {
                context: i,
                family: format!("family{}", i / 2),
                partition: if i == 4 {
                    "meta_validation"
                } else {
                    "meta_train"
                }
                .into(),
                records: (i * 8..i * 8 + 8).collect(),
                max_length: 8,
            })
            .collect(),
    }
}

#[test]
fn cli_policy_rejects_legacy_or_mismatched_labels_before_gpu_admission() {
    use jope_parallel::manifest::{file_hash, write_json};
    use serde_json::json;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let corpus = root.join("corpus");
    std::fs::create_dir(&corpus).unwrap();
    let policy = root.join("policy.json");
    let run = root.join("run");
    for strict in [false, true] {
        write_json(
            &corpus.join("index.json"),
            &json!({"label_policy":if strict {"strict_midrank_v4"} else {"reference_v1"},
            "admission_policy":{}}),
        )
        .unwrap();
        write_json(&policy,&json!({"schema":"parallel.replay_policy.v2","eligible":true,"final_tests_opened":false,
            "corpus_sha256":if strict {"changed".to_string()} else {file_hash(&corpus.join("index.json")).unwrap()}})).unwrap();
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
            .arg("--root")
            .arg(root)
            .args(["train", "--corpus"])
            .arg(&corpus)
            .arg("--output")
            .arg(&run)
            .arg("--replay-policy")
            .arg(&policy)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(
            String::from_utf8_lossy(&result.stderr).contains("matching strictly admitted corpus"),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!run.exists());
        assert!(!root.join("runs/encoding-service.lock").exists());
    }
}

#[test]
fn ungranted_valid_policy_is_still_refused_by_the_dispatch_hold() {
    // Positive input control for the test above: a replay policy that matches
    // a strictly admitted corpus passes input validation and must then be
    // refused by the original-GPU-grant hold before any lease, lock, run
    // directory or rank process exists.
    use jope_parallel::manifest::{file_hash, write_json};
    use serde_json::json;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let corpus = root.join("corpus");
    std::fs::create_dir(&corpus).unwrap();
    write_json(
        &corpus.join("index.json"),
        &json!({"label_policy":"strict_midrank_v4","admission_policy":{}}),
    )
    .unwrap();
    let policy = root.join("policy.json");
    write_json(
        &policy,
        &json!({"schema":"parallel.replay_policy.v2","eligible":true,"final_tests_opened":false,
        "corpus_sha256":file_hash(&corpus.join("index.json")).unwrap()}),
    )
    .unwrap();
    let run = root.join("run");
    let before = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<std::collections::BTreeSet<_>>();
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
        .arg("--root")
        .arg(root)
        .args(["train", "--corpus"])
        .arg(&corpus)
        .arg("--output")
        .arg(&run)
        .arg("--replay-policy")
        .arg(&policy)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success());
    assert!(
        stderr.contains("training-controller: original native GPU grant verifier unavailable"),
        "{stderr}"
    );
    assert!(
        !stderr.contains("matching strictly admitted corpus"),
        "{stderr}"
    );
    assert!(!run.exists());
    assert!(!root.join("runs/encoding-service.lock").exists());
    let after = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(before, after, "the refused request created workspace state");
}

#[test]
fn balanced_portions_exact_resume_and_broad_fallback() {
    let index = index();
    let broad = BTreeMap::from([("a".into(), vec![0, 1]), ("b".into(), vec![2, 3])]);
    let fresh = BTreeMap::from([("a".into(), vec![1]), ("b".into(), vec![3])]);
    let all = sample_replay(&index, &broad, &fresh, 0, 8192, 123, true);
    let mut resumed = sample_replay(&index, &broad, &fresh, 0, 2048, 123, true);
    resumed.extend(sample_replay(&index, &broad, &fresh, 2048, 6144, 123, true));
    assert_eq!(all, resumed);
    let forced = all
        .chunks(8)
        .step_by(2)
        .map(|ids| ids[0])
        .collect::<Vec<_>>();
    assert_eq!(forced.iter().filter(|&&id| id == 8).count(), 256);
    assert_eq!(forced.iter().filter(|&&id| id == 24).count(), 256);
    assert!(all.iter().all(|&id| id < 32));
    assert_ne!(
        all,
        sample_replay(&index, &broad, &fresh, 0, 8192, 456, true)
    );
    assert_eq!(
        sample_replay(&index, &broad, &BTreeMap::new(), 0, 1024, 123, true),
        sample(&index, &broad, 0, 1024, 123)
    );
}

#[test]
fn reward_representation_is_order_invariant_diagnostic() {
    use jope_parallel::neural::{Config, Policy};
    use tch::{nn, Device, Kind, Tensor};
    let vs = nn::VarStore::new(Device::Cpu);
    let mut config = Config::regression(401);
    config.width = 32;
    config.ffn = 128;
    config.layers = 1;
    config.dropout = 0.0;
    let policy = Policy::new(vs.root(), config).unwrap();
    let dataset = Tensor::zeros([1, 1536], (Kind::Float, Device::Cpu));
    let target = Tensor::zeros([1, 512], (Kind::Float, Device::Cpu));
    let a = Tensor::from_slice(&[1i64, 2, 3, 4]).reshape([1, 4]);
    let b = Tensor::from_slice(&[4i64, 3, 2, 1]).reshape([1, 4]);
    let pad = Tensor::ones([1, 4], (Kind::Bool, Device::Cpu));
    let (ra, _) = policy.reward_runtime(&dataset, &target, &a, &pad).unwrap();
    let (rb, _) = policy.reward_runtime(&dataset, &target, &b, &pad).unwrap();
    assert!((ra - rb).abs().max().double_value(&[]) < 1e-6);
}

#[test]
fn fresh_portions_include_rare_fresh_programs_and_resume_exactly() {
    use jope_parallel::training::sample_replay_fresh;
    let mut index = index();
    index.groups[0].records = (0..10000).collect();
    let families = BTreeMap::from([("a".into(), vec![0])]);
    let fresh = BTreeMap::from([(0, vec![9998])]);
    let sample = |start, count| {
        sample_replay_fresh(
            &index, &families, &families, &fresh, start, count, 123, true,
        )
    };
    let all = sample(0, 8192);
    for ids in all.chunks(8).step_by(2) {
        assert_eq!(ids[0], 0);
        assert!(ids.contains(&9998));
        assert_eq!(
            ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
            8
        );
    }
    assert_eq!(all, [sample(0, 2048), sample(2048, 6144)].concat());
}

#[test]
fn compact_identifiers_preserve_all_sixteen_bits_after_gather() {
    use tch::{Kind, Tensor};
    let ids = [0i64, 1, 32767, 32768, 65534, 65535];
    let compact = Tensor::from_slice(&ids.map(|id| id as u16 as i16));
    assert_eq!(compact.kind(), Kind::Int16);
    let order = Tensor::from_slice(&[5i64, 3, 0, 2, 1, 4]);
    let gathered = compact
        .index_select(0, &order)
        .to_kind(Kind::Int64)
        .bitwise_and(0xffffi64);
    assert_eq!(
        Vec::<i64>::try_from(gathered).unwrap(),
        vec![65535, 32768, 0, 32767, 1, 65534]
    );
}
