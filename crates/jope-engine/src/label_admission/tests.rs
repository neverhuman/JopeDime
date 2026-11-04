use super::*;

fn sha(c: char) -> String {
    c.to_string().repeat(64)
}

fn policy() -> Value {
    json!({
        "schema": POLICY_SCHEMA,
        "evaluator_protocol": PROTOCOL,
        "authority_instance": sha('a'),
        "runtime_sha256": sha('b'),
        "numerical_contract_sha256": sha('c'),
        "evaluator_sha256": sha('d'),
        "evaluator": "evaluator/bin",
        "authority_root": "authority",
        "authority_witness": "witness.json",
        "native_fit_policy": {"path": "native.json", "sha256": sha('e')}
    })
}

#[test]
fn strict_v2_and_native_fit_binding_preflight() {
    preflight_policy(&policy()).unwrap();
    let mut without_native = policy();
    without_native["native_fit_policy"] = Value::Null;
    preflight_policy(&without_native).unwrap();

    type Mutation = (&'static str, fn(&mut Value));
    let negatives: &[Mutation] = &[
        ("schema", |p| {
            p["schema"] = json!("parallel.label_admission.v1")
        }),
        ("protocol", |p| p["evaluator_protocol"] = json!("other")),
        ("instance", |p| p["authority_instance"] = json!("xyz")),
        ("evaluator_sha", |p| {
            p["evaluator_sha256"] = json!(sha('D').to_uppercase())
        }),
        ("witness", |p| p["authority_witness"] = json!("  ")),
        ("native_path", |p| {
            p["native_fit_policy"]["path"] = json!("")
        }),
        ("native_sha", |p| {
            p["native_fit_policy"]["sha256"] = json!("short")
        }),
    ];
    for (name, mutate) in negatives {
        let mut p = policy();
        mutate(&mut p);
        assert!(
            preflight_policy(&p).is_err(),
            "negative case {name} accepted"
        );
    }
}

#[test]
fn historical_census_closure_verified() {
    let closed = json!({
        "directories_declared": 7877,
        "expected_member_keys": 23631,
        "located_member_keys": 23631,
        "missing_member_keys": 0,
        "all_7877_payload_hashes_verified": false
    });
    census_closure(&closed).unwrap();
    let mut missing = closed.clone();
    missing["located_member_keys"] = json!(23630);
    missing["missing_member_keys"] = json!(1);
    assert!(census_closure(&missing).is_err());
    let mut short = closed.clone();
    short["directories_declared"] = json!(7876);
    assert!(census_closure(&short).is_err());
}

#[test]
fn shard_directory_payload_hashes_verified() {
    let dir = tempfile::tempdir().unwrap();
    let shard = dir.path();
    std::fs::write(shard.join("shard.json"), b"{}").unwrap();
    std::fs::write(shard.join("records.bin"), [1u8, 2, 3]).unwrap();
    std::fs::write(shard.join("labels.jsonl"), b"{\"y\":1}\n").unwrap();
    let (s, r, l) = (hash(b"{}"), hash(&[1u8, 2, 3]), hash(b"{\"y\":1}\n"));
    verify_shard_directory_payload_hashes(shard, &s, &r, &l).unwrap();
    std::fs::write(shard.join("records.bin"), [1u8, 2, 4]).unwrap();
    assert!(
        verify_shard_directory_payload_hashes(shard, &s, &r, &l).is_err(),
        "tampered byte accepted"
    );
    std::fs::remove_file(shard.join("labels.jsonl")).unwrap();
    assert!(
        verify_shard_directory_payload_hashes(shard, &s, &r, &l).is_err(),
        "missing member accepted"
    );
}

#[test]
fn census_payload_samples_and_promotion_gate() {
    let good = json!({
        "all_7877_payload_hashes_verified": false,
        "sample_exact_version_payload_verification": [
            {"hash_matches": true, "manifest_metadata_matches_parent": true}
        ]
    });
    verify_census_payload_samples(&good).unwrap();
    let mut bad = good.clone();
    bad["sample_exact_version_payload_verification"][0]["hash_matches"] = json!(false);
    assert!(verify_census_payload_samples(&bad).is_err());
    assert!(
        require_full_payload_closure(&good).is_err(),
        "unverified closure promoted"
    );
}
