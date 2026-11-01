//! The appliance's JSON documents match their committed schemas (schemas/jope.appliance-*.json),
//! checked with the repository's own validator.
use jope_dist::pack::{pack, Output, PackRequest};
use jope_dist::{fixture, report};
use serde_json::Value;

fn schema(name: &str) -> Value {
    let path = jope_dist::repo_root().join("schemas").join(name);
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn valid(name: &str, instance: &Value) {
    let errors = jope_datasets::schema::validate(&schema(name), instance);
    assert!(errors.is_empty(), "{name}: {errors:#?}");
}

#[test]
fn the_committed_lock_matches_its_schema() {
    let lock: Value = serde_json::from_slice(
        &std::fs::read(jope_dist::repo_root().join("deploy/appliance/models.lock.json")).unwrap(),
    )
    .unwrap();
    valid("jope.appliance-model-lock.v1.schema.json", &lock);
}

#[test]
fn a_packed_manifest_matches_its_schema() {
    let repo = jope_dist::repo_root();
    let dir = tempfile::tempdir().unwrap();
    let code = dir.path().join("code");
    let mut elf = vec![0u8; 64];
    elf[0..4].copy_from_slice(b"\x7fELF");
    elf[4] = 2; // ELFCLASS64
    elf[5] = 1; // ELFDATA2LSB
    elf[6] = 1; // EV_CURRENT
    elf[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
    std::fs::write(&code, elf).unwrap();
    let models = dir.path().join("models");
    fixture::write(&models, &repo.join("deploy/appliance/fixtures/bundles")).unwrap();
    let notices = dir.path().join("notices.json");
    std::fs::write(
        &notices,
        format!(r#"{{"schema":"{}"}}"#, jope_dist::notices::SCHEMA),
    )
    .unwrap();
    let out = dir.path().join("jopedime");
    pack(
        &PackRequest {
            profile: &repo.join("deploy/appliance/profiles/fixture.toml"),
            lock: &repo.join("deploy/appliance/models.lock.json"),
            repo: &repo,
            code: &code,
            models: &models,
            notices: &notices,
            source_sha: "89abcdef0123456789abcdef0123456789abcdef",
            target: "x86_64-unknown-linux-gnu",
            release_key: None,
        },
        Output::Development { out: &out },
        None,
    )
    .unwrap();
    let inspected = report::inspect(&out, &report::trusted(&[], true, false).unwrap()).unwrap();
    valid(
        "jope.appliance-manifest.v1.schema.json",
        &inspected["manifest"],
    );
}

#[test]
fn a_fitted_header_matches_its_schema() {
    use jope_runtime::api::{fit, FitOptions};
    use jope_runtime::budget::Budget;
    use jope_runtime::limits::Limits;
    use jope_runtime::table::{parse_csv, training};
    use jope_runtime::task::TaskChoice;
    let mut text = String::from("a,b,y\n");
    for i in 0..30 {
        text.push_str(&format!(
            "{},{},{}\n",
            i,
            ["x", "y"][i % 2],
            if i % 3 == 0 { "yes" } else { "no" }
        ));
    }
    let limits = Limits::default();
    let table = parse_csv(text.as_bytes(), &limits).unwrap();
    let training = training(&table, "y", TaskChoice::Auto, &limits).unwrap();
    let outcome = fit(
        &training,
        &FitOptions::default(),
        &Budget::new(None, 1),
        &"0".repeat(64),
        Default::default(),
    )
    .unwrap();
    valid(
        "jope.appliance-fitted.v1.schema.json",
        &serde_json::to_value(&outcome.fitted.header).unwrap(),
    );
}

#[test]
fn landmark_companion_receipt_schema_and_derived_coverage_are_explicit() {
    let value = serde_json::json!({
        "schema":"jope.landmark-encoding/v1", "status":"none", "mask":vec![false;23],
        "subsample_n":300,"source_digest":"a".repeat(64),"conditioning_digest":"b".repeat(64),
        "weights_sha256":"d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c",
        "embedding_identity":"dope-mlcv3-high-f32-descriptor256-task128-dist128/v1",
        "device":"Cuda(0)","version":0
    });
    valid("jope.landmark-encoding.v1.schema.json", &value);
    let mut candidate = value.clone();
    candidate["embedding_identity"] = serde_json::json!(
        "dope-mlcv3-unqualified-candidate-high-f32-descriptor256-task128-dist128/v1"
    );
    candidate["weights_sha256"] = serde_json::json!("c".repeat(64));
    assert!(!jope_datasets::schema::validate(
        &schema("jope.landmark-encoding.v1.schema.json"),
        &candidate
    )
    .is_empty());
    candidate["training_receipt_sha256"] = serde_json::json!("d".repeat(64));
    valid("jope.landmark-encoding.v1.schema.json", &candidate);
    candidate["weights_sha256"] = value["weights_sha256"].clone();
    assert!(!jope_datasets::schema::validate(
        &schema("jope.landmark-encoding.v1.schema.json"),
        &candidate
    )
    .is_empty());
    let mut malformed = value;
    malformed["status"] = serde_json::json!({"partial":23});
    assert!(!jope_datasets::schema::validate(
        &schema("jope.landmark-encoding.v1.schema.json"),
        &malformed
    )
    .is_empty());
}

#[test]
fn explicit_consumer_schemas_keep_raw_and_learned_semantics_distinct() {
    let digest = "a".repeat(64);
    let mut consumer = serde_json::json!({
        "mode":"encoding-only", "recipe":"dope512-direct-consumer/v1",
        "checkpoint_identity":{"lineage":"released-v1"},
        "jope_weights_sha256":null,"components_digest":null,
        "source_digest":digest,"input_conditioning_digest":digest,"conditioning_digest":digest
    });
    let name = "jope.dope-consumer.v1.schema.json";
    valid(name, &consumer);
    consumer["mode"] = "jope-dl".into();
    assert!(!jope_datasets::schema::validate(&schema(name), &consumer).is_empty());
    consumer["recipe"] = "dope512-foundation1024-target512-jope-dl512/v1".into();
    consumer["jope_weights_sha256"] = digest.clone().into();
    consumer["components_digest"] = digest.clone().into();
    consumer["projection_identity"] = serde_json::json!({
        "schema":"jope.policy-assets-projection/v1","parent_manifest_sha256":digest,
        "projection_digest":digest,"policy_sha256":digest
    });
    consumer["component_admission"] = serde_json::json!({
        "schema":"jope.component-admission/v1","subject_sha256":digest,
        "exposure_receipt_sha256":digest,"policy_sha256":digest,"authority_sha256":digest,
        "fit_scope_digest":digest,"episode":"isolated-fixture","purpose":"episode_fit"
    });
    valid(name, &consumer);
    consumer["component_admission"]["eligible"] = true.into();
    assert!(!jope_datasets::schema::validate(&schema(name), &consumer).is_empty());
    for other in [
        "jope.landmark-encoding.v1.schema.json",
        "jope.landmark-event.v1.schema.json",
    ] {
        assert_eq!(
            schema(name)["$defs"]["consumer"],
            schema(other)["$defs"]["consumer"]
        );
        assert_eq!(
            schema(name)["$defs"]["fitScope"],
            schema(other)["$defs"]["fitScope"]
        );
    }
}

#[test]
fn component_request_schema_cannot_select_its_own_authority_or_admission() {
    let name = "jope.fusion-component-requests.v1.schema.json";
    let request = serde_json::json!({"schema":"jope.fusion-component-requests/v1",
        "entries":[{"fit_scope_digest":"a".repeat(64),"subject_sha256":"b".repeat(64),
            "exposure_receipt_sha256":"c".repeat(64),"episode":"isolated-fixture"}]});
    valid(name, &request);
    for key in [
        "authority_root",
        "issuer",
        "policy_sha256",
        "eligible",
        "authenticated",
    ] {
        let mut forged = request.clone();
        forged[key] = "caller-controlled".into();
        assert!(!jope_datasets::schema::validate(&schema(name), &forged).is_empty());
    }
}
