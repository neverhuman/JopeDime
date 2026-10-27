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
    std::fs::write(&code, b"\x7fELF-not-really-but-long-enough").unwrap();
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
    let mut malformed = value;
    malformed["status"] = serde_json::json!({"partial":23});
    assert!(!jope_datasets::schema::validate(
        &schema("jope.landmark-encoding.v1.schema.json"),
        &malformed
    )
    .is_empty());
}
