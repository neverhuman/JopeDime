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
