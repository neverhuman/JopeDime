//! Explicit A1 no-engine client leg. The provider supplies verified release bytes.
#![cfg(not(feature = "engine"))]

use jope_client::{Http, Request, Transport};
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

#[derive(Deserialize)]
struct Fixture {
    operation: u8,
    route: usize,
    method: String,
    target: String,
    parameters: BTreeMap<String, String>,
    query: BTreeMap<String, String>,
    body: Option<Value>,
    idempotency_key: Option<String>,
}

struct OwnedEngine(Child);
impl Drop for OwnedEngine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
#[ignore = "requires the A1 provider's verified candidate and source bindings"]
fn a1_lite_registry_and_candidate_background_contract() {
    let fixtures: Vec<Fixture> =
        serde_json::from_str(include_str!("contract/operations.json")).unwrap();
    assert_eq!(
        fixtures
            .iter()
            .map(|row| row.operation)
            .collect::<BTreeSet<_>>(),
        (1..=17).collect()
    );
    // Compile and exercise every shared request vector without the engine feature.
    for row in fixtures {
        let parameters: Vec<_> = row
            .parameters
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let query: Vec<_> = row
            .query
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let mut request = Request::new(row.operation, row.route, &parameters)
            .unwrap()
            .query(&query)
            .key(row.idempotency_key.as_deref())
            .unwrap();
        if let Some(body) = row.body.as_ref() {
            request = request.json(body).unwrap();
        }
        assert_eq!(request.method().as_str(), row.method);
        assert_eq!(request.target(), row.target);
        if let Some(body) = row.body {
            assert_eq!(
                serde_json::from_slice::<Value>(request.body()).unwrap(),
                body
            );
        } else {
            assert!(request.body().is_empty());
        }
    }

    let binary =
        PathBuf::from(std::env::var_os("JOPE_REL07_ARTIFACT").expect("candidate required"));
    assert!(!fs::symlink_metadata(&binary)
        .unwrap()
        .file_type()
        .is_symlink());
    assert!(binary.is_absolute() && binary.is_file());
    let expected = std::env::var("JOPE_REL07_SOURCE_SHA").expect("source pin required");
    let version = Command::new(&binary)
        .args(["--version", "--json"])
        .output()
        .unwrap();
    assert!(version.status.success());
    let version: Value = serde_json::from_slice(&version.stdout).unwrap();
    assert_eq!(version["source_sha40"], expected);

    let root = PathBuf::from(
        std::env::var_os("JOPE_QUALIFICATION_TEST_TMP").expect("owned scratch required"),
    );
    fs::create_dir_all(&root).unwrap();
    let home = tempfile::Builder::new()
        .prefix("a1-")
        .tempdir_in(root)
        .unwrap();
    let child = Command::new(&binary)
        .args(["serve", "--idle-exit", "120s"])
        .env("JOPEDIME_HOME", home.path())
        .env_remove("JOPEDIME_ENDPOINT")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut engine = OwnedEngine(child);
    let deadline = Instant::now() + Duration::from_secs(20);
    let record: Value = loop {
        assert!(
            engine.0.try_wait().unwrap().is_none(),
            "candidate engine exited"
        );
        if let Ok(bytes) = fs::read(home.path().join("daemon.json")) {
            if let Ok(record) = serde_json::from_slice::<Value>(&bytes) {
                break record;
            }
        }
        assert!(
            Instant::now() < deadline,
            "candidate engine did not become ready"
        );
        thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(record["pid"].as_u64(), Some(u64::from(engine.0.id())));
    let endpoint = record["url"].as_str().unwrap();
    assert!(endpoint.starts_with("http://127.0.0.1:"));
    let client = Http::new(endpoint, record["token"].as_str().unwrap()).unwrap();
    let mut response = loop {
        if let Ok(response) = client.execute(Request::new(1, 0, &[]).unwrap()) {
            if response.status == 200 {
                break response;
            }
        }
        assert!(
            Instant::now() < deadline,
            "candidate HTTP did not become ready"
        );
        thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(response.status, 200);
    let mut bytes = vec![];
    response.body.read_to_end(&mut bytes).unwrap();
    let system: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(system["api_version"].as_str().unwrap().starts_with("1."));
    // Same real-router invalid registration vector as the full in-process/HTTP case.
    let response = client
        .execute(
            Request::new(2, 0, &[])
                .unwrap()
                .key(Some("clt04-real-router-parity"))
                .unwrap(),
        )
        .unwrap();
    assert!((400..500).contains(&response.status));
    let stop = Command::new(&binary)
        .args(["serve", "--stop"])
        .env("JOPEDIME_HOME", home.path())
        .env_remove("JOPEDIME_ENDPOINT")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(stop.success());
}
