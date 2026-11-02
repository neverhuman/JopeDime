use super::*;
use serde_json::json;

use super::*;
use serde_json::json;

fn digest_text(text: &str) -> String {
    digest(text.as_bytes())
}

fn fixture() -> Value {
    let anchors: Vec<Value> = (0..ANCHOR_COUNT)
        .map(|index| {
            let genome = format!("anchor-{index}");
            json!({
                "anchor_index": index,
                "engine": if index % 2 == 0 { "featmath" } else { "newmath" },
                "semantic_sha256": digest_text(&genome),
                "canonical_genome": genome,
                "algorithm_family": format!("family-{index}"),
                "performance_quartile": index,
                "median_runtime_seconds": index as f64 + 0.25,
                "real_dataset_coverage": 100,
            })
        })
        .collect();
    let roles = [
        "anchor", "anchor", "anchor", "anchor", "transfer", "transfer", "policy", "policy",
        "diverse", "diverse",
    ];
    let mut slots = Vec::new();
    for (index, role) in roles.into_iter().enumerate() {
        let genome = if role == "anchor" {
            anchors[index]["canonical_genome"]
                .as_str()
                .unwrap()
                .to_owned()
        } else {
            format!("slot-{index}")
        };
        let semantic = if role == "anchor" {
            anchors[index]["semantic_sha256"]
                .as_str()
                .unwrap()
                .to_owned()
        } else {
            digest_text(&genome)
        };
        slots.push(json!({
                "slot": index,
                "role": role,
                "engine": if role == "anchor" { anchors[index]["engine"].clone() } else { json!("featmath") },
                "semantic_sha256": semantic,
                "canonical_genome": genome,
                "source_dataset_hash": if role == "transfer" { json!("0123456789abcdef") } else { Value::Null },
                "anchor_index": if role == "anchor" { json!(index) } else { Value::Null },
            }));
    }
    let mut slate = json!({
        "schema": PROBE_SLATE_SCHEMA,
        "slate_sha256": "0".repeat(64),
        "catalog_sha256": "1".repeat(64),
        "created_at_ms": 1,
        "seed": 0,
        "screening_protocol": {
            "fidelity": "screening", "folds": 3, "row_cap": 768,
            "timeout_seconds": 30, "omp_threads": 1, "train_only": true
        },
        "exact_protocol": {
            "fidelity": "exact", "folds": 5, "row_cap": null,
            "timeout_seconds": null, "omp_threads": 1, "train_only": true
        },
        "anchors": anchors,
        "datasets": [{
            "dataset_hash": "aaaaaaaaaaaaaaaa", "synthetic_split": "train", "slots": slots
        }],
    });
    reseal(&mut slate);
    slate
}

fn reseal(slate: &mut Value) {
    let mut unhashed = slate.as_object().unwrap().clone();
    unhashed.remove("slate_sha256");
    let mut canonical = String::new();
    canonical_json(&Value::Object(unhashed), &mut canonical).unwrap();
    slate["slate_sha256"] = json!(digest_text(&canonical));
}

#[test]
fn accepts_fixture_and_enforces_production_size() {
    let slate = fixture();
    assert_eq!(validate_probe_slate(&slate, false), Ok(()));
    assert_eq!(
        validate_probe_slate(&slate, true).unwrap_err(),
        "production pilot requires exactly 2,000 synthetic datasets"
    );
}

#[test]
fn content_hash_and_exact_fields_are_enforced() {
    let mut slate = fixture();
    slate["seed"] = json!(1);
    assert_eq!(
        validate_probe_slate(&slate, false).unwrap_err(),
        "probe slate content hash differs"
    );
    slate["unexpected"] = json!(true);
    reseal(&mut slate);
    assert_eq!(
        validate_probe_slate(&slate, false).unwrap_err(),
        "probe slate has unknown or missing fields"
    );
}

#[test]
fn protocol_and_anchor_semantics_are_bound() {
    let mut slate = fixture();
    slate["screening_protocol"]["row_cap"] = json!(767);
    reseal(&mut slate);
    assert_eq!(
        validate_probe_slate(&slate, false).unwrap_err(),
        "screening protocol must be CV3/768 rows/30 seconds"
    );

    let mut slate = fixture();
    slate["anchors"][0]["canonical_genome"] = json!("changed-genome");
    reseal(&mut slate);
    assert_eq!(
        validate_probe_slate(&slate, false).unwrap_err(),
        "anchor semantic hash does not bind its genome"
    );
}

#[test]
fn slots_and_dataset_order_are_enforced() {
    let mut slate = fixture();
    slate["datasets"][0]["slots"][4]["role"] = json!("policy");
    reseal(&mut slate);
    assert_eq!(
        validate_probe_slate(&slate, false).unwrap_err(),
        "probe dataset role counts must be 4/2/2/2"
    );

    let mut slate = fixture();
    let duplicate = slate["datasets"][0].clone();
    slate["datasets"].as_array_mut().unwrap().push(duplicate);
    reseal(&mut slate);
    assert_eq!(
        validate_probe_slate(&slate, false).unwrap_err(),
        "probe datasets must be strictly sorted and unique"
    );
}

#[test]
fn canonical_json_matches_python_compact_sorted_encoding() {
    let value = json!({"é": 0.00001, "v": 0.0001});
    let mut encoded = String::new();
    canonical_json(&value, &mut encoded).unwrap();
    assert_eq!(encoded, "{\"v\":0.0001,\"é\":1e-05}");
}
