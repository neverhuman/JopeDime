use super::*;

fn digest(value: &str) -> String {
    sha256_hex(value.as_bytes())
}

fn anchor(index: usize) -> Value {
    let genome = format!("anchor-{index}");
    json!({
        "anchor_index": index, "engine": if index.is_multiple_of(2) { "featmath" } else { "newmath" },
        "semantic_sha256": digest(&genome), "canonical_genome": genome,
        "algorithm_family": format!("family-{index}"), "performance_quartile": index,
        "median_runtime_seconds": 1.0, "real_dataset_coverage": 100,
    })
}

fn candidate(label: &str, source: Option<&str>) -> Value {
    json!({
        "engine": "featmath", "semantic_sha256": digest(label), "canonical_genome": label,
        "source_dataset_hash": source,
    })
}

fn fixture(refill_only: bool) -> (Vec<Value>, Vec<Value>, Value) {
    let anchors = (0..4).map(anchor).collect::<Vec<_>>();
    let pilot = vec![json!({"dataset_hash":"0123456789abcdef", "split":"train"})];
    let transfer = if refill_only {
        vec![candidate("anchor-0", Some("aaaaaaaaaaaaaaaa"))]
    } else {
        vec![
            candidate("transfer-a", Some("aaaaaaaaaaaaaaaa")),
            candidate("transfer-b", Some("aaaaaaaaaaaaaaaa")),
        ]
    };
    let pools = json!({"0123456789abcdef": {
        "transfer": transfer,
        "policy": [candidate("policy-a", None), candidate("policy-b", None)],
        "diverse": [candidate("diverse-a", None), candidate("diverse-b", None)],
        "refill": [candidate("transfer-c", Some("bbbbbbbbbbbbbbbb")), candidate("transfer-d", Some("bbbbbbbbbbbbbbbb")),
            candidate("transfer-e", Some("bbbbbbbbbbbbbbbb"))],
    }});
    (anchors, pilot, pools)
}

#[test]
fn deterministic_probe_refill_produces_role_counts_and_canonical_hash() {
    let (anchors, pilot, pools) = fixture(true);
    let first = build_probe_slate(&"c".repeat(64), &anchors, &pilot, &pools, 17, 1, false).unwrap();
    let second =
        build_probe_slate(&"c".repeat(64), &anchors, &pilot, &pools, 17, 1, false).unwrap();
    assert_eq!(first, second);
    let slots = first["datasets"][0]["slots"].as_array().unwrap();
    assert_eq!(slots.len(), 10);
    assert_eq!(
        slots.iter().filter(|slot| slot["role"] == "anchor").count(),
        4
    );
    assert_eq!(
        slots
            .iter()
            .filter(|slot| slot["role"] == "transfer")
            .count(),
        2
    );
    assert_eq!(
        slots.iter().filter(|slot| slot["role"] == "policy").count(),
        2
    );
    assert_eq!(
        slots
            .iter()
            .filter(|slot| slot["role"] == "diverse")
            .count(),
        2
    );
    assert_ne!(
        slots[4]["semantic_sha256"].as_str().unwrap(),
        digest("anchor-0")
    );
    let claimed = first["slate_sha256"].as_str().unwrap();
    let mut unhashed = first.clone();
    unhashed.as_object_mut().unwrap().remove("slate_sha256");
    assert_eq!(claimed, canonical_hash(&unhashed).unwrap());
}

#[test]
fn insufficient_unique_candidates_keep_the_python_error() {
    let anchors = (0..4).map(anchor).collect::<Vec<_>>();
    let pilot = vec![json!({"dataset_hash":"0123456789abcdef", "split":"train"})];
    let pools =
        json!({"0123456789abcdef": {"transfer": [], "policy": [], "diverse": [], "refill": []}});
    let error =
        build_probe_slate(&"c".repeat(64), &anchors, &pilot, &pools, 17, 1, false).unwrap_err();
    assert!(error
        .to_string()
        .contains("dataset 0123456789abcdef cannot deterministically refill transfer"));
}

#[test]
fn generated_slate_sorts_pilot_datasets() {
    let (anchors, mut pilot, pools) = fixture(false);
    pilot.push(json!({"dataset_hash":"0000000000000001", "split":"train"}));
    let pools = json!({
        "0123456789abcdef": pools["0123456789abcdef"].clone(),
        "0000000000000001": pools["0123456789abcdef"].clone(),
    });
    let slate = build_probe_slate(&"c".repeat(64), &anchors, &pilot, &pools, 17, 1, false).unwrap();
    assert_eq!(slate["datasets"][0]["dataset_hash"], "0000000000000001");
}
