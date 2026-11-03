use super::fixture::sign;
use super::*;
use serde_json::Map;
use std::collections::BTreeMap;

/// SplitMix64, for a scenario set that is the same on every host.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
    fn subset(&mut self, items: &[String], max: usize) -> Vec<String> {
        let mut chosen = vec![];
        for _ in 0..self.below(max + 1) {
            if !items.is_empty() {
                chosen.push(items[self.below(items.len())].clone());
            }
        }
        chosen
    }
}

/// Decision faults of the random set: each one is a refusal (or a malformed input) of the
/// policy that the Python module rejected for exactly one reason.
const DECISION_FAULTS: [&str; 38] = [
    "schema",
    "admission",
    "graph-sha",
    "parent-sha",
    "policy-sha",
    "task",
    "no-target",
    "blank-units",
    "empty-predictors",
    "numeric-predictor",
    "string-predictors",
    "no-contract",
    "null-contract",
    "grouped-no-entity",
    "temporal-no-gap",
    "bad-strategy",
    "source-mismatch",
    "prior-status",
    "prior-parent",
    "prior-empty",
    "unknown-component",
    "numeric-component",
    "unlisted-reviewer",
    "path-reviewer",
    "numeric-reviewer",
    "wrong-key",
    "list-decision",
    "bad-signature",
    "upper-signature",
    "numeric-signature",
    "extra-ref-key",
    "list-ref",
    "duplicate",
    "tamper-source",
    "escape-path",
    "tamper-prior",
    "key-symlink",
    "missing-key",
];

/// One fixed-seed random world: 0-3 frozen parent families (with random partitions and alias
/// sets), 1-3 new components with durable source objects, signed decisions for most of them
/// (about a third carrying one fault), random requested/live/sealed alias sets, and rare
/// policy, graph and receipt faults.
pub(crate) fn random_case(
    index: usize,
    base: &Path,
    root: &Path,
    absolute: bool,
) -> (Value, Tamper) {
    let mut rng = Rng(RANDOM_SEED ^ (index as u64 + 1).wrapping_mul(0xd1b5_4a32_d192_ed03));
    let w = World {
        root: root.to_path_buf(),
        absolute,
    };
    let reviewers = ["reviewer-a", "review_b"];
    let mut keys = BTreeMap::new();
    for (i, reviewer) in reviewers.iter().enumerate() {
        let len = if rng.chance(4) {
            16
        } else {
            32 + rng.below(12)
        };
        let key: Vec<u8> = (0..len)
            .map(|j| b'!' + ((j * 7 + i * 13 + index) % 90) as u8)
            .collect();
        w.put_bytes(
            &format!("credentials/provenance-review/{reviewer}.key"),
            &key,
        );
        keys.insert(*reviewer, key);
    }
    let mut policy =
        json!({"schema": family_growth::POLICY_SCHEMA, "seed": SEED, "reviewers": reviewers});
    match rng.below(40) {
        0 => policy["seed"] = json!(SEED + 1),
        1 => policy["schema"] = json!("parallel.family_growth_policy.v2"),
        2 => policy["reviewers"] = json!("reviewer-a review_b"),
        3 => {
            policy.as_object_mut().unwrap().remove("reviewers");
        }
        4 => policy["reviewers"] = json!(null),
        _ => {}
    }
    let policy_ref = if rng.chance(3) {
        let real = w.put("manifests/policy-real.json", &policy);
        std::os::unix::fs::symlink("policy-real.json", root.join(family_growth::POLICY_PATH))
            .unwrap();
        real
    } else {
        w.put(family_growth::POLICY_PATH, &policy)
    };
    let partitions = ["meta_train", "meta_validation", "meta_test"];
    let mut old: Vec<(String, Vec<String>, &str)> = vec![];
    for j in 0..rng.below(4) {
        let keys: Vec<String> = (0..rng.below(3)).map(|m| format!("fam{j}-k{m}")).collect();
        old.push((format!("fam{j}"), keys, partitions[rng.below(3)]));
    }
    let mut families = Map::new();
    for (name, keys, partition) in &old {
        families.insert(
            name.clone(),
            json!({"source_keys": keys, "partition": partition}),
        );
    }
    let parent_schema = if rng.chance(2) {
        "parallel.family_registry.v0"
    } else {
        "parallel.family_registry.v1"
    };
    let parent_ref = w.put(
        "parent.json",
        &json!({"schema": parent_schema, "families": families}),
    );
    let mut components = vec![];
    for (name, keys, _) in &old {
        let mut aliases: BTreeSet<String> = keys.iter().cloned().collect();
        aliases.insert(name.clone());
        components.push(json!({"component_id": name, "aliases": aliases, "source_objects": []}));
    }
    if components.len() >= 2 && rng.chance(12) {
        let second = components.remove(1);
        let mut merged: BTreeSet<String> = BTreeSet::new();
        for c in [&components[0], &second] {
            for a in c["aliases"].as_array().unwrap() {
                merged.insert(a.as_str().unwrap().to_owned());
            }
        }
        components[0]["aliases"] = json!(merged);
    }
    if !components.is_empty() && rng.chance(4) {
        let aliases = components[0]["aliases"].as_array_mut().unwrap();
        if aliases.len() > 1 {
            aliases.pop();
        }
    }
    let mut new_ids = vec![];
    for i in 0..1 + rng.below(3) {
        let cid = format!("cc{index}-{i}");
        let mut aliases = vec![cid.clone()];
        aliases.extend((0..rng.below(3)).map(|m| format!("{cid}-alt{m}")));
        let sources: Vec<Value> = (0..1 + rng.below(2))
            .map(|m| {
                w.put(
                    &format!("sources/{cid}-{m}.json"),
                    &json!({"component": cid, "m": m}),
                )
            })
            .collect();
        components
            .push(json!({"component_id": cid, "aliases": aliases, "source_objects": sources}));
        new_ids.push(cid);
    }
    match rng.below(50) {
        0 => {
            let last = components.last().unwrap().clone();
            components.push(last);
        }
        1 if components.len() >= 2 => {
            let alias = components[0]["aliases"][0].clone();
            components.last_mut().unwrap()["aliases"]
                .as_array_mut()
                .unwrap()
                .push(alias);
        }
        2 => components.last_mut().unwrap()["aliases"] = json!([]),
        3 => components.last_mut().unwrap()["aliases"]
            .as_array_mut()
            .unwrap()
            .push(json!(7)),
        4 => {
            let last = components.last_mut().unwrap();
            let cid = last["component_id"].clone();
            last["aliases"] = json!([cid, cid]);
        }
        _ => {}
    }
    let graph_ref = w.put(
        "graph.json",
        &json!({"schema": "parallel.component_graph.v1", "components": components}),
    );
    let prior = w.put(
        "prior-use.json",
        &json!({"scope": "random-scenario", "index": index}),
    );
    let mut decisions = vec![];
    for cid in &new_ids {
        if !rng.chance(80) {
            continue;
        }
        let reviewer = reviewers[rng.below(2)];
        let sources = components
            .iter()
            .find(|c| c["component_id"] == cid.as_str())
            .map(|c| c["source_objects"].clone())
            .unwrap();
        let contract = match rng.below(3) {
            0 => json!({"strategy": "iid"}),
            1 => json!({"strategy": "grouped", "entity": "site"}),
            _ => {
                json!({"strategy": "temporal", "entity": "site", "order": "t", "horizon": 1, "gap": 0})
            }
        };
        let mut decision = json!({
            "schema": "parallel.provenance_decision.v3", "component_id": cid,
            "admission": "approved", "reviewer": reviewer,
            "graph_sha256": graph_ref["sha256"], "parent_registry_sha256": parent_ref["sha256"],
            "policy_sha256": policy_ref["sha256"], "task": "regression", "target": "y",
            "predictors": ["x0", "x1"], "units": "units", "entity_time_contract": contract,
            "source_objects": sources,
            "prior_use_findings": {"status": "unused", "parent_registry_sha256": parent_ref["sha256"], "evidence": [prior]},
        });
        let mut signer = keys[reviewer].clone();
        let fault = if rng.chance(35) {
            DECISION_FAULTS[rng.below(DECISION_FAULTS.len())]
        } else {
            "none"
        };
        {
            let d = decision.as_object_mut().unwrap();
            match fault {
                "schema" => {
                    d.insert("schema".into(), json!("parallel.provenance_decision.v2"));
                }
                "admission" => {
                    d.insert("admission".into(), json!("rejected"));
                }
                "graph-sha" => {
                    d.insert("graph_sha256".into(), json!("0".repeat(64)));
                }
                "parent-sha" => {
                    d.insert("parent_registry_sha256".into(), json!("1".repeat(64)));
                }
                "policy-sha" => {
                    d.insert("policy_sha256".into(), json!("2".repeat(64)));
                }
                "task" => {
                    d.insert("task".into(), json!("binary"));
                }
                "no-target" => {
                    d.remove("target");
                }
                "blank-units" => {
                    d.insert("units".into(), json!(" \t\u{a0}\u{1f}"));
                }
                "empty-predictors" => {
                    d.insert("predictors".into(), json!([]));
                }
                "numeric-predictor" => {
                    d.insert("predictors".into(), json!(["x", 3]));
                }
                "string-predictors" => {
                    d.insert("predictors".into(), json!("xy"));
                }
                "no-contract" => {
                    d.remove("entity_time_contract");
                }
                "null-contract" => {
                    d.insert("entity_time_contract".into(), Value::Null);
                }
                "grouped-no-entity" => {
                    d.insert(
                        "entity_time_contract".into(),
                        json!({"strategy": "grouped"}),
                    );
                }
                "temporal-no-gap" => {
                    d.insert(
                        "entity_time_contract".into(),
                        json!({"strategy": "temporal", "entity": "site", "order": "t", "horizon": 1}),
                    );
                }
                "bad-strategy" => {
                    d.insert("entity_time_contract".into(), json!({"strategy": "random"}));
                }
                "source-mismatch" => {
                    d.insert("source_objects".into(), json!([prior]));
                }
                "prior-status" => {
                    d["prior_use_findings"]["status"] = json!("used");
                }
                "prior-parent" => {
                    d["prior_use_findings"]["parent_registry_sha256"] = json!("3".repeat(64));
                }
                "prior-empty" => {
                    d["prior_use_findings"]["evidence"] = json!([]);
                }
                "unknown-component" => {
                    d.insert("component_id".into(), json!("nowhere"));
                }
                "numeric-component" => {
                    d.insert("component_id".into(), json!(5));
                }
                "unlisted-reviewer" => {
                    d.insert("reviewer".into(), json!("mallory"));
                }
                "path-reviewer" => {
                    d.insert("reviewer".into(), json!("../reviewer-a"));
                }
                "numeric-reviewer" => {
                    d.insert("reviewer".into(), json!(12));
                }
                "wrong-key" => signer = b"a different key, long enough to sign!!".to_vec(),
                _ => {}
            }
        }
        if fault == "list-decision" {
            decision = json!([cid, reviewer]);
        }
        let rel = format!("decisions/{cid}.json");
        let mut reference = w.put(&rel, &decision);
        super::fixture::sign(&signer, &mut reference);
        match fault {
            "bad-signature" => reference["signature"] = json!("0".repeat(64)),
            "upper-signature" => {
                let upper = reference["signature"].as_str().unwrap().to_uppercase();
                reference["signature"] = json!(upper);
            }
            "numeric-signature" => reference["signature"] = json!(42),
            "extra-ref-key" => reference["note"] = json!("extra"),
            "list-ref" => reference = json!(["path", "sha256", "signature"]),
            "duplicate" => decisions.push(reference.clone()),
            "tamper-source" => {
                let first = sources[0]["path"].as_str().unwrap();
                fs::write(root.join(first), "tampered").unwrap();
            }
            "escape-path" => {
                fs::write(
                    base.join("outside.json"),
                    fs::read(root.join(&rel)).unwrap(),
                )
                .unwrap();
                reference["path"] = json!(if absolute {
                    base.join("outside.json").to_str().unwrap().to_owned()
                } else {
                    "../outside.json".to_owned()
                });
            }
            "tamper-prior" => fs::write(root.join("prior-use.json"), "changed").unwrap(),
            "key-symlink" => {
                let link = root.join(format!("credentials/provenance-review/{reviewer}.key"));
                if fs::symlink_metadata(&link).is_ok_and(|m| m.is_file()) {
                    fs::rename(&link, root.join(format!("{reviewer}.key"))).unwrap();
                    std::os::unix::fs::symlink(format!("../../{reviewer}.key"), &link).unwrap();
                }
            }
            "missing-key" => {
                let _ = fs::remove_file(
                    root.join(format!("credentials/provenance-review/{reviewer}.key")),
                );
            }
            _ => {}
        }
        decisions.push(reference);
    }
    let mut known: Vec<String> = vec![];
    for c in &components {
        for a in c["aliases"].as_array().unwrap() {
            if let Some(a) = a.as_str() {
                known.push(a.to_owned());
            }
        }
    }
    let old_aliases: Vec<String> = old
        .iter()
        .flat_map(|(n, k, _)| std::iter::once(n.clone()).chain(k.iter().cloned()))
        .collect();
    let mut requested = rng.subset(&known, 5);
    if requested.is_empty() {
        requested.push(new_ids[0].clone());
    }
    if rng.chance(4) {
        requested.push("unknown-alias".into());
    }
    let live = rng.subset(&old_aliases, 2);
    let sealed = rng.subset(&old_aliases, 2);
    let tamper = if rng.chance(10) {
        [
            Tamper::RowPartition,
            Tamper::NTest,
            Tamper::ExtraKey,
            Tamper::DropKey("rows"),
            Tamper::DropKey("root"),
            Tamper::DropKey("decisions"),
            Tamper::Schema,
        ][rng.below(7)]
    } else {
        Tamper::None
    };
    let inputs = json!({
        "requested_families": requested, "additional_live_families": live,
        "additional_sealed_families": sealed, "graph": graph_ref,
        "parent_registry": parent_ref, "decisions": decisions,
    });
    (inputs, tamper)
}
