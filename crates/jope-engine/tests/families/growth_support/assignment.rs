use super::*;
use serde_json::Map;

pub(crate) fn assign(root: &Path, inputs: &Value) -> Result<Value, GrowthError> {
    family_growth::assign_new_components(&Request {
        families: &inputs["requested_families"],
        live_families: &inputs["additional_live_families"],
        sealed_test_families: &inputs["additional_sealed_families"],
        root: &json!(root.to_str().unwrap()),
        graph: &inputs["graph"],
        parent_registry: &inputs["parent_registry"],
        decisions: &inputs["decisions"],
    })
}

/// A receipt edited after it was produced.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Tamper {
    None,
    RowPartition,
    NTest,
    ExtraKey,
    DropKey(&'static str),
    Schema,
}

impl Tamper {
    pub(crate) fn apply(self, request: &mut Value) {
        let map = request.as_object_mut().unwrap();
        match self {
            Tamper::None => {}
            Tamper::RowPartition => {
                if let Some(row) = map
                    .get_mut("rows")
                    .and_then(Value::as_array_mut)
                    .and_then(|r| r.first_mut())
                {
                    row["partition"] = json!("meta_test");
                }
            }
            Tamper::NTest => {
                map.insert("n_test".into(), json!(1));
            }
            Tamper::ExtraKey => {
                map.insert("runtime_closure_note".into(), json!("added after the fact"));
            }
            Tamper::DropKey(key) => {
                map.remove(key);
            }
            Tamper::Schema => {
                map.insert(
                    "schema".into(),
                    json!("parallel.family_growth_assignment.v2"),
                );
            }
        }
    }
}

/// The cases of the retired Python module's tests (`FamilyGrowthPolicyTests`), with the same
/// fixture and mutations. `registry-parent` is the engine's shape: a frozen registry parent and
/// a `development-<i>` component.
pub(crate) fn existing_case(case: &str, root: &Path, absolute: bool) -> (Value, Tamper) {
    let mut f = GrowthFixture::new(root, None, absolute);
    let mut tamper = Tamper::None;
    let names: Vec<String> = match case {
        "live-and-sealed-aliases" => vec!["old-train-alias", "old-test-alias", "new-alias", "new"]
            .into_iter()
            .map(String::from)
            .collect(),
        "alias-invariance-canonical" => vec!["new".into()],
        "alias-invariance-alias" => vec!["new-alias".into(), "new".into()],
        "forged-signature" => {
            f.decisions[0]["signature"] = json!("0".repeat(64));
            vec!["new".into()]
        }
        "duplicate-decision" => {
            let first = f.decisions[0].clone();
            f.decisions.push(first);
            vec!["new".into()]
        }
        "unresolved-component"
        | "stale-parent"
        | "missing-units"
        | "empty-prior-evidence"
        | "empty-source-objects" => {
            let d = f.decision.as_object_mut().unwrap();
            match case {
                "unresolved-component" => {
                    d.insert("component_id".into(), json!("missing"));
                }
                "stale-parent" => {
                    d.insert("parent_registry_sha256".into(), json!("a".repeat(64)));
                }
                "missing-units" => {
                    d.remove("units");
                }
                "empty-prior-evidence" => {
                    d["prior_use_findings"]["evidence"] = json!([]);
                }
                _ => {
                    d.insert("source_objects".into(), json!([]));
                }
            }
            f.sign();
            vec!["new".into()]
        }
        "graph-merge-quarantine" => {
            let components = f.graph["components"].as_array().unwrap().clone();
            let find = |id: &str| {
                components
                    .iter()
                    .find(|c| c["component_id"] == id)
                    .unwrap()
                    .clone()
            };
            let (train, test, new) = (find("old-train"), find("old-test"), find("new"));
            let mut aliases: Vec<Value> = train["aliases"].as_array().unwrap().clone();
            aliases.extend(test["aliases"].as_array().unwrap().iter().cloned());
            f.graph["components"] = json!([
                {"component_id": "old-train", "aliases": aliases, "source_objects": []},
                new,
            ]);
            f.graph_ref = f.w.put("graph.json", &f.graph.clone());
            f.decision["graph_sha256"] = f.graph_ref["sha256"].clone();
            f.sign();
            vec!["old-train-alias".into()]
        }
        "forged-partition" => {
            tamper = Tamper::RowPartition;
            vec!["new".into()]
        }
        "tampered-source" => {
            fs::write(f.w.root.join("source.json"), "tampered").unwrap();
            vec!["new".into()]
        }
        "registry-parent" => {
            let mut families = Map::new();
            for (name, key, partition) in [
                ("family-a", "fixture:old-a", "meta_train"),
                ("family-b", "fixture:old-b", "meta_validation"),
                ("family-c", "fixture:old-c", "meta_test"),
            ] {
                families.insert(name.into(), json!({
                    "family": name, "partition": partition, "assignment_method": "stable_sha256_v1",
                    "assignment_hash": null, "bucket": null, "source_keys": [key],
                    "datasets": [], "holds": [],
                }));
            }
            let parent = json!({
                "schema": "parallel.family_registry.v1", "seed": SEED, "registry_id": "fixture",
                "qualification": true, "full_inventory_complete": false,
                "independent_selection_eligible": false, "parent": null,
                "audit": {"path": "audit.json", "sha256": "0".repeat(64)}, "decisions": [],
                "reviews": [], "prepared_inputs": [], "families": families,
                "held_review_datasets": {},
            });
            f = GrowthFixture::new(root, Some(parent), absolute);
            let cid = (0..1000)
                .map(|i| format!("development-{i}"))
                .find(|c| family_growth::stable_assignment_v2(c).2 == "development_validation")
                .unwrap();
            let last = f.graph["components"]
                .as_array_mut()
                .unwrap()
                .last_mut()
                .unwrap();
            last["component_id"] = json!(cid);
            last["aliases"] = json!([cid, "new-alias"]);
            f.graph_ref = f.w.put("graph.json", &f.graph.clone());
            f.decision["component_id"] = json!(cid);
            f.decision["graph_sha256"] = f.graph_ref["sha256"].clone();
            f.sign();
            vec![cid, "new-alias".into()]
        }
        other => panic!("unknown existing case {other}"),
    };
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    (f.inputs(&names), tamper)
}
