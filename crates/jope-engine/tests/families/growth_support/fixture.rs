use super::*;

pub(super) fn sign(key: &[u8], reference: &mut Value) {
    let message = format!(
        "{}{}",
        family_growth::SIGNATURE_DOMAIN,
        reference["sha256"].as_str().unwrap()
    );
    reference["signature"] = json!(family_growth::hmac_sha256_hex(key, message.as_bytes()));
}

/// Port of the `GrowthFixture` test double of the retired
/// `python/src/jope/tools/test_family_growth_policy.py`: trusted policy and reviewer-key
/// custody, a parent registry, a component graph and one signed approving decision for `new`.
pub(crate) struct GrowthFixture {
    pub(crate) w: World,
    pub(crate) parent: Value,
    pub(crate) graph: Value,
    pub(crate) graph_ref: Value,
    pub(crate) decision: Value,
    pub(crate) decisions: Vec<Value>,
}

impl GrowthFixture {
    pub(crate) fn new(root: &Path, parent: Option<Value>, absolute: bool) -> Self {
        let w = World {
            root: root.to_path_buf(),
            absolute,
        };
        w.put_bytes("credentials/provenance-review/test-reviewer.key", TEST_KEY);
        let policy = w.put(
            family_growth::POLICY_PATH,
            &json!({"schema": family_growth::POLICY_SCHEMA, "seed": SEED, "reviewers": ["test-reviewer"]}),
        );
        let parent = parent.unwrap_or_else(|| {
            json!({"schema": "parallel.family_registry.v1", "families": {
                "old-train": {"source_keys": ["old-train-alias"], "partition": "meta_train"},
                "old-test": {"source_keys": ["old-test-alias"], "partition": "meta_test"},
            }})
        });
        let parent_ref = w.put("parent.json", &parent);
        let source = w.put("source.json", &json!({"synthetic_unit_fixture": true}));
        let mut components = vec![];
        for (family, row) in parent["families"].as_object().unwrap() {
            let mut aliases: BTreeSet<&str> = row["source_keys"]
                .as_array()
                .unwrap()
                .iter()
                .map(|k| k.as_str().unwrap())
                .collect();
            aliases.insert(family);
            components
                .push(json!({"component_id": family, "aliases": aliases, "source_objects": []}));
        }
        components.push(json!({"component_id": "new", "aliases": ["new", "new-alias"], "source_objects": [source]}));
        let graph = json!({"schema": "parallel.component_graph.v1", "components": components});
        let graph_ref = w.put("graph.json", &graph);
        let prior = w.put(
            "prior-use.json",
            &json!({"component_id": "new", "occurrences": [], "scope": "synthetic-test-fixture"}),
        );
        let decision = json!({
            "schema": "parallel.provenance_decision.v3", "component_id": "new",
            "admission": "approved", "reviewer": "test-reviewer",
            "graph_sha256": graph_ref["sha256"], "parent_registry_sha256": parent_ref["sha256"],
            "policy_sha256": policy["sha256"], "task": "regression", "target": "y",
            "predictors": ["x"], "units": "fixture-units",
            "entity_time_contract": {"strategy": "iid"}, "source_objects": [source],
            "prior_use_findings": {"status": "unused", "parent_registry_sha256": parent_ref["sha256"], "evidence": [prior]},
        });
        let mut fixture = Self {
            w,
            parent: parent_ref,
            graph,
            graph_ref,
            decision,
            decisions: vec![],
        };
        fixture.sign();
        fixture
    }
    pub(crate) fn sign(&mut self) {
        let mut reference = self.w.put("decision.json", &self.decision.clone());
        sign(TEST_KEY, &mut reference);
        self.decisions = vec![reference];
    }
    pub(crate) fn inputs(&self, names: &[&str]) -> Value {
        json!({
            "requested_families": names, "additional_live_families": [],
            "additional_sealed_families": [], "graph": self.graph_ref,
            "parent_registry": self.parent, "decisions": self.decisions,
        })
    }
}
