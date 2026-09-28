use jope_parallel::{
    families::{self, Registry},
    family_growth::{self, GrowthError, Request},
    manifest::{
        file_hash, hash, read_json, write_json, Admission, AuditReport, Dataset, Decision,
        Evidence, SEED,
    },
};
use serde_json::{json, Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};
use tempfile::TempDir;

struct Fixture {
    dir: TempDir,
    audit: PathBuf,
    evidence: Evidence,
}
impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let doc = dir.path().join("documentation.html");
        fs::write(&doc, "Unit-test documentation fixture").unwrap();
        let evidence = Evidence {
            path: doc.clone(),
            sha256: file_hash(&doc).unwrap(),
            source_url: "https://example.invalid/fixture".into(),
            finding: "Temporary unit-test fixture".into(),
        };
        let mut datasets = vec![];
        for (id, family, partition, admission) in [
            ("old-a", "family-a", Some("meta_train"), Admission::Approved),
            (
                "old-b",
                "family-b",
                Some("meta_validation"),
                Admission::Approved,
            ),
            ("old-c", "family-c", Some("meta_test"), Admission::Approved),
            ("new-a", "unresolved-a", None, Admission::Unresolved),
            ("new-b", "unresolved-b", None, Admission::Unresolved),
        ] {
            let source_dir = dir.path().join(id);
            fs::create_dir(&source_dir).unwrap();
            fs::write(
                source_dir.join("meta.json"),
                "{\"target_generation\":\"original_numeric_column\"}",
            )
            .unwrap();
            fs::write(source_dir.join("train.csv"), "0,1\n1,2\n").unwrap();
            if let Some(partition) = partition {
                write_json(&source_dir.join("prepared.json"), &json!({"dataset":id,"family":family,"meta_partition":partition,"clean_evaluation":true,"storage":{}})).unwrap();
            }
            // No final test file exists: freezing must never require opening it.
            datasets.push(Dataset {
                id: id.into(),
                task: "regression".into(),
                source_dir: source_dir.clone(),
                source_key: format!("fixture:{id}"),
                family: family.into(),
                parents: vec![],
                admission,
                reason: "unit test".into(),
                evidence: vec![evidence.clone()],
                metadata_sha256: file_hash(&source_dir.join("meta.json")).unwrap(),
                train: None,
                test: None,
                split_feature_overlap: 0,
                issues: vec![],
                meta_partition: partition.map(str::to_string),
            });
        }
        let audit = dir.path().join("audit.json");
        write_json(
            &audit,
            &AuditReport {
                schema: "parallel.audit.v1".into(),
                seed: SEED,
                roots: vec![],
                datasets,
                artifact_counts: BTreeMap::new(),
                missing_roots: vec![],
                complete: false,
            },
        )
        .unwrap();
        Self {
            dir,
            audit,
            evidence,
        }
    }
    fn review(&self, label: &str, ids: &[&str], aliases: &[&str]) -> (PathBuf, PathBuf) {
        let audit: AuditReport = read_json(&self.audit).unwrap();
        let mut rows = vec![];
        let mut decisions = vec![];
        let mut probes = vec![];
        for id in ids {
            let d = audit.datasets.iter().find(|d| d.id == *id).unwrap();
            let family = format!("verified-{id}");
            let reconstruction = json!({"id":id,"reconstructed":true,"metadata_sha256":d.metadata_sha256,"source":d.source_dir.join("train.csv"),"source_sha256":file_hash(&d.source_dir.join("train.csv")).unwrap(),"training_sha256":file_hash(&d.source_dir.join("train.csv")).unwrap(),"max_match_abs_error":0.0});
            probes.push(reconstruction.clone());
            let aliases:Vec<Value> = aliases.iter().map(|key| json!({"source_key":key,"relationship":"same_source_family","evidence":self.evidence})).collect();
            rows.push(json!({"id":id,"source_key":d.source_key,"proposed_family":family,"pilot_candidate_status":"eligible_for_fresh_cohort_preparation","metadata":{"path":d.source_dir.join("meta.json"),"sha256":d.metadata_sha256,"target_generation":"original_numeric_column"},"reconstruction":reconstruction,"documentation":self.evidence,"related_source_keys":aliases}));
            decisions.push(Decision {
                key: d.source_key.clone(),
                admission: Admission::Approved,
                family,
                reason: "unit test".into(),
                evidence: vec![self.evidence.clone()],
            });
        }
        let probe = self.dir.path().join(format!("{label}-probe.json"));
        write_json(&probe, &json!({"datasets":probes})).unwrap();
        let review = self.dir.path().join(format!("{label}-review.json"));
        write_json(&review, &json!({"schema":"parallel.provenance_review.v2","active":false,"original_final_test_csv_opened":false,"reconstruction_probe":{"path":probe,"sha256":file_hash(&probe).unwrap()},"datasets":rows})).unwrap();
        let decisions_path = self.dir.path().join(format!("{label}-decisions.json"));
        write_json(&decisions_path, &decisions).unwrap();
        (review, decisions_path)
    }
    fn freeze(
        &self,
        name: &str,
        ids: &[&str],
        aliases: &[&str],
        parent: Option<&Path>,
    ) -> (PathBuf, Registry) {
        let (review, decisions) = self.review(name, ids, aliases);
        let output = self.dir.path().join(format!("{name}.json"));
        let prepared =
            ["old-a", "old-b", "old-c"].map(|id| self.dir.path().join(id).join("prepared.json"));
        let registry =
            families::freeze(&self.audit, &prepared, &review, &decisions, parent, &output).unwrap();
        (output, registry)
    }
}

#[test]
fn append_preserves_every_assignment_and_binds_dataset_review() {
    let f = Fixture::new();
    let (p0, r0) = f.freeze("base", &[], &[], None);
    let (p1, r1) = f.freeze("one", &["new-a"], &[], Some(&p0));
    let (p2, r2) = f.freeze("two", &["new-b"], &[], Some(&p1));
    for (key, old) in &r0.families {
        assert_eq!(r2.families[key].partition, old.partition);
    }
    assert_eq!(
        r1.families["verified-new-a"].partition,
        r2.families["verified-new-a"].partition
    );
    assert_eq!(r0.registry_id, r2.registry_id);
    assert_eq!(r2.parent.unwrap().sha256, file_hash(&p1).unwrap());
    let old = families::binding(&p2, "old-a", "fixture:old-a", "family-a").unwrap();
    assert!(old.entity_review_sha256.is_none());
    let new = families::binding(&p2, "new-a", "fixture:new-a", "unresolved-a").unwrap();
    assert!(new.entity_review_sha256.is_some());
    assert!(new.qualification && !new.full_inventory_complete);
    assert!(families::binding(&p2, "new-a", "fixture:old-a", "unresolved-a").is_err());
    assert!(families::binding(&p2, "new-a", "fixture:new-a", "wrong-family").is_err());
    assert_eq!(r2.families.len(), 5);
}

#[test]
fn cross_partition_aliases_are_quarantined_without_reassignment() {
    let f = Fixture::new();
    let (p0, r0) = f.freeze("base", &[], &[], None);
    let (p1, r1) = f.freeze(
        "conflict",
        &["new-a"],
        &["fixture:old-a", "fixture:old-c"],
        Some(&p0),
    );
    for key in ["family-a", "family-c"] {
        assert_eq!(r0.families[key].partition, r1.families[key].partition);
        assert!(!r1.families[key].holds.is_empty());
    }
    assert!(families::binding(&p1, "old-a", "fixture:old-a", "family-a").is_err());
    assert!(families::binding(&p1, "new-a", "fixture:new-a", "unresolved-a").is_err());
    assert!(families::binding(&p1, "old-b", "fixture:old-b", "family-b").is_ok());
}

#[test]
fn output_cannot_be_overwritten_and_prepared_partition_mismatch_fails() {
    let f = Fixture::new();
    let (p0, _) = f.freeze("base", &[], &[], None);
    let before = file_hash(&p0).unwrap();
    let (review, decisions) = f.review("empty", &[], &[]);
    assert!(families::freeze(&f.audit, &[], &review, &decisions, None, &p0).is_err());
    assert_eq!(before, file_hash(&p0).unwrap());
    let p = f.dir.path().join("prepared.json");
    write_json(&p,&json!({"dataset":"old-a","family":"family-a","meta_partition":"meta_test","clean_evaluation":true,"storage":{}})).unwrap();
    let output = f.dir.path().join("mismatch.json");
    assert!(families::freeze(&f.audit, &[p], &review, &decisions, None, &output).is_err());
    assert!(!output.exists());
}

#[test]
fn changed_evidence_or_qualification_promotion_is_rejected() {
    let f = Fixture::new();
    let (p0, mut r0) = f.freeze("base", &[], &[], None);
    r0.qualification = false;
    assert!(r0.validate().is_err());
    let (review, decisions) = f.review("new", &["new-a"], &[]);
    fs::write(&f.evidence.path, "changed").unwrap();
    assert!(families::freeze(
        &f.audit,
        &[],
        &review,
        &decisions,
        Some(&p0),
        &f.dir.path().join("bad.json")
    )
    .is_err());
}

#[test]
fn bucket_specification_is_fixed_and_prepared_quotas_count_families() {
    let family = "example-family";
    let (digest, bucket, partition) = families::stable_assignment(family);
    assert_eq!(digest, hash(format!("20260907:family:{family}").as_bytes()));
    let raw = hex::decode(&digest).unwrap();
    assert_eq!(
        bucket,
        u64::from_be_bytes(raw[..8].try_into().unwrap()) % 1_000_000
    );
    assert_eq!(
        partition,
        if bucket < 800_000 {
            "meta_train"
        } else if bucket < 900_000 {
            "meta_validation"
        } else {
            "meta_test"
        }
    );
    let f = Fixture::new();
    let (review, decisions) = f.review("empty", &[], &[]);
    let mut prepared = vec![];
    for i in 0..2 {
        let p = f.dir.path().join(format!("prepared-{i}.json"));
        write_json(&p,&json!({"dataset":"old-a","family":"family-a","meta_partition":"meta_train","clean_evaluation":true,"storage":{},"view":i})).unwrap();
        prepared.push(p);
    }
    let r = families::freeze(
        &f.audit,
        &prepared,
        &review,
        &decisions,
        None,
        &f.dir.path().join("registry.json"),
    )
    .unwrap();
    assert_eq!(
        r.quota_report()["families_with_verified_preparation_candidates"]["meta_train"],
        1
    );
    assert_eq!(
        r.quota_report()["remaining_families_by_partition"]["meta_train"],
        47
    );
}

#[test]
fn canonical_metadata_wins_over_earlier_acquisition_stubs() {
    let f = Fixture::new();
    let mut audit: AuditReport = read_json(&f.audit).unwrap();
    let original = audit.datasets[0].clone();
    let canonical = f.dir.path().join("regression/old-a");
    fs::create_dir_all(&canonical).unwrap();
    let mut stub = original.clone();
    stub.source_dir = f.dir.path().join("acquisition/quarantine/old-a");
    stub.metadata_sha256 = hash(b"obsolete metadata");
    audit.datasets[0].source_dir = canonical;
    audit.datasets.insert(0, stub);
    write_json(&f.audit, &audit).unwrap();
    let (p, registry) = f.freeze("canonical", &[], &[], None);
    assert_eq!(
        registry.families["family-a"].datasets[0].metadata_sha256,
        original.metadata_sha256
    );
    assert_eq!(
        families::binding(&p, "old-a", "fixture:old-a", "family-a")
            .unwrap()
            .metadata_sha256,
        original.metadata_sha256
    );
}

/// The engine-side growth fixture: a `GrowthFixture` over the frozen registry at `parent_path`
/// whose new component is renamed to the first `development-<i>` id the component bucket rule
/// sends to `development_validation`, then assigned by the policy and written to
/// `growth-assignment.json` beside it. References are absolute, as the Python fixture wrote them.
fn growth_fixture(f: &Fixture, parent_path: &Path) -> (PathBuf, PathBuf, String) {
    let root = f.dir.path().canonicalize().unwrap();
    let parent: Value = serde_json::from_slice(&fs::read(parent_path).unwrap()).unwrap();
    let mut g = GrowthFixture::new(&root, Some(parent), true);
    let cid = (0..1000)
        .map(|i| format!("development-{i}"))
        .find(|c| family_growth::stable_assignment_v2(c).2 == "development_validation")
        .unwrap();
    let last = g.graph["components"]
        .as_array_mut()
        .unwrap()
        .last_mut()
        .unwrap();
    last["component_id"] = json!(cid);
    last["aliases"] = json!([cid, "new-alias"]);
    g.graph_ref = g.w.put("graph.json", &g.graph.clone());
    g.decision["component_id"] = json!(cid);
    g.decision["graph_sha256"] = g.graph_ref["sha256"].clone();
    g.sign();
    let assignment = assign(&root, &g.inputs(&[cid.as_str(), "new-alias"])).unwrap();
    g.w.put("growth-assignment.json", &assignment);
    (
        root.join("parent.json"),
        root.join("growth-assignment.json"),
        cid,
    )
}

#[test]
fn authenticated_component_growth_preserves_panels_and_is_membership_only() {
    let f = Fixture::new();
    let (original, before) = f.freeze("growth-base", &[], &[], None);
    let (parent, assignment, component) = growth_fixture(&f, &original);
    let output = f.dir.path().join("grown.json");
    let grown = families::freeze_growth(&parent, &assignment, &output).unwrap();
    for (family, prior) in &before.families {
        assert_eq!(
            serde_json::to_value(&grown.families[family]).unwrap(),
            serde_json::to_value(prior).unwrap()
        );
    }
    let new = &grown.families[&component];
    assert_eq!(new.partition, "development_validation");
    assert_eq!(new.assignment_method, "stable_sha256_component_v2");
    assert!(new.datasets.is_empty());
    assert_eq!(
        grown.quota_report()["remaining_families_by_partition"],
        before.quota_report()["remaining_families_by_partition"]
    );
    assert!(families::binding(&output, "new", "new-alias", &component).is_err());
    let mut forged = grown.clone();
    forged.families.get_mut(&component).unwrap().partition = "meta_test".into();
    assert!(forged.validate().is_err());
    let mut forged = grown.clone();
    forged.families.get_mut("family-c").unwrap().partition = "meta_train".into();
    assert!(forged.validate().is_err());
    let mut forged = grown.clone();
    forged.growth.clear();
    assert!(forged.validate().is_err());
    let mut forged = grown.clone();
    let dataset = before.families["family-a"].datasets[0].clone();
    forged
        .families
        .get_mut(&component)
        .unwrap()
        .datasets
        .push(dataset);
    assert!(forged.validate().is_err());
}

#[test]
fn component_growth_refuses_forged_summaries_changed_helper_and_stale_parent() {
    let f = Fixture::new();
    let (original, _) = f.freeze("growth-base", &[], &[], None);
    let (parent, assignment, _) = growth_fixture(&f, &original);
    let genuine = fs::read(&assignment).unwrap();
    let mut value: Value = read_json(&assignment).unwrap();
    value["n_test"] = json!(1);
    write_json(&assignment, &value).unwrap();
    assert!(
        families::freeze_growth(&parent, &assignment, &f.dir.path().join("forged.json")).is_err()
    );
    // The verifier is compiled into the release. A helper planted where the retired Python
    // verifier used to be looked up, printing the forged summary, is never consulted: it can
    // neither authorize the forgery nor block the genuine receipt.
    fs::create_dir_all(f.dir.path().join("tools")).unwrap();
    fs::write(
        f.dir.path().join("tools/family_growth_policy.py"),
        format!("print({:?})", serde_json::to_string(&value).unwrap()),
    )
    .unwrap();
    assert!(families::freeze_growth(
        &parent,
        &assignment,
        &f.dir.path().join("helper-changed.json")
    )
    .is_err());
    assert!(!f.dir.path().join("helper-changed.json").exists());
    fs::write(&assignment, &genuine).unwrap();
    assert!(families::freeze_growth(
        &parent,
        &assignment,
        &f.dir.path().join("helper-ignored.json")
    )
    .is_ok());
    let mut stale: Value = read_json(&parent).unwrap();
    stale["registry_id"] = json!("changed");
    let stale_path = f.dir.path().join("stale-parent.json");
    write_json(&stale_path, &stale).unwrap();
    assert!(
        families::freeze_growth(&stale_path, &assignment, &f.dir.path().join("stale.json"))
            .is_err()
    );
}

// ---- the grown-family policy against the retired Python implementation ----------------------
//
// `family_growth` is the native port of `crates/jope-engine/policy/family_growth_policy.py`
// (archived at 8a55c4fef; restore with `git checkout 8a55c4fef -- <path>`). Before that file
// was deleted, `capture_growth_policy_outputs_from_python` ran it (`python3 -I -B <policy>
// --verify <request>`, exactly as `families::freeze_growth` used to) on every scenario below and
// recorded its outcome in `tests/fixtures/family-growth/python-verify-outputs.jsonl`. The
// scenarios are the cases of the Python module's own test file (`existing/*`) and a fixed-seed
// random set (`random/*`). `growth_policy_matches_the_captured_python_outputs` replays them
// against the native verifier, so the parity outlives the Python.

const TEST_KEY: &[u8] = b"test-only-key-never-production!!!";
const PARITY_FIXTURE: &str = "tests/fixtures/family-growth/python-verify-outputs.jsonl";
const RANDOM_SEED: u64 = 20_260_926;
const RANDOM_SCENARIOS: usize = 128;
const EXISTING: [&str; 14] = [
    "live-and-sealed-aliases",
    "alias-invariance-canonical",
    "alias-invariance-alias",
    "forged-signature",
    "duplicate-decision",
    "unresolved-component",
    "stale-parent",
    "missing-units",
    "empty-prior-evidence",
    "empty-source-objects",
    "graph-merge-quarantine",
    "forged-partition",
    "tampered-source",
    "registry-parent",
];

/// Python's `json.dumps(value, sort_keys=True, separators=(",", ":"))` (with `ensure_ascii`)
/// for the values these fixtures hold. They hold no floats.
fn py_compact(value: &Value) -> String {
    fn string(s: &str, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                ' '..='~' => out.push(c),
                _ => {
                    for unit in c.encode_utf16(&mut [0u16; 2]) {
                        write!(out, "\\u{unit:04x}").unwrap();
                    }
                }
            }
        }
        out.push('"');
    }
    fn walk(value: &Value, out: &mut String) {
        match value {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => {
                assert!(
                    n.is_i64() || n.is_u64(),
                    "fixture values hold no floats: {n}"
                );
                out.push_str(&n.to_string());
            }
            Value::String(s) => string(s, out),
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    walk(item, out);
                }
                out.push(']');
            }
            Value::Object(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    string(k, out);
                    out.push(':');
                    walk(v, out);
                }
                out.push('}');
            }
        }
    }
    let mut out = String::new();
    walk(value, &mut out);
    out
}

/// Files of one scenario root, referenced relative to it or (as the Python fixture did)
/// absolutely.
struct World {
    root: PathBuf,
    absolute: bool,
}

impl World {
    fn reference(&self, rel: &str) -> String {
        if self.absolute {
            self.root.join(rel).to_str().unwrap().to_owned()
        } else {
            rel.to_owned()
        }
    }
    fn put_bytes(&self, rel: &str, bytes: &[u8]) -> Value {
        let path = self.root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        json!({"path": self.reference(rel), "sha256": hash(bytes)})
    }
    /// `GrowthFixture.put`: compact sorted JSON plus a newline.
    fn put(&self, rel: &str, value: &Value) -> Value {
        self.put_bytes(rel, format!("{}\n", py_compact(value)).as_bytes())
    }
}

fn sign(key: &[u8], reference: &mut Value) {
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
struct GrowthFixture {
    w: World,
    parent: Value,
    graph: Value,
    graph_ref: Value,
    decision: Value,
    decisions: Vec<Value>,
}

impl GrowthFixture {
    fn new(root: &Path, parent: Option<Value>, absolute: bool) -> Self {
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
    fn sign(&mut self) {
        let mut reference = self.w.put("decision.json", &self.decision.clone());
        sign(TEST_KEY, &mut reference);
        self.decisions = vec![reference];
    }
    fn inputs(&self, names: &[&str]) -> Value {
        json!({
            "requested_families": names, "additional_live_families": [],
            "additional_sealed_families": [], "graph": self.graph_ref,
            "parent_registry": self.parent, "decisions": self.decisions,
        })
    }
}

fn assign(root: &Path, inputs: &Value) -> Result<Value, GrowthError> {
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
enum Tamper {
    None,
    RowPartition,
    NTest,
    ExtraKey,
    DropKey(&'static str),
    Schema,
}

impl Tamper {
    fn apply(self, request: &mut Value) {
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
fn existing_case(case: &str, root: &Path, absolute: bool) -> (Value, Tamper) {
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
fn random_case(index: usize, base: &Path, root: &Path, absolute: bool) -> (Value, Tamper) {
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
        sign(&signer, &mut reference);
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

fn scenario_names() -> Vec<String> {
    EXISTING
        .iter()
        .map(|case| format!("existing/{case}"))
        .chain((0..RANDOM_SCENARIOS).map(|i| format!("random/{i:03}")))
        .collect()
}

/// Build scenario `name` under `base` and return its root and the receipt to verify: the
/// native assignment when the policy produces one, otherwise the bare request (schema, root
/// and inputs), in both cases with the scenario's tampering applied.
fn build_scenario(name: &str, base: &Path, absolute: bool) -> (PathBuf, Value) {
    fs::create_dir_all(base.join("root")).unwrap();
    let root = base.join("root").canonicalize().unwrap();
    let (inputs, tamper) = match name.split_once('/') {
        Some(("existing", case)) => existing_case(case, &root, absolute),
        Some(("random", index)) => random_case(index.parse().unwrap(), base, &root, absolute),
        _ => panic!("unknown scenario {name}"),
    };
    let mut request = assign(&root, &inputs).unwrap_or_else(|_| {
        let mut skeleton = inputs.clone();
        skeleton["schema"] = json!(family_growth::ASSIGNMENT_SCHEMA);
        skeleton["root"] = json!(root.to_str().unwrap());
        skeleton
    });
    tamper.apply(&mut request);
    (root, request)
}

/// SHA-256 over every file and link under `base` (relative path, then bytes or link target)
/// and the root-independent receipt, so a changed scenario generator is reported as such.
fn scenario_digest(base: &Path, root: &Path, request: &Value) -> String {
    fn walk(dir: &Path, base: &Path, out: &mut Vec<u8>) {
        let mut entries: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        entries.sort();
        for path in entries {
            let meta = fs::symlink_metadata(&path).unwrap();
            out.extend(
                path.strip_prefix(base)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .as_bytes(),
            );
            out.push(0);
            if meta.file_type().is_symlink() {
                out.extend(b"link:");
                out.extend(fs::read_link(&path).unwrap().to_str().unwrap().as_bytes());
            } else if meta.is_dir() {
                walk(&path, base, out);
                continue;
            } else {
                out.extend(fs::read(&path).unwrap());
            }
            out.push(0);
        }
    }
    let mut bytes = vec![];
    walk(base, base, &mut bytes);
    bytes.extend(
        py_compact(request)
            .replace(root.to_str().unwrap(), "@ROOT@")
            .as_bytes(),
    );
    hash(&bytes)
}

/// Whether the native outcome equals a captured Python outcome.
fn agrees(outcome: &Result<Value, GrowthError>, root: &Path, record: &Value) -> Result<(), String> {
    let root = root.to_str().unwrap();
    match (outcome, record["exit"].as_i64().unwrap()) {
        (Ok(value), 0) => {
            let native = py_compact(value).replace(root, "@ROOT@");
            (native == record["stdout"].as_str().unwrap())
                .then_some(())
                .ok_or_else(|| {
                    format!(
                        "output differs:\n native {native}\n python {}",
                        record["stdout"]
                    )
                })
        }
        (Err(GrowthError::Refused(message)), code) if code != 0 => {
            (record["error_type"] == "ValueError" && record["error"] == message.as_str())
                .then_some(())
                .ok_or_else(|| {
                    format!(
                        "native refused {message:?}; python {} {}",
                        record["error_type"], record["error"]
                    )
                })
        }
        (Err(GrowthError::Malformed(message)), code) if code != 0 => (record["error_type"]
            != "ValueError")
            .then_some(())
            .ok_or_else(|| {
                format!(
                    "native malformed {message:?}; python ValueError {}",
                    record["error"]
                )
            }),
        (outcome, code) => Err(format!(
            "native {outcome:?}; python exit {code} {} {}",
            record["error_type"], record["error"]
        )),
    }
}

/// Differential proof, replayed: the native verifier reproduces the outcome the Python policy
/// produced on every captured scenario, byte for byte on success and with the same refusal
/// message otherwise.
#[test]
fn growth_policy_matches_the_captured_python_outputs() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(PARITY_FIXTURE);
    let records: Vec<Value> = fs::read_to_string(&fixture)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let names = scenario_names();
    assert_eq!(
        records
            .iter()
            .map(|r| r["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        names,
        "the captured scenario list differs from this test's"
    );
    let (mut accepted, mut refused) = (0, 0);
    for (name, record) in names.iter().zip(&records) {
        let dir = tempfile::tempdir().unwrap();
        let (root, request) = build_scenario(name, dir.path(), false);
        assert_eq!(
            scenario_digest(dir.path(), &root, &request),
            record["inputs_sha256"].as_str().unwrap(),
            "{name}: the scenario inputs differ from the captured ones"
        );
        let outcome = family_growth::verify_assignment(&request);
        if let Err(why) = agrees(&outcome, &root, record) {
            panic!("{name}: {why}");
        }
        match outcome {
            Ok(_) => accepted += 1,
            Err(GrowthError::Refused(_)) => refused += 1,
            Err(_) => {}
        }
    }
    // The set is not vacuous: it grows families and refuses forgeries.
    assert!(
        accepted >= 30 && refused >= 30,
        "accepted {accepted}, refused {refused}"
    );
}

// ---- custody that cannot be resolved (review round five of PR #117) --------------------------
//
// `escapes()` took a `canonicalize()` error for "does not escape": a policy or reviewer-key
// directory linked out of the workspace to a directory so deep that the resolved name is longer
// than `PATH_MAX` could not be resolved, yet opened through the root's short path, so the native
// check passed and read custody from outside the root. The retired Python refused it
// (`pinned provenance policy escaped workspace custody`): its non-strict `resolve()` built the
// long name anyway and found it outside the root. `tests/fixtures/family-growth/
// python-custody-r5.jsonl` holds the Python's outcome on each case below (captured by
// `capture_growth_custody_outputs_from_python`, run `p1-sweep5-capture-*`).

const CUSTODY_FIXTURE: &str = "tests/fixtures/family-growth/python-custody-r5.jsonl";
const CUSTODY_CASES: [&str; 7] = [
    "custody-control",
    "policy-link-out-too-long",
    "policy-link-out",
    "key-link-out-too-long",
    "key-link-out",
    "policy-directory-unsearchable",
    "key-directory-unsearchable",
];

/// A directory under `base` whose absolute name is `len` bytes long (components of 200 bytes).
fn deep_directory(base: &Path, len: usize) -> PathBuf {
    let mut dir = base.join("outside");
    while dir.as_os_str().len() + 201 < len {
        dir = dir.join("d".repeat(200));
    }
    let pad = len - dir.as_os_str().len() - 1;
    dir.join("e".repeat(pad))
}

/// Build custody case `name` under `base`: a grown-family assignment produced while custody was
/// intact, then the policy (`manifests`) or reviewer-key (`credentials/provenance-review`)
/// directory moved out of the root behind a symlink (deep enough that the resolved names exceed
/// `PATH_MAX`, or not), or made unsearchable. Returns the root, the assignment and the directory
/// whose mode must be restored after the run.
fn custody_case(name: &str, base: &Path) -> (PathBuf, Value, Option<PathBuf>) {
    use std::os::unix::fs::PermissionsExt;
    fs::create_dir_all(base.join("root")).unwrap();
    let root = base.join("root").canonicalize().unwrap();
    let base = base.canonicalize().unwrap();
    let fixture = GrowthFixture::new(&root, None, false);
    let assignment = assign(&root, &fixture.inputs(&["new"])).unwrap();
    let (dir, how) = match name.split_once('-') {
        Some(("custody", _)) => return (root, assignment, None),
        Some(("policy", how)) => ("manifests", how),
        Some(("key", how)) => (family_growth::REVIEWER_KEYS, how),
        _ => panic!("unknown custody case {name}"),
    };
    let inside = root.join(dir);
    match how {
        "link-out-too-long" | "link-out" => {
            let outside = if how == "link-out" {
                base.join("outside-short")
            } else {
                deep_directory(&base, 4090)
            };
            fs::create_dir_all(outside.parent().unwrap()).unwrap();
            fs::rename(&inside, &outside).unwrap();
            std::os::unix::fs::symlink(&outside, &inside).unwrap();
            (root, assignment, None)
        }
        "directory-unsearchable" => {
            fs::set_permissions(&inside, fs::Permissions::from_mode(0o000)).unwrap();
            (root, assignment, Some(inside))
        }
        _ => panic!("unknown custody case {name}"),
    }
}

fn restore(dir: Option<PathBuf>) {
    use std::os::unix::fs::PermissionsExt;
    if let Some(dir) = dir {
        fs::set_permissions(dir, fs::Permissions::from_mode(0o755)).unwrap();
    }
}

/// Custody the native check cannot resolve is refused, never taken as custody kept: each case
/// agrees with the retired Python's captured outcome (the escape refusals word for word; a
/// directory that may not be searched as a malformed-input error, where the Python raised
/// `PermissionError`). A long link out of the workspace was accepted before this round.
#[test]
fn custody_that_cannot_be_resolved_is_refused_as_python_refused_it() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(CUSTODY_FIXTURE);
    let records: Vec<Value> = fs::read_to_string(&fixture)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        records
            .iter()
            .map(|r| r["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        CUSTODY_CASES,
        "every captured custody case is replayed"
    );
    let mut refused = 0;
    for (name, record) in CUSTODY_CASES.iter().zip(&records) {
        let dir = tempfile::tempdir().unwrap();
        let (root, request, locked) = custody_case(name, dir.path());
        // A user who may search anything (root) cannot reproduce an unsearchable directory.
        let privileged = locked.as_ref().is_some_and(|d| fs::read_dir(d).is_ok());
        let outcome = family_growth::verify_assignment(&request);
        restore(locked);
        if privileged {
            eprintln!("{name}: skipped, this user may search a directory of mode 000");
            continue;
        }
        if let Err(why) = agrees(&outcome, &root, record) {
            panic!("{name}: {why}");
        }
        if outcome.is_err() {
            refused += 1;
        }
    }
    assert!(refused >= 4, "refused {refused}");
    // The long link resolves through the root's short path: it is the check, not the read, that
    // refuses it.
    let dir = tempfile::tempdir().unwrap();
    let (root, request, _) = custody_case("policy-link-out-too-long", dir.path());
    assert!(fs::read(root.join(family_growth::POLICY_PATH)).is_ok());
    assert!(root
        .join(family_growth::POLICY_PATH)
        .canonicalize()
        .is_err());
    assert_eq!(
        family_growth::verify_assignment(&request),
        Err(GrowthError::Refused(
            "pinned provenance policy escaped workspace custody".into()
        ))
    );
}

/// Records the retired Python policy's outcome on the custody cases (`python3 -I -B <policy>
/// --verify <request>`) as `<JOPE_GROWTH_CAPTURE_DIR>/custody-r5.jsonl`, the layout of
/// `python-custody-r5.jsonl`, and checks the native verifier agrees with each.
#[test]
#[ignore = "capture: needs JOPE_GROWTH_POLICY_PY (the archived Python policy) and JOPE_GROWTH_CAPTURE_DIR"]
fn capture_growth_custody_outputs_from_python() {
    let policy = PathBuf::from(std::env::var("JOPE_GROWTH_POLICY_PY").unwrap());
    let out = PathBuf::from(std::env::var("JOPE_GROWTH_CAPTURE_DIR").unwrap());
    let (mut lines, mut disagreements) = (vec![], vec![]);
    for name in CUSTODY_CASES {
        let base = out.join("custody").join(name);
        let (root, request, locked) = custody_case(name, &base);
        let request_path = base.join("request.json");
        fs::write(&request_path, format!("{}\n", py_compact(&request))).unwrap();
        let output = std::process::Command::new("/usr/bin/python3")
            .args(["-I", "-B"])
            .arg(&policy)
            .arg("--verify")
            .arg(&request_path)
            .output()
            .unwrap();
        let native = family_growth::verify_assignment(&request);
        restore(locked);
        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        fs::write(base.join("python.stdout"), &stdout).unwrap();
        fs::write(base.join("python.stderr"), &stderr).unwrap();
        let code = output.status.code().unwrap();
        let last = stderr
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        let (error_type, error) = if code == 0 {
            ("", "")
        } else {
            last.split_once(": ").unwrap_or((last, ""))
        };
        let root_text = root.to_str().unwrap();
        let record = json!({
            "name": name, "exit": code,
            "stdout": stdout.trim_end_matches('\n').replace(root_text, "@ROOT@"),
            "error_type": error_type,
            "error": error.replace(root_text, "@ROOT@"),
        });
        if let Err(why) = agrees(&native, &root, &record) {
            disagreements.push(json!({"name": name, "why": why}));
        }
        lines.push(py_compact(&record));
    }
    fs::write(out.join("custody-r5.jsonl"), lines.join("\n") + "\n").unwrap();
    fs::write(
        out.join("custody-r5-disagreements.json"),
        serde_json::to_vec_pretty(&disagreements).unwrap(),
    )
    .unwrap();
    assert!(disagreements.is_empty(), "{disagreements:?}");
}

/// Regenerates `tests/fixtures/family-growth/python-verify-outputs.jsonl` from the archived
/// Python policy and checks both reference forms (relative, the committed one, and absolute,
/// as the Python fixture wrote them). Restore the policy with
/// `git show 8a55c4fef:crates/jope-engine/policy/family_growth_policy.py > <file>`.
#[test]
#[ignore = "capture: needs JOPE_GROWTH_POLICY_PY (the archived Python policy) and JOPE_GROWTH_CAPTURE_DIR"]
fn capture_growth_policy_outputs_from_python() {
    let policy = PathBuf::from(std::env::var("JOPE_GROWTH_POLICY_PY").unwrap());
    let out = PathBuf::from(std::env::var("JOPE_GROWTH_CAPTURE_DIR").unwrap());
    let mut report = Map::new();
    for mode in ["relative", "absolute"] {
        let (mut lines, mut disagreements) = (vec![], vec![]);
        let mut outcomes: BTreeMap<String, usize> = BTreeMap::new();
        for name in scenario_names() {
            let base = out.join(mode).join(name.replace('/', "-"));
            let (root, request) = build_scenario(&name, &base, mode == "absolute");
            // Digest the scenario before the request and the Python outputs join it.
            let digest = scenario_digest(&base, &root, &request);
            let request_path = base.join("request.json");
            fs::write(&request_path, format!("{}\n", py_compact(&request))).unwrap();
            let output = std::process::Command::new("/usr/bin/python3")
                .args(["-I", "-B"])
                .arg(&policy)
                .arg("--verify")
                .arg(&request_path)
                .output()
                .unwrap();
            let stdout = String::from_utf8(output.stdout).unwrap();
            let stderr = String::from_utf8(output.stderr).unwrap();
            fs::write(base.join("python.stdout"), &stdout).unwrap();
            fs::write(base.join("python.stderr"), &stderr).unwrap();
            let code = output.status.code().unwrap();
            let last = stderr
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("");
            let (error_type, error) = if code == 0 {
                ("", "")
            } else {
                last.split_once(": ").unwrap_or((last, ""))
            };
            let record = json!({
                "name": name, "inputs_sha256": digest,
                "exit": code,
                "stdout": stdout.trim_end_matches('\n').replace(root.to_str().unwrap(), "@ROOT@"),
                "error_type": error_type, "error": error,
            });
            *outcomes
                .entry(if code == 0 {
                    "accepted".into()
                } else {
                    error_type.to_owned()
                })
                .or_default() += 1;
            let native = family_growth::verify_assignment(&request);
            if let Err(why) = agrees(&native, &root, &record) {
                disagreements.push(json!({"name": name, "why": why}));
            }
            lines.push(py_compact(&record));
        }
        fs::write(out.join(format!("{mode}.jsonl")), lines.join("\n") + "\n").unwrap();
        report.insert(
            mode.into(),
            json!({"scenarios": lines.len(), "outcomes": outcomes, "disagreements": disagreements}),
        );
    }
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    for mode in ["relative", "absolute"] {
        assert!(
            report[mode]["disagreements"].as_array().unwrap().is_empty(),
            "{mode}: {}",
            report[mode]["disagreements"]
        );
    }
}
