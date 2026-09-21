use jope_parallel::{
    families::{self, Registry},
    manifest::{
        file_hash, hash, read_json, write_json, Admission, AuditReport, Dataset, Decision,
        Evidence, SEED,
    },
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    ffi::OsString,
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

/// The policy helper `src/families.rs` hashes (`include_bytes!("../policy/family_growth_policy.py")`)
/// and copies into each fixture's `tools/` directory.
fn policy_helper() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("policy/family_growth_policy.py")
}

/// ADR-0003 split the two modules the fixture script imports across two trees: the policy helper
/// stays crate-local because the crate embeds it, while its `GrowthFixture` test double moved into
/// the uv package. Both directories have to be on the path or the script cannot import either.
fn growth_pythonpath() -> OsString {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut path = crate_dir.join("policy").into_os_string();
    path.push(":");
    path.push(crate_dir.join("../../python/src/jope/tools"));
    path
}

fn growth_fixture(f: &Fixture, parent_path: &Path) -> (PathBuf, PathBuf, String) {
    let root = f.dir.path();
    fs::create_dir_all(root.join("tools")).unwrap();
    fs::copy(policy_helper(), root.join("tools/family_growth_policy.py")).unwrap();
    let script = r#"
import json,sys
from pathlib import Path
from test_family_growth_policy import GrowthFixture
import family_growth_policy as g
root,parent=map(Path,sys.argv[1:])
f=GrowthFixture(root,json.loads(parent.read_bytes()))
cid=next('development-'+str(i) for i in range(1000) if g.stable_assignment_v2('development-'+str(i))[2]=='development_validation')
f.graph['components'][-1]['component_id']=cid
f.graph['components'][-1]['aliases']=[cid,'new-alias']
f.graph_ref=f.put('graph.json',f.graph)
f.decision['component_id']=cid
f.decision['graph_sha256']=f.graph_ref['sha256']
f.sign()
f.put('growth-assignment.json',f.assign([cid,'new-alias']))
print(cid)
"#;
    let output = std::process::Command::new("/usr/bin/python3")
        .args(["-B", "-c", script])
        .arg(root)
        .arg(parent_path)
        .env("PYTHONPATH", growth_pythonpath())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    (
        root.join("parent.json"),
        root.join("growth-assignment.json"),
        String::from_utf8(output.stdout).unwrap().trim().to_owned(),
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
    let mut value: Value = read_json(&assignment).unwrap();
    value["n_test"] = json!(1);
    write_json(&assignment, &value).unwrap();
    assert!(
        families::freeze_growth(&parent, &assignment, &f.dir.path().join("forged.json")).is_err()
    );
    value["n_test"] = json!(0);
    write_json(&assignment, &value).unwrap();
    fs::write(
        f.dir.path().join("tools/family_growth_policy.py"),
        "print('forged')",
    )
    .unwrap();
    assert!(families::freeze_growth(
        &parent,
        &assignment,
        &f.dir.path().join("helper-changed.json")
    )
    .is_err());
    fs::copy(
        policy_helper(),
        f.dir.path().join("tools/family_growth_policy.py"),
    )
    .unwrap();
    let mut stale: Value = read_json(&parent).unwrap();
    stale["registry_id"] = json!("changed");
    let stale_path = f.dir.path().join("stale-parent.json");
    write_json(&stale_path, &stale).unwrap();
    assert!(
        families::freeze_growth(&stale_path, &assignment, &f.dir.path().join("stale.json"))
            .is_err()
    );
}
