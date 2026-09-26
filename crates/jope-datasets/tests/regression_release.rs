//! W1-T02 acceptance: the ten-regression release binds exact bytes and
//! interpretation, reopens byte-identically, and every changed source, target,
//! row, schema, group, fold or family mapping yields a new identity or a refusal.

use jope_datasets::hash::sha256_hex;
use jope_datasets::regression_release::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

const PROTOCOL: &str = "fixture-grouped-development-v1";
const SEED: u64 = 7;
const PREPARATION_SOURCE: &[u8] = b"# fixture preparation implementation\n";

#[derive(Clone)]
struct RowData {
    id: String,
    group: String,
    fold: Option<u64>,
    values: Vec<Option<String>>,
    target: String,
}

#[derive(Clone)]
struct MemberData {
    id: String,
    features: Value,
    group_columns: Vec<String>,
    raw: Vec<u8>,
    metadata: Vec<u8>,
    rows: Vec<RowData>,
}

fn hex(label: &str) -> String {
    sha256_hex(label.as_bytes())
}

fn member_data(id: &str) -> MemberData {
    let rows = (0..12u64)
        .map(|i| RowData {
            id: hex(&format!("{id}:row:{i}")),
            group: hex(&format!("{id}:group:{i}")),
            fold: if i < 10 { Some(i % 5) } else { None },
            values: vec![
                Some(format!("{}.5", i * 3)),
                if i % 4 == 0 {
                    None
                } else {
                    Some(["red", "NA", ""][(i % 3) as usize].to_owned())
                },
            ],
            target: format!("{}", 100 + i * 7),
        })
        .collect();
    MemberData {
        id: id.to_owned(),
        features: json!([{"name":"x","kind":"numeric"},{"name":"c","kind":"categorical"}]),
        group_columns: vec![],
        raw: format!("raw-object:{id}").into_bytes(),
        metadata: format!("metadata:{id}").into_bytes(),
        rows,
    }
}

fn canonical(value: &Value) -> Vec<u8> {
    canonical_json(value).unwrap()
}

/// Staged objects and the matching frozen panel entry for one member.
fn materialize(m: &MemberData) -> (StagedMember, Value) {
    let source = sha256_hex(&m.raw);
    let row = |r: &RowData, index: usize| {
        json!({"id":r.id,"source_index":index,"group":r.group,"fold":r.fold,
               "values":r.values,"target":r.target})
    };
    let common = |partition: &str, rows: Vec<Value>| {
        json!({"schema":"jope.prepared-table/v1","dataset":m.id,"task":"regression",
               "source_sha256":source,"features":m.features,"classes":[],
               "partition":partition,"rows":rows})
    };
    let development = canonical(&common(
        "development",
        m.rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.fold.is_some())
            .map(|(i, r)| row(r, i))
            .collect(),
    ));
    let sealed = canonical(&common(
        "sealed_final",
        m.rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.fold.is_none())
            .map(|(i, r)| row(r, i))
            .collect(),
    ));
    let split = canonical(
        &json!({"protocol":PROTOCOL,"seed":SEED,"source_sha256":source,
        "rows":m.rows.iter().enumerate().map(|(i, r)| json!({"id":r.id,"source_index":i,"group":r.group,"fold":r.fold})).collect::<Vec<_>>()}),
    );
    let mut fold_rows = [0u64; 5];
    for r in &m.rows {
        if let Some(f) = r.fold {
            fold_rows[f as usize] += 1;
        }
    }
    let development_rows: u64 = fold_rows.iter().sum();
    let entry = json!({
        "id": m.id, "task": "regression", "metric": "rmse", "rows": m.rows.len(),
        "features": m.features, "classes": [], "group_columns": m.group_columns,
        "source": {"url": format!("https://example.test/{}.tsv.gz", m.id), "sha256": source,
                   "bytes": m.raw.len(), "metadata_url": format!("https://example.test/{}.yaml", m.id),
                   "metadata_sha256": sha256_hex(&m.metadata)},
        "split": {"protocol": PROTOCOL, "sha256": sha256_hex(&split),
                  "development_sha256": sha256_hex(&development), "sealed_sha256": sha256_hex(&sealed),
                  "development_rows": development_rows,
                  "sealed_rows": m.rows.len() as u64 - development_rows, "fold_rows": fold_rows},
        "license": {"original": "synthetic"}, "checkpoint_exposure": "unknown",
        "description": "synthetic fixture", "group_limitations": "synthetic"
    });
    (
        StagedMember {
            raw: m.raw.clone(),
            metadata: m.metadata.clone(),
            development,
            split,
            sealed,
        },
        entry,
    )
}

fn classification_entry(id: &str, task: &str) -> Value {
    json!({
        "id": id, "task": task, "metric": if task == "binary" {"auroc"} else {"log_loss"},
        "rows": 20, "features": [{"name":"x","kind":"numeric"}], "classes": ["0","1"],
        "group_columns": [],
        "source": {"url": "https://example.test/c", "sha256": hex(id), "bytes": 1,
                   "metadata_url": "https://example.test/m", "metadata_sha256": hex(&format!("{id}:m"))},
        "split": {"protocol": PROTOCOL, "sha256": hex(&format!("{id}:s")),
                  "development_sha256": hex(&format!("{id}:d")), "sealed_sha256": hex(&format!("{id}:x")),
                  "development_rows": 16, "sealed_rows": 4, "fold_rows": [4,3,3,3,3]},
        "license": {"original": "synthetic"}, "checkpoint_exposure": "unknown",
        "description": "synthetic fixture", "group_limitations": "synthetic"
    })
}

fn unknown(basis: &str) -> Exposure {
    Exposure {
        state: ExposureState::Unknown,
        basis: basis.into(),
        evidence: vec![],
    }
}

fn registry_for(ids: &[String]) -> FamilyRegistry {
    FamilyRegistry {
        schema: REGISTRY_SCHEMA.into(),
        registry_version: 1,
        families: ids
            .iter()
            .map(|id| Family {
                family: format!("family:{id}"),
                members: vec![id.clone()],
                source_keys: vec![format!("origin:{id}")],
                relations: vec![],
            })
            .collect(),
        exposures: ids
            .iter()
            .map(|id| ExposureRecord {
                dataset: id.clone(),
                catalog: CatalogAvailability {
                    state: CatalogState::Unknown,
                    basis: "fixture catalog not inspected".into(),
                    evidence: vec![],
                },
                encoder: unknown("no encoder read logs"),
                teacher: unknown("no teacher corpus audit"),
                checkpoint: unknown("pretrained ancestry untraceable"),
            })
            .collect(),
    }
}

struct Fixture {
    members: Vec<MemberData>,
    panel: Value,
    staged: BTreeMap<String, StagedMember>,
    registry: FamilyRegistry,
    preparation: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        let members: Vec<MemberData> = (0..10).map(|i| member_data(&format!("reg{i}"))).collect();
        let ids: Vec<String> = members.iter().map(|m| m.id.clone()).collect();
        let mut f = Self {
            members,
            panel: Value::Null,
            staged: BTreeMap::new(),
            registry: registry_for(&ids),
            preparation: PREPARATION_SOURCE.to_vec(),
        };
        f.rebuild();
        f
    }

    /// Regression entries interleave with binary ones to prove panel order is kept.
    fn rebuild(&mut self) {
        let mut datasets = Vec::new();
        self.staged.clear();
        for (i, m) in self.members.iter().enumerate() {
            let (staged, entry) = materialize(m);
            self.staged.insert(m.id.clone(), staged);
            datasets.push(entry);
            datasets.push(classification_entry(&format!("bin{i}"), "binary"));
        }
        for i in 0..10 {
            datasets.push(classification_entry(&format!("multi{i}"), "multiclass"));
        }
        self.panel = json!({"schema":"jope.benchmark-panel/v1","panel_id":"fixture-30",
            "repository":"example/pmlb","revision":"0".repeat(40),"seed":SEED,
            "selection":"synthetic","folds":5,"datasets":datasets});
    }

    fn panel_bytes(&self) -> Vec<u8> {
        serde_json::to_vec_pretty(&self.panel).unwrap()
    }

    fn derive(&self) -> anyhow::Result<RegressionRelease> {
        derive(&ReleaseSources {
            panel: &self.panel_bytes(),
            preparation_source_path: "benchmarks/prepare.py",
            preparation_source: &self.preparation,
            registry: &self.registry,
            staged: &self.staged,
        })
    }
}

fn err_text<T: std::fmt::Debug>(result: anyhow::Result<T>) -> String {
    format!("{:#}", result.unwrap_err())
}

#[test]
fn derived_release_reopens_to_identical_bytes_and_identity() {
    let f = Fixture::new();
    let release = f.derive().unwrap();
    let bytes = release.encode().unwrap();
    assert_eq!(bytes.last(), Some(&b'\n'));
    let reopened = RegressionRelease::open(&bytes).unwrap();
    assert_eq!(reopened, release);
    assert_eq!(reopened.encode().unwrap(), bytes);
    assert_eq!(
        f.derive().unwrap().encode().unwrap(),
        bytes,
        "derivation is deterministic"
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("release.json");
    std::fs::write(&path, &bytes).unwrap();
    let stored = std::fs::read(&path).unwrap();
    assert_eq!(
        RegressionRelease::open(&stored).unwrap().encode().unwrap(),
        stored
    );
    assert_eq!(release.release_id, release.compute_id().unwrap());

    let names: Vec<&str> = release.members.iter().map(|m| m.dataset.as_str()).collect();
    assert_eq!(
        names,
        (0..10).map(|i| format!("reg{i}")).collect::<Vec<_>>()
    );
    assert_eq!(
        release
            .members
            .iter()
            .map(|m| m.panel_index)
            .collect::<Vec<_>>(),
        (0..10).map(|i| i * 2).collect::<Vec<u64>>()
    );
    assert_eq!(release.exclusions.len(), 20);
    assert!(release
        .exclusions
        .iter()
        .all(|e| e.reason == format!("task_not_regression:{}", e.task)));
    assert_eq!(
        release.derived_from.task_counts,
        BTreeMap::from([
            ("binary".to_owned(), 10),
            ("multiclass".to_owned(), 10),
            ("regression".to_owned(), 10),
            ("timeseries".to_owned(), 0)
        ])
    );
    assert_eq!(
        release.family_registry_sha256,
        sha256_hex(&f.registry.encode().unwrap())
    );
    for (id, staged) in &f.staged {
        release.verify_staged(id, staged).unwrap();
        release.verify_development(id, &staged.development).unwrap();
    }
    check_against_panel(&release, &f.panel_bytes()).unwrap();
    let m = release.member("reg0").unwrap();
    assert_eq!(m.target.column, TARGET_COLUMN);
    assert_eq!(m.target.development_label_role, DEVELOPMENT_LABEL_ROLE);
    assert_eq!(m.target.sealed_label_role, SEALED_LABEL_ROLE);
    assert_eq!(
        (m.rows.total, m.rows.development, m.rows.sealed),
        (12, 10, 2)
    );
}

type Mutation = (&'static str, fn(&mut MemberData));

fn consistent_mutations() -> Vec<Mutation> {
    vec![
        ("changed_raw_source_bytes", |m| m.raw.push(b'!')),
        ("changed_source_metadata", |m| m.metadata.push(b'!')),
        ("changed_one_target", |m| m.rows[3].target = "999".into()),
        ("swapped_two_targets_row_label_misalignment", |m| {
            let t = m.rows[1].target.clone();
            m.rows[1].target = m.rows[2].target.clone();
            m.rows[2].target = t;
        }),
        ("changed_feature_kind", |m| {
            m.features[0]["kind"] = json!("categorical")
        }),
        ("renamed_feature", |m| {
            m.features[1]["name"] = json!("colour")
        }),
        ("changed_feature_value", |m| {
            m.rows[4].values[0] = Some("0.25".into())
        }),
        ("changed_row_identity", |m| {
            m.rows[5].id = hex("replacement")
        }),
        ("merged_two_groups_within_a_fold", |m| {
            m.rows[5].group = m.rows[0].group.clone()
        }),
        ("reassigned_folds_with_equal_counts", |m| {
            m.rows[0].fold = Some(1);
            m.rows[1].fold = Some(0);
        }),
        ("moved_a_development_row_to_sealed", |m| {
            m.rows[10].fold = Some(0);
            m.rows[0].fold = None;
        }),
        ("reordered_rows", |m| m.rows.swap(0, 1)),
    ]
}

#[test]
fn changed_source_target_rows_schema_groups_folds_or_family_mapping_is_a_new_identity_or_rejected()
{
    let base = Fixture::new();
    let original = base.derive().unwrap();
    let original_bytes = original.encode().unwrap();
    let mut decisions = Vec::new();
    for (label, mutate) in consistent_mutations() {
        // (1) Inputs change but the frozen declarations do not: refusal.
        let mut f = Fixture::new();
        let panel = f.panel.clone();
        mutate(&mut f.members[0]);
        f.rebuild();
        f.panel = panel;
        let refused = err_text(f.derive());
        // (2) The old release never re-interprets the changed bytes.
        let old_check = original.verify_staged("reg0", &f.staged["reg0"]);
        assert!(
            old_check.is_err(),
            "{label}: old release accepted changed bytes"
        );
        // (3) A consistently re-declared input is a different release identity.
        let mut g = Fixture::new();
        mutate(&mut g.members[0]);
        g.rebuild();
        let changed = g.derive().unwrap_or_else(|e| panic!("{label}: {e:#}"));
        assert_ne!(changed.release_id, original.release_id, "{label}");
        assert_ne!(changed.encode().unwrap(), original_bytes, "{label}");
        decisions.push(
            json!({"mutation":label,"stale_declaration":"rejected","stale_reason":refused,
            "old_release_verify":"rejected","redeclared":"new_release_identity",
            "new_release_id":changed.release_id}),
        );
    }

    // A declaration-only change (grouping columns) is a new identity as well.
    let mut f = Fixture::new();
    f.members[0].group_columns = vec!["c".into()];
    f.rebuild();
    assert_ne!(f.derive().unwrap().release_id, original.release_id);
    decisions.push(json!({"mutation":"declared_group_column","decision":"new_release_identity"}));

    // A reordered split (source indices no longer positional) is refused outright.
    let mut staged = base.staged["reg0"].clone();
    let mut split: Value = serde_json::from_slice(&staged.split).unwrap();
    split["rows"].as_array_mut().unwrap().swap(0, 1);
    staged.split = canonical(&split);
    let mut g = Fixture::new();
    g.panel["datasets"][0]["split"]["sha256"] = json!(sha256_hex(&staged.split));
    g.staged.insert("reg0".into(), staged);
    assert!(err_text(g.derive()).contains("not in original source order"));
    decisions.push(json!({"mutation":"reordered_rows","decision":"rejected"}));

    // A family mapping or exposure change is a new release identity.
    for (label, change) in [
        ("added_unresolved_relation", 0),
        ("changed_exposure_basis", 1),
        ("added_family_source_key", 2),
    ] {
        let mut f = Fixture::new();
        match change {
            0 => f.registry.families[0].relations.push(Relation {
                dataset: "reg0".into(),
                other: "external:variant".into(),
                kind: RelationKind::AlternateTarget,
                status: RelationStatus::Unresolved,
                evidence: "name similarity only".into(),
            }),
            1 => f.registry.exposures[0].encoder.basis = "different basis".into(),
            _ => f.registry.families[0]
                .source_keys
                .push("origin:alias".into()),
        }
        let changed = f.derive().unwrap();
        assert_ne!(
            changed.family_registry_sha256,
            original.family_registry_sha256
        );
        assert_ne!(changed.release_id, original.release_id, "{label}");
        decisions.push(json!({"mutation":label,"decision":"new_release_identity"}));
    }

    // Changed preparation implementation is a new identity.
    let mut f = Fixture::new();
    f.preparation.extend_from_slice(b"# changed\n");
    assert_ne!(f.derive().unwrap().release_id, original.release_id);
    decisions
        .push(json!({"mutation":"changed_preparation_source","decision":"new_release_identity"}));

    // Missing, substituted or surplus objects never fall back.
    let mut f = Fixture::new();
    f.staged.remove("reg3");
    assert!(err_text(f.derive()).contains("no substitution"));
    let mut f = Fixture::new();
    let other = f.staged["reg4"].clone();
    f.staged.insert("reg3".into(), other);
    assert!(f.derive().is_err());
    let mut f = Fixture::new();
    let extra = f.staged["reg4"].clone();
    f.staged.insert("bin0".into(), extra);
    assert!(err_text(f.derive()).contains("non-members"));
    decisions
        .push(json!({"mutation":"missing_substituted_or_surplus_object","decision":"rejected"}));

    if let Ok(path) = std::env::var("W1_MUTATION_RECEIPT") {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&json!({"schema":"jope.w1.regression-release-mutations/v1",
                "original_release_id":original.release_id,"decisions":decisions}))
            .unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn development_receiver_rejects_changed_bytes_and_realigned_labels() {
    let f = Fixture::new();
    let release = f.derive().unwrap();
    let dev = &f.staged["reg2"].development;
    release.verify_development("reg2", dev).unwrap();
    let mut table: Value = serde_json::from_slice(dev).unwrap();
    let rows = table["rows"].as_array_mut().unwrap();
    let t = rows[0]["target"].clone();
    rows[0]["target"] = rows[1]["target"].clone();
    rows[1]["target"] = t;
    let swapped = canonical(&table);
    assert!(err_text(release.verify_development("reg2", &swapped)).contains("development bytes"));
    assert!(release.verify_development("reg9", dev).is_err());
    assert!(release.verify_development("bin0", dev).is_err());
}

#[test]
fn unknown_fields_versions_and_noncanonical_encodings_are_rejected() {
    let f = Fixture::new();
    let release = f.derive().unwrap();
    let bytes = release.encode().unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    let reject = |v: &Value, needle: &str| {
        let text = err_text(RegressionRelease::open(&canonical(v)));
        assert!(text.contains(needle), "{needle}: {text}");
    };
    let mut v = value.clone();
    v["future_field"] = json!(true);
    reject(&v, "fields");
    let mut v = value.clone();
    v["members"][0]["raw"]["mirror"] = json!("s3://latest");
    reject(&v, "fields");
    let mut v = value.clone();
    v["schema"] = json!("jope.regression-release/v2");
    reject(&v, "unsupported regression release schema");
    let mut v = value.clone();
    v["release_id"] = json!(hex("forged"));
    reject(&v, "identity");
    let mut v = value.clone();
    v["members"][0]["rows"]["development"] = json!(9);
    reject(&v, "");
    let mut v = value.clone();
    v["members"][0]["target"]["column"] = json!("target_alias");
    reject(&v, "target identity");
    let mut v = value.clone();
    v["members"][0]["target"]["development_label_role"] = json!("outer_test");
    reject(&v, "label roles");
    let mut v = value.clone();
    v["members"][0]["rows"]["total"] = json!(12.0);
    assert!(RegressionRelease::open(&serde_json::to_vec(&v).unwrap()).is_err());
    assert!(err_text(RegressionRelease::open(
        &serde_json::to_vec_pretty(&value).unwrap()
    ))
    .contains("canonical"));
    assert!(RegressionRelease::open(&bytes[..bytes.len() - 1]).is_err());
    let text = String::from_utf8(bytes.clone()).unwrap();
    let duplicated = text.replacen('{', "{\"schema\":\"jope.regression-release/v0\",", 1);
    assert!(RegressionRelease::open(duplicated.as_bytes()).is_err());

    let registry = f.registry.encode().unwrap();
    assert_eq!(FamilyRegistry::open(&registry).unwrap(), f.registry);
    let rv: Value = serde_json::from_slice(&registry).unwrap();
    let mut v = rv.clone();
    v["exposures"][0]["encoder"]["assumed"] = json!("unseen");
    assert!(FamilyRegistry::open(&canonical(&v)).is_err());
    let mut v = rv.clone();
    v["schema"] = json!("jope.regression-family-registry/v2");
    assert!(FamilyRegistry::open(&canonical(&v)).is_err());
    assert!(FamilyRegistry::open(&serde_json::to_vec_pretty(&rv).unwrap()).is_err());
}

#[test]
fn membership_is_every_regression_entry_of_the_frozen_thirty_entry_panel() {
    let f = Fixture::new();
    assert_eq!(
        panel_regression_members(&f.panel_bytes()).unwrap(),
        (0..10).map(|i| format!("reg{i}")).collect::<Vec<_>>()
    );
    let release = f.derive().unwrap();
    let mut g = Fixture::new();
    g.panel["datasets"][0]["task"] = json!("binary");
    assert!(err_text(g.derive()).contains("exactly ten regression"));
    let mut g = Fixture::new();
    g.panel["datasets"][1]["id"] = json!("reg0");
    assert!(g.derive().is_err());
    let mut g = Fixture::new();
    g.panel["datasets"].as_array_mut().unwrap().pop();
    assert!(err_text(g.derive()).contains("thirty"));
    let mut g = Fixture::new();
    g.panel["datasets"][0]["replacement_for"] = json!("reg1");
    assert!(g.derive().is_err());
    let mut g = Fixture::new();
    g.panel["datasets"][4]["source"]["sha256"] = json!(hex("other version"));
    assert!(check_against_panel(&release, &g.panel_bytes()).is_err());
    let mut g = Fixture::new();
    g.panel["datasets"][0]["split"]["protocol"] = json!("other-protocol");
    assert!(err_text(g.derive()).contains("preparation protocol"));
}

#[test]
fn family_registry_keeps_unknown_distinct_from_unseen_and_blocks_never_seen_claims() {
    let f = Fixture::new();
    f.registry.validate().unwrap();
    for id in f.registry.members() {
        let blockers = f.registry.never_seen_claim_blockers(id).unwrap();
        assert_eq!(blockers.len(), 3, "{id}: {blockers:?}");
        assert!(blockers[0].starts_with("encoder exposure unknown"));
    }
    assert_ne!(ExposureState::Unknown, ExposureState::Unseen);

    let unseen = || Exposure {
        state: ExposureState::Unseen,
        basis: "auditable clean root".into(),
        evidence: vec![EvidenceRef {
            description: "complete read log".into(),
            sha256: Some(hex("read log")),
        }],
    };
    let mut clean = f.registry.clone();
    for e in &mut clean.exposures {
        e.encoder = unseen();
        e.teacher = unseen();
        e.checkpoint = unseen();
    }
    clean.validate().unwrap();
    assert!(clean.never_seen_claim_blockers("reg0").unwrap().is_empty());

    // Unseen or exposed without evidence is not a claim; it must stay unknown.
    let mut r = clean.clone();
    r.exposures[0].checkpoint.evidence.clear();
    assert!(err_text(r.validate()).contains("requires evidence"));
    let mut r = clean.clone();
    r.exposures[0].encoder = Exposure {
        state: ExposureState::Exposed,
        basis: "trained".into(),
        evidence: vec![],
    };
    assert!(r.validate().is_err());

    // One unknown component blocks the claim even when the others are unseen.
    let mut r = clean.clone();
    r.exposures[0].checkpoint = unknown("ancestry untraceable");
    assert_eq!(r.never_seen_claim_blockers("reg0").unwrap().len(), 1);
    assert!(r.never_seen_claim_blockers("reg1").unwrap().is_empty());

    // Unresolved relationships block both sides.
    let mut r = clean.clone();
    r.families[0].relations.push(Relation {
        dataset: "reg0".into(),
        other: "reg1".into(),
        kind: RelationKind::SharedEntity,
        status: RelationStatus::Unresolved,
        evidence: "same collection name".into(),
    });
    r.validate().unwrap();
    assert_eq!(r.never_seen_claim_blockers("reg0").unwrap().len(), 1);
    assert_eq!(r.never_seen_claim_blockers("reg1").unwrap().len(), 1);
    assert!(r.never_seen_claim_blockers("reg2").unwrap().is_empty());

    // A resolved alias/alternate target cannot straddle the exclusion boundary.
    let mut r = clean.clone();
    r.families[0].relations.push(Relation {
        dataset: "reg0".into(),
        other: "reg1".into(),
        kind: RelationKind::Alias,
        status: RelationStatus::Resolved,
        evidence: "documented alias".into(),
    });
    assert!(err_text(r.validate()).contains("straddles"));
    let mut r = clean.clone();
    r.families[0].relations.push(Relation {
        dataset: "reg0".into(),
        other: "external:derived".into(),
        kind: RelationKind::AlternateTarget,
        status: RelationStatus::Resolved,
        evidence: "documented alternate target".into(),
    });
    assert!(r.validate().is_err());
    r.families[0].source_keys.push("external:derived".into());
    r.validate().unwrap();
    // The same external key cannot belong to two families.
    r.families[1].source_keys.push("external:derived".into());
    assert!(err_text(r.validate()).contains("cannot span families"));

    // Every member needs exactly one exposure record; releases need full coverage.
    let mut r = clean.clone();
    r.exposures.remove(3);
    assert!(r.validate().is_err());
    let mut g = Fixture::new();
    g.registry.families.pop();
    g.registry.exposures.pop();
    g.registry.validate().unwrap();
    assert!(err_text(g.derive()).contains("cover exactly"));
}

const COMMITTED_RELEASE: &[u8] = include_bytes!("../releases/regression-10-v1/release.json");
const COMMITTED_REGISTRY: &[u8] =
    include_bytes!("../releases/regression-10-v1/family-registry.json");
const FROZEN_PANEL: &[u8] = include_bytes!("../../../benchmarks/panel.json");
const PREPARATION_IMPLEMENTATION: &[u8] = include_bytes!("../../../benchmarks/prepare.py");

/// Membership proof for the committed release against the unchanged frozen
/// panel. Byte-level re-derivation from the staged custody objects is the
/// `jope-datasets regression-release verify` receipt, not this CI test.
#[test]
fn committed_ten_regression_release_matches_the_frozen_panel() {
    let release = RegressionRelease::open(COMMITTED_RELEASE).unwrap();
    assert_eq!(release.encode().unwrap(), COMMITTED_RELEASE);
    assert_eq!(
        release.release_id,
        "859ce6895f6ca2b016103465f44e4f84a3dd54dcce8b2c832ba15ec536454d4a"
    );
    let registry = FamilyRegistry::open(COMMITTED_REGISTRY).unwrap();
    assert_eq!(registry.encode().unwrap(), COMMITTED_REGISTRY);
    assert_eq!(registry.sha256().unwrap(), release.family_registry_sha256);
    check_against_panel(&release, FROZEN_PANEL).unwrap();
    assert_eq!(
        release.preparation.source_sha256,
        sha256_hex(PREPARATION_IMPLEMENTATION)
    );
    let members: Vec<String> = release.members.iter().map(|m| m.dataset.clone()).collect();
    assert_eq!(members, panel_regression_members(FROZEN_PANEL).unwrap());
    assert_eq!(
        members,
        [
            "197_cpu_act",
            "225_puma8NH",
            "230_machine_cpu",
            "522_pm10",
            "529_pollen",
            "579_fri_c0_250_5",
            "auto_insurance_price",
            "690_visualizing_galaxy",
            "712_chscase_geyser1",
            "503_wind"
        ]
    );
    let binary = release.exclusions.iter().filter(|e| e.task == "binary");
    let multiclass = release.exclusions.iter().filter(|e| e.task == "multiclass");
    assert_eq!((binary.count(), multiclass.count()), (10, 10));
    assert_eq!(release.derived_from.task_counts["timeseries"], 0);
    assert_eq!(
        registry.members(),
        members
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>()
    );
    // Historical exposure is unknown, never unseen: every never-seen claim is blocked.
    for m in &members {
        let blockers = registry.never_seen_claim_blockers(m).unwrap();
        assert!(blockers
            .iter()
            .any(|b| b.starts_with("encoder exposure unknown")));
        assert!(blockers
            .iter()
            .any(|b| b.starts_with("checkpoint exposure unknown")));
    }
}

fn write_staged(root: &Path, f: &Fixture) {
    for dir in ["downloads", "metadata", "development", "splits", "sealed"] {
        std::fs::create_dir_all(root.join(dir)).unwrap();
    }
    for (id, s) in &f.staged {
        std::fs::write(root.join("downloads").join(sha256_hex(&s.raw)), &s.raw).unwrap();
        std::fs::write(
            root.join("metadata").join(format!("{id}.yaml")),
            &s.metadata,
        )
        .unwrap();
        for (dir, bytes) in [
            ("development", &s.development),
            ("splits", &s.split),
            ("sealed", &s.sealed),
        ] {
            std::fs::write(root.join(dir).join(format!("{id}.json")), bytes).unwrap();
        }
    }
}

#[test]
fn staged_loader_reads_only_exact_regular_objects_without_fallback() {
    let f = Fixture::new();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("prepared");
    write_staged(&root, &f);
    // Classification members have no staged objects at all; they are never opened.
    let loaded = load_staged(&root, &f.panel_bytes()).unwrap();
    assert_eq!(loaded.len(), 10);
    for (id, s) in &loaded {
        assert_eq!(s.development, f.staged[id].development);
        assert_eq!(s.raw, f.staged[id].raw);
    }

    // A symlinked leaf is refused even when it points at identical bytes.
    let dev = root.join("development/reg1.json");
    let copy = dir.path().join("elsewhere.json");
    std::fs::rename(&dev, &copy).unwrap();
    std::os::unix::fs::symlink(&copy, &dev).unwrap();
    assert!(err_text(load_staged(&root, &f.panel_bytes())).contains("symlink"));
    std::fs::remove_file(&dev).unwrap();
    std::fs::rename(&copy, &dev).unwrap();

    // A missing exact object is not replaced by a `latest` or differently named copy.
    let raw = root
        .join("downloads")
        .join(sha256_hex(&f.staged["reg2"].raw));
    std::fs::rename(&raw, root.join("downloads/latest")).unwrap();
    assert!(err_text(load_staged(&root, &f.panel_bytes())).contains("absent"));
    std::fs::rename(root.join("downloads/latest"), &raw).unwrap();

    // A different version under the exact name loads but cannot derive a release.
    std::fs::write(&raw, b"new version under the old key").unwrap();
    let substituted = load_staged(&root, &f.panel_bytes()).unwrap();
    let result = derive(&ReleaseSources {
        panel: &f.panel_bytes(),
        preparation_source_path: "benchmarks/prepare.py",
        preparation_source: &f.preparation,
        registry: &f.registry,
        staged: &substituted,
    });
    assert!(err_text(result).contains("raw object differs"));

    // A symlinked partition directory is refused.
    let moved = dir.path().join("splits-real");
    std::fs::rename(root.join("splits"), &moved).unwrap();
    std::os::unix::fs::symlink(&moved, root.join("splits")).unwrap();
    assert!(err_text(load_staged(&root, &f.panel_bytes())).contains("symlink"));
}
