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

include!("regression_release/mutations.rs");
include!("regression_release/receivers.rs");
include!("regression_release/committed_fixtures.rs");
