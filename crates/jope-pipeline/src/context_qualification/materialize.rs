//! Parent-only custody preparation. Full backing files are verified before the
//! exact native fitting rows are selected. No corpus path reaches a worker.
use super::*;
use jope_policy::context_producer::{NativeComponent, ReferenceCase, ReferenceManifest};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Component,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceSources {
    pub index: Artifact,
    pub registry: Artifact,
    pub assembly: Artifact,
    /// The index's prepared/receipt paths resolve only within this custody root.
    pub metadata_root: PathBuf,
    /// Component receipt/vector paths retain their original root-relative names.
    pub component_root: PathBuf,
    /// Final2048 reference vectors retain their original root-relative names.
    pub context_root: PathBuf,
    /// Original backing training source paths retain their original relative names.
    pub source_root: PathBuf,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceSelection {
    pub schema: String,
    pub index_sha256: String,
    pub registry_sha256: String,
    pub assembly_sha256: String,
    pub cases: Vec<ReferenceSelectionCase>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceSelectionCase {
    pub dataset: String,
    pub family: String,
    pub cut: u8,
    pub prepared_sha256: String,
    pub index_entry: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterializedReferences {
    pub selection: Artifact,
    pub manifest: Artifact,
    pub cases: Vec<Artifact>,
    pub job_identities: Vec<ReferenceJobIdentity>,
    /// All actual opened inputs and their full hashes, including original source.
    pub custody_inputs: Vec<Artifact>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceJobIdentity {
    pub dataset_sha256: String,
    pub split_sha256: String,
    pub fitting_rows_sha256: String,
    pub fitting_rows: usize,
    pub fold: u8,
}

pub fn materialize_references(
    sources: &ReferenceSources,
    directory: &Path,
) -> Result<MaterializedReferences> {
    physical_directory(directory)?;
    for root in [
        &sources.metadata_root,
        &sources.component_root,
        &sources.context_root,
        &sources.source_root,
    ] {
        physical_directory(root)?;
    }
    ensure!(
        sources.index.sha256 == context_producer::REFERENCE_INDEX_SHA256
            && sources.registry.sha256 == context_producer::REFERENCE_REGISTRY_SHA256,
        "reference index/registry differs from retained native lineage"
    );
    let index_bytes = read_artifact(&sources.index, 32 << 20)?;
    let registry_bytes = read_artifact(&sources.registry, 16 << 20)?;
    let assembly_bytes = read_artifact(&sources.assembly, 16 << 20)?;
    ensure!(
        sources.assembly.sha256 == context_producer::ASSEMBLY_SHA256,
        "wrong accepted context assembly"
    );
    let index: Value = serde_json::from_slice(&index_bytes)?;
    let registry: Value = serde_json::from_slice(&registry_bytes)?;
    let assembly: Value = serde_json::from_slice(&assembly_bytes)?;
    ensure!(
        index["schema"] == "parallel.context_index.v1"
            && index["assembly_sha256"] == sources.assembly.sha256
            && index["registry"]["sha256"] == sources.registry.sha256,
        "reference index binding changed"
    );
    ensure!(
        assembly["final_test_targets_used"] == false,
        "assembly input closure lacks explicit train-only proof"
    );
    let entries = index["encodings"]
        .as_array()
        .context("reference index entries absent")?;
    let mut candidates: BTreeMap<(String, String, String), BTreeMap<u8, Value>> = BTreeMap::new();
    for entry in entries {
        if entry["partition"] != "meta_train" {
            continue;
        }
        let family = string(entry, "family")?;
        let dataset = string(entry, "dataset")?;
        if registry["families"][family]["partition"] != "meta_train"
            || registry["held_review_datasets"].get(dataset).is_some()
        {
            continue;
        }
        let Some(cut) = entry["cut"].as_u64().filter(|v| *v < 4) else {
            continue;
        };
        let prepared = string(entry, "prepared")?;
        let group = candidates
            .entry((family.into(), dataset.into(), prepared.into()))
            .or_default();
        ensure!(
            group.insert(cut as u8, entry.clone()).is_none(),
            "duplicate native reference dataset/cut"
        );
    }
    let mut selected_families = BTreeSet::new();
    let mut selected = Vec::new();
    let mut custody = vec![
        sources.index.clone(),
        sources.registry.clone(),
        sources.assembly.clone(),
    ];
    // Choice depends only on index/registry/prepared metadata. No reference
    // vectors, original source cells, or encoding losses are read here.
    for ((family, dataset, prepared), cuts) in candidates {
        if cuts.len() != 4 || selected_families.contains(&family) {
            continue;
        }
        let first = &cuts[&0];
        let prepared_file = Artifact {
            path: resolve(&sources.metadata_root, &format!("{prepared}/prepared.json"))?,
            sha256: string(first, "prepared_sha256")?.into(),
        };
        let prepared_bytes = read_artifact(&prepared_file, 32 << 20)?;
        let meta: Value = serde_json::from_slice(&prepared_bytes)?;
        if meta["task"] != "regression" {
            continue;
        }
        ensure!(
            meta["dataset"] == dataset
                && meta["family"] == family
                && meta["meta_partition"] == "meta_train"
                && meta["target_last"] == true,
            "native prepared metadata differs from training index"
        );
        let registered = registry["families"][&family]["datasets"]
            .as_array()
            .context("family membership absent")?
            .iter()
            .any(|d| {
                d["dataset"] == dataset
                    && d["preparation_eligible"] == true
                    && d["prepared"]
                        .as_array()
                        .is_some_and(|ps| ps.iter().any(|p| p["sha256"] == prepared_file.sha256))
            });
        ensure!(
            registered,
            "reference prepared metadata not admitted by training registry"
        );
        ensure!(
            cuts.values()
                .all(|e| e["prepared_sha256"] == prepared_file.sha256),
            "reference cuts disagree on prepared identity"
        );
        custody.push(prepared_file);
        for (cut, entry) in cuts {
            selected.push(ReferenceSelectionCase {
                dataset: dataset.clone(),
                family: family.clone(),
                cut,
                prepared_sha256: string(&entry, "prepared_sha256")?.into(),
                index_entry: entry,
            });
        }
        selected_families.insert(family);
        if selected_families.len() == 2 {
            break;
        }
    }
    ensure!(
        selected.len() == 8,
        "fewer than two complete admitted regression training families"
    );
    let selection = ReferenceSelection {
        schema: "jope.cpu-context-reference-selection/v1".into(),
        index_sha256: sources.index.sha256.clone(),
        registry_sha256: sources.registry.sha256.clone(),
        assembly_sha256: sources.assembly.sha256.clone(),
        cases: selected,
    };
    // Persist the choice before reading any selected vector or original cells.
    let selection_file = write_immutable(
        &directory.join("selection.json"),
        &serde_json::to_vec(&selection)?,
    )?;
    let mut cases = Vec::new();
    let mut job_identities = Vec::new();
    for choice in &selection.cases {
        let entry = &choice.index_entry;
        let prepared_relative = string(entry, "prepared")?;
        let prepared_file = Artifact {
            path: resolve(
                &sources.metadata_root,
                &format!("{prepared_relative}/prepared.json"),
            )?,
            sha256: choice.prepared_sha256.clone(),
        };
        let meta: Value = serde_json::from_slice(&read_artifact(&prepared_file, 32 << 20)?)?;
        let cut = meta["cuts"]
            .as_array()
            .context("native cuts missing")?
            .iter()
            .find(|c| c["index"] == choice.cut)
            .context("selected cut missing")?;
        let vector_relative = string(entry, "path")?;
        let receipt_path = Path::new(vector_relative).with_extension("json");
        let receipt = Artifact {
            path: resolve(
                &sources.metadata_root,
                receipt_path.to_str().context("receipt path UTF8")?,
            )?,
            sha256: string(entry, "receipt_sha256")?.into(),
        };
        let receipt_bytes = read_artifact(&receipt, 2 << 20)?;
        let context: Value = serde_json::from_slice(&receipt_bytes)?;
        ensure!(
            context["dataset"] == choice.dataset
                && context["family"] == choice.family
                && context["cut"] == choice.cut
                && context["partition"] == "meta_train"
                && context["prepared_sha256"] == choice.prepared_sha256,
            "selected reference receipt changed"
        );
        let train_indices = indices(&cut["train_indices"])?;
        ensure!(
            context["selected_training_rows_sha256"]
                == sha256_hex(&serde_json::to_vec(&train_indices)?),
            "native selected training indices differ"
        );
        let training = indices(&meta["selected_train_indices"])?
            .into_iter()
            .collect::<BTreeSet<_>>();
        ensure!(
            !train_indices.is_empty()
                && train_indices.iter().all(|i| training.contains(i))
                && train_indices.iter().collect::<BTreeSet<_>>().len() == train_indices.len(),
            "cut contains nontraining/duplicate rows"
        );
        let prep = Artifact {
            path: resolve(
                &sources.metadata_root,
                &format!("{prepared_relative}/cut-{}/preprocessing.json", choice.cut),
            )?,
            sha256: string(cut, "preprocessing_sha256")?.into(),
        };
        let preprocessing: NativePreprocessing =
            serde_json::from_slice(&read_artifact(&prep, 16 << 20)?)?;
        let rows =
            load_training_rows(sources, &meta, &train_indices, &preprocessing, &mut custody)?;
        job_identities.push(ReferenceJobIdentity {
            dataset_sha256: string(&meta["storage"], "source_sha256")?.into(),
            split_sha256: choice.prepared_sha256.clone(),
            fitting_rows_sha256: sha256_hex(&serde_json::to_vec(&train_indices)?),
            fitting_rows: rows.len(),
            fold: choice.cut,
        });
        let csv = numeric_csv(&rows);
        ensure!(
            sha256_hex(&csv) == string(cut, "training_sha256")?
                && context["training_sha256"] == string(cut, "training_sha256")?
                && context["preprocessing_sha256"] == prep.sha256,
            "materialized rows differ from frozen native cut"
        );
        let output = Artifact {
            path: resolve(&sources.context_root, vector_relative)?,
            sha256: string(entry, "sha256")?.into(),
        };
        let context_vector = read_artifact(&output, 8192)?;
        let mut components = BTreeMap::new();
        for name in ["hyperion", "foundation", "target"] {
            let record = &context["components"][name];
            let receipt_file = Artifact {
                path: resolve(&sources.component_root, string(record, "receipt")?)?,
                sha256: string(record, "receipt_sha256")?.into(),
            };
            let vector_file = Artifact {
                path: resolve(&sources.component_root, string(record, "signature")?)?,
                sha256: string(record, "signature_sha256")?.into(),
            };
            components.insert(
                name.into(),
                NativeComponent {
                    receipt: read_artifact(&receipt_file, 2 << 20)?,
                    vector: read_artifact(&vector_file, 8192)?,
                },
            );
            custody.extend([receipt_file, vector_file]);
        }
        cases.push(ReferenceCase {
            id: format!("{}/cut-{}", choice.dataset, choice.cut),
            prepared_rows_sha256: sha256_hex(&serde_json::to_vec(&rows)?),
            prepared_rows: rows,
            context_receipt: receipt_bytes,
            context_vector,
            components,
        });
        custody.extend([prepared_file, prep, receipt, output]);
    }
    let manifest = ReferenceManifest {
        schema: "jope.cpu-context-reference/v1".into(),
        verification_policy_sha256: context_producer::verification_policy_sha256(),
        registry_bytes,
        index_bytes,
        cases,
    };
    let bytes = serde_json::to_vec(&manifest)?;
    let manifest_sha = sha256_hex(&bytes);
    let inputs = context_producer::case_inputs(&bytes, &manifest_sha)?;
    let manifest_file = write_immutable(&directory.join("reference-manifest.json"), &bytes)?;
    let mut case_files = Vec::new();
    for (i, bytes) in inputs.iter().enumerate() {
        case_files.push(write_immutable(
            &directory.join(format!("case-{i}.json")),
            bytes,
        )?);
    }
    let mut unique = BTreeMap::new();
    for file in custody {
        if let Some(old) = unique.insert(file.path.clone(), file.sha256.clone()) {
            ensure!(
                old == file.sha256,
                "custody path changed during preparation"
            );
        }
    }
    let result = MaterializedReferences {
        selection: selection_file,
        manifest: manifest_file,
        cases: case_files,
        job_identities,
        custody_inputs: unique
            .into_iter()
            .map(|(path, sha256)| Artifact { path, sha256 })
            .collect(),
    };
    write_immutable(
        &directory.join("materialization.json"),
        &serde_json::to_vec(&result)?,
    )?;
    Ok(result)
}

#[derive(Deserialize)]
struct NativePreprocessing {
    schema: String,
    columns: Vec<NativeColumn>,
    labels: Vec<String>,
    target_mean: f64,
    target_scale: f64,
}
#[derive(Deserialize)]
struct NativeColumn {
    kind: String,
    impute: f64,
    mean: f64,
    scale: f64,
    categories: Vec<String>,
}
fn load_training_rows(
    sources: &ReferenceSources,
    meta: &Value,
    train_indices: &[usize],
    pp: &NativePreprocessing,
    custody: &mut Vec<Artifact>,
) -> Result<Vec<Vec<f64>>> {
    ensure!(
        pp.schema == "parallel.preprocessing.v1"
            && pp.labels.is_empty()
            && pp.target_mean.is_finite()
            && pp.target_scale.is_finite()
            && pp.target_scale > 0.0,
        "native regression preprocessing invalid"
    );
    let storage = &meta["storage"];
    ensure!(
        storage.is_object(),
        "reference preparation requires explicit original row mapping"
    );
    let file = Artifact {
        path: resolve(&sources.source_root, string(storage, "source")?)?,
        sha256: string(storage, "source_sha256")?.into(),
    };
    // Hash the full source, then stream only selected records into numerical input.
    context_producer::worker::verify_file(&policy_artifact(&file), 2u64 << 30)?;
    let map = indices(&storage["original_training_row_map"])?;
    let selected = train_indices
        .iter()
        .map(|&i| {
            map.get(i)
                .copied()
                .context("native row mapping out of range")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        selected.iter().collect::<BTreeSet<_>>().len() == selected.len(),
        "duplicate original source fitting rows"
    );
    let wanted = selected.iter().copied().collect::<BTreeSet<_>>();
    let mut reader = csv::Reader::from_path(&file.path)?;
    let headers = reader.headers()?.clone();
    let feature_names = storage["feature_columns"]
        .as_array()
        .context("native features absent")?
        .iter()
        .map(|v| v.as_str().context("invalid feature name"))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        feature_names.len() == pp.columns.len()
            && feature_names.iter().collect::<BTreeSet<_>>().len() == feature_names.len(),
        "native feature schema width/duplicates"
    );
    let position = |name: &str| {
        let found = headers
            .iter()
            .enumerate()
            .filter(|(_, h)| *h == name)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        ensure!(found.len() == 1, "source column absent/duplicate: {name}");
        Ok(found[0])
    };
    let features = feature_names
        .iter()
        .map(|name| position(name))
        .collect::<Result<Vec<_>>>()?;
    let target = position(string(storage, "target_column")?)?;
    let mut fitted = BTreeMap::new();
    for (i, record) in reader.records().enumerate() {
        let record = record?;
        if !wanted.contains(&i) {
            continue;
        }
        let mut row = Vec::new();
        for (&column, p) in features.iter().zip(&pp.columns) {
            let value = record.get(column).context("short source record")?.trim();
            let encoded = match p.kind.as_str() {
                "numeric" => {
                    ensure!(
                        p.impute.is_finite()
                            && p.mean.is_finite()
                            && p.scale.is_finite()
                            && p.scale > 0.0,
                        "invalid native numeric preprocessor"
                    );
                    (value
                        .parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite())
                        .unwrap_or(p.impute)
                        - p.mean)
                        / p.scale
                }
                "categorical" => {
                    ensure!(
                        p.categories.windows(2).all(|w| w[0] < w[1]),
                        "native categories are not sorted unique"
                    );
                    p.categories
                        .binary_search_by(|c| c.as_str().cmp(value))
                        .map(|i| i as f64)
                        .unwrap_or(-1.0)
                }
                _ => anyhow::bail!("unsupported native preprocessing kind"),
            };
            row.push(encoded);
        }
        let y = record
            .get(target)
            .context("short target record")?
            .trim()
            .parse::<f64>()?;
        row.push((y - pp.target_mean) / pp.target_scale);
        ensure!(
            row.iter().all(|v| v.is_finite()),
            "nonfinite native prepared row"
        );
        fitted.insert(i, row);
    }
    // Detect source mutation during selection without parsing unselected targets.
    context_producer::worker::verify_file(&policy_artifact(&file), 2u64 << 30)?;
    custody.push(file);
    selected
        .iter()
        .map(|i| {
            fitted
                .get(i)
                .cloned()
                .context("selected original training row missing")
        })
        .collect()
}
fn numeric_csv(rows: &[Vec<f64>]) -> Vec<u8> {
    rows.iter()
        .map(|r| {
            let mut s = r
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",");
            s.push('\n');
            s
        })
        .collect::<String>()
        .into_bytes()
}
fn indices(v: &Value) -> Result<Vec<usize>> {
    v.as_array()
        .context("row index list missing")?
        .iter()
        .map(|v| {
            usize::try_from(v.as_u64().context("invalid row index")?).context("row index overflow")
        })
        .collect()
}
fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .with_context(|| format!("missing metadata string {key}"))
}
fn resolve(root: &Path, relative: &str) -> Result<PathBuf> {
    let p = Path::new(relative);
    ensure!(
        !relative.is_empty() && p.components().all(|c| matches!(c, Component::Normal(_))),
        "custody reference path escapes root"
    );
    let p = root.join(p);
    ensure!(p.canonicalize()? == p, "custody reference must be physical");
    Ok(p)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numeric_csv_matches_native_float_display() {
        assert_eq!(
            numeric_csv(&[vec![1.0, -0.0, 1.25], vec![-2.0, 0.0, 3.0]]),
            b"1,-0,1.25\n-2,0,3\n"
        );
    }
    #[test]
    fn indices_refuse_negative_and_fractional() {
        assert!(indices(&serde_json::json!([-1])).is_err());
        assert!(indices(&serde_json::json!([0.5])).is_err());
    }
    #[test]
    fn parent_selects_only_original_fitting_rows_after_full_hash_verification() {
        let fixture_root =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-fixtures");
        std::fs::create_dir_all(&fixture_root).unwrap();
        let scratch = tempfile::Builder::new()
            .prefix("cpu-context-input-")
            .tempdir_in(&fixture_root)
            .unwrap();
        let root = scratch.path().canonicalize().unwrap();
        let csv = b"number,category,target\n1,A,10\n999,HOLD,DO_NOT_PARSE_TARGET\n3,B,30\n";
        let source = root.join("training-source.csv");
        std::fs::write(&source, csv).unwrap();
        let unused = Artifact {
            path: source.clone(),
            sha256: sha256_hex(csv),
        };
        let sources = ReferenceSources {
            index: unused.clone(),
            registry: unused.clone(),
            assembly: unused,
            metadata_root: root.clone(),
            component_root: root.clone(),
            context_root: root.clone(),
            source_root: root,
        };
        let meta = serde_json::json!({"storage":{"source":"training-source.csv","source_sha256":sha256_hex(csv),"feature_columns":["number","category"],"target_column":"target","original_training_row_map":[2,0]}});
        let pp = NativePreprocessing {
            schema: "parallel.preprocessing.v1".into(),
            columns: vec![
                NativeColumn {
                    kind: "numeric".into(),
                    impute: 0.0,
                    mean: 1.0,
                    scale: 2.0,
                    categories: vec![],
                },
                NativeColumn {
                    kind: "categorical".into(),
                    impute: -1.0,
                    mean: 0.0,
                    scale: 1.0,
                    categories: vec!["A".into(), "B".into()],
                },
            ],
            labels: vec![],
            target_mean: 10.0,
            target_scale: 10.0,
        };
        let mut custody = Vec::new();
        let actual = load_training_rows(&sources, &meta, &[1, 0], &pp, &mut custody).unwrap();
        assert_eq!(actual, vec![vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 2.0]]);
        assert_eq!(custody.len(), 1);
        // A mutation outside selected fitting rows still invalidates the full
        // frozen source identity; it never licenses a silently changed input.
        std::fs::write(
            &source,
            b"number,category,target\n1,A,10\n0,HOLD,CHANGED\n3,B,30\n",
        )
        .unwrap();
        assert!(load_training_rows(&sources, &meta, &[1, 0], &pp, &mut custody).is_err());
    }
}
