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

mod inputs;
#[cfg(test)]
mod tests;

#[cfg(test)]
use inputs::NativeColumn;
use inputs::{indices, load_training_rows, numeric_csv, resolve, string, NativePreprocessing};
