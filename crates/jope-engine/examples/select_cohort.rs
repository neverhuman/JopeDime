//! Frozen, score-blind qualification family selection and context pointer subsets.
use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use jope_parallel::{
    families::Registry,
    manifest::*,
    prepare::Prepared,
    service::{GpJob, GpJobs},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    root: PathBuf,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Freeze {
        #[arg(long)]
        registry: PathBuf,
        #[arg(long, required = true)]
        jobs: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    Finalize {
        #[arg(long)]
        selection: PathBuf,
        #[arg(long)]
        contexts: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Binding {
    path: PathBuf,
    sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Candidate {
    dataset: String,
    family: String,
    partition: String,
    source_key: String,
    metadata_sha256: String,
    source_identity_sha256: String,
    rows: usize,
    features: usize,
    eligible: bool,
    holds: Vec<String>,
    prepared_versions: Vec<Binding>,
}
#[derive(Serialize, Deserialize)]
struct Selection {
    schema: String,
    seed: u64,
    qualification_only: bool,
    independent_selection_eligible: bool,
    registry: Binding,
    job_manifests: Vec<Binding>,
    vocabulary: Binding,
    method: String,
    projection_limit: String,
    quotas: BTreeMap<String, usize>,
    candidates: Vec<Candidate>,
    selected: Vec<Candidate>,
    original_final_test_targets_opened: bool,
    model_scores_consulted: bool,
}

fn tie(scope: &str, value: &str) -> String {
    hash(format!("{SEED}:{scope}:{value}").as_bytes())
}
fn stratum(c: &Candidate) -> (u8, u8) {
    let rows = if c.rows < 256 {
        0
    } else if c.rows < 1024 {
        1
    } else if c.rows < 4096 {
        2
    } else if c.rows < 16384 {
        3
    } else {
        4
    };
    let features = if c.features <= 4 {
        0
    } else if c.features <= 16 {
        1
    } else if c.features <= 64 {
        2
    } else {
        3
    };
    (rows, features)
}
fn quotas() -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("meta_train".into(), 48),
        ("meta_validation".into(), 8),
        ("meta_test".into(), 8),
    ])
}

fn select(candidates: &[Candidate]) -> Result<Vec<Candidate>> {
    let mut families: BTreeMap<&str, Vec<&Candidate>> = BTreeMap::new();
    let mut ids = BTreeSet::new();
    for c in candidates {
        ensure!(
            c.eligible && c.holds.is_empty(),
            "held or unadmitted candidate: {}",
            c.dataset
        );
        ensure!(ids.insert(&c.dataset), "duplicate candidate dataset");
        ensure!(
            c.rows > 0 && c.rows <= ROW_LIMIT && c.features > 0,
            "invalid candidate dimensions"
        );
        ensure!(
            quotas().contains_key(&c.partition),
            "unknown candidate partition"
        );
        families.entry(&c.family).or_default().push(c);
    }
    let mut representatives = Vec::new();
    for (family, mut versions) in families {
        ensure!(
            versions
                .iter()
                .all(|c| c.partition == versions[0].partition),
            "family crosses partitions"
        );
        versions.sort_by_key(|c| tie(&format!("representative:{family}"), &c.dataset));
        representatives.push(versions[0].clone());
    }
    let mut selected = Vec::new();
    for (partition, quota) in quotas() {
        let eligible = representatives
            .iter()
            .filter(|c| c.partition == partition)
            .count();
        ensure!(
            eligible >= quota,
            "insufficient {partition} families: {eligible}, required {quota}"
        );
        let mut buckets: BTreeMap<(u8, u8), Vec<Candidate>> = BTreeMap::new();
        for c in representatives.iter().filter(|c| c.partition == partition) {
            buckets.entry(stratum(c)).or_default().push((*c).clone());
        }
        let mut buckets = buckets
            .into_iter()
            .map(|(key, mut values)| {
                values.sort_by_key(|c| tie(&format!("family:{partition}"), &c.family));
                (key, VecDeque::from(values))
            })
            .collect::<Vec<_>>();
        buckets.sort_by_key(|(key, _)| {
            tie(
                &format!("stratum:{partition}"),
                &format!("{}:{}", key.0, key.1),
            )
        });
        let mut count = 0;
        while count < quota {
            for (_, bucket) in &mut buckets {
                if let Some(c) = bucket.pop_front() {
                    selected.push(c);
                    count += 1;
                }
                if count == quota {
                    break;
                }
            }
        }
    }
    selected.sort_by(|a, b| (&a.partition, &a.family).cmp(&(&b.partition, &b.family)));
    ensure!(
        selected.len() == 64
            && selected
                .iter()
                .map(|c| &c.family)
                .collect::<BTreeSet<_>>()
                .len()
                == 64,
        "cohort family collision"
    );
    Ok(selected)
}

fn binding(root: &Path, path: &Path) -> Result<Binding> {
    let full = resolve(root, path)?;
    Ok(Binding {
        path: path.into(),
        sha256: file_hash(&full)?,
    })
}

fn freeze(root: &Path, registry_path: &Path, paths: &[PathBuf], output: &Path) -> Result<()> {
    let output = resolve(root, output)?;
    ensure!(!output.exists(), "selection receipt already exists");
    let registry: Registry = read_json(&resolve(root, registry_path)?)?;
    registry.validate()?;
    let mut manifests = paths
        .iter()
        .map(|p| binding(root, p))
        .collect::<Result<Vec<_>>>()?;
    manifests.sort_by(|a, b| a.path.cmp(&b.path));
    ensure!(
        manifests.windows(2).all(|w| w[0].path != w[1].path),
        "duplicate input job manifest"
    );
    let mut by_dataset: BTreeMap<String, Candidate> = BTreeMap::new();
    let mut prepared = BTreeMap::<PathBuf, Prepared>::new();
    let mut prepared_hashes = BTreeMap::<PathBuf, String>::new();
    let mut vocabulary: Option<Binding> = None;
    let mut cuts = BTreeMap::<(String, usize), String>::new();
    for input in &manifests {
        let jobs: GpJobs = read_json(&resolve(root, &input.path)?)?;
        ensure!(
            jobs.schema == "parallel.gp_jobs.v1" && jobs.qualification,
            "expected qualification jobs"
        );
        let vocab = binding(
            root,
            jobs.vocabulary.as_ref().context("job vocabulary absent")?,
        )?;
        if let Some(old) = &vocabulary {
            ensure!(old.sha256 == vocab.sha256, "job vocabularies differ");
        } else {
            vocabulary = Some(vocab);
        }
        for job in jobs.jobs {
            let manifest = job.prepared.join("prepared.json");
            if !prepared.contains_key(&job.prepared) {
                ensure!(
                    file_hash(&resolve(root, &manifest)?)? == job.prepared_sha256,
                    "prepared input changed"
                );
                let p: Prepared = read_json(&resolve(root, &manifest)?)?;
                ensure!(p.schema == "parallel.prepared.v1" && p.task == "regression" && p.target_last && p.clean_evaluation && !p.qualification_only,
                    "cohort requires reconstructed regression cuts without known original split overlap");
                let storage = p
                    .storage
                    .as_ref()
                    .context("cohort requires pointer-only preparations")?;
                let dataset_id = &p.dataset;
                let matches = registry
                    .families
                    .values()
                    .flat_map(|f| {
                        f.datasets
                            .iter()
                            .filter(move |d| &d.dataset == dataset_id)
                            .map(move |d| (f, d))
                    })
                    .collect::<Vec<_>>();
                ensure!(
                    matches.len() == 1,
                    "dataset registry identity missing or ambiguous: {}",
                    p.dataset
                );
                let (family, dataset) = matches[0];
                ensure!(
                    family.holds.is_empty() && dataset.preparation_eligible,
                    "held or unadmitted registry family"
                );
                ensure!(
                    family.family == p.family && family.partition == p.meta_partition,
                    "prepared family partition changed"
                );
                if let Some(b) = &p.family_registry_binding {
                    ensure!(
                        b.registry_id == registry.registry_id
                            && b.family == p.family
                            && b.partition == p.meta_partition
                            && b.metadata_sha256 == dataset.metadata_sha256
                            && b.entity_review_sha256 == dataset.entity_review_sha256,
                        "prepared registry lineage mismatch"
                    );
                }
                let candidate = Candidate {
                    dataset: p.dataset.clone(),
                    family: p.family.clone(),
                    partition: p.meta_partition.clone(),
                    source_key: dataset.source_key.clone(),
                    metadata_sha256: dataset.metadata_sha256.clone(),
                    source_identity_sha256: hash(&serde_json::to_vec(
                        &json!({"storage":p.storage,"original_train_sha256":p.original_train_sha256,"original_test_sha256":p.original_test_sha256,"selected_train_indices":p.selected_train_indices,"selected_test_indices":p.selected_test_indices,"entity_columns":p.entity_columns}),
                    )?),
                    rows: p.rows_total,
                    features: storage.feature_columns.len(),
                    eligible: true,
                    holds: vec![],
                    prepared_versions: vec![Binding {
                        path: manifest,
                        sha256: job.prepared_sha256.clone(),
                    }],
                };
                if let Some(previous) = by_dataset.get_mut(&p.dataset) {
                    ensure!(
                        previous.family == candidate.family
                            && previous.partition == candidate.partition
                            && previous.rows == candidate.rows
                            && previous.features == candidate.features
                            && previous.source_identity_sha256 == candidate.source_identity_sha256,
                        "prepared versions disagree on source identity, cohort dimensions, or family"
                    );
                    previous
                        .prepared_versions
                        .extend(candidate.prepared_versions);
                } else {
                    by_dataset.insert(p.dataset.clone(), candidate);
                }
                prepared.insert(job.prepared.clone(), p);
                prepared_hashes.insert(job.prepared.clone(), job.prepared_sha256.clone());
            }
            ensure!(
                prepared_hashes[&job.prepared] == job.prepared_sha256,
                "job prepared hash changes within manifest"
            );
            let p = &prepared[&job.prepared];
            let cut = p
                .cuts
                .iter()
                .find(|c| c.index == job.cut)
                .context("job cut absent")?;
            ensure!(
                cut.train_indices
                    .iter()
                    .all(|i| !cut.validation_indices.contains(i)),
                "cut row overlap"
            );
            let key = (p.dataset.clone(), job.cut);
            if let Some(old) = cuts.insert(key, job.prepared_sha256.clone()) {
                ensure!(
                    old == job.prepared_sha256,
                    "conflicting prepared versions for a cut"
                );
            }
        }
    }
    let mut candidates = by_dataset.into_values().collect::<Vec<_>>();
    for c in &mut candidates {
        c.prepared_versions.sort_by(|a, b| a.path.cmp(&b.path));
    }
    let selected = select(&candidates)?;
    let receipt=Selection {schema:"parallel.qualification_cohort.v1".into(),seed:SEED,qualification_only:true,independent_selection_eligible:false,
        registry:binding(root,registry_path)?,job_manifests:manifests,vocabulary:vocabulary.context("no vocabulary")?,
        method:"one dataset per family chosen by seeded SHA256; per partition round-robin fixed row bins [<256,<1024,<4096,<16384,larger] and feature bins [<=4,<=16,<=64,larger], with seeded stratum/family ordering; no outcomes or scores".into(),
        projection_limit:"Projection v2 was fitted on 55 meta-training families, exceeding the selected 48; this cohort remains qualification-only and does not establish fresh independent pilot evaluation.".into(),
        quotas:quotas(),candidates,selected,original_final_test_targets_opened:false,model_scores_consulted:false};
    write_json(&output, &receipt)?;
    println!(
        "{}",
        json!({"selected_families":64,"qualification_only":true,"receipt":output,"sha256":file_hash(&output)?})
    );
    Ok(())
}

fn assembly_path(root: &Path, index: &Value) -> Result<PathBuf> {
    let expected = index["assembly_sha256"]
        .as_str()
        .context("assembly identity absent")?;
    let mut pending = vec![(index.clone(), 0usize)];
    let mut found = None;
    while let Some((current, depth)) = pending.pop() {
        ensure!(
            depth < 8 && current["assembly_sha256"] == expected,
            "context source assembly mismatch or recursive sources"
        );
        if let Some(path) = current["assembly"].as_str() {
            let path = PathBuf::from(path);
            ensure!(
                file_hash(&resolve(root, &path)?)? == expected,
                "context assembly changed"
            );
            found = Some(path);
        } else {
            let sources = current["sources"]
                .as_array()
                .context("context assembly binding absent")?;
            ensure!(!sources.is_empty(), "empty context sources");
            for source in sources {
                let path = PathBuf::from(
                    source["path"]
                        .as_str()
                        .context("source index path absent")?,
                );
                ensure!(
                    file_hash(&resolve(root, &path)?)? == source["sha256"],
                    "source context index changed"
                );
                pending.push((read_json(&resolve(root, &path)?)?, depth + 1));
            }
        }
    }
    found.context("context assembly path absent")
}

fn finalize(root: &Path, selection_path: &Path, contexts_path: &Path, output: &Path) -> Result<()> {
    let output = resolve(root, output)?;
    ensure!(!output.exists(), "cohort subset destination exists");
    let selection: Selection = read_json(&resolve(root, selection_path)?)?;
    ensure!(
        selection.schema == "parallel.qualification_cohort.v1"
            && selection.seed == SEED
            && selection.qualification_only
            && !selection.independent_selection_eligible
            && selection.quotas == quotas()
            && !selection.original_final_test_targets_opened
            && !selection.model_scores_consulted,
        "invalid cohort qualification"
    );
    ensure!(
        select(&selection.candidates)? == selection.selected,
        "frozen cohort selection changed"
    );
    ensure!(
        file_hash(&resolve(root, &selection.registry.path)?)? == selection.registry.sha256,
        "registry receipt changed"
    );
    ensure!(
        file_hash(&resolve(root, &selection.vocabulary.path)?)? == selection.vocabulary.sha256,
        "selected vocabulary changed"
    );
    let selected = selection
        .selected
        .iter()
        .map(|c| (c.dataset.clone(), c))
        .collect::<BTreeMap<_, _>>();
    let mut selected_jobs: BTreeMap<(String, usize), GpJob> = BTreeMap::new();
    let mut known_prepared = BTreeMap::<PathBuf, Prepared>::new();
    for c in &selection.selected {
        for b in &c.prepared_versions {
            ensure!(
                file_hash(&resolve(root, &b.path)?)? == b.sha256,
                "selected preparation changed"
            );
            known_prepared.insert(
                b.path.parent().unwrap().to_path_buf(),
                read_json(&resolve(root, &b.path)?)?,
            );
        }
    }
    for input in &selection.job_manifests {
        ensure!(
            file_hash(&resolve(root, &input.path)?)? == input.sha256,
            "input GP jobs changed"
        );
        let jobs: GpJobs = read_json(&resolve(root, &input.path)?)?;
        for job in jobs.jobs {
            if let Some(p) = known_prepared.get(&job.prepared) {
                selected_jobs
                    .entry((p.dataset.clone(), job.cut))
                    .or_insert(job);
            }
        }
    }
    let mut index: Value = read_json(&resolve(root, contexts_path)?)?;
    ensure!(
        index["schema"] == "parallel.context_index.v1"
            && index["qualification"] == true
            && index["dimensions"] == 2048,
        "expected qualification context index"
    );
    let assembly_path = assembly_path(root, &index)?;
    ensure!(
        file_hash(&resolve(root, &assembly_path)?)? == index["assembly_sha256"],
        "context assembly changed"
    );
    let assembly: Value = read_json(&resolve(root, &assembly_path)?)?;
    let fit_families = assembly["meta_training_families"]
        .as_object()
        .context("projection family identities absent")?;
    ensure!(
        fit_families.len() == 55,
        "projection v2 training-family count differs from frozen qualification limitation"
    );
    let mut contexts = Vec::new();
    let mut seen = BTreeSet::new();
    for row in index["encodings"]
        .as_array()
        .context("context rows absent")?
    {
        let dataset = row["dataset"].as_str().context("context dataset absent")?;
        if !selected.contains_key(dataset) {
            continue;
        }
        let cut = row["cut"].as_u64().context("context cut absent")? as usize;
        let key = (dataset.to_string(), cut);
        let job = selected_jobs
            .get(&key)
            .context("selected context is not in frozen GP jobs")?;
        ensure!(seen.insert(key), "duplicate selected context");
        ensure!(
            row["prepared"] == json!(job.prepared)
                && row["prepared_sha256"] == job.prepared_sha256
                && row["family"] == selected[dataset].family
                && row["partition"] == selected[dataset].partition,
            "selected context identity mismatch"
        );
        let path = PathBuf::from(row["path"].as_str().context("context path absent")?);
        let bytes = fs::read(resolve(root, &path)?)?;
        ensure!(
            bytes.len() == 2048 * 4
                && hash(&bytes) == row["sha256"]
                && bytes
                    .chunks_exact(4)
                    .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "context bytes changed"
        );
        let receipt_path = resolve(root, &path.with_extension("json"))?;
        if let Some(expected) = row["receipt_sha256"].as_str() {
            ensure!(
                file_hash(&receipt_path)? == expected,
                "context receipt changed"
            );
        }
        let meta: Value = read_json(&receipt_path)?;
        let p = &known_prepared[&job.prepared];
        let c = p
            .cuts
            .iter()
            .find(|c| c.index == cut)
            .context("prepared cut absent")?;
        ensure!(
            meta["prepared_sha256"] == job.prepared_sha256
                && meta["training_sha256"] == c.training_sha256
                && meta["preprocessing_sha256"] == c.preprocessing_sha256
                && meta["encoder_spec_sha256"] == index["assembly_sha256"]
                && meta["output_sha256"] == row["sha256"],
            "context receipt cut binding mismatch"
        );
        contexts.push(row.clone());
    }
    ensure!(
        seen.len() == selected_jobs.len(),
        "selected cuts are missing verified saved contexts"
    );
    let selection_sha = file_hash(&resolve(root, selection_path)?)?;
    index["assembly"] = json!(assembly_path);
    index["encodings"] = json!(contexts);
    index["cohort_selection_sha256"] = json!(selection_sha);
    index["source_context_index"] = json!(binding(root, contexts_path)?);
    let final_output = output;
    let output = final_output.with_file_name(format!(
        ".{}.staging-{}",
        final_output
            .file_name()
            .context("output name absent")?
            .to_string_lossy(),
        std::process::id()
    ));
    ensure!(!output.exists(), "cohort staging collision");
    fs::create_dir_all(&output)?;
    write_json(
        &output.join("gp-jobs.json"),
        &GpJobs {
            schema: "parallel.gp_jobs.v1".into(),
            qualification: true,
            jobs: selected_jobs.into_values().collect(),
            vocabulary: Some(selection.vocabulary.path.clone()),
        },
    )?;
    write_json(&output.join("contexts.json"), &index)?;
    write_json(
        &output.join("receipt.json"),
        &json!({"schema":"parallel.cohort_pointer_subsets.v1","qualification_only":true,"independent_selection_eligible":false,"selection_sha256":selection_sha,"source_context_index":binding(root,contexts_path)?,"gp_jobs_sha256":file_hash(&output.join("gp-jobs.json"))?,"contexts_sha256":file_hash(&output.join("contexts.json"))?,"selected_families":64,"selected_views":seen.len(),"projection_meta_training_families":fit_families.keys().collect::<Vec<_>>(),"projection_fit_family_count":55,"selected_meta_training_families":48,"copied_dataset_bytes":0,"copied_encoding_bytes":0,"final_test_targets_opened":false,"enqueued":false}),
    )?;
    fs::File::open(&output)?.sync_all()?;
    fs::rename(&output, &final_output)?;
    fs::File::open(final_output.parent().context("output parent absent")?)?.sync_all()?;
    println!(
        "{}",
        json!({"families":64,"verified_context_pointers":seen.len(),"receipt":final_output.join("receipt.json")})
    );
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Freeze {
            registry,
            jobs,
            output,
        } => freeze(&cli.root, &registry, &jobs, &output),
        Command::Finalize {
            selection,
            contexts,
            output,
        } => finalize(&cli.root, &selection, &contexts, &output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn candidates() -> Vec<Candidate> {
        let mut result = Vec::new();
        for (partition, n) in [("meta_train", 55), ("meta_validation", 9), ("meta_test", 8)] {
            for i in 0..n {
                result.push(Candidate {
                    dataset: format!("{partition}-{i}"),
                    family: format!("{partition}-family-{i}"),
                    partition: partition.into(),
                    source_key: format!("source-{partition}-{i}"),
                    metadata_sha256: "0".repeat(64),
                    source_identity_sha256: "1".repeat(64),
                    rows: 64 << (i % 6),
                    features: 1 << (i % 7),
                    eligible: true,
                    holds: vec![],
                    prepared_versions: vec![],
                });
            }
        }
        result
    }
    #[test]
    fn stable_under_input_permutations_and_one_dataset_per_family() {
        let mut input = candidates();
        let mut alias = input[0].clone();
        alias.dataset = "alternate-target".into();
        input.push(alias);
        let a = select(&input).unwrap();
        input.reverse();
        input.rotate_left(13);
        assert_eq!(a, select(&input).unwrap());
        assert_eq!(a.len(), 64);
        assert_eq!(
            a.iter().map(|c| &c.family).collect::<BTreeSet<_>>().len(),
            64
        );
        for (partition, n) in quotas() {
            assert_eq!(a.iter().filter(|c| c.partition == partition).count(), n);
        }
    }
    #[test]
    fn holds_and_insufficient_partition_quotas_are_rejected() {
        let mut input = candidates();
        input[0].holds.push("unresolved_alias".into());
        assert!(select(&input).is_err());
        input[0].holds.clear();
        input.retain(|c| c.dataset != "meta_test-0");
        assert!(select(&input)
            .unwrap_err()
            .to_string()
            .contains("insufficient meta_test"));
    }
    #[test]
    fn family_partition_collisions_are_rejected() {
        let mut input = candidates();
        input[55].family = input[0].family.clone();
        assert!(select(&input)
            .unwrap_err()
            .to_string()
            .contains("crosses partitions"));
    }
}
