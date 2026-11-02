use super::*;

pub(crate) fn run_root(campaign_root: &Path, run_id: &str) -> PathBuf {
    campaign_root.join(run_id)
}

pub(crate) fn plan(
    run_id: &str,
    campaign_root: &Path,
    corpus_manifest: &Path,
    comparator_root: &Path,
    seed_catalog: &Path,
) -> Result<()> {
    let root = run_root(campaign_root, run_id);
    let path = root.join("plan.v1.json");
    let plan = if path.exists() {
        let existing = read_plan(campaign_root, run_id)?;
        if existing.corpus_manifest != corpus_manifest.display().to_string()
            || existing.comparator_root != comparator_root.display().to_string()
            || existing.seed_catalog_sha256 != seed_sources_sha256(seed_catalog)?
            || existing.scq_binary != scq_str()
        {
            bail!(
                "immutable multiclass plan input drift at {}",
                path.display()
            );
        }
        existing
    } else {
        let plan = build_plan(run_id, corpus_manifest, comparator_root, seed_catalog)?;
        let bytes = canonical_json(&plan)?;
        immutable_write(&path, &bytes)?;
        plan
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "plan": path,
            "datasets": plan.datasets.len(),
            "priority": PRIORITY_DATASETS,
            "tail": plan.datasets.len() - PRIORITY_DATASETS,
            "fast_baseline_released": plan.datasets.iter().filter(|row| row.fast_baselines_complete).count(),
            "blocked_on_fast_baselines": plan.datasets.iter().filter(|row| !row.fast_baselines_complete).count(),
        }))?
    );
    Ok(())
}

pub(crate) fn read_plan(campaign_root: &Path, run_id: &str) -> Result<Plan> {
    let path = run_root(campaign_root, run_id).join("plan.v1.json");
    let plan: Plan = serde_json::from_slice(&fs::read(&path)?)
        .with_context(|| format!("read {}", path.display()))?;
    let gate = coverage_gate(Path::new(&plan.comparator_root))?;
    if plan.schema != PLAN_SCHEMA
        || plan.run_id != RUN_ID
        || !plan.test_adaptive
        || plan.selection_bias != "test-selected"
        || plan.benchmark_valid
        || plan.official_combined_rank_eligible
        || plan.config != Config::fixed()
        || plan.datasets.len() != 272
        || plan.seeds.len() != POPULATION
        || plan.corpus_eval != corpus_eval_str()
        || plan.release_binary_sha256 != current_binary_sha256()?
        || plan.scq_binary != scq_str()
        || sha256_file(Path::new(&plan.scq_binary))? != plan.scq_binary_sha256
        || plan.corpus_manifest != mc_coverage_manifest_str()
        || plan.corpus_manifest_sha256 != MC_COVERAGE_MANIFEST_SHA256
        || plan.multiclass_coverage_final_receipt_sha256 != gate.multiclass_receipt_sha256
        || plan.timeseries_coverage_manifest_sha256 != TS_COVERAGE_MANIFEST_SHA256
        || plan.timeseries_coverage_final_receipt_sha256 != gate.timeseries_receipt_sha256
        || plan.comparator_set_path != gate.comparator_set_path
        || plan.comparator_set_sha256 != gate.comparator_set_sha256
        || plan.seed_catalog_sha256
            != seed_sources_sha256(Path::new(
                &plan.seed_catalog.split(" + ").next().unwrap_or(""),
            ))?
        || plan.seed_population_sha256 != sha256_bytes(&serde_json::to_vec(&plan.seeds)?)
        || sha256_file(Path::new(&plan.corpus_eval))? != plan.corpus_eval_sha256
    {
        bail!("multiclass plan identity or policy drift");
    }
    Ok(plan)
}

pub(crate) fn ready_dataset_bindings(plan: &Plan, root: &Path) -> Result<Vec<DatasetBinding>> {
    let comparator_root = Path::new(&plan.comparator_root);
    let gate = coverage_gate(comparator_root)?;
    let mut ready = Vec::new();
    for planned in &plan.datasets {
        let Some(binding) = bind_frozen_comparator(planned.clone(), comparator_root, &gate)? else {
            if planned.fast_baselines_complete {
                bail!(
                    "{} planned comparator lock disappeared",
                    planned.dataset_hash
                );
            }
            continue;
        };
        if binding != *planned {
            bail!("{} planned comparator drift", planned.dataset_hash);
        }
        let path = root
            .join("datasets")
            .join(&binding.dataset_hash)
            .join("comparator-binding.v1.json");
        let bytes = canonical_json(&binding)?;
        if path.is_file() {
            if fs::read(&path)? != bytes {
                bail!(
                    "immutable multiclass comparator binding collision at {}",
                    path.display()
                );
            }
        } else {
            immutable_write(&path, &bytes)?;
        }
        ready.push(binding);
    }
    Ok(ready)
}
