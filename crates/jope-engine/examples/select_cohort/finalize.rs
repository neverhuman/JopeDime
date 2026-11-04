use super::*;

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

pub(crate) fn finalize(
    root: &Path,
    selection_path: &Path,
    contexts_path: &Path,
    output: &Path,
) -> Result<()> {
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
