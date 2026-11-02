use super::*;

/// Apply the original cohort's frozen transforms. This never fits a projection,
/// changes an existing vector, copies old contexts, or reads held-out targets.

pub fn apply(
    root: &Path,
    jobs_path: &Path,
    hyperion: &Path,
    foundation: &Path,
    target: &Path,
    assembly: &Path,
    output: &Path,
    qualification: bool,
) -> Result<()> {
    crate::task_contract::require_original_gpu_grant("apply-contexts")?;
    let _lease = crate::controller::gpu_lease(root, "apply frozen context transforms")?;
    apply_impl(
        root,
        jobs_path,
        hyperion,
        foundation,
        target,
        assembly,
        output,
        qualification,
        None,
    )
}

/// Explicit binary qualification application. The scheduler owns the GPU lease,
/// as for direct encoder ranks. Regression projection fitting is never repeated.
pub fn apply_binary_qualification(
    root: &Path,
    jobs_path: &Path,
    hyperion: &Path,
    foundation: &Path,
    target: &Path,
    assembly: &Path,
    output: &Path,
    aliases: &Path,
) -> Result<()> {
    apply_impl(
        root,
        jobs_path,
        hyperion,
        foundation,
        target,
        assembly,
        output,
        true,
        Some(aliases),
    )
}

fn apply_impl(
    root: &Path,
    jobs_path: &Path,
    hyperion: &Path,
    foundation: &Path,
    target: &Path,
    assembly: &Path,
    output: &Path,
    qualification: bool,
    binary_aliases: Option<&Path>,
) -> Result<()> {
    crate::task_contract::require_original_gpu_grant("apply-contexts-numerical")?;
    ensure!(
        tch::Cuda::is_available(),
        "frozen context application requires CUDA numerical parity"
    );
    ensure!(
        output.is_absolute() && assembly.is_absolute(),
        "context output and frozen assembly paths must be absolute"
    );
    let assembly_path = assembly.join("assembly.json");
    let assembly_sha = file_hash(&assembly_path)?;
    let original_bytes = fs::read(assembly.join("index.json"))?;
    let original: Value = serde_json::from_slice(&original_bytes)?;
    let spec: Value = read_json(&assembly_path)?;
    ensure!(
        original["schema"] == "parallel.context_index.v1"
            && original["assembly_sha256"] == assembly_sha
            && spec["schema"] == "parallel.context_assembly.v1",
        "frozen assembly identity mismatch"
    );
    ensure!(
        qualification || (spec["qualification"] == false && original["qualification"] == false),
        "provisional assembly cannot produce clean contexts"
    );
    ensure!(
        spec["dimensions"] == json!({"hyperion":512,"foundation":1024,"target":512})
            && spec["final_test_targets_used"] == false,
        "unsupported frozen assembly contract"
    );
    let transforms_path = assembly.join("transforms.ot");
    let transforms_sha = file_hash(&transforms_path)?;
    ensure!(
        spec["transforms_sha256"] == transforms_sha,
        "frozen transforms changed"
    );
    let registry: Option<crate::families::Registry> = if spec["registry"].is_null() {
        None
    } else {
        let path = resolve(
            root,
            Path::new(
                spec["registry"]["path"]
                    .as_str()
                    .context("assembly registry path absent")?,
            ),
        )?;
        ensure!(
            file_hash(&path)? == spec["registry"]["sha256"],
            "frozen assembly registry changed"
        );
        let registry: crate::families::Registry = read_json(&path)?;
        registry.validate()?;
        ensure!(
            qualification || !registry.qualification,
            "qualification registry cannot produce independent contexts"
        );
        Some(registry)
    };
    let reference_rows = training_parity_rows(
        &original,
        registry
            .as_ref()
            .context("frozen parity requires a verified family registry")?,
    )?;
    let reference_selection = json!({
        "schema":"parallel.frozen_context_reference_selection.v1",
        "assembly_sha256":assembly_sha,
        "original_index_sha256":hash(&original_bytes),
        "registry":spec["registry"],
        "selection_rule":"first two sorted admitted training families with four cuts; first four sorted dataset/cut identities per family",
        "reference_partition":"meta_train",
        "reference_views":reference_rows.len(),
        "rows":reference_rows,
    });
    let clip = spec["clip_abs"]
        .as_f64()
        .context("frozen clipping bound absent")?;
    ensure!(
        clip.is_finite() && clip > 0.,
        "invalid frozen clipping bound"
    );
    let jobs: GpJobs = read_json(jobs_path)?;
    ensure!(
        jobs.schema == "parallel.gp_jobs.v1"
            && !jobs.jobs.is_empty()
            && (qualification || !jobs.qualification),
        "encoding job qualification mismatch"
    );
    let mut apply_receipt = json!({"schema":"parallel.frozen_context_application.v1","jobs_sha256":file_hash(jobs_path)?,"assembly":assembly_path.strip_prefix(root)?,"assembly_sha256":assembly_sha,"transforms":transforms_path.strip_prefix(root)?,"transforms_sha256":transforms_sha,"component_roots":{"hyperion":hyperion.strip_prefix(root)?,"foundation":foundation.strip_prefix(root)?,"target":target.strip_prefix(root)?},"qualification":qualification,"projection_refitted":false,"cohort_statistics_refitted":false,"old_vectors_copied":0,"final_test_targets_used":false,"numerical_backend":"Rust/LibTorch 2.7 CUDA FP32; original frozen transforms"});
    let aliases: Option<Value> = binary_aliases.map(read_json).transpose()?;
    if let Some(aliases) = &aliases {
        ensure!(
            aliases["schema"] == "parallel.binary_family_aliases.v1"
                && aliases["qualification"] == true
                && aliases["frozen_before_partitioning"] == true
                && aliases["existing_assignments_preserved"] == true,
            "invalid frozen binary family aliases"
        );
        apply_receipt["binary_qualification"] = true.into();
        apply_receipt["binary_aliases_sha256"] = file_hash(binary_aliases.unwrap())?.into();
        apply_receipt["projection_scope"] =
            "frozen regression projection applied to separate binary qualification index".into();
    }
    if registry.is_some() {
        apply_receipt["registry"] = spec["registry"].clone();
    }
    // All components are verified before creating the new version or using CUDA.
    let mut prepared = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut components = Vec::new();
    let (mut h, mut f, mut t) = (Vec::new(), Vec::new(), Vec::new());
    for job in &jobs.jobs {
        if !prepared.contains_key(&job.prepared) {
            let path = resolve(root, &job.prepared)?.join("prepared.json");
            ensure!(
                file_hash(&path)? == job.prepared_sha256,
                "prepared cohort changed"
            );
            let p: Prepared = read_json(&path)?;
            ensure!(
                qualification || p.family_partition_frozen,
                "family partition must be frozen before clean contexts"
            );
            if let Some(aliases) = &aliases {
                let raw: Value = read_json(&path)?;
                ensure!(
                    p.task == "binary"
                        && p.qualification_only
                        && raw["binary_aliases_sha256"] == apply_receipt["binary_aliases_sha256"],
                    "binary prepared cut must bind frozen aliases and qualification"
                );
                let entries = aliases["families"]
                    .as_array()
                    .context("binary families absent")?
                    .iter()
                    .filter(|f| f["dataset"] == p.dataset)
                    .collect::<Vec<_>>();
                ensure!(
                    entries.len() == 1
                        && entries[0]["family"] == p.family
                        && entries[0]["partition"] == p.meta_partition,
                    "binary cut family or partition differs from frozen aliases"
                );
                if let Some(family) = registry.as_ref().and_then(|r| r.families.get(&p.family)) {
                    ensure!(
                        family.partition == p.meta_partition,
                        "binary qualification changed an existing regression family assignment"
                    );
                }
            } else if let Some(registry) = &registry {
                registered_prepared(registry, &p)?;
            }
            prepared.insert(job.prepared.clone(), p);
        }
        let p = &prepared[&job.prepared];
        ensure!(
            seen.insert((p.dataset.clone(), job.cut)),
            "duplicate dataset/cut in context jobs"
        );
        let cut = p
            .cuts
            .iter()
            .find(|c| c.index == job.cut)
            .context("cut absent")?;
        let suffix = Path::new(&p.dataset).join(format!("cut-{}", job.cut));
        let mut bindings = serde_json::Map::new();
        for (name, dir, receipt, file, dimensions, destination) in [
            (
                "hyperion",
                hyperion,
                "hyperion.json",
                "hyperion.f32",
                512,
                &mut h,
            ),
            (
                "foundation",
                foundation,
                "foundation.json",
                "foundation_raw.f32",
                2048,
                &mut f,
            ),
            ("target", target, "target.json", "target.f32", 512, &mut t),
        ] {
            let dir = dir.join(&suffix);
            let receipt_path = dir.join(receipt);
            let meta: Value = read_json(&receipt_path)?;
            ensure!(
                meta["dataset"] == p.dataset
                    && meta["family"] == p.family
                    && meta["cut"] == job.cut
                    && meta["prepared_sha256"] == job.prepared_sha256
                    && meta["final_test_targets_used"] == false
                    && meta["shape"] == json!([dimensions]),
                "component cut receipt mismatch: {}",
                receipt_path.display()
            );
            destination.extend(signature(
                &dir,
                receipt,
                file,
                dimensions,
                &cut.training_sha256,
                &cut.preprocessing_sha256,
            )?);
            bindings.insert(name.into(),json!({"receipt":receipt_path.strip_prefix(root)?,"receipt_sha256":file_hash(&receipt_path)?,"signature":dir.join(file).strip_prefix(root)?,"signature_sha256":meta["signature_sha256"],"spec_sha256":meta["spec_sha256"]}));
        }
        components.push(Value::Object(bindings));
    }
    fs::create_dir_all(output)?;
    immutable_json(
        &output.join("reference-selection.json"),
        &reference_selection,
    )?;
    apply_receipt["reference_selection_sha256"] =
        file_hash(&output.join("reference-selection.json"))?.into();
    immutable_json(&output.join("application.json"), &apply_receipt)?;
    tch::set_num_threads(2);
    let device = Device::Cuda(0);
    let transforms = load_transforms(&transforms_path, device)?;
    let parity =
        verify_original_contexts(root, &original, &reference_rows, &transforms, device, clip)?;
    write_json(&output.join("reference-parity.json"), &parity)?;
    ensure!(
        parity["passed"] == true,
        "frozen context application differs from original saved contexts: {parity}"
    );
    let n = jobs.jobs.len() as i64;
    let normalized = tch::no_grad(|| {
        normalize_frozen(
            &Tensor::from_slice(&h).view([n, 512]).to_device(device),
            &Tensor::from_slice(&f).view([n, 2048]).to_device(device),
            &Tensor::from_slice(&t).view([n, 512]).to_device(device),
            &transforms,
            clip,
        )
    });
    ensure!(
        normalized.size() == [n, 2048] && normalized.isfinite().all().int64_value(&[]) != 0,
        "invalid frozen context output"
    );
    let values: Vec<f32> = Vec::try_from(normalized.to_device(Device::Cpu).flatten(0, -1))?;
    let mut rows = Vec::new();
    for ((job, values), bindings) in jobs
        .jobs
        .iter()
        .zip(values.chunks_exact(2048))
        .zip(components)
    {
        let p = &prepared[&job.prepared];
        let cut = p.cuts.iter().find(|c| c.index == job.cut).unwrap();
        let path = output.join(&p.dataset).join(format!("cut-{}.f32", job.cut));
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        if path.exists() {
            ensure!(
                file_hash(&path)? == hash(&bytes),
                "existing context vector changed"
            );
        } else {
            atomic_write(&path, &bytes)?;
        }
        let encoding = Encoding {
            schema: "parallel.encoding.v1".into(),
            dataset: p.dataset.clone(),
            family: p.family.clone(),
            partition: p.meta_partition.clone(),
            cut: job.cut,
            prepared_sha256: job.prepared_sha256.clone(),
            training_sha256: cut.training_sha256.clone(),
            selected_training_rows_sha256: hash(&serde_json::to_vec(&cut.train_indices)?),
            preprocessing_sha256: cut.preprocessing_sha256.clone(),
            encoder_spec_sha256: assembly_sha.clone(),
            numerical_backend: "Rust/LibTorch 2.7 CUDA FP32; original frozen transforms".into(),
            dataset_dimensions: 1536,
            target_dimensions: 512,
            output_sha256: hash(&bytes),
        };
        let components_sha = hash(&serde_json::to_vec(&bindings)?);
        let mut meta = serde_json::to_value(&encoding)?;
        if aliases.is_some() {
            meta["task"] = "binary".into();
            meta["qualification"] = true.into();
            meta["binary_aliases_sha256"] = apply_receipt["binary_aliases_sha256"].clone();
        }
        meta["components"] = bindings;
        meta["components_sha256"] = components_sha.clone().into();
        meta["application_sha256"] = file_hash(&output.join("application.json"))?.into();
        immutable_json(&path.with_extension("json"), &meta)?;
        rows.push(json!({"dataset":p.dataset,"family":p.family,"partition":p.meta_partition,"cut":job.cut,"path":path.strip_prefix(root)?,"sha256":encoding.output_sha256,"prepared":job.prepared,"prepared_sha256":job.prepared_sha256,"components_sha256":components_sha,"receipt_sha256":file_hash(&path.with_extension("json"))?}));
    }
    let mut index = json!({"schema":"parallel.context_index.v1","assembly_sha256":assembly_sha,"assembly":assembly_path.strip_prefix(root)?,"qualification":qualification,"dimensions":2048,"application_sha256":file_hash(&output.join("application.json"))?,"reference_parity_sha256":file_hash(&output.join("reference-parity.json"))?,"encodings":rows});
    if aliases.is_some() {
        index["task"] = "binary".into();
        index["binary_qualification"] = true.into();
    }
    immutable_json(&output.join("index.json"), &index)
}
