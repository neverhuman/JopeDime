use super::*;
fn signature(
    dir: &Path,
    receipt: &str,
    file: &str,
    dimensions: usize,
    training: &str,
    preprocessing: &str,
) -> Result<Vec<f32>> {
    let meta: serde_json::Value = read_json(&dir.join(receipt))?;
    ensure!(
        meta["training_sha256"] == training && meta["preprocessing_sha256"] == preprocessing,
        "component input identity mismatch"
    );
    let bytes = fs::read(dir.join(file))?;
    ensure!(
        bytes.len() == dimensions * 4 && meta["signature_sha256"] == hash(&bytes),
        "component bytes changed"
    );
    let values = bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    ensure!(values.iter().all(|x| x.is_finite()), "nonfinite component");
    Ok(values)
}
pub fn assemble(
    root: &Path,
    jobs_path: &Path,
    hyperion: &Path,
    foundation: &Path,
    target: &Path,
    output: &Path,
    qualification: bool,
) -> Result<()> {
    assemble_impl(
        root,
        jobs_path,
        hyperion,
        foundation,
        target,
        output,
        qualification,
        None,
    )
}

pub fn assemble_registered(
    root: &Path,
    jobs_path: &Path,
    hyperion: &Path,
    foundation: &Path,
    target: &Path,
    registry_path: &Path,
    output: &Path,
    qualification: bool,
) -> Result<()> {
    assemble_impl(
        root,
        jobs_path,
        hyperion,
        foundation,
        target,
        output,
        qualification,
        Some(registry_path),
    )
}

pub(super) fn registered_prepared(
    registry: &crate::families::Registry,
    p: &Prepared,
) -> Result<Value> {
    let matches: Vec<_> = registry
        .families
        .values()
        .flat_map(|family| {
            family
                .datasets
                .iter()
                .filter(move |d| d.dataset == p.dataset)
                .map(move |d| (family, d))
        })
        .collect();
    ensure!(
        matches.len() == 1,
        "prepared dataset is missing or ambiguous in family registry: {}",
        p.dataset
    );
    let (family, dataset) = matches[0];
    ensure!(
        family.family == p.family && family.partition == p.meta_partition,
        "prepared family/partition differs from frozen registry for {}",
        p.dataset
    );
    ensure!(
        family.holds.is_empty() && dataset.preparation_eligible,
        "prepared dataset is not admitted by frozen registry: {}",
        p.dataset
    );
    if let Some(binding) = &p.family_registry_binding {
        ensure!(
            binding.family == family.family
                && binding.partition == family.partition
                && binding.metadata_sha256 == dataset.metadata_sha256
                && binding.entity_review_sha256 == dataset.entity_review_sha256,
            "prepared registry lineage differs from current preserved assignment"
        );
    }
    Ok(
        json!({"dataset":p.dataset,"family":family.family,"partition":family.partition,"source_key":dataset.source_key,"metadata_sha256":dataset.metadata_sha256,"entity_review_sha256":dataset.entity_review_sha256,"prepared_registry_binding":p.family_registry_binding}),
    )
}

fn assemble_impl(
    root: &Path,
    jobs_path: &Path,
    hyperion: &Path,
    foundation: &Path,
    target: &Path,
    output: &Path,
    qualification: bool,
    registry_path: Option<&Path>,
) -> Result<()> {
    crate::task_contract::require_original_gpu_grant("assemble-contexts")?;
    let _lease = crate::controller::gpu_lease(root, "assemble cohort context transforms")?;
    ensure!(
        output.is_absolute() && output.starts_with(root),
        "context output must be inside workspace"
    );
    ensure!(!output.exists(), "context version already exists");
    let registry: Option<crate::families::Registry> = registry_path.map(read_json).transpose()?;
    if let Some(registry) = &registry {
        registry.validate()?;
        ensure!(
            qualification || !registry.qualification,
            "qualification registry cannot produce independent contexts"
        );
    }
    let registry_receipt = registry_path.map(|path| -> Result<Value> { Ok(json!({"path":path.strip_prefix(root)?,"sha256":file_hash(path)?,"registry_id":registry.as_ref().unwrap().registry_id})) }).transpose()?;
    let jobs: GpJobs = read_json(jobs_path)?;
    ensure!(
        jobs.schema == "parallel.gp_jobs.v1"
            && !jobs.jobs.is_empty()
            && (qualification || !jobs.qualification),
        "invalid encoding cohort or qualification"
    );
    let mut prepared = BTreeMap::new();
    let mut registry_members = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut component_receipts = Vec::new();
    let mut h = Vec::new();
    let mut f = Vec::new();
    let mut t = Vec::new();
    let mut training = Vec::new();
    let mut families = BTreeMap::<String, usize>::new();
    for (j, job) in jobs.jobs.iter().enumerate() {
        if !prepared.contains_key(&job.prepared) {
            let path = resolve(root, &job.prepared)?.join("prepared.json");
            ensure!(
                file_hash(&path)? == job.prepared_sha256,
                "prepared cohort changed"
            );
            let p: Prepared = read_json(&path)?;
            ensure!(
                p.family_partition_frozen || qualification || registry.is_some(),
                "family partition must be frozen before clean contexts"
            );
            if let Some(registry) = &registry {
                registry_members.insert(job.prepared.clone(), registered_prepared(registry, &p)?);
            }
            prepared.insert(job.prepared.clone(), p);
        }
        let p = &prepared[&job.prepared];
        ensure!(
            seen.insert((p.dataset.clone(), job.cut)),
            "duplicate dataset/cut in context assembly"
        );
        let cut = p
            .cuts
            .iter()
            .find(|c| c.index == job.cut)
            .context("cut missing")?;
        let suffix = Path::new(&p.dataset).join(format!("cut-{}", job.cut));
        let mut bindings = serde_json::Map::new();
        for (name, root_dir, receipt, file, dimensions, destination) in [
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
            let dir = root_dir.join(&suffix);
            let receipt_path = dir.join(receipt);
            let meta: Value = read_json(&receipt_path)?;
            ensure!(
                meta["dataset"] == p.dataset
                    && meta["family"] == p.family
                    && meta["cut"] == job.cut
                    && meta["prepared_sha256"] == job.prepared_sha256
                    && meta["final_test_targets_used"] == false
                    && meta["shape"] == json!([dimensions]),
                "component receipt does not bind this exact prepared cut: {}",
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
        component_receipts.push(Value::Object(bindings));
        if p.meta_partition == "meta_train" {
            training.push((j as i64, p.family.clone()));
            *families.entry(p.family.clone()).or_default() += 1;
        }
    }
    ensure!(
        training.len() >= 1024 && families.len() >= 2,
        "1024-component projection needs at least 1024 meta-training views from multiple families"
    );
    let final_output = output;
    let staging = final_output.with_file_name(format!(
        ".{}.staging",
        final_output
            .file_name()
            .context("context output name missing")?
            .to_string_lossy()
    ));
    if staging.exists() {
        let old: Value = read_json(&staging.join("build.json"))?;
        ensure!(
            old["schema"] == "parallel.context_build.v1",
            "unknown context staging directory"
        );
        let retired = staging.with_extension(format!(
            "retired-{}-{}",
            crate::telemetry::now(),
            std::process::id()
        ));
        ensure!(!retired.exists(), "context staging retirement collision");
        fs::rename(&staging, &retired)?;
        write_json(
            &retired.join("retirement.json"),
            &json!({"reason":"previous incomplete context assembly; preserved before retry","timestamp":crate::telemetry::now()}),
        )?;
    }
    let output = staging.as_path();
    fs::create_dir_all(output)?;
    write_json(
        &output.join("build.json"),
        &json!({"schema":"parallel.context_build.v1","jobs_sha256":file_hash(jobs_path)?,"registry":registry_receipt,"output":final_output.strip_prefix(root)?,"qualification":qualification}),
    )?;
    let device = Device::Cuda(0);
    tch::set_num_threads(2);
    let n = jobs.jobs.len() as i64;
    let ht = Tensor::from_slice(&h).view([n, 512]).to_device(device);
    let ft = Tensor::from_slice(&f).view([n, 2048]).to_device(device);
    let tt = Tensor::from_slice(&t).view([n, 512]).to_device(device);
    let ids =
        Tensor::from_slice(&training.iter().map(|x| x.0).collect::<Vec<_>>()).to_device(device);
    let weights = Tensor::from_slice(
        &training
            .iter()
            .map(|(_, family)| 1.0 / (families.len() * families[family]) as f64)
            .collect::<Vec<_>>(),
    )
    .to_device(device)
    .unsqueeze(1);
    let moments = |x: &Tensor| {
        let train = x.index_select(0, &ids).to_kind(Kind::Double);
        let mean = (&train * &weights).sum_dim_intlist(&[0i64][..], false, Kind::Double);
        let scale = ((&train - &mean).square() * &weights)
            .sum_dim_intlist(&[0i64][..], false, Kind::Double)
            .sqrt()
            .clamp_min(1e-6);
        (mean.to_kind(Kind::Float), scale.to_kind(Kind::Float))
    };
    let train_f = ft.index_select(0, &ids).to_kind(Kind::Double);
    let center = (&train_f * &weights).sum_dim_intlist(&[0i64][..], false, Kind::Double);
    let centered = (&train_f - &center) * weights.sqrt();
    let covariance = centered.transpose(0, 1).matmul(&centered);
    let (eigenvalues, eigenvectors) = covariance.linalg_eigh("U");
    let mut projection = eigenvectors.narrow(1, 1024, 1024).flip([1]).transpose(0, 1);
    let max_index = projection.abs().argmax(1, true);
    let signs = projection.gather(1, &max_index, false).sign();
    projection *= signs;
    let projection = projection.to_kind(Kind::Float);
    let center = center.to_kind(Kind::Float);
    let z = (&ft - &center).matmul(&projection.transpose(0, 1));
    let (h_mean, h_scale) = moments(&ht);
    let (z_mean, z_scale) = moments(&z);
    let (t_mean, t_scale) = moments(&tt);
    fs::create_dir_all(output)?;
    let projection_path = output.join("transforms.ot");
    Tensor::save_multi(
        &[
            ("projection", &projection.to_device(Device::Cpu)),
            ("foundation_mean", &center.to_device(Device::Cpu)),
            ("hyperion_mean", &h_mean.to_device(Device::Cpu)),
            ("hyperion_scale", &h_scale.to_device(Device::Cpu)),
            ("projected_mean", &z_mean.to_device(Device::Cpu)),
            ("projected_scale", &z_scale.to_device(Device::Cpu)),
            ("target_mean", &t_mean.to_device(Device::Cpu)),
            ("target_scale", &t_scale.to_device(Device::Cpu)),
        ],
        &projection_path,
    )?;
    fs::File::open(&projection_path)?.sync_all()?;
    write_json(&output.join("registry-members.json"), &registry_members)?;
    let spec = json!({"schema":"parallel.context_assembly.v1","qualification":qualification,"selection_eligible":!qualification,"jobs_sha256":file_hash(jobs_path)?,"transforms_sha256":file_hash(&projection_path)?,"registry":registry_receipt,"registry_members_sha256":file_hash(&output.join("registry-members.json"))?,"meta_training_families":families,"meta_training_views":training.len(),"family_weighting":"equal mass per original source family","dimensions":{"hyperion":512,"foundation":1024,"target":512},"foundation_raw_dimensions":2048,"scale_floor":1e-6,"clip_abs":32.0,"numerical_backend":"Rust/LibTorch 2.7 CUDA; FP64 weighted covariance and eigensystem; FP32 output","explained_fraction":eigenvalues.narrow(0,1024,1024).sum(Kind::Double).double_value(&[])/eigenvalues.sum(Kind::Double).double_value(&[]),"final_test_targets_used":false});
    write_json(&output.join("assembly.json"), &spec)?;
    let spec_sha = file_hash(&output.join("assembly.json"))?;
    let normalized = Tensor::cat(
        &[
            (&ht - &h_mean) / h_scale,
            (&z - z_mean) / z_scale,
            (&tt - t_mean) / t_scale,
        ],
        1,
    )
    .clamp(-32.0, 32.0);
    ensure!(
        normalized.isfinite().all().int64_value(&[]) != 0,
        "nonfinite assembled context"
    );
    let values: Vec<f32> = Vec::try_from(normalized.to_device(Device::Cpu).flatten(0, -1))?;
    let mut index = Vec::new();
    for ((job, values), bindings) in jobs
        .jobs
        .iter()
        .zip(values.chunks_exact(2048))
        .zip(component_receipts)
    {
        let p = &prepared[&job.prepared];
        let cut = p.cuts.iter().find(|c| c.index == job.cut).unwrap();
        let dir = output.join(&p.dataset);
        fs::create_dir_all(&dir)?;
        let path = dir.join(format!("cut-{}.f32", job.cut));
        let bytes = values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>();
        atomic_write(&path, &bytes)?;
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
            encoder_spec_sha256: spec_sha.clone(),
            numerical_backend: "Rust/LibTorch 2.7 CUDA FP32 assembled frozen components".into(),
            dataset_dimensions: 1536,
            target_dimensions: 512,
            output_sha256: hash(&bytes),
        };
        let components_sha = hash(&serde_json::to_vec(&bindings)?);
        let mut meta = serde_json::to_value(&encoding)?;
        meta["components"] = bindings;
        meta["components_sha256"] = components_sha.clone().into();
        meta["registry"] = registry_receipt.clone().into();
        write_json(&path.with_extension("json"), &meta)?;
        let public_path = final_output
            .join(&p.dataset)
            .join(format!("cut-{}.f32", job.cut));
        index.push(json!({"dataset":p.dataset,"family":p.family,"cut":job.cut,"partition":p.meta_partition,"path":public_path.strip_prefix(root).context("contexts must be in workspace")?,"sha256":encoding.output_sha256,"prepared":job.prepared,"prepared_sha256":job.prepared_sha256,"components_sha256":components_sha,"receipt_sha256":file_hash(&path.with_extension("json"))?}));
    }
    write_json(
        &output.join("index.json"),
        &json!({"schema":"parallel.context_index.v1","assembly_sha256":spec_sha,"assembly":final_output.join("assembly.json").strip_prefix(root)?,"qualification":qualification,"registry":registry_receipt,"dimensions":2048,"encodings":index}),
    )?;
    fs::File::open(output)?.sync_all()?;
    fs::rename(output, final_output)?;
    fs::File::open(final_output.parent().context("context parent missing")?)?.sync_all()?;
    Ok(())
}
