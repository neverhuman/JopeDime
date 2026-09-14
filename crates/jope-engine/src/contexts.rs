//! Fit cohort transforms on meta-training families, then persist full contexts.
use crate::{encoding::Encoding, manifest::*, prepare::Prepared, service::GpJobs};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use tch::{Device, Kind, Tensor};
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

fn registered_prepared(registry: &crate::families::Registry, p: &Prepared) -> Result<Value> {
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

fn immutable_json(path: &Path, value: &Value) -> Result<()> {
    if path.exists() {
        ensure!(
            read_json::<Value>(path)? == *value,
            "immutable context receipt changed: {}",
            path.display()
        );
        Ok(())
    } else {
        write_json(path, value)
    }
}

fn load_transforms(path: &Path, device: Device) -> Result<BTreeMap<String, Tensor>> {
    let tensors: BTreeMap<_, _> = Tensor::load_multi_with_device(path, device)?
        .into_iter()
        .collect();
    for (name, shape) in [
        ("projection", vec![1024, 2048]),
        ("foundation_mean", vec![2048]),
        ("hyperion_mean", vec![512]),
        ("hyperion_scale", vec![512]),
        ("projected_mean", vec![1024]),
        ("projected_scale", vec![1024]),
        ("target_mean", vec![512]),
        ("target_scale", vec![512]),
    ] {
        let tensor = tensors
            .get(name)
            .with_context(|| format!("frozen transform {name} absent"))?;
        ensure!(
            tensor.size() == shape
                && tensor.kind() == Kind::Float
                && tensor.isfinite().all().int64_value(&[]) != 0,
            "invalid frozen transform {name}"
        );
        if name.ends_with("scale") {
            ensure!(
                tensor.gt(0).all().int64_value(&[]) != 0,
                "nonpositive frozen scale {name}"
            );
        }
    }
    Ok(tensors)
}

fn normalize_frozen(
    h: &Tensor,
    f: &Tensor,
    t: &Tensor,
    transforms: &BTreeMap<String, Tensor>,
    clip: f64,
) -> Tensor {
    let z = (f - &transforms["foundation_mean"]).matmul(&transforms["projection"].transpose(0, 1));
    Tensor::cat(
        &[
            (h - &transforms["hyperion_mean"]) / &transforms["hyperion_scale"],
            (&z - &transforms["projected_mean"]) / &transforms["projected_scale"],
            (t - &transforms["target_mean"]) / &transforms["target_scale"],
        ],
        1,
    )
    .clamp(-clip, clip)
}

/// Choose the bounded training panel using index/registry metadata only.
/// In particular, never open a candidate reference to discover its partition.
fn training_parity_rows<'a>(
    original: &'a Value,
    registry: &crate::families::Registry,
) -> Result<Vec<&'a Value>> {
    let mut families = BTreeMap::<String, BTreeMap<(String, u64), &Value>>::new();
    for row in original["encodings"]
        .as_array()
        .context("original context rows absent")?
    {
        if row["partition"] != "meta_train" {
            continue;
        }
        let name = row["family"].as_str().context("reference family absent")?;
        let family = registry
            .families
            .get(name)
            .context("reference family is not registered")?;
        ensure!(
            family.partition == "meta_train" && family.holds.is_empty(),
            "training reference has protected or held registry membership"
        );
        let dataset = row["dataset"]
            .as_str()
            .context("reference dataset absent")?;
        ensure!(
            family
                .datasets
                .iter()
                .any(|d| d.dataset == dataset && d.preparation_eligible),
            "training reference dataset is not admitted"
        );
        let key = (
            dataset.to_string(),
            row["cut"].as_u64().context("reference cut absent")?,
        );
        ensure!(
            families
                .entry(name.into())
                .or_default()
                .insert(key, row)
                .is_none(),
            "duplicate training reference"
        );
    }
    let selected: Vec<_> = families
        .values()
        .filter(|rows| rows.len() >= 4)
        .take(2)
        .flat_map(|rows| rows.values().take(4).copied())
        .collect();
    ensure!(
        selected.len() == 8,
        "parity requires four deterministic training cuts from each of two families"
    );
    Ok(selected)
}

fn verify_original_contexts(
    root: &Path,
    original: &Value,
    rows: &[&Value],
    transforms: &BTreeMap<String, Tensor>,
    device: Device,
    clip: f64,
) -> Result<Value> {
    let (mut h, mut f, mut t, mut expected) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for row in rows {
        let path = resolve(
            root,
            Path::new(
                row["path"]
                    .as_str()
                    .context("original context path absent")?,
            ),
        )?;
        let metadata_path = path.with_extension("json");
        let metadata: Value = read_json(&metadata_path)?;
        let encoding: Encoding = serde_json::from_value(metadata.clone())?;
        if let Some(expected) = row["receipt_sha256"].as_str() {
            ensure!(
                file_hash(&metadata_path)? == expected,
                "original context metadata changed"
            );
        }
        ensure!(
            encoding.partition == "meta_train"
                && encoding.family == row["family"]
                && encoding.dataset == row["dataset"]
                && encoding.cut == row["cut"]
                && encoding.prepared_sha256 == row["prepared_sha256"],
            "selected reference metadata changed membership or identity"
        );
        let bytes = fs::read(&path)?;
        ensure!(
            bytes.len() == 2048 * 4
                && hash(&bytes) == row["sha256"]
                && encoding.output_sha256 == row["sha256"]
                && encoding.encoder_spec_sha256 == original["assembly_sha256"],
            "original context reference changed"
        );
        expected.extend(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        );
        for (name, file, dimensions, destination) in [
            ("hyperion", "hyperion.f32", 512, &mut h),
            ("foundation", "foundation_raw.f32", 2048, &mut f),
            ("target", "target.f32", 512, &mut t),
        ] {
            destination.extend(reference_component(
                root, &metadata, &encoding, name, file, dimensions,
            )?);
        }
    }
    let n = rows.len() as i64;
    let actual = tch::no_grad(|| {
        normalize_frozen(
            &Tensor::from_slice(&h).view([n, 512]).to_device(device),
            &Tensor::from_slice(&f).view([n, 2048]).to_device(device),
            &Tensor::from_slice(&t).view([n, 512]).to_device(device),
            transforms,
            clip,
        )
    });
    let expected = Tensor::from_slice(&expected)
        .view([n, 2048])
        .to_device(device);
    let error = (actual - &expected).abs().max().double_value(&[]);
    Ok(
        json!({"schema":"parallel.frozen_context_apply_parity.v1","passed":error.is_finite() && error<=3e-4,"max_abs_error":error,"tolerance":3e-4,"reference_views":rows.len(),"matrix_rows":n,"dimensions":2048,"assembly_sha256":original["assembly_sha256"],"batch_protocol":"four deterministic training cuts per family; two families; selected before file reads","reference_partition":"meta_train","backend":format!("Rust/LibTorch {:?} FP32", device),"projection_refitted":false,"reference_vectors_overwritten":false}),
    )
}

/// A different component root is not evidence for the original saved vector.
/// Validate all metadata before opening a selected component's numerical payload.
fn reference_component(
    root: &Path,
    metadata: &Value,
    encoding: &Encoding,
    name: &str,
    file: &str,
    dimensions: usize,
) -> Result<Vec<f32>> {
    let binding = &metadata["components"][name];
    let receipt = resolve(
        root,
        Path::new(
            binding["receipt"]
                .as_str()
                .context("original component binding absent; unbound fallback is unavailable")?,
        ),
    )?;
    let payload = resolve(
        root,
        Path::new(
            binding["signature"]
                .as_str()
                .context("reference component signature absent")?,
        ),
    )?;
    ensure!(
        receipt.file_name().and_then(|n| n.to_str()) == Some(&format!("{name}.json"))
            && payload.file_name().and_then(|n| n.to_str()) == Some(file)
            && receipt.parent() == payload.parent(),
        "reference component path mismatch"
    );
    let raw = fs::read(&receipt)?;
    ensure!(
        hash(&raw) == binding["receipt_sha256"],
        "original bound component receipt changed"
    );
    let meta: Value = serde_json::from_slice(&raw)?;
    ensure!(
        meta["dataset"] == encoding.dataset
            && meta["family"] == encoding.family
            && meta["cut"] == encoding.cut
            && meta["prepared_sha256"] == encoding.prepared_sha256
            && meta["training_sha256"] == encoding.training_sha256
            && meta["preprocessing_sha256"] == encoding.preprocessing_sha256
            && meta["final_test_targets_used"] == false
            && meta["shape"] == json!([dimensions])
            && meta["spec_sha256"].is_string()
            && meta["spec_sha256"] == binding["spec_sha256"]
            && meta["signature_sha256"] == binding["signature_sha256"],
        "reference component receipt does not bind the exact training cut"
    );
    let bytes = fs::read(&payload)?;
    ensure!(
        bytes.len() == dimensions * 4 && hash(&bytes) == binding["signature_sha256"],
        "original bound component bytes changed"
    );
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite reference component"
    );
    Ok(values)
}

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

/// Merge indices by reference. No vectors, source rows, or model assets are copied.
pub fn union(root: &Path, indexes: &[PathBuf], output: &Path) -> Result<()> {
    ensure!(
        !indexes.is_empty(),
        "context union requires at least one index"
    );
    output
        .strip_prefix(root)
        .context("context union must be inside workspace")?;
    let mut rows = BTreeMap::<(String, u64), Value>::new();
    let mut assembly = None;
    let mut qualification = false;
    let mut sources = Vec::new();
    for path in indexes {
        let index: Value = read_json(path)?;
        ensure!(
            index["schema"] == "parallel.context_index.v1" && index["dimensions"] == 2048,
            "invalid source context index"
        );
        let sha = index["assembly_sha256"]
            .as_str()
            .context("assembly hash absent")?
            .to_string();
        if let Some(old) = &assembly {
            ensure!(*old == sha, "context union has different frozen assemblies");
        } else {
            assembly = Some(sha.clone());
        }
        qualification |= index["qualification"]
            .as_bool()
            .context("qualification flag absent")?;
        for row in index["encodings"]
            .as_array()
            .context("context rows absent")?
        {
            let key = (
                row["dataset"]
                    .as_str()
                    .context("dataset absent")?
                    .to_string(),
                row["cut"].as_u64().context("cut absent")?,
            );
            let payload = resolve(
                root,
                Path::new(row["path"].as_str().context("context path absent")?),
            )?;
            ensure!(
                file_hash(&payload)? == row["sha256"],
                "source context bytes changed"
            );
            let encoding: Encoding = read_json(&payload.with_extension("json"))?;
            if let Some(receipt_sha) = row["receipt_sha256"].as_str() {
                ensure!(
                    file_hash(&payload.with_extension("json"))? == receipt_sha,
                    "source context metadata changed"
                );
            }
            ensure!(
                encoding.encoder_spec_sha256 == sha
                    && encoding.output_sha256 == row["sha256"]
                    && encoding.prepared_sha256 == row["prepared_sha256"]
                    && encoding.dataset == key.0
                    && encoding.cut as u64 == key.1
                    && encoding.family == row["family"]
                    && encoding.partition == row["partition"],
                "source context receipt mismatch"
            );
            if let Some(old) = rows.get(&key) {
                ensure!(*old == *row, "conflicting duplicate context in union");
            } else {
                rows.insert(key, row.clone());
            }
        }
        sources.push(json!({"path":path.strip_prefix(root)?,"sha256":file_hash(path)?}));
    }
    immutable_json(
        output,
        &json!({"schema":"parallel.context_index.v1","assembly_sha256":assembly.context("assembly absent")?,"qualification":qualification,"dimensions":2048,"sources":sources,"copied_vector_bytes":0,"encodings":rows.into_values().collect::<Vec<_>>()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_assembly_accepts_old_metadata_only_with_preserved_assignment() {
        let registry: crate::families::Registry = serde_json::from_value(json!({
            "schema":"parallel.family_registry.v1","seed":SEED,"registry_id":"test-registry","qualification":true,"full_inventory_complete":false,"independent_selection_eligible":false,"parent":null,"audit":{"path":"audit.json","sha256":"audit"},"decisions":[],"reviews":[],"prepared_inputs":[],"held_review_datasets":{},
            "families":{"family":{"family":"family","partition":"meta_train","assignment_method":"preserved_audit_assignment","assignment_hash":null,"bucket":null,"source_keys":["real-source"],"holds":[],"datasets":[{"dataset":"dataset","source_key":"real-source","audit_family":"family","metadata_sha256":"metadata","entity_review_sha256":null,"preparation_eligible":true,"eligibility_reason":"verified","prepared":[],"evidence":[]}]}}
        })).unwrap();
        registry.validate().unwrap();
        let mut prepared: Prepared = serde_json::from_value(json!({"schema":"parallel.prepared.v1","dataset":"dataset","family":"family","task":"regression","meta_partition":"meta_train","seed":SEED,"family_partition_frozen":false,"clean_evaluation":false,"clean_status":"qualification","original_train_sha256":"training","original_test_sha256":"sealed","selected_train_indices":[],"selected_test_indices":[],"target_last":true,"rows_total":0,"cuts":[]})).unwrap();
        assert!(prepared.family_registry_binding.is_none());
        registered_prepared(&registry, &prepared).unwrap();
        prepared.meta_partition = "meta_validation".into();
        assert!(registered_prepared(&registry, &prepared).is_err());
        prepared.meta_partition = "meta_train".into();
        let mut held = registry.clone();
        held.families
            .get_mut("family")
            .unwrap()
            .holds
            .insert("unresolved origin".into());
        assert!(registered_prepared(&held, &prepared).is_err());
        prepared.dataset = "unregistered".into();
        assert!(registered_prepared(&registry, &prepared).is_err());
    }

    #[test]
    fn parity_selection_is_training_only_bounded_and_registry_checked() {
        let mut families = serde_json::Map::new();
        let mut rows = Vec::new();
        for name in ["alpha", "beta", "gamma"] {
            families.insert(name.into(), json!({"family":name,"partition":"meta_train","assignment_method":"preserved_audit_assignment","assignment_hash":null,"bucket":null,"source_keys":[name],"holds":[],"datasets":[{"dataset":name,"source_key":name,"audit_family":name,"metadata_sha256":"metadata","entity_review_sha256":null,"preparation_eligible":true,"eligibility_reason":"verified","prepared":[],"evidence":[]}]}));
            for cut in (0..6).rev() {
                rows.push(json!({"dataset":name,"family":name,"cut":cut,"partition":"meta_train","path":"never-open-this"}));
            }
        }
        // A protected reference need not have a readable path or valid dataset.
        rows.push(json!({"partition":"meta_test","path":"protected-file-does-not-exist"}));
        rows.push(json!({"partition":"meta_validation","path":"protected-file-does-not-exist"}));
        let registry: crate::families::Registry = serde_json::from_value(json!({"schema":"parallel.family_registry.v1","seed":SEED,"registry_id":"test","qualification":true,"full_inventory_complete":false,"independent_selection_eligible":false,"parent":null,"audit":{"path":"audit","sha256":"audit"},"decisions":[],"reviews":[],"prepared_inputs":[],"held_review_datasets":{},"families":families})).unwrap();
        let original = json!({"encodings":rows});
        let selected = training_parity_rows(&original, &registry).unwrap();
        assert_eq!(selected.len(), 8);
        assert!(selected.iter().all(|row| row["partition"] == "meta_train"));
        assert_eq!(selected[0]["family"], "alpha");
        assert_eq!(selected[0]["cut"], 0);
        assert_eq!(selected[7]["family"], "beta");
        assert_eq!(selected[7]["cut"], 3);
        let mut protected = registry;
        protected.families.get_mut("alpha").unwrap().partition = "meta_test".into();
        assert!(training_parity_rows(&original, &protected).is_err());
    }

    #[test]
    fn reference_component_rejects_unbound_and_wrong_cut_before_payload_access() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let encoding: Encoding = serde_json::from_value(json!({
            "schema":"parallel.encoding.v1", "dataset":"dataset", "family":"family",
            "partition":"meta_train", "cut":2, "prepared_sha256":"prepared",
            "training_sha256":"training", "selected_training_rows_sha256":"rows",
            "preprocessing_sha256":"preprocessing", "encoder_spec_sha256":"assembly",
            "numerical_backend":"fixture", "dataset_dimensions":1536,
            "target_dimensions":512, "output_sha256":"context",
        }))
        .unwrap();
        let err = reference_component(root, &json!({}), &encoding, "target", "target.f32", 512)
            .unwrap_err();
        assert!(err.to_string().contains("unbound fallback"));
        let receipt = root.join("target.json");
        let mut meta = json!({"dataset":"dataset", "family":"family", "cut":3,
            "prepared_sha256":"prepared", "training_sha256":"training",
            "preprocessing_sha256":"preprocessing", "final_test_targets_used":false,
            "shape":[512], "spec_sha256":"spec", "signature_sha256":"unopened"});
        write_json(&receipt, &meta).unwrap();
        let mut metadata = json!({"components":{"target":{
            "receipt":"target.json", "receipt_sha256":file_hash(&receipt).unwrap(),
            "signature":"target.f32", "signature_sha256":"unopened", "spec_sha256":"spec",
        }}});
        let err = reference_component(root, &metadata, &encoding, "target", "target.f32", 512)
            .unwrap_err();
        assert!(err.to_string().contains("exact training cut"));
        assert!(!root.join("target.f32").exists());
        meta["cut"] = 2.into();
        let bytes = vec![0_u8; 512 * 4];
        fs::write(root.join("target.f32"), &bytes).unwrap();
        meta["signature_sha256"] = hash(&bytes).into();
        write_json(&receipt, &meta).unwrap();
        metadata["components"]["target"]["receipt_sha256"] = file_hash(&receipt).unwrap().into();
        metadata["components"]["target"]["signature_sha256"] = hash(&bytes).into();
        assert_eq!(
            reference_component(root, &metadata, &encoding, "target", "target.f32", 512).unwrap(),
            vec![0_f32; 512]
        );
        fs::write(root.join("target.f32"), vec![1_u8; 512 * 4]).unwrap();
        assert!(
            reference_component(root, &metadata, &encoding, "target", "target.f32", 512)
                .unwrap_err()
                .to_string()
                .contains("bytes changed")
        );
    }

    #[test]
    fn frozen_transform_applies_stored_statistics_without_refitting() {
        tch::set_num_threads(2);
        let cpu = (Kind::Float, Device::Cpu);
        let transforms = BTreeMap::from([
            (
                "projection".into(),
                Tensor::cat(
                    &[Tensor::eye(1024, cpu), Tensor::zeros([1024, 1024], cpu)],
                    1,
                ),
            ),
            ("foundation_mean".into(), Tensor::ones([2048], cpu)),
            ("hyperion_mean".into(), Tensor::ones([512], cpu)),
            ("hyperion_scale".into(), Tensor::full([512], 2., cpu)),
            ("projected_mean".into(), Tensor::ones([1024], cpu)),
            ("projected_scale".into(), Tensor::full([1024], 2., cpu)),
            ("target_mean".into(), Tensor::full([512], 3., cpu)),
            ("target_scale".into(), Tensor::full([512], 4., cpu)),
        ]);
        let output = normalize_frozen(
            &Tensor::full([2, 512], 3., cpu),
            &Tensor::full([2, 2048], 4., cpu),
            &Tensor::full([2, 512], 7., cpu),
            &transforms,
            32.,
        );
        assert_eq!(output.size(), [2, 2048]);
        assert_eq!((output - 1.).abs().max().double_value(&[]), 0.);
    }

    #[test]
    fn union_references_existing_vectors_and_rejects_changed_sources() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let mut indexes = Vec::new();
        for cut in 0..2 {
            let path = root.join(format!("saved-{cut}/cut-{cut}.f32"));
            atomic_write(&path, &vec![0u8; 2048 * 4]).unwrap();
            let encoding = Encoding {
                schema: "parallel.encoding.v1".into(),
                dataset: "real-source".into(),
                family: "source-family".into(),
                partition: "meta_train".into(),
                cut,
                prepared_sha256: "prepared".into(),
                training_sha256: "train".into(),
                selected_training_rows_sha256: "rows".into(),
                preprocessing_sha256: "pp".into(),
                encoder_spec_sha256: "frozen-assembly".into(),
                numerical_backend: "test".into(),
                dataset_dimensions: 1536,
                target_dimensions: 512,
                output_sha256: file_hash(&path).unwrap(),
            };
            write_json(&path.with_extension("json"), &encoding).unwrap();
            let index = root.join(format!("index-{cut}.json"));
            write_json(&index,&json!({"schema":"parallel.context_index.v1","assembly_sha256":"frozen-assembly","qualification":true,"dimensions":2048,"encodings":[{"dataset":encoding.dataset,"family":encoding.family,"partition":encoding.partition,"cut":cut,"path":path.strip_prefix(root).unwrap(),"sha256":encoding.output_sha256,"prepared":"prepared","prepared_sha256":"prepared"}]})).unwrap();
            indexes.push(index);
        }
        let output = root.join("union/index.json");
        union(root, &indexes, &output).unwrap();
        let merged: Value = read_json(&output).unwrap();
        assert_eq!(merged["encodings"].as_array().unwrap().len(), 2);
        assert_eq!(merged["copied_vector_bytes"], 0);
        assert_eq!(fs::read_dir(output.parent().unwrap()).unwrap().count(), 1);
        assert_eq!(merged["assembly_sha256"], "frozen-assembly");
        fs::write(root.join("saved-1/cut-1.f32"), b"changed").unwrap();
        assert!(union(root, &indexes, &root.join("changed.json")).is_err());
    }
}
