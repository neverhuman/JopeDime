use super::*;

pub(super) fn immutable_json(path: &Path, value: &Value) -> Result<()> {
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

pub(super) fn load_transforms(path: &Path, device: Device) -> Result<BTreeMap<String, Tensor>> {
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

pub(super) fn normalize_frozen(
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

pub(super) fn verify_original_contexts(
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
pub(super) fn reference_component(
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
