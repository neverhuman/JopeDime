use super::*;

pub(crate) fn campaign_root(run_id: &str, dataset: &str) -> PathBuf {
    run_root(run_id).join(dataset)
}

pub(crate) fn generation_root(root: &Path, generation: usize) -> PathBuf {
    root.join("generations")
        .join(format!("generation-{generation:06}"))
}

pub(crate) fn immutable_or_equal(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        if fs::read(path)? != bytes {
            bail!("immutable artifact collision at {}", path.display());
        }
        return Ok(());
    }
    immutable_write(path, bytes)
}

pub(crate) fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<String> {
    let bytes = canonical_json(value)?;
    immutable_or_equal(path, &bytes)?;
    Ok(sha256_bytes(&bytes))
}

pub(crate) fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path)?).with_context(|| format!("parse {}", path.display()))
}

pub(crate) fn read_cohort() -> Result<(FrozenCohort, String)> {
    let manifest: FrozenCohort =
        serde_json::from_slice(COHORT_BYTES).context("parse frozen Binary42 cohort")?;
    if manifest.schema != "JopeBinary42FrozenCohort.v1"
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.track_id != "tabular-binary-v1"
        || manifest.datasets.len() != 42
        || manifest
            .datasets
            .iter()
            .enumerate()
            .any(|(order, dataset)| dataset.order != order)
        || manifest
            .datasets
            .iter()
            .filter(|dataset| dataset.leader_auc == 1.0)
            .count()
            != 6
        || manifest.datasets.iter().any(|dataset| {
            if dataset.leader_auc.to_bits() == 1.0f64.to_bits() {
                dataset.win_policy != "exact_auc_1_tie"
            } else {
                dataset.win_policy != "strictly_greater_auc"
            }
        })
    {
        bail!("frozen Binary42 cohort contract mismatch");
    }
    let ids = manifest
        .datasets
        .iter()
        .map(|dataset| dataset.id.as_str())
        .collect::<BTreeSet<_>>();
    if ids.len() != 42 {
        bail!("frozen Binary42 cohort contains duplicate datasets");
    }
    let ilpd = manifest
        .datasets
        .iter()
        .find(|dataset| dataset.id == ILPD)
        .context("frozen Binary42 cohort omits ILPD")?;
    if ilpd.train_rows != 485
        || ilpd.test_rows != 98
        || ilpd.features != 10
        || ilpd.train_sha256 != "f2a030306747738f28cb60c9f2f3fe1591724427ee63cb9d0bf8fb43b75cc114"
        || ilpd.test_sha256 != "0b86467f1692851213f7adbf6194e91c2eda39ff3e340a4ef81805b53e388aac"
        || ilpd.leader_method != "pytorch_tabular_fttransformer_epochs5"
        || ilpd.leader_auc.to_bits() != 0.7681159420289855f64.to_bits()
        || ilpd.leader_report_sha256
            != "5018f76223d4445f87960a076c9766bfe4140bb43d8b8bf9ae68ebb5ac85bad4"
    {
        bail!("frozen ILPD binding differs from the pilot contract");
    }
    Ok((manifest, sha256_bytes(COHORT_BYTES)))
}

pub(crate) fn selected_dataset(dataset: &str) -> Result<(FrozenDataset, String)> {
    if dataset != ILPD {
        bail!("the Binary42 pilot is frozen to ILPD ({ILPD}); no other dataset may launch");
    }
    let (cohort, sha256) = read_cohort()?;
    Ok((
        cohort
            .datasets
            .into_iter()
            .find(|record| record.id == dataset)
            .context("selected dataset is absent from Binary42")?,
        sha256,
    ))
}

pub(crate) fn cohort_dataset(dataset: &str) -> Result<(FrozenDataset, String)> {
    let (cohort, sha256) = read_cohort()?;
    Ok((
        cohort
            .datasets
            .into_iter()
            .find(|record| record.id == dataset)
            .with_context(|| format!("dataset {dataset} is absent from frozen Binary42"))?,
        sha256,
    ))
}

pub(crate) fn admission_order(cohort: &FrozenCohort) -> Vec<FrozenDataset> {
    let mut datasets = cohort.datasets.clone();
    datasets.sort_by(|left, right| {
        let left_cost = (left.train_rows as u128) * (left.features as u128);
        let right_cost = (right.train_rows as u128) * (right.features as u128);
        right_cost
            .cmp(&left_cost)
            .then_with(|| left.order.cmp(&right.order))
    });
    datasets
}

pub(crate) fn verify_dataset_binding(spec: &FrozenDataset) -> Result<()> {
    let train_path = Path::new(&spec.train_path);
    let test_path = Path::new(&spec.test_path);
    let report_path = Path::new(&spec.leader_report_path);
    if sha256_file(train_path)? != spec.train_sha256
        || sha256_file(test_path)? != spec.test_sha256
        || sha256_file(report_path)? != spec.leader_report_sha256
    {
        bail!(
            "Binary42 dataset or leader report hash mismatch for {}",
            spec.id
        );
    }
    let train = read_binary_csv(train_path)?;
    let test = read_binary_csv(test_path)?;
    if train.x.len() != spec.train_rows
        || test.x.len() != spec.test_rows
        || train.features != spec.features
        || test.features != spec.features
    {
        bail!("Binary42 dataset shape mismatch for {}", spec.id);
    }
    Ok(())
}
