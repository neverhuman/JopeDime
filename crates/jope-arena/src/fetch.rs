//! Dataset fetch with SHA-256 verification (`/home/ubuntu/sotarena/src/main.rs::fetch`).
//!
//! Each reporter dataset (`data/<task>/<hash>/dataset.json`) names an HTTPS CSV per split with
//! its exact byte count and SHA-256. Files land in `<cache>/<task>/<hash>/{train,test}.csv`
//! (`SOTARENA_BENCHMARK_CACHE`, default `$JOPE_HOME/benchmarks/cache`); a cached file is
//! re-verified on every fetch and never trusted by name alone. The cache must lie outside the
//! reporter root.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use anyhow::{ensure, Context, Result};

use crate::reporter::{Config, DataFile, Dataset, DatasetCatalog};

/// Corpus counts the reporter prints and (for `--all`) asserts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CorpusSummary {
    /// Every dataset.
    pub datasets: usize,
    /// Datasets with both split URLs.
    pub downloadable_datasets: usize,
    /// Datasets without URLs.
    pub metadata_only_datasets: usize,
    /// Downloadable files.
    pub files: usize,
    /// Downloadable bytes.
    pub bytes: u64,
}

/// Summarise a dataset catalogue.
pub fn corpus_summary(datasets: &DatasetCatalog) -> CorpusSummary {
    let datasets_count = datasets.values().map(BTreeMap::len).sum();
    let downloadable = datasets
        .values()
        .flat_map(BTreeMap::values)
        .filter(|dataset| dataset.train.url.is_some() && dataset.test.url.is_some())
        .collect::<Vec<_>>();
    CorpusSummary {
        datasets: datasets_count,
        downloadable_datasets: downloadable.len(),
        metadata_only_datasets: datasets_count - downloadable.len(),
        files: downloadable.len() * 2,
        bytes: downloadable
            .iter()
            .map(|dataset| dataset.train.bytes + dataset.test.bytes)
            .sum(),
    }
}

/// Byte count and SHA-256 must both match.
pub fn verify_bytes(bytes: &[u8], expected: &DataFile) -> Result<()> {
    ensure!(
        bytes.len() as u64 == expected.bytes,
        "byte count mismatch: {} vs expected {}",
        bytes.len(),
        expected.bytes
    );
    let actual = crate::io::sha256_hex(bytes);
    ensure!(
        actual == expected.sha256,
        "SHA-256 mismatch: {actual} vs expected {}",
        expected.sha256
    );
    Ok(())
}

fn lexical_absolute(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::RootDir => normalized.push(Path::new("/")),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(value) => normalized.push(value),
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
        }
    }
    Ok(normalized)
}

/// The cache directory: `requested`, else [`crate::paths::benchmark_cache`]; never inside `root`.
pub fn external_cache(root: &Path, requested: Option<PathBuf>) -> Result<PathBuf> {
    let path = requested.unwrap_or_else(crate::paths::benchmark_cache);
    let repository = fs::canonicalize(root)
        .with_context(|| format!("reporter root {} does not exist", root.display()))?;
    let normalized = lexical_absolute(&path)?;
    ensure!(
        !normalized.starts_with(&repository),
        "cache directory {} must be outside the reporter root {}",
        normalized.display(),
        repository.display()
    );
    Ok(normalized)
}

/// Datasets selected by task / dataset id, downloadable ones only.
pub fn fetch_targets<'a>(
    config: &Config,
    datasets: &'a DatasetCatalog,
    task_id: Option<&'a str>,
    wanted: Option<&str>,
) -> Result<Vec<(&'a str, &'a Dataset)>> {
    ensure!(
        wanted.is_none() || task_id.is_some(),
        "--dataset requires --task"
    );
    if let Some(task) = task_id {
        ensure!(
            config.tasks.iter().any(|candidate| candidate.id == task),
            "unknown task {task}"
        );
    }
    if let (Some(task), Some(wanted)) = (task_id, wanted) {
        let dataset = datasets[task]
            .get(wanted)
            .or_else(|| {
                datasets[task]
                    .values()
                    .find(|dataset| dataset.dataset_id == wanted)
            })
            .with_context(|| format!("unknown dataset {wanted} in task {task}"))?;
        ensure!(
            dataset.train.url.is_some() && dataset.test.url.is_some(),
            "dataset {wanted} is metadata-only"
        );
        return Ok(vec![(task, dataset)]);
    }
    Ok(datasets
        .iter()
        .filter(|(task, _)| task_id.is_none_or(|wanted| wanted == task.as_str()))
        .flat_map(|(task_id, task_datasets)| {
            task_datasets
                .values()
                .filter(|dataset| dataset.train.url.is_some() && dataset.test.url.is_some())
                .map(move |dataset| (task_id.as_str(), dataset))
        })
        .collect())
}

/// How bytes are obtained for a URL (HTTP by default; tests inject a map).
pub trait Downloader {
    /// Fetch the whole body.
    fn get(&self, url: &str) -> Result<Vec<u8>>;
}

/// `ureq` GET.
pub struct HttpDownloader;

impl Downloader for HttpDownloader {
    fn get(&self, url: &str) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        ureq::get(url)
            .call()
            .with_context(|| format!("download failed: {url}"))?
            .into_reader()
            .read_to_end(&mut bytes)?;
        Ok(bytes)
    }
}

/// What a fetch did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FetchReport {
    /// Datasets verified.
    pub datasets: usize,
    /// Files verified.
    pub files: usize,
    /// Bytes verified.
    pub bytes: u64,
    /// Files downloaded (the rest were already cached).
    pub downloaded: usize,
    /// Metadata-only datasets skipped.
    pub metadata_only: usize,
}

/// Fetch and verify the selected datasets into `cache`.
pub fn fetch_into(
    downloader: &dyn Downloader,
    selected: &[(&str, &Dataset)],
    cache: &Path,
    log: &mut dyn FnMut(&str),
) -> Result<FetchReport> {
    let mut report = FetchReport {
        datasets: selected.len(),
        ..FetchReport::default()
    };
    for (task_id, dataset) in selected {
        for (split, source) in [("train", &dataset.train), ("test", &dataset.test)] {
            let url = source
                .url
                .as_ref()
                .context("selected dataset is not downloadable")?;
            let destination = cache
                .join(task_id)
                .join(&dataset.dataset_hash)
                .join(format!("{split}.csv"));
            let bytes = if destination.exists() {
                fs::read(&destination)?
            } else {
                let bytes = downloader.get(url)?;
                verify_bytes(&bytes, source)
                    .with_context(|| format!("{url} ({} {split})", dataset.dataset_hash))?;
                fs::create_dir_all(destination.parent().expect("cache path has a parent"))?;
                fs::write(&destination, &bytes)?;
                report.downloaded += 1;
                bytes
            };
            verify_bytes(&bytes, source).with_context(|| {
                format!(
                    "cached {} does not match its manifest",
                    destination.display()
                )
            })?;
            report.files += 1;
            report.bytes += source.bytes;
            log(&format!("verified {} {split}", dataset.dataset_hash));
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elo::Direction;
    use crate::reporter::fixtures::{data_file, test_config};
    use std::collections::HashMap;

    struct MapDownloader(HashMap<&'static str, Vec<u8>>);

    impl Downloader for MapDownloader {
        fn get(&self, url: &str) -> Result<Vec<u8>> {
            self.0
                .get(url)
                .cloned()
                .with_context(|| format!("no fixture for {url}"))
        }
    }

    fn dataset(hash: &str, train: DataFile, test: DataFile) -> Dataset {
        Dataset {
            schema: "sotarena.dataset".to_owned(),
            dataset_hash: hash.to_owned(),
            dataset_id: hash.to_owned(),
            task: "binary".to_owned(),
            metric: "roc_auc".to_owned(),
            train,
            test,
        }
    }

    #[test]
    fn fetch_verifies_downloads_and_the_cache() {
        let train_bytes = b"1,2\n3,4\n".to_vec();
        let test_bytes = b"5,6\n".to_vec();
        let mut train = data_file(Some("https://example.test/train.csv"));
        train.sha256 = crate::io::sha256_hex(&train_bytes);
        train.bytes = train_bytes.len() as u64;
        let mut test = data_file(Some("https://example.test/test.csv"));
        test.sha256 = crate::io::sha256_hex(&test_bytes);
        test.bytes = test_bytes.len() as u64;
        let catalog: DatasetCatalog = BTreeMap::from([(
            "binary".to_owned(),
            BTreeMap::from([
                (
                    "0123456789abcdef".to_owned(),
                    dataset("0123456789abcdef", train, test),
                ),
                (
                    "fedcba9876543210".to_owned(),
                    dataset("fedcba9876543210", data_file(None), data_file(None)),
                ),
            ]),
        )]);
        let config = test_config(Direction::Maximize);
        let summary = corpus_summary(&catalog);
        assert_eq!(
            summary,
            CorpusSummary {
                datasets: 2,
                downloadable_datasets: 1,
                metadata_only_datasets: 1,
                files: 2,
                bytes: 12
            }
        );
        let selected = fetch_targets(&config, &catalog, Some("binary"), None).unwrap();
        assert_eq!(selected.len(), 1);
        assert!(
            fetch_targets(&config, &catalog, Some("binary"), Some("fedcba9876543210")).is_err()
        );
        assert!(fetch_targets(&config, &catalog, None, Some("x")).is_err());

        let downloader = MapDownloader(HashMap::from([
            ("https://example.test/train.csv", train_bytes.clone()),
            ("https://example.test/test.csv", test_bytes.clone()),
        ]));
        let cache = tempfile::tempdir().unwrap();
        let mut lines = Vec::new();
        let report = fetch_into(&downloader, &selected, cache.path(), &mut |l| {
            lines.push(l.to_owned())
        })
        .unwrap();
        assert_eq!(
            report,
            FetchReport {
                datasets: 1,
                files: 2,
                bytes: 12,
                downloaded: 2,
                metadata_only: 0
            }
        );
        assert_eq!(
            lines,
            [
                "verified 0123456789abcdef train",
                "verified 0123456789abcdef test"
            ]
        );
        assert_eq!(
            fs::read(cache.path().join("binary/0123456789abcdef/train.csv")).unwrap(),
            train_bytes
        );
        // Second run: nothing downloaded, everything re-verified.
        let empty = MapDownloader(HashMap::new());
        let report = fetch_into(&empty, &selected, cache.path(), &mut |_| {}).unwrap();
        assert_eq!(report.downloaded, 0);
        assert_eq!(report.files, 2);
        // A corrupted cache file is refused, not silently reused.
        fs::write(
            cache.path().join("binary/0123456789abcdef/test.csv"),
            b"tampered",
        )
        .unwrap();
        let error = fetch_into(&empty, &selected, cache.path(), &mut |_| {}).unwrap_err();
        assert!(
            error.to_string().contains("does not match its manifest"),
            "{error}"
        );
        // A wrong download is refused before it is cached.
        let bad = MapDownloader(HashMap::from([
            ("https://example.test/train.csv", b"wrong".to_vec()),
            ("https://example.test/test.csv", test_bytes),
        ]));
        let fresh = tempfile::tempdir().unwrap();
        assert!(fetch_into(&bad, &selected, fresh.path(), &mut |_| {}).is_err());
        assert!(!fresh
            .path()
            .join("binary/0123456789abcdef/train.csv")
            .exists());
    }

    #[test]
    fn cache_must_be_outside_the_root() {
        let root = tempfile::tempdir().unwrap();
        assert!(external_cache(root.path(), Some(root.path().join("cache"))).is_err());
        let elsewhere = tempfile::tempdir().unwrap();
        assert_eq!(
            external_cache(root.path(), Some(elsewhere.path().join("x/../y"))).unwrap(),
            lexical_absolute(&elsewhere.path().join("y")).unwrap()
        );
    }
}
