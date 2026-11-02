//! Host, path, and worker configuration for the legacy GP factory.
//!
//! Prime-owned paths come from the caller's resolved prime configuration so this module does
//! not duplicate the Python `prime_config` parser or bake host-specific filesystem state into the
//! engine.

use anyhow::{bail, Result};
use std::path::PathBuf;

pub const SELF_ROWS_GLOB: &str = "self_rows_*_g*.jsonl";
pub const SELF_ROWS_GLOBS: [&str; 2] = ["self_rows_g*.jsonl", SELF_ROWS_GLOB];
pub const MODEL_SEED_RECENT: usize = 3;
pub const GP_HOST: &str = "xbabe2";
pub const GP_HOST2: &str = "xbabe3";
pub const GP_HOST2_WORKERS: usize = 5;
pub const MODEL_NAME: &str = "jope_gp_v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostMap(Vec<(String, usize)>);

impl HostMap {
    pub fn get(&self, host: &str) -> Option<usize> {
        self.0
            .iter()
            .find_map(|(name, workers)| (name == host).then_some(*workers))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, usize)> {
        self.0
            .iter()
            .map(|(host, workers)| (host.as_str(), *workers))
    }

    fn set(&mut self, host: String, workers: usize) {
        if let Some((_, current)) = self.0.iter_mut().find(|(name, _)| name == &host) {
            *current = workers;
        } else {
            self.0.push((host, workers));
        }
    }
}

fn python_int(value: &str) -> Result<usize> {
    let value = value.trim();
    let bytes = value.as_bytes();
    for (index, byte) in bytes.iter().enumerate() {
        if *byte == b'_' {
            if index == 0
                || index + 1 == bytes.len()
                || !bytes[index - 1].is_ascii_digit()
                || !bytes[index + 1].is_ascii_digit()
            {
                bail!("invalid literal for int() with base 10: '{value}'");
            }
        }
    }
    let compact = value.replace('_', "");
    let parsed = compact
        .parse::<i128>()
        .map_err(|_| anyhow::anyhow!("invalid literal for int() with base 10: '{value}'"))?;
    Ok(parsed.max(0).try_into().unwrap_or(usize::MAX))
}

/// Parse `JOPE_GP_HOSTS`, retaining Python dict insertion order and duplicate-key behavior.
pub fn gp_host_map(spec: Option<&str>, default_workers: usize) -> Result<HostMap> {
    let spec = spec.unwrap_or_default().trim();
    if spec.is_empty() {
        return Ok(HostMap(vec![
            (GP_HOST.to_owned(), default_workers),
            (GP_HOST2.to_owned(), GP_HOST2_WORKERS),
        ]));
    }
    let mut output = HostMap(Vec::new());
    for part in spec.split(',') {
        let part = part.trim();
        let (host, workers) = part.split_once(':').unwrap_or((part, ""));
        if !host.is_empty() {
            let workers = if workers.is_empty() {
                default_workers
            } else {
                python_int(workers)?
            };
            output.set(host.to_owned(), workers);
        }
    }
    if output.0.is_empty() {
        output.set(GP_HOST.to_owned(), default_workers);
    }
    Ok(output)
}

/// Read `JOPE_GP_HOSTS` using the same default and parsing rules as the Python module.
pub fn gp_host_map_from_env(default_workers: usize) -> Result<HostMap> {
    match std::env::var("JOPE_GP_HOSTS") {
        Ok(spec) => gp_host_map(Some(&spec), default_workers),
        Err(std::env::VarError::NotPresent) => gp_host_map(None, default_workers),
        Err(error) => Err(error.into()),
    }
}

/// The subset of `PrimeConfig` used by the GP factory configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimePaths {
    pub jain_binary: PathBuf,
    pub gpu_cache: PathBuf,
    pub regression_dataset_root: PathBuf,
    pub cache: PathBuf,
    pub legacy_root: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpFactoryConfig {
    pub jain_bin: PathBuf,
    pub gpu_cache: PathBuf,
    pub reg_root: PathBuf,
    pub gp_work_root: PathBuf,
    pub model_name: &'static str,
    pub candidate_pools_glob: PathBuf,
}

impl GpFactoryConfig {
    pub fn from_prime_paths(prime: &PrimePaths) -> Self {
        Self {
            jain_bin: prime.jain_binary.clone(),
            gpu_cache: prime.gpu_cache.clone(),
            reg_root: prime.regression_dataset_root.clone(),
            gp_work_root: prime.cache.join("gp_datasets"),
            model_name: MODEL_NAME,
            candidate_pools_glob: prime
                .legacy_root
                .join("runs/self_play_v33/candidate_pools/candidate_pools_g*.jsonl"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_map_defaults_match_python_order_and_worker_counts() {
        let hosts = gp_host_map(None, 4).unwrap();
        assert_eq!(
            hosts.iter().collect::<Vec<_>>(),
            vec![("xbabe2", 4), ("xbabe3", 5)]
        );
    }

    #[test]
    fn host_map_overrides_preserve_insertion_order_and_python_dict_updates() {
        let hosts = gp_host_map(Some(" xbabe1:4,xbabe3:4,xbabe2:2,xbabe1:7 "), 9).unwrap();
        assert_eq!(
            hosts.iter().collect::<Vec<_>>(),
            vec![("xbabe1", 7), ("xbabe3", 4), ("xbabe2", 2)]
        );
        let hosts = gp_host_map(Some("a:-2,b:,a:5"), 3).unwrap();
        assert_eq!(hosts.iter().collect::<Vec<_>>(), vec![("a", 5), ("b", 3)]);
        let hosts = gp_host_map(Some(" , :2 "), 6).unwrap();
        assert_eq!(hosts.iter().collect::<Vec<_>>(), vec![("xbabe2", 6)]);
        assert!(gp_host_map(Some("xbabe1:nope"), 4).is_err());
    }

    #[test]
    fn prime_paths_derive_factory_roots_and_glob() {
        let prime = PrimePaths {
            jain_binary: "/opt/jain".into(),
            gpu_cache: "/cache/gpu".into(),
            regression_dataset_root: "/data/regression".into(),
            cache: "/prime/cache".into(),
            legacy_root: "/legacy".into(),
        };
        let config = GpFactoryConfig::from_prime_paths(&prime);
        assert_eq!(config.jain_bin, PathBuf::from("/opt/jain"));
        assert_eq!(config.gpu_cache, PathBuf::from("/cache/gpu"));
        assert_eq!(config.reg_root, PathBuf::from("/data/regression"));
        assert_eq!(
            config.gp_work_root,
            PathBuf::from("/prime/cache/gp_datasets")
        );
        assert_eq!(config.model_name, "jope_gp_v1");
        assert_eq!(
            config.candidate_pools_glob,
            PathBuf::from("/legacy/runs/self_play_v33/candidate_pools/candidate_pools_g*.jsonl")
        );
    }
}
