//! Synthetic CANARY_FEATURES negative controls.
//!
//! Canary columns are fitted from training feature marginals only, then regenerated deterministically
//! for train, holdout, and prediction domains. The specs are small and are stored through
//! `FeatureInfo`, so artifacts never embed raw random arrays.
use crate::buildmatrix::FeatureInfo;
use crate::data::{Dataset, Matrix, MultiTargets};
use crate::dsl::Expr;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub const CANARY_PREFIX: &str = "CANARY_FEATURE_";
const CANARY_VERSION: u32 = 1;
const CONTINUOUS_QUANTILES: usize = 65;
const LOW_CARDINALITY_MAX: usize = 32;
const INTEGER_DISCRETE_MAX: usize = 256;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CanaryConfig {
    pub enabled: bool,
}

impl Default for CanaryConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl CanaryConfig {
    pub fn disabled() -> Self {
        Self { enabled: false }
    }

    pub fn effective_enabled(&self) -> bool {
        self.enabled && !canary_env_disabled()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CanarySpec {
    pub version: u32,
    pub name: String,
    pub source_name: String,
    pub source_index: usize,
    pub canary_index: usize,
    #[serde(default)]
    pub search_width: usize,
    pub seed: u64,
    pub profile: CanaryProfile,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CanaryProfile {
    Continuous {
        missing_rate: f32,
        quantiles: Vec<f32>,
    },
    Discrete {
        missing_rate: f32,
        values: Vec<f32>,
        cumulative: Vec<f32>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanaryDomain {
    Train,
    Holdout,
    Predict,
}

pub struct CanaryAugmentation {
    pub train: Dataset,
    pub hold: Dataset,
    pub base_features: Vec<FeatureInfo>,
    pub specs: Vec<CanarySpec>,
    pub enabled: bool,
    pub seed: u64,
}

pub fn append_canaries(
    train: &Dataset,
    hold: &Dataset,
    cfg: &CanaryConfig,
    seed: u64,
) -> CanaryAugmentation {
    if !cfg.effective_enabled() || train.p() == 0 {
        return CanaryAugmentation {
            train: train.clone(),
            hold: hold.clone(),
            base_features: base_feature_infos_without_canaries(train),
            specs: Vec::new(),
            enabled: false,
            seed,
        };
    }
    let specs = fit_specs(train, seed);
    let train_x = append_canary_columns(&train.x, &specs, CanaryDomain::Train);
    let hold_x = append_canary_columns(&hold.x, &specs, CanaryDomain::Holdout);
    let mut names = train.names.clone();
    names.extend(specs.iter().map(|s| s.name.clone()));
    let base_features = base_feature_infos_with_canaries(train, &specs);
    CanaryAugmentation {
        train: dataset_with_matrix(train, train_x, names.clone()),
        hold: dataset_with_matrix(hold, hold_x, names),
        base_features,
        specs,
        enabled: true,
        seed,
    }
}

pub fn fit_specs(train: &Dataset, seed: u64) -> Vec<CanarySpec> {
    (0..train.p())
        .map(|j| {
            let source_name = train.names[j].clone();
            CanarySpec {
                version: CANARY_VERSION,
                name: canary_name(j, &source_name),
                source_name,
                source_index: j,
                canary_index: j,
                search_width: train.p(),
                seed,
                profile: fit_profile(train.x.col(j)),
            }
        })
        .collect()
}

pub fn append_canary_columns(base: &Matrix, specs: &[CanarySpec], domain: CanaryDomain) -> Matrix {
    if specs.is_empty() {
        return base.clone();
    }
    let expected_full_width = specs
        .iter()
        .map(|s| s.search_width.max(s.source_index + 1) + s.canary_index + 1)
        .max()
        .unwrap_or(base.c);
    if base.c >= expected_full_width {
        return base.clone();
    }
    let search_width = specs
        .iter()
        .map(|s| s.search_width.max(s.source_index + 1))
        .max()
        .unwrap_or(base.c)
        .max(base.c);
    let max_canary = specs.iter().map(|s| s.canary_index).max().unwrap_or(0);
    let mut data = Vec::with_capacity(base.n * (search_width + max_canary + 1));
    data.extend_from_slice(base.data.as_ref().as_slice());
    for _ in base.c..search_width {
        data.extend(std::iter::repeat_n(f32::NAN, base.n));
    }
    for canary_index in 0..=max_canary {
        if let Some(spec) = specs.iter().find(|s| s.canary_index == canary_index) {
            data.extend(generate_column(spec, base.n, domain));
        } else {
            data.extend(std::iter::repeat_n(f32::NAN, base.n));
        }
    }
    Matrix::new(data, base.n, search_width + max_canary + 1)
}

pub fn virtual_base_for_domain(
    base: &Matrix,
    features: &[FeatureInfo],
    domain: CanaryDomain,
) -> Matrix {
    let specs = specs_from_features(features);
    append_canary_columns(base, &specs, domain)
}

pub fn specs_from_features(features: &[FeatureInfo]) -> Vec<CanarySpec> {
    let mut specs = features
        .iter()
        .flat_map(|f| f.canary.iter().chain(f.canary_refs.iter()).cloned())
        .collect::<Vec<_>>();
    specs.sort_by_key(|s| s.canary_index);
    specs.dedup_by_key(|s| s.canary_index);
    specs
}

pub fn is_canary_name(name: &str) -> bool {
    name.starts_with(CANARY_PREFIX)
}

pub fn feature_is_direct_canary(f: &FeatureInfo) -> bool {
    f.canary.is_some() || f.source == "canary" || is_canary_name(&f.formula)
}

/// The column a canary spec occupies in the GP *search* matrix. Mirrors the layout
/// [`append_canary_columns`] builds: the base/search width, then one appended slot per canary.
pub fn canary_search_col(spec: &CanarySpec) -> usize {
    spec.search_width.max(spec.source_index + 1) + spec.canary_index
}

/// True when a GP expression's referenced search-matrix columns (`refs`, from `Expr::col_refs`)
/// touch any CANARY negative-control column. Such a survivor "leans on" a synthetic sentinel and is
/// quarantined from the model matrix before feature reduction, so it can never reach selection.
pub fn expr_refs_canary(refs: &[usize], specs: &[CanarySpec]) -> bool {
    specs
        .iter()
        .any(|spec| refs.contains(&canary_search_col(spec)))
}

pub fn generate_column(spec: &CanarySpec, rows: usize, domain: CanaryDomain) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            if unit(spec.seed, spec.canary_index, row, domain, 0) < missing_rate(&spec.profile) {
                f32::NAN
            } else {
                sample_profile(
                    &spec.profile,
                    unit(spec.seed, spec.canary_index, row, domain, 1),
                )
            }
        })
        .collect()
}

fn base_feature_infos_without_canaries(train: &Dataset) -> Vec<FeatureInfo> {
    train
        .names
        .iter()
        .enumerate()
        .map(|(j, name)| FeatureInfo {
            source: "base".into(),
            formula: name.clone(),
            expr: Expr::Col(j as u16),
            fill: 0.0,
            scale_min: 0.0,
            scale_span: 1.0,
            canary: None,
            canary_refs: Vec::new(),
        })
        .collect()
}

fn base_feature_infos_with_canaries(train: &Dataset, specs: &[CanarySpec]) -> Vec<FeatureInfo> {
    let mut features = base_feature_infos_without_canaries(train);
    let offset = train.p();
    features.extend(specs.iter().enumerate().map(|(i, spec)| FeatureInfo {
        source: "canary".into(),
        formula: spec.name.clone(),
        expr: Expr::Col((offset + i) as u16),
        fill: 0.0,
        scale_min: 0.0,
        scale_span: 1.0,
        canary: Some(spec.clone()),
        canary_refs: Vec::new(),
    }));
    features
}

fn dataset_with_matrix(src: &Dataset, x: Matrix, names: Vec<String>) -> Dataset {
    Dataset {
        x,
        y: src.y.clone(),
        y_multi: src.y_multi.as_ref().map(|m| MultiTargets {
            data: m.data.clone(),
            k: m.k,
            primary: m.primary,
            names: m.names.clone(),
        }),
        names,
    }
}

fn fit_profile(col: &[f32]) -> CanaryProfile {
    let n = col.len().max(1);
    let missing = col.iter().filter(|v| !v.is_finite()).count();
    let missing_rate = missing as f32 / n as f32;
    let mut finite = col
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    if finite.is_empty() {
        return CanaryProfile::Discrete {
            missing_rate,
            values: vec![0.0],
            cumulative: vec![1.0],
        };
    }
    let unique = unique_values(&finite);
    let integer_like = finite.iter().all(|v| (*v - v.round()).abs() <= 1e-5);
    if unique.len() <= LOW_CARDINALITY_MAX || (integer_like && unique.len() <= INTEGER_DISCRETE_MAX)
    {
        let total = finite.len() as f32;
        let mut values = Vec::with_capacity(unique.len());
        let mut cumulative = Vec::with_capacity(unique.len());
        let mut acc = 0usize;
        for (value, count) in unique {
            acc += count;
            values.push(value);
            cumulative.push(acc as f32 / total);
        }
        if let Some(last) = cumulative.last_mut() {
            *last = 1.0;
        }
        CanaryProfile::Discrete {
            missing_rate,
            values,
            cumulative,
        }
    } else {
        CanaryProfile::Continuous {
            missing_rate,
            quantiles: (0..CONTINUOUS_QUANTILES)
                .map(|i| quantile(&finite, i as f32 / (CONTINUOUS_QUANTILES - 1) as f32))
                .collect(),
        }
    }
}

fn unique_values(sorted: &[f32]) -> Vec<(f32, usize)> {
    let mut out: Vec<(f32, usize)> = Vec::new();
    for &v in sorted {
        if let Some((last, count)) = out.last_mut() {
            if last.to_bits() == v.to_bits() {
                *count += 1;
                continue;
            }
        }
        out.push((v, 1));
    }
    out
}

fn quantile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let t = pos - lo as f32;
        sorted[lo] * (1.0 - t) + sorted[hi] * t
    }
}

fn missing_rate(profile: &CanaryProfile) -> f64 {
    match profile {
        CanaryProfile::Continuous { missing_rate, .. }
        | CanaryProfile::Discrete { missing_rate, .. } => *missing_rate as f64,
    }
}

fn sample_profile(profile: &CanaryProfile, u: f64) -> f32 {
    match profile {
        CanaryProfile::Discrete {
            values, cumulative, ..
        } => {
            let idx = cumulative
                .iter()
                .position(|&p| u <= p as f64)
                .unwrap_or_else(|| values.len().saturating_sub(1));
            values.get(idx).copied().unwrap_or(0.0)
        }
        CanaryProfile::Continuous { quantiles, .. } => {
            if quantiles.is_empty() {
                return 0.0;
            }
            if quantiles.len() == 1 {
                return quantiles[0];
            }
            let pos = u.clamp(0.0, 1.0) as f32 * (quantiles.len() - 1) as f32;
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            if lo == hi {
                quantiles[lo]
            } else {
                let t = pos - lo as f32;
                quantiles[lo] * (1.0 - t) + quantiles[hi] * t
            }
        }
    }
}

fn canary_name(index: usize, source_name: &str) -> String {
    format!(
        "{CANARY_PREFIX}{index:04}__{}",
        sanitize_source_name(source_name)
    )
}

fn sanitize_source_name(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars().take(64) {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    while out.contains("__") {
        out = out.replace("__", "_");
    }
    let out = out.trim_matches('_');
    if out.is_empty() {
        "feature".into()
    } else {
        out.into()
    }
}

fn canary_env_disabled() -> bool {
    std::env::var("JAIN_CANARY_FEATURES")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "off" | "false" | "no"
            )
        })
        .unwrap_or(false)
}

fn unit(seed: u64, canary_index: usize, row: usize, domain: CanaryDomain, stream: u64) -> f64 {
    let domain = match domain {
        CanaryDomain::Train => 0x5452_4149_4e21_2121,
        CanaryDomain::Holdout => 0x484f_4c44_4f55_5421,
        CanaryDomain::Predict => 0x5052_4544_4943_5421,
    };
    let x = seed
        ^ domain
        ^ ((canary_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        ^ ((row as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9))
        ^ stream.wrapping_mul(0x94d0_49bb_1331_11eb);
    let h = mix64(x);
    ((h >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ds() -> Dataset {
        Dataset {
            x: Matrix::new(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 2.0, 2.0, f32::NAN],
                5,
                2,
            ),
            y: vec![0.0, 1.0, 0.0, 1.0, 0.0],
            y_multi: None,
            names: vec!["continuous value".into(), "group".into()],
        }
    }

    #[test]
    fn canary_generation_shape_names_and_determinism() {
        let train = ds();
        let aug = append_canaries(&train, &train, &CanaryConfig::default(), 99);
        assert!(aug.enabled);
        assert_eq!(aug.specs.len(), 2);
        assert_eq!(aug.train.x.n, 5);
        assert_eq!(aug.train.x.c, 4);
        assert_eq!(aug.hold.x.c, 4);
        assert!(aug.train.names[2].starts_with("CANARY_FEATURE_0000__continuous_value"));
        let again = append_canaries(&train, &train, &CanaryConfig::default(), 99);
        assert_eq!(aug.train.x.col(2), again.train.x.col(2));
        assert_eq!(aug.train.x.col(3), again.train.x.col(3));
    }

    #[test]
    fn domains_are_reproducible_but_distinct() {
        let specs = fit_specs(&ds(), 7);
        let tr = generate_column(&specs[0], 8, CanaryDomain::Train);
        let tr2 = generate_column(&specs[0], 8, CanaryDomain::Train);
        let ho = generate_column(&specs[0], 8, CanaryDomain::Holdout);
        let pr = generate_column(&specs[0], 8, CanaryDomain::Predict);
        assert_eq!(tr, tr2);
        assert_ne!(tr, ho);
        assert_ne!(tr, pr);
    }

    #[test]
    fn discrete_profile_uses_empirical_support_and_missingness() {
        let specs = fit_specs(&ds(), 11);
        let col = generate_column(&specs[1], 128, CanaryDomain::Train);
        assert!(col
            .iter()
            .filter(|v| v.is_finite())
            .all(|v| *v == 1.0 || *v == 2.0));
        assert!(col.iter().any(|v| !v.is_finite()));
    }

    #[test]
    fn continuous_profile_stays_within_training_range() {
        let specs = fit_specs(&ds(), 12);
        let col = generate_column(&specs[0], 128, CanaryDomain::Predict);
        assert!(col
            .iter()
            .filter(|v| v.is_finite())
            .all(|v| (1.0..=5.0).contains(v)));
    }
}
