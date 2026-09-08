use std::collections::BTreeMap;

use super::contracts::{
    CampaignArmV1, CatalogCountsV1, CatalogDatasetV1, CatalogFreezeV1, CatalogPartitionV1,
    DatasetOriginV1, GpTaskV1, PilotDatasetFreezeV1, TranchePlanV1, TrancheV1,
};

/// Benchmark eligibility is intentionally independent of row count.  A real
/// dataset that is too small or malformed stays visible and produces explicit
/// invalid evidence later; it is never silently removed from the benchmark.
pub fn eligible_real_datasets(
    datasets: impl IntoIterator<Item = CatalogDatasetV1>,
) -> Vec<CatalogDatasetV1> {
    let mut eligible: Vec<_> = datasets
        .into_iter()
        .filter(|dataset| dataset.origin == DatasetOriginV1::Real)
        .collect();
    eligible.sort_by(|left, right| left.dataset_sha256.cmp(&right.dataset_sha256));
    eligible
}

pub fn freeze_catalog(
    observation_snapshot_sha256: impl Into<String>,
    datasets: impl IntoIterator<Item = CatalogDatasetV1>,
) -> Result<CatalogFreezeV1, String> {
    let datasets = eligible_real_datasets(datasets);
    let mut counts = CatalogCountsV1::default();
    for dataset in &datasets {
        match (dataset.partition, dataset.task) {
            (CatalogPartitionV1::Development, GpTaskV1::Regression) => {
                counts.development_regression += 1
            }
            (CatalogPartitionV1::Development, GpTaskV1::Binary) => counts.development_binary += 1,
            (CatalogPartitionV1::Validation, GpTaskV1::Regression) => {
                counts.validation_regression += 1
            }
            (CatalogPartitionV1::Validation, GpTaskV1::Binary) => counts.validation_binary += 1,
        }
    }
    let freeze = CatalogFreezeV1 {
        schema: CatalogFreezeV1::SCHEMA.into(),
        observation_snapshot_sha256: observation_snapshot_sha256.into(),
        counts,
        datasets,
    };
    freeze.validate()?;
    Ok(freeze)
}

#[derive(Clone)]
struct CoveragePoint<'a> {
    dataset: &'a CatalogDatasetV1,
    numeric: [f64; 4],
}

fn normalized_points<'a>(datasets: &'a [&'a CatalogDatasetV1]) -> Vec<CoveragePoint<'a>> {
    let raw: Vec<_> = datasets
        .iter()
        .map(|dataset| {
            [
                (dataset.row_count as f64).ln_1p(),
                (dataset.feature_count as f64).ln_1p(),
                dataset.missing_fraction,
                dataset.target_distribution,
            ]
        })
        .collect();
    let mut lows = [f64::INFINITY; 4];
    let mut highs = [f64::NEG_INFINITY; 4];
    for point in &raw {
        for index in 0..4 {
            lows[index] = lows[index].min(point[index]);
            highs[index] = highs[index].max(point[index]);
        }
    }
    datasets
        .iter()
        .zip(raw)
        .map(|(dataset, raw)| {
            let mut numeric = [0.0; 4];
            for index in 0..4 {
                let width = highs[index] - lows[index];
                numeric[index] = if width <= f64::EPSILON {
                    0.5
                } else {
                    (raw[index] - lows[index]) / width
                };
            }
            CoveragePoint { dataset, numeric }
        })
        .collect()
}

fn distance(left: &CoveragePoint<'_>, right: &CoveragePoint<'_>) -> f64 {
    let numeric = left
        .numeric
        .iter()
        .zip(right.numeric)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>();
    let source = f64::from(left.dataset.source_family != right.dataset.source_family);
    (numeric + source).sqrt()
}

fn max_min_order(datasets: &[&CatalogDatasetV1]) -> Vec<String> {
    let points = normalized_points(datasets);
    if points.is_empty() {
        return Vec::new();
    }
    let source_support = points
        .iter()
        .fold(BTreeMap::<&str, usize>::new(), |mut map, point| {
            *map.entry(&point.dataset.source_family).or_default() += 1;
            map
        });
    // The deterministic first point is far from the numeric center and gives
    // a small bonus to rare source families.  Subsequent points maximize the
    // distance to their closest selected point (classic max-min coverage).
    let first = (0..points.len())
        .max_by(|left, right| {
            let score = |index: usize| {
                points[index]
                    .numeric
                    .iter()
                    .map(|value| (value - 0.5).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    + 1.0 / source_support[points[index].dataset.source_family.as_str()] as f64
            };
            score(*left).total_cmp(&score(*right)).then_with(|| {
                points[*right]
                    .dataset
                    .dataset_sha256
                    .cmp(&points[*left].dataset.dataset_sha256)
            })
        })
        .unwrap();
    let mut selected = vec![first];
    let mut remaining: Vec<_> = (0..points.len()).filter(|index| *index != first).collect();
    while !remaining.is_empty() {
        let next_position = remaining
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                let closest = |candidate: usize| {
                    selected
                        .iter()
                        .map(|chosen| distance(&points[candidate], &points[*chosen]))
                        .min_by(f64::total_cmp)
                        .unwrap()
                };
                closest(**left).total_cmp(&closest(**right)).then_with(|| {
                    points[**right]
                        .dataset
                        .dataset_sha256
                        .cmp(&points[**left].dataset.dataset_sha256)
                })
            })
            .map(|(position, _)| position)
            .unwrap();
        selected.push(remaining.remove(next_position));
    }
    selected
        .into_iter()
        .map(|index| points[index].dataset.dataset_sha256.clone())
        .collect()
}

pub fn select_max_min_pilot(
    catalog: &CatalogFreezeV1,
    alternates_per_task: usize,
) -> Result<PilotDatasetFreezeV1, String> {
    catalog.validate()?;
    let required = 10usize
        .checked_add(alternates_per_task)
        .ok_or_else(|| "pilot alternate count overflows".to_string())?;
    let mut selected = BTreeMap::new();
    let mut alternates = BTreeMap::new();
    for task in [GpTaskV1::Regression, GpTaskV1::Binary] {
        let task_datasets: Vec<_> = catalog
            .datasets
            .iter()
            .filter(|dataset| {
                dataset.partition == CatalogPartitionV1::Development && dataset.task == task
            })
            .collect();
        if task_datasets.len() < required {
            return Err(format!(
                "{} development catalog has {} datasets; pilot plus alternates requires {required}",
                task.as_str(),
                task_datasets.len()
            ));
        }
        let order = max_min_order(&task_datasets);
        selected.insert(task, order[..10].to_vec());
        alternates.insert(task, order[10..required].to_vec());
    }
    let freeze = PilotDatasetFreezeV1 {
        schema: PilotDatasetFreezeV1::SCHEMA.into(),
        catalog_freeze_sha256: catalog.identity()?,
        selected,
        alternates,
        first_arm_started: false,
    };
    freeze.validate()?;
    Ok(freeze)
}

#[allow(clippy::too_many_arguments)]
pub fn build_tranche_plan(
    catalog: &CatalogFreezeV1,
    campaign_freeze_sha256: &str,
    selected_arm: CampaignArmV1,
    partition: CatalogPartitionV1,
    pilot_comparison_sha256: &str,
    development_completion_sha256: Option<String>,
    reused_pilot_dataset_sha256s: Vec<String>,
) -> Result<TranchePlanV1, String> {
    catalog.validate()?;
    let mut datasets: Vec<_> = catalog
        .datasets
        .iter()
        .filter(|dataset| dataset.partition == partition)
        .map(|dataset| dataset.dataset_sha256.clone())
        .collect();
    datasets.sort();
    let tranches = datasets
        .chunks(128)
        .enumerate()
        .map(|(index, chunk)| TrancheV1 {
            index: index as u32,
            partition,
            dataset_sha256s: chunk.to_vec(),
        })
        .collect();
    let plan = TranchePlanV1 {
        schema: TranchePlanV1::SCHEMA.into(),
        campaign_freeze_sha256: campaign_freeze_sha256.into(),
        selected_arm,
        partition,
        pilot_comparison_sha256: pilot_comparison_sha256.into(),
        development_completion_sha256,
        reused_pilot_dataset_sha256s,
        tranches,
    };
    plan.validate()?;
    Ok(plan)
}
