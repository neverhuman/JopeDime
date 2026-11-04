use super::*;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClusteredConfidenceBoundsV3 {
    pub resamples: usize,
    pub seed: u64,
    pub candidate_vs_accepted_lower: f64,
    pub candidate_vs_gp_lower: f64,
    pub conditional_advantage_lower: f64,
}

#[derive(Clone, Copy)]
struct BootstrapFamily {
    candidate_accepted_sum: f64,
    candidate_gp_sum: f64,
    conditional_sum: f64,
    count: usize,
}

#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
}

fn lower_confidence_bound(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    // A one-sided 95% deterministic percentile bound.
    values[(values.len() * 5 / 100).min(values.len() - 1)]
}

pub fn clustered_confidence_bounds_v3(
    observations: &[PromotionObservationV3],
    seed: u64,
) -> Result<ClusteredConfidenceBoundsV3> {
    ensure!(
        !observations.is_empty(),
        "clustered confidence requires observations"
    );
    let mut grouped: BTreeMap<&Sha256Digest, BootstrapFamily> = BTreeMap::new();
    for row in observations {
        row.validate()?;
        let family = grouped
            .entry(&row.source_family_sha256)
            .or_insert(BootstrapFamily {
                candidate_accepted_sum: 0.0,
                candidate_gp_sum: 0.0,
                conditional_sum: 0.0,
                count: 0,
            });
        family.candidate_accepted_sum += row.candidate_score - row.accepted_score;
        family.candidate_gp_sum += row.candidate_score - row.gp_score;
        family.conditional_sum +=
            row.candidate_conditional_advantage - row.accepted_conditional_advantage;
        family.count += 1;
    }
    let families = grouped.values().copied().collect::<Vec<_>>();
    ensure!(
        families.len() >= 2,
        "clustered confidence requires at least two source families"
    );
    let mut random = SplitMix64::new(seed);
    let mut accepted = Vec::with_capacity(PROMOTION_BOOTSTRAP_RESAMPLES);
    let mut gp = Vec::with_capacity(PROMOTION_BOOTSTRAP_RESAMPLES);
    let mut conditional = Vec::with_capacity(PROMOTION_BOOTSTRAP_RESAMPLES);
    for _ in 0..PROMOTION_BOOTSTRAP_RESAMPLES {
        let mut accepted_sum = 0.0;
        let mut gp_sum = 0.0;
        let mut conditional_sum = 0.0;
        let mut count = 0;
        for _ in 0..families.len() {
            let family = families[random.next() as usize % families.len()];
            accepted_sum += family.candidate_accepted_sum;
            gp_sum += family.candidate_gp_sum;
            conditional_sum += family.conditional_sum;
            count += family.count;
        }
        accepted.push(accepted_sum / count as f64);
        gp.push(gp_sum / count as f64);
        conditional.push(conditional_sum / count as f64);
    }
    Ok(ClusteredConfidenceBoundsV3 {
        resamples: PROMOTION_BOOTSTRAP_RESAMPLES,
        seed,
        candidate_vs_accepted_lower: lower_confidence_bound(accepted),
        candidate_vs_gp_lower: lower_confidence_bound(gp),
        conditional_advantage_lower: lower_confidence_bound(conditional),
    })
}
