//! Immutable ranked-reward views derived from raw utility measurements.
//!
//! A view records every input needed to reproduce its ranking. It never edits
//! source measurements and cannot qualify a policy or authorize promotion.

use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const RANKED_REWARD_SCHEMA: &str = "jope.selfplay-ranked-reward-view/v1";
pub const MAX_BUFFER_ITEMS: usize = 16_384;
pub const MAX_ID_BYTES: usize = 256;
pub const MAX_ALPHA: f64 = 8.0;

#[derive(Clone, Debug, PartialEq)]
pub struct RawUtilityMeasurement {
    pub measurement_id: String,
    pub family_id: String,
    pub raw_utility: f64,
    pub synthetic: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmptyBufferRule {
    ExplicitUnavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TieRule {
    PinnedSeededHashOrder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankedViewStatus {
    Ready,
    EmptyUnavailable,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankedUtility {
    /// Original utility, copied without rank transformation.
    pub raw_utility: f64,
    pub rank: usize,
    pub normalized_rank: f64,
    pub reward_weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankedRewardView {
    pub schema: &'static str,
    pub view_id: String,
    pub raw_snapshot_sha256: String,
    pub policy_generation: String,
    pub alpha: f64,
    pub tie_seed: u64,
    pub tie_rule: TieRule,
    pub empty_rule: EmptyBufferRule,
    pub status: RankedViewStatus,
    pub entries: Vec<(RawUtilityMeasurement, RankedUtility)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RankedError {
    InvalidAlpha,
    MissingPolicyGeneration,
    BufferTooLarge,
    InvalidMeasurement,
    DuplicateMeasurementId,
}

/// Build a reproducible view over an immutable copy of the raw measurements.
/// Higher utility ranks first. Equal utilities are ordered by a deterministic
/// SplitMix64 key derived from the pinned seed and measurement ID.
pub fn rank_reward_view(
    raw: &[RawUtilityMeasurement],
    alpha: f64,
    tie_seed: u64,
    policy_generation: &str,
) -> Result<RankedRewardView, RankedError> {
    if !alpha.is_finite() || !(0.0..=MAX_ALPHA).contains(&alpha) {
        return Err(RankedError::InvalidAlpha);
    }
    if policy_generation.trim().is_empty() || policy_generation.len() > MAX_ID_BYTES {
        return Err(RankedError::MissingPolicyGeneration);
    }
    if raw.len() > MAX_BUFFER_ITEMS {
        return Err(RankedError::BufferTooLarge);
    }
    let mut ids = BTreeSet::new();
    for measurement in raw {
        if measurement.measurement_id.trim().is_empty()
            || measurement.measurement_id.len() > MAX_ID_BYTES
            || measurement.family_id.trim().is_empty()
            || measurement.family_id.len() > MAX_ID_BYTES
            || !measurement.raw_utility.is_finite()
        {
            return Err(RankedError::InvalidMeasurement);
        }
        if !ids.insert(measurement.measurement_id.as_str()) {
            return Err(RankedError::DuplicateMeasurementId);
        }
    }

    let raw_snapshot_sha256 = raw_digest(raw);
    let empty = raw.is_empty();
    let mut ordered: Vec<(RawUtilityMeasurement, u64)> = raw
        .iter()
        .cloned()
        .map(|item| {
            let mut rng = SplitMix64::new(tie_seed ^ stable_id_hash(&item.measurement_id));
            (item, rng.next_u64())
        })
        .collect();
    ordered.sort_by(|(a, a_tie), (b, b_tie)| {
        b.raw_utility
            .total_cmp(&a.raw_utility)
            .then_with(|| a_tie.cmp(b_tie))
            .then_with(|| a.measurement_id.cmp(&b.measurement_id))
    });

    let n = ordered.len();
    let mut entries = Vec::with_capacity(n);
    let mut total_weight = 0.0;
    for (index, (measurement, _)) in ordered.iter().enumerate() {
        let normalized_rank = if n <= 1 {
            1.0
        } else {
            (n - 1 - index) as f64 / (n - 1) as f64
        };
        total_weight += (alpha * normalized_rank).exp();
        entries.push((
            measurement.clone(),
            RankedUtility {
                raw_utility: measurement.raw_utility,
                rank: index + 1,
                normalized_rank,
                reward_weight: 0.0,
            },
        ));
    }
    for (_, utility) in &mut entries {
        utility.reward_weight = (alpha * utility.normalized_rank).exp() / total_weight;
    }

    let status = if empty {
        RankedViewStatus::EmptyUnavailable
    } else {
        RankedViewStatus::Ready
    };
    let view_id = view_digest(
        &raw_snapshot_sha256,
        policy_generation,
        alpha,
        tie_seed,
        status,
    );
    Ok(RankedRewardView {
        schema: RANKED_REWARD_SCHEMA,
        view_id,
        raw_snapshot_sha256,
        policy_generation: policy_generation.to_owned(),
        alpha,
        tie_seed,
        tie_rule: TieRule::PinnedSeededHashOrder,
        empty_rule: EmptyBufferRule::ExplicitUnavailable,
        status,
        entries,
    })
}

fn raw_digest(raw: &[RawUtilityMeasurement]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"jopedime.selfplay-raw-utility.v1\0");
    hasher.update((raw.len() as u64).to_le_bytes());
    for item in raw {
        put_string(&mut hasher, &item.measurement_id);
        put_string(&mut hasher, &item.family_id);
        hasher.update(item.raw_utility.to_bits().to_le_bytes());
        hasher.update([u8::from(item.synthetic)]);
    }
    hex::encode(hasher.finalize())
}

fn view_digest(raw: &str, policy: &str, alpha: f64, seed: u64, status: RankedViewStatus) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"jopedime.selfplay-ranked-view.v1\0");
    put_string(&mut hasher, raw);
    put_string(&mut hasher, policy);
    hasher.update(alpha.to_bits().to_le_bytes());
    hasher.update(seed.to_le_bytes());
    hasher.update([match status {
        RankedViewStatus::Ready => 1,
        RankedViewStatus::EmptyUnavailable => 0,
    }]);
    hasher.update([1, 1]); // pinned seeded tie rule and explicit empty rule versions
    hex::encode(hasher.finalize())
}

fn put_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn stable_id_hash(id: &str) -> u64 {
    let digest = Sha256::digest(id.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("fixed digest prefix"))
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}
