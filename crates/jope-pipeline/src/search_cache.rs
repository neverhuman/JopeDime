//! Search-scoped stage-cache identity and bounded cache ownership.
//!
//! Callers build a context from exact fit inputs. The digest includes row IDs
//! and row contents in order, fold plan, seed, evaluator/registry identity,
//! effects, and numeric profile. A cache instance belongs to one search lane.

use feat_math::stage_cache::{StageCache, StageCacheCtx};
use feat_math::{GenomeIr, Model, Rows};
use sha2::{Digest, Sha256};
use std::sync::Arc;

const IDENTITY_DOMAIN: &[u8] = b"jopedime.search-stage-cache.identity.v1\0";
pub const DEFAULT_LANE_BYTE_CAP: usize = 256 * 1024 * 1024;

/// Complete caller-supplied identity for a fit's reusable stage work.
#[derive(Clone, Debug)]
pub struct FitIdentity<'a> {
    pub registry_identity: &'a [u8],
    pub evaluator_identity: &'a [u8],
    pub train_row_ids: &'a [u64],
    pub fold_plan: &'a [u8],
    pub fit_seed: u64,
    pub effect_closure: &'a [u8],
    pub numeric_profile: &'a [u8],
    pub folds: usize,
    pub fold: usize,
}

impl FitIdentity<'_> {
    /// Build the cache context, binding actual ordered TRAIN values as well as
    /// caller row IDs and plan metadata.
    pub fn stage_context(&self, train_x: &Rows, train_y: &[f64]) -> StageCacheCtx {
        let mut hash = Sha256::new();
        hash.update(IDENTITY_DOMAIN);
        field(&mut hash, self.registry_identity);
        field(&mut hash, self.evaluator_identity);
        hash.update((self.train_row_ids.len() as u64).to_le_bytes());
        for id in self.train_row_ids {
            hash.update(id.to_le_bytes());
        }
        hash.update((train_x.len() as u64).to_le_bytes());
        for row in train_x {
            hash.update((row.len() as u64).to_le_bytes());
            for value in row {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        hash.update((train_y.len() as u64).to_le_bytes());
        for value in train_y {
            hash.update(value.to_bits().to_le_bytes());
        }
        field(&mut hash, self.fold_plan);
        hash.update(self.fit_seed.to_le_bytes());
        field(&mut hash, self.effect_closure);
        field(&mut hash, self.numeric_profile);
        hash.update((self.folds as u64).to_le_bytes());
        hash.update((self.fold as u64).to_le_bytes());
        StageCacheCtx {
            context_tag: format!("search-fit-v1:{}", hex(&hash.finalize())),
            folds: self.folds,
            fold: self.fold,
        }
    }
}

fn field(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// One bounded cache with explicit generation boundaries.
pub struct LaneStageCache {
    cache: Arc<StageCache>,
    byte_cap: usize,
}

impl LaneStageCache {
    pub fn new(byte_cap: usize) -> Self {
        Self {
            cache: Arc::new(StageCache::new(byte_cap)),
            byte_cap,
        }
    }

    pub fn default_lane() -> Self {
        Self::new(DEFAULT_LANE_BYTE_CAP)
    }

    pub fn cache(&self) -> &StageCache {
        &self.cache
    }

    pub fn begin_generation(&self, generation: u64) {
        self.cache.begin_generation(generation);
    }

    pub fn evict_stale(&self) {
        self.cache.sweep();
    }

    /// Aggregate cache instrumentation. Deltas are per-fit only under serialized
    /// access; this is not the pending I16 effort-ledger adapter.
    pub fn profile(&self) -> CacheProfile {
        let (hits, misses, resident_bytes) = self.cache.stats();
        CacheProfile {
            hits,
            misses,
            resident_bytes,
            byte_cap: self.byte_cap,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheProfile {
    pub hits: u64,
    pub misses: u64,
    pub resident_bytes: usize,
    pub byte_cap: usize,
}

/// Fit through feat-math's existing cache API using the complete input context.
pub fn fit_cached(
    genome: &GenomeIr,
    train_x: &Rows,
    train_y: &[f64],
    identity: &FitIdentity<'_>,
    cache: &LaneStageCache,
) -> anyhow::Result<Model> {
    let context = identity.stage_context(train_x, train_y);
    feat_math::model::fit_with_stage_cache(genome, train_x, train_y, cache.cache(), &context)
}
