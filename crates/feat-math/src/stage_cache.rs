//! Stage-prefix cache: content-addressed reuse of fitted transform stages and
//! their train-pass output blocks across genome evaluations that share a
//! pipeline prefix on the same CV fold.
//!
//! The input block to stage i is fully determined by (train-data identity,
//! fold plan, fold, target-stage canonical text, canonical stages 0..i-1), so
//! a chain hash over canonical stage text is a sound content address — no
//! data hashing is needed, and a cache hit reproduces bit-identical results
//! because it returns the output of the identical computation. The cache is
//! therefore invisible in every score, prediction, and receipt.

use crate::dsl::Stage;
use crate::ops::FittedOp;
use crate::Rows;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const SHARDS: usize = 64;

/// Default in-memory budget: 6 GiB.
pub const DEFAULT_BYTE_CAP: usize = 6 * 1024 * 1024 * 1024;

/// Everything besides the stage chain itself that pins a stage computation:
/// the train-data identity plus the fold coordinates (which pin both the
/// fold-scoped x and, together with the target stage, the effective y).
#[derive(Clone, Debug)]
pub struct StageCacheCtx {
    pub context_tag: String,
    pub folds: usize,
    pub fold: usize,
}

/// Chain key over (context, target, stage prefix). Opaque by design: callers
/// can only obtain one from `chain_root` and extend it stage by stage.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ChainKey([u8; 32]);

struct StageCacheEntry {
    fitted: FittedOp,
    block: Arc<Rows>,
    bytes: usize,
    last_used: AtomicU64,
}

pub struct StageCache {
    shards: Vec<Mutex<HashMap<[u8; 32], Arc<StageCacheEntry>>>>,
    byte_cap: usize,
    bytes: AtomicUsize,
    generation: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl StageCache {
    pub fn new(byte_cap: usize) -> Self {
        Self {
            shards: (0..SHARDS).map(|_| Mutex::new(HashMap::new())).collect(),
            byte_cap,
            bytes: AtomicUsize::new(0),
            generation: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Stamp for entry recency; call at each generation boundary before the
    /// generation's evaluations begin.
    pub fn begin_generation(&self, generation: u64) {
        self.generation.store(generation, Ordering::Relaxed);
    }

    pub fn chain_root(&self, ctx: &StageCacheCtx, target: Option<&Stage>) -> ChainKey {
        let mut hasher = Sha256::new();
        hasher.update(b"jope-stage-cache.v1\0");
        hasher.update(ctx.context_tag.as_bytes());
        hasher.update(b"\0");
        hasher.update((ctx.folds as u64).to_le_bytes());
        hasher.update((ctx.fold as u64).to_le_bytes());
        if let Some(stage) = target {
            hasher.update(stage.to_string().as_bytes());
        }
        ChainKey(hasher.finalize().into())
    }

    pub fn chain_extend(&self, key: &ChainKey, stage: &Stage) -> ChainKey {
        let mut hasher = Sha256::new();
        hasher.update(key.0);
        hasher.update(stage.to_string().as_bytes());
        ChainKey(hasher.finalize().into())
    }

    fn shard(&self, key: &ChainKey) -> &Mutex<HashMap<[u8; 32], Arc<StageCacheEntry>>> {
        &self.shards[key.0[0] as usize % SHARDS]
    }

    /// Fitted op + train-pass output block for this chain key, if cached.
    pub fn get(&self, key: &ChainKey) -> Option<(FittedOp, Arc<Rows>)> {
        let shard = self.shard(key).lock().expect("stage cache shard");
        match shard.get(&key.0) {
            Some(entry) => {
                entry
                    .last_used
                    .store(self.generation.load(Ordering::Relaxed), Ordering::Relaxed);
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some((entry.fitted.clone(), Arc::clone(&entry.block)))
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Keeps the entry when its bytes fit under the cap with everything already resident; an
    /// entry that does not fit is dropped, and a later miss recomputes the identical block. The
    /// resident total therefore never passes the cap, between sweeps included. The bytes counted
    /// are the f64 payload and the serialized fit, not allocation overhead.
    pub fn insert(&self, key: ChainKey, fitted: FittedOp, block: Arc<Rows>) {
        let block_bytes = block.iter().map(|row| row.len() * 8).sum::<usize>();
        let fitted_bytes = serde_json::to_vec(&fitted).map_or(1024, |bytes| bytes.len());
        let bytes = block_bytes + fitted_bytes;
        let reserved = self
            .bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |resident| {
                resident
                    .checked_add(bytes)
                    .filter(|total| *total <= self.byte_cap)
            });
        if reserved.is_err() {
            return;
        }
        let entry = Arc::new(StageCacheEntry {
            fitted,
            block,
            bytes,
            last_used: AtomicU64::new(self.generation.load(Ordering::Relaxed)),
        });
        let mut shard = self.shard(&key).lock().expect("stage cache shard");
        // Racing builders produce identical entries (the key pins the whole
        // computation); replacing just swaps equal values.
        if let Some(previous) = shard.insert(key.0, entry) {
            self.bytes.fetch_sub(previous.bytes, Ordering::Relaxed);
        }
    }

    /// Generation-boundary maintenance: drop entries not used in the current
    /// or previous generation. The byte cap needs no eviction here: [`insert`]
    /// admits an entry only while it fits, so a full cache refuses new keys
    /// until resident ones go stale and a sweep returns their bytes.
    ///
    /// [`insert`]: StageCache::insert
    pub fn sweep(&self) {
        let generation = self.generation.load(Ordering::Relaxed);
        let keep_from = generation.saturating_sub(1);
        for shard in &self.shards {
            let mut shard = shard.lock().expect("stage cache shard");
            shard.retain(|_, entry| {
                let keep = entry.last_used.load(Ordering::Relaxed) >= keep_from;
                if !keep {
                    self.bytes.fetch_sub(entry.bytes, Ordering::Relaxed);
                }
                keep
            });
        }
    }

    /// (hits, misses, resident bytes) — instrumentation only.
    pub fn stats(&self) -> (u64, u64, usize) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.bytes.load(Ordering::Relaxed),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::parse_dsl;

    fn stage(src: &str) -> Stage {
        parse_dsl(&format!("(pipe X {src})")).unwrap().stages[0].clone()
    }

    fn ctx() -> StageCacheCtx {
        StageCacheCtx {
            context_tag: "fixture|rows=10|features=3".to_string(),
            folds: 5,
            fold: 0,
        }
    }

    #[test]
    fn chain_keys_separate_context_target_and_stages() {
        let cache = StageCache::new(DEFAULT_BYTE_CAP);
        let base = cache.chain_root(&ctx(), None);
        let with_target = cache.chain_root(&ctx(), Some(&stage("(target kind=log1p)")));
        assert!(base != with_target, "target stage must alter the root");
        let other_fold = cache.chain_root(&StageCacheCtx { fold: 1, ..ctx() }, None);
        assert!(base != other_fold, "fold must alter the root");
        let a = cache.chain_extend(&base, &stage("(standardize)"));
        let b = cache.chain_extend(&base, &stage("(standardize append=true)"));
        assert!(a != b, "stage params must alter the chain");
        // Canonical text is parameter-order independent by construction.
        let c1 = cache.chain_extend(&base, &stage("(rff_cos gamma=1.5 n=8 seed=3)"));
        let c2 = cache.chain_extend(&base, &stage("(rff_cos seed=3 n=8 gamma=1.5)"));
        assert!(c1 == c2, "param order must not alter the chain");
    }

    #[test]
    fn insert_get_sweep_and_byte_accounting() {
        let cache = StageCache::new(1_000_000);
        cache.begin_generation(0);
        let key = cache.chain_extend(&cache.chain_root(&ctx(), None), &stage("(standardize)"));
        let block = Arc::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let fitted = FittedOp::Sigmoid { append: false };
        assert!(cache.get(&key).is_none());
        cache.insert(key, fitted, Arc::clone(&block));
        let (_, cached) = cache.get(&key).expect("hit after insert");
        assert!(Arc::ptr_eq(&cached, &block));
        let (hits, misses, bytes) = cache.stats();
        assert_eq!((hits, misses), (1, 1));
        assert!(bytes > 32);
        // Entry used in generation 0 survives the generation-1 sweep but not
        // the generation-3 sweep.
        cache.begin_generation(1);
        cache.sweep();
        assert!(cache.get(&key).is_some());
        cache.begin_generation(3);
        cache.sweep();
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().2, 0, "sweep must return all bytes");
    }

    #[test]
    fn admission_under_pressure_is_deterministic_and_a_sweep_frees_budget() {
        // Each entry holds 64 block bytes and a 28-byte fit, so a 400-byte cap admits four.
        let cache = StageCache::new(400);
        cache.begin_generation(0);
        let root = cache.chain_root(&ctx(), None);
        let keys: Vec<ChainKey> = (0..5)
            .map(|index| cache.chain_extend(&root, &stage(&format!("(lag k={})", index + 1))))
            .collect();
        for (index, key) in keys.iter().enumerate() {
            cache.insert(
                *key,
                FittedOp::Sigmoid { append: false },
                Arc::new(vec![vec![index as f64; 8]]),
            );
        }
        let resident: Vec<bool> = keys.iter().map(|key| cache.get(key).is_some()).collect();
        assert_eq!(resident, vec![true, true, true, true, false]);
        assert_eq!(cache.stats().2, 4 * 92);
        // Two generations on, the unused entries are stale: the sweep returns their bytes and
        // the refused key now fits.
        cache.begin_generation(2);
        cache.sweep();
        assert_eq!(cache.stats().2, 0);
        cache.insert(
            keys[4],
            FittedOp::Sigmoid { append: false },
            Arc::new(vec![vec![4.0; 8]]),
        );
        assert!(cache.get(&keys[4]).is_some());
        assert_eq!(cache.stats().2, 92);
    }

    #[test]
    fn insertions_never_pass_the_byte_cap_between_sweeps() {
        let cache = StageCache::new(400);
        cache.begin_generation(0);
        let root = cache.chain_root(&ctx(), None);
        let mut kept = 0;
        for index in 0..20 {
            let key = cache.chain_extend(&root, &stage(&format!("(lag k={})", index + 1)));
            cache.insert(
                key,
                FittedOp::Sigmoid { append: false },
                Arc::new(vec![vec![index as f64; 8]]),
            );
            assert!(
                cache.stats().2 <= 400,
                "resident bytes exceed the cap at insert {index}"
            );
            if cache.get(&key).is_some() {
                kept += 1;
            }
        }
        // The first entries fit and stay retrievable; the rest are dropped, never half-kept.
        assert!((1..20).contains(&kept), "{kept}");
        // Replacing an entry with an equal one keeps the accounting exact. The cache has room
        // for the second reservation, so the replacement really runs.
        let roomy = StageCache::new(1_000);
        roomy.begin_generation(0);
        let key = roomy.chain_extend(&roomy.chain_root(&ctx(), None), &stage("(lag k=1)"));
        roomy.insert(
            key,
            FittedOp::Sigmoid { append: false },
            Arc::new(vec![vec![0.0; 8]]),
        );
        let one = roomy.stats().2;
        assert_eq!(one, 92);
        let replacement = Arc::new(vec![vec![0.0; 8]]);
        roomy.insert(
            key,
            FittedOp::Sigmoid { append: false },
            Arc::clone(&replacement),
        );
        let (_, cached) = roomy.get(&key).expect("replacement resident");
        assert!(
            Arc::ptr_eq(&cached, &replacement),
            "the replacement is stored"
        );
        assert_eq!(roomy.stats().2, one, "a replacement does not double-count");
    }
}
