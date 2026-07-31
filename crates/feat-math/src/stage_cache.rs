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

    pub fn insert(&self, key: ChainKey, fitted: FittedOp, block: Arc<Rows>) {
        let block_bytes = block.iter().map(|row| row.len() * 8).sum::<usize>();
        let fitted_bytes = serde_json::to_vec(&fitted).map_or(1024, |bytes| bytes.len());
        let bytes = block_bytes + fitted_bytes;
        if bytes > self.byte_cap {
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
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Generation-boundary maintenance: drop entries not used in the current
    /// or previous generation, then evict oldest-first (ties broken by key,
    /// so eviction order is deterministic) until under the byte cap.
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
        if self.bytes.load(Ordering::Relaxed) <= self.byte_cap {
            return;
        }
        let mut candidates: Vec<(u64, [u8; 32], usize)> = Vec::new();
        for shard in &self.shards {
            let shard = shard.lock().expect("stage cache shard");
            for (key, entry) in shard.iter() {
                candidates.push((entry.last_used.load(Ordering::Relaxed), *key, entry.bytes));
            }
        }
        candidates.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
        for (_, key, bytes) in candidates {
            if self.bytes.load(Ordering::Relaxed) <= self.byte_cap {
                break;
            }
            let mut shard = self.shards[key[0] as usize % SHARDS]
                .lock()
                .expect("stage cache shard");
            if shard.remove(&key).is_some() {
                self.bytes.fetch_sub(bytes, Ordering::Relaxed);
            }
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
    fn eviction_under_pressure_is_deterministic_and_total() {
        // Cap fits roughly two entries; inserting five forces evictions.
        let cache = StageCache::new(400);
        cache.begin_generation(0);
        let root = cache.chain_root(&ctx(), None);
        let mut keys = Vec::new();
        for index in 0..5 {
            let key = cache.chain_extend(&root, &stage(&format!("(lag k={})", index + 1)));
            let block = Arc::new(vec![vec![index as f64; 8]]);
            cache.insert(key, FittedOp::Sigmoid { append: false }, block);
            keys.push(key);
        }
        cache.sweep();
        let resident = keys.iter().filter(|key| cache.get(key).is_some()).count();
        assert!(resident < 5, "eviction must have removed entries");
        assert!(cache.stats().2 <= 400, "resident bytes exceed the cap");
    }
}
