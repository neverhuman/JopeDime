//! Stage-prefix cache: content-addressed reuse of fitted transform stages and
//! their train-pass output blocks across genome evaluations that share a
//! pipeline prefix on the same CV fold.
//!
//! The input block to stage i is fully determined by (train-data identity,
//! fold plan, fold, target stage, stages 0..i-1), so a chain hash over each
//! stage's exact encoding (`write_exact_stage`: its op and every parameter,
//! floats as their IEEE-754 bits) is a sound content address — no data hashing
//! is needed, and a cache hit reproduces bit-identical results because it
//! returns the output of the identical computation. Canonical stage text is not
//! a key: it prints a float within 1e-12 of an integer as that integer, so two
//! different stages can print alike. Every field of a key (context tag, folds,
//! fold, optional target, each stage) is length-prefixed or fixed-width, and a
//! key is the SHA-256 of that encoding, so two different chains share a key
//! only through a SHA-256 collision. The cache is therefore invisible in every
//! score, prediction, and receipt.

use crate::dsl::{ParamValue, Stage};
use crate::ops::FittedOp;
use crate::Rows;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const SHARDS: usize = 64;

/// Default accounted payload budget: 6 GiB. Heap overhead and caller-held clones are additional.
pub const DEFAULT_BYTE_CAP: usize = if usize::BITS >= 64 {
    (6_u64 * 1024 * 1024 * 1024) as usize
} else {
    256 * 1024 * 1024
};

/// Everything besides the stage chain itself that pins a stage computation:
/// the train-data identity plus the fold coordinates (which pin both the
/// fold-scoped x and, together with the target stage, the effective y).
#[derive(Clone, Debug)]
pub struct StageCacheCtx {
    pub context_tag: String,
    pub folds: usize,
    pub fold: usize,
}

/// Appends the exact encoding of `stage` to `out`: its op, then every parameter in key order,
/// each name and string length-prefixed and each value type-tagged, with every float as its
/// IEEE-754 bits. Two stages encode alike only when their ops and parameters match exactly,
/// floats bit for bit, so an encoding names one computation.
pub(crate) fn write_exact_stage(stage: &Stage, out: &mut Vec<u8>) {
    write_prefixed(out, stage.op.as_bytes());
    out.extend_from_slice(&(stage.params.len() as u64).to_le_bytes());
    for (name, value) in &stage.params {
        write_prefixed(out, name.as_bytes());
        match value {
            ParamValue::Int(v) => {
                out.push(0);
                out.extend_from_slice(&v.to_le_bytes());
            }
            ParamValue::Float(v) => {
                out.push(1);
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            ParamValue::Bool(v) => {
                out.push(2);
                out.push(u8::from(*v));
            }
            ParamValue::Str(v) => {
                out.push(3);
                write_prefixed(out, v.as_bytes());
            }
        }
    }
}

fn write_prefixed(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn exact_stage(stage: &Stage) -> Vec<u8> {
    let mut out = Vec::new();
    write_exact_stage(stage, &mut out);
    out
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
        hasher.update(b"jope-stage-cache.v2\0");
        hasher.update((ctx.context_tag.len() as u64).to_le_bytes());
        hasher.update(ctx.context_tag.as_bytes());
        hasher.update((ctx.folds as u64).to_le_bytes());
        hasher.update((ctx.fold as u64).to_le_bytes());
        match target {
            Some(stage) => {
                hasher.update([1u8]);
                hasher.update(exact_stage(stage));
            }
            None => hasher.update([0u8]),
        }
        ChainKey(hasher.finalize().into())
    }

    pub fn chain_extend(&self, key: &ChainKey, stage: &Stage) -> ChainKey {
        let mut hasher = Sha256::new();
        hasher.update(key.0);
        hasher.update(exact_stage(stage));
        ChainKey(hasher.finalize().into())
    }

    fn shard(&self, key: &ChainKey) -> &Mutex<HashMap<[u8; 32], Arc<StageCacheEntry>>> {
        &self.shards[key.0[0] as usize % SHARDS]
    }

    /// Fitted op + train-pass output block for this chain key, if cached.
    pub fn get(&self, key: &ChainKey) -> Option<(FittedOp, Arc<Rows>)> {
        let entry = {
            let shard = self.shard(key).lock().expect("stage cache shard");
            let entry = shard.get(&key.0).cloned();
            if let Some(entry) = &entry {
                entry
                    .last_used
                    .store(self.generation.load(Ordering::Relaxed), Ordering::Relaxed);
            }
            entry
        };
        // Fitted transforms can own large matrices. Pin the entry under the lock, then do
        // the deep clone outside it so other workers on this shard can make progress.
        match entry {
            Some(entry) => {
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
        let Some(block_bytes) = block.iter().try_fold(0usize, |total, row| {
            total.checked_add(row.len().checked_mul(std::mem::size_of::<f64>())?)
        }) else {
            return;
        };
        let Some(fitted_limit) = self.byte_cap.checked_sub(block_bytes) else {
            return;
        };
        // Count the existing JSON representation without allocating a second copy of a fitted
        // matrix. Stop at the entry limit and reject serialization errors instead of guessing
        // 1024 bytes, which could undercount a large fit.
        let mut fitted_bytes = ByteCounter {
            bytes: 0,
            limit: fitted_limit,
        };
        if serde_json::to_writer(&mut fitted_bytes, &fitted).is_err() {
            return;
        }
        let bytes = block_bytes + fitted_bytes.bytes;
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

/// A bounded counting sink: cache accounting uses serialized bytes, not a transient JSON Vec.
struct ByteCounter {
    bytes: usize,
    limit: usize,
}

impl std::io::Write for ByteCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .filter(|total| *total <= self.limit)
            .ok_or_else(|| std::io::Error::other("stage cache entry exceeds byte cap"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
