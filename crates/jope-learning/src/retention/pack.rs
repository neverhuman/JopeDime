//! Write-side packs (`jope.s3-pack/v1`): many small result objects become one >= 64 MiB pack
//! plus an index, cutting PUTs about 2,600x (strategy §4 "Dedup/compaction", §8.3).
//!
//! A pack is a concatenation of independent zstd frames, one per ~1 MiB chunk of records, so
//! `zstd -d` on the whole pack yields the records back to back, and one record needs only its
//! chunk decoded. The index names every record (key, source version, sha256, size, chunk,
//! offset) and every chunk (offset, length, decoded size, sha256), plus the pack's own sha256
//! and its write-time tags. [`verify`] re-derives all of it; a pack copy only counts as the
//! kept copy for an E1 expiry of the loose originals after `verify` passes on the bytes read
//! back from S3 (strategy §8.2 E1).
//!
//! Nothing here touches S3. Writers upload `pack_key`/`index_key` with the tags themselves.

use super::tags::WriteTags;
use crate::canon::sha256_hex;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Schema identifier for pack values.
pub const PACK_SCHEMA: &str = "jope.s3-pack/v1";
/// Schema identifier for pack index values.
pub const PACK_INDEX_SCHEMA: &str = "jope.s3-pack-index/v1";
/// Production packs are at least this large before compression (strategy §4).
pub const MIN_PACK_BYTES: u64 = 64 * 1024 * 1024;
/// Decoded chunk size target; one zstd frame per chunk.
pub const CHUNK_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `ChunkRef` is a record for the chunk ref contract in this module: Write-side packs (jope.s3-pack/v1): many small result objects become one >= 64 MiB pack plus an index, cutting PUTs about 2,600x (strategy §4 "Dedup/compaction", §8.3).
pub struct ChunkRef {
    /// Offset recorded for this chunk ref.
    pub offset: u64,
    /// Length recorded for this chunk ref.
    pub length: u64,
    /// Decoded recorded for this chunk ref.
    pub decoded: u64,
    /// Sha-256 recorded for this chunk ref.
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `Entry` is a record for the entry contract in this module: Write-side packs (jope.s3-pack/v1): many small result objects become one >= 64 MiB pack plus an index, cutting PUTs about 2,600x (strategy §4 "Dedup/compaction", §8.3).
pub struct Entry {
    /// Key recorded for this entry.
    pub key: String,
    /// Identifier for the associated version.
    pub version_id: Option<String>,
    /// Sha-256 recorded for this entry.
    pub sha256: String,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
    /// Chunk recorded for this entry.
    pub chunk: u32,
    /// Offset recorded for this entry.
    pub offset: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `PackIndex` is a record for the pack index contract in this module: Write-side packs (jope.s3-pack/v1): many small result objects become one >= 64 MiB pack plus an index, cutting PUTs about 2,600x (strategy §4 "Dedup/compaction", §8.3).
pub struct PackIndex {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Pack schema recorded for this pack index.
    pub pack_schema: String,
    /// Family recorded for this pack index.
    pub family: String,
    /// Identifier for the associated run.
    pub run_id: String,
    /// Sequence recorded for this pack index.
    pub sequence: u32,
    /// True for packs below [`MIN_PACK_BYTES`] written by fixtures; never a production pack.
    pub fixture: bool,
    /// SHA-256 digest identifying the associated pack.
    pub pack_sha256: String,
    /// Byte count for the associated pack.
    pub pack_bytes: u64,
    /// Byte count for the associated decoded.
    pub decoded_bytes: u64,
    /// Records recorded for this pack index.
    pub records: u64,
    /// Collection of tags values carried by this pack index.
    pub tags: Vec<(String, String)>,
    /// Collection of chunks values carried by this pack index.
    pub chunks: Vec<ChunkRef>,
    /// Collection of entries values carried by this pack index.
    pub entries: Vec<Entry>,
}

impl PackIndex {
    /// Performs the pack key operation and returns an error if it cannot be completed.
    pub fn pack_key(&self) -> String {
        format!(
            "packs/v1/{}/{}/{:05}-{}.zst",
            self.family, self.run_id, self.sequence, self.pack_sha256
        )
    }

    /// Performs the index key operation and returns an error if it cannot be completed.
    pub fn index_key(&self) -> String {
        format!(
            "packs/v1/{}/{}/{:05}-{}.index.json",
            self.family, self.run_id, self.sequence, self.pack_sha256
        )
    }
}

/// `SealedPack` is a record for the sealed pack contract in this module: Write-side packs (jope.s3-pack/v1): many small result objects become one >= 64 MiB pack plus an index, cutting PUTs about 2,600x (strategy §4 "Dedup/compaction", §8.3).
pub struct SealedPack {
    /// Encoded byte count associated with this object or operation.
    pub bytes: Vec<u8>,
    /// Index recorded for this sealed pack.
    pub index: PackIndex,
}

fn safe_segment(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
}

/// Accumulates records and seals a pack each time the decoded size reaches the target.
pub struct PackWriter {
    family: String,
    run_id: String,
    tags: WriteTags,
    target: u64,
    fixture: bool,
    sequence: u32,
    chunk: Vec<u8>,
    frames: Vec<u8>,
    chunks: Vec<ChunkRef>,
    entries: Vec<Entry>,
    keys: BTreeSet<String>,
    decoded: u64,
}

impl PackWriter {
    /// A production writer: packs seal at `target` decoded bytes, at least [`MIN_PACK_BYTES`].
    pub fn new(family: &str, run_id: &str, tags: WriteTags, target: u64) -> Result<PackWriter> {
        ensure!(
            target >= MIN_PACK_BYTES,
            "pack target {target} is below the 64 MiB minimum"
        );
        PackWriter::build(family, run_id, tags, target, false)
    }

    /// A fixture writer with a small target; its packs say `fixture: true`.
    pub fn fixture(family: &str, run_id: &str, tags: WriteTags, target: u64) -> Result<PackWriter> {
        PackWriter::build(family, run_id, tags, target.max(1), true)
    }

    fn build(
        family: &str,
        run_id: &str,
        tags: WriteTags,
        target: u64,
        fixture: bool,
    ) -> Result<PackWriter> {
        ensure!(
            safe_segment(family) && safe_segment(run_id),
            "family/run_id must be [A-Za-z0-9._-]{{1,128}}"
        );
        Ok(PackWriter {
            family: family.into(),
            run_id: run_id.into(),
            tags,
            target,
            fixture,
            sequence: 0,
            chunk: Vec::new(),
            frames: Vec::new(),
            chunks: Vec::new(),
            entries: Vec::new(),
            keys: BTreeSet::new(),
            decoded: 0,
        })
    }

    fn flush_chunk(&mut self) {
        if self.chunk.is_empty() {
            return;
        }
        let frame = jope_records::zstd_frame::compress(&self.chunk);
        self.chunks.push(ChunkRef {
            offset: self.frames.len() as u64,
            length: frame.len() as u64,
            decoded: self.chunk.len() as u64,
            sha256: sha256_hex(&self.chunk),
        });
        self.frames.extend_from_slice(&frame);
        self.chunk.clear();
    }

    fn seal(&mut self) -> Option<SealedPack> {
        self.flush_chunk();
        if self.entries.is_empty() {
            return None;
        }
        let bytes = std::mem::take(&mut self.frames);
        let index = PackIndex {
            schema: PACK_INDEX_SCHEMA.into(),
            pack_schema: PACK_SCHEMA.into(),
            family: self.family.clone(),
            run_id: self.run_id.clone(),
            sequence: self.sequence,
            fixture: self.fixture,
            pack_sha256: sha256_hex(&bytes),
            pack_bytes: bytes.len() as u64,
            decoded_bytes: self.decoded,
            records: self.entries.len() as u64,
            tags: self.tags.pairs().to_vec(),
            chunks: std::mem::take(&mut self.chunks),
            entries: std::mem::take(&mut self.entries),
        };
        self.sequence += 1;
        self.decoded = 0;
        Some(SealedPack { bytes, index })
    }

    /// Adds one record; returns a sealed pack when this record brings it to the target.
    pub fn add(
        &mut self,
        key: &str,
        version_id: Option<&str>,
        body: &[u8],
    ) -> Result<Option<SealedPack>> {
        ensure!(!key.is_empty(), "record key is empty");
        if !self.keys.insert(key.to_string()) {
            bail!("record key {key:?} added twice to run {}", self.run_id);
        }
        if !self.chunk.is_empty() && self.chunk.len() + body.len() > CHUNK_BYTES {
            self.flush_chunk();
        }
        self.entries.push(Entry {
            key: key.into(),
            version_id: version_id.map(str::to_string),
            sha256: sha256_hex(body),
            bytes: body.len() as u64,
            chunk: self.chunks.len() as u32,
            offset: self.chunk.len() as u64,
        });
        self.chunk.extend_from_slice(body);
        self.decoded += body.len() as u64;
        Ok(if self.decoded >= self.target {
            self.seal()
        } else {
            None
        })
    }

    /// Seals the last, possibly short, pack.
    pub fn finish(mut self) -> Option<SealedPack> {
        self.seal()
    }
}

fn decode_chunk(pack: &[u8], c: &ChunkRef) -> Result<Vec<u8>> {
    let start = usize::try_from(c.offset)?;
    let end = start
        .checked_add(usize::try_from(c.length)?)
        .context("chunk range overflows")?;
    ensure!(
        end <= pack.len(),
        "chunk {start}..{end} is outside the {}-byte pack",
        pack.len()
    );
    let data = jope_records::zstd_frame::decompress(&pack[start..end], usize::try_from(c.decoded)?)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        data.len() as u64 == c.decoded,
        "chunk decodes to {} bytes, index says {}",
        data.len(),
        c.decoded
    );
    ensure!(
        sha256_hex(&data) == c.sha256,
        "chunk at {start} does not match its sha256"
    );
    Ok(data)
}

/// Reads one record back, verifying its chunk and its own sha256.
pub fn read_entry(pack: &[u8], index: &PackIndex, i: usize) -> Result<Vec<u8>> {
    let e = index
        .entries
        .get(i)
        .with_context(|| format!("no entry {i}"))?;
    let c = index
        .chunks
        .get(e.chunk as usize)
        .with_context(|| format!("entry {i} names missing chunk {}", e.chunk))?;
    let data = decode_chunk(pack, c)?;
    let start = usize::try_from(e.offset)?;
    let end = start + usize::try_from(e.bytes)?;
    ensure!(end <= data.len(), "entry {i} runs past its chunk");
    let body = data[start..end].to_vec();
    ensure!(
        sha256_hex(&body) == e.sha256,
        "entry {} ({}) does not match its sha256",
        i,
        e.key
    );
    Ok(body)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
/// `PackVerify` is a record for the pack verify contract in this module: Write-side packs (jope.s3-pack/v1): many small result objects become one >= 64 MiB pack plus an index, cutting PUTs about 2,600x (strategy §4 "Dedup/compaction", §8.3).
pub struct PackVerify {
    /// Records recorded for this pack verify.
    pub records: u64,
    /// Byte count for the associated decoded.
    pub decoded_bytes: u64,
    /// Chunks recorded for this pack verify.
    pub chunks: u64,
}

/// Re-derives everything the index claims from the pack bytes. Any disagreement is an error.
pub fn verify(pack: &[u8], index: &PackIndex) -> Result<PackVerify> {
    ensure!(
        index.schema == PACK_INDEX_SCHEMA && index.pack_schema == PACK_SCHEMA,
        "unknown pack schema"
    );
    ensure!(
        sha256_hex(pack) == index.pack_sha256,
        "pack bytes do not match pack_sha256"
    );
    ensure!(
        pack.len() as u64 == index.pack_bytes,
        "pack size disagrees with the index"
    );
    let mut next = 0u64;
    let mut decoded = Vec::with_capacity(index.chunks.len());
    for c in &index.chunks {
        ensure!(
            c.offset == next,
            "chunks are not contiguous at {}",
            c.offset
        );
        next = c.offset + c.length;
        decoded.push(decode_chunk(pack, c)?);
    }
    ensure!(next == index.pack_bytes, "chunks do not cover the pack");
    let mut keys = BTreeSet::new();
    let mut total = 0u64;
    let mut cursor = (0u32, 0u64);
    for (i, e) in index.entries.iter().enumerate() {
        ensure!(keys.insert(&e.key), "duplicate key {} in pack index", e.key);
        ensure!(
            (e.chunk, e.offset) == cursor || (e.chunk == cursor.0 + 1 && e.offset == 0),
            "entry {i} is not contiguous"
        );
        let data = decoded
            .get(e.chunk as usize)
            .with_context(|| format!("entry {i} names missing chunk"))?;
        let end = usize::try_from(e.offset + e.bytes)?;
        ensure!(end <= data.len(), "entry {i} runs past its chunk");
        ensure!(
            sha256_hex(&data[e.offset as usize..end]) == e.sha256,
            "entry {i} ({}) sha256 mismatch",
            e.key
        );
        cursor = (e.chunk, e.offset + e.bytes);
        total += e.bytes;
    }
    ensure!(
        total == index.decoded_bytes,
        "entries sum to {total} bytes, index says {}",
        index.decoded_bytes
    );
    ensure!(
        index.entries.len() as u64 == index.records,
        "record count disagrees with the index"
    );
    let chunk_total: u64 = index.chunks.iter().map(|c| c.decoded).sum();
    ensure!(
        chunk_total == total,
        "chunks decode to {chunk_total} bytes, entries hold {total}"
    );
    Ok(PackVerify {
        records: index.records,
        decoded_bytes: total,
        chunks: index.chunks.len() as u64,
    })
}
