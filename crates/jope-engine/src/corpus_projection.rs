//! Training-only projection of a sharded teaching corpus. Contexts outside the
//! admitted partitions are removed by position; their files are never opened.
use crate::{
    corpus::{ContextRow, Group, RECORD_BYTES},
    manifest::{file_hash, hash, read_json, resolve, write_json},
    teaching_store::{self, Manifest, Shard, SCHEMA},
};
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
};

pub const ADMITTED_PARTITIONS: [&str; 2] = ["meta_train", "meta_validation"];
pub const PROJECTION_SCHEMA: &str = "jope.corpus-projection/v1";

pub fn admitted_partition(partition: &str) -> bool {
    ADMITTED_PARTITIONS.contains(&partition)
}

/// Refuses any context whose partition is missing, not a string, or not admitted.
/// An empty or absent `contexts` array carries no context and is accepted.
pub fn verify_admitted_context_closure(index: &Value) -> Result<()> {
    let Some(contexts) = index.get("contexts") else {
        return Ok(());
    };
    let contexts = contexts.as_array().context("contexts are not an array")?;
    for (position, row) in contexts.iter().enumerate() {
        let partition = &row["partition"];
        ensure!(
            partition.as_str().is_some_and(admitted_partition),
            "non-admitted context partition at position {position}: {partition}"
        );
    }
    Ok(())
}

/// Every panel record must carry the context of the group that owns it, and
/// that context must be a validation context. The caller reads the records.
pub fn verify_panel_record_contexts(
    panel_ids: &[usize],
    record_context: impl Fn(usize) -> Result<usize>,
    groups: &[Group],
    contexts: &[ContextRow],
) -> Result<()> {
    let wanted: BTreeSet<usize> = panel_ids.iter().copied().collect();
    let mut owners = BTreeMap::new();
    for group in groups {
        for &id in group.records.iter().filter(|id| wanted.contains(*id)) {
            ensure!(
                owners.insert(id, group.context).is_none(),
                "panel record {id} belongs to more than one group"
            );
        }
    }
    for &id in panel_ids {
        let Some(&owner) = owners.get(&id) else {
            bail!("panel record {id} belongs to no group");
        };
        let context = record_context(id)?;
        ensure!(
            context == owner,
            "panel record {id} context {context} differs from its group context {owner}"
        );
        let Some(row) = contexts.get(context) else {
            bail!("panel record {id} context {context} is outside the index");
        };
        ensure!(
            row.partition == "meta_validation",
            "panel record {id} is not on a meta_validation context"
        );
    }
    Ok(())
}

/// `corpus` is relative to `source_root` and `output` to `root`. The output is
/// self-contained: shards, retained contexts and metadata are copied and verified.
pub fn project(root: &Path, source_root: &Path, corpus: &Path, output: &Path) -> Result<Value> {
    let source = resolve(source_root, corpus)?;
    let target = resolve(root, output)?;
    let mut staging = output.as_os_str().to_owned();
    staging.push(".partial");
    let staging = resolve(root, Path::new(&staging))?;
    ensure!(
        !target.starts_with(&source) && !source.starts_with(&target),
        "projection output overlaps its source corpus"
    );
    ensure!(
        fs::symlink_metadata(&target).is_err() && fs::symlink_metadata(&staging).is_err(),
        "projection output already exists"
    );
    let index_bytes = fs::read(source.join("index.json"))?;
    let src: Value = serde_json::from_slice(&index_bytes)?;
    ensure!(
        src["schema"] == SCHEMA,
        "projection requires a sharded teaching corpus"
    );
    let m = Manifest::deserialize(&src)?;
    let mut new_ids = Vec::with_capacity(m.contexts.len());
    let mut kept = Vec::new();
    let mut removed = BTreeMap::<String, usize>::new();
    for row in m.contexts {
        if admitted_partition(&row.partition) {
            new_ids.push(Some(kept.len()));
            kept.push(row);
        } else {
            // Counted by partition only: an excluded path is never resolved or opened.
            new_ids.push(None);
            *removed.entry(row.partition).or_default() += 1;
        }
    }
    let mut contexts = Vec::with_capacity(m.shards.len());
    for shard in &m.shards {
        let context = new_ids
            .get(shard.context)
            .copied()
            .flatten()
            .context("shard references a non-admitted context")?;
        contexts.push(context);
    }
    teaching_store::read_index(source_root, &source)?;
    let source_index_sha256 = hash(&index_bytes);
    let provenance: Value = read_json(&source.join("provenance.json"))?;
    ensure!(
        provenance["index_sha256"] == source_index_sha256
            && m.records_sha256 == hash(&serde_json::to_vec(&m.shards)?),
        "source teaching manifest changed"
    );
    let remapped = m
        .shards
        .iter()
        .zip(&contexts)
        .filter(|(shard, context)| shard.context != **context)
        .count();
    let projection = json!({"schema":PROJECTION_SCHEMA,"source_index_sha256":source_index_sha256,
        "source_context_index_sha256":m.context_index_sha256,
        "admitted_partitions":ADMITTED_PARTITIONS,"removed_contexts_by_partition":removed,
        "contexts_in":new_ids.len(),"contexts_out":kept.len(),"shards":m.shards.len(),
        "records":m.records,"remapped_shards":remapped,
        "first_removed_position":new_ids.iter().position(Option::is_none)});
    let admitted: Vec<String> = kept.iter().map(|c| c.sha256.clone()).collect();
    let parent = target.parent().context("projection parent absent")?;
    fs::create_dir_all(parent)?;
    fs::create_dir(&staging)?;
    let built = (|| -> Result<(String, String)> {
        fs::create_dir(staging.join("shards"))?;
        let mut shards = Vec::with_capacity(m.shards.len());
        for (shard, &context) in m.shards.iter().zip(&contexts) {
            let name = shard.path.file_name().context("unnamed shard path")?;
            let directory = staging.join("shards").join(name);
            fs::create_dir(&directory)?;
            let from = resolve(source_root, &shard.path)?;
            rewrite_records(&from, &directory, shard, context)?;
            rewrite_labels(&from, &directory, shard, context)?;
            let mut projected = Shard {
                context,
                path: output.join("shards").join(name),
                sha256: String::new(),
                records: shard.records,
                max_length: shard.max_length,
                records_sha256: file_hash(&directory.join("records.bin"))?,
                labels_sha256: file_hash(&directory.join("labels.jsonl"))?,
                legal_sets: shard.legal_sets.clone(),
            };
            write_json(&directory.join("shard.json"), &projected)?;
            projected.sha256 = file_hash(&directory.join("shard.json"))?;
            shards.push(projected);
        }
        let mut offsets = Vec::new();
        let mut records = 0usize;
        for shard in &shards {
            offsets.extend((records as u64).to_le_bytes());
            offsets.extend((shard.records as u64).to_le_bytes());
            records += shard.records;
        }
        ensure!(
            records == m.records
                && hash(&offsets) == m.groups_sha256
                && fs::read(source.join("groups.bin"))? == offsets,
            "group offsets changed"
        );
        fs::write(staging.join("groups.bin"), &offsets)?;
        let vocabulary = fs::read(source.join("vocabulary.json"))?;
        ensure!(
            hash(&vocabulary) == m.vocabulary_sha256,
            "source vocabulary changed"
        );
        fs::write(staging.join("vocabulary.json"), &vocabulary)?;
        for row in &mut kept {
            copy_verified(
                &resolve(source_root, &row.path)?,
                &staging.join("contexts").join(&row.path),
                &row.sha256,
            )?;
            row.path = output.join("contexts").join(&row.path);
        }
        // The resident context tensor is keyed on this digest alone.
        let context_index = json!({"schema":"parallel.context_index.v1",
            "qualification":m.qualification,"encodings":kept});
        let context_bytes = serde_json::to_vec(&context_index)?;
        let context_index_sha256 = hash(&context_bytes);
        ensure!(
            context_index_sha256 != m.context_index_sha256,
            "projected context index is not distinct from its source"
        );
        fs::write(staging.join("contexts.json"), &context_bytes)?;
        let records_sha256 = hash(&serde_json::to_vec(&shards)?);
        // Edited as a value: every source key is kept and no typed default is added.
        let mut index = src.clone();
        index["contexts"] = serde_json::to_value(&kept)?;
        index["shards"] = serde_json::to_value(&shards)?;
        index["context_index_sha256"] = json!(context_index_sha256);
        index["records_sha256"] = json!(records_sha256);
        index["projection"] = projection.clone();
        write_json(&staging.join("index.json"), &index)?;
        let provenance = json!({"index_sha256":file_hash(&staging.join("index.json"))?,
            "projection":projection});
        write_json(&staging.join("provenance.json"), &provenance)?;
        sync_tree(&staging)?;
        Ok((context_index_sha256, records_sha256))
    })();
    if built.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    let (context_index_sha256, records_sha256) = built?;
    fs::rename(&staging, &target)?;
    fs::File::open(parent)?.sync_all()?;
    teaching_store::read_index(root, &target)?;
    let written: Value = read_json(&target.join("index.json"))?;
    verify_admitted_context_closure(&written)?;
    let rows: Vec<ContextRow> = serde_json::from_value(written["contexts"].clone())?;
    ensure!(
        rows.iter().map(|c| &c.sha256).eq(admitted.iter()),
        "projected contexts differ from the admitted source contexts"
    );
    let mut receipt = projection;
    receipt["output_index_sha256"] = json!(file_hash(&target.join("index.json"))?);
    receipt["context_index_sha256"] = json!(context_index_sha256);
    receipt["records_sha256"] = json!(records_sha256);
    Ok(receipt)
}

/// Stream fixed-width records; only the context id at bytes 516..520 changes.
fn rewrite_records(from: &Path, to: &Path, shard: &Shard, context: usize) -> Result<()> {
    let source = fs::File::open(from.join("records.bin"))?;
    ensure!(
        source.metadata()?.len() == (shard.records * RECORD_BYTES) as u64,
        "shard record length changed"
    );
    let old = u32::try_from(shard.context)?.to_le_bytes();
    let new = u32::try_from(context)?.to_le_bytes();
    let mut reader = BufReader::new(source);
    let mut writer = BufWriter::new(fs::File::create_new(to.join("records.bin"))?);
    let mut digest = Sha256::new();
    let mut record = [0u8; RECORD_BYTES];
    for _ in 0..shard.records {
        reader.read_exact(&mut record)?;
        digest.update(record);
        ensure!(
            record[516..520] == old,
            "record context differs from its shard context"
        );
        record[516..520].copy_from_slice(&new);
        writer.write_all(&record)?;
    }
    ensure!(
        hex::encode(digest.finalize()) == shard.records_sha256,
        "shard records changed"
    );
    writer.flush()?;
    Ok(())
}

/// Only each label's context changes. A line that does not survive a parse
/// round trip byte for byte is refused rather than silently re-encoded.
fn rewrite_labels(from: &Path, to: &Path, shard: &Shard, context: usize) -> Result<()> {
    let mut reader = BufReader::new(fs::File::open(from.join("labels.jsonl"))?);
    let mut writer = BufWriter::new(fs::File::create_new(to.join("labels.jsonl"))?);
    let mut digest = Sha256::new();
    let mut line = Vec::new();
    let mut local = 0usize;
    while reader.read_until(b'\n', &mut line)? != 0 {
        digest.update(&line);
        let text = line.strip_suffix(b"\n").context("unterminated label")?;
        let mut row: Value = serde_json::from_slice(text)?;
        ensure!(
            serde_json::to_vec(&row)? == text
                && row["record"] == local
                && row["context"] == shard.context,
            "label coordinates or encoding changed"
        );
        row["context"] = json!(context);
        serde_json::to_writer(&mut writer, &row)?;
        writer.write_all(b"\n")?;
        local += 1;
        line.clear();
    }
    ensure!(
        local == shard.records && hex::encode(digest.finalize()) == shard.labels_sha256,
        "shard labels changed"
    );
    writer.flush()?;
    Ok(())
}

/// Copy one retained context file, hashing exactly the bytes that are written.
fn copy_verified(from: &Path, to: &Path, sha256: &str) -> Result<()> {
    fs::create_dir_all(to.parent().context("context parent absent")?)?;
    let mut reader = fs::File::open(from)?;
    let mut writer = fs::File::create_new(to)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
        writer.write_all(&buffer[..n])?;
    }
    ensure!(
        hex::encode(digest.finalize()) == sha256,
        "retained context changed"
    );
    Ok(())
}

/// Children first, so every entry is durable before the directory naming it.
fn sync_tree(path: &Path) -> Result<()> {
    if fs::symlink_metadata(path)?.is_dir() {
        for entry in fs::read_dir(path)? {
            sync_tree(&entry?.path())?;
        }
    }
    fs::File::open(path)?.sync_all()?;
    Ok(())
}
