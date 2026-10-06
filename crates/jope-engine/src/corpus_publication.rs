//! Atomic CPU corpus publication with resumable, hash-verified staging.
use crate::{
    corpus,
    manifest::*,
    telemetry::{now, StageHeartbeat},
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub fn verify(path: &Path, contexts_sha: &str, vocabulary_sha: &str) -> Result<()> {
    let index: corpus::Index = read_json(&path.join("index.json"))?;
    let provenance: Value = read_json(&path.join("provenance.json"))?;
    ensure!(
        index.schema == "parallel.teaching_corpus.v1",
        "unrecognized qualification corpus"
    );
    ensure!(
        index.context_index_sha256 == contexts_sha && index.vocabulary_sha256 == vocabulary_sha,
        "corpus input identities changed"
    );
    ensure!(
        file_hash(&path.join("index.json"))? == provenance["index_sha256"]
            && file_hash(&path.join("records.bin"))? == index.records_sha256
            && file_hash(&path.join("labels.jsonl"))? == provenance["labels_sha256"]
            // W1 round 8: an index that binds its labels must bind these ones.
            && index
                .labels_sha256
                .as_ref()
                .is_none_or(|bound| provenance["labels_sha256"] == bound.as_str())
            && file_hash(&path.join("vocabulary.json"))? == vocabulary_sha,
        "corpus payload or receipt hash changed"
    );
    ensure!(
        path.join("records.bin").metadata()?.len() == (index.records * index.record_bytes) as u64,
        "truncated corpus records"
    );
    Ok(())
}

fn retire_partial(path: &Path, reason: &str) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let name = path
        .file_name()
        .context("partial corpus name absent")?
        .to_string_lossy();
    let parent = path.parent().context("partial corpus parent absent")?;
    let retired = (0..100)
        .map(|n| {
            parent.join(format!(
                "{name}.retired-{}-{}-{n}",
                now(),
                std::process::id()
            ))
        })
        .find(|p| !p.exists())
        .context("no free corpus retirement path")?;
    fs::rename(path, &retired)?;
    write_json(
        &retired.join("retirement.json"),
        &json!({"schema":"parallel.retired_corpus_build.v1","original_path":path,"retired_at":now(),"reason":reason,"qualification":true,"payload_preserved":true}),
    )?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

/// Context, vocabulary and observation inputs for atomic corpus publication.
#[derive(Clone, Copy)]
pub struct PublicationSources<'a> {
    pub contexts: &'a Path,
    pub vocab: &'a Path,
    pub observations: &'a [PathBuf],
}

pub fn build(
    root: &Path,
    sources: PublicationSources<'_>,
    output: &Path,
    qualification: bool,
    cache: Option<&Path>,
    heartbeat: Option<&StageHeartbeat>,
) -> Result<()> {
    let PublicationSources {
        contexts,
        vocab,
        observations,
    } = sources;

    crate::campaign::check_deadline()?;
    let contexts_sha = file_hash(contexts)?;
    let vocab_sha = file_hash(vocab)?;
    fs::create_dir_all(output.parent().context("corpus parent absent")?)?;
    let name = output
        .file_name()
        .context("corpus name absent")?
        .to_string_lossy();
    use std::os::fd::AsRawFd;
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(output.with_file_name(format!(".{name}.lock")))?;
    ensure!(
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "another packer owns this corpus publication"
    );
    let request = json!({"schema":"parallel.corpus_publication.v1", "contexts_sha256":contexts_sha,
        "vocabulary_sha256":vocab_sha,"observations":observations,"qualification":qualification});
    let verified = |path: &Path| -> Result<()> {
        verify(path, &contexts_sha, &vocab_sha)?;
        ensure!(
            read_json::<Value>(&path.join("publication.json"))? == request,
            "corpus publication inputs changed"
        );
        ensure!(
            read_json::<corpus::Index>(&path.join("index.json"))?.qualification == qualification,
            "corpus qualification changed"
        );
        Ok(())
    };
    if output.join("provenance.json").exists() {
        return verified(output);
    }
    retire_partial(
        output,
        "Interrupted final publication; preserve before rebuilding",
    )?;
    let staging = output.with_file_name(format!(".{name}.staging"));
    if staging.exists() {
        if let Err(error) = verified(&staging) {
            retire_partial(&staging, &format!("Incomplete staging: {error:#}"))?;
        }
    }
    if !staging.exists() {
        let mut progress = |operation: &str, done, total| {
            heartbeat.map_or(Ok(()), |h| h.advance(operation, done, total))
        };
        let result = if let Some(cache) = cache {
            corpus::build_observed_cached(
                root,
                crate::corpus::CorpusSources {
                    contexts_path: contexts,
                    vocab_path: vocab,
                    observations,
                },
                &staging,
                qualification,
                cache,
                &mut progress,
            )
            .map(|_| ())
        } else {
            corpus::build_observed(
                root,
                contexts,
                vocab,
                observations,
                &staging,
                qualification,
                &mut progress,
            )
        };
        if let Err(error) = result {
            let reason = format!("Corpus staging build failed: {error:#}");
            retire_partial(&staging, &reason)?;
            write_json(
                &output.with_file_name(format!(
                    "{name}-build-failure-{}-{}.json",
                    now(),
                    std::process::id()
                )),
                &json!({"timestamp":now(),"failure":reason,"request":request,"resumable":true}),
            )?;
            return Err(error);
        }
        write_json(&staging.join("publication.json"), &request)?;
    }
    verified(&staging)?;
    fs::File::open(&staging)?.sync_all()?;
    fs::rename(&staging, output)?;
    fs::File::open(output.parent().context("corpus parent absent")?)?.sync_all()?;
    Ok(())
}
