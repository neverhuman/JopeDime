//! Bounded, read-only progress for the explicitly admitted proposal-library jobs.
use crate::{manifest::{file_hash, read_json}, service::{GpJob, GpJobs}, telemetry::now};
use serde_json::{json, Value};
use std::{collections::BTreeSet, path::{Path, PathBuf}, sync::{Mutex, OnceLock}};

#[derive(Default)]
struct Evidence {
    checked: bool,
    started: bool,
    complete: bool,
    evaluations: u64,
    mismatch: bool,
}
struct Entry { job: GpJob, evidence: Evidence }
struct Cache { key: (PathBuf, String), entries: Vec<Entry>, cursor: usize }
static CACHE: OnceLock<Mutex<Option<Cache>>> = OnceLock::new();

fn unavailable(reason:&str)->Value { json!({"status":"unavailable","reason":reason,"observation_ledgers_read":false}) }

fn probe(root:&Path, entry:&mut Entry) {
    entry.evidence.checked=true;
    let output=root.join(&entry.job.output);
    let progress:Value=read_json(&output.join("progress.json")).unwrap_or_default();
    if !progress.is_object() {return;}
    let matches=progress["prepared_sha256"]==entry.job.prepared_sha256
        && progress["cut"].as_u64()==Some(entry.job.cut as u64)
        && progress["seeds_sha256"].as_str()==entry.job.seeds_sha256.as_deref();
    if !matches {
        entry.evidence=Evidence{checked:true,mismatch:true,..Default::default()};
        return;
    }
    let Some(evaluations)=progress["evaluations"].as_u64() else {return;};
    entry.evidence.started=true;
    entry.evidence.evaluations=evaluations;
    entry.evidence.mismatch=false;
    let summary:Value=read_json(&output.join("search.json")).unwrap_or_default();
    entry.evidence.complete=summary["schema"]=="parallel.search.v1"
        && summary["completed_evaluations"].as_u64()==Some(evaluations)
        && summary["seeds_sha256"]==progress["seeds_sha256"]
        && summary["vocabulary_sha256"].is_string()
        && summary["vocabulary_sha256"]==progress["vocabulary_sha256"];
}

pub(super) fn progress(root:&Path, receipt_path:&Path, receipt:&Value)->Value {
    if receipt["outcome"]!="admitted" {return unavailable("no committed library GP admission");}
    let Some(expected)=receipt["jobs_sha256"].as_str() else {return unavailable("admission job-manifest hash missing");};
    let Some(parent)=receipt_path.parent() else {return unavailable("admission receipt location missing");};
    let manifest=parent.join("combined-jobs.json");
    let key=(manifest.clone(),expected.to_owned());
    let mut guard=CACHE.get_or_init(||Mutex::new(None)).lock().unwrap_or_else(|error|error.into_inner());
    if guard.as_ref().is_none_or(|cache|cache.key!=key) {
        if file_hash(&manifest).ok().as_deref()!=Some(expected) {return unavailable("admitted job-manifest hash mismatch");}
        let Ok(jobs)=read_json::<GpJobs>(&manifest) else {return unavailable("admitted job manifest is unreadable");};
        if jobs.schema!="parallel.gp_jobs.v1" || receipt["requested_jobs"].as_u64()!=Some(jobs.jobs.len() as u64) {
            return unavailable("admitted job count or schema mismatch");
        }
        let outputs=jobs.jobs.iter().map(|job|root.join(&job.output)).collect::<BTreeSet<_>>();
        if outputs.len()!=jobs.jobs.len() {return unavailable("admitted output paths are not unique");}
        *guard=Some(Cache{key,entries:jobs.jobs.into_iter().map(|job|Entry{job,evidence:Evidence::default()}).collect(),cursor:0});
    }
    let cache=guard.as_mut().unwrap();
    let mut indices=cache.entries.iter().enumerate().filter(|(_,entry)|entry.evidence.started && !entry.evidence.complete).map(|(index,_)|index).collect::<BTreeSet<_>>();
    for _ in 0..cache.entries.len().min(256) {
        let index=cache.cursor;cache.cursor=(cache.cursor+1)%cache.entries.len();
        if !cache.entries[index].evidence.complete {indices.insert(index);}
    }
    for &index in &indices {probe(root,&mut cache.entries[index]);}
    json!({"schema":"parallel.dashboard_library_gp.v1","status":"observed","timestamp":now(),
        "requested_jobs":cache.entries.len(),"checked_jobs":cache.entries.iter().filter(|entry|entry.evidence.checked).count(),
        "started_jobs":cache.entries.iter().filter(|entry|entry.evidence.started).count(),
        "completed_jobs":cache.entries.iter().filter(|entry|entry.evidence.complete).count(),
        "observed_evaluations":cache.entries.iter().map(|entry|entry.evidence.evaluations).sum::<u64>(),
        "identity_mismatches":cache.entries.iter().filter(|entry|entry.evidence.mismatch).count(),
        "outputs_probed_this_refresh":indices.len(),"jobs_sha256":expected,
        "scope":"admitted proposal library jobs only; observed lower bound from bounded compact-receipt scans",
        "observation_ledgers_read":false})
}
