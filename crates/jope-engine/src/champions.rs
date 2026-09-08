//! Durable best-found teachers. Scores never cross a cut or evaluator identity.
use crate::{corpus::Index, manifest::*, prepare::Prepared};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, os::fd::AsRawFd, path::{Path, PathBuf}};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Identity {
    pub dataset: String,
    pub family: String,
    pub partition: String,
    pub cut: usize,
    pub prepared_sha256: String,
    pub training_sha256: String,
    pub validation_sha256: String,
    pub evaluator_sha256: String,
    pub vocabulary_sha256: String,
}
impl Identity {
    pub fn key(&self) -> Result<String> { Ok(hash(&serde_json::to_vec(self)?)) }
    pub fn observation(value: &Value) -> Result<Option<Self>> {
        if value["vocabulary_sha256"].is_null() { return Ok(None); }
        let string = |name: &str| -> Result<String> {
            Ok(value[name].as_str().with_context(|| format!("teacher {name} absent"))?.to_string())
        };
        let result = Self {
            dataset: string("dataset")?, family: string("family")?, partition: string("partition")?,
            cut: value["cut"].as_u64().context("teacher cut absent")? as usize,
            prepared_sha256: string("prepared_sha256")?, training_sha256: string("training_sha256")?,
            validation_sha256: string("validation_sha256")?, evaluator_sha256: string("evaluator_sha256")?,
            vocabulary_sha256: string("vocabulary_sha256")?,
        };
        for digest in [&result.prepared_sha256, &result.training_sha256, &result.validation_sha256,
                       &result.evaluator_sha256, &result.vocabulary_sha256] {
            ensure!(digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()), "invalid teacher identity hash");
        }
        Ok(Some(result))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Champion {
    pub schema: String,
    pub identity: Identity,
    pub observation: Value,
    pub observation_sha256: String,
    pub source: PathBuf,
    pub rmse_original_units: f64,
    pub previous_event_sha256: Option<String>,
    pub previous_rmse_original_units: Option<f64>,
    pub improvement_number: usize,
    pub timestamp: u64,
}
impl Champion {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == "parallel.cut_champion.v1"
            && Identity::observation(&self.observation)?.as_ref() == Some(&self.identity)
            && hash(&serde_json::to_vec(&self.observation)?) == self.observation_sha256
            && score(&self.observation)? == Some(self.rmse_original_units), "champion evidence changed");
        Ok(())
    }
    pub fn genome(&self) -> &str { self.observation["genome"].as_str().unwrap() }
}

fn score(o: &Value) -> Result<Option<f64>> {
    if !o["failure"].is_null() || o["evaluation_bias"] != "adaptive_cut_heldout_teacher" {
        return Ok(None);
    }
    let rmse = o["metrics"]["rmse"].as_f64().context("teacher RMSE absent")?;
    let scale = o["target_scale"].as_f64().context("teacher target scale absent")?;
    let original = rmse * scale;
    ensure!(rmse.is_finite() && rmse >= 0.0 && scale.is_finite() && scale > 0.0
        && original.is_finite(), "invalid teacher score");
    let genome = o["genome"].as_str().context("teacher program absent")?;
    ensure!(hash(genome.as_bytes()) == o["program_sha256"], "teacher program hash changed");
    ensure!(o["independent_evaluation"] == false, "teacher bias identity absent");
    Ok(Some(original))
}

/// The caller has already fsynced the observation. Store its full evidence so
/// journal rotation or a later corpus rebuild cannot erase the incumbent.
pub fn record(root: &Path, observation: &Value, source: &Path) -> Result<bool> {
    let Some(identity) = Identity::observation(observation)? else { return Ok(false); };
    let Some(rmse) = score(observation)? else { return Ok(false); };
    let directory = root.join("runs/champions").join(identity.key()?);
    fs::create_dir_all(&directory)?;
    let lock = fs::OpenOptions::new().create(true).truncate(false).write(true).open(directory.join("lock"))?;
    ensure!(unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } == 0, "champion lock failed");
    let current = directory.join("current.json");
    let previous = if current.exists() {
        let old: Champion = read_json(&current)?;
        old.validate()?;
        ensure!(old.identity == identity, "champion namespace collision");
        if old.rmse_original_units <= rmse { return Ok(false); }
        Some(old)
    } else { None };
    let champion = Champion {
        schema: "parallel.cut_champion.v1".into(), identity,
        observation: observation.clone(), observation_sha256: hash(&serde_json::to_vec(observation)?),
        source: source.strip_prefix(root).unwrap_or(source).to_path_buf(), rmse_original_units: rmse,
        previous_event_sha256: previous.as_ref().map(|p| hash(&serde_json::to_vec(p).unwrap())),
        previous_rmse_original_units: previous.as_ref().map(|p| p.rmse_original_units),
        improvement_number: previous.as_ref().map_or(0, |p| p.improvement_number + 1),
        timestamp: crate::telemetry::now(),
    };
    champion.validate()?;
    // Event identities use canonical compact serialization independently of
    // the human-readable file representation.
    let event = hash(&serde_json::to_vec(&champion)?);
    write_json(&directory.join(format!("event-{event}.json")), &champion)?;
    write_json(&current, &champion)?;
    Ok(true)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub schema: String,
    pub request: Value,
    pub evaluator_sha256: String,
    pub vocabulary_sha256: String,
    pub entries: Vec<Champion>,
}

/// Both proposal arms share this immutable cycle-level snapshot. The request
/// binds their common corpus/checkpoint/cut range; changes fail on resume.
pub fn freeze_snapshot(root: &Path, corpus: &Path, checkpoint: &Path,
                       output: &Path, cuts: usize, cut_start: usize) -> Result<PathBuf> {
    let destination = output.parent().context("proposal parent absent")?.join("champion-snapshot.json");
    let lock = fs::OpenOptions::new().create(true).truncate(false).write(true).open(destination.with_extension("lock"))?;
    ensure!(unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } == 0, "snapshot lock failed");
    let request = json!({"corpus_sha256":file_hash(&corpus.join("index.json"))?,
        "checkpoint_sha256":file_hash(&checkpoint.join("state.json"))?,"cuts":cuts,"cut_start":cut_start,
        "vocabulary_sha256":file_hash(&corpus.join("vocabulary.json"))?});
    if destination.exists() {
        let snapshot: Snapshot = read_json(&destination)?;
        ensure!(snapshot.schema == "parallel.champion_snapshot.v1" && snapshot.request == request,
            "cycle champion snapshot request changed");
        for entry in &snapshot.entries { entry.validate()?; }
        bind_snapshot(&destination)?;
        return Ok(destination);
    }
    let service: Value = read_json(&root.join("runs/gp-service.json"))?;
    let evaluator = service["executable_sha256"].as_str().context("GP service evaluator identity absent")?;
    let executable = PathBuf::from(service["executable"].as_str().context("GP service executable absent")?);
    let executable = if executable.is_absolute() { executable } else { root.join(executable) };
    ensure!(executable.starts_with(root) && file_hash(&executable)? == evaluator
        && crate::telemetry::now().saturating_sub(service["timestamp"].as_u64().unwrap_or(0)) <= 10,
        "GP evaluator receipt is stale or changed");
    let vocabulary = request["vocabulary_sha256"].as_str().unwrap().to_string();
    ensure!(service["vocabulary_sha256"] == vocabulary, "GP service vocabulary differs from corpus");
    let index: Index = read_json(&corpus.join("index.json"))?;
    let mut available = Vec::<Champion>::new();
    let store = root.join("runs/champions");
    if store.exists() {
        for directory in fs::read_dir(store)? {
            let path = directory?.path().join("current.json");
            if !path.is_file() { continue; }
            let entry: Champion = read_json(&path)?;
            entry.validate()?;
            if entry.identity.vocabulary_sha256 == vocabulary { available.push(entry); }
        }
    }
    // Current evaluator first, then a deterministic version order. Scores from
    // different evaluator versions are never compared; every seed is refitted.
    available.sort_by(|a,b| (a.identity.evaluator_sha256 != evaluator, &a.identity.evaluator_sha256)
        .cmp(&(b.identity.evaluator_sha256 != evaluator, &b.identity.evaluator_sha256)));
    let mut prepared = BTreeMap::<PathBuf, Prepared>::new();
    let mut entries = Vec::new();
    for c in index.contexts.iter().filter(|c| c.partition != "meta_test" && c.cut >= cut_start && c.cut-cut_start < cuts) {
        if !prepared.contains_key(&c.prepared) {
            ensure!(file_hash(&resolve(root, &c.prepared)?.join("prepared.json"))? == c.prepared_sha256,
                "snapshot preparation changed");
            prepared.insert(c.prepared.clone(), read_json(&resolve(root, &c.prepared)?.join("prepared.json"))?);
        }
        let cut = prepared[&c.prepared].cuts.iter().find(|p| p.index == c.cut).context("snapshot cut absent")?;
        let identity = Identity { dataset:c.dataset.clone(), family:c.family.clone(), partition:c.partition.clone(),
            cut:c.cut, prepared_sha256:c.prepared_sha256.clone(), training_sha256:cut.training_sha256.clone(),
            validation_sha256:cut.validation_sha256.clone(), evaluator_sha256:evaluator.into(), vocabulary_sha256:vocabulary.clone() };
        for entry in &available {
            let mut compatible = entry.identity.clone();
            compatible.evaluator_sha256 = evaluator.into();
            if compatible == identity { entries.push(entry.clone()); }
        }
    }
    write_json(&destination, &Snapshot { schema:"parallel.champion_snapshot.v1".into(), request,
        evaluator_sha256:evaluator.into(), vocabulary_sha256:vocabulary, entries })?;
    bind_snapshot(&destination)?;
    Ok(destination)
}

fn bind_snapshot(path: &Path) -> Result<()> {
    let receipt = path.with_file_name("champion-snapshot-binding.json");
    let expected = json!({"schema":"parallel.champion_snapshot_binding.v1","sha256":file_hash(path)?});
    if receipt.exists() {
        ensure!(read_json::<Value>(&receipt)? == expected, "champion snapshot binding changed");
    } else { write_json(&receipt, &expected)?; }
    Ok(())
}

pub fn verify_cycle_snapshot(output: &Path, corpus_sha: &str, cuts: usize, cut_start: usize) -> Result<()> {
    let path = output.join("champion-snapshot.json");
    let snapshot: Snapshot = read_json(&path)?;
    let binding: Value = read_json(&output.join("champion-snapshot-binding.json"))?;
    ensure!(binding["sha256"] == file_hash(&path)?
        && snapshot.request["corpus_sha256"] == corpus_sha
        && snapshot.request["cuts"] == cuts && snapshot.request["cut_start"] == cut_start,
        "cycle champion snapshot identity changed");
    for entry in &snapshot.entries { entry.validate()?; }
    Ok(())
}

/// One bounded pass over explicitly declared completed ledgers. Only the best
/// observation for each exact identity survives in RAM; no test CSV is opened.
pub fn import(root: &Path, sources: &[PathBuf]) -> Result<Value> {
    let mut best = BTreeMap::<Identity, (f64, Value, PathBuf)>::new();
    let mut ledgers = 0usize;
    let mut observations = 0usize;
    for source in sources {
        for entry in walkdir::WalkDir::new(resolve(root, source)?).follow_links(false) {
            let entry = entry?;
            if !entry.file_type().is_file() || entry.file_name() != "observations.json" { continue; }
            let summary = entry.path().parent().unwrap().join("search.json");
            if !summary.is_file() { continue; }
            ensure!(read_json::<Value>(&summary)?["schema"] == "parallel.search.v1", "invalid completed teacher search");
            let rows: Vec<Value> = read_json(entry.path())?;
            ledgers += 1;
            for row in rows {
                observations += 1;
                let Some(identity) = Identity::observation(&row)? else { continue; };
                let Some(rmse) = score(&row)? else { continue; };
                if best.get(&identity).is_none_or(|(old,_,_)| rmse < *old) {
                    best.insert(identity, (rmse, row, entry.path().into()));
                }
            }
        }
    }
    let mut changes = 0;
    for (_, (_, row, path)) in &best { changes += record(root, row, path)? as usize; }
    Ok(json!({"schema":"parallel.champion_import.v1","ledgers":ledgers,"observations":observations,
        "identities":best.len(),"new_or_improved_champions":changes,"source_roots":sources,
        "qualification_only":true,"final_tests_opened":false}))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn observation(rmse: f64) -> Value {
        let genome="(pipe X (ridge))";
        json!({"dataset":"d","family":"f","partition":"meta_train","cut":0,
            "prepared_sha256":"1".repeat(64),"training_sha256":"2".repeat(64),
            "validation_sha256":"3".repeat(64),"evaluator_sha256":"4".repeat(64),
            "vocabulary_sha256":"5".repeat(64),"genome":genome,"program_sha256":hash(genome.as_bytes()),
            "metrics":{"rmse":rmse},"target_scale":2.0,"failure":null,
            "evaluation_bias":"adaptive_cut_heldout_teacher","independent_evaluation":false})
    }
    #[test]
    fn only_improvements_replace_exact_identity_and_preserve_evidence() {
        let tmp=tempfile::tempdir().unwrap();let root=tmp.path();
        let first=observation(3.0);assert!(record(root,&first,Path::new("run1/observations.json")).unwrap());
        let identity=Identity::observation(&first).unwrap().unwrap();
        let dir=root.join("runs/champions").join(identity.key().unwrap());
        let original=fs::read(dir.join("current.json")).unwrap();
        assert!(!record(root,&observation(4.0),Path::new("worse/observations.json")).unwrap());
        assert_eq!(original,fs::read(dir.join("current.json")).unwrap());
        assert!(record(root,&observation(1.0),Path::new("better/observations.json")).unwrap());
        let best:Champion=read_json(&dir.join("current.json")).unwrap();best.validate().unwrap();
        assert_eq!(best.rmse_original_units,2.0);assert_eq!(best.previous_rmse_original_units,Some(6.0));
        let previous:Champion=read_json(&dir.join(format!("event-{}.json",best.previous_event_sha256.unwrap()))).unwrap();
        assert_eq!(previous.observation,first);
    }
    #[test]
    fn different_heldout_or_evaluator_cannot_replace_an_incumbent() {
        let tmp=tempfile::tempdir().unwrap();let first=observation(3.0);
        record(tmp.path(),&first,Path::new("original")).unwrap();
        for field in ["validation_sha256","training_sha256","prepared_sha256","evaluator_sha256","vocabulary_sha256"] {
            let mut other=observation(0.0);other[field]=json!("a".repeat(64));
            assert!(record(tmp.path(),&other,Path::new("other")).unwrap());
        }
        let id=Identity::observation(&first).unwrap().unwrap();
        let saved:Champion=read_json(&tmp.path().join("runs/champions").join(id.key().unwrap()).join("current.json")).unwrap();
        assert_eq!(saved.rmse_original_units,6.0);assert_eq!(saved.improvement_number,0);
    }
    #[test]
    fn changed_program_or_score_bias_is_never_admitted() {
        let tmp=tempfile::tempdir().unwrap();let mut o=observation(1.0);
        o["program_sha256"]=json!("0".repeat(64));assert!(record(tmp.path(),&o,Path::new("x")).is_err());
        o=observation(1.0);o["evaluation_bias"]=json!("independent_test");
        assert!(!record(tmp.path(),&o,Path::new("x")).unwrap());
    }
    #[test]
    fn partial_ledgers_are_not_imported() {
        let tmp=tempfile::tempdir().unwrap();let ledger=tmp.path().join("runs/incomplete/observations.json");
        write_json(&ledger,&vec![observation(1.0)]).unwrap();
        let report=import(tmp.path(),&[PathBuf::from("runs/incomplete")]).unwrap();
        assert_eq!(report["identities"],0);assert!(!tmp.path().join("runs/champions").exists());
    }
    #[test]
    fn both_arms_freeze_same_programs_and_versions_never_share_scores() {
        let tmp=tempfile::tempdir().unwrap();let root=tmp.path();
        for path in ["runs","corpus","checkpoint","cycle/neural","cycle/shuffled","prepared"] {fs::create_dir_all(root.join(path)).unwrap();}
        fs::write(root.join("evaluator"),b"new evaluator").unwrap();
        fs::write(root.join("corpus/vocabulary.json"),b"{}").unwrap();
        fs::write(root.join("checkpoint/state.json"),b"checkpoint").unwrap();
        let vocab=file_hash(&root.join("corpus/vocabulary.json")).unwrap();
        let evaluator=file_hash(&root.join("evaluator")).unwrap();
        let prepared=json!({"schema":"parallel.prepared.v1","dataset":"d","family":"f","task":"regression",
            "meta_partition":"meta_train","seed":SEED,"family_partition_frozen":false,"clean_evaluation":true,
            "clean_status":"raw_reconstructed","original_train_sha256":"0","original_test_sha256":"1",
            "selected_train_indices":[0,1],"selected_test_indices":[],"target_last":true,"rows_total":2,
            "cuts":[{"index":0,"train_indices":[0],"validation_indices":[1],"training_sha256":"2".repeat(64),
                "validation_sha256":"3".repeat(64),"preprocessing_sha256":"6".repeat(64)}]});
        write_json(&root.join("prepared/prepared.json"),&prepared).unwrap();
        let prepared_sha=file_hash(&root.join("prepared/prepared.json")).unwrap();
        let mut old=observation(1.0);old["prepared_sha256"]=json!(prepared_sha);old["vocabulary_sha256"]=json!(vocab);
        record(root,&old,Path::new("old/observations.json")).unwrap();
        let mut current=old.clone();current["evaluator_sha256"]=json!(evaluator);current["metrics"]["rmse"]=json!(100.0);
        record(root,&current,Path::new("new/observations.json")).unwrap();
        write_json(&root.join("runs/gp-service.json"),&json!({"executable":"evaluator","executable_sha256":evaluator,
            "vocabulary_sha256":vocab,"timestamp":crate::telemetry::now()})).unwrap();
        write_json(&root.join("corpus/index.json"),&json!({"schema":"parallel.teaching_corpus.v1","qualification":true,
            "vocabulary_sha256":vocab,"context_index_sha256":"x","records_sha256":"x","record_bytes":0,"records":0,
            "legal_sets":[],"groups":[],"contexts":[{"dataset":"d","family":"f","cut":0,"partition":"meta_train",
                "path":"x","sha256":"x","prepared":"prepared","prepared_sha256":prepared_sha}]})).unwrap();
        let a=freeze_snapshot(root,&root.join("corpus"),&root.join("checkpoint"),&root.join("cycle/neural"),2,0).unwrap();
        let frozen=fs::read(&a).unwrap();let s:Snapshot=read_json(&a).unwrap();
        assert_eq!(s.entries.len(),2);assert_eq!(s.entries[0].rmse_original_units,200.0);
        assert_eq!(s.entries[1].rmse_original_units,2.0);
        old["metrics"]["rmse"]=json!(0.1);record(root,&old,Path::new("improved/observations.json")).unwrap();
        let b=freeze_snapshot(root,&root.join("corpus"),&root.join("checkpoint"),&root.join("cycle/shuffled"),2,0).unwrap();
        assert_eq!(a,b);assert_eq!(frozen,fs::read(b).unwrap());
        assert!(freeze_snapshot(root,&root.join("corpus"),&root.join("checkpoint"),&root.join("cycle/shuffled"),3,0).is_err());
    }
}
