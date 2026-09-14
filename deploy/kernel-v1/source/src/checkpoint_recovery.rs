//! Recover interrupted checkpoint publication without deleting any saved work.
//! Call only after all previous training ranks have been reaped, before spawning
//! the next attempt. The controller's exclusive GPU lease must remain held.
use crate::manifest::*;
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{fs, path::{Path, PathBuf}, time::{SystemTime, UNIX_EPOCH}};

fn validate_checkpoint(path: &Path, step: u64) -> Result<String> {
    let state_path = path.join("state.json");
    ensure!(!fs::symlink_metadata(&state_path)?.file_type().is_symlink(), "checkpoint state is a symlink");
    let state: Value = read_json(&state_path)?;
    ensure!(state["schema"] == "parallel.checkpoint.v1" && state["step"] == step,
        "checkpoint step or schema changed: {}", path.display());
    for (name, key) in [("model.ot","model_sha256"),("optimizer.ot","optimizer_sha256"),
        ("optimizer.json","optimizer_metadata_sha256"),("config.json","config_sha256")] {
        let artifact = path.join(name);
        ensure!(!fs::symlink_metadata(&artifact)?.file_type().is_symlink()
            && file_hash(&artifact)? == state[key], "checkpoint artifact changed: {}", artifact.display());
    }
    Ok(file_hash(&state_path)?)
}

pub fn reconcile(root: &Path, run: &Path) -> Result<()> {
    let root = root.canonicalize()?;
    let relative_run = if run.is_absolute() { run.strip_prefix(&root)?.to_path_buf() } else { run.to_path_buf() };
    let run = resolve(&root, &relative_run)?;
    let directory = resolve(&root, &relative_run.join("checkpoints"))?;
    let latest_path = resolve(&root, &relative_run.join("latest.json"))?;
    if !directory.exists() {
        ensure!(!latest_path.exists(), "published checkpoint directory is missing");
        return Ok(());
    }
    let latest_hash = if latest_path.exists() { Some(file_hash(&latest_path)?) } else { None };
    let published_step = if latest_hash.is_some() {
        let latest: Value = read_json(&latest_path)?;
        let step = latest["step"].as_u64().context("published checkpoint step absent")?;
        let path = resolve(&root, Path::new(latest["checkpoint"].as_str().context("published checkpoint path absent")?))?;
        ensure!(path == directory.join(format!("step-{step}")), "latest checkpoint is outside this run or has a mismatched step");
        ensure!(validate_checkpoint(&path,step)? == latest["state_sha256"], "published checkpoint state changed");
        Some(step)
    } else {
        ensure!(!run.join("complete.json").exists(), "completed training run lost latest.json; refusing to quarantine committed checkpoints");
        None
    };
    let mut moves = Vec::<Value>::new();
    for entry in fs::read_dir(&directory)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let (step, kind) = if let Some(number) = name.strip_prefix("step-") {
            let Ok(step) = number.parse::<u64>() else { continue; };
            if published_step.is_some_and(|published| step <= published) { continue; }
            (step,"unpublished_complete_checkpoint")
        } else if let Some(rest) = name.strip_prefix(".step-") {
            let Some((number, suffix)) = rest.split_once('-') else { continue; };
            let Ok(step) = number.parse::<u64>() else { continue; };
            if suffix.is_empty() { continue; }
            (step,"interrupted_staging_checkpoint")
        } else { continue; };
        let metadata = entry.file_type()?;
        ensure!(metadata.is_dir() && !metadata.is_symlink(), "unsafe checkpoint recovery entry: {}", entry.path().display());
        let state_sha = if kind == "unpublished_complete_checkpoint" {
            Some(validate_checkpoint(&entry.path(),step)?)
        } else if entry.path().join("state.json").is_file() {
            Some(file_hash(&entry.path().join("state.json"))?)
        } else { None };
        moves.push(json!({"name":name,"step":step,"kind":kind,"state_sha256":state_sha,"moved":false}));
    }
    if moves.is_empty() { return Ok(()); }
    moves.sort_by(|a,b|a["name"].as_str().cmp(&b["name"].as_str()));
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let recovery = resolve(&root,&relative_run.join("checkpoint-recovery").join(format!("{stamp}-{}",std::process::id())))?;
    fs::create_dir_all(recovery.parent().context("quarantine parent absent")?)?;
    fs::create_dir(&recovery)?;
    let mut receipt = json!({"schema":"parallel.checkpoint_recovery.v1","run":relative_run,
        "published_step":published_step,"latest_sha256":latest_hash,"caller_boundary":"all ranks reaped before next spawn",
        "moves":moves,"completed":false,"deleted_bytes":0});
    write_json(&recovery.join("receipt.json"),&receipt)?;
    for index in 0..receipt["moves"].as_array().unwrap().len() {
        ensure!(if let Some(expected)=&latest_hash { latest_path.exists() && file_hash(&latest_path)? == *expected }
            else { !latest_path.exists() }, "latest checkpoint changed during stopped-rank recovery");
        let name = receipt["moves"][index]["name"].as_str().unwrap();
        let destination = recovery.join(name);
        ensure!(!destination.exists(), "checkpoint quarantine collision");
        fs::rename(directory.join(name),&destination)?;
        fs::File::open(&directory)?.sync_all()?;
        fs::File::open(&recovery)?.sync_all()?;
        receipt["moves"][index]["moved"] = true.into();
        write_json(&recovery.join("receipt.json"),&receipt)?;
    }
    receipt["completed"] = true.into();
    write_json(&recovery.join("receipt.json"),&receipt)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn checkpoint(root:&Path, step:u64)->PathBuf {
        let directory=root.join(format!("run/checkpoints/step-{step}"));fs::create_dir_all(&directory).unwrap();
        let mut state=json!({"schema":"parallel.checkpoint.v1","step":step});
        for (file,key) in [("model.ot","model_sha256"),("optimizer.ot","optimizer_sha256"),
            ("optimizer.json","optimizer_metadata_sha256"),("config.json","config_sha256")] {
            fs::write(directory.join(file),format!("{step}:{file}")).unwrap();
            state[key]=file_hash(&directory.join(file)).unwrap().into();
        }
        write_json(&directory.join("state.json"),&state).unwrap();directory
    }
    fn latest(root:&Path, step:u64) {
        let directory=checkpoint(root,step);
        write_json(&root.join("run/latest.json"),&json!({"step":step,
            "checkpoint":format!("run/checkpoints/step-{step}"),"state_sha256":file_hash(&directory.join("state.json")).unwrap()})).unwrap();
    }
    #[test]
    fn orphan_and_staging_are_preserved_and_published_history_stays_put() {
        let tmp=tempfile::tempdir().unwrap();let root=tmp.path();
        checkpoint(root,5);latest(root,10);let orphan=checkpoint(root,20);
        let bytes=fs::read(orphan.join("model.ot")).unwrap();
        let staging=root.join("run/checkpoints/.step-30-123");fs::create_dir(&staging).unwrap();fs::write(staging.join("partial"),b"saved").unwrap();
        let published=fs::read(root.join("run/latest.json")).unwrap();
        reconcile(root,&root.join("run")).unwrap();
        assert!(root.join("run/checkpoints/step-5").exists());assert!(root.join("run/checkpoints/step-10").exists());
        assert!(!orphan.exists());assert!(!staging.exists());assert_eq!(published,fs::read(root.join("run/latest.json")).unwrap());
        let quarantine=fs::read_dir(root.join("run/checkpoint-recovery")).unwrap().next().unwrap().unwrap().path();
        assert_eq!(bytes,fs::read(quarantine.join("step-20/model.ot")).unwrap());
        assert_eq!(fs::read(quarantine.join(".step-30-123/partial")).unwrap(),b"saved");
        let receipt:Value=read_json(&quarantine.join("receipt.json")).unwrap();assert_eq!(receipt["completed"],true);
        reconcile(root,&root.join("run")).unwrap();assert_eq!(fs::read_dir(root.join("run/checkpoint-recovery")).unwrap().count(),1);
    }
    #[test]
    fn corrupted_latest_aborts_before_moving_anything() {
        let tmp=tempfile::tempdir().unwrap();let root=tmp.path();latest(root,10);checkpoint(root,20);
        fs::write(root.join("run/checkpoints/step-10/model.ot"),b"corrupt").unwrap();
        assert!(reconcile(root,&root.join("run")).is_err());assert!(root.join("run/checkpoints/step-20").exists());
        assert!(!root.join("run/checkpoint-recovery").exists());
    }
    #[test]
    fn fresh_unpublished_work_can_be_quarantined_but_completed_run_without_latest_cannot() {
        let tmp=tempfile::tempdir().unwrap();let root=tmp.path();checkpoint(root,1);
        reconcile(root,&root.join("run")).unwrap();assert!(!root.join("run/checkpoints/step-1").exists());
        checkpoint(root,2);fs::write(root.join("run/complete.json"),b"{}").unwrap();
        assert!(reconcile(root,&root.join("run")).is_err());assert!(root.join("run/checkpoints/step-2").exists());
    }
    #[test]
    fn published_path_cannot_point_to_a_different_run() {
        let tmp=tempfile::tempdir().unwrap();let root=tmp.path();latest(root,10);checkpoint(root,20);
        let mut marker:Value=read_json(&root.join("run/latest.json")).unwrap();marker["checkpoint"]=json!("other/checkpoints/step-10");
        write_json(&root.join("run/latest.json"),&marker).unwrap();
        assert!(reconcile(root,&root.join("run")).is_err());assert!(root.join("run/checkpoints/step-20").exists());
    }
    #[test]
    fn published_marker_with_missing_checkpoint_directory_is_rejected() {
        let tmp=tempfile::tempdir().unwrap();let root=tmp.path();
        write_json(&root.join("run/latest.json"),&json!({"step":10,"checkpoint":"run/checkpoints/step-10","state_sha256":"x"})).unwrap();
        assert!(reconcile(root,&root.join("run")).unwrap_err().to_string().contains("directory is missing"));
    }
}
