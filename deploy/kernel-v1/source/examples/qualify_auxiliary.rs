//! Exercise actual eight-GPU auxiliary optimizer continuation on a saved corpus.
#[cfg(feature = "neural")]
fn main() -> anyhow::Result<()> {
    use anyhow::{ensure, Result};
    use clap::Parser;
    use jope_parallel::{auxiliary_training::{Auxiliary, Options, load_selection}, manifest::*};
    use serde_json::{json, Value};
    use std::{path::PathBuf, time::{Duration, Instant}};
    #[derive(Parser)]
    struct Args {
        #[arg(long)] root: PathBuf,
        #[arg(long)] corpus: PathBuf,
        #[arg(long)] checkpoint: PathBuf,
        #[arg(long)] output: PathBuf,
        #[arg(long)] executable: PathBuf,
    }
    let a = Args::parse();
    ensure!(a.executable.is_absolute() && a.executable.starts_with(&a.root), "training executable outside workspace");
    std::env::set_var("JOPE_TRAIN_EXECUTABLE", &a.executable);
    ensure!(!a.output.exists(), "qualification output exists");
    let checkpoint_sha = file_hash(&a.checkpoint.join("state.json"))?;
    let before: Value = read_json(&a.root.join("runs/gp-service.json"))?;
    let begun = Instant::now();
    let worker = Auxiliary::start(&a.root, &a.corpus, &a.checkpoint, &a.output,
        Options { block_steps: 10, max_steps: 20, patience: 3, relative_improvement: 0.001 }, true)?;
    let wait = (|| -> Result<()> {
        while !a.output.join("complete.json").exists() {
            ensure!(begun.elapsed() < Duration::from_secs(300), "auxiliary GPU qualification timed out");
            std::thread::sleep(Duration::from_millis(250));
        }
        Ok(())
    })();
    let selected = worker.finish()?;
    wait?;
    let verified = load_selection(&a.root, &a.output)?;
    ensure!(selected.state_sha256 == verified.state_sha256, "selection identity mismatch");
    let complete: Value = read_json(&a.output.join("complete.json"))?;
    ensure!(complete["completed_optimizer_steps"] == 20, "two auxiliary blocks did not finish");
    let training: Value = read_json(&a.output.join("train/complete.json"))?;
    ensure!(training["world"] == 8 && training["steps"] == 20, "auxiliary rank topology changed");
    ensure!(file_hash(&a.checkpoint.join("state.json"))? == checkpoint_sha, "frozen proposal checkpoint changed");
    let after: Value = read_json(&a.root.join("runs/gp-service.json"))?;
    ensure!(after["timestamp"].as_u64() > before["timestamp"].as_u64(), "background GP heartbeat did not advance");
    let receipt = json!({"schema":"parallel.auxiliary_gpu_qualification.v1","passed":true,
        "world":8,"optimizer_steps":20,"block_steps":10,"seconds":begun.elapsed().as_secs_f64(),
        "selected_checkpoint":selected,"completion_sha256":file_hash(&a.output.join("complete.json"))?,
        "training_completion_sha256":file_hash(&a.output.join("train/complete.json"))?,
        "training_executable_sha256":file_hash(&a.executable)?,
        "frozen_proposal_checkpoint_sha256":checkpoint_sha,"gp_heartbeat_advanced":true,
        "qualification":true,"final_tests_opened":false});
    write_json(&a.output.join("qualification.json"), &receipt)?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}

#[cfg(not(feature = "neural"))]
fn main() { eprintln!("This qualification requires --features neural"); std::process::exit(2); }
