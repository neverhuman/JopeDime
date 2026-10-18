//! Tiny CPU-only synthetic mechanism fixture for I32.
//!
//! This uses the engine's native Policy, composite objective, AdamW update and
//! atomic checkpoint writer. It is a mechanism exercise only: its data is
//! synthetic TRAIN data and its permission never qualifies science or serving.

use anyhow::{ensure, Context, Result};
use jope_parallel::{
    neural::{AdamW, Config, Policy},
    training::{
        checkpoint_commit::{self, Expectation, Phase},
        objective::{self, Outputs, RowLabels},
    },
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};
use tch::{nn, Device, Kind, Tensor};

const BATCH: usize = 8;
const VOCAB: i64 = 8;
const SEALED_CANARY: &str = "I32-SEALED-CANARY-MUST-NOT-ENTER-TRAINING";

fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn parameters_sha256(store: &nn::VarStore) -> Result<String> {
    let mut digest = Sha256::new();
    let variables: BTreeMap<_, _> = store.variables().into_iter().collect();
    for (name, value) in variables {
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        let values = Vec::<f32>::try_from(
            value
                .detach()
                .to_device(Device::Cpu)
                .to_kind(Kind::Float)
                .flatten(0, -1),
        )?;
        for value in values {
            digest.update(value.to_le_bytes());
        }
    }
    Ok(hex::encode(digest.finalize()))
}

fn configure() -> Config {
    Config {
        dataset_dim: 4,
        target_dim: 4,
        target_keep_dim: 4,
        width: 8,
        layers: 1,
        heads: 2,
        ffn: 16,
        vocab_size: VOCAB,
        embedding_rows: VOCAB,
        max_seq: 4,
        dropout: 0.0,
        dataset_rms: 1.0,
        target_rms: 1.0,
    }
}

fn commit_checkpoint(
    root: &Path,
    run: &Path,
    step: u64,
    store: &nn::VarStore,
    optimizer: &AdamW,
    config: &Config,
    stop_phase: Option<Phase>,
) -> Result<Value> {
    checkpoint_commit::commit(
        root,
        run,
        step,
        |staging| {
            store.save(staging.join("model.ot"))?;
            optimizer.save(&staging.join("optimizer.ot"))?;
            fs::write(staging.join("config.json"), serde_json::to_vec(config)?)?;
            Ok(())
        },
        |staging| {
            let config_sha256 = jope_parallel::manifest::file_hash(&staging.join("config.json"))?;
            let optimizer_sha256 =
                jope_parallel::manifest::file_hash(&staging.join("optimizer.ot"))?;
            let optimizer_metadata_sha256 =
                jope_parallel::manifest::file_hash(&staging.join("optimizer.json"))?;
            let model_sha256 = jope_parallel::manifest::file_hash(&staging.join("model.ot"))?;
            let state = json!({
                "schema": checkpoint_commit::SCHEMA,
                "step": step,
                "world": 1,
                "global_batch": BATCH,
                "corpus_sha256": hash(b"synthetic-train-only-v1"),
                "vocabulary_sha256": hash(b"synthetic-vocabulary-v1"),
                "seed": 20261018,
                "sampler_id": "i32_synthetic_train_only/v1",
                "initialization": {"kind":"synthetic_fixture", "accepted_parent": "step-0"},
                "autocast_cache_lifetime": "cpu_disabled",
                "ranking_backward": jope_parallel::neural::RANKING_BACKWARD_ID,
                "runtime_objective": objective::RUNTIME_OBJECTIVE_ID,
                "execution_partition": jope_parallel::task_contract::training_partition_of(
                    jope_parallel::task_contract::PARTITION_V1_SCHEMA, 1, BATCH, BATCH
                )?,
                "config_sha256": config_sha256,
                "optimizer_metadata_sha256": optimizer_metadata_sha256,
                "optimizer_sha256": optimizer_sha256,
                "model_sha256": model_sha256,
                "objective": objective::identity(),
                "objective_sha256": objective::identity_sha256()
            });
            Ok(state)
        },
        &mut |phase| {
            if Some(phase) == stop_phase {
                anyhow::bail!("injected I32 interruption at {phase:?}");
            }
            Ok(())
        },
    )
}

fn fixture_inputs() -> (Tensor, Tensor, Tensor, Tensor, Tensor, Vec<RowLabels>) {
    let dataset = Tensor::arange((BATCH * 4) as i64, (Kind::Float, Device::Cpu))
        .reshape([BATCH as i64, 4])
        / 17.0;
    let target = Tensor::ones([BATCH as i64, 4], (Kind::Float, Device::Cpu)) * 0.25;
    let mut token_data = Vec::with_capacity(BATCH * 2);
    for row in 0..BATCH {
        token_data.extend([1i64, 2 + (row % 5) as i64]);
    }
    let tokens = Tensor::from_slice(&token_data).reshape([BATCH as i64, 2]);
    let pad = Tensor::ones([BATCH as i64, 2], (Kind::Bool, Device::Cpu));
    let legal = Tensor::ones([BATCH as i64, 1, VOCAB], (Kind::Bool, Device::Cpu));
    let rows = (0..BATCH)
        .map(|record| RowLabels {
            record,
            context: 0,
            value: 0.1 + record as f32 * 0.1,
            percentile: record as f32 / (BATCH - 1) as f32,
            runtime: 0.1 + record as f32 * 0.01,
        })
        .collect();
    (dataset, target, tokens, pad, legal, rows)
}

fn actor_behavior(
    policy: &Policy,
    dataset: &Tensor,
    target: &Tensor,
    tokens: &Tensor,
    pad: &Tensor,
) -> Result<(String, Vec<i64>)> {
    let (logits, _, _, _, _) = policy.forward(dataset, target, tokens, pad, false)?;
    let next = logits.select(1, 2);
    let values = Vec::<f32>::try_from(
        next.detach()
            .to_device(Device::Cpu)
            .to_kind(Kind::Float)
            .flatten(0, -1),
    )?;
    let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let proposals = Vec::<i64>::try_from(next.argmax(-1, false))?;
    Ok((hash(&bytes), proposals))
}

fn verify_independent_reload(checkpoint: &Path, output: &Path) -> Result<()> {
    let root = checkpoint
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .context("checkpoint must be under RUN/checkpoints/step-N")?;
    let state: Value = serde_json::from_slice(&fs::read(checkpoint.join("state.json"))?)?;
    let step = state["step"]
        .as_u64()
        .context("checkpoint step is absent")?;
    checkpoint_commit::verify_dir(root, checkpoint, step, &Expectation::default())?;

    let config: Config = serde_json::from_slice(&fs::read(checkpoint.join("config.json"))?)?;
    let mut store = nn::VarStore::new(Device::Cpu);
    let policy = Policy::new(store.root(), config)?;
    store.load(checkpoint.join("model.ot"))?;
    let reloaded_parameters_sha256 = parameters_sha256(&store)?;
    let (dataset, target, tokens, pad, _, _) = fixture_inputs();
    let (child_actor_sha256, _) = actor_behavior(&policy, &dataset, &target, &tokens, &pad)?;

    let capsule: Value = serde_json::from_slice(&fs::read(root.join("capsule.json"))?)?;
    ensure!(
        capsule["child_parameters_sha256"] == reloaded_parameters_sha256,
        "reloaded parameters differ from the engine capsule"
    );
    ensure!(
        capsule["parent_actor_logits_sha256"] != child_actor_sha256,
        "next actor behavior did not change from the parent"
    );
    let receipt = json!({
        "independent_reload_verified": true,
        "parameter_digest_matches_engine": true,
        "next_actor_behavior_changed": true,
        "dry_run_only": true,
        "reloaded_parameters_sha256": reloaded_parameters_sha256,
        "next_actor_logits_sha256": child_actor_sha256
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().collect();
    if args
        .get(1)
        .is_some_and(|arg| arg == "--i32-reload-checkpoint")
    {
        ensure!(
            args.len() == 7 && args[3] == "--output" && args[5] == "--cpu"
                && args[6] == "--dry-run-next-actor",
            "usage: research_loop_fixture --i32-reload-checkpoint CHECKPOINT --output RELOAD.json --cpu --dry-run-next-actor"
        );
        return verify_independent_reload(&PathBuf::from(&args[2]), &PathBuf::from(&args[4]));
    }
    ensure!(
        args.len() == 3 && args[1] == "--output",
        "usage: research_loop_fixture --output DIR"
    );
    let output = PathBuf::from(&args[2]);
    fs::create_dir_all(&output)?;
    tch::manual_seed(20261018);

    // The leakage canary is deliberately presented to role admission and refused before any
    // tokenization, objective construction, gradient or optimizer update.
    let canary_refused = reject_non_train_role("sealed_meta_test", SEALED_CANARY).is_err();
    ensure!(canary_refused, "sealed canary unexpectedly admitted");

    let config = configure();
    let parent_store = nn::VarStore::new(Device::Cpu);
    let parent_policy = Policy::new(parent_store.root(), config.clone())?;
    let parent_parameters_sha256 = parameters_sha256(&parent_store)?;
    let accepted_run = output.join("accepted-parent");
    let initial_optimizer = AdamW::new(1e-3, 0.0);
    let _ = commit_checkpoint(
        &output,
        &accepted_run,
        0,
        &parent_store,
        &initial_optimizer,
        &config,
        None,
    )?;
    let parent_latest_before = fs::read(accepted_run.join("latest.json"))?;
    let (dataset, target, tokens, pad, legal, rows) = fixture_inputs();
    let (parent_actor_sha256, parent_proposals) =
        actor_behavior(&parent_policy, &dataset, &target, &tokens, &pad)?;

    let (logits, rewards, runtime, stability, ablation) =
        parent_policy.forward(&dataset, &target, &tokens, &pad, true)?;
    let outputs = Outputs {
        logits: &logits,
        rewards: &rewards,
        runtime: &runtime,
        stability: &stability,
        ablation: &ablation,
    };
    let (loss, _) = objective::composite(
        &outputs,
        parent_policy.loss_log_vars.values(),
        &tokens,
        &pad,
        &legal,
        &rows,
    )?;
    let loss_value = loss.double_value(&[]);
    loss.backward();
    let mut optimizer = AdamW::new(1e-3, 0.0);
    optimizer.update_reference(&parent_store)?;
    ensure!(
        optimizer.step == 1,
        "native optimizer did not apply exactly one update"
    );
    let child_parameters_sha256 = parameters_sha256(&parent_store)?;
    ensure!(
        child_parameters_sha256 != parent_parameters_sha256,
        "native update did not change parameters"
    );

    let child_run = output.join("rejected-child");
    let child_commit = commit_checkpoint(
        &output,
        &child_run,
        1,
        &parent_store,
        &optimizer,
        &config,
        Some(Phase::LatestStaged),
    );
    ensure!(
        child_commit.is_err(),
        "publication crash injection did not interrupt"
    );
    let child_checkpoint = child_run.join("checkpoints/step-1");
    let committed =
        checkpoint_commit::verify_dir(&output, &child_checkpoint, 1, &Expectation::default())?;
    ensure!(
        fs::read(accepted_run.join("latest.json"))? == parent_latest_before,
        "rejected child changed accepted parent pointer"
    );

    // Reload the committed child into a fresh native actor. The external I14 Candle fixture must
    // independently consume this checkpoint before a complete closure can be reported.
    let mut reloaded_store = nn::VarStore::new(Device::Cpu);
    let reloaded_policy = Policy::new(reloaded_store.root(), config.clone())?;
    reloaded_store.load(child_checkpoint.join("model.ot"))?;
    let reloaded_parameters_sha256 = parameters_sha256(&reloaded_store)?;
    ensure!(
        reloaded_parameters_sha256 == child_parameters_sha256,
        "native reload changed parameters"
    );
    let (child_actor_sha256, child_proposals) =
        actor_behavior(&reloaded_policy, &dataset, &target, &tokens, &pad)?;
    ensure!(
        child_actor_sha256 != parent_actor_sha256,
        "next actor behavior did not change after reload"
    );

    // The engine fixture emits facts. It does not substitute a local reload for I14's independent
    // Candle reload, or mark the resulting dry-run/publication as accepted.
    let capsule = json!({
        "schema": "jope.research_loop_smoke.v1",
        "status": "mechanism_fixture_partial_pending_independent_reload",
        "permission": {"kind":"synthetic_cpu_fixture", "qualifying":false},
        "dataset_role": "synthetic_train_only",
        "trace": ["propose", "exact_train_objective", "nonqualifying_admission", "native_cpu_update", "checkpoint_commit", "native_reload", "next_actor", "dry_run_judge", "capsule", "publication_pending"],
        "objective_loss": loss_value,
        "native_optimizer_steps": optimizer.step,
        "parent_parameters_sha256": parent_parameters_sha256,
        "child_parameters_sha256": child_parameters_sha256,
        "parameters_changed": parent_parameters_sha256 != child_parameters_sha256,
        "parent_actor_logits_sha256": parent_actor_sha256,
        "child_actor_logits_sha256": child_actor_sha256,
        "actor_behavior_changed": parent_actor_sha256 != child_actor_sha256,
        "parent_proposals": parent_proposals,
        "child_proposals": child_proposals,
        "accepted_parent_latest": accepted_run.join("latest.json"),
        "child_checkpoint": child_checkpoint,
        "committed_child_state_sha256": committed.state_sha256,
        "leakage_canary": {"canary_sha256":hash(SEALED_CANARY.as_bytes()), "refused_before_objective":canary_refused, "split":"sealed_meta_test"},
        "publication_crash": {"injected_at":"LatestStaged", "child_checkpoint_complete":true, "accepted_parent_pointer_unchanged":true},
        "judge": {"kind":"dry_run", "decision":"rejected", "accepted_parent_step":0, "rejected_child_step":1},
        "independent_candle_reload": "pending external I14 fixture",
        "scientific_benefit_claim": false,
        "serving_promotion_claim": false,
        "publication": "pending"
    });
    let capsule_path = output.join("capsule.json");
    fs::write(&capsule_path, serde_json::to_vec_pretty(&capsule)?)?;
    ensure!(
        !fs::read(&capsule_path)?
            .windows(SEALED_CANARY.len())
            .any(|w| w == SEALED_CANARY.as_bytes()),
        "leakage canary appeared in exported capsule"
    );
    println!("{}", serde_json::to_string(&capsule)?);
    Ok(())
}

fn reject_non_train_role(role: &str, _canary: &str) -> Result<()> {
    ensure!(
        role == "train",
        "refuse non-TRAIN source role before proposal construction"
    );
    Ok(())
}
