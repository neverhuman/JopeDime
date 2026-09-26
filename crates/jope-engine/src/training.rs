//! Synchronous training on saved real contexts and adaptive GP teaching labels.
use crate::{
    corpus::{Index, SEQUENCE},
    manifest::*,
    neural::{AdamW, Bf16, Config, Nccl, Policy},
    telemetry::now,
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::{Duration, Instant},
};
use tch::{nn, Device, Kind, Tensor};

pub mod ancestry;
pub mod checkpoint_commit;
pub mod consumption;
pub mod continuation;
pub mod inputs;
pub mod objective;
pub mod partition;
pub mod topology;

pub const SAMPLER_ID: &str = "family_uniform_champion_plus_seven_quantiles_v2";
pub const REPLAY_SAMPLER_ID: &str = "lineage_fresh_program_champion_plus_seven_v2";
pub const SEARCH_BEATING_SAMPLER_ID: &str = "search_beating_champion_plus_seven_v1";

pub fn active_sampler_id(policy: Option<&Value>) -> &'static str {
    match policy {
        Some(p)
            if p["schema"] == crate::checkpoint_recovery::CONTINUATION_SCHEMA
                && p.get("replay_policy_ref").is_none() =>
        {
            SAMPLER_ID
        }
        Some(p)
            if p["discoveries"] == "teacher_improved"
                || p["discoveries"] == "search_beating"
                || p["search_beating"] == true =>
        {
            SEARCH_BEATING_SAMPLER_ID
        }
        Some(_) => REPLAY_SAMPLER_ID,
        None => SAMPLER_ID,
    }
}
thread_local! {
    static RESIDENT_TENSORS: std::cell::RefCell<Option<(String,Vec<Tensor>)>> = const { std::cell::RefCell::new(None) };
    static RESIDENT_CONTEXT: std::cell::RefCell<Option<(String,Tensor)>> = const { std::cell::RefCell::new(None) };
}

pub fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
/// Every rank-local comparison contains the best observed teacher and seven diverse comparators.
/// Records are sorted by increasing cut RMSE by the immutable corpus builder. Families receive equal mass.
pub fn sample(
    index: &Index,
    families: &BTreeMap<String, Vec<usize>>,
    serial: usize,
    count: usize,
    seed: u64,
) -> Vec<usize> {
    assert!(count % 8 == 0 && serial % 8 == 0 && !families.is_empty());
    let families = families.values().collect::<Vec<_>>();
    let mut result = Vec::with_capacity(count);
    for g in serial / 8..(serial + count) / 8 {
        let cycle = g / families.len();
        let family = (g + mix(seed ^ cycle as u64) as usize) % families.len();
        let candidates = families[family];
        let h = mix(seed ^ g as u64);
        let group = &index.groups[candidates[h as usize % candidates.len()]];
        assert!(
            group.records.len() >= 8,
            "teaching comparison needs eight distinct programs"
        );
        result.push(group.records[0]);
        for band in 0..7 {
            let lo = 1 + band * (group.records.len() - 1) / 7;
            let hi = 1 + (band + 1) * (group.records.len() - 1) / 7;
            result.push(
                group.records
                    [lo + mix(h ^ ((band + 1) as u64).wrapping_mul(7919)) as usize % (hi - lo)],
            );
        }
    }
    result
}

/// Alternate equal-size fresh and historical portions, balancing families in each.
pub fn sample_replay(
    index: &Index,
    families: &BTreeMap<String, Vec<usize>>,
    fresh: &BTreeMap<String, Vec<usize>>,
    serial: usize,
    count: usize,
    seed: u64,
    balanced: bool,
) -> Vec<usize> {
    assert!(serial % 8 == 0 && count % 8 == 0);
    if !balanced || fresh.is_empty() {
        return sample(index, families, serial, count, seed);
    }
    let mut result = Vec::with_capacity(count);
    for comparison in serial / 8..(serial + count) / 8 {
        let portion = if comparison % 2 == 0 { fresh } else { families };
        result.extend(sample(index, portion, comparison / 2 * 8, 8, seed));
    }
    result
}

/// A designated fresh comparison must actually consume a newly admitted or
/// improved program. Preserve the champion and seven distinct comparators.
pub fn sample_replay_fresh(
    index: &Index,
    families: &BTreeMap<String, Vec<usize>>,
    fresh: &BTreeMap<String, Vec<usize>>,
    fresh_by_teacher: &BTreeMap<usize, Vec<usize>>,
    serial: usize,
    count: usize,
    seed: u64,
    balanced: bool,
) -> Vec<usize> {
    let mut result = sample_replay(index, families, fresh, serial, count, seed, balanced);
    if !balanced || fresh.is_empty() {
        return result;
    }
    for (n, ids) in result.chunks_mut(8).enumerate() {
        let comparison = serial / 8 + n;
        if comparison % 2 != 0 {
            continue;
        }
        let eligible = &fresh_by_teacher[&ids[0]];
        assert!(!eligible.is_empty(), "fresh group needs a fresh program");
        if !ids.iter().any(|id| eligible.contains(id)) {
            let fresh_id =
                eligible[mix(seed ^ comparison as u64 ^ 0x6672657368) as usize % eligible.len()];
            // Replace the closest sampled score band, keeping the champion.
            let slot = (1..8)
                .min_by_key(|&slot| ids[slot].abs_diff(fresh_id))
                .unwrap();
            ids[slot] = fresh_id;
        }
    }
    result
}
fn validation_families(index: &Index) -> Result<BTreeMap<String, Vec<usize>>> {
    let mut families = BTreeMap::<String, Vec<usize>>::new();
    for (i, group) in index.groups.iter().enumerate() {
        if group.partition == "meta_validation" {
            ensure!(
                group.records.len() >= 8,
                "validation group lacks comparators"
            );
            families.entry(group.family.clone()).or_default().push(i);
        }
    }
    Ok(families)
}

/// Fixed global validation panel, identical on every evaluation of a corpus
/// (partition v1: 256 rows per rank).
pub fn validation_panel(index: &Index, world: usize, seed: u64) -> Result<Vec<Vec<usize>>> {
    let families = validation_families(index)?;
    ensure!(
        !families.is_empty() && [1, 2, 4, 8].contains(&world),
        "validation panel requires families and a supported world size"
    );
    Ok((0..world)
        .map(|rank| sample(index, &families, rank * 256, 256, seed ^ 0x76616c))
        .collect())
}

/// The fixed validation panel of one attempt.
enum ValidationPanel {
    /// Partition v1: 256 rows per rank.
    V1(Vec<Vec<usize>>),
    /// Partition v2: every rank's microbatches of the fixed global panel,
    /// the panel in global order, and its global denominators.
    V2 {
        per_rank: Vec<Vec<Vec<usize>>>,
        global_ids: Vec<usize>,
        denominators: partition::Denominators,
    },
}

/// Partition v2 (W7-T06): one fixed global panel of
/// `partition::VALIDATION_PANEL_ROWS` rows, whatever the rank count, split
/// by `plan` into each rank's microbatches (`[rank][microbatch]` record ids).
/// Its rows are those of the v1 eight-rank panel, so the panel and the
/// globally normalized validation objective do not depend on the rank count.
pub fn validation_panel_partitioned(
    index: &Index,
    plan: &partition::Plan,
    seed: u64,
) -> Result<Vec<Vec<Vec<usize>>>> {
    ensure!(
        plan.global_batch() == partition::VALIDATION_PANEL_ROWS,
        "the validation panel has {} rows",
        partition::VALIDATION_PANEL_ROWS
    );
    let families = validation_families(index)?;
    ensure!(
        !families.is_empty(),
        "validation panel requires families and a supported world size"
    );
    (0..plan.world())
        .map(|rank| {
            Ok(plan
                .slots(rank, seed, 0)?
                .iter()
                .map(|slot| {
                    sample(
                        index,
                        &families,
                        slot.serial as usize,
                        slot.rows,
                        seed ^ 0x76616c,
                    )
                })
                .collect())
        })
        .collect()
}

pub(crate) fn parameter_hash(vs: &nn::VarStore) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for (name, p) in vs.variables().into_iter().collect::<BTreeMap<_, _>>() {
        h.update(name);
        hash_parameter_values(
            &mut h,
            &Vec::<f32>::try_from(p.to_device(Device::Cpu).flatten(0, -1))?,
        );
    }
    Ok(hex::encode(h.finalize()))
}
fn hash_parameter_values(h: &mut sha2::Sha256, values: &[f32]) {
    use sha2::Digest;
    #[cfg(target_endian = "little")]
    {
        // f32 has no padding and this read-only byte view stays within the live
        // slice. It preserves the previous per-value to_le_bytes representation.
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        h.update(bytes);
    }
    #[cfg(target_endian = "big")]
    for value in values {
        h.update(value.to_le_bytes());
    }
}
fn checkpoint(
    root: &Path,
    run: &Path,
    launch: &Path,
    rank: usize,
    world: usize,
    comm: &Nccl,
    vs: &nn::VarStore,
    opt: &AdamW,
    config: &Config,
    corpus_hash: &str,
    vocab_hash: &str,
    seed: u64,
    epoch_steps: usize,
    initialization: &Value,
    validation: &Value,
    global_batch: usize,
    execution: &partition::Execution,
    partition_state: Option<&partition::State>,
    batch_qualification: bool,
    exposure: &Option<crate::training_exposure::Counts>,
    family_groups: &crate::training_exposure::FamilyGroups,
    family_groups_known: bool,
    consumption: &Value,
    training_inputs_sha256: &str,
    binding: &topology::Binding,
) -> Result<()> {
    let step = opt.step;
    let receipt = launch.join(format!("checkpoint-{step}-rank-{rank}.json"));
    let mut rank_receipt = json!({"rank":rank,"step":step,"parameter_sha256":parameter_hash(vs)?,"exposure":exposure,
        "committed_family_groups":family_groups,"attempt":launch.strip_prefix(root)?,
        "consumption_journal":consumption,"topology":binding.receipt()});
    if !family_groups_known {
        rank_receipt["family_groups_baseline"] = json!("unknown");
    }
    write_json(&receipt, &rank_receipt)?;
    let mut barrier = Tensor::ones([1], (Kind::Float, Device::Cuda(binding.device_index())));
    comm.reduce(&mut barrier)?;
    if rank == 0 {
        let receipts = (0..world)
            .map(|r| launch.join(format!("checkpoint-{step}-rank-{r}.json")))
            .collect::<Vec<_>>();
        let hashes = receipts
            .iter()
            .map(|path| read_json::<Value>(path))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            hashes.iter().enumerate().all(|(rank, r)| r["rank"] == rank
                && r["step"] == step
                && r["parameter_sha256"] == hashes[0]["parameter_sha256"]),
            "checkpoint ranks diverged"
        );
        let rank_receipts = receipts
            .iter()
            .enumerate()
            .map(|(r, path)| {
                Ok(json!({"rank":r,"path":path.strip_prefix(root)?,"sha256":file_hash(path)?}))
            })
            .collect::<Result<Vec<_>>>()?;
        // W6-T05: staged objects, then the manifest, verified as a loader
        // would, then an atomic rename and a uniquely staged latest.json.
        checkpoint_commit::commit(
            root,
            run,
            step,
            |staging| {
                vs.save(staging.join("model.ot"))?;
                opt.save(&staging.join("optimizer.ot"))?;
                write_json(&staging.join("config.json"), config)
            },
            |staging| {
                let state = json!({"schema":"parallel.checkpoint.v1","step":step,"world":world,"global_batch":global_batch,"model_sha256":file_hash(&staging.join("model.ot"))?,"optimizer_sha256":file_hash(&staging.join("optimizer.ot"))?,"optimizer_metadata_sha256":file_hash(&staging.join("optimizer.json"))?,"config_sha256":file_hash(&staging.join("config.json"))?,"corpus_sha256":corpus_hash,"vocabulary_sha256":vocab_hash,"parameter_sha256":hashes[0]["parameter_sha256"],"seed":seed,"rng":"counter reset before every microbatch: mix(seed ^ step*65537 ^ rank*257 ^ microbatch); LibTorch CPU/CUDA manual_seed","sampler":format!("family-uniform best teacher plus seven quantile comparators; next global example = step*{global_batch}"),"sampler_id":SAMPLER_ID,"scheduler":{"type":"constant","learning_rate":opt.lr,"epoch_steps":epoch_steps},"initialization":initialization,"validation":validation,"autocast_cache_lifetime":"scoped_v2","qualification":true,"independent_evaluation":false,"final_tests_opened":false});
                let mut state = state;
                if initialization.get("replay_policy_sha256").is_some() {
                    state["sampler_id"] = json!(initialization
                        .get("resolved_sampler_id")
                        .and_then(Value::as_str)
                        .unwrap_or_else(|| active_sampler_id(initialization.get("replay_policy"))));
                    state["replay_policy_sha256"] = initialization["replay_policy_sha256"].clone();
                    state["local_step"] = json!(
                        step - initialization["optimizer_parent_step"]
                            .as_u64()
                            .unwrap_or(0)
                    );
                    state["stochastic_counter_basis"] =
                        json!("optimizer_step_minus_optimizer_parent_step");
                    state["next_global_example"] = json!((step
                        - initialization["optimizer_parent_step"]
                            .as_u64()
                            .unwrap_or(0))
                    .checked_mul(global_batch as u64)
                    .context("checkpoint sampler offset overflow")?);
                }
                if batch_qualification {
                    state["batch_qualification"] = json!(true);
                    state["eligible_checkpoint"] = json!(false);
                }
                state["ranking_backward"] = json!(crate::neural::RANKING_BACKWARD_ID);
                state["runtime_objective"] = json!("huber_log1p_seconds_v2");
                // v1 records the same value as before W7-T06.
                state["execution_partition"] = execution.record().clone();
                state["committed_exposure_by_rank"] = json!(hashes
            .iter()
            .map(|row| {
                let mut committed = json!({"rank":row["rank"],"counts":row["exposure"],
                    "committed_family_groups":row["committed_family_groups"],"attempt":row["attempt"],
                    "consumption_journal":row["consumption_journal"]});
                if let Some(baseline) = row.get("family_groups_baseline") {
                    committed["family_groups_baseline"] = baseline.clone();
                }
                committed
            })
            .collect::<Vec<_>>());
                state["training_inputs_sha256"] = json!(training_inputs_sha256);
                state["objective"] = execution.objective_identity();
                state["objective_sha256"] = json!(execution.objective_sha256());
                state["checkpoint_format"] = json!(checkpoint_commit::FORMAT);
                state["required_objects"] = json!(checkpoint_commit::required_object_names());
                state["rank_receipts"] = json!(rank_receipts);
                if let Some(partition_state) = partition_state {
                    state["partition_state"] = serde_json::to_value(partition_state)?;
                }
                Ok(state)
            },
            &mut checkpoint_commit::no_stop,
        )?;
        // Publication never deletes durable recovery history. Replication and
        // boundary certification are separate from this local checkpoint commit.
    }
    comm.reduce(&mut barrier)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::hash_parameter_values;
    use sha2::{Digest, Sha256};

    #[test]
    fn batched_parameter_hash_preserves_every_float_bit() {
        let values = [
            0u32, 0x80000000, 0x3f800000, 0xbf800000, 0x00000001, 0x7f800000, 0xff800000,
            0x7fc00001, 0x7f800001, 0xffffffff,
        ]
        .map(f32::from_bits);
        let mut old = Sha256::new();
        let mut new = Sha256::new();
        old.update("tensor-name");
        new.update("tensor-name");
        for value in values {
            old.update(value.to_le_bytes());
        }
        hash_parameter_values(&mut new, &values);
        assert_eq!(old.finalize(), new.finalize());
    }

    #[test]
    #[ignore = "explicit full-policy CPU checkpoint hash benchmark"]
    fn benchmark_parameter_hash() {
        let values = vec![0.25f32; 27_796_494];
        let mut timings = Vec::new();
        let mut digests = Vec::new();
        for batched in [false, true] {
            let begun = std::time::Instant::now();
            for _ in 0..3 {
                let mut h = Sha256::new();
                if batched {
                    hash_parameter_values(&mut h, &values);
                } else {
                    for value in &values {
                        h.update(value.to_le_bytes());
                    }
                }
                digests.push(h.finalize());
            }
            timings.push(begun.elapsed().as_secs_f64() / 3.0);
        }
        assert!(digests.iter().all(|x| *x == digests[0]));
        eprintln!("checkpoint SHA256 benchmark: 27,796,494 FP32 parameters, per-float {:.3} ms, batched {:.3} ms, {:.2}x speedup, identical digests",
            timings[0]*1000.0,timings[1]*1000.0,timings[0]/timings[1]);
    }
}

pub fn rank(
    root: &Path,
    corpus: &Path,
    run: &Path,
    launch: &Path,
    rank: usize,
    world: usize,
    micro: usize,
    steps: usize,
    checkpoint_steps: usize,
    resume: bool,
    initialize: Option<&Path>,
) -> Result<()> {
    rank_with_policy(
        root,
        corpus,
        run,
        launch,
        rank,
        world,
        micro,
        steps,
        checkpoint_steps,
        resume,
        initialize,
        None,
        None,
    )
}

/// `continuation_mode` is the request's optional explicit lineage-mode
/// declaration; `None` keeps the implicit meaning of older requests.
pub fn rank_with_policy(
    root: &Path,
    corpus: &Path,
    run: &Path,
    launch: &Path,
    rank: usize,
    world: usize,
    micro: usize,
    steps: usize,
    checkpoint_steps: usize,
    resume: bool,
    initialize: Option<&Path>,
    policy: Option<&Path>,
    continuation_mode: Option<continuation::Mode>,
) -> Result<()> {
    crate::task_contract::require_original_gpu_grant("train-rank")?;
    let _deadline = crate::gpu_worker::deadline_watchdog()?;
    if std::env::var_os("JOPE_GPU_WORKER_ACTIVE").is_some() {
        crate::gpu_worker::require_accepted_task(
            root,
            &crate::gpu_worker::Task::Train {
                corpus: corpus.into(),
                run: run.into(),
                launch: launch.into(),
                world,
                micro,
                steps,
                checkpoint_steps,
                resume,
                initialize: initialize.map(Into::into),
                replay_policy: policy.map(Into::into),
                continuation_mode,
            },
        )?;
    }
    let (_lease, partition_version) = crate::gpu_worker::verify_training_launch(
        root, corpus, run, launch, world, micro, steps, resume, initialize, policy,
    )?;
    if crate::gpu_worker::enabled() {
        return crate::gpu_worker::submit(
            root,
            rank,
            crate::gpu_worker::Task::Train {
                corpus: corpus.into(),
                run: run.into(),
                launch: launch.into(),
                world,
                micro,
                steps,
                checkpoint_steps,
                resume,
                initialize: initialize.map(Into::into),
                replay_policy: policy.map(Into::into),
                continuation_mode,
            },
        );
    }
    rank_impl(
        root,
        corpus,
        run,
        launch,
        rank,
        world,
        micro,
        steps,
        checkpoint_steps,
        resume,
        initialize,
        2048,
        false,
        policy,
        None,
        false,
        continuation_mode,
        None,
        partition_version,
    )
}

/// Explicit policy warm-start experiment; ordinary production/resume stays unchanged.
pub fn rank_replay(
    root: &Path,
    corpus: &Path,
    run: &Path,
    launch: &Path,
    rank: usize,
    world: usize,
    steps: usize,
    resume: bool,
    initialize: Option<&Path>,
    policy: &Path,
) -> Result<()> {
    rank_with_policy(
        root,
        corpus,
        run,
        launch,
        rank,
        world,
        256,
        steps,
        1000,
        resume,
        initialize,
        Some(policy),
        None,
    )
}

/// Isolated real-corpus qualification; ordinary controllers keep their existing batch.
pub fn qualify_batch_rank(
    root: &Path,
    corpus: &Path,
    run: &Path,
    launch: &Path,
    rank: usize,
    micro: usize,
    global_batch: usize,
    steps: usize,
    initialize: &Path,
) -> Result<()> {
    crate::task_contract::require_qualified_gpu_experiment("qualify-batch-rank")?;
    ensure!(
        [256, 512, 1024, 2048].contains(&micro) && global_batch == micro * 8,
        "qualification requires an explicit eight-rank global batch"
    );
    let experiment = std::env::var_os("JOPE_EXPERIMENT_DIR")
        .context("batch qualification requires an isolated experiment directory")?;
    let experiment = Path::new(&experiment).canonicalize()?;
    for path in [root, run, launch] {
        ensure!(
            path.canonicalize()?.starts_with(&experiment),
            "batch qualification root, output and launch must stay inside its experiment"
        );
    }
    rank_impl(
        root,
        corpus,
        run,
        launch,
        rank,
        8,
        micro,
        steps,
        steps,
        false,
        Some(initialize),
        global_batch,
        true,
        None,
        None,
        false,
        None,
        None,
        partition::Version::V1,
    )
}

/// SmartCluster's opaque process-local admission is distinct from legacy funding.
pub(crate) fn smartcluster_rank(
    task: &crate::smartcluster_worker::VerifiedTask,
    receipt: &Path,
) -> Result<()> {
    use crate::smartcluster_worker::Operation;
    let Operation::Train {
        corpus,
        run,
        launch,
        micro,
        checkpoint_steps,
        initialize,
        replay_policy,
        initial_optimizer_step,
        requested_step_increment,
        target_optimizer_step,
        continuation_mode,
    } = task.operation()
    else {
        anyhow::bail!("training operation required")
    };
    ensure!(
        !task.stop().load(std::sync::atomic::Ordering::Acquire),
        "native control already lost"
    );
    ensure!(
        initial_optimizer_step.checked_add(*requested_step_increment)
            == Some(*target_optimizer_step),
        "original training target changed"
    );
    let root = task.root();
    let corpus = resolve(root, corpus)?;
    let run = resolve(root, run)?;
    let launch_root = resolve(root, launch)?;
    let initial = resolve(root, initialize)?;
    let policy_path = resolve(root, replay_policy)?;
    let parent_bytes = fs::read(initial.join("state.json"))?;
    let parent: Value = serde_json::from_slice(&parent_bytes)?;
    let policy: Value = read_json(&policy_path)?;
    // An optimizer carry continues the parent's optimizer step. A `jope.native_loop.v2` weight
    // warm start (OWNER-DECISIONS.md D12) restarts it at zero from the bound parent weights and
    // is admitted only as a declared warm start over a verified native continuation corpus.
    let warm_start =
        crate::native_loop::warm_start::policy_mode(&policy, &hash(&parent_bytes), &parent)?;
    ensure!(
        (if warm_start.is_some() {
            *initial_optimizer_step == 0
        } else {
            parent["step"] == *initial_optimizer_step
        }) && file_hash(&initial.join("model.ot"))? == parent["model_sha256"]
            && file_hash(&initial.join("optimizer.ot"))? == parent["optimizer_sha256"],
        "original checkpoint model/optimizer changed"
    );
    ensure!(
        warm_start.is_some() || policy["optimizer"] == "carry",
        "native continuation must preserve optimizer bytes"
    );
    let corpus_index: Value = read_json(&corpus.join("index.json"))?;
    let native_continuation = corpus_index["qualification"] == false;
    if warm_start.is_some() {
        ensure!(
            native_continuation && *continuation_mode == Some(continuation::Mode::WeightWarmStart),
            "a weight warm start is admitted only as a declared native continuation"
        );
    }
    if native_continuation {
        verify_native_training_eligibility(
            task,
            &policy,
            warm_start
                .as_ref()
                .map(|binding| binding["parent_step"].as_u64().context("parent step"))
                .transpose()?,
        )?;
    }
    let original = json!({"schema":"jope.smartcluster_optimizer_window.v1","initial_optimizer_step":initial_optimizer_step,"requested_step_increment":requested_step_increment,"target_optimizer_step":target_optimizer_step,"parent_state_sha256":file_hash(&initial.join("state.json"))?,"corpus_sha256":file_hash(&corpus.join("index.json"))?,"policy_sha256":file_hash(&policy_path)?,"microbatch":micro,"global_batch":2048});
    fs::create_dir_all(&run)?;
    fs::create_dir_all(&launch_root)?;
    let window = run.join("smartcluster-window.json");
    if window.exists() {
        ensure!(
            read_json::<Value>(&window)? == original,
            "original native optimizer window changed"
        );
    } else {
        crate::smartcluster_worker::write_immutable_record(&window, &original)?;
    }
    if task.recovering() {
        crate::checkpoint_recovery::reconcile(root, &run)?;
    }
    let launch = native_attempt_launch(&launch_root, task.attempt(), &original)?;
    let resume = task.recovering() && run.join("latest.json").exists();
    if resume {
        let latest: Value = read_json(&run.join("latest.json"))?;
        let checkpoint = resolve(
            root,
            Path::new(
                latest["checkpoint"]
                    .as_str()
                    .context("recovery checkpoint absent")?,
            ),
        )?;
        ensure!(
            file_hash(&checkpoint.join("state.json"))? == latest["state_sha256"],
            "recovery state changed"
        );
        let state: Value = read_json(&checkpoint.join("state.json"))?;
        let step = state["step"]
            .as_u64()
            .context("recovery optimizer step absent")?;
        ensure!(
            step >= *initial_optimizer_step && step <= *target_optimizer_step,
            "recovery outside original optimizer window"
        );
        write_json(&launch.join("resume-input.json"), &latest)?;
    }
    rank_impl(
        root,
        &corpus,
        &run,
        &launch,
        0,
        1,
        *micro,
        (*requested_step_increment).try_into()?,
        *checkpoint_steps,
        resume,
        if resume { None } else { Some(initialize) },
        2048,
        false,
        Some(&policy_path),
        Some(task.stop()),
        native_continuation,
        *continuation_mode,
        None,
        partition::Version::V1,
    )?;
    let complete: Value = read_json(&launch.join("complete-0.json"))?;
    ensure!(
        complete["steps"] == *target_optimizer_step,
        "training did not finish original target"
    );
    let mut completion = json!({"schema":"jope.smartcluster_training_completion.v1","window":original,"numerical_receipt":complete,"latest":read_json::<Value>(&run.join("latest.json"))?});
    // A weight warm start's receipt names the parent weights it started from (D12).
    if let Some(binding) = warm_start {
        completion["warm_start"] = binding;
    }
    write_json(receipt, &completion)
}

/// NCCL rendezvous and rank receipts belong to one actual native allocation.
/// Preserve the original logical optimizer window and every old launch verbatim.
fn native_attempt_launch(
    root: &Path,
    attempt: jain_smartcluster::FencedAttempt,
    window: &Value,
) -> Result<std::path::PathBuf> {
    let launch = root.join(format!(
        "native-{}-attempt-{}-fence-{}",
        attempt.job_id,
        attempt.number.get(),
        attempt.fence.get()
    ));
    fs::create_dir(&launch)
        .context("native training allocation launch already exists or cannot be created")?;
    fs::File::open(root)?.sync_all()?;
    crate::smartcluster_worker::write_immutable_record(
        &launch.join("native-attempt.json"),
        &json!({"schema":"jope.native_training_launch.v1","attempt":attempt,"original_window_sha256":hash(&serde_json::to_vec(window)?)}),
    )?;
    Ok(launch)
}

/// This path is reachable only through the authenticated native worker. Legacy
/// rank entry points retain their original explicitly-qualified-corpus gate.
fn verify_native_training_eligibility(
    task: &crate::smartcluster_worker::VerifiedTask,
    policy: &Value,
    warm_start_parent_step: Option<u64>,
) -> Result<()> {
    use crate::{
        native_graph::Artifact,
        native_loop::{Request, Role},
        smartcluster_worker::Operation,
    };
    let Operation::Train {
        corpus,
        initialize,
        initial_optimizer_step,
        requested_step_increment,
        target_optimizer_step,
        ..
    } = task.operation()
    else {
        anyhow::bail!("native Train required")
    };
    let root = task.root();
    let manifest = task.manifest();
    let corpus = Artifact {
        path: corpus.join("index.json"),
        sha256: file_hash(&resolve(root, corpus)?.join("index.json"))?,
    };
    let parent = Artifact {
        path: initialize.join("state.json"),
        sha256: file_hash(&resolve(root, initialize)?.join("state.json"))?,
    };
    ensure!(
        manifest.parent_checkpoint_sha256.as_deref() == Some(parent.sha256.as_str())
            && manifest.corpus_sha256.as_deref() == Some(corpus.sha256.as_str()),
        "native training parent/corpus manifest differs"
    );
    let index: Value = crate::native_genome::load_corpus_index(root, &corpus)?;
    ensure!(
        index["qualification"] == false
            && index["label_policy"] == "strict_continuation_midrank_v1",
        "unqualified corpus is not native verified continuation targets"
    );
    let evidence = &policy["eligibility_evidence"];
    ensure!(
        evidence["schema"] == "jope.native_continuation_eligibility.v1",
        "native eligibility evidence absent"
    );
    let mut parameters = json!({"checkpoint":parent,"corpus":corpus,"initial_step":initial_optimizer_step,
        "requested_increment":requested_step_increment,"target_step":target_optimizer_step,"global_batch":2048});
    if let Some(step) = warm_start_parent_step {
        parameters["optimizer"] = json!("reset");
        parameters["parent_step"] = json!(step);
    }
    ensure!(
        evidence["original_window"] == parameters,
        "native replay original window differs"
    );
    let scientific: Artifact = serde_json::from_value(evidence["scientific_policy"].clone())?;
    let request = Request {
        schema: "jope.native_loop_request.v1".into(),
        run: manifest.run.clone(),
        generation: manifest.generation,
        id: manifest.task_id.clone(),
        role: Role::Train,
        parameters,
        predecessors: serde_json::from_value(evidence["verified_corpus_dependencies"].clone())?,
    };
    let expected = crate::native_loop::materialize::training_eligibility(
        root,
        &corpus,
        &parent,
        &scientific,
        &request,
    )?;
    ensure!(
        *policy == expected,
        "native replay differs from complete verified target reconstruction"
    );
    Ok(())
}

fn rank_impl(
    root: &Path,
    corpus: &Path,
    run: &Path,
    launch: &Path,
    rank: usize,
    world: usize,
    micro: usize,
    steps: usize,
    checkpoint_steps: usize,
    resume: bool,
    initialize: Option<&Path>,
    global_batch: usize,
    batch_qualification: bool,
    replay_policy: Option<&Path>,
    native_stop: Option<&std::sync::atomic::AtomicBool>,
    verified_native_continuation: bool,
    declared_mode: Option<continuation::Mode>,
    assignment: Option<&topology::Assignment>,
    partition_version: partition::Version,
) -> Result<()> {
    // W7-T03: a frozen host/device assignment is validated before any input,
    // device or communicator work. Without one, the pre-W7 single-host
    // mapping applies and is recorded as such.
    if let Some(assignment) = assignment {
        topology::validate(assignment)?;
        ensure!(
            assignment.world == world,
            "requested world differs from the frozen assignment"
        );
    }
    ensure!(
        rank < world && steps > 0 && checkpoint_steps > 0,
        "invalid training topology"
    );
    // W7-T06: the v1 gate is unchanged; partition v2 runs only when the
    // request selects it, with any world from one to eight.
    let execution = partition::execution(
        partition_version,
        world,
        micro,
        global_batch,
        batch_qualification,
    )?;
    crate::neural::configure_runtime();
    let policy: Option<Value> = replay_policy.map(read_json).transpose()?;
    let policy_sha = replay_policy.map(file_hash).transpose()?;
    let replay = policy
        .as_ref()
        .map(|p| crate::checkpoint_recovery::continuation_replay(root, p))
        .transpose()?
        .flatten();
    let seed = match &policy {
        Some(p) => p["sampling_seed"]
            .as_u64()
            .context("replay sampling seed")?,
        None => SEED,
    };
    let corpus_hash = file_hash(&corpus.join("index.json"))?;
    let index_bytes = fs::read(corpus.join("index.json"))?;
    ensure!(hash(&index_bytes) == corpus_hash, "corpus index changed");
    let verified_index: Value = serde_json::from_slice(&index_bytes)?;
    crate::corpus_projection::verify_admitted_context_closure(&verified_index)?;
    // W6 follow-up: the resident view is loaded from the verified bytes, so a
    // file substituted after verification cannot be loaded; T01's binding
    // stays as a second check.
    let resident = crate::teaching_store::load_resident_verified(root, corpus, &index_bytes)?;
    let (index, records) = resident.as_ref();
    inputs::bind_resident(&verified_index, index)?;
    ensure!(
        ["parallel.teaching_corpus.v1", crate::teaching_store::SCHEMA]
            .contains(&index.schema.as_str())
            && (index.qualification || (native_stop.is_some() && verified_native_continuation)),
        "training currently requires explicitly qualified real corpus"
    );
    // Parse the vocabulary from the bytes that were verified.
    let vocabulary: crate::grammar::Vocab = serde_json::from_slice(&inputs::read_verified(
        &corpus.join("vocabulary.json"),
        &index.vocabulary_sha256,
    )?)?;
    let frozen_inputs = inputs::freeze(index, &corpus_hash, &index.vocabulary_sha256, None)?;
    let frozen_inputs_sha = inputs::digest(&frozen_inputs)?;
    let mut families = BTreeMap::<String, Vec<usize>>::new();
    let mut validation = BTreeMap::<String, Vec<usize>>::new();
    let mut teacher_family = BTreeMap::<usize, String>::new();
    let mut train_records = 0;
    for (i, g) in index.groups.iter().enumerate() {
        if g.partition == "meta_train" {
            families.entry(g.family.clone()).or_default().push(i);
            train_records += g.records.len();
            if let Some(&teacher) = g.records.first() {
                ensure!(
                    teacher_family
                        .insert(teacher, g.family.clone())
                        .is_none_or(|family| family == g.family),
                    "comparison teacher belongs to two training families"
                );
            }
        } else if g.partition == "meta_validation" {
            validation.entry(g.family.clone()).or_default().push(i);
        }
    }
    ensure!(
        families.len() >= 2,
        "insufficient independent training families"
    );
    let mut fresh = BTreeMap::<String, Vec<usize>>::new();
    let mut fresh_by_teacher = BTreeMap::<usize, Vec<usize>>::new();
    let mut fresh_records = std::collections::BTreeSet::<usize>::new();
    let balanced = replay
        .as_ref()
        .is_some_and(|p| p["replay"] == "fresh_balanced");
    if let Some(p) = &replay {
        ensure!(
            p["schema"] == "parallel.replay_policy.v2"
                && p["eligible"] == true
                && p["corpus_sha256"] == corpus_hash
                && p["vocabulary_sha256"] == index.vocabulary_sha256
                && p["final_tests_opened"] == false,
            "replay policy identity or eligibility changed"
        );
        ensure!(
            ["broad", "fresh_balanced"]
                .iter()
                .any(|v| p["replay"] == *v)
                && ["reset", "carry"].iter().any(|v| p["optimizer"] == *v),
            "unknown replay/optimizer mode"
        );
        let mut seen = std::collections::BTreeSet::new();
        for value in p["fresh_groups"]
            .as_array()
            .context("fresh group indices")?
        {
            let i = value.as_u64().context("fresh group index")? as usize;
            ensure!(seen.insert(i), "duplicate fresh group");
            let g = index.groups.get(i).context("fresh group outside corpus")?;
            ensure!(
                g.partition == "meta_train" && g.records.len() >= 8,
                "fresh replay includes validation/test or insufficient comparators"
            );
            fresh.entry(g.family.clone()).or_default().push(i);
            let records = p["fresh_records"][i.to_string()]
                .as_array()
                .context("fresh program records absent")?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|id| id as usize)
                        .context("invalid fresh program record")
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                !records.is_empty()
                    && records.iter().all(|id| g.records.contains(id))
                    && records
                        .iter()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                        == records.len(),
                "fresh records must be distinct members of the selected training group"
            );
            fresh_records.extend(records.iter().copied());
            fresh_by_teacher.insert(g.records[0], records);
        }
    }
    let epoch_steps = train_records.div_ceil(global_batch).max(1);
    let fixed_panel = if validation.is_empty() {
        None
    } else {
        Some(match execution.plan() {
            None => ValidationPanel::V1(validation_panel(&index, world, SEED)?),
            Some(_) => {
                // W7-T06: one fixed global panel split by the plan, with its
                // global denominators from labels alone.
                let plan = partition::Plan::new(world, partition::VALIDATION_PANEL_ROWS, micro)?;
                let per_rank = validation_panel_partitioned(&index, &plan, SEED)?;
                let global_ids = per_rank
                    .iter()
                    .flatten()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>();
                let labels = global_ids
                    .iter()
                    .map(|&id| objective::RowLabels::from_record(id, &records.get(id)))
                    .collect::<Vec<_>>();
                let denominators = partition::denominators(&labels)?;
                ValidationPanel::V2 {
                    per_rank,
                    global_ids,
                    denominators,
                }
            }
        })
    };
    let panel_sha256 = fixed_panel
        .as_ref()
        .map(|panel| {
            match panel {
                ValidationPanel::V1(ids) => serde_json::to_vec(ids),
                ValidationPanel::V2 { global_ids, .. } => serde_json::to_vec(global_ids),
            }
            .map(|bytes| hash(&bytes))
        })
        .transpose()?;
    let config = Config::regression(vocabulary.tokens.len() as i64);
    let recovered_continuation = policy
        .as_ref()
        .filter(|p| p["schema"] == crate::checkpoint_recovery::CONTINUATION_SCHEMA)
        .map(|p| {
            crate::checkpoint_recovery::verify_continuation_partition(
                root,
                p,
                &corpus_hash,
                &index.vocabulary_sha256,
                execution.record(),
                steps,
            )
        })
        .transpose()?;
    ensure!(
        !(resume && initialize.is_some()),
        "resume and initialize are mutually exclusive"
    );
    // W6-T06: each attempt is exactly one of the four continuation modes.
    let mode = continuation::classify(resume, initialize.is_some(), policy.as_ref())?;
    // W6 follow-up: an explicit lineage-mode declaration must match the
    // request (or, on resume, the recorded lineage); none is recorded implicit.
    let declaration = continuation::declared_lineage(
        declared_mode,
        resume,
        initialize.is_some(),
        policy.as_ref(),
        if resume {
            Some(read_json::<Value>(&run.join("initialization.json"))?)
        } else {
            None
        }
        .as_ref(),
    )?;
    // The objective is the one the partition's reductions define.
    let objective_sha = execution.objective_sha256();
    let mut continuity = Value::Null;
    let selected_checkpoint = if resume {
        let latest: Value = read_json(&launch.join("resume-input.json"))?;
        let cp = resolve(
            root,
            Path::new(latest["checkpoint"].as_str().context("checkpoint path")?),
        )?;
        ensure!(
            file_hash(&cp.join("state.json"))? == latest["state_sha256"],
            "checkpoint selection changed"
        );
        Some(cp)
    } else {
        initialize.map(|path| resolve(root, path)).transpose()?
    };
    // All compatibility and serialized-content checks precede CUDA/NCCL.
    // Deserialize the chosen state once; never reselect from live latest.json.
    let selected_state: Option<Value> = selected_checkpoint
        .as_ref()
        .map(|cp| read_json(&cp.join("state.json")))
        .transpose()?;
    if let (Some(cp), Some(state)) = (&selected_checkpoint, &selected_state) {
        if resume {
            // W6-T05: exact resume loads only a complete, internally
            // consistent checkpoint of this corpus and topology.
            checkpoint_commit::verify_dir(
                root,
                cp,
                state["step"].as_u64().context("checkpoint step")?,
                &checkpoint_commit::Expectation {
                    corpus_sha256: Some(&corpus_hash),
                    vocabulary_sha256: Some(&index.vocabulary_sha256),
                    world: Some(world),
                    global_batch: Some(global_batch),
                    require_complete: false,
                },
            )?;
            ensure!(
                state
                    .get("training_inputs_sha256")
                    .is_none_or(|frozen| *frozen == frozen_inputs_sha.as_str()),
                "exact resume requires the same frozen training inputs"
            );
        }
        // Full objective identity, partition, vocabulary and architecture
        // against the requested mode; undeclared changes are refused.
        continuity = continuation::check_parent_partition(
            mode,
            state,
            &read_json::<Value>(&cp.join("config.json"))?,
            &continuation::Current {
                corpus_sha256: &corpus_hash,
                vocabulary_sha256: &index.vocabulary_sha256,
                world,
                microbatch: micro,
                global_batch,
                seed,
                sampler_id: active_sampler_id(replay.as_ref()),
                config: &serde_json::to_value(&config)?,
                objective_sha256: &objective_sha,
            },
            execution.record(),
            policy.as_ref(),
        )?;
        ensure!(
            state["autocast_cache_lifetime"] == "scoped_v2"
                && state["vocabulary_sha256"] == index.vocabulary_sha256,
            "checkpoint autocast or vocabulary mismatch before CUDA"
        );
        let previous: Config = read_json(&cp.join("config.json"))?;
        ensure!(
            serde_json::to_value(previous)? == serde_json::to_value(&config)?,
            "checkpoint architecture mismatch before CUDA"
        );
        for (name, field) in [
            ("model.ot", "model_sha256"),
            ("config.json", "config_sha256"),
        ] {
            ensure!(
                file_hash(&cp.join(name))? == state[field],
                "checkpoint {name} changed before CUDA"
            );
        }
        if resume || policy.as_ref().is_some_and(|p| p["optimizer"] == "carry") {
            for (name, field) in [
                ("optimizer.ot", "optimizer_sha256"),
                ("optimizer.json", "optimizer_metadata_sha256"),
            ] {
                ensure!(
                    file_hash(&cp.join(name))? == state[field],
                    "checkpoint {name} changed before CUDA"
                );
            }
            let optimizer: Value = read_json(&cp.join("optimizer.json"))?;
            let expected = if resume {
                json!({"execution_partition":execution.record(),
                    "sampler_id":active_sampler_id(replay.as_ref()),"corpus_sha256":corpus_hash,
                    "world":world,"global_batch":global_batch,"seed":seed,
                    "ranking_backward":crate::neural::RANKING_BACKWARD_ID,
                    "runtime_objective":"huber_log1p_seconds_v2"})
            } else {
                json!({"ranking_backward":crate::neural::RANKING_BACKWARD_ID,"runtime_objective":"huber_log1p_seconds_v2"})
            };
            crate::task_contract::verify_resume_metadata(state, &expected, &optimizer)?;
        }
        if resume {
            let lineage: Value = read_json(&run.join("initialization.json"))?;
            ensure!(
                state["initialization"] == lineage,
                "resume initialization lineage changed before CUDA"
            );
            if policy.is_some() {
                ensure!(
                    lineage["replay_policy_sha256"] == json!(policy_sha)
                        && lineage["replay_policy"] == json!(policy),
                    "resume replay policy changed before CUDA"
                );
            } else {
                ensure!(
                    lineage.get("replay_policy_sha256").is_none(),
                    "resume original replay policy absent before CUDA"
                );
            }
            let target = steps
                .checked_add(lineage["optimizer_parent_step"].as_u64().unwrap_or(0) as usize)
                .context("training step overflow")?;
            ensure!(
                state["step"].as_u64().context("checkpoint step")? <= target as u64,
                "checkpoint exceeds requested training window before CUDA"
            );
        } else if let Some(p) = &policy {
            ensure!(
                p["parent_checkpoint_sha256"] == file_hash(&cp.join("state.json"))?,
                "replay initialization changed before CUDA"
            );
        }
    }
    let exposure_baseline = if resume {
        crate::training_exposure::checkpoint_baseline_rows(
            selected_state
                .as_ref()
                .context("resume exposure state absent")?,
            rank,
            &execution.rows_per_rank(),
        )?
    } else {
        Some(crate::training_exposure::Counts::default())
    };
    let (family_baseline, family_baseline_known) = if resume {
        crate::training_exposure::checkpoint_family_baseline(
            selected_state
                .as_ref()
                .context("resume exposure state absent")?,
            rank,
            exposure_baseline.as_ref(),
        )?
    } else {
        (crate::training_exposure::FamilyGroups::new(), true)
    };
    // The communicator rank is not the CUDA ordinal: bind this worker to its
    // owned device and check visibility before the communicator exists.
    let binding = topology::resolve(assignment, rank, world, crate::neural::local_devices)?;
    let comm = Nccl::init(&binding, &launch.join("nccl.id"))?;
    let device = Device::Cuda(binding.device_index());
    tch::manual_seed(seed as i64);
    let mut vs = nn::VarStore::new(device);
    let model = Policy::new(vs.root(), config.clone())?;
    let mut optimizer = AdamW::new(1e-4, 0.01);
    ensure!(
        !(resume && initialize.is_some()),
        "resume and initialize are mutually exclusive"
    );
    let initialization_path = run.join("initialization.json");
    let mut initialization = json!({"kind":"fresh","seed":seed,"initialization":"pytorch_linear_mha_embedding_v2","autocast_cache_lifetime":"scoped_v2"});
    if initialize.is_some() {
        let checkpoint = selected_checkpoint
            .as_ref()
            .context("initialization selection absent")?;
        let state = selected_state
            .as_ref()
            .context("initialization state absent")?;
        ensure!(
            state["autocast_cache_lifetime"] == "scoped_v2",
            "initialization checkpoint predates corrected autocast cache lifetime"
        );
        ensure!(
            state["vocabulary_sha256"] == index.vocabulary_sha256,
            "warm-start vocabulary mismatch; no token remapping is implicit"
        );
        let previous: Config = read_json(&checkpoint.join("config.json"))?;
        ensure!(
            serde_json::to_value(previous)? == serde_json::to_value(&config)?,
            "warm-start architecture mismatch"
        );
        crate::neural::load_bound_model(
            &mut vs,
            &checkpoint.join("model.ot"),
            &state["model_sha256"],
            "initialization model changed",
        )?;
        initialization = json!({"kind":"warm_start","checkpoint":checkpoint.strip_prefix(root)?,
            "state_sha256":file_hash(&checkpoint.join("state.json"))?,"model_sha256":state["model_sha256"],
            "parent_step":state["step"],"parent_corpus_sha256":state["corpus_sha256"],
            "optimizer":crate::native_loop::warm_start::FRESH_OPTIMIZER, "sampler":"new corpus starts at step zero"});
        if let Some(p) = &policy {
            ensure!(
                p["parent_checkpoint_sha256"] == file_hash(&checkpoint.join("state.json"))?,
                "replay initialization checkpoint changed"
            );
            initialization["replay_policy_sha256"] = json!(policy_sha);
            initialization["replay_policy"] = p.clone();
            if p["optimizer"] == "carry" {
                ensure!(
                    state["ranking_backward"] == crate::neural::RANKING_BACKWARD_ID
                        && state["runtime_objective"] == "huber_log1p_seconds_v2",
                    "optimizer warm-start requires compatible objective identity"
                );
                optimizer = AdamW::load_bound(
                    &checkpoint.join("optimizer.ot"),
                    &state["optimizer_sha256"],
                    &state["optimizer_metadata_sha256"],
                    device,
                    "warm-start optimizer missing or changed",
                )?;
                ensure!(
                    optimizer.step == state["step"].as_u64().context("parent optimizer step")?,
                    "warm-start optimizer/model step mismatch"
                );
                optimizer.validate_parameters(&vs)?;
                initialization["optimizer"] = json!("carried FP32 AdamW with bias-correction step");
                initialization["optimizer_parent_step"] = json!(optimizer.step);
            }
            initialization["sampler"] =
                json!("lineage-bound counter; local block step starts at zero");
            initialization["resolved_sampler_id"] = json!(active_sampler_id(replay.as_ref()));
            if let Some(recovered) = &recovered_continuation {
                initialization["continuation"] = recovered.clone();
                initialization["exact_same_experiment_resume"] = json!(false);
            }
        }
    }
    if !resume {
        // The lineage record names its mode; a new lineage never inherits an
        // exact-resume claim from its parent.
        initialization["continuation_mode"] = json!(mode.as_str());
        initialization["continuation_mode_declaration"] = declaration["declaration"].clone();
        initialization["objective_sha256"] = json!(objective_sha);
        initialization["exact_same_experiment_resume"] = json!(false);
        if !continuity.is_null() {
            initialization["continuity"] = continuity.clone();
        }
    }
    if resume {
        let cp = selected_checkpoint
            .as_ref()
            .context("resume selection absent")?;
        let state = selected_state.as_ref().context("resume state absent")?;
        ensure!(state["execution_partition"] == *execution.record(),
            "exact resume requires the original microbatch, accumulation and RNG/objective partition");
        ensure!(state["sampler_id"] == active_sampler_id(replay.as_ref()),
                "exact resume requires the same sampler; use an explicit warm-start across sampler versions");
        ensure!(state["ranking_backward"]==crate::neural::RANKING_BACKWARD_ID&&state["runtime_objective"]=="huber_log1p_seconds_v2","exact resume requires the same ranking/runtime objectives; use an explicit model warm-start across versions");
        ensure!(
            state["autocast_cache_lifetime"] == "scoped_v2",
            "cannot resume a checkpoint trained with stale autocast parameter casts"
        );
        ensure!(
            state["corpus_sha256"] == corpus_hash
                && state["vocabulary_sha256"] == index.vocabulary_sha256
                && state["world"] == world
                && state["global_batch"] == global_batch
                && state["seed"] == seed,
            "resume identity mismatch"
        );
        ensure!(
            file_hash(&cp.join("model.ot"))? == state["model_sha256"],
            "checkpoint model changed"
        );
        ensure!(
            file_hash(&cp.join("optimizer.ot"))? == state["optimizer_sha256"]
                && file_hash(&cp.join("optimizer.json"))? == state["optimizer_metadata_sha256"]
                && file_hash(&cp.join("config.json"))? == state["config_sha256"],
            "checkpoint optimizer or configuration changed"
        );
        // W2-T04 (G-10c): the configuration, model and optimizer consumed are
        // exactly the bytes this checkpoint's state binds, each read once.
        let previous: Config = serde_json::from_slice(&crate::manifest::read_bound(
            &cp.join("config.json"),
            &state["config_sha256"],
            "checkpoint optimizer or configuration changed",
        )?)?;
        ensure!(
            serde_json::to_value(previous)? == serde_json::to_value(&config)?,
            "resume architecture mismatch"
        );
        crate::neural::load_bound_model(
            &mut vs,
            &cp.join("model.ot"),
            &state["model_sha256"],
            "checkpoint model changed",
        )?;
        optimizer = AdamW::load_bound(
            &cp.join("optimizer.ot"),
            &state["optimizer_sha256"],
            &state["optimizer_metadata_sha256"],
            device,
            "checkpoint optimizer or configuration changed",
        )?;
        ensure!(
            optimizer.step == state["step"].as_u64().context("checkpoint step")?,
            "optimizer/model step mismatch"
        );
        initialization = read_json(&initialization_path)?;
        ensure!(
            state["initialization"] == initialization,
            "initialization lineage mismatch"
        );
        if policy.is_some() {
            ensure!(
                initialization["replay_policy_sha256"] == json!(policy_sha)
                    && initialization["replay_policy"] == json!(policy),
                "exact resume replay policy changed"
            );
        } else {
            ensure!(
                initialization.get("replay_policy_sha256").is_none(),
                "exact resume requires original replay policy"
            );
        }
    }
    ensure!(
        policy.is_none() || initialize.is_some() || resume,
        "replay experiment requires explicit initialization or exact resume"
    );
    let sampling_offset = initialization["optimizer_parent_step"]
        .as_u64()
        .unwrap_or(0) as usize;
    let steps = steps
        .checked_add(sampling_offset)
        .context("training step overflow")?;
    ensure!(
        optimizer.step as usize <= steps,
        "checkpoint exceeds the requested training window"
    );
    if rank == 0 && !resume {
        checkpoint_commit::replace_json(&initialization_path, &initialization)?;
    }
    if rank == 0 {
        checkpoint_commit::replace_json(&launch.join("training-inputs.json"), &frozen_inputs)?;
    }
    ensure!(
        index
            .contexts
            .iter()
            .all(|c| crate::corpus_projection::admitted_partition(&c.partition)),
        "non-admitted context partition in training closure"
    );
    // W6-T01: cached device staging is keyed by the content it was built
    // from; a cold load verifies every byte through the same functions.
    // W2-T04 (G-11a/b): a staged tensor lives on one device; the device is part
    // of every resident key.
    let context_key = format!("{device:?}:{}", inputs::context_identity(index)?);
    let context_tensor = RESIDENT_CONTEXT.with(|cache| -> Result<Tensor> {
        if let Some((identity, tensor)) = cache.borrow().as_ref() {
            if *identity == context_key {
                return Ok(tensor.shallow_clone());
            }
        }
        // Release an obsolete epoch before allocating its replacement.
        cache.borrow_mut().take();
        let contexts = inputs::stage_contexts(root, index)?;
        // Small immutable artifacts are staged once on each GPU; no CSV loading in training steps.
        let tensor = Tensor::from_slice(&contexts)
            .view([-1, 2048])
            .to_device(device);
        // Device tensors own their copied storage; release host staging before training.
        drop(contexts);
        *cache.borrow_mut() = Some((context_key.clone(), tensor.shallow_clone()));
        Ok(tensor)
    })?;
    let tensor_key = format!("{device:?}:{corpus_hash}:{frozen_inputs_sha}");
    let (resident_cache_hit, tensors) =
        RESIDENT_TENSORS.with(|cache| -> Result<(bool, Vec<Tensor>)> {
            if let Some((identity, tensors)) = cache.borrow().as_ref() {
                if *identity == tensor_key {
                    return Ok((true, tensors.iter().map(Tensor::shallow_clone).collect()));
                }
            }
            cache.borrow_mut().take();
            let staged = inputs::stage_records(index, vocabulary.tokens.len(), |i| records.get(i))?;
            let token_tensor = Tensor::from_slice(&staged.tokens)
                .view([-1, SEQUENCE as i64])
                .to_device(device);
            let legal_id_tensor = Tensor::from_slice(&staged.legal)
                .view([-1, SEQUENCE as i64])
                .to_device(device);
            let context_id_tensor = Tensor::from_slice(&staged.context_ids).to_device(device);
            let mask_table = Tensor::from_slice(&staged.masks)
                .view([-1, vocabulary.tokens.len() as i64])
                .to_kind(Kind::Bool)
                .to_device(device);
            drop(staged);
            let tensors = vec![token_tensor, legal_id_tensor, context_id_tensor, mask_table];
            *cache.borrow_mut() = Some((
                tensor_key.clone(),
                tensors.iter().map(Tensor::shallow_clone).collect(),
            ));
            Ok((false, tensors))
        })?;
    let [token_tensor, legal_id_tensor, context_id_tensor, mask_table]: [Tensor; 4] =
        tensors.try_into().ok().unwrap();
    // Validate every sampled comparison against the frozen index before its
    // records are read or any loss is formed (W6-T03).
    let comparisons = objective::SampledComparisons::new(index)?;
    // `global` selects the objective: `None` is v1's microbatch-local
    // composite; partition v2 passes the step's global denominators.
    let loss_for = |ids: &[usize],
                    train: bool,
                    global: Option<&partition::Denominators>|
     -> Result<(Tensor, [f64; 5])> {
        let corpus_partition = if train {
            "meta_train"
        } else {
            "meta_validation"
        };
        comparisons.check_ids(index, ids, corpus_partition)?;
        let rows = ids.iter().map(|&i| records.get(i)).collect::<Vec<_>>();
        let labels = ids
            .iter()
            .zip(&rows)
            .map(|(&id, r)| objective::RowLabels::from_record(id, r))
            .collect::<Vec<_>>();
        comparisons.check_rows(index, &labels)?;
        let length = rows.iter().map(|r| r.length).max().unwrap().div_ceil(8) * 8;
        let length = length.min(SEQUENCE) as i64;
        let id = Tensor::from_slice(&ids.iter().map(|&i| i as i64).collect::<Vec<_>>())
            .to_device(device);
        let tokens = token_tensor
            .index_select(0, &id)
            .narrow(1, 0, length)
            .to_kind(Kind::Int64)
            .bitwise_and(0xffffi64);
        let pad = tokens.ne(0);
        let context = context_tensor.index_select(0, &context_id_tensor.index_select(0, &id));
        let (logits, rewards, runtime, stability, ablation) = model.forward(
            &context.narrow(1, 0, 1536),
            &context.narrow(1, 1536, 512),
            &tokens,
            &pad,
            train,
        )?;
        let legal = mask_table
            .index_select(
                0,
                &legal_id_tensor
                    .index_select(0, &id)
                    .narrow(1, 1, length - 1)
                    .reshape([-1])
                    .to_kind(Kind::Int64)
                    .bitwise_and(0xffffi64),
            )
            .view([ids.len() as i64, length - 1, vocabulary.tokens.len() as i64]);
        let outputs = objective::Outputs {
            logits: &logits,
            rewards: &rewards,
            runtime: &runtime,
            stability: &stability,
            ablation: &ablation,
        };
        match global {
            None => objective::composite(
                &outputs,
                model.loss_log_vars.values(),
                &tokens,
                &pad,
                &legal,
                &labels,
            ),
            Some(global) => partition::composite(
                &outputs,
                model.loss_log_vars.values(),
                &tokens,
                &pad,
                &legal,
                &labels,
                global,
            ),
        }
    };
    let accumulation = execution.accumulation(rank)?;
    let started = Instant::now();
    let initial_step = optimizer.step as usize;
    let mut last_live_write = Instant::now() - Duration::from_secs(1);
    let mut measured_step_seconds = 0.0;
    let mut telemetry_seconds = 0.0;
    let mut coordination_seconds = 0.0;
    let mut checkpoint_seconds = 0.0;
    let mut metrics = if rank == 0 {
        Some(BufWriter::new(
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(run.join("metrics.jsonl"))?,
        ))
    } else {
        None
    };
    let mut last_checkpoint = Instant::now();
    let mut validation_metrics = Value::Null;
    let mut latest: Value = if run.join("latest.json").is_file() {
        read_json(&run.join("latest.json"))?
    } else {
        Value::Null
    };
    let mut observed_exposure = crate::training_exposure::Counts::default();
    let mut committed_exposure = crate::training_exposure::Counts::default();
    let mut observed_families = crate::training_exposure::FamilyGroups::new();
    write_json(
        &launch.join(format!("rank-{rank}.json")),
        &json!({"schema":"parallel.neural_training.v1",
        "rank":rank,"world":world,"status":"active","step":initial_step,"steps":steps,
        "initial_optimizer_step":initial_step,"corpus_sha256":corpus_hash,"timestamp":now(),
        "exposure":crate::training_exposure::attempt_snapshot(&observed_exposure,&committed_exposure,false)?}),
    )?;
    // W6-T04: every consumed microbatch, attempted update and durable commit
    // of this rank and attempt, bound into each checkpoint receipt.
    let mut journal = consumption::Journal::create_with_shard(
        &launch.join(format!("consumption-rank-{rank}.jsonl")),
        consumption::Header {
            schema: consumption::SCHEMA.into(),
            rank,
            world,
            microbatch: micro,
            global_batch,
            accumulation,
            seed,
            sampler_id: active_sampler_id(replay.as_ref()).into(),
            rng: consumption::RNG_ID.into(),
            sampling_offset: sampling_offset as u64,
            initial_optimizer_step: initial_step as u64,
            target_optimizer_step: steps as u64,
            corpus_sha256: corpus_hash.clone(),
            replay_policy_sha256: policy_sha.clone(),
            resume_state_sha256: if resume {
                Some(file_hash(
                    &selected_checkpoint
                        .as_ref()
                        .context("resume selection absent")?
                        .join("state.json"),
                )?)
            } else {
                None
            },
        },
        execution.shard(rank)?,
    )?;
    let journal_path = journal.path().strip_prefix(root)?.to_path_buf();
    // The optimizer step whose update is in flight; cleared once applied.
    let mut attempting: Option<usize> = None;
    let outcome = (|| -> Result<()> {
        for step in optimizer.step as usize..steps {
            attempting = Some(step);
            let begun = Instant::now();
            let mut values = [0.0; 5];
            let mut loss_value = 0.0;
            // W7-T06: partition v2 draws the whole global batch once, forms
            // the global denominators from labels alone, and runs this rank's
            // contiguous shard of whole comparisons. Each microbatch's
            // objective is scaled by the world size, so the mean all-reduce
            // delivers the sum of rank gradients of the global objective.
            let mut step_denominators = None;
            if let Some(plan) = execution.plan() {
                let local_step = (step - sampling_offset) as u64;
                let scale = plan.local_scale(partition::Transport::Mean);
                let global_ids = sample_replay_fresh(
                    &index,
                    &families,
                    &fresh,
                    &fresh_by_teacher,
                    usize::try_from(plan.global_serial(local_step)?)?,
                    global_batch,
                    seed,
                    balanced,
                );
                comparisons.check_ids(index, &global_ids, "meta_train")?;
                let global_labels = global_ids
                    .iter()
                    .map(|&id| objective::RowLabels::from_record(id, &records.get(id)))
                    .collect::<Vec<_>>();
                comparisons.check_rows(index, &global_labels)?;
                let global = partition::denominators(&global_labels)?;
                let slots = plan.slots(rank, seed, local_step)?;
                if slots.is_empty() {
                    // An empty rank contributes exact zero gradients with the
                    // same packed layout and still joins every collective.
                    partition::empty_rank_objective(vs.trainable_variables().iter())?.backward();
                }
                for slot in &slots {
                    tch::manual_seed(slot.dropout_seed);
                    let first = slot.first_comparison * objective::COMPARISON_ROWS;
                    let ids = global_ids[first..first + slot.rows].to_vec();
                    let serial = usize::try_from(slot.serial)?;
                    let guard = Bf16::enter();
                    let (loss, contributions) = loss_for(&ids, true, Some(&global))?;
                    let loss = loss * scale;
                    loss_value += loss.double_value(&[]);
                    for i in 0..5 {
                        values[i] += contributions[i] * scale;
                    }
                    drop(guard);
                    loss.backward();
                    observed_exposure.observe(
                        &ids,
                        &fresh_records,
                        serial,
                        balanced && !fresh.is_empty(),
                    )?;
                    crate::training_exposure::observe_families(
                        &mut observed_families,
                        &teacher_family,
                        &ids,
                    )?;
                    let consumed = ids.iter().map(|&i| records.get(i)).collect::<Vec<_>>();
                    journal.microbatch(
                        step as u64,
                        slot.microbatch,
                        slot.serial,
                        slot.dropout_seed,
                        &ids,
                        objective::supervision(
                            &ids.iter()
                                .zip(&consumed)
                                .map(|(&id, r)| objective::RowLabels::from_record(id, r))
                                .collect::<Vec<_>>(),
                            &consumed.iter().map(|r| r.length).collect::<Vec<_>>(),
                        )?,
                    )?;
                }
                step_denominators = Some(global);
            }
            // Partition v1: equal rank shards of `accumulation` microbatches.
            let v1_microbatches = if execution.plan().is_none() {
                accumulation
            } else {
                0
            };
            for mb in 0..v1_microbatches {
                let dropout_seed =
                    consumption::dropout_seed(seed, (step - sampling_offset) as u64, rank, mb);
                tch::manual_seed(dropout_seed);
                let serial = (step - sampling_offset) * global_batch
                    + rank * (global_batch / world)
                    + mb * micro;
                let ids = sample_replay_fresh(
                    &index,
                    &families,
                    &fresh,
                    &fresh_by_teacher,
                    serial,
                    micro,
                    seed,
                    balanced,
                );
                let guard = Bf16::enter();
                let (loss, m) = loss_for(&ids, true, None)?;
                let loss = loss / accumulation as f64;
                loss_value += loss.double_value(&[]);
                for i in 0..5 {
                    values[i] += m[i] / accumulation as f64;
                }
                drop(guard);
                loss.backward();
                observed_exposure.observe(
                    &ids,
                    &fresh_records,
                    serial,
                    balanced && !fresh.is_empty(),
                )?;
                crate::training_exposure::observe_families(
                    &mut observed_families,
                    &teacher_family,
                    &ids,
                )?;
                let consumed = ids.iter().map(|&i| records.get(i)).collect::<Vec<_>>();
                journal.microbatch(
                    step as u64,
                    mb,
                    serial as u64,
                    dropout_seed,
                    &ids,
                    objective::supervision(
                        &ids.iter()
                            .zip(&consumed)
                            .map(|(&id, r)| objective::RowLabels::from_record(id, r))
                            .collect::<Vec<_>>(),
                        &consumed.iter().map(|r| r.length).collect::<Vec<_>>(),
                    )?,
                )?;
            }
            let mut variables = vs.variables().into_iter().collect::<BTreeMap<_, _>>();
            let gradients = variables.values().map(|p| p.grad()).collect::<Vec<_>>();
            ensure!(
                gradients.iter().all(|g| g.defined()),
                "undefined rank gradient"
            );
            let mut packed = Tensor::cat(
                &gradients
                    .iter()
                    .map(|g| g.flatten(0, -1))
                    .collect::<Vec<_>>(),
                0,
            );
            ensure!(
                packed.isfinite().all().int64_value(&[]) != 0,
                "nonfinite training gradients"
            );
            let communication = Instant::now();
            comm.reduce(&mut packed)?;
            let communication_seconds = communication.elapsed().as_secs_f64();
            let norm = packed.norm().double_value(&[]);
            if norm > 1.0 {
                packed /= norm;
            }
            let mut offset = 0;
            for p in variables.values_mut() {
                let size = p.numel() as i64;
                p.grad()
                    .copy_(&packed.narrow(0, offset, size).reshape(p.size()));
                offset += size;
            }
            optimizer.update(&vs)?;
            // Parameters may now differ from the step's start: a later failure
            // is not a rejected update. The journal then ends with this step's
            // microbatches and no outcome, and recovery re-executes it.
            attempting = None;
            tch::Cuda::synchronize(binding.device_index() as i64);
            ensure!(
                optimizer.step as usize == step + 1,
                "optimizer step did not advance by exactly one update"
            );
            journal.applied(step as u64, norm)?;
            let mut totals = Tensor::from_slice(
                &[
                    loss_value, values[0], values[1], values[2], values[3], values[4],
                ]
                .map(|x| x as f32),
            )
            .to_device(device);
            comm.reduce(&mut totals)?;
            let totals: Vec<f32> = Vec::try_from(totals.to_device(Device::Cpu))?;
            // Every rank votes on every step; only rank zero contributes the wall-clock trigger.
            let native_checkpoint =
                native_stop.is_some_and(|stop| stop.load(std::sync::atomic::Ordering::Acquire));
            let due = native_checkpoint
                || (step + 1) % checkpoint_steps == 0
                || (step + 1) % epoch_steps == 0
                || step + 1 == steps
                || (rank == 0 && last_checkpoint.elapsed().as_secs() >= 540);
            let vote_started = Instant::now();
            let mut checkpoint_vote =
                Tensor::from_slice(&[if due { 1f32 } else { 0f32 }]).to_device(device);
            comm.reduce(&mut checkpoint_vote)?;
            let checkpoint_due = checkpoint_vote.double_value(&[0]) > 0.0;
            coordination_seconds += vote_started.elapsed().as_secs_f64();
            if checkpoint_due && !validation.is_empty() {
                let (v, sampled_examples) =
                    match fixed_panel.as_ref().context("validation panel missing")? {
                        ValidationPanel::V1(panel) => {
                            let (_, v) = tch::no_grad(|| {
                                let _guard = Bf16::enter();
                                loss_for(&panel[rank], false, None)
                            })?;
                            (v, world * 256)
                        }
                        // Partition v2: this rank's contributions to the
                        // globally normalized panel objective, scaled for the
                        // mean all-reduce; an empty rank contributes zeros.
                        ValidationPanel::V2 {
                            per_rank,
                            denominators,
                            ..
                        } => {
                            let mut v = [0.0; 5];
                            for ids in &per_rank[rank] {
                                let (_, contributions) = tch::no_grad(|| {
                                    let _guard = Bf16::enter();
                                    loss_for(ids, false, Some(denominators))
                                })?;
                                for i in 0..5 {
                                    v[i] += contributions[i] * world as f64;
                                }
                            }
                            (v, partition::VALIDATION_PANEL_ROWS)
                        }
                    };
                let mut t = Tensor::from_slice(&v.map(|x| x as f32)).to_device(device);
                comm.reduce(&mut t)?;
                let v: Vec<f32> = Vec::try_from(t.to_device(Device::Cpu))?;
                validation_metrics = json!({"sequence_nll":v[0],"ranking_bt":v[1],"ranking_listwise":v[2],"runtime_huber":v[3],"contextfree_bt":v[4],"weighted_losses":objective::weighted_losses(&v),"source_families":validation.len(),"evaluated_step":step+1,"panel_sha256":panel_sha256,"sampler_id":SAMPLER_ID,"sampled_examples":sampled_examples,"family_balanced":true,"evaluation_bias":"adaptive_cut_heldout_teacher","independent_evaluation":false});
            }
            let seconds = begun.elapsed().as_secs_f64();
            measured_step_seconds += seconds;
            let mut state = json!({"schema":"parallel.neural_training.v1","phase":"training on real GP teaching corpus","status":"active","run":run.strip_prefix(root)?,"latest_receipt":run.join("latest.json").strip_prefix(root)?,"checkpoint":latest["checkpoint"],"checkpoint_step":latest["step"],"rank":rank,"world":world,"step":step+1,"steps":steps,"epoch":(step+1) as f64/epoch_steps as f64,"microbatch":micro,"accumulation":accumulation,"global_batch":global_batch,"loss":totals[0],"sequence_nll":totals[1],"ranking_bt":totals[2],"ranking_listwise":totals[3],"runtime_huber":totals[4],"contextfree_bt":totals[5],"weighted_losses":objective::weighted_losses(&totals[1..]),"inactive_objectives":["stability","loss_log_vars"],"gradient_clipped":norm>1.0,"validation":validation_metrics,"gradient_norm":norm,"step_seconds":seconds,"communication_seconds":communication_seconds,"examples_per_second":global_batch as f64/seconds,"elapsed_seconds":started.elapsed().as_secs_f64(),"timestamp":now(),"corpus_sha256":corpus_hash,"training_families":families.len(),"sampler_id":SAMPLER_ID,"teacher_in_each_comparison":true,"qualification":true,"fresh_initialization":initialization["kind"] == "fresh","initialization":initialization,"autocast_cache_lifetime":"scoped_v2"});
            if policy.is_some() {
                state["sampler_id"] = json!(active_sampler_id(replay.as_ref()));
                state["replay_policy_sha256"] = json!(policy_sha);
                state["local_step"] = json!(step + 1 - sampling_offset);
                state["replay_comparisons_this_rank_attempt"] = json!({"total":observed_exposure.comparison_groups,
                "fresh_comparisons":observed_exposure.groups_with_actual_fresh_program,
                "fresh_programs":observed_exposure.actual_fresh_program_presentations,
                "historical_comparisons":observed_exposure.comparison_groups-observed_exposure.groups_with_actual_fresh_program,
                "fresh_portion_fraction":if balanced && !fresh.is_empty() {0.5} else {0.0},
                "fallback_to_broad":balanced && fresh.is_empty()});
            }
            if batch_qualification {
                state["batch_qualification"] = json!(true);
                state["eligible_checkpoint"] = json!(false);
            }
            if checkpoint_due {
                let checkpoint_started = Instant::now();
                crate::training_exposure::validate_families(
                    &observed_families,
                    &observed_exposure,
                )?;
                // Consumption through this update is durable before any
                // checkpoint can reference it.
                let consumed = journal.sync()?;
                // Partition v2 persists every rank's RNG/sampler state and
                // the global committed update this checkpoint covers.
                let partition_state = match (execution.plan(), &step_denominators) {
                    (Some(plan), Some(global)) => Some(plan.state(
                        seed,
                        sampling_offset as u64,
                        partition::CommittedUpdate::new(
                            plan,
                            optimizer.step,
                            (step - sampling_offset) as u64,
                            global,
                            [totals[1], totals[2], totals[3], totals[4], totals[5]].map(f64::from),
                            &packed,
                            norm,
                            &parameter_hash(&vs)?,
                        )?,
                    )?),
                    _ => None,
                };
                checkpoint(
                    root,
                    run,
                    launch,
                    rank,
                    world,
                    &comm,
                    &vs,
                    &optimizer,
                    &config,
                    &corpus_hash,
                    &index.vocabulary_sha256,
                    seed,
                    epoch_steps,
                    &initialization,
                    &validation_metrics,
                    global_batch,
                    &execution,
                    partition_state.as_ref(),
                    batch_qualification,
                    &exposure_baseline
                        .as_ref()
                        .map(|base| base.add(&observed_exposure))
                        .transpose()?,
                    &crate::training_exposure::combine_families(
                        &family_baseline,
                        &observed_families,
                    )?,
                    family_baseline_known,
                    &json!({"path":journal_path,"binding":consumed}),
                    &frozen_inputs_sha,
                    &binding,
                )?;
                committed_exposure = observed_exposure.clone();
                checkpoint_seconds += checkpoint_started.elapsed().as_secs_f64();
                last_checkpoint = Instant::now();
                latest = read_json(&run.join("latest.json"))?;
                ensure!(
                    latest["step"].as_u64() == Some(optimizer.step),
                    "published checkpoint does not cover the applied updates"
                );
                journal.commit(
                    optimizer.step,
                    latest["state_sha256"]
                        .as_str()
                        .context("published checkpoint state digest absent")?,
                )?;
                state["checkpoint"] = latest["checkpoint"].clone();
                state["checkpoint_step"] = latest["step"].clone();
            }
            state["exposure"] = crate::training_exposure::attempt_snapshot(
                &observed_exposure,
                &committed_exposure,
                false,
            )?;
            state["initial_optimizer_step"] = json!(initial_step);
            if step + 1 == steps {
                state["status"] = json!("complete");
                state["phase"] = json!("real GP teaching corpus training complete");
            }
            let elapsed = started.elapsed().as_secs_f64();
            state["end_to_end_examples_per_second"] =
                json!((step + 1 - initial_step) as f64 * global_batch as f64 / elapsed);
            state["elapsed_seconds"] = json!(elapsed);
            state["telemetry_seconds"] = json!(telemetry_seconds);
            state["coordination_seconds"] = json!(coordination_seconds);
            state["checkpoint_seconds"] = json!(checkpoint_seconds);
            state["measured_step_seconds"] = json!(measured_step_seconds);
            let publish = checkpoint_due
                || step + 1 == steps
                || last_live_write.elapsed() >= Duration::from_secs(1);
            let telemetry_started = Instant::now();
            if let Some(writer) = &mut metrics {
                serde_json::to_writer(&mut *writer, &state)?;
                writer.write_all(b"\n")?;
                if publish {
                    writer.flush()?;
                }
            }
            if publish {
                write_json(&launch.join(format!("rank-{rank}.json")), &state)?;
                if rank == 0 {
                    write_json(&root.join("runs/neural-service.json"), &state)?;
                }
                last_live_write = Instant::now();
            }
            telemetry_seconds += telemetry_started.elapsed().as_secs_f64();
            if native_checkpoint && step + 1 < steps {
                ensure!(checkpoint_due, "native recovery lacks committed checkpoint");
                anyhow::bail!(
                    "native control lost: checkpoint committed at original optimizer step {}",
                    step + 1
                );
            }
        }
        let mut complete = json!({"rank":rank,"world":world,"steps":optimizer.step,"parameter_sha256":parameter_hash(&vs)?,"seconds":started.elapsed().as_secs_f64(),"executed_steps":optimizer.step as usize-initial_step,"end_to_end_examples_per_second":(optimizer.step as usize-initial_step) as f64*global_batch as f64/started.elapsed().as_secs_f64(),"telemetry_seconds":telemetry_seconds,"coordination_seconds":coordination_seconds,"checkpoint_seconds":checkpoint_seconds,"measured_step_seconds":measured_step_seconds,"corpus_sha256":corpus_hash});
        complete["resident_tensor_cache_hit"] = json!(resident_cache_hit);
        complete["initial_optimizer_step"] = json!(initial_step);
        complete["local_steps"] = json!(optimizer.step as usize - sampling_offset);
        complete["continuation"] = initialization
            .get("continuation")
            .cloned()
            .unwrap_or(Value::Null);
        complete["exact_same_experiment_resume"] =
            json!(resume && recovered_continuation.is_none());
        complete["training_inputs_sha256"] = json!(frozen_inputs_sha);
        complete["topology"] = binding.receipt();
        complete["continuation_mode"] = json!(mode.as_str());
        complete["continuation_declaration"] = declaration.clone();
        complete["continuity"] = continuity.clone();
        complete["objective_sha256"] = json!(objective_sha);
        complete["resident_identifier_bytes"] = json!(index.records * SEQUENCE * 4);
        complete["exposure"] = crate::training_exposure::attempt_snapshot(
            &observed_exposure,
            &committed_exposure,
            true,
        )?;
        complete["committed_exposure_this_task_rank"] = json!(exposure_baseline
            .as_ref()
            .map(|base| base.add(&observed_exposure))
            .transpose()?);
        crate::training_exposure::validate_families(&observed_families, &observed_exposure)?;
        let committed_families =
            crate::training_exposure::combine_families(&family_baseline, &observed_families)?;
        complete["committed_family_groups_this_task_rank"] = json!(committed_families);
        if !family_baseline_known {
            complete["family_groups_baseline"] = json!("unknown");
        }
        if policy.is_some() {
            complete["replay_policy_sha256"] = json!(policy_sha);
            complete["actual_fresh_comparisons_this_rank_attempt"] =
                json!(observed_exposure.groups_with_actual_fresh_program);
            complete["actual_fresh_programs_this_rank_attempt"] =
                json!(observed_exposure.actual_fresh_program_presentations);
            complete["total_comparisons_this_rank_attempt"] =
                json!(observed_exposure.comparison_groups);
        }
        if batch_qualification {
            complete["microbatch"] = json!(micro);
            complete["global_batch"] = json!(global_batch);
            complete["batch_qualification"] = json!(true);
            complete["eligible_checkpoint"] = json!(false);
        }
        complete["consumption_journal"] = json!({"path":journal_path,"binding":journal.sync()?});
        if execution.plan().is_some() {
            // Added only for partition v2; v1 completion receipts are unchanged.
            complete["execution_partition"] = execution.record().clone();
        }
        write_json(&launch.join(format!("complete-{rank}.json")), &complete)?;
        Ok(())
    })();
    if let Err(error) = &outcome {
        // An update that was attempted but not applied is recorded as
        // rejected; the journal refuses anything after it. Best effort: the
        // original error is what this rank returns.
        if let Some(step) = attempting {
            let _ = journal.rejected(step as u64, &format!("{error:#}"));
        }
        let _ = journal.sync();
    }
    if outcome.is_err() {
        // Normal error returns preserve all observed completed forwards. A
        // hard process death can leave an unreported tail and must remain
        // unknown during reconciliation; never infer zero lost exposure.
        if let Ok(mut exposure) = crate::training_exposure::attempt_snapshot(
            &observed_exposure,
            &committed_exposure,
            false,
        ) {
            exposure["failure_returned"] = json!(true);
            exposure["aborted_exposure_requires_stopped_rank_checkpoint_reconciliation"] =
                json!(true);
            let _ = write_json(
                &launch.join(format!("rank-{rank}.json")),
                &json!({
                "schema":"parallel.neural_training.v1","rank":rank,"world":world,"status":"failed",
                "initial_optimizer_step":initial_step,"step":optimizer.step,"corpus_sha256":corpus_hash,
                "exposure":exposure,"timestamp":now()}),
            );
        }
    }
    outcome
}

#[cfg(test)]
mod sampler_id_tests {
    use super::{active_sampler_id, REPLAY_SAMPLER_ID, SAMPLER_ID, SEARCH_BEATING_SAMPLER_ID};
    use serde_json::json;

    #[test]
    fn search_beating_identity_is_opt_in() {
        assert_eq!(active_sampler_id(None), SAMPLER_ID);
        assert_eq!(
            active_sampler_id(Some(&json!({"replay":"fresh_balanced"}))),
            REPLAY_SAMPLER_ID
        );
        assert_eq!(
            active_sampler_id(Some(&json!({"discoveries":"teacher_improved"}))),
            SEARCH_BEATING_SAMPLER_ID
        );
        assert_eq!(
            active_sampler_id(Some(&json!({"search_beating":true}))),
            SEARCH_BEATING_SAMPLER_ID
        );
    }
}

#[cfg(test)]
mod native_launch_tests {
    use super::*;
    #[test]
    fn new_fence_preserves_original_rendezvous_and_optimizer_window() {
        let t = tempfile::tempdir().unwrap();
        let job = jain_smartcluster::JobId::new();
        let first = jain_smartcluster::FencedAttempt {
            job_id: job,
            number: jain_smartcluster::AttemptNumber::new(1).unwrap(),
            fence: jain_smartcluster::FenceToken::new(24).unwrap(),
        };
        let second = jain_smartcluster::FencedAttempt {
            job_id: job,
            number: jain_smartcluster::AttemptNumber::new(2).unwrap(),
            fence: jain_smartcluster::FenceToken::new(25).unwrap(),
        };
        let window = json!({"initial_optimizer_step":60000,"requested_step_increment":100,"target_optimizer_step":60100});
        let a = native_attempt_launch(t.path(), first, &window).unwrap();
        fs::write(a.join("nccl.id"), b"old rendezvous").unwrap();
        let b = native_attempt_launch(t.path(), second, &window).unwrap();
        assert_ne!(a, b);
        assert!(!b.join("nccl.id").exists());
        assert_eq!(fs::read(a.join("nccl.id")).unwrap(), b"old rendezvous");
        let old: Value = read_json(&a.join("native-attempt.json")).unwrap();
        let new: Value = read_json(&b.join("native-attempt.json")).unwrap();
        assert_eq!(old["original_window_sha256"], new["original_window_sha256"]);
        assert!(native_attempt_launch(t.path(), first, &window).is_err());
    }
}
