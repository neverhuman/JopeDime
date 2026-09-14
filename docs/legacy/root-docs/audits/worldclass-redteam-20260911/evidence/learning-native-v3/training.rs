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

pub const SAMPLER_ID: &str = "family_uniform_champion_plus_seven_quantiles_v2";
pub const REPLAY_SAMPLER_ID: &str = "lineage_fresh_program_champion_plus_seven_v2";
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
pub fn sample_replay(index: &Index, families: &BTreeMap<String, Vec<usize>>,
    fresh: &BTreeMap<String, Vec<usize>>, serial: usize, count: usize, seed: u64,
    balanced: bool) -> Vec<usize> {
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
pub fn sample_replay_fresh(index: &Index, families: &BTreeMap<String, Vec<usize>>,
    fresh: &BTreeMap<String, Vec<usize>>, fresh_by_teacher: &BTreeMap<usize, Vec<usize>>,
    serial: usize, count: usize, seed: u64, balanced: bool) -> Vec<usize> {
    let mut result = sample_replay(index,families,fresh,serial,count,seed,balanced);
    if !balanced || fresh.is_empty() { return result; }
    for (n, ids) in result.chunks_mut(8).enumerate() {
        let comparison = serial/8+n;
        if comparison % 2 != 0 { continue; }
        let eligible = &fresh_by_teacher[&ids[0]];
        assert!(!eligible.is_empty(), "fresh group needs a fresh program");
        if !ids.iter().any(|id| eligible.contains(id)) {
            let fresh_id = eligible[mix(seed ^ comparison as u64 ^ 0x6672657368) as usize % eligible.len()];
            // Replace the closest sampled score band, keeping the champion.
            let slot = (1..8).min_by_key(|&slot| ids[slot].abs_diff(fresh_id)).unwrap();
            ids[slot] = fresh_id;
        }
    }
    result
}
/// Fixed global validation panel, identical on every evaluation of a corpus.
pub fn validation_panel(index: &Index, world: usize, seed: u64) -> Result<Vec<Vec<usize>>> {
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
    ensure!(
        !families.is_empty() && [1, 2, 4, 8].contains(&world),
        "validation panel requires families and a supported world size"
    );
    Ok((0..world)
        .map(|rank| sample(index, &families, rank * 256, 256, seed ^ 0x76616c))
        .collect())
}

fn parameter_hash(vs: &nn::VarStore) -> Result<String> {
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
    batch_qualification: bool,
) -> Result<()> {
    let step = opt.step;
    let receipt = launch.join(format!("checkpoint-{step}-rank-{rank}.json"));
    write_json(
        &receipt,
        &json!({"rank":rank,"step":step,"parameter_sha256":parameter_hash(vs)?}),
    )?;
    let mut barrier = Tensor::ones([1], (Kind::Float, Device::Cuda(rank)));
    comm.reduce(&mut barrier)?;
    if rank == 0 {
        let hashes = (0..world)
            .map(|r| read_json::<Value>(&launch.join(format!("checkpoint-{step}-rank-{r}.json"))))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            hashes.iter().all(
                |r| r["step"] == step && r["parameter_sha256"] == hashes[0]["parameter_sha256"]
            ),
            "checkpoint ranks diverged"
        );
        let staging = run.join(format!("checkpoints/.step-{step}-{}", now()));
        let dest = run.join(format!("checkpoints/step-{step}"));
        ensure!(!dest.exists(), "checkpoint already exists");
        fs::create_dir_all(&staging)?;
        vs.save(staging.join("model.ot"))?;
        opt.save(&staging.join("optimizer.ot"))?;
        write_json(&staging.join("config.json"), config)?;
        let state = json!({"schema":"parallel.checkpoint.v1","step":step,"world":world,"global_batch":global_batch,"model_sha256":file_hash(&staging.join("model.ot"))?,"optimizer_sha256":file_hash(&staging.join("optimizer.ot"))?,"optimizer_metadata_sha256":file_hash(&staging.join("optimizer.json"))?,"config_sha256":file_hash(&staging.join("config.json"))?,"corpus_sha256":corpus_hash,"vocabulary_sha256":vocab_hash,"parameter_sha256":hashes[0]["parameter_sha256"],"seed":seed,"rng":"counter reset before every microbatch: mix(seed ^ step*65537 ^ rank*257 ^ microbatch); LibTorch CPU/CUDA manual_seed","sampler":format!("family-uniform best teacher plus seven quantile comparators; next global example = step*{global_batch}"),"sampler_id":SAMPLER_ID,"scheduler":{"type":"constant","learning_rate":opt.lr,"epoch_steps":epoch_steps},"initialization":initialization,"validation":validation,"autocast_cache_lifetime":"scoped_v2","qualification":true,"independent_evaluation":false,"final_tests_opened":false});
        let mut state = state;
        if initialization.get("replay_policy_sha256").is_some() {
            state["sampler_id"] = json!(REPLAY_SAMPLER_ID);
            state["replay_policy_sha256"] = initialization["replay_policy_sha256"].clone();
            state["local_step"] = json!(step - initialization["optimizer_parent_step"].as_u64().unwrap_or(0));
        }
        if batch_qualification {
            state["batch_qualification"] = json!(true);
            state["eligible_checkpoint"] = json!(false);
        }
        state["ranking_backward"] = json!(crate::neural::RANKING_BACKWARD_ID);
        state["runtime_objective"] = json!("huber_log1p_seconds_v2");
        write_json(&staging.join("state.json"), &state)?;
        for name in [
            "model.ot",
            "optimizer.ot",
            "optimizer.json",
            "config.json",
            "state.json",
        ] {
            fs::File::open(staging.join(name))?.sync_all()?;
        }
        fs::File::open(&staging)?.sync_all()?;
        fs::rename(&staging, &dest)?;
        fs::File::open(dest.parent().unwrap())?.sync_all()?;
        write_json(
            &run.join("latest.json"),
            &json!({"checkpoint":dest.strip_prefix(root)?,"state_sha256":file_hash(&dest.join("state.json"))?,"step":step}),
        )?;
        retain_checkpoints(root, run)?;
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

/// Keep recent recovery points, thousand-step milestones and selected best/parent checkpoints.
fn retain_checkpoints(root: &Path, run: &Path) -> Result<()> {
    let mut protected = std::collections::BTreeSet::new();
    for entry in fs::read_dir(root.join("runs"))? {
        let path = entry?.path().join("initialization.json");
        if path.is_file() {
            let receipt: Value = read_json(&path)?;
            if let Some(path) = receipt["checkpoint"].as_str() {
                protected.insert(root.join(path));
            }
        }
    }
    let best = run.join("best.json");
    if best.is_file() {
        let receipt: Value = read_json(&best)?;
        if let Some(path) = receipt["checkpoint"].as_str() {
            protected.insert(resolve(root, Path::new(path))?);
        }
    }
    let mut checkpoints = Vec::new();
    for entry in fs::read_dir(run.join("checkpoints"))? {
        let path = entry?.path();
        if let Some(step) = path
            .file_name()
            .and_then(|x| x.to_str())
            .and_then(|x| x.strip_prefix("step-"))
            .and_then(|x| x.parse::<u64>().ok())
        {
            if path.join("state.json").is_file() {
                checkpoints.push((step, path));
            }
        }
    }
    checkpoints.sort_by_key(|(step, _)| *step);
    for (step, path) in checkpoints.iter().take(checkpoints.len().saturating_sub(3)) {
        if step % 1000 != 0 && !protected.contains(path) {
            fs::remove_dir_all(path)?;
        }
    }
    Ok(())
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
    if crate::gpu_worker::enabled() {
        return crate::gpu_worker::submit(root,rank,crate::gpu_worker::Task::Train {
            corpus:corpus.into(),run:run.into(),launch:launch.into(),world,micro,steps,checkpoint_steps,
            resume,initialize:initialize.map(Into::into)});
    }
    rank_impl(root, corpus, run, launch, rank, world, micro, steps, checkpoint_steps,
        resume, initialize, 2048, false, None)
}

/// Explicit policy warm-start experiment; ordinary production/resume stays unchanged.
pub fn rank_replay(root: &Path, corpus: &Path, run: &Path, launch: &Path,
    rank: usize, world: usize, steps: usize, resume: bool, initialize: Option<&Path>,
    policy: &Path) -> Result<()> {
    rank_impl(root, corpus, run, launch, rank, world, 256, steps, 1000,
                    resume, initialize, 2048, false, Some(policy))
}

/// Isolated real-corpus qualification; ordinary controllers keep their existing batch.
pub fn qualify_batch_rank(
    root: &Path, corpus: &Path, run: &Path, launch: &Path, rank: usize,
    micro: usize, global_batch: usize, steps: usize, initialize: &Path,
) -> Result<()> {
    ensure!([256, 512, 1024, 2048].contains(&micro) && global_batch == micro * 8,
        "qualification requires an explicit eight-rank global batch");
    let experiment = std::env::var_os("JOPE_EXPERIMENT_DIR")
        .context("batch qualification requires an isolated experiment directory")?;
    let experiment = Path::new(&experiment).canonicalize()?;
    for path in [root, run, launch] {
        ensure!(path.canonicalize()?.starts_with(&experiment),
            "batch qualification root, output and launch must stay inside its experiment");
    }
    rank_impl(root, corpus, run, launch, rank, 8, micro, steps, steps,
        false, Some(initialize), global_batch, true, None)
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
) -> Result<()> {
    ensure!(
        [1, 2, 4, 8].contains(&world)
            && rank < world
            && (if batch_qualification { [256, 512, 1024, 2048].contains(&micro) } else { [64, 128, 256].contains(&micro) })
            && [2048, 4096, 8192, 16384].contains(&global_batch)
            && global_batch % (world * micro) == 0
            && steps > 0
            && checkpoint_steps > 0,
        "invalid training topology"
    );
    crate::neural::configure_runtime();
    let policy: Option<Value> = replay_policy.map(read_json).transpose()?;
    let policy_sha = replay_policy.map(file_hash).transpose()?;
    let seed = match &policy {
        Some(p) => p["sampling_seed"].as_u64().context("replay sampling seed")?,
        None => SEED,
    };
    let corpus_hash = file_hash(&corpus.join("index.json"))?;
    let resident = crate::teaching_store::load_resident(root, corpus)?;
    let (index, records) = resident.as_ref();
    ensure!(
        ["parallel.teaching_corpus.v1", crate::teaching_store::SCHEMA].contains(&index.schema.as_str()) && index.qualification,
        "training currently requires explicitly qualified real corpus"
    );
    ensure!(
        file_hash(&corpus.join("vocabulary.json"))? == index.vocabulary_sha256,
        "corpus changed"
    );
    let vocabulary: crate::grammar::Vocab = read_json(&corpus.join("vocabulary.json"))?;
    let mut families = BTreeMap::<String, Vec<usize>>::new();
    let mut validation = BTreeMap::<String, Vec<usize>>::new();
    let mut train_records = 0;
    for (i, g) in index.groups.iter().enumerate() {
        if g.partition == "meta_train" {
            families.entry(g.family.clone()).or_default().push(i);
            train_records += g.records.len();
        } else if g.partition == "meta_validation" {
            validation.entry(g.family.clone()).or_default().push(i);
        }
    }
    ensure!(
        families.len() >= 2,
        "insufficient independent training families"
    );
    let mut fresh = BTreeMap::<String, Vec<usize>>::new();
    let mut fresh_by_teacher = BTreeMap::<usize,Vec<usize>>::new();
    let mut fresh_records = std::collections::BTreeSet::<usize>::new();
    let balanced = policy.as_ref().is_some_and(|p| p["replay"] == "fresh_balanced");
    if let Some(p) = &policy {
        ensure!(p["schema"] == "parallel.replay_policy.v2" && p["eligible"] == true
            && p["corpus_sha256"] == corpus_hash && p["vocabulary_sha256"] == index.vocabulary_sha256
            && p["final_tests_opened"] == false, "replay policy identity or eligibility changed");
        ensure!(["broad", "fresh_balanced"].iter().any(|v| p["replay"] == *v)
            && ["reset", "carry"].iter().any(|v| p["optimizer"] == *v), "unknown replay/optimizer mode");
        let mut seen = std::collections::BTreeSet::new();
        for value in p["fresh_groups"].as_array().context("fresh group indices")? {
            let i = value.as_u64().context("fresh group index")? as usize;
            ensure!(seen.insert(i), "duplicate fresh group");
            let g = index.groups.get(i).context("fresh group outside corpus")?;
            ensure!(g.partition == "meta_train" && g.records.len() >= 8,
                    "fresh replay includes validation/test or insufficient comparators");
            fresh.entry(g.family.clone()).or_default().push(i);
            let records = p["fresh_records"][i.to_string()].as_array().context("fresh program records absent")?
                .iter().map(|v| v.as_u64().map(|id|id as usize).context("invalid fresh program record"))
                .collect::<Result<Vec<_>>>()?;
            ensure!(!records.is_empty() && records.iter().all(|id| g.records.contains(id))
                && records.iter().collect::<std::collections::BTreeSet<_>>().len() == records.len(),
                "fresh records must be distinct members of the selected training group");
            fresh_records.extend(records.iter().copied());
            fresh_by_teacher.insert(g.records[0],records);
        }
    }
    let epoch_steps = train_records.div_ceil(global_batch).max(1);
    let fixed_panel = if validation.is_empty() {
        None
    } else {
        Some(validation_panel(&index, world, SEED)?)
    };
    let panel_sha256 = fixed_panel
        .as_ref()
        .map(|ids| serde_json::to_vec(ids).map(|bytes| hash(&bytes)))
        .transpose()?;
    let comm = Nccl::init(rank, world, &launch.join("nccl.id"))?;
    let device = Device::Cuda(rank);
    tch::manual_seed(seed as i64);
    let mut vs = nn::VarStore::new(device);
    let config = Config::regression(vocabulary.tokens.len() as i64);
    let model = Policy::new(vs.root(), config.clone())?;
    let mut optimizer = AdamW::new(1e-4, 0.01);
    ensure!(
        !(resume && initialize.is_some()),
        "resume and initialize are mutually exclusive"
    );
    let initialization_path = run.join("initialization.json");
    let mut initialization = json!({"kind":"fresh","seed":seed,"initialization":"pytorch_linear_mha_embedding_v2","autocast_cache_lifetime":"scoped_v2"});
    if let Some(checkpoint) = initialize {
        let checkpoint = resolve(root, checkpoint)?;
        let state: Value = read_json(&checkpoint.join("state.json"))?;
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
        ensure!(
            file_hash(&checkpoint.join("model.ot"))? == state["model_sha256"],
            "initialization model changed"
        );
        vs.load(checkpoint.join("model.ot"))?;
        initialization = json!({"kind":"warm_start","checkpoint":checkpoint.strip_prefix(root)?,
            "state_sha256":file_hash(&checkpoint.join("state.json"))?,"model_sha256":state["model_sha256"],
            "parent_step":state["step"],"parent_corpus_sha256":state["corpus_sha256"],
            "optimizer":"fresh FP32 AdamW", "sampler":"new corpus starts at step zero"});
        if let Some(p) = &policy {
            ensure!(p["parent_checkpoint_sha256"] == file_hash(&checkpoint.join("state.json"))?,
                    "replay initialization checkpoint changed");
            initialization["replay_policy_sha256"] = json!(policy_sha);
            initialization["replay_policy"] = p.clone();
            if p["optimizer"] == "carry" {
                ensure!(state["ranking_backward"] == crate::neural::RANKING_BACKWARD_ID
                    && state["runtime_objective"] == "huber_log1p_seconds_v2",
                    "optimizer warm-start requires compatible objective identity");
                ensure!(file_hash(&checkpoint.join("optimizer.ot"))? == state["optimizer_sha256"]
                    && file_hash(&checkpoint.join("optimizer.json"))? == state["optimizer_metadata_sha256"],
                    "warm-start optimizer missing or changed");
                optimizer = AdamW::load(&checkpoint.join("optimizer.ot"), device)?;
                ensure!(optimizer.step == state["step"].as_u64().context("parent optimizer step")?,
                        "warm-start optimizer/model step mismatch");
                optimizer.validate_parameters(&vs)?;
                initialization["optimizer"] = json!("carried FP32 AdamW with bias-correction step");
                initialization["optimizer_parent_step"] = json!(optimizer.step);
            }
            initialization["sampler"] = json!("lineage-bound counter; local block step starts at zero");
        }
    }
    if resume {
        let latest: Value = read_json(&run.join("latest.json"))?;
        let cp = resolve(
            root,
            Path::new(latest["checkpoint"].as_str().context("checkpoint path")?),
        )?;
        ensure!(
            file_hash(&cp.join("state.json"))? == latest["state_sha256"],
            "checkpoint receipt changed"
        );
        let state: Value = read_json(&cp.join("state.json"))?;
        ensure!(state["sampler_id"] == if policy.is_some() { REPLAY_SAMPLER_ID } else { SAMPLER_ID },
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
        let previous: Config = read_json(&cp.join("config.json"))?;
        ensure!(
            serde_json::to_value(previous)? == serde_json::to_value(&config)?,
            "resume architecture mismatch"
        );
        vs.load(cp.join("model.ot"))?;
        optimizer = AdamW::load(&cp.join("optimizer.ot"), device)?;
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
            ensure!(initialization["replay_policy_sha256"] == json!(policy_sha)
                && initialization["replay_policy"] == json!(policy), "exact resume replay policy changed");
        } else {
            ensure!(initialization.get("replay_policy_sha256").is_none(), "exact resume requires original replay policy");
        }
    }
    ensure!(policy.is_none() || initialize.is_some() || resume, "replay experiment requires explicit initialization or exact resume");
    let sampling_offset = initialization["optimizer_parent_step"].as_u64().unwrap_or(0) as usize;
    let steps = steps.checked_add(sampling_offset).context("training step overflow")?;
    if rank == 0 && !resume {
        write_json(&initialization_path, &initialization)?;
    }
    let context_tensor = RESIDENT_CONTEXT.with(|cache| -> Result<Tensor> {
        if let Some((identity,tensor)) = cache.borrow().as_ref() {
            if *identity == index.context_index_sha256 {return Ok(tensor.shallow_clone());}
        }
    let mut contexts = Vec::with_capacity(index.contexts.len() * 2048);
    for c in &index.contexts {
        let bytes = fs::read(resolve(root, &c.path)?)?;
        ensure!(
            hash(&bytes) == c.sha256 && bytes.len() == 8192,
            "context artifact changed"
        );
        contexts.extend(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        );
    }
    // Small immutable artifacts are staged once on each GPU; no CSV loading in training steps.
    let tensor = Tensor::from_slice(&contexts)
        .view([-1, 2048])
        .to_device(device);
    // Device tensors own their copied storage; release host staging before training.
    drop(contexts);
        *cache.borrow_mut() = Some((index.context_index_sha256.clone(),tensor.shallow_clone()));
        Ok(tensor)
    })?;
    let (resident_cache_hit, tensors) = RESIDENT_TENSORS.with(|cache| -> Result<(bool,Vec<Tensor>)> {
        if let Some((identity,tensors)) = cache.borrow().as_ref() {
            if *identity == corpus_hash {return Ok((true,tensors.iter().map(Tensor::shallow_clone).collect()));}
        }
    let mut tokens = Vec::with_capacity(index.records * SEQUENCE);
    let mut legal_ids = Vec::with_capacity(index.records * SEQUENCE);
    let mut context_ids = Vec::with_capacity(index.records);
    for i in 0..index.records {
        let r = records.get(i);
        // LibTorch's short tensor is signed. Preserve all sixteen identifier
        // bits and restore unsigned values only after gathering a minibatch.
        tokens.extend(r.tokens.map(|id| id as u16 as i16));
        legal_ids.extend(r.legal.map(|id| id as u16 as i16));
        context_ids.push(r.context as i64);
    }
    let token_tensor = Tensor::from_slice(&tokens)
        .view([-1, SEQUENCE as i64])
        .to_device(device);
    drop(tokens);
    let legal_id_tensor = Tensor::from_slice(&legal_ids)
        .view([-1, SEQUENCE as i64])
        .to_device(device);
    drop(legal_ids);
    let context_id_tensor = Tensor::from_slice(&context_ids).to_device(device);
    drop(context_ids);
    let mut masks = vec![0u8; index.legal_sets.len() * vocabulary.tokens.len()];
    for (i, set) in index.legal_sets.iter().enumerate() {
        for &t in set {
            masks[i * vocabulary.tokens.len() + t] = 1;
        }
    }
    let mask_table = Tensor::from_slice(&masks)
        .view([-1, vocabulary.tokens.len() as i64])
        .to_kind(Kind::Bool)
        .to_device(device);
        let tensors = vec![token_tensor,legal_id_tensor,context_id_tensor,mask_table];
        *cache.borrow_mut() = Some((corpus_hash.clone(),tensors.iter().map(Tensor::shallow_clone).collect()));
        Ok((false,tensors))
    })?;
    let [token_tensor,legal_id_tensor,context_id_tensor,mask_table]: [Tensor;4] = tensors.try_into().ok().unwrap();
    let loss_for = |ids: &[usize], train: bool| -> Result<(Tensor, [f64; 4])> {
        let rows = ids.iter().map(|&i| records.get(i)).collect::<Vec<_>>();
        let length = rows.iter().map(|r| r.length).max().unwrap().div_ceil(8) * 8;
        let length = length.min(SEQUENCE) as i64;
        let id = Tensor::from_slice(&ids.iter().map(|&i| i as i64).collect::<Vec<_>>())
            .to_device(device);
        let tokens = token_tensor.index_select(0, &id).narrow(1, 0, length)
            .to_kind(Kind::Int64).bitwise_and(0xffffi64);
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
                    .to_kind(Kind::Int64).bitwise_and(0xffffi64),
            )
            .view([ids.len() as i64, length - 1, vocabulary.tokens.len() as i64]);
        let weights = Tensor::from_slice(
            &rows
                .iter()
                .map(|r| 0.05 + r.percentile.powi(4))
                .collect::<Vec<_>>(),
        )
        .to_device(device);
        let nll = crate::neural::sequence_nll(
            &logits.narrow(1, 1, length - 1),
            &tokens.narrow(1, 1, length - 1),
            &legal,
            &pad.narrow(1, 1, length - 1),
            &weights,
        )?;
        let reward = rewards.mean_dim(&[1i64][..], false, Kind::Float);
        let values = rows.iter().map(|r| r.value as f64).collect::<Vec<_>>();
        let percentiles = rows.iter().map(|r| r.percentile).collect::<Vec<_>>();
        let groups = rows
            .iter()
            .enumerate()
            .map(|(i, r)| format!("{}:{}", r.context, i / 8))
            .collect::<Vec<_>>();
        let (bt, list) = crate::neural::ranking(&reward, &values, &percentiles, &groups)?;
        let abt = crate::neural::ranking_pairwise(&ablation, &values, &percentiles, &groups)?;
        let runtime_label = Tensor::from_slice(&rows.iter().map(|r| r.runtime).collect::<Vec<_>>())
            .to_device(device);
        let rt = runtime.to_kind(Kind::Float).log1p().smooth_l1_loss(
            &runtime_label,
            tch::Reduction::Mean,
            1.0,
        );
        let mut loss = &nll
            + &bt * 0.25
            + &list * 0.1
            + &rt * 0.1
            + abt * 0.05
            + stability.sum(Kind::Float) * 0.0;
        // Inactive optional objectives stay connected for identical packed gradient layouts.
        for x in model.loss_log_vars.values() {
            loss += x * 0.0;
        }
        Ok((
            loss,
            [
                nll.double_value(&[]),
                bt.double_value(&[]),
                list.double_value(&[]),
                rt.double_value(&[]),
            ],
        ))
    };
    let accumulation = global_batch / (world * micro);
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
    let mut actual_fresh = 0usize;
    let mut actual_fresh_programs = 0usize;
    let mut comparisons = 0usize;
    for step in optimizer.step as usize..steps {
        let begun = Instant::now();
        let mut values = [0.0; 4];
        let mut loss_value = 0.0;
        for mb in 0..accumulation {
            tch::manual_seed(mix(seed
                ^ ((step - sampling_offset) as u64).wrapping_mul(65537)
                ^ (rank as u64).wrapping_mul(257)
                ^ mb as u64) as i64);
            let ids = sample_replay_fresh(
                &index,
                &families,
                &fresh,
                &fresh_by_teacher,
                (step - sampling_offset) * global_batch + rank * (global_batch / world) + mb * micro,
                micro,
                seed,
                balanced,
            );
            comparisons += ids.len() / 8;
            actual_fresh += ids.chunks(8).filter(|ids| ids.iter().any(|id| fresh_records.contains(id))).count();
            actual_fresh_programs += ids.iter().filter(|id| fresh_records.contains(id)).count();
            let guard = Bf16::enter();
            let (loss, m) = loss_for(&ids, true)?;
            let loss = loss / accumulation as f64;
            loss_value += loss.double_value(&[]);
            for i in 0..4 {
                values[i] += m[i] / accumulation as f64;
            }
            drop(guard);
            loss.backward();
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
        tch::Cuda::synchronize(rank as i64);
        let mut totals = Tensor::from_slice(
            &[loss_value, values[0], values[1], values[2], values[3]].map(|x| x as f32),
        )
        .to_device(device);
        comm.reduce(&mut totals)?;
        let totals: Vec<f32> = Vec::try_from(totals.to_device(Device::Cpu))?;
        // Every rank votes on every step; only rank zero contributes the wall-clock trigger.
        let due = (step + 1) % checkpoint_steps == 0
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
            let ids = &fixed_panel.as_ref().context("validation panel missing")?[rank];
            let (_, v) = tch::no_grad(|| {
                let _guard = Bf16::enter();
                loss_for(ids, false)
            })?;
            let mut t = Tensor::from_slice(&v.map(|x| x as f32)).to_device(device);
            comm.reduce(&mut t)?;
            let v: Vec<f32> = Vec::try_from(t.to_device(Device::Cpu))?;
            validation_metrics = json!({"sequence_nll":v[0],"ranking_bt":v[1],"ranking_listwise":v[2],"runtime_huber":v[3],"source_families":validation.len(),"evaluated_step":step+1,"panel_sha256":panel_sha256,"sampler_id":SAMPLER_ID,"sampled_examples":world*256,"family_balanced":true,"evaluation_bias":"adaptive_cut_heldout_teacher","independent_evaluation":false});
        }
        let seconds = begun.elapsed().as_secs_f64();
        measured_step_seconds += seconds;
        let mut state = json!({"schema":"parallel.neural_training.v1","phase":"training on real GP teaching corpus","status":"active","run":run.strip_prefix(root)?,"latest_receipt":run.join("latest.json").strip_prefix(root)?,"checkpoint":latest["checkpoint"],"checkpoint_step":latest["step"],"rank":rank,"world":world,"step":step+1,"steps":steps,"epoch":(step+1) as f64/epoch_steps as f64,"microbatch":micro,"accumulation":accumulation,"global_batch":global_batch,"loss":totals[0],"sequence_nll":totals[1],"ranking_bt":totals[2],"ranking_listwise":totals[3],"runtime_huber":totals[4],"validation":validation_metrics,"gradient_norm":norm,"step_seconds":seconds,"communication_seconds":communication_seconds,"examples_per_second":global_batch as f64/seconds,"elapsed_seconds":started.elapsed().as_secs_f64(),"timestamp":now(),"corpus_sha256":corpus_hash,"training_families":families.len(),"sampler_id":SAMPLER_ID,"teacher_in_each_comparison":true,"qualification":true,"fresh_initialization":initialization["kind"] == "fresh","initialization":initialization,"autocast_cache_lifetime":"scoped_v2"});
        if policy.is_some() {
            state["sampler_id"] = json!(REPLAY_SAMPLER_ID);
            state["replay_policy_sha256"] = json!(policy_sha);
            state["local_step"] = json!(step + 1 - sampling_offset);
            state["replay_comparisons_this_rank_attempt"] = json!({"total":comparisons,
                "fresh_comparisons":actual_fresh,"fresh_programs":actual_fresh_programs,
                "historical_comparisons":comparisons-actual_fresh,
                "fresh_portion_fraction":if balanced && !fresh.is_empty() {0.5} else {0.0},
                "fallback_to_broad":balanced && fresh.is_empty()});
        }
        if batch_qualification {
            state["batch_qualification"] = json!(true);
            state["eligible_checkpoint"] = json!(false);
        }
        if checkpoint_due {
            let checkpoint_started = Instant::now();
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
                batch_qualification,
            )?;
            checkpoint_seconds += checkpoint_started.elapsed().as_secs_f64();
            last_checkpoint = Instant::now();
            latest = read_json(&run.join("latest.json"))?;
            state["checkpoint"] = latest["checkpoint"].clone();
            state["checkpoint_step"] = latest["step"].clone();
        }
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
    }
    let mut complete = json!({"rank":rank,"world":world,"steps":optimizer.step,"parameter_sha256":parameter_hash(&vs)?,"seconds":started.elapsed().as_secs_f64(),"executed_steps":optimizer.step as usize-initial_step,"end_to_end_examples_per_second":(optimizer.step as usize-initial_step) as f64*global_batch as f64/started.elapsed().as_secs_f64(),"telemetry_seconds":telemetry_seconds,"coordination_seconds":coordination_seconds,"checkpoint_seconds":checkpoint_seconds,"measured_step_seconds":measured_step_seconds,"corpus_sha256":corpus_hash});
    complete["resident_tensor_cache_hit"] = json!(resident_cache_hit);
    complete["resident_identifier_bytes"] = json!(index.records * SEQUENCE * 4);
    if policy.is_some() {
        complete["replay_policy_sha256"] = json!(policy_sha);
        complete["local_steps"] = json!(optimizer.step as usize - sampling_offset);
        complete["actual_fresh_comparisons_this_rank_attempt"] = json!(actual_fresh);
        complete["actual_fresh_programs_this_rank_attempt"] = json!(actual_fresh_programs);
        complete["total_comparisons_this_rank_attempt"] = json!(comparisons);
    }
    if batch_qualification {
        complete["microbatch"] = json!(micro);
        complete["global_batch"] = json!(global_batch);
        complete["batch_qualification"] = json!(true);
        complete["eligible_checkpoint"] = json!(false);
    }
    write_json(&launch.join(format!("complete-{rank}.json")), &complete)?;
    Ok(())
}
