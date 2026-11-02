use super::*;

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
pub(super) fn hash_parameter_values(h: &mut sha2::Sha256, values: &[f32]) {
    use sha2::Digest;
    for value in values {
        h.update(value.to_le_bytes());
    }
}
pub(super) fn checkpoint(
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
