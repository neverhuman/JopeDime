use super::*;

macro_rules! training_loop_steps {
    (
        (
        $root:ident,
        $run:ident,
        $launch:ident,
        $rank:ident,
        $world:ident,
        $micro:ident,
        $steps:ident,
        $checkpoint_steps:ident,
        $global_batch:ident,
        $batch_qualification:ident,
        $native_stop:ident,
        $execution:ident,
        $policy:ident,
        $policy_sha:ident,
        $replay:ident,
        $seed:ident,
        $corpus_hash:ident,
        $index:ident,
        $records:ident,
        $frozen_inputs_sha:ident,
        $families:ident,
        $validation:ident,
        $teacher_family:ident,
        $fresh:ident,
        $fresh_by_teacher:ident,
        $fresh_records:ident,
        $balanced:ident,
        $epoch_steps:ident,
        $fixed_panel:ident,
        $panel_sha256:ident,
        $config:ident,
        $exposure_baseline:ident,
        $family_baseline:ident,
        $family_baseline_known:ident,
        $binding:ident,
        $comm:ident,
        $device:ident,
        $vs:ident,
        $optimizer:ident,
        $initialization:ident,
        $sampling_offset:ident,
        $comparisons:ident,
        $loss_for:ident,
        $accumulation:ident,
        $started:ident,
        $initial_step:ident,
        $last_live_write:ident,
        $measured_step_seconds:ident,
        $telemetry_seconds:ident,
        $coordination_seconds:ident,
        $checkpoint_seconds:ident,
        $metrics:ident,
        $telemetry:ident,
        $last_checkpoint:ident,
        $validation_metrics:ident,
        $latest:ident,
        $observed_exposure:ident,
        $committed_exposure:ident,
        $observed_families:ident,
        $journal:ident,
        $journal_path:ident,
        $attempting:ident
        )
    ) => {{
        for step in $optimizer.step as usize..$steps {
            $attempting = Some(step);
            let begun = Instant::now();
            let mut values = [0.0; 5];
            let mut loss_value = 0.0;
            // W7-T06: partition v2 draws the whole global batch once, forms
            // the global denominators from labels alone, and runs this rank's
            // contiguous shard of whole comparisons. Each microbatch's
            // objective is scaled by the world size, so the mean all-reduce
            // delivers the sum of rank gradients of the global objective.
            let mut step_denominators = None;
            if let Some(plan) = $execution.plan() {
                let local_step = (step - $sampling_offset) as u64;
                let scale = plan.local_scale(partition::Transport::Mean);
                let global_ids = sample_replay_fresh(
                    &$index,
                    &$families,
                    &$fresh,
                    &$fresh_by_teacher,
                    usize::try_from(plan.global_serial(local_step)?)?,
                    $global_batch,
                    $seed,
                    $balanced,
                );
                $comparisons.check_ids($index, &global_ids, "meta_train")?;
                let global_labels = global_ids
                    .iter()
                    .map(|&id| objective::RowLabels::from_record(id, &$records.get(id)))
                    .collect::<Vec<_>>();
                $comparisons.check_rows($index, &global_labels)?;
                let global = partition::denominators(&global_labels)?;
                let slots = plan.slots($rank, $seed, local_step)?;
                if slots.is_empty() {
                    // An empty rank contributes exact zero gradients with the
                    // same packed layout and still joins every collective.
                    partition::empty_rank_objective($vs.trainable_variables().iter())?.backward();
                }
                for slot in &slots {
                    tch::manual_seed(slot.dropout_seed);
                    let first = slot.first_comparison * objective::COMPARISON_ROWS;
                    let ids = global_ids[first..first + slot.rows].to_vec();
                    let serial = usize::try_from(slot.serial)?;
                    let guard = Bf16::enter();
                    let (loss, contributions) = $loss_for(&ids, true, Some(&global))?;
                    let loss = loss * scale;
                    loss_value += loss.double_value(&[]);
                    for i in 0..5 {
                        values[i] += contributions[i] * scale;
                    }
                    drop(guard);
                    loss.backward();
                    $observed_exposure.observe(
                        &ids,
                        &$fresh_records,
                        serial,
                        $balanced && !$fresh.is_empty(),
                    )?;
                    crate::training_exposure::observe_families(
                        &mut $observed_families,
                        &$teacher_family,
                        &ids,
                    )?;
                    let consumed = ids.iter().map(|&i| $records.get(i)).collect::<Vec<_>>();
                    $journal.microbatch(
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
            let v1_microbatches = if $execution.plan().is_none() {
                $accumulation
            } else {
                0
            };
            for mb in 0..v1_microbatches {
                let dropout_seed =
                    consumption::dropout_seed($seed, (step - $sampling_offset) as u64, $rank, mb);
                tch::manual_seed(dropout_seed);
                let serial = (step - $sampling_offset) * $global_batch
                    + $rank * ($global_batch / $world)
                    + mb * $micro;
                let ids = sample_replay_fresh(
                    &$index,
                    &$families,
                    &$fresh,
                    &$fresh_by_teacher,
                    serial,
                    $micro,
                    $seed,
                    $balanced,
                );
                let guard = Bf16::enter();
                let (loss, m) = $loss_for(&ids, true, None)?;
                let loss = loss / $accumulation as f64;
                loss_value += loss.double_value(&[]);
                for i in 0..5 {
                    values[i] += m[i] / $accumulation as f64;
                }
                drop(guard);
                loss.backward();
                $observed_exposure.observe(
                    &ids,
                    &$fresh_records,
                    serial,
                    $balanced && !$fresh.is_empty(),
                )?;
                crate::training_exposure::observe_families(
                    &mut $observed_families,
                    &$teacher_family,
                    &ids,
                )?;
                let consumed = ids.iter().map(|&i| $records.get(i)).collect::<Vec<_>>();
                $journal.microbatch(
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
            let mut variables = $vs.variables().into_iter().collect::<BTreeMap<_, _>>();
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
            $comm.reduce(&mut packed)?;
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
            $optimizer.update(&$vs)?;
            // Parameters may now differ from the step's start: a later failure
            // is not a rejected update. The journal then ends with this step's
            // microbatches and no outcome, and recovery re-executes it.
            $attempting = None;
            tch::Cuda::synchronize($binding.device_index() as i64);
            ensure!(
                $optimizer.step as usize == step + 1,
                "optimizer step did not advance by exactly one update"
            );
            $journal.applied(step as u64, norm)?;
            let mut totals = Tensor::from_slice(
                &[
                    loss_value, values[0], values[1], values[2], values[3], values[4],
                ]
                .map(|x| x as f32),
            )
            .to_device($device);
            $comm.reduce(&mut totals)?;
            let totals: Vec<f32> = Vec::try_from(totals.to_device(Device::Cpu))?;
            // Every rank votes on every step; only rank zero contributes the wall-clock trigger.
            let native_checkpoint =
                $native_stop.is_some_and(|stop| stop.load(std::sync::atomic::Ordering::Acquire));
            let due = native_checkpoint
                || (step + 1) % $checkpoint_steps == 0
                || (step + 1) % $epoch_steps == 0
                || step + 1 == $steps
                || ($rank == 0 && $last_checkpoint.elapsed().as_secs() >= 540);
            let vote_started = Instant::now();
            let mut checkpoint_vote =
                Tensor::from_slice(&[if due { 1f32 } else { 0f32 }]).to_device($device);
            $comm.reduce(&mut checkpoint_vote)?;
            let checkpoint_due = checkpoint_vote.double_value(&[0]) > 0.0;
            $coordination_seconds += vote_started.elapsed().as_secs_f64();
            if checkpoint_due && !$validation.is_empty() {
                let (v, sampled_examples) =
                    match $fixed_panel.as_ref().context("validation panel missing")? {
                        ValidationPanel::V1(panel) => {
                            let (_, v) = tch::no_grad(|| {
                                let _guard = Bf16::enter();
                                $loss_for(&panel[$rank], false, None)
                            })?;
                            (v, $world * 256)
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
                            for ids in &per_rank[$rank] {
                                let (_, contributions) = tch::no_grad(|| {
                                    let _guard = Bf16::enter();
                                    $loss_for(ids, false, Some(denominators))
                                })?;
                                for i in 0..5 {
                                    v[i] += contributions[i] * $world as f64;
                                }
                            }
                            (v, partition::VALIDATION_PANEL_ROWS)
                        }
                    };
                let mut t = Tensor::from_slice(&v.map(|x| x as f32)).to_device($device);
                $comm.reduce(&mut t)?;
                let v: Vec<f32> = Vec::try_from(t.to_device(Device::Cpu))?;
                $validation_metrics = json!({"sequence_nll":v[0],"ranking_bt":v[1],"ranking_listwise":v[2],"runtime_huber":v[3],"contextfree_bt":v[4],"weighted_losses":objective::weighted_losses(&v),"source_families":$validation.len(),"evaluated_step":step+1,"panel_sha256":$panel_sha256,"sampler_id":SAMPLER_ID,"sampled_examples":sampled_examples,"family_balanced":true,"evaluation_bias":"adaptive_cut_heldout_teacher","independent_evaluation":false});
            }
            let seconds = begun.elapsed().as_secs_f64();
            $measured_step_seconds += seconds;
            #[cfg(feature = "telemetry")]
            if let Some((_, producer)) = $telemetry.as_mut() {
                let at = (step + 1) as u64;
                use jope_telemetry::train as tm;
                producer.emit_scalar(tm::TRAIN_LOSS, 0, at, totals[0] as f64, false);
                for (group, metric_index) in (1_u16..=5).zip(1_usize..=5) {
                    producer.emit_scalar(
                        tm::TRAIN_LOSS_COMPONENT,
                        group - 1,
                        at,
                        totals[metric_index] as f64,
                        false,
                    );
                }
                producer.emit_scalar(tm::GRAD_NORM_GLOBAL, 0, at, norm, false);
                producer.emit_scalar(tm::GRAD_CLIPPED, 0, at, (norm > 1.0) as u8 as f64, false);
                producer.emit_scalar(
                    tm::GRAD_CLIP_COEF,
                    0,
                    at,
                    if norm > 1.0 { 1.0 / norm } else { 1.0 },
                    false,
                );
                producer.emit_scalar(tm::STEP_WALL_MS, 0, at, seconds * 1e3, false);
                producer.emit_scalar(
                    tm::SAMPLES_PER_S,
                    0,
                    at,
                    $global_batch as f64 / seconds.max(f64::MIN_POSITIVE),
                    false,
                );
                let nonfinite = totals.iter().filter(|v| !v.is_finite()).count()
                    + usize::from(!norm.is_finite());
                producer.emit_scalar(tm::TRAIN_NONFINITE, 0, at, nonfinite as f64, false);
            }
            let mut state = json!({"schema":"parallel.neural_training.v1","phase":"training on real GP teaching corpus","status":"active","run":$run.strip_prefix($root)?,"latest_receipt":$run.join("latest.json").strip_prefix($root)?,"checkpoint":$latest["checkpoint"],"checkpoint_step":$latest["step"],"rank":$rank,"world":$world,"step":step+1,"steps":$steps,"epoch":(step+1) as f64/$epoch_steps as f64,"microbatch":$micro,"accumulation":$accumulation,"global_batch":$global_batch,"loss":totals[0],"sequence_nll":totals[1],"ranking_bt":totals[2],"ranking_listwise":totals[3],"runtime_huber":totals[4],"contextfree_bt":totals[5],"weighted_losses":objective::weighted_losses(&totals[1..]),"inactive_objectives":["stability","loss_log_vars"],"gradient_clipped":norm>1.0,"validation":$validation_metrics,"gradient_norm":norm,"step_seconds":seconds,"communication_seconds":communication_seconds,"examples_per_second":$global_batch as f64/seconds,"elapsed_seconds":$started.elapsed().as_secs_f64(),"timestamp":now(),"corpus_sha256":$corpus_hash,"training_families":$families.len(),"sampler_id":SAMPLER_ID,"teacher_in_each_comparison":true,"qualification":true,"fresh_initialization":$initialization["kind"] == "fresh","initialization":$initialization,"autocast_cache_lifetime":"scoped_v2"});
            if $policy.is_some() {
                state["sampler_id"] = json!(active_sampler_id($replay.as_ref()));
                state["replay_policy_sha256"] = json!($policy_sha);
                state["local_step"] = json!(step + 1 - $sampling_offset);
                state["replay_comparisons_this_rank_attempt"] = json!({"total":$observed_exposure.comparison_groups,
                "fresh_comparisons":$observed_exposure.groups_with_actual_fresh_program,
                "fresh_programs":$observed_exposure.actual_fresh_program_presentations,
                "historical_comparisons":$observed_exposure.comparison_groups-$observed_exposure.groups_with_actual_fresh_program,
                "fresh_portion_fraction":if $balanced && !$fresh.is_empty() {0.5} else {0.0},
                "fallback_to_broad":$balanced && $fresh.is_empty()});
            }
            if $batch_qualification {
                state["batch_qualification"] = json!(true);
                state["eligible_checkpoint"] = json!(false);
            }
            if checkpoint_due {
                let checkpoint_started = Instant::now();
                crate::training_exposure::validate_families(
                    &$observed_families,
                    &$observed_exposure,
                )?;
                // Consumption through this update is durable before any
                // checkpoint can reference it.
                let consumed = $journal.sync()?;
                // Partition v2 persists every rank's RNG/sampler state and
                // the global committed update this checkpoint covers.
                let partition_state = match ($execution.plan(), &step_denominators) {
                    (Some(plan), Some(global)) => Some(plan.state(
                        $seed,
                        $sampling_offset as u64,
                        partition::CommittedUpdate::new(
                            plan,
                            $optimizer.step,
                            (step - $sampling_offset) as u64,
                            global,
                            [totals[1], totals[2], totals[3], totals[4], totals[5]].map(f64::from),
                            &packed,
                            norm,
                            &parameter_hash(&$vs)?,
                        )?,
                    )?),
                    _ => None,
                };
                checkpoint(
                    $root,
                    $run,
                    $launch,
                    $rank,
                    $world,
                    &$comm,
                    &$vs,
                    &$optimizer,
                    &$config,
                    &$corpus_hash,
                    &$index.vocabulary_sha256,
                    $seed,
                    $epoch_steps,
                    &$initialization,
                    &$validation_metrics,
                    $global_batch,
                    &$execution,
                    partition_state.as_ref(),
                    $batch_qualification,
                    &$exposure_baseline
                        .as_ref()
                        .map(|base| base.add(&$observed_exposure))
                        .transpose()?,
                    &crate::training_exposure::combine_families(
                        &$family_baseline,
                        &$observed_families,
                    )?,
                    $family_baseline_known,
                    &json!({"path":$journal_path,"binding":consumed}),
                    &$frozen_inputs_sha,
                    &$binding,
                )?;
                $committed_exposure = $observed_exposure.clone();
                $checkpoint_seconds += checkpoint_started.elapsed().as_secs_f64();
                $last_checkpoint = Instant::now();
                $latest = read_json(&$run.join("latest.json"))?;
                ensure!(
                    $latest["step"].as_u64() == Some($optimizer.step),
                    "published checkpoint does not cover the applied updates"
                );
                $journal.commit(
                    $optimizer.step,
                    $latest["state_sha256"]
                        .as_str()
                        .context("published checkpoint state digest absent")?,
                )?;
                state["checkpoint"] = $latest["checkpoint"].clone();
                state["checkpoint_step"] = $latest["step"].clone();
            }
            state["exposure"] = crate::training_exposure::attempt_snapshot(
                &$observed_exposure,
                &$committed_exposure,
                false,
            )?;
            state["initial_optimizer_step"] = json!($initial_step);
            if step + 1 == $steps {
                state["status"] = json!("complete");
                state["phase"] = json!("real GP teaching corpus training complete");
            }
            let elapsed = $started.elapsed().as_secs_f64();
            state["end_to_end_examples_per_second"] =
                json!((step + 1 - $initial_step) as f64 * $global_batch as f64 / elapsed);
            state["elapsed_seconds"] = json!(elapsed);
            state["telemetry_seconds"] = json!($telemetry_seconds);
            state["coordination_seconds"] = json!($coordination_seconds);
            state["checkpoint_seconds"] = json!($checkpoint_seconds);
            state["measured_step_seconds"] = json!($measured_step_seconds);
            let publish = checkpoint_due
                || step + 1 == $steps
                || $last_live_write.elapsed() >= Duration::from_secs(1);
            let telemetry_started = Instant::now();
            if let Some(writer) = &mut $metrics {
                serde_json::to_writer(&mut *writer, &state)?;
                writer.write_all(b"\n")?;
                if publish {
                    writer.flush()?;
                }
            }
            if publish {
                write_json(&$launch.join(format!("rank-{rank}.json")), &state)?;
                if $rank == 0 {
                    write_json(&$root.join("runs/neural-service.json"), &state)?;
                }
                $last_live_write = Instant::now();
            }
            $telemetry_seconds += telemetry_started.elapsed().as_secs_f64();
            if native_checkpoint && step + 1 < $steps {
                ensure!(checkpoint_due, "native recovery lacks committed checkpoint");
                anyhow::bail!(
                    "native control lost: checkpoint committed at original optimizer step {}",
                    step + 1
                );
            }
        }
    }};
}
