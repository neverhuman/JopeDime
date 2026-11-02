use super::*;
use anyhow::ensure;
use serde_json::json;

#[macro_use]
#[path = "training_loop/steps.rs"]
mod steps;

macro_rules! training_loop {
    ($context:expr) => {{
        let (
            root,
            run,
            launch,
            rank,
            world,
            micro,
            steps,
            checkpoint_steps,
            resume,
            global_batch,
            batch_qualification,
            native_stop,
            execution,
            policy,
            policy_sha,
            replay,
            seed,
            corpus_hash,
            index,
            records,
            frozen_inputs_sha,
            families,
            validation,
            teacher_family,
            fresh,
            fresh_by_teacher,
            fresh_records,
            balanced,
            epoch_steps,
            fixed_panel,
            panel_sha256,
            config,
            recovered_continuation,
            mode,
            declaration,
            objective_sha,
            continuity,
            exposure_baseline,
            family_baseline,
            family_baseline_known,
            binding,
            comm,
            device,
            vs,
            mut optimizer,
            initialization,
            sampling_offset,
            resident_cache_hit,
            comparisons,
            loss_for,
            accumulation,
            started,
            initial_step,
            mut last_live_write,
            mut measured_step_seconds,
            mut telemetry_seconds,
            mut coordination_seconds,
            mut checkpoint_seconds,
            mut metrics,
            mut telemetry,
            mut last_checkpoint,
            mut validation_metrics,
            mut latest,
            mut observed_exposure,
            mut committed_exposure,
            mut observed_families,
            mut journal,
            journal_path,
            mut attempting
        ) = $context;
    let outcome = (|| -> Result<()> {
        training_loop_steps!((root, run, launch, rank, world, micro, steps, checkpoint_steps, global_batch, batch_qualification, native_stop, execution, policy, policy_sha, replay, seed, corpus_hash, index, records, frozen_inputs_sha, families, validation, teacher_family, fresh, fresh_by_teacher, fresh_records, balanced, epoch_steps, fixed_panel, panel_sha256, config, exposure_baseline, family_baseline, family_baseline_known, binding, comm, device, vs, optimizer, initialization, sampling_offset, comparisons, loss_for, accumulation, started, initial_step, last_live_write, measured_step_seconds, telemetry_seconds, coordination_seconds, checkpoint_seconds, metrics, telemetry, last_checkpoint, validation_metrics, latest, observed_exposure, committed_exposure, observed_families, journal, journal_path, attempting));
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
        #[cfg(feature = "telemetry")]
        if let Some((recorder, producer)) = telemetry.take() {
            drop(producer);
            if let Err(error) = recorder.close() {
                eprintln!("jope-telemetry finalization failed: {error:#}");
            }
        }
        Ok(())
    })();
        (
            outcome,
            optimizer,
            journal,
            attempting,
            observed_exposure,
            committed_exposure,
            observed_families,
            corpus_hash,
        )
    }};
}
