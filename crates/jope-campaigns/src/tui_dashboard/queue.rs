use super::*;

pub(super) fn parse_queue(raw: &RawSources, scq: &Value, families: &[FamilyRow]) -> QueueSummary {
    let queue = raw.queue.as_ref().unwrap_or(&Value::Null);
    let freshness = raw.freshness.as_ref().unwrap_or(&Value::Null);
    let counters = scq.get("counters").unwrap_or(&Value::Null);
    QueueSummary {
        queued: u64_at(counters, &["jobs_queued"])
            .unwrap_or_else(|| families.iter().map(|row| row.queued).sum()),
        starting: families.iter().map(|row| row.starting).sum(),
        dispatching: u64_at(queue, &["dispatching"])
            .unwrap_or_else(|| families.iter().map(|row| row.dispatching).sum()),
        running: u64_at(counters, &["jobs_running"])
            .or_else(|| u64_at(queue, &["running"]))
            .unwrap_or_else(|| families.iter().map(|row| row.running).sum()),
        held: u64_at(counters, &["jobs_held"])
            .or_else(|| u64_at(queue, &["held"]))
            .unwrap_or_else(|| families.iter().map(|row| row.held).sum()),
        useful_depth: u64_at(queue, &["queued_starting"]).unwrap_or(0),
        target: u64_at(queue, &["queued_target"]).unwrap_or(0),
        hard_cap: u64_at(queue, &["queued_hard_cap"]).unwrap_or(0),
        maximum_age_seconds: f64_at(queue, &["queue_maximum_age_seconds"]).unwrap_or(0.0),
        epoch: string_at(queue, &["checkpoint_epoch"])
            .or_else(|| string_at(freshness, &["checkpoint_epoch"]))
            .unwrap_or("-")
            .to_owned(),
        stale_epochs: u64_at(freshness, &["stale_epoch_count"]).unwrap_or(0),
        stale_held: u64_at(freshness, &["stale_held"]).unwrap_or(0),
        admission_allowed: bool_at(queue, &["admission_allowed"]),
        mode: string_at(queue, &["mode"]).unwrap_or("degraded").to_owned(),
    }
}

pub(super) fn parse_learning(raw: &RawSources, scq: &Value) -> LearningSummary {
    let status = raw.status.as_ref().unwrap_or(&Value::Null);
    let trainer = status.get("trainer").unwrap_or(&Value::Null);
    let ingestion = status.get("ingestion").unwrap_or(&Value::Null);
    let fallback = raw.trainer_fallback.as_ref().unwrap_or(&Value::Null);
    let proposal = raw.proposal_summary.as_ref().unwrap_or(&Value::Null);
    let flywheel = raw.flywheel.as_ref().unwrap_or(&Value::Null);
    let trainer_id = trainer
        .get("active_trainers")
        .and_then(Value::as_array)
        .and_then(|values| values.first())
        .and_then(Value::as_str)
        .or_else(|| string_at(trainer, &["trainer"]))
        .or_else(|| string_at(fallback, &["job_id"]))
        .unwrap_or("-");
    let trainer_state = scq
        .get("jobs")
        .and_then(Value::as_array)
        .and_then(|jobs| {
            jobs.iter().find(|job| {
                string_at(job, &["id"]).is_some_and(|candidate| candidate == trainer_id)
            })
        })
        .and_then(|job| string_at(job, &["state"]))
        .unwrap_or("not in SCQ");
    let promotion = status.get("promotion").unwrap_or(&Value::Null);
    LearningSummary {
        checkpoint: string_at(
            raw.queue.as_ref().unwrap_or(&Value::Null),
            &["active_checkpoint_sha256"],
        )
        .or_else(|| {
            string_at(
                raw.freshness.as_ref().unwrap_or(&Value::Null),
                &["checkpoint_sha256"],
            )
        })
        .or_else(|| string_at(proposal, &["model_checkpoint_sha256"]))
        .unwrap_or("-")
        .to_owned(),
        trainer: trainer_id.to_owned(),
        trainer_state: trainer_state.to_owned(),
        labels: u64_at(trainer, &["labels"])
            .or_else(|| u64_at(status, &["adaptive_fitness_watermark"]))
            .or_else(|| u64_at(ingestion, &["exact_rows_total"]))
            .unwrap_or(0),
        frozen_watermark: u64_at(flywheel, &["last_frozen_watermark"]).unwrap_or(0),
        internal_promotion: promotion_label(promotion.get("internal_learning")),
        milestone_promotion: promotion_label(promotion.get("milestone")),
        proposal_regression: u64_at(proposal, &["counts", "regression"]).unwrap_or(0),
        proposal_binary: u64_at(proposal, &["counts", "binary"]).unwrap_or(0),
        proposal_validity: f64_at(proposal, &["proposal_validity"]),
        consecutive_rejections: u64_at(status, &["evolution", "consecutive_rejections"])
            .or_else(|| u64_at(flywheel, &["consecutive_rejections"]))
            .unwrap_or(0),
    }
}

pub(super) fn parse_wave(raw: &RawSources) -> WaveSummary {
    let status = raw.status.as_ref().unwrap_or(&Value::Null);
    let queue = raw.queue.as_ref().unwrap_or(&Value::Null);
    let supply = raw.supply.as_ref().unwrap_or(&Value::Null);
    let catalog = status.get("catalog").unwrap_or(&Value::Null);
    let sealed = status.get("sealed_audit").unwrap_or(&Value::Null);
    let ingestion = status.get("ingestion").unwrap_or(&Value::Null);
    let pairs = status.get("pair_registry").unwrap_or(&Value::Null);
    WaveSummary {
        active_wave: string_at(queue, &["newest_micro_wave"])
            .or_else(|| string_at(status, &["active_micro_cohort", "wave"]))
            .unwrap_or("-")
            .to_owned(),
        admission_waves: queue
            .get("admission_waves")
            .and_then(Value::as_array)
            .map_or(0, Vec::len),
        unclaimed_manifests: u64_at(queue, &["unclaimed_current_checkpoint_manifests"])
            .or_else(|| u64_at(queue, &["manifests_unclaimed"]))
            .unwrap_or(0),
        ready_manifests: u64_at(supply, &["ready_manifests"]).unwrap_or(0),
        awaiting_keeper_ack: string_at(supply, &["awaiting_keeper_ack"])
            .unwrap_or("-")
            .to_owned(),
        adaptive_regression: u64_at(catalog, &["task_counts", "regression"]).unwrap_or(0),
        adaptive_binary: u64_at(catalog, &["task_counts", "binary"]).unwrap_or(0),
        sealed_isolated: bool_at(sealed, &["isolated_from_training"]),
        sealed_scored: u64_at(sealed, &["states", "scored"]).unwrap_or(0),
        corpus_rows: u64_at(status, &["adaptive_fitness_watermark"])
            .or_else(|| u64_at(ingestion, &["exact_rows_total"]))
            .unwrap_or(0),
        strict_wins: u64_at(ingestion, &["strict_wins"]).unwrap_or(0),
        pair_count: u64_at(pairs, &["pairs"]).unwrap_or(0),
        duplicate_pairs_prevented: u64_at(pairs, &["duplicates_prevented"]).unwrap_or(0),
        evidence_bytes: raw.evidence_bytes.unwrap_or(0),
    }
}

pub(super) fn parse_events(scq: &Value, local: &[Value]) -> Vec<String> {
    let mut events = scq
        .get("events")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .rev()
        .take(8)
        .map(format_event)
        .collect::<Vec<_>>();
    events.extend(local.iter().take(6).map(format_event));
    events.truncate(12);
    events
}

pub(super) fn parse_failures(scq: &Value) -> Vec<String> {
    scq.get("jobs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|job| string_at(job, &["state"]) == Some("failed"))
        .rev()
        .take(8)
        .map(|job| {
            format!(
                "{} {} {}",
                short(string_at(job, &["id"]).unwrap_or("-"), 24),
                string_at(job, &["failure_class"]).unwrap_or("failed"),
                short(string_at(job, &["message"]).unwrap_or(""), 48)
            )
        })
        .collect()
}
