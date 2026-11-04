use super::*;

/// The two guardrails `ops/generalization-plan.md` asked for, plus the one that
/// would have caught the outage they were written during.
///
/// 1. `corpus_dataset_count_fell` — evidence only ever accumulates, so a corpus
///    whose dataset count DROPS between freezes is always a sampler or catalog
///    defect. `test_exact_train.publish_freeze_guard` judges this at freeze time
///    and leaves the verdict in `ops/corpus-freeze-guard.v1.json`; this lifts it
///    where operators already look. It is an alarm and not a refusal to freeze
///    on purpose: refusing would pin the trainer to the OLD corpus, which is the
///    starvation the guardrail exists to detect.
/// 2. `corpus_split_median_divergence` — train and validation per-dataset
///    epoch-draw medians within 2x. `build_corpus` already QUARANTINES a freeze
///    that violates this, so a live alarm here means an older corpus is still
///    the one being trained on.
/// 3. `corpus_freeze_starved` — the alarm neither guardrail would have raised.
///    Both of them only fire when a freeze RUNS. For two days the live failure
///    was that no freeze ran at all: `freeze_corpus` is `PHASES[0]` of the
///    promotion tournament and a new tournament only starts once the previous
///    one terminates, so a tournament parked at `phase_index 9` pinned the
///    corpus at watermark 1,805,437 while the observations table reached
///    4,690,000+. Nothing anywhere went red. This measures the pointer gap
///    directly, so it is true whether or not anybody freezes.
pub(super) fn check_corpus_freeze_guard(root: &Path, alarms: &mut Alarms) {
    let guard_path = root.join("ops/corpus-freeze-guard.v1.json");
    if let Ok(bytes) = std::fs::read(&guard_path) {
        if let Ok(guard) = serde_json::from_slice::<Value>(&bytes) {
            if guard.get("regressed").and_then(Value::as_bool) == Some(true) {
                alarms.push(
                    "corpus_dataset_count_fell",
                    "critical",
                    guard
                        .get("observed")
                        .and_then(Value::as_str)
                        .unwrap_or("corpus dataset count regressed")
                        .to_owned(),
                    guard
                        .get("threshold")
                        .and_then(Value::as_str)
                        .unwrap_or("no prior freeze recorded")
                        .to_owned(),
                );
            }
            if let Some(scopes) = guard
                .get("split_median_divergence")
                .and_then(Value::as_object)
            {
                for (scope, values) in scopes {
                    let Some(ratio) = values.get("ratio").and_then(Value::as_f64) else {
                        continue;
                    };
                    if ratio <= SPLIT_MEDIAN_RATIO_MAX {
                        continue;
                    }
                    let train = values.get("train").and_then(Value::as_f64).unwrap_or(0.0);
                    let validation = values
                        .get("validation")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    alarms.push(
                        &format!("corpus_split_median_divergence:{scope}"),
                        "critical",
                        format!(
                            "{scope} epoch-draw medians train {train} vs validation \
                             {validation} ({ratio:.2}x)"
                        ),
                        format!("<={SPLIT_MEDIAN_RATIO_MAX:.1}x"),
                    );
                }
            }
        }
    }

    // The starvation measure. Both pointers are single small JSON files.
    let frozen = std::fs::read(root.join("training/current-corpus.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|value| value.get("watermark").and_then(Value::as_u64));
    let labels = std::fs::read(root.join("ops/last-ingest.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|value| {
            value
                .get("exact_rows_total")
                .or_else(|| value.get("watermark"))
                .and_then(Value::as_u64)
        });
    if let (Some(frozen), Some(labels)) = (frozen, labels) {
        let gap = labels.saturating_sub(frozen);
        let severity = if gap >= CORPUS_STARVED_CRIT {
            Some("critical")
        } else if gap >= CORPUS_STARVED_WARN {
            Some("warning")
        } else {
            None
        };
        if let Some(severity) = severity {
            alarms.push(
                "corpus_freeze_starved",
                severity,
                format!(
                    "{gap} labels outside the frozen corpus (pointer {frozen}, ingested \
                     {labels}); the trainer cannot see any of them"
                ),
                format!("<{CORPUS_STARVED_WARN}"),
            );
        }
    }

    // Only judged when the freezer is actually deployed on this root, so a root
    // without it does not carry a permanent alarm for a unit it never had.
    let status = root.join("ops/corpus-freezer-v1.json");
    if status.is_file() {
        if let Some(age) = file_age_secs(&status) {
            if age > CORPUS_FREEZER_STALE_SECS {
                alarms.push(
                    "status_stale:ops/corpus-freezer-v1.json",
                    "warning",
                    format!("{age}s"),
                    format!("{CORPUS_FREEZER_STALE_SECS}s"),
                );
            }
        }
        if let Ok(bytes) = std::fs::read(&status) {
            if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
                if value.get("lifecycle").and_then(Value::as_str) == Some("error") {
                    alarms.push(
                        "corpus_freezer_error",
                        "critical",
                        value
                            .get("error")
                            .and_then(Value::as_str)
                            .unwrap_or("corpus freezer reported an error")
                            .to_owned(),
                        "lifecycle=ok".to_owned(),
                    );
                }
                if let Some(alarm) = value.get("alarm").and_then(Value::as_str) {
                    alarms.push(
                        alarm,
                        "critical",
                        value
                            .get("detail")
                            .and_then(Value::as_str)
                            .unwrap_or(alarm)
                            .to_owned(),
                        "corpus pointer <= exact_rows_total".to_owned(),
                    );
                }
            }
        }
    }
}

/// A trainer submission intent that can NEVER succeed, however long it retries.
///
/// `jope_selfplay_trainer_watchdog.rs::render_spec` sha256s every spec input at
/// render time; `write_immutable` (:1512) then refuses to rewrite a spec whose
/// content differs, and `validate_intent` (:1355) re-checks only the watermark,
/// the checkpoint and the job id — never the input digests. So if any declared
/// input changes in the seconds between render and a successful submit, `scq
/// check` rejects the spec with `input-hash-mismatch` forever and the watchdog
/// retries the identical dead spec every ~13 s until a human clears the state.
///
/// This is not theoretical and it is not rare: it happened TWICE inside 90
/// minutes on 2026-07-28 (22:53Z and 23:33Z), both times because a workstream
/// edited `scripts/test_exact_train.py` while the watchdog was live. The first
/// occurrence cost 34 minutes of trainer downtime and was found only by reading
/// the journal by hand — every alarm in this file stayed green, because from the
/// outside a watchdog retrying looks exactly like a watchdog working.
///
/// The durable fix belongs in the watchdog (drop the intent and re-render when
/// the rejection is `input-hash-mismatch`). Until then, this makes it loud in
/// 60 seconds and prints the recovery.
pub(super) fn check_trainer_intent(root: &Path, alarms: &mut Alarms) {
    let path = root.join("ops/trainer-watchdog-state.v1.json");
    let Ok(bytes) = std::fs::read(&path) else {
        return;
    };
    let Ok(state) = serde_json::from_slice::<Value>(&bytes) else {
        return;
    };
    let Some(intent) = state.get("intent").filter(|value| !value.is_null()) else {
        return;
    };
    let Some(error) = intent.get("last_error").and_then(Value::as_str) else {
        return;
    };
    if !error.contains("input-hash-mismatch") {
        return;
    }
    let job = intent
        .get("job_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let spec = intent.get("spec").and_then(Value::as_str).unwrap_or("");
    alarms.push(
        "trainer_intent_unsubmittable",
        "critical",
        format!(
            "{job} can never submit: a spec input changed after the spec was frozen. \
             Recover by stopping jope-selfplay-trainer-watchdog, setting \
             \"intent\": null in ops/trainer-watchdog-state.v1.json, and restarting. \
             spec={spec}"
        ),
        "no unsubmittable trainer intent".to_owned(),
    );
}

pub(super) fn check_corpus_size(alarms: &mut Alarms) {
    if let Ok(meta) = std::fs::metadata(corpus_str()) {
        if meta.len() > CORPUS_WARN_BYTES {
            alarms.push(
                "corpus_size",
                "warning",
                format!("{:.1} GiB", meta.len() as f64 / f64::from(1u32 << 30)),
                format!("{} GiB", CORPUS_WARN_BYTES / (1 << 30)),
            );
        }
    }
}

/// The archive surviving GC is not the same as the archive surviving the disk.
///
/// `training/history/` lives on xbabe2's `/dev/nvme2n1p2` — the same filesystem
/// as the `var/runs` tree it exists to outlive. `jope-training-replica.timer`
/// mirrors it to xbabe3 and xbabe4 every 15 minutes and writes
/// `ops/training-replica.v1.json` every cycle, pass or fail. Three ways that
/// can quietly stop, each made loud here:
///
/// 1. `training_replica_missing` / `_stale` — the replication timer stopped, so
///    the archive is silently back to a single copy.
/// 2. `training_replica_target_stale:<node>` — one target has not accepted a
///    complete, verified publish for hours (unreachable, full, or refusing).
/// 3. `training_replica_degraded` — fewer than two targets are healthy. Two is
///    the point: one replica plus the source is a single failure away from a
///    single copy.
pub(super) fn check_training_replica(root: &Path, alarms: &mut Alarms) {
    let path = root.join("ops/training-replica.v1.json");
    let Ok(bytes) = std::fs::read(&path) else {
        alarms.push(
            "training_replica_missing",
            "critical",
            "no ops/training-replica.v1.json".into(),
            "replica timer writes it every 15 min".into(),
        );
        return;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return;
    };
    if value.get("schema").and_then(Value::as_str) != Some("JopeTrainingReplicaStatus.v1") {
        return;
    }
    if let Some(age) = file_age_secs(&path) {
        if age > REPLICA_RECEIPT_STALE_SECS {
            alarms.push(
                "training_replica_stale",
                "critical",
                format!("receipt {age}s old"),
                format!("<{REPLICA_RECEIPT_STALE_SECS}s"),
            );
        }
    }

    let now = now_ms() as u64;
    let targets = value
        .get("targets")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for target in targets {
        let name = target.get("target").and_then(Value::as_str).unwrap_or("?");
        if target.get("ok").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        // A target that has never succeeded reports 0; treat that as infinitely
        // stale rather than as a fresh success at the epoch.
        let last = target
            .get("last_success_ms")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let age = if last == 0 {
            u64::MAX
        } else {
            now.saturating_sub(last) / 1000
        };
        if age > REPLICA_TARGET_STALE_SECS {
            let observed = match last {
                0 => "never replicated successfully".to_string(),
                _ => format!("last good publish {age}s ago"),
            };
            alarms.push(
                &format!("training_replica_target_stale:{name}"),
                "critical",
                format!(
                    "{observed}; {}",
                    target
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("no error recorded")
                ),
                format!("a verified publish within {REPLICA_TARGET_STALE_SECS}s"),
            );
        }
    }

    let healthy = value
        .get("healthy_targets")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if healthy < REPLICA_MIN_HEALTHY {
        alarms.push(
            "training_replica_degraded",
            "critical",
            format!(
                "{healthy} of {} replica targets healthy",
                value
                    .get("target_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
            ),
            format!(">={REPLICA_MIN_HEALTHY} off-node replicas"),
        );
    }
}
