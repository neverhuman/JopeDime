use super::*;

/// Proposal-set health, read off the summary the generator already writes.
///
/// Context. Until 2026-07-28 a single dataset that the policy could not give
/// `--proposals-per-dataset` distinct genomes for was a FATAL error inside
/// `scripts/test_exact_dl_generate.py`; `seed_next_strict_wave` failed 848
/// consecutive times on `binary:0208f980ee49a057 produced only 34 unique
/// proposals` and the tournament parked at phase_index 9. That check is now a
/// deterministic cross-dataset backfill, so the job completes — which is
/// exactly why the shortfall has to become loud somewhere else. This is that
/// somewhere else.
///
/// What it reads (`proposals/current.json` -> `summary`, both the controller
/// micro-cohort lane and the propose driver publish it):
///   * `deficiency_by_task[task].deficient_fraction` — share of that task's
///     contexts the model could not fill on its own.
///   * `deficiency_by_task[task].min_native_unique` — the worst context.
///   * `native_head_counts_by_task[task]` — terminal-head histogram over
///     MODEL-SAMPLED rows only. One head at ~100% is mode collapse.
///   * `proposal_validity_by_task[task]` — share of decoded sequences whose
///     head is even admissible for the task.
///
/// Measured basis for the thresholds, both sets from checkpoint fbcd5459:
///   * micro-cohort `3a89b11ac7da` (128 contexts, 2026-07-28T23:11Z):
///     regression 0/64 contexts deficient, binary 19/64 (0.297), worst context
///     20 of 64 native uniques, 583 of 8,192 rows borrowed; binary head mix
///     logit_ridge 0.867 / tab_boost 0.093 / extra_trees 0.024 / gam_boost
///     0.015 / logistic 0.0002.
///   * whole-catalog `catalog-62054d7b05cb` (14,629 contexts, same day
///     20:47Z): binary heads 132,669 of 132,669 `logistic` (1.0000) and binary
///     decode validity 0.0407 — 96% of everything the model emitted for a
///     binary context carried a regression head.
///
/// So a 0.10 deficient fraction is already off the healthy branch, 1.0000 head
/// share is a state this system has genuinely reached, and the two coexist
/// under one checkpoint: the alarms have to be per task and per set.
pub(super) const PROPOSAL_DEFICIENT_FRACTION_WARN: f64 = 0.10;
pub(super) const PROPOSAL_DEFICIENT_FRACTION_CRIT: f64 = 0.50;
/// A context whose model-sampled uniques fall to a quarter of the budget is not
/// "thin", it is a collapsed conditional distribution.
pub(super) const PROPOSAL_MIN_NATIVE_RATIO_CRIT: f64 = 0.25;
/// One terminal head holding this share of a task's model-sampled proposals is
/// mode collapse: the binary allowlist has six admissible heads and regression
/// nine, and a healthy regression branch measures 0.23 for its modal head.
///
/// The WARN floor is set ABOVE the binary branch's normal skew, not at some
/// round number. Every non-collapsed binary reading taken on 2026-07-28 sits at
/// 0.878 (micro-cohort, 128 contexts) to 0.883 (4-context offline run) for
/// `logit_ridge`; the collapsed whole-catalog set sits at 1.0000 for `logistic`.
/// A floor at 0.80 would therefore be lit permanently and mean nothing. 0.90
/// clears the measured healthy band and still fires long before the 0.95
/// critical.
pub(super) const PROPOSAL_HEAD_SHARE_CRIT: f64 = 0.95;
pub(super) const PROPOSAL_HEAD_SHARE_WARN: f64 = 0.90;
/// Below this the decoder is spending most of its samples on heads that cannot
/// run for the task at all.
pub(super) const PROPOSAL_VALIDITY_WARN: f64 = 0.50;

pub(super) fn proposal_summary_path(root: &Path) -> Option<PathBuf> {
    let pointer_path = root.join("proposals/current.json");
    let bytes = std::fs::read(&pointer_path).ok()?;
    let pointer: Value = serde_json::from_slice(&bytes).ok()?;
    // Preferred: the summary the publisher recorded. It can live under
    // training/tournament-work/, which GC may reap, so fall back to the summary
    // that sits beside the archived proposal file itself.
    if let Some(path) = pointer.get("summary").and_then(Value::as_str) {
        let candidate = PathBuf::from(path);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    let proposals = pointer.get("proposals").and_then(Value::as_str)?;
    let sibling = PathBuf::from(proposals.strip_suffix(".jsonl")?.to_owned() + ".summary.json");
    sibling.is_file().then_some(sibling)
}

pub(super) fn check_proposal_deficiency(root: &Path, alarms: &mut Alarms) {
    let Some(path) = proposal_summary_path(root) else {
        return;
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return;
    };
    if value.get("schema").and_then(Value::as_str) != Some("JopeTestOnlyDlProposalSet.v1") {
        return;
    }
    let budget = value
        .get("proposals_per_dataset")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let label = value
        .get("model_checkpoint_sha256")
        .and_then(Value::as_str)
        .map(|digest| digest[..digest.len().min(12)].to_owned())
        .unwrap_or_else(|| "unknown".to_owned());

    // 1. Deficiency: how much of the set the model could not produce itself.
    // Pre-deficiency summaries have no `deficiency_by_task`; those runs are
    // simply not measured here rather than silently reported as healthy.
    if let Some(by_task) = value.get("deficiency_by_task").and_then(Value::as_object) {
        for (task, row) in by_task {
            let contexts = row.get("contexts").and_then(Value::as_i64).unwrap_or(0);
            if contexts <= 0 {
                continue;
            }
            let deficient = row
                .get("deficient_contexts")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let fraction = row
                .get("deficient_fraction")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let worst = row
                .get("min_native_unique")
                .and_then(Value::as_i64)
                .unwrap_or(budget);
            let borrowed = row
                .get("backfilled_rows")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let unfilled = row
                .get("unfilled_contexts")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let worst_ratio = if budget > 0 {
                worst as f64 / budget as f64
            } else {
                1.0
            };
            let severity = if fraction >= PROPOSAL_DEFICIENT_FRACTION_CRIT
                || worst_ratio <= PROPOSAL_MIN_NATIVE_RATIO_CRIT
            {
                Some("critical")
            } else if fraction >= PROPOSAL_DEFICIENT_FRACTION_WARN {
                Some("warning")
            } else {
                None
            };
            if let Some(severity) = severity {
                alarms.push(
                    &format!("proposal_deficiency:{task}"),
                    severity,
                    format!(
                        "{deficient}/{contexts} contexts short ({:.3}); worst {worst}/{budget} \
                         native uniques; {borrowed} rows borrowed cross-dataset; \
                         checkpoint {label}",
                        fraction
                    ),
                    format!(
                        "<{:.2} deficient fraction and worst context >{:.2} of budget",
                        PROPOSAL_DEFICIENT_FRACTION_WARN, PROPOSAL_MIN_NATIVE_RATIO_CRIT
                    ),
                );
            }
            if unfilled > 0 {
                // The cross-dataset pool itself ran dry: rows below the budget
                // reached disk, which every wave consumer rejects.
                alarms.push(
                    &format!("proposal_unfilled:{task}"),
                    "critical",
                    format!("{unfilled} contexts below {budget} rows after backfill"),
                    "0 contexts below budget".into(),
                );
            }
        }
    }

    // 2. Mode collapse: the histogram, not the count. A set can be 100%
    // complete and still be one genome shape repeated.
    if let Some(by_task) = value
        .get("native_head_counts_by_task")
        .and_then(Value::as_object)
    {
        for (task, histogram) in by_task {
            let Some(histogram) = histogram.as_object() else {
                continue;
            };
            let total: i64 = histogram.values().filter_map(Value::as_i64).sum();
            if total <= 0 {
                continue;
            }
            let mut ranked: Vec<(&String, i64)> = histogram
                .iter()
                .filter_map(|(head, count)| count.as_i64().map(|count| (head, count)))
                .collect();
            ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));
            let (head, count) = ranked[0];
            let share = count as f64 / total as f64;
            let severity = if share >= PROPOSAL_HEAD_SHARE_CRIT {
                Some("critical")
            } else if share >= PROPOSAL_HEAD_SHARE_WARN {
                Some("warning")
            } else {
                None
            };
            if let Some(severity) = severity {
                let mix = ranked
                    .iter()
                    .take(4)
                    .map(|(head, count)| format!("{head} {:.4}", *count as f64 / total as f64))
                    .collect::<Vec<_>>()
                    .join(", ");
                alarms.push(
                    &format!("proposal_head_collapse:{task}"),
                    severity,
                    format!(
                        "{head} holds {share:.4} of {total} model-sampled {task} proposals \
                         across {} distinct heads ({mix}); checkpoint {label}",
                        ranked.len()
                    ),
                    format!("modal head <{:.2}", PROPOSAL_HEAD_SHARE_WARN),
                );
            }
        }
    }

    // 3. Admissibility: how much of the decoder's output the task filter throws
    // away before diversity is even considered.
    if let Some(by_task) = value
        .get("proposal_validity_by_task")
        .and_then(Value::as_object)
    {
        for (task, reading) in by_task {
            let Some(validity) = reading.as_f64() else {
                continue;
            };
            if validity >= PROPOSAL_VALIDITY_WARN {
                continue;
            }
            alarms.push(
                &format!("proposal_task_validity:{task}"),
                if validity < 0.10 { "critical" } else { "warning" },
                format!("{validity:.4} of decoded {task} sequences carry an admissible head; checkpoint {label}"),
                format!(">={PROPOSAL_VALIDITY_WARN:.2}"),
            );
        }
    }
}
