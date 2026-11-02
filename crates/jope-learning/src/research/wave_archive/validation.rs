use super::*;

pub(super) fn validate(input: &WaveInput) -> anyhow::Result<()> {
    anyhow::ensure!(
        safe_segment(&input.wave_id) && !input.version.is_empty() && safe_segment(&input.version),
        "unsafe wave id/version"
    );
    anyhow::ensure!(
        input.events.len() <= MAX_EVENTS,
        "event count exceeds limit"
    );
    if input.new_wave {
        anyhow::ensure!(
            !input.events.is_empty(),
            "new-wave evidence has no producer events"
        );
        for key in [
            "source",
            "config",
            "grammar",
            "evaluator",
            "registry",
            "context",
            "support",
            "optimizer_recipe",
            "rng_sampler",
            "data_role",
        ] {
            anyhow::ensure!(
                matches!(input.identity.get(key), Some(Known::Known(value)) if !value.is_empty()),
                "new wave missing required identity {key}"
            );
        }
    }
    for event in &input.events {
        anyhow::ensure!(!event.kind.trim().is_empty(), "event kind is required");
        for metric in event.metrics.values() {
            anyhow::ensure!(
                !metric.unit.is_empty() && !metric.source_field.is_empty(),
                "metric unit/source are required"
            );
            if let Some(value) = metric.value {
                anyhow::ensure!(
                    value.is_finite() && metric.denominator.is_some_and(|n| n > 0),
                    "finite metric requires positive denominator"
                );
            } else {
                anyhow::ensure!(
                    metric
                        .missing_reason
                        .as_ref()
                        .is_some_and(|s| !s.trim().is_empty()),
                    "null metric requires missing reason"
                );
            }
        }
        if input.new_wave {
            for key in ["computed", "applied", "committed"] {
                anyhow::ensure!(
                    event.counters.contains_key(key),
                    "new-wave event lacks {key} counter"
                );
            }
        }
    }
    let mut sorted = input.events.clone();
    sorted.sort_by_key(|e| (e.sequence, e.rank, e.attempt));
    validate_event_order(&sorted)
}

pub(super) fn validate_event_order(events: &[Event]) -> anyhow::Result<()> {
    let mut keys = BTreeSet::new();
    for event in events {
        anyhow::ensure!(
            keys.insert((event.sequence, event.rank, event.attempt)),
            "duplicate event identity"
        );
    }
    Ok(())
}
pub(super) fn estimated_event_bytes(event: &Event) -> usize {
    let strings = event.kind.len()
        + event
            .fields
            .iter()
            .map(|(k, v)| {
                k.len()
                    + match v {
                        Known::Known(s) => s.len(),
                        Known::Missing { reason } => reason.len(),
                    }
            })
            .sum::<usize>()
        + event
            .counters
            .iter()
            .map(|(k, v)| {
                k.len()
                    + match v {
                        Known::Known(_) => 24,
                        Known::Missing { reason } => reason.len(),
                    }
            })
            .sum::<usize>()
        + event
            .metrics
            .iter()
            .map(|(k, m)| {
                k.len()
                    + m.unit.len()
                    + m.status.len()
                    + m.source_field.len()
                    + m.missing_reason.as_ref().map_or(0, String::len)
                    + 64
            })
            .sum::<usize>();
    // JSON escaping can expand each control character to six bytes; use a conservative bound.
    strings.saturating_mul(6).saturating_add(256)
}
pub(super) fn safe_segment(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
}
