//! Counters for observed presentations, independent of sampler arithmetic.
//! A designated fresh comparison is not a promise that all its programs are fresh.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    pub program_presentations: u64,
    pub comparison_groups: u64,
    pub actual_fresh_program_presentations: u64,
    pub groups_with_actual_fresh_program: u64,
    pub designated_fresh_groups: u64,
    pub designated_broad_groups: u64,
}

impl Counts {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.comparison_groups.checked_mul(8) == Some(self.program_presentations)
                && self.actual_fresh_program_presentations <= self.program_presentations
                && self.groups_with_actual_fresh_program <= self.comparison_groups
                && self.groups_with_actual_fresh_program <= self.actual_fresh_program_presentations
                && self
                    .groups_with_actual_fresh_program
                    .checked_mul(8)
                    .is_some_and(|bound| self.actual_fresh_program_presentations <= bound)
                && self
                    .designated_fresh_groups
                    .checked_add(self.designated_broad_groups)
                    == Some(self.comparison_groups),
            "invalid training exposure counters"
        );
        Ok(())
    }
    pub fn observe(
        &mut self,
        ids: &[usize],
        fresh: &BTreeSet<usize>,
        serial: usize,
        balanced: bool,
    ) -> Result<()> {
        ensure!(
            ids.len() % 8 == 0 && serial % 8 == 0,
            "exposure requires complete comparison groups"
        );
        let groups = ids.len() / 8;
        (serial / 8)
            .checked_add(groups)
            .context("training exposure serial overflow")?;
        let fresh_groups = if balanced {
            (0..groups).filter(|n| (serial / 8 + n) % 2 == 0).count()
        } else {
            0
        };
        let batch = Self {
            program_presentations: ids.len().try_into()?,
            comparison_groups: groups.try_into()?,
            actual_fresh_program_presentations: ids
                .iter()
                .filter(|id| fresh.contains(id))
                .count()
                .try_into()?,
            groups_with_actual_fresh_program: ids
                .chunks(8)
                .filter(|group| group.iter().any(|id| fresh.contains(id)))
                .count()
                .try_into()?,
            designated_fresh_groups: fresh_groups.try_into()?,
            designated_broad_groups: (groups - fresh_groups).try_into()?,
        };
        *self = self.add(&batch)?;
        Ok(())
    }
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.combine(other, |left, right| left.checked_add(right))
    }
    pub fn subtract(&self, other: &Self) -> Result<Self> {
        self.combine(other, |left, right| left.checked_sub(right))
    }
    fn combine(&self, other: &Self, operation: impl Fn(u64, u64) -> Option<u64>) -> Result<Self> {
        self.validate()?;
        other.validate()?;
        let op = |left, right| {
            operation(left, right).context("training exposure overflow or nonmonotone commit")
        };
        let value = Self {
            program_presentations: op(self.program_presentations, other.program_presentations)?,
            comparison_groups: op(self.comparison_groups, other.comparison_groups)?,
            actual_fresh_program_presentations: op(
                self.actual_fresh_program_presentations,
                other.actual_fresh_program_presentations,
            )?,
            groups_with_actual_fresh_program: op(
                self.groups_with_actual_fresh_program,
                other.groups_with_actual_fresh_program,
            )?,
            designated_fresh_groups: op(
                self.designated_fresh_groups,
                other.designated_fresh_groups,
            )?,
            designated_broad_groups: op(
                self.designated_broad_groups,
                other.designated_broad_groups,
            )?,
        };
        value.validate()?;
        Ok(value)
    }
}

/// Historical checkpoints may lack exposure evidence. Preserve that unknown;
/// do not manufacture a cumulative total when resuming one of those states.
pub fn checkpoint_baseline(
    state: &Value,
    rank: usize,
    world: usize,
    batch_per_rank: usize,
) -> Result<Option<Counts>> {
    ensure!(
        world > 0 && rank < world && batch_per_rank > 0,
        "invalid checkpoint exposure partition"
    );
    let Some(rows) = state.get("committed_exposure_by_rank") else {
        return Ok(None);
    };
    let rows = rows.as_array().context("checkpoint exposure rows absent")?;
    ensure!(
        rank < world && rows.len() == world,
        "checkpoint exposure rank set differs"
    );
    let parent = state["initialization"]["optimizer_parent_step"]
        .as_u64()
        .unwrap_or(0);
    let steps = state["step"]
        .as_u64()
        .context("checkpoint exposure step absent")?
        .checked_sub(parent)
        .context("checkpoint exposure predates parent")?;
    for (i, row) in rows.iter().enumerate() {
        ensure!(
            row["rank"].as_u64() == Some(i as u64),
            "checkpoint exposure rank identity changed"
        );
        if !row["counts"].is_null() {
            let counts: Counts = serde_json::from_value(row["counts"].clone())?;
            counts.validate()?;
            ensure!(
                steps.checked_mul(batch_per_rank as u64) == Some(counts.program_presentations),
                "checkpoint exposure differs from committed step window"
            );
        }
    }
    if rows[rank]["counts"].is_null() {
        Ok(None)
    } else {
        Ok(Some(serde_json::from_value(rows[rank]["counts"].clone())?))
    }
}

pub fn attempt_snapshot(observed: &Counts, committed: &Counts, finished: bool) -> Result<Value> {
    let uncommitted = observed.subtract(committed)?;
    ensure!(
        !finished || uncommitted == Counts::default(),
        "completed task has uncommitted exposure"
    );
    Ok(
        json!({"schema":"parallel.training_attempt_exposure.v1", "scope":"this rank and attempt only",
        "observed_completed_microbatches":observed,"checkpoint_committed":committed,
        "uncommitted_or_checkpoint_acknowledgment_unknown":uncommitted,
        "aborted_exposure":if finished {Some(Counts::default())} else {None},
        "unknown_inflight_or_unreported_tail":!finished,"attempt_finished":finished,
        "reconciliation":"On failure retain these lower bounds, join durable checkpoint rank counts, and keep unobserved tail unknown"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_fresh_groups_can_have_one_fresh_program_among_sixteen_presentations() {
        let mut counts = Counts::default();
        counts
            .observe(&(0..16).collect::<Vec<_>>(), &BTreeSet::from([3]), 0, true)
            .unwrap();
        assert_eq!(counts.designated_fresh_groups, 1);
        assert_eq!(counts.designated_broad_groups, 1);
        assert_eq!(counts.actual_fresh_program_presentations, 1);
        assert_eq!(counts.program_presentations, 16);
    }
    #[test]
    fn observed_uncommitted_and_unknown_abort_are_not_checkpoint_credit() {
        let mut first = Counts::default();
        first
            .observe(&(0..16).collect::<Vec<_>>(), &BTreeSet::from([3]), 0, true)
            .unwrap();
        let mut second = Counts::default();
        second
            .observe(
                &(16..32).collect::<Vec<_>>(),
                &BTreeSet::from([21, 22]),
                16,
                true,
            )
            .unwrap();
        let all = first.add(&second).unwrap();
        let active = attempt_snapshot(&all, &first, false).unwrap();
        assert_eq!(
            active["uncommitted_or_checkpoint_acknowledgment_unknown"],
            serde_json::to_value(second).unwrap()
        );
        assert!(active["aborted_exposure"].is_null());
        assert_eq!(active["unknown_inflight_or_unreported_tail"], true);
        assert!(attempt_snapshot(&all, &first, true).is_err());
        assert_eq!(
            attempt_snapshot(&all, &all, true).unwrap()["unknown_inflight_or_unreported_tail"],
            false
        );
    }
    #[test]
    fn resume_counts_bind_rank_and_local_committed_window_while_legacy_stays_unknown() {
        assert!(checkpoint_baseline(&json!({"step":3000}), 0, 8, 256)
            .unwrap()
            .is_none());
        let mut counts = Counts::default();
        counts
            .observe(&(0..16).collect::<Vec<_>>(), &BTreeSet::from([3]), 0, true)
            .unwrap();
        let mut state = json!({"step":3001,"initialization":{"optimizer_parent_step":3000},
            "committed_exposure_by_rank":[{"rank":0,"counts":counts}]});
        assert_eq!(checkpoint_baseline(&state, 0, 1, 16).unwrap(), Some(counts));
        state["step"] = json!(3002);
        assert!(checkpoint_baseline(&state, 0, 1, 16).is_err());
        state["step"] = json!(3001);
        state["committed_exposure_by_rank"][0]["rank"] = json!(true);
        assert!(checkpoint_baseline(&state, 0, 1, 16).is_err());
    }
}
