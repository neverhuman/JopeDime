use super::*;

impl BoundedRunContract {
    /// Plan-time validation: structure plus the wall clock. Refuses a plan whose remaining
    /// original window or grants cannot fund the frozen ceilings.
    pub fn validate_at(&self, now_unix_ms: u64) -> Check {
        self.validate_structure()?;
        need!(
            now_unix_ms < self.original_deadline_unix_ms,
            Invalid,
            "original deadline already elapsed"
        );
        let start = now_unix_ms.max(self.not_before_unix_ms);
        let longest = self
            .participants
            .iter()
            .map(|p| p.ceilings.wall_ms)
            .max()
            .unwrap_or(0);
        need!(
            self.original_deadline_unix_ms - start >= longest,
            Underfunded,
            "remaining original window cannot fund the longest wall ceiling"
        );
        for p in &self.participants {
            let g = &p.grant.body;
            need!(
                now_unix_ms < g.expires_unix_ms,
                Invalid,
                "grant for rank {} already expired",
                p.global_rank
            );
            need!(
                g.expires_unix_ms
                    .saturating_sub(start.max(g.not_before_unix_ms))
                    >= p.ceilings.wall_ms,
                Underfunded,
                "remaining grant window for rank {} cannot fund its wall ceiling",
                p.global_rank
            );
        }
        Ok(())
    }
    /// A later attempt continues the same run, deadline, topology and counters.
    pub fn validate_successor(&self, previous: &Self, outcome: &AttemptOutcome) -> Check {
        let Some(p) = &self.predecessor else {
            return Err(PlanRefusal::Invalid(
                "successor names no predecessor".into(),
            ));
        };
        need!(
            outcome.terminal,
            Invalid,
            "predecessor attempt is not terminal; reconcile it first"
        );
        need!(
            self.run == previous.run
                && self.attempt == previous.attempt + 1
                && p.plan_sha256 == outcome.plan_sha256
                && p.journal_tip_sha256 == outcome.journal_tip_sha256
                && p.committed_updates == outcome.committed_updates_total
                && self.counters.committed_updates_at_start == outcome.committed_updates_total,
            Invalid,
            "successor does not continue the exact predecessor journal and counters"
        );
        need!(
            self.original_deadline_unix_ms == previous.original_deadline_unix_ms,
            Invalid,
            "a later attempt cannot renew the original deadline"
        );
        let topology = |c: &Self| -> Vec<(u32, u32, String, Option<String>)> {
            c.participants
                .iter()
                .map(|p| {
                    (
                        p.global_rank,
                        p.local_rank,
                        p.host.host.clone(),
                        p.device.as_ref().map(|d| d.uuid.clone()),
                    )
                })
                .collect()
        };
        // The selected training partition is topology too: a changed rank count, microbatch
        // or partition version is never an exact continuation (W7-T06).
        need!(
            topology(self) == topology(previous) && self.training == previous.training,
            Invalid,
            "a topology change requires an explicit transition and requalification"
        );
        Ok(())
    }
    /// Reviewable summary printed by Plan; it is derived, not a second authority.
    pub fn review(&self, plan_sha256: &str) -> Value {
        let participants: Vec<Value> = self
            .participants
            .iter()
            .map(|p| {
                json!({
                    "global_rank": p.global_rank,
                    "local_rank": p.local_rank,
                    "host": p.host.host,
                    "boot_id": p.host.boot_id,
                    "device": p.device,
                    "ceilings": p.ceilings,
                    "grant_expires_unix_ms": p.grant.body.expires_unix_ms,
                    "result_destination": p.result_destination,
                })
            })
            .collect();
        let mut kinds: BTreeMap<String, u64> = BTreeMap::new();
        for w in &self.work_census {
            // The kind's serde name (`numerical`), never its JSON rendering (`"numerical"`).
            let kind = serde_json::to_value(w.kind)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();
            *kinds.entry(kind).or_default() += w.wall_ms;
        }
        let mut review = json!({
            "schema": "jope.native_bounded_plan_review.v1",
            "plan_sha256": plan_sha256,
            "run": self.run,
            "attempt": self.attempt,
            "world_size": self.world_size(),
            "coordinator_host": self.coordinator_host,
            "not_before_unix_ms": self.not_before_unix_ms,
            "original_deadline_unix_ms": self.original_deadline_unix_ms,
            "participants": participants,
            "artifact_closure_entries": self.artifact_closure.len(),
            "work_census_wall_ms_by_kind": kinds,
            "counters": self.counters,
            "predecessor": self.predecessor,
            "dispatch_started": false,
        });
        // Only a plan that selects training gains the key; every other review is unchanged.
        if let Some(t) = &self.training {
            review["training"] = json!({
                "selection": t,
                "execution_partition": self.execution_partition().ok().flatten(),
            });
        }
        review
    }
}
