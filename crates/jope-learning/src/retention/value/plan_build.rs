use super::*;

impl RetentionPlanV2 {
    /// Build a deterministic, in-memory plan. No operation is performed on source objects.
    pub fn build(
        as_of_unix: u64,
        inventory_max_age_secs: u64,
        source_inventory_sha256: &str,
        policy: RetentionPolicyV2,
        mut candidates: Vec<Candidate>,
    ) -> Result<Self> {
        policy.validate()?;
        validate_digest(source_inventory_sha256, "source inventory digest")?;
        ensure!(
            inventory_max_age_secs > 0,
            "inventory maximum age must be positive"
        );
        candidates.sort_by(|a, b| a.sources.cmp(&b.sources));
        let current_objects: BTreeSet<PhysicalObject> = candidates
            .iter()
            .flat_map(|candidate| candidate.sources.iter().cloned())
            .collect();
        let keeper_objects: BTreeSet<PhysicalObject> = candidates
            .iter()
            .filter(|candidate| {
                candidate.hard_t0
                    || candidate.active_pin
                    || candidate.active_promise
                    || candidate.score_row
                    || candidate.reservoir_selected == Some(true)
                    || candidate.ranking_current
                        && (candidate
                            .context_rank
                            .is_some_and(|rank| rank <= policy.k_ctx)
                            || candidate
                                .lineage_rank
                                .is_some_and(|rank| rank <= policy.k_lineage))
            })
            .flat_map(|candidate| candidate.sources.iter().cloned())
            .collect();
        let mut seen = BTreeSet::new();
        let mut items = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            ensure!(
                !candidate.sources.is_empty(),
                "candidate has no source objects"
            );
            for source in &candidate.sources {
                source.validate()?;
                ensure!(
                    seen.insert(source.clone()),
                    "duplicate physical source in candidates: {}",
                    source.key
                );
            }
            if let Some(signals) = &candidate.signals {
                signals.validate()?;
            }
            let current_bytes = sum_bytes(&candidate.sources)?;
            let pinned = candidate.hard_t0 || candidate.active_pin || candidate.active_promise;
            let values = candidate
                .sources
                .iter()
                .map(|source| {
                    RetentionValue::classify(
                        source,
                        pinned,
                        candidate.score_row,
                        candidate.signals.as_ref(),
                        &policy,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let mut unresolved = Vec::new();
            let mut recommendation = Recommendation::Hold;
            let mut replacement = None;
            let mut reason = String::new();
            if candidate.hard_t0
                || candidate.active_pin
                || candidate.active_promise
                || candidate.score_row
            {
                recommendation = if candidate.hard_t0 {
                    Recommendation::KeepT0
                } else if candidate.score_row {
                    Recommendation::KeepScoreRows
                } else {
                    Recommendation::Keep
                };
                reason = if candidate.hard_t0 {
                    "hard T0 obligation"
                } else if candidate.score_row {
                    "score rows are retained"
                } else if candidate.active_pin {
                    "active evidence pin"
                } else {
                    "active future promise"
                }
                .into();
            } else if !candidate.ranking_current
                || candidate.context_rank.is_none()
                || candidate.lineage_rank.is_none()
                || candidate.reservoir_selected.is_none()
            {
                unresolved
                    .push("current keeper ranking and reservoir evidence is incomplete".into());
                reason = "no change proposed because keeper status is unknown".into();
            } else if candidate.context_rank.unwrap_or(u64::MAX) <= policy.k_ctx
                || candidate.lineage_rank.unwrap_or(u64::MAX) <= policy.k_lineage
                || candidate.reservoir_selected == Some(true)
            {
                recommendation = Recommendation::Keep;
                reason = "retained by context, lineage, or reservoir obligation".into();
            } else {
                match candidate.inventory_at_unix {
                    Some(at) if at <= as_of_unix && as_of_unix - at <= inventory_max_age_secs => {}
                    Some(_) => unresolved.push("inventory is stale for this plan date".into()),
                    None => unresolved.push("current inventory evidence is missing".into()),
                }
                match &candidate.consumer_closure {
                    Some(scan)
                        if scan.scanned_at_unix <= as_of_unix
                            && !scan.observed_root_generation.is_empty()
                            && scan.observed_root_generation == scan.current_root_generation
                            && scan.active_consumers.is_empty()
                            && scan.unresolved.is_empty() => {}
                    Some(scan) => {
                        if scan.scanned_at_unix > as_of_unix {
                            unresolved.push("consumer scan postdates plan".into());
                        }
                        if scan.observed_root_generation.is_empty()
                            || scan.observed_root_generation != scan.current_root_generation
                        {
                            unresolved
                                .push("consumer root generation changed or is unknown".into());
                        }
                        unresolved.extend(
                            scan.active_consumers
                                .iter()
                                .map(|c| format!("active consumer: {c}")),
                        );
                        unresolved
                            .extend(scan.unresolved.iter().map(|g| format!("consumer gap: {g}")));
                    }
                    None => unresolved.push("consumer and recovery closure is missing".into()),
                }
                if let Some((kind, object)) =
                    replacement_for(&candidate, &policy, &current_objects, &keeper_objects)?
                {
                    replacement = Some(object);
                    if unresolved.is_empty() {
                        recommendation = kind;
                        reason = match kind {
                            Recommendation::ReviewDuplicate => "exact duplicate has verified bytes and tested recovery; owner review required".into(),
                            Recommendation::ReviewPack => "pack manifest and restore are verified; owner review required".into(),
                            Recommendation::ReviewSupersession => "qualified champion dominance passed margin and grace; score rows and recovery remain; owner review required".into(),
                            _ => unreachable!(),
                        };
                    }
                } else {
                    unresolved.push(replacement_gap(
                        &candidate,
                        &policy,
                        &current_objects,
                        &keeper_objects,
                    ));
                }
                if !unresolved.is_empty() {
                    reason = "no change proposed because current proof is incomplete".into();
                }
            }
            let replacement_bytes = match recommendation {
                Recommendation::ReviewDuplicate | Recommendation::ReviewSupersession => 0,
                Recommendation::ReviewPack => {
                    replacement.as_ref().map_or(current_bytes, |r| r.bytes)
                }
                _ => current_bytes,
            };
            let hypothetical_bytes_after_review = replacement_bytes;
            items.push(PlanItem {
                source: candidate.sources,
                replacement,
                recommendation,
                reason,
                unresolved,
                values,
                current_bytes,
                hypothetical_bytes_after_review,
            });
        }
        items.sort_by(|a, b| {
            fn priority(item: &PlanItem) -> u8 {
                match item.recommendation {
                    Recommendation::ReviewDuplicate => 0,
                    Recommendation::ReviewSupersession => 1,
                    Recommendation::ReviewPack => 2,
                    Recommendation::Hold => 3,
                    Recommendation::KeepT0 => 4,
                    Recommendation::KeepScoreRows => 5,
                    Recommendation::Keep => 6,
                }
            }
            let max_value = |item: &PlanItem| {
                item.values
                    .iter()
                    .filter_map(|v| v.marginal_value)
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            let max_surprise = |item: &PlanItem| {
                item.values
                    .iter()
                    .filter_map(|v| v.surprise)
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            priority(a)
                .cmp(&priority(b))
                .then_with(|| max_value(b).total_cmp(&max_value(a)))
                .then_with(|| max_surprise(b).total_cmp(&max_surprise(a)))
                .then_with(|| a.source.cmp(&b.source))
        });
        let current_objects = items.iter().try_fold(0u64, |n, item| {
            n.checked_add(item.source.len() as u64)
                .ok_or_else(|| anyhow::anyhow!("object total overflow"))
        })?;
        let current_bytes = items.iter().try_fold(0u64, |n, item| {
            n.checked_add(item.current_bytes)
                .ok_or_else(|| anyhow::anyhow!("byte total overflow"))
        })?;
        let hypothetical_objects_after_review = items.iter().try_fold(0u64, |n, item| {
            let count = match item.recommendation {
                Recommendation::ReviewDuplicate | Recommendation::ReviewSupersession => 0,
                Recommendation::ReviewPack => 1,
                _ => item.source.len() as u64,
            };
            n.checked_add(count)
                .ok_or_else(|| anyhow::anyhow!("object total overflow"))
        })?;
        let hypothetical_bytes_after_review = items.iter().try_fold(0u64, |n, item| {
            n.checked_add(item.hypothetical_bytes_after_review)
                .ok_or_else(|| anyhow::anyhow!("byte total overflow"))
        })?;
        let hypothetical_bytes_avoided =
            current_bytes.saturating_sub(hypothetical_bytes_after_review);
        let mut plan = Self {
            schema: RETENTION_PLAN_V2_SCHEMA.into(),
            as_of_unix,
            inventory_max_age_secs,
            source_inventory_sha256: source_inventory_sha256.into(),
            policy,
            items,
            current_objects,
            current_bytes,
            hypothetical_objects_after_review,
            hypothetical_bytes_after_review,
            hypothetical_bytes_avoided,
            net_bytes_delta: current_bytes as i128 - hypothetical_bytes_after_review as i128,
            plan_sha256: String::new(),
        };
        let bytes = serde_json::to_vec(&plan)?;
        plan.plan_sha256 = sha256_hex(&bytes);
        Ok(plan)
    }
}
