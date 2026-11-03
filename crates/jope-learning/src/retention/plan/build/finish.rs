use super::super::diff_previous;
use super::state::{Analysis, Evaluated};
use super::*;

pub(crate) fn finish(cfg: &Config, evaluated: Evaluated, analysis: Analysis) -> Result<Built> {
    let Evaluated {
        loaded,
        lifecycles,
        inv,
        rows,
        ..
    } = evaluated;
    let super::state::Loaded {
        policy,
        policy_rec,
        ledger,
        agg,
        inputs,
        refs,
        ref_counts,
        unmatched_in_scope,
        unmatched_sample,
        other_buckets,
        bucketless_resolved,
        bucketless_unresolved,
        ..
    } = loaded;
    let Analysis {
        inventory_summary,
        findings,
        plan_jsonl,
        plan_sha,
        hazards,
        notes,
    } = analysis;
    let mut totals = Totals::default();
    let mut by_action: BTreeMap<&'static str, Totals> = Action::ALL
        .iter()
        .map(|a| (a.as_str(), Totals::default()))
        .collect();
    let mut by_tier: BTreeMap<&'static str, Totals> = BTreeMap::new();
    for r in &rows {
        totals.add(r);
        by_action.entry(r.action.as_str()).or_default().add(r);
        by_tier.entry(r.tier).or_default().add(r);
    }
    totals.finish();
    by_action.values_mut().for_each(Totals::finish);
    by_tier.values_mut().for_each(Totals::finish);
    ensure!(
        totals.objects == ledger.total_objects && totals.bytes == ledger.total_bytes,
        "plan totals do not reconcile with the ledger"
    );
    let to_notes = |m: BTreeMap<(String, String), (u64, f64)>| -> Vec<Note> {
        m.into_iter()
            .map(|((kind, detail), (n, gb))| Note {
                kind,
                detail,
                groups: n,
                gb: round4(gb),
            })
            .collect()
    };

    let lock = InputsLock {
        schema: INPUTS_LOCK_SCHEMA,
        as_of: cfg.as_of.clone(),
        tool: crate::TOOL_VERSION.into(),
        source_commit: cfg.source_commit.clone(),
        policy: policy_rec.clone(),
        inputs,
        lifecycle_files: lifecycles
            .files
            .iter()
            .map(|(p, s, n)| InputRecord {
                role: "lifecycle".into(),
                path: p.clone(),
                bytes: *n,
                sha256: s.clone(),
                check: "parsed".into(),
            })
            .collect(),
        scans: refs.scans.clone(),
        inventory: inv
            .as_ref()
            .map(|i| {
                i.reports
                    .iter()
                    .map(|r| InventoryLock {
                        bucket: r.source_bucket.clone(),
                        report_created_at: r.created_at.clone(),
                        manifest: r.manifest.clone(),
                        files: r.files.clone(),
                        read: i.buckets[&r.source_bucket].read.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default(),
    };
    let mut lock_json = serde_json::to_vec_pretty(&lock)?;
    lock_json.push(b'\n');

    let diff = match &cfg.previous_plan {
        Some(p) => Some(diff_previous(p, &rows)?),
        None => None,
    };
    let mut ref_totals: BTreeMap<String, u64> = BTreeMap::new();
    for r in &refs.objects {
        *ref_totals.entry(r.role.as_str().to_string()).or_default() += 1;
    }
    ref_totals.insert("contexts".into(), refs.contexts.len() as u64);
    ref_totals.insert("programs".into(), refs.programs.len() as u64);
    let summary = Summary {
        schema: PLAN_SUMMARY_SCHEMA,
        as_of: cfg.as_of.clone(),
        mode: "DRY-RUN: read-only; no S3 calls; this tool has no apply, tag, transition or delete path",
        granularity: match &inv {
            None => format!(
                "prefix groups from the S3 object ledger generated {} (S3 Inventory not yet available)",
                ledger.generated_at
            ),
            Some(i) => format!(
                "prefix groups from the S3 object ledger generated {}, with object-level facts from S3 Inventory reports ({}) for {} buckets",
                ledger.generated_at,
                i.reports
                    .iter()
                    .map(|r| r.created_at.as_str())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .join(", "),
                i.reports.len()
            ),
        },
        plan_file: "retention-plan.jsonl",
        plan_sha256: plan_sha,
        rows: rows.len() as u64,
        inputs_lock_sha256: sha256_hex(&lock_json),
        policy_sha256: policy_rec.sha256.clone(),
        prices: format!("{} (estimates; billed S3 is currently credited)", policy.prices.label),
        totals,
        by_action,
        by_tier,
        expire: "0 groups: EXPIRE needs TV=false, an E1-E5 rule proven with object-level evidence (inventory + GET+sha256), and an agreeing independent reference scan; none is available at prefix level".into(),
        e_rule_status: BTreeMap::from([
            ("E1", match &inv {
                None => "not evaluable: needs S3 Inventory key/size join plus GET+sha256 of both copies (etag is not enough)".to_string(),
                Some(i) => format!(
                    "candidates only: {}; proof needs GET+sha256 of both copies (a separate, approved read job), so nothing expires",
                    if i.e1.is_empty() {
                        "no E1 candidate pair has both buckets inventoried".to_string()
                    } else {
                        i.e1.iter()
                            .map(|o| format!(
                                "s3://{}/{} vs s3://{}/{}: {} of {} keys same key+size+ETag ({:.1} GB)",
                                o.bucket, o.prefix, o.counterpart_bucket, o.counterpart_prefix,
                                o.same_key_size_etag, o.objects, o.same_key_size_etag_bytes as f64 / 1e9
                            ))
                            .collect::<Vec<_>>()
                            .join("; ")
                    }
                ),
            }),
            ("E2", "not evaluable: no checkpoint->L6 curve table (gp_distilled/v1 not built)".into()),
            ("E3", "not evaluable: no distilled table hash in a T0 manifest".into()),
            ("E4", "not evaluable: needs program_sha subset proof over full bodies".into()),
            ("E5", match &inv {
                None => "delegated to the applied lifecycle rules p0-noncurrent-expire-* (ledger lists current versions only)".to_string(),
                Some(_) => "delegated to the applied lifecycle rules p0-noncurrent-expire-*; noncurrent versions per group are listed from S3 Inventory (summary.inventory, row.inventory); T0 groups' noncurrent bytes are in notes t0-noncurrent-versions (D2)".to_string(),
            }),
        ]),
        hazards: to_notes(hazards),
        notes: to_notes(notes),
        findings,
        refs: ref_totals,
        refs_unmatched_in_scope: unmatched_in_scope,
        refs_unmatched_sample: unmatched_sample,
        refs_other_buckets: other_buckets,
        refs_bucketless_resolved: bucketless_resolved,
        refs_bucketless_unresolved: bucketless_unresolved,
        prefix_mentions: refs.prefix_mentions,
        index_rows: BTreeMap::from([
            ("total", agg.rows_total),
            ("s3", agg.rows_s3),
            ("local", agg.rows_local),
            ("unmatched", agg.rows_unmatched.len() as u64),
            ("qualified", agg.qualified_total),
        ]),
        index_rows_unmatched_sample: agg.rows_unmatched.iter().take(20).cloned().collect(),
        diff_vs_previous: diff,
        inventory: inventory_summary,
    };
    let report_md = super::super::report(&summary, &rows, &lock, &ref_counts);
    Ok(Built {
        rows,
        plan_jsonl,
        summary,
        lock,
        lock_json,
        report_md,
    })
}
