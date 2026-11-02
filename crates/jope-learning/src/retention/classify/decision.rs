use super::*;

/// Performs the decide operation and returns an error if it cannot be completed.
pub fn decide(f: &Facts, p: &Policy) -> Result<Decision> {
    let g = f.group;
    let rule = p.rule_for(&g.bucket, &g.prefix);
    let kind = rule.map(|r| r.kind);
    let family = rule.and_then(|r| r.family.clone());
    let rule_src = rule.map_or_else(
        || "default".to_string(),
        |r| format!("policy s3://{}/{}", r.bucket, r.prefix),
    );
    let mean = g.bytes.checked_div(g.objects).unwrap_or(0);
    let exact = exact_max(f);
    let live = is_live(
        exact.or(g.max_last_modified.as_deref()),
        f.as_of_unix,
        p.open_run_window_hours,
    );

    let t0_refs: BTreeMap<Role, u64> = f
        .refs_by_role
        .iter()
        .filter(|(r, _)| r.is_t0())
        .map(|(r, n)| (*r, *n))
        .collect();
    let t0_ctx: BTreeMap<Role, u64> = f
        .ctx_roles
        .iter()
        .filter(|(r, _)| r.is_t0())
        .map(|(r, n)| (*r, *n))
        .collect();
    let t0_prog: BTreeMap<Role, u64> = f
        .prog_roles
        .iter()
        .filter(|(r, _)| r.is_t0())
        .map(|(r, n)| (*r, *n))
        .collect();
    let qualified = f.index.map_or(0, |i| i.qualified_rows);
    let t0 = hard_t0(&g.bucket, &g.prefix)
        .or_else(|| {
            (!t0_refs.is_empty()).then(|| {
                format!(
                    "referenced by key from T0 sources ({})",
                    roles_str(&t0_refs)
                )
            })
        })
        .or_else(|| {
            (!t0_ctx.is_empty())
                .then(|| format!("holds GP rows of T0 contexts ({})", roles_str(&t0_ctx)))
        })
        .or_else(|| {
            (!t0_prog.is_empty()).then(|| {
                format!(
                    "holds programs pinned by T0 sources ({})",
                    roles_str(&t0_prog)
                )
            })
        })
        .or_else(|| (qualified > 0).then(|| format!("holds {qualified} qualified rows")))
        .or_else(|| {
            p.soft_t0_for(&g.bucket, &g.prefix)
                .map(|n| format!("policy soft T0: {}", n.why))
        });
    let tv = tv_for(f, kind, t0.is_some(), p);
    let active_key_refs: BTreeMap<Role, u64> = f
        .refs_by_role
        .iter()
        .filter(|(r, _)| r.is_active())
        .map(|(r, n)| (*r, *n))
        .collect();

    let (action, why): (Action, String) = if let Some(t) = &t0 {
        (
            Action::KeepT0,
            format!("{t}; never delete, no transition proposed"),
        )
    } else if !p.in_scope(&g.bucket) {
        (
            Action::Hold,
            "out of JopeDime retention scope (legacy/other system); its own lifecycle is untouched"
                .into(),
        )
    } else if !active_key_refs.is_empty() {
        (
            Action::KeepHot,
            format!(
                "referenced by key/version from active rosters/seeds ({}); packing or archiving would break those references",
                roles_str(&active_key_refs)
            ),
        )
    } else if let Some(i) = f.inv.filter(|_| f.inv_conflict) {
        (
            Action::Hold,
            format!(
                "the ledger ({} obj, {}, last write {}) and the S3 Inventory report ({} current obj, {}, {} latest delete markers) disagree with no write after the report: one input is wrong, so no change until they agree",
                g.objects,
                human_bytes(g.bytes),
                g.max_last_modified.as_deref().unwrap_or("?"),
                i.current_objects,
                human_bytes(i.current_bytes),
                i.delete_markers_latest
            ),
        )
    } else if live && g.direct && exact.is_none() {
        // A direct group only knows its parent prefix's date range unless an unchanged
        // inventory listed its objects.
        (
            Action::Hold,
            format!(
                "direct objects under a prefix written at {}; their own dates are unknown {}",
                g.max_last_modified.as_deref().unwrap_or("?"),
                if f.inv.is_some() {
                    "(they changed after the S3 Inventory report)"
                } else {
                    "until S3 Inventory"
                }
            ),
        )
    } else if live {
        (
            Action::KeepHot,
            format!(
                "writer active (last write {} within {}h of as-of); classify again after the run closes",
                exact.or(g.max_last_modified.as_deref()).unwrap_or("?"),
                p.open_run_window_hours
            ),
        )
    } else {
        match kind {
            Some(Kind::Encoding) => (Action::KeepHot, "encodings: kNN-teacher / training input, read in place".into()),
            Some(Kind::DatasetPayload) => (Action::KeepHot, "dataset/training payload, read in place".into()),
            Some(Kind::GpResult | Kind::Receipts) if mean >= p.compact_max_mean_object_bytes => (
                Action::KeepHot,
                format!("objects already large (mean {}); packing gains nothing", human_bytes(mean)),
            ),
            Some(Kind::GpResult | Kind::Receipts) => (
                Action::Compact,
                format!(
                    "closed run, mean object {}: pack into >= {} zstd shards + index, keep HWM receipt/program/score loose",
                    human_bytes(mean),
                    human_bytes(p.pack_bytes)
                ),
            ),
            Some(Kind::GpResultUnconsumed | Kind::Telemetry) => (
                Action::DistillCold,
                "no recorded consumer; distill every row (incl. failures) to gp_distilled/v1, then pack raw bytes to Deep Archive; nothing deleted".into(),
            ),
            Some(Kind::Archive) if mean < p.cold_min_mean_object_bytes => (
                Action::DistillCold,
                format!("archive objects small (mean {}): pack before Deep Archive", human_bytes(mean)),
            ),
            Some(Kind::Archive) => (Action::Cold, "opaque archive: Deep Archive (180-day minimum, 12-48 h restore)".into()),
            Some(Kind::Checkpoint) => (
                Action::Hold,
                "checkpoints: no checkpoint->L6/authority linkage yet; frozen v1/promoted must stay T0 (§7.3), so no transition".into(),
            ),
            Some(Kind::Report) => (Action::Hold, "research/report bytes: T3 only after the report is sealed (strategy §3)".into()),
            Some(Kind::Replica) => (Action::Hold, "replica: keep/drop is owner decision 6".into()),
            Some(Kind::Custody) => (Action::Hold, "preservation custody: dev/prod duplicate check is owner decision 6".into()),
            Some(Kind::Review) => (Action::Hold, "contents not classified; review before any action".into()),
            None => (Action::Hold, "no policy rule matched; default HOLD + review (§7.3 rule 6)".into()),
        }
    };
    let e = e_candidates(f, kind, p);
    let cost = cost(f, action, p)?;

    let mut reason = format!(
        "{}: {} [{}{}] {} obj, {}, mean {}; {}",
        action.as_str(),
        g.id(),
        kind.map_or("unclassified", Kind::as_str),
        family
            .as_deref()
            .map(|s| format!("/{s}"))
            .unwrap_or_default(),
        g.objects,
        human_bytes(g.bytes),
        human_bytes(mean),
        why
    );
    reason.push_str(&format!(
        "; TV={} ({})",
        tv.value,
        if tv.uses.is_empty() {
            "none".into()
        } else {
            tv.uses.join(",")
        }
    ));
    reason.push_str(&format!("; refs {}", roles_str(&f.refs_by_role)));
    if let Some(i) = f.index {
        reason.push_str(&format!(
            "; index {} rows ({} scored, {} qualified)",
            i.rows, i.scored_rows, i.qualified_rows
        ));
    }
    reason.push_str("; champion: no qualified champion, so no supersession");
    if !f.coverage.full.is_empty() {
        let rules: Vec<&str> = f
            .coverage
            .full
            .iter()
            .map(|s| s.split(':').next().unwrap_or(""))
            .collect();
        reason.push_str(&format!("; lifecycle {}", rules.join(",")));
    }
    if let Some(i) = f.inv {
        reason.push_str(&format!(
            "; inventory {} ({} noncurrent versions{})",
            if f.inv_fresh {
                "unchanged since report"
            } else if f.inv_conflict {
                "disagrees with the ledger without a later write"
            } else {
                "changed since report"
            },
            i.noncurrent_versions,
            exact
                .filter(|_| g.direct)
                .map(|m| format!(", direct objects last written {m}"))
                .unwrap_or_default()
        ));
        if !e.is_empty() {
            reason.push_str("; no E-rule proven (E1 needs GET+sha256; E2-E4 need gp_distilled/v1)");
        }
    } else if !e.is_empty() {
        reason.push_str("; E-rules not provable at prefix level");
    }
    Ok(Decision {
        action,
        kind: kind.map(Kind::as_str),
        family,
        rule: rule_src,
        t0,
        live,
        tv,
        e_candidates: e,
        cost,
        reason,
    })
}
