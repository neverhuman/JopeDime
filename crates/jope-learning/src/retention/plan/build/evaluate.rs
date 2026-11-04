use super::state::{Evaluated, Loaded};
use super::*;

pub(crate) fn evaluate(cfg: &Config, loaded: Loaded) -> Result<Evaluated> {
    let as_of_unix = loaded.as_of_unix;
    let policy = &loaded.policy;
    let ledger = &loaded.ledger;
    let lookup = &loaded.lookup;
    let agg = &loaded.agg;
    let refs = &loaded.refs;
    let refs_by_group = &loaded.refs_by_group;
    let sources_by_group = &loaded.sources_by_group;
    let pins = &loaded.pins;
    let lifecycles = match &cfg.lifecycle_dir {
        Some(d) => Lifecycles::load_dir(d)?,
        None => Lifecycles::default(),
    };
    let mut classes = assign_classes(ledger, policy);

    // S3 Inventory: object-level facts per group, pin existence, E1 overlap.
    let inv: Option<Inventory> = if cfg.inventories.is_empty() {
        None
    } else {
        let mut reports = Vec::new();
        for spec in &cfg.inventories {
            reports.push(inventory::load_report(
                &spec.path,
                spec.expect_prefix.as_deref(),
            )?);
        }
        let pinned: HashSet<(String, String)> = pins
            .iter()
            .map(|(_, b, k, _)| (b.clone(), k.clone()))
            .collect();
        let e1_pairs: Vec<E1Pair> = policy
            .e1_candidates
            .iter()
            .filter_map(|c| {
                let rest = c.counterpart.strip_prefix("s3://")?;
                let (cb, cp) = rest.split_once('/').unwrap_or((rest, ""));
                Some((
                    c.bucket.clone(),
                    c.prefix.clone(),
                    cb.to_string(),
                    cp.to_string(),
                ))
            })
            .collect();
        Some(inventory::aggregate(
            reports,
            lookup,
            ledger.groups.len(),
            &pinned,
            &e1_pairs,
        )?)
    };
    let fresh = |gi: usize| -> bool {
        let g = &ledger.groups[gi];
        inv.as_ref()
            .and_then(|i| i.per_group[gi].as_ref())
            .is_some_and(|x| x.current_objects == g.objects && x.current_bytes == g.bytes)
    };
    let conflict = |gi: usize| -> bool {
        let g = &ledger.groups[gi];
        inv.as_ref().is_some_and(|i| {
            i.per_group[gi]
                .as_ref()
                .zip(i.buckets.get(&g.bucket))
                .is_some_and(|(x, b)| {
                    x.conflicts_with_ledger(
                        g.objects,
                        g.bytes,
                        g.max_last_modified.as_deref(),
                        &b.report_created_at,
                    )
                })
        })
    };
    if let Some(i) = &inv {
        for (gi, x) in i.per_group.iter().enumerate() {
            let Some(x) = x else { continue };
            if let Some((class, _)) = x.class_bytes.iter().max_by_key(|(_, b)| **b) {
                classes[gi] = (
                    class.clone(),
                    if fresh(gi) {
                        "inventory"
                    } else if conflict(gi) {
                        "inventory-ledger-conflict"
                    } else {
                        "inventory-changed-since-report"
                    },
                );
            }
        }
    }

    // Decide every group.
    let empty = BTreeMap::new();
    let mut rows = Vec::with_capacity(ledger.groups.len());
    for (gi, g) in ledger.groups.iter().enumerate() {
        let gidx = agg.per_group.get(&gi);
        let mut ctx_roles: BTreeMap<Role, u64> = BTreeMap::new();
        let mut prog_roles: BTreeMap<Role, u64> = BTreeMap::new();
        if let Some(i) = gidx {
            for c in &i.contexts {
                for role in refs.contexts.get(c).into_iter().flatten() {
                    *ctx_roles.entry(*role).or_default() += 1;
                }
            }
            for p in &i.programs {
                for role in refs.programs.get(p).into_iter().flatten() {
                    *prog_roles.entry(*role).or_default() += 1;
                }
            }
        }
        let family_scored = gidx.and_then(|i| {
            i.families
                .iter()
                .map(|f| agg.scored_by_family.get(f).copied().unwrap_or(0))
                .max()
        });
        let facts = Facts {
            group: g,
            storage_class: classes[gi].0.clone(),
            refs_by_role: refs_by_group
                .get(&gi)
                .cloned()
                .unwrap_or_else(|| empty.clone()),
            ctx_roles,
            prog_roles,
            index: gidx,
            family_scored_rows: family_scored,
            coverage: coverage(&lifecycles, &g.bucket, &g.prefix, g.direct),
            as_of_unix,
            inv: inv.as_ref().and_then(|i| i.per_group[gi].as_ref()),
            inv_fresh: fresh(gi),
            inv_conflict: conflict(gi),
            e1: inv.as_ref().and_then(|i| {
                let c = policy.e1_candidate_for(&g.bucket, &g.prefix)?;
                i.e1.iter()
                    .find(|o| o.bucket == g.bucket && o.prefix == c.prefix)
            }),
        };
        let d: Decision = decide(&facts, policy)?;
        if d.action == Action::Expire {
            bail!(
                "internal: EXPIRE reached for {} without object-level evidence",
                g.id()
            );
        }
        if super::super::super::classify::hard_t0(&g.bucket, &g.prefix).is_some()
            && d.action != Action::KeepT0
        {
            bail!(
                "internal: hard T0 group {} got {}",
                g.id(),
                d.action.as_str()
            );
        }
        let sources = sources_by_group
            .get(&gi)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();
        let mut r = row(g, &facts, d, classes[gi].1, sources);
        r.inventory = facts.inv.map(|x| RowInventory {
            report_created_at: inv
                .as_ref()
                .and_then(|i| i.buckets.get(&g.bucket))
                .map(|b| b.report_created_at.clone())
                .unwrap_or_default(),
            unchanged_since_report: facts.inv_fresh,
            ledger_conflict: facts.inv_conflict,
            facts: x.clone(),
        });
        rows.push(r);
    }

    Ok(Evaluated {
        loaded,
        lifecycles,
        inv,
        rows,
    })
}
