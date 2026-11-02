use super::*;

/// The policy value for a family/key, as it should appear in a rendered spec.
fn policy_value(family: Family, key: &str) -> Option<String> {
    let p = spec_policy::policy(family);
    Some(match key {
        "gpus_per_task" => p.gpus_per_task.to_string(),
        "cpus_reserved" => p.cpus_reserved.to_string(),
        "omp_threads" => p.omp_threads.to_string(),
        "ram_mb" => p.ram_mb.to_string(),
        "gang_size" => p.gang_size.to_string(),
        "placement" => p.placement.to_owned(),
        _ => return None,
    })
}

/// Free RAM of the smallest GPU node, MB. Measured 2026-07-28 23:1xZ via
/// `scq nodes`: xbabe1 43G, xbabe2 45G, xbabe3 77G.
pub const SMALLEST_GPU_NODE_FREE_MB: u32 = 43 * 1024;

/// Run every drift check. An empty result means every renderer agrees with every
/// other renderer AND with `spec_policy.rs`.
pub fn audit(sites: &[RenderSite]) -> Vec<Finding> {
    let mut findings = vec![];

    // 1. Cross-renderer disagreement. The headline check.
    for family in Family::all() {
        let members: Vec<&RenderSite> = sites.iter().filter(|s| s.family == Some(family)).collect();
        if members.len() < 2 {
            continue;
        }
        for key in SHARED_KEYS {
            let mut seen: Vec<(String, String)> = vec![];
            for m in &members {
                if let Some(v) = m.params.get(*key) {
                    seen.push((m.location(), v.clone()));
                }
            }
            let distinct: std::collections::BTreeSet<&String> =
                seen.iter().map(|(_, v)| v).collect();
            if distinct.len() > 1 {
                findings.push(Finding::Disagreement {
                    family,
                    key: (*key).to_owned(),
                    values: seen,
                });
            }
        }
    }

    // 2. Every renderer must match the declared policy.
    for s in sites {
        let Some(family) = s.family else { continue };
        for key in SHARED_KEYS {
            let (Some(actual), Some(expected)) = (s.params.get(*key), policy_value(family, key))
            else {
                continue;
            };
            if *actual != expected {
                findings.push(Finding::PolicyMismatch {
                    site: s.location(),
                    family,
                    key: (*key).to_owned(),
                    expected,
                    actual: actual.clone(),
                });
            }
        }
    }

    // 3. Forms scq silently drops or rejects.
    for s in sites {
        if s.dep_form == DepForm::HostnameNoOp {
            findings.push(Finding::HostnameNoOp { site: s.location() });
        }
        if s.emits_resource_node {
            findings.push(Finding::ResourceNodePin { site: s.location() });
        }
        if s.emits_ram_gb && s.params.contains_key("ram_mb") {
            findings.push(Finding::RamUnitConflict { site: s.location() });
        }
        if s.probes_etc_hostname {
            findings.push(Finding::HostnamePinnedProbe { site: s.location() });
        }
    }

    // 4. Reservations must bracket the measured peak.
    for family in Family::all() {
        let p = spec_policy::policy(family);
        if let (Some(peak), Some(margin)) = (p.measured_peak_mb, p.margin()) {
            if !(spec_policy::MIN_MARGIN..=spec_policy::MAX_MARGIN).contains(&margin) {
                findings.push(Finding::MarginOutOfBounds {
                    family,
                    ram_mb: p.ram_mb,
                    peak_mb: peak,
                    margin,
                });
            }
        }
        if p.gpus_per_task > 0 && p.placement_demand_mb() > SMALLEST_GPU_NODE_FREE_MB {
            findings.push(Finding::Unplaceable {
                family,
                demand_mb: p.placement_demand_mb(),
                smallest_node_mb: SMALLEST_GPU_NODE_FREE_MB,
            });
        }
    }

    // 5. Registry completeness, both directions.
    for s in sites {
        if registry_entry(&s.file, &s.function).is_none() {
            findings.push(Finding::UnregisteredSite { site: s.location() });
        }
    }
    for (file, function, _) in REGISTRY {
        let matched = sites
            .iter()
            .any(|s| s.file == *file && s.function == *function);
        if !matched {
            findings.push(Finding::StaleRegistryEntry {
                file: (*file).to_owned(),
                function: (*function).to_owned(),
            });
        }
    }

    findings
}
