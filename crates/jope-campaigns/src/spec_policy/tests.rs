use super::*;

/// Reservations must bracket the measured peak. This is the check that fails
/// on `test_exact_train.py`'s 8192 (below the 9,174 MB trainer peak) and on
/// the 49152 propose reservation (46.6x peak).
///
/// A family listed in `UNDER_MARGIN_ACCEPTED` is exempt from the lower bound
/// ONLY here, where the exemption is reviewable. `audit()` still raises
/// `MarginOutOfBounds` for it, so `jope_spec_drift` keeps reporting it and the
/// risk stays visible to ops.
#[test]
fn reservations_are_within_measured_margin_bounds() {
    for f in Family::all() {
        let p = policy(f);
        let Some(margin) = p.margin() else { continue };
        if under_margin_accepted(f).is_some() {
            assert!(
                margin < MIN_MARGIN,
                "{}: listed in UNDER_MARGIN_ACCEPTED but its margin {margin:.2}x now \
                     meets MIN_MARGIN {MIN_MARGIN}x -- remove the exemption",
                f.as_str()
            );
            continue;
        }
        assert!(
            margin >= MIN_MARGIN,
            "{}: ram_mb {} is only {margin:.2}x the measured peak {} MB (min {MIN_MARGIN}x)",
            f.as_str(),
            p.ram_mb,
            p.measured_peak_mb.unwrap()
        );
        assert!(
            margin <= MAX_MARGIN,
            "{}: ram_mb {} is {margin:.2}x the measured peak {} MB (max {MAX_MARGIN}x) \
                 -- an over-reservation this large stops the job being placeable",
            f.as_str(),
            p.ram_mb,
            p.measured_peak_mb.unwrap()
        );
    }
}

/// Every GPU family must fit the SMALLEST GPU node's free RAM, or it silently
/// pins to one host. Measured 2026-07-28: xbabe1 44 GB free is the tightest.
#[test]
fn gpu_families_fit_the_smallest_gpu_node() {
    use crate::spec_drift::SMALLEST_GPU_NODE_FREE_MB;
    for f in Family::all() {
        let p = policy(f);
        if p.gpus_per_task == 0 {
            continue;
        }
        assert!(
            p.placement_demand_mb() <= SMALLEST_GPU_NODE_FREE_MB,
            "{}: demands {} MB free (ram_mb {} + {SCQ_SAFETY_FLOOR_MB} floor) but the \
                 smallest GPU node has {SMALLEST_GPU_NODE_FREE_MB} MB -- it would pin to one host",
            f.as_str(),
            p.placement_demand_mb(),
            p.ram_mb
        );
    }
}

/// scq requires cpus_reserved > 0 whenever gpus_per_task > 0
/// (scq-core/src/spec.rs:2605-2608), else submit is rejected.
#[test]
fn gpu_jobs_reserve_cpus() {
    for f in Family::all() {
        let p = policy(f);
        if p.gpus_per_task > 0 {
            assert!(
                p.cpus_reserved > 0,
                "{}: gpus>0 requires cpus_reserved>0",
                f.as_str()
            );
        }
    }
}

/// OMP_NUM_THREADS is set from omp_threads; a mismatch oversubscribes the node.
#[test]
fn omp_threads_match_cpu_reservation() {
    for f in Family::all() {
        let p = policy(f);
        assert_eq!(
            p.omp_threads,
            p.cpus_reserved,
            "{}: omp_threads {} != cpus_reserved {}",
            f.as_str(),
            p.omp_threads,
            p.cpus_reserved
        );
    }
}

/// Probes must test a CAPABILITY, never a hostname. A hostname probe would
/// re-create the hard-coding that stranded 8,007 datasets on one node.
#[test]
fn probes_are_capability_based_not_hostname_based() {
    for f in Family::all() {
        for s in policy(f).smoke {
            assert!(
                !s.args.iter().any(|a| a.render().contains("/etc/hostname")),
                "{}: probe '{}' reads /etc/hostname -- probe a capability instead",
                f.as_str(),
                s.name
            );
            assert!(
                s.command == "/usr/bin/test",
                "{}: probe '{}' uses {} -- capability probes are /usr/bin/test -f",
                f.as_str(),
                s.name,
                s.command
            );
        }
    }
}

/// The rendered placement block must use the enforced form.
#[test]
fn placement_block_uses_the_enforced_dependency_form() {
    let block = TRAINER.render_placement_block();
    assert!(
        block.contains("[[dependencies.smoke]]"),
        "must use the smoke form"
    );
    assert!(
        !block.contains("hostname"),
        "hostname is a serde no-op in scq"
    );
    assert!(block.contains("timeout_secs"));
}

#[test]
fn resource_block_never_emits_controller_only_node_key() {
    for f in Family::all() {
        let block = policy(f).render_resource_block();
        assert!(
            !block.contains("\nnode ") && !block.contains("\nnode="),
            "{}: [resource] node is rejected by scq as controller-only",
            f.as_str()
        );
    }
}

#[test]
fn cohort_schema_is_v2_and_excludes_sealed() {
    assert_eq!(COHORT_SCHEMA, "JopeMicroCohort.v2");
    const { assert!(EXCLUDE_SEALED_FAMILIES) };
}

#[test]
fn policy_json_is_parseable_and_complete() {
    let j = policy_json();
    for f in Family::all() {
        assert!(
            j.contains(&format!("\"{}\"", f.as_str())),
            "missing {}",
            f.as_str()
        );
    }
    assert!(j.contains("JopeSpecPolicy.v2"));
    // balanced braces == well-formed enough to catch a truncated writer
    let opens = j.chars().filter(|c| *c == '{').count();
    let closes = j.chars().filter(|c| *c == '}').count();
    assert_eq!(opens, closes, "unbalanced JSON braces");
}
