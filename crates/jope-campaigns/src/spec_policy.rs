//! WS-2: the SINGLE SOURCE OF TRUTH for scq job specs.
//!
//! Every stall on 2026-07-28 was one bug class: the same job family was
//! rendered by several independent code paths that had silently diverged.
//! Five `ram_mb` values existed for jobs whose measured peak was ~1 GB; three
//! renderers used a `[dependencies] hostname` key that scq silently DROPS.
//!
//! This module owns the four things that drifted:
//!   1. resource reservations (derived from MEASURED peak x a stated margin),
//!   2. placement constraints (capability smoke probes -- never hostnames),
//!   3. dependency FORM (`[[dependencies.smoke]]`, the only enforced one),
//!   4. schema versions (cohort schema, sealed-family exclusion).
//!
//! It deliberately does NOT own job ids, args, inputs or outputs: those legitimately
//! differ per call site, and rewriting them on a live cluster is unnecessary risk.
//!
//! `jope_spec_drift` (src/bin/jope_spec_drift.rs) is the enforcement half: it fails
//! when any renderer disagrees with this file, and would have caught all four of
//! today's stalls before they shipped.
//!
//! ## Why `hostname` is forbidden (measured, scq source)
//!
//! `scq-core/src/deps.rs:16-31` defines `DependencySpec` with exactly seven fields:
//! `binaries, python, python_imports, python_packages, shared_libraries,
//! system_packages, smoke`. There is NO `hostname`, and no `deny_unknown_fields`,
//! so serde silently discards it. `DependencySpec::is_empty()` (deps.rs:48-55) is
//! then true, and `prepare_dependencies` (scq-controller/src/state.rs:3037-3040)
//! runs `rec.dependency_allowed_nodes.clear()` -- placement becomes UNCONSTRAINED.
//! That is how one job black-holed xbabe3 92 times.
//!
//! `smoke` IS enforced, at placement: state.rs:3088 sets `dependency_allowed_nodes`
//! from the probe results and state.rs:2612-2613 does
//! `eligible.retain(|(node, _)| j.dependency_allowed_nodes.contains(node))`.
//!
//! `[resource] node` is rejected outright at submit
//! (`scq-core/src/spec.rs:288-294`: "node is controller-only").

use std::fmt::Write as _;

/// scq adds this much on top of `ram_mb` when deciding whether a node can host a
/// job. A 49152 MB reservation therefore demands 56 GB free, which on 2026-07-28
/// only xbabe3 ever had -- the direct cause of the propose-lane stall.
pub const SCQ_SAFETY_FLOOR_MB: u32 = 8192;

/// A reservation below the measured peak risks OOM/eviction; above the upper bound
/// it silently makes the job unplaceable. Both bounds are enforced by the drift test.
pub const MIN_MARGIN: f64 = 1.5;

/// Families whose reservation is knowingly BELOW `MIN_MARGIN` over the measured
/// peak, each with the reason it was accepted.
///
/// This is an explicit, reviewed acceptance of OOM risk -- NOT a way to silence
/// the check. `audit()` still raises `MarginOutOfBounds` for every family listed
/// here, so `jope_spec_drift` keeps reporting it. The only thing an entry
/// suppresses is the unit-test hard failure, so that a reservation change which
/// needs an operator capacity decision does not masquerade as a code defect.
pub const UNDER_MARGIN_ACCEPTED: &[(Family, &str)] = &[];

/// The recorded reason this family may sit below `MIN_MARGIN`, if any.
pub fn under_margin_accepted(family: Family) -> Option<&'static str> {
    UNDER_MARGIN_ACCEPTED
        .iter()
        .find(|(f, _)| *f == family)
        .map(|(_, why)| *why)
}
pub const MAX_MARGIN: f64 = 16.0;

/// Cohort schema. v1 is retired: a v1 planner never filtered sealed families, so
/// 1,054 of 19,951 catalog rows were unbuildable and ~99.9% of 128-draws failed.
pub const COHORT_SCHEMA: &str = "JopeMicroCohort.v2";

/// Sealed families are never eligible for cohort draws. Non-negotiable.
pub const EXCLUDE_SEALED_FAMILIES: bool = true;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Family {
    Trainer,
    Propose,
    Eval,
    Encode,
}

impl Family {
    pub fn as_str(self) -> &'static str {
        match self {
            Family::Trainer => "trainer",
            Family::Propose => "propose",
            Family::Eval => "eval",
            Family::Encode => "encode",
        }
    }

    pub fn all() -> [Family; 4] {
        [
            Family::Trainer,
            Family::Propose,
            Family::Eval,
            Family::Encode,
        ]
    }
}

/// A capability smoke probe. Probe what the job NEEDS, never which host it is on,
/// so a newly provisioned node qualifies automatically.
#[derive(Clone, Copy, Debug)]
pub struct SmokeProbe {
    pub name: &'static str,
    pub command: &'static str,
    pub args: &'static [ProbeArg],
    pub timeout_secs: u32,
}

/// One probe argument: a literal, or a path relative to the lineage-A
/// self-play tree (`crate::paths::flywheel_root()`), resolved when the spec is
/// rendered so the policy tables stay `const`.
#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeArg {
    Literal(&'static str),
    UnderFlywheelRoot(&'static str),
}

impl ProbeArg {
    pub fn render(&self) -> String {
        match self {
            Self::Literal(value) => (*value).to_owned(),
            Self::UnderFlywheelRoot(relative) => {
                jope_paths::to_string(&crate::paths::flywheel_root().join(relative))
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FamilyPolicy {
    pub family: Family,
    pub gpus_per_task: u32,
    pub cpus_reserved: u32,
    pub omp_threads: u32,
    pub ram_mb: u32,
    pub gang_size: u32,
    pub placement: &'static str,
    /// Highest RSS ever observed for this family, in MB. `None` = no telemetry
    /// retained; the drift test reports it as UNMEASURED instead of inventing a margin.
    pub measured_peak_mb: Option<u32>,
    /// Provenance of `measured_peak_mb`, so the number can be re-derived.
    pub peak_evidence: &'static str,
    pub smoke: &'static [SmokeProbe],
    pub max_runtime_hours: u32,
    pub preemptible: bool,
}

impl FamilyPolicy {
    /// Reservation / measured peak. `None` when the peak is unmeasured.
    pub fn margin(&self) -> Option<f64> {
        self.measured_peak_mb
            .filter(|p| *p > 0)
            .map(|p| f64::from(self.ram_mb) / f64::from(p))
    }

    /// Free RAM a node must have for scq to place this job.
    pub fn placement_demand_mb(&self) -> u32 {
        self.ram_mb + SCQ_SAFETY_FLOOR_MB
    }

    /// The `[resource]` block. Byte-identical for every renderer of this family.
    pub fn render_resource_block(&self) -> String {
        let mut s = String::new();
        let _ = write!(
            s,
            "[resource]\n\
             gpus_per_task = {}\n\
             cpus_reserved = {}\n\
             omp_threads = {}\n\
             ram_mb = {}\n\
             gang_size = {}\n\
             placement = \"{}\"\n",
            self.gpus_per_task,
            self.cpus_reserved,
            self.omp_threads,
            self.ram_mb,
            self.gang_size,
            self.placement,
        );
        s
    }

    /// The `[[dependencies.smoke]]` blocks. This is the ONLY placement mechanism
    /// scq enforces; `hostname` is a no-op and `[resource] node` is rejected.
    pub fn render_placement_block(&self) -> String {
        let mut s = String::new();
        for p in self.smoke {
            let args = p
                .args
                .iter()
                .map(|a| serde_json::to_string(&a.render()).expect("serialize probe argument"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = write!(
                s,
                "[[dependencies.smoke]]\n\
                 name = \"{}\"\n\
                 command = \"{}\"\n\
                 args = [{}]\n\
                 timeout_secs = {}\n",
                p.name, p.command, args, p.timeout_secs,
            );
        }
        s
    }

    pub fn render_limits_block(&self) -> String {
        format!(
            "[limits]\nmax_runtime_hours = {}\nmax_restarts = 1\n",
            self.max_runtime_hours
        )
    }
}

// ---------------------------------------------------------------------------
// Capability probes. Each asserts a FILE the job genuinely needs, so provisioning
// that file on a new node makes the node eligible with no code change.
// ---------------------------------------------------------------------------

/// The GPU-job capability probe. The immutable runtime bundle is synchronized
/// to every execution node before activation, and trainer/proposer imports use
/// its bundled `pytorch` tree.
///
/// Everything else the job needs -- the script, corpus manifest, checkpoint,
/// packed shards -- is STAGED by scq via `[[input]]`, so it must NOT be probed.
pub const PROBE_PRIME_POLICY_SRC: SmokeProbe = SmokeProbe {
    name: "runtime bundle JOPE policy present",
    command: "/usr/bin/test",
    args: &[
        ProbeArg::Literal("-f"),
        ProbeArg::UnderFlywheelRoot("runtime-bundles/current/pytorch/jope/policy/model_v3.py"),
    ],
    timeout_secs: 5,
};

/// DO NOT USE. Kept named so the reason survives.
///
/// `jope_selfplay_trainer_watchdog.rs:1448` probes the live flywheel root. That
/// path exists ONLY on xbabe2 (MEASURED 2026-07-28 23:2xZ: xbabe1 ABSENT,
/// xbabe3 ABSENT, xbabe2 PRESENT), so it is a hostname pin wearing a capability
/// probe's clothes -- it defeats the very property the smoke form exists for,
/// namely that a newly provisioned node qualifies automatically.
///
/// Worse, it is not merely restrictive: `prepare_dependencies` probes only nodes
/// that ALREADY have the requested GPU/cores/RAM free
/// (scq-controller/src/state.rs:3092-3140). When xbabe2 is busy and xbabe3 is
/// the only node with a free GPU, the probe fails there and the submit is
/// REJECTED outright rather than queued. MEASURED: 44 consecutive
/// `cycle_failed` records with
/// `error[dependency-unsatisfied]: ... xbabe3: smoke flywheel live root present:
/// smoke test exited with exit 1`.
pub const PROBE_FLYWHEEL_ROOT_XBABE2_ONLY: SmokeProbe = SmokeProbe {
    name: "flywheel live root present",
    command: "/usr/bin/test",
    args: &[
        ProbeArg::Literal("-f"),
        ProbeArg::UnderFlywheelRoot("catalog/current.json"),
    ],
    timeout_secs: 5,
};

// ---------------------------------------------------------------------------
// THE POLICY. Change a number here and every renderer changes with it.
// ---------------------------------------------------------------------------

pub const TRAINER: FamilyPolicy = FamilyPolicy {
    family: Family::Trainer,
    gpus_per_task: 1,
    cpus_reserved: 4,
    omp_threads: 4,
    // MEASURED peak 9,174 MB -> 1.79x margin.
    // NOTE scripts/test_exact_train.py:1155 reserves 8192, which is BELOW the
    // observed peak: that job can be OOM-killed by its own cgroup.
    ram_mb: 16384,
    gang_size: 1,
    placement: "single_node",
    measured_peak_mb: Some(9174),
    peak_evidence: "scq show jope-selfplay-train-001805437-c-r000024 --json: \
                    peak_ram_mb = 9174 (2026-07-28 23:1xZ)",
    smoke: &[PROBE_PRIME_POLICY_SRC],
    max_runtime_hours: 168,
    preemptible: false,
};

pub const PROPOSE: FamilyPolicy = FamilyPolicy {
    family: Family::Propose,
    gpus_per_task: 1,
    cpus_reserved: 4,
    omp_threads: 4,
    // MEASURED peak 1,055 MB across all three retained propose jobs (p50 965).
    // 12288 is the production-proven value -- the currently-RUNNING propose job
    // uses it. The 11.6x margin covers catalog growth: the context blob scales
    // with dataset count, which the operator intends to grow indefinitely.
    // 49152 was 46.6x peak and demanded 56 GB free, so it could only ever land
    // on xbabe3. That is the propose-lane stall.
    ram_mb: 12288,
    gang_size: 1,
    placement: "single_node",
    measured_peak_mb: Some(1055),
    peak_evidence: "scq show jope-selfplay-propose-fbcd5459f2c1-c4255cdc7963d --json: \
                    peak_ram_mb = 1055 (n=3 retained, p50 965, 2026-07-28 23:1xZ)",
    smoke: &[PROBE_PRIME_POLICY_SRC],
    max_runtime_hours: 24,
    preemptible: true,
};

pub const EVAL: FamilyPolicy = FamilyPolicy {
    family: Family::Eval,
    gpus_per_task: 0,
    cpus_reserved: 1,
    omp_threads: 1,
    // MEASURED 2026-07-29 00:0xZ over the FULL retained `scq --json ls` population:
    // n=2154 eval jobs (p50 32 / p95 97 / max 483) and n=284 coverage jobs
    // (p50 46 / p95 105 / max 217). Worst observed across both lanes = 483 MB.
    //
    // The earlier figure recorded here was 270 MB, taken from an n=323 sample. It
    // was not representative: widening to the full population raised the observed
    // max by 1.79x. 8 of 2154 eval jobs exceed 270 MB and 5 exceed the 341 MB
    // implied by MIN_MARGIN, so the sample had simply not seen the tail.
    //
    // 768 MB is the smallest round reservation above the 1.5x measured-peak
    // floor (483 * 1.5 = 724.5 MB). The scheduler's independent safety floor
    // remains unchanged.
    //
    // Deliberately NOT given a smoke probe -- eval jobs carry every input via
    // `[[input]]`, and a dependency block would make submits FAIL under
    // saturation instead of queueing (see PROBE_FLYWHEEL_ROOT_XBABE2_ONLY).
    ram_mb: 768,
    gang_size: 1,
    placement: "single_node",
    measured_peak_mb: Some(483),
    peak_evidence: "scq --json ls, full retained population: n=2154 eval (max 483 MB, \
                    p95 97) + n=284 coverage (max 217 MB), 2026-07-29 00:0xZ",
    smoke: &[],
    max_runtime_hours: 6,
    preemptible: true,
};

pub const ENCODE: FamilyPolicy = FamilyPolicy {
    family: Family::Encode,
    gpus_per_task: 1,
    cpus_reserved: 4,
    omp_threads: 4,
    ram_mb: 16384,
    gang_size: 1,
    placement: "single_node",
    // No encode job survives in scq's retained window (0 of 4,330 records), so
    // there is no measured peak. Reported as UNMEASURED rather than back-fitting
    // a margin to the existing number: `margin()` returns None and the drift
    // test skips the band check instead of asserting against a guess.
    measured_peak_mb: None,
    peak_evidence: "UNMEASURED: 0 jope-binexp-* records in `scq ls --json` (n=4330, \
                    2026-07-28 23:1xZ)",
    smoke: &[PROBE_PRIME_POLICY_SRC],
    max_runtime_hours: 12,
    preemptible: true,
};

pub fn policy(family: Family) -> &'static FamilyPolicy {
    match family {
        Family::Trainer => &TRAINER,
        Family::Propose => &PROPOSE,
        Family::Eval => &EVAL,
        Family::Encode => &ENCODE,
    }
}

/// Machine-readable projection consumed by the scripts/ lane
/// (`scripts/jope-spec-policy.json`), so Python renderers read these exact numbers
/// instead of keeping their own copy. The drift test asserts the checked-in JSON
/// still matches this module, so it cannot go stale.
pub fn policy_json() -> String {
    let mut fams = String::new();
    for (i, f) in Family::all().iter().enumerate() {
        let p = policy(*f);
        if i > 0 {
            fams.push_str(",\n");
        }
        let smoke = p
            .smoke
            .iter()
            .map(|s| {
                let args = s
                    .args
                    .iter()
                    .map(|a| serde_json::to_string(a).expect("serialize typed probe argument"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "        {{\"name\": \"{}\", \"command\": \"{}\", \"args\": [{}], \"timeout_secs\": {}}}",
                    s.name, s.command, args, s.timeout_secs
                )
            })
            .collect::<Vec<_>>()
            .join(",\n");
        let peak = match p.measured_peak_mb {
            Some(v) => v.to_string(),
            None => "null".to_owned(),
        };
        let _ = write!(
            fams,
            "    \"{}\": {{\n\
             \x20     \"gpus_per_task\": {},\n\
             \x20     \"cpus_reserved\": {},\n\
             \x20     \"omp_threads\": {},\n\
             \x20     \"ram_mb\": {},\n\
             \x20     \"gang_size\": {},\n\
             \x20     \"placement\": \"{}\",\n\
             \x20     \"measured_peak_mb\": {},\n\
             \x20     \"peak_evidence\": \"{}\",\n\
             \x20     \"max_runtime_hours\": {},\n\
             \x20     \"preemptible\": {},\n\
             \x20     \"smoke\": [\n{}\n      ]\n    }}",
            p.family.as_str(),
            p.gpus_per_task,
            p.cpus_reserved,
            p.omp_threads,
            p.ram_mb,
            p.gang_size,
            p.placement,
            peak,
            p.peak_evidence,
            p.max_runtime_hours,
            p.preemptible,
            smoke
        );
    }
    format!(
        "{{\n  \"schema\": \"JopeSpecPolicy.v2\",\n  \
         \"generated_by\": \"jope_spec_drift --emit-json (src/spec_policy.rs)\",\n  \
         \"scq_safety_floor_mb\": {SCQ_SAFETY_FLOOR_MB},\n  \
         \"cohort_schema\": \"{COHORT_SCHEMA}\",\n  \
         \"exclude_sealed_families\": {EXCLUDE_SEALED_FAMILIES},\n  \
         \"families\": {{\n{fams}\n  }}\n}}\n"
    )
}

#[cfg(test)]
mod tests;
