use super::*;

/// Resource keys that MUST agree across every renderer of one family.
/// These are the parameters whose divergence caused a measured outage.
pub const SHARED_KEYS: &[&str] = &[
    "gpus_per_task",
    "cpus_reserved",
    "omp_threads",
    "ram_mb",
    "gang_size",
    "placement",
];

/// How a render site expresses placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepForm {
    /// `[[dependencies.smoke]]` -- the ONLY form scq enforces at placement.
    Smoke,
    /// `[dependencies]` + `smoke = [{ ... }]` on one line. Exactly equivalent:
    /// `smoke` IS a field of DependencySpec, so this is enforced too. Kept as a
    /// distinct variant because an earlier version of this scanner recognised
    /// only the `[[dependencies.smoke]]` spelling and therefore reported
    /// `python/test-support/jope/research/test_exact_canary.py` as having NO placement constraint when it
    /// in fact has an enforced one.
    SmokeInline,
    /// `[dependencies] hostname = [...]` -- a serde NO-OP. `hostname` is not a
    /// field of scq's `DependencySpec` (scq-core/src/deps.rs:17-31) and there is
    /// no `deny_unknown_fields`, so serde discards it, `DependencySpec::is_empty()`
    /// becomes true, and `prepare_dependencies`
    /// (scq-controller/src/state.rs:3038-3040) CLEARS `dependency_allowed_nodes`.
    /// Placement ends up completely unconstrained.
    HostnameNoOp,
    /// No dependency block at all: placement unconstrained, but honestly so.
    None,
}

impl fmt::Display for DepForm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DepForm::Smoke => write!(f, "[[dependencies.smoke]]"),
            DepForm::SmokeInline => write!(f, "[dependencies] smoke = [{{...}}] (enforced)"),
            DepForm::HostnameNoOp => write!(f, "[dependencies] hostname (SERDE NO-OP)"),
            DepForm::None => write!(f, "none"),
        }
    }
}

/// One place in the tree that renders an scq job spec.
#[derive(Clone, Debug)]
pub struct RenderSite {
    /// Repo-relative path, e.g. `crates/jope-campaigns/src/bin/jope_selfplay_promotion.rs`.
    pub file: String,
    /// 1-indexed line of the `[resource]` marker.
    pub line: usize,
    /// Enclosing `fn` / `def` name -- the stable key used by `REGISTRY`.
    pub function: String,
    /// `None` = a registered render site that is deliberately OUTSIDE the
    /// flywheel families (an offline campaign). Still checked for the forms scq
    /// drops or rejects; never compared against a family reservation.
    pub family: Option<Family>,
    pub params: BTreeMap<String, String>,
    pub dep_form: DepForm,
    /// True when the site emits `[resource] node`, which scq rejects at submit
    /// ("node is controller-only", scq-core/src/spec.rs:286-294).
    pub emits_resource_node: bool,
    /// True when the site emits `ram_gb`. scq rejects a spec that sets BOTH
    /// `ram_mb` and `ram_gb` (scq-core/src/spec.rs:295-299).
    pub emits_ram_gb: bool,
    /// True when a smoke probe reads `/etc/hostname`. The probe is then ENFORCED
    /// but pins the job to named hosts, so a newly provisioned node never
    /// qualifies -- the anti-pattern the smoke form exists to avoid.
    pub probes_etc_hostname: bool,
}

impl RenderSite {
    pub fn location(&self) -> String {
        format!("{}:{} ({})", self.file, self.line, self.function)
    }
}

/// THE REGISTRY: (repo-relative file, enclosing function, family).
///
/// Keyed on the function name rather than a line number so ordinary edits above
/// a render site do not invalidate it. Every render site the scanner finds must
/// appear here; anything else fails as `UnregisteredSite`.
pub const REGISTRY: &[(&str, &str, Option<Family>)] = &[
    (
        "python/test-support/jope/research/test_exact_backlog.py",
        "render_spec",
        Some(Family::Eval),
    ),
    (
        "python/test-support/jope/research/test_exact_canary.py",
        "render",
        Some(Family::Eval),
    ),
    (
        "python/test-support/jope/research/test_exact_ops.py",
        "render_checkpoint_proposal_spec",
        Some(Family::Propose),
    ),
    (
        "python/test-support/jope/research/test_exact_train.py",
        "render_spec",
        Some(Family::Trainer),
    ),
    // The offline train-only red-team renderer (train_only_redteam_scq.py) and the
    // genome-boost campaign (genome_boost_cluster_campaign.py) were registered here
    // until the Python removal P1 (ADR-0006) archived both; restore with
    // `git checkout 8a55c4fef -- <path>` (docs/archive/python-archive-manifest.json)
    // and re-register them if either campaign is revived.
    // Smart-v3 CPU study dispatcher. It is a frozen, development-only
    // campaign renderer outside the flywheel families, but it still belongs
    // in the registry so its SCQ syntax is audited by the repository test.
    // The shared spec renderers replaced duplicated inline blocks.
    (
        "python/test-support/jope/research/test_exact_train_config.py",
        "render_resource_block",
        Some(Family::Trainer),
    ),
    (
        "python/test-support/jope/research/smart_v3_cpu_scq_specs.py",
        "render_spec",
        None,
    ),
    // Offline adaptive-panel submitters, outside the flywheel families.
    (
        "crates/jope-campaigns/src/test_adaptive/scq_batch.rs",
        "submit_true",
        None,
    ),
    (
        "crates/jope-campaigns/src/test_adaptive/scq_batch.rs",
        "run_batch_flavored",
        None,
    ),
];

/// Directories the scanner reads. Kept explicit so a new renderer is a
/// deliberate act.
///
/// `crates/jope-campaigns/src/test_adaptive` is here because omitting it was a REAL hole, not a
/// hypothetical one: the first version of this list covered only `src/bin` and
/// `scripts`, and `scan_targets` reads each directory NON-recursively. That hid
/// five spec render sites in `crates/jope-campaigns/src/test_adaptive/{scq_batch,binary42}.rs` --
/// compiled code, reachable from `src/main.rs` CLI subcommands -- every one of
/// which emits `[resource] node`, the form scq refuses outright ("node is
/// controller-only"). A completeness check that cannot see a whole directory of
/// renderers is exactly the blind spot this module exists to close.
///
/// Verified 2026-07-29 by grepping the tree for `[resource]`: outside these
/// directories the only remaining hits in `src/` are `spec_policy.rs` and
/// `spec_drift.rs` themselves (the source of truth, see `EXEMPT_FILES`), and in
/// `python/test-support/jope/research/` there are none below the top level.
pub const SCAN_DIRS: &[&str] = &[
    "crates/jope-campaigns/src/bin",
    "crates/jope-campaigns/src/test_adaptive",
    "python/test-support/jope/research",
];

/// The shared spec builder itself, and its checker. These files necessarily
/// contain the literals they exist to police, so scanning them yields phantom
/// render sites. They are the source of truth, not a competing renderer.
pub const EXEMPT_FILES: &[&str] = &[
    "python/test-support/jope/research/jope_spec_policy.py",
    "crates/jope-campaigns/src/bin/jope_spec_drift.rs",
];

#[derive(Clone, Debug)]
pub enum Finding {
    /// Two renderers of one family emit different values for a shared key.
    /// THIS is the check that would have caught all four of today's stalls.
    Disagreement {
        family: Family,
        key: String,
        values: Vec<(String, String)>,
    },
    /// A renderer disagrees with `spec_policy.rs`.
    PolicyMismatch {
        site: String,
        family: Family,
        key: String,
        expected: String,
        actual: String,
    },
    /// `[dependencies] hostname` -- silently dropped, placement unconstrained.
    HostnameNoOp { site: String },
    /// `[resource] node` -- rejected outright at submit.
    ResourceNodePin { site: String },
    /// Both `ram_mb` and `ram_gb` set. scq refuses the whole submit.
    /// VERIFIED 2026-07-29: `scq check` on a spec setting both answers
    /// "invalid spec: set only one of resource.ram_mb and resource.ram_gb".
    RamUnitConflict { site: String },
    /// An ENFORCED smoke probe that greps `/etc/hostname`: a host pin, so a
    /// newly provisioned node never qualifies.
    HostnamePinnedProbe { site: String },
    /// Reservation is below the measured peak, or so far above it that the job
    /// cannot be placed.
    MarginOutOfBounds {
        family: Family,
        ram_mb: u32,
        peak_mb: u32,
        margin: f64,
    },
    /// A GPU family whose reservation + scq's 8 GiB safety floor exceeds the
    /// free RAM of the smallest GPU node: it can only ever land on one host.
    Unplaceable {
        family: Family,
        demand_mb: u32,
        smallest_node_mb: u32,
    },
    /// A render site that is not in `REGISTRY`.
    UnregisteredSite { site: String },
    /// A registry entry that matched no render site (stale after a refactor).
    StaleRegistryEntry { file: String, function: String },
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Finding::Disagreement {
                family,
                key,
                values,
            } => {
                writeln!(
                    f,
                    "DRIFT  {} renderers disagree on `{key}`:",
                    family.as_str()
                )?;
                for (site, v) in values {
                    writeln!(f, "         {v:>12}   {site}")?;
                }
                write!(
                    f,
                    "       -> one job family must have ONE reservation. Fix every \
                     renderer to spec_policy::{}.{key}.",
                    family.as_str().to_uppercase()
                )
            }
            Finding::PolicyMismatch {
                site,
                family,
                key,
                expected,
                actual,
            } => write!(
                f,
                "POLICY {site} renders {}.{key} = {actual}, but spec_policy.rs says {expected}",
                family.as_str()
            ),
            Finding::HostnameNoOp { site } => write!(
                f,
                "NO-OP  {site} uses `[dependencies] hostname`. scq's DependencySpec has no \
                 such field (scq-core/src/deps.rs:17-31), serde drops it, and \
                 prepare_dependencies CLEARS dependency_allowed_nodes \
                 (scq-controller/src/state.rs:3038-3040). Placement is NOT constrained."
            ),
            Finding::ResourceNodePin { site } => write!(
                f,
                "REJECT {site} emits `[resource] node`. scq refuses the whole submit: \
                 \"node is controller-only\" (scq-core/src/spec.rs:286-294)."
            ),
            Finding::RamUnitConflict { site } => write!(
                f,
                "RAMDUP {site} sets BOTH `ram_mb` and `ram_gb` in [resource]. scq refuses \
                 the whole submit: \"set only one of resource.ram_mb and resource.ram_gb\"."
            ),
            Finding::HostnamePinnedProbe { site } => write!(
                f,
                "PIN    {site} declares an ENFORCED smoke probe that greps /etc/hostname. \
                 That pins the job to named hosts, so a newly provisioned node never \
                 qualifies -- and when the named host is busy the submit is REJECTED \
                 rather than queued. Probe a CAPABILITY the job actually dereferences."
            ),
            Finding::MarginOutOfBounds {
                family,
                ram_mb,
                peak_mb,
                margin,
            } => write!(
                f,
                "MARGIN {} reserves {ram_mb} MB against a MEASURED {peak_mb} MB peak \
                 ({margin:.2}x); allowed band is {:.1}x..{:.1}x",
                family.as_str(),
                spec_policy::MIN_MARGIN,
                spec_policy::MAX_MARGIN
            ),
            Finding::Unplaceable {
                family,
                demand_mb,
                smallest_node_mb,
            } => write!(
                f,
                "PLACE  {} demands {demand_mb} MB free (reservation + scq's {} MB safety \
                 floor) but the smallest GPU node has {smallest_node_mb} MB -- it can only \
                 ever land on one host",
                family.as_str(),
                spec_policy::SCQ_SAFETY_FLOOR_MB
            ),
            Finding::UnregisteredSite { site } => write!(
                f,
                "NEW    {site} renders an scq spec but is not in spec_drift::REGISTRY. \
                 Adding a renderer without registering it is how this outage class starts. \
                 Register it against its job family."
            ),
            Finding::StaleRegistryEntry { file, function } => write!(
                f,
                "STALE  spec_drift::REGISTRY names {file}::{function}, which renders no \
                 spec any more. Remove the entry."
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Scanner
// ---------------------------------------------------------------------------
