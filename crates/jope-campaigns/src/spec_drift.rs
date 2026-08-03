//! WS-2: the DRIFT TEST. Fails when two renderers of the same job family
//! disagree on a shared scq spec parameter.
//!
//! `spec_policy.rs` declares what each job family SHOULD reserve. This module is
//! the enforcement half: it reads every spec-rendering source file in the repo,
//! extracts the `[resource]` / `[dependencies]` block each one emits, and fails
//! when two renderers of the same family disagree, or when any renderer
//! disagrees with the policy, or when a renderer uses a form scq silently drops.
//!
//! # Why this exists
//!
//! Every cluster stall on 2026-07-28 was ONE bug class: the same job family was
//! rendered by several independent code paths that had silently diverged. Nobody
//! could see the divergence because no single file contained two of the values.
//! This module is the file that contains all of them.
//!
//! Measured drift at the time of writing (see `tests`):
//!   * trainer `ram_mb`: THREE values -- 49152, 16384, 8192.
//!   * propose `ram_mb`: TWO values  -- 49152, 12288.
//!   * placement form: 1 renderer of 10 used the enforced `[[dependencies.smoke]]`;
//!     5 used `[dependencies] hostname`, which scq DROPS.
//!
//! # Scanner contract
//!
//! A "render site" is any line containing the literal `[resource]` in a source
//! file under `src/bin/` or `scripts/`. Comments and Rust `mod tests` blocks are
//! stripped first, because a commented-out `ram_mb = 49152` is documentation,
//! not a reservation -- an earlier version of this scanner reported exactly that
//! false positive against `scripts/test_exact_ops.py`.
//!
//! Every render site MUST be listed in `REGISTRY`. An unregistered site fails the
//! test: adding a sixth renderer is precisely the event that caused this outage
//! class, so it has to be a deliberate, reviewed act.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::spec_policy::{self, Family};

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
    /// `scripts/test_exact_canary.py` as having NO placement constraint when it
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
    /// Repo-relative path, e.g. `src/bin/jope_selfplay_promotion.rs`.
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
        "src/bin/jope_selfplay_promotion.rs",
        "render_checkpoint_proposal_spec",
        Some(Family::Propose),
    ),
    (
        "src/bin/jope_selfplay_promotion.rs",
        "render_trainer_spec",
        Some(Family::Trainer),
    ),
    (
        "src/bin/jope_selfplay_propose_driver.rs",
        "render_spec",
        Some(Family::Propose),
    ),
    (
        "src/bin/jope_selfplay_supply.rs",
        "render_cohort_proposer_spec",
        Some(Family::Propose),
    ),
    (
        "src/bin/jope_selfplay_expansion_driver.rs",
        "submit_encode_chunk",
        Some(Family::Encode),
    ),
    // Codex-owned (do not edit; propose changes in FLYWHEEL_CHAT.md). Scanned
    // read-only so its drift is still visible here.
    (
        "src/bin/jope_selfplay_queue.rs",
        "render_spec",
        Some(Family::Eval),
    ),
    (
        "src/bin/jope_selfplay_trainer_watchdog.rs",
        "render_spec",
        Some(Family::Trainer),
    ),
    // Sealed-holdout zero-shot evaluation. Appeared 2026-07-28 23:34Z and was
    // caught by the UnregisteredSite check within minutes of being written --
    // which is the whole point of that check. CPU eval shape (0 GPU, 1 core,
    // 512 MB), so it is the Eval family.
    (
        "src/bin/jope_sealed_zeroshot.rs",
        "plan",
        Some(Family::Eval),
    ),
    (
        "scripts/test_exact_backlog.py",
        "render_spec",
        Some(Family::Eval),
    ),
    ("scripts/test_exact_canary.py", "render", Some(Family::Eval)),
    (
        "scripts/test_exact_ops.py",
        "render_checkpoint_proposal_spec",
        Some(Family::Propose),
    ),
    (
        "scripts/test_exact_train.py",
        "render_spec",
        Some(Family::Trainer),
    ),
    // Offline train-only red-team campaign renderer. It is not part of the
    // flywheel families, but its SCQ resource/placement syntax is still
    // audited so a future campaign cannot reintroduce ignored host pins.
    ("scripts/train_only_redteam_scq.py", "render_spec", None),
    // Coverage recovery renderers are isolated campaign controllers. Their
    // reservations are manifest-driven rather than flywheel-family policy,
    // but their SCQ syntax remains subject to the generic rejection checks.
    ("scripts/coverage_control_plane.py", "render_spec", None),
    ("scripts/mc_union_coverage.py", "render_spec", None),
    // Smart-v3 CPU study dispatcher. It is a frozen, development-only
    // campaign renderer outside the flywheel families, but it still belongs
    // in the registry so its SCQ syntax is audited by the repository test.
    ("scripts/smart_v3_cpu_scq.py", "render_spec", None),
    // Offline genome-boost campaign. NOT a flywheel family: it is a manually
    // driven benchmark sweep with its own cadence, so its reservations are not
    // compared against trainer/propose/eval/encode. It IS still checked for the
    // forms scq drops or rejects -- and it currently emits a `[resource] node`
    // pin, which scq refuses outright.
    (
        "scripts/genome_boost_cluster_campaign.py",
        "render_spec",
        None,
    ),
    // Binary42 / adaptive-panel campaign submitters. Compiled code, reachable
    // from `src/main.rs` CLI subcommands (`run_binary42`, `binary42_batch_worker`,
    // ...), NOT daemons -- so they are out of the flywheel families and their
    // reservations are not compared against trainer/propose/eval/encode.
    //
    // They were invisible to this scanner until 2026-07-29 because `SCAN_DIRS`
    // omitted `src/test_adaptive`. All five pin `[resource] node`, which scq
    // refuses outright ("node is controller-only"), so these campaign
    // subcommands cannot submit as written. Registered here so the pin is
    // REPORTED rather than merely unseen; fixing them is a separate change owned
    // by whoever runs those campaigns.
    ("src/test_adaptive/binary42.rs", "submit_canary", None),
    ("src/test_adaptive/binary42.rs", "submit_canary_at", None),
    ("src/test_adaptive/binary42.rs", "scq_spec", None),
    ("src/test_adaptive/scq_batch.rs", "submit_true", None),
    ("src/test_adaptive/scq_batch.rs", "run_batch_flavored", None),
    ("src/test_adaptive/timeseries42.rs", "scq_spec", None),
    ("src/test_adaptive/multiclass60.rs", "render_spec", None),
    (
        "src/test_adaptive/multiclass60.rs",
        "render_replay_spec",
        None,
    ),
];

/// Directories the scanner reads. Kept explicit so a new renderer is a
/// deliberate act.
///
/// `src/test_adaptive` is here because omitting it was a REAL hole, not a
/// hypothetical one: the first version of this list covered only `src/bin` and
/// `scripts`, and `scan_targets` reads each directory NON-recursively. That hid
/// five spec render sites in `src/test_adaptive/{scq_batch,binary42}.rs` --
/// compiled code, reachable from `src/main.rs` CLI subcommands -- every one of
/// which emits `[resource] node`, the form scq refuses outright ("node is
/// controller-only"). A completeness check that cannot see a whole directory of
/// renderers is exactly the blind spot this module exists to close.
///
/// Verified 2026-07-29 by grepping the tree for `[resource]`: outside these
/// directories the only remaining hits in `src/` are `spec_policy.rs` and
/// `spec_drift.rs` themselves (the source of truth, see `EXEMPT_FILES`), and in
/// `scripts/` there are none below the top level.
pub const SCAN_DIRS: &[&str] = &["src/bin", "src/test_adaptive", "scripts"];

/// The shared spec builder itself, and its checker. These files necessarily
/// contain the literals they exist to police, so scanning them yields phantom
/// render sites. They are the source of truth, not a competing renderer.
pub const EXEMPT_FILES: &[&str] = &["scripts/jope_spec_policy.py", "src/bin/jope_spec_drift.rs"];

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

/// Strip whole-line comments so a commented-out value is never read as a
/// reservation. This is load-bearing: `scripts/test_exact_ops.py` documents the
/// old 49152 value in a comment directly above the live 12288, and a naive
/// scanner reports the comment.
fn strip_comments(src: &str, is_py: bool) -> Vec<String> {
    src.lines()
        .map(|ln| {
            let t = ln.trim_start();
            let is_comment = if is_py {
                t.starts_with('#')
            } else {
                t.starts_with("//")
            };
            if is_comment {
                String::new()
            } else {
                ln.to_owned()
            }
        })
        .collect()
}

/// Rust `#[cfg(test)]` modules assert against these same literals, so scanning
/// them yields phantom render sites. Cut the file at the test module.
fn cut_test_module(src: &str) -> &str {
    for marker in ["\n#[cfg(test)]", "\nmod tests"] {
        if let Some(i) = src.find(marker) {
            return &src[..i];
        }
    }
    src
}

/// Value of `key` in a window of rendered-TOML-inside-source-code.
///
/// Handles all three literal styles in this tree simultaneously:
///   Python list item : `"ram_mb = 12288",`
///   Rust owned string: `"ram_mb = 49152".to_owned(),`
///   Rust raw block   : `ram_mb = 16384\n\`
fn find_value(window: &str, key: &str) -> Option<String> {
    for ln in window.lines() {
        // Every occurrence, not just the first: `placement = "single_node"`
        // contains "node", and stopping at the first hit would hide a real
        // `node = "xbabe2"` later on the same line.
        let mut from = 0usize;
        while let Some(rel) = ln[from..].find(key) {
            let pos = from + rel;
            from = pos + key.len();
            // Reject substring hits such as `packed_ram_mb` or `single_node`.
            if pos > 0 {
                let prev = ln.as_bytes()[pos - 1];
                if prev == b'_' || prev.is_ascii_alphanumeric() {
                    continue;
                }
            }
            let rest = ln[pos + key.len()..].trim_start();
            let Some(rest) = rest.strip_prefix('=') else {
                continue;
            };
            let v: String = rest
                .trim_start()
                .trim_start_matches('\\')
                .trim_start_matches('"')
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                .collect();
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    None
}

/// The `[resource]` section of an unescaped window: everything from the
/// `[resource]` header to the next TOML section header.
///
/// Needed because a bare `find_value(window, "node")` is not a reliable test for
/// a `[resource] node` pin. It matched Rust `format!` NAMED ARGUMENTS that
/// happen to be spelled `node = node,` below the template, and missed the real
/// pin whenever the value was a placeholder (`node = {node:?}`), whose `{` makes
/// the extracted value empty. Measured on this tree: three of the five
/// `src/test_adaptive` render sites emit `[resource] node` and were NOT flagged,
/// while the two that were flagged matched on the format-argument list rather
/// than on the spec text. Scoping the test to the section, and looking for the
/// key at the start of a line, is what TOML actually means.
fn resource_section(window: &str) -> &str {
    let Some(start) = window.find("[resource]") else {
        return "";
    };
    let rest = &window[start + "[resource]".len()..];
    match rest.find("\n[") {
        Some(end) => &rest[..end],
        None => rest,
    }
}

/// Is `key` assigned at the start of a line inside this section?
fn section_declares(section: &str, key: &str) -> bool {
    section.lines().any(|ln| {
        let t = ln.trim();
        t.strip_prefix(key)
            .is_some_and(|r| r.trim_start().starts_with('='))
    })
}

fn enclosing_function(lines: &[String], idx: usize, is_py: bool) -> String {
    for i in (0..=idx).rev() {
        let t = lines[i].trim_start();
        let rest = if is_py {
            t.strip_prefix("def ")
        } else {
            t.strip_prefix("fn ")
                .or_else(|| t.strip_prefix("pub fn "))
                .or_else(|| t.strip_prefix("async fn "))
                .or_else(|| t.strip_prefix("pub async fn "))
        };
        if let Some(rest) = rest {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return name;
            }
        }
    }
    "<module>".to_owned()
}

/// `Some(entry)` when the site is registered; the inner `Option<Family>` is the
/// family it belongs to, or `None` for a deliberately out-of-scope renderer.
fn registry_entry(file: &str, function: &str) -> Option<Option<Family>> {
    REGISTRY
        .iter()
        .find(|(f, fun, _)| *f == file && *fun == function)
        .map(|(_, _, fam)| *fam)
}

/// Scan one source file for render sites.
fn scan_file(repo_root: &Path, rel: &str) -> std::io::Result<Vec<RenderSite>> {
    let full: PathBuf = repo_root.join(rel);
    let raw = std::fs::read_to_string(&full)?;
    if !raw.contains("[resource]") {
        return Ok(vec![]);
    }
    let is_py = rel.ends_with(".py");
    let body = if is_py {
        raw.as_str()
    } else {
        cut_test_module(&raw)
    };
    let lines = strip_comments(body, is_py);

    let mut out = vec![];
    for (i, ln) in lines.iter().enumerate() {
        if !ln.contains("[resource]") {
            continue;
        }
        // Unescape literal `\n` before extracting values. Several renderers emit
        // an ENTIRE spec as one single-line `format!` string
        // (`"[job]\nid = ...\n\n[resource]\ngpus_per_task = 0\n..."`), so every
        // key is preceded by the two characters `\` `n`. `find_value` rejects a
        // hit whose preceding byte is alphanumeric -- the guard that stops
        // `packed_ram_mb` matching `ram_mb` -- and `n` is alphanumeric, so every
        // key in those specs was silently skipped and the site reported `params
        // {}`. Turning the escape into a real newline makes the one-line and
        // multi-line styles parse identically. Line numbers are unaffected: they
        // come from the physical index `i`, not from this window.
        let unescape = |s: String| s.replace("\\n", "\n");
        let hi = (i + 60).min(lines.len());
        let window = unescape(lines[i..hi].join("\n"));
        let hi2 = (i + 100).min(lines.len());
        let window2 = unescape(lines[i..hi2].join("\n"));

        let mut params = BTreeMap::new();
        for k in SHARED_KEYS {
            if let Some(v) = find_value(&window, k) {
                params.insert((*k).to_owned(), v);
            }
        }
        let dep_form = if window2.contains("[[dependencies.smoke]]") {
            DepForm::Smoke
        } else if window2.contains("smoke = [") {
            DepForm::SmokeInline
        } else if window2.contains("hostname = [") {
            DepForm::HostnameNoOp
        } else {
            DepForm::None
        };
        // A probe reading /etc/hostname is a host pin whichever form carries it.
        let probes_etc_hostname = matches!(dep_form, DepForm::Smoke | DepForm::SmokeInline)
            && window2.contains("/etc/hostname");
        let function = enclosing_function(&lines, i, is_py);
        let family = registry_entry(rel, &function).flatten();
        out.push(RenderSite {
            file: rel.to_owned(),
            line: i + 1,
            function,
            family,
            params,
            dep_form,
            emits_resource_node: section_declares(resource_section(&window2), "node")
                || window.contains("node_line"),
            emits_ram_gb: section_declares(resource_section(&window2), "ram_gb"),
            probes_etc_hostname,
        });
    }
    Ok(out)
}

/// Every source file the scanner considers, repo-relative and sorted.
pub fn scan_targets(repo_root: &Path) -> Vec<String> {
    let mut out = vec![];
    for dir in SCAN_DIRS {
        let Ok(rd) = std::fs::read_dir(repo_root.join(dir)) else {
            continue;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            let ok = p
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e == "rs" || e == "py");
            if ok && p.is_file() {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    let rel = format!("{dir}/{name}");
                    if !EXEMPT_FILES.contains(&rel.as_str()) {
                        out.push(rel);
                    }
                }
            }
        }
    }
    out.sort();
    out
}

/// Scan the whole repo for spec render sites.
pub fn scan_repo(repo_root: &Path) -> std::io::Result<Vec<RenderSite>> {
    let mut sites = vec![];
    for rel in scan_targets(repo_root) {
        sites.extend(scan_file(repo_root, &rel)?);
    }
    sites.sort_by(|a, b| (&a.file, a.line).cmp(&(&b.file, b.line)));
    Ok(sites)
}

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
            if margin < spec_policy::MIN_MARGIN || margin > spec_policy::MAX_MARGIN {
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

/// Repo root, resolved from the crate manifest so the test works from anywhere.
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sites() -> Vec<RenderSite> {
        scan_repo(&repo_root()).expect("scan the repo")
    }

    /// The scanner must not regress into finding nothing: a silent scan failure
    /// would make every other check in this file vacuous.
    #[test]
    fn scanner_finds_the_known_render_sites() {
        let s = sites();
        assert!(
            s.len() >= 8,
            "expected >=8 spec render sites, found {}: {:#?}",
            s.len(),
            s.iter().map(|x| x.location()).collect::<Vec<_>>()
        );
        for (file, function, _) in REGISTRY {
            assert!(
                s.iter().any(|x| x.file == *file && x.function == *function),
                "registry entry {file}::{function} matched no render site -- the scanner \
                 or the file moved"
            );
        }
    }

    /// A commented-out reservation must never be read as a live one. This
    /// regression is real: `scripts/test_exact_ops.py` documents the old 49152
    /// directly above the live 12288.
    #[test]
    fn commented_out_values_are_not_parsed() {
        let py = "[resource]\n# ram_mb = 49152 was the old value\nram_mb = 12288\n";
        let lines = strip_comments(py, true);
        let joined = lines.join("\n");
        assert_eq!(find_value(&joined, "ram_mb").as_deref(), Some("12288"));

        let rs = "[resource]\n// ram_mb = 49152 stalled the lane\nram_mb = 16384\n";
        let lines = strip_comments(rs, false);
        let joined = lines.join("\n");
        assert_eq!(find_value(&joined, "ram_mb").as_deref(), Some("16384"));
    }

    /// All three literal styles used in this tree must parse identically.
    #[test]
    fn value_extraction_handles_every_literal_style() {
        assert_eq!(
            find_value("        \"ram_mb = 12288\",", "ram_mb").as_deref(),
            Some("12288")
        );
        assert_eq!(
            find_value("        \"ram_mb = 49152\".to_owned(),", "ram_mb").as_deref(),
            Some("49152")
        );
        assert_eq!(
            find_value("         ram_mb = 16384\\n\\", "ram_mb").as_deref(),
            Some("16384")
        );
        assert_eq!(
            find_value("         placement = \\\"single_node\\\"\\n", "placement").as_deref(),
            Some("single_node")
        );
        // must not match a longer identifier that merely ends in the key
        assert_eq!(find_value("packed_ram_mb = 99", "ram_mb"), None);
    }

    /// THE DELIVERABLE. Two renderers of one family must never disagree on a
    /// shared parameter, and no renderer may use a form scq drops or rejects.
    ///
    /// This is the test that would have caught all four of 2026-07-28's stalls
    /// before they shipped.
    #[test]
    fn no_two_renderers_disagree_and_none_use_a_dropped_form() {
        let findings = audit(&sites());
        if !findings.is_empty() {
            let mut msg = format!(
                "\n{} spec-drift finding(s) -- a job family with two reservations is an \
                 outage waiting to happen:\n\n",
                findings.len()
            );
            for f in &findings {
                msg.push_str(&format!("{f}\n\n"));
            }
            panic!("{msg}");
        }
    }

    /// The scanner itself must be able to SEE drift. Without this, a scanner bug
    /// that returns nothing would make the deliverable silently pass forever.
    #[test]
    fn audit_detects_injected_disagreement() {
        let mk = |file: &str, ram: &str| RenderSite {
            file: file.to_owned(),
            line: 1,
            function: "render_spec".to_owned(),
            family: Some(Family::Trainer),
            params: BTreeMap::from([("ram_mb".to_owned(), ram.to_owned())]),
            dep_form: DepForm::Smoke,
            emits_resource_node: false,
            emits_ram_gb: false,
            probes_etc_hostname: false,
        };
        let injected = vec![mk("a.rs", "16384"), mk("b.py", "8192")];
        let findings = audit(&injected);
        assert!(
            findings.iter().any(|f| matches!(
                f,
                Finding::Disagreement { key, .. } if key == "ram_mb"
            )),
            "audit failed to notice 16384 vs 8192 -- the drift test is broken: {findings:#?}"
        );
    }

    /// The checked-in JSON the scripts/ lane reads must match this crate exactly.
    /// Without this the Python renderers would silently keep a stale copy of the
    /// numbers -- which is the very failure mode this workstream exists to end.
    ///
    /// Regenerate with:
    ///   cargo run --release --bin jope_spec_drift -- --emit-json \
    ///     > scripts/jope-spec-policy.json
    #[test]
    fn checked_in_policy_json_matches_this_module() {
        let path = repo_root().join("scripts/jope-spec-policy.json");
        let on_disk = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{} is missing ({e}). Regenerate: cargo run --release --bin \
                 jope_spec_drift -- --emit-json > scripts/jope-spec-policy.json",
                path.display()
            )
        });
        assert_eq!(
            on_disk,
            spec_policy::policy_json(),
            "scripts/jope-spec-policy.json is STALE -- the Python lane is reading \
             different numbers from the Rust policy. Regenerate: cargo run --release \
             --bin jope_spec_drift -- --emit-json > scripts/jope-spec-policy.json"
        );
    }

    /// Both spellings of a smoke dependency are enforced by scq and must both be
    /// recognised. The inline `smoke = [{ ... }]` form is REAL: it is how
    /// `scripts/test_exact_canary.py` pins itself to a host, and an earlier
    /// version of this scanner reported that file as having no placement
    /// constraint at all.
    #[test]
    fn both_smoke_spellings_are_recognised_and_hostname_pins_are_flagged() {
        let s = sites();
        let canary = s
            .iter()
            .find(|x| x.file == "scripts/test_exact_canary.py")
            .expect("canary render site");
        assert_eq!(
            canary.dep_form,
            DepForm::SmokeInline,
            "inline `smoke = [{{...}}]` must be seen as an ENFORCED dependency, not as none"
        );
        assert!(
            canary.probes_etc_hostname,
            "a probe grepping /etc/hostname is a host pin and must be flagged"
        );
        assert!(
            audit(std::slice::from_ref(canary))
                .iter()
                .any(|f| matches!(f, Finding::HostnamePinnedProbe { .. })),
            "the audit must raise HostnamePinnedProbe for an /etc/hostname probe"
        );
    }

    /// A whole spec rendered as ONE `\n`-escaped line must parse exactly like a
    /// multi-line one. Five real render sites in `src/test_adaptive` are written
    /// that way; before the window was unescaped they reported `params {}`,
    /// so every reservation in them was invisible to the drift check.
    #[test]
    fn single_line_escaped_specs_are_parsed() {
        let s = sites();
        let site = s
            .iter()
            .find(|x| x.file == "src/test_adaptive/scq_batch.rs" && x.function == "submit_true")
            .expect("scq_batch::submit_true render site");
        assert_eq!(
            site.params.get("gpus_per_task").map(String::as_str),
            Some("0"),
            "a one-line `\\n`-escaped spec must yield its [resource] values: {:?}",
            site.params
        );
        assert!(
            site.emits_resource_node,
            "`node = {{node:?}}` inside [resource] is a controller-only pin and must be flagged"
        );
    }

    /// `[resource] node` must be detected from the SECTION, not from a stray
    /// `format!` named argument, and a placeholder value must not hide it.
    #[test]
    fn resource_node_is_detected_from_the_section_only() {
        let spec = "[resource]\ngpus_per_task = 0\nnode = {node:?}\ngang_size = 1\n\n\
                    [limits]\nmax_restarts = 0\n";
        assert!(section_declares(resource_section(spec), "node"));
        // A `node` key in a LATER section is not a [resource] pin.
        let other = "[resource]\ngpus_per_task = 0\n\n[meta]\nnode = \"xbabe3\"\n";
        assert!(!section_declares(resource_section(other), "node"));
        // `placement = "single_node"` must never read as a node pin.
        let placement = "[resource]\ngang_size = 1\nplacement = \"single_node\"\n";
        assert!(!section_declares(resource_section(placement), "node"));
    }

    /// Setting both RAM units is a hard submit rejection and must be a finding.
    #[test]
    fn both_ram_units_is_a_finding() {
        let site = RenderSite {
            file: "src/bin/whatever.rs".to_owned(),
            line: 1,
            function: "render_spec".to_owned(),
            family: Some(Family::Eval),
            params: BTreeMap::from([("ram_mb".to_owned(), "512".to_owned())]),
            dep_form: DepForm::None,
            emits_resource_node: false,
            emits_ram_gb: true,
            probes_etc_hostname: false,
        };
        assert!(
            audit(&[site])
                .iter()
                .any(|f| matches!(f, Finding::RamUnitConflict { .. })),
            "a spec setting ram_mb AND ram_gb must be reported -- scq refuses it"
        );
    }

    /// An unregistered renderer must fail, so a sixth spec path cannot appear
    /// unreviewed.
    #[test]
    fn audit_rejects_an_unregistered_render_site() {
        let orphan = RenderSite {
            file: "src/bin/brand_new_thing.rs".to_owned(),
            line: 42,
            function: "render_spec".to_owned(),
            family: Some(Family::Eval),
            params: BTreeMap::new(),
            dep_form: DepForm::None,
            emits_resource_node: false,
            emits_ram_gb: false,
            probes_etc_hostname: false,
        };
        assert!(
            audit(&[orphan])
                .iter()
                .any(|f| matches!(f, Finding::UnregisteredSite { .. })),
            "an unregistered render site must fail the drift test"
        );
    }
}
