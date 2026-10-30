//! Pinned portfolio member registry (`core_v1`) — the fixed member set the
//! compact `(portfolio preset=core ...)` head expands to at fit time.
//!
//! The registry is DATA, not grammar: names, inner pipelines, dataset-stat
//! gates (train stats only: n, p, dup_frac — blessed by the game card), and
//! per-member feature-view flags. Registry versions are append-only; every
//! training label that references a portfolio baseline carries this version
//! string (`baseline_id`).
//!
//! # core_v1 substitutions (documented deviations from the frozen spec table)
//!
//! The spec (docs/specs/portfolio_head.md §3) pins member pipelines in terms
//! of heads/params that partially land later (G1b/G1c). Until those exist in
//! this crate, `core_v1` substitutes as follows — full fidelity replaces these
//! in a future append-only registry version:
//!
//! * `knn_loo` — a LOO-k kNN head does not exist yet; substituted with the
//!   Nadaraya-Watson smoother `(ksmooth k=32)` (nearest-neighbor family,
//!   fixed k).
//! * `enet_quad` — `(elasticnet cv=3)` self-tuning does not exist yet. The
//!   task sheet suggested `(ridge_gcv)`, but that would make `enet_quad` an
//!   exact duplicate of `ridge_quad`; instead the existing fixed-parameter
//!   `(elasticnet)` head (alpha=0.001, l1_ratio=0.5, sweeps=60 — the stack
//!   library config) keeps the L1 diversity the member exists for. FLAGGED
//!   as a deviation from the task sheet.
//! * `gam_tbe` — `target_bin_encode oof=5` (inner-OOF de-biasing) does not
//!   exist yet; plain `(target_bin_encode bins=12)` is used. The portfolio's
//!   own per-fold member refits keep it leak-safe under OOF measurement.
//! * `gpr_lin` / `gpr_blom` — `exact=true` does not exist yet; the default
//!   inducing-point budget m=500 already gives a full-train exact solve for
//!   n <= 500 and a 500-point basis above.
//! * `gpr_iso` — `ard=false` does not exist yet; substituted with the
//!   dedicated ARD-RBF path `(gpr kernel=rbf steps=40)`, which stays distinct
//!   from `gpr_lin`'s composite `rbf+linear` kernel.
//! * `ridge_quad` / `enet_quad` — the spec's `pair_prod topk=40` exceeds the
//!   engine clamp (2..24); pinned at `topk=24`.
//! * `histgbm_d6` — the `raw+screen` view needs `histgbm2 screen=` (G1);
//!   plain raw view until then.

use crate::dsl::{parse_dsl, DslExpr, ParamValue, Stage};
use anyhow::{bail, Result};

/// Registry version stamped into diagnostics / training labels.
pub const REGISTRY_VERSION: &str = "core_v1";

/// Hard cap on simultaneously-active members (spec §2: 1..12; the always-on
/// core is 8 and the conditional gates cannot all fire together).
pub const MAX_ACTIVE_MEMBERS: usize = 12;

/// Dataset-stat gate: a pure function of the training array (n rows, p kept
/// columns, duplicate-row fraction). No dataset-identity dispatch.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Gate {
    Always,
    /// Active iff n >= threshold.
    NAtLeast(usize),
    /// Active iff n < threshold.
    NBelow(usize),
    /// Active iff p <= threshold.
    PAtMost(usize),
    /// Active iff duplicate-row fraction > threshold.
    DupFracAbove(f64),
}

impl Gate {
    pub fn passes(&self, n: usize, p: usize, dup_frac: f64) -> bool {
        match self {
            Gate::Always => true,
            Gate::NAtLeast(t) => n >= *t,
            Gate::NBelow(t) => n < *t,
            Gate::PAtMost(t) => p <= *t,
            Gate::DupFracAbove(t) => dup_frac > *t,
        }
    }
}

/// Feature view a member sees when the kept width exceeds 300 columns:
/// `Proj` members get the shared 256-component projection; `Raw` members
/// always see the raw kept block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum View {
    Raw,
    Proj,
}

/// How a member computes: a DSL pipeline executed through `model::fit`, or
/// the built-in duplicate-key shrunk group mean (not expressible as a pipe).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberKind {
    Pipe,
    GroupMean,
}

#[derive(Clone, Copy, Debug)]
pub struct MemberDef {
    pub name: &'static str,
    /// DSL source of the inner pipeline; empty for `GroupMean`.
    pub pipe: &'static str,
    pub kind: MemberKind,
    pub gate: Gate,
    pub view: View,
}

/// The pinned `core_v1` member table (order is part of the contract).
pub const CORE_V1: [MemberDef; 13] = [
    MemberDef {
        name: "ridge_gcv_raw",
        pipe: "(pipe X (ridge_gcv))",
        kind: MemberKind::Pipe,
        gate: Gate::Always,
        view: View::Proj,
    },
    MemberDef {
        name: "ridge_quad",
        // spec: pair_prod topk=40 -> engine clamp pins 24 (see module docs).
        pipe: "(pipe X (cheb degree=3 append=true) (pair_prod topk=24 append=true) (ridge_gcv))",
        kind: MemberKind::Pipe,
        gate: Gate::Always,
        view: View::Raw,
    },
    MemberDef {
        name: "gpr_lin",
        // spec: exact=true -> default m=500 (full-train solve for n<=500).
        pipe: "(pipe X (gpr kernel=rbf+linear steps=40))",
        kind: MemberKind::Pipe,
        gate: Gate::Always,
        view: View::Proj,
    },
    MemberDef {
        name: "gpr_blom",
        pipe: "(pipe X (blom) (gpr kernel=rbf+linear steps=40))",
        kind: MemberKind::Pipe,
        gate: Gate::Always,
        view: View::Proj,
    },
    MemberDef {
        name: "knn_loo",
        // Substituted: LOO-k kNN head lands in G1b (see module docs).
        pipe: "(pipe X (ksmooth k=32))",
        kind: MemberKind::Pipe,
        gate: Gate::Always,
        view: View::Proj,
    },
    MemberDef {
        name: "rff_ridge",
        pipe: "(pipe X (rff_cos n=160 append=false) (rff_sin n=160 append=true) (ridge_gcv))",
        kind: MemberKind::Pipe,
        gate: Gate::Always,
        view: View::Proj,
    },
    MemberDef {
        name: "histgbm_d6",
        // spec view raw+screen: corr-screen lands with `histgbm2 screen=` (G1).
        pipe: "(pipe X (histgbm2 leaves=63 lr=0.06 rounds=400 bags=3))",
        kind: MemberKind::Pipe,
        gate: Gate::Always,
        view: View::Raw,
    },
    MemberDef {
        name: "histgbm_d3",
        pipe: "(pipe X (histgbm2 leaves=15 lr=0.08 rounds=500))",
        kind: MemberKind::Pipe,
        gate: Gate::Always,
        view: View::Raw,
    },
    MemberDef {
        name: "histgbm_slow",
        pipe: "(pipe X (histgbm2 leaves=31 lr=0.03 rounds=900 col_subsample=0.7))",
        kind: MemberKind::Pipe,
        gate: Gate::NAtLeast(500),
        view: View::Raw,
    },
    MemberDef {
        name: "enet_quad",
        // Substituted head: elasticnet cv=3 lands in G1c (see module docs).
        pipe: "(pipe X (cheb degree=3 append=true) (pair_prod topk=24 append=true) (elasticnet))",
        kind: MemberKind::Pipe,
        gate: Gate::PAtMost(40),
        view: View::Raw,
    },
    MemberDef {
        name: "gpr_iso",
        // Substituted: ard=false lands in G1; ARD-RBF dedicated path meanwhile.
        pipe: "(pipe X (gpr kernel=rbf steps=40))",
        kind: MemberKind::Pipe,
        gate: Gate::NBelow(150),
        view: View::Proj,
    },
    MemberDef {
        name: "gam_tbe",
        // Substituted: oof=5 inner de-biasing lands with the G1 upgrade.
        pipe: "(pipe X (target_bin_encode bins=12) (ridge_gcv))",
        kind: MemberKind::Pipe,
        gate: Gate::NAtLeast(80),
        view: View::Raw,
    },
    MemberDef {
        name: "group_mean",
        pipe: "",
        kind: MemberKind::GroupMean,
        gate: Gate::DupFracAbove(0.3),
        view: View::Raw,
    },
];

/// Registry lookup by member name.
pub fn find(name: &str) -> Option<&'static MemberDef> {
    CORE_V1.iter().find(|m| m.name == name)
}

/// A member resolved for one portfolio fit: parsed pipeline + gate + view.
#[derive(Clone, Debug)]
pub struct ResolvedMember {
    pub name: String,
    /// Parsed inner pipeline (`None` for the built-in group-mean member).
    pub pipe: Option<DslExpr>,
    pub gate: Gate,
    pub view: View,
}

impl ResolvedMember {
    pub fn from_def(def: &MemberDef) -> Result<ResolvedMember> {
        let pipe = match def.kind {
            MemberKind::Pipe => Some(parse_dsl(def.pipe)?),
            MemberKind::GroupMean => None,
        };
        Ok(ResolvedMember {
            name: def.name.to_string(),
            pipe,
            gate: def.gate,
            view: def.view,
        })
    }

    /// An explicit-form member: gate Always, raw view (views/gates are
    /// registry data; the nested grammar carries neither).
    pub fn from_pipe(name: String, pipe: DslExpr) -> ResolvedMember {
        ResolvedMember {
            name,
            pipe: Some(pipe),
            gate: Gate::Always,
            view: View::Raw,
        }
    }

    pub fn canonical(&self) -> String {
        match &self.pipe {
            Some(p) => p.to_string(),
            None => "(group mean over duplicate rows)".to_string(),
        }
    }
}

/// The built-in small-sample fallback member (spec §4 step 6: n < 20 fits a
/// single `ridge_gcv_raw`), independent of preset/add/drop edits.
pub fn small_n_fallback() -> ResolvedMember {
    ResolvedMember::from_def(&CORE_V1[0]).expect("ridge_gcv_raw must parse")
}

fn str_param<'a>(stage: &'a Stage, key: &str) -> Option<&'a str> {
    match stage.params.get(key) {
        Some(ParamValue::Str(s)) => Some(s.as_str()),
        _ => None,
    }
}

/// Compact-head parameter validation shared by the engine, the emitter plan,
/// and jope-validate. Returns (preset, adds, drop) after checking names.
pub fn compact_params(stage: &Stage) -> Result<(String, Vec<String>, Option<String>)> {
    for key in stage.params.keys() {
        if !matches!(
            key.as_str(),
            "preset" | "folds" | "repeats" | "blend" | "add1" | "add2" | "drop1"
        ) {
            bail!("portfolio: unknown parameter '{key}'");
        }
    }
    let preset = match str_param(stage, "preset") {
        None => "core".to_string(),
        Some("core") => "core".to_string(),
        Some("none") => "none".to_string(),
        Some(other) => bail!("portfolio: unknown preset '{other}' (expected core or none)"),
    };
    if let Some(v) = stage.params.get("folds") {
        if v.as_usize() != Some(5) {
            bail!("portfolio: folds must be 5");
        }
    }
    match stage.params.get("repeats") {
        None => {}
        Some(ParamValue::Str(s)) if s == "auto" => {}
        Some(v) if v.as_usize() == Some(1) => {}
        Some(_) => bail!("portfolio: repeats must be auto or 1"),
    }
    if let Some(v) = str_param(stage, "blend") {
        if v != "nnls" {
            bail!("portfolio: blend must be nnls (caruana is not implemented in v1)");
        }
    } else if stage.params.contains_key("blend") {
        bail!("portfolio: blend must be nnls");
    }
    let mut adds = Vec::new();
    for key in ["add1", "add2"] {
        if let Some(v) = stage.params.get(key) {
            let name = match v {
                ParamValue::Str(s) => s.as_str(),
                _ => bail!("portfolio: {key} must be a registry member name"),
            };
            if find(name).is_none() {
                bail!("portfolio: {key}='{name}' is not in the {REGISTRY_VERSION} registry");
            }
            if adds.iter().any(|a| a == name) {
                bail!("portfolio: duplicate add '{name}'");
            }
            adds.push(name.to_string());
        }
    }
    let drop = match stage.params.get("drop1") {
        None => None,
        Some(ParamValue::Str(s)) => {
            if find(s).is_none() {
                bail!("portfolio: drop1='{s}' is not in the {REGISTRY_VERSION} registry");
            }
            Some(s.clone())
        }
        Some(_) => bail!("portfolio: drop1 must be a registry member name"),
    };
    for a in &adds {
        if Some(a) == drop.as_ref() {
            bail!("portfolio: '{a}' is both added and dropped");
        }
    }
    Ok((preset, adds, drop))
}

/// True iff the compact stage asked for repeats=1 (default: auto).
pub fn compact_repeats_auto(stage: &Stage) -> bool {
    !matches!(stage.params.get("repeats"), Some(v) if v.as_usize() == Some(1))
}

/// Expand the compact head into the resolved member roster (pre-gating).
/// `preset=core` starts from the full registry, `preset=none` from empty;
/// `drop1` removes by name, then `add1`/`add2` force-include registry members
/// (an add is an explicit request: it overrides the member's dataset gate to
/// Always, appending the member when the preset did not already carry it).
pub fn resolve_compact(stage: &Stage) -> Result<Vec<ResolvedMember>> {
    let (preset, adds, drop) = compact_params(stage)?;
    let mut roster: Vec<ResolvedMember> = Vec::new();
    if preset == "core" {
        for def in &CORE_V1 {
            if Some(def.name) == drop.as_deref() {
                continue;
            }
            roster.push(ResolvedMember::from_def(def)?);
        }
    }
    for name in &adds {
        if let Some(existing) = roster.iter_mut().find(|m| &m.name == name) {
            existing.gate = Gate::Always;
            continue;
        }
        let def = find(name).expect("validated above");
        let mut m = ResolvedMember::from_def(def)?;
        m.gate = Gate::Always;
        roster.push(m);
    }
    if roster.is_empty() {
        bail!("portfolio: preset=none needs at least one add");
    }
    Ok(roster)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::parse_param_value;

    fn stage(params: &[(&str, &str)]) -> Stage {
        let mut s = Stage::new("portfolio");
        for (k, v) in params {
            s.params.insert(k.to_string(), parse_param_value(v));
        }
        s
    }

    #[test]
    fn registry_pipes_all_parse_and_end_in_heads() {
        for def in &CORE_V1 {
            match def.kind {
                MemberKind::Pipe => {
                    let dsl = parse_dsl(def.pipe)
                        .unwrap_or_else(|e| panic!("member {} does not parse: {e}", def.name));
                    let last = dsl.stages.last().unwrap();
                    assert!(
                        crate::heads::is_head_op(&last.op),
                        "member {} does not end in a head",
                        def.name
                    );
                    assert!(
                        dsl.stages.iter().all(|s| s.op != "portfolio"),
                        "member {} must not nest portfolios",
                        def.name
                    );
                }
                MemberKind::GroupMean => assert!(def.pipe.is_empty()),
            }
        }
        // Names are unique.
        for (i, a) in CORE_V1.iter().enumerate() {
            for b in &CORE_V1[i + 1..] {
                assert_ne!(a.name, b.name);
            }
        }
    }

    #[test]
    fn gates_never_exceed_the_member_cap() {
        // Adversarial sweep over (n, p, dup) shows the active roster never
        // exceeds MAX_ACTIVE_MEMBERS (n>=500 and n<150 are exclusive).
        for n in [5, 20, 79, 80, 149, 150, 299, 300, 499, 500, 5000] {
            for p in [1, 40, 41, 300, 301, 1000] {
                for dup in [0.0, 0.31] {
                    let active = CORE_V1.iter().filter(|m| m.gate.passes(n, p, dup)).count();
                    assert!(
                        active <= MAX_ACTIVE_MEMBERS,
                        "n={n} p={p} dup={dup}: {active}"
                    );
                    assert!(active >= 8);
                }
            }
        }
    }

    #[test]
    fn compact_resolution_and_validation() {
        // Defaults: full core roster.
        let r = resolve_compact(&stage(&[])).unwrap();
        assert_eq!(r.len(), 13);
        // Drop + add.
        let r = resolve_compact(&stage(&[("drop1", "gam_tbe"), ("add1", "histgbm_slow")])).unwrap();
        assert_eq!(r.len(), 12);
        assert!(r.iter().all(|m| m.name != "gam_tbe"));
        // The added member's gate is overridden to Always (force-include).
        let slow = r.iter().find(|m| m.name == "histgbm_slow").unwrap();
        assert_eq!(slow.gate, Gate::Always);
        // Untouched members keep their registry gates.
        let iso = r.iter().find(|m| m.name == "gpr_iso").unwrap();
        assert_eq!(iso.gate, Gate::NBelow(150));
        // Bad names / params rejected.
        assert!(resolve_compact(&stage(&[("add1", "bogus")])).is_err());
        assert!(resolve_compact(&stage(&[("drop1", "bogus")])).is_err());
        assert!(resolve_compact(&stage(&[("folds", "3")])).is_err());
        assert!(resolve_compact(&stage(&[("blend", "caruana")])).is_err());
        assert!(resolve_compact(&stage(&[("repeats", "2")])).is_err());
        assert!(resolve_compact(&stage(&[("preset", "none")])).is_err());
        assert!(resolve_compact(&stage(&[("nonsense", "1")])).is_err());
        assert!(resolve_compact(&stage(&[("add1", "gpr_iso"), ("add2", "gpr_iso")])).is_err());
        // preset=none with adds works.
        let r = resolve_compact(&stage(&[("preset", "none"), ("add1", "ridge_gcv_raw")])).unwrap();
        assert_eq!(r.len(), 1);
        // Adding a member already in the core roster is an idempotent
        // gate-override, never a duplicate.
        let r = resolve_compact(&stage(&[("add1", "ridge_gcv_raw")])).unwrap();
        assert_eq!(r.len(), 13);
        // repeats=1 accepted and detected.
        assert!(compact_repeats_auto(&stage(&[])));
        assert!(!compact_repeats_auto(&stage(&[("repeats", "1")])));
        assert!(compact_repeats_auto(&stage(&[("repeats", "auto")])));
    }
}
