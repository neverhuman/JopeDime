//! Opt-in engine methods for the profile route. The default selects none of them, and a run with
//! the default records nothing new, so its receipt and sidecars keep their bytes.

use anyhow::{bail, ensure, Result};
use feat_math::kernel::expression_search::{LmJacobianCharge, LmOptions};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::search::Refinement;
// laneD:B18-pareto-archive begin
use super::archive::COST_VERSION as ARCHIVE_COST_VERSION;
// laneD:B18-pareto-archive end

// laneD:B17-lm-refinement begin
/// The damped least-squares constant refinement's stable method id.
pub const DAMPED_LS: &str = "feat-math.constant-refinement.damped-ls.v1";

/// The fewest objective trials per program the damped least-squares selection accepts. It is a
/// floor for the selection, not the method's minimum: a `derive` stage is fitted only when its
/// remaining allowance holds the two whole-pipeline passes, the surrogate's initial value, one
/// Jacobian, one trial and the re-score (literal sites plus five under the default `sites` charge,
/// six under `one`). Under that, and for every `map` stage, the coordinate search runs and the
/// trace names why, so with fewer than six trials no stage is fitted by the method.
pub const DAMPED_LS_MIN_TRIALS: usize = 3;

/// How a stage's literals are refined.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum RefineMethod {
    /// The coordinate search.
    #[default]
    Coordinate,
    /// Damped least squares on the stage's training-prefix rows, kept only when the whole
    /// pipeline's training loss strictly falls. `map` stages, and stages whose remaining allowance
    /// is under the method's minimum, use the coordinate search.
    Lm(LmOptions),
}

/// Every opt-in engine method. The default is all off.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EngineSelection {
    pub refine: RefineMethod,
    // laneD:B18-pareto-archive begin
    /// The loss × cost front of the run's measured programs; none by default. Only publication
    /// builds it (`run::run`, through `publication::publish`): `search::run_cut` records the
    /// selection in its result's `engine` and builds nothing, and its search is the same with or
    /// without it. A caller that runs cuts itself builds the front with `archive::build`.
    pub archive: Option<ArchivePolicy>,
    // laneD:B18-pareto-archive end
}

impl EngineSelection {
    /// Refuse options the methods cannot run with. The search options add [`Self::validate`].
    pub fn check_options(&self) -> Result<()> {
        if let RefineMethod::Lm(options) = self.refine {
            ensure!(
                options.tau.is_finite() && options.tau > 0.0,
                "lm refinement needs a finite positive tau"
            );
        }
        Ok(())
    }

    /// Refuse a selection under its floor: no refinement parent, or fewer than
    /// [`DAMPED_LS_MIN_TRIALS`] trials per program. An accepted allowance can still be under a
    /// stage's minimum; that stage then uses the coordinate search.
    pub fn validate(&self, refinement: Refinement) -> Result<()> {
        self.check_options()?;
        if matches!(self.refine, RefineMethod::Lm(_)) {
            ensure!(
                refinement.top_k > 0 && refinement.trials_per_program >= DAMPED_LS_MIN_TRIALS,
                "lm refinement requires --refine-top-k of at least 1 and --refine-trials of at \
                 least 3; a derive stage with fewer trials left than its surrogate minimum \
                 (literal sites plus five under the default charge) uses the coordinate search"
            );
        }
        Ok(())
    }

    /// The receipt record: `None` for the default, so a default receipt keeps its bytes.
    pub fn record(&self) -> Option<EngineRecord> {
        let refine = match self.refine {
            RefineMethod::Coordinate => None,
            RefineMethod::Lm(options) => Some(RefineRecord {
                id: DAMPED_LS.into(),
                jacobian: match options.jacobian {
                    LmJacobianCharge::Sites => "sites",
                    LmJacobianCharge::One => "one",
                }
                .into(),
                tau: options.tau,
            }),
        };
        // laneD:B18-pareto-archive begin
        let archive = self.archive.map(ArchiveRecord::selected);
        (refine.is_some() || archive.is_some()).then_some(EngineRecord { refine, archive })
        // laneD:B18-pareto-archive end
    }
}

/// The opt-in engine methods a run selected. Absent from a receipt for the default selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EngineRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refine: Option<RefineRecord>,
    // laneD:B18-pareto-archive begin
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archive: Option<ArchiveRecord>,
    // laneD:B18-pareto-archive end
}

/// The constant refinement method, when it is not the coordinate search.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RefineRecord {
    pub id: String,
    /// What one Jacobian is charged: `sites` (one evaluation per literal site) or `one`.
    pub jacobian: String,
    /// The initial damping relative to the largest scaled curvature.
    pub tau: f64,
}

impl EngineRecord {
    /// The selection this record names. Unknown ids or values, and a record the selection would
    /// not write back byte for byte, are refused.
    pub fn selection(&self) -> Result<EngineSelection> {
        let mut selection = EngineSelection::default();
        if let Some(refine) = &self.refine {
            ensure!(refine.id == DAMPED_LS, "unknown refinement method");
            let jacobian = match refine.jacobian.as_str() {
                "sites" => LmJacobianCharge::Sites,
                "one" => LmJacobianCharge::One,
                _ => bail!("unknown refinement jacobian charge"),
            };
            selection.refine = RefineMethod::Lm(LmOptions {
                tau: refine.tau,
                jacobian,
            });
        }
        // laneD:B18-pareto-archive begin
        if let Some(archive) = &self.archive {
            ensure!(
                archive.id == FRONT_LOSS_COST && archive.cost_version == ARCHIVE_COST_VERSION,
                "unknown archive method or cost version"
            );
            selection.archive = Some(ArchivePolicy::ArchiveOnly);
        }
        // The archive's digest is evidence, not selection: the selection writes its record back
        // without one, and publication adds it.
        let mut canonical = self.clone();
        if let Some(archive) = &mut canonical.archive {
            archive.sha256 = None;
        }
        // laneD:B18-pareto-archive end
        selection.check_options()?;
        ensure!(
            selection.record().as_ref() == Some(&canonical),
            "engine record is not canonical"
        );
        Ok(selection)
    }

    /// The damped least-squares record, when that method was selected.
    pub fn damped_ls(&self) -> Option<&RefineRecord> {
        self.refine.as_ref().filter(|r| r.id == DAMPED_LS)
    }
}
// laneD:B17-lm-refinement end

// laneD:B18-pareto-archive begin
/// The loss × cost front's stable method id.
pub const FRONT_LOSS_COST: &str = "feat-math.front-loss-cost.v1";

/// What a run does with the loss × cost front of its measured programs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchivePolicy {
    /// Publish the front as `search/cut-0/archive.json`, as evidence only: ranking, parents and
    /// the selected program never read it, so every measurement and refinement is unchanged.
    /// Publication writes it; `search::run_cut` alone writes and builds nothing.
    ArchiveOnly,
}

/// The loss × cost front a run publishes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArchiveRecord {
    /// The method id, `feat-math.front-loss-cost.v1`.
    pub id: String,
    /// The cost version of every member's complexity.
    pub cost_version: String,
    /// The sha256 of `search/cut-0/archive.json`. Publication adds it, so a receipt's record
    /// always has one; the selection's own record, before anything is published, has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

impl ArchiveRecord {
    /// The record a selection writes, before publication adds the digest.
    fn selected(policy: ArchivePolicy) -> Self {
        match policy {
            ArchivePolicy::ArchiveOnly => Self {
                id: FRONT_LOSS_COST.into(),
                cost_version: ARCHIVE_COST_VERSION.into(),
                sha256: None,
            },
        }
    }
}
// laneD:B18-pareto-archive end
