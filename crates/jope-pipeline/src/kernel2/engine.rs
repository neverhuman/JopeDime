//! Opt-in engine methods for the profile route. The default selects none of them, and a run with
//! the default records nothing new, so its receipt and sidecars keep their bytes.

use anyhow::{bail, ensure, Result};
use feat_math::kernel::expression_search::{LmJacobianCharge, LmOptions};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::search::Refinement;

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
        refine.map(|refine| EngineRecord {
            refine: Some(refine),
        })
    }
}

/// The opt-in engine methods a run selected. Absent from a receipt for the default selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EngineRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refine: Option<RefineRecord>,
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
        selection.check_options()?;
        ensure!(
            selection.record().as_ref() == Some(self),
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
