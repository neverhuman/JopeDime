//! Verify and render the committed core-IP remediation records under
//! `docs/internal/core-ip/`.
//!
//! Each module is the Rust port of one former `docs/internal/core-ip/tools/*.py` verifier and
//! keeps its rules and failure messages, so records accepted or refused there are accepted or
//! refused here for the same reason:
//!
//! | Module | Command | Record |
//! |---|---|---|
//! | [`findings`] | `verify-findings` | `findings/register.json` + `findings/sources/` |
//! | [`dispositions`] | `verify-dispositions` | `findings/dispositions.json` + `findings/evidence/` |
//! | [`claims`] | `verify-claims` | `claims/claims.json` |
//! | [`claim_inventory`] | `claim-inventory` | `claims/claims.json` → `claims/INVENTORY.md` |
//! | [`test_inventory`] | `verify-test-inventory` | `findings/test-inventory.json` vs `deploy/ci/coverage.json` |
//! | [`task_status`] | `task-status` | `work-orders/task-status.json` → `work-orders/STATUS.md` |
//! | [`progress_view`] | `progress-view` | all of the above → `work-orders/PROGRESS.md` |
//! | [`inventory`] | `verify-inventory` | `data/tensors.json` vs the accepted safetensors asset |
//! | [`elo_history`] | `elo-history` | `elo-history/histories.json` + its digest-pinned inputs |
//!
//! Records were written and read by Python, so [`pyjson`], [`pycsv`] and [`py`] reproduce the
//! parts of Python's `json` and `csv` modules and built-ins the rules depend on
//! (document-ordered objects, `==`, truthiness, `repr`, `json.dumps`, `round`, `math.fsum`).
//! These verifiers check record structure and evidence binding; they qualify no product
//! behaviour and close no finding by themselves.

pub mod claim_inventory;
pub mod claims;
pub mod dispositions;
pub mod elo_history;
mod error;
pub mod findings;
pub mod inventory;
pub mod layout;
pub mod progress_view;
pub mod py;
pub mod pycsv;
pub mod pyjson;
pub mod synthetic;
pub mod task_status;
pub mod test_inventory;

pub use error::{Error, Result};
