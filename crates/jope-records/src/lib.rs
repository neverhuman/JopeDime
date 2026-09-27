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
//! | [`test_inventory`] | `verify-test-inventory` | `findings/test-inventory.json` vs `deploy/ci/coverage.json` |
//! | [`task_status`] | `task-status` | `work-orders/task-status.json` → `work-orders/STATUS.md` |
//! | [`progress_view`] | `progress-view` | all of the above → `work-orders/PROGRESS.md` |
//! | [`inventory`] | `verify-inventory` | `data/tensors.json` vs the accepted safetensors asset |
//! | [`elo_history`] | `elo-history` | `elo-history/histories.json` + its digest-pinned inputs |
//!
//! Four repository hygiene guards live here too, ported from `deploy/ci` Python (archived at
//! `8a55c4fef`) with their output and exit statuses unchanged:
//!
//! | Module | Command | Guards |
//! |---|---|---|
//! | [`coverage`] | `coverage discover\|validate` | `deploy/ci/coverage.json` against Cargo and git |
//! | [`scope_check`] | `rust-scope-check` | `deploy/ci/rust-test-scope.sh` partitions the workspace |
//! | [`parity`] | `parity-check` | `docs/parity/capability-matrix.md` rows and counts |
//! | [`readme`] | `readme-benchmarks --check\|--write` | the README's measured-results blocks |
//!
//! Records were written and read by Python, so [`pyjson`], [`pycsv`] and [`py`] reproduce the
//! parts of Python's `json` and `csv` modules and built-ins the rules depend on
//! (document-ordered objects, `==`, truthiness, `repr`, `json.dumps`, `round`, `math.fsum`).
//! These verifiers check record structure and evidence binding; they qualify no product
//! behaviour and close no finding by themselves.

pub mod claims;
pub mod coverage;
pub mod dispositions;
pub mod elo_history;
mod error;
pub mod findings;
pub mod inventory;
pub mod layout;
pub mod parity;
pub mod progress_view;
pub mod py;
pub mod pycsv;
pub mod pyjson;
pub mod readme;
pub mod scope_check;
pub mod synthetic;
pub mod task_status;
pub mod test_inventory;

pub use error::{Error, Result};
