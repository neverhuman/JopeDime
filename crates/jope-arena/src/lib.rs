//! SOTArena plumbing for JopeDime (plan WP12; red-team item 8; blockers B18/B20).
//!
//! Two different "sotarena" trees existed historically and are kept apart here:
//!
//! * the **benchmark clone** (`SOTARENA_ROOT`, default `$JOPE_HOME/benchmarks/sotarena`):
//!   `registry/`, `tracks/<track>/track.json`, `results/<track>[/<version>]/{scores.csv,
//!   leaderboard.csv, evidence-manifest.json, resolution-ledger.jsonl, …}` — the data that proves
//!   `jope-dime`'s rank-1 Elo on `tabular-regression-v2`. [`leaderboard`] recomputes its official
//!   ladder byte-for-byte; [`verify`] checks its evidence manifest; [`publish`] snapshots a results
//!   directory into an object store as `scores/sotarena/<track>/<version>/…`.
//! * the **Elo reporter** (`SOTARENA_REPORTER_ROOT`, default `$JOPE_STATE/arena`): `config.json` +
//!   `data/<task>/<hash>/{dataset.json,results/*.json}` — the adaptive / best-known family tables.
//!   [`reporter`] loads it, [`family`] merges methods into families and rates them, [`fetch`]
//!   downloads its datasets with SHA-256 verification into `$JOPE_HOME/benchmarks/cache`.
//!
//! [`scores`] is the `scores.csv` schema both trees share, plus emission from campaign receipts.
//! [`elo`] holds both Bradley–Terry fits: the per-match one that produced the committed
//! leaderboards and the reporter's pair-aggregated one; they agree to floating-point noise.

pub mod elo;
pub mod family;
pub mod fetch;
pub mod io;
pub mod leaderboard;
pub mod paths;
pub mod publish;
pub mod pyfmt;
pub mod reporter;
pub mod scores;
pub mod verify;
