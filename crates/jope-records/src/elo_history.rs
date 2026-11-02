//! `elo-history`: replay retained historical Elo ladders and keep unrecoverable history open
//! (W12-T04).
//!
//! Every history in `elo-history/histories.json` carries exactly one disposition:
//!
//! - `reproducible-with-receipts`: the original scores, track and registries are retained with
//!   digests. This module re-implements the documented procedure independently (aggregated pair
//!   statistics, not the per-match port in `crates/jope-arena/src/elo.rs`, which it never calls)
//!   and must reproduce the committed leaderboard bytes and snapshot rows from those inputs alone.
//! - `scorer-parity-only`: a current scorer agrees with a reference implementation on fixtures.
//!   That is implementation parity; no historical match graph was recovered.
//! - `irrecoverable`: original matches, participants or settings are not retained. The retained
//!   observations may be quoted as attributable history; ranking, rating and continuity claims
//!   are refused.
//!
//! Every rating is historical-ranking-only and is refused to teaching, HWM, transfer and
//! promotion consumers. Ratings are relative to one method universe, so continuity across
//! histories is refused. Lineage tables are recomputed from their retained bytes: a parent edge
//! whose parent is not retained blocks continuity across it, and exposure totals that need
//! unretained corpus membership are refused. A digest authenticates retained bytes, not the truth
//! of the producer that wrote them.
//!
//! Numerics follow the replay this port replaces operation for operation: IEEE-754 doubles in
//! the same accumulation order, `exp`/`log` from the platform C library, CPython's `math.fsum`
//! for the centring mean, and Python's `round`/`repr` spellings in every written cell.

include!("elo_history/shared.rs");
include!("elo_history/ratings.rs");
include!("elo_history/replay.rs");
include!("elo_history/validation.rs");
include!("elo_history/lineage.rs");
include!("elo_history/continuity.rs");
