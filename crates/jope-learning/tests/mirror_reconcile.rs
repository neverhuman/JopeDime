//! A mirror that is merely *behind* must be able to catch up; a mirror that has *diverged* must
//! still be refused.
//!
//! Making the `HEAD.json` write conditional on the writer's own pre-append HEAD stops a second
//! host's batch being orphaned, but on its own it is a one-way door: after a single
//! `hwm ingest` run without `--store` (or one transient push failure) the local pre-append HEAD
//! never equals the mirror's again, and every later push — and every retry of it — fails with
//! `mirror pointer changed`. There is no push/pull/backfill verb to get out of it, so that host's
//! mirror is wedged for good while each retry seals another local batch and leaves another object
//! on the mirror that the mirror's own HEAD cannot reach.
//!
//! `a_local_only_ingest_no_longer_wedges_the_mirror` is the three-step reproduction; it fails on
//! the pre-fix code at step 3 with rc=2. `a_diverged_mirror_is_refused_and_nothing_is_written` is
//! the negative control: catching up must not become "overwrite whatever is there".
//!
//! Reconciling correctly is not enough on its own, because `HEAD.json` is also the *only* input the
//! reconcile has: `plan` returns the batches after the mirror's tip and nothing else, so anything a
//! push skips under a tip it has already moved is never offered to a later push.
//! `a_failed_blob_push_leaves_the_mirrors_head_where_it_was` is the ordering that follows from
//! that, with an object-store failure injected mid-push.
//!
//! That ordering covers a blob missing from the *mirror*. It does not cover a blob missing from
//! the *local store*, which the push used to skip in silence —
//! `a_blob_the_local_store_cannot_supply_is_never_published_over` is that case,
//! `a_blob_only_the_mirror_still_has_does_not_wedge_the_push` is its negative control, and
//! `a_wrong_object_under_the_immutable_key_cannot_stand_in_for_the_blob` is the boundary between
//! the two: the mirror backs the batch only when its bytes *are* the record.

mod common;

use common::{jl, Run};
use jope_learning::hwm::{self, Candidate};
use jope_learning::ledger::Ledger;
use jope_learning::metric::Metric;
use jope_learning::mirror::Mirror;
use jope_learning::record::*;
use jope_learning::state::{PointerKey, State};
use serde_json::json;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

// --------------------------------------------------------------------------------------
#[path = "mirror_reconcile/mirror_blob_reconcile.rs"]
mod mirror_blob_reconcile;
#[path = "mirror_reconcile/mirror_cli.rs"]
mod mirror_cli;
#[path = "mirror_reconcile/mirror_head_reconcile.rs"]
mod mirror_head_reconcile;
