//! W8-T04: reconcile a run publication that lost a race.
//!
//! When two publishers race the same predecessor, one commit wins the run pointer and the other
//! gets an explicit [`Publication::Conflict`]. The losing commit stays on the remote as an
//! immutable, unreferenced object and in its outbox, and nothing is overwritten (W8-T03). The
//! losing host is then stuck: its next commit names the stale parent, and a commit on the winner
//! names a commit its outbox never recorded.
//!
//! Reconciliation is the procedure the offline fault matrix proved (W8-T06), as an operator
//! verb:
//!
//! 1. adopt the winner's committed closure into the local outbox ([`restore`]), verifying every
//!    item again;
//! 2. commit this host's own work that the committed closure does not contain on top of the
//!    winner, through the ordinary publisher and its receiving checks.
//!
//! Nothing is rewritten or deleted: the winner is never displaced, the losing commit stays where
//! it is, and work the winner already committed is not committed again.
//!
//! [`plan`] is read-only. It reads the remote and this host's outbox and reports:
//! - the committed head;
//! - this host's commits that lost;
//! - what the restore would adopt;
//! - the work a rebased commit would carry;
//! - whether that commit would pass the graph rules ([`check_rules`]) and the HWM-ledger check.
//!
//! It writes nothing, and refuses when there is no local outbox rather than create one.
//! [`apply`] performs the plan, and only an admissible one.

use crate::ledger::Ledger;
use crate::progress::inventory::{
    blob_key, decode_as, encode, Commit, Dependency, Encoded, Head, HwmDecision, Kind, ObjectRef,
    RunManifest, MAX_REFS,
};
use crate::progress::outbox::{Limits, Outbox};
use crate::progress::product::{Config, Driver};
use crate::progress::publish::{check_rules, reusable, Graph, Publication};
use crate::progress::rebuild::{restore, Restored};
use crate::progress::remote::Remote;
use crate::state::State;
use anyhow::{bail, ensure, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// What reconciliation found, and with `--apply`, what it did.
#[derive(Clone, Debug, Serialize)]
pub struct Plan {
    /// Identifier for the associated run.
    pub run_id: String,
    /// The run pointer: the winner.
    pub head: Option<Head>,
    /// This host's commits of the run that the committed chain does not contain: races it lost.
    pub divergent: Vec<String>,
    /// Committed items this host's outbox does not hold; the restore adopts them.
    pub adopt: Vec<String>,
    /// This host's objects of the run that the committed closure does not contain: the work a
    /// rebased commit carries.
    pub recommit: Vec<String>,
    /// The rebased commit would pass the graph rules and the HWM-ledger check.
    pub admissible: bool,
    /// Why it would not, or why there is nothing to do.
    pub refusal: Option<String>,
    /// Only with apply: what the restore and the publication did.
    pub applied: Option<Applied>,
}

#[derive(Clone, Debug, Serialize)]
/// `Applied` is a record for the applied contract in this module: W8-T04: reconcile a run publication that lost a race.
pub struct Applied {
    /// Optional restored value; `None` means it is not available.
    pub restored: Option<Restored>,
    /// Optional publication value; `None` means it is not available.
    pub publication: Option<Publication>,
}

/// This host's objects of `run_id`, by reference.
fn local_objects(outbox: &Outbox, run_id: &str) -> Result<BTreeMap<ObjectRef, Encoded>> {
    let mut out = BTreeMap::new();
    for seq in 1..=outbox.records() {
        if let Some(o) = outbox.object(seq)? {
            if o.run_id == run_id {
                out.insert(o.reference(), o);
            }
        }
    }
    Ok(out)
}

fn plan_with(outbox: &Outbox, remote: &dyn Remote, run_id: &str, publisher: &str) -> Result<Plan> {
    let local = local_objects(outbox, run_id)?;
    let manifests: Vec<&Encoded> = local
        .values()
        .filter(|o| o.kind == Kind::RunManifest)
        .collect();
    let [manifest] = manifests.as_slice() else {
        bail!(
            "this host holds {} manifests for run {run_id}; reconciliation needs exactly one",
            manifests.len()
        );
    };
    let manifest = decode_as::<RunManifest>(&manifest.bytes, &manifest.digest)?;
    ensure!(
        manifest.value().publisher == publisher,
        "{publisher} is not the publisher run {run_id} names ({})",
        manifest.value().publisher
    );
    let committed = reusable(remote, run_id)?;
    let (head, graph) = match committed {
        Some(c) => (Some(c.head), c.graph),
        None => (None, Graph::default()),
    };
    // The committed chain, commit by commit from the head.
    let mut chain = BTreeSet::new();
    let mut next = head.as_ref().map(|h| h.commit.clone());
    while let Some(c) = next {
        next = graph
            .get::<Commit>(&c)
            .and_then(|commit| commit.value().parent.clone());
        chain.insert(c);
    }
    let divergent: Vec<String> = local
        .values()
        .filter(|o| o.kind == Kind::Commit && !chain.contains(&o.digest))
        .map(|o| o.digest.clone())
        .collect();
    let mut adopt: Vec<String> = graph
        .objects
        .keys()
        .filter(|r| outbox.seq_of(&Dependency::Object((*r).clone())).is_none())
        .map(|r| crate::progress::inventory::object_key(r.kind, &r.digest))
        .collect();
    adopt.extend(
        graph
            .blobs
            .iter()
            .filter(|b| outbox.seq_of(&Dependency::Blob((*b).clone())).is_none())
            .map(|b| blob_key(&b.sha256)),
    );
    let recommit: Vec<&Encoded> = local
        .values()
        .filter(|o| !matches!(o.kind, Kind::RunManifest | Kind::Commit | Kind::Summary))
        .filter(|o| !graph.objects.contains_key(&o.reference()))
        .collect();
    let mut plan = Plan {
        run_id: run_id.into(),
        head: head.clone(),
        divergent,
        adopt,
        recommit: recommit.iter().map(|o| o.key()).collect(),
        admissible: false,
        refusal: None,
        applied: None,
    };
    if recommit.is_empty() {
        plan.refusal = Some(
            "nothing to reconcile: the committed closure holds all of this host's work".into(),
        );
        return Ok(plan);
    }
    if recommit.len() > MAX_REFS {
        plan.refusal = Some(format!(
            "{} objects to re-commit, more than one commit may add ({MAX_REFS})",
            recommit.len()
        ));
        return Ok(plan);
    }
    // The rebased commit, checked exactly as publication would check its graph.
    let mut objects: Vec<ObjectRef> = recommit.iter().map(|o| o.reference()).collect();
    objects.sort();
    let rebased = encode(&Commit {
        schema: Kind::Commit.schema().into(),
        run_id: run_id.into(),
        run_manifest: manifest.digest().into(),
        parent: head.as_ref().map(|h| h.commit.clone()),
        height: head.as_ref().map_or(0, |h| h.height + 1),
        objects,
    })?;
    let mut would_be = graph.clone();
    for o in &recommit {
        would_be.objects.insert(o.reference(), o.bytes.clone());
    }
    would_be
        .objects
        .insert(rebased.reference(), rebased.bytes.clone());
    if let Err(e) = check_rules(&would_be, &rebased.reference()) {
        plan.refusal = Some(format!(
            "the rebased commit would not pass the graph rules: {e:#}"
        ));
        return Ok(plan);
    }
    let ledger = Ledger::open(outbox.state().root(), "hwm");
    for o in recommit.iter().filter(|o| o.kind == Kind::HwmDecision) {
        let d = decode_as::<HwmDecision>(&o.bytes, &o.digest)?;
        if let Err(e) = d.value().verify_against(&ledger) {
            plan.refusal = Some(format!(
                "{} is not a decision this host's HWM ledger recorded: {e:#}",
                o.key()
            ));
            return Ok(plan);
        }
    }
    plan.admissible = true;
    Ok(plan)
}

/// Read-only: what reconciling `run_id` as `publisher` would do. Refuses when this host has no
/// outbox rather than create one.
pub fn plan(
    state: &State,
    remote: &dyn Remote,
    run_id: &str,
    publisher: &str,
    limits: Limits,
) -> Result<Plan> {
    ensure!(
        Outbox::dir(state).is_dir(),
        "{} holds no progress outbox: there is nothing on this host to reconcile",
        state.root().display()
    );
    let outbox = Outbox::open(state, limits)?;
    plan_with(&outbox, remote, run_id, publisher)
}

/// Perform an admissible plan: adopt the winner's committed closure, then commit this host's
/// remaining work on top of it through the ordinary publisher. An inadmissible plan is returned
/// unapplied, with nothing written.
pub fn apply(
    state: &State,
    remote: &dyn Remote,
    config: Config,
    run_id: &str,
    publisher: &str,
    now_ms: u64,
) -> Result<Plan> {
    let mut plan = plan(state, remote, run_id, publisher, config.limits)?;
    if !plan.admissible {
        return Ok(plan);
    }
    let restored = {
        let mut outbox = Outbox::open(state, config.limits)?;
        restore(remote, run_id, &mut outbox)?
    };
    let mut driver = Driver::open(state, remote, config, run_id, publisher, now_ms)?;
    let publication = driver.commit()?;
    plan.applied = Some(Applied {
        restored,
        publication,
    });
    Ok(plan)
}
