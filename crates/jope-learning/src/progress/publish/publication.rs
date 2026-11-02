use super::*;

/// The outcome of one publication.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "outcome")]
pub enum Publication {
    /// The pointer moved from `from` to the commit.
    Advanced {
        /// Commit recorded for this publication.
        commit: String,
        /// Height recorded for this publication.
        height: u64,
        /// Optional from value; `None` means it is not available.
        from: Option<String>,
    },
    /// The pointer already named this commit, as after a retry once a response was lost.
    /// Field value recorded by this public API type.
    AlreadyPublished {
        /// Commit recorded for this outcome.
        commit: String,
    },
    /// The pointer names something other than the commit's parent. Nothing was written. The
    /// commit stays on the remote and in the outbox for reconciliation.
    Conflict {
        /// Commit recorded for this publication.
        commit: String,
        /// Optional expected value; `None` means it is not available.
        expected: Option<String>,
        /// Optional found value; `None` means it is not available.
        found: Option<String>,
    },
    /// The commit's closure is not durable and verified on the remote, or the pointer is not a
    /// valid pointer of this run. The pointer was not moved.
    /// Field value recorded by this public API type.
    /// Field value recorded by this public API type.
    Blocked {
        /// Commit recorded for this outcome.
        commit: String,
        /// Explanation accompanying this outcome.
        reason: String,
    },
}

/// Every digest `commit`'s closure reaches through this outbox's records: each item's reference
/// digest and the sha256 of its recorded bytes (an object's bytes and domain digest differ).
fn local_closure_digests(outbox: &Outbox, commit: &Encoded) -> Vec<String> {
    let mut digests = vec![commit.digest.clone(), sha256_hex(&commit.bytes)];
    let mut seen = BTreeSet::new();
    let mut pending: Vec<Dependency> = commit.dependencies.clone();
    while let Some(item) = pending.pop() {
        if !seen.insert(item.clone()) {
            continue;
        }
        digests.push(match &item {
            Dependency::Blob(b) => b.sha256.clone(),
            Dependency::Object(o) => o.digest.clone(),
        });
        if let Some(record) = outbox.seq_of(&item).and_then(|seq| outbox.record(seq)) {
            digests.push(record.bytes_sha256.clone());
            pending.extend(record.dependencies.iter().cloned());
        }
    }
    digests
}

/// A publication that has done everything except move the pointer.
#[derive(Debug)]
pub struct Plan {
    /// Identifier for the associated run.
    pub run_id: String,
    /// Head key recorded for this plan.
    pub head_key: String,
    /// Commit recorded for this plan.
    pub commit: String,
    /// Height recorded for this plan.
    pub height: u64,
    /// Optional parent value; `None` means it is not available.
    pub parent: Option<String>,
    /// Objects missing remotely that were uploaded again from the local outbox.
    pub reuploaded: Vec<String>,
    /// Remote reads made while verifying the closure.
    pub reads: u64,
    observed: Option<ObservedObject>,
}

#[derive(Debug)]
/// `Prepared` is an enum for the prepared contract in this module: W8-T03: dependency-first conditional publication.
pub enum Prepared {
    /// The `Ready` case of `Prepared`.
    Ready(Plan),
    /// The `Done` case of `Prepared`.
    Done(Publication),
}

/// Steps 1 and 2, then one read of the pointer: everything up to, but not including, the pointer
/// write. Refuses (errors, writing nothing) when `publisher` is not the run's publisher or the
/// commit is not the run's, and when the commit names anything the outbox has not recorded.
pub fn prepare(
    outbox: &mut Outbox,
    remote: &dyn Remote,
    publisher: &str,
    manifest: &Decoded<RunManifest>,
    commit: &Encoded,
) -> Result<Prepared> {
    ensure!(
        commit.kind == Kind::Commit,
        "only a commit advances a run pointer"
    );
    let c = decode_as::<Commit>(&commit.bytes, &commit.digest)?.into_value();
    let m = manifest.value();
    ensure!(
        publisher == m.publisher,
        "{publisher} is not the publisher of run {} ({}); nothing was written",
        m.run_id,
        m.publisher
    );
    ensure!(
        c.run_id == m.run_id && c.run_manifest == manifest.digest(),
        "commit {} belongs to run {} manifest {}, not {} {}; nothing was written",
        commit.digest,
        c.run_id,
        c.run_manifest,
        m.run_id,
        manifest.digest()
    );
    let blocked = |reason: String| {
        Ok(Prepared::Done(Publication::Blocked {
            commit: commit.digest.clone(),
            reason,
        }))
    };

    // W12-T05: nothing revoked is delivered or committed. The commit's whole closure, ancestors
    // included, is checked as this outbox recorded it, before anything is appended or delivered.
    // The outbox records every dependency before its dependent and delivery verifies that the
    // remote holds exactly those bytes, so the verified remote closure is the same set.
    let revocations = Revocations::load(outbox.state())?;
    if !revocations.is_empty() {
        if let Err(e) = revocations.refuse(
            &format!("commit {}", commit.digest),
            local_closure_digests(outbox, commit)
                .iter()
                .map(String::as_str),
        ) {
            return blocked(format!("{e:#}; nothing was delivered"));
        }
    }

    outbox.append_object(commit)?;
    let delivery = outbox.deliver(remote, None)?;
    if let Some(f) = delivery.failure {
        return blocked(format!(
            "delivery of outbox record {} ({}) failed: {}",
            f.seq, f.key, f.error
        ));
    }

    let root = Dependency::Object(ObjectRef {
        kind: Kind::Commit,
        digest: commit.digest.clone(),
    });
    let mut reuploaded: Vec<String> = Vec::new();
    let graph = loop {
        let error = match verify_closure(remote, &root) {
            Ok(graph) => break graph,
            Err(error) => error,
        };
        let Some(d) = error.downcast_ref::<Defect>() else {
            return Err(error);
        };
        if d.kind != DefectKind::Missing || reuploaded.contains(&d.key) {
            return blocked(format!(
                "the commit's closure does not verify on the remote: {d}"
            ));
        }
        match outbox.redeliver(remote, &d.key) {
            Ok(true) => reuploaded.push(d.key.clone()),
            Ok(false) => {
                return blocked(format!(
                    "the commit's closure does not verify on the remote: {d}; this outbox \
                     cannot supply it either"
                ))
            }
            Err(e) => return blocked(format!("re-uploading {} failed: {e:#}", d.key)),
        }
    };
    let reads = graph.reads;

    // An HWM decision is a link to the W3 HWM ledger, which stays the authority. Every decision
    // this commit adds must be one this host's own HWM ledger (the state tree of its outbox)
    // recorded, with that record, identity, transition and qualification. Decisions already
    // committed by an ancestor were checked by the host that added them, so a host restored from
    // the remote can publish on top of them without their batches.
    let ledger = Ledger::open(outbox.state().root(), "hwm");
    for o in c.objects.iter().filter(|o| o.kind == Kind::HwmDecision) {
        let key = object_key(o.kind, &o.digest);
        let Some(decision) = graph.get::<HwmDecision>(&o.digest) else {
            return blocked(format!("{key} is not in the verified closure"));
        };
        if let Err(e) = decision.value().verify_against(&ledger) {
            return blocked(format!(
                "{key} is not a decision this host's HWM ledger ({}) recorded: {e:#}",
                ledger.root().display()
            ));
        }
    }

    let head_key = head_key(&c.run_id)?;
    let observed = remote.get_observed(&head_key)?;
    let found = match &observed {
        None => None,
        Some(o) => match Head::decode(&o.bytes) {
            Ok(h) if h.run_id == c.run_id => Some(h),
            Ok(h) => {
                return blocked(format!(
                    "{head_key} names run {}; it is left as it is",
                    h.run_id
                ))
            }
            Err(e) => {
                return blocked(format!(
                    "{head_key} is not a valid run pointer ({e:#}); it is left as it is"
                ))
            }
        },
    };
    if found.as_ref().is_some_and(|h| h.commit == commit.digest) {
        return Ok(Prepared::Done(Publication::AlreadyPublished {
            commit: commit.digest.clone(),
        }));
    }
    if found.as_ref().map(|h| &h.commit) != c.parent.as_ref() {
        return Ok(Prepared::Done(Publication::Conflict {
            commit: commit.digest.clone(),
            expected: c.parent.clone(),
            found: found.map(|h| h.commit),
        }));
    }
    if let Some(h) = &found {
        if h.height.checked_add(1) != Some(c.height) {
            return blocked(format!(
                "{head_key} claims height {} for the parent, which is at height {}; it is left \
                 as it is",
                h.height,
                c.height - 1
            ));
        }
    }
    Ok(Prepared::Ready(Plan {
        run_id: c.run_id,
        head_key,
        commit: commit.digest.clone(),
        height: c.height,
        parent: c.parent,
        reuploaded,
        reads,
        observed,
    }))
}

/// Step 3: move the pointer from the observed predecessor to the commit, conditionally.
///
/// When the write fails, only the pointer itself can say what happened:
/// - it names the commit: the write landed and its response was lost, so the result is
///   `Advanced`;
/// - it still names the predecessor: nothing was written, and the error is returned;
/// - it names anything else: another writer won, and the result is a `Conflict`.
pub fn advance(remote: &dyn Remote, plan: Plan) -> Result<Publication> {
    let bytes = Head {
        schema: HEAD_SCHEMA.into(),
        run_id: plan.run_id.clone(),
        commit: plan.commit.clone(),
        height: plan.height,
    }
    .encode()?;
    let Err(error) = remote.put_pointer(&plan.head_key, &bytes, plan.observed.as_ref()) else {
        return Ok(Publication::Advanced {
            commit: plan.commit,
            height: plan.height,
            from: plan.parent,
        });
    };
    let now = remote
        .get_observed(&plan.head_key)?
        .map(|o| Head::decode(&o.bytes))
        .transpose()
        .with_context(|| format!("{}: unreadable after a failed write", plan.head_key))?
        .map(|h| h.commit);
    match now {
        Some(c) if c == plan.commit => Ok(Publication::Advanced {
            commit: plan.commit,
            height: plan.height,
            from: plan.parent,
        }),
        now if now == plan.parent => Err(error.context(format!(
            "{}: the conditional pointer write failed and the pointer is unchanged; nothing \
             was written, and the publication can be retried",
            plan.head_key
        ))),
        now => Ok(Publication::Conflict {
            commit: plan.commit,
            expected: plan.parent,
            found: now,
        }),
    }
}

/// Publish `commit` for `manifest`'s run: [`prepare`], then [`advance`].
pub fn publish(
    outbox: &mut Outbox,
    remote: &dyn Remote,
    publisher: &str,
    manifest: &Decoded<RunManifest>,
    commit: &Encoded,
) -> Result<Publication> {
    match prepare(outbox, remote, publisher, manifest, commit)? {
        Prepared::Ready(plan) => advance(remote, plan),
        Prepared::Done(publication) => Ok(publication),
    }
}

/// A run's remotely committed state, verified end to end.
#[derive(Debug)]
pub struct Reusable {
    /// Head recorded for this reusable.
    pub head: Head,
    /// Graph recorded for this reusable.
    pub graph: Graph,
}

/// Read the run pointer, then verify the commit it names and that commit's whole closure. Only
/// what this returns may be reused. Every failure is an error, never a partial result. That
/// includes a corrupt pointer, a dangling or misdescribed one, and any missing or corrupt
/// object, blob or broken rule.
pub fn reusable(remote: &dyn Remote, run_id: &str) -> Result<Option<Reusable>> {
    let key = head_key(run_id)?;
    let Some(bytes) = remote.get(&key)? else {
        return Ok(None);
    };
    let head = Head::decode(&bytes).with_context(|| format!("{key} is not a valid run pointer"))?;
    ensure!(head.run_id == run_id, "{key} names run {}", head.run_id);
    let root = Dependency::Object(ObjectRef {
        kind: Kind::Commit,
        digest: head.commit.clone(),
    });
    let graph = verify_closure(remote, &root).with_context(|| {
        format!(
            "{key} names commit {} whose closure does not verify",
            head.commit
        )
    })?;
    let commit = graph
        .get::<Commit>(&head.commit)
        .context("the pointer's commit did not decode")?;
    let commit = commit.value();
    if commit.height != head.height || commit.run_id != run_id {
        bail!(
            "{key} claims height {} for commit {} of run {} at height {}",
            head.height,
            head.commit,
            commit.run_id,
            commit.height
        );
    }
    Ok(Some(Reusable { head, graph }))
}
