use super::*;
use crate::progress::inventory::{encode, Commit, Head, HEAD_SCHEMA};

impl<'r> Driver<'r> {
    /// Append a mapping: every blob, then every object, staging the objects for the next commit.
    /// A mapping is admitted whole or not at all: if its new items would pass the outbox maximum,
    /// it is refused ([`Refused::Full`]) before any of them is written.
    fn append(&mut self, mapped: &Mapped, step: &mut Step) -> Result<()> {
        // W12-T05: revoked evidence is never appended, let alone published.
        let revocations = crate::revocation::Revocations::load(self.outbox.state())?;
        if !revocations.is_empty() {
            let mut digests: Vec<String> = mapped.blobs.iter().map(|b| sha256_hex(b)).collect();
            for o in &mapped.objects {
                digests.push(o.digest.clone());
                digests.push(sha256_hex(&o.bytes));
            }
            revocations.refuse("an engine publication", digests.iter().map(String::as_str))?;
        }
        let (mut entries, mut bytes) = (0u64, 0u64);
        for blob in &mapped.blobs {
            if self
                .outbox
                .seq_of(&Dependency::Blob(BlobRef::of(blob)))
                .is_none()
            {
                entries += 1;
                bytes += blob.len() as u64;
            }
        }
        for object in &mapped.objects {
            if self
                .outbox
                .seq_of(&Dependency::Object(object.reference()))
                .is_none()
            {
                entries += 1;
                bytes += object.bytes.len() as u64;
            }
        }
        let limits = self.outbox.limits();
        let (pending_entries, pending_bytes) =
            (self.outbox.pending_entries(), self.outbox.pending_bytes());
        if pending_entries + entries > limits.max_entries
            || pending_bytes + bytes > limits.max_bytes
        {
            return Err(Refused::Full {
                pending_entries,
                pending_bytes,
                item_bytes: bytes,
            }
            .into());
        }
        for blob in &mapped.blobs {
            let key = Dependency::Blob(BlobRef::of(blob)).key();
            match self.outbox.append_blob(blob)? {
                Appended::Added { .. } => step.appended.push(key),
                Appended::Existing { .. } => step.existing.push(key),
            }
        }
        for object in &mapped.objects {
            match self.outbox.append_object(object)? {
                Appended::Added { .. } => step.appended.push(object.key()),
                Appended::Existing { .. } => step.existing.push(object.key()),
            }
            let r = object.reference();
            if !self.committed.contains(&r) && object.kind != Kind::RunManifest {
                self.staged.insert(r);
            }
        }
        Ok(())
    }

    /// Commit every staged object on top of the committed tip. `None` when nothing is staged.
    /// Only an advanced (or already published) pointer moves the staged objects to committed;
    /// a conflict or a block leaves them staged, and the publication says why.
    pub fn commit(&mut self) -> Result<Option<Publication>> {
        if self.staged.is_empty() {
            return Ok(None);
        }
        let objects: Vec<ObjectRef> = self.staged.iter().cloned().collect();
        ensure!(
            objects.len() <= MAX_REFS,
            "{} objects are staged, more than one commit may add ({MAX_REFS}); flush more often",
            objects.len()
        );
        let (parent, height) = match &self.head {
            Some(h) => (
                Some(h.commit.clone()),
                h.height.checked_add(1).context("height overflow")?,
            ),
            None => (None, 0),
        };
        if let Some(parent) = &parent {
            let recorded = self.outbox.seq_of(&Dependency::Object(ObjectRef {
                kind: Kind::Commit,
                digest: parent.clone(),
            }));
            ensure!(
                recorded.is_some(),
                "run {} is committed at {parent}, which this host never recorded: another \
                 publisher advanced the run. `jope-learning progress reconcile` adopts it and \
                 commits this host's work on top; nothing was written",
                self.manifest.value().run_id
            );
        }
        let commit = encode(&Commit {
            schema: Kind::Commit.schema().into(),
            run_id: self.manifest.value().run_id.clone(),
            run_manifest: self.manifest.digest().into(),
            parent,
            height,
            objects,
        })?;
        let publication = publish(
            &mut self.outbox,
            self.remote,
            &self.publisher,
            &self.manifest,
            &commit,
        )?;
        if matches!(
            publication,
            Publication::Advanced { .. } | Publication::AlreadyPublished { .. }
        ) {
            self.head = Some(Head {
                schema: HEAD_SCHEMA.into(),
                run_id: self.manifest.value().run_id.clone(),
                commit: commit.digest.clone(),
                height,
            });
            self.committed.extend(std::mem::take(&mut self.staged));
            self.committed.insert(commit.reference());
        }
        Ok(Some(publication))
    }

    /// Flush if the cadence says one is due. Staged objects count as pending: a flush commits
    /// them, which delivers everything they depend on first, and then delivers anything else
    /// still pending. A failed or refused commit leaves everything owed and the urgency set.
    pub fn tick(&mut self, clock: &dyn Clock) -> Result<Step> {
        let now = clock.now_ms();
        let mut step = Step::default();
        let pending = self.outbox.pending_entries() + self.staged.len() as u64;
        step.due = self
            .cadence
            .due_for(now, pending, self.outbox.pending_bytes());
        if step.due.is_some() && !self.staged.is_empty() {
            step.publication = self.commit()?;
            if self.staged.is_empty() && self.checkpoint_pending {
                self.checkpoints.checkpointed(now);
                self.checkpoint_pending = false;
            }
        }
        // Delivery (and the cadence's own bookkeeping) only once nothing is left uncommitted: a
        // staged object whose commit failed or was refused keeps the flush owed, and its
        // urgency, until a commit takes it. Everything pending belongs to a staged object or to
        // the manifest, which the next commit delivers first.
        if self.staged.is_empty() {
            step.tick = Some(self.cadence.tick(clock, &mut self.outbox, self.remote)?);
        }
        step.flushed =
            step.due.is_some() && self.staged.is_empty() && self.outbox.pending_entries() == 0;
        Ok(self.finish(step))
    }

    /// Request an urgent flush for `why` and flush now.
    fn urgent(&mut self, why: Urgent, clock: &dyn Clock, step: Step) -> Result<Step> {
        self.cadence.request(why);
        let flushed = self.tick(clock)?;
        Ok(self.finish(Step {
            due: flushed.due,
            publication: flushed.publication,
            tick: flushed.tick,
            flushed: flushed.flushed,
            ..step
        }))
    }
}
