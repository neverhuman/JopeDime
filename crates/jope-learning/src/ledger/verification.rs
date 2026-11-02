use super::*;

impl Ledger {
    /// Walk the chain, recomputing every hash.
    ///
    /// Self-contained: it compares the tree against itself. See the module docs for the one thing
    /// that cannot detect — a removed tail with `HEAD.json` rewritten to match — and what does.
    pub fn verify(&self) -> Result<LedgerReport> {
        let mut report = LedgerReport {
            ledger: self.name.clone(),
            ..Default::default()
        };
        let batches = self.batches()?;
        let head = match self.head() {
            Ok(h) => h,
            Err(e) => {
                report.breaks.push(Break {
                    batch: "HEAD".into(),
                    kind: "head-unreadable".into(),
                    detail: e.to_string(),
                });
                None
            }
        };
        let mut prev: Option<(String, String)> = None;
        let mut last_rel = None;
        for (rel, path) in &batches {
            let (events, seal, file_sha, events_sha) = match Ledger::read_batch(path) {
                Ok(x) => x,
                Err(e) => {
                    report.breaks.push(Break {
                        batch: rel.clone(),
                        kind: "unreadable".into(),
                        detail: e.to_string(),
                    });
                    continue;
                }
            };
            let stem = rel
                .rsplit('/')
                .next()
                .unwrap_or(rel)
                .trim_end_matches(".jsonl");
            let mut brk = |kind: &str, detail: String| {
                report.breaks.push(Break {
                    batch: rel.clone(),
                    kind: kind.into(),
                    detail,
                })
            };
            if seal.batch != stem {
                brk(
                    "batch-name",
                    format!("seal names {} in file {stem}", seal.batch),
                );
            }
            if seal.ledger != self.name {
                brk("ledger-name", format!("seal ledger {}", seal.ledger));
            }
            if seal.events != events.len() as u64 {
                brk(
                    "event-count",
                    format!(
                        "seal says {} events, file has {}",
                        seal.events,
                        events.len()
                    ),
                );
            }
            if seal.events_sha256 != events_sha {
                brk(
                    "events-sha256",
                    format!("seal {} != recomputed {events_sha}", seal.events_sha256),
                );
            }
            match (&prev, &seal.prev_batch, &seal.prev_batch_sha256) {
                (None, None, None) => {}
                (None, pb, ps) => brk(
                    "prev-link",
                    format!("first batch names a previous batch {pb:?} {ps:?}"),
                ),
                (Some((pb, ps)), Some(sb), Some(ss)) => {
                    if pb != sb || ps != ss {
                        brk(
                            "prev-link",
                            format!("seal prev {sb}/{ss} != previous batch {pb}/{ps}"),
                        );
                    }
                }
                (Some((pb, _)), _, _) => {
                    brk(
                        "prev-link",
                        format!("batch after {pb} has no previous link"),
                    );
                }
            }
            report.batches += 1;
            report.events += events.len() as u64;
            prev = Some((seal.batch.clone(), file_sha));
            last_rel = Some(rel.clone());
        }
        match (&head, &prev) {
            (None, None) => {}
            (None, Some((b, _))) => report.breaks.push(Break {
                batch: b.clone(),
                kind: "head-missing".into(),
                detail: "batches exist but HEAD.json is absent".into(),
            }),
            (Some(h), None) => report.breaks.push(Break {
                batch: h.batch.clone(),
                kind: "head-dangling".into(),
                detail: "HEAD.json names a batch but none exist".into(),
            }),
            (Some(h), Some((b, s))) => {
                if &h.batch != b || &h.sha256 != s {
                    report.breaks.push(Break {
                        batch: h.batch.clone(),
                        kind: "head-mismatch".into(),
                        detail: format!("HEAD {}/{} != tip {b}/{s}", h.batch, h.sha256),
                    });
                }
                if Some(&h.path) != last_rel.as_ref() {
                    report.breaks.push(Break {
                        batch: h.batch.clone(),
                        kind: "head-path".into(),
                        detail: format!("HEAD path {} != tip path {last_rel:?}", h.path),
                    });
                }
                if h.batches != report.batches || h.events_total != report.events {
                    report.breaks.push(Break {
                        batch: h.batch.clone(),
                        kind: "head-counts".into(),
                        detail: format!(
                            "HEAD counts {}/{} != walked {}/{}",
                            h.batches, h.events_total, report.batches, report.events
                        ),
                    });
                }
            }
        }
        report.head = head;
        Ok(report)
    }

    /// Where a mirror's `HEAD.json` sits relative to *this host's* chain.
    ///
    /// This is the whole of the push path's safety argument, so it is deliberately narrow: the
    /// only thing that may be caught up is a mirror whose tip is a batch **this host already
    /// has**, at the position and with the file hash the mirror's own `HEAD.json` claims.
    /// Anything else — a tip this host cannot produce, a tip whose bytes differ, a tip that
    /// claims a different depth — is a real divergence and is refused. `head_bytes` is a fresh
    /// read of the key that is about to be written; using it as the expected-previous value is
    /// only sound *because* this function proves it is an ancestor of the local chain first.
    pub fn mirror_position(&self, head_bytes: Option<&[u8]>) -> Result<MirrorPosition> {
        let batches = self.batches()?;
        let diverged = |detail: String| Ok(MirrorPosition::Diverged { detail });
        let Some(bytes) = head_bytes else {
            // The mirror has no HEAD for this ledger: everything local is missing.
            return Ok(if batches.is_empty() {
                MirrorPosition::UpToDate
            } else {
                MirrorPosition::Behind { missing: batches }
            });
        };
        let head: Head = match serde_json::from_slice(bytes) {
            Ok(h) => h,
            Err(e) => {
                return diverged(format!(
                    "its HEAD.json does not parse as {HEAD_SCHEMA}: {e}"
                ))
            }
        };
        if head.schema != HEAD_SCHEMA {
            return diverged(format!("its HEAD.json has schema {}", head.schema));
        }
        if head.ledger != self.name {
            return diverged(format!(
                "its HEAD.json is for ledger {}, not {}",
                head.ledger, self.name
            ));
        }
        if head.path.rsplit('/').next().unwrap_or(&head.path) != format!("{}.jsonl", head.batch) {
            return diverged(format!(
                "its HEAD.json names batch {} at path {}",
                head.batch, head.path
            ));
        }
        let Some(i) = batches.iter().position(|(rel, _)| rel == &head.path) else {
            return diverged(format!(
                "its tip {} is not a batch of this host's {} chain (local tip {})",
                head.batch,
                self.name,
                batches
                    .last()
                    .map_or("(empty)".to_string(), |(rel, _)| rel.clone())
            ));
        };
        let sha = sha256_hex(&fs::read(&batches[i].1)?);
        if sha != head.sha256 {
            return diverged(format!(
                "its tip {} hashes to {} there and {sha} here",
                head.batch, head.sha256
            ));
        }
        if head.batches != (i + 1) as u64 {
            return diverged(format!(
                "its tip {} is batch {} of this host's chain but its HEAD.json counts {}",
                head.batch,
                i + 1,
                head.batches
            ));
        }
        Ok(if i + 1 == batches.len() {
            MirrorPosition::UpToDate
        } else {
            MirrorPosition::Behind {
                missing: batches[i + 1..].to_vec(),
            }
        })
    }

    /// Fail unless the chain verifies (used before any append in strict mode).
    ///
    /// An interrupted append is recovered first ([`Ledger::recover`]): a head that lags a
    /// sealed continuation of its own chain is crash residue, not tampering. Refusing it forever
    /// wedged every later write of this ledger.
    pub fn require_intact(&self) -> Result<LedgerReport> {
        self.recover()?;
        let r = self.verify()?;
        if !r.ok() {
            let mut kinds: Vec<&str> = r.breaks.iter().map(|b| b.kind.as_str()).collect();
            kinds.sort_unstable();
            kinds.dedup();
            bail!(
                "{} ledger chain broken: {} break(s) [{}] across {} batch(es): {}",
                self.name,
                r.breaks.len(),
                kinds.join(", "),
                r.batches,
                r.breaks
                    .iter()
                    .map(|b| format!("{}:{}:{}", b.batch, b.kind, b.detail))
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
        Ok(r)
    }
}
