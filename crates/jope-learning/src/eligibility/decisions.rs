use super::*;

/// Durable `qualified` decisions: (comparison, record, stored qualification digest). A stored
/// qualification is eligible only once its decision event is durable; a crash between the
/// immutable file and the event leaves it pending until `qualify` is redelivered.
pub(crate) struct Decisions(BTreeSet<(String, String, String)>);
impl Decisions {
    pub(crate) fn load(state: &State) -> Result<Self> {
        let ledger = Ledger::open(state.root(), "eligibility");
        ledger.require_intact()?;
        Ok(Self::of(&ledger.events()?))
    }
    fn of(events: &[serde_json::Value]) -> Self {
        Self(
            events
                .iter()
                .filter(|event| event["outcome"] == "qualified")
                .filter_map(|event| {
                    Some((
                        event["comparison_sha256"].as_str()?.to_owned(),
                        event["record_sha256"].as_str()?.to_owned(),
                        event["qualification_sha256"].as_str()?.to_owned(),
                    ))
                })
                .collect(),
        )
    }
    fn contains(&self, record: &HwmRecord, qualification_sha256: &str) -> bool {
        self.0.contains(&(
            record.identity_sha256.clone(),
            record.provenance.record_sha256.clone(),
            qualification_sha256.to_owned(),
        ))
    }
}

/// Recompute a stored qualification and require its durable decision; returns its digest.
fn revalidate(
    state: &State,
    record: &HwmRecord,
    decisions: &Decisions,
) -> Result<(Qualification, String)> {
    let bytes = fs::read(qualification_path(state, record))?;
    let stored: StoredQualification = serde_json::from_slice(&bytes)?;
    let current = measured(state, record, &stored.evidence, false)?;
    ensure!(
        current == stored.qualification && current.guard.admits(&current),
        "retained teaching qualification changed"
    );
    let sha = sha256_hex(&bytes);
    ensure!(
        decisions.contains(record, &sha),
        "incomplete teaching linkage: the stored qualification has no durable qualified \
         decision; redeliver it through qualify"
    );
    Ok((current, sha))
}

/// Recompute a retained admission without opening mutable worker paths or changing ledgers.
pub fn revalidate_qualification(state: &State, record: &HwmRecord) -> Result<Qualification> {
    Ok(revalidate(state, record, &Decisions::load(state)?)?.0)
}
fn qualification_path(state: &State, record: &HwmRecord) -> std::path::PathBuf {
    state
        .root()
        .join("eligibility/qualified")
        .join(&record.identity_sha256)
        .join(format!("{}.json", record.provenance.record_sha256))
}

/// Recompute every fold and mean loss; no structural-only witness can call itself qualified.
pub fn qualify(state: &State, record: &HwmRecord, evidence: &OofEvidence) -> Result<Qualification> {
    let _lock = state.lock("eligibility")?;
    let ledger = Ledger::open(state.root(), "eligibility");
    ledger.require_intact()?;
    let decided = Decisions::of(&ledger.events()?);
    let measurement = measured(state, record, evidence, true);
    let outcome = (|| -> Result<(Qualification, String)> {
        let qualification = measurement
            .as_ref()
            .map_err(|error| anyhow::anyhow!("{error:#}"))?
            .clone();
        ensure!(qualification.guard.admits(&qualification), "no eligible signal: mean-fold gain and four of five gains above the frozen numerical margin are required");
        require_blob(state, &record.provenance.record_sha256)?;
        let stored = canonical_json(&StoredQualification {
            evidence: evidence.clone(),
            qualification: qualification.clone(),
        })?;
        let path = qualification_path(state, record);
        // A byte-identical redelivery stays idempotent; anything else is requalification.
        ensure!(
            !path.exists() || fs::read(&path)? == stored,
            "{}",
            requalification_refused(record)
        );
        immutable_write(&path, &stored)?;
        Ok((qualification, sha256_hex(&stored)))
    })();
    // Redelivering a durable decision is idempotent; a pending file gains its event here.
    if let Ok((qualification, sha)) = &outcome {
        if decided.contains(record, sha) {
            return Ok(qualification.clone());
        }
    }
    ledger.append(&[serde_json::json!({"schema":"jope.teaching-decision/v1",
        "comparison_sha256":record.identity_sha256, "record_sha256":record.provenance.record_sha256,
        "evidence_sha256":canonical_sha256(evidence)?, "guard":GuardPolicy::current(),
        "outcome":if outcome.is_ok(){"qualified"}else{"rejected"},
        "qualification_sha256":outcome.as_ref().ok().map(|(_, sha)| sha),
        "reason":outcome.as_ref().err().map(|error| format!("{error:#}")),
        "measurement":measurement.as_ref().ok()
    })], serde_json::json!({"source":"measured-teaching-scorer"}))?;
    outcome.map(|(qualification, _)| qualification)
}

/// The digest of the durable qualification an HWM decision relies on.
pub(crate) fn require_qualified(
    state: &State,
    record: &HwmRecord,
    decisions: &Decisions,
) -> Result<String> {
    record.validate()?;
    Ok(revalidate(state, record, decisions)?.1)
}

fn retain_artifact(state: &State, artifact: &Artifact, live: bool) -> Result<()> {
    if live {
        artifact.verify()?;
        let bytes = fs::read(&artifact.path)?;
        ensure!(
            state.put_blob(&bytes)? == artifact.sha256,
            "artifact changed during retention"
        );
    } else {
        require_blob(state, &artifact.sha256)?;
    }
    Ok(())
}

/// Explicit historical replay: every entry must have an intact accepted ledger
/// event and recomputable measured evidence. Current-pointer admission is unchanged.
pub fn revalidate_replay(state: &State, entries: &[Qualification]) -> Result<TeachingManifest> {
    Ledger::open(state.root(), "hwm").require_intact()?;
    Ledger::open(state.root(), "eligibility").require_intact()?;
    ensure!(!entries.is_empty(), "empty replay admission");
    let events = crate::hwm::events(state)?;
    let decisions = Decisions::load(state)?;
    for entry in entries {
        let mut found = false;
        for event in &events {
            if event.outcome != "accepted"
                || event.pointer != PointerKey::of(&event.record)
                || event.record.sealed
                || event.record.identity != entry.comparison
            {
                continue;
            }
            event.record.validate()?;
            if revalidate(state, &event.record, &decisions)?.0 == *entry {
                found = true;
                break;
            }
        }
        ensure!(
            found,
            "replay lacks exact accepted historical qualification"
        );
    }
    Ok(TeachingManifest {
        schema: "jope.qualified-teaching-freeze/v1",
        entries: entries.to_vec(),
        excluded_diagnostic: 0,
        exclusions: vec![],
    })
}

/// Only exact current pointers that equal their ledger tip, with recomputable measured
/// admissions bound to that HWM decision, can enter a teaching manifest. Incomplete linkage
/// (a pointer behind its tip, or a tip with no pointer) is an exclusion, never an omission.
pub fn freeze_teaching(state: &State) -> Result<TeachingManifest> {
    Ledger::open(state.root(), "hwm").require_intact()?;
    let decisions = Decisions::load(state)?;
    let tips = crate::hwm::accepted_tips(state)?;
    let pointers = state.pointers()?;
    let mut manifest = TeachingManifest {
        schema: "jope.qualified-teaching-freeze/v1",
        excluded_diagnostic: 0,
        exclusions: Vec::new(),
        entries: Vec::new(),
    };
    for (key, record) in &pointers {
        let candidate = (|| -> Result<Qualification> {
            record.validate()?;
            ensure!(
                *key == PointerKey::of(record)
                    && record.identity.comparison.is_some()
                    && !record.sealed,
                "legacy, mismatched or sealed diagnostic record"
            );
            let tip = tips
                .get(key)
                .context("teaching pointer lacks an accepted exact-identity ledger event")?;
            ensure!(
                tip.record == *record,
                "incomplete teaching linkage: the pointer lags its HWM ledger tip {}; run hwm \
                 recovery",
                tip.record.provenance.record_sha256
            );
            require_blob(state, &record.provenance.record_sha256)?;
            let (qualified, sha) = revalidate(state, record, &decisions)?;
            ensure!(
                tip.qualification_sha256.as_deref() == Some(sha.as_str()),
                "incomplete teaching linkage: the HWM decision is not bound to this stored \
                 qualification"
            );
            Ok(qualified)
        })();
        match candidate {
            Ok(qualified) => manifest.entries.push(qualified),
            Err(error) => {
                manifest.excluded_diagnostic += 1;
                manifest.exclusions.push((key.rel(), format!("{error:#}")));
            }
        }
    }
    for (key, tip) in &tips {
        if tip.record.identity.comparison.is_some()
            && *key == PointerKey::of(&tip.record)
            && !pointers.iter().any(|(pointer, _)| pointer == key)
        {
            manifest.excluded_diagnostic += 1;
            manifest.exclusions.push((
                key.rel(),
                "incomplete teaching linkage: the accepted HWM tip has no pointer; run hwm \
                 recovery"
                    .into(),
            ));
        }
    }
    Ok(manifest)
}
