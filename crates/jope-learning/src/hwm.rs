//! The high-water-mark ledger: admission rules, ingest, show, verify and freeze.

use crate::canon::sha256_hex;
use crate::champion::Champion;
use crate::ledger::{atomic_write, now_unix, Ledger, LedgerReport};
use crate::mirror::{Mirror, MirrorReport};
use crate::record::HwmRecord;
use crate::state::{PointerKey, State};
use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Schema identifier for event values.
pub const EVENT_SCHEMA: &str = "jope.hwm-event/v1";
/// Schema identifier for freeze values.
pub const FREEZE_SCHEMA: &str = "jope.hwm-freeze/v1";

/// One record proposed to the ledger with the bytes that become its blob.
pub struct Candidate {
    /// Record recorded for this candidate.
    pub record: HwmRecord,
    /// Collection of blob values carried by this candidate.
    pub blob: Vec<u8>,
    /// True when `prev_event_sha256`/`improvement_number` are copied from the source's own chain
    /// (champions); false when the ledger assigns them from the record it displaces.
    pub chain_from_source: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "outcome", content = "reason")]
/// `Decision` is an enum for the decision contract in this module: The high-water-mark ledger: admission rules, ingest, show, verify and freeze.
pub enum Decision {
    /// The `Accepted` case of `Decision`.
    Accepted,
    /// The `Rejected` case of `Decision`.
    Rejected(RejectReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
/// `RejectReason` is an enum for the reject reason contract in this module: The high-water-mark ledger: admission rules, ingest, show, verify and freeze.
pub enum RejectReason {
    /// Same score as the incumbent: the older record stays.
    Tie,
    /// Worse than the incumbent under the cut's metric direction.
    NotBetter,
    /// An unbound evaluator can never displace a bound one.
    UnboundCannotDisplaceBound,
    /// The incumbent was scored under a different metric.
    MetricMismatch,
    /// Scores under different full comparison identities cannot displace one another.
    IdentityMismatch,
    /// Train-side OOF candidates require recomputable measured teaching admission.
    TeachingIneligible,
    /// The record failed its structural checks.
    Invalid,
    /// The record is revoked evidence (W12-T05): it can neither initialize nor improve.
    Revoked,
}

impl RejectReason {
    /// Returns the stable string representation of this value.
    pub fn as_str(self) -> &'static str {
        match self {
            RejectReason::Tie => "tie",
            RejectReason::NotBetter => "not-better",
            RejectReason::UnboundCannotDisplaceBound => "unbound-cannot-displace-bound",
            RejectReason::MetricMismatch => "metric-mismatch",
            RejectReason::IdentityMismatch => "identity-mismatch",
            RejectReason::TeachingIneligible => "teaching-ineligible",
            RejectReason::Invalid => "invalid",
            RejectReason::Revoked => "revoked",
        }
    }
}

/// The explicit HWM outcome of one candidate. Teaching eligibility is decided before this and
/// never counts as an improvement: only `Improved` advances an existing pointer's chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Transition {
    /// First accepted record of a pointer: a bootstrap, not an improvement.
    Initialized,
    /// Strictly better than the comparable incumbent it displaced.
    Improved,
    /// The `Tie` case of `Transition`.
    Tie,
    /// Comparable with the incumbent and not better.
    NotImproved,
    /// Metric or evaluator semantics differ from the incumbent's.
    Noncomparable,
    /// Train-side OOF record without a recomputable measured qualification.
    Ineligible,
    /// The `Invalid` case of `Transition`.
    Invalid,
    /// A record this pointer already accepted, delivered again: never another improvement.
    Duplicate,
    /// A record that is revoked evidence (W12-T05).
    Revoked,
}

impl Transition {
    /// Performs the of operation.
    pub fn of(incumbent: Option<&HwmRecord>, decision: &Decision) -> Self {
        match decision {
            Decision::Accepted if incumbent.is_none() => Self::Initialized,
            Decision::Accepted => Self::Improved,
            Decision::Rejected(RejectReason::Tie) => Self::Tie,
            Decision::Rejected(RejectReason::NotBetter) => Self::NotImproved,
            Decision::Rejected(
                RejectReason::UnboundCannotDisplaceBound
                | RejectReason::MetricMismatch
                | RejectReason::IdentityMismatch,
            ) => Self::Noncomparable,
            Decision::Rejected(RejectReason::TeachingIneligible) => Self::Ineligible,
            Decision::Rejected(RejectReason::Invalid) => Self::Invalid,
            Decision::Rejected(RejectReason::Revoked) => Self::Revoked,
        }
    }

    /// Returns the stable string representation of this value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Initialized => "initialized",
            Self::Improved => "improved",
            Self::Tie => "tie",
            Self::NotImproved => "not-improved",
            Self::Noncomparable => "noncomparable",
            Self::Ineligible => "ineligible",
            Self::Invalid => "invalid",
            Self::Duplicate => "duplicate",
            Self::Revoked => "revoked",
        }
    }
}

/// The admission rule (`champions.rs:452` + the plan's unbound rule).
pub fn decide(incumbent: Option<&HwmRecord>, candidate: &HwmRecord) -> Decision {
    if candidate.validate().is_err() {
        return Decision::Rejected(RejectReason::Invalid);
    }
    let Some(inc) = incumbent else {
        return Decision::Accepted;
    };
    if inc.evaluator.is_bound() && !candidate.evaluator.is_bound() {
        return Decision::Rejected(RejectReason::UnboundCannotDisplaceBound);
    }
    if inc.score.metric != candidate.score.metric {
        return Decision::Rejected(RejectReason::MetricMismatch);
    }
    if inc.identity != candidate.identity {
        return Decision::Rejected(RejectReason::IdentityMismatch);
    }
    let dir = candidate.score.metric.direction();
    if dir.strictly_improves(candidate.score.value, inc.score.value) {
        Decision::Accepted
    } else if candidate.score.value == inc.score.value {
        Decision::Rejected(RejectReason::Tie)
    } else {
        Decision::Rejected(RejectReason::NotBetter)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `HwmEvent` is a record for the HWM event contract in this module: The high-water-mark ledger: admission rules, ingest, show, verify and freeze.
pub struct HwmEvent {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Position within the batch.
    pub seq: u64,
    /// Outcome recorded for this HWM event.
    pub outcome: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Explanation associated with the recorded disposition.
    pub reason: Option<String>,
    /// State pointer associated with this record.
    pub pointer: PointerKey,
    /// Record recorded for this HWM event.
    pub record: HwmRecord,
    /// The record this one replaced at the pointer (accepted events only).
    pub displaced_record_sha256: Option<String>,
    /// Unix timestamp when this event was recorded.
    pub ts_unix: u64,
    /// Absent on events written before transitions were recorded; `verify` rederives it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transition: Option<Transition>,
    /// Digest of the stored teaching qualification a train-side OOF decision relied on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qualification_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
/// `Reject` is a record for the reject contract in this module: The high-water-mark ledger: admission rules, ingest, show, verify and freeze.
pub struct Reject {
    /// SHA-256 digest identifying the associated identity.
    pub identity_sha256: String,
    /// SHA-256 digest identifying the associated legacy identity.
    pub legacy_identity_sha256: Option<String>,
    /// State pointer associated with this record.
    pub pointer: String,
    /// Explanation associated with the recorded disposition.
    pub reason: String,
    /// Measured or derived value carried by this result.
    pub value: f64,
    /// Optional incumbent value value; `None` means it is not available.
    pub incumbent_value: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
/// `IngestReport` is a record for the ingest report contract in this module: The high-water-mark ledger: admission rules, ingest, show, verify and freeze.
pub struct IngestReport {
    /// Provenance source for this value.
    pub source: Value,
    /// Candidates recorded for this ingest report.
    pub candidates: u64,
    /// Accepted recorded for this ingest report.
    pub accepted: u64,
    /// Rejected recorded for this ingest report.
    pub rejected: u64,
    /// Every candidate's [`Transition`]; `accepted` = `initialized` + `improved`.
    pub transitions: BTreeMap<String, u64>,
    /// Pointers written recorded for this ingest report.
    pub pointers_written: u64,
    /// Pointers that lagged their ledger tip (a crash before a pointer write) and were rolled
    /// forward; included in `pointers_written`.
    pub pointers_recovered: u64,
    /// Blobs written recorded for this ingest report.
    pub blobs_written: u64,
    /// Ledger batch receipt produced by the operation, if one was appended.
    pub batch: Option<crate::ledger::BatchReceipt>,
    /// Optional mirror value; `None` means it is not available.
    pub mirror: Option<MirrorReport>,
    /// Collection of rejects values carried by this ingest report.
    pub rejects: Vec<Reject>,
    /// Seconds recorded for this ingest report.
    pub seconds: f64,
}

/// Every field of a record that is an IEEE double and is not finite, as `field: value`.
///
/// `serde_json` has no representation for NaN or an infinity: it writes them as `null` and
/// refuses to read them back. A record carrying one is therefore not a record that happens to be
/// wrong, it is a record that *cannot be written down* — and an append-only hash-chained ledger
/// has no way to take one back. One such event makes every later `events()` call fail with
/// `invalid type: null, expected f64`, which is to say the whole ledger, not just that event.
fn unrepresentable_floats(r: &HwmRecord) -> Vec<String> {
    let mut out = Vec::new();
    let mut check = |name: &str, v: f64| {
        if !v.is_finite() {
            out.push(format!("{name}={v}"));
        }
    };
    check("score.value", r.score.value);
    if let Some(v) = r.cost.runtime_seconds {
        check("cost.runtime_seconds", v);
    }
    if let Some(v) = r.cost.search_seconds {
        check("cost.search_seconds", v);
    }
    out
}

/// Refuse any candidate the ledger could not read back, before a single byte of it is written.
///
/// This is deliberately *not* a [`Decision`]: a rejection event is itself a ledger write, and the
/// whole point is that this record cannot be put in the ledger in any shape. Writing it as a
/// rejection would poison the chain exactly as accepting it would; writing it with a substituted
/// score would be a fabricated number in an append-only record of what was measured. So the run
/// fails, names the candidate and the field, and nothing moves — the source is wrong and only the
/// source can fix it. Ordinary ties and inferior candidates are unaffected: they are
/// representable, so they keep their readable rejection event.
fn refuse_unrepresentable(candidates: &[Candidate]) -> Result<()> {
    let mut bad: Vec<String> = Vec::new();
    for c in candidates {
        let fields = unrepresentable_floats(&c.record);
        if !fields.is_empty() {
            bad.push(format!(
                "{} ({}) at {}: {}",
                c.record.identity_sha256,
                c.record.provenance.source,
                PointerKey::of(&c.record).rel(),
                fields.join(", ")
            ));
        }
    }
    ensure!(
        bad.is_empty(),
        "refusing to ingest {} candidate(s) whose score is not a finite number, because JSON \
         cannot represent one: serde writes NaN and the infinities as `null` and cannot read them \
         back, so a single such event would make every later read of this append-only ledger fail \
         with `invalid type: null, expected f64`. Nothing was written — no blob, no batch, no \
         pointer, nothing on the mirror. Fix or drop these rows at the source and re-run:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
    Ok(())
}

/// Admit `candidates` in (ts, identity) order, append one sealed batch, then move the pointers.
pub fn ingest(
    state: &State,
    mirror: Option<&Mirror>,
    mut candidates: Vec<Candidate>,
    source: Value,
) -> Result<IngestReport> {
    let started = Instant::now();
    // Before the lock, before the mirror preflight, before any blob: a candidate that cannot be
    // serialised and read back has no admissible outcome at all, so it never reaches the ledger.
    refuse_unrepresentable(&candidates)?;
    let _lock = state.lock("hwm")?;
    let ledger = Ledger::open(state.root(), "hwm");
    ledger.require_intact()?;
    // Before a single local byte moves: if the mirror has diverged from this host's chain there is
    // nothing this run can do about it, and appending anyway only widens the gap and leaves batch
    // objects on the mirror that its own HEAD cannot reach.
    if let Some(m) = mirror {
        crate::sync::preflight(m, &ledger)?;
    }
    candidates.sort_by(|a, b| {
        (a.record.ts_unix, &a.record.identity_sha256)
            .cmp(&(b.record.ts_unix, &b.record.identity_sha256))
    });
    // Every touched pointer is resolved against the ledger before a single byte moves: the
    // incumbent is the ledger's tip, never a pointer that lags it or that it never accepted.
    let (history, rejected_history) = decided_events(&ledger)?;
    let mut on_disk: BTreeMap<PointerKey, Option<HwmRecord>> = BTreeMap::new();
    let mut current: BTreeMap<PointerKey, HwmRecord> = BTreeMap::new();
    let mut seen: BTreeMap<PointerKey, BTreeSet<String>> = BTreeMap::new();
    // Records each pointer already rejected. A record rejected again is a redelivery (W3-T06):
    // counted as a duplicate delivery, never as a second decision.
    let mut refused: Rejected = BTreeMap::new();
    let revocations = crate::revocation::Revocations::load(state)?;
    let mut lagging = BTreeSet::new();
    for c in &candidates {
        let key = PointerKey::of(&c.record);
        if on_disk.contains_key(&key) {
            continue;
        }
        let disk = state.read_pointer(&key)?;
        let accepted = history.get(&key).map_or(&[][..], Vec::as_slice);
        if let Some(tip) = resolve(&key, disk.as_ref(), accepted)? {
            if disk.as_ref() != Some(tip) {
                lagging.insert(key.clone());
            }
            current.insert(key.clone(), tip.clone());
        }
        seen.insert(
            key.clone(),
            accepted
                .iter()
                .map(|e| e.record.provenance.record_sha256.clone())
                .collect(),
        );
        refused.insert(
            key.clone(),
            rejected_history.get(&key).cloned().unwrap_or_default(),
        );
        on_disk.insert(key, disk);
    }
    let mut decisions = None;
    let mut events = Vec::with_capacity(candidates.len());
    let mut rejects = Vec::new();
    let mut transitions = BTreeMap::<String, u64>::new();
    let mut blobs_written = 0u64;
    let ts = now_unix();
    for (seq, mut c) in candidates.into_iter().enumerate() {
        let blob_sha = sha256_hex(&c.blob);
        ensure!(
            c.record.provenance.record_sha256 == blob_sha,
            "candidate {} record_sha256 {} does not hash its blob {blob_sha}",
            c.record.identity_sha256,
            c.record.provenance.record_sha256
        );
        if !state.blob_path(&blob_sha).exists() {
            blobs_written += 1;
        }
        state.put_blob(&c.blob)?;
        let key = PointerKey::of(&c.record);
        let duplicate = seen[&key].contains(&c.record.provenance.record_sha256);
        // Rejected before at this pointer. A redelivery the ledger now accepts (a teaching
        // qualification that has since become durable) is a new decision and is unaffected.
        let redelivered = refused[&key].contains(&c.record.provenance.record_sha256);
        let incumbent = current.get(&key);
        if !c.chain_from_source {
            match incumbent {
                Some(inc) => {
                    c.record.prev_event_sha256 = Some(inc.provenance.record_sha256.clone());
                    c.record.improvement_number = inc.improvement_number + 1;
                }
                None => {
                    c.record.prev_event_sha256 = None;
                    c.record.improvement_number = 0;
                }
            }
        }
        let train_oof = c
            .record
            .identity
            .comparison
            .as_ref()
            .is_some_and(|context| context.stage == "train_oof");
        // W12-T05: a revoked record is refused before any comparison. A revoked piece of a
        // train-side teacher's deeper closure is refused by its qualification below.
        let revoked = revocations.revoked(&blob_sha).is_some();
        let qualification_sha256 = if train_oof && !revoked {
            if decisions.is_none() {
                decisions = Some(crate::eligibility::Decisions::load(state)?);
            }
            crate::eligibility::require_qualified(state, &c.record, decisions.as_ref().unwrap())
                .ok()
        } else {
            None
        };
        let decision = if revoked {
            Decision::Rejected(RejectReason::Revoked)
        } else if train_oof && qualification_sha256.is_none() {
            Decision::Rejected(RejectReason::TeachingIneligible)
        } else {
            match decide(incumbent, &c.record) {
                Decision::Accepted if duplicate => Decision::Rejected(RejectReason::NotBetter),
                decision => decision,
            }
        };
        let transition = match decision {
            Decision::Rejected(RejectReason::Tie | RejectReason::NotBetter) if duplicate => {
                Transition::Duplicate
            }
            Decision::Rejected(_) if redelivered => Transition::Duplicate,
            _ => Transition::of(incumbent, &decision),
        };
        *transitions.entry(transition.as_str().into()).or_default() += 1;
        let (outcome, reason, displaced) = match &decision {
            Decision::Accepted => (
                "accepted",
                None,
                incumbent.map(|r| r.provenance.record_sha256.clone()),
            ),
            Decision::Rejected(reason) => {
                rejects.push(Reject {
                    identity_sha256: c.record.identity_sha256.clone(),
                    legacy_identity_sha256: c.record.legacy_identity_sha256.clone(),
                    pointer: key.rel(),
                    reason: reason.as_str().into(),
                    value: c.record.score.value,
                    incumbent_value: incumbent.map(|r| r.score.value),
                });
                ("rejected", Some(reason.as_str().to_string()), None)
            }
        };
        let event = HwmEvent {
            schema: EVENT_SCHEMA.into(),
            seq: seq as u64,
            outcome: outcome.into(),
            reason,
            pointer: key.clone(),
            record: c.record.clone(),
            displaced_record_sha256: displaced,
            ts_unix: ts,
            transition: Some(transition),
            qualification_sha256,
        };
        events.push(serde_json::to_value(&event)?);
        if decision != Decision::Accepted {
            if let Some(refused) = refused.get_mut(&key) {
                refused.insert(c.record.provenance.record_sha256.clone());
            }
        }
        if decision == Decision::Accepted {
            if let Some(seen) = seen.get_mut(&key) {
                seen.insert(c.record.provenance.record_sha256.clone());
            }
            current.insert(key, c.record);
        }
    }
    let accepted = events.iter().filter(|e| e["outcome"] == "accepted").count() as u64;
    let rejected = events.len() as u64 - accepted;
    let batch = if events.is_empty() {
        None
    } else {
        Some(ledger.append(&events, source.clone())?)
    };
    // Pointers move only after their events are durable.
    let mut pointers_written = 0u64;
    let mut pointers_recovered = 0u64;
    for (key, record) in &current {
        let before = on_disk.get(key).cloned().flatten();
        if before.as_ref() == Some(record) {
            continue;
        }
        state.write_pointer(
            key,
            record,
            before.as_ref().map(|r| r.provenance.record_sha256.as_str()),
        )?;
        pointers_written += 1;
        pointers_recovered += u64::from(lagging.contains(key));
    }
    let mirror_report = match mirror {
        Some(m) => Some(push_mirror(m, state, &ledger)?),
        None => None,
    };
    Ok(IngestReport {
        source,
        candidates: events.len() as u64,
        accepted,
        rejected,
        transitions,
        pointers_written,
        pointers_recovered,
        blobs_written,
        batch,
        mirror: mirror_report,
        rejects,
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// Parse one ledger event, saying *which* event failed and why in terms a reader can act on.
///
/// A bare `invalid type: null, expected f64` names neither the batch, the event nor the field, and
/// is indistinguishable from a dozen other schema problems; on a 12k-event corpus it is not enough
/// to find the offending line. So on failure the raw JSON is re-read for the identifying fields it
/// still has, and the one cause that cannot be fixed by re-ingesting — a float written as `null` —
/// is named for what it is.
pub fn parse_event(index: usize, v: &Value) -> Result<HwmEvent> {
    match serde_json::from_value::<HwmEvent>(v.clone()) {
        Ok(e) => Ok(e),
        Err(err) => {
            let at = |p: &[&str]| {
                let mut cur = v;
                for k in p {
                    cur = cur.get(*k)?;
                }
                cur.as_str().map(str::to_string)
            };
            let mut detail = format!(
                "event #{index} (seq {}, pointer {}, identity {}, record {})",
                v.get("seq").map_or("?".into(), |x| x.to_string()),
                at(&["pointer", "task"]).map_or("?".into(), |t| format!(
                    "{t}/{}/{}",
                    at(&["pointer", "dataset_hash"]).unwrap_or_else(|| "?".into()),
                    at(&["pointer", "cut_id"]).unwrap_or_else(|| "?".into())
                )),
                at(&["record", "identity_sha256"]).unwrap_or_else(|| "?".into()),
                at(&["record", "provenance", "record_sha256"]).unwrap_or_else(|| "?".into()),
            );
            if v.pointer("/record/score/value") == Some(&Value::Null) {
                detail.push_str(
                    ": its `record.score.value` is JSON `null`, which is how serde writes a NaN \
                     or an infinity and is why it cannot be read back. This event was written by \
                     a build that admitted a non-finite score; it cannot be repaired in place, \
                     because the ledger is append-only and hash-chained. Quarantine the batch \
                     that holds it (`hwm verify` reports it as `event-unparseable` without \
                     failing) and re-ingest its source",
                );
            }
            Err(anyhow!("parse hwm event: {detail}: {err}"))
        }
    }
}

/// For every pointer, the `record_sha256`s this host's ledger accepted at it, oldest first. The
/// only evidence that a pointer the mirror holds is an *earlier state of ours* rather than another
/// host's.
fn accepted_history(ledger: &Ledger) -> Result<BTreeMap<PointerKey, Vec<String>>> {
    Ok(accepted_events(ledger)?
        .into_iter()
        .map(|(key, events)| {
            let shas = events
                .into_iter()
                .map(|e| e.record.provenance.record_sha256)
                .collect();
            (key, shas)
        })
        .collect())
}

/// Accepted events at each pointer, oldest first.
type Accepted = BTreeMap<PointerKey, Vec<HwmEvent>>;
/// The records each pointer rejected.
type Rejected = BTreeMap<PointerKey, BTreeSet<String>>;

/// Every accepted event at each pointer, oldest first: the ledger's own HWM history.
fn accepted_events(ledger: &Ledger) -> Result<Accepted> {
    Ok(decided_events(ledger)?.0)
}

/// The ledger's accepted history and, at each pointer, every record it rejected there.
fn decided_events(ledger: &Ledger) -> Result<(Accepted, Rejected)> {
    let mut accepted: Accepted = BTreeMap::new();
    let mut rejected: Rejected = BTreeMap::new();
    for (i, v) in ledger.events()?.iter().enumerate() {
        let e = parse_event(i, v)?;
        if e.outcome == "accepted" {
            accepted.entry(e.pointer.clone()).or_default().push(e);
        } else {
            rejected
                .entry(e.pointer.clone())
                .or_default()
                .insert(e.record.provenance.record_sha256.clone());
        }
    }
    Ok((accepted, rejected))
}

/// The ledger tip at every pointer (its last accepted event).
pub(crate) fn accepted_tips(state: &State) -> Result<BTreeMap<PointerKey, HwmEvent>> {
    Ok(accepted_events(&Ledger::open(state.root(), "hwm"))?
        .into_iter()
        .filter_map(|(key, mut events)| events.pop().map(|tip| (key, tip)))
        .collect())
}

/// The ledger's tip at `key`. The pointer on disk may lag it (a crash after the batch append,
/// before the pointer write) and is then rolled forward; it may never hold a record the ledger
/// did not accept at that pointer.
fn resolve<'a>(
    key: &PointerKey,
    disk: Option<&HwmRecord>,
    accepted: &'a [HwmEvent],
) -> Result<Option<&'a HwmRecord>> {
    if let Some(disk) = disk {
        ensure!(
            accepted.iter().any(|e| e.record == *disk),
            "{}: the pointer holds record {} that this host's HWM ledger never accepted there. \
             Nothing was written. Run `hwm verify`, then restore the pointer from the ledger or \
             re-ingest its source",
            key.rel(),
            disk.provenance.record_sha256
        );
    }
    Ok(accepted.last().map(|e| &e.record))
}

#[derive(Clone, Debug, Default, Serialize)]
/// `RecoverReport` is a record for the recover report contract in this module: The high-water-mark ledger: admission rules, ingest, show, verify and freeze.
pub struct RecoverReport {
    /// False for a plan ([`recover_plan`]): nothing was written and no lock was taken.
    pub applied: bool,
    /// The ledger batch a lagging `HEAD.json` was (or would be) rolled forward to.
    pub ledger_head: Option<String>,
    /// Pointers recorded for this recover report.
    pub pointers: u64,
    /// Pointers rolled forward (or, in a plan, that would be) to their ledger tip.
    pub recovered: Vec<String>,
    /// Pointers holding a record the ledger never accepted there; left untouched.
    pub refused: Vec<(String, String)>,
}

/// Roll every pointer that lags its ledger tip forward: the recovery for a crash between a
/// batch append and its pointer writes. Legacy pointer keys are not rewritten, and a pointer
/// the ledger never accepted is reported rather than overwritten.
pub fn recover(state: &State) -> Result<RecoverReport> {
    recover_with(state, true)
}

/// What [`recover`] would do, without writing anything or taking any lock: a ledger head that
/// lags a sealed continuation of its chain, the pointers that lag their ledger tip, and the
/// pointers it would refuse. A chain that is broken in any other way is refused, exactly as
/// [`recover`] refuses it.
pub fn recover_plan(state: &State) -> Result<RecoverReport> {
    recover_with(state, false)
}

fn recover_with(state: &State, apply: bool) -> Result<RecoverReport> {
    let _lock = if apply {
        Some(state.lock("hwm")?)
    } else {
        None
    };
    let ledger = Ledger::open(state.root(), "hwm");
    let mut report = RecoverReport {
        applied: apply,
        ..RecoverReport::default()
    };
    if apply {
        report.ledger_head = ledger.recover()?.map(|h| h.path);
        ledger.require_intact()?;
    } else {
        let lagging = ledger.lagging_head()?;
        // A lagging head shows up as `head-*` breaks, which rolling forward repairs; any other
        // break refuses, as `require_intact` would.
        let verified = ledger.verify()?;
        let breaks: Vec<String> = verified
            .breaks
            .iter()
            .filter(|b| !(lagging.is_some() && b.kind.starts_with("head-")))
            .map(|b| format!("{}:{}:{}", b.batch, b.kind, b.detail))
            .collect();
        ensure!(
            breaks.is_empty(),
            "hwm ledger chain broken: {} break(s): {}",
            breaks.len(),
            breaks.join("; ")
        );
        report.ledger_head = lagging.map(|h| h.path);
    }
    let history = accepted_events(&ledger)?;
    for (key, accepted) in &history {
        let Some(tip) = accepted.last().map(|e| &e.record) else {
            continue;
        };
        if *key != PointerKey::of(tip) {
            continue;
        }
        report.pointers += 1;
        let disk = state.read_pointer(key)?;
        match resolve(key, disk.as_ref(), accepted) {
            Err(error) => report.refused.push((key.rel(), format!("{error:#}"))),
            Ok(_) if disk.as_ref() == Some(tip) => {}
            Ok(_) => {
                if apply {
                    state.write_pointer(
                        key,
                        tip,
                        disk.as_ref().map(|r| r.provenance.record_sha256.as_str()),
                    )?;
                }
                report.recovered.push(key.rel());
            }
        }
    }
    for (key, _) in state.pointers()? {
        if key.identity_sha256.is_some() && !history.contains_key(&key) {
            report
                .refused
                .push((key.rel(), "no accepted ledger event at this pointer".into()));
        }
    }
    Ok(report)
}

/// Bring the mirror up to this host's tip: the ledger batches it is missing, the blobs those
/// batches name, and the `current.json` pointers they moved.
///
/// Everything is derived from the batches [`crate::sync::push`] backfills rather than from what
/// this one run happened to write, so a mirror left behind by an earlier local-only ingest is
/// caught up completely — batches, blobs and pointers — instead of being wedged for ever.
///
/// The blobs go up *inside* that backfill, ahead of the batch object that names them and therefore
/// ahead of the `HEAD.json` move, because the mirror's `HEAD.json` is the only input
/// [`crate::sync::plan`] has: once it names a batch, that batch is never in `missing` again and
/// nothing will ever come back for its blobs. Pushing them afterwards would mean one failed blob
/// PUT leaves the mirror's ledger permanently naming records the mirror does not hold, with every
/// later push reporting success. Keeping them ahead of it costs nothing — they are
/// content-addressed and immutable, so a blob the mirror already has is an `Exists` no-op — and it
/// is what keeps the mirror's ledger self-contained (`hwm verify` reads a blob per event).
///
/// Ordering is only half of it, because it only covers a blob missing from the *mirror*. A blob
/// missing from the *local store* is the same permanent damage arriving through a different door:
/// the upload is skipped, the batch object goes up regardless, the tip moves past it, and the gap
/// is outside every future backfill window. So the local store is asked first, the mirror second,
/// and a record neither can supply stops the push before that batch's object is written — the one
/// point at which stopping still leaves the mirror repairable.
///
/// The `current.json` pointers stay *after* the `HEAD.json` move, and are deliberately *not* every
/// pointer: the real corpus has ~12k of them and each would cost a GET per push. Only the pointers
/// that the backfilled batches moved are pushed. Once `HEAD.json` names those batches, no later
/// `plan` offers them again, so each such pointer is recorded as **owed** to this mirror
/// ([`owed_path`]) durably, *before* anything goes up. A pointer leaves the owed set only once
/// the mirror holds exactly the local bytes. So a pointer write that fails, or never happens
/// because the process died after `HEAD.json` moved, is retried on every later push, including a
/// push with nothing new. Before W8-T06 it stayed stale on the mirror, or absent, until some
/// later batch happened to accept a record at that exact pointer. Blobs cannot be left that way
/// either, which is why they go up ahead of the batch that names them.
fn push_mirror(m: &Mirror, state: &State, ledger: &Ledger) -> Result<MirrorReport> {
    let mut report = MirrorReport {
        url: m.url.clone(),
        ..Default::default()
    };
    let plan = crate::sync::plan(m, ledger)?;
    let mut owed = Owed::load(state, &m.url)?;
    // What the owed record on disk says; it is rewritten whenever that would be wrong.
    let mut on_disk = owed.pointers.clone();
    for (_, path) in &plan.missing {
        let (events, _, _, _) = Ledger::read_batch(path)?;
        for (i, v) in events.iter().enumerate() {
            let e = parse_event(i, v)?;
            if e.outcome == "accepted" {
                owed.pointers.insert(e.pointer);
            }
        }
    }
    if owed.pointers != on_disk {
        owed.save(state)?;
        on_disk = owed.pointers.clone();
    }
    crate::sync::push(m, ledger, plan, &mut report, |path, report| {
        let (events, _, _, _) = Ledger::read_batch(path)?;
        for (i, v) in events.iter().enumerate() {
            let e = parse_event(i, v)?;
            let sha = &e.record.provenance.record_sha256;
            match state.read_blob(sha)? {
                Some(bytes) => report.count(m.put_immutable(&State::blob_rel(sha), &bytes)?),
                // The local store cannot supply a blob this batch names. Skipping it is the one
                // thing this push must not do: the batch object would go up, `HEAD.json` would
                // move past it, and from then on `sync::plan` never returns that batch again — the
                // gap is outside every future backfill window and therefore permanent, while every
                // later push reports success. So the mirror is asked whether it can back the
                // record itself, and otherwise the push fails before the batch object is written,
                // leaving the mirror's tip where it was so that a push made after the blob is
                // restored closes the same gap.
                //
                // "Can back it" means the mirror's bytes *hash to the sha the event names*, not
                // merely that something exists at that key. Every other path into this store
                // proves identity — `State::read_blob` re-hashes what it reads and
                // `Mirror::put_immutable` compares bytes on an `Exists` — and this one must too,
                // because it is the only path that accepts an object it did not itself write.
                // Presence alone would let one wrong object under the immutable key launder
                // itself into a published, HEAD-reachable batch that no reader can verify.
                None => {
                    let rel = State::blob_rel(sha);
                    let refuse = |detail: String| {
                        anyhow!(
                            "{}: names record {sha}, and this host's blob store ({}) does not \
                             hold it — {detail}. Refusing to publish a batch the mirror could \
                             not back: the mirror's HEAD.json would name a record no reader can \
                             verify, and because the backfill window is derived from that \
                             HEAD.json alone, no later push would ever come back for it. The \
                             mirror's tip and its existing bytes are unchanged; restore the blob \
                             (`hwm verify` lists it as `blob-missing`) from a bundle or by \
                             re-ingesting its source, then re-run this push.",
                            path.display(),
                            state.blob_path(sha).display(),
                        )
                    };
                    match m.get(&rel)? {
                        None => bail!(refuse(format!("and neither does the mirror at {}", m.url))),
                        Some(theirs) => {
                            let got = sha256_hex(&theirs);
                            if got != *sha {
                                bail!(refuse(format!(
                                    "and the {} bytes the mirror at {} holds at {rel} hash to \
                                     {got}, not {sha}, so that object is not this record. It is \
                                     left exactly as it is — an immutable key is never \
                                     overwritten to make a push succeed",
                                    theirs.len(),
                                    m.url
                                )));
                            }
                            // The mirror's copy is the record, so the batch is already backed
                            // there and there is nothing to upload for it.
                            report.existed += 1;
                        }
                    }
                }
            }
        }
        Ok(())
    })?;
    // Settle every owed pointer. Whatever does not settle stays owed, durably, including when the
    // push stops at an error.
    let settled = settle_owed(m, state, ledger, &mut owed, &mut report);
    if owed.pointers != on_disk {
        owed.save(state)?;
    }
    report.pointers_owed = owed.pointers.len() as u64;
    settled.map(|()| report)
}

/// Push each owed pointer whose local bytes differ from the mirror's, under the ancestor rule, and
/// drop it from `owed` once the mirror holds exactly the local bytes. A pointer absent locally (a
/// local break `hwm verify` reports as `pointer-missing`) cannot be settled and stays owed. The
/// first failed write stops the settlement, and everything not yet settled stays owed.
fn settle_owed(
    m: &Mirror,
    state: &State,
    ledger: &Ledger,
    owed: &mut Owed,
    report: &mut MirrorReport,
) -> Result<()> {
    let mut history: Option<BTreeMap<PointerKey, Vec<String>>> = None;
    for key in owed.pointers.clone() {
        let local_bytes = match fs::read(state.pointer_path(&key)?) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e).with_context(|| format!("read {}", key.rel())),
        };
        let observed = m.get_observed(&key.rel())?;
        if observed.as_ref().map(|object| object.bytes.as_slice()) != Some(local_bytes.as_slice()) {
            if let Some(theirs) = &observed {
                let hist = match &history {
                    Some(h) => h,
                    None => history.insert(accepted_history(ledger)?),
                };
                check_pointer_ancestor(m, &key, &theirs.bytes, &local_bytes, hist.get(&key))?;
            }
            m.put_pointer(&key.rel(), &local_bytes, observed.as_ref())?;
            report.pointers += 1;
        }
        owed.pointers.remove(&key);
    }
    Ok(())
}

/// Schema identifier for owed values.
pub const OWED_SCHEMA: &str = "jope.hwm-mirror-owed/v1";

/// `<state>/progress/mirror-owed/hwm-<sha256(url)>.json`: the `current.json` pointers this host
/// owes the mirror at `url`.
pub fn owed_path(state: &State, url: &str) -> PathBuf {
    state
        .root()
        .join("progress")
        .join("mirror-owed")
        .join(format!("hwm-{}.json", sha256_hex(url.as_bytes())))
}

/// Pointers whose batches the mirror's `HEAD.json` may already name, but whose `current.json`
/// the mirror is not yet known to hold.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Owed {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Url recorded for this owed.
    pub url: String,
    /// Pointers recorded for this owed.
    pub pointers: BTreeSet<PointerKey>,
}

impl Owed {
    /// Performs the load operation and returns an error if it cannot be completed.
    pub fn load(state: &State, url: &str) -> Result<Owed> {
        let path = owed_path(state, url);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Owed {
                    schema: OWED_SCHEMA.into(),
                    url: url.into(),
                    pointers: BTreeSet::new(),
                })
            }
            Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
        };
        let owed: Owed = serde_json::from_slice(&bytes).with_context(|| {
            format!(
                "{} does not parse. It is the record of mirror pointers still owed; refusing to \
                 push without it, because a push that forgets a debt reports success over a \
                 stale mirror",
                path.display()
            )
        })?;
        ensure!(
            owed.schema == OWED_SCHEMA && owed.url == url,
            "{} is not the owed-pointer record for {url}",
            path.display()
        );
        Ok(owed)
    }

    /// Performs the save operation and returns an error if it cannot be completed.
    pub fn save(&self, state: &State) -> Result<()> {
        atomic_write(
            &owed_path(state, &self.url),
            &crate::canon::canonical_json(self)?,
        )
    }
}

/// The mirror's copy of a pointer may only be replaced when it is an *earlier* state of that
/// pointer on this host's chain (or the same record). Anything else is another host's decision and
/// overwriting it would lose a record that is the tip of a chain we do not have.
fn check_pointer_ancestor(
    m: &Mirror,
    key: &PointerKey,
    theirs_bytes: &[u8],
    ours_bytes: &[u8],
    history: Option<&Vec<String>>,
) -> Result<()> {
    let rel = key.rel();
    let refuse = |detail: String| -> anyhow::Error {
        anyhow::anyhow!(
            "{rel}: the mirror at {} holds a record this host's ledger never accepted at that \
             pointer — {detail}. Refusing to overwrite it. Ingest the source that produced it on \
             this host first, or push to a mirror this host's chain covers.",
            m.url
        )
    };
    let theirs: HwmRecord = serde_json::from_slice(theirs_bytes)
        .map_err(|e| refuse(format!("its bytes do not parse as a record: {e}")))?;
    let ours: HwmRecord =
        serde_json::from_slice(ours_bytes).context("parse this host's own pointer")?;
    if theirs.provenance.record_sha256 == ours.provenance.record_sha256 {
        return Ok(());
    }
    let hist = history.map(Vec::as_slice).unwrap_or(&[]);
    let at = |sha: &str| hist.iter().position(|h| h == sha);
    match (
        at(&theirs.provenance.record_sha256),
        at(&ours.provenance.record_sha256),
    ) {
        (Some(t), Some(o)) if t < o => Ok(()),
        (Some(_), Some(_)) => Err(refuse(format!(
            "it holds {} which this host's ledger accepted *after* the {} we would write",
            theirs.provenance.record_sha256, ours.provenance.record_sha256
        ))),
        _ => Err(refuse(format!(
            "it holds record {}, which is not in this host's accepted history for that pointer",
            theirs.provenance.record_sha256
        ))),
    }
}

/// The ledger's events parsed.
pub fn events(state: &State) -> Result<Vec<HwmEvent>> {
    Ledger::open(state.root(), "hwm")
        .events()?
        .iter()
        .enumerate()
        .map(|(i, v)| parse_event(i, v))
        .collect()
}

#[derive(Clone, Debug, Serialize)]
/// `ShowEntry` is a record for the show entry contract in this module: The high-water-mark ledger: admission rules, ingest, show, verify and freeze.
pub struct ShowEntry {
    /// State pointer associated with this record.
    pub pointer: PointerKey,
    /// Record recorded for this show entry.
    pub record: HwmRecord,
}

/// Pointers for a dataset (all tasks unless `task`), optionally one cut. Bound before unbound,
/// then by score under the metric direction, so the first entry is the dataset's best.
pub fn show(
    state: &State,
    dataset_hash: &str,
    task: Option<&str>,
    cut: Option<&str>,
) -> Result<Vec<ShowEntry>> {
    let mut out: Vec<ShowEntry> = state
        .pointers()?
        .into_iter()
        .filter(|(k, _)| k.dataset_hash == dataset_hash)
        .filter(|(k, _)| task.is_none_or(|t| k.task == t))
        .filter(|(k, _)| cut.is_none_or(|c| k.cut_id == c))
        .map(|(pointer, record)| ShowEntry { pointer, record })
        .collect();
    out.sort_by(|a, b| {
        b.record
            .evaluator
            .is_bound()
            .cmp(&a.record.evaluator.is_bound())
            .then_with(|| {
                let dir = a.record.score.metric.direction();
                let (x, y) = (a.record.score.value, b.record.score.value);
                match dir {
                    crate::metric::Direction::Lower => x.total_cmp(&y),
                    crate::metric::Direction::Higher => y.total_cmp(&x),
                }
            })
    });
    Ok(out)
}

/// The latest ledger event for an identity (`identity_sha256` or the source's legacy key).
pub fn find_identity(state: &State, id: &str) -> Result<Option<HwmEvent>> {
    Ok(events(state)?.into_iter().rfind(|e| {
        e.record.identity_sha256 == id || e.record.legacy_identity_sha256.as_deref() == Some(id)
    }))
}

#[derive(Clone, Debug, Serialize, Default)]
/// `VerifyReport` is a record for the verify report contract in this module: The high-water-mark ledger: admission rules, ingest, show, verify and freeze.
pub struct VerifyReport {
    /// Ledger recorded for this verify report.
    pub ledger: LedgerReport,
    /// Events recorded for this verify report.
    pub events: u64,
    /// Accepted recorded for this verify report.
    pub accepted: u64,
    /// Rejected recorded for this verify report.
    pub rejected: u64,
    /// Rederived from ledger order, never from a recorded flag: `accepted` split into
    /// pointer bootstraps and committed improvements.
    pub initialized: u64,
    /// Improved recorded for this verify report.
    pub improved: u64,
    /// Identities recorded for this verify report.
    pub identities: u64,
    /// Pointers on disk recorded for this verify report.
    pub pointers_on_disk: u64,
    /// Blobs checked recorded for this verify report.
    pub blobs_checked: u64,
    /// Champion records recorded for this verify report.
    pub champion_records: u64,
    /// Champion chain links walked recorded for this verify report.
    pub champion_chain_links_walked: u64,
    /// Champion chain links unresolved recorded for this verify report.
    pub champion_chain_links_unresolved: u64,
    /// Collection of breaks values carried by this verify report.
    pub breaks: Vec<VerifyBreak>,
}

impl VerifyReport {
    /// Performs the ok operation.
    pub fn ok(&self) -> bool {
        self.breaks.is_empty() && self.ledger.ok()
    }
    /// Performs the total breaks operation.
    pub fn total_breaks(&self) -> usize {
        self.breaks.len() + self.ledger.breaks.len()
    }
}

#[derive(Clone, Debug, Serialize)]
/// `VerifyBreak` is a record for the verify break contract in this module: The high-water-mark ledger: admission rules, ingest, show, verify and freeze.
pub struct VerifyBreak {
    /// Record kind or category represented by this value.
    pub kind: String,
    /// Subject recorded for this verify break.
    pub subject: String,
    /// Detail recorded for this verify break.
    pub detail: String,
}

/// Walk every chain: the batch chain, the per-pointer ledger chain, on-disk pointers and blobs,
/// every record's self-consistency, and (with `event_dirs`) each champion's engine chain back to
/// `improvement_number` 0 through `event-<sha>.json` files.
pub fn verify(state: &State, event_dirs: &[PathBuf]) -> Result<VerifyReport> {
    let ledger = Ledger::open(state.root(), "hwm");
    let mut report = VerifyReport {
        ledger: ledger.verify()?,
        ..Default::default()
    };
    let mut brk = |kind: &str, subject: String, detail: String| {
        report.breaks.push(VerifyBreak {
            kind: kind.into(),
            subject,
            detail,
        });
    };
    // Lenient throughout: a malformed event is a break, not an abort, and so is a batch that
    // cannot be read at all — `ledger.verify()` above has already recorded an `unreadable` break
    // for each of those, so reading the rest adds diagnosis instead of replacing it.
    let mut events = Vec::new();
    for (i, v) in ledger.events_readable()?.iter().enumerate() {
        match parse_event(i, v) {
            Ok(e) => events.push(e),
            Err(err) => brk(
                "event-unparseable",
                format!("event #{i}"),
                format!("{err:#}"),
            ),
        }
    }
    let mut expected: BTreeMap<PointerKey, HwmRecord> = BTreeMap::new();
    let mut identities: BTreeSet<String> = BTreeSet::new();
    let mut chain_cache: BTreeMap<String, Result<Champion, String>> = BTreeMap::new();
    let mut walked = 0u64;
    let mut unresolved = 0u64;
    let mut champion_records = 0u64;
    let mut blobs_checked = 0u64;
    for e in &events {
        report.events += 1;
        let r = &e.record;
        let subject = r.identity_sha256.clone();
        identities.insert(subject.clone());
        if e.schema != EVENT_SCHEMA {
            brk("event-schema", subject.clone(), e.schema.clone());
        }
        if let Err(err) = r.validate() {
            brk("record-invalid", subject.clone(), err.to_string());
        }
        if !e.pointer.matches(r) {
            brk(
                "pointer-key",
                subject.clone(),
                "event pointer != record identity".into(),
            );
        }
        let blob = match state.read_blob(&r.provenance.record_sha256) {
            Ok(Some(b)) => {
                blobs_checked += 1;
                Some(b)
            }
            Ok(None) => {
                brk(
                    "blob-missing",
                    subject.clone(),
                    r.provenance.record_sha256.clone(),
                );
                None
            }
            Err(err) => {
                brk("blob-corrupt", subject.clone(), err.to_string());
                None
            }
        };
        let is_champion = r.provenance.source.starts_with("champions");
        if is_champion {
            champion_records += 1;
            if let Some(blob) = &blob {
                match Champion::parse(blob) {
                    Ok(ch) => {
                        let (w, u) =
                            verify_champion(&ch, r, event_dirs, &mut chain_cache, &mut |k, d| {
                                brk(k, subject.clone(), d)
                            });
                        walked += w;
                        unresolved += u;
                    }
                    Err(err) => brk("champion-parse", subject.clone(), err.to_string()),
                }
            }
        }
        let derived = match (e.outcome.as_str(), expected.contains_key(&e.pointer)) {
            ("accepted", false) => Some(Transition::Initialized),
            ("accepted", true) => Some(Transition::Improved),
            _ => None,
        };
        if let Some(recorded) = e.transition {
            let committed = matches!(recorded, Transition::Initialized | Transition::Improved);
            if derived.map_or(committed, |d| d != recorded) {
                brk(
                    "transition-mismatch",
                    subject.clone(),
                    format!(
                        "{} event records {} at {}",
                        e.outcome,
                        recorded.as_str(),
                        e.pointer.rel()
                    ),
                );
            }
        }
        match e.outcome.as_str() {
            "accepted" => {
                report.accepted += 1;
                let prev = expected.get(&e.pointer);
                if prev.is_some() {
                    report.improved += 1;
                } else {
                    report.initialized += 1;
                }
                let prev_sha = prev.map(|p| p.provenance.record_sha256.clone());
                if e.displaced_record_sha256 != prev_sha {
                    brk(
                        "pointer-chain",
                        subject.clone(),
                        format!(
                            "displaced {:?} != previous accepted {:?} at {}",
                            e.displaced_record_sha256,
                            prev_sha,
                            e.pointer.rel()
                        ),
                    );
                }
                if let Some(p) = prev {
                    if !r
                        .score
                        .metric
                        .direction()
                        .strictly_improves(r.score.value, p.score.value)
                    {
                        brk(
                            "non-improvement-accepted",
                            subject.clone(),
                            format!("{} vs incumbent {}", r.score.value, p.score.value),
                        );
                    }
                    if p.evaluator.is_bound() && !r.evaluator.is_bound() {
                        brk("unbound-displaced-bound", subject.clone(), e.pointer.rel());
                    }
                    if !is_champion
                        && (r.prev_event_sha256.as_deref() != Some(&p.provenance.record_sha256)
                            || r.improvement_number != p.improvement_number + 1)
                    {
                        brk(
                            "record-chain",
                            subject.clone(),
                            format!(
                                "prev {:?}/{} does not continue {}/{}",
                                r.prev_event_sha256,
                                r.improvement_number,
                                p.provenance.record_sha256,
                                p.improvement_number
                            ),
                        );
                    }
                } else if !is_champion
                    && (r.prev_event_sha256.is_some() || r.improvement_number != 0)
                {
                    brk(
                        "record-chain",
                        subject.clone(),
                        "first accepted record of a pointer is not improvement 0".into(),
                    );
                }
                expected.insert(e.pointer.clone(), r.clone());
            }
            "rejected" => report.rejected += 1,
            other => brk("event-outcome", subject.clone(), other.to_string()),
        }
    }
    let (on_disk, pointer_errors) = state.pointers_readable()?;
    for (path, error) in pointer_errors {
        brk("pointer-unreadable", path, error);
    }
    report.pointers_on_disk = on_disk.len() as u64;
    let mut seen = BTreeSet::new();
    for (key, record) in &on_disk {
        if let Err(error) = record.validate() {
            brk("pointer-invalid", key.rel(), format!("{error:#}"));
        }
        if !key.matches(record) {
            brk(
                "pointer-identity",
                key.rel(),
                "pointer key differs from record".into(),
            );
        }
        seen.insert(key.clone());
        match expected.get(key) {
            Some(exp) if exp.provenance.record_sha256 == record.provenance.record_sha256 => {}
            Some(exp) => brk(
                "pointer-stale",
                key.rel(),
                format!(
                    "disk {} != ledger {}",
                    record.provenance.record_sha256, exp.provenance.record_sha256
                ),
            ),
            None => brk("pointer-orphan", key.rel(), "no accepted event".into()),
        }
    }
    for key in expected.keys() {
        if !seen.contains(key) {
            brk(
                "pointer-missing",
                key.rel(),
                "accepted in ledger, absent on disk".into(),
            );
        }
    }
    report.identities = identities.len() as u64;
    report.blobs_checked = blobs_checked;
    report.champion_records = champion_records;
    report.champion_chain_links_walked = walked;
    report.champion_chain_links_unresolved = unresolved;
    Ok(report)
}

fn load_event(
    dirs: &[PathBuf],
    key: &str,
    sha: &str,
    cache: &mut BTreeMap<String, Result<Champion, String>>,
) -> Result<Champion, String> {
    if let Some(c) = cache.get(sha) {
        return c.clone();
    }
    let mut result: Result<Champion, String> = Err(format!("event-{sha}.json not found for {key}"));
    for d in dirs {
        let p = d.join(key).join(format!("event-{sha}.json"));
        if p.exists() {
            result = fs::read(&p)
                .map_err(|e| e.to_string())
                .and_then(|b| Champion::parse(&b).map_err(|e| e.to_string()))
                .and_then(|c| {
                    let got = c.event_sha256().map_err(|e| e.to_string())?;
                    if got == sha {
                        Ok(c)
                    } else {
                        Err(format!("{} hashes to {got}, not its name", p.display()))
                    }
                });
            break;
        }
    }
    cache.insert(sha.to_string(), result.clone());
    result
}

/// (links walked, links unresolved for lack of event files)
fn verify_champion(
    ch: &Champion,
    r: &HwmRecord,
    dirs: &[PathBuf],
    cache: &mut BTreeMap<String, Result<Champion, String>>,
    brk: &mut dyn FnMut(&str, String),
) -> (u64, u64) {
    if let Err(e) = ch.validate() {
        brk("champion-invalid", e.to_string());
    }
    let key = match ch.identity.key() {
        Ok(k) => k,
        Err(e) => {
            brk("champion-key", e.to_string());
            return (0, 0);
        }
    };
    if r.legacy_identity_sha256.as_deref() != Some(&key) {
        brk(
            "legacy-identity",
            format!(
                "record {:?} != Identity::key {key}",
                r.legacy_identity_sha256
            ),
        );
    }
    if r.sealed != ch.identity.is_sealed() {
        brk(
            "sealed-flag",
            format!(
                "sealed {} for partition {}",
                r.sealed, ch.identity.partition
            ),
        );
    }
    if r.score.value != ch.rmse_original_units
        || r.prev_event_sha256 != ch.previous_event_sha256
        || r.improvement_number != ch.improvement_number
        || r.ts_unix != ch.timestamp
    {
        brk(
            "champion-fields",
            "score/chain/timestamp differ from the champion".into(),
        );
    }
    if dirs.is_empty() {
        return (0, u64::from(ch.previous_event_sha256.is_some()));
    }
    let mut walked = 0u64;
    // The record's own event file must hold these exact bytes.
    match ch.event_sha256() {
        Ok(own) => {
            if let Err(e) = load_event(dirs, &key, &own, cache) {
                brk("champion-own-event", e);
            }
        }
        Err(e) => brk("champion-hash", e.to_string()),
    }
    let mut cur = ch.clone();
    let mut hops = 0u64;
    while let Some(prev_sha) = cur.previous_event_sha256.clone() {
        hops += 1;
        if hops > 10_000 {
            brk("chain-loop", key.clone());
            break;
        }
        let prev = match load_event(dirs, &key, &prev_sha, cache) {
            Ok(p) => p,
            Err(e) => {
                brk("chain-link", e);
                break;
            }
        };
        walked += 1;
        if prev.identity != cur.identity {
            brk(
                "chain-identity",
                format!("event {prev_sha} has another identity"),
            );
        }
        if prev.improvement_number + 1 != cur.improvement_number {
            brk(
                "chain-number",
                format!(
                    "{} -> {} at {prev_sha}",
                    prev.improvement_number, cur.improvement_number
                ),
            );
        }
        if cur.previous_rmse_original_units != Some(prev.rmse_original_units)
            || cur.rmse_original_units >= prev.rmse_original_units
        {
            brk(
                "chain-score",
                format!(
                    "{} did not strictly improve on {} at {prev_sha}",
                    cur.rmse_original_units, prev.rmse_original_units
                ),
            );
        }
        if let Err(e) = prev.validate() {
            brk("chain-event-invalid", format!("{prev_sha}: {e}"));
        }
        cur = prev;
    }
    if cur.improvement_number != 0 && cur.previous_event_sha256.is_none() {
        brk(
            "chain-root",
            format!("root has improvement_number {}", cur.improvement_number),
        );
    }
    (walked, 0)
}

#[derive(Clone, Debug, Serialize)]
/// `FreezeEntry` is a record for the freeze entry contract in this module: The high-water-mark ledger: admission rules, ingest, show, verify and freeze.
pub struct FreezeEntry {
    /// State pointer associated with this record.
    pub pointer: PointerKey,
    /// SHA-256 digest identifying the associated identity.
    pub identity_sha256: String,
    /// SHA-256 digest identifying the associated legacy identity.
    pub legacy_identity_sha256: Option<String>,
    /// SHA-256 digest identifying the associated record.
    pub record_sha256: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Metric used to evaluate the represented result.
    pub metric: String,
    /// Measured or derived value carried by this result.
    pub value: f64,
    /// SHA-256 digest identifying the evaluator used for this evidence.
    pub evaluator_sha256: String,
    /// Blob recorded for this freeze entry.
    pub blob: String,
}

#[derive(Clone, Debug, Serialize)]
/// `FreezeManifest` is a record for the freeze manifest contract in this module: The high-water-mark ledger: admission rules, ingest, show, verify and freeze.
pub struct FreezeManifest {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Archive snapshots do not establish train-side OOF teaching eligibility.
    pub scope: String,
    /// Teaching eligible recorded for this freeze manifest.
    pub teaching_eligible: bool,
    /// Optional ledger head value; `None` means it is not available.
    pub ledger_head: Option<crate::ledger::Head>,
    /// Include unbound recorded for this freeze manifest.
    pub include_unbound: bool,
    /// Pointers recorded for this freeze manifest.
    pub pointers: u64,
    /// Included recorded for this freeze manifest.
    pub included: u64,
    /// Excluded sealed recorded for this freeze manifest.
    pub excluded_sealed: u64,
    /// Excluded unbound recorded for this freeze manifest.
    pub excluded_unbound: u64,
    /// Unix timestamp when this event was recorded.
    pub ts_unix: u64,
    /// Collection of entries values carried by this freeze manifest.
    pub entries: Vec<FreezeEntry>,
}

/// A snapshot restricted to HWM winners: every pointer that is not sealed and (unless
/// `include_unbound`) is replayable, with its blob copied under `<out>/blobs/sha256/`.
///
/// Verifies the chain first, like every other path that writes or exports. This is the export
/// boundary — a frozen corpus is the artifact a later reader trusts *without* re-deriving it, and
/// the `ledger_head` this manifest carries reads as provenance for the chain it came from. It used
/// to be the one path in the crate that took `HEAD.json` at its word: `ingest` and
/// `scores::append` call `require_intact`, `find` and `rows` go through the fail-closed
/// `events()`, and `freeze` parsed `HEAD.json`, walked the pointers and wrote a manifest stamping
/// `batches` and `events_total` it had never checked.
///
/// The blobs were never the problem — `read_blob` re-hashes every one it copies — so the exported
/// *content* was sound. What was unsound was the manifest's implicit claim about where it came
/// from, which is exactly the claim a corpus consumer cannot re-derive for itself.
///
/// The check is `require_intact`, the same one `ingest` uses, and it buys the same thing: the
/// batch chain this host holds is internally consistent. It is not a claim that the tree is
/// perfect. Two things it does not cover, both documented rather than papered over: a removed tail
/// with `HEAD.json` rewritten to match is invisible to any self-contained check (see the
/// [`crate::ledger`] module docs), and a pointer-level break with an intact chain — a
/// `pointer-orphan` standing on its own — is reported by `hwm verify` and not by this. Run
/// `hwm verify` before freezing a corpus you intend to publish; the runbook says so.
///
/// There is deliberately no `--force`. No workflow in this tree freezes a ledger it has reason to
/// believe is broken: `freeze` has exactly two callers, both of them tests that build a clean tree
/// first, and salvage from a damaged store is `hwm blob` and `hwm show --raw`, which read one
/// record at a time and make no provenance claim at all.
pub fn freeze(state: &State, out: &Path, include_unbound: bool) -> Result<FreezeManifest> {
    let ledger = Ledger::open(state.root(), "hwm");
    ledger.require_intact().with_context(|| {
        format!(
            "refusing to freeze into {}: nothing was written. A frozen corpus is trusted without \
             being re-derived, and its manifest's `ledger_head` is a provenance claim about the \
             chain it came from, so it must not be written from a chain this host cannot walk. \
             Run `hwm verify` for the full report and repair the ledger first",
            out.display()
        )
    })?;
    let head = ledger.head()?;
    let pointers = state.pointers()?;
    let mut manifest = FreezeManifest {
        schema: FREEZE_SCHEMA.into(),
        scope: "diagnostic_hwm_archive".into(),
        teaching_eligible: false,
        ledger_head: head,
        include_unbound,
        pointers: pointers.len() as u64,
        included: 0,
        excluded_sealed: 0,
        excluded_unbound: 0,
        ts_unix: now_unix(),
        entries: Vec::new(),
    };
    fs::create_dir_all(out)?;
    for (pointer, r) in pointers {
        if r.sealed {
            manifest.excluded_sealed += 1;
            continue;
        }
        if !r.replayable && !include_unbound {
            manifest.excluded_unbound += 1;
            continue;
        }
        let sha = r.provenance.record_sha256.clone();
        let blob = state
            .read_blob(&sha)?
            .with_context(|| format!("blob {sha} missing for {}", pointer.rel()))?;
        let rel = State::blob_rel(&sha);
        let dest = out.join(&rel);
        // A destination path is content-addressed, but existence is not integrity: skipping the
        // write because the path was occupied published a manifest asserting `record_sha256`
        // over whatever bytes happened to sit there -- a truncated, corrupted or planted file
        // exported as though it had been verified, with `verify` unable to catch it because the
        // source ledger it checks is intact. Re-export stays idempotent (identical bytes are
        // accepted in silence); disagreeing bytes fail the freeze before any manifest is written.
        if dest.exists() {
            let existing = fs::read(&dest)
                .with_context(|| format!("reading existing export {}", dest.display()))?;
            let existing_sha = sha256_hex(&existing);
            ensure!(
                existing_sha == sha,
                "refusing to freeze {}: {} already holds bytes hashing to {existing_sha}, \
                 not the {sha} its path claims",
                pointer.rel(),
                dest.display()
            );
        } else {
            atomic_write(&dest, &blob)?;
        }
        manifest.entries.push(FreezeEntry {
            pointer,
            identity_sha256: r.identity_sha256.clone(),
            legacy_identity_sha256: r.legacy_identity_sha256.clone(),
            record_sha256: sha,
            program_sha256: r.genome.program_sha256.clone(),
            metric: r.score.metric.to_string(),
            value: r.score.value,
            evaluator_sha256: r.evaluator.sha256.clone(),
            blob: rel,
        });
        manifest.included += 1;
    }
    atomic_write(
        &out.join("manifest.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

/// Directory size and file count (for reports).
pub fn tree_size(root: &Path) -> Result<(u64, u64)> {
    let (mut bytes, mut files) = (0u64, 0u64);
    for e in walkdir::WalkDir::new(root) {
        let e = e?;
        if e.file_type().is_file() {
            bytes += e.metadata()?.len();
            files += 1;
        }
    }
    Ok((bytes, files))
}

/// Performs the source value operation.
pub fn source_value(kind: &str, path: &Path, extra: Value) -> Value {
    let mut v =
        json!({"kind": kind, "path": path.display().to_string(), "tool": crate::TOOL_VERSION});
    if let (Some(dst), Some(src)) = (v.as_object_mut(), extra.as_object()) {
        for (k, val) in src {
            dst.insert(k.clone(), val.clone());
        }
    }
    v
}

/// Performs the fail if operation and returns an error if it cannot be completed.
pub fn fail_if(cond: bool, msg: &str) -> Result<()> {
    if cond {
        bail!("{msg}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{Direction, Metric};
    use crate::record::*;

    pub fn record(
        dataset: &str,
        cut: &str,
        evaluator: &str,
        metric: Metric,
        value: f64,
    ) -> HwmRecord {
        let identity = HwmIdentity {
            comparison: None,
            task: "regression".into(),
            dataset_hash: dataset.into(),
            cut_id: crate::canon::sha256_hex(cut.as_bytes()),
            evaluator_sha256: evaluator.into(),
            vocabulary_sha256: UNBOUND.into(),
            protocol: "test".into(),
        };
        let dsl = format!("(pipe X (ridge alpha={value}))");
        let blob = format!("{dataset}/{cut}/{evaluator}/{value}");
        HwmRecord {
            schema: HWM_SCHEMA.into(),
            identity_sha256: identity.sha256().unwrap(),
            identity,
            legacy_identity_sha256: None,
            genome: Genome {
                program_sha256: crate::canon::sha256_hex(dsl.as_bytes()),
                dsl,
                semantic_sha256: None,
            },
            score: Score {
                metric,
                value,
                direction: metric.direction(),
                units: "original_target_units".into(),
            },
            cost: Cost::default(),
            evaluator: Evaluator {
                sha256: evaluator.into(),
                version: "test".into(),
            },
            benchmark_valid: false,
            sealed: false,
            replayable: evaluator != UNBOUND,
            provenance: Provenance {
                source: "test".into(),
                run_id: "run".into(),
                git_sha: UNBOUND.into(),
                bundle_id: UNBOUND.into(),
                record_sha256: crate::canon::sha256_hex(blob.as_bytes()),
            },
            prev_event_sha256: None,
            improvement_number: 0,
            ts_unix: 1,
        }
    }

    #[test]
    fn comparison_identity_never_collides_across_evaluator_vocabulary_protocol_or_context() {
        use crate::record::ComparisonContext;
        let mut base = record("d", "c", BOUND, Metric::Auc, 0.5);
        base.identity.task = "binary".into();
        base.identity.comparison = Some(ComparisonContext {
            numerical_policy: "strict-v1".into(),
            class_order: vec!["no".into(), "yes".into()],
            checkpoint_sha256: None,
            stage: "train_oof".into(),
            aggregation: "mean_fold_loss".into(),
            label_role: "development".into(),
            sealed: false,
            benchmark_valid: false,
            recipe_encoding: "native-canonical-dsl-sha256/v1".into(),
        });
        base.identity_sha256 = base.identity.sha256().unwrap();
        let mut variants = Vec::new();
        let mut changed = base.clone();
        changed.identity.evaluator_sha256 = "a".repeat(64);
        changed.evaluator.sha256 = changed.identity.evaluator_sha256.clone();
        variants.push(changed);
        let mut changed = base.clone();
        changed.identity.vocabulary_sha256 = "b".repeat(64);
        variants.push(changed);
        let mut changed = base.clone();
        changed.identity.protocol.push_str("-other");
        variants.push(changed);
        let mut changed = base.clone();
        changed
            .identity
            .comparison
            .as_mut()
            .unwrap()
            .numerical_policy
            .push_str("-other");
        variants.push(changed);
        let mut changed = base.clone();
        changed
            .identity
            .comparison
            .as_mut()
            .unwrap()
            .class_order
            .reverse();
        variants.push(changed);
        let mut changed = base.clone();
        changed
            .identity
            .comparison
            .as_mut()
            .unwrap()
            .checkpoint_sha256 = Some("c".repeat(64));
        variants.push(changed);
        let mut changed = base.clone();
        changed.identity.comparison.as_mut().unwrap().stage = "confirm".into();
        variants.push(changed);
        let mut changed = base.clone();
        changed.identity.comparison.as_mut().unwrap().aggregation = "pooled".into();
        variants.push(changed);
        let mut changed = base.clone();
        changed.identity.comparison.as_mut().unwrap().label_role = "test".into();
        variants.push(changed);
        let mut changed = base.clone();
        changed.sealed = true;
        changed.identity.comparison.as_mut().unwrap().sealed = true;
        variants.push(changed);
        let mut changed = base.clone();
        changed.benchmark_valid = true;
        changed
            .identity
            .comparison
            .as_mut()
            .unwrap()
            .benchmark_valid = true;
        variants.push(changed);
        let key = PointerKey::of(&base);
        for mut candidate in variants {
            candidate.identity_sha256 = candidate.identity.sha256().unwrap();
            candidate.score.value = 0.9;
            assert_ne!(PointerKey::of(&candidate), key);
            assert_ne!(PointerKey::of(&candidate).rel(), key.rel());
            assert_eq!(
                decide(Some(&base), &candidate),
                Decision::Rejected(RejectReason::IdentityMismatch)
            );
        }
    }

    #[test]
    fn raw_negative_r2_gains_are_not_clipped() {
        let old = record("d", "c", BOUND, Metric::R2, -5.0);
        let improved = record("d", "c", BOUND, Metric::R2, -2.0);
        assert_eq!(decide(Some(&old), &improved), Decision::Accepted);
        assert_eq!(
            decide(Some(&improved), &old),
            Decision::Rejected(RejectReason::NotBetter)
        );
    }

    #[test]
    fn legacy_pointer_remains_diagnostic_without_fallback_or_rewrite() {
        let dir = tempfile::tempdir().unwrap();
        let state = State::at(dir.path());
        let old = record("d", "c", UNBOUND, Metric::Rmse, 0.1);
        let mut legacy = PointerKey::of(&old);
        legacy.identity_sha256 = None;
        let bytes = serde_json::to_vec(&old).unwrap();
        atomic_write(&state.pointer_path(&legacy).unwrap(), &bytes).unwrap();
        assert!(state.read_pointer(&PointerKey::of(&old)).unwrap().is_none());
        assert!(state
            .write_pointer(&legacy, &old, Some(&old.provenance.record_sha256))
            .is_err());
        let current = record("d", "c", UNBOUND, Metric::Rmse, 0.2);
        let rep = ingest(
            &state,
            None,
            vec![Candidate {
                record: current.clone(),
                blob: b"d/c/unbound/0.2".to_vec(),
                chain_from_source: false,
            }],
            json!({"test":"legacy-preservation"}),
        )
        .unwrap();
        assert_eq!(rep.accepted, 1);
        assert_eq!(
            state
                .read_pointer(&PointerKey::of(&current))
                .unwrap()
                .unwrap()
                .score
                .value,
            0.2
        );
        assert_eq!(
            fs::read(state.pointer_path(&legacy).unwrap()).unwrap(),
            bytes
        );
        assert_eq!(state.pointers().unwrap().len(), 2);
        let manifest = crate::eligibility::freeze_teaching(&state).unwrap();
        assert!(manifest.entries.is_empty());
        assert_eq!(manifest.excluded_diagnostic, 2);
    }

    const BOUND: &str = "b5ccee8e33af34313c51896f52466f87f2f8544dcb01672e6d6b4832d1ac4393";

    #[test]
    fn admission_rules() {
        let inc = record("d", "c", BOUND, Metric::Rmse, 0.5);
        assert_eq!(decide(None, &inc), Decision::Accepted);
        assert_eq!(
            decide(Some(&inc), &record("d", "c", BOUND, Metric::Rmse, 0.4)),
            Decision::Accepted
        );
        assert_eq!(
            decide(Some(&inc), &record("d", "c", BOUND, Metric::Rmse, 0.5)),
            Decision::Rejected(RejectReason::Tie)
        );
        assert_eq!(
            decide(Some(&inc), &record("d", "c", BOUND, Metric::Rmse, 0.6)),
            Decision::Rejected(RejectReason::NotBetter)
        );
        // Unbound never displaces bound, even with a better score.
        assert_eq!(
            decide(Some(&inc), &record("d", "c", UNBOUND, Metric::Rmse, 0.1)),
            Decision::Rejected(RejectReason::UnboundCannotDisplaceBound)
        );
        // Evaluator changes start separate comparisons; historical scores cannot displace them.
        let unb = record("d", "c", UNBOUND, Metric::R2, 0.8);
        assert_eq!(
            decide(Some(&unb), &record("d", "c", BOUND, Metric::R2, 0.9)),
            Decision::Rejected(RejectReason::IdentityMismatch)
        );
        assert_eq!(
            decide(Some(&unb), &record("d", "c", BOUND, Metric::R2, 0.8)),
            Decision::Rejected(RejectReason::IdentityMismatch)
        );
        assert_eq!(
            decide(Some(&unb), &record("d", "c", UNBOUND, Metric::R2, 0.7)),
            Decision::Rejected(RejectReason::NotBetter)
        );
        assert_eq!(
            decide(Some(&unb), &record("d", "c", UNBOUND, Metric::Auc, 0.99)),
            Decision::Rejected(RejectReason::MetricMismatch)
        );
        let mut bad = record("d", "c", BOUND, Metric::Rmse, 0.1);
        bad.score.direction = Direction::Higher;
        assert_eq!(
            decide(None, &bad),
            Decision::Rejected(RejectReason::Invalid)
        );
    }

    #[test]
    fn transitions_separate_initialization_ties_non_improvement_and_improvement() {
        let dir = tempfile::tempdir().unwrap();
        let state = State::at(dir.path());
        let cand = |v: f64, ts: u64, variant: &str| {
            let mut r = record("d", "c", UNBOUND, Metric::R2, v);
            r.genome.dsl = format!("(pipe X (ridge alpha={v}) {variant})");
            r.genome.program_sha256 = crate::canon::sha256_hex(r.genome.dsl.as_bytes());
            r.ts_unix = ts;
            let blob = format!("d/c/{v}/{variant}").into_bytes();
            r.provenance.record_sha256 = crate::canon::sha256_hex(&blob);
            Candidate {
                record: r,
                blob,
                chain_from_source: false,
            }
        };
        let mut noncomparable = cand(0.9, 4, "other-metric");
        noncomparable.record.score.metric = Metric::Auc;
        let mut invalid = cand(0.95, 5, "invalid");
        invalid.record.score.direction = Direction::Lower;
        let mut ineligible = cand(0.99, 6, "unqualified-train-oof");
        ineligible.record.identity.comparison = Some(ComparisonContext {
            numerical_policy: "strict-v1".into(),
            stage: "train_oof".into(),
            aggregation: "mean_fold_loss".into(),
            label_role: "development".into(),
            sealed: false,
            benchmark_valid: false,
            recipe_encoding: "native-canonical-dsl-sha256/v1".into(),
            class_order: vec![],
            checkpoint_sha256: None,
        });
        ineligible.record.identity_sha256 = ineligible.record.identity.sha256().unwrap();
        let first = ingest(
            &state,
            None,
            vec![
                cand(0.5, 1, "first"),
                cand(0.5, 2, "equal"),
                cand(0.4, 3, "worse"),
                noncomparable,
                invalid,
                ineligible,
            ],
            json!({"kind": "test"}),
        )
        .unwrap();
        // The same improving record delivered twice is one improvement and one duplicate.
        let second = ingest(
            &state,
            None,
            vec![cand(0.7, 7, "better"), cand(0.7, 7, "better")],
            json!({"kind": "test"}),
        )
        .unwrap();
        let counts = |pairs: &[(&str, u64)]| {
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect::<BTreeMap<_, _>>()
        };
        assert_eq!(
            first.transitions,
            counts(&[
                ("initialized", 1),
                ("tie", 1),
                ("not-improved", 1),
                ("noncomparable", 1),
                ("invalid", 1),
                ("ineligible", 1)
            ])
        );
        assert_eq!((first.accepted, first.rejected), (1, 5));
        assert_eq!(
            second.transitions,
            counts(&[("improved", 1), ("duplicate", 1)])
        );
        let v = verify(&state, &[]).unwrap();
        assert_eq!((v.accepted, v.initialized, v.improved), (2, 1, 1));
        assert!(!v.breaks.iter().any(|b| b.kind == "transition-mismatch"));
        let transitions: Vec<Value> = Ledger::open(state.root(), "hwm")
            .events()
            .unwrap()
            .into_iter()
            .map(|event| event["transition"].clone())
            .collect();
        assert_eq!(
            transitions,
            [
                "initialized",
                "tie",
                "not-improved",
                "noncomparable",
                "invalid",
                "ineligible",
                "improved",
                "duplicate"
            ]
            .map(|t| json!(t))
        );
        let key = PointerKey::of(&cand(0.7, 7, "better").record);
        let current = state.read_pointer(&key).unwrap().unwrap();
        assert_eq!((current.score.value, current.improvement_number), (0.7, 1));
    }

    fn delivery(v: f64, ts: u64) -> Candidate {
        let mut record = record("d", "c", UNBOUND, Metric::R2, v);
        record.ts_unix = ts;
        Candidate {
            blob: format!("d/c/unbound/{v}").into_bytes(),
            record,
            chain_from_source: false,
        }
    }

    fn pointer_file(state: &State) -> PathBuf {
        state
            .pointer_path(&PointerKey::of(&delivery(0.0, 0).record))
            .unwrap()
    }

    fn head(state: &State) -> Option<Vec<u8>> {
        fs::read(Ledger::open(state.root(), "hwm").root().join("HEAD.json")).ok()
    }

    fn outcomes(state: &State) -> Vec<(Value, Value)> {
        Ledger::open(state.root(), "hwm")
            .events()
            .unwrap()
            .into_iter()
            .map(|event| (event["transition"].clone(), event["reason"].clone()))
            .collect()
    }

    #[test]
    fn a_pointer_behind_the_ledger_is_rolled_forward_before_any_decision() {
        let dir = tempfile::tempdir().unwrap();
        let state = State::at(dir.path());
        ingest(&state, None, vec![delivery(0.5, 1)], json!({"k": 1})).unwrap();
        let lagging = fs::read(pointer_file(&state)).unwrap();
        ingest(&state, None, vec![delivery(0.9, 2)], json!({"k": 2})).unwrap();
        // Crash after the batch accepting 0.9 was durable, before its pointer moved.
        fs::write(pointer_file(&state), &lagging).unwrap();
        let report = ingest(&state, None, vec![delivery(0.7, 3)], json!({"k": 3})).unwrap();
        assert_eq!(
            report.accepted, 0,
            "a candidate weaker than the ledger's HWM was accepted against a stale pointer"
        );
        assert_eq!((report.pointers_written, report.pointers_recovered), (1, 1));
        let current = state
            .read_pointer(&PointerKey::of(&delivery(0.0, 0).record))
            .unwrap()
            .unwrap();
        assert_eq!((current.score.value, current.improvement_number), (0.9, 1));
        let v = verify(&state, &[]).unwrap();
        assert!(v.ok(), "{:?}", v.breaks);
        assert_eq!((v.initialized, v.improved), (1, 1));
        // The first acceptance of a pointer can be lost the same way.
        fs::remove_file(pointer_file(&state)).unwrap();
        let report = ingest(&state, None, vec![delivery(0.8, 4)], json!({"k": 4})).unwrap();
        assert_eq!(report.accepted, 0);
        assert!(verify(&state, &[]).unwrap().ok());
    }

    #[test]
    fn a_pointer_the_ledger_never_accepted_is_refused_before_any_write() {
        let dir = tempfile::tempdir().unwrap();
        let state = State::at(dir.path());
        ingest(&state, None, vec![delivery(0.5, 1)], json!({"k": 1})).unwrap();
        let accepted = delivery(0.5, 1).record.provenance.record_sha256;
        state
            .write_pointer(
                &PointerKey::of(&delivery(0.0, 0).record),
                &delivery(0.95, 2).record,
                Some(&accepted),
            )
            .unwrap();
        let (forged, before) = (fs::read(pointer_file(&state)).unwrap(), head(&state));
        let result = ingest(&state, None, vec![delivery(0.6, 3)], json!({"k": 3}));
        assert!(
            result.is_err(),
            "decided against a pointer the ledger never accepted"
        );
        assert_eq!(head(&state), before);
        assert_eq!(fs::read(pointer_file(&state)).unwrap(), forged);
    }

    #[test]
    fn recovery_rolls_lagging_pointers_forward_and_leaves_unaccepted_ones() {
        let dir = tempfile::tempdir().unwrap();
        let state = State::at(dir.path());
        ingest(&state, None, vec![delivery(0.5, 1)], json!({"k": 1})).unwrap();
        let lagging = fs::read(pointer_file(&state)).unwrap();
        ingest(&state, None, vec![delivery(0.9, 2)], json!({"k": 2})).unwrap();
        fs::write(pointer_file(&state), &lagging).unwrap();
        assert!(verify(&state, &[])
            .unwrap()
            .breaks
            .iter()
            .any(|b| b.kind == "pointer-stale"));
        let report = recover(&state).unwrap();
        assert_eq!(
            (
                report.pointers,
                report.recovered.len(),
                report.refused.len()
            ),
            (1, 1, 0)
        );
        let v = verify(&state, &[]).unwrap();
        assert!(v.ok(), "{:?}", v.breaks);
        assert!(recover(&state).unwrap().recovered.is_empty());
        let tip = delivery(0.9, 2).record.provenance.record_sha256;
        state
            .write_pointer(
                &PointerKey::of(&delivery(0.0, 0).record),
                &delivery(0.95, 3).record,
                Some(&tip),
            )
            .unwrap();
        let forged = fs::read(pointer_file(&state)).unwrap();
        let report = recover(&state).unwrap();
        assert!(report.recovered.is_empty() && report.refused.len() == 1);
        assert_eq!(fs::read(pointer_file(&state)).unwrap(), forged);
    }

    #[test]
    fn redelivery_of_an_accepted_record_is_a_duplicate_not_a_tie() {
        let dir = tempfile::tempdir().unwrap();
        let state = State::at(dir.path());
        ingest(&state, None, vec![delivery(0.5, 1)], json!({"k": 1})).unwrap();
        ingest(&state, None, vec![delivery(0.5, 1)], json!({"k": 2})).unwrap();
        ingest(&state, None, vec![delivery(0.9, 2)], json!({"k": 3})).unwrap();
        ingest(&state, None, vec![delivery(0.5, 1)], json!({"k": 4})).unwrap();
        let mut equal = delivery(0.9, 5);
        equal.blob = b"another record with an equal score".to_vec();
        equal.record.provenance.record_sha256 = crate::canon::sha256_hex(&equal.blob);
        ingest(&state, None, vec![equal], json!({"k": 5})).unwrap();
        assert_eq!(
            outcomes(&state),
            [
                (json!("initialized"), Value::Null),
                (json!("duplicate"), json!("tie")),
                (json!("improved"), Value::Null),
                (json!("duplicate"), json!("not-better")),
                (json!("tie"), json!("tie")),
            ]
        );
    }

    #[test]
    fn redelivery_of_a_rejected_record_is_a_duplicate_not_a_second_decision() {
        let dir = tempfile::tempdir().unwrap();
        let state = State::at(dir.path());
        let counts = |pairs: &[(&str, u64)]| {
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect::<BTreeMap<_, _>>()
        };
        ingest(&state, None, vec![delivery(0.5, 1)], json!({"k": 1})).unwrap();
        // A worse record is decided once; delivered again, in a later batch or twice in one
        // batch, it is a duplicate delivery, never a second not-improved decision.
        let first = ingest(&state, None, vec![delivery(0.4, 2)], json!({"k": 2})).unwrap();
        let again = ingest(&state, None, vec![delivery(0.4, 2)], json!({"k": 3})).unwrap();
        let twice = ingest(
            &state,
            None,
            vec![delivery(0.3, 3), delivery(0.3, 3)],
            json!({"k": 4}),
        )
        .unwrap();
        assert_eq!(first.transitions, counts(&[("not-improved", 1)]));
        assert_eq!(again.transitions, counts(&[("duplicate", 1)]));
        assert_eq!(
            twice.transitions,
            counts(&[("not-improved", 1), ("duplicate", 1)])
        );
        // An invalid record redelivered is a duplicate too.
        let mut invalid = delivery(0.2, 4);
        invalid.record.score.direction = crate::metric::Direction::Lower;
        let mut invalid_again = delivery(0.2, 4);
        invalid_again.record.score.direction = crate::metric::Direction::Lower;
        ingest(&state, None, vec![invalid], json!({"k": 5})).unwrap();
        let report = ingest(&state, None, vec![invalid_again], json!({"k": 6})).unwrap();
        assert_eq!(report.transitions, counts(&[("duplicate", 1)]));
        // Positive control: another record with the same worse score is its own decision.
        let mut other = delivery(0.4, 7);
        other.blob = b"another record, the same worse score".to_vec();
        other.record.provenance.record_sha256 = crate::canon::sha256_hex(&other.blob);
        let report = ingest(&state, None, vec![other], json!({"k": 7})).unwrap();
        assert_eq!(report.transitions, counts(&[("not-improved", 1)]));
        assert_eq!(
            outcomes(&state),
            [
                (json!("initialized"), Value::Null),
                (json!("not-improved"), json!("not-better")),
                (json!("duplicate"), json!("not-better")),
                (json!("not-improved"), json!("not-better")),
                (json!("duplicate"), json!("not-better")),
                (json!("invalid"), json!("invalid")),
                (json!("duplicate"), json!("invalid")),
                (json!("not-improved"), json!("not-better")),
            ]
        );
        // The pointer is untouched by every rejection, and the ledger verifies.
        let current = state
            .read_pointer(&PointerKey::of(&delivery(0.0, 0).record))
            .unwrap()
            .unwrap();
        assert_eq!((current.score.value, current.improvement_number), (0.5, 0));
        // The chain verifies; the only breaks are the invalid record's own, as before.
        let v = verify(&state, &[]).unwrap();
        assert!(v.ledger.ok(), "{:?}", v.ledger.breaks);
        assert!(
            v.breaks.iter().all(|b| b.kind == "record-invalid"),
            "{:?}",
            v.breaks
        );
        assert_eq!((v.accepted, v.rejected), (1, 7));
    }

    #[test]
    fn a_revoked_record_is_rejected_and_never_initializes_or_improves() {
        let dir = tempfile::tempdir().unwrap();
        let state = State::at(dir.path());
        let repo = tempfile::tempdir().unwrap();
        let revoked = delivery(0.9, 1);
        crate::revocation::testing::revoke(
            &state,
            repo.path(),
            &[
                ("record", &revoked.blob, &[]),
                ("unrelated", b"unrelated", &[]),
            ],
            &["record"],
        );
        let key = PointerKey::of(&delivery(0.0, 0).record);
        // Rejected as `revoked`, before any comparison, and the pointer never exists.
        let report = ingest(&state, None, vec![revoked], json!({"k": 1})).unwrap();
        assert_eq!((report.accepted, report.rejected), (0, 1));
        assert_eq!(report.rejects[0].reason, "revoked");
        assert!(state.read_pointer(&key).unwrap().is_none());
        // Positive control: an unrevoked record initializes the same pointer.
        let report = ingest(&state, None, vec![delivery(0.5, 2)], json!({"k": 2})).unwrap();
        assert_eq!(report.accepted, 1);
        // The revoked record never improves on it either; its redelivery is a duplicate.
        let report = ingest(&state, None, vec![delivery(0.9, 1)], json!({"k": 3})).unwrap();
        assert_eq!(
            (report.accepted, report.rejects[0].reason.as_str()),
            (0, "revoked")
        );
        assert_eq!(
            outcomes(&state),
            [
                (json!("revoked"), json!("revoked")),
                (json!("initialized"), Value::Null),
                (json!("duplicate"), json!("revoked")),
            ]
        );
        let current = state.read_pointer(&key).unwrap().unwrap();
        assert_eq!(current.score.value, 0.5);
        let v = verify(&state, &[]).unwrap();
        assert!(v.ok(), "{:?}", v.breaks);
    }

    #[test]
    fn ingest_moves_pointers_only_on_strict_improvement_and_logs_rejects() {
        let dir = tempfile::tempdir().unwrap();
        let state = State::at(dir.path());
        let cand = |v: f64, ts: u64| {
            let mut r = record("d", "c", UNBOUND, Metric::R2, v);
            r.ts_unix = ts;
            let blob = format!("d/c/unbound/{v}").into_bytes();
            Candidate {
                record: r,
                blob,
                chain_from_source: false,
            }
        };
        let rep = ingest(
            &state,
            None,
            vec![cand(0.5, 3), cand(0.7, 1), cand(0.7, 2), cand(0.6, 4)],
            json!({"kind": "test"}),
        )
        .unwrap();
        // Ordered by ts: 0.7 (accepted, #0), 0.7 (tie), 0.5 (not better), 0.6 (not better).
        assert_eq!(
            (rep.accepted, rep.rejected, rep.pointers_written),
            (1, 3, 1)
        );
        assert_eq!(
            rep.rejects
                .iter()
                .map(|r| r.reason.as_str())
                .collect::<Vec<_>>(),
            vec!["tie", "not-better", "not-better"]
        );
        let key = PointerKey::of(&cand(0.7, 1).record);
        let cur = state.read_pointer(&key).unwrap().unwrap();
        assert_eq!((cur.score.value, cur.improvement_number), (0.7, 0));
        let rep = ingest(&state, None, vec![cand(0.9, 9)], json!({"kind": "test"})).unwrap();
        assert_eq!((rep.accepted, rep.rejected), (1, 0));
        let cur = state.read_pointer(&key).unwrap().unwrap();
        assert_eq!((cur.score.value, cur.improvement_number), (0.9, 1));
        assert_eq!(
            cur.prev_event_sha256.as_deref(),
            Some(crate::canon::sha256_hex(b"d/c/unbound/0.7").as_str())
        );
        let v = verify(&state, &[]).unwrap();
        assert!(v.ok(), "{:?}", v.breaks);
        assert_eq!(
            (v.events, v.accepted, v.rejected, v.pointers_on_disk),
            (5, 2, 3, 1)
        );
        // A stale pointer on disk is a break.
        let stale = cand(0.7, 1).record;
        let mut bytes = serde_json::to_vec_pretty(&stale).unwrap();
        bytes.push(b'\n');
        atomic_write(&state.pointer_path(&key).unwrap(), &bytes).unwrap();
        let v = verify(&state, &[]).unwrap();
        assert!(v.breaks.iter().any(|b| b.kind == "pointer-stale"));
    }
}
