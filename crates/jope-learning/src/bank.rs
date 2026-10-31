//! The champion bank ("future targets"): pipeline champions the HWM ledger accepted as
//! improvements, kept content-addressed and append-only, and exported, ranked within one HWM
//! comparison identity at a time, as the `jope.pipeline-seeds/v1` file
//! `jope pipeline run --seed-genomes` and `jope pipeline evolve --seeds` consume.
//!
//! Admission (`bank add`) re-verifies everything it cites before a byte is written:
//!
//! * the pipeline receipt re-derives from its run directory
//!   ([`jope_pipeline::evidence::verify_champion_run`]);
//! * the HWM ledger walks clean ([`crate::hwm::verify`]) and holds an *accepted* event for exactly
//!   this receipt whose transition is `improved` — a first record of a pointer (`initialized`)
//!   bootstraps the high-water mark without beating anything, and every rejection is refused;
//! * that event's record is exactly the mapping of this receipt
//!   ([`crate::sources::pipeline::identity`], genome, score) and its blob is the receipt's bytes;
//! * when `--lineage` is given, the lineage verifies and names this receipt.
//!
//! Entries live as events of the `bank` ledger (`$JOPE_STATE/bank/ledger`), the same sealed,
//! hash-chained JSONL batches as the HWM ledger, so the bank is append-only by construction. An
//! entry's address is the sha256 of its canonical JSON; adding an entry that is already banked
//! writes nothing.

use crate::canon::{canonical_sha256, is_hex64, sha256_hex};
use crate::hwm::{self, HwmEvent, Transition};
use crate::ledger::{now_unix, Ledger};
use crate::state::State;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

/// Schema identifier for entry values.
pub const ENTRY_SCHEMA: &str = "jope.pipeline-bank-entry/v1";
/// Schema identifier for event values.
pub const EVENT_SCHEMA: &str = "jope.pipeline-bank-event/v1";
/// Ledger name used in bank entries and events.
pub const LEDGER: &str = "bank";

/// The HWM decision an entry rests on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HwmDecision {
    /// The HWM ledger batch (ULID) and the event's position in it.
    pub batch: String,
    /// Monotone sequence number assigned to this event or row.
    pub seq: u64,
    /// sha256 of the event's canonical JSON as the ledger holds it.
    pub event_sha256: String,
    /// Always `improved`.
    pub transition: String,
    /// State pointer associated with this record.
    pub pointer: String,
    /// The record the improvement displaced.
    pub displaced_record_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `BankEntry` is a record for the bank entry contract in this module: The champion bank ("future targets"): pipeline champions the HWM ledger accepted as improvements, kept content-addressed and append-only, and exported, ranked within one HWM comparison identity at a time, as the.
pub struct BankEntry {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Task identity for the comparison represented by this record.
    pub task: String,
    /// Hash identifying the dataset used by this record.
    pub dataset_hash: String,
    /// The HWM comparison identity the champion improved under.
    pub identity_sha256: String,
    /// Identifier for the data cut represented by this record.
    pub cut_id: String,
    /// Canonical program or genome representation.
    pub genome: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Mean held-out RMSE on that identity's cuts.
    pub validation_rmse: f64,
    /// sha256 of the pipeline receipt (also the HWM record's blob).
    pub receipt_sha256: String,
    /// sha256 of the `jope.pipeline-lineage/v1` that produced it, when given.
    pub lineage_sha256: Option<String>,
    /// High-water-mark decision associated with this candidate.
    pub hwm: HwmDecision,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `BankEvent` is a record for the bank event contract in this module: The champion bank ("future targets"): pipeline champions the HWM ledger accepted as improvements, kept content-addressed and append-only, and exported, ranked within one HWM comparison identity at a time, as the.
pub struct BankEvent {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// The entry's content address: sha256 of its canonical JSON.
    pub entry_sha256: String,
    /// Bank entry represented by this event.
    pub entry: BankEntry,
    /// Unix timestamp when this event was recorded.
    pub ts_unix: u64,
}

#[derive(Clone, Debug, Serialize)]
/// `AddReport` is a record for the add report contract in this module: The champion bank ("future targets"): pipeline champions the HWM ledger accepted as improvements, kept content-addressed and append-only, and exported, ranked within one HWM comparison identity at a time, as the.
pub struct AddReport {
    /// SHA-256 digest identifying the associated entry.
    pub entry_sha256: String,
    /// Whether the entry was already present before this operation.
    pub already_banked: bool,
    /// Bank entry represented by this event.
    pub entry: BankEntry,
    /// Ledger batch receipt produced by the operation, if one was appended.
    pub batch: Option<crate::ledger::BatchReceipt>,
}

/// Every banked entry, oldest first, each re-checked against its content address.
pub fn list(state: &State) -> Result<Vec<BankEvent>> {
    let ledger = Ledger::open(state.root(), LEDGER);
    if ledger.head()?.is_none() && ledger.batches()?.is_empty() {
        return Ok(Vec::new());
    }
    ledger.require_intact()?;
    let mut out = Vec::new();
    for (index, value) in ledger.events()?.into_iter().enumerate() {
        let event: BankEvent = serde_json::from_value(value)
            .with_context(|| format!("bank event #{index} does not parse"))?;
        ensure!(
            event.schema == EVENT_SCHEMA
                && event.entry.schema == ENTRY_SCHEMA
                && event.entry_sha256 == canonical_sha256(&event.entry)?,
            "bank event #{index}: its entry does not hash to its address {}",
            event.entry_sha256
        );
        out.push(event);
    }
    Ok(out)
}

/// The accepted HWM event for `receipt_sha256`, with its batch id, among the ledger's batches.
fn find_decision(state: &State, receipt_sha256: &str) -> Result<(HwmEvent, HwmDecision)> {
    let ledger = Ledger::open(state.root(), "hwm");
    let mut found: Vec<(HwmEvent, HwmDecision)> = Vec::new();
    for (_, path) in ledger.batches()? {
        let (events, seal, _, _) = Ledger::read_batch(&path)?;
        for (index, value) in events.iter().enumerate() {
            let event = hwm::parse_event(index, value)?;
            if event.record.provenance.source != crate::sources::pipeline::SOURCE
                || event.record.provenance.record_sha256 != receipt_sha256
            {
                continue;
            }
            let transition = event
                .transition
                .map(Transition::as_str)
                .unwrap_or("unrecorded")
                .to_owned();
            let decision = HwmDecision {
                batch: seal.batch.clone(),
                seq: event.seq,
                event_sha256: canonical_sha256(value)?,
                transition,
                pointer: event.pointer.rel(),
                displaced_record_sha256: event.displaced_record_sha256.clone().unwrap_or_default(),
            };
            found.push((event, decision));
        }
    }
    let Some(accepted) = found.iter().find(|(e, _)| e.outcome == "accepted") else {
        match found.last() {
            Some((event, decision)) => bail!(
                "the HWM ledger rejected this receipt ({}: {}); only HWM-improved champions may be \
                 banked",
                decision.transition,
                event.reason.as_deref().unwrap_or("no reason recorded")
            ),
            None => bail!(
                "the HWM ledger holds no decision for receipt {receipt_sha256}; ingest it first \
                 with `jope-learning hwm ingest --source pipeline <receipt>`"
            ),
        }
    };
    ensure!(
        accepted.1.transition == Transition::Improved.as_str(),
        "the HWM decision for this receipt is `{}`, not `improved`: only a champion that beat \
         the comparable incumbent may be banked",
        accepted.1.transition
    );
    Ok(accepted.clone())
}

/// Admit one pipeline champion. Everything is verified before anything is written.
pub fn add(state: &State, receipt_path: &Path, lineage_path: Option<&Path>) -> Result<AddReport> {
    let run = jope_pipeline::evidence::verify_champion_run(receipt_path)
        .context("the receipt does not verify; nothing was banked")?;
    let receipt_bytes = std::fs::read(receipt_path)?;
    ensure!(
        sha256_hex(&receipt_bytes) == run.receipt_sha256,
        "{} changed while it was being verified",
        receipt_path.display()
    );
    let report = hwm::verify(state, &[])?;
    ensure!(
        report.ok(),
        "the HWM ledger does not verify ({} break(s)); nothing was banked. Run `hwm verify`",
        report.total_breaks()
    );
    let (event, decision) = find_decision(state, &run.receipt_sha256)?;
    let receipt = &run.receipt;
    let best = receipt
        .search
        .best
        .as_ref()
        .context("no measured champion")?;
    let identity = crate::sources::pipeline::identity(receipt)?;
    ensure!(
        event.record.identity == identity
            && event.record.genome.dsl == best.genome
            && event.record.score.value.to_bits() == best.validation_rmse.to_bits(),
        "the HWM event's record is not this receipt's champion under its comparison identity"
    );
    ensure!(
        state.read_blob(&run.receipt_sha256)?.as_deref() == Some(receipt_bytes.as_slice()),
        "the HWM record's blob is not this receipt"
    );
    let lineage_bytes = match lineage_path {
        Some(path) => {
            let bytes = std::fs::read(path)?;
            jope_pipeline::lineage::verify(path, false)
                .context("the lineage does not verify; nothing was banked")?;
            let lineage = jope_pipeline::lineage::read_lineage(path)?;
            ensure!(
                sha256_hex(&bytes) == sha256_hex(&std::fs::read(path)?),
                "{} changed while it was being verified",
                path.display()
            );
            ensure!(
                lineage
                    .rounds
                    .iter()
                    .chain(lineage.control.iter())
                    .any(|r| r.receipt_sha256 == run.receipt_sha256),
                "the lineage does not name this receipt"
            );
            Some(bytes)
        }
        None => None,
    };
    let lineage_sha256 = lineage_bytes.as_deref().map(sha256_hex);
    let entry = BankEntry {
        schema: ENTRY_SCHEMA.into(),
        task: event.record.identity.task.clone(),
        dataset_hash: event.record.identity.dataset_hash.clone(),
        identity_sha256: event.record.identity_sha256.clone(),
        cut_id: event.record.identity.cut_id.clone(),
        genome: best.genome.clone(),
        program_sha256: best.program_sha256.clone(),
        validation_rmse: best.validation_rmse,
        receipt_sha256: run.receipt_sha256.clone(),
        lineage_sha256,
        hwm: decision,
    };
    let entry_sha256 = canonical_sha256(&entry)?;
    let _lock = state.lock(LEDGER)?;
    if list(state)?.iter().any(|e| e.entry_sha256 == entry_sha256) {
        return Ok(AddReport {
            entry_sha256,
            already_banked: true,
            entry,
            batch: None,
        });
    }
    // The lineage the entry cites is kept, content-addressed, next to the receipt blob the HWM
    // ingest already stored.
    if let Some(bytes) = &lineage_bytes {
        state.put_blob(bytes)?;
    }
    let event = BankEvent {
        schema: EVENT_SCHEMA.into(),
        entry_sha256: entry_sha256.clone(),
        entry: entry.clone(),
        ts_unix: now_unix(),
    };
    let batch = Ledger::open(state.root(), LEDGER).append(
        &[serde_json::to_value(&event)?],
        json!({"kind": "pipeline-bank", "receipt": receipt_path.display().to_string(),
               "tool": crate::TOOL_VERSION}),
    )?;
    Ok(AddReport {
        entry_sha256,
        already_banked: false,
        entry,
        batch: Some(batch),
    })
}

#[derive(Clone, Debug, Serialize)]
/// `ExportReport` is a record for the export report contract in this module: The champion bank ("future targets"): pipeline champions the HWM ledger accepted as improvements, kept content-addressed and append-only, and exported, ranked within one HWM comparison identity at a time, as the.
pub struct ExportReport {
    /// Destination path for the generated output.
    pub out: String,
    /// SHA-256 digest identifying the associated seeds.
    pub seeds_sha256: String,
    /// Hash identifying the dataset used by this record.
    pub dataset_hash: String,
    /// The one HWM comparison identity the exported champions were ranked within.
    pub identity_sha256: String,
    /// Genomes recorded for this export report.
    pub genomes: usize,
    /// SHA-256 digest identifying the associated bank head.
    pub bank_head_sha256: Option<String>,
}

/// This wrapper has no Deserialize implementation. A diagnostic `SeedsFile` or copied role
/// string cannot be presented as a global export. Serialize only after filtering, before ranking.
#[derive(Debug, Serialize)]
pub struct GlobalBankExport {
    purpose: crate::replay::admission::Purpose,
    entries: Vec<(BankEntry, crate::replay::admission::ReplayAdmission)>,
}
impl GlobalBankExport {
    /// Performs the purpose operation and returns an error if it cannot be completed.
    pub fn purpose(&self) -> crate::replay::admission::Purpose {
        self.purpose
    }
    /// Performs the entries operation and returns an error if it cannot be completed.
    pub fn entries(&self) -> &[(BankEntry, crate::replay::admission::ReplayAdmission)] {
        &self.entries
    }
}

/// Global learning/retrieval companion to the legacy search-seed export. Fail atomically on
/// any missing/restricted proof; never rank protected entries or silently discard refusals.
pub fn export_for(
    state: &State,
    verifier: &crate::replay::admission::AdmissionVerifier<'_>,
    purpose: crate::replay::admission::Purpose,
    proofs: &std::collections::BTreeMap<String, String>,
    at_ms: u64,
) -> Result<GlobalBankExport> {
    ensure!(
        purpose.is_global(),
        "bank global export requires a global purpose"
    );
    verifier.require_state(state)?;
    let mut entries = Vec::new();
    for event in list(state)? {
        let subject = canonical_sha256(&event.entry)?;
        let proof = proofs
            .get(&subject)
            .context("HELD: bank entry lacks exposure proof")?;
        let admission = verifier.admit(&subject, proof, purpose, None, at_ms)?;
        let quality = admission
            .labels()
            .quality
            .value()
            .context("HELD: bank score is not measured")?;
        ensure!(
            quality.metric == "rmse"
                && quality.value.f64_bits
                    == format!("{:016x}", event.entry.validation_rmse.to_bits()),
            "HELD: bank score disagrees with admitted measurement"
        );
        entries.push((event.entry, admission));
    }
    ensure!(!entries.is_empty(), "HELD: empty global bank export");
    Ok(GlobalBankExport { purpose, entries })
}

/// Write the best `top` distinct banked genomes of `dataset_hash` (optionally one task) under
/// one HWM comparison identity as a `jope.pipeline-seeds/v1` file that does not exist yet.
///
/// Scores measured under different identities (other cuts, fold seed, protocol or source commit)
/// are never ranked against each other: a mean over three folds is not comparable with a single
/// easier fold's. When the dataset's entries span more than one identity, `identity` must name
/// the one to rank within; the refusal lists them.
pub fn export(
    state: &State,
    dataset_hash: &str,
    task: Option<&str>,
    identity: Option<&str>,
    top: usize,
    out: &Path,
) -> Result<ExportReport> {
    ensure!(top >= 1, "--top must be at least 1");
    let entries: Vec<BankEntry> = list(state)?
        .into_iter()
        .map(|e| e.entry)
        .filter(|e| e.dataset_hash == dataset_hash && task.is_none_or(|t| e.task == t))
        .collect();
    ensure!(
        !entries.is_empty(),
        "the bank holds no champion for dataset {dataset_hash}"
    );
    let mut identities: std::collections::BTreeMap<&str, Vec<&BankEntry>> =
        std::collections::BTreeMap::new();
    for entry in &entries {
        identities
            .entry(entry.identity_sha256.as_str())
            .or_default()
            .push(entry);
    }
    let identity_sha256 = match identity {
        Some(wanted) => {
            ensure!(
                identities.contains_key(wanted),
                "the bank holds no champion for dataset {dataset_hash} under comparison identity \
                 {wanted}"
            );
            wanted.to_owned()
        }
        None => {
            if identities.len() > 1 {
                let listed: Vec<String> = identities
                    .iter()
                    .map(|(sha, banked)| {
                        let best = banked
                            .iter()
                            .map(|e| e.validation_rmse)
                            .fold(f64::INFINITY, f64::min);
                        format!(
                            "{sha} (cuts {}, {} entr{}, best mean held-out RMSE {best})",
                            banked[0].cut_id,
                            banked.len(),
                            if banked.len() == 1 { "y" } else { "ies" }
                        )
                    })
                    .collect();
                bail!(
                    "dataset {dataset_hash} is banked under {} comparison identities, whose scores \
                     are not comparable; choose one with --identity: {}",
                    identities.len(),
                    listed.join("; ")
                );
            }
            entries[0].identity_sha256.clone()
        }
    };
    let entries: Vec<BankEntry> = entries
        .into_iter()
        .filter(|e| e.identity_sha256 == identity_sha256)
        .collect();
    let champions: Vec<jope_pipeline::islands::Champion> = entries
        .iter()
        .map(|e| jope_pipeline::islands::Champion::by_rmse(e.genome.clone(), e.validation_rmse))
        .collect();
    let chosen = jope_pipeline::islands::next_seeds(&champions, top);
    let mut genomes = Vec::new();
    for champion in &chosen {
        let entry = entries
            .iter()
            .find(|e| e.genome == champion.genome && e.validation_rmse == champion.rmse)
            .context("an exported champion has no entry")?;
        genomes.push(jope_pipeline::seeds::SeedGenome {
            genome: entry.genome.clone(),
            program_sha256: entry.program_sha256.clone(),
            bank_entry_sha256: Some(canonical_sha256(entry)?),
            receipt_sha256: Some(entry.receipt_sha256.clone()),
        });
    }
    ensure!(is_hex64(&genomes[0].program_sha256), "malformed bank entry");
    let file = jope_pipeline::seeds::SeedsFile::new(Some(dataset_hash.to_owned()), genomes);
    let seeds_sha256 = jope_pipeline::seeds::write(out, &file)?;
    Ok(ExportReport {
        out: out.display().to_string(),
        seeds_sha256,
        dataset_hash: dataset_hash.to_owned(),
        identity_sha256,
        genomes: file.genomes.len(),
        bank_head_sha256: Ledger::open(state.root(), LEDGER).head()?.map(|h| h.sha256),
    })
}

/// The bank's entries as a JSON value for `bank list`.
pub fn listing(state: &State, dataset_hash: Option<&str>) -> Result<Value> {
    let entries: Vec<Value> = list(state)?
        .into_iter()
        .filter(|e| dataset_hash.is_none_or(|d| e.entry.dataset_hash == d))
        .map(|e| json!({"entry_sha256": e.entry_sha256, "ts_unix": e.ts_unix, "entry": e.entry}))
        .collect();
    Ok(json!({
        "bank_head": Ledger::open(state.root(), LEDGER).head()?,
        "entries": entries,
    }))
}
