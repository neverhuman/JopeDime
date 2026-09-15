//! Durable collection selection. This is scheduling evidence, never fit authority.
use super::GpJob;
use crate::{corpus::ContextRow, manifest::*};
use anyhow::{ensure, Context as _, Result};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};

pub const PROTOCOL: &str = "family_dataset_cut_strict_v1";
const LEDGER_VERSION: u32 = 2;
const IMPLEMENTATION: &str = "weighted_inflight_v2";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Context {
    pub family: String,
    pub dataset: String,
    pub partition: String,
    pub prepared_sha256: String,
    pub cut: usize,
}
impl Context {
    pub fn from_row(row: &ContextRow) -> Self {
        Self {
            family: row.family.clone(),
            dataset: row.dataset.clone(),
            partition: row.partition.clone(),
            prepared_sha256: row.prepared_sha256.clone(),
            cut: row.cut,
        }
    }
    fn for_job(root: &Path, job: &GpJob) -> Result<Self> {
        let path = resolve(root, &job.prepared)?.join("prepared.json");
        let bytes = fs::read(path)?;
        ensure!(
            hash(&bytes) == job.prepared_sha256,
            "coverage preparation changed"
        );
        let row: serde_json::Value = serde_json::from_slice(&bytes)?;
        ensure!(
            row["schema"] == "parallel.prepared.v1",
            "coverage preparation schema absent"
        );
        let text = |key| -> Result<String> {
            Ok(row[key]
                .as_str()
                .filter(|s| !s.is_empty())
                .context("coverage identity absent")?
                .into())
        };
        let value = Self {
            family: text("family")?,
            dataset: text("dataset")?,
            partition: text("meta_partition")?,
            prepared_sha256: job.prepared_sha256.clone(),
            cut: job.cut,
        };
        ensure!(
            value.partition != "meta_test",
            "sealed contexts cannot enter collection"
        );
        ensure!(
            row["cuts"]
                .as_array()
                .context("coverage cuts absent")?
                .iter()
                .any(|c| c["index"].as_u64() == Some(job.cut as u64)),
            "coverage cut absent"
        );
        Ok(value)
    }
    fn key(&self) -> Result<String> {
        Ok(hash(&serde_json::to_vec(self)?))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Counts {
    // None is deliberately distinct from a verified zero. Recovery must supply
    // the complete independently verified census before weighted admission.
    pub unique_strict_commits: Option<u64>,
    pub attempted: u64,
    // Dispatch attempts plus pending generation plans. Active requests are
    // already in attempted, so they must not be added a second time here.
    pub selection_exposure: u64,
    pub in_flight: u64,
    pub unknown: u64,
    pub ready_at: u64,
    // Population sampling is without replacement within that population only.
    // Work reserved by other populations remains eligible and is downweighted.
    pub selected_in_population: bool,
}

/// Frozen cut probability, applied after family and dataset balancing.
pub fn probabilities(counts: &[Counts]) -> Result<Vec<f64>> {
    ensure!(!counts.is_empty(), "empty coverage population");
    let weights = counts
        .iter()
        .map(|c| -> Result<f64> {
            let commits = c
                .unique_strict_commits
                .context("verified strict-commit census unavailable")?;
            let load = 1.0 + commits as f64 + c.in_flight as f64;
            Ok(1.0 / (load * load))
        })
        .collect::<Result<Vec<_>>>()?;
    let sum: f64 = weights.iter().sum();
    ensure!(sum.is_finite() && sum > 0.0, "invalid coverage weights");
    Ok(weights
        .iter()
        .map(|w| 0.10 / weights.len() as f64 + 0.90 * w / sum)
        .collect())
}

fn draw(sequence: u64) -> f64 {
    let bytes = hash(&[SEED.to_le_bytes(), sequence.to_le_bytes()].concat());
    let value = u64::from_str_radix(&bytes[..16], 16).expect("SHA256 hex");
    (value >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Canonical ordering makes the sweep and random selection independent of queue order.
fn select(rows: &[(Context, Counts)], sequence: u64, now: u64) -> Result<Option<usize>> {
    let mut families = BTreeMap::<&str, (u64, BTreeMap<&str, (u64, Vec<usize>)>)>::new();
    for (i, (context, counts)) in rows.iter().enumerate() {
        let family = families.entry(&context.family).or_default();
        family.0 = family.0.saturating_add(counts.selection_exposure);
        let dataset = family.1.entry(&context.dataset).or_default();
        dataset.0 = dataset.0.saturating_add(counts.selection_exposure);
        if !counts.selected_in_population && counts.unknown == 0 && counts.ready_at <= now {
            dataset.1.push(i);
        }
    }
    for (_, (_, datasets)) in families.iter_mut() {
        datasets.retain(|_, (_, indices)| !indices.is_empty());
    }
    let Some((_, (_, datasets))) = families
        .iter()
        .filter(|(_, (_, d))| !d.is_empty())
        .min_by_key(|(name, (attempts, _))| (*attempts, **name))
    else {
        return Ok(None);
    };
    let (_, (_, candidates)) = datasets
        .iter()
        .min_by_key(|(name, (attempts, _))| (*attempts, **name))
        .unwrap();
    let mut candidates = candidates.clone();
    candidates.sort_by(|a, b| rows[*a].0.cmp(&rows[*b].0));
    let weights = probabilities(
        &candidates
            .iter()
            .map(|i| rows[*i].1.clone())
            .collect::<Vec<_>>(),
    )?;
    // A deterministic sweep every tenth reservation prevents stochastic starvation.
    if sequence % 10 == 0 {
        return Ok(candidates
            .into_iter()
            .min_by_key(|i| (rows[*i].1.selection_exposure, rows[*i].0.clone())));
    }
    let mut cursor = draw(sequence);
    for (candidate, weight) in candidates.iter().zip(weights) {
        if cursor < weight {
            return Ok(Some(*candidate));
        }
        cursor -= weight;
    }
    Ok(candidates.last().copied())
}

pub struct Ledger {
    db: Connection,
}
impl Ledger {
    pub fn open(root: &Path) -> Result<Self> {
        fs::create_dir_all(root.join("runs"))?;
        let mut db = Connection::open(root.join("runs/gp-coverage.sqlite3"))?;
        db.busy_timeout(Duration::from_secs(5))?;
        // V1 excluded all busy contexts and gave each context one global plan.
        // It is not silently reinterpreted under the corrected selection rule.
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let version: u32 = tx.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        let tables: u64 = tx.query_row(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            [],
            |r| r.get(0),
        )?;
        ensure!(version == LEDGER_VERSION || (version == 0 && tables == 0),
            "coverage ledger version unavailable or incompatible; retain original ledger for explicit reconciliation");
        if version == 0 {
            tx.execute_batch(
                "CREATE TABLE coverage_contexts (
                identity TEXT PRIMARY KEY, metadata TEXT NOT NULL, strict_commits INTEGER,
                attempted INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0,
                unknown INTEGER NOT NULL DEFAULT 0, reused INTEGER NOT NULL DEFAULT 0,
                in_flight INTEGER NOT NULL DEFAULT 0, ready_at INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE coverage_requests (
                output TEXT PRIMARY KEY, job_sha256 TEXT NOT NULL, context TEXT NOT NULL,
                state TEXT NOT NULL, diagnosis TEXT, updated_at INTEGER NOT NULL,
                request_identity TEXT UNIQUE NOT NULL);
            CREATE TABLE coverage_events (
                sequence INTEGER PRIMARY KEY, output TEXT NOT NULL, state TEXT NOT NULL,
                diagnosis TEXT, timestamp INTEGER NOT NULL);
            CREATE TABLE coverage_plans (
                context TEXT NOT NULL, owner TEXT NOT NULL, timestamp INTEGER NOT NULL,
                PRIMARY KEY(owner,context));
            CREATE TABLE coverage_populations (
                owner TEXT PRIMARY KEY, request TEXT NOT NULL, selection TEXT NOT NULL);
            CREATE TABLE coverage_version (implementation TEXT PRIMARY KEY);
            INSERT INTO coverage_version VALUES('weighted_inflight_v2');
            PRAGMA user_version=2;",
            )?;
        }
        ensure!(
            tx.query_row("SELECT implementation FROM coverage_version", [], |r| {
                r.get::<_, String>(0)
            })? == IMPLEMENTATION,
            "coverage ledger implementation changed"
        );
        tx.commit()?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;")?;
        Ok(Self { db })
    }

    /// Selection and reservation share one immediate transaction across controllers.
    /// Reservations never expire into permission to retry an ambiguous start.
    pub fn reserve(&mut self, root: &Path, jobs: &[GpJob], now: u64) -> Result<Option<usize>> {
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut candidates = Vec::new();
        let mut indices = Vec::new();
        let mut unique = BTreeMap::new();
        let mut request_outputs = BTreeMap::new();
        for (index, job) in jobs.iter().enumerate() {
            let context = Context::for_job(root, job)?;
            let key = context.key()?;
            let request_identity = request_identity(job, &context)?;
            if let Some(output) = request_outputs.insert(request_identity.clone(), &job.output) {
                ensure!(
                    output == &job.output,
                    "duplicate collection request under different outputs"
                );
            }
            let bound_output: Option<String> = tx
                .query_row(
                    "SELECT output FROM coverage_requests WHERE request_identity=?1",
                    [&request_identity],
                    |r| r.get(0),
                )
                .optional()?;
            ensure!(bound_output.as_deref().is_none_or(|p| Some(p) == job.output.to_str()),
                "collection request already bound to another output; reuse or reconcile original request");
            let metadata = serde_json::to_string(&context)?;
            tx.execute(
                "INSERT OR IGNORE INTO coverage_contexts(identity,metadata) VALUES(?1,?2)",
                params![key, metadata],
            )?;
            let old: Option<(String, String)> = tx
                .query_row(
                    "SELECT job_sha256,state FROM coverage_requests WHERE output=?1",
                    [job.output.to_string_lossy()],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;
            if let Some((digest, state)) = old {
                ensure!(
                    digest == hash(&serde_json::to_vec(job)?),
                    "coverage request identity changed"
                );
                if state != "failed_before_start" {
                    continue;
                }
            }
            // Existing uncommitted search output predates this ledger and needs
            // authority reconciliation. A renamed destination is never a retry permit.
            let output = resolve(root, &job.output)?;
            if old_is_untracked(&tx, job)?
                && ["search-request.json", "progress.json", "attempts.jsonl"]
                    .iter()
                    .any(|name| output.join(name).exists())
                && !output.join("search.json").exists()
            {
                tx.execute(
                    "UPDATE coverage_contexts SET unknown=unknown+1 WHERE identity=?1",
                    [&key],
                )?;
                tx.execute("INSERT INTO coverage_requests VALUES(?1,?2,?3,'unknown','unreconciled historical output',?4,?5)",
                    params![job.output.to_string_lossy(), hash(&serde_json::to_vec(job)?), key, now, request_identity])?;
                continue;
            }
            if unique.insert(key.clone(), ()).is_some() {
                continue;
            }
            let counts = context_counts(&tx, &key)?;
            candidates.push((context, counts));
            indices.push(index);
        }
        let sequence: u64 = tx.query_row(
            "SELECT count(*) FROM coverage_events WHERE state='reserved'",
            [],
            |r| r.get(0),
        )?;
        // Commit discovered ambiguous history even if missing counts reject selection.
        let selected = select(&candidates, sequence, now);
        let result = match selected {
            Ok(Some(index)) => {
                let job = &jobs[indices[index]];
                let key = candidates[index].0.key()?;
                tx.execute("INSERT INTO coverage_requests VALUES(?1,?2,?3,'reserved',NULL,?4,?5)
                    ON CONFLICT(output) DO UPDATE SET state='reserved',diagnosis=NULL,updated_at=excluded.updated_at",
                    params![job.output.to_string_lossy(),hash(&serde_json::to_vec(job)?),key,now,
                        request_identity(job, &candidates[index].0)?])?;
                tx.execute("UPDATE coverage_contexts SET attempted=attempted+1,in_flight=in_flight+1 WHERE identity=?1", [&key])?;
                // Transfer this population's pending plan into a dispatch, so
                // the same obligation is not counted as both planned and active.
                tx.execute("DELETE FROM coverage_plans WHERE context=?1 AND substr(?2,1,length(owner)+1)=owner||'/'",
                    params![key,job.output.to_string_lossy()])?;
                tx.execute(
                    "INSERT INTO coverage_events(output,state,timestamp) VALUES(?1,'reserved',?2)",
                    params![job.output.to_string_lossy(), now],
                )?;
                Ok(Some(indices[index]))
            }
            Ok(None) => Ok(None),
            Err(error) => Err(error),
        };
        tx.commit()?;
        result
    }

    pub(super) fn finish(
        &mut self,
        job: &GpJob,
        state: &str,
        diagnosis: &str,
        now: u64,
    ) -> Result<()> {
        ensure!(
            matches!(state, "completed" | "unknown" | "failed_before_start"),
            "invalid coverage outcome"
        );
        ensure!(
            state == "completed" || !diagnosis.trim().is_empty(),
            "coverage failure diagnosis absent"
        );
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (key, digest, old): (String, String, String) = tx.query_row(
            "SELECT context,job_sha256,state FROM coverage_requests WHERE output=?1",
            [job.output.to_string_lossy()],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        ensure!(
            digest == hash(&serde_json::to_vec(job)?),
            "coverage completion identity changed"
        );
        ensure!(
            old == "reserved",
            "coverage outcome already recorded; reconciliation required"
        );
        tx.execute(
            "UPDATE coverage_requests SET state=?1,diagnosis=?2,updated_at=?3 WHERE output=?4",
            params![state, diagnosis, now, job.output.to_string_lossy()],
        )?;
        tx.execute(
            "UPDATE coverage_contexts SET in_flight=in_flight-1,
            failed=failed+?1,unknown=unknown+?2,ready_at=max(ready_at,?3) WHERE identity=?4",
            params![
                u64::from(state == "failed_before_start"),
                u64::from(state == "unknown"),
                if state == "failed_before_start" {
                    now.saturating_add(30)
                } else {
                    0
                },
                key
            ],
        )?;
        tx.execute(
            "INSERT INTO coverage_events(output,state,diagnosis,timestamp) VALUES(?1,?2,?3,?4)",
            params![job.output.to_string_lossy(), state, diagnosis, now],
        )?;
        if state == "completed" {
            tx.execute("DELETE FROM coverage_plans WHERE context=?1 AND substr(?2,1,length(owner)+1)=owner||'/'",
                params![key,job.output.to_string_lossy()])?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn status(&self) -> Result<serde_json::Value> {
        self.db.query_row("SELECT count(*),sum(strict_commits IS NULL),coalesce(sum(attempted),0),
            coalesce(sum(failed),0),coalesce(sum(unknown),0),coalesce(sum(reused),0),coalesce(sum(in_flight),0)
            FROM coverage_contexts", [], |r| Ok(serde_json::json!({"protocol":PROTOCOL,"ledger_version":LEDGER_VERSION,"implementation":IMPLEMENTATION,
                "contexts":r.get::<_,u64>(0)?,"unavailable_strict_counts":r.get::<_,Option<u64>>(1)?.unwrap_or(0),
                "attempted":r.get::<_,u64>(2)?,"failed_before_start":r.get::<_,u64>(3)?,
                "unknown":r.get::<_,u64>(4)?,"reused":r.get::<_,u64>(5)?,"unreconciled_reservations":r.get::<_,u64>(6)?,
                "strict_count_source":"recovered independent authority census required"}))).map_err(Into::into)
    }

    pub(super) fn tracks(&self, job: &GpJob) -> Result<bool> {
        Ok(self.db.query_row(
            "SELECT EXISTS(SELECT 1 FROM coverage_requests WHERE output=?1)",
            [job.output.to_string_lossy()],
            |r| r.get(0),
        )?)
    }

    /// An existing immutable search can resolve this scheduler's interrupted
    /// dispatch. This does not add strict commits or declare scientific validity.
    pub(super) fn reused(&mut self, job: &GpJob, now: u64, restored: bool) -> Result<()> {
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row: Option<(String, String, String)> = tx
            .query_row(
                "SELECT context,job_sha256,state FROM coverage_requests WHERE output=?1",
                [job.output.to_string_lossy()],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        if let Some((key, digest, state)) = row {
            ensure!(
                digest == hash(&serde_json::to_vec(job)?),
                "reused coverage identity changed"
            );
            tx.execute("UPDATE coverage_contexts SET reused=reused+?1,in_flight=in_flight-?2 WHERE identity=?3",
                params![u64::from(restored),u64::from(state == "reserved"),key])?;
            tx.execute("UPDATE coverage_requests SET state='completed',diagnosis='compatible durable search restored',updated_at=?1 WHERE output=?2",
                params![now,job.output.to_string_lossy()])?;
            tx.execute(
                "INSERT INTO coverage_events(output,state,timestamp) VALUES(?1,?2,?3)",
                params![
                    job.output.to_string_lossy(),
                    if restored { "reused" } else { "completed" },
                    now
                ],
            )?;
            tx.execute("DELETE FROM coverage_plans WHERE context=?1 AND substr(?2,1,length(owner)+1)=owner||'/'",
                params![key,job.output.to_string_lossy()])?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Freeze training-only identities before opening context payloads. The DB
    /// transaction retains the selection if file publication is interrupted.
    pub fn freeze_population(
        &mut self,
        root: &Path,
        corpus: &Path,
        owner: &Path,
        contexts: &[ContextRow],
        cuts_per_dataset: usize,
        now: u64,
    ) -> Result<serde_json::Value> {
        ensure!(cuts_per_dataset > 0, "coverage population size absent");
        let owner = owner
            .strip_prefix(root)
            .context("coverage owner outside workspace")?;
        relative(owner)?;
        let request = serde_json::json!({"protocol":PROTOCOL,"implementation":IMPLEMENTATION,"corpus_sha256":file_hash(&corpus.join("index.json"))?,
            "cuts_per_dataset":cuts_per_dataset,"partition":"meta_train","seed":SEED});
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<(String, String)> = tx
            .query_row(
                "SELECT request,selection FROM coverage_populations WHERE owner=?1",
                [owner.to_string_lossy()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        if let Some((bound, selection)) = existing {
            ensure!(
                serde_json::from_str::<serde_json::Value>(&bound)? == request,
                "coverage population request changed"
            );
            return Ok(serde_json::from_str(&selection)?);
        }
        let mut rows = Vec::new();
        let mut source_indices = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for (i, row) in contexts
            .iter()
            .enumerate()
            .filter(|(_, c)| c.partition == "meta_train")
        {
            let context = Context::from_row(row);
            let key = context.key()?;
            ensure!(
                seen.insert(key.clone()),
                "duplicate coverage context identity"
            );
            tx.execute(
                "INSERT OR IGNORE INTO coverage_contexts(identity,metadata) VALUES(?1,?2)",
                params![key, serde_json::to_string(&context)?],
            )?;
            let counts = context_counts(&tx, &key)?;
            rows.push((context, counts));
            source_indices.push(i);
        }
        let datasets = rows
            .iter()
            .map(|(c, _)| (&c.family, &c.dataset))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let requested = datasets
            .checked_mul(cuts_per_dataset)
            .context("coverage population overflow")?;
        let mut selected = Vec::new();
        let mut sequence: u64 = tx.query_row(
            "SELECT count(*) FROM coverage_events WHERE state='planned'",
            [],
            |r| r.get(0),
        )?;
        for _ in 0..requested {
            let Some(i) = select(&rows, sequence, now)? else {
                break;
            };
            let key = rows[i].0.key()?;
            tx.execute(
                "INSERT INTO coverage_plans VALUES(?1,?2,?3)",
                params![key, owner.to_string_lossy(), now],
            )?;
            selected
                .push(serde_json::json!({"context_index":source_indices[i],"identity":rows[i].0}));
            tx.execute("INSERT INTO coverage_events(output,state,diagnosis,timestamp) VALUES(?1,'planned',?2,?3)",
                params![owner.to_string_lossy(),key,now])?;
            rows[i].1.selected_in_population = true;
            rows[i].1.in_flight += 1;
            rows[i].1.selection_exposure += 1;
            sequence += 1;
        }
        ensure!(
            !selected.is_empty(),
            "no available training contexts for coverage population"
        );
        let selection = serde_json::json!({"schema":"parallel.coverage_selection.v1","request":request,
            "owner":owner,"contexts":selected,"requested_contexts":requested,"selected_contexts":selected.len(),
            "payloads_opened":false,"final_tests_opened":false});
        tx.execute(
            "INSERT INTO coverage_populations VALUES(?1,?2,?3)",
            params![
                owner.to_string_lossy(),
                serde_json::to_string(&request)?,
                serde_json::to_string(&selection)?
            ],
        )?;
        tx.commit()?;
        Ok(selection)
    }
}

fn context_counts(tx: &rusqlite::Transaction<'_>, key: &str) -> Result<Counts> {
    Ok(tx.query_row(
        "SELECT strict_commits,attempted,
        (SELECT count(*) FROM coverage_requests WHERE context=?1 AND state='reserved') +
        (SELECT count(*) FROM coverage_plans WHERE context=?1),
        (SELECT count(*) FROM coverage_requests WHERE context=?1 AND state='unknown'),ready_at,
        (SELECT count(*) FROM coverage_plans WHERE context=?1)
        FROM coverage_contexts WHERE identity=?1",
        [key],
        |r| {
            let attempted: u64 = r.get(1)?;
            let pending_plans: u64 = r.get(5)?;
            Ok(Counts {
                unique_strict_commits: r.get(0)?,
                attempted,
                selection_exposure: attempted.saturating_add(pending_plans),
                in_flight: r.get(2)?,
                unknown: r.get(3)?,
                ready_at: r.get(4)?,
                selected_in_population: false,
            })
        },
    )?)
}

/// Read and bind a frozen selection on both controller and rank paths.
pub fn selected_indices(path: &Path, corpus: &Path, contexts: &[ContextRow]) -> Result<Vec<usize>> {
    let selection: serde_json::Value = read_json(path)?;
    if selection["schema"] == "jope.native_generation_selection.v1" {
        let request = &selection["request"];
        ensure!(
            request["protocol"] == "all_eligible_training_contexts_v1"
                && request["corpus_sha256"] == file_hash(&corpus.join("index.json"))?,
            "native selection corpus/protocol changed"
        );
        let root = std::path::PathBuf::from(request["root"].as_str().context("selection root")?);
        ensure!(
            root.is_absolute() && root.canonicalize()? == root && corpus.starts_with(&root),
            "selection physical root differs"
        );
        let policy: crate::native_loop::ScientificPolicy = crate::native_genome::load(
            &root,
            &serde_json::from_value(request["scientific_policy"].clone())?,
        )?;
        ensure!(
            policy.schema == "jope.native_learning_policy.v1"
                && policy.sealed_partition == "meta_test",
            "native scientific selection policy differs"
        );
        let start = request["cut_start"]
            .as_u64()
            .context("selection cut start")?;
        let cuts = request["cuts"].as_u64().context("selection cuts")?;
        ensure!(cuts > 0, "empty selection cut budget");
        let overlay: Option<crate::native_graph::Artifact> =
            serde_json::from_value(request["preparation_overlay"].clone())?;
        let views = overlay
            .as_ref()
            .map(|a| {
                crate::preparation_overlay::views(
                    &root,
                    a,
                    request["run"].as_str().context("overlay selection run")?,
                )
            })
            .transpose()?;
        let expected: Vec<_> = contexts
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                views
                    .as_ref()
                    .is_none_or(|v| v.contains(&(c.dataset.clone(), c.cut)))
                    && c.partition == "meta_train"
                    && !policy.excluded_families.contains(&c.family)
                    && c.cut as u64 >= start
                    && (c.cut as u64) - start < cuts
            })
            .map(|(i, c)| serde_json::json!({"context_index":i,"identity":Context::from_row(c)}))
            .collect();
        ensure!(
            !expected.is_empty() && selection["contexts"] == serde_json::to_value(&expected)?,
            "native selection omitted or added eligible contexts"
        );
        return expected
            .iter()
            .map(|v| Ok(v["context_index"].as_u64().context("context index")? as usize))
            .collect();
    }

    ensure!(
        selection["schema"] == "parallel.coverage_selection.v1"
            && selection["request"]["protocol"] == PROTOCOL
            && selection["request"]["implementation"] == IMPLEMENTATION
            && selection["request"]["corpus_sha256"] == file_hash(&corpus.join("index.json"))?,
        "coverage selection corpus or protocol changed"
    );
    let mut seen = std::collections::BTreeSet::new();
    selection["contexts"]
        .as_array()
        .context("coverage selection identities absent")?
        .iter()
        .map(|entry| {
            let i = usize::try_from(
                entry["context_index"]
                    .as_u64()
                    .context("coverage index absent")?,
            )?;
            let row = contexts.get(i).context("coverage index out of range")?;
            ensure!(
                seen.insert(i)
                    && row.partition == "meta_train"
                    && entry["identity"] == serde_json::to_value(Context::from_row(row))?,
                "coverage selection identity changed"
            );
            Ok(i)
        })
        .collect()
}

fn old_is_untracked(tx: &rusqlite::Transaction<'_>, job: &GpJob) -> Result<bool> {
    Ok(!tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM coverage_requests WHERE output=?1)",
        [job.output.to_string_lossy()],
        |r| r.get::<_, bool>(0),
    )?)
}

/// Artifact locations are aliases, not new computation identities. Distinct
/// seeded/generation/parent requests may share a context; renamed copies may not.
fn request_identity(job: &GpJob, context: &Context) -> Result<String> {
    let mut binding = serde_json::to_value(job)?;
    for key in ["output", "prepared", "seeds", "parent_checkpoint"] {
        binding.as_object_mut().unwrap().remove(key);
    }
    Ok(hash(&serde_json::to_vec(&(
        IMPLEMENTATION,
        context,
        binding,
    ))?))
}

#[cfg(test)]
mod tests;
