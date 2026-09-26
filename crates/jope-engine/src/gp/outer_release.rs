//! Outer score release authority (W1-T04).
//!
//! Sealed final-test labels have exactly one permitted consumer: an outer
//! scorer holding a grant issued here. Upstream producers — audit,
//! preparation, views, search, export — never receive them. The authority is
//! separate from the scorer in two ways:
//!
//! * **Scoring capability.** [`Authority::grant`] issues a grant only for an
//!   authenticated fitted subject: the predictor exported from a sealed,
//!   independently replayed selection ([`super::handoff`]) whose final refit is
//!   verified against its fold fit ([`super::lineage::verify_final_refit`]).
//!   The grant binds that subject, its preparation and the preparation's own
//!   sealed-test identity (`Prepared.original_test_sha256`) with its
//!   root-relative path. One grant exists per subject and label set, and a
//!   grant admits one scoring attempt: the attempt is recorded as an exposure
//!   *before* the first label byte is read, so a retry, crash or resume can
//!   never obtain an unrecorded second look.
//! * **Result visibility.** [`Authority::score`] writes the computed score
//!   only into the authority's withheld store and hands the caller a receipt
//!   without metrics. [`Authority::release`] is the only path that makes the
//!   score visible, once.
//!
//! Every grant, refusal, exposure, score and release is appended to a
//! hash-chained ledger (`ledger.jsonl`). A refused request is recorded with
//! whether any label byte had been read. Released results carry aggregate
//! metrics only, never per-row targets, and repeat the preparation's
//! evaluation limitations.
//!
//! Fixture scope: the separation is a component contract on one host, not OS
//! isolation of the scorer; managed-worker isolation is W7.
use super::{handoff, lineage, metrics, Predictor};
use crate::{
    manifest::{
        after_verification, atomic_write, hash, read_bound, read_json, resolve, write_json,
    },
    prepare::{Prepared, Table},
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

pub const AUTHORITY_SCHEMA: &str = "parallel.outer_release_authority.v1";
pub const GRANT_SCHEMA: &str = "parallel.outer_release_grant.v1";
pub const EVENT_SCHEMA: &str = "parallel.outer_release_event.v1";
pub const RESULT_SCHEMA: &str = "parallel.outer_score.v1";
pub const RECEIPT_SCHEMA: &str = "parallel.outer_score_receipt.v1";
/// Original label role of the scored labels (the release contract's term).
pub const LABEL_ROLE: &str = "sealed_final";
/// The CLI `score` command finds its release authority here; nothing else does.
pub const AUTHORITY_ENV: &str = "JOPE_OUTER_RELEASE_AUTHORITY";
const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A path under `root` as its root-relative form, refusing any symlink
/// component (including the leaf) and anything outside the root.
fn rooted(root: &Path, path: &Path) -> Result<(PathBuf, PathBuf)> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let relative = absolute
        .strip_prefix(root)
        .context("sealed label file outside the granted root")?
        .to_path_buf();
    Ok((resolve(root, &relative)?, relative))
}

/// Read `path` once and record it as `relative_artifact` does (its path under
/// the canonical `root` and the digest of exactly the bytes returned).
fn read_artifact(root: &Path, path: &Path) -> Result<(Value, Vec<u8>)> {
    let absolute = path.canonicalize()?;
    let relative = absolute
        .strip_prefix(root)
        .context("original fit artifact outside workspace")?
        .to_path_buf();
    let bytes = fs::read(&absolute)?;
    Ok((json!({"path":relative,"sha256":hash(&bytes)}), bytes))
}

/// Who the grant is for, and which completed search sealed its selection.
pub struct GrantRequest<'a> {
    pub search: handoff::Inputs<'a>,
    /// The exported predictor (`*.handoff.json` must sit beside it).
    pub predictor: &'a Path,
    /// The sealed final-test file, under `search.root`.
    pub test: &'a Path,
}

/// The hash-chained event ledger, read and verified under the authority lock.
struct Ledger {
    path: PathBuf,
    instance: String,
    events: Vec<Value>,
    tip: String,
}

impl Ledger {
    fn read(path: PathBuf, instance: &str) -> Result<Self> {
        let bytes = fs::read(&path).context("release ledger unavailable")?;
        ensure!(
            bytes.is_empty() || bytes.ends_with(b"\n"),
            "release ledger has a torn tail"
        );
        let mut events = Vec::new();
        let mut tip = GENESIS.to_string();
        for line in bytes.split_inclusive(|b| *b == b'\n') {
            let event: Value = serde_json::from_slice(line).context("release ledger event")?;
            ensure!(
                event["schema"] == EVENT_SCHEMA
                    && event["sequence"] == events.len()
                    && event["previous_sha256"] == tip.as_str()
                    && event["authority_instance"] == instance,
                "release ledger chain broken at event {}",
                events.len()
            );
            tip = hash(line);
            events.push(event);
        }
        Ok(Self {
            path,
            instance: instance.into(),
            events,
            tip,
        })
    }

    fn append(&mut self, mut event: Value) -> Result<Value> {
        event["schema"] = json!(EVENT_SCHEMA);
        event["sequence"] = json!(self.events.len());
        event["previous_sha256"] = json!(self.tip);
        event["authority_instance"] = json!(self.instance);
        event["recorded_at"] = json!(crate::telemetry::now());
        let mut line = serde_json::to_vec(&event)?;
        line.push(b'\n');
        let mut file = fs::OpenOptions::new().append(true).open(&self.path)?;
        file.write_all(&line)?;
        file.sync_data()?;
        self.tip = hash(&line);
        self.events.push(event.clone());
        Ok(event)
    }

    fn find(&self, kind: &str, grant: &Value) -> Option<&Value> {
        self.events
            .iter()
            .find(|e| e["event"] == kind && e["grant_id"] == *grant)
    }
}

#[derive(Debug)]
pub struct Authority {
    dir: PathBuf,
    instance: String,
}

impl Authority {
    /// A new, empty authority with its own instance identity.
    pub fn initialize(dir: &Path) -> Result<Self> {
        fs::create_dir(dir).context("outer release authority already exists or has no parent")?;
        let dir = dir.canonicalize()?;
        for sub in ["grants", "withheld"] {
            fs::create_dir(dir.join(sub))?;
        }
        let instance = hash(
            format!(
                "{}:{}:{:?}",
                dir.display(),
                std::process::id(),
                std::time::SystemTime::now()
            )
            .as_bytes(),
        );
        fs::write(dir.join("ledger.jsonl"), b"")?;
        write_json(
            &dir.join("authority.json"),
            &json!({"schema":AUTHORITY_SCHEMA,"instance":instance,
                "permitted_consumer":"outer_scorer","label_role":LABEL_ROLE}),
        )?;
        Self::open(&dir)
    }

    pub fn open(dir: &Path) -> Result<Self> {
        let dir = dir
            .canonicalize()
            .context("outer release authority unavailable")?;
        let identity: Value = read_json(&dir.join("authority.json"))?;
        ensure!(
            identity["schema"] == AUTHORITY_SCHEMA,
            "not an outer release authority"
        );
        let instance = identity["instance"]
            .as_str()
            .filter(|i| is_sha256(i))
            .context("outer release authority instance malformed")?
            .to_string();
        let authority = Self { dir, instance };
        authority.ledger()?;
        Ok(authority)
    }

    pub fn instance(&self) -> &str {
        &self.instance
    }

    fn ledger(&self) -> Result<Ledger> {
        Ledger::read(self.dir.join("ledger.jsonl"), &self.instance)
    }

    /// Serialize every mutation of this authority.
    fn lock(&self) -> Result<fs::File> {
        use std::os::fd::AsRawFd;
        let lock = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join("ledger.lock"))?;
        ensure!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } == 0,
            "outer release authority lock failed"
        );
        Ok(lock)
    }

    /// Every recorded event, verified.
    pub fn events(&self) -> Result<Vec<Value>> {
        Ok(self.ledger()?.events)
    }

    /// Issue the one scoring grant for an authenticated fitted subject. The
    /// sealed test file is named, never opened.
    pub fn grant(&self, request: &GrantRequest) -> Result<PathBuf> {
        let _lock = self.lock()?;
        let mut ledger = self.ledger()?;
        let issued = self.grant_body(&ledger, request);
        let body = match issued {
            Ok(body) => body,
            Err(error) => {
                ledger.append(json!({"event":"grant_refused","grant_id":null,
                    "predictor":request.predictor,"labels_opened":false,
                    "reason":format!("{error:#}")}))?;
                return Err(error);
            }
        };
        let grant_id = hash(&serde_json::to_vec(&body)?);
        let mut grant = body;
        grant["grant_id"] = json!(grant_id);
        let path = self.dir.join("grants").join(format!("{grant_id}.json"));
        // Route R8: the ledger binds exactly the bytes written, never a reread.
        let bytes = serde_json::to_vec_pretty(&grant)?;
        atomic_write(&path, &bytes)?;
        after_verification(&path);
        ledger.append(json!({"event":"granted","grant_id":grant_id,
            "grant_sha256":hash(&bytes),
            "subject_sha256":grant["subject"]["predictor"]["sha256"],
            "labels_sha256":grant["scoring_labels"]["sha256"]}))?;
        Ok(path)
    }

    fn grant_body(&self, ledger: &Ledger, request: &GrantRequest) -> Result<Value> {
        let root = request.search.root.canonicalize()?;
        let predictor = request.predictor;
        let handoff_path = predictor.with_extension("handoff.json");
        // Route R8 (review of #111): every artifact the grant binds is read
        // once; its checks, its recorded digest and the grant's fields all
        // come from those bytes, never from a reopened path.
        let (handoff_artifact, recorded) = read_artifact(&root, &handoff_path)
            .and_then(|(artifact, bytes)| Ok((artifact, serde_json::from_slice::<Value>(&bytes)?)))
            .context("an outer score needs an exported handoff")?;
        after_verification(&handoff_path);
        ensure!(
            recorded["schema"] == handoff::HANDOFF_SCHEMA && recorded["status"] == "exported",
            "an outer score needs an exported sealed selection"
        );
        let (subject, predictor_bytes) = read_artifact(&root, predictor)?;
        ensure!(
            recorded["predictor"] == subject,
            "the predictor is not the one its handoff exported"
        );
        after_verification(predictor);
        let seal_path = resolve(
            &root,
            Path::new(recorded["seal"]["path"].as_str().context("handoff seal")?),
        )?;
        let (seal_artifact, seal_bytes) = read_artifact(&root, &seal_path)?;
        ensure!(
            seal_artifact == recorded["seal"],
            "the seal changed after its handoff"
        );
        after_verification(&seal_path);
        // The fitted subject is authenticated by replaying its selection and
        // re-verifying its final refit, never by trusting a receipt field.
        let (seal, verified) = handoff::verify_seal_bytes(&request.search, &seal_bytes)?;
        let refit =
            lineage::verify_final_refit_bytes(&root, &verified, predictor, &predictor_bytes)?;
        let scope = &seal["final_fit_scope"];
        let prepared_dir = resolve(
            &root,
            Path::new(scope["prepared"].as_str().context("sealed preparation")?),
        )?;
        let p: Prepared = serde_json::from_slice(&read_bound(
            &prepared_dir.join("prepared.json"),
            &scope["prepared_sha256"],
            "the preparation changed after its sealed selection",
        )?)?;
        ensure!(
            p.clean_evaluation,
            "an outer score needs a clean evaluation preparation"
        );
        ensure!(
            is_sha256(&p.original_test_sha256),
            "the preparation binds no sealed final-test identity"
        );
        let (_, test_relative) = rooted(&root, request.test)?;
        ensure!(
            !ledger.events.iter().any(|e| e["event"] == "granted"
                && e["subject_sha256"] == subject["sha256"]
                && e["labels_sha256"] == p.original_test_sha256.as_str()),
            "an outer score for this fitted subject and these sealed labels was already granted"
        );
        Ok(
            json!({"schema":GRANT_SCHEMA,"authority_instance":self.instance,
            "sequence":ledger.events.len(),"root":root,
            "subject":{"predictor":subject,"handoff":handoff_artifact,
                "seal":recorded["seal"],"program_sha256":seal["selected"]["program_sha256"],
                "training_sha256":scope["training_sha256"],"fit_key":refit.fit_key,
                "model_sha256":recorded["final_refit"]["model_sha256"]},
            "preparation":{"prepared":scope["prepared"],"prepared_sha256":scope["prepared_sha256"],
                "dataset":p.dataset,"family":p.family,"partition":p.meta_partition,"cut":scope["cut"],
                "clean_status":p.clean_status,"evaluation_limitations":p.evaluation_limitations},
            "scoring_labels":{"role":LABEL_ROLE,"path":test_relative,
                "sha256":p.original_test_sha256},
            "permitted_consumer":"outer_scorer","upstream_label_access":"denied",
            "attempts":1,"release":"withheld_until_authority_release"}),
        )
    }

    /// Score the granted subject on exactly the granted sealed labels. The
    /// score is withheld; `receipt` receives the outcome without metrics.
    pub fn score(&self, grant: &Path, model: &Path, test: &Path, receipt: &Path) -> Result<Value> {
        let _lock = self.lock()?;
        let mut ledger = self.ledger()?;
        let mut grant_id = Value::Null;
        let mut labels_opened = false;
        let outcome = self.score_granted(
            &mut ledger,
            grant,
            model,
            test,
            &mut grant_id,
            &mut labels_opened,
        );
        match outcome {
            Ok(scored) => {
                let event = ledger.append(json!({"event":"scored","grant_id":grant_id,
                    "result_sha256":scored["result_sha256"],"rows":scored["rows"]}))?;
                let out = json!({"schema":RECEIPT_SCHEMA,"status":"scored_withheld",
                    "grant_id":grant_id,"result_sha256":scored["result_sha256"],
                    "released":false,"ledger_sequence":event["sequence"],
                    "access":{"model_sha256":scored["model_sha256"],"labels_role":LABEL_ROLE,
                        "labels_sha256":scored["labels_sha256"],"rows":scored["rows"],
                        "labels_opened":true}});
                write_json(receipt, &out)?;
                Ok(out)
            }
            Err(error) => {
                let event = ledger.append(json!({"event":"score_refused","grant_id":grant_id,
                    "labels_opened":labels_opened,"reason":format!("{error:#}")}))?;
                write_json(
                    receipt,
                    &json!({"schema":RECEIPT_SCHEMA,"status":"refused","grant_id":grant_id,
                        "labels_opened":labels_opened,"reason":format!("{error:#}"),
                        "ledger_sequence":event["sequence"],"released":false}),
                )?;
                Err(error)
            }
        }
    }

    fn score_granted(
        &self,
        ledger: &mut Ledger,
        grant_path: &Path,
        model: &Path,
        test: &Path,
        grant_id: &mut Value,
        labels_opened: &mut bool,
    ) -> Result<Value> {
        let bytes = fs::read(grant_path).context("outer release grant unavailable")?;
        let grant: Value = serde_json::from_slice(&bytes).context("outer release grant")?;
        ensure!(
            grant["schema"] == GRANT_SCHEMA,
            "not an outer release grant"
        );
        *grant_id = grant["grant_id"].clone();
        ensure!(
            grant["authority_instance"] == self.instance.as_str()
                && ledger
                    .find("granted", grant_id)
                    .is_some_and(|e| e["grant_sha256"] == hash(&bytes)),
            "the grant is not recorded by this release authority"
        );
        ensure!(
            ledger.find("score_started", grant_id).is_none(),
            "the grant is consumed: its one scoring attempt is already recorded"
        );
        let model_bytes = fs::read(model).context("granted predictor unavailable")?;
        ensure!(
            hash(&model_bytes) == grant["subject"]["predictor"]["sha256"],
            "the model is not the granted fitted subject"
        );
        let p: Predictor = serde_json::from_slice(&model_bytes)?;
        let root = PathBuf::from(grant["root"].as_str().context("grant root")?);
        let granted = resolve(
            &root,
            Path::new(
                grant["scoring_labels"]["path"]
                    .as_str()
                    .context("granted labels")?,
            ),
        )?;
        let (supplied, _) = rooted(&root, test)?;
        ensure!(
            supplied == granted,
            "the sealed label file is not the granted one"
        );
        // The exposure is recorded before the first label byte is read: from
        // here on the grant is consumed whatever happens next.
        ledger.append(json!({"event":"score_started","grant_id":grant_id,
            "model_sha256":hash(&model_bytes),"labels_path":grant["scoring_labels"]["path"]}))?;
        *labels_opened = true;
        let labels = fs::read(&granted).context("granted sealed labels unavailable")?;
        let labels_sha256 = hash(&labels);
        ensure!(
            labels_sha256 == grant["scoring_labels"]["sha256"],
            "sealed label bytes differ from the granted identity; not parsed"
        );
        let t = Table::parse(&labels, &Value::Null)?;
        ensure!(
            p.schema == "parallel.predictor.v1"
                && p.task == "regression"
                && t.width == p.feature_count + 1,
            "predictor/test schema mismatch"
        );
        let x = t
            .rows
            .iter()
            .map(|r| p.preprocessing.features(&r[..r.len() - 1]))
            .collect::<Result<Vec<_>>>()?;
        let y = t
            .rows
            .iter()
            .map(|r| {
                r.last()
                    .unwrap()
                    .parse::<f64>()
                    .context("invalid test target")
            })
            .collect::<Result<Vec<_>>>()?;
        let predictions = p
            .model
            .predict(&x)
            .iter()
            .map(|v| v * p.preprocessing.target_scale + p.preprocessing.target_mean)
            .collect::<Vec<_>>();
        let result = json!({"schema":RESULT_SCHEMA,"grant_id":grant_id,
            "subject":grant["subject"],"preparation":grant["preparation"],
            "scoring_labels":{"role":LABEL_ROLE,"sha256":labels_sha256,"rows":t.rows.len()},
            "metrics":metrics(&y, &predictions)?,"per_row_values_included":false});
        let id = grant_id.as_str().context("grant id")?;
        let withheld = self.dir.join("withheld").join(format!("{id}.json"));
        // Route R8: the ledger binds exactly the bytes written, never a reread.
        let bytes = serde_json::to_vec_pretty(&result)?;
        atomic_write(&withheld, &bytes)?;
        after_verification(&withheld);
        Ok(json!({"result_sha256":hash(&bytes),"rows":t.rows.len(),
            "model_sha256":hash(&model_bytes),"labels_sha256":labels_sha256}))
    }

    /// Make one withheld score visible at `output`, exactly once.
    pub fn release(&self, grant_id: &str, output: &Path) -> Result<Value> {
        let _lock = self.lock()?;
        let mut ledger = self.ledger()?;
        let id = json!(grant_id);
        let outcome = (|| -> Result<(Vec<u8>, Value)> {
            let scored = ledger
                .find("scored", &id)
                .context("no withheld score exists for this grant")?
                .clone();
            ensure!(
                ledger.find("released", &id).is_none(),
                "this score was already released"
            );
            ensure!(
                !output.exists(),
                "release destination exists; a release never overwrites"
            );
            let bytes = fs::read(self.dir.join("withheld").join(format!("{grant_id}.json")))?;
            ensure!(
                hash(&bytes) == scored["result_sha256"],
                "withheld score changed after scoring"
            );
            Ok((bytes, scored))
        })();
        match outcome {
            Ok((bytes, scored)) => {
                atomic_write(output, &bytes)?;
                ledger.append(json!({"event":"released","grant_id":grant_id,
                    "result_sha256":scored["result_sha256"],"output":output}))
            }
            Err(error) => {
                ledger.append(json!({"event":"release_refused","grant_id":grant_id,
                    "reason":format!("{error:#}")}))?;
                Err(error)
            }
        }
    }
}
