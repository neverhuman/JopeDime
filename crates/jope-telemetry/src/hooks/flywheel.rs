//! Full-cohort flywheel reporting. Clocks, aggregation, rates and sinks stay native.
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Clone, Deserialize, Serialize)]
pub struct Context {
    pub context_sha256: String,
    pub dataset_id: String,
    pub family_id: String,
    pub dl_split: String,
    pub source_split: String,
    pub task: String,
}
#[derive(Deserialize)]
struct Packet {
    train: Vec<Context>,
    dev: Vec<Context>,
}

pub struct FullFlywheel {
    contexts: BTreeMap<String, Context>,
    train: Vec<String>,
    dev: Vec<String>,
    output: PathBuf,
    history: PathBuf,
    status: PathBuf,
    start: Instant,
    pack_start: Instant,
    pack_seconds: f64,
    bytes: u64,
    presentations: usize,
    seen: BTreeSet<String>,
    recent: VecDeque<(f64, u64)>,
    previous: Option<Value>,
    telemetry: Option<(
        crate::core::Recorder,
        crate::core::ProducerHandle,
        crate::collectors::SystemCollector,
    )>,
}

fn shuffled<T>(xs: &mut [T], seed: &mut u64) {
    for i in (1..xs.len()).rev() {
        *seed = seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        xs.swap(i, (z % (i as u64 + 1)) as usize);
    }
}

fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    let temporary = path.with_extension("json.new");
    let mut file = fs::File::create(&temporary)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    writeln!(file)?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

impl FullFlywheel {
    pub fn new(packet: &Path, output: PathBuf, history: PathBuf) -> Result<Self> {
        let packet: Packet = serde_json::from_slice(&fs::read(packet)?)?;
        ensure!(
            !packet.train.is_empty() && !packet.dev.is_empty(),
            "empty cohort"
        );
        let train_families: BTreeSet<_> = packet.train.iter().map(|r| &r.family_id).collect();
        ensure!(
            packet
                .dev
                .iter()
                .all(|r| !train_families.contains(&r.family_id)),
            "family leakage"
        );
        let mut contexts = BTreeMap::new();
        for (rows, split) in [(&packet.train, "DL-Train"), (&packet.dev, "DL-Val")] {
            for r in rows {
                ensure!(
                    r.dl_split == split && matches!(r.source_split.as_str(), "train" | "inner"),
                    "outside approved closure"
                );
                ensure!(
                    matches!(r.task.as_str(), "regression" | "binary"),
                    "deferred task"
                );
                ensure!(
                    contexts
                        .insert(r.context_sha256.clone(), r.clone())
                        .is_none(),
                    "duplicate context"
                );
            }
        }
        fs::create_dir_all(&output)?;
        if let Some(parent) = history.parent() {
            fs::create_dir_all(parent)?;
        }
        let telemetry = if let Some(root) = std::env::var_os("JOPE_TELEMETRY_RUN_DIR") {
            let path = PathBuf::from(root).join(format!("full-cohort-{}", ulid::Ulid::new()));
            let metadata = [
                ("host".into(), json!("xbabe1")),
                ("trainer".into(), json!("full-cohort-frozen-dope")),
                ("phase2_kernel_used".into(), json!(false)),
                (
                    "data_identity".into(),
                    json!(format!(
                        "{:x}",
                        <sha2::Sha256 as sha2::Digest>::digest(serde_json::to_vec(&contexts)?)
                    )),
                ),
            ]
            .into_iter()
            .collect();
            let mut builder = crate::core::RecorderBuilder::new(
                path,
                crate::manifest::RunMetadata {
                    metadata,
                    metric_groups: crate::hooks::trainers::flywheel_metric_groups(
                        crate::hooks::trainers::cpu_count(),
                    ),
                    ..Default::default()
                },
            );
            builder.add_producer("full-flywheel")?;
            builder.add_producer("host-probe")?;
            let (recorder, mut producers) = builder.start()?;
            let host = crate::collectors::SystemCollector::start(
                producers.remove(1),
                std::time::Duration::from_secs(1),
            )?;
            Some((recorder, producers.remove(0), host))
        } else {
            None
        };
        Ok(Self {
            contexts,
            train: packet
                .train
                .iter()
                .map(|r| r.context_sha256.clone())
                .collect(),
            dev: packet
                .dev
                .iter()
                .map(|r| r.context_sha256.clone())
                .collect(),
            output,
            history,
            status: Path::new(&std::env::var("HOME")?)
                .join("jope_plan/ns/FLYWHEEL_GP_FUSED_STATUS.md"),
            start: Instant::now(),
            pack_start: Instant::now(),
            pack_seconds: 0.,
            bytes: 0,
            presentations: 0,
            seen: BTreeSet::new(),
            recent: VecDeque::new(),
            previous: None,
            telemetry,
        })
    }
    /// Whole family blocks; every context appears in each complete shuffled pass.
    pub fn training_order(&self, mut seed: u64, limit: usize, gp: bool) -> Result<Vec<String>> {
        ensure!(
            gp || limit >= self.train.len(),
            "batch cannot cover full cohort"
        );
        let mut groups: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        for c in &self.train {
            groups
                .entry(&self.contexts[c].family_id)
                .or_default()
                .push(c.clone());
        }
        let mut groups: Vec<_> = groups.into_values().collect();
        let mut out = Vec::new();
        loop {
            shuffled(&mut groups, &mut seed);
            for group in &mut groups {
                shuffled(group, &mut seed);
                if gp {
                    out.push(group[0].clone());
                } else if out.len() + group.len() <= limit {
                    out.extend(group.iter().cloned());
                } else {
                    return Ok(out);
                }
                if out.len() >= limit {
                    return Ok(out);
                }
            }
            if gp {
                return Ok(out);
            }
        }
    }
    pub fn utility(
        &self,
        cids: &[String],
        fitness: &[f64],
        gates: &[bool],
        programs: &[String],
        role: &str,
    ) -> Result<Value> {
        let expected = if role == "train" {
            &self.train
        } else {
            ensure!(role == "dev", "unknown role");
            &self.dev
        };
        ensure!(
            cids == expected
                && cids.len() == fitness.len()
                && cids.len() == gates.len()
                && cids.len() == programs.len(),
            "incomplete or reordered utility"
        );
        let mut families: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut tasks: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut values = Vec::new();
        for ((c, f), gate) in cids.iter().zip(fitness).zip(gates) {
            ensure!(!*gate || f.is_finite(), "nonfinite passing fitness");
            let v = if *gate { -f } else { -2. };
            families
                .entry(self.contexts[c].family_id.clone())
                .or_default()
                .push(v);
            tasks
                .entry(self.contexts[c].task.clone())
                .or_default()
                .push(v);
            values.push(v);
        }
        let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
        let per_family: BTreeMap<_, _> = families.iter().map(|(f, xs)| (f, mean(xs))).collect();
        // The legacy scalar gate is supported only for a single-task cohort. A joint cohort
        // requires a vector admission rule before deployment; never silently blend tasks.
        ensure!(
            tasks.len() == 1,
            "joint cohort requires per-task admission gate"
        );
        Ok(
            json!({"mean": mean(&values), "per_context": values, "selected_programs": programs, "per_family": per_family,
            "val_reg": tasks.get("regression").map(|xs| mean(xs)), "val_bin": tasks.get("binary").map(|xs| mean(xs)),
            "n_reg": tasks.get("regression").map_or(0, Vec::len), "n_bin": tasks.get("binary").map_or(0, Vec::len),
            "scope": format!("{} fixed contexts; negative training-CV fitness; repeatedly selected, not independent test", if role == "dev" { "DL-Val" } else { "DL-Train" })}),
        )
    }
    pub fn begin(&mut self) {
        self.start = Instant::now();
        self.pack_seconds = 0.;
        self.bytes = 0;
        self.presentations = 0;
        self.seen.clear();
    }
    pub fn begin_pack(&mut self) {
        self.pack_start = Instant::now();
    }
    pub fn packed(&mut self, cids: &[String], payload_bytes: &[u64]) -> Result<()> {
        ensure!(
            cids.iter().all(|c| self.train.contains(c)),
            "nontrain optimizer batch"
        );
        self.seen.extend(cids.iter().cloned());
        self.presentations += cids.len();
        self.bytes += payload_bytes.iter().sum::<u64>();
        self.pack_seconds += self.pack_start.elapsed().as_secs_f64();
        Ok(())
    }
    /// Append actual raw scores under the shared append lock. No test rows accepted.
    #[allow(clippy::too_many_arguments)] // Raw score boundary mirrors the historical row identity.
    pub fn scores(
        &self,
        cids: &[String],
        values: &[f64],
        metric: &str,
        higher: bool,
        model: &str,
        model_sha: &str,
        round: u64,
    ) -> Result<()> {
        ensure!(cids.len() == values.len(), "score lengths");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.history.with_extension("jsonl.lock"))?;
        lock.lock()?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.history)?;
        for (c, value) in cids.iter().zip(values) {
            let r = self
                .contexts
                .get(c)
                .ok_or_else(|| anyhow::anyhow!("unapproved context"))?;
            ensure!(value.is_finite(), "nonfinite history score");
            let row = json!({"schema":"jope.score-history/v1", "ts":crate::alarm::utc_now_rfc3339(), "model":format!("{model}:{model_sha}"), "model_sha":model_sha,"task":r.task,"dataset":r.dataset_id,"family":r.family_id,"split":format!("{}/{}",r.dl_split,c),"metric":metric,"value":value,"higher_is_better":higher,"source":"flywheel:full-cohort/v1:train-only-two-fold-cv","round":round,"host":"xbabe1"});
            serde_json::to_writer(&mut file, &row)?;
            writeln!(file)?;
        }
        file.sync_data()?;
        Ok(())
    }
    pub fn finish(&mut self, mut row: Value) -> Result<Value> {
        ensure!(
            self.seen.len() == self.train.len(),
            "round omitted train contexts"
        );
        let wall = self.start.elapsed().as_secs_f64();
        let promoted = row["promotions"].as_u64().unwrap_or(0);
        self.recent.push_back((wall, promoted));
        while self.recent.len() > 20 {
            self.recent.pop_front();
        }
        row["wall_seconds"] = json!(wall);
        row["datasets_per_second"] = json!(self.seen.len() as f64 / wall);
        row["training_context_presentations"] = json!(self.presentations);
        row["distinct_training_contexts"] = json!(self.seen.len());
        row["training_contexts_per_second"] =
            json!(self.presentations as f64 / row["train_wall_seconds"].as_f64().unwrap_or(wall));
        row["promotions_per_hour"] = json!(
            3600. * self.recent.iter().map(|x| x.1).sum::<u64>() as f64
                / self.recent.iter().map(|x| x.0).sum::<f64>()
        );
        row["promotions_rate_scope"] =
            json!("up to 20 completed rounds since process start; includes qualification wait");
        row["batch_gb_s"] = if self.pack_seconds > 0. {
            json!(self.bytes as f64 / 1e9 / self.pack_seconds)
        } else {
            Value::Null
        };
        row["batch_bytes"] = json!(self.bytes);
        row["batch_gb_s_scope"] = json!("actual cold pool buffers plus indices / completed pack delivery wall; not PCIe-only bandwidth");
        row["gp_dsl_tokens_per_second"] =
            json!(row["tokens_count"].as_u64().unwrap_or(0) as f64 / wall);
        for key in [
            "rust_tokens_per_second",
            "all_languages_tokens_per_second",
            "reasoning_tokens_per_second",
        ] {
            row[key] = json!("N/A (fused path unused)");
        }
        row["phase2_kernel_used"] = json!(false);
        row["regression_elo"] = json!(
            "N/A: Golden45/panel fitting unavailable within this round's 64 native-CV budget"
        );
        row["binary_elo"] = json!("N/A: no binary contexts in approved 1219/91 corpus");
        row["elo_every_n_rounds"] = json!(10);
        row["percentiles"] = json!("percentiles pending JRB-02: no historical other-model scores under these exact DL-Val context/split/metric identities");
        let candidate = row["candidate_dev_utility"].clone();
        row["val_reg"] = candidate["val_reg"].clone();
        row["val_bin"] = candidate["val_bin"].clone();
        row["n_reg"] = candidate["n_reg"].clone();
        row["n_bin"] = candidate["n_bin"].clone();
        for task in ["reg", "bin"] {
            let key = format!("val_{task}");
            row[format!("train_vs_val_gap_{task}")] = match (
                row["candidate_train_utility"][&key].as_f64(),
                candidate[&key].as_f64(),
            ) {
                (Some(t), Some(v)) => json!(t - v),
                _ => Value::Null,
            };
            row[format!("val_{task}_delta")] = match (
                self.previous.as_ref().and_then(|r| r[&key].as_f64()),
                candidate[&key].as_f64(),
            ) {
                (Some(p), Some(v)) => json!(v - p),
                _ => Value::Null,
            };
        }
        row["val_per_family"] = candidate["per_family"].clone();
        let rnd = row["round"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("missing round"))?;
        if let Some((recorder, producer, _host)) = &mut self.telemetry {
            for (metric, key) in [
                (crate::train::IO_BATCH_GB_S, "batch_gb_s"),
                (crate::train::SAMPLES_PER_S, "datasets_per_second"),
            ] {
                if let Some(value) = row[key].as_f64() {
                    producer.emit_scalar(metric, 0, rnd, value, false);
                }
            }
            for name in [
                "throughput.tokens_per_s.rust",
                "throughput.tokens_per_s.all_languages",
                "throughput.reasoning_tok_per_s",
            ] {
                if let Some(metric) = crate::train::metric_by_name(name) {
                    producer.emit_unavailable(metric.id, 0, rnd, 10, false);
                }
            }
            let compact: serde_json::Map<String, Value> = [
                "round",
                "dope_checksum_match",
                "candidate_parameter_sha256",
                "val_reg",
                "val_bin",
                "n_reg",
                "n_bin",
                "val_per_family",
                "train_vs_val_gap_reg",
                "train_vs_val_gap_bin",
                "regression_elo",
                "binary_elo",
                "percentiles",
                "gp_dsl_tokens_per_second",
                "promotions_per_hour",
            ]
            .into_iter()
            .map(|key| (key.to_owned(), row[key].clone()))
            .collect();
            ensure!(
                crate::record::record(recorder, "jope.flywheel.full-cohort/v1", &compact),
                "native round record dropped"
            );
        }
        atomic_json(&self.output.join(format!("round-{rnd:02}.json")), &row)?;
        let mut summary = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.output.join("summary.jsonl"))?;
        serde_json::to_writer(&mut summary, &row)?;
        writeln!(summary)?;
        summary.sync_data()?;

        fs::write(&self.status, format!("# Fused flywheel status\n\nRound {rnd}; receipts {}.\nDOPE {} match {}; trainable {}.\nTrain {} / validation {} (regression {}; binary {}).\nRegression validation {}; binary validation {}.\nRegression Elo {}; binary Elo {}.\nPercentiles: {}.\nCandidate SHA {}. Adapter action {}.\nDistinct contexts/s {}; promotions/hour {}; input GB/s {}.\nRust/all-language/reasoning tok/s: N/A (fused path unused).\n",self.output.display(),row["dope_sha256"],row["dope_checksum_match"],row["trainable_params"],self.train.len(),self.dev.len(),row["n_reg"],row["n_bin"],row["val_reg"],row["val_bin"],row["regression_elo"],row["binary_elo"],row["percentiles"],row["candidate_parameter_sha256"],row["adapter_admission"]["action"],row["datasets_per_second"],row["promotions_per_hour"],row["batch_gb_s"]))?;
        self.previous = Some(row.clone());
        Ok(row)
    }
}

impl Drop for FullFlywheel {
    fn drop(&mut self) {
        if let Some((recorder, producer, mut host)) = self.telemetry.take() {
            host.stop();
            drop(producer);
            // A continuous worker has no natural completed run boundary.
            let _ = recorder.interrupt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn packet(dir: &Path) -> PathBuf {
        let path = dir.join("selected.json");
        fs::write(&path, json!({"train":[{"context_sha256":"a","dataset_id":"a","family_id":"f1","source_split":"train","dl_split":"DL-Train","task":"regression"},{"context_sha256":"b","dataset_id":"b","family_id":"f1","source_split":"inner","dl_split":"DL-Train","task":"regression"},{"context_sha256":"c","dataset_id":"c","family_id":"f2","source_split":"train","dl_split":"DL-Train","task":"regression"}],"dev":[{"context_sha256":"d","dataset_id":"d","family_id":"f3","source_split":"train","dl_split":"DL-Val","task":"regression"}]}).to_string()).unwrap();
        path
    }
    #[test]
    fn shuffle_covers_every_context_and_preserves_family_blocks() {
        let d = tempfile::tempdir().unwrap();
        let p = packet(d.path());
        let r =
            FullFlywheel::new(&p, d.path().join("out"), d.path().join("history.jsonl")).unwrap();
        let a = r.training_order(7, 3, false).unwrap();
        assert_eq!(a.iter().collect::<BTreeSet<_>>().len(), 3);
        assert_eq!(
            (a.iter().position(|x| x == "a").unwrap() as i32
                - a.iter().position(|x| x == "b").unwrap() as i32)
                .abs(),
            1
        );
        assert!(r.training_order(7, 2, false).is_err());
        let g = r.training_order(1, 16, true).unwrap();
        assert_eq!(g.len(), 2);
    }
    #[test]
    fn rejects_holdout_multiclass_and_family_overlap() {
        for (from, to) in [
            ("DL-Val", "DL-Holdout"),
            ("regression", "multiclass"),
            ("f3", "f1"),
        ] {
            let d = tempfile::tempdir().unwrap();
            let p = packet(d.path());
            let text = fs::read_to_string(&p).unwrap().replace(from, to);
            fs::write(&p, text).unwrap();
            assert!(
                FullFlywheel::new(&p, d.path().join("out"), d.path().join("history.jsonl"))
                    .is_err()
            );
        }
    }
    #[test]
    fn utility_has_task_counts_and_history_uses_exact_context_split() {
        let d = tempfile::tempdir().unwrap();
        let p = packet(d.path());
        let history = d.path().join("history.jsonl");
        let r = FullFlywheel::new(&p, d.path().join("out"), history.clone()).unwrap();
        let u = r
            .utility(&["d".into()], &[-1.5], &[true], &["p".into()], "dev")
            .unwrap();
        assert_eq!(u["val_reg"], 1.5);
        assert!(u["val_bin"].is_null());
        assert_eq!(u["n_bin"], 0);
        r.scores(
            &["d".into()],
            &[-1.5],
            "cv_fitness",
            false,
            "candidate",
            "sha",
            1,
        )
        .unwrap();
        let row: Value = serde_json::from_str(fs::read_to_string(history).unwrap().trim()).unwrap();
        assert_eq!(row["split"], "DL-Val/d");
        assert_eq!(row["task"], "regression");
        assert_eq!(row["model"], "candidate:sha");
    }
    #[test]
    fn round_report_requires_coverage_and_emits_native_rates_and_na() {
        let d = tempfile::tempdir().unwrap();
        let p = packet(d.path());
        let mut r =
            FullFlywheel::new(&p, d.path().join("out"), d.path().join("history.jsonl")).unwrap();
        r.status = d.path().join("STATUS.md");
        r.begin();
        assert!(r.finish(json!({"round":1})).is_err());
        r.begin_pack();
        r.packed(&["a".into(), "b".into(), "c".into()], &[1000, 800])
            .unwrap();
        let row = r.finish(json!({"round":1,"promotions":2,"tokens_count":100,"train_wall_seconds":1.,"candidate_dev_utility":{"val_reg":1.5,"val_bin":null,"n_reg":1,"n_bin":0,"per_family":{"f3":1.5}},"candidate_train_utility":{"val_reg":2.,"val_bin":null}})).unwrap();
        assert_eq!(row["distinct_training_contexts"], 3);
        assert_eq!(row["batch_bytes"], 1800);
        assert_eq!(row["train_vs_val_gap_reg"], 0.5);
        assert!(row["train_vs_val_gap_bin"].is_null());
        assert_eq!(row["rust_tokens_per_second"], "N/A (fused path unused)");
        assert!(row["promotions_per_hour"].as_f64().unwrap() > 0.);
        assert_eq!(
            fs::read_to_string(d.path().join("out/summary.jsonl"))
                .unwrap()
                .lines()
                .count(),
            1
        );
    }
    #[test]
    fn scientific_receipt_floats_round_trip_bit_exactly() {
        for original in [
            1.1073772627425547_f64,
            1.2591052389885997_f64,
            0.9088130596333871_f64,
        ] {
            let text = serde_json::to_string(&original).unwrap();
            let parsed: Value = serde_json::from_str(&text).unwrap();
            assert_eq!(
                parsed.as_f64().unwrap().to_bits(),
                original.to_bits(),
                "{text}"
            );
        }
    }
}
