//! Immutable teaching corpus. Scores remain explicitly adaptive to each cut.
use crate::{
    encoding::Encoding,
    grammar::{State, Vocab},
    manifest::*,
    prepare::Prepared,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

pub const SEQUENCE: usize = 128;
pub const RECORD_BYTES: usize = 532;

/// A vocabulary is frozen for the whole build, so exact-genome tokenization
/// and its rejection reason can be reused across cuts. Limit both entry count
/// and retained payload; cache exhaustion affects speed only, never labels.
#[derive(Default)]
struct TokenCache {
    entries: HashMap<String, std::result::Result<Vec<usize>, String>>,
    payload_bytes: usize,
}
impl TokenCache {
    fn encode(&mut self, vocab: &Vocab, genome: &str) -> std::result::Result<Vec<usize>, String> {
        if let Some(result) = self.entries.get(genome) {
            return result.clone();
        }
        let result = match vocab.encode(genome) {
            Ok(tokens) if tokens.len() <= SEQUENCE => Ok(tokens),
            Ok(_) => Err("sequence_too_long".into()),
            Err(e) => Err(format!(
                "unrepresentable:{}",
                format!("{e:#}").split(':').next().unwrap_or("unknown")
            )),
        };
        let bytes = genome.len()
            + match &result {
                Ok(tokens) => tokens.len() * std::mem::size_of::<usize>(),
                Err(reason) => reason.len(),
            };
        if self.entries.len() < 65_536 && self.payload_bytes + bytes <= 32 * 1024 * 1024 {
            self.payload_bytes += bytes;
            self.entries.insert(genome.to_owned(), result.clone());
        }
        result
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn cache_reuses_exact_tokens_and_preserves_unsnapped_rejections() {
        let vocab: Vocab =
            read_json(&Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"))
                .unwrap();
        let accepted = "(pipe X (standardize) (ridge))";
        let rejected = "(pipe X (rff_cos n=17 gamma=0.513 seed=7) (ridge))";
        let expected = vocab.encode(accepted).unwrap();
        let mut cache = TokenCache::default();
        assert_eq!(cache.encode(&vocab, accepted).unwrap(), expected);
        let mut returned = cache.encode(&vocab, accepted).unwrap();
        returned[0] = usize::MAX;
        assert_eq!(cache.encode(&vocab, accepted).unwrap(), expected);
        let reason = cache.encode(&vocab, rejected).unwrap_err();
        assert_eq!(reason, "unrepresentable:exact parameter has no frozen token; scores cannot be attached to a snapped program");
        assert_eq!(cache.encode(&vocab, rejected).unwrap_err(), reason);
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn exhausted_cache_preserves_results_without_retaining_more_payload() {
        let vocab: Vocab =
            read_json(&Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"))
                .unwrap();
        let genome = "(pipe X (standardize) (ridge))";
        let mut cache = TokenCache {
            payload_bytes: 32 * 1024 * 1024,
            ..Default::default()
        };
        assert_eq!(
            cache.encode(&vocab, genome).unwrap(),
            vocab.encode(genome).unwrap()
        );
        assert!(cache.entries.is_empty());
        assert_eq!(cache.payload_bytes, 32 * 1024 * 1024);
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct ContextRow {
    pub dataset: String,
    pub family: String,
    pub cut: usize,
    pub partition: String,
    pub path: PathBuf,
    pub sha256: String,
    pub prepared: PathBuf,
    pub prepared_sha256: String,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Group {
    pub context: usize,
    pub family: String,
    pub partition: String,
    pub records: Vec<usize>,
    pub max_length: usize,
}
#[derive(Serialize, Deserialize)]
pub struct Index {
    pub schema: String,
    pub qualification: bool,
    pub vocabulary_sha256: String,
    pub context_index_sha256: String,
    pub records_sha256: String,
    pub record_bytes: usize,
    pub records: usize,
    pub legal_sets: Vec<Vec<usize>>,
    pub contexts: Vec<ContextRow>,
    pub groups: Vec<Group>,
}
#[derive(Clone, Debug)]
pub struct Record {
    pub tokens: [i64; SEQUENCE],
    pub legal: [i64; SEQUENCE],
    pub length: usize,
    pub context: usize,
    pub value: f32,
    pub percentile: f32,
    pub runtime: f32,
}
impl Record {
    pub fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(RECORD_BYTES);
        for x in self.tokens.iter().chain(self.legal.iter()) {
            out.extend((*x as u16).to_le_bytes());
        }
        out.extend((self.length as u16).to_le_bytes());
        out.extend(0u16.to_le_bytes());
        out.extend((self.context as u32).to_le_bytes());
        for x in [self.value, self.percentile, self.runtime] {
            out.extend(x.to_le_bytes());
        }
        out
    }
    pub fn parse(b: &[u8]) -> Self {
        assert_eq!(b.len(), RECORD_BYTES);
        let mut r = Self {
            tokens: [0; SEQUENCE],
            legal: [0; SEQUENCE],
            length: u16::from_le_bytes(b[512..514].try_into().unwrap()) as usize,
            context: u32::from_le_bytes(b[516..520].try_into().unwrap()) as usize,
            value: f32::from_le_bytes(b[520..524].try_into().unwrap()),
            percentile: f32::from_le_bytes(b[524..528].try_into().unwrap()),
            runtime: f32::from_le_bytes(b[528..532].try_into().unwrap()),
        };
        for (i, x) in r.tokens.iter_mut().chain(r.legal.iter_mut()).enumerate() {
            *x = u16::from_le_bytes(b[i * 2..i * 2 + 2].try_into().unwrap()) as i64;
        }
        r
    }
}
/// Read-only memory mapping of the immutable training artifact.
pub struct Records {
    ptr: *mut libc::c_void,
    len: usize,
    _file: fs::File,
}
impl Records {
    pub fn open(path: &Path, count: usize) -> Result<Self> {
        use std::os::fd::AsRawFd;
        let file = fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        ensure!(
            count > 0 && len == count * RECORD_BYTES,
            "record artifact dimensions changed"
        );
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        ensure!(ptr != libc::MAP_FAILED, "mmap failed");
        Ok(Self {
            ptr,
            len,
            _file: file,
        })
    }
    pub fn get(&self, i: usize) -> Record {
        assert!((i + 1) * RECORD_BYTES <= self.len);
        Record::parse(unsafe {
            std::slice::from_raw_parts((self.ptr as *const u8).add(i * RECORD_BYTES), RECORD_BYTES)
        })
    }
}
impl Drop for Records {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr, self.len);
        }
    }
}

pub fn build(
    root: &Path,
    contexts_path: &Path,
    vocab_path: &Path,
    observations: &[PathBuf],
    output: &Path,
    qualification: bool,
) -> Result<()> {
    build_observed(root, contexts_path, vocab_path, observations, output, qualification, |_, _, _| Ok(()))
}

/// Observation affects telemetry only; corpus bytes and admission rules are shared.
pub fn build_observed(
    root: &Path,
    contexts_path: &Path,
    vocab_path: &Path,
    observations: &[PathBuf],
    output: &Path,
    qualification: bool,
    mut progress: impl FnMut(&str, usize, Option<usize>) -> Result<()>,
) -> Result<()> {
    crate::campaign::check_deadline()?;
    ensure!(!output.exists(), "corpus version exists");
    let ci: Value = read_json(contexts_path)?;
    ensure!(
        ci["schema"] == "parallel.context_index.v1"
            && (qualification || ci["qualification"] == false),
        "context qualification mismatch"
    );
    let contexts: Vec<ContextRow> = serde_json::from_value(ci["encodings"].clone())?;
    let vocab: Vocab = read_json(vocab_path)?;
    ensure!(
        vocab.tokens.len() < u16::MAX as usize,
        "vocabulary too large"
    );
    let mut by_cut = BTreeMap::new();
    let mut prepared = BTreeMap::new();
    for (i, c) in contexts.iter().enumerate() {
        crate::campaign::check_deadline()?;
        ensure!(
            file_hash(&resolve(root, &c.path)?)? == c.sha256,
            "context changed"
        );
        let e: Encoding = read_json(&resolve(root, &c.path)?.with_extension("json"))?;
        ensure!(
            e.encoder_spec_sha256 == ci["assembly_sha256"]
                && e.prepared_sha256 == c.prepared_sha256,
            "context identity mismatch"
        );
        if !prepared.contains_key(&c.prepared) {
            let p = resolve(root, &c.prepared)?.join("prepared.json");
            ensure!(file_hash(&p)? == c.prepared_sha256, "cut manifest changed");
            prepared.insert(c.prepared.clone(), read_json::<Prepared>(&p)?);
        }
        by_cut.insert((c.dataset.clone(), c.cut), i);
        if i % 128 == 0 || i + 1 == contexts.len() {
            progress("contexts verified", i + 1, Some(contexts.len()))?;
        }
    }
    let mut selected: BTreeMap<usize, BTreeMap<String, (Vec<usize>, f64, f64, String)>> =
        BTreeMap::new();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut token_cache = TokenCache::default();
    let mut source_receipts = Vec::new();
    for directory in observations {
        let mut paths = walkdir::WalkDir::new(directory)
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|e| {
                e.file_name() == "observations.json"
                    && e.path().parent().unwrap().join("search.json").exists()
            })
            .map(|e| e.into_path())
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths {
            crate::campaign::check_deadline()?;
            // Only completed ledgers are ingested, binding an immutable snapshot.
            let bytes = fs::read(&path)?;
            let rows: Vec<Value> = serde_json::from_slice(&bytes)?;
            let relative_path = if path.is_absolute() {
                path.strip_prefix(root)?
            } else {
                path.as_path()
            };
            source_receipts.push(
                json!({"path":relative_path,"sha256":hash(&bytes),"observations":rows.len()}),
            );
            for o in rows {
                crate::campaign::check_deadline()?;
                *counts.entry("observations_seen".into()).or_default() += 1;
                let key = (
                    o["dataset"].as_str().unwrap_or("").to_string(),
                    o["cut"].as_u64().unwrap_or(u64::MAX) as usize,
                );
                let Some(&context) = by_cut.get(&key) else {
                    *counts.entry("no_matching_context".into()).or_default() += 1;
                    continue;
                };
                let c = &contexts[context];
                if c.partition == "meta_test" {
                    *counts.entry("meta_test_withheld".into()).or_default() += 1;
                    continue;
                }
                let p = &prepared[&c.prepared];
                let cut = p
                    .cuts
                    .iter()
                    .find(|x| x.index == c.cut)
                    .context("cut absent")?;
                ensure!(
                    o["family"] == c.family
                        && o["partition"] == c.partition
                        && o["training_sha256"] == cut.training_sha256
                        && o["validation_sha256"] == cut.validation_sha256,
                    "observation/cut identity mismatch: {}",
                    path.display()
                );
                ensure!(
                    qualification
                        || (o["family_partition_frozen"] == true && o["qualification"] == false),
                    "provisional observation in clean corpus"
                );
                if !o["failure"].is_null() {
                    *counts.entry("failed_program".into()).or_default() += 1;
                    continue;
                }
                let rmse = o["metrics"]["rmse"].as_f64().context("RMSE absent")?;
                let runtime = o["runtime_seconds"].as_f64().context("runtime absent")?;
                ensure!(
                    rmse.is_finite() && rmse >= 0.0 && runtime.is_finite() && runtime >= 0.0,
                    "invalid score"
                );
                let genome = o["genome"].as_str().context("genome absent")?;
                ensure!(
                    hash(genome.as_bytes()) == o["program_sha256"],
                    "program identity mismatch"
                );
                // Validate every observation before skipping duplicates. The
                // first score/runtime/bias for this exact program and cut wins.
                if selected
                    .get(&context)
                    .is_some_and(|rows| rows.contains_key(genome))
                {
                    continue;
                }
                let tokens = match token_cache.encode(&vocab, genome) {
                    Ok(tokens) => tokens,
                    Err(reason) => {
                        *counts.entry(reason).or_default() += 1;
                        continue;
                    }
                };
                let rows = selected.entry(context).or_default();
                // Exact same program/cut has one teaching label; repeated runs do not increase its weight.
                rows.entry(genome.to_string()).or_insert((
                    tokens,
                    rmse,
                    runtime,
                    o["evaluation_bias"]
                        .as_str()
                        .unwrap_or("unknown")
                        .to_string(),
                ));
            }
            progress("observations scanned", counts.get("observations_seen").copied().unwrap_or(0), None)?;
        }
    }
    fs::create_dir_all(output)?;
    let mut file = BufWriter::new(fs::File::create(output.join("records.bin"))?);
    let mut label_file = BufWriter::new(fs::File::create(output.join("labels.jsonl"))?);
    let mut sets = BTreeMap::from([(vec![0usize], 0usize)]);
    let mut legal_sets = vec![vec![0]];
    let mut groups = Vec::new();
    let mut count = 0;
    let mut training_families = BTreeSet::new();
    for (context, programs) in selected {
        crate::campaign::check_deadline()?;
        if programs.len() < 8 {
            *counts.entry("groups_below_eight".into()).or_default() += 1;
            continue;
        }
        let c = &contexts[context];
        let mut rows = programs.into_iter().collect::<Vec<_>>();
        rows.sort_by(|a, b| a.1 .1.total_cmp(&b.1 .1).then(a.0.cmp(&b.0)));
        let median = rows[rows.len() / 2].1 .1.max(1e-8);
        let n = rows.len();
        let mut g = Group {
            context,
            family: c.family.clone(),
            partition: c.partition.clone(),
            records: Vec::new(),
            max_length: 0,
        };
        for (i, (program, (tokens, rmse, runtime, bias))) in rows.into_iter().enumerate() {
            crate::campaign::check_deadline()?;
            let percentile = 1.0 - i as f32 / (n - 1) as f32;
            let mut record = Record {
                tokens: [0; SEQUENCE],
                legal: [0; SEQUENCE],
                length: tokens.len(),
                context,
                value: (1.0 / (1.0 + rmse / median)) as f32,
                percentile,
                runtime: runtime.ln_1p() as f32,
            };
            let mut state = State::default();
            state.sampling = false;
            for (j, &token) in tokens.iter().enumerate() {
                record.tokens[j] = token as i64;
                if j == 0 {
                    continue;
                }
                let legal = state.legal(&vocab);
                let id = if let Some(&id) = sets.get(&legal) {
                    id
                } else {
                    let id = legal_sets.len();
                    legal_sets.push(legal.clone());
                    sets.insert(legal, id);
                    id
                };
                record.legal[j] = id as i64;
                state.advance(&vocab, token)?;
            }
            ensure!(
                legal_sets.len() < u16::MAX as usize,
                "too many grammar states"
            );
            file.write_all(&record.bytes())?;
            serde_json::to_writer(
                &mut label_file,
                &json!({"record":count,"context":context,"program":program,"rmse_cut_normalized":rmse,"runtime_seconds":runtime,"evaluation_bias":bias}),
            )?;
            label_file.write_all(b"\n")?;
            g.records.push(count);
            g.max_length = g.max_length.max(tokens.len());
            count += 1;
        }
        if g.partition == "meta_train" {
            training_families.insert(g.family.clone());
        }
        groups.push(g);
        progress("records packed; publication pending", count, None)?;
    }
    crate::campaign::check_deadline()?;
    file.flush()?;
    file.get_ref().sync_all()?;
    label_file.flush()?;
    label_file.get_ref().sync_all()?;
    ensure!(
        count > 0 && training_families.len() >= 2,
        "insufficient teaching corpus"
    );
    fs::copy(vocab_path, output.join("vocabulary.json"))?;
    let index = Index {
        schema: "parallel.teaching_corpus.v1".into(),
        qualification,
        vocabulary_sha256: file_hash(vocab_path)?,
        context_index_sha256: file_hash(contexts_path)?,
        records_sha256: file_hash(&output.join("records.bin"))?,
        record_bytes: RECORD_BYTES,
        records: count,
        legal_sets,
        contexts,
        groups,
    };
    write_json(&output.join("index.json"), &index)?;
    write_json(
        &output.join("provenance.json"),
        &json!({"sources":source_receipts,"counts":counts,"selected_programs":count,"groups":index.groups.len(),"meta_training_families":training_families,"index_sha256":file_hash(&output.join("index.json"))?,"labels_sha256":file_hash(&output.join("labels.jsonl"))?,"evaluation_bias":"adaptive_cut_heldout_teacher","label_policy":"exact program parameters only; no numeric snapping; duplicate program/cut labels have one vote","qualification":qualification,"independent_evaluation":false,"final_tests_opened":false}),
    )?;
    println!(
        "{}",
        json!({"records":count,"groups":index.groups.len(),"training_families":training_families.len(),"grammar_states":index.legal_sets.len(),"counts":counts})
    );
    Ok(())
}
