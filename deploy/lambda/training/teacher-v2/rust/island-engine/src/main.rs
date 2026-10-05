use anyhow::{Context, Result, anyhow, bail};
use base64::Engine as _;
use flate2::read::GzDecoder;
use rand::SeedableRng;
use rand::prelude::{IndexedRandom, SliceRandom};
use rand_chacha::ChaCha20Rng;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const ISLANDS: usize = 8;
// S1 uses the research campaign's 64-candidate population, within 96 G0 slots.
const ISLAND_SIZE: usize = 8;
const POPULATION: usize = ISLANDS * ISLAND_SIZE;
const CASES: usize = 192;
const SIGNATURE: usize = 64;
const MIGRATION_INTERVAL: u64 = 3 * POPULATION as u64;
const SELECTION_PROTOCOL: &str = "jope.s1-inner-cv-selection/v1";

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Program {
    dsl: String,
    program_sha256: String,
    birth_seq: u64,
    #[serde(default)]
    parents: Vec<String>,
    operator: String,
}

#[derive(Clone, Debug, Deserialize)]
struct SemanticRow {
    selection_protocol: String,
    context_sha256: String,
    program_sha256: String,
    program: Program,
    full_prediction_sha256: String,
    semantic_signature_f32_b64: String,
    sampled_absolute_error_f32_b64: String,
    fitness: f64,
    cv_gate_passed: bool,
    cv_fold_wins: u64,
    cv_geomean_rmse_ratio: f64,
    cv_mean_log_ratio: f64,
    cv_tail_log_ratio: f64,
    cv_log_ratio_sd: f64,
    test_rmse: Option<f64>,
    test_rmse_ratio: Option<f64>,
    complexity_open_parens: u64,
    billed_ms: Option<u64>,
    max_memory_used_mb: Option<u64>,
    result_key: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Individual {
    program: Program,
    full_prediction_sha256: String,
    signature: Vec<f32>,
    cases: Vec<f32>,
    fitness: f64,
    cv_gate_passed: bool,
    cv_fold_wins: u64,
    cv_geomean_rmse_ratio: f64,
    cv_mean_log_ratio: f64,
    cv_tail_log_ratio: f64,
    cv_log_ratio_sd: f64,
    test_rmse: Option<f64>,
    test_rmse_ratio: Option<f64>,
    complexity_open_parens: u64,
    billed_ms: Option<u64>,
    max_memory_used_mb: Option<u64>,
    result_key: String,
    roots: Vec<String>,
    born_completed: u64,
    origin_island: Option<usize>,
}

impl Individual {
    fn from_semantic(row: SemanticRow) -> Result<Self> {
        if row.selection_protocol != SELECTION_PROTOCOL {
            bail!("historical/test-derived semantic cache is not S1")
        }
        if row.program_sha256 != row.program.program_sha256 {
            bail!("semantic row program mismatch")
        }
        Ok(Self {
            program: row.program,
            full_prediction_sha256: row.full_prediction_sha256,
            signature: decode_f32(&row.semantic_signature_f32_b64, SIGNATURE)?,
            cases: decode_f32(&row.sampled_absolute_error_f32_b64, CASES)?,
            fitness: row.fitness,
            cv_gate_passed: row.cv_gate_passed,
            cv_fold_wins: row.cv_fold_wins,
            cv_geomean_rmse_ratio: row.cv_geomean_rmse_ratio,
            cv_mean_log_ratio: row.cv_mean_log_ratio,
            cv_tail_log_ratio: row.cv_tail_log_ratio,
            cv_log_ratio_sd: row.cv_log_ratio_sd,
            test_rmse: row.test_rmse,
            test_rmse_ratio: row.test_rmse_ratio,
            complexity_open_parens: row.complexity_open_parens,
            billed_ms: row.billed_ms,
            max_memory_used_mb: row.max_memory_used_mb,
            result_key: row.result_key,
            roots: Vec::new(),
            born_completed: 0,
            origin_island: None,
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Policy {
    Balanced,
    CvGated,
    CvStable,
    EpsilonLexicase,
    NoveltyQuality,
    AgeRoot,
}

impl Policy {
    fn name(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::CvGated => "cv_gated",
            Self::CvStable => "cv_stable",
            Self::EpsilonLexicase => "epsilon_lexicase",
            Self::NoveltyQuality => "novelty_quality",
            Self::AgeRoot => "age_root",
        }
    }
}

const POLICIES: [Policy; ISLANDS] = [
    Policy::Balanced,
    Policy::Balanced,
    Policy::CvGated,
    Policy::CvStable,
    Policy::EpsilonLexicase,
    Policy::EpsilonLexicase,
    Policy::NoveltyQuality,
    Policy::AgeRoot,
];

#[derive(Clone, Debug)]
struct Island {
    policy: Policy,
    population: Vec<String>,
    archive: Vec<String>,
    completed: u64,
}

#[derive(Debug, Deserialize)]
struct SampleManifest {
    context_sha256: String,
    oof_rows: usize,
    oof_case_indices: Vec<usize>,
    test_case_indices: Vec<usize>,
    oof_target_std: f64,
    projection_rows: usize,
    projection_cols: usize,
    projection_f32_sha256: String,
}

#[derive(Clone, Debug)]
struct Pending {
    island_id: usize,
    proposal_seq: u64,
    proposal_draw: u64,
}

struct Engine {
    context: String,
    proposal_binary: PathBuf,
    all: HashMap<String, Individual>,
    islands: Vec<Island>,
    seen: HashSet<String>,
    active: HashSet<String>,
    active_semantics: HashSet<String>,
    pending: HashMap<String, Pending>,
    projection: Vec<f32>,
    sample: SampleManifest,
    rng: ChaCha20Rng,
    next_birth: u64,
    next_draw: u64,
    proposal_seq: u64,
    new_completed: u64,
    migration_epoch: u64,
    number_re: Regex,
}

#[derive(Debug, Deserialize)]
struct ProposalRaw {
    dsl: String,
    program_sha256: String,
    operator: String,
    draw: u64,
}

#[derive(Debug, Deserialize)]
struct ProposalBatch {
    proposals: Vec<ProposalRaw>,
}

#[derive(Debug, Deserialize)]
struct CommandIn {
    cmd: String,
    island_id: Option<usize>,
    count: Option<usize>,
    result_path: Option<String>,
    result_key: Option<String>,
    program_sha256: Option<String>,
}

fn decode_f32(encoded: &str, expected: usize) -> Result<Vec<f32>> {
    let bytes = base64::engine::general_purpose::STANDARD.decode(encoded)?;
    if bytes.len() != expected * 4 {
        bail!("f32 vector length mismatch")
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect())
}

fn encode_f32(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn f64_hash(values: &[f64]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn diagnostic_cmp(a: Option<f64>, b: Option<f64>) -> Ordering {
    match (a, b) {
        (Some(a), Some(b)) => a.total_cmp(&b),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn cmp_f64(a: f64, b: f64) -> Ordering {
    a.total_cmp(&b)
}

fn distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

impl Engine {
    fn load(
        context: String,
        index: &Path,
        seen_jsonl: &Path,
        resume_jsonl: Option<&Path>,
        projection_path: &Path,
        sample_path: &Path,
        proposal_binary: PathBuf,
    ) -> Result<Self> {
        let sample: SampleManifest = serde_json::from_slice(&fs::read(sample_path)?)?;
        if sample.context_sha256 != context || sample.projection_rows != SIGNATURE {
            bail!("semantic sample/context mismatch")
        }
        validate_sample(&sample)?;
        let projection_bytes = fs::read(projection_path)?;
        if sha256(&projection_bytes) != sample.projection_f32_sha256 {
            bail!("semantic projection hash mismatch")
        }
        if projection_bytes.len() != sample.projection_rows * sample.projection_cols * 4 {
            bail!("semantic projection shape mismatch")
        }
        let projection = projection_bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        if projection.iter().any(|v| !v.is_finite()) {
            bail!("nonfinite projection")
        }
        let file = File::open(index)?;
        let decoder = GzDecoder::new(file);
        let mut all = HashMap::new();
        let mut max_birth = 0;
        for line in BufReader::new(decoder).lines() {
            let row: SemanticRow = serde_json::from_str(&line?)?;
            if row.context_sha256 != context {
                bail!("semantic index contains foreign context")
            }
            max_birth = max_birth.max(row.program.birth_seq);
            let individual = Individual::from_semantic(row)?;
            all.insert(individual.program.program_sha256.clone(), individual);
        }
        let mut roots_memo = HashMap::<String, Vec<String>>::new();
        let keys = all.keys().cloned().collect::<Vec<_>>();
        for hash in keys {
            let roots = Self::roots_for(&hash, &all, &mut roots_memo, &mut HashSet::new());
            all.get_mut(&hash).unwrap().roots = roots;
        }
        let mut seen = HashSet::new();
        for line in BufReader::new(File::open(seen_jsonl)?).lines() {
            let value: Value = serde_json::from_str(&line?)?;
            validate_history(&value, &context)?;
            if let Some(hash) = value
                .get("program")
                .and_then(|x| x.get("program_sha256"))
                .and_then(Value::as_str)
            {
                seen.insert(hash.to_owned());
                if let Some(birth) = value
                    .get("program")
                    .and_then(|x| x.get("birth_seq"))
                    .and_then(Value::as_u64)
                {
                    max_birth = max_birth.max(birth);
                }
            }
        }
        let seed = u64::from_str_radix(&context[..16], 16)? ^ 0x15A1_A5E5_2026_0002;
        let islands = POLICIES
            .iter()
            .map(|policy| Island {
                policy: *policy,
                population: Vec::new(),
                archive: Vec::new(),
                completed: 0,
            })
            .collect();
        let mut engine = Self {
            context,
            proposal_binary,
            all,
            islands,
            seen,
            active: HashSet::new(),
            active_semantics: HashSet::new(),
            pending: HashMap::new(),
            projection,
            sample,
            rng: ChaCha20Rng::seed_from_u64(seed),
            next_birth: max_birth + 1,
            next_draw: 0,
            proposal_seq: 0,
            new_completed: 0,
            migration_epoch: 0,
            number_re: Regex::new(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")?,
        };
        engine.seed_islands()?;
        if let Some(path) = resume_jsonl
            && path.exists()
        {
            engine.load_resume(path)?;
        }
        Ok(engine)
    }

    fn roots_for(
        hash: &str,
        all: &HashMap<String, Individual>,
        memo: &mut HashMap<String, Vec<String>>,
        visiting: &mut HashSet<String>,
    ) -> Vec<String> {
        if let Some(roots) = memo.get(hash) {
            return roots.clone();
        }
        if !visiting.insert(hash.to_owned()) {
            return vec![format!("cycle:{hash}")];
        }
        let mut roots = HashSet::new();
        match all.get(hash) {
            Some(individual) if !individual.program.parents.is_empty() => {
                for parent in &individual.program.parents {
                    if all.contains_key(parent) {
                        roots.extend(Self::roots_for(parent, all, memo, visiting));
                    } else {
                        roots.insert(parent.clone());
                    }
                }
            }
            _ => {
                roots.insert(hash.to_owned());
            }
        }
        visiting.remove(hash);
        let mut roots = roots.into_iter().collect::<Vec<_>>();
        roots.sort();
        memo.insert(hash.to_owned(), roots.clone());
        roots
    }

    fn seed_islands(&mut self) -> Result<()> {
        let mut representatives = HashMap::<String, String>::new();
        for (hash, individual) in &self.all {
            if !individual.cv_gate_passed {
                continue;
            }
            let replace = representatives
                .get(&individual.full_prediction_sha256)
                .and_then(|old| self.all.get(old))
                .map(|old| Self::balanced_cmp(individual, old) == Ordering::Less)
                .unwrap_or(true);
            if replace {
                representatives.insert(individual.full_prediction_sha256.clone(), hash.clone());
            }
        }
        let candidate_hashes = representatives.into_values().collect::<Vec<_>>();
        if candidate_hashes.len() < POPULATION {
            bail!("not enough semantically unique eligible seeds")
        }
        let mut used_program = HashSet::new();
        let mut used_semantic = HashSet::new();
        for island_id in 0..ISLANDS {
            let selected = self.select_seed_set(
                island_id,
                &candidate_hashes,
                &used_program,
                &used_semantic,
                ISLAND_SIZE,
            )?;
            for hash in &selected {
                used_program.insert(hash.clone());
                used_semantic.insert(self.all[hash].full_prediction_sha256.clone());
            }
            self.islands[island_id].archive = selected.clone();
            self.islands[island_id].population = selected;
        }
        self.rebuild_active()?;
        Ok(())
    }

    fn balanced_cmp(a: &Individual, b: &Individual) -> Ordering {
        a.fitness
            .total_cmp(&b.fitness)
            .then_with(|| a.program.program_sha256.cmp(&b.program.program_sha256))
    }

    /// S1 (TEST-STEER-01): no arm reads test_rmse or test_rmse_ratio; those fields are reported, never selected on.
    fn policy_cmp(policy: Policy, a: &Individual, b: &Individual) -> Ordering {
        match policy {
            Policy::Balanced
            | Policy::EpsilonLexicase
            | Policy::NoveltyQuality
            | Policy::AgeRoot => Self::balanced_cmp(a, b),
            Policy::CvGated => b
                .cv_gate_passed
                .cmp(&a.cv_gate_passed)
                .then_with(|| cmp_f64(a.cv_geomean_rmse_ratio, b.cv_geomean_rmse_ratio))
                .then_with(|| cmp_f64(a.cv_log_ratio_sd, b.cv_log_ratio_sd))
                .then_with(|| a.program.program_sha256.cmp(&b.program.program_sha256)),
            Policy::CvStable => cmp_f64(a.cv_geomean_rmse_ratio, b.cv_geomean_rmse_ratio)
                .then_with(|| cmp_f64(a.cv_log_ratio_sd, b.cv_log_ratio_sd))
                .then_with(|| a.program.program_sha256.cmp(&b.program.program_sha256)),
        }
    }

    fn select_seed_set(
        &mut self,
        island_id: usize,
        candidates: &[String],
        used_program: &HashSet<String>,
        used_semantic: &HashSet<String>,
        count: usize,
    ) -> Result<Vec<String>> {
        let policy = self.islands[island_id].policy;
        let mut available = candidates
            .iter()
            .filter(|h| {
                !used_program.contains(*h)
                    && !used_semantic.contains(&self.all[*h].full_prediction_sha256)
            })
            .cloned()
            .collect::<Vec<_>>();
        available.sort_by(|a, b| Self::policy_cmp(policy, &self.all[a], &self.all[b]));
        let mut selected = Vec::new();
        let mut root_counts = HashMap::<String, usize>::new();
        match policy {
            Policy::EpsilonLexicase => {
                let pool = available.iter().take(4096).cloned().collect::<Vec<_>>();
                let mut local = HashSet::new();
                while selected.len() < count {
                    let Some(hash) = self.lexicase_pick(&pool, &local) else {
                        break;
                    };
                    if self.root_allowed(&self.all[&hash], &root_counts, 14) {
                        for root in &self.all[&hash].roots {
                            *root_counts.entry(root.clone()).or_default() += 1;
                        }
                        local.insert(hash.clone());
                        selected.push(hash);
                    } else {
                        local.insert(hash);
                    }
                }
            }
            Policy::NoveltyQuality => {
                let pool = available.iter().take(8192).cloned().collect::<Vec<_>>();
                if let Some(first) = pool.first() {
                    selected.push(first.clone());
                }
                while selected.len() < count {
                    let mut best: Option<(f64, String)> = None;
                    for hash in &pool {
                        if selected.contains(hash) {
                            continue;
                        }
                        let individual = &self.all[hash];
                        if !self.root_allowed(individual, &root_counts, 14) {
                            continue;
                        }
                        let novelty = selected
                            .iter()
                            .map(|s| distance(&individual.signature, &self.all[s].signature))
                            .fold(f64::INFINITY, f64::min);
                        let quality = (-individual.fitness).clamp(0.0, 3.0);
                        let score = novelty + 0.05 * quality;
                        if best.as_ref().map(|x| score > x.0).unwrap_or(true) {
                            best = Some((score, hash.clone()));
                        }
                    }
                    let Some((_, hash)) = best else { break };
                    for root in &self.all[&hash].roots {
                        *root_counts.entry(root.clone()).or_default() += 1;
                    }
                    selected.push(hash);
                }
            }
            Policy::AgeRoot => {
                available.sort_by(|a, b| {
                    self.all[a]
                        .program
                        .birth_seq
                        .cmp(&self.all[b].program.birth_seq)
                        .then_with(|| Self::policy_cmp(policy, &self.all[a], &self.all[b]))
                });
                self.scan_select(&available, &mut selected, &mut root_counts, count, 8);
            }
            _ => self.scan_select(&available, &mut selected, &mut root_counts, count, 14),
        }
        if selected.len() < count {
            for hash in available {
                if selected.len() == count {
                    break;
                }
                if !selected.contains(&hash) {
                    selected.push(hash);
                }
            }
        }
        if selected.len() != count {
            bail!("could not seed island {island_id}")
        }
        Ok(selected)
    }

    fn scan_select(
        &self,
        available: &[String],
        selected: &mut Vec<String>,
        root_counts: &mut HashMap<String, usize>,
        count: usize,
        root_cap: usize,
    ) {
        for hash in available {
            if selected.len() == count {
                break;
            }
            let individual = &self.all[hash];
            if self.root_allowed(individual, root_counts, root_cap) {
                for root in &individual.roots {
                    *root_counts.entry(root.clone()).or_default() += 1;
                }
                selected.push(hash.clone());
            }
        }
    }

    fn root_allowed(
        &self,
        individual: &Individual,
        counts: &HashMap<String, usize>,
        cap: usize,
    ) -> bool {
        individual
            .roots
            .iter()
            .all(|root| counts.get(root).copied().unwrap_or(0) < cap)
    }

    fn lexicase_pick(&mut self, pool: &[String], excluded: &HashSet<String>) -> Option<String> {
        let mut survivors = pool
            .iter()
            .filter(|h| !excluded.contains(*h))
            .cloned()
            .collect::<Vec<_>>();
        if survivors.is_empty() {
            return None;
        }
        let mut cases = (0..CASES).collect::<Vec<_>>();
        cases.shuffle(&mut self.rng);
        for case in cases {
            if survivors.len() <= 1 {
                break;
            }
            let best = survivors
                .iter()
                .map(|h| self.all[h].cases[case])
                .fold(f32::INFINITY, f32::min);
            let mut errors = survivors
                .iter()
                .map(|h| self.all[h].cases[case])
                .collect::<Vec<_>>();
            let med = median(&mut errors);
            for error in &mut errors {
                *error = (*error - med).abs();
            }
            let epsilon = median(&mut errors).max(1e-6);
            survivors.retain(|h| self.all[h].cases[case] <= best + epsilon);
        }
        survivors.choose(&mut self.rng).cloned()
    }

    fn rebuild_active(&mut self) -> Result<()> {
        self.active.clear();
        self.active_semantics.clear();
        for island in &self.islands {
            if island.population.len() != ISLAND_SIZE {
                bail!("island population size invariant")
            }
            for hash in &island.population {
                if !self.active.insert(hash.clone()) {
                    bail!("program active in multiple islands")
                }
                if !self
                    .active_semantics
                    .insert(self.all[hash].full_prediction_sha256.clone())
                {
                    bail!("semantic duplicate across active islands")
                }
            }
        }
        if self.active.len() != POPULATION {
            bail!("global population invariant")
        }
        Ok(())
    }

    fn load_resume(&mut self, path: &Path) -> Result<()> {
        for line in BufReader::new(File::open(path)?).lines() {
            let value: Value = serde_json::from_str(&line?)?;
            validate_history(&value, &self.context)?;
            let hash = value
                .get("program")
                .and_then(|x| x.get("program_sha256"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            if !hash.is_empty() {
                self.seen.insert(hash);
            }
            if value.get("status").and_then(Value::as_str) != Some("completed") {
                continue;
            }
            let island_id = value
                .get("island_id")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("resume completed row missing island_id"))?
                as usize;
            let individual = self.individual_from_scored(&value)?;
            self.next_birth = self.next_birth.max(individual.program.birth_seq + 1);
            self.proposal_seq = self.proposal_seq.max(
                value
                    .get("proposal_seq")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            );
            self.next_draw = self.next_draw.max(
                value
                    .get("proposal_draw")
                    .and_then(Value::as_u64)
                    .map(|draw| draw + 1)
                    .unwrap_or(0),
            );
            let hash = individual.program.program_sha256.clone();
            self.all.insert(hash.clone(), individual);
            self.islands[island_id].archive.push(hash);
            self.islands[island_id].completed += 1;
            self.new_completed += 1;
        }
        for island_id in 0..ISLANDS {
            self.reselect(island_id)?;
        }
        self.migration_epoch = self.new_completed / MIGRATION_INTERVAL;
        Ok(())
    }

    fn individual_from_scored(&self, value: &Value) -> Result<Individual> {
        if value["selection_protocol"] != SELECTION_PROTOCOL
            || value["schema"] != "jope.island-scored-evaluation/s1-v1"
        {
            bail!("historical/test-derived resume is not S1")
        }
        let program: Program = serde_json::from_value(value["program"].clone())?;
        Ok(Individual {
            program,
            full_prediction_sha256: value["full_prediction_sha256"]
                .as_str()
                .ok_or_else(|| anyhow!("missing semantic hash"))?
                .to_owned(),
            signature: decode_f32(
                value["semantic_signature_f32_b64"].as_str().unwrap(),
                SIGNATURE,
            )?,
            cases: decode_f32(
                value["sampled_absolute_error_f32_b64"].as_str().unwrap(),
                CASES,
            )?,
            fitness: value["fitness"].as_f64().unwrap(),
            cv_gate_passed: value["cv_gate_passed"].as_bool().unwrap(),
            cv_fold_wins: value["cv_fold_wins"].as_u64().unwrap(),
            cv_geomean_rmse_ratio: value["cv_geomean_rmse_ratio"].as_f64().unwrap(),
            cv_mean_log_ratio: value["cv_mean_log_ratio"].as_f64().unwrap(),
            cv_tail_log_ratio: value["cv_tail_log_ratio"].as_f64().unwrap(),
            cv_log_ratio_sd: value["cv_log_ratio_sd"].as_f64().unwrap(),
            test_rmse: value["test_rmse"].as_f64(),
            test_rmse_ratio: value["test_rmse_ratio"].as_f64(),
            complexity_open_parens: value["complexity_open_parens"].as_u64().unwrap(),
            billed_ms: value.get("billed_ms").and_then(Value::as_u64),
            max_memory_used_mb: value.get("max_memory_used_mb").and_then(Value::as_u64),
            result_key: value["result_key"].as_str().unwrap_or_default().to_owned(),
            roots: serde_json::from_value(value["roots"].clone())?,
            born_completed: value["born_completed"].as_u64().unwrap_or(0),
            origin_island: value
                .get("origin_island")
                .and_then(Value::as_u64)
                .map(|x| x as usize),
        })
    }

    fn parent_pool(&mut self, island_id: usize) -> Vec<String> {
        let island = &self.islands[island_id];
        let policy = island.policy;
        let mut hashes = island.population.clone();
        match policy {
            Policy::EpsilonLexicase => {
                let mut out = Vec::new();
                let mut excluded = HashSet::new();
                while out.len() < 64 {
                    let Some(hash) = self.lexicase_pick(&hashes, &excluded) else {
                        break;
                    };
                    excluded.insert(hash.clone());
                    out.push(hash);
                }
                if out.len() >= 2 {
                    hashes = out;
                }
            }
            Policy::NoveltyQuality => {
                let comparison_population = hashes.clone();
                hashes.sort_by(|a, b| {
                    let na = self.novelty_in(a, &comparison_population);
                    let nb = self.novelty_in(b, &comparison_population);
                    nb.total_cmp(&na)
                        .then_with(|| Self::policy_cmp(policy, &self.all[a], &self.all[b]))
                });
                hashes.truncate(64);
            }
            Policy::AgeRoot => {
                hashes.sort_by_key(|h| std::cmp::Reverse(self.all[h].program.birth_seq));
                hashes.truncate(64);
            }
            _ => {
                hashes.sort_by(|a, b| Self::policy_cmp(policy, &self.all[a], &self.all[b]));
                let mut out = hashes.iter().take(32).cloned().collect::<Vec<_>>();
                let mut tail = hashes.iter().skip(32).cloned().collect::<Vec<_>>();
                tail.shuffle(&mut self.rng);
                out.extend(tail.into_iter().take(32));
                hashes = out;
            }
        }
        hashes
    }

    fn novelty_in(&self, hash: &str, population: &[String]) -> f64 {
        let mut distances = population
            .iter()
            .filter(|other| other.as_str() != hash)
            .map(|other| distance(&self.all[hash].signature, &self.all[other].signature))
            .collect::<Vec<_>>();
        distances.sort_by(|a, b| a.total_cmp(b));
        let k = distances.len().min(15);
        if k == 0 {
            return 0.0;
        }
        distances[..k].iter().sum::<f64>() / k as f64
    }

    fn propose(&mut self, island_id: usize, count: usize) -> Result<Value> {
        if island_id >= ISLANDS || count == 0 || count > ISLAND_SIZE {
            bail!("invalid proposal request")
        }
        let parent_hashes = self.parent_pool(island_id);
        if parent_hashes.len() < 2 {
            bail!("parent pool too small")
        }
        let parents = parent_hashes
            .iter()
            .map(|hash| self.all[hash].program.dsl.clone())
            .collect::<Vec<_>>();
        let mut output = Vec::new();
        let mut structure_counts = HashMap::<String, usize>::new();
        let mut rounds = 0;
        while output.len() < count && rounds < 32 {
            rounds += 1;
            let ask = ((count - output.len()) * 4).clamp(32, 256);
            let request = json!({
                "parents": parents,
                "seed": u64::from_str_radix(&self.context[..16], 16)?,
                "generation": self.new_completed / POPULATION as u64,
                "offset": self.next_draw,
                "count": ask,
            });
            let mut child = Command::new(&self.proposal_binary)
                .arg("--propose")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .with_context(|| format!("spawn {}", self.proposal_binary.display()))?;
            child
                .stdin
                .as_mut()
                .unwrap()
                .write_all(&serde_json::to_vec(&request)?)?;
            let raw = child.wait_with_output()?;
            if !raw.status.success() {
                bail!("proposal engine failed")
            }
            let batch: ProposalBatch = serde_json::from_slice(&raw.stdout)?;
            if let Some(max_draw) = batch.proposals.iter().map(|x| x.draw).max() {
                self.next_draw = max_draw + 1;
            }
            for proposal in batch.proposals {
                if output.len() == count {
                    break;
                }
                if self.seen.contains(&proposal.program_sha256) || !self.admit_dsl(&proposal.dsl) {
                    continue;
                }
                let structure = self.number_re.replace_all(&proposal.dsl, "#").to_string();
                let structure_count = structure_counts.entry(structure).or_default();
                if *structure_count >= 4 {
                    continue;
                }
                *structure_count += 1;
                let index = proposal.draw as usize % parent_hashes.len();
                let parent_programs = match proposal.operator.as_str() {
                    "grammar_random" => Vec::new(),
                    "typed_crossover" => vec![
                        parent_hashes[index].clone(),
                        parent_hashes[(index + 1) % parent_hashes.len()].clone(),
                    ],
                    _ => vec![parent_hashes[index].clone()],
                };
                let program = Program {
                    dsl: proposal.dsl,
                    program_sha256: proposal.program_sha256.clone(),
                    birth_seq: self.next_birth,
                    parents: parent_programs,
                    operator: proposal.operator,
                };
                self.next_birth += 1;
                self.proposal_seq += 1;
                self.seen.insert(program.program_sha256.clone());
                self.pending.insert(
                    program.program_sha256.clone(),
                    Pending {
                        island_id,
                        proposal_seq: self.proposal_seq,
                        proposal_draw: proposal.draw,
                    },
                );
                output.push(json!({
                    "candidate": program,
                    "island_id": island_id,
                    "policy": self.islands[island_id].policy.name(),
                    "proposal_seq": self.proposal_seq,
                    "proposal_draw": proposal.draw,
                    "island_completed": self.islands[island_id].completed,
                    "migration_epoch": self.migration_epoch,
                }));
            }
        }
        if output.len() != count {
            bail!("unique admitted proposal stream exhausted")
        }
        Ok(
            json!({"schema":"jope.island-proposal-batch/s1-v1","selection_protocol":SELECTION_PROTOCOL,"context_sha256":self.context,"proposals":output}),
        )
    }

    fn admit_dsl(&self, dsl: &str) -> bool {
        let complexity = dsl.bytes().filter(|x| *x == b'(').count();
        if !(3..=10).contains(&complexity) {
            return false;
        }
        if dsl.contains("(histgbm2") || dsl.contains("(gpr ") {
            return false;
        }
        if dsl.matches("(winsor_clip").count() > 3 {
            return false;
        }
        true
    }

    fn accept(
        &mut self,
        island_id: usize,
        result_path: &Path,
        result_key: String,
    ) -> Result<Value> {
        if island_id >= ISLANDS {
            bail!("invalid island")
        }
        let mut compressed = Vec::new();
        File::open(result_path)?.read_to_end(&mut compressed)?;
        let mut decoder = GzDecoder::new(&compressed[..]);
        let mut raw = Vec::new();
        decoder.read_to_end(&mut raw)?;
        let pack: Value = serde_json::from_slice(&raw)?;
        let record = &pack["record"];
        if record["context_sha256"].as_str() != Some(&self.context) {
            bail!("accepted result context mismatch")
        }
        let program: Program = serde_json::from_value(record["program"].clone())?;
        let pending = self.pending.remove(&program.program_sha256);
        if let Some(pending) = &pending
            && pending.island_id != island_id
        {
            bail!("pending island mismatch")
        }
        let status = record["status"].as_str().unwrap_or("missing");
        if status != "completed" {
            return Ok(json!({
                "schema":"jope.island-scored-evaluation/s1-v1",
            "selection_protocol":SELECTION_PROTOCOL,
                "status":status,
                "context_sha256":self.context,
                "program":program,
                "island_id":island_id,
                "policy":self.islands[island_id].policy.name(),
                "proposal_seq":pending.as_ref().map(|x|x.proposal_seq),
                "proposal_draw":pending.as_ref().map(|x|x.proposal_draw),
                "result_key":result_key,
            }));
        }
        let mut individual = self.score_record(record, program, result_key)?;
        individual.origin_island = Some(island_id);
        individual.born_completed = self.new_completed;
        let hash = individual.program.program_sha256.clone();
        let full_semantic = individual.full_prediction_sha256.clone();
        let roots = individual.roots.clone();
        self.all.insert(hash.clone(), individual);
        self.islands[island_id].archive.push(hash.clone());
        if self.islands[island_id].archive.len() > 4096 {
            self.islands[island_id].archive.remove(0);
        }
        self.islands[island_id].completed += 1;
        self.new_completed += 1;
        self.reselect(island_id)?;
        if self.new_completed / MIGRATION_INTERVAL > self.migration_epoch {
            self.migrate()?;
        }
        let selected = self.active.contains(&hash);
        let individual = &self.all[&hash];
        Ok(json!({
            "schema":"jope.island-scored-evaluation/s1-v1",
            "selection_protocol":SELECTION_PROTOCOL,
            "status":"completed",
            "context_sha256":self.context,
            "program":individual.program,
            "island_id":island_id,
            "policy":self.islands[island_id].policy.name(),
            "proposal_seq":pending.as_ref().map(|x|x.proposal_seq),
            "proposal_draw":pending.as_ref().map(|x|x.proposal_draw),
            "result_key":individual.result_key,
            "fitness":individual.fitness,
            "cv_gate_passed":individual.cv_gate_passed,
            "cv_fold_wins":individual.cv_fold_wins,
            "cv_geomean_rmse_ratio":individual.cv_geomean_rmse_ratio,
            "cv_mean_log_ratio":individual.cv_mean_log_ratio,
            "cv_tail_log_ratio":individual.cv_tail_log_ratio,
            "cv_log_ratio_sd":individual.cv_log_ratio_sd,
            "test_rmse":individual.test_rmse,
            "test_rmse_ratio":individual.test_rmse_ratio,
            "complexity_open_parens":individual.complexity_open_parens,
            "full_prediction_sha256":full_semantic,
            "semantic_signature_f32_b64":encode_f32(&individual.signature),
            "sampled_absolute_error_f32_b64":encode_f32(&individual.cases),
            "roots":roots,
            "born_completed":individual.born_completed,
            "origin_island":individual.origin_island,
            "population_selected":selected,
            "semantic_duplicate_of_active":!selected && self.active_semantics.contains(&individual.full_prediction_sha256),
            "migration_epoch":self.migration_epoch,
        }))
    }

    fn score_record(
        &self,
        record: &Value,
        program: Program,
        result_key: String,
    ) -> Result<Individual> {
        validate_sample(&self.sample)?;
        if record["schema"] != "jope.s1-evaluation/v1" || record["status"] != "completed" {
            bail!("S1 needs its versioned completed evaluation")
        }
        let folds = record["inner_cv"]
            .as_array()
            .ok_or_else(|| anyhow!("inner_cv missing"))?;
        if folds.len() != 7 {
            bail!("expected seven folds")
        }
        let mut logs = Vec::new();
        let mut cv_rmse = Vec::new();
        let mut oof_predictions = Vec::new();
        let mut oof_targets = Vec::new();
        let mut wins = 0_u64;
        for fold in folds {
            let rmse = fold["metrics"]["rmse"]
                .as_f64()
                .ok_or_else(|| anyhow!("fold rmse"))?;
            let baseline = fold["baseline_metrics"]["rmse"]
                .as_f64()
                .ok_or_else(|| anyhow!("fold baseline"))?;
            let ratio = rmse / baseline;
            if !rmse.is_finite()
                || rmse < 0.0
                || !baseline.is_finite()
                || baseline <= 0.0
                || !ratio.is_finite()
            {
                bail!("invalid inner-CV loss/baseline")
            }
            if ratio < 1.0 {
                wins += 1;
            }
            logs.push(ratio.max(1e-300).ln());
            cv_rmse.push(rmse);
            oof_predictions.extend(array_f64(&fold["predictions"])?);
            oof_targets.extend(array_f64(&fold["targets"])?);
        }
        let mean = logs.iter().sum::<f64>() / logs.len() as f64;
        let variance =
            logs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / logs.len() as f64;
        let sd = variance.sqrt();
        logs.sort_by(|a, b| b.total_cmp(a));
        let tail = logs[..3].iter().sum::<f64>() / 3.0;
        let geo = mean.exp();
        // Test diagnostics may be absent or failed. Neither their value nor their
        // availability can censor an otherwise valid CV candidate.
        let test = &record["official_test"];
        let test_rmse = test["metrics"]["rmse"]
            .as_f64()
            .filter(|v| v.is_finite() && *v >= 0.0);
        let test_baseline = test["baseline_metrics"]["rmse"]
            .as_f64()
            .filter(|v| v.is_finite() && *v > 0.0);
        let test_ratio = test_rmse
            .zip(test_baseline)
            .map(|(r, b)| r / b)
            .filter(|v| v.is_finite());
        let gate = geo <= 0.90 && wins >= 5;
        let complexity = program.dsl.bytes().filter(|x| *x == b'(').count() as u64;
        let fitness = if gate {
            0.80 * mean + 0.10 * tail + 0.10 * sd + 0.0005 * complexity as f64
        } else {
            100.0 + mean + tail + sd
        };
        let _test_predictions = array_f64(&test["predictions"]).unwrap_or_default();
        if oof_predictions.len() != self.sample.oof_rows
            || oof_targets.len() != self.sample.oof_rows
        {
            bail!("OOF result shape mismatch")
        }
        let mut residual = Vec::with_capacity(self.sample.projection_cols);
        for (prediction, target) in oof_predictions.iter().zip(&oof_targets) {
            residual.push(((prediction - target) / self.sample.oof_target_std) as f32);
        }
        // S1: test residuals never enter the signature, the cases or the semantic key (TEST-STEER-01).
        if residual.len() != self.sample.projection_cols {
            bail!("S1: the projection must cover OOF residuals only")
        }
        let mut signature = vec![0_f32; SIGNATURE];
        for (row, output) in signature.iter_mut().enumerate() {
            let projection = &self.projection[row * residual.len()..(row + 1) * residual.len()];
            *output = projection.iter().zip(&residual).map(|(a, b)| a * b).sum();
        }
        let mut cases = Vec::with_capacity(CASES);
        for index in &self.sample.oof_case_indices {
            cases.push(residual[*index].abs());
        }
        if !self.sample.test_case_indices.is_empty() {
            bail!("S1: test case indices are not admitted")
        }
        if cases.len() != CASES {
            bail!("semantic cases mismatch")
        }
        let prediction_values = oof_predictions;
        let roots = if program.parents.is_empty() {
            vec![program.program_sha256.clone()]
        } else {
            let mut roots = HashSet::new();
            for parent in &program.parents {
                match self.all.get(parent) {
                    Some(individual) => roots.extend(individual.roots.iter().cloned()),
                    None => {
                        roots.insert(parent.clone());
                    }
                }
            }
            let mut roots = roots.into_iter().collect::<Vec<_>>();
            roots.sort();
            roots
        };
        Ok(Individual {
            program,
            full_prediction_sha256: f64_hash(&prediction_values),
            signature,
            cases,
            fitness,
            cv_gate_passed: gate,
            cv_fold_wins: wins,
            cv_geomean_rmse_ratio: geo,
            cv_mean_log_ratio: mean,
            cv_tail_log_ratio: tail,
            cv_log_ratio_sd: sd,
            test_rmse,
            test_rmse_ratio: test_ratio,
            complexity_open_parens: complexity,
            billed_ms: None,
            max_memory_used_mb: None,
            result_key,
            roots,
            born_completed: self.new_completed,
            origin_island: None,
        })
    }

    fn reselect(&mut self, island_id: usize) -> Result<()> {
        let policy = self.islands[island_id].policy;
        let other_active_semantics = self
            .islands
            .iter()
            .enumerate()
            .filter(|(id, _)| *id != island_id)
            .flat_map(|(_, island)| island.population.iter())
            .map(|hash| self.all[hash].full_prediction_sha256.clone())
            .collect::<HashSet<_>>();
        let mut best_by_semantic = HashMap::<String, String>::new();
        for hash in self.islands[island_id]
            .archive
            .iter()
            .chain(self.islands[island_id].population.iter())
        {
            let individual = &self.all[hash];
            if !individual.cv_gate_passed
                || other_active_semantics.contains(&individual.full_prediction_sha256)
            {
                continue;
            }
            let replace = best_by_semantic
                .get(&individual.full_prediction_sha256)
                .map(|old| Self::policy_cmp(policy, individual, &self.all[old]) == Ordering::Less)
                .unwrap_or(true);
            if replace {
                best_by_semantic.insert(individual.full_prediction_sha256.clone(), hash.clone());
            }
        }
        let mut candidates = best_by_semantic.into_values().collect::<Vec<_>>();
        candidates.sort_by(|a, b| Self::policy_cmp(policy, &self.all[a], &self.all[b]));
        let old = self.islands[island_id].population.clone();
        let mut selected = Vec::new();
        match policy {
            Policy::EpsilonLexicase => {
                let mut excluded = HashSet::new();
                while selected.len() < ISLAND_SIZE {
                    let Some(hash) = self.lexicase_pick(&candidates, &excluded) else {
                        break;
                    };
                    excluded.insert(hash.clone());
                    selected.push(hash);
                }
            }
            Policy::NoveltyQuality => {
                if let Some(first) = candidates.first() {
                    selected.push(first.clone());
                }
                while selected.len() < ISLAND_SIZE {
                    let mut best: Option<(f64, String)> = None;
                    for hash in candidates.iter().take(1024) {
                        if selected.contains(hash) {
                            continue;
                        }
                        let novelty = selected
                            .iter()
                            .map(|s| distance(&self.all[hash].signature, &self.all[s].signature))
                            .fold(f64::INFINITY, f64::min);
                        let score = novelty - 0.02 * self.all[hash].fitness;
                        if best.as_ref().map(|x| score > x.0).unwrap_or(true) {
                            best = Some((score, hash.clone()));
                        }
                    }
                    let Some((_, hash)) = best else { break };
                    selected.push(hash);
                }
            }
            Policy::AgeRoot => {
                candidates.sort_by(|a, b| {
                    std::cmp::Reverse(self.all[a].program.birth_seq)
                        .cmp(&std::cmp::Reverse(self.all[b].program.birth_seq))
                        .then_with(|| Self::policy_cmp(policy, &self.all[a], &self.all[b]))
                });
                selected.extend(candidates.into_iter().take(ISLAND_SIZE));
            }
            _ => selected.extend(candidates.into_iter().take(ISLAND_SIZE)),
        }
        if selected.len() < ISLAND_SIZE {
            for hash in old {
                if selected.len() == ISLAND_SIZE {
                    break;
                }
                let semantic = &self.all[&hash].full_prediction_sha256;
                if !selected.contains(&hash)
                    && !other_active_semantics.contains(semantic)
                    && !selected
                        .iter()
                        .any(|x| self.all[x].full_prediction_sha256 == *semantic)
                {
                    selected.push(hash);
                }
            }
        }
        if selected.len() != ISLAND_SIZE {
            bail!("island reselection could not fill population")
        }
        self.islands[island_id].population = selected;
        self.rebuild_active()?;
        Ok(())
    }

    fn migrate(&mut self) -> Result<()> {
        let mut migrants = Vec::new();
        for island_id in 0..ISLANDS {
            let mut population = self.islands[island_id].population.clone();
            let policy = self.islands[island_id].policy;
            population.sort_by(|a, b| Self::policy_cmp(policy, &self.all[a], &self.all[b]));
            let mut chosen = population.iter().take(3).cloned().collect::<Vec<_>>();
            let mut novel = population.clone();
            novel.sort_by(|a, b| {
                self.novelty_in(b, &population)
                    .total_cmp(&self.novelty_in(a, &population))
            });
            for hash in novel {
                if chosen.len() == 6 {
                    break;
                }
                if !chosen.contains(&hash) {
                    chosen.push(hash);
                }
            }
            migrants.push(chosen);
        }
        for island_id in 0..ISLANDS {
            let incoming = migrants[(island_id + ISLANDS - 1) % ISLANDS].clone();
            let outgoing = migrants[island_id].clone();
            self.islands[island_id]
                .population
                .retain(|hash| !outgoing.contains(hash));
            self.islands[island_id]
                .population
                .extend(incoming.iter().cloned());
            self.islands[island_id].archive.extend(incoming);
        }
        self.migration_epoch += 1;
        self.rebuild_active()?;
        Ok(())
    }

    fn status(&self) -> Value {
        let mut root_counts = HashMap::<String, usize>::new();
        let mut policies = Vec::new();
        for (island_id, island) in self.islands.iter().enumerate() {
            let semantics = island
                .population
                .iter()
                .map(|h| self.all[h].full_prediction_sha256.clone())
                .collect::<HashSet<_>>();
            for hash in &island.population {
                for root in &self.all[hash].roots {
                    *root_counts.entry(root.clone()).or_default() += 1;
                }
            }
            let best = island
                .population
                .iter()
                .min_by(|a, b| Self::policy_cmp(island.policy, &self.all[*a], &self.all[*b]))
                .unwrap();
            policies.push(json!({
                "island_id":island_id,
                "policy":island.policy.name(),
                "population":island.population.len(),
                "unique_semantics":semantics.len(),
                "completed":island.completed,
                "best_program_sha256":best,
                "best_fitness":self.all[best].fitness,
                "best_test_rmse":self.all[best].test_rmse,
                "best_cv_ratio":self.all[best].cv_geomean_rmse_ratio,
            }));
        }
        let balanced = self
            .active
            .iter()
            .min_by(|a, b| Self::balanced_cmp(&self.all[*a], &self.all[*b]))
            .unwrap();
        let best_test = self
            .active
            .iter()
            .min_by(|a, b| diagnostic_cmp(self.all[*a].test_rmse, self.all[*b].test_rmse))
            .unwrap();
        let mut roots = root_counts.into_iter().collect::<Vec<_>>();
        roots.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        json!({
            "schema":"jope.island-engine-status/s1-v1",
            "selection_protocol":SELECTION_PROTOCOL,
            "budget_policy":"fixed_equal_18",
            "test_fields":"diagnostic_only",
            "context_sha256":self.context,
            "population":self.active.len(),
            "unique_active_semantics":self.active_semantics.len(),
            "distinct_root_ancestors":roots.len(),
            "largest_root_families":roots.into_iter().take(10).map(|(root,members)|json!({"root":root,"members":members})).collect::<Vec<_>>(),
            "new_completed":self.new_completed,
            "equivalent_generation":self.new_completed as f64 / POPULATION as f64,
            "migration_epoch":self.migration_epoch,
            "seen_programs":self.seen.len(),
            "pending_programs":self.pending.len(),
            "balanced_leader":{"program_sha256":balanced,"fitness":self.all[balanced].fitness,"test_rmse":self.all[balanced].test_rmse,"cv_ratio":self.all[balanced].cv_geomean_rmse_ratio},
            "best_test_leader":{"program_sha256":best_test,"fitness":self.all[best_test].fitness,"test_rmse":self.all[best_test].test_rmse,"cv_ratio":self.all[best_test].cv_geomean_rmse_ratio},
            "islands":policies,
        })
    }

    fn snapshot(&self) -> Value {
        let islands = self
            .islands
            .iter()
            .enumerate()
            .map(|(id, island)| {
                json!({
                    "island_id":id,
                    "policy":island.policy.name(),
                    "completed":island.completed,
                    "population":island.population,
                    "archive_tail":island.archive.iter().rev().take(512).cloned().collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        json!({
            "schema":"jope.island-engine-snapshot/s1-v1",
            "selection_protocol":SELECTION_PROTOCOL,
            "status":self.status(),
            "next_birth":self.next_birth,
            "next_draw":self.next_draw,
            "proposal_seq":self.proposal_seq,
            "islands":islands,
        })
    }

    fn abandon(&mut self, program_sha256: &str) -> Result<Value> {
        let pending = self
            .pending
            .remove(program_sha256)
            .ok_or_else(|| anyhow!("program is not pending"))?;
        Ok(json!({
            "schema":"jope.island-abandonment/v1",
            "program_sha256":program_sha256,
            "island_id":pending.island_id,
            "proposal_seq":pending.proposal_seq,
            "proposal_draw":pending.proposal_draw,
        }))
    }
}

fn validate_history(value: &Value, context: &str) -> Result<()> {
    if value["selection_protocol"] != SELECTION_PROTOCOL || value["context_sha256"] != context {
        bail!("history is not the admitted S1 context/protocol")
    }
    Ok(())
}

fn validate_sample(sample: &SampleManifest) -> Result<()> {
    if sample.oof_rows == 0
        || sample.projection_rows != SIGNATURE
        || sample.projection_cols != sample.oof_rows
        || !sample.test_case_indices.is_empty()
        || sample.oof_case_indices.len() != CASES
        || sample
            .oof_case_indices
            .iter()
            .any(|i| *i >= sample.oof_rows)
        || !sample.oof_target_std.is_finite()
        || sample.oof_target_std <= 0.0
    {
        bail!("S1 requires OOF-only projection and case indices")
    }
    Ok(())
}

fn array_f64(value: &Value) -> Result<Vec<f64>> {
    value
        .as_array()
        .ok_or_else(|| anyhow!("expected float array"))?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| anyhow!("non-float array value")))
        .collect()
}

fn arg_value(args: &[String], name: &str) -> Result<String> {
    let index = args
        .iter()
        .position(|x| x == name)
        .ok_or_else(|| anyhow!("missing {name}"))?;
    args.get(index + 1)
        .cloned()
        .ok_or_else(|| anyhow!("missing value for {name}"))
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let context = arg_value(&args, "--context")?;
    let index = PathBuf::from(arg_value(&args, "--control-index")?);
    let seen = PathBuf::from(arg_value(&args, "--seen-jsonl")?);
    let resume = args
        .iter()
        .position(|x| x == "--resume-jsonl")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);
    let projection = PathBuf::from(arg_value(&args, "--projection")?);
    let sample = PathBuf::from(arg_value(&args, "--sample")?);
    let proposal = PathBuf::from(arg_value(&args, "--proposal-binary")?);
    let mut engine = Engine::load(
        context,
        &index,
        &seen,
        resume.as_deref(),
        &projection,
        &sample,
        proposal,
    )?;
    println!(
        "{}",
        serde_json::to_string(&json!({"event":"ready","status":engine.status()}))?
    );
    io::stdout().flush()?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let command: CommandIn = match serde_json::from_str(&line) {
            Ok(command) => command,
            Err(error) => {
                println!("{}", json!({"ok":false,"error":error.to_string()}));
                io::stdout().flush()?;
                continue;
            }
        };
        let result = match command.cmd.as_str() {
            "status" => Ok(engine.status()),
            "snapshot" => Ok(engine.snapshot()),
            "propose" => engine.propose(
                command
                    .island_id
                    .ok_or_else(|| anyhow!("island_id required"))?,
                command.count.ok_or_else(|| anyhow!("count required"))?,
            ),
            "accept" => engine.accept(
                command
                    .island_id
                    .ok_or_else(|| anyhow!("island_id required"))?,
                Path::new(
                    command
                        .result_path
                        .as_deref()
                        .ok_or_else(|| anyhow!("result_path required"))?,
                ),
                command.result_key.unwrap_or_default(),
            ),
            "abandon" => engine.abandon(
                command
                    .program_sha256
                    .as_deref()
                    .ok_or_else(|| anyhow!("program_sha256 required"))?,
            ),
            "shutdown" => {
                println!("{}", json!({"ok":true,"result":engine.snapshot()}));
                io::stdout().flush()?;
                break;
            }
            _ => Err(anyhow!("unknown command")),
        };
        match result {
            Ok(value) => println!(
                "{}",
                serde_json::to_string(&json!({"ok":true,"result":value}))?
            ),
            Err(error) => println!(
                "{}",
                serde_json::to_string(&json!({"ok":false,"error":format!("{error:#}")}))?
            ),
        }
        io::stdout().flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const OOF: usize = 210; // 7 folds x 30 rows
    const TEST: usize = 40;

    fn s1_engine() -> Engine {
        let mut s = 0x2545F4914F6CDD1D_u64;
        let projection = (0..SIGNATURE * OOF)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 2001) as f32 / 1000.0 - 1.0
            })
            .collect();
        Engine {
            context: "c".into(),
            proposal_binary: PathBuf::new(),
            all: HashMap::new(),
            islands: Vec::new(),
            seen: HashSet::new(),
            active: HashSet::new(),
            active_semantics: HashSet::new(),
            pending: HashMap::new(),
            projection,
            sample: SampleManifest {
                context_sha256: "c".into(),
                oof_rows: OOF,
                oof_case_indices: (0..CASES).map(|i| (i * 7) % OOF).collect(),
                test_case_indices: Vec::new(),
                oof_target_std: 1.0,
                projection_rows: SIGNATURE,
                projection_cols: OOF,
                projection_f32_sha256: String::new(),
            },
            rng: ChaCha20Rng::seed_from_u64(1),
            next_birth: 0,
            next_draw: 0,
            proposal_seq: 0,
            new_completed: 0,
            migration_epoch: 0,
            number_re: Regex::new("x").unwrap(),
        }
    }

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// An evaluation record: seven inner folds (the CV part) and an official_test part drawn from `test_seed`.
    fn record(cv_seed: u64, test_seed: u64) -> (Value, Program) {
        record_as(cv_seed, test_seed, cv_seed)
    }

    /// `program_id` lets two programs share one CV part exactly, so CV ties exist and a test tie-break would show.
    fn record_as(cv_seed: u64, test_seed: u64, program_id: u64) -> (Value, Program) {
        let mut r = Lcg(cv_seed);
        let scale = 0.5 + r.f();
        let folds: Vec<Value> = (0..7).map(|_| {
            let targets: Vec<f64> = (0..30).map(|_| r.f()).collect();
            let preds: Vec<f64> = targets.iter().map(|t| t + (r.f() - 0.5) * scale * 0.2).collect();
            json!({"metrics": {"rmse": 0.05 + r.f() * 0.1 * scale}, "baseline_metrics": {"rmse": 0.12 + r.f() * 0.02}, "predictions": preds, "targets": targets})
        }).collect();
        let mut q = Lcg(test_seed);
        let targets: Vec<f64> = (0..TEST).map(|_| q.f()).collect();
        let preds: Vec<f64> = targets.iter().map(|t| t + (q.f() - 0.5)).collect();
        let rec = json!({"schema":"jope.s1-evaluation/v1","status":"completed","inner_cv": folds, "official_test": {"metrics": {"rmse": 0.01 + q.f()}, "baseline_metrics": {"rmse": 0.2 + q.f() * 0.1},
                         "predictions": preds, "targets": targets}});
        let dsl = format!("(pipe X (ridge lambda_frac={}))", program_id);
        (
            rec,
            Program {
                program_sha256: format!("{:064x}", program_id),
                dsl,
                birth_seq: program_id,
                parents: Vec::new(),
                operator: "test".into(),
            },
        )
    }

    fn selection_view(i: &Individual) -> (String, Vec<u32>, Vec<u32>, u64, bool, u64, u64, u64) {
        (
            i.full_prediction_sha256.clone(),
            i.signature.iter().map(|v| v.to_bits()).collect(),
            i.cases.iter().map(|v| v.to_bits()).collect(),
            i.fitness.to_bits(),
            i.cv_gate_passed,
            i.cv_fold_wins,
            i.cv_geomean_rmse_ratio.to_bits(),
            i.cv_log_ratio_sd.to_bits(),
        )
    }

    /// S1 (TEST-STEER-01): scoring the same CV part with unrelated official_test parts yields identical selection
    /// inputs, and every policy orders a population identically.
    #[test]
    fn selection_is_blind_to_the_official_test() {
        let e = s1_engine();
        let score = |cv: u64, test: u64, id: u64| {
            let (r, p) = record_as(cv, test, id);
            e.score_record(&r, p, String::new()).unwrap()
        };
        // 60 distinct CV parts plus 20 programs that copy one of them exactly (CV ties)
        let ids: Vec<(u64, u64)> = (1..=60)
            .map(|i| (i, i))
            .chain((1..=20).map(|i| (i, 100 + i)))
            .collect();
        let a: Vec<Individual> = ids
            .iter()
            .map(|&(cv, id)| score(cv, 1000 + id, id))
            .collect();
        let b: Vec<Individual> = ids
            .iter()
            .map(|&(cv, id)| score(cv, 5000 + 3 * id, id))
            .collect();
        for (x, y) in a.iter().zip(&b) {
            assert_ne!(x.test_rmse.map(f64::to_bits), y.test_rmse.map(f64::to_bits));
            assert_eq!(selection_view(x), selection_view(y));
        }
        for policy in POLICIES {
            let order = |v: &[Individual]| {
                let mut v = v.to_vec();
                v.sort_by(|p, q| Engine::policy_cmp(policy, p, q));
                v.into_iter()
                    .map(|i| i.program.program_sha256)
                    .collect::<Vec<_>>()
            };
            assert_eq!(order(&a), order(&b), "{}", policy.name());
        }
    }

    #[test]
    fn missing_failed_or_malformed_test_diagnostics_cannot_censor_cv() {
        let e = s1_engine();
        let (r, p) = record(3, 9);
        let expected = e.score_record(&r, p.clone(), String::new()).unwrap();
        for diagnostic in [
            Value::Null,
            json!({"status":"failed"}),
            json!({"status":"timeout"}),
            json!({"metrics":{"rmse":null},"predictions":[],"targets":["invalid"]}),
        ] {
            let mut changed = r.clone();
            changed["official_test"] = diagnostic;
            let actual = e.score_record(&changed, p.clone(), String::new()).unwrap();
            assert_eq!(selection_view(&expected), selection_view(&actual));
            assert_eq!(actual.test_rmse, None);
            for policy in POLICIES {
                assert_eq!(
                    Engine::policy_cmp(policy, &expected, &actual),
                    Ordering::Equal
                );
            }
        }
    }

    #[test]
    fn selection_still_responds_to_the_cv_part() {
        let e = s1_engine();
        let (r, p) = record(3, 9);
        let base = e.score_record(&r, p.clone(), String::new()).unwrap();
        let mut worse = r.clone();
        for f in worse["inner_cv"].as_array_mut().unwrap() {
            let v = f["metrics"]["rmse"].as_f64().unwrap();
            f["metrics"]["rmse"] = json!(v * 1.5);
        }
        let w = e.score_record(&worse, p, String::new()).unwrap();
        assert!(w.cv_geomean_rmse_ratio > base.cv_geomean_rmse_ratio);
        assert_ne!(w.fitness.to_bits(), base.fitness.to_bits());
    }

    #[test]
    fn invalid_inner_scores_cannot_be_selected() {
        let e = s1_engine();
        let (r, p) = record(3, 9);
        for (field, value) in [("rmse", -1.0), ("baseline", 0.0), ("baseline", -1.0)] {
            let mut changed = r.clone();
            if field == "rmse" {
                changed["inner_cv"][0]["metrics"]["rmse"] = json!(value);
            } else {
                changed["inner_cv"][0]["baseline_metrics"]["rmse"] = json!(value);
            }
            assert!(e.score_record(&changed, p.clone(), String::new()).is_err());
        }
    }

    #[test]
    fn test_cases_and_test_wide_projections_are_refused() {
        let mut e = s1_engine();
        e.sample.test_case_indices = vec![0];
        let (r, p) = record(2, 4);
        assert!(e.score_record(&r, p.clone(), String::new()).is_err());
        let mut e = s1_engine();
        e.sample.projection_cols = OOF + TEST;
        assert!(e.score_record(&r, p, String::new()).is_err());
    }

    #[test]
    fn stale_resume_and_semantic_cache_are_refused() {
        let e = s1_engine();
        assert!(
            e.individual_from_scored(&json!({"schema":"jope.island-scored-evaluation/v1"}))
                .is_err()
        );
        let (record, p) = record(3, 9);
        let individual = e.score_record(&record, p.clone(), String::new()).unwrap();
        let row = json!({"context_sha256":"c", "program_sha256":p.program_sha256, "program":p,
            "full_prediction_sha256":individual.full_prediction_sha256, "semantic_signature_f32_b64":encode_f32(&individual.signature),
            "sampled_absolute_error_f32_b64":encode_f32(&individual.cases), "fitness":individual.fitness,
            "cv_gate_passed":individual.cv_gate_passed, "cv_fold_wins":individual.cv_fold_wins,
            "cv_geomean_rmse_ratio":individual.cv_geomean_rmse_ratio, "cv_mean_log_ratio":individual.cv_mean_log_ratio,
            "cv_tail_log_ratio":individual.cv_tail_log_ratio, "cv_log_ratio_sd":individual.cv_log_ratio_sd,
            "test_rmse":individual.test_rmse, "test_rmse_ratio":individual.test_rmse_ratio,
            "complexity_open_parens":individual.complexity_open_parens, "result_key":""});
        assert!(serde_json::from_value::<SemanticRow>(row.clone()).is_err());
        let mut qualified = row;
        qualified["selection_protocol"] = json!(SELECTION_PROTOCOL);
        assert!(
            Individual::from_semantic(serde_json::from_value(qualified.clone()).unwrap()).is_ok()
        );
        qualified["selection_protocol"] = json!("test-primary-v1");
        assert!(Individual::from_semantic(serde_json::from_value(qualified).unwrap()).is_err());
    }

    #[test]
    fn legacy_failed_resume_cannot_change_seen_programs() {
        let mut e = s1_engine();
        let dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../../../target/s1-history");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("resume-{}.jsonl", std::process::id()));
        fs::write(&path,serde_json::to_vec(&json!({"schema":"jope.island-scored-evaluation/v1","context_sha256":"c","status":"failed","program":{"program_sha256":"old-test-derived"}})).unwrap()).unwrap();
        assert!(e.load_resume(&path).is_err());
        assert!(e.seen.is_empty());
        assert!(validate_history(&json!({"selection_protocol":SELECTION_PROTOCOL,"context_sha256":"c","status":"failed"}),"c").is_ok());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn s1_population_and_generation_slots_match_the_owner_policy() {
        assert_eq!(POPULATION, 64);
        assert_eq!(ISLANDS * ISLAND_SIZE, 64);
        assert_eq!(MIGRATION_INTERVAL, 192);
    }

    #[test]
    fn f32_round_trip() {
        let values = vec![1.25_f32, -9.0, 0.0];
        let encoded = encode_f32(&values);
        assert_eq!(decode_f32(&encoded, values.len()).unwrap(), values);
    }

    #[test]
    fn behavior_hash_is_value_sensitive() {
        assert_ne!(f64_hash(&[1.0, 2.0]), f64_hash(&[1.0, 2.0000001]));
        assert_eq!(
            f64_hash(&[1.0, 2.0]),
            "dc91ce9a50ddc828740aa26743716897fdb2bb64f1db662fe263a59be56145ae"
        );
    }
}
