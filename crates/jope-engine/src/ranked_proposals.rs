//! Isolated, manifest-bound seed experiment. No champion store or GP outcomes
//! are inputs to pure predictions; all six controls are frozen together.
use crate::{
    corpus::{ContextRow, Index},
    grammar::{State, Vocab},
    manifest::{file_hash, hash, read_hashed, read_json, write_json, SEED},
    neural::{Bf16, Config, Policy},
    proposal_ranking::{score_pool, Bindings, Options, Ranking},
    training::mix,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    time::Instant,
};
use tch::{nn, Device, Kind, Tensor};

pub const PROTOCOL: &str = "larger_exact_pool_training_context_v1";

#[derive(Clone, Deserialize)]
pub struct Artifact {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Deserialize)]
pub struct View {
    pub dataset: String,
    pub family: String,
    pub cut: usize,
    pub prepared: PathBuf,
    pub prepared_sha256: String,
    pub context_path: PathBuf,
    pub context_sha256: String,
}
#[derive(Deserialize)]
pub struct Manifest {
    pub schema: String,
    pub output: PathBuf,
    pub corpus: PathBuf,
    pub checkpoint: PathBuf,
    pub earlier_checkpoint: PathBuf,
    pub views: Vec<View>,
    pub artifacts: Vec<Artifact>,
    #[serde(default)]
    pub champion_snapshot: Option<PathBuf>,
    #[serde(default = "default_seed")]
    pub seed: u64,
}
fn default_seed() -> u64 {
    20260908
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BankEntry {
    pub program: String,
    pub program_sha256: String,
    pub family: String,
    pub dataset: String,
    pub cut: usize,
    pub record: usize,
    pub token_count: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Origin {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bank_record: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_family: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draw: Option<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub program: String,
    pub program_sha256: String,
    pub origins: Vec<Origin>,
}

fn path(root: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}
fn immutable_json(p: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if p.exists() {
        ensure!(
            fs::read(p)? == bytes,
            "frozen artifact changed: {}",
            p.display()
        );
    } else {
        crate::manifest::atomic_write(p, &bytes)?;
    }
    Ok(())
}

/// Exact champions are already first in each immutable corpus group. Round
/// robin by family, with each family's cut groups sorted by dataset/cut/record.
/// No parameter projection and no old reward or RMSE is transferred.
pub fn round_robin_bank(mut entries: Vec<BankEntry>, cap: usize) -> Vec<BankEntry> {
    entries.sort_by(|a, b| {
        (&a.family, &a.dataset, a.cut, a.record).cmp(&(&b.family, &b.dataset, b.cut, b.record))
    });
    let mut families = BTreeMap::<String, VecDeque<BankEntry>>::new();
    for entry in entries {
        families
            .entry(entry.family.clone())
            .or_default()
            .push_back(entry);
    }
    let mut seen = BTreeSet::new();
    let mut bank = Vec::new();
    while bank.len() < cap {
        let mut advanced = false;
        for rows in families.values_mut() {
            while let Some(entry) = rows.pop_front() {
                advanced = true;
                if seen.insert(entry.program.clone()) {
                    bank.push(entry);
                    break;
                }
            }
            if bank.len() == cap {
                break;
            }
        }
        if !advanced {
            break;
        }
    }
    bank
}

fn bank(index: &Index, corpus: &Path, vocab: &Vocab) -> Result<(Vec<BankEntry>, Value)> {
    let mut wanted = BTreeMap::<usize, &ContextRow>::new();
    for group in &index.groups {
        if group.partition != "meta_train" {
            continue;
        }
        let context = index
            .contexts
            .get(group.context)
            .context("bank context absent")?;
        ensure!(
            context.partition == "meta_train" && context.family == group.family,
            "bank partition mismatch"
        );
        wanted.insert(
            *group.records.first().context("empty training group")?,
            context,
        );
    }
    #[derive(Deserialize)]
    struct Label {
        record: usize,
        context: usize,
        program: String,
    }
    let mut entries = Vec::new();
    let mut rejected = Vec::new();
    for line in BufReader::new(fs::File::open(corpus.join("labels.jsonl"))?).lines() {
        // Scores are deliberately not deserialized. Only meta-training group
        // champions become candidates; current validation teachers are ignored.
        let label: Label = serde_json::from_str(&line?)?;
        let Some(c) = wanted.remove(&label.record) else {
            continue;
        };
        ensure!(
            index
                .contexts
                .get(label.context)
                .is_some_and(|x| x.dataset == c.dataset && x.cut == c.cut && x.family == c.family),
            "bank record context mismatch"
        );
        match vocab.encode(&label.program).and_then(|t| {
            feat_math::GenomeIr::parse(&label.program)?;
            Ok(t)
        }) {
            Ok(tokens) => entries.push(BankEntry {
                program_sha256: hash(label.program.as_bytes()),
                program: label.program,
                family: c.family.clone(),
                dataset: c.dataset.clone(),
                cut: c.cut,
                record: label.record,
                token_count: tokens.len(),
            }),
            Err(e) => rejected.push(json!({"record":label.record,"reason":format!("{e:#}")})),
        }
    }
    ensure!(wanted.is_empty(), "training champion labels absent");
    let exact = entries.len();
    let result = round_robin_bank(entries, 1024);
    Ok((
        result,
        json!({"exact_group_champions":exact,"rejected":rejected,"selection":"deterministic family round robin; groups ordered dataset/cut/record; unique exact programs","parameter_projection":false,"scores_transferred":false}),
    ))
}

fn insert(
    pool: &mut Vec<Candidate>,
    lookup: &mut BTreeMap<String, usize>,
    program: String,
    origin: Origin,
) {
    if let Some(&i) = lookup.get(&program) {
        pool[i].origins.push(origin);
    } else {
        lookup.insert(program.clone(), pool.len());
        pool.push(Candidate {
            program_sha256: hash(program.as_bytes()),
            program,
            origins: vec![origin],
        });
    }
}

pub fn grammar(vocab: &Vocab, seed: u64, ridge_first: bool) -> Result<Vec<String>> {
    let mut programs = Vec::new();
    let mut seen = BTreeSet::new();
    if ridge_first {
        let ridge = "(pipe X (ridge))".to_string();
        vocab.encode(&ridge)?;
        seen.insert(ridge.clone());
        programs.push(ridge);
    }
    for offset in 0..100000u64 {
        if programs.len() == 64 {
            return Ok(programs);
        }
        let raw = feat_math::evolve::random_canonical_pipeline(seed.wrapping_add(offset), 6);
        let Ok(program) = vocab.project_proposal(&raw) else {
            continue;
        };
        if feat_math::GenomeIr::parse(&program).is_ok() && seen.insert(program.clone()) {
            programs.push(program);
        }
    }
    anyhow::bail!("unable to produce deterministic grammar64")
}

fn sample(
    policy: &Policy,
    vocab: &Vocab,
    values: &[f32],
    device: Device,
    seed: u64,
) -> Result<(Vec<(usize, String)>, Value)> {
    tch::manual_seed(seed as i64);
    let batch = 128usize;
    let contexts = Tensor::from_slice(values)
        .view([1, 2048])
        .repeat([batch as i64, 1])
        .to_device(device);
    let task_id = vocab.task_id("regression")?;
    let mut sequences = vec![vec![1usize, vocab.engine_ids["featmath"], task_id]; batch];
    let mut states = (0..batch)
        .map(|_| {
            let mut state = State::default();
            state.advance(vocab, vocab.engine_ids["featmath"]).unwrap();
            state.advance(vocab, task_id).unwrap();
            state
        })
        .collect::<Vec<_>>();
    for _ in 3..128 {
        crate::campaign::check_deadline()?;
        if states.iter().all(State::done) {
            break;
        }
        let length = sequences[0].len();
        let tokens = Tensor::from_slice(
            &sequences
                .iter()
                .flat_map(|s| s.iter().map(|&t| t as i64))
                .collect::<Vec<_>>(),
        )
        .view([batch as i64, length as i64])
        .to_device(device);
        let mut legal = vec![0u8; batch * vocab.tokens.len()];
        for (i, s) in states.iter().enumerate() {
            for id in if s.done() { vec![0] } else { s.legal(vocab) } {
                legal[i * vocab.tokens.len() + id] = 1;
            }
        }
        let legal = Tensor::from_slice(&legal)
            .view([batch as i64, vocab.tokens.len() as i64])
            .to_kind(Kind::Bool)
            .to_device(device);
        let next = tch::no_grad(|| -> Result<Vec<i64>> {
            let _guard = matches!(device, Device::Cuda(_)).then(Bf16::enter);
            let (logits, _, _, _, _) = policy.forward(
                &contexts.narrow(1, 0, 1536),
                &contexts.narrow(1, 1536, 512),
                &tokens,
                &tokens.ne(0),
                false,
            )?;
            let logits = logits
                .select(1, length as i64)
                .to_kind(Kind::Float)
                .masked_fill(&legal.logical_not(), f64::NEG_INFINITY);
            Ok(Vec::<i64>::try_from(
                logits
                    .softmax(-1, Kind::Float)
                    .multinomial(1, false)
                    .to_device(Device::Cpu)
                    .flatten(0, -1),
            )?)
        })?;
        for i in 0..batch {
            if !states[i].done() {
                states[i].advance(vocab, next[i] as usize)?;
            }
            sequences[i].push(next[i] as usize);
        }
    }
    let mut valid = Vec::new();
    let mut invalid = 0;
    for (draw, (mut sequence, state)) in sequences.into_iter().zip(states).enumerate() {
        if !state.done() {
            invalid += 1;
            continue;
        }
        sequence.truncate(
            sequence
                .iter()
                .position(|&t| t == 2)
                .context("EOS absent")?
                + 1,
        );
        let Ok(program) = vocab.decode(&sequence) else {
            invalid += 1;
            continue;
        };
        if feat_math::GenomeIr::parse(&program).is_err() || vocab.encode(&program).is_err() {
            invalid += 1;
            continue;
        }
        valid.push((draw, program));
    }
    let unique = valid.iter().map(|(_, p)| p).collect::<BTreeSet<_>>().len();
    let receipt = json!({"attempted":batch,"valid":valid.len(),"invalid":invalid,"unique":unique,"duplicates":valid.len()-unique,"sampling_seed":seed,"generation_mask":"existing regression grammar mask unchanged; exact bank may include training-admitted heads outside sampling mask"});
    Ok((valid, receipt))
}

fn values(root: &Path, p: &Path, sha: &str) -> Result<Vec<f32>> {
    let bytes = fs::read(path(root, p))?;
    ensure!(
        hash(&bytes) == sha && bytes.len() == 2048 * 4,
        "saved context identity changed"
    );
    let values = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite context");
    Ok(values)
}

fn load(
    root: &Path,
    checkpoint: &Path,
    vocab_sha: &str,
    device: Device,
) -> Result<(nn::VarStore, Policy, String, Value)> {
    let checkpoint = path(root, checkpoint);
    let state: Value = read_json(&checkpoint.join("state.json"))?;
    ensure!(
        state["model_sha256"] == file_hash(&checkpoint.join("model.ot"))?
            && state["config_sha256"] == file_hash(&checkpoint.join("config.json"))?
            && state["vocabulary_sha256"] == vocab_sha
            && state["autocast_cache_lifetime"] == "scoped_v2",
        "frozen checkpoint identity mismatch"
    );
    let config: Config = read_json(&checkpoint.join("config.json"))?;
    let mut vs = nn::VarStore::new(device);
    let model = Policy::new(vs.root(), config)?;
    vs.load(checkpoint.join("model.ot"))?;
    Ok((vs, model, file_hash(&checkpoint.join("state.json"))?, state))
}

fn ordered(ranking: &Ranking) -> Vec<String> {
    ranking.ranked.iter().map(|p| p.program.clone()).collect()
}

fn scoring_parity(
    policy: &Policy,
    vocab: &Vocab,
    context: &[f32],
    programs: &[String],
    device: Device,
) -> Result<Value> {
    let rows = programs
        .iter()
        .map(|p| vocab.encode(p))
        .collect::<Result<Vec<_>>>()?;
    let length = rows
        .iter()
        .map(Vec::len)
        .max()
        .context("empty parity pool")?;
    let mut ids = vec![0i64; rows.len() * length];
    for (i, row) in rows.iter().enumerate() {
        for (j, &id) in row.iter().enumerate() {
            ids[i * length + j] = id as i64;
        }
    }
    let tokens = Tensor::from_slice(&ids)
        .view([rows.len() as i64, length as i64])
        .to_device(device);
    let contexts = Tensor::from_slice(context)
        .view([1, 2048])
        .repeat([rows.len() as i64, 1])
        .to_device(device);
    let (reward_error, runtime_error, order_matches) =
        tch::no_grad(|| -> Result<(f64, f64, bool)> {
            let _guard = matches!(device, Device::Cuda(_)).then(Bf16::enter);
            let dataset = contexts.narrow(1, 0, 1536);
            let target = contexts.narrow(1, 1536, 512);
            let present = tokens.ne(0);
            let (_, full_reward, full_runtime, _, _) =
                policy.forward(&dataset, &target, &tokens, &present, false)?;
            let (fast_reward, fast_runtime) =
                policy.reward_runtime(&dataset, &target, &tokens, &present)?;
            let reward_error = (&full_reward - &fast_reward).abs().max().double_value(&[]);
            let runtime_error = (&full_runtime - &fast_runtime)
                .abs()
                .max()
                .double_value(&[]);
            let order = |reward: &Tensor| -> Result<Vec<usize>> {
                let quality = Vec::<f32>::try_from(
                    reward
                        .mean_dim(&[1i64][..], false, Kind::Float)
                        .to_device(Device::Cpu),
                )?;
                let mut indices = (0..programs.len()).collect::<Vec<_>>();
                indices.sort_by(|&a, &b| {
                    quality[b]
                        .total_cmp(&quality[a])
                        .then(hash(programs[a].as_bytes()).cmp(&hash(programs[b].as_bytes())))
                });
                Ok(indices)
            };
            Ok((
                reward_error,
                runtime_error,
                order(&full_reward)? == order(&fast_reward)?,
            ))
        })?;
    ensure!(reward_error<=2e-6 && runtime_error<=2e-6 && order_matches,"decoder-free checkpoint scoring parity failed: reward={reward_error}, runtime={runtime_error}, ordering={order_matches}");
    Ok(
        json!({"reference":"same frozen checkpoint full Policy.forward","programs":programs.len(),"reward_max_abs_error":reward_error,"runtime_max_abs_error":runtime_error,"tolerance":2e-6,"selected_order_identical":order_matches}),
    )
}

/// Separate CPU-only phase. Verify every view's pure ranking is durable before
/// loading any retained teacher, then use the same 10/6 tail in all GP arms.
pub fn assemble(root: &Path, manifest_path: &Path) -> Result<()> {
    use crate::{
        gp::{SeedCandidate, SeedPopulation},
        prepare::Prepared,
        proposal_population,
    };
    let canonical_root = root.canonicalize()?;
    let root = canonical_root.as_path();
    let manifest_path = path(root, manifest_path);
    let m: Manifest = read_json(&manifest_path)?;
    ensure!(
        matches!(
            m.schema.as_str(),
            "parallel.seed_experiment.v1" | "parallel.seed_experiment.v2"
        ),
        "invalid seed experiment manifest"
    );
    let manifest_sha = file_hash(&manifest_path)?;
    let output = path(root, &m.output).canonicalize()?;
    ensure!(
        output.starts_with(root.canonicalize()?)
            && !output.starts_with(root.join("runs/champions")),
        "assembly output must be isolated"
    );
    let mut rankings = Vec::new();
    for view in &m.views {
        let destination = output
            .join("proposals")
            .join(&view.dataset)
            .join(format!("cut-{}", view.cut));
        let complete: Value = read_json(&destination.join("complete.json"))
            .context("all pure rankings must complete before teacher assembly")?;
        let ranking: Value = read_json(&destination.join("rankings.json"))?;
        ensure!(
            complete["manifest_sha256"] == manifest_sha
                && complete["rankings_sha256"] == file_hash(&destination.join("rankings.json"))?
                && ranking["manifest_sha256"] == manifest_sha
                && ranking["retained_teachers_consulted"] == false,
            "pure ranking freeze changed"
        );
        rankings.push((ranking, complete["rankings_sha256"].clone()));
    }
    let corpus = path(root, &m.corpus);
    for artifact in &m.artifacts {
        ensure!(
            file_hash(&path(root, &artifact.path))? == artifact.sha256,
            "frozen assembly input changed"
        );
    }
    // W1 route R8: the vocabulary is read once; its digest is of the parsed bytes.
    let (vocab_bytes, vocab_sha) =
        read_hashed(&corpus.join("vocabulary.json"), "corpus vocabulary")?;
    let vocab: Vocab = serde_json::from_slice(&vocab_bytes)?;
    let index: Index = read_json(&corpus.join("index.json"))?;
    let champions = if let Some(snapshot) = &m.champion_snapshot {
        let snapshot = path(root, snapshot);
        ensure!(
            m.artifacts.iter().any(|a| path(root, &a.path) == snapshot),
            "champion snapshot absent from frozen artifacts"
        );
        let snapshot: crate::champions::Snapshot = read_json(&snapshot)?;
        ensure!(
            snapshot.schema == "parallel.champion_snapshot.v1"
                && snapshot.vocabulary_sha256 == vocab_sha,
            "champion snapshot vocabulary mismatch"
        );
        snapshot.entries
    } else {
        Vec::new()
    };
    let mut wanted = BTreeSet::new();
    for view in &m.views {
        if let Some((context, _)) = index
            .contexts
            .iter()
            .enumerate()
            .find(|(_, c)| c.dataset == view.dataset && c.cut == view.cut)
        {
            for group in index.groups.iter().filter(|g| g.context == context) {
                wanted.extend(group.records.iter().take(32).copied());
            }
        }
    }
    let mut labels = BTreeMap::new();
    #[derive(Deserialize)]
    struct Label {
        record: usize,
        program: String,
    }
    for line in BufReader::new(fs::File::open(corpus.join("labels.jsonl"))?).lines() {
        let label: Label = serde_json::from_str(&line?)?;
        if wanted.contains(&label.record) {
            labels.insert(label.record, label.program);
        }
    }
    let mut receipts = Vec::new();
    for (j, (view, (ranking, ranking_sha))) in m.views.iter().zip(rankings).enumerate() {
        let prepared_path = path(root, &view.prepared).join("prepared.json");
        ensure!(
            file_hash(&prepared_path)? == view.prepared_sha256,
            "assembly prepared identity changed"
        );
        let prepared: Prepared = read_json(&prepared_path)?;
        let cut = prepared
            .cuts
            .iter()
            .find(|c| c.index == view.cut)
            .context("assembly cut absent")?;
        let mut tail = Vec::new();
        let mut seen = BTreeSet::new();
        for champion in champions
            .iter()
            .filter(|c| c.identity.dataset == view.dataset && c.identity.cut == view.cut)
        {
            if tail.len() == 10 {
                break;
            }
            champion.validate()?;
            ensure!(
                champion.identity.family == view.family
                    && champion.identity.prepared_sha256 == view.prepared_sha256
                    && champion.identity.training_sha256 == cut.training_sha256
                    && champion.identity.validation_sha256 == cut.validation_sha256
                    && champion.identity.vocabulary_sha256 == vocab_sha,
                "retained champion cut mismatch"
            );
            vocab.encode(champion.genome())?;
            if seen.insert(champion.genome().to_string()) {
                tail.push(SeedCandidate {
                    genome: champion.genome().into(),
                    origin: "incumbent_cut_teacher".into(),
                    policy_logprob: None,
                });
            }
        }
        if let Some((context, _)) = index
            .contexts
            .iter()
            .enumerate()
            .find(|(_, c)| c.dataset == view.dataset && c.cut == view.cut)
        {
            for id in index
                .groups
                .iter()
                .filter(|g| g.context == context)
                .flat_map(|g| g.records.iter().take(32))
            {
                if tail.len() == 10 {
                    break;
                }
                let genome = labels.get(id).context("retained label absent")?;
                vocab.encode(genome)?;
                if seen.insert(genome.clone()) {
                    tail.push(SeedCandidate {
                        genome: genome.clone(),
                        origin: "retained_cut_teacher".into(),
                        policy_logprob: None,
                    });
                }
            }
        }
        let tail_seed = mix(m.seed ^ j as u64 ^ 0x7461696c);
        for offset in 0..100000u64 {
            if tail.len() == 16 {
                break;
            }
            let raw =
                feat_math::evolve::random_canonical_pipeline(tail_seed.wrapping_add(offset), 6);
            let Ok(genome) = vocab.project_proposal(&raw) else {
                continue;
            };
            if seen.insert(genome.clone()) {
                tail.push(SeedCandidate {
                    genome,
                    origin: if tail.len() < 10 {
                        "retained_slot_fallback"
                    } else {
                        "random_exploration"
                    }
                    .into(),
                    policy_logprob: None,
                });
            }
        }
        ensure!(tail.len() == 16, "cannot fill shared tail");
        let tail_sha = hash(&serde_json::to_vec(&tail)?);
        let overlap_seed = mix(m.seed ^ j as u64 ^ 0x67705f6f7665726c);
        for arm in ["current", "larger_pool", "generic"] {
            let programs = ranking["rankings"][arm]
                .as_array()
                .context("frozen arm absent")?;
            ensure!(programs.len() >= 64, "frozen arm lacks64");
            let pure = programs
                .iter()
                .take(64)
                .map(|p| {
                    Ok(SeedCandidate {
                        genome: p.as_str().context("program text absent")?.into(),
                        origin: "neural_policy".into(),
                        policy_logprob: None,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let mut assembly = proposal_population::assemble(&pure, &tail, &vocab, overlap_seed)?;
            if arm == "generic" {
                for c in &mut assembly.neural_slots {
                    c.origin = if c.origin == "neural_overlap_fallback" {
                        "generic_overlap_fallback"
                    } else {
                        "generic_grammar"
                    }
                    .into();
                }
            }
            let mut candidates = assembly.neural_slots.clone();
            candidates.extend(tail.clone());
            let population = SeedPopulation {
                schema: "parallel.seed_population.v1".into(),
                prepared_sha256: view.prepared_sha256.clone(),
                cut: view.cut,
                training_sha256: cut.training_sha256.clone(),
                context_sha256: view.context_sha256.clone(),
                checkpoint_sha256: ranking["checkpoint_sha256"]
                    .as_str()
                    .context("checkpoint hash absent")?
                    .into(),
                candidates,
            };
            let destination = output
                .join("downstream")
                .join(&view.dataset)
                .join(format!("cut-{}", view.cut))
                .join(arm);
            let seeds = destination.join("seeds.json");
            immutable_json(&seeds, &population)?;
            let receipt = json!({"schema":"parallel.ranked_gp_assembly.v1","manifest_sha256":manifest_sha,"rankings_sha256":ranking_sha,"arm":arm,"shared_tail_sha256":tail_sha,"shared_tail":tail,"overlap_seed":overlap_seed,"assembly":assembly,"seeds_sha256":file_hash(&seeds)?,"teacher_scores_transferred":false,"all_pure_rankings_frozen_before_teacher_access":true});
            immutable_json(&destination.join("assembly.json"), &receipt)?;
            receipts.push(json!({"dataset":view.dataset,"family":view.family,"cut":view.cut,"arm":arm,"seeds":seeds,"seeds_sha256":file_hash(&seeds)?,"shared_tail_sha256":tail_sha}));
        }
    }
    immutable_json(
        &output.join("assemblies-complete.json"),
        &json!({"schema":"parallel.ranked_gp_assemblies_complete.v1","manifest_sha256":manifest_sha,"assemblies":receipts}),
    )?;
    Ok(())
}

/// The device a ranked-proposal worker uses: CPU only when `JOPE_RANKED_DEVICE`
/// is exactly `cpu`, otherwise the CUDA device its binding owns (W7-T03
/// follow-up; formerly `Device::Cuda(rank)`).
pub fn worker_device(
    binding: &crate::training::topology::Binding,
    requested: Option<&str>,
) -> Device {
    if requested == Some("cpu") {
        Device::Cpu
    } else {
        binding.cuda_device()
    }
}

pub fn run(root: &Path, manifest_path: &Path, rank: usize, world: usize) -> Result<()> {
    crate::task_contract::require_qualified_gpu_experiment("propose-ranked")?;
    let canonical_root = root.canonicalize()?;
    let root = canonical_root.as_path();
    ensure!(
        world > 0 && world <= 8 && rank < world,
        "invalid proposal topology"
    );
    // No caller delivers an assignment yet: the labelled legacy mapping
    // applies and is recorded in the completion receipt.
    let binding =
        crate::training::topology::resolve(None, rank, world, crate::neural::local_devices)?;
    tch::set_num_threads(2);
    tch::set_num_interop_threads(1);
    tch::jit::set_graph_executor_optimize(false);
    let manifest_path = path(root, manifest_path);
    let m: Manifest = read_json(&manifest_path)?;
    ensure!(
        matches!(
            m.schema.as_str(),
            "parallel.seed_experiment.v1" | "parallel.seed_experiment.v2"
        ) && !m.views.is_empty()
            && !m.artifacts.is_empty(),
        "invalid seed experiment manifest"
    );
    let manifest_sha = file_hash(&manifest_path)?;
    let output = path(root, &m.output);
    fs::create_dir_all(&output)?;
    let output = output.canonicalize()?;
    ensure!(
        output.starts_with(root.canonicalize()?)
            && output != root.canonicalize()?
            && !output.starts_with(root.join("runs/champions")),
        "experiment output must be isolated inside workspace"
    );
    let corpus = path(root, &m.corpus);
    ensure!(
        !corpus.starts_with(output.join("proposals"))
            && !corpus.starts_with(output.join("downstream")),
        "proposal outputs overlap frozen corpus"
    );
    let mut frozen = BTreeMap::new();
    for artifact in &m.artifacts {
        let p = path(root, &artifact.path);
        ensure!(
            file_hash(&p)? == artifact.sha256,
            "frozen input changed: {}",
            p.display()
        );
        frozen.insert(p.canonicalize()?, artifact.sha256.clone());
    }
    for directory in [
        &corpus,
        &path(root, &m.checkpoint),
        &path(root, &m.earlier_checkpoint),
    ] {
        let names: &[&str] = if directory == &corpus {
            &[
                "index.json",
                "vocabulary.json",
                "provenance.json",
                "labels.jsonl",
            ]
        } else {
            &["state.json", "config.json", "model.ot"]
        };
        for name in names {
            ensure!(
                frozen.contains_key(&directory.join(name).canonicalize()?),
                "required input absent from frozen manifest: {}",
                directory.join(name).display()
            );
        }
    }
    let index: Index = read_json(&corpus.join("index.json"))?;
    // W1 route R8: the vocabulary is read once; its digest is of the parsed bytes.
    let (vocab_bytes, vocab_sha) =
        read_hashed(&corpus.join("vocabulary.json"), "corpus vocabulary")?;
    let vocab: Vocab = serde_json::from_slice(&vocab_bytes)?;
    ensure!(
        index.vocabulary_sha256 == vocab_sha,
        "corpus vocabulary mismatch"
    );
    let provenance: Value = read_json(&corpus.join("provenance.json"))?;
    ensure!(
        provenance["index_sha256"] == file_hash(&corpus.join("index.json"))?
            && provenance["labels_sha256"] == file_hash(&corpus.join("labels.jsonl"))?,
        "corpus provenance mismatch"
    );
    let started = Instant::now();
    let (bank, bank_selection) = bank(&index, &corpus, &vocab)?;
    let bank_value = json!({"schema":"parallel.exact_champion_bank.v1","manifest_sha256":manifest_sha,"corpus_sha256":file_hash(&corpus.join("index.json"))?,"vocabulary_sha256":vocab_sha,"count":bank.len(),"selection":bank_selection,"entries":bank,"partition":"meta_train","query_source_family_excluded":true});
    let bank_path = output.join("frozen-bank.json");
    immutable_json(&bank_path, &bank_value)?;
    let bank_sha = file_hash(&bank_path)?;
    let device = worker_device(
        &binding,
        std::env::var("JOPE_RANKED_DEVICE").ok().as_deref(),
    );
    let (_vs, model, checkpoint_sha, state) = load(root, &m.checkpoint, &vocab_sha, device)?;
    ensure!(
        state["corpus_sha256"] == file_hash(&corpus.join("index.json"))?,
        "selected checkpoint corpus differs from frozen corpus"
    );
    let (_earlier_vs, earlier, earlier_sha, _) =
        load(root, &m.earlier_checkpoint, &vocab_sha, device)?;
    let donors = index
        .contexts
        .iter()
        .filter(|c| c.partition == "meta_train")
        .collect::<Vec<_>>();
    ensure!(
        !donors.is_empty(),
        "no training contexts for shuffle control"
    );
    let mut completed = Vec::new();
    for (j, view) in m.views.iter().enumerate().filter(|(j, _)| binding.owns(*j)) {
        crate::campaign::check_deadline()?;
        ensure!(
            !view.dataset.is_empty() && !view.dataset.contains('/') && view.dataset != "..",
            "unsafe dataset ID"
        );
        let c = index
            .contexts
            .iter()
            .find(|c| c.dataset == view.dataset && c.cut == view.cut)
            .context("query absent from frozen corpus contexts")?;
        ensure!(
            c.partition == "meta_validation"
                && c.family == view.family
                && c.sha256 == view.context_sha256
                && c.prepared_sha256 == view.prepared_sha256
                && path(root, &c.path).canonicalize()?
                    == path(root, &view.context_path).canonicalize()?
                && path(root, &c.prepared).canonicalize()?
                    == path(root, &view.prepared).canonicalize()?,
            "query differs from frozen validation context"
        );
        ensure!(
            frozen.get(&path(root, &view.context_path).canonicalize()?)
                == Some(&view.context_sha256)
                && frozen.get(
                    &path(root, &view.prepared)
                        .join("prepared.json")
                        .canonicalize()?
                ) == Some(&view.prepared_sha256),
            "query context or prepared identity absent from frozen artifacts"
        );
        let destination = output
            .join("proposals")
            .join(&view.dataset)
            .join(format!("cut-{}", view.cut));
        let ranking_path = destination.join("rankings.json");
        let complete_path = destination.join("complete.json");
        if complete_path.exists() {
            let complete: Value = read_json(&complete_path)?;
            ensure!(
                complete["manifest_sha256"] == manifest_sha
                    && complete["rankings_sha256"] == file_hash(&ranking_path)?
                    && complete["bank_sha256"] == bank_sha,
                "saved ranked proposal changed"
            );
            completed.push(json!({"dataset":view.dataset,"cut":view.cut,"path":ranking_path,"sha256":file_hash(&ranking_path)?}));
            continue;
        }
        let begun = Instant::now();
        let context = values(root, &view.context_path, &view.context_sha256)?;
        let donor = donors
            .iter()
            .filter(|d| d.family != view.family)
            .min_by_key(|d| hash(format!("{}:{}:{}:{}", m.seed, j, d.dataset, d.cut).as_bytes()))
            .context("no distinct-family shuffle donor")?;
        let shuffled = values(root, &donor.path, &donor.sha256)?;
        let sampling_seed = mix(m.seed ^ j as u64);
        let (draws, sampling) = sample(&model, &vocab, &context, device, sampling_seed)?;
        let generation_seconds = begun.elapsed().as_secs_f64();
        let mut pool = Vec::new();
        let mut lookup = BTreeMap::new();
        for (draw, program) in &draws {
            insert(
                &mut pool,
                &mut lookup,
                program.clone(),
                Origin {
                    kind: "neural_policy".into(),
                    bank_record: None,
                    source_family: None,
                    draw: Some(*draw),
                },
            );
        }
        let unique_neural = pool.len();
        let mut baseline = pool
            .iter()
            .take(64)
            .map(|c| c.program.clone())
            .collect::<Vec<_>>();
        let mut baseline_seen = pool
            .iter()
            .map(|c| c.program.clone())
            .collect::<BTreeSet<_>>();
        let fallback_seed = mix(SEED ^ j as u64 ^ 0x707572655f66696c);
        for offset in 0..10000u64 {
            if baseline.len() == 64 {
                break;
            }
            let raw =
                feat_math::evolve::random_canonical_pipeline(fallback_seed.wrapping_add(offset), 6);
            let Ok(program) = vocab.project_proposal(&raw) else {
                continue;
            };
            if baseline_seen.insert(program.clone()) {
                baseline.push(program.clone());
                insert(
                    &mut pool,
                    &mut lookup,
                    program,
                    Origin {
                        kind: "neural_rejection_fallback".into(),
                        bank_record: None,
                        source_family: None,
                        draw: None,
                    },
                );
            }
        }
        ensure!(baseline.len() == 64, "baseline could not fill64");
        for entry in bank.iter().filter(|entry| entry.family != view.family) {
            insert(
                &mut pool,
                &mut lookup,
                entry.program.clone(),
                Origin {
                    kind: "meta_training_champion".into(),
                    bank_record: Some(entry.record),
                    source_family: Some(entry.family.clone()),
                    draw: None,
                },
            );
        }
        let grammar_seed = mix(m.seed ^ j as u64 ^ 0x6772616d6d6172);
        for program in grammar(&vocab, grammar_seed, false)? {
            insert(
                &mut pool,
                &mut lookup,
                program,
                Origin {
                    kind: "deterministic_grammar".into(),
                    bank_record: None,
                    source_family: None,
                    draw: None,
                },
            );
        }
        let programs = pool.iter().map(|c| c.program.clone()).collect::<Vec<_>>();
        let bindings = Bindings {
            checkpoint_sha256: checkpoint_sha.clone(),
            conditioning_context_sha256: view.context_sha256.clone(),
            vocabulary_sha256: vocab_sha.clone(),
        };
        let options = || Options {
            device,
            batch_size: 64,
            keep: 64,
        };
        let score_started = Instant::now();
        let parity = scoring_parity(&model, &vocab, &context, &baseline, device)?;
        let current = score_pool(
            &model,
            &vocab,
            &context,
            &baseline,
            bindings.clone(),
            options(),
        )?;
        let larger = score_pool(
            &model,
            &vocab,
            &context,
            &programs,
            bindings.clone(),
            options(),
        )?;
        let earlier_ranking = score_pool(
            &earlier,
            &vocab,
            &context,
            &programs,
            Bindings {
                checkpoint_sha256: earlier_sha.clone(),
                ..bindings.clone()
            },
            options(),
        )?;
        let shuffled_ranking = score_pool(
            &model,
            &vocab,
            &shuffled,
            &programs,
            Bindings {
                conditioning_context_sha256: donor.sha256.clone(),
                ..bindings.clone()
            },
            options(),
        )?;
        let mut random = programs.clone();
        random.sort_by_key(|p| hash(format!("{}:{}:{}", m.seed, j, p).as_bytes()));
        let generic = grammar(&vocab, SEED, true)?;
        let receipt = json!({
            "schema":"parallel.seed_rankings.v1","protocol":PROTOCOL,"manifest_sha256":manifest_sha,
            "dataset":view.dataset,"family":view.family,"cut":view.cut,"prepared_sha256":view.prepared_sha256,
            "context_sha256":view.context_sha256,"checkpoint_sha256":checkpoint_sha,"earlier_checkpoint_sha256":earlier_sha,
            "vocabulary_sha256":vocab_sha,"bank_sha256":bank_sha,"candidate_pool":pool,"unique_neural":unique_neural,
            "rankings":{"current":ordered(&current),"larger_pool":ordered(&larger),"earlier_checkpoint":ordered(&earlier_ranking),"shuffled_context":ordered(&shuffled_ranking),"random":random,"generic":generic},
            "predictions":{"current":current,"larger_pool":larger,"earlier_checkpoint":earlier_ranking,"shuffled_context":shuffled_ranking},
            "sampling":sampling,"grammar_seed":grammar_seed,"fallback_seed":fallback_seed,
            "reward_runtime_parity":parity,
            "shuffle_donor":{"dataset":donor.dataset,"family":donor.family,"cut":donor.cut,"context_sha256":donor.sha256},
            "costs":{"generation_seconds":generation_seconds,"scoring_seconds":score_started.elapsed().as_secs_f64(),"total_seconds":begun.elapsed().as_secs_f64()},
            "selected_programs_per_arm":64,"training_context_only":true,"retained_teachers_consulted":false,
            "heldout_labels_read":false,"gp_outcomes_read":false,"old_bank_scores_transferred":false,
            "qualification":true,"previously_used_validation_families":true,"final_tests_opened":false
        });
        write_json(&ranking_path, &receipt)?;
        immutable_json(
            &complete_path,
            &json!({"schema":"parallel.seed_rankings_complete.v1","manifest_sha256":manifest_sha,"rankings_sha256":file_hash(&ranking_path)?,"bank_sha256":bank_sha}),
        )?;
        completed.push(json!({"dataset":view.dataset,"cut":view.cut,"path":ranking_path,"sha256":file_hash(&ranking_path)?}));
        write_json(
            &output.join(format!("rank-{rank}-progress.json")),
            &json!({"rank":rank,"world":world,"phase":"freezing seed rankings","completed":completed.len(),"seconds":started.elapsed().as_secs_f64()}),
        )?;
    }
    write_json(
        &output.join(format!("rank-{rank}-complete.json")),
        &json!({"schema":"parallel.ranked_proposals_complete.v1","manifest_sha256":manifest_sha,"bank_sha256":bank_sha,"rank":rank,"world":world,"views":completed,"seconds":started.elapsed().as_secs_f64(),"topology":binding.receipt()}),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn entry(family: &str, cut: usize, program: &str) -> BankEntry {
        BankEntry {
            family: family.into(),
            dataset: format!("dataset_{family}"),
            cut,
            record: cut,
            program: program.into(),
            program_sha256: hash(program.as_bytes()),
            token_count: 4,
        }
    }
    #[test]
    fn bank_round_robin_is_deterministic_balanced_and_exact() {
        let entries = vec![
            entry("b", 2, "b2"),
            entry("a", 9, "shared"),
            entry("a", 1, "a1"),
            entry("b", 1, "shared"),
            entry("a", 3, "a3"),
        ];
        let bank = round_robin_bank(entries.clone(), 4);
        assert_eq!(
            bank.iter().map(|e| e.program.as_str()).collect::<Vec<_>>(),
            vec!["a1", "shared", "a3", "b2"]
        );
        assert_eq!(
            serde_json::to_vec(&bank).unwrap(),
            serde_json::to_vec(&round_robin_bank(entries.into_iter().rev().collect(), 4)).unwrap()
        );
        assert!(bank
            .iter()
            .filter(|e| e.family != "a")
            .all(|e| e.family == "b"));
    }
    #[test]
    fn pool_keeps_all128_and_accumulates_origins_before_ranking() {
        let mut pool = Vec::new();
        let mut lookup = BTreeMap::new();
        for draw in 0..128 {
            insert(
                &mut pool,
                &mut lookup,
                format!("program{draw}"),
                Origin {
                    kind: "neural_policy".into(),
                    draw: Some(draw),
                    bank_record: None,
                    source_family: None,
                },
            );
        }
        insert(
            &mut pool,
            &mut lookup,
            "program100".into(),
            Origin {
                kind: "meta_training_champion".into(),
                draw: None,
                bank_record: Some(7),
                source_family: Some("train".into()),
            },
        );
        assert_eq!(pool.len(), 128);
        assert_eq!(pool[100].origins.len(), 2);
    }
    #[test]
    fn frozen_vocabulary_accepts_gcv_without_snapping() {
        let vocab: Vocab =
            read_json(&Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"))
                .unwrap();
        assert!(vocab.encode("(pipe X (ridge_gcv))").is_ok());
        assert!(vocab
            .encode("(pipe X (rff_cos n=17 gamma=0.513 seed=7) (ridge))")
            .is_err());
        let generic = grammar(&vocab, SEED, true).unwrap();
        assert_eq!(generic.len(), 64);
        assert_eq!(generic[0], "(pipe X (ridge))");
        assert_eq!(generic.iter().collect::<BTreeSet<_>>().len(), 64);
    }
}
