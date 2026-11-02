use super::prelude::*;

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
pub(crate) fn default_seed() -> u64 {
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

pub(crate) fn path(root: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}
pub(crate) fn immutable_json(p: &Path, value: &impl Serialize) -> Result<()> {
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

pub(crate) fn bank(index: &Index, corpus: &Path, vocab: &Vocab) -> Result<(Vec<BankEntry>, Value)> {
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

pub(crate) fn insert(
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
