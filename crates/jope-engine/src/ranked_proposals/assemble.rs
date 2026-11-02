use super::catalog::{immutable_json, path, Manifest};
use super::prelude::*;

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
                    proposal_record_sha256: None,
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
                        proposal_record_sha256: None,
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
                    proposal_record_sha256: None,
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
                        proposal_record_sha256: None,
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
