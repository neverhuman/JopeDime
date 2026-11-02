use super::catalog::{bank, immutable_json, insert, path, Manifest, Origin, PROTOCOL};
use super::prelude::*;
use super::sampling::{grammar, load, ordered, sample, scoring_parity, values};

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
    run_admitted(root, manifest_path, rank, world)
}

// Private post-admission implementation. Public callers must cross the hold above.
pub(super) fn run_admitted(
    root: &Path,
    manifest_path: &Path,
    rank: usize,
    world: usize,
) -> Result<()> {
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
