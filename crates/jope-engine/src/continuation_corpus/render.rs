use super::*;

/// Actual existing strict validators decide every source outcome before deriving a new target.
/// This function cannot create an admission policy or change original fitted program identity.
pub fn build(root: &Path, manifest: &Artifact, output: &Path) -> Result<()> {
    crate::manifest::relative(output)?;
    let output = root.join(output);
    let mut files = render(root, manifest)?;
    let index = files
        .remove("index.json")
        .context("rendered index absent")?;
    fs::create_dir_all(&output)?;
    for (name, bytes) in files {
        immutable(&output.join(name), &bytes)?;
    }
    immutable(&output.join("index.json"), &index)
}
/// Pure deterministic rendering also drives independent native Graph verification.
pub fn render(root: &Path, manifest: &Artifact) -> Result<BTreeMap<String, Vec<u8>>> {
    let set: TargetSet = native_genome::load(root, manifest)?;
    ensure!(
        set.schema == "jope.continuation_target_set.v1"
            && !set.run.is_empty()
            && !set.targets.is_empty()
            && set.targets.len() <= 100000
            && !set.training_arms.is_empty(),
        "target-set bounds/schema"
    );
    let parent_index: Value = native_genome::load_corpus_index(root, &set.parent_corpus)?;
    let previous_set = set
        .parent_corpus
        .path
        .parent()
        .context("parent corpus directory")?
        .join("target-set.json");
    if parent_index["label_policy"] == "strict_continuation_midrank_v1" {
        ensure!(
            crate::manifest::file_hash(&resolve(root, &previous_set)?)?
                == parent_index["target_set_sha256"],
            "previous continuation target set changed"
        );
        let previous: TargetSet = crate::manifest::read_json(&resolve(root, &previous_set)?)?;
        ensure!(
            previous.run == set.run
                && previous.generation < set.generation
                && previous.vocabulary == set.vocabulary
                && previous.frozen_scientific_policy == set.frozen_scientific_policy
                && previous.training_arms == set.training_arms
                && previous.qualification == set.qualification,
            "continuation corpus changed prior objective or run"
        );
        let identities: BTreeSet<_> = set.targets.iter().map(|a| (&a.path, &a.sha256)).collect();
        ensure!(
            previous
                .targets
                .iter()
                .all(|a| identities.contains(&(&a.path, &a.sha256))),
            "previous continuation targets omitted"
        );
    }

    let scientific: Value = native_genome::load(root, &set.frozen_scientific_policy)?;
    ensure!(
        scientific["schema"] == "jope.native_learning_policy.v1"
            && scientific["training_target"] == "continuation_only_v1"
            && scientific["training_arms"] == serde_json::to_value(&set.training_arms)?,
        "continuation objective not explicitly frozen"
    );
    let vocab_bytes = native_genome::checked(root, &set.vocabulary, 16 * 1024 * 1024)?;
    let vocab: crate::grammar::Vocab = serde_json::from_slice(&vocab_bytes)?;
    let mut groups = BTreeMap::<String, Vec<(Artifact, ContinuationTarget, Candidate)>>::new();
    let excluded: BTreeSet<String> =
        serde_json::from_value(scientific["excluded_families"].clone())?;
    let mut budgets = BTreeMap::new();
    let mut seen_candidates = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut refused_targets = Vec::new();
    for reference in &set.targets {
        ensure!(
            seen.insert(reference.sha256.clone()),
            "duplicate target identity"
        );
        let target: ContinuationTarget = native_genome::load(root, reference)?;
        let verified = native_genome::target(root, &target.continuation)?;
        ensure!(
            serde_json::to_value(&target)? == serde_json::to_value(&verified)?,
            "continuation target differs from original results"
        );
        let task_result: native_genome::GenomeResult =
            native_genome::load(root, &target.continuation)?;
        let task: native_genome::GenomeTask = native_genome::load(root, &task_result.task)?;
        ensure!(
            task.admission_policy == set.admission_policy
                && task.vocabulary == set.vocabulary
                && task.qualification == set.qualification
                && task.preparation_overlay == set.preparation_overlay,
            "target scope/vocabulary changed"
        );
        let candidate: Candidate = native_genome::load(root, &target.candidate)?;
        ensure!(
            candidate.run == set.run
                && candidate.generation <= set.generation
                && candidate.context.partition == "meta_train",
            "foreign generation or sealed target"
        );
        ensure!(
            !excluded.contains(&candidate.context.family),
            "excluded family target"
        );
        ensure!(
            seen_candidates.insert((
                candidate.generation,
                candidate.arm,
                hash(&serde_json::to_vec(&candidate.context)?),
                candidate.genome_sha256.clone()
            )),
            "duplicate candidate outcome cannot become independent training evidence"
        );
        let budget = (task.evaluations, task.seconds);
        if let Some(old) = budgets.insert(candidate.generation, budget) {
            ensure!(
                old == budget,
                "unmatched continuation budget within generation"
            );
        }
        // Every outcome remains in the target set, including failures and comparison-only arms.
        if !set.training_arms.contains(&candidate.arm) {
            continue;
        }
        let key = hash(&serde_json::to_vec(&candidate.context)?);
        if let Some(reason) = training_refusal(&target) {
            refused_targets.push(json!({"target":reference,"context_sha256":key,"arm":candidate.arm,
                "reason":reason,"completed_children":target.completed_children,
                "completed_evaluations":target.completed_evaluations,"failed_evaluations":target.failed_evaluations,
                "fit_runtime_seconds":target.fit_runtime_seconds}));
            continue;
        }
        groups
            .entry(key)
            .or_default()
            .push((reference.clone(), target, candidate));
    }
    ensure!(
        !groups.is_empty(),
        "no admitted continuation training targets; training held"
    );
    let (groups, undersized) = eligible_contexts(groups, MINIMUM_ADMITTED_TARGETS_PER_CONTEXT);
    let mut excluded_contexts = Vec::new();
    for (key, group) in &undersized {
        let row = &group[0].2.context;
        excluded_contexts.push(json!({"context_sha256":key,"dataset":row.dataset,"family":row.family,
            "cut":row.cut,"admitted_targets":group.len(),"minimum":MINIMUM_ADMITTED_TARGETS_PER_CONTEXT,
            "reason":"fewer_than_minimum_admitted_targets",
            "targets":group.iter().map(|x| x.0.clone()).collect::<Vec<_>>()}));
    }
    ensure!(
        !groups.is_empty(),
        "no usable continuation context has eight admitted targets; training held"
    );
    let mut records = Vec::new();
    let mut labels = Vec::new();
    let mut contexts = Vec::new();
    let mut index_groups = Vec::new();
    let mut legal_sets = vec![vec![0usize]];
    let mut legal_ids = BTreeMap::from([(vec![0usize], 0usize)]);
    let mut families = BTreeSet::new();
    let mut record_count = 0usize;
    for (_, mut group) in groups {
        ensure!(
            group.len() >= MINIMUM_ADMITTED_TARGETS_PER_CONTEXT,
            "continuation context lacks eight independently identified targets"
        );
        group.sort_by(|a, b| {
            a.1.quality_rmse_cut_normalized
                .unwrap()
                .total_cmp(&b.1.quality_rmse_cut_normalized.unwrap())
                .then(a.0.sha256.cmp(&b.0.sha256))
        });
        let context = contexts.len();
        let row = group[0].2.context.clone();
        families.insert(row.family.clone());
        contexts.push(row.clone());
        let scores: Vec<f64> = group
            .iter()
            .map(|x| x.1.quality_rmse_cut_normalized.unwrap())
            .collect();
        let percentiles = crate::teaching_store::midranks(&scores);
        let median = scores[scores.len() / 2].max(1e-8);
        let start = record_count;
        let mut max_length = 0;
        for (i, (reference, target, candidate)) in group.into_iter().enumerate() {
            let tokens = vocab.encode(&candidate.genome)?;
            ensure!(
                tokens.len() <= crate::corpus::SEQUENCE,
                "unrepresentable exact parent program"
            );
            let mut record = crate::corpus::Record {
                tokens: [0; crate::corpus::SEQUENCE],
                legal: [0; crate::corpus::SEQUENCE],
                length: tokens.len(),
                context,
                value: (1. / (1. + scores[i] / median)) as f32,
                percentile: percentiles[i],
                runtime: target.fit_runtime_seconds.ln_1p() as f32,
            };
            let mut grammar = crate::grammar::State::default();
            grammar.sampling = false;
            for (j, token) in tokens.iter().enumerate() {
                record.tokens[j] = *token as i64;
                if j == 0 {
                    continue;
                }
                let legal = grammar.legal(&vocab);
                let id = *legal_ids.entry(legal.clone()).or_insert_with(|| {
                    let n = legal_sets.len();
                    legal_sets.push(legal);
                    n
                });
                ensure!(
                    id < u16::MAX as usize,
                    "continuation grammar table overflow"
                );
                record.legal[j] = id as i64;
                grammar.advance(&vocab, *token)?;
            }
            records.extend(record.bytes());
            max_length = max_length.max(record.length);
            serde_json::to_writer(
                &mut labels,
                &json!({"schema":"jope.continuation_training_label.v1","record":record_count,"context":context,
                "program":candidate.genome,"program_sha256":candidate.genome_sha256,"target":reference,
                "target_kind":target.target_kind,"protocol":target.protocol,"completed_children":target.completed_children,
                "rmse_cut_normalized":scores[i],"fit_runtime_seconds":target.fit_runtime_seconds,"runtime_seconds":target.fit_runtime_seconds,
                "direct_scores_are_separate":true,"heldout_final_test_used":false}),
            )?;
            labels.push(b'\n');
            record_count += 1;
        }
        index_groups.push(crate::corpus::Group {
            context,
            family: row.family,
            partition: row.partition,
            records: (start..record_count).collect(),
            max_length,
        });
    }
    ensure!(
        families.len() >= 2,
        "continuation corpus lacks two training families"
    );
    let context_bytes = serde_json::to_vec(
        &json!({"schema":"parallel.context_index.v1","qualification":set.qualification,"encodings":contexts}),
    )?;
    let index = crate::corpus::Index {
        schema: "parallel.teaching_corpus.v1".into(),
        qualification: set.qualification,
        vocabulary_sha256: set.vocabulary.sha256.clone(),
        context_index_sha256: hash(&context_bytes),
        records_sha256: hash(&records),
        record_bytes: crate::corpus::RECORD_BYTES,
        records: record_count,
        legal_sets,
        contexts,
        labels_sha256: None,
        groups: index_groups,
    };
    let mut files = BTreeMap::from([
        ("records.bin".into(), records.clone()),
        ("labels.jsonl".into(), labels.clone()),
        ("vocabulary.json".into(), vocab_bytes),
        ("contexts.json".into(), context_bytes),
        (
            "target-set.json".into(),
            native_genome::checked(root, manifest, 16 * 1024 * 1024)?,
        ),
    ]);
    let mut index_value = serde_json::to_value(&index)?;
    index_value["label_policy"] = json!("strict_continuation_midrank_v1");
    index_value["target_set_sha256"] = json!(manifest.sha256);
    if let Some(a) = &set.preparation_overlay {
        index_value["preparation_overlay"] = serde_json::to_value(a)?;
    }
    let index_bytes = serde_json::to_vec(&index_value)?;
    files.insert("provenance.json".into(),serde_json::to_vec(&json!({"schema":"jope.continuation_corpus_provenance.v1","index_sha256":hash(&index_bytes),"labels_sha256":hash(&labels),
        "parent_corpus":set.parent_corpus,"target_set":manifest,"label_policy":"strict_continuation_midrank_v1","frozen_scientific_policy":set.frozen_scientific_policy,
        "direct_history_preserved":true,"protocol":native_genome::CONTINUATION_PROTOCOL,
        "training_admission":{"minimum_completed_children":native_genome::MINIMUM_COMPLETED_CHILDREN,
            "completed_child":"continuation row with null original failure and measured RMSE; parent row excluded",
            "minimum_admitted_targets_per_context":MINIMUM_ADMITTED_TARGETS_PER_CONTEXT},
        "refused_targets":refused_targets,"excluded_contexts":excluded_contexts,"targets":set.targets}))?);
    files.insert("index.json".into(), index_bytes);
    Ok(files)
}
