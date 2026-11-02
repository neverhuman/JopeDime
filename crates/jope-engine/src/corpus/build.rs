use super::*;

pub fn build(
    root: &Path,
    contexts_path: &Path,
    vocab_path: &Path,
    observations: &[PathBuf],
    output: &Path,
    qualification: bool,
) -> Result<()> {
    build_observed(
        root,
        contexts_path,
        vocab_path,
        observations,
        output,
        qualification,
        |_, _, _| Ok(()),
    )
}

/// Observation affects telemetry only; corpus bytes and admission rules are shared.
pub fn build_observed(
    root: &Path,
    contexts_path: &Path,
    vocab_path: &Path,
    observations: &[PathBuf],
    output: &Path,
    qualification: bool,
    progress: impl FnMut(&str, usize, Option<usize>) -> Result<()>,
) -> Result<()> {
    build_impl(
        root,
        crate::corpus::CorpusSources {
            contexts_path,
            vocab_path,
            observations,
        },
        output,
        qualification,
        None,
        progress,
    )
    .map(|_| ())
}

/// Pinned context, vocabulary and observation inputs for a corpus build.
#[derive(Clone, Copy)]
pub struct CorpusSources<'a> {
    pub contexts_path: &'a Path,
    pub vocab_path: &'a Path,
    pub observations: &'a [PathBuf],
}

/// Reuse validated ledger contributions in a private cache under `root/runs`.
/// Every ledger is SHA256-checked on every call; reuse avoids JSON parsing and
/// tokenization, not source I/O. This never changes corpus bytes or provenance.
/// The caller retains responsibility for atomic publication of `output`.
pub fn build_observed_cached(
    root: &Path,
    sources: CorpusSources<'_>,
    output: &Path,
    qualification: bool,
    cache_path: &Path,
    progress: impl FnMut(&str, usize, Option<usize>) -> Result<()>,
) -> Result<ScanCacheStats> {
    let CorpusSources {
        contexts_path,
        vocab_path,
        observations,
    } = sources;

    ensure!(qualification, "production corpus requires teaching_store::publish_admitted and a frozen admission manifest");
    let cache = LedgerCache::open(root, cache_path, output, observations)?;
    build_impl(
        root,
        crate::corpus::CorpusSources {
            contexts_path,
            vocab_path,
            observations,
        },
        output,
        qualification,
        Some(cache),
        progress,
    )
}

fn build_impl(
    root: &Path,
    sources: CorpusSources<'_>,
    output: &Path,
    qualification: bool,
    cache: Option<LedgerCache>,
    mut progress: impl FnMut(&str, usize, Option<usize>) -> Result<()>,
) -> Result<ScanCacheStats> {
    let CorpusSources {
        contexts_path,
        vocab_path,
        observations,
    } = sources;

    crate::campaign::check_deadline()?;
    ensure!(qualification, "production corpus requires teaching_store::publish_admitted and a frozen admission manifest");
    ensure!(!output.exists(), "corpus version exists");
    let context_bytes = fs::read(contexts_path)?;
    let contexts_sha = hash(&context_bytes);
    let ci: Value = serde_json::from_slice(&context_bytes)?;
    ensure!(
        ci["schema"] == "parallel.context_index.v1"
            && (qualification || ci["qualification"] == false),
        "context qualification mismatch"
    );
    let contexts: Vec<ContextRow> = serde_json::from_value(ci["encodings"].clone())?;
    let vocab_bytes = fs::read(vocab_path)?;
    let vocab_sha = hash(&vocab_bytes);
    let vocab: Vocab = serde_json::from_slice(&vocab_bytes)?;
    // Bind validation/tokenization implementation as well as all data inputs.
    let scan_policy = hash(&serde_json::to_vec(&(
        hash(
            concat!(
                include_str!("../corpus.rs"),
                include_str!("token_cache.rs"),
                include_str!("ledger.rs"),
                include_str!("build.rs"),
                include_str!("records.rs"),
                include_str!("grammar.rs"),
                include_str!("grammar/automaton.rs"),
                include_str!("task_contract.rs"),
                include_str!("prepare.rs"),
                include_str!("encoding.rs"),
                include_str!("manifest.rs")
            )
            .as_bytes(),
        ),
        file_hash(Path::new("/proc/self/exe"))?,
    ))?);
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
    let mut selected = Selected::new();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut token_cache = TokenCache::default();
    let mut source_receipts = Vec::new();
    let mut cache_stats = ScanCacheStats::default();
    let mut hashing_ns = 0u64;
    let mut parsing_ns = 0u64;
    let mut merging_ns = 0u64;
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
            let relative_path = if path.is_absolute() {
                path.strip_prefix(root)?
            } else {
                path.as_path()
            };
            let (source_sha, contribution) = if let Some(cache) = &cache {
                // Never accept metadata (mtime/size) as proof of score identity.
                let safe_path = resolve(root, relative_path)?;
                let hashing_started = std::time::Instant::now();
                let source_sha = file_hash(&safe_path)?;
                cache_stats.ledgers_hashed += 1;
                let binding = LedgerBinding {
                    schema: "parallel.corpus_scan_binding.v1".into(),
                    policy_sha256: scan_policy.clone(),
                    context_index_sha256: contexts_sha.clone(),
                    vocabulary_sha256: vocab_sha.clone(),
                    qualification,
                    source_path: relative_path.to_path_buf(),
                    source_sha256: source_sha.clone(),
                    search_sha256: file_hash(&resolve(
                        root,
                        &relative_path.with_file_name("search.json"),
                    )?)?,
                };
                hashing_ns += hashing_started.elapsed().as_nanos() as u64;
                let parsing_started = std::time::Instant::now();
                if let Some(contribution) = cache.load(&binding, &contexts, &vocab)? {
                    cache_stats.ledgers_reused += 1;
                    parsing_ns += parsing_started.elapsed().as_nanos() as u64;
                    (source_sha, contribution)
                } else {
                    let bytes = fs::read(&safe_path)?;
                    ensure!(
                        hash(&bytes) == source_sha,
                        "observation ledger changed during cache scan: {}",
                        path.display()
                    );
                    let rows: Vec<Value> = serde_json::from_slice(&bytes)?;
                    let contribution = scan_ledger(
                        rows,
                        &path,
                        crate::corpus::ScanInputs {
                            by_cut: &by_cut,
                            contexts: &contexts,
                            prepared: &prepared,
                            vocab: &vocab,
                        },
                        qualification,
                        &mut token_cache,
                    )?;
                    cache_stats.ledgers_parsed += 1;
                    cache.publish(&binding, &contribution, &mut progress)?;
                    cache_stats.entries_committed += 1;
                    parsing_ns += parsing_started.elapsed().as_nanos() as u64;
                    (source_sha, contribution)
                }
            } else {
                let parsing_started = std::time::Instant::now();
                let bytes = fs::read(&path)?;
                let rows: Vec<Value> = serde_json::from_slice(&bytes)?;
                cache_stats.ledgers_parsed += 1;
                let contribution = scan_ledger(
                    rows,
                    &path,
                    crate::corpus::ScanInputs {
                        by_cut: &by_cut,
                        contexts: &contexts,
                        prepared: &prepared,
                        vocab: &vocab,
                    },
                    qualification,
                    &mut token_cache,
                )?;
                parsing_ns += parsing_started.elapsed().as_nanos() as u64;
                let hashing_started = std::time::Instant::now();
                let source_sha = hash(&bytes);
                hashing_ns += hashing_started.elapsed().as_nanos() as u64;
                (source_sha, contribution)
            };
            let merging_started = std::time::Instant::now();
            source_receipts.push(json!({"path":relative_path,"sha256":source_sha,
                "observations":contribution.counts.get("observations_seen").copied().unwrap_or(0)}));
            // Each ledger validates every row before local deduplication. Merge
            // in original source order: an exact program/cut's first label wins.
            for (reason, n) in contribution.counts {
                let count = counts.entry(reason).or_default();
                *count = count.checked_add(n).context("corpus count overflow")?;
            }
            for (context, programs) in contribution.selected {
                let target = selected.entry(context).or_default();
                for (genome, label) in programs {
                    match target.entry(genome) {
                        std::collections::btree_map::Entry::Occupied(_) => {
                            *counts.entry("duplicate_program_cut".into()).or_default() += 1;
                        }
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(label);
                        }
                    }
                }
            }
            merging_ns += merging_started.elapsed().as_nanos() as u64;
            progress(
                "observations scanned",
                counts.get("observations_seen").copied().unwrap_or(0),
                None,
            )?;
        }
    }
    let writing_started = std::time::Instant::now();
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
    ensure!(
        file_hash(contexts_path)? == contexts_sha && file_hash(vocab_path)? == vocab_sha,
        "corpus context or vocabulary changed during build"
    );
    fs::write(output.join("vocabulary.json"), &vocab_bytes)?;
    let index = Index {
        schema: "parallel.teaching_corpus.v1".into(),
        qualification,
        vocabulary_sha256: vocab_sha,
        context_index_sha256: contexts_sha,
        records_sha256: file_hash(&output.join("records.bin"))?,
        record_bytes: RECORD_BYTES,
        records: count,
        legal_sets,
        contexts,
        groups,
        labels_sha256: Some(file_hash(&output.join("labels.jsonl"))?),
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
    write_json(
        &output.join("packing-timing.json"),
        &json!({
            "schema":"parallel.packing_timing.v1","source_hashing_ns":hashing_ns,
            "source_read_parse_cache_ns":parsing_ns,"merging_ns":merging_ns,
            "packing_writing_verification_ns":writing_started.elapsed().as_nanos() as u64,
            "source_content_verified":true,"mtime_used_as_identity":false,
            "timings_excluded_from_corpus_identity":true
        }),
    )?;
    Ok(cache_stats)
}
