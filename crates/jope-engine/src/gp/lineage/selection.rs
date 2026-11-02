use super::*;

/// The selected subject's authenticated chain within one completed search.
#[derive(Clone, Debug, Serialize)]
pub struct SelectedLineage {
    pub selection: selection_replay::SearchReplay,
    pub lineage: Lineage,
    /// `None` when no candidate was measured (a typed no-result, not a
    /// substitute) or when the inherited incumbent remains selected.
    pub selected: Option<Selected>,
    pub inherited_selected: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Selected {
    pub chain: Chain,
    /// Every ancestor ledger position reachable through variation parents.
    pub ancestry: Vec<i64>,
    pub fold_fit_completion: Value,
    pub fold_model: Value,
    /// The selected fold fit's training partition (the cut's training rows).
    pub training_sha256: String,
}

/// Verify a completed search's selection and every candidate's lineage. The
/// seed population and vocabulary are the request's frozen inputs, supplied
/// explicitly and authenticated by their request digests. Each supplied input
/// is read once, so the digest checked is of the bytes parsed (route R8).
pub fn verify_search(
    root: &Path,
    output: &Path,
    seeds: Option<&Path>,
    vocabulary: Option<&Path>,
) -> Result<SelectedLineage> {
    let read =
        |path: &Path| std::fs::read(path).with_context(|| format!("open {}", path.display()));
    let seeds = seeds.map(read).transpose()?;
    let vocabulary = vocabulary.map(read).transpose()?;
    verify_search_inputs(root, output, seeds.as_deref(), vocabulary.as_deref())
}

/// W5-T01: [`verify_search`] against the seed population and vocabulary the search
/// froze beside its request ([`super::SEARCH_INPUTS`]), so the verifier needs no
/// caller paths. An input the request does not pin is not read. A search written
/// before its inputs were frozen still needs [`verify_search`] with the original files.
pub fn verify_search_recorded(root: &Path, output: &Path) -> Result<SelectedLineage> {
    let request: Value = read_json(&output.join("search-request.json"))?;
    let frozen = |key: &str, name: &str| -> Result<Option<Vec<u8>>> {
        if request[key].is_null() {
            return Ok(None);
        }
        let path = output.join(super::SEARCH_INPUTS).join(name);
        std::fs::read(&path)
            .map(Some)
            .with_context(|| format!("frozen search input {} is absent", path.display()))
    };
    let seeds = frozen("seeds_sha256", super::SEARCH_SEEDS)?;
    let vocabulary = frozen("vocabulary_sha256", super::SEARCH_VOCABULARY)?;
    verify_search_inputs(root, output, seeds.as_deref(), vocabulary.as_deref())
}

fn verify_search_inputs(
    root: &Path,
    output: &Path,
    seeds: Option<&[u8]>,
    vocabulary: Option<&[u8]>,
) -> Result<SelectedLineage> {
    let selection = selection_replay::verify_search(root, output)?;
    let root = root.canonicalize()?;
    let output = output.canonicalize()?;
    let request: Value = read_json(&output.join("search-request.json"))?;
    ensure!(
        request["random_protocol"] == RANDOM_PROTOCOL,
        "lineage is verified under the slot-keyed random stream protocol only"
    );
    ensure!(
        request["seeds_sha256"] == json!(seeds.map(hash)),
        "seed population differs from the frozen request"
    );
    ensure!(
        request["vocabulary_sha256"] == json!(vocabulary.map(hash)),
        "vocabulary differs from the frozen request"
    );
    let seeds: Option<SeedPopulation> = seeds.map(serde_json::from_slice).transpose()?;
    let vocabulary: Option<crate::grammar::Vocab> =
        vocabulary.map(serde_json::from_slice).transpose()?;
    let native = match request.get("native_genome_stage").and_then(Value::as_str) {
        None => None,
        Some(stage) => {
            let reference: crate::native_graph::Artifact =
                serde_json::from_value(request["native_genome_task"].clone())?;
            let task: crate::native_genome::GenomeTask =
                crate::native_genome::load(&root, &reference)?;
            let candidate: crate::native_genome::Candidate =
                crate::native_genome::load(&root, &task.candidate)?;
            Some((
                serde_json::to_value(&task.candidate)?,
                candidate.genome,
                stage == "continuation",
            ))
        }
    };
    let rows: Vec<Value> = read_json(&output.join("observations.json"))?;
    // Each envelope is the one its pre-dispatch attempt recorded, and that
    // attempt is the one the original completion binds where one exists.
    for (position, row) in rows.iter().enumerate() {
        let path = output.join(format!("candidate-{position}.attempt.json"));
        let attempt: Value = read_json(&path)?;
        ensure!(
            attempt["lineage"] == row["lineage"] && attempt["genome"] == row["genome"],
            "candidate {position}: ledger lineage differs from its pre-dispatch attempt"
        );
        if !row["clean_completion"].is_null() {
            let completion = crate::label_admission::artifact(&root, &row["clean_completion"])?;
            ensure!(
                completion["attempt"]["sha256"] == file_hash(&path)?,
                "candidate {position}: attempt changed after its completion"
            );
        }
    }
    let inherited = selection
        .replay
        .ranking
        .iter()
        .find(|entry| entry.position == -1)
        .map(|entry| (entry.objective, entry.genome.clone()));
    let context = LineageContext {
        rng_seed: request["rng_seed"].as_u64().context("request seed")?,
        vocabulary: vocabulary.as_ref(),
        vocabulary_sha256: request["vocabulary_sha256"].clone(),
        seeds: seeds.as_ref(),
        seeds_sha256: request["seeds_sha256"].clone(),
        native,
        inherited,
    };
    let lineage = verify_rows(&context, &rows)?;
    let inherited_selected = selection
        .replay
        .winner
        .as_ref()
        .is_some_and(|winner| winner.position < 0);
    let selected = match &selection.replay.winner {
        // No measured candidate is a typed no-result; an inherited incumbent's
        // chain is authenticated by its own direct search, not this ledger.
        None => None,
        Some(winner) if winner.position < 0 => None,
        Some(winner) => {
            let position = winner.position as usize;
            let chain = lineage.chains[position].clone();
            let mut ancestry = BTreeSet::new();
            let mut pending = chain.parents.clone();
            while let Some(parent) = pending.pop() {
                if ancestry.insert(parent) && parent >= 0 {
                    pending.extend(lineage.chains[parent as usize].parents.iter().copied());
                }
            }
            let completion =
                crate::label_admission::artifact(&root, &rows[position]["clean_completion"])?;
            Some(Selected {
                chain,
                ancestry: ancestry.into_iter().collect(),
                fold_fit_completion: completion["fit_completion"].clone(),
                fold_model: completion["model"].clone(),
                training_sha256: rows[position]["training_sha256"]
                    .as_str()
                    .context("selected training identity")?
                    .into(),
            })
        }
    };
    Ok(SelectedLineage {
        selection,
        lineage,
        selected,
        inherited_selected,
    })
}
