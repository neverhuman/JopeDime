use super::*;

/// Revalidate every original strict observation; no parent score is accepted from the planner.
pub(crate) fn verified_rows(
    root: &Path,
    reference: &Artifact,
) -> Result<(GenomeTask, Candidate, GenomeResult, Vec<Value>)> {
    let result: GenomeResult = load(root, reference)?;
    let task: GenomeTask = load(root, &result.task)?;
    let candidate = task.candidate(root)?;
    let rows = verify_result(root, &task, &result)?;
    Ok((task, candidate, result, rows))
}
pub(crate) fn verify_result(
    root: &Path,
    task: &GenomeTask,
    result: &GenomeResult,
) -> Result<Vec<Value>> {
    ensure!(
        result.schema == "jope.native_genome_completion.v1",
        "native genome receipt schema"
    );
    let candidate = task.candidate(root)?;
    ensure!(
        result.stage == task.stage
            && result.candidate == task.candidate
            && result.direct_parent == task.direct_parent,
        "native genome result changes original task"
    );
    let directory = resolve(root, &task.output)?;
    ensure!(
        result.search.path == task.output.join("search.json")
            && result.observations.path == task.output.join("observations.json"),
        "native genome result outside original output"
    );
    ensure!(
        crate::smartcluster_worker::hash_directory(&directory)? == result.artifacts,
        "native genome output closure changed"
    );
    let search: Value = load(root, &result.search)?;
    let rows: Vec<Value> = load(root, &result.observations)?;
    ensure!(
        search["terminal"] == true
            && search["final_tests_opened"] == false
            && search["observations_sha256"] == result.observations.sha256
            && rows.len() <= task.evaluations
            && search["completed_evaluations"] == rows.len(),
        "incomplete native genome search"
    );
    let request: Value = crate::label_admission::artifact(root, &search["search_request"])?;
    ensure!(
        request["native_genome_task"] == serde_json::to_value(&result.task)?
            && request["evaluations"] == task.evaluations
            && request["seconds"] == task.seconds
            && request["rng_seed"] == task.rng_seed,
        "original genome request budget changed"
    );
    let expected: Value = load(root, &task.admission_policy)?;
    // Route R8: the preparation these observations are admitted against is the
    // bound one; `verify_inputs` refuses any parse that is not those bytes.
    let prepared: crate::prepare::Prepared = serde_json::from_slice(&read_verified(
        &resolve(root, &candidate.context.prepared)?.join("prepared.json"),
        &candidate.context.prepared_sha256,
        "admission prepared artifact",
    )?)?;
    let training = crate::label_admission::verify_inputs(root, &candidate.context, &prepared)?;
    let mut authority = crate::label_admission::approved_authority(root, &expected)?;
    for (i, row) in rows.iter().enumerate() {
        ensure!(
            row["evaluation_index"] == i
                && row["native_genome_task"] == serde_json::to_value(&result.task)?
                && row["native_candidate"] == serde_json::to_value(&task.candidate)?
                && row["comparison_arm"] == serde_json::to_value(candidate.arm)?
                && row["continuation_parent"] == serde_json::to_value(&task.direct_parent)?,
            "native genome observation ordering changed"
        );
        crate::label_admission::validate_receipt(&search, row, rows.len(), &expected)?;
        crate::label_admission::validate_preverified(
            root,
            row,
            &candidate.context,
            &prepared,
            crate::label_admission::AdmissionInputs {
                vocabulary_sha: &task.vocabulary.sha256,
                expected: &expected,
                fit_training: &training,
            },
            &mut authority,
        )?;
        if task.stage == GenomeEvaluationStage::Direct {
            ensure!(
                row["genome"] == candidate.genome,
                "direct task evaluated another program"
            );
        }
    }
    if task.stage == GenomeEvaluationStage::Direct {
        ensure!(
            rows.len() == 1,
            "direct task lacks its exact original evaluation outcome"
        );
    }
    Ok(rows)
}
pub(crate) fn direct_parent(
    root: &Path,
    task: &GenomeTask,
    c: &Candidate,
) -> Result<Option<(Value, Artifact)>> {
    let Some(parent) = &task.direct_parent else {
        return Ok(None);
    };
    let (old, prior, result, rows) = verified_rows(root, parent)?;
    ensure!(
        old.stage == GenomeEvaluationStage::Direct
            && result.candidate == task.candidate
            && prior.genome_sha256 == c.genome_sha256
            && old.vocabulary == task.vocabulary
            && old.admission_policy == task.admission_policy
            && old.qualification == task.qualification
            && old.preparation_overlay == task.preparation_overlay,
        "continuation changed original direct evaluation binding"
    );
    Ok(Some((
        rows.into_iter().next().context("direct row absent")?,
        result.observations,
    )))
}
/// Include every transitive immutable JSON artifact reference plus actual prepared bytes.
/// Live authority databases are not dependencies; original committed receipts are.
pub(crate) fn required_inputs(
    root: &Path,
    reference: &Artifact,
) -> Result<BTreeMap<PathBuf, String>> {
    let task: GenomeTask = load(root, reference)?;
    let c = task.candidate(root)?;
    let mut refs = vec![
        reference.clone(),
        task.candidate.clone(),
        task.vocabulary.clone(),
        task.admission_policy.clone(),
        c.proposal.clone(),
        c.parent_checkpoint.clone(),
        c.origin_capture.clone(),
    ];
    refs.extend(task.direct_parent.iter().cloned());
    refs.extend(task.preparation_overlay.iter().cloned());
    let mut out = artifact_closure(root, refs)?;
    prepared_closure(root, &c.context, &mut out)?;
    Ok(out)
}
/// Every transitive immutable JSON artifact reference reachable from `refs`.
pub(super) fn artifact_closure(
    root: &Path,
    mut refs: Vec<Artifact>,
) -> Result<BTreeMap<PathBuf, String>> {
    let mut out = BTreeMap::new();
    fn nested(v: &Value, refs: &mut Vec<Artifact>) {
        if let Some(o) = v.as_object() {
            if let (Some(p), Some(h)) = (
                o.get("path").and_then(Value::as_str),
                o.get("sha256").and_then(Value::as_str),
            ) {
                if !Path::new(p).is_absolute() && valid_hash(h) {
                    refs.push(Artifact {
                        path: p.into(),
                        sha256: h.into(),
                    });
                }
            }
            // Genome output maps use paths relative to their declared search directory.
            if o.get("schema").and_then(Value::as_str) == Some("jope.native_genome_completion.v1") {
                if let (Some(map), Some(search)) = (
                    o.get("artifacts").and_then(Value::as_object),
                    o.get("search")
                        .and_then(|s| s.get("path"))
                        .and_then(Value::as_str),
                ) {
                    if let Some(base) = Path::new(search).parent() {
                        for (path, digest) in map {
                            if let Some(digest) = digest.as_str().filter(|h| valid_hash(h)) {
                                refs.push(Artifact {
                                    path: base.join(path),
                                    sha256: digest.into(),
                                });
                            }
                        }
                    }
                }
            }
            for x in o.values() {
                nested(x, refs);
            }
        } else if let Some(a) = v.as_array() {
            for x in a {
                nested(x, refs);
            }
        }
    }
    while let Some(a) = refs.pop() {
        ensure!(out.len() < 100000, "native genome closure too large");
        if let Some(old) = out.get(&a.path) {
            ensure!(old == &a.sha256, "conflicting native genome artifact path");
            continue;
        }
        let path = resolve(root, &a.path)?;
        // Route R8: nested references are taken from exactly the verified
        // bytes. Inputs above the parse bound are hashed and never parsed.
        match read_parse_bounded(&path, 16 * 1024 * 1024)? {
            Some(bytes) => {
                ensure!(hash(&bytes) == a.sha256, "native genome input changed");
                crate::manifest::after_verification(&path);
                if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
                    nested(&value, &mut refs);
                }
            }
            None => ensure!(file_hash(&path)? == a.sha256, "native genome input changed"),
        }
        out.insert(a.path, a.sha256);
    }
    Ok(out)
}
/// The whole file through one handle when it is within `limit` bytes; `None`
/// when it is larger (it is then hashed by the caller and never parsed).
fn read_parse_bounded(path: &Path, limit: u64) -> Result<Option<Vec<u8>>> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    if file.metadata()?.len() > limit {
        return Ok(None);
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    Ok((bytes.len() as u64 <= limit).then_some(bytes))
}
/// The prepared directory's actual bytes and its original observation source.
pub(super) fn prepared_closure(
    root: &Path,
    context: &ContextRow,
    out: &mut BTreeMap<PathBuf, String>,
) -> Result<()> {
    let dir = resolve(root, &context.prepared)?;
    for e in walkdir::WalkDir::new(&dir).follow_links(false) {
        let e = e?;
        ensure!(!e.file_type().is_symlink(), "prepared symlink");
        if e.file_type().is_file() {
            out.insert(
                e.path().strip_prefix(root)?.to_owned(),
                file_hash(e.path())?,
            );
        }
    }
    // Route R8: the closure records the preparation the candidate bound, and
    // its original observation source is read from exactly those bytes.
    let prepared = dir.join("prepared.json");
    let bytes = read_verified(&prepared, &context.prepared_sha256, "native genome input")?;
    ensure!(
        out.get(prepared.strip_prefix(root)?) == Some(&context.prepared_sha256),
        "native genome input changed"
    );
    let p: crate::prepare::Prepared = serde_json::from_slice(&bytes)?;
    if let Some(s) = p.storage {
        let path = if s.source.is_absolute() {
            s.source.strip_prefix(root)?.to_owned()
        } else {
            s.source
        };
        ensure!(
            file_hash(&resolve(root, &path)?)? == s.source_sha256,
            "original observation source changed"
        );
        out.insert(path, s.source_sha256);
    }
    Ok(())
}
