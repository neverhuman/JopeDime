#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    tasks: Vec<String>,
    seed: u64,
    family_registry: Artifact,
    vocabulary: Artifact,
    encoder: Artifact,
    checkpoint: Option<Artifact>,
    /// One independently pinned proof per eligible comparison, never a context vector.
    contexts: BTreeMap<String, CpuProof>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CpuProof {
    producer: Artifact,
    producer_source_sha40: String,
    input: Artifact,
    qualification: Artifact,
    reference_manifest: Artifact,
    vector: Artifact,
    artifact: Artifact,
    frozen_transforms: Artifact,
    bundle: Artifact,
    context_execution: ExecutionProof,
    case_executions: Vec<ExecutionProof>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyRegistry {
    schema: String,
    policy: String,
    entries: Vec<FamilyEntry>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyEntry {
    dataset_id: String,
    source_sha256: String,
    source_url: String,
    metadata_url: String,
    metadata_sha256: String,
    source_family: String,
    rationale: String,
    /// Exact frozen metadata plus upstream descriptions, not source data or sealed labels.
    evidence: Vec<FamilyEvidence>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyEvidence {
    url: String,
    artifact: Artifact,
}
fn checked(reference: &Artifact, maximum: u64) -> Result<Vec<u8>> {
    ensure!(
        fs::metadata(&reference.path)?.len() <= maximum,
        "training input exceeds metadata bound"
    );
    reference.verify()?;
    let bytes = fs::read(&reference.path)?;
    ensure!(
        bytes.len() as u64 <= maximum && sha256_hex(&bytes) == reference.sha256,
        "training input changed during read"
    );
    Ok(bytes)
}
fn value(reference: &Artifact) -> Result<Value> {
    Ok(serde_json::from_slice(&checked(reference, 32 << 20)?)?)
}
fn blob(state: &State, sha: &str) -> Result<Artifact> {
    let reference = Artifact {
        path: state.blob_path(sha),
        sha256: sha.to_owned(),
    };
    reference.verify()?;
    Ok(reference)
}
fn bytes_file(path: &Path, bytes: &[u8]) -> Result<Artifact> {
    if path.try_exists()? {
        ensure!(fs::read(path)? == bytes, "immutable corpus output changed");
    } else {
        let mut out = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?;
        out.write_all(bytes)?;
        out.sync_all()?;
        File::open(path.parent().context("output parent absent")?)?.sync_all()?;
    }
    artifact(path)
}
fn object_file(path: &Path, object: &impl Serialize) -> Result<Artifact> {
    immutable(path, object)?;
    artifact(path)
}
fn float_sha(values: &[f32]) -> String {
    sha256_hex(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn family<'a>(registry: &'a FamilyRegistry, dataset: &PanelDataset) -> Result<&'a str> {
    let matches = registry
        .entries
        .iter()
        .filter(|entry| entry.dataset_id == dataset.id)
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "dataset {} needs one explicit resolved family",
        dataset.id
    );
    let entry = matches[0];
    ensure!(
        entry.source_sha256 == dataset.source.sha256
            && entry.source_url == dataset.source.url
            && entry.metadata_url == dataset.source.metadata_url
            && entry.metadata_sha256 == dataset.source.metadata_sha256,
        "family evidence belongs to a different frozen source: {}",
        dataset.id
    );
    ensure!(
        !entry.source_family.trim().is_empty()
            && !entry.rationale.trim().is_empty()
            && !entry.evidence.is_empty(),
        "unresolved family provenance"
    );
    ensure!(
        entry
            .evidence
            .iter()
            .any(|e| e.url == dataset.source.metadata_url
                && e.artifact.sha256 == dataset.source.metadata_sha256),
        "original frozen metadata evidence absent"
    );
    for evidence in &entry.evidence {
        ensure!(
            evidence.url.starts_with("https://") || evidence.url.starts_with("http://"),
            "family evidence URL absent"
        );
        checked(&evidence.artifact, 8 << 20)?;
    }
    Ok(&entry.source_family)
}
/// Find a deterministic family partition covering each requested category on both
/// sides. At most three witness families are needed for three tasks; do not use
/// labels, scores, dataset hashes or row counts to choose the split.
fn partition(
    families: &BTreeMap<String, BTreeSet<String>>,
    seed: u64,
) -> Result<BTreeMap<String, String>> {
    build_training_family_partition(families, seed)
}
/// Conservative exact parameter-bin gate. Python performs an additional complete
/// token encode/decode check at plan/run. Free expressions without an exact frozen
/// bin are refused here; a future expression parser may widen this capability.
fn compatible(genome: &str, task: &str, vocabulary: &Value) -> Result<()> {
    if vocabulary["schema"] == "parallel.action_vocab.v1" {
        let vocab = jope_policy::grammar::Vocab::from_json(&serde_json::to_vec(vocabulary)?)?;
        let tokens = vocab.encode_for_task(genome, task)?;
        let replay = vocab.decode(&tokens)?;
        // Native exact encoder already checks expression-token equivalence as well
        // as every original parameter; retain original bytes separately.
        ensure!(
            vocab.encode_for_task(&replay, task)? == tokens,
            "native token roundtrip changed"
        );
        return Ok(());
    }

    let GenomeIr::Pipeline(pipeline) = GenomeIr::parse(genome)? else {
        bail!("nonpipeline teaching proposal is unsupported")
    };
    let spec = &vocabulary["engines"]["featmath"];
    ensure!(
        spec["engine"] == "featmath",
        "frozen featmath vocabulary absent"
    );
    ensure!(
        !pipeline.stages.is_empty()
            && pipeline.stages.len() - 1
                <= spec["max_transforms"]
                    .as_u64()
                    .context("frozen transform bound absent")? as usize,
        "proposal exceeds frozen grammar"
    );
    for (i, stage) in pipeline.stages.iter().enumerate() {
        let group = if i + 1 == pipeline.stages.len() {
            "heads"
        } else {
            "transforms"
        };
        let op = &spec[group][&stage.op];
        ensure!(
            op["tasks"]
                .as_array()
                .is_some_and(|tasks| tasks.iter().any(|v| v == task))
                && op["generate"].is_boolean(),
            "operator {} lacks exact frozen task policy",
            stage.op
        );
        let parameters = op["params"]
            .as_array()
            .context("frozen parameters absent")?;
        for (name, supplied) in &stage.params {
            if name == "append" && op["allow_append"] == true {
                ensure!(
                    serde_json::to_value(supplied)?.is_boolean(),
                    "append flag must be boolean"
                );
                continue;
            }
            let param = parameters
                .iter()
                .find(|p| p["name"] == *name)
                .context("original parameter absent from frozen grammar")?;
            let supplied = serde_json::to_value(supplied)?;
            ensure!(
                param["bins"]
                    .as_array()
                    .is_some_and(|bins| bins.iter().any(|bin| bin == &supplied
                        || (bin.is_number()
                            && supplied.is_number()
                            && bin.as_f64() == supplied.as_f64()))),
                "original parameter {}.{} is outside exact frozen bins; snapping is forbidden",
                stage.op,
                name
            );
        }
        for param in parameters {
            if param["required"] == true {
                ensure!(
                    stage
                        .params
                        .contains_key(param["name"].as_str().context("parameter name absent")?),
                    "required parameter would be implicitly defaulted"
                );
            }
        }
    }
    Ok(())
}
