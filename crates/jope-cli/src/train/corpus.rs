//! Materialize measured, family-disjoint real training inputs. No encoder/model is run.
//! All private execution proofs are re-admitted before any corpus is published.
use super::{artifact, directory, immutable};
use anyhow::{bail, ensure, Context, Result};
use clap::Args;
use feat_math::GenomeIr;
use jope_learning::{
    canon::{canonical_json, canonical_sha256},
    eligibility::{freeze_teaching, Qualification},
    state::State,
};
use jope_pipeline::{
    cpu_context_bridge::{load_qualified, AdmissionExpectation, CpuContextFiles, ExecutionProof},
    execution::Artifact,
    experiment::PanelDataset,
    neural_search::FrozenRecipe,
    prepared_data::parse_development,
    search_context::TrainingIdentity,
    sha256_hex,
};
use jope_policy::context_producer::{
    self, CpuContextArtifact, EncoderIdentity, TrainingInput, TrainingRow,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

#[derive(Args)]
pub(super) struct PrepareArgs {
    #[arg(long)]
    learning_state: PathBuf,
    #[arg(long)]
    request: PathBuf,
    #[arg(long)]
    request_sha256: String,
    #[arg(long)]
    output: PathBuf,
}
#[derive(Args)]
pub(super) struct RequestArgs {
    #[arg(long)]
    learning_state: PathBuf,
    #[arg(long)]
    family_registry: PathBuf,
    #[arg(long)]
    vocabulary: PathBuf,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long)]
    expect_source: String,
    #[arg(long, required = true)]
    context_files: Vec<PathBuf>,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    output: PathBuf,
}
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
    ensure!(
        !families.is_empty() && families.len() <= 3,
        "unsupported requested task inventory"
    );
    ensure!(
        families.values().all(|s| s.len() >= 2),
        "real training requires at least two independently evidenced families per category"
    );
    let mut all = families
        .values()
        .flat_map(|s| s.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    ensure!(
        all.len() <= 30,
        "family registry exceeds bounded panel scope"
    );
    all.sort_by_key(|name| {
        sha256_hex(format!("jope.family-partition/v1\0{seed}\0{name}").as_bytes())
    });
    let acceptable = |selected: &BTreeSet<String>| {
        families
            .values()
            .all(|set| !set.is_disjoint(selected) && !set.is_subset(selected))
    };
    let mut selected = None;
    'outer: for a in 0..all.len() {
        for b in a..all.len() {
            for c in b..all.len() {
                let set = [all[a].clone(), all[b].clone(), all[c].clone()]
                    .into_iter()
                    .collect();
                if acceptable(&set) {
                    selected = Some(set);
                    break 'outer;
                }
            }
        }
    }
    let mut selected =
        selected.context("no family-disjoint partition covers every requested category")?;
    let target = all.len().div_ceil(5);
    for name in &all {
        if selected.len() >= target {
            break;
        }
        let mut trial = selected.clone();
        trial.insert(name.clone());
        if acceptable(&trial) {
            selected = trial;
        }
    }
    Ok(all
        .into_iter()
        .map(|family| {
            let side = if selected.contains(&family) {
                "development"
            } else {
                "train"
            };
            (family, side.to_owned())
        })
        .collect())
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
struct Selected {
    qualification: Qualification,
    dataset: PanelDataset,
    family: String,
    recipe_ref: Artifact,
    recipe: FrozenRecipe,
    proof: CpuProof,
}
fn verify_context(
    state: &State,
    selected: &Selected,
    encoder: &EncoderIdentity,
) -> Result<(Vec<f32>, TrainingIdentity)> {
    let q = &selected.qualification;
    let dataset = &selected.dataset;
    let proof = &selected.proof;
    let policy = value(&blob(state, &q.freeze.freeze_sha256)?)?;
    let inner = &policy["inner_partition"];
    ensure!(
        inner["schema"] == "jope.teaching-inner-partition/v1"
            && canonical_sha256(inner)? == q.comparison.cut_id,
        "real corpus requires authenticated inner-OOF teaching partition"
    );
    let fold = q
        .context_outer_fold
        .context("exact context outer fold absent")?;
    ensure!(
        inner["original_split_sha256"] == dataset.split.sha256
            && inner["development_sha256"] == dataset.split.development_sha256
            && inner["outer_fold"] == fold
            && inner["dataset_sha256"] == dataset.source.sha256,
        "inner context cut changed"
    );
    let development = blob(state, &dataset.split.development_sha256)?;
    let data = parse_development(&checked(&development, 512 << 20)?, dataset)?;
    let cut = data.cut(fold)?;
    let ids = cut.train.rows.iter().map(|r| &r.id).collect::<Vec<_>>();
    let ids_value = serde_json::to_value(&ids)?;
    let row_sha = q
        .context_fitting_rows_sha256
        .as_deref()
        .context("exact context fitting row digest absent")?;
    ensure!(
        row_sha == cut.train.rows_sha256
            && value(&blob(state, row_sha)?)? == ids_value
            && inner["fitting_rows"] == ids_value,
        "context row IDs differ from original development assignment"
    );
    let rows = cut
        .train
        .rows
        .iter()
        .map(|row| TrainingRow {
            id: row.id.clone(),
            values: row.values.clone(),
            label: row.target.clone(),
        })
        .collect::<Vec<_>>();
    let input: TrainingInput = serde_json::from_slice(&checked(&proof.input, 128 << 20)?)?;
    ensure!(
        serde_json::to_value(&input.rows)? == serde_json::to_value(&rows)?
            && serde_json::to_value(&input.features)? == serde_json::to_value(&dataset.features)?,
        "CPU context does not use the exact original fitting values/labels/schema"
    );
    let training_sha = context_producer::training_rows_sha256(&rows)?;
    ensure!(
        input.identity.dataset_sha256 == dataset.source.sha256
            && input.identity.prepared_sha256 == dataset.split.development_sha256
            && input.identity.training_sha256 == training_sha
            && input.identity.fitting_rows_sha256 == row_sha
            && input.identity.cut == fold,
        "CPU training identity differs from authenticated original rows"
    );
    let preprocessing_sha = sha256_hex(&serde_json::to_vec(&input.prepare()?.preprocessing)?);
    let training = TrainingIdentity {
        prepared_sha256: dataset.split.development_sha256.clone(),
        training_sha256: training_sha,
        selected_training_rows_sha256: row_sha.to_owned(),
        preprocessing_sha256: preprocessing_sha,
        cut: fold.into(),
    };
    for reference in [
        &proof.producer,
        &proof.qualification,
        &proof.reference_manifest,
    ] {
        reference.verify()?;
    }
    let request = &proof.context_execution.job.request;
    ensure!(
        request.dataset_sha256 == dataset.source.sha256
            && request.split_sha256 == dataset.split.sha256
            && request.fitting_rows_sha256 == row_sha
            && request.fitting_rows == cut.train.rows.len()
            && request.fold == fold
            && request.task == dataset.task,
        "actual context worker differs from frozen fitting partition"
    );
    ensure!(
        proof.context_execution.job.command.program.path == proof.producer.path
            && proof.context_execution.job.command.program.sha256 == proof.producer.sha256,
        "CPU producer path or executable changed"
    );
    let expected = AdmissionExpectation {
        source_sha40: proof.producer_source_sha40.clone(),
        producer_sha256: proof.producer.sha256.clone(),
        input_sha256: proof.input.sha256.clone(),
        qualification_sha256: proof.qualification.sha256.clone(),
        reference_manifest_sha256: proof.reference_manifest.sha256.clone(),
        dataset_sha256: dataset.source.sha256.clone(),
        training: training.clone(),
    };
    let files = CpuContextFiles {
        vector: proof.vector.clone(),
        artifact: proof.artifact.clone(),
        frozen_transforms: proof.frozen_transforms.clone(),
        bundle: proof.bundle.clone(),
        context_execution: proof.context_execution.clone(),
        case_executions: proof.case_executions.clone(),
        training: training.clone(),
    };
    let token = load_qualified(&files, &expected)?;
    let produced: CpuContextArtifact =
        serde_json::from_slice(&checked(&files.artifact, 128 << 20)?)?;
    ensure!(
        &produced.encoder == encoder,
        "training encoder differs from qualified component encoder"
    );
    Ok((token.context().to_vec(), training))
}

pub(super) fn prepare(args: PrepareArgs) -> Result<Value> {
    let request_ref = Artifact {
        path: args.request,
        sha256: args.request_sha256,
    };
    let request: Request = serde_json::from_slice(&checked(&request_ref, 32 << 20)?)?;
    ensure!(
        request.schema == "jope.real-training-corpus-request/v1",
        "unsupported real corpus request"
    );
    // Regression is the initial positively supported route. Classification has a
    // separate experimental token and must never be routed through this producer.
    ensure!(request.tasks == ["regression"], "this corpus producer supports real regression; classification requires its separate experimental context route");
    let registry: FamilyRegistry =
        serde_json::from_slice(&checked(&request.family_registry, 8 << 20)?)?;
    ensure!(
        registry.schema == "jope.source-family-registry/v1"
            && registry.policy == "explicit-upstream-provenance/conservative-shared-source/v1",
        "unsupported family registry"
    );
    let vocabulary = value(&request.vocabulary)?;
    let encoder: EncoderIdentity = serde_json::from_slice(&checked(&request.encoder, 1 << 20)?)?;
    if let Some(checkpoint) = &request.checkpoint {
        checkpoint.verify()?;
    }
    let state = State::at(&args.learning_state);
    let teaching = freeze_teaching(&state)?;
    let frozen_bytes = canonical_json(&teaching)?;
    let mut selected = Vec::new();
    let mut families: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for q in teaching
        .entries
        .iter()
        .filter(|q| request.tasks.contains(&q.comparison.task))
    {
        ensure!(
            q.comparison.vocabulary_sha256 == request.vocabulary.sha256,
            "eligible comparison uses incompatible vocabulary; no retroactive conversion"
        );
        ensure!(
            q.label_role == "development" && q.candidate_loss.is_finite(),
            "nonfinite or report-only teaching entry"
        );
        let policy = value(&blob(&state, &q.freeze.freeze_sha256)?)?;
        let dataset: PanelDataset = serde_json::from_value(policy["dataset"].clone())?;
        ensure!(
            dataset.source.sha256 == q.comparison.dataset_hash
                && serde_json::to_value(dataset.task)? == q.comparison.task,
            "teaching task/source identity differs"
        );
        let source_family = family(&registry, &dataset)?.to_owned();
        let recipe_ref = blob(&state, &q.record_sha256)?;
        let recipe: FrozenRecipe = serde_json::from_slice(&checked(&recipe_ref, 32 << 20)?)?;
        recipe.candidate.recipe.validate()?;
        ensure!(
            recipe.candidate.genome == recipe.candidate.recipe.original_proposal
                && recipe.candidate.program_sha256 == q.program_sha256
                && recipe.candidate.recipe.executable_sha256 == q.program_sha256,
            "original proposal/derived measured program binding differs"
        );
        compatible(&recipe.candidate.genome, &q.comparison.task, &vocabulary)?;
        let proof = request
            .contexts
            .get(&q.freeze.comparison_sha256)
            .context("eligible comparison has no fresh qualified fitting-row context")?
            .clone();
        families
            .entry(q.comparison.task.clone())
            .or_default()
            .insert(source_family.clone());
        selected.push(Selected {
            qualification: q.clone(),
            dataset,
            family: source_family,
            recipe_ref,
            recipe,
            proof,
        });
    }
    ensure!(
        !selected.is_empty(),
        "no admitted real regression teaching rows; no synthetic regression substitution"
    );
    let comparisons = selected
        .iter()
        .map(|s| s.qualification.freeze.comparison_sha256.clone())
        .collect::<BTreeSet<_>>();
    ensure!(
        comparisons.len() == selected.len()
            && comparisons == request.contexts.keys().cloned().collect(),
        "context inventory must exactly cover admitted comparisons"
    );
    let splits = partition(&families, request.seed)?;
    selected.sort_by(|a, b| {
        a.qualification
            .freeze
            .comparison_sha256
            .cmp(&b.qualification.freeze.comparison_sha256)
    });
    // Validate every actual proof before writing even a partial consumable corpus.
    let verified = selected
        .iter()
        .map(|s| verify_context(&state, s, &encoder))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        canonical_json(&freeze_teaching(&state)?)? == frozen_bytes,
        "eligible HWM changed during preparation; start a new immutable request"
    );
    let controller = directory(&args.output, true)?;
    // Full admission/exclusion and process evidence remains controller-only.
    let original_admission =
        bytes_file(&controller.join("controller-admission.json"), &frozen_bytes)?;
    let output = directory(&controller.join("worker"), true)?;
    let mut closure = Vec::<Value>::new();
    let admission = object_file(
        &output.join("admission.json"),
        &json!({
            "schema":"jope.qualified-teaching-freeze/v1",
            "projection_of_sha256":original_admission.sha256,
            "entries":selected.iter().map(|s| &s.qualification).collect::<Vec<_>>()
        }),
    )?;
    closure.push(json!({"artifact":admission,"role":"teaching-admission"}));
    let partition_ref = object_file(
        &output.join("family-partition.json"),
        &json!({"schema":"jope.training-family-partition/v1","policy":"seeded-family-order/three-task-witness/development-ceil20percent/v1","seed":request.seed,"registry_sha256":request.family_registry.sha256,"families":splits,"requested_tasks":request.tasks,"admission":admission}),
    )?;
    closure.push(json!({"artifact":partition_ref,"role":"family-partition"}));
    let mut corpus = Vec::new();
    let mut contexts = BTreeMap::new();
    for (selected, (values, training)) in selected.iter().zip(verified) {
        let q = &selected.qualification;
        let context_id = &q.freeze.comparison_sha256;
        let proof = &selected.proof;
        ensure!(
            values.len() == 2048 && values.iter().all(|v| v.is_finite()),
            "qualified context dimension or finiteness differs"
        );
        let wrapper = object_file(
            &output.join(format!("context-{context_id}.json")),
            &json!({"lineage":"asm.experimental-regression-context/v1","policy_compatible":false,"training":training,"values":values,"context_sha256":float_sha(&values)}),
        )?;
        let fitting_input = bytes_file(
            &output.join(format!("fitting-input-{context_id}.json")),
            &checked(&proof.input, 128 << 20)?,
        )?;
        let recipe = bytes_file(
            &output.join(format!("recipe-{context_id}.json")),
            &checked(&selected.recipe_ref, 32 << 20)?,
        )?;
        let producer = object_file(
            &output.join(format!("producer-{context_id}.json")),
            &json!({"schema":"jope.training-context-producer/v1","output_sha256":wrapper.sha256,
            "fitting_rows_sha256":training.selected_training_rows_sha256,"encoder_sha256":request.encoder.sha256,"training":training,
            "backend_lineage":context_producer::BACKEND,"qualified_bundle_sha256":proof.bundle.sha256,
            "inputs":[fitting_input]}),
        )?;
        for (artifact, role) in [
            (&wrapper, "context-values"),
            (&producer, "context-producer"),
            (&fitting_input, "fitting-input"),
            (&recipe, "teaching-recipe"),
        ] {
            closure.push(json!({"artifact":artifact,"role":role,"split":splits[&selected.family],"source_family":selected.family,"context_id":context_id}));
        }
        contexts.insert(context_id.clone(),json!({"task":q.comparison.task,"dataset_id":q.comparison.dataset_hash,"comparison_sha256":context_id,"partition":"fit",
            "fitting_rows_sha256":training.selected_training_rows_sha256,"encoder_sha256":request.encoder.sha256,"context_encoding":"f32le/v1",
            "embedding_f32le_sha256":float_sha(&values[..1536]),"target_embedding_f32le_sha256":float_sha(&values[1536..]),
            "lineage":"asm.experimental-regression-context/v1","context_artifact":wrapper,"producer_receipt":producer,"cpu_proof_sha256":canonical_sha256(proof)?}));
        let row = json!({"genome":selected.recipe.candidate.genome,"engine":"featmath","dataset_id":q.comparison.dataset_hash,"source":"asm-qualified-real-teaching",
            "embedding":values[..1536],"target_embedding":values[1536..],"metadata":{"task_type":q.comparison.task,"split":splits[&selected.family],"source_family":selected.family,
            "context_id":context_id,"dataset_name":selected.dataset.id,"family_registry_sha256":request.family_registry.sha256,"family_partition":partition_ref,
            "teaching_recipe":recipe,"original_proposal_sha256":selected.recipe.candidate.recipe.original_bytes_sha256,
            "executable_program_sha256":q.program_sha256,"reward_pair_identity":context_id,"measured_oof_loss":q.candidate_loss,"score_robust":-q.candidate_loss}});
        corpus.extend(canonical_json(&row)?);
        corpus.push(b'\n');
    }
    let corpus_ref = bytes_file(&output.join("corpus.jsonl"), &corpus)?;
    let index = object_file(
        &output.join("contexts.json"),
        &json!({"schema":"jope.training-context-index/v1","contexts":contexts}),
    )?;
    let encoder_ref = bytes_file(
        &output.join("encoder.json"),
        &checked(&request.encoder, 1 << 20)?,
    )?;
    let vocabulary_ref = bytes_file(
        &output.join("active_vocab.json"),
        &checked(&request.vocabulary, 32 << 20)?,
    )?;
    for (artifact, role) in [
        (&corpus_ref, "corpus"),
        (&index, "context-index"),
        (&encoder_ref, "encoder-identity"),
        (&vocabulary_ref, "vocabulary"),
    ] {
        closure.push(json!({"artifact":artifact,"role":role}));
    }
    let mut inputs = json!({"schema":"jope.experimental-training-inputs/v1","corpus":corpus_ref,"contexts":index,"encoder":encoder_ref,"vocabulary":vocabulary_ref,"admission":admission});
    if let Some(checkpoint) = request.checkpoint {
        let projected = bytes_file(
            &output.join("initial-checkpoint.pt"),
            &checked(&checkpoint, 2 << 30)?,
        )?;
        closure.push(json!({"artifact":projected,"role":"checkpoint"}));
        inputs["checkpoint"] = serde_json::to_value(projected)?;
    }
    let input_closure = object_file(
        &output.join("input-closure.json"),
        &json!({
            "schema":"jope.training-input-closure/v1","root":output,"artifacts":closure,
            "families":splits,"contexts":comparisons,
            "excluded_roles":["outer-test","sealed-meta-test","outside-selection-evaluation"],
            "controller_admission_sha256":original_admission.sha256
        }),
    )?;
    inputs["input_closure"] = serde_json::to_value(input_closure)?;
    let manifest = object_file(&output.join("inputs.json"), &inputs)?;
    let result = json!({"schema":"jope.real-training-corpus-result/v1","request":request_ref,"inputs":manifest,"family_partition":partition_ref,
        "rows":selected.len(),"families":splits.len(),"training_families":splits.values().filter(|v|*v=="train").count(),
        "development_families":splits.values().filter(|v|*v=="development").count(),"fits":0,"optimizer_updates":0,
        "token_validation":"exact-bin gate passed; complete Python token roundtrip is required by train plan/run"});
    object_file(&controller.join("receipt.json"), &result)?;
    Ok(result)
}

/// Derive opaque comparison/proof pins from actual retained artifacts, then run
/// the same admission consumer used by prepare-inputs. No fits or clock admission.
pub(super) fn prepare_request(args: RequestArgs) -> Result<Value> {
    ensure!(
        args.expect_source.len() == 40 && args.expect_source.bytes().all(|c| c.is_ascii_hexdigit()),
        "full expected source required"
    );
    let state = State::at(&args.learning_state);
    let teaching = freeze_teaching(&state)?;
    let selected = teaching
        .entries
        .iter()
        .filter(|q| q.comparison.task == "regression")
        .collect::<Vec<_>>();
    ensure!(
        !selected.is_empty(),
        "no admitted real regression comparisons"
    );
    let vocabulary = artifact(&fs::canonicalize(&args.vocabulary)?)?;
    let registry = artifact(&fs::canonicalize(&args.family_registry)?)?;
    ensure!(
        selected
            .iter()
            .all(|q| q.comparison.vocabulary_sha256 == vocabulary.sha256),
        "admitted vocabulary differs"
    );
    let mut retained = Vec::new();
    let mut seen = BTreeSet::new();
    for path in &args.context_files {
        let reference = artifact(&fs::canonicalize(path)?)?;
        ensure!(
            seen.insert(reference.path.clone()),
            "duplicate context-files artifact"
        );
        let files: CpuContextFiles = serde_json::from_slice(&checked(&reference, 32 << 20)?)?;
        let bundle: jope_policy::producer::QualifiedContextBundle =
            serde_json::from_slice(&checked(&files.bundle, 128 << 20)?)?;
        ensure!(
            bundle.pins.source_sha256 == args.expect_source,
            "context source differs from expected source"
        );
        // The final corpus consumer rechecks all nine actual executions against
        // independently admitted fitting rows; this pass never supplies a token.
        retained.push((files, bundle, reference));
    }
    let output = directory(&args.output, true)?;
    let mut contexts = BTreeMap::new();
    let mut used = BTreeSet::new();
    let mut encoder: Option<EncoderIdentity> = None;
    for q in selected {
        let matches = retained
            .iter()
            .enumerate()
            .filter(|(_, (files, _, _))| {
                let request = &files.context_execution.job.request;
                request.dataset_sha256 == q.comparison.dataset_hash
                    && q.context_fitting_rows_sha256.as_ref() == Some(&request.fitting_rows_sha256)
                    && q.context_outer_fold == Some(request.fold)
            })
            .collect::<Vec<_>>();
        ensure!(
            matches.len() == 1,
            "admitted comparison requires exactly one matching context execution"
        );
        let (index, (files, bundle, _)) = matches[0];
        used.insert(index);
        let inputs = files
            .context_execution
            .job
            .command
            .inputs
            .iter()
            .filter(|a| a.sha256 == bundle.pins.input_sha256)
            .collect::<Vec<_>>();
        ensure!(
            inputs.len() == 1,
            "context execution input is absent or ambiguous"
        );
        let produced: CpuContextArtifact =
            serde_json::from_slice(&checked(&files.artifact, 128 << 20)?)?;
        if let Some(identity) = &encoder {
            ensure!(
                identity == &produced.encoder,
                "context encoder identities differ"
            );
        } else {
            encoder = Some(produced.encoder);
        }
        let key = q.freeze.comparison_sha256.clone();
        let proof = CpuProof {
            producer: files.context_execution.job.command.program.clone(),
            producer_source_sha40: args.expect_source.clone(),
            input: inputs[0].clone(),
            qualification: bytes_file(
                &output.join(format!("qualification-{key}.json")),
                &bundle.qualification,
            )?,
            reference_manifest: bytes_file(
                &output.join(format!("references-{key}.json")),
                &bundle.reference_manifest,
            )?,
            vector: files.vector.clone(),
            artifact: files.artifact.clone(),
            frozen_transforms: files.frozen_transforms.clone(),
            bundle: files.bundle.clone(),
            context_execution: files.context_execution.clone(),
            case_executions: files.case_executions.clone(),
        };
        ensure!(
            contexts.insert(key, proof).is_none(),
            "duplicate admitted comparison identity"
        );
    }
    ensure!(
        used.len() == retained.len(),
        "unrequested context execution supplied"
    );
    let request = Request {
        schema: "jope.real-training-corpus-request/v1".into(),
        tasks: vec!["regression".into()],
        seed: args.seed,
        family_registry: registry,
        vocabulary,
        encoder: object_file(
            &output.join("encoder.json"),
            &encoder.context("encoder absent")?,
        )?,
        checkpoint: args
            .checkpoint
            .as_ref()
            .map(|p| artifact(&fs::canonicalize(p)?))
            .transpose()?,
        contexts,
    };
    let reference = object_file(&output.join("request.json"), &request)?;
    let inputs = prepare(PrepareArgs {
        learning_state: args.learning_state,
        request: reference.path.clone(),
        request_sha256: reference.sha256.clone(),
        output: output.join("inputs"),
    })?;
    let result = json!({"schema":"jope.training-request-preparation/v1","request":reference,
        "context_sources":retained.iter().map(|(_,_,r)|r).collect::<Vec<_>>(),
        "validated_inputs":inputs,"fits":0,"execution_budget_started":false});
    object_file(&output.join("receipt.json"), &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sets(values: &[(&str, &[&str])]) -> BTreeMap<String, BTreeSet<String>> {
        values
            .iter()
            .map(|(task, names)| {
                (
                    task.to_string(),
                    names.iter().map(|s| s.to_string()).collect(),
                )
            })
            .collect()
    }
    #[test]
    fn partition_is_family_disjoint_and_deterministic() {
        let family = sets(&[
            ("regression", &["robot", "pollution", "wind"]),
            ("binary", &["robot", "credit"]),
        ]);
        let first = partition(&family, 7).unwrap();
        assert_eq!(first, partition(&family, 7).unwrap());
        for members in family.values() {
            assert_eq!(
                members
                    .iter()
                    .map(|f| first[f].as_str())
                    .collect::<BTreeSet<_>>(),
                ["train", "development"].into_iter().collect()
            );
        }
    }
    #[test]
    fn absent_one_family_and_impossible_cross_task_partitions_fail() {
        assert!(partition(&BTreeMap::new(), 7).is_err());
        assert!(partition(&sets(&[("regression", &["same-source"])]), 7).is_err());
        assert!(partition(
            &sets(&[
                ("regression", &["a", "b"]),
                ("binary", &["b", "c"]),
                ("multiclass", &["a", "c"])
            ]),
            7
        )
        .is_err());
    }
    #[test]
    fn exact_bins_refuse_snapping_unknown_params_and_wrong_tasks() {
        let vocab = json!({"engines":{"featmath":{"engine":"featmath","max_transforms":4,"heads":{"ridge":{"tasks":["regression"],"generate":true,"params":[{"name":"lambda","bins":[1.0,2.0],"required":false}]}}}}});
        assert!(compatible(" (pipe X (ridge lambda=1)) ", "regression", &vocab).is_ok());
        for (genome, task) in [
            ("(pipe X (ridge lambda=1.1))", "regression"),
            ("(pipe X (ridge other=1))", "regression"),
            ("(pipe X (ridge lambda=1))", "binary"),
        ] {
            assert!(compatible(genome, task, &vocab).is_err());
        }
    }
    #[test]
    fn f32_identity_is_little_endian_and_preserves_signed_zero() {
        assert_ne!(float_sha(&[0.0]), float_sha(&[-0.0]));
        assert_eq!(float_sha(&[1.0]), sha256_hex(&[0, 0, 128, 63]));
    }
}
