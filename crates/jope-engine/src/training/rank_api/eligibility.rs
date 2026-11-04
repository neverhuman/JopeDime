use super::*;

/// This path is reachable only through the authenticated native worker. Legacy
/// rank entry points retain their original explicitly-qualified-corpus gate.
pub(super) fn verify_native_training_eligibility(
    task: &crate::smartcluster_worker::VerifiedTask,
    policy: &Value,
    warm_start_parent_step: Option<u64>,
) -> Result<()> {
    use crate::{
        native_graph::Artifact,
        native_loop::{Request, Role},
        smartcluster_worker::Operation,
    };
    let Operation::Train {
        corpus,
        initialize,
        initial_optimizer_step,
        requested_step_increment,
        target_optimizer_step,
        ..
    } = task.operation()
    else {
        anyhow::bail!("native Train required")
    };
    let root = task.root();
    let manifest = task.manifest();
    let corpus = Artifact {
        path: corpus.join("index.json"),
        sha256: file_hash(&resolve(root, corpus)?.join("index.json"))?,
    };
    let parent = Artifact {
        path: initialize.join("state.json"),
        sha256: file_hash(&resolve(root, initialize)?.join("state.json"))?,
    };
    ensure!(
        manifest.parent_checkpoint_sha256.as_deref() == Some(parent.sha256.as_str())
            && manifest.corpus_sha256.as_deref() == Some(corpus.sha256.as_str()),
        "native training parent/corpus manifest differs"
    );
    let index: Value = crate::native_genome::load_corpus_index(root, &corpus)?;
    ensure!(
        index["qualification"] == false
            && index["label_policy"] == "strict_continuation_midrank_v1",
        "unqualified corpus is not native verified continuation targets"
    );
    let evidence = &policy["eligibility_evidence"];
    ensure!(
        evidence["schema"] == "jope.native_continuation_eligibility.v1",
        "native eligibility evidence absent"
    );
    let mut parameters = json!({"checkpoint":parent,"corpus":corpus,"initial_step":initial_optimizer_step,
        "requested_increment":requested_step_increment,"target_step":target_optimizer_step,"global_batch":2048});
    if let Some(step) = warm_start_parent_step {
        parameters["optimizer"] = json!("reset");
        parameters["parent_step"] = json!(step);
    }
    ensure!(
        evidence["original_window"] == parameters,
        "native replay original window differs"
    );
    let scientific: Artifact = serde_json::from_value(evidence["scientific_policy"].clone())?;
    let request = Request {
        schema: "jope.native_loop_request.v1".into(),
        run: manifest.run.clone(),
        generation: manifest.generation,
        id: manifest.task_id.clone(),
        role: Role::Train,
        parameters,
        predecessors: serde_json::from_value(evidence["verified_corpus_dependencies"].clone())?,
    };
    let expected = crate::native_loop::materialize::training_eligibility(
        root,
        &corpus,
        &parent,
        &scientific,
        &request,
    )?;
    ensure!(
        *policy == expected,
        "native replay differs from complete verified target reconstruction"
    );
    Ok(())
}
