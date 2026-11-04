use super::*;

pub(crate) fn execute(
    root: &Path,
    reference: &Artifact,
    expected: GenomeEvaluationStage,
    receipt: &Path,
) -> Result<()> {
    let task: GenomeTask = load(root, reference)?;
    ensure!(
        task.stage == expected,
        "fixed native genome service mismatch"
    );
    let c = task.candidate(root)?;
    let parent = direct_parent(root, &task, &c)?;
    // A crash after the original numerical search fsync but before wrapper fsync
    // must not execute the direct fit or continuation budget a second time.
    if !resolve(root, &task.output)?
        .join("search.json")
        .try_exists()?
    {
        crate::gp::search_native_genome(root, reference, &task, &c, parent.as_ref().map(|p| &p.0))?;
    }
    let result = GenomeResult {
        schema: "jope.native_genome_completion.v1".into(),
        task: reference.clone(),
        candidate: task.candidate.clone(),
        stage: task.stage,
        search: Artifact {
            path: task.output.join("search.json"),
            sha256: file_hash(&resolve(root, &task.output)?.join("search.json"))?,
        },
        observations: Artifact {
            path: task.output.join("observations.json"),
            sha256: file_hash(&resolve(root, &task.output)?.join("observations.json"))?,
        },
        direct_parent: task.direct_parent.clone(),
        artifacts: crate::smartcluster_worker::hash_directory(&resolve(root, &task.output)?)?,
    };
    verify_result(root, &task, &result)?;
    crate::smartcluster_worker::write_immutable_record(receipt, &serde_json::to_value(result)?)
}
/// Verified continuation outcomes have a separate identity and retain all original rows.
/// Under plain_seeded_continuation_v1 a zero-child, all-failed or timeout-only continuation is still
/// returned with its rows and cost, but `training_admission` refuses it as an ordinary training label.
pub fn target(root: &Path, continuation: &Artifact) -> Result<ContinuationTarget> {
    let (task, c, result, rows) = verified_rows(root, continuation)?;
    ensure!(
        task.stage == GenomeEvaluationStage::Continuation,
        "population/direct score is not a continuation target"
    );
    let direct = task
        .direct_parent
        .clone()
        .context("original direct parent absent")?;
    let (parent, parent_rows) = direct_parent(root, &task, &c)?.context("direct parent absent")?;
    let children = completed_children(&rows)?;
    let mut all = vec![(parent_rows, 0, parent)];
    all.extend(
        rows.into_iter()
            .enumerate()
            .map(|(i, r)| (result.observations.clone(), i, r)),
    );
    let (best, runtime, failures) = summarize_outcomes(&all)?;
    Ok(ContinuationTarget {
        schema: "jope.continuation_target.v2".into(),
        protocol: CONTINUATION_PROTOCOL.into(),
        candidate: task.candidate,
        direct,
        continuation: continuation.clone(),
        parent_genome_sha256: c.genome_sha256,
        winning_original_observation: best.as_ref().map(|x| x.1.clone()),
        winning_row: best.as_ref().map(|x| x.2),
        quality_rmse_cut_normalized: best.map(|x| x.0),
        fit_runtime_seconds: runtime,
        completed_evaluations: all.len() - 1,
        failed_evaluations: failures,
        completed_children: children,
        training_admission: training_admission(children).into(),
        evaluation_budget: task.evaluations,
        search_seconds_budget: task.seconds,
        target_kind: "best_observed_outcome_after_plain_seeded_continuation_v1".into(),
    })
}
/// Children only: the verified direct parent row never counts toward continuation admission.
pub(super) fn completed_children(rows: &[Value]) -> Result<usize> {
    let mut completed = 0;
    for row in rows {
        if !row["failure"].is_null() {
            continue;
        }
        let quality = row["metrics"]["rmse"]
            .as_f64()
            .context("original child quality absent")?;
        ensure!(
            quality.is_finite() && quality >= 0.,
            "invalid original child quality"
        );
        completed += 1;
    }
    Ok(completed)
}
pub(crate) fn training_admission(completed_children: usize) -> &'static str {
    if completed_children >= MINIMUM_COMPLETED_CHILDREN {
        TRAINING_ADMITTED
    } else {
        REFUSED_NO_COMPLETED_CHILD
    }
}
// Called only after original receipt validation; keeps parent and descendant identities distinct.
pub(super) type OutcomeSummary = (Option<(f64, Artifact, usize)>, f64, usize);
pub(super) fn summarize_outcomes(all: &[(Artifact, usize, Value)]) -> Result<OutcomeSummary> {
    let mut best: Option<(f64, Artifact, usize)> = None;
    let mut runtime = 0.;
    let mut failures = 0;
    for (artifact, index, row) in all {
        let cost = row["runtime_seconds"]
            .as_f64()
            .context("original measured cost absent")?;
        ensure!(
            cost.is_finite() && cost >= 0.,
            "invalid original measured cost"
        );
        runtime += cost;
        if !row["failure"].is_null() {
            failures += 1;
            continue;
        }
        let quality = row["metrics"]["rmse"]
            .as_f64()
            .context("original quality absent")?;
        ensure!(
            quality.is_finite() && quality >= 0.,
            "invalid original quality"
        );
        if best.as_ref().is_none_or(|old| quality < old.0) {
            best = Some((quality, artifact.clone(), *index));
        }
    }
    ensure!(runtime.is_finite(), "continuation cost overflow");
    Ok((best, runtime, failures))
}
