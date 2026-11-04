use super::*;

pub(super) fn authenticated_rows(
    state: &State,
    plan: &FreezePlan,
) -> Result<(PanelDataset, Vec<TargetRow>)> {
    let panel: Panel = blob(state, &plan.panel_sha256)?;
    panel.validate()?;
    let dataset = panel
        .datasets
        .into_iter()
        .find(|dataset| dataset.id == plan.dataset_id)
        .context("dataset absent from frozen panel")?;
    ensure!(
        dataset.source.sha256 == plan.comparison.dataset_hash
            && dataset.split.sha256 == plan.comparison.cut_id
            && dataset.task.as_str() == plan.comparison.task
            && dataset.classes
                == plan
                    .comparison
                    .comparison
                    .as_ref()
                    .context("missing comparison")?
                    .class_order,
        "comparison differs from authenticated panel dataset"
    );
    let bytes = state
        .read_blob(&dataset.split.development_sha256)?
        .context("prepared development bytes unavailable")?;
    let data = parse_development(&bytes, &dataset)?;
    let rows = data
        .rows
        .into_iter()
        .map(|row| TargetRow {
            id: row.id,
            group: row.group,
            fold: row.fold,
            target: row.target,
        })
        .collect();
    Ok((dataset, rows))
}

pub(super) fn burn_path(state: &State, dataset: &str, cut: &str) -> Result<std::path::PathBuf> {
    ensure!(
        is_hex64(dataset) && is_hex64(cut),
        "invalid label partition identity"
    );
    Ok(state
        .root()
        .join("eligibility/burns")
        .join(dataset)
        .join(format!("{cut}.json")))
}
/// Test-headline callers must check this before presenting a partition as untouched.
pub fn require_unburned_test_partition(state: &State, dataset: &str, cut: &str) -> Result<()> {
    let ledger = Ledger::open(state.root(), "eligibility");
    ledger.require_intact()?;
    let mut burned = burn_path(state, dataset, cut)?.exists();
    for event in ledger.events()? {
        if let Some(sha) = event
            .get("label_burn_sha256")
            .and_then(|value| value.as_str())
        {
            let receipt: serde_json::Value = blob(state, sha)?;
            burned |= receipt
                .get("dataset_sha256")
                .and_then(|value| value.as_str())
                == Some(dataset)
                && receipt.get("cut_id").and_then(|value| value.as_str()) == Some(cut);
        }
    }
    ensure!(
        !burned,
        "partition labels were consumed by optimization and are development data"
    );
    Ok(())
}

/// Historical label/burn classification only. This helper never issues v2 global authority.
pub fn effective_label_role(original: &str, optimization_visible: bool) -> Result<&'static str> {
    match (original, optimization_visible) {
        ("sealed" | "meta_test", _) => bail!("sealed labels cannot enter optimization or teaching"),
        ("train", _) => Ok("train"),
        ("development" | "validation", _) | ("test", true) => Ok("development"),
        ("test", false) => Ok("test"),
        _ => bail!("unknown label role"),
    }
}

/// Purpose-bound companion to historical replay revalidation. A Qualification's role strings
/// and accepted HWM decision do not constitute exposure proof. Every subject and its context
/// are revalidated before this companion returns anything to a global consumer.
pub fn qualify_for(
    state: &State,
    record: &HwmRecord,
    evidence: &OofEvidence,
    verifier: &crate::replay::admission::AdmissionVerifier<'_>,
    proofs: &BTreeMap<String, String>,
    at_ms: u64,
) -> Result<(Qualification, crate::replay::admission::ReplayAdmission)> {
    use crate::replay::admission::Purpose;
    verifier.require_state(state)?;
    let qualified = qualify(state, record, evidence)?;
    let subject = canonical_sha256(&qualified)?;
    let proof = proofs
        .get(&subject)
        .context("HELD: qualification lacks exposure proof")?;
    let admission = verifier.admit(&subject, proof, Purpose::GlobalPolicyTraining, None, at_ms)?;
    context::verify_global_exposure(
        state,
        &qualified,
        verifier,
        Purpose::GlobalPolicyTraining,
        proofs,
        at_ms,
    )?;
    Ok((qualified, admission))
}

/// Revalidate an accepted historical set and its independently admitted exposure closure.
pub fn revalidate_replay_for(
    state: &State,
    entries: &[Qualification],
    verifier: &crate::replay::admission::AdmissionVerifier<'_>,
    purpose: crate::replay::admission::Purpose,
    proofs: &BTreeMap<String, String>,
    at_ms: u64,
) -> Result<Vec<crate::replay::admission::ReplayAdmission>> {
    ensure!(
        purpose.is_global(),
        "global replay requires an explicit global purpose"
    );
    verifier.require_state(state)?;
    // Historical guards remain necessary, independently of the new role/exposure boundary.
    revalidate_replay(state, entries)?;
    let mut admitted = Vec::new();
    for entry in entries {
        let subject = canonical_sha256(entry)?;
        let proof = proofs
            .get(&subject)
            .context("HELD: qualification lacks exposure proof")?;
        let admission = verifier.admit(&subject, proof, purpose, None, at_ms)?;
        context::verify_global_exposure(state, entry, verifier, purpose, proofs, at_ms)?;
        admitted.push(admission);
    }
    Ok(admitted)
}

/// Current-pointer companion; legacy manifests remain historical compatibility records.
pub fn freeze_teaching_for(
    state: &State,
    verifier: &crate::replay::admission::AdmissionVerifier<'_>,
    purpose: crate::replay::admission::Purpose,
    proofs: &BTreeMap<String, String>,
    at_ms: u64,
) -> Result<Vec<crate::replay::admission::ReplayAdmission>> {
    let manifest = freeze_teaching(state)?;
    revalidate_replay_for(state, &manifest.entries, verifier, purpose, proofs, at_ms)
}

pub(super) fn task(identity: &HwmIdentity) -> Result<Task> {
    let context = identity
        .comparison
        .as_ref()
        .context("legacy comparison is diagnostic only")?;
    ensure!(
        context.numerical_policy == NUMERICAL_POLICY
            && context.stage == "train_oof"
            && context.aggregation == "mean_fold_loss"
            && context.label_role == "development"
            && context.recipe_encoding == "native-canonical-dsl-sha256/v1"
            && !context.sealed
            && !context.benchmark_valid,
        "numerical policy differs from scorer"
    );
    ensure!(
        is_hex64(&identity.evaluator_sha256)
            && is_hex64(&identity.vocabulary_sha256)
            && is_hex64(&identity.dataset_hash)
            && is_hex64(&identity.cut_id)
            && !identity.protocol.trim().is_empty(),
        "incomplete comparison identity"
    );
    ensure!(
        context
            .checkpoint_sha256
            .as_ref()
            .is_none_or(|sha| is_hex64(sha)),
        "invalid checkpoint identity"
    );
    match identity.task.as_str() {
        "regression" => Ok(Task::Regression),
        "binary" => Ok(Task::Binary),
        "multiclass" => Ok(Task::Multiclass),
        _ => bail!("task has no admitted OOF scorer"),
    }
}

pub(super) fn blob<T: for<'de> Deserialize<'de>>(state: &State, sha: &str) -> Result<T> {
    ensure!(is_hex64(sha), "invalid artifact identity");
    let bytes = state
        .read_blob(sha)?
        .context("required prediction/target artifact is missing")?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub(super) fn require_blob(state: &State, sha: &str) -> Result<()> {
    ensure!(is_hex64(sha), "invalid fitted artifact identity");
    ensure!(
        !state
            .read_blob(sha)?
            .context("required fitted evidence is missing")?
            .is_empty(),
        "empty fitted evidence"
    );
    Ok(())
}

pub(super) fn freeze_path(state: &State, identity: &str) -> Result<std::path::PathBuf> {
    ensure!(is_hex64(identity), "invalid comparison digest");
    Ok(state
        .root()
        .join("eligibility/freezes")
        .join(identity)
        .join("freeze.json"))
}

pub(super) fn immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::create_dir_all(path.parent().context("immutable output parent missing")?)?;
    if path.exists() {
        ensure!(
            fs::read(path)? == bytes,
            "immutable eligibility evidence already differs"
        );
        return Ok(());
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::File::open(path.parent().unwrap())?.sync_all()?;
    Ok(())
}

pub(super) fn fold_rows(rows: &[TargetRow], fold: u8) -> (Vec<&TargetRow>, Vec<&TargetRow>) {
    rows.iter().partition(|row| row.fold != fold)
}

pub(super) fn rows_sha(rows: &[&TargetRow]) -> Result<String> {
    canonical_sha256(&rows.iter().map(|row| &row.id).collect::<Vec<_>>())
}

pub(super) fn baseline(
    task: Task,
    classes: &[String],
    fitting: &[&TargetRow],
    n: usize,
) -> Result<Predictions> {
    ensure!(!fitting.is_empty() && n > 0, "empty OOF partition");
    match task {
        Task::Regression => {
            let ys = fitting
                .iter()
                .map(|row| row.target.parse::<f64>().context("invalid target"))
                .collect::<Result<Vec<_>>>()?;
            ensure!(ys.iter().all(|y| y.is_finite()), "nonfinite target");
            let mean = ys.iter().map(|y| y / ys.len() as f64).sum::<f64>();
            ensure!(mean.is_finite(), "nonfinite trivial mean");
            Ok(Predictions::Values(vec![mean; n]))
        }
        Task::Binary | Task::Multiclass => {
            let mut counts = vec![0usize; classes.len()];
            for row in fitting {
                let index = classes
                    .iter()
                    .position(|class| class == &row.target)
                    .context("unknown training class")?;
                counts[index] += 1;
            }
            ensure!(
                counts.iter().all(|count| *count > 0),
                "training fold lacks a frozen class"
            );
            let probabilities = counts
                .iter()
                .map(|count| *count as f64 / fitting.len() as f64)
                .collect();
            Ok(Predictions::Probabilities {
                classes: classes.to_vec(),
                values: vec![probabilities; n],
            })
        }
        _ => bail!("unsupported trivial baseline task"),
    }
}

/// Call before optimization. Repeating the same plan retains the original receipt/time;
/// no plan is reconstructed from a candidate record, and differing plans cannot overwrite it.
pub fn freeze_optimization(state: &State, plan: &FreezePlan) -> Result<FreezeReceipt> {
    freeze_impl(state, plan, None)
}
pub(super) fn freeze_impl(
    state: &State,
    plan: &FreezePlan,
    inner_partition: Option<InnerPartition>,
) -> Result<FreezeReceipt> {
    let selected_task = task(&plan.comparison)?;
    if let Some(context) = &plan.context {
        context.validate()?;
    }
    let classes = &plan.comparison.comparison.as_ref().unwrap().class_order;
    let label_role = effective_label_role(&plan.original_label_role, true)?;
    ensure!(
        label_role == plan.comparison.comparison.as_ref().unwrap().label_role,
        "frozen label role differs from comparison identity"
    );
    let identity = plan.comparison.sha256()?;
    let (dataset, rows) = match &inner_partition {
        Some(inner) => inner::authenticated_inner_rows(state, plan, inner)?,
        None => authenticated_rows(state, plan)?,
    };
    let plan_sha256 = match &inner_partition {
        Some(inner) => canonical_sha256(&(plan, inner))?,
        None => canonical_sha256(plan)?,
    };
    let path = freeze_path(state, &identity)?;
    let _hwm_lock = state.lock("hwm")?;
    let _lock = state.lock("eligibility")?;
    Ledger::open(state.root(), "eligibility").require_intact()?;
    if path.exists() {
        let bytes = fs::read(&path)?;
        let existing: FrozenPolicy = serde_json::from_slice(&bytes)?;
        ensure!(
            existing.schema == FREEZE && existing.plan_sha256 == plan_sha256,
            "optimization freeze is already pinned to different inputs"
        );
        let receipt = FreezeReceipt {
            comparison_sha256: identity,
            freeze_sha256: sha256_hex(&bytes),
        };
        load_freeze(state, &receipt)?;
        state.put_blob(&bytes)?;
        return Ok(receipt);
    }
    ensure!(
        !state
            .pointers()?
            .iter()
            .any(|(key, _)| key.identity_sha256.as_ref() == Some(&identity)),
        "cannot freeze policy after an incumbent exists"
    );
    let mut ids = BTreeSet::new();
    let mut groups = BTreeMap::new();
    ensure!(!rows.is_empty(), "no training rows");
    for row in &rows {
        ensure!(
            !row.id.is_empty() && ids.insert(&row.id) && !row.group.is_empty() && row.fold < 5,
            "invalid/duplicate OOF row"
        );
        ensure!(
            *groups.entry(&row.group).or_insert(row.fold) == row.fold,
            "group crosses OOF folds"
        );
    }
    let mut folds = Vec::new();
    for fold in 0..5 {
        let (train, hold) = fold_rows(&rows, fold);
        let predictions = baseline(selected_task, classes, &train, hold.len())?;
        let targets = hold
            .iter()
            .map(|row| row.target.clone())
            .collect::<Vec<_>>();
        score_predictions(selected_task, classes, &targets, &predictions)?;
        folds.push(FrozenFold {
            fold,
            training_rows_sha256: rows_sha(&train)?,
            scoring_rows_sha256: rows_sha(&hold)?,
            baseline_predictions_sha256: state.put_blob(&canonical_json(&predictions)?)?,
        });
    }
    let label_burn_sha256 = if plan.original_label_role == "test" {
        let bytes = canonical_json(&serde_json::json!({"schema":"jope.label-burn/v1",
            "dataset_sha256":plan.comparison.dataset_hash, "cut_id":dataset.split.sha256,
            "panel_sha256":plan.panel_sha256, "development_sha256":dataset.split.development_sha256,
            "original_role":"test", "effective_role":"development"}))?;
        immutable_write(
            &burn_path(state, &plan.comparison.dataset_hash, &dataset.split.sha256)?,
            &bytes,
        )?;
        Some(state.put_blob(&bytes)?)
    } else {
        None
    };
    if let Some(inner) = &inner_partition {
        state.put_blob(&canonical_json(&inner.fitting_rows)?)?;
    }
    let policy = FrozenPolicy {
        schema: FREEZE.into(),
        plan_sha256,
        comparison: plan.comparison.clone(),
        label_role: label_role.into(),
        panel_sha256: plan.panel_sha256.clone(),
        dataset,
        original_label_role: plan.original_label_role.clone(),
        label_burn_sha256,
        guard: GuardPolicy::current(),
        evaluation_seed: plan.evaluation_seed,
        targets_sha256: state.put_blob(&canonical_json(&rows)?)?,
        folds,
        created_at_unix: now_unix(),
        inner_partition,
        context: plan.context.clone(),
    };
    let bytes = canonical_json(&policy)?;
    immutable_write(&path, &bytes)?;
    state.put_blob(&bytes)?;
    Ledger::open(state.root(), "eligibility").append(
        &[serde_json::json!({
            "outcome":"frozen", "comparison_sha256":identity, "freeze_sha256":sha256_hex(&bytes),
            "original_label_role":policy.original_label_role, "label_role":policy.label_role,
            "label_burn_sha256":policy.label_burn_sha256, "guard":policy.guard
        })],
        serde_json::json!({"source":"controller-owned-preoptimization-freeze"}),
    )?;
    Ok(FreezeReceipt {
        comparison_sha256: identity,
        freeze_sha256: sha256_hex(&bytes),
    })
}

/// W3-T04: the fitting input a fit worker receives for `fold` of the frozen dataset, recomputed
/// from its authenticated prepared development bytes (parsed once, and only when needed).
pub(super) fn fold_fitting_input(
    state: &State,
    dataset: &PanelDataset,
    development: &mut Option<DevelopmentData>,
    fold: u8,
) -> Result<String> {
    if development.is_none() {
        let bytes = state
            .read_blob(&dataset.split.development_sha256)?
            .context("prepared development bytes unavailable")?;
        *development = Some(parse_development(&bytes, dataset)?);
    }
    match development {
        Some(data) => data.cut(fold)?.train.fitting_input_sha256(),
        None => bail!("prepared development bytes unavailable"),
    }
}

pub(super) fn load_freeze(state: &State, receipt: &FreezeReceipt) -> Result<FrozenPolicy> {
    ensure!(is_hex64(&receipt.freeze_sha256), "invalid freeze receipt");
    let bytes = fs::read(freeze_path(state, &receipt.comparison_sha256)?)?;
    ensure!(
        sha256_hex(&bytes) == receipt.freeze_sha256,
        "independent optimization freeze changed"
    );
    let policy: FrozenPolicy = serde_json::from_slice(&bytes)?;
    let ledger = Ledger::open(state.root(), "eligibility");
    ledger.require_intact()?;
    ensure!(
        ledger.events()?.iter().any(
            |event| event.get("outcome").and_then(|value| value.as_str()) == Some("frozen")
                && event.get("freeze_sha256").and_then(|value| value.as_str())
                    == Some(&receipt.freeze_sha256)
        ),
        "freeze has no durable preoptimization ledger receipt"
    );
    ensure!(
        policy.schema == FREEZE
            && policy.comparison.sha256()? == receipt.comparison_sha256
            && matches!(policy.label_role.as_str(), "train" | "development"),
        "invalid frozen comparison policy"
    );
    Ok(policy)
}
