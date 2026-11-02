use super::*;

pub(super) const MINIMUM_CANARY_VALIDITY: f64 = 0.99;
pub(super) const MAXIMUM_CANARY_LATENCY_RATIO: f64 = 1.25;
pub(super) const SHADOW_COMPARISON_LOWER_BOUND_METHOD: &str = "context_cluster_normal_95_v1";

pub(super) fn shadow_summary_eligible(summary: &Value) -> bool {
    let metric = summary.get("candidate_vs_accepted").unwrap_or(&Value::Null);
    summary
        .get("contexts")
        .and_then(Value::as_array)
        .is_some_and(|contexts| contexts.len() == 16)
        && summary
            .get("accepted_validity")
            .and_then(Value::as_f64)
            .is_some_and(f64::is_finite)
        && summary
            .get("gp_validity")
            .and_then(Value::as_f64)
            .is_some_and(f64::is_finite)
        && metric
            .get("validity")
            .and_then(Value::as_f64)
            .is_some_and(|candidate| candidate >= MINIMUM_CANARY_VALIDITY)
        && metric
            .get("pairwise_lift_lower_bound")
            .and_then(Value::as_f64)
            .is_some_and(|lower| lower.is_finite() && lower > 0.0)
        && summary
            .pointer("/candidate_vs_gp/pairwise_lift_lower_bound")
            .and_then(Value::as_f64)
            .is_some_and(|lower| lower.is_finite() && lower > 0.0)
        && metric
            .get("latency_ratio")
            .and_then(Value::as_f64)
            .is_some_and(|ratio| {
                ratio.is_finite() && ratio > 0.0 && ratio <= MAXIMUM_CANARY_LATENCY_RATIO
            })
        && metric
            .get("stability")
            .and_then(Value::as_f64)
            .is_some_and(|value| value >= 0.99)
        && summary
            .get("replay_verification_complete")
            .and_then(Value::as_bool)
            == Some(true)
        && summary
            .get("matched_comparisons_complete")
            .and_then(Value::as_bool)
            == Some(true)
        && summary
            .get("comparison_lower_bound_method")
            .and_then(Value::as_str)
            == Some(SHADOW_COMPARISON_LOWER_BOUND_METHOD)
}

/// Keys in a trainer recipe that vary run-to-run without changing what the
/// trainer actually DOES (corpus counters, evidence tallies, the parent the
/// plateau already keys on). Stripped before hashing so a recipe identity
/// names the training procedure, not the moment it ran.
pub(super) const RECIPE_VOLATILE_KEYS: &[&str] = &[
    "complete_parent_pairs",
    "completed_descendant_rows",
    "completed_descendant_receipts",
    "hard_negative_rows",
    "new_eligible_labels",
    "new_completed_descendant_receipts",
    "accepted_parent_checkpoint_sha256",
];

pub(super) fn recipe_identity_sha256(recipe: &Value) -> Result<String> {
    let object = recipe
        .as_object()
        .context("trainer recipe is not an object")?;
    let canonical = object
        .iter()
        .filter(|(key, _)| !RECIPE_VOLATILE_KEYS.contains(&key.as_str()))
        .collect::<BTreeMap<_, _>>();
    Ok(sha256_bytes(&serde_json::to_vec(&canonical)?))
}

/// Resolve the recipe identity a rejected candidate was trained with:
/// candidate checkpoint -> ShadowCheckpoint.v1 (trainer_intent_sha256) ->
/// content-addressed TrainerIntent receipt -> its recipe. None when any
/// link is missing (legacy receipts predating recipe stamping).
pub(super) fn resolve_candidate_recipe_identity(
    root: &Path,
    task: Task,
    candidate: &Sha256Digest,
) -> Result<Option<String>> {
    let checkpoint_doc = root
        .join("training/shadow-checkpoints")
        .join(task.as_str())
        .join(format!("{}.v1.json", candidate.as_str()));
    if !checkpoint_doc.is_file() {
        return Ok(None);
    }
    let checkpoint = read_json(&checkpoint_doc)?;
    let Some(intent_sha256) = checkpoint
        .get("trainer_intent_sha256")
        .and_then(Value::as_str)
    else {
        return Ok(None);
    };
    let intents = root.join("training/trainer-intents");
    if !intents.is_dir() {
        return Ok(None);
    }
    for entry in fs::read_dir(&intents)?.filter_map(|entry| entry.ok()) {
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("json")
            || sha256_file(&path)? != intent_sha256
        {
            continue;
        }
        let intent = read_json(&path)?;
        return match intent.get("recipe") {
            Some(recipe) if !recipe.is_null() => Ok(Some(recipe_identity_sha256(recipe)?)),
            _ => Ok(None),
        };
    }
    Ok(None)
}

/// A valid operator directive for (task, parent) at ops/recipe-changes/.
/// Malformed documents are treated as absent (the pause stays), never as
/// a bridge.
pub(super) fn recipe_change_directive(
    root: &Path,
    task: Task,
    parent: &Sha256Digest,
    current_recipe_identity: Option<&String>,
) -> bool {
    let diagnosis_path = root.join("training/plateau-diagnoses").join(format!(
        "{}-{}.v2.json",
        task.as_str(),
        parent.as_str()
    ));
    let v2_path = root.join("ops/recipe-changes").join(format!(
        "{}-{}.v2.json",
        task.as_str(),
        parent.as_str()
    ));
    if diagnosis_path.is_file() && v2_path.is_file() {
        let Ok(diagnosis_value) = read_json(&diagnosis_path) else {
            return false;
        };
        let Ok(diagnosis) = serde_json::from_value::<PlateauDiagnosisV2>(diagnosis_value) else {
            return false;
        };
        let Ok(value) = read_json(&v2_path) else {
            return false;
        };
        let Ok(directive) = serde_json::from_value::<RecipeChangeDirectiveV2>(value) else {
            return false;
        };
        let Ok(diagnosis_sha256) = sha256_file(&diagnosis_path).and_then(Sha256Digest::parse)
        else {
            return false;
        };
        let current_matches = current_recipe_identity
            .and_then(|identity| Sha256Digest::parse(identity.clone()).ok())
            .is_some_and(|identity| identity == directive.next_recipe_identity_sha256);
        let previous_matches = diagnosis
            .current_recipe_identity_sha256
            .as_ref()
            .and_then(|identity| Sha256Digest::parse(identity.clone()).ok())
            .map_or(
                directive.previous_recipe_identity_sha256.is_none(),
                |identity| directive.previous_recipe_identity_sha256.as_ref() == Some(&identity),
            );
        return diagnosis.validate().is_ok()
            && directive.validate(&diagnosis).is_ok()
            && directive.answers_diagnosis_sha256 == diagnosis_sha256
            && current_matches
            && previous_matches;
    }

    // V1 can bridge only pre-v2 legacy diagnoses. Once a v2 diagnosis exists,
    // an enumerated but unaudited free-form directive is never sufficient.
    if diagnosis_path.is_file() {
        return false;
    }
    let path = root.join("ops/recipe-changes").join(format!(
        "{}-{}.v1.json",
        task.as_str(),
        parent.as_str()
    ));
    read_json(&path)
        .ok()
        .and_then(|value| serde_json::from_value::<RecipeChangeDirectiveV1>(value).ok())
        .is_some_and(|directive| {
            directive.validate().is_ok()
                && directive.task == task
                && directive.unchanged_parent_checkpoint_sha256 == *parent
        })
}

pub(super) fn publish_plateau_diagnoses(
    root: &Path,
    accepted: &BTreeMap<Task, AcceptedPointer>,
    current_recipe_identities: &BTreeMap<Task, String>,
) -> Result<BTreeSet<Task>> {
    let directory = root.join("training/shadow-canaries/completed");
    let mut plateau_tasks = BTreeSet::new();
    if !directory.is_dir() {
        return Ok(plateau_tasks);
    }
    for task in Task::ALL {
        let parent = match accepted.get(&task) {
            Some(pointer) => &pointer.digest,
            // Legacy tasks must always have an accepted pointer; the new
            // tasks are skipped until their first anchor is promoted.
            None if matches!(task, Task::Binary | Task::Regression) => {
                bail!("accepted pointer lacks {task}")
            }
            None => continue,
        };
        let mut by_candidate = BTreeMap::<Sha256Digest, (u64, Value)>::new();
        for entry in fs::read_dir(&directory)?.filter_map(|entry| entry.ok()) {
            let Ok(value) = read_json(&entry.path()) else {
                continue;
            };
            if value.get("schema").and_then(Value::as_str) != Some("ShadowBattleSummary.v1")
                || value.get("mode").and_then(Value::as_str) != Some("shadow")
                || value.get("task").and_then(Value::as_str) != Some(task.as_str())
                || value
                    .get("accepted_parent_checkpoint_sha256")
                    .and_then(Value::as_str)
                    != Some(parent.as_str())
            {
                continue;
            }
            let Some(candidate) = value
                .get("candidate_checkpoint_sha256")
                .and_then(Value::as_str)
                .and_then(|digest| Sha256Digest::parse(digest.to_owned()).ok())
            else {
                continue;
            };
            let created = value
                .get("created_at_ms")
                .and_then(Value::as_u64)
                .unwrap_or_default();
            let replace = by_candidate
                .get(&candidate)
                .is_none_or(|(known, _)| created >= *known);
            if replace {
                by_candidate.insert(candidate, (created, value));
            }
        }
        let (eligible, mut rejected): (Vec<_>, Vec<_>) = by_candidate
            .into_iter()
            .partition(|(_, (_, summary))| shadow_summary_eligible(summary));
        rejected.sort_by_key(|(_, (created, _))| *created);
        // Recipe scoping: a rejected shadow counts toward THIS plateau only
        // if it was trained with the recipe the lane would use right now.
        // Unresolvable identities (legacy receipts) count conservatively,
        // unless an operator RecipeChangeDirective bridges them. An unknown
        // CURRENT identity disables scoping entirely (v1 behavior).
        let current_identity = current_recipe_identities.get(&task);
        let directive_present = recipe_change_directive(root, task, parent, current_identity);
        let scoped = |candidate: &Sha256Digest| -> Result<(bool, Option<String>)> {
            let identity = resolve_candidate_recipe_identity(root, task, candidate)?;
            let counts = match (&identity, current_identity) {
                (Some(candidate_identity), Some(current)) => candidate_identity == current,
                (Some(_), None) => true,
                (None, _) => !directive_present,
            };
            Ok((counts, identity))
        };
        // A lane that has a PASSING shadow on this parent is not plateaued,
        // however many were rejected alongside it. The diagnosis published below
        // asserts that the recipe is exhausted and must be changed; a candidate
        // that cleared every bar on the same parent, under the same recipe
        // identity, is direct evidence that it is not. Counting only rejections
        // let three historical losses pause a lane that already had a winner
        // sitting on it -- the loop then trains nothing, and the winner is never
        // chained, so the parent never changes and the plateau never clears.
        // The eligible candidate is held to the SAME recipe scoping as the
        // rejections, so a stale pass cannot unlatch a genuinely stuck lane.
        let mut passing = false;
        for (candidate, _) in &eligible {
            if scoped(candidate)?.0 {
                passing = true;
                break;
            }
        }
        if passing {
            continue;
        }
        let mut counted = Vec::new();
        for (candidate, (created, summary)) in rejected {
            let (counts, identity) = scoped(&candidate)?;
            if counts {
                counted.push((candidate, identity, created, summary));
            }
        }
        if counted.len() < 3 {
            continue;
        }
        let counted_slice = &counted[counted.len() - 3..];
        let selected = counted_slice
            .iter()
            .map(|(candidate, _, created, summary)| {
                (candidate.clone(), (*created, summary.clone()))
            })
            .collect::<Vec<_>>();
        let selected = &selected[..];
        let proposal_validity = selected
            .iter()
            .filter_map(|(_, (_, summary))| {
                summary.pointer("/candidate_vs_accepted/validity")?.as_f64()
            })
            .fold(1.0, f64::min);
        let gp_disadvantage = selected
            .iter()
            .filter_map(|(_, (_, summary))| {
                summary.pointer("/candidate_vs_gp/pairwise_lift")?.as_f64()
            })
            .map(|lift| (-lift).max(0.0))
            .fold(0.0, f64::max);
        let latency = selected
            .iter()
            .filter_map(|(_, (_, summary))| {
                summary
                    .pointer("/candidate_vs_accepted/latency_ratio")?
                    .as_f64()
            })
            .fold(0.0, f64::max);
        let replayed = selected
            .iter()
            .filter(|(_, (_, summary))| {
                summary
                    .get("replay_verification_complete")
                    .and_then(Value::as_bool)
                    == Some(true)
            })
            .count();
        let diagnosis = PlateauDiagnosisV1 {
            task,
            unchanged_parent_checkpoint_sha256: parent.clone(),
            rejected_candidates: selected
                .iter()
                .map(|(candidate, _)| candidate.clone())
                .collect(),
            proposal_validity,
            gp_disadvantage,
            calibration_summary:
                "learned uncertainty/risk ranking remains disabled until its family-disjoint calibration gate passes"
                    .to_owned(),
            data_slice_summary:
                "the recycled 16-context diagnostic has no authoritative provenance-v2 slice result; single-use external panels are required"
                    .to_owned(),
            latency_summary: format!("worst candidate/accepted latency ratio {latency:.6}"),
            replay_summary: format!("{replayed}/3 rejected shadow receipts replay-verified"),
            diagnosed_at_ms: now_ms(),
        };
        diagnosis.validate()?;
        let path = root.join("training/plateau-diagnoses").join(format!(
            "{}-{}.v1.json",
            task.as_str(),
            parent.as_str()
        ));
        if path.is_file() {
            let existing: PlateauDiagnosisV1 = serde_json::from_value(read_json(&path)?)?;
            existing.validate()?;
            ensure!(
                existing.unchanged_parent_checkpoint_sha256 == *parent,
                "plateau diagnosis parent drift"
            );
        } else {
            atomic_json(&path, &serde_json::to_value(&diagnosis)?)?;
        }
        let v2 = PlateauDiagnosisV2 {
            schema: PLATEAU_DIAGNOSIS_V2_SCHEMA.to_owned(),
            task,
            unchanged_parent_checkpoint_sha256: parent.clone(),
            current_recipe_identity_sha256: current_identity.cloned(),
            counted_candidates: counted_slice
                .iter()
                .map(|(candidate, identity, _, _)| PlateauCountedCandidateV1 {
                    candidate_checkpoint_sha256: candidate.clone(),
                    recipe_identity_sha256: identity.clone(),
                })
                .collect(),
            recipe_change_directive_present: directive_present,
            proposal_validity: diagnosis.proposal_validity,
            gp_disadvantage: diagnosis.gp_disadvantage,
            calibration_summary: diagnosis.calibration_summary.clone(),
            data_slice_summary: diagnosis.data_slice_summary.clone(),
            latency_summary: diagnosis.latency_summary.clone(),
            replay_summary: diagnosis.replay_summary.clone(),
            failure_classes: [
                (proposal_validity < 0.99).then_some(PlateauFailureClassV2::ValidityLoss),
                (gp_disadvantage > 0.0).then_some(PlateauFailureClassV2::GpDisadvantage),
                (latency > 1.25).then_some(PlateauFailureClassV2::LatencyRegression),
            ]
            .into_iter()
            .flatten()
            .collect(),
            diagnosed_at_ms: diagnosis.diagnosed_at_ms,
        };
        v2.validate()?;
        let v2_path = root.join("training/plateau-diagnoses").join(format!(
            "{}-{}.v2.json",
            task.as_str(),
            parent.as_str()
        ));
        if v2_path.is_file() {
            let existing: PlateauDiagnosisV2 = serde_json::from_value(read_json(&v2_path)?)?;
            existing.validate()?;
            ensure!(
                existing.unchanged_parent_checkpoint_sha256 == *parent,
                "plateau diagnosis v2 parent drift"
            );
        } else {
            atomic_json(&v2_path, &serde_json::to_value(&v2)?)?;
        }
        plateau_tasks.insert(task);
    }
    Ok(plateau_tasks)
}
