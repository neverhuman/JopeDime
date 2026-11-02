/// The live deadlock this pins, measured on the production root on
/// 2026-08-04: the regression lane held three rejected shadows AND one that
/// cleared every bar (`8cadebd2`, lift +0.1748, stability 1.0, validity 1.0)
/// on the same unchanged parent. Counting only rejections declared the recipe
/// exhausted, so the controller trained nothing, the winner was never
/// chained, the parent never moved -- and the plateau that the winner
/// disproved kept the lane paused indefinitely.
#[test]
fn a_passing_shadow_on_the_same_parent_defeats_the_plateau() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let parent = Sha256Digest::parse("a".repeat(64)).unwrap();
    let accepted = binary_accepted(&parent);
    for (index, character) in ["1", "2", "3"].into_iter().enumerate() {
        write_reject_summary(root, &parent, &character.repeat(64), index as u64 + 1);
    }
    // Three rejections alone are a plateau -- this is the existing contract.
    assert_eq!(
        publish_plateau_diagnoses(root, &accepted, &BTreeMap::new()).unwrap(),
        BTreeSet::from([Task::Binary])
    );

    // One passing shadow on the same parent withdraws the diagnosis.
    write_passing_summary(root, &parent, &"4".repeat(64), 4);
    assert_eq!(
        publish_plateau_diagnoses(root, &accepted, &BTreeMap::new()).unwrap(),
        BTreeSet::new(),
        "a lane with a winner on the current parent must not be plateaued"
    );
}

/// The eligible candidate is scoped exactly like the rejections, so a pass
/// recorded under a DIFFERENT recipe than the lane would use now cannot
/// unlatch a genuinely stuck lane.
#[test]
fn a_pass_from_a_different_recipe_does_not_defeat_the_plateau() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let parent = Sha256Digest::parse("a".repeat(64)).unwrap();
    let accepted = binary_accepted(&parent);
    let current = json!({"recipe_name": "e", "exact_causal_pair_fraction": 1.0});
    let stale = json!({"recipe_name": "a", "broad_replay_fraction": 1.0});
    for (index, character) in ["1", "2", "3"].into_iter().enumerate() {
        let candidate = character.repeat(64);
        write_reject_summary(root, &parent, &candidate, index as u64 + 1);
        install_candidate_recipe(root, &candidate, &current);
    }
    let winner = "4".repeat(64);
    write_passing_summary(root, &parent, &winner, 4);
    install_candidate_recipe(root, &winner, &stale);

    let identity = recipe_identity_sha256(&current).unwrap();
    assert_eq!(
        publish_plateau_diagnoses(root, &accepted, &BTreeMap::from([(Task::Binary, identity)]))
            .unwrap(),
        BTreeSet::from([Task::Binary]),
        "a pass under a superseded recipe must not clear the plateau"
    );
}

fn install_candidate_recipe(root: &Path, candidate: &str, recipe: &Value) {
    let intents = root.join("training/trainer-intents");
    fs::create_dir_all(&intents).unwrap();
    let intent_path = intents.join(format!("intent-{}.v3.json", &candidate[..8]));
    atomic_json(
        &intent_path,
        &json!({"schema": "TrainerIntent.v3", "recipe": recipe}),
    )
    .unwrap();
    let checkpoint_dir = root.join("training/shadow-checkpoints/binary");
    fs::create_dir_all(&checkpoint_dir).unwrap();
    atomic_json(
        &checkpoint_dir.join(format!("{candidate}.v1.json")),
        &json!({"trainer_intent_sha256": sha256_file(&intent_path).unwrap()}),
    )
    .unwrap();
}

fn binary_accepted(parent: &Sha256Digest) -> BTreeMap<Task, AcceptedPointer> {
    BTreeMap::from([
        (
            Task::Binary,
            AcceptedPointer {
                digest: parent.clone(),
                path: PathBuf::new(),
            },
        ),
        (
            Task::Regression,
            AcceptedPointer {
                digest: Sha256Digest::parse("b".repeat(64)).unwrap(),
                path: PathBuf::new(),
            },
        ),
    ])
}

#[test]
fn a_changed_recipe_reschedules_training_instead_of_pausing() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let parent = Sha256Digest::parse("a".repeat(64)).unwrap();
    let old_recipe = json!({
        "recipe_name": "a",
        "broad_replay_fraction": 1.0,
        "complete_parent_pairs": 17,
    });
    for (index, character) in ["1", "2", "3"].into_iter().enumerate() {
        let candidate = character.repeat(64);
        write_reject_summary(root, &parent, &candidate, index as u64 + 1);
        install_candidate_recipe(root, &candidate, &old_recipe);
    }
    let accepted = binary_accepted(&parent);
    // Same recipe as the rejects (volatile counters differ): still pauses.
    let same = recipe_identity_sha256(&json!({
        "recipe_name": "a",
        "broad_replay_fraction": 1.0,
        "complete_parent_pairs": 99,
    }))
    .unwrap();
    let plateau =
        publish_plateau_diagnoses(root, &accepted, &BTreeMap::from([(Task::Binary, same)]))
            .unwrap();
    assert_eq!(plateau, BTreeSet::from([Task::Binary]));
    // A genuinely different recipe re-schedules training for the task.
    let changed = recipe_identity_sha256(&json!({
        "recipe_name": "d",
        "broad_replay_fraction": 0.75,
        "exact_causal_pair_fraction": 0.25,
    }))
    .unwrap();
    fs::remove_dir_all(root.join("training/plateau-diagnoses")).unwrap();
    let plateau =
        publish_plateau_diagnoses(root, &accepted, &BTreeMap::from([(Task::Binary, changed)]))
            .unwrap();
    assert!(plateau.is_empty());
    assert!(!root.join("training/plateau-diagnoses").is_dir());
}

#[test]
fn a_recipe_change_directive_v2_bridges_only_the_exact_recipe_plateau() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let parent = Sha256Digest::parse("a".repeat(64)).unwrap();
    let old_recipe = "c".repeat(64);
    let changed_recipe = "d".repeat(64);
    for (index, character) in ["1", "2", "3"].into_iter().enumerate() {
        // No shadow-checkpoint/intent fixtures: identities unresolvable.
        write_reject_summary(root, &parent, &character.repeat(64), index as u64 + 1);
    }
    let accepted = binary_accepted(&parent);
    let directive_dir = root.join("ops/recipe-changes");
    fs::create_dir_all(&directive_dir).unwrap();
    let directive_path = directive_dir.join(format!("binary-{}.v1.json", parent.as_str()));
    // A malformed directive (no enumerated changes) is treated as absent.
    atomic_json(
        &directive_path,
        &json!({
            "schema": "RecipeChangeDirective.v1",
            "task": "binary",
            "unchanged_parent_checkpoint_sha256": parent,
            "answers_diagnosis": format!("binary-{}.v1.json", parent.as_str()),
            "enumerated_changes": [],
            "created_at_ms": 2,
        }),
    )
    .unwrap();
    let plateau = publish_plateau_diagnoses(
        root,
        &accepted,
        &BTreeMap::from([(Task::Binary, old_recipe.clone())]),
    )
    .unwrap();
    assert_eq!(plateau, BTreeSet::from([Task::Binary]));
    // Once the controller publishes a v2 diagnosis, even a valid free-form
    // v1 directive cannot release the pause.
    atomic_json(
        &directive_path,
        &json!({
            "schema": "RecipeChangeDirective.v1",
            "task": "binary",
            "unchanged_parent_checkpoint_sha256": parent,
            "answers_diagnosis": format!("binary-{}.v1.json", parent.as_str()),
            "enumerated_changes": [
                "ShadowBattleSummary.v2: median latency gate vs matched GP comparators",
            ],
            "created_at_ms": 3,
        }),
    )
    .unwrap();
    let plateau = publish_plateau_diagnoses(
        root,
        &accepted,
        &BTreeMap::from([(Task::Binary, changed_recipe.clone())]),
    )
    .unwrap();
    assert_eq!(plateau, BTreeSet::from([Task::Binary]));

    // The v2 directive binds the exact immutable diagnosis, old and new
    // recipe identities, failure classes, and their required knobs.
    let diagnosis_path = root
        .join("training/plateau-diagnoses")
        .join(format!("binary-{}.v2.json", parent.as_str()));
    let v2_path = directive_dir.join(format!("binary-{}.v2.json", parent.as_str()));
    atomic_json(
        &v2_path,
        &json!({
            "schema": "RecipeChangeDirective.v2",
            "task": "binary",
            "unchanged_parent_checkpoint_sha256": parent,
            "answers_diagnosis_sha256": sha256_file(&diagnosis_path).unwrap(),
            "previous_recipe_identity_sha256": old_recipe,
            "next_recipe_identity_sha256": changed_recipe.clone(),
            "failure_classes": ["validity_loss", "gp_disadvantage"],
            "changed_knobs": [
                "hard_negative_validity_sampling",
                "exact_causal_pair_gp_comparator_mixture"
            ],
            "created_at_ms": 4,
        }),
    )
    .unwrap();
    let plateau = publish_plateau_diagnoses(
        root,
        &accepted,
        &BTreeMap::from([(Task::Binary, changed_recipe)]),
    )
    .unwrap();
    assert!(plateau.is_empty());
}

#[test]
fn recipe_identity_ignores_volatile_counters_only() {
    let base = json!({
        "recipe_name": "d",
        "broad_replay_fraction": 0.75,
        "complete_parent_pairs": 32,
        "new_eligible_labels": 51_000,
        "accepted_parent_checkpoint_sha256": "a".repeat(64),
    });
    let counters_shifted = json!({
        "recipe_name": "d",
        "broad_replay_fraction": 0.75,
        "complete_parent_pairs": 640,
        "new_eligible_labels": 1,
        "accepted_parent_checkpoint_sha256": "c".repeat(64),
    });
    let fraction_changed = json!({
        "recipe_name": "d",
        "broad_replay_fraction": 0.50,
        "complete_parent_pairs": 32,
    });
    let base_identity = recipe_identity_sha256(&base).unwrap();
    assert_eq!(
        base_identity,
        recipe_identity_sha256(&counters_shifted).unwrap()
    );
    assert_ne!(
        base_identity,
        recipe_identity_sha256(&fraction_changed).unwrap()
    );
}
