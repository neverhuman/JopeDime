#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> Sha256Digest {
        Sha256Digest::parse(character.to_string().repeat(64)).unwrap()
    }

    #[test]
    fn task_wire_names_and_conversions_are_stable() {
        for task in Task::ALL {
            assert_eq!(
                serde_json::to_value(task).unwrap(),
                serde_json::Value::String(task.as_str().to_owned())
            );
            assert_eq!(task.as_str().parse::<Task>().unwrap(), task);
            assert_eq!(crate::task::TaskKind::from(task), task);
        }
        assert_eq!(Task::Multiclass.corpus_dir_name(), "multi_label");
        assert_eq!(Task::Timeseries.corpus_dir_name(), "timeseries_reduced_v2");
        assert!("multi_label".parse::<Task>().is_err());
    }

    #[test]
    fn rotation_successor_with_legacy_eligibility_reproduces_strict_alternation() {
        let legacy = |task: Task| matches!(task, Task::Regression | Task::Binary);
        let mut position = 0;
        let mut picks = Vec::new();
        for _ in 0..8 {
            let (task, next) = Task::rotation_successor(position, legacy);
            picks.push(task);
            position = next;
        }
        // Exactly the pre-widening R,B alternation, for eight successive calls.
        assert_eq!(
            picks,
            vec![
                Task::Regression,
                Task::Binary,
                Task::Regression,
                Task::Binary,
                Task::Regression,
                Task::Binary,
                Task::Regression,
                Task::Binary,
            ]
        );
    }

    #[test]
    fn rotation_successor_visits_every_eligible_task_and_never_deadlocks() {
        let mut position = 0;
        let mut picks = Vec::new();
        for _ in 0..8 {
            let (task, next) = Task::rotation_successor(position, |_| true);
            picks.push(task);
            position = next;
        }
        assert_eq!(picks, Task::ROTATION_CYCLE.to_vec());

        // Starvation-skip: with multiclass still ineligible the cycle covers
        // the other three tasks and returns to the top.
        let no_multiclass = |task: Task| task != Task::Multiclass;
        let mut position = 4;
        let mut picks = Vec::new();
        for _ in 0..4 {
            let (task, next) = Task::rotation_successor(position, no_multiclass);
            picks.push(task);
            position = next;
        }
        assert_eq!(
            picks,
            vec![
                Task::Timeseries,
                Task::Regression,
                Task::Binary,
                Task::Regression
            ]
        );

        // No eligible task at all: the current slot is returned unconditionally.
        let (task, next) = Task::rotation_successor(4, |_| false);
        assert_eq!(task, Task::Timeseries);
        assert_eq!(next, 5);
    }

    #[test]
    fn evaluation_key_is_protocol_exact() {
        let base = EvaluationKeyV2::new(
            Task::Binary,
            digest('a'),
            digest('b'),
            digest('c'),
            digest('d'),
        );
        let changed = EvaluationKeyV2::new(
            Task::Binary,
            digest('a'),
            digest('b'),
            digest('c'),
            digest('e'),
        );
        assert_ne!(base.id(), changed.id());
        assert!(base.same_semantic_pair(&changed));
    }

    #[test]
    fn cohort_proposal_pointer_scope_is_immutable_and_task_exact() {
        let pointer = CohortProposalPointerV1 {
            schema: COHORT_PROPOSAL_POINTER_SCHEMA.into(),
            task_scopes: vec![CohortProposalTaskScope {
                task: Task::Binary,
                accepted_checkpoint_sha256: digest('a'),
                context_count: 64,
            }],
            cohort_sha256: digest('b'),
            catalog_sha256: digest('c'),
            decoder_configuration_sha256: digest('d'),
            seed: 7,
            proposal_sha256: digest('e'),
            proposal_job_id: "cohort-proposer-binary-a".into(),
        };
        pointer
            .validate_scope(
                Task::Binary,
                &digest('a'),
                &digest('b'),
                &digest('c'),
                &digest('d'),
                7,
            )
            .unwrap();
        assert!(pointer
            .validate_scope(
                Task::Regression,
                &digest('a'),
                &digest('b'),
                &digest('c'),
                &digest('d'),
                7,
            )
            .is_err());
        assert!(pointer
            .validate_scope(
                Task::Binary,
                &digest('a'),
                &digest('f'),
                &digest('c'),
                &digest('d'),
                7,
            )
            .is_err());
        assert_eq!(pointer.id().unwrap(), pointer.id().unwrap());
    }

    #[test]
    fn shared_proposal_job_requires_a_shared_checkpoint() {
        let pointer = CohortProposalPointerV1 {
            schema: COHORT_PROPOSAL_POINTER_SCHEMA.into(),
            task_scopes: vec![
                CohortProposalTaskScope {
                    task: Task::Binary,
                    accepted_checkpoint_sha256: digest('a'),
                    context_count: 64,
                },
                CohortProposalTaskScope {
                    task: Task::Regression,
                    accepted_checkpoint_sha256: digest('b'),
                    context_count: 64,
                },
            ],
            cohort_sha256: digest('c'),
            catalog_sha256: digest('d'),
            decoder_configuration_sha256: digest('e'),
            seed: 9,
            proposal_sha256: digest('f'),
            proposal_job_id: "invalid-shared-job".into(),
        };
        assert!(pointer.validate().is_err());
    }

    #[test]
    fn uncertainty_has_zero_action_weight() {
        let mut signals = SelectionSignals {
            conditional_reward: 1.0,
            stability: 0.5,
            incumbent_lift_probability: 0.75,
            diversity: 0.2,
            raw_uncertainty: -100.0,
        };
        let score = signals.action_score();
        signals.raw_uncertainty = 100.0;
        assert_eq!(signals.action_score(), score);
    }

    #[test]
    fn cache_hits_are_not_new_evidence() {
        let receipt = EvaluationCacheReceiptV2::exact_hit(
            EvaluationKeyV2::new(
                Task::Regression,
                digest('a'),
                digest('b'),
                digest('c'),
                digest('d'),
            ),
            digest('e'),
        );
        receipt.validate().unwrap();
        assert!(!receipt.counts_as_action);
        assert!(!receipt.counts_as_evaluation);
        assert!(!receipt.counts_as_statistical_evidence);
    }

    #[test]
    fn origin_attribution_rejects_gp_descendants_of_dl() {
        let context = DatasetContext {
            task: Task::Binary,
            dataset_id: "dataset".into(),
            dataset_content_sha256: digest('a'),
            source_family: "family".into(),
            context_sha256: digest('b'),
        };
        let program = "(pipe X (ridge))".to_owned();
        let candidate = PolicyCandidate {
            task: Task::Binary,
            dataset_id: "dataset".into(),
            dataset_content_sha256: digest('a'),
            context_sha256: digest('b'),
            semantic_sha256: Sha256Digest::of_bytes(program.as_bytes()),
            canonical_program: program,
            source_family: "family".into(),
            origin: Origin::GpComparator,
            parentage: Parentage {
                parent_semantic_sha256: vec![digest('c')],
                has_dl_parent_or_ancestor: true,
                dl_ancestry_kind: DlAncestryKind::AcceptedDl,
            },
            producer_checkpoint_sha256: None,
            behavior_checkpoint_sha256: None,
            decoder_log_probability: None,
            proposal_rank: 0,
            seed: 1,
            signals: SelectionSignals {
                conditional_reward: 0.1,
                stability: 0.1,
                incumbent_lift_probability: 0.5,
                diversity: 0.1,
                raw_uncertainty: 0.1,
            },
        };
        assert!(candidate.validate_for(&context, &digest('d')).is_err());
    }
}
