#[cfg(test)]
mod tests {
    use super::*;

    fn request_for(role: GpuLaneRole) -> GpuLaneAdmissionV1 {
        GpuLaneAdmissionV1::new(
            role,
            "/tmp/spec".into(),
            Sha256Digest::parse("a".repeat(64)).unwrap(),
            format!("{}-job", role.as_str()),
            1,
        )
        .unwrap()
    }

    fn request() -> GpuLaneAdmissionV1 {
        request_for(GpuLaneRole::CohortProposer)
    }

    #[test]
    fn capacity_wait_does_not_enter_the_pause_transaction() {
        let state = request();
        for capacity in [
            NodeCapacity {
                node: "xbabe1".to_owned(),
                up: false,
                gpus_free: 1,
                free_core_budget: 4,
                gpu_companion_core_reserve: 4,
            },
            NodeCapacity {
                node: "xbabe1".to_owned(),
                up: true,
                gpus_free: 0,
                free_core_budget: 4,
                gpu_companion_core_reserve: 4,
            },
            NodeCapacity {
                node: "xbabe1".to_owned(),
                up: true,
                gpus_free: 1,
                free_core_budget: 3,
                gpu_companion_core_reserve: 4,
            },
        ] {
            assert!(!capacity_available(&state, &capacity));
        }
        assert!(capacity_available(
            &state,
            &NodeCapacity {
                node: "xbabe1".to_owned(),
                up: true,
                gpus_free: 1,
                free_core_budget: 4,
                gpu_companion_core_reserve: 4,
            }
        ));
        assert!(!capacity_available(
            &state,
            &NodeCapacity {
                node: "xbabe1".to_owned(),
                up: true,
                gpus_free: 1,
                free_core_budget: 4,
                gpu_companion_core_reserve: 0,
            }
        ));
    }

    #[test]
    fn submitted_recovery_requires_an_exact_archived_release() {
        let root = std::env::temp_dir().join(format!(
            "jope-gpu-release-recovery-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let completed = root.join("ops/gpu-lane-admission/completed");
        fs::create_dir_all(&completed).unwrap();
        let mut state = request();
        state.held_evaluator_ids = vec!["eval-a".to_owned(), "eval-b".to_owned()];
        state.stage = GpuAdmissionStage::Submitted;
        state.recovery_disposition = RecoveryDisposition::ConfirmSubmissionThenRelease;
        let mut sentinel = PauseSentinel::new(vec![GpuLaneRole::CohortProposer]);
        sentinel.held_evaluator_ids = state.held_evaluator_ids.clone();
        sentinel.released_evaluator_ids = vec!["eval-a".to_owned()];
        atomic_json(
            &completed.join("pause-1.json"),
            &serde_json::to_value(&sentinel).unwrap(),
        )
        .unwrap();
        assert!(archived_exact_release(&root, &state).unwrap().is_none());

        sentinel.released_evaluator_ids = state.held_evaluator_ids.clone();
        atomic_json(
            &completed.join("pause-2.json"),
            &serde_json::to_value(&sentinel).unwrap(),
        )
        .unwrap();
        assert_eq!(
            archived_exact_release(&root, &state).unwrap().unwrap(),
            state.held_evaluator_ids
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn low_runway_prioritizes_proposer_without_freezing_independent_roles() {
        let mut states = BTreeMap::new();
        for role in [
            GpuLaneRole::CohortProposer,
            GpuLaneRole::AlternatingTrainer,
            GpuLaneRole::BreadthEncoder,
        ] {
            states.insert(role, request_for(role));
        }
        let ready = states.keys().copied().collect::<Vec<_>>();
        let low = GpuArbitrationEvidence {
            candidate_evaluation_work: 100,
            candidate_evaluation_work_target: 200,
            protected_complete_waves: 1,
        };
        assert_eq!(
            select_gpu_roles(&states, &ready, low),
            vec![
                GpuLaneRole::CohortProposer,
                GpuLaneRole::AlternatingTrainer,
                GpuLaneRole::BreadthEncoder,
            ]
        );

        let green = GpuArbitrationEvidence {
            candidate_evaluation_work: 200,
            candidate_evaluation_work_target: 200,
            protected_complete_waves: 2,
        };
        assert_eq!(
            select_gpu_roles(&states, &ready, green),
            vec![
                GpuLaneRole::AlternatingTrainer,
                GpuLaneRole::CohortProposer,
                GpuLaneRole::BreadthEncoder,
            ]
        );
        states.remove(&GpuLaneRole::AlternatingTrainer);
        assert_eq!(
            select_gpu_roles(&states, &ready, green),
            vec![GpuLaneRole::CohortProposer, GpuLaneRole::BreadthEncoder]
        );
        states.remove(&GpuLaneRole::CohortProposer);
        assert_eq!(
            select_gpu_roles(&states, &ready, low),
            vec![GpuLaneRole::BreadthEncoder]
        );
    }

    #[test]
    fn node_policy_reserves_four_gpu_only_companion_cores() {
        let policy = include_str!("../../manifests/selfplay/scq-nodes.v3.toml");
        let assignment = |section: &str, key: &str| {
            section
                .lines()
                .find_map(|line| line.trim().strip_prefix(&format!("{key} = ")))
                .unwrap()
                .trim_matches('"')
                .to_owned()
        };
        let reserves = policy
            .split("[[node]]")
            .skip(1)
            .map(|section| {
                (
                    assignment(section, "name"),
                    assignment(section, "gpu_companion_core_reserve")
                        .parse::<i64>()
                        .unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for node in ["xbabe0", "xbabe1", "xbabe2", "xbabe3"] {
            assert_eq!(reserves.get(node), Some(&4));
        }
        assert_eq!(reserves.get("xbabe4"), Some(&0));
    }
}
