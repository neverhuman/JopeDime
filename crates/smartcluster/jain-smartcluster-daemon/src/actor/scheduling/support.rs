use super::*;

pub(super) fn prelaunch_rejection(
    job: &JobRecord,
    message: String,
    at: Timestamp,
) -> TransitionMutation {
    TransitionMutation {
        transition: FencedTransition {
            job_id: job.id,
            from: job.state,
            to: JobState::Failed,
            attempt: None,
            at,
            reason: Some(message.clone()),
        },
        terminal: Some(TerminalStatus::Failed {
            class: FailureClass::Dependency,
            message,
            exit_code: None,
            at,
        }),
        failure: None,
    }
}

pub(super) fn completion_ready(
    leader_exited: bool,
    tree_drained: bool,
    output_drains_pending: u8,
    control_drained: bool,
) -> bool {
    leader_exited && tree_drained && output_drains_pending == 0 && control_drained
}

pub(super) fn known_jain_gpu_compatibility(
    known_jain: bool,
    mps_shareable: bool,
    mig_inventory: bool,
) -> GpuCompatibility {
    GpuCompatibility {
        mps: known_jain && mps_shareable,
        mig: known_jain && mig_inventory,
        // Registration permits learning; the scheduler still requires five clean
        // exclusive runs before selecting shared MPS.
        trusted: known_jain,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_registered_jain_templates_can_unlock_gpu_sharing() {
        let known = known_jain_gpu_compatibility(true, true, true);
        assert!(known.trusted);
        assert!(known.mps);
        assert!(known.mig);

        let administrator_argv = known_jain_gpu_compatibility(false, true, true);
        assert!(!administrator_argv.trusted);
        assert!(!administrator_argv.mps);
        assert!(!administrator_argv.mig);
    }

    #[test]
    fn completion_waits_for_process_tree_and_every_output_reader() {
        assert!(!completion_ready(false, true, 0, true));
        assert!(!completion_ready(true, false, 0, true));
        assert!(!completion_ready(true, true, 1, true));
        assert!(!completion_ready(true, true, 0, false));
        assert!(completion_ready(true, true, 0, true));
    }
}
