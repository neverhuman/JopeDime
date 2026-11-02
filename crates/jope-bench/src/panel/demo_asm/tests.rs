use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn warm_wrapper_does_not_extend_global_deadline_or_reporting_budget() {
        let mut resources = Resources {
            workers: 20,
            cpus: 20,
            memory_bytes: 64u64 << 30,
            swap_bytes: 0,
            gpu_tasks: 0,
            threads_per_worker: 1,
            total_seconds: 60,
            reporting_seconds: 5,
            per_fit_seconds: 30,
        };
        warm_limits(&resources).unwrap();
        warm_pair_limits(&resources, 20).unwrap();
        assert!(warm_pair_limits(&resources, 28).is_err());
        assert!(warm_pair_limits(&resources, 120).is_err());
        assert!(warm_pair_limits(&resources, 0).is_err());
        resources.total_seconds = 300;
        warm_limits(&resources).unwrap();
        warm_pair_limits(&resources, 120).unwrap();
        resources.total_seconds = 240;
        assert!(warm_pair_limits(&resources, 120).is_err());
        resources.total_seconds = 7201;
        assert!(warm_limits(&resources).is_err());
        resources.total_seconds = 60;
        resources.reporting_seconds = 4;
        assert!(warm_limits(&resources).is_err());
        resources.reporting_seconds = 5;
        resources.gpu_tasks = 1;
        assert!(warm_limits(&resources).is_err());
    }

    #[test]
    fn warm_latency_target_is_measured_separately_from_completed_execution() {
        assert!(latency_target_met(true, 59.9));
        for elapsed in [60., 61., 230.13, f64::NAN, -1.] {
            assert!(!latency_target_met(true, elapsed));
        }
        assert!(!latency_target_met(false, 20.));
    }

    fn allowance(role: &'static str, start: u64, cutoff: u64) -> AvailableArmBudget {
        let arm = ArmBudget {
            id_sha256: "a".repeat(64),
            seconds: 20,
        };
        let clock = ArmClock {
            id_sha256: arm.id_sha256.clone(),
            seconds: arm.seconds,
            started_unix_ms: start,
            deadline_unix_ms: start + 20000,
        };
        observed_arm_budget(role, &arm, &clock, cutoff.min(clock.deadline_unix_ms)).unwrap()
    }
    #[test]
    fn warm_comparison_rejects_unequal_and_equally_clipped_allowances() {
        let complete = [
            allowance("neural", 1000, 56000),
            allowance("generic", 24000, 56000),
        ];
        assert!(matched_full_allowances(&complete));
        assert!(!matched_full_allowances(&[]));
        assert!(!matched_full_allowances(&complete[..1]));
        let mut unequal = [
            allowance("neural", 1000, 56000),
            allowance("generic", 24000, 56000),
        ];
        unequal[1].declared_seconds = 21;
        assert!(!matched_full_allowances(&unequal));
        assert_eq!(complete[0].available_milliseconds, 20000);
        assert_eq!(complete[1].available_milliseconds, 20000);
        let clipped = [
            allowance("neural", 1000, 56000),
            allowance("generic", 40000, 56000),
        ];
        assert!(!matched_full_allowances(&clipped));
        assert_eq!(clipped[1].available_milliseconds, 16000);
        assert!(clipped[1].truncated);
        let both_clipped = [
            allowance("neural", 40000, 56000),
            allowance("generic", 40000, 56000),
        ];
        assert!(!matched_full_allowances(&both_clipped));
        assert!(!matched_full_allowances(&[
            allowance("neural", 1000, 56000),
            allowance("generic", 57000, 56000)
        ]));
    }
    #[test]
    fn warm_comparison_rejects_changed_original_arm_identity_or_clock() {
        let arm = ArmBudget {
            id_sha256: "a".repeat(64),
            seconds: 20,
        };
        let mut clock = ArmClock {
            id_sha256: arm.id_sha256.clone(),
            seconds: 20,
            started_unix_ms: 1000,
            deadline_unix_ms: 21000,
        };
        assert!(observed_arm_budget("neural", &arm, &clock, 21001).is_err());
        clock.deadline_unix_ms += 1;
        assert!(observed_arm_budget("neural", &arm, &clock, 21000).is_err());
        clock.deadline_unix_ms = 21000;
        clock.id_sha256 = "b".repeat(64);
        assert!(observed_arm_budget("neural", &arm, &clock, 21000).is_err());
    }
}
