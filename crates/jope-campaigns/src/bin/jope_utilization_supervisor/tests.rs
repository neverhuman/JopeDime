use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args::parse_from(["jope_utilization_supervisor"])
    }

    fn stage(name: &'static str, health: Health) -> Stage {
        Stage {
            name,
            health,
            detail: String::new(),
            receipt: String::new(),
            repair: Some(Repair::restart("unit.service")),
        }
    }

    #[test]
    fn iso_utc_matches_known_epoch() {
        assert_eq!(iso_utc(1_785_251_604_304), "2026-07-28T15:13:24Z");
        assert_eq!(iso_utc(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn core_pct_uses_budget_not_slot_target() {
        // Live 2026-07-28T15:18Z reading: 1,052 slot targets, 317 slots busy,
        // but only 1 of 344 core-budget units free. Slot occupancy says 30%;
        // the cluster was in fact saturated. Judging on slots would have made
        // this supervisor cry idle on a full cluster.
        let util = Utilization {
            slots_in_use: 317,
            slots_total: 1052,
            core_budget_total: 344,
            core_budget_free: 1,
            ..Utilization::default()
        };
        assert!((util.slot_pct() - 30.13).abs() < 0.1);
        assert!(util.core_pct() > 99.0);
    }

    #[test]
    fn idle_with_work_is_idle_and_without_work_is_not_an_alarm() {
        let a = args();
        let util = Utilization {
            scq_reachable: true,
            core_budget_total: 344,
            core_budget_free: 344,
            ..Utilization::default()
        };
        let stages = vec![stage("keeper", Health::Producing)];
        let with_work = WorkOnDisk {
            wave_manifests: 7040,
            ..WorkOnDisk::default()
        };
        assert_eq!(judge(&a, &util, &with_work, &stages), Verdict::Idle);
        assert_eq!(
            judge(&a, &util, &WorkOnDisk::default(), &stages),
            Verdict::IdleNoWork
        );
    }

    #[test]
    fn busy_but_broken_producer_is_degraded() {
        // 2026-07-28T15:18Z: 552 active jobs and a saturated core budget while
        // BOTH the keeper and supply were in error loops. "Busy" alone must not
        // read as healthy or the cliff arrives unannounced.
        let a = args();
        let util = Utilization {
            scq_reachable: true,
            core_budget_total: 344,
            core_budget_free: 1,
            jobs_queued: 73,
            jobs_running: 315,
            ..Utilization::default()
        };
        let stages = vec![
            stage("scq", Health::Producing),
            stage("keeper", Health::Erroring),
        ];
        let work = WorkOnDisk {
            wave_manifests: 7040,
            ..WorkOnDisk::default()
        };
        assert_eq!(judge(&a, &util, &work, &stages), Verdict::Degraded);
    }

    #[test]
    fn all_producing_and_busy_is_ok() {
        let a = args();
        let util = Utilization {
            scq_reachable: true,
            core_budget_total: 344,
            core_budget_free: 20,
            jobs_queued: 900,
            ..Utilization::default()
        };
        let stages = vec![
            stage("scq", Health::Producing),
            stage("keeper", Health::Producing),
        ];
        assert_eq!(
            judge(
                &a,
                &util,
                &WorkOnDisk {
                    wave_manifests: 10,
                    ..WorkOnDisk::default()
                },
                &stages
            ),
            Verdict::Ok
        );
    }

    #[test]
    fn backoff_doubles_then_saturates() {
        let a = args();
        assert_eq!(backoff_ms(&a, 0), 0);
        assert_eq!(backoff_ms(&a, 1), 300_000);
        assert_eq!(backoff_ms(&a, 2), 600_000);
        assert_eq!(backoff_ms(&a, 4), 2_400_000);
        assert_eq!(backoff_ms(&a, 9), 3_600_000); // clamped at --backoff-max-seconds
        assert_eq!(backoff_ms(&a, 40), 3_600_000); // shift is clamped, no overflow
    }

    #[test]
    fn hourly_cap_suppresses_every_action() {
        let a = args();
        let now = now_ms();
        let state = State {
            action_log: vec![now; 6],
            ..State::default()
        };
        let stages = vec![stage("keeper", Health::Erroring)];
        let (chosen, note) = plan_repair(&a, Verdict::Idle, &stages, &state, now);
        assert!(chosen.is_none());
        assert!(note.unwrap().contains("max-actions-per-hour"));
    }

    #[test]
    fn backoff_defers_a_repeat_of_the_same_action() {
        let a = args();
        let now = now_ms();
        let stages = vec![stage("keeper", Health::Erroring)];
        let mut state = State::default();
        let key = stages[0].repair.as_ref().unwrap().key.clone();
        state.attempts.insert(key, (1, now - 10_000));
        let (chosen, note) = plan_repair(&a, Verdict::Idle, &stages, &state, now);
        assert!(chosen.is_none());
        assert!(note.unwrap().contains("backoff"));
    }

    #[test]
    fn one_repair_per_cycle_and_pipeline_order_wins() {
        let a = args();
        let stages = vec![
            stage("scq", Health::Producing),
            stage("keeper", Health::Erroring),
            stage("supply", Health::Erroring),
        ];
        let (chosen, _) = plan_repair(&a, Verdict::Idle, &stages, &State::default(), now_ms());
        assert_eq!(chosen.map(|stage| stage.name), Some("keeper"));
    }

    #[test]
    fn expansion_retrigger_waits_until_the_cluster_is_actually_idle() {
        let a = args();
        let stages = vec![stage("expansion", Health::Blocked)];
        let (chosen, note) =
            plan_repair(&a, Verdict::Degraded, &stages, &State::default(), now_ms());
        assert!(chosen.is_none());
        assert!(note.unwrap().contains("not idle"));
        let (chosen, _) = plan_repair(&a, Verdict::Idle, &stages, &State::default(), now_ms());
        assert_eq!(chosen.map(|stage| stage.name), Some("expansion"));
    }

    #[test]
    fn a_stage_without_a_safe_repair_is_never_improvised_on() {
        let a = args();
        let stages = vec![Stage {
            name: "scq",
            health: Health::Down,
            detail: String::new(),
            receipt: String::new(),
            repair: None,
        }];
        let (chosen, note) = plan_repair(&a, Verdict::Idle, &stages, &State::default(), now_ms());
        assert!(chosen.is_none());
        assert!(note.unwrap().contains("no safe automatic repair"));
    }

    #[test]
    fn healthy_verdicts_never_plan_a_repair() {
        let a = args();
        let stages = vec![stage("keeper", Health::Erroring)];
        for verdict in [Verdict::Ok, Verdict::IdleNoWork] {
            assert!(
                plan_repair(&a, verdict, &stages, &State::default(), now_ms())
                    .0
                    .is_none()
            );
        }
    }

    #[test]
    fn simulate_idle_forces_the_idle_verdict_on_a_saturated_cluster() {
        let mut a = args();
        a.simulate_idle = true;
        let util = Utilization {
            scq_reachable: true,
            core_budget_total: 344,
            core_budget_free: 0,
            ..Utilization::default()
        };
        let stages = vec![stage("keeper", Health::Producing)];
        let work = WorkOnDisk {
            wave_manifests: 7040,
            ..WorkOnDisk::default()
        };
        assert_eq!(judge(&a, &util, &work, &stages), Verdict::Idle);
    }
}
