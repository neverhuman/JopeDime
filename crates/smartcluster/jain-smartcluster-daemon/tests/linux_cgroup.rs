use jain_smartcluster_daemon::capacity::CpuSet;
use jain_smartcluster_daemon::cgroup::{AttemptKey, CgroupLimits, CgroupManager, IsolationMode};

#[test]
#[ignore = "requires a delegated cgroup-v2 host lane"]
fn real_delegated_cgroup_controls_are_writable() {
    let root = std::env::var_os("SCQ_TEST_CGROUP_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/sys/fs/cgroup/jain-smartcluster-test.slice"));
    let manager = match CgroupManager::initialize(&root, false) {
        Ok(manager) if manager.mode() == IsolationMode::Delegated => manager,
        Ok(manager) => {
            eprintln!(
                "SKIP: cgroup delegation unavailable: initialized in {:?} mode",
                manager.mode()
            );
            return;
        }
        Err(error) => {
            eprintln!("SKIP: cgroup delegation unavailable: {error}");
            return;
        }
    };
    let cpus = std::fs::read_to_string(root.join("cpuset.cpus.effective"))
        .ok()
        .and_then(|value| CpuSet::parse(&value).ok())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| CpuSet::parse("0").expect("fallback CPU set"));
    let key = AttemptKey {
        job_id: format!("hostlane{}", std::process::id()),
        attempt: 1,
        fence: "1".to_owned(),
    };
    let cgroup = manager
        .prepare(
            &key,
            &CgroupLimits {
                cpus,
                cpu_weight: 100,
                cpu_max_millicores: 100,
                memory_low_bytes: 0,
                memory_high_bytes: 64 * 1024 * 1024,
                memory_max_bytes: 128 * 1024 * 1024,
            },
        )
        .expect("create real test cgroup");
    manager.freeze(&cgroup).expect("freeze control");
    manager.resume(&cgroup).expect("resume control");
    assert!(cgroup.path.join("memory.max").is_file());
    manager
        .remove_if_empty(&cgroup)
        .expect("remove empty test cgroup");
}

#[test]
#[ignore = "requires Linux PSI trigger registration permissions"]
fn real_psi_trigger_registers_when_delegated() {
    let cgroup = std::env::var_os("SCQ_TEST_PRESSURE_CGROUP")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/sys/fs/cgroup"));
    match jain_smartcluster_daemon::psi::PsiMonitor::register(
        &cgroup,
        jain_smartcluster_daemon::config::PsiConfig::default(),
    ) {
        Ok(_monitor) => {}
        Err(error) => eprintln!("SKIP: PSI triggers unavailable: {error}"),
    }
}
