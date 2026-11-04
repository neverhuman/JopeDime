use super::*;

#[test]
fn cpuset_round_trip_compacts_ranges() {
    let set = CpuSet::parse("0-3,8,10-11,3").expect("valid set");
    assert_eq!(set.len(), 7);
    assert_eq!(set.to_list(), "0-3,8,10-11");
}

#[test]
fn cpu_limit_uses_ceiling_millicores() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("cpu.max");
    fs::write(&path, "25001 100000").expect("write");
    assert_eq!(read_cpu_limit(&path).expect("parse"), Some(251));
}

#[test]
fn invalid_descending_cpu_range_is_rejected() {
    assert!(matches!(
        CpuSet::parse("4-2"),
        Err(CapacityError::CpuSet(_))
    ));
}

#[test]
fn affinity_parser_uses_kernel_allowed_list() {
    let set = process_affinity_from("Name:\ttest\nCpus_allowed_list:\t2-3,7\n").expect("affinity");
    assert_eq!(set.to_list(), "2-3,7");
}

#[test]
fn cpuset_walks_from_leaf_to_parent() {
    let directory = tempfile::tempdir().expect("tempdir");
    let parent = directory.path().join("parent");
    let leaf = parent.join("leaf");
    fs::create_dir_all(&leaf).expect("directories");
    fs::write(parent.join("cpuset.cpus.effective"), "1-2").expect("cpuset");
    let cpus = hierarchy_cpuset(&[leaf, parent])
        .expect("walk")
        .expect("parent cpuset");
    assert_eq!(cpus.to_list(), "1-2");
}

#[test]
fn allocation_root_capacity_includes_tighter_child_limits() {
    let directory = tempfile::tempdir().expect("tempdir");
    let mount = directory.path().join("cgroup");
    let daemon = mount.join("daemon");
    let workloads = daemon.join("workloads");
    let scratch = directory.path().join("scratch");
    fs::create_dir_all(&workloads).expect("workload root");
    fs::create_dir(&scratch).expect("scratch");
    fs::write(
        workloads.join("cpuset.cpus.effective"),
        process_affinity().expect("affinity").to_list(),
    )
    .expect("cpuset");
    fs::write(workloads.join("cpu.max"), "100000 100000").expect("cpu limit");
    fs::write(workloads.join("memory.max"), "67108864").expect("memory limit");
    let proc_cgroup = directory.path().join("self.cgroup");
    fs::write(&proc_cgroup, "0::/daemon\n").expect("proc cgroup");
    let numa = directory.path().join("numa");

    let capacity = CapacityDiscovery {
        cgroup_mount: &mount,
        proc_self_cgroup: &proc_cgroup,
        numa_root: &numa,
        scratch_path: &scratch,
    }
    .discover_for_allocation_root(&workloads)
    .expect("allocation capacity");

    assert_eq!(capacity.cgroup_path, workloads.canonicalize().unwrap());
    assert_eq!(capacity.cpu_millicores, 1_000);
    assert_eq!(capacity.memory_max_bytes, 64 * 1024 * 1024);
}

#[test]
fn systemd_supervisor_parent_uses_delegated_capacity() {
    let directory = tempfile::tempdir().unwrap();
    let mount = directory.path().join("cgroup");
    let delegated = mount.join("system.slice/scqd.service");
    let supervisor = delegated.join("supervisor");
    let scratch = directory.path().join("scratch");
    fs::create_dir_all(&supervisor).unwrap();
    fs::create_dir(&scratch).unwrap();
    fs::write(delegated.join("cgroup.procs"), "").unwrap();
    fs::write(
        delegated.join("cgroup.controllers"),
        "cpuset cpu io memory pids\n",
    )
    .unwrap();
    let affinity = process_affinity().unwrap();
    fs::write(delegated.join("cpuset.cpus.effective"), affinity.to_list()).unwrap();
    fs::write(delegated.join("cpu.max"), "200000 100000").unwrap();
    fs::write(delegated.join("memory.max"), "134217728").unwrap();
    let proc_cgroup = directory.path().join("self.cgroup");
    fs::write(&proc_cgroup, "0::/system.slice/scqd.service/supervisor\n").unwrap();
    let capacity = CapacityDiscovery {
        cgroup_mount: &mount,
        proc_self_cgroup: &proc_cgroup,
        numa_root: &directory.path().join("numa"),
        scratch_path: &scratch,
    }
    .discover_for_allocation_root(&delegated)
    .unwrap();
    assert_eq!(capacity.cgroup_path, delegated.canonicalize().unwrap());
    assert_eq!(
        capacity.cpu_millicores,
        2_000.min(affinity.len() as u64 * 1_000)
    );
    assert_eq!(capacity.memory_max_bytes, 128 * 1024 * 1024);
}

#[test]
fn delegated_parent_exception_refuses_other_layouts_and_missing_controls() {
    let directory = tempfile::tempdir().unwrap();
    let parent = directory.path().join("scqd.service");
    let supervisor = parent.join("supervisor");
    fs::create_dir_all(&supervisor).unwrap();
    fs::write(parent.join("cgroup.procs"), "").unwrap();
    fs::write(parent.join("cgroup.controllers"), "cpu memory cpuset").unwrap();
    assert!(is_systemd_delegated_root(&parent, &supervisor).unwrap());
    assert!(!is_systemd_delegated_root(&parent, &parent.join("worker")).unwrap());
    assert!(!is_systemd_delegated_root(directory.path(), &supervisor).unwrap());
    fs::write(parent.join("cgroup.procs"), "123\n").unwrap();
    assert!(!is_systemd_delegated_root(&parent, &supervisor).unwrap());
    fs::write(parent.join("cgroup.procs"), "").unwrap();
    fs::write(parent.join("cgroup.controllers"), "cpu memory").unwrap();
    assert!(!is_systemd_delegated_root(&parent, &supervisor).unwrap());
    fs::remove_file(parent.join("cgroup.controllers")).unwrap();
    assert!(is_systemd_delegated_root(&parent, &supervisor).is_err());
}

#[test]
fn allocation_root_outside_daemon_cgroup_fails_closed() {
    let directory = tempfile::tempdir().expect("tempdir");
    let mount = directory.path().join("cgroup");
    let daemon = mount.join("daemon");
    let sibling = mount.join("sibling");
    fs::create_dir_all(&daemon).expect("daemon cgroup");
    fs::create_dir(&sibling).expect("sibling cgroup");
    let proc_cgroup = directory.path().join("self.cgroup");
    fs::write(&proc_cgroup, "0::/daemon\n").expect("proc cgroup");

    let error = CapacityDiscovery {
        cgroup_mount: &mount,
        proc_self_cgroup: &proc_cgroup,
        numa_root: directory.path(),
        scratch_path: directory.path(),
    }
    .discover_for_allocation_root(&sibling)
    .expect_err("sibling root must fail");

    assert!(matches!(
        error,
        CapacityError::AllocationRootOutsideDaemon { .. }
    ));
}
