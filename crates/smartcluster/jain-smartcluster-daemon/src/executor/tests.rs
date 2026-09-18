use super::*;
use jain_smartcluster_core::Environment;

#[test]
fn jain_templates_resolve_to_fixed_worker_argv() {
    let resolver = WorkloadResolver::new(WorkerConfig::default());
    let plan = resolver
        .resolve(&Workload::Jain {
            template: JainWorkload::Training,
            arguments: vec!["dataset-1".to_owned()],
            environment: Environment::default(),
        })
        .expect("resolve");
    assert_eq!(
        plan.program,
        std::path::Path::new("/usr/libexec/jain/jain-worker")
    );
    assert_eq!(plan.arguments, ["training", "dataset-1"]);
}

#[test]
fn administrator_argv_is_not_wrapped_in_a_shell() {
    let resolver = WorkloadResolver::new(WorkerConfig::default());
    let plan = resolver
        .resolve(&Workload::AdministratorArgv {
            program: "/usr/bin/printf".to_owned(),
            arguments: vec!["%s".to_owned(), "$(not-expanded)".to_owned()],
            environment: Environment::default(),
            working_directory: Some("/tmp".to_owned()),
        })
        .expect("resolve");
    assert_eq!(plan.program, std::path::Path::new("/usr/bin/printf"));
    assert_eq!(plan.arguments[1], "$(not-expanded)");
}

#[test]
fn relative_programs_fail_closed() {
    let plan = LaunchPlan {
        program: PathBuf::from("worker"),
        arguments: Vec::new(),
        environment: BTreeMap::new(),
        working_directory: None,
    };
    assert!(matches!(
        plan.validate(),
        Err(ExecutorError::ProgramNotAbsolute(_))
    ));
}

#[test]
fn reserved_environment_cannot_replace_fencing_or_loader_policy() {
    for name in [
        "SCQ_CONTROL_FD",
        "SCQ_CUSTOM",
        "PATH",
        "LD_PRELOAD",
        "LD_AUDIT",
        "DYLD_INSERT_LIBRARIES",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_MPS_PIPE_DIRECTORY",
    ] {
        assert!(reserved_environment_name(name), "{name}");
    }
    assert!(!reserved_environment_name("OMP_NUM_THREADS"));
}

#[test]
fn helper_identity_prefix_is_unambiguous() {
    let arguments = vec![
        std::process::id().to_string(),
        "993".to_owned(),
        "994".to_owned(),
        "/usr/bin/printf".to_owned(),
        "--literal".to_owned(),
    ];
    let invocation = parse_helper_arguments(&arguments).expect("parse");
    assert_eq!(invocation.expected_parent, nix::unistd::getpid());
    assert_eq!(invocation.identity, Some((993, 994)));
    assert_eq!(invocation.program, "/usr/bin/printf");
    assert_eq!(invocation.arguments, ["--literal"]);

    let incomplete = vec![
        std::process::id().to_string(),
        "-".to_owned(),
        "994".to_owned(),
        "/usr/bin/true".to_owned(),
    ];
    assert!(matches!(
        parse_helper_arguments(&incomplete),
        Err(ExecutorError::HelperIdentity)
    ));

    let invalid_parent = vec![
        "0".to_owned(),
        "-".to_owned(),
        "-".to_owned(),
        "/usr/bin/true".to_owned(),
    ];
    assert!(matches!(
        parse_helper_arguments(&invalid_parent),
        Err(ExecutorError::HelperParent)
    ));
}

#[test]
fn cpu_allocations_hide_every_cuda_device() {
    assert_eq!(cuda_visible_devices(&[]), "-1");
    assert_eq!(
        cuda_visible_devices(&["GPU-a".to_owned(), "GPU-b".to_owned()]),
        "GPU-a,GPU-b"
    );
}

#[test]
fn process_group_pid_conversion_rejects_values_outside_linux_pid_range() {
    assert!(matches!(
        signal_process_group(u32::MAX, Signal::SIGTERM),
        Err(ExecutorError::ProcessIdOutOfRange(u32::MAX))
    ));
}
