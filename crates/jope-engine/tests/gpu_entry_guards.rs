//! Rejection-only public entry tests; none may initialize CUDA or run a fit.
#![cfg(feature = "neural")]
use std::{fs, os::fd::AsRawFd, process::Command};

#[test]
#[ignore = "subprocess isolates caller environment"]
fn entry_guard_child() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path();
    let missing = root.join("must-not-be-created");
    let absent = missing.as_path();
    fs::create_dir(root.join("runs")).unwrap();
    let lease = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(root.join("runs/encoding-service.lock"))
        .unwrap();
    assert_eq!(
        unsafe { libc::flock(lease.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    // A real exclusive lease, future deadline and caller declarations cannot
    // substitute for the absent original native grant verifier.
    std::env::set_var("JOPE_GPU_TASK_DEADLINE_MS", u64::MAX.to_string());
    std::env::set_var("JOPE_GPU_LEASE_FD", lease.as_raw_fd().to_string());
    std::env::set_var("JOPE_GPU_WORKER_LEASE_FD", lease.as_raw_fd().to_string());
    std::env::set_var("JOPE_GPU_WORKER_ACTIVE", "1");
    std::env::set_var("JOPE_GPU_POOL", absent);
    std::env::set_var("JOPE_EXPERIMENT_DIR", root);
    let route = std::env::var("JOPE_ENTRY_TEST_ROUTE").unwrap();
    let result = match route.as_str() {
        "proposal" => {
            jope_parallel::proposals::rank(absent, absent, absent, absent, 0, 8, 2, 0, false)
        }
        "monitor" => jope_parallel::monitoring::evaluate(absent, absent, absent, absent, absent),
        "train" => jope_parallel::training::rank(
            absent, absent, absent, absent, 0, 8, 256, 1, 1, false, None,
        ),
        "batch" => jope_parallel::training::qualify_batch_rank(
            absent, absent, absent, absent, 0, 256, 2048, 1, absent,
        ),
        "ranked" => jope_parallel::ranked_proposals::run(absent, absent, 0, 8),
        "nccl-probe" => jope_parallel::neural::nccl_probe(0, 8, absent),
        "training-probe" => {
            jope_parallel::neural::training_probe(0, 8, absent, absent, 256, 2048, 8)
        }
        "probe-controller" => jope_parallel::controller::probe(8, absent, 60, None),
        "training-probe-controller" => {
            jope_parallel::controller::training_probe(absent, absent, 256, 2048, 8, 60)
        }
        "train-controller" => jope_parallel::controller::train(
            root,
            absent,
            absent,
            jope_parallel::controller::TrainingOptions {
                world: 8,
                micro: 256,
                steps: 1,
                checkpoint_steps: 1,
                resume: false,
                fail_rank: None,
                initialize: None,
            },
        ),
        "proposal-controller" => {
            jope_parallel::controller::propose(root, absent, absent, absent, 2, 0, false)
        }
        "encode" => jope_parallel::encoding::encode(root, absent, 0, absent, absent, 0).map(|_| ()),
        "target-rank" => {
            jope_parallel::components::target_rank(root, absent, absent, absent, 0, 8, 256)
        }
        "foundation-rank" => {
            jope_parallel::components::foundation_rank(root, absent, absent, absent, 0, 8, 1)
        }
        "component-service" => {
            jope_parallel::components::service(root, absent, absent, absent, 1, "target")
        }
        "assemble-contexts" => {
            jope_parallel::contexts::assemble(root, absent, absent, absent, absent, absent, true)
        }
        "assemble-registered" => jope_parallel::contexts::assemble_registered(
            root, absent, absent, absent, absent, absent, absent, true,
        ),
        "apply-contexts" => jope_parallel::contexts::apply(
            root, absent, absent, absent, absent, absent, absent, true,
        ),
        "binary-contexts" => jope_parallel::contexts::apply_binary_qualification(
            root, absent, absent, absent, absent, absent, absent, absent,
        ),
        "campaign" => jope_parallel::campaign::run(root, absent),
        "cycle" => {
            jope_parallel::pipeline::run(root, absent, absent, 1, 2, 0, None, None, &[], true)
        }
        "resident-submit" => jope_parallel::gpu_worker::submit(
            root,
            0,
            jope_parallel::gpu_worker::Task::Monitor {
                corpus: absent.into(),
                checkpoint: absent.into(),
                panel: absent.into(),
                output: absent.into(),
            },
        ),
        "forged-worker" => {
            std::env::set_var("JOPE_GPU_WORKER_ACTIVE", "1");
            jope_parallel::gpu_worker::require_accepted_task(
                absent,
                &jope_parallel::gpu_worker::Task::Monitor {
                    corpus: absent.into(),
                    checkpoint: absent.into(),
                    panel: absent.into(),
                    output: absent.into(),
                },
            )
        }
        _ => panic!("unknown test route"),
    };
    let error = format!("{:#}", result.unwrap_err());
    assert!(
        error.contains("original native GPU grant verifier unavailable")
            || error.contains("verified resident acceptance"),
        "wrong rejection: {error}"
    );
    assert!(
        !missing.exists(),
        "held public route created its output before rejection"
    );
}

#[test]
fn public_gpu_routes_reject_caller_declarations_before_input_or_cuda_access() {
    for route in [
        "proposal",
        "monitor",
        "train",
        "batch",
        "ranked",
        "forged-worker",
        "nccl-probe",
        "training-probe",
        "probe-controller",
        "training-probe-controller",
        "train-controller",
        "proposal-controller",
        "encode",
        "target-rank",
        "foundation-rank",
        "component-service",
        "assemble-contexts",
        "assemble-registered",
        "apply-contexts",
        "binary-contexts",
        "campaign",
        "cycle",
        "resident-submit",
    ] {
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "entry_guard_child"])
            .env("JOPE_ENTRY_TEST_ROUTE", route)
            .env_remove("JOPE_GPU_TASK_DEADLINE_MS")
            .env_remove("JOPE_GPU_LEASE_FD")
            .env_remove("JOPE_GPU_POOL")
            .env_remove("JOPE_GPU_WORKER_ACTIVE")
            .env_remove("JOPE_GPU_WORKER_LEASE_FD")
            .status()
            .unwrap();
        assert!(status.success(), "public route {route} bypassed its guard");
    }
}
