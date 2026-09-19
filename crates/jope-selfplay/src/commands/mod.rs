//! The programs of the `jope-selfplay` binary (former `src/bin/*.rs` of the root crate).

pub mod jope_selfplay_alarms;
pub mod jope_selfplay_bundle;
pub mod jope_selfplay_corpus_freezer;
pub mod jope_selfplay_expansion_driver;
pub mod jope_selfplay_freshness;
pub mod jope_selfplay_gc;
pub mod jope_selfplay_mcp;
pub mod jope_selfplay_promotion;
pub mod jope_selfplay_propose_driver;
pub mod jope_selfplay_queue;
pub mod jope_selfplay_shadow;
pub mod jope_selfplay_status;
pub mod jope_selfplay_supply;
pub mod jope_selfplay_trainer_watchdog;
pub mod jope_selfplay_tui;
pub mod jope_selfplay_wave;
pub mod jope_task_learning_controller;
pub mod jope_training_archive;
pub mod jope_training_history;
pub mod jope_training_replica;
pub mod jope_queue_control;
pub mod jope_ledger_reconciler;
pub mod jope_flywheel_liveness;
pub mod jope_effective_unit_audit;
pub mod jope_ops_health_check;
pub mod jope_utilization_supervisor;
pub mod jope_gpu_lane_admission;
pub mod jope_reporting;
pub mod jope_v3_control;
pub mod scq_runs_gc;

use jope_campaigns::dispatch::Program;

/// Every program this binary carries, by its historical `src/bin` name.
pub const PROGRAMS: &[Program] = &[
    Program {
        name: "jope_selfplay_alarms",
        about: "one-shot health scan for the self-play flywheel",
        run: jope_selfplay_alarms::main,
    },
    Program {
        name: "jope_selfplay_bundle",
        about: "content-addressed self-play runtime bundler",
        run: jope_selfplay_bundle::main,
    },
    Program {
        name: "jope_selfplay_corpus_freezer",
        about: "advance the training corpus watermark on its schedule",
        run: jope_selfplay_corpus_freezer::main,
    },
    Program {
        name: "jope_selfplay_expansion_driver",
        about: "recurring, guardrailed dataset-unlock cycle",
        run: jope_selfplay_expansion_driver::main,
    },
    Program {
        name: "jope_selfplay_freshness",
        about: "checkpoint-freshness authority for the internal self-play queue",
        run: jope_selfplay_freshness::main,
    },
    Program {
        name: "jope_selfplay_gc",
        about: "run-dir and receipts garbage collector",
        run: jope_selfplay_gc::main,
    },
    Program {
        name: "jope_selfplay_mcp",
        about: "stdio MCP adapter for the freshness authority",
        run: jope_selfplay_mcp::main,
    },
    Program {
        name: "jope_selfplay_promotion",
        about: "promotion gate and pointer publisher",
        run: jope_selfplay_promotion::main,
    },
    Program {
        name: "jope_selfplay_propose_driver",
        about: "proposal driver",
        run: jope_selfplay_propose_driver::main,
    },
    Program {
        name: "jope_selfplay_queue",
        about: "SCQ queue keeper (submit/once cycles)",
        run: jope_selfplay_queue::main,
    },
    Program {
        name: "jope_selfplay_shadow",
        about: "shadow flywheel driver",
        run: jope_selfplay_shadow::main,
    },
    Program {
        name: "jope_selfplay_status",
        about: "queue status reconciliation and completed-job fetch",
        run: jope_selfplay_status::main,
    },
    Program {
        name: "jope_selfplay_supply",
        about: "micro-wave supply builder",
        run: jope_selfplay_supply::main,
    },
    Program {
        name: "jope_selfplay_trainer_watchdog",
        about: "continuous trainer watchdog",
        run: jope_selfplay_trainer_watchdog::main,
    },
    Program {
        name: "jope_selfplay_tui",
        about: "terminal dashboard for the self-play flywheel",
        run: jope_selfplay_tui::main,
    },
    Program {
        name: "jope_selfplay_wave",
        about: "wave builder",
        run: jope_selfplay_wave::main,
    },
    Program {
        name: "jope_task_learning_controller",
        about: "per-task learning controller (trainer submission)",
        run: jope_task_learning_controller::main,
    },
    Program {
        name: "jope_training_archive",
        about: "training archive ingest/export",
        run: jope_training_archive::main,
    },
    Program {
        name: "jope_training_history",
        about: "training history capture",
        run: jope_training_history::main,
    },
    Program {
        name: "jope_training_replica",
        about: "training replica maintenance",
        run: jope_training_replica::main,
    },
    Program {
        name: "jope_queue_control",
        about: "fail-closed SCQ queue audit/hold control and robust cross-dataset seed export",
        run: jope_queue_control::main,
    },
    Program {
        name: "jope_ledger_reconciler",
        about: "three-ledger reconciler",
        run: jope_ledger_reconciler::main,
    },
    Program {
        name: "jope_flywheel_liveness",
        about: "join the flywheel's control-plane status into one fail-closed view",
        run: jope_flywheel_liveness::main,
    },
    Program {
        name: "jope_effective_unit_audit",
        about: "audit effective user-systemd units against the active runtime bundle",
        run: jope_effective_unit_audit::main,
    },
    Program {
        name: "jope_ops_health_check",
        about: "turn a failing status file into a systemd failure",
        run: jope_ops_health_check::main,
    },
    Program {
        name: "jope_utilization_supervisor",
        about: "cluster-utilization control loop",
        run: jope_utilization_supervisor::main,
    },
    Program {
        name: "jope_gpu_lane_admission",
        about: "crash-safe admission for the three reserved self-play GPU lanes",
        run: jope_gpu_lane_admission::main,
    },
    Program {
        name: "jope_reporting",
        about: "operational status/telemetry materialization loop",
        run: jope_reporting::main,
    },
    Program {
        name: "jope_v3_control",
        about: "fail-closed activation and validation for the v3 flywheel proof path",
        run: jope_v3_control::main,
    },
    Program {
        name: "scq_runs_gc",
        about: "retention for smart_cluster/var/runs",
        run: scq_runs_gc::main,
    },
];
