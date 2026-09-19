//! The programs of the `jope-campaigns` binary (former `src/bin/*.rs` and the
//! former root binary `jope-dime-regressor`), plus the campaign command modules
//! the driver shares.

pub(crate) mod binary_cv;
pub(crate) mod exact_cv;
pub(crate) mod exact_cv_v2;
pub(crate) mod jain_regressor;
pub(crate) mod train_only_redteam;
pub(crate) mod train_only_study;

pub mod jope_dime_regressor;
pub mod aspfm_recovery_evolve;
pub mod corpus_eval;
pub mod jope_canary_executor;
pub mod jope_capacity;
pub mod jope_causal_evidence;
pub mod jope_causal_pair_repair;
pub mod jope_compile_cache;
pub mod jope_dlgp_best5;
pub mod jope_dlgp_best5_worker;
pub mod jope_dlgp_budget_supervisor;
pub mod jope_encoder_supervisor;
pub mod jope_encoder_v2_capacity;
pub mod jope_encoder_v2_import;
pub mod jope_expanded_catalog;
pub mod jope_generalized_gp;
pub mod jope_gp_bridge;
pub mod jope_hyperion_cache;
pub mod jope_hyperion_gate;
pub mod jope_sealed_zeroshot;
pub mod jope_spec_drift;
pub mod jope_strategy_screen;
pub mod jope_task_elo;
pub mod jope_task_rating;
pub mod jope_timeseries_export;
pub mod search_v3;

use crate::dispatch::Program;

/// Every program this binary carries, by its historical `src/bin` name.
pub const PROGRAMS: &[Program] = &[
    Program {
        name: "jope-dime-regressor",
        about: "the former root binary: Golden42/46 panels, exact-cv, test-adaptive, train-only study/red-team and ratchet verbs",
        run: jope_dime_regressor::main,
    },
    Program {
        name: "aspfm_recovery_evolve",
        about: "ASPFM recovery evolution run over one dataset",
        run: aspfm_recovery_evolve::main,
    },
    Program {
        name: "corpus_eval",
        about: "massive direct-test eval lane worker: evaluate a shard of genomes against one frozen dataset",
        run: corpus_eval::main,
    },
    Program {
        name: "jope_canary_executor",
        about: "execute pending three-lane canaries through pinned proposal and evaluator jobs",
        run: jope_canary_executor::main,
    },
    Program {
        name: "jope_capacity",
        about: "pressure-gated self-play capacity supervisor (audit by default, --apply records decisions)",
        run: jope_capacity::main,
    },
    Program {
        name: "jope_causal_evidence",
        about: "rebuild receipt-accurate, task-scoped causal evidence; fails promotion closed",
        run: jope_causal_evidence::main,
    },
    Program {
        name: "jope_causal_pair_repair",
        about: "execute exact, receipt-level causal pair repairs without imputing outcomes",
        run: jope_causal_pair_repair::main,
    },
    Program {
        name: "jope_compile_cache",
        about: "promising-candidate replay and fitted-artifact cache (audit by default)",
        run: jope_compile_cache::main,
    },
    Program {
        name: "jope_dlgp_best5",
        about: "DLGP best-5 campaign driver",
        run: jope_dlgp_best5::main,
    },
    Program {
        name: "jope_dlgp_best5_worker",
        about: "DLGP best-5 campaign worker",
        run: jope_dlgp_best5_worker::main,
    },
    Program {
        name: "jope_dlgp_budget_supervisor",
        about: "persistent, fail-closed supervisor for the budget-first DLGP campaign",
        run: jope_dlgp_budget_supervisor::main,
    },
    Program {
        name: "jope_encoder_supervisor",
        about: "fail-closed supervisor for one Foundation Encoder V2 dataset (sandboxed PyTorch child)",
        run: jope_encoder_supervisor::main,
    },
    Program {
        name: "jope_encoder_v2_capacity",
        about: "read-only Encoder V2 admission and graceful-preemption decision publisher",
        run: jope_encoder_v2_capacity::main,
    },
    Program {
        name: "jope_encoder_v2_import",
        about: "offline importer for fetched Encoder V2 SCQ output capsules",
        run: jope_encoder_v2_import::main,
    },
    Program {
        name: "jope_expanded_catalog",
        about: "expanded cache/catalog operations",
        run: jope_expanded_catalog::main,
    },
    Program {
        name: "jope_generalized_gp",
        about: "generalized GP campaign driver",
        run: jope_generalized_gp::main,
    },
    Program {
        name: "jope_gp_bridge",
        about: "bridge between checkpoint proposals and long-running GP populations",
        run: jope_gp_bridge::main,
    },
    Program {
        name: "jope_hyperion_cache",
        about: "freeze, build and verify the Hyperion train-only prediction cache",
        run: jope_hyperion_cache::main,
    },
    Program {
        name: "jope_hyperion_gate",
        about: "validate a Hyperion train-only campaign registry against the gate",
        run: jope_hyperion_gate::main,
    },
    Program {
        name: "jope_sealed_zeroshot",
        about: "evaluate the frozen sealed cohort (the only benchmark-valid zero-shot estimate) per checkpoint",
        run: jope_sealed_zeroshot::main,
    },
    Program {
        name: "jope_spec_drift",
        about: "WS-2 scq spec drift checker (--list, --emit-json, --verify-probes)",
        run: jope_spec_drift::main,
    },
    Program {
        name: "jope_strategy_screen",
        about: "race five DL-seeded GP search protocols on one dataset",
        run: jope_strategy_screen::main,
    },
    Program {
        name: "jope_task_elo",
        about: "per-task ELO over the strict corpus",
        run: jope_task_elo::main,
    },
    Program {
        name: "jope_task_rating",
        about: "build one immutable within-task adaptive or diagnostic rating snapshot",
        run: jope_task_rating::main,
    },
    Program {
        name: "jope_timeseries_export",
        about: "export the v2 direct multi-horizon reduction of every timeseries dataset",
        run: jope_timeseries_export::main,
    },
    Program {
        name: "search-v3",
        about: "search v3 campaign driver",
        run: search_v3::main,
    },
];
