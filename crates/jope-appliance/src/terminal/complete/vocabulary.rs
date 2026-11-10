//! Canonical terminal command, flag and enum vocabulary used by completion.

/// Canonical command and flag entries shared by terminal parsing and completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommandSpec {
    /// Command spelling without its leading `\\` or `/`.
    pub name: &'static str,
    /// Flags accepted by the command.
    pub flags: &'static [&'static str],
}

/// Canonical terminal command and flag vocabulary for parser/completion use.
pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "help",
        flags: &[],
    },
    CommandSpec {
        name: "open",
        flags: &["--name", "--json"],
    },
    CommandSpec {
        name: "demo",
        flags: &[],
    },
    CommandSpec {
        name: "usage",
        flags: &[],
    },
    CommandSpec {
        name: "target",
        flags: &["--task", "--ignore", "--group", "--time"],
    },
    CommandSpec {
        name: "objective",
        flags: &["--json"],
    },
    CommandSpec {
        name: "fitness",
        flags: &["--json"],
    },
    CommandSpec {
        name: "effort",
        flags: &["--evals", "--time"],
    },
    CommandSpec {
        name: "landmarks",
        flags: &[],
    },
    CommandSpec {
        name: "comparators",
        flags: &[],
    },
    CommandSpec {
        name: "lang",
        flags: &["--best-effort"],
    },
    CommandSpec {
        name: "language",
        flags: &["--best-effort"],
    },
    CommandSpec {
        name: "plan",
        flags: &["--json"],
    },
    CommandSpec {
        name: "run",
        flags: &["--seed", "--local", "--cloud", "--json"],
    },
    CommandSpec {
        name: "pause",
        flags: &["--json"],
    },
    CommandSpec {
        name: "continue",
        flags: &["--json"],
    },
    CommandSpec {
        name: "cancel",
        flags: &["--json"],
    },
    CommandSpec {
        name: "restart",
        flags: &["--same-seed", "--new-seed", "--warm", "--json"],
    },
    CommandSpec {
        name: "status",
        flags: &["--json"],
    },
    CommandSpec {
        name: "metrics",
        flags: &["--detailed", "--json"],
    },
    CommandSpec {
        name: "reasoning",
        flags: &[],
    },
    CommandSpec {
        name: "code",
        flags: &["--json"],
    },
    CommandSpec {
        name: "diff",
        flags: &["--json"],
    },
    CommandSpec {
        name: "export",
        flags: &["--kind", "--lang", "--alg", "--out", "--force", "--json"],
    },
    CommandSpec {
        name: "predict",
        flags: &["--input", "--output", "--model", "--json"],
    },
    CommandSpec {
        name: "runs",
        flags: &["--all", "--json"],
    },
    CommandSpec {
        name: "attach",
        flags: &["--json"],
    },
    CommandSpec {
        name: "doctor",
        flags: &["--json"],
    },
    CommandSpec {
        name: "auto",
        flags: &[],
    },
    CommandSpec {
        name: "copy",
        flags: &[],
    },
    CommandSpec {
        name: "transcript",
        flags: &[],
    },
    CommandSpec {
        name: "mouse",
        flags: &[],
    },
    CommandSpec {
        name: "login",
        flags: &[],
    },
    CommandSpec {
        name: "logout",
        flags: &[],
    },
    CommandSpec {
        name: "sync",
        flags: &[],
    },
    CommandSpec {
        name: "feedback",
        flags: &[],
    },
    CommandSpec {
        name: "quit",
        flags: &["--pause"],
    },
    CommandSpec {
        name: "exit",
        flags: &["--pause"],
    },
];

pub(super) const TASKS: &[&str] = &["regression", "binary", "multiclass"];
pub(super) const EFFORTS: &[&str] = &["fast", "medium", "high", "ultra"];
pub(super) const LANDMARKS: &[&str] = &["full", "none"];
/// `off`, `all`, then every roster key; comparators never run at fast effort.
pub(super) const COMPARATORS: &[&str] = &[
    "off",
    "all",
    "autogluon_style",
    "flaml_style",
    "h2o_style",
    "mljar_style",
    "xgboost_style",
    "lightgbm_style",
    "catboost_style",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
];
pub(super) const LANGUAGES: &[&str] = &["rust", "python", "javascript", "typescript"];
pub(super) const REASONING: &[&str] = &["off", "concise", "detailed"];
pub(super) const EXPORT_KINDS: &[&str] = &["bundle", "model", "source"];
pub(super) const COPY_ITEMS: &[&str] = &["link", "run", "result", "all"];
pub(super) const ON_OFF: &[&str] = &["on", "off"];
pub(super) const FEEDBACK_KINDS: &[&str] = &["issue", "idea", "good", "other"];
