//! Rust counterpart of jope.policy.tokens_build.
//!
//! Builds the append-only action-token table from ordered grammar descriptions. The
//! Python implementation remains in use while its product importer is present.

mod construction;
mod extensions;
mod spec;

pub use construction::build_action_vocab;
pub use extensions::{
    append_missing_task_tokens, append_missing_task_tokens_from, extend_action_vocab_append_only,
    extend_engine_ops_append_only, reconcile_head_tasks_from_grammar,
};
pub use spec::{
    format_param_token, ActionVocab, ExprSpec, GrammarSpec, OpSpec, ParamSpec, ABSENT,
    CANONICAL_TASKS, SPECIALS,
};

#[cfg(test)]
mod tests;
