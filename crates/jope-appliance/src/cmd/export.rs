use crate::cli::{ExportKindArg, LanguageArg};
use jope_api::{
    ids::{AlgorithmId, RunId},
    ExportKind, ExportRequest, Language, Refit, RunSnapshot,
};
use jope_client::Transport;
use serde_json::Value;
use std::str::FromStr;

pub(crate) struct Options<'a> {
    pub run: &'a str,
    pub alg: Option<&'a str>,
    pub kind: ExportKindArg,
    pub language: LanguageArg,
    pub out: &'a std::path::Path,
    pub force: bool,
    pub idempotency_key: Option<&'a str>,
    pub json_out: bool,
}

pub(crate) fn run(options: Options<'_>) -> anyhow::Result<()> {
    let Options {
        run,
        alg,
        kind,
        language,
        out,
        force,
        idempotency_key,
        json_out,
    } = options;
    let client = super::dataset::resolve_client()?;
    let run_id = RunId::from_str(run).map_err(|_| anyhow::anyhow!("invalid run id"))?;
    let algorithm = if let Some(alg) = alg {
        AlgorithmId::from_str(alg).map_err(|_| anyhow::anyhow!("invalid algorithm id"))?
    } else {
        let snapshot: RunSnapshot = client.run_status(&run_id)?;
        snapshot
            .best
            .ok_or_else(|| anyhow::anyhow!("run has no best algorithm"))?
            .algorithm_id
    };
    let kind = match kind {
        ExportKindArg::Model => ExportKind::Model,
        ExportKindArg::Source => ExportKind::Source,
        ExportKindArg::Bundle => ExportKind::Bundle,
    };
    let language = match language {
        LanguageArg::Rust => Language::Rust,
        LanguageArg::Python => Language::Python,
        LanguageArg::Javascript => Language::Javascript,
        LanguageArg::Typescript => Language::Typescript,
    };
    let generated = idempotency_key.is_none();
    let key = super::common::resolve_idempotency_key(idempotency_key);
    if generated {
        super::common::write_idempotency_key(&mut std::io::stderr().lock(), &key)?;
    }
    let request = ExportRequest {
        algorithm_id: algorithm,
        kind,
        languages: vec![language],
        require_all_languages: true,
        refit: Refit::Fit,
    };
    let manifest = client.model_export(&run_id, &request, Some(&key))?;
    materialize(&client, &manifest, out, force)?;
    let mut value: Value = serde_json::to_value(manifest)?;
    if let Some(object) = value.as_object_mut() {
        super::common::add_idempotency_key(object, &key);
    }
    super::dataset::emit(&value, json_out)
}

pub(crate) use crate::common::materialize_export as materialize;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kind_flags_match_wire_values() {
        assert_eq!(serde_json::to_value(ExportKind::Bundle).unwrap(), "bundle");
    }
}
