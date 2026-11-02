use super::calibration::CalibrationModel;
use super::heads::{ExtraNode, V3Head};
use super::input::{CategoricalMode, FittedColumn, FittedInput};
use super::pipeline::{BoundModel, BoundPredictor};
use super::transforms::{GroupTail, SemanticKey, V3Op};
use super::types::{text_hash, TaskKind, TaskSpec, ARTIFACT_SCHEMA};
use crate::heads::FittedHead;
use crate::ops::FittedOp;
use crate::target_transform::FittedTarget;
use anyhow::{bail, Result};
use serde::Serialize;

mod bound;
mod python;
#[cfg(test)]
mod quotation_regression_tests;
mod rust;
mod rust_head;
mod rust_input;
mod rust_ops;
#[cfg(test)]
mod tests;

pub use bound::{emit_bound, emit_bound_rust_only, BoundEmitted};
pub(crate) use bound::{lint_shipped_masked, user_text};
pub(crate) use python::emit_bound_python;
pub(crate) use rust::emit_bound_rust;

#[cfg(test)]
mod typescript_support {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/typescript_support/mod.rs"
    ));
}

#[cfg(test)]
mod go_support {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/go_support/mod.rs"
    ));
}

#[cfg(test)]
mod quotation_regression_tests;
#[cfg(test)]
mod tests;

/// Opt-in v3 binding around unchanged v2 bound sources. The caller supplies the
/// selected canonical program; its digest must match the fitted model's binding.
pub fn emit_bound_targets_v3(
    model: &BoundModel,
    program: &crate::GenomeIr,
    options: &crate::transpile::EmitOptions,
    requested: &[crate::artifact_manifest::Target],
) -> Result<crate::artifact_manifest::TargetEmission> {
    use crate::artifact_manifest::{
        sha256, ArtifactManifestV3, FittedIdentity, Target, TargetEmission, TargetRecord,
    };
    anyhow::ensure!(
        sha256(program.canonical().as_bytes()) == model.genome_sha256,
        "bound model does not identify selected program"
    );
    let bound = emit_bound(model, options)?;
    let old: serde_json::Value = serde_json::from_str(&bound.manifest)?;
    let emitter = old["emitter_version"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing bound emitter identity"))?;
    let fitted = FittedIdentity {
        state_sha256: sha256(&serde_json::to_vec(model)?),
        input_sha256: sha256(&serde_json::to_vec(&model.input)?),
        task_sha256: model.task_sha256.clone(),
        schema_sha256: model.schema_sha256.clone(),
        vocabulary_sha256: model.vocabulary_sha256.clone(),
    };
    let mut records = Vec::new();
    let mut sources = std::collections::BTreeMap::new();
    for &target in requested {
        if target == Target::Cuda {
            let bundle = crate::cuda::emit_bound(model, options)?;
            let inputs: Vec<_> = bundle
                .files
                .iter()
                .map(|(p, s)| (p.as_str(), s.as_str()))
                .collect();
            records.push(TargetRecord::supported(
                target,
                "advanced_v3",
                crate::cuda::EMITTER,
                &inputs,
                Some(fitted.clone()),
            ));
            sources.extend(bundle.files);
            continue;
        }
        if target == Target::TypeScript {
            let bundle = crate::typescript::emit_bound(model, options)?;
            records.push(TargetRecord::supported(
                target,
                "advanced_v3",
                crate::typescript::EMITTER_VERSION,
                &[
                    ("model.ts", &bundle.source),
                    ("model.kernel.rs", &bundle.kernel),
                ],
                Some(fitted.clone()),
            ));
            sources.insert("model.ts".into(), bundle.source);
            sources.insert("model.kernel.rs".into(), bundle.kernel);
            continue;
        }
        if target == Target::Go {
            let bundle = crate::go::emit_bound(model, options)?;
            records.push(TargetRecord::supported(
                target,
                "advanced_v3",
                crate::go::EMITTER_VERSION,
                &crate::go::sources(&bundle),
                Some(fitted.clone()),
            ));
            for (path, source) in crate::go::sources(&bundle) {
                sources.insert(path.into(), source.into());
            }
            continue;
        }
        let source = match target {
            Target::Python => Some(&bound.python),
            Target::Rust => Some(&bound.rust),
            _ => None,
        };
        if let Some(source) = source {
            records.push(TargetRecord::supported(
                target,
                "advanced_v3",
                emitter,
                &[(target.path(), source)],
                Some(fitted.clone()),
            ));
            sources.insert(target.path().into(), source.clone());
        } else {
            records.push(TargetRecord::unsupported(
                target,
                "bound-v2 adapter supports Python, Rust, TypeScript, Go and CUDA C++".into(),
            ));
        }
    }
    Ok(TargetEmission {
        manifest: ArtifactManifestV3::new(
            &program.canonical(),
            "advanced_v3",
            options,
            requested,
            records,
        )?,
        sources,
    })
}

impl BoundEmitted {
    /// Versioned target binding without changing the existing bound emitter API.
    pub fn emit_targets_v3(
        model: &BoundModel,
        program: &crate::GenomeIr,
        options: &crate::transpile::EmitOptions,
        requested: &[crate::artifact_manifest::Target],
    ) -> Result<crate::artifact_manifest::TargetEmission> {
        emit_bound_targets_v3(model, program, options, requested)
    }
}
