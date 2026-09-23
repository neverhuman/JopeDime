//! Public human explanation facade for pipeline genomes.
//!
//! The wording is sourced from the same planned-stage/head tables used by the
//! Python/Rust emitters, so reports describe the exact implementation that is
//! emitted in appendices.

use anyhow::{bail, Result};

use crate::spec::{
    build_plan, head_doc, head_short, op_doc, op_short, target_doc, target_short, Plan,
};
use crate::GenomeIr;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Explanation {
    pub title: String,
    pub lines: Vec<String>,
}

pub fn explain_genome(src: &str) -> Result<Vec<String>> {
    Ok(explain(&GenomeIr::parse(src)?)?.lines)
}

pub fn explain(genome: &GenomeIr) -> Result<Explanation> {
    let plan = build_plan(genome).map_err(|problems| anyhow::anyhow!(problems.join("; ")))?;
    match plan {
        Plan::Pipeline {
            target,
            stages,
            head,
        } => {
            let mut lines = Vec::new();
            if let Some(target) = &target {
                lines.push(format!(
                    "Target: {}; {}",
                    target_short(target.kind),
                    target_doc(target)
                ));
            }
            if stages.is_empty() {
                lines.push("Uses the input columns directly.".to_string());
            }
            for (idx, stage) in stages.iter().enumerate() {
                lines.push(format!(
                    "Stage {}: {}; {} {}",
                    idx + 1,
                    op_short(&stage.op),
                    op_doc(&stage.op),
                    if stage.append {
                        "Outputs are appended to the incoming columns."
                    } else {
                        "Outputs replace the incoming columns."
                    }
                ));
            }
            lines.push(
                "After every stage, non-finite cells become 0.0 and width is capped at 512 columns."
                    .to_string(),
            );
            lines.push(format!("Head: {}; {}", head_short(&head), head_doc(&head)));
            Ok(Explanation {
                title: format!("Pipeline ending in {}", head_short(&head)),
                lines,
            })
        }
        Plan::Symbolic { source, affine, .. } => Ok(Explanation {
            title: "Symbolic expression".into(),
            lines: vec![format!(
                "Evaluates the guarded symbolic formula `{source}`{}.",
                if affine {
                    " and fits an affine scale and intercept on the training target"
                } else {
                    ""
                }
            )],
        }),
    }
}

pub fn explain_canonical(src: &str) -> Result<Explanation> {
    let genome = GenomeIr::parse(src)?;
    if genome.canonical().is_empty() {
        bail!("empty program");
    }
    explain(&genome)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explains_pipeline_stages_and_head() {
        let e = explain_canonical("(pipe X (robust_scale) (ridge))").unwrap();
        assert!(e.title.contains("ridge"));
        assert!(e.lines.iter().any(|line| line.contains("robust scaling")));
        assert!(e.lines.iter().any(|line| line.contains("Head:")));
    }
}
