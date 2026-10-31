//! The only way programs enter: independently qualified records, admitted with exact checks.
//!
//! Fields are private and nothing here implements `Deserialize` for a program, so a program
//! reaches the table builder only through the crate-private `QualifiedProgram::admit`, which
//! the eligible-State reader (`teaching`) calls.

use crate::{is_hex64, sha256_hex, Code};
use anyhow::{bail, ensure, Result};
use feat_math::advanced_v3::{Objective, TaskSpec};
use feat_math::GenomeIr;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Task of a qualified record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskLabel {
    /// Regression task scored by RMSE.
    Regression,
    /// Binary classification task scored by ROC AUC.
    Binary,
    /// Multiclass task scored by macro F1.
    Multiclass,
}

impl TaskLabel {
    /// Every task, in declaration order.
    pub const ALL: [Self; 3] = [Self::Regression, Self::Binary, Self::Multiclass];

    /// `regression`, `binary` or `multiclass`.
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "regression" => Ok(Self::Regression),
            "binary" => Ok(Self::Binary),
            "multiclass" => Ok(Self::Multiclass),
            _ => bail!(Code("input_join_failed")),
        }
    }

    /// Return the serialized lowercase task label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Regression => "regression",
            Self::Binary => "binary",
            Self::Multiclass => "multiclass",
        }
    }

    /// The engine task contract, as the frozen panel binds it: RMSE, ROC AUC and macro F1.
    pub fn task_spec(self) -> Result<TaskSpec> {
        match self {
            Self::Regression => TaskSpec::regression(Objective::Rmse),
            Self::Binary => TaskSpec::binary(Objective::RocAuc),
            Self::Multiclass => TaskSpec::multiclass(Objective::MacroF1),
        }
    }
}

/// Where the labels of a record came from.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelOrigin {
    /// Original role recorded for the labels, such as `development` or `test`.
    pub original_role: String,
    /// Digest of the burn receipt when optimization-visible labels were burned.
    pub burn_sha256: Option<String>,
}

impl LabelOrigin {
    /// Test labels that optimization could see: an original test role or a recorded burn.
    pub fn is_adaptive(&self) -> bool {
        self.original_role == "test" || self.burn_sha256.is_some()
    }
}

/// The fields of one qualified record, borrowed from the state reader.
pub(crate) struct Admission<'a> {
    pub program: &'a str,
    pub recipe_sha256: &'a str,
    pub task: &'a str,
    pub dataset_sha256: &'a str,
    pub cut_sha256: &'a str,
    pub comparison_sha256: &'a str,
    pub record_sha256: &'a str,
    pub label: LabelOrigin,
}

/// One admitted program with its provenance.
#[derive(Clone, Debug)]
pub struct QualifiedProgram {
    program: String,
    recipe_sha256: String,
    task: TaskLabel,
    dataset_sha256: String,
    cut_sha256: String,
    comparison_sha256: String,
    record_sha256: String,
    label: LabelOrigin,
}

impl QualifiedProgram {
    /// Admit a record. The program must be its own canonical text, the recipe digest must be
    /// the sha256 of that text, every digest must be 64 lowercase hex digits and the task must
    /// be known.
    pub(crate) fn admit(a: Admission<'_>) -> Result<Self> {
        let reparsed = GenomeIr::parse(a.program).ok().map(|ir| ir.canonical());
        ensure!(
            reparsed.as_deref() == Some(a.program),
            Code("noncanonical_program")
        );
        ensure!(
            is_hex64(a.recipe_sha256) && sha256_hex(a.program.as_bytes()) == a.recipe_sha256,
            Code("recipe_hash_mismatch")
        );
        for digest in [
            a.dataset_sha256,
            a.cut_sha256,
            a.comparison_sha256,
            a.record_sha256,
        ] {
            ensure!(is_hex64(digest), Code("input_join_failed"));
        }
        let task = TaskLabel::parse(a.task)?;
        ensure!(
            a.label.burn_sha256.as_deref().is_none_or(is_hex64),
            Code("input_join_failed")
        );
        Ok(Self {
            program: a.program.to_owned(),
            recipe_sha256: a.recipe_sha256.to_owned(),
            task,
            dataset_sha256: a.dataset_sha256.to_owned(),
            cut_sha256: a.cut_sha256.to_owned(),
            comparison_sha256: a.comparison_sha256.to_owned(),
            record_sha256: a.record_sha256.to_owned(),
            label: a.label,
        })
    }

    /// Test-only: the same record with its program text and recipe digest replaced, so the
    /// table builder's own re-checks can be exercised.
    #[cfg(test)]
    pub(crate) fn tampered(&self, program: &str, recipe_sha256: &str) -> Self {
        Self {
            program: program.to_owned(),
            recipe_sha256: recipe_sha256.to_owned(),
            ..self.clone()
        }
    }

    /// Return the admitted canonical program text.
    pub fn program(&self) -> &str {
        &self.program
    }

    /// Return the digest of the admitted canonical program text.
    pub fn recipe_sha256(&self) -> &str {
        &self.recipe_sha256
    }

    /// Return the task under which the program was qualified.
    pub fn task(&self) -> TaskLabel {
        self.task
    }

    /// Return the digest of the source dataset.
    pub fn dataset_sha256(&self) -> &str {
        &self.dataset_sha256
    }

    /// Return the digest identifying the source dataset cut.
    pub fn cut_sha256(&self) -> &str {
        &self.cut_sha256
    }

    /// Return the digest of the measured comparison identity.
    pub fn comparison_sha256(&self) -> &str {
        &self.comparison_sha256
    }

    /// Return the digest of the qualified source record.
    pub fn record_sha256(&self) -> &str {
        &self.record_sha256
    }

    /// Return the label provenance attached to the source record.
    pub fn label(&self) -> &LabelOrigin {
        &self.label
    }
}

/// Every admitted program of one source read, with the source identity.
#[derive(Clone, Debug)]
pub struct QualifiedSet {
    programs: Vec<QualifiedProgram>,
    source_schema: String,
    source_sha256: String,
    excluded_at_source: u32,
}

impl QualifiedSet {
    /// A set of admitted programs. Record digests must be unique.
    pub(crate) fn new(
        programs: Vec<QualifiedProgram>,
        source_schema: &str,
        source_sha256: &str,
        excluded_at_source: u32,
    ) -> Result<Self> {
        ensure!(
            !source_schema.is_empty() && is_hex64(source_sha256),
            Code("input_join_failed")
        );
        ensure!(
            u32::try_from(programs.len()).is_ok(),
            Code("input_join_failed")
        );
        {
            let mut records = BTreeSet::new();
            for program in &programs {
                ensure!(
                    records.insert(program.record_sha256.as_str()),
                    Code("input_join_failed")
                );
            }
        }
        Ok(Self {
            programs,
            source_schema: source_schema.to_owned(),
            source_sha256: source_sha256.to_owned(),
            excluded_at_source,
        })
    }

    /// Return every admitted program in source order.
    pub fn programs(&self) -> &[QualifiedProgram] {
        &self.programs
    }

    /// Return the schema identifier of the qualified source.
    pub fn source_schema(&self) -> &str {
        &self.source_schema
    }

    /// Return the digest of the qualified source read.
    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }

    /// Return the number of source records excluded before qualification.
    pub fn excluded_at_source(&self) -> u32 {
        self.excluded_at_source
    }
}

/// A reader of qualified programs. The eligible-State reader (`teaching::StateTeaching`)
/// implements it.
pub(crate) trait QualifiedSource {
    fn qualified(&self) -> Result<QualifiedSet>;
}

/// Private map from dataset digest to a family label, with the sha256 of the bytes it was
/// read from. The table builder records that digest, so a table's recorded map digest is
/// always the digest of the map that labelled it.
///
/// A label equal to a key or constant of the shareable summary (for example `passed`,
/// `checks` or `schema`) makes the pre-write scan refuse every summary of a table that
/// carries it. That fails closed; choose labels that are not summary words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyMap {
    labels: BTreeMap<String, String>,
    sha256: String,
}

impl FamilyMap {
    /// Keys must be 64 lowercase hex digits and labels must match `[a-z0-9_-]{6,64}`. The
    /// digest is that of the map's compact JSON bytes.
    pub fn from_map(map: BTreeMap<String, String>) -> Result<Self> {
        let bytes = serde_json::to_vec(&map).map_err(|_| Code("invalid_family_map"))?;
        Self::build(map, &bytes)
    }

    /// Parse the private JSON object `{ "<dataset sha256>": "<label>", ... }`. The digest is
    /// that of `bytes`, exactly as read.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let map: BTreeMap<String, String> =
            serde_json::from_slice(bytes).map_err(|_| Code("invalid_family_map"))?;
        Self::build(map, bytes)
    }

    fn build(labels: BTreeMap<String, String>, bytes: &[u8]) -> Result<Self> {
        for (dataset, label) in &labels {
            ensure!(
                is_hex64(dataset) && is_family_label(label),
                Code("invalid_family_map")
            );
        }
        Ok(Self {
            labels,
            sha256: sha256_hex(bytes),
        })
    }

    /// Lowercase hex sha256 of the bytes the map was read from.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// The label of one dataset, if the map has one.
    pub fn label(&self, dataset_sha256: &str) -> Option<&str> {
        self.labels.get(dataset_sha256).map(String::as_str)
    }
}

/// `[a-z0-9_-]{6,64}`.
pub(crate) fn is_family_label(label: &str) -> bool {
    (6..=64).contains(&label.len())
        && label.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_of;
    use crate::fixture;

    fn admission<'a>(program: &'a str, recipe: &'a str, digests: &'a [String; 4]) -> Admission<'a> {
        Admission {
            program,
            recipe_sha256: recipe,
            task: "regression",
            dataset_sha256: &digests[0],
            cut_sha256: &digests[1],
            comparison_sha256: &digests[2],
            record_sha256: &digests[3],
            label: fixture::validation_label(),
        }
    }

    fn digests() -> [String; 4] {
        ["dataset", "cut", "comparison", "record"].map(fixture::digest)
    }

    #[test]
    fn admit_rejects_noncanonical_program() {
        // The member table's strings are not canonical as written (parameter order).
        let written = feat_math::members::CORE_V1[1].pipe;
        let canonical = fixture::canonical(written);
        assert_ne!(written, canonical);
        let digests = digests();
        let written_recipe = sha256_hex(written.as_bytes());
        let error = QualifiedProgram::admit(admission(written, &written_recipe, &digests))
            .expect_err("non-canonical text is refused");
        assert_eq!(code_of(&error), Some("noncanonical_program"));

        let error = QualifiedProgram::admit(admission("(pipe X", &written_recipe, &digests))
            .expect_err("unparsable text is refused");
        assert_eq!(code_of(&error), Some("noncanonical_program"));

        let recipe = sha256_hex(canonical.as_bytes());
        let admitted = QualifiedProgram::admit(admission(&canonical, &recipe, &digests))
            .expect("the canonical text is admitted");
        assert_eq!(admitted.program(), canonical);
        assert_eq!(admitted.recipe_sha256(), recipe);
        assert_eq!(admitted.task(), TaskLabel::Regression);
    }

    #[test]
    fn admit_rejects_recipe_hash_mismatch() {
        let program = fixture::canonical("(pipe X (standardize) (ridge))");
        let digests = digests();
        let other = sha256_hex(b"(pipe X (robust_scale) (ridge))");
        let error = QualifiedProgram::admit(admission(&program, &other, &digests))
            .expect_err("a foreign digest is refused");
        assert_eq!(code_of(&error), Some("recipe_hash_mismatch"));

        let upper = sha256_hex(program.as_bytes()).to_uppercase();
        let error = QualifiedProgram::admit(admission(&program, &upper, &digests))
            .expect_err("an uppercase digest is refused");
        assert_eq!(code_of(&error), Some("recipe_hash_mismatch"));
    }

    #[test]
    fn admit_rejects_bad_digests_and_unknown_task() {
        let program = fixture::canonical("(pipe X (standardize) (ridge))");
        let recipe = sha256_hex(program.as_bytes());
        for position in 0..4 {
            for bad in [
                String::from("abc"),
                fixture::digest("x").to_uppercase(),
                format!("{}g", &fixture::digest("x")[..63]),
            ] {
                let mut digests = digests();
                digests[position] = bad;
                let error = QualifiedProgram::admit(admission(&program, &recipe, &digests))
                    .expect_err("a malformed digest is refused");
                assert_eq!(code_of(&error), Some("input_join_failed"));
            }
        }

        let digests = digests();
        let mut unknown = admission(&program, &recipe, &digests);
        unknown.task = "timeseries";
        let error = QualifiedProgram::admit(unknown).expect_err("an unknown task is refused");
        assert_eq!(code_of(&error), Some("input_join_failed"));

        let mut burned = admission(&program, &recipe, &digests);
        burned.label = LabelOrigin {
            original_role: "test".into(),
            burn_sha256: Some("not-a-digest".into()),
        };
        let error = QualifiedProgram::admit(burned).expect_err("a malformed burn is refused");
        assert_eq!(code_of(&error), Some("input_join_failed"));

        for task in TaskLabel::ALL {
            assert_eq!(TaskLabel::parse(task.as_str()).unwrap(), task);
            assert!(task.task_spec().is_ok());
        }
    }

    #[test]
    fn label_origin_marks_test_role_and_burn_as_adaptive() {
        let plain = |role: &str| LabelOrigin {
            original_role: role.into(),
            burn_sha256: None,
        };
        assert!(plain("test").is_adaptive());
        assert!(!plain("validation").is_adaptive());
        assert!(!plain("train").is_adaptive());
        let burned = LabelOrigin {
            original_role: "validation".into(),
            burn_sha256: Some(fixture::digest("burn")),
        };
        assert!(burned.is_adaptive());
        assert!(fixture::adaptive_label().is_adaptive());
        assert!(!fixture::validation_label().is_adaptive());
    }
}
