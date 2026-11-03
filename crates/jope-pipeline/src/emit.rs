//! Stage 6: emit standalone sources from the selected genome.
//!
//! This reproduces `crates/feat-math/examples/emit.rs`: `transpile::emit` for the
//! bit-identical Python/Rust pair plus the `battle-artifact-v1` manifest,
//! `emit_cpp17` and `emit_javascript` behind their own coverage gates, and a
//! `targets.txt` naming what was emitted. The legacy entrypoint uses
//! `EmitOptions::default()`; the explicit option entrypoint binds opt-in changes.
//! The title, feature names, context and `fast_runtime` flag all
//! change the Python and Rust bytes, and those bytes are what
//! `deploy/ci/emitters.sh` pins.
//!
//! There is exactly one C++ emitter. C++20 is a compile flag over the same
//! source, so `cpp17` and `cpp20` are two artifact rows sharing one `path` and
//! differing only in the recorded `build` command.
//!
//! A requested language the genome is not covered for is recorded as an
//! exclusion with the emitter's own reason, never silently dropped.

use anyhow::{bail, ensure, Context, Result};
use feat_math::{
    cxx::{self, CxxStd},
    genome::GenomeIr,
    transpile::{self, EmitOptions},
};
use std::path::{Path, PathBuf};

use crate::sha256_hex as hash;

pub mod fitted;

mod legacy;
mod target_emission;

pub use legacy::{cpp_source, emit, emit_with_options, emit_with_training};
pub use target_emission::{
    emit_numeric_v2, emit_v3, publish_v3, read_identity_v3, reload_numeric_v2, EmittedV3,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Language {
    Python,
    Rust,
    Cpp17,
    Cpp20,
    JavaScript,
    Go,
    TypeScript,
    Cpp17Fitted,
    C17,
    Cuda,
}

impl Language {
    pub const ALL: [Language; 5] = [
        Language::Rust,
        Language::Python,
        Language::Cpp17,
        Language::Cpp20,
        Language::JavaScript,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Language::Python => "python",
            Language::Rust => "rust",
            Language::Cpp17 => "cpp17",
            Language::Cpp20 => "cpp20",
            Language::JavaScript => "javascript",
            Language::TypeScript => "typescript",
            Language::Go => "go",
            Language::Cpp17Fitted => "cpp17-fitted",
            Language::C17 => "c17",
            Language::Cuda => "cuda",
        }
    }

    pub fn parse(value: &str) -> Option<Language> {
        match value.trim().to_ascii_lowercase().as_str() {
            "python" | "py" => Some(Language::Python),
            "rust" | "rs" => Some(Language::Rust),
            "cpp17" | "c++17" => Some(Language::Cpp17),
            "cpp20" | "c++20" => Some(Language::Cpp20),
            "javascript" | "js" => Some(Language::JavaScript),
            "typescript" | "ts" => Some(Language::TypeScript),
            "go" | "golang" => Some(Language::Go),
            "cpp17-fitted" => Some(Language::Cpp17Fitted),
            "c17" | "c" => Some(Language::C17),
            "cuda" | "cuda-cpp" | "cu" => Some(Language::Cuda),
            _ => None,
        }
    }

    /// Legacy recipes, `cpp17-fitted`, or `all` (the legacy recipes). Duplicates collapse; the order is
    /// fixed so the emitted set does not depend on how the flag was written.
    pub fn parse_list(value: &str) -> Result<Vec<Language>> {
        let mut chosen = std::collections::BTreeSet::new();
        for part in value.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            if part.eq_ignore_ascii_case("all") {
                chosen.extend(Language::ALL);
                continue;
            }
            let language = Language::parse(part).with_context(|| {
                format!(
                    "unknown emit target {part:?}; expected rust, python, cpp17, cpp20, js, ts, go, c17, cuda, cpp17-fitted or all"
                )
            })?;
            chosen.insert(language);
        }
        if chosen.is_empty() {
            bail!("at least one emit target is required");
        }
        Ok(Language::ALL
            .into_iter()
            .chain([
                Language::Go,
                Language::TypeScript,
                Language::Cpp17Fitted,
                Language::C17,
                Language::Cuda,
            ])
            .filter(|l| chosen.contains(l))
            .collect())
    }
}

/// One emitted source file, as the receipt records it.
#[derive(Clone, Debug)]
pub struct Artifact {
    pub language: &'static str,
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    /// How the artifact is turned into a program, when that needs a compiler.
    pub build: Option<String>,
}

/// A requested language the genome is not covered for.
#[derive(Clone, Debug)]
pub struct Exclusion {
    pub language: &'static str,
    pub reason: String,
}

pub struct Emitted {
    pub canonical_genome: String,
    pub genome_sha256: String,
    pub artifacts: Vec<Artifact>,
    pub exclusions: Vec<Exclusion>,
}

/// Test-only hook: run `body` with the transpiler lowering `genome` instead of the one requested.
///
/// This exists so the emit-to-selection binding can be tested at all. It is compiled out of any
/// non-test build -- `emit()`'s `#[cfg(not(test))]` arm has no branch on it.
#[cfg(test)]
pub(crate) mod seam {
    use super::*;
    use std::cell::RefCell;

    thread_local! {
        static RENDER_INSTEAD: RefCell<Option<String>> = const { RefCell::new(None) };
    }

    pub(crate) fn swapped_genome() -> Result<Option<GenomeIr>> {
        let raw = RENDER_INSTEAD.with(|c| c.borrow().clone());
        match raw {
            Some(g) => Ok(Some(
                GenomeIr::parse(g.trim()).with_context(|| format!("parse swapped genome {g}"))?,
            )),
            None => Ok(None),
        }
    }

    /// Render `genome` instead of whatever `emit()` is asked for, for the duration of `body`.
    pub(crate) fn render_instead<T>(genome: &str, body: impl FnOnce() -> T) -> T {
        RENDER_INSTEAD.with(|c| *c.borrow_mut() = Some(genome.to_owned()));
        let out = body();
        RENDER_INSTEAD.with(|c| *c.borrow_mut() = None);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::legacy::read_identity;
    use super::*;

    #[test]
    fn c17_is_explicit_and_publishes_complete_native_bundle() {
        assert!(!Language::parse_list("all")
            .unwrap()
            .contains(&Language::C17));
        assert_eq!(Language::parse_list("c,c17").unwrap(), vec![Language::C17]);
        let dir = tempfile::tempdir().unwrap();
        let e = emit(
            "(pipe X (standardize) (histgbm rounds=2))",
            &[Language::C17],
            dir.path(),
        )
        .unwrap();
        assert_eq!(e.artifacts.len(), 4);
        assert!(e.exclusions.is_empty());
        for a in e.artifacts {
            assert_eq!(a.language, "c17");
            assert_eq!(
                hash(&std::fs::read(dir.path().join(a.path)).unwrap()),
                a.sha256
            );
        }
        assert_eq!(
            std::fs::read_to_string(dir.path().join("targets.txt")).unwrap(),
            "c17\n"
        );
    }

    /// The identity `emit()` reports must describe the artifacts it wrote, not the genome it was
    /// asked for. Reported on PR #44 by the independent reviewer, who demonstrated the gap by
    /// making the transpiler lower `(pipe X (ridge alpha=999.0))` while the receipt went on
    /// claiming the selected genome -- exit 0, `ok: true`, every `ensure!` satisfied and
    /// `verify-receipt` green, with `model.rs` containing a completely different model.
    #[test]
    fn the_reported_identity_follows_the_artifacts_not_the_request() {
        let dir = tempfile::tempdir().unwrap();
        let requested = "(pipe X (rank append=true) (gpr m=400 steps=60))";
        let actually_rendered = "(pipe X (ridge alpha=999.0))";

        // Control: with no swap, the reported identity is the requested genome.
        let honest = emit(
            requested,
            &[Language::Python, Language::Rust],
            dir.path().join("honest").as_path(),
        )
        .unwrap();
        assert_eq!(honest.canonical_genome, requested);

        // With the transpiler lowering something else, the reported identity must follow the
        // artifacts. Before this fix it echoed `requested` and the pipeline shipped happily.
        let swapped = seam::render_instead(actually_rendered, || {
            emit(
                requested,
                &[Language::Python, Language::Rust],
                dir.path().join("swapped").as_path(),
            )
        })
        .unwrap();
        assert_eq!(
            swapped.canonical_genome, actually_rendered,
            "emit() reported the genome it was asked for, not the one it wrote"
        );
        assert_ne!(
            swapped.genome_sha256, honest.genome_sha256,
            "two different emissions reported the same genome digest"
        );
    }

    /// A manifest is only evidence if it describes the files beside it.
    #[test]
    fn a_manifest_that_disagrees_with_the_file_it_names_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("emit");
        let good = emit(
            "(pipe X (ridge alpha=1.0))",
            &[Language::Python, Language::Rust],
            out.as_path(),
        )
        .unwrap();

        // Rewrite model.py underneath a manifest that still names the original digest.
        std::fs::write(out.join("model.py"), "# not the emitted model\n").unwrap();
        let artifacts: Vec<Artifact> = good
            .artifacts
            .iter()
            .map(|a| Artifact {
                sha256: if a.path == "model.py" {
                    hash(b"# not the emitted model\n")
                } else {
                    a.sha256.clone()
                },
                language: a.language,
                path: a.path.clone(),
                bytes: a.bytes,
                build: a.build.clone(),
            })
            .collect();
        let err = read_identity(out.as_path(), &artifacts)
            .expect_err("a manifest was accepted over a file it does not describe");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("model.py") && msg.contains("but the file written is"),
            "the refusal must name the file and both digests, got: {msg}"
        );
    }

    #[test]
    fn language_lists_parse_into_a_fixed_order() {
        assert_eq!(
            Language::parse_list("js,rust,cpp20,py,cpp17").unwrap(),
            Language::ALL.to_vec()
        );
        assert_eq!(Language::parse_list("all").unwrap(), Language::ALL.to_vec());
        assert_eq!(
            Language::parse_list("rs, rs ,rust").unwrap(),
            vec![Language::Rust]
        );
        assert_eq!(Language::parse_list("golang").unwrap(), [Language::Go]);
        assert!(Language::parse_list(" ").is_err());
    }

    #[test]
    fn the_bootstrap_genome_emits_every_language_and_shares_one_cpp_source() {
        let temp = tempfile::tempdir().unwrap();
        let emitted = emit(crate::select::BOOTSTRAP_GENOME, &Language::ALL, temp.path()).unwrap();
        assert!(emitted.exclusions.is_empty(), "{:?}", emitted.exclusions);
        let languages: Vec<&str> = emitted.artifacts.iter().map(|a| a.language).collect();
        assert_eq!(
            languages,
            ["python", "rust", "cpp17", "cpp20", "javascript"]
        );
        let cpp: Vec<&Artifact> = emitted
            .artifacts
            .iter()
            .filter(|a| a.path == "model.cpp")
            .collect();
        assert_eq!(cpp.len(), 2);
        assert_eq!(cpp[0].sha256, cpp[1].sha256);
        assert!(cpp[0].build.as_ref().unwrap().contains("-std=c++17"));
        assert!(cpp[1].build.as_ref().unwrap().contains("-std=c++20"));
        for artifact in &emitted.artifacts {
            let path = temp.path().join(&artifact.path);
            assert_eq!(std::fs::metadata(&path).unwrap().len(), artifact.bytes);
        }
        assert_eq!(
            std::fs::read_to_string(temp.path().join("targets.txt")).unwrap(),
            "py rs cpp js\n"
        );
        let typescript_dir = temp.path().join("typescript");
        let typescript = emit(
            crate::select::BOOTSTRAP_GENOME,
            &[Language::TypeScript],
            &typescript_dir,
        )
        .unwrap();
        assert!(typescript.exclusions.is_empty());
        assert_eq!(typescript.artifacts.len(), 2);
        assert_eq!(
            std::fs::read_to_string(typescript_dir.join("targets.txt")).unwrap(),
            "ts\n"
        );
        assert!(temp.path().join("manifest.json").is_file());
        assert_eq!(
            emitted.genome_sha256,
            hash(emitted.canonical_genome.as_bytes())
        );
        assert_eq!(
            emitted.canonical_genome,
            feat_math::genome::GenomeIr::parse(crate::select::BOOTSTRAP_GENOME)
                .unwrap()
                .canonical()
        );
        let go_dir = temp.path().join("go");
        let go = emit(crate::select::BOOTSTRAP_GENOME, &[Language::Go], &go_dir).unwrap();
        assert!(go.exclusions.is_empty());
        assert_eq!(go.artifacts.len(), 4);
        assert_eq!(
            std::fs::read_to_string(go_dir.join("targets.txt")).unwrap(),
            "golang\n"
        );
        assert!(temp.path().join("manifest.json").is_file());
        assert_eq!(
            emitted.genome_sha256,
            hash(emitted.canonical_genome.as_bytes())
        );
        assert_eq!(
            emitted.canonical_genome,
            feat_math::genome::GenomeIr::parse(crate::select::BOOTSTRAP_GENOME)
                .unwrap()
                .canonical()
        );
    }

    #[test]
    fn a_head_the_cpp_and_js_emitters_do_not_cover_is_recorded_not_dropped() {
        let temp = tempfile::tempdir().unwrap();
        let genome = "(pipe X (histgbm depth=5 lr=0.15 rounds=150))";
        let emitted = emit(genome, &Language::ALL, temp.path()).unwrap();
        let languages: Vec<&str> = emitted.artifacts.iter().map(|a| a.language).collect();
        assert_eq!(languages, ["python", "rust"]);
        let excluded: Vec<&str> = emitted.exclusions.iter().map(|e| e.language).collect();
        assert_eq!(excluded, ["cpp17", "cpp20", "javascript"]);
        assert!(emitted.exclusions.iter().all(|e| !e.reason.is_empty()));
        assert_eq!(
            std::fs::read_to_string(temp.path().join("targets.txt")).unwrap(),
            "py rs\n"
        );
    }
}

#[cfg(test)]
mod fitted_tests;
