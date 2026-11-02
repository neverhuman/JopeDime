use super::*;

fn v3_language(target: feat_math::artifact_manifest::Target, path: &str) -> &'static str {
    use feat_math::artifact_manifest::Target;
    match target {
        Target::Python => "python",
        Target::Rust => "rust",
        Target::Cpp17 => "cpp17",
        Target::Cpp20 => "cpp20",
        Target::JavaScript => "javascript",
        Target::C17 => "c17",
        Target::Cuda => "cuda",
        Target::TypeScript if path.ends_with(".kernel.rs") => "typescript-kernel",
        Target::TypeScript => "typescript",
        Target::Go if path.ends_with(".kernel.rs") => "go-kernel",
        Target::Go if path == "go.mod" => "go-module",
        Target::Go if path == "go.sum" => "go-checksums",
        Target::Go => "go",
        Target::Cpp17RecipeV2 => "cpp17-recipe-v2",
        Target::Cpp17Fitted if path.ends_with(".codec.json") => "cpp17-fitted-codec",
        Target::Cpp17Fitted if path.ends_with("standalone.cpp") => "cpp17-fitted-standalone",
        Target::Cpp17Fitted => "cpp17-fitted",
    }
}

pub(super) fn check_artifact_manifest(
    manifest: &serde_json::Value,
    artifact: &crate::receipt::Artifact,
) -> Result<&'static str> {
    use crate::emit::Language;
    let (expected_path, language) = match Language::parse(&artifact.language) {
        Some(Language::Cuda) => {
            ensure!(
                [
                    "cuda/CMakeLists.txt",
                    "cuda/manifest.json",
                    "cuda/model.cu",
                    "cuda/model.hpp",
                    "cuda/model_core.rs",
                    "cuda/model_native.rs"
                ]
                .contains(&artifact.path.as_str()),
                "unknown CUDA artifact"
            );
            (artifact.path.as_str(), "cuda")
        }
        Some(Language::C17) => {
            ensure!(
                [
                    "c17/model.h",
                    "c17/model_core.rs",
                    "c17/model_ffi.rs",
                    "c17/manifest.json"
                ]
                .contains(&artifact.path.as_str()),
                "unknown C17 artifact"
            );
            (artifact.path.as_str(), "c17")
        }
        Some(Language::Rust) => ("model.rs", "rust"),
        Some(Language::Python) => ("model.py", "python"),
        Some(Language::JavaScript) => ("model.js", "javascript"),
        Some(Language::TypeScript) => ("model.ts", "typescript"),
        None if artifact.language == "typescript-kernel" => {
            ("model.kernel.rs", "typescript-kernel")
        }
        Some(Language::Go) => ("model.go", "go"),
        None if artifact.language == "go-module" => ("go.mod", "go-module"),
        None if artifact.language == "go-checksums" => ("go.sum", "go-checksums"),
        None if artifact.language == "go-kernel" => ("model.go.kernel.rs", "go-kernel"),
        Some(language @ (Language::Cpp17 | Language::Cpp20)) => ("model.cpp", language.as_str()),
        Some(Language::Cpp17Fitted) => ("cpp17-fitted/model.cpp", "cpp17-fitted"),
        None if artifact.language == "cpp17-fitted-codec" => {
            ("cpp17-fitted/model.codec.json", "cpp17-fitted-codec")
        }
        None if artifact.language == "cpp17-fitted-standalone" => {
            ("cpp17-fitted/standalone.cpp", "cpp17-fitted-standalone")
        }
        None => anyhow::bail!("unknown emitted language"),
    };
    ensure!(
        artifact.path == expected_path,
        "emitted artifact path disagrees"
    );
    let digest_field = match expected_path {
        "model.rs" => Some("rs_sha256"),
        "model.py" => Some("py_sha256"),
        _ => None,
    };
    if let Some(field) = digest_field {
        ensure!(
            manifest[field].as_str() == Some(artifact.sha256.as_str()),
            "emitted artifact digest disagrees with selected manifest"
        );
    }
    Ok(language)
}

pub(super) fn check_artifact_targets(root: &Path, languages: &BTreeSet<&str>) -> Result<()> {
    let retained = String::from_utf8(read_evidence(root, Path::new("emit/targets.txt"))?)?;
    let mut targets = BTreeSet::new();
    for target in retained.split_whitespace() {
        ensure!(
            [
                "py",
                "rs",
                "cpp",
                "js",
                "ts",
                "golang",
                "c17",
                "cuda",
                "cpp17-fitted"
            ]
            .contains(&target)
                && targets.insert(target),
            "unknown or duplicate emitted target"
        );
    }
    let actual: BTreeSet<_> = languages
        .iter()
        .map(|language| match *language {
            "python" => "py",
            "rust" => "rs",
            "cpp17" | "cpp20" => "cpp",
            "javascript" => "js",
            "c17" => "c17",
            "cuda" => "cuda",
            "typescript" | "typescript-kernel" => "ts",
            "go" | "go-kernel" | "go-module" | "go-checksums" => "golang",
            _ => "cpp17-fitted",
        })
        .collect();
    ensure!(
        targets == actual,
        "artifact languages disagree with emitted targets"
    );
    for language in ["typescript", "typescript-kernel"] {
        ensure!(
            languages.contains(language) == targets.contains("ts"),
            "incomplete TypeScript artifact set"
        );
    }
    for language in ["go", "go-kernel", "go-module", "go-checksums"] {
        ensure!(
            languages.contains(language) == targets.contains("golang"),
            "incomplete Go artifact set"
        );
    }
    for language in [
        "cpp17-fitted",
        "cpp17-fitted-codec",
        "cpp17-fitted-standalone",
    ] {
        ensure!(
            languages.contains(language) == targets.contains("cpp17-fitted"),
            "incomplete fitted artifact set"
        );
    }
    Ok(())
}
