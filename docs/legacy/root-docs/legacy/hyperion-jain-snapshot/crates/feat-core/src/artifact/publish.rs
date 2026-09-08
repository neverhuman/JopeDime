use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub fn artifact_staging_dir(model_out: &Path) -> PathBuf {
    model_out.join(format!(".jain-staging-{}", std::process::id()))
}

pub fn publish_artifact(staging: &Path, model_out: &Path) -> Result<()> {
    std::fs::create_dir_all(model_out)
        .with_context(|| format!("creating {}", model_out.display()))?;
    let final_manifest = model_out.join("manifest.json");
    let _ = std::fs::remove_file(&final_manifest);
    let final_models = model_out.join("models");
    if final_models.exists() {
        std::fs::remove_dir_all(&final_models)
            .with_context(|| format!("removing {}", final_models.display()))?;
    }
    std::fs::rename(staging.join("models"), &final_models).with_context(|| {
        format!(
            "publishing model files from {} to {}",
            staging.display(),
            final_models.display()
        )
    })?;

    let staging_starforge = staging.join("starforge");
    if staging_starforge.exists() {
        let final_starforge = model_out.join("starforge");
        if final_starforge.exists() {
            std::fs::remove_dir_all(&final_starforge)
                .with_context(|| format!("removing {}", final_starforge.display()))?;
        }
        std::fs::rename(&staging_starforge, &final_starforge).with_context(|| {
            format!(
                "publishing Starforge contexts from {} to {}",
                staging_starforge.display(),
                final_starforge.display()
            )
        })?;
    }

    std::fs::rename(staging.join("manifest.json"), &final_manifest).with_context(|| {
        format!(
            "publishing manifest from {} to {}",
            staging.display(),
            final_manifest.display()
        )
    })?;
    let _ = std::fs::remove_dir_all(staging);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn publish_artifact_preserves_starforge_context_dir() {
        let base = std::env::temp_dir().join(format!(
            "jain_core_publish_test_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&base);
        let staging = base.join("staging");
        let model_out = base.join("model");
        fs::create_dir_all(staging.join("models")).unwrap();
        fs::create_dir_all(staging.join("starforge")).unwrap();
        fs::write(staging.join("models/00_native.cbm"), b"m").unwrap();
        fs::write(
            staging.join("starforge/00_starforge_reduced_context.json"),
            b"{}",
        )
        .unwrap();
        fs::write(staging.join("manifest.json"), b"{}").unwrap();

        publish_artifact(&staging, &model_out).unwrap();

        assert!(model_out.join("manifest.json").is_file());
        assert!(model_out.join("models/00_native.cbm").is_file());
        assert!(model_out
            .join("starforge/00_starforge_reduced_context.json")
            .is_file());
        assert!(!staging.exists());

        let _ = fs::remove_dir_all(&base);
    }
}
