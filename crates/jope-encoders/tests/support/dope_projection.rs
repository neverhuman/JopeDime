//! Explicit, local artifact qualification. Never materialize the full governed bundle.
use candle_core::Device;
use starforge::bundle::{LoadedModelBundle, VerifiedJopePolicyAssets};
use starforge::jope_policy::JopePolicyModel;
use std::path::{Path, PathBuf};

fn minimal_projection() -> anyhow::Result<PathBuf> {
    let source = std::env::var_os("JOPE_MODELS_ROOT")
        .expect("explicit policy projection qualification requires JOPE_MODELS_ROOT");
    let source = Path::new(&source);
    let layout = VerifiedJopePolicyAssets::reviewed_layout()?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/landmarks/p4-policy-projection")
        .join(format!("process-{}", std::process::id()));
    std::fs::create_dir_all(&root)?;
    let parent = if source.join("model-bundle.v1.json").exists() {
        source.join("model-bundle.v1.json")
    } else {
        source.join("artifacts/model-bundle.v1.json")
    };
    std::fs::copy(parent, root.join("model-bundle.v1.json"))?;
    for path in layout
        .files
        .iter()
        .map(|f| &f.path)
        .chain(layout.notices.keys())
    {
        let src = if source.join(path).exists() {
            source.join(path)
        } else {
            source.join(path.strip_prefix("artifacts/").unwrap_or(path))
        };
        let dst = root.join(path);
        std::fs::create_dir_all(dst.parent().unwrap())?;
        // Copy into a private fixture: corruption tests must never alter governed cache inodes.
        std::fs::copy(src, dst)?;
    }
    Ok(root)
}

#[test]
#[ignore = "requires the governed selected policy artifacts; missing artifacts fail qualification"]
fn minimal_projection_verifies_without_opening_unrelated_assets() -> anyhow::Result<()> {
    let root = minimal_projection()?;
    let forbidden = root.join("artifacts/starforge");
    std::fs::create_dir_all(&forbidden)?;
    #[cfg(unix)]
    std::os::unix::fs::symlink(
        "/does-not-exist",
        forbidden.join("starlight_core_regressor.safetensors"),
    )?;
    let assets = VerifiedJopePolicyAssets::load(&root)?;
    let proposal: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../docs/qualification/optional-landmarks-policy-assets.json"
    ))?;
    assert_eq!(
        assets.identity().projection_digest,
        "eeae7b5b7e35f4976db4bd0b424685eb6e5d4a42152e16c92d7d6b088485217c"
    );
    assert_eq!(
        proposal["parent_manifest_sha256"],
        assets.identity().parent_manifest_sha256
    );
    std::fs::write(
        root.join("verified-layout.json"),
        serde_json::to_vec_pretty(assets.layout())?,
    )?;
    assert!(LoadedModelBundle::load(&root).is_err());
    let model = JopePolicyModel::load_policy_assets(&assets, Device::Cpu)?;
    assert_eq!(model.weights_sha256(), assets.identity().policy_sha256);
    assert_eq!(model.projection_identity(), Some(assets.identity()));
    let context = model.context(&vec![0.25; 1536], &vec![0.5; 512])?;
    assert!(model
        .context_values(&context)?
        .iter()
        .all(|v| v.is_finite()));
    let notice = root.join("docs/notices/jain-governed-models.txt");
    std::fs::write(&notice, b"changed notice")?;
    assert!(VerifiedJopePolicyAssets::load(&root).is_err());
    // The already validated capability owns immutable exact model bytes.
    let again = JopePolicyModel::load_policy_assets(&assets, Device::Cpu)?;
    assert_eq!(
        model.context_values(&context)?,
        again.context_values(&again.context(&vec![0.25; 1536], &vec![0.5; 512])?)?
    );
    Ok(())
}
