use std::{fs, path::Path};
fn copy_tree(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for e in fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        let p = e.path();
        let out = dst.join(e.file_name());
        if p.is_dir() {
            copy_tree(&p, &out);
        } else if p.extension().is_some_and(|s| s == "rs") {
            // Preserve the encoder's original 24k-cell activation cap. The 240k
            // override could stall CUDA on narrow tables despite high GPU load.
            let text = fs::read_to_string(&p).unwrap();
            fs::write(out, text).unwrap();
        }
    }
}
fn main() {
    let base = Path::new("../../assets/source/hyperion-jain/crates/feat-core/src")
        .canonicalize()
        .unwrap();
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed={}", base.display());
    let text = fs::read_to_string(base.join("hyperion.rs"))
        .unwrap()
        .replace("fn load_settings(", "pub fn load_settings(")
        .replace("fn fit_quantile_bounds(", "pub fn fit_quantile_bounds(");
    fs::write(out.join("hyperion.rs"), text).unwrap();
    copy_tree(&base.join("hyperion_v7"), &out.join("hyperion_v7"));
    fs::write(
        out.join("modules.rs"),
        format!(
            "#[path={:?}] pub mod hyperion;\n#[path={:?}] pub mod hyperion_v7;\n",
            out.join("hyperion.rs"),
            out.join("hyperion_v7/mod.rs")
        ),
    )
    .unwrap();
}
