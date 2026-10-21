//! Records the build target and the source commit the builder declares.
fn main() {
    println!("cargo:rerun-if-env-changed=JOPE_BUILD_SOURCE_SHA");
    let target = std::env::var("TARGET").unwrap_or_default();
    println!("cargo:rustc-env=JOPE_BUILD_TARGET={target}");
    let profile = std::env::var("PROFILE").unwrap_or_default();
    println!("cargo:rustc-env=JOPE_BUILD_PROFILE_KIND={profile}");
}
