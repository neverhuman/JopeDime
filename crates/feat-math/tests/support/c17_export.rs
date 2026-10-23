//! C17 source identity, boundary safety, native callers, and option propagation.
use super::c17;
use feat_math::{
    artifact_manifest::{Target, TargetState},
    transpile::{self, EmitOptions},
    GenomeIr,
};
use std::{fs, path::PathBuf, process::Command};

fn dir(label: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("c17-{label}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}
fn fixture() -> GenomeIr {
    GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap()
}

#[test]
fn bundle_is_deterministic_complete_and_binds_unchanged_rust() {
    let g = fixture();
    let opts = EmitOptions::default();
    let bundle = transpile::emit_c17(&g, &opts).unwrap();
    assert_eq!(bundle.files, transpile::emit_c17(&g, &opts).unwrap().files);
    assert_eq!(bundle.files.len(), 4);
    let pair = transpile::emit(&g, &opts).unwrap();
    assert!(bundle.files["c17/model_core.rs"].starts_with(&pair.rust));
    assert!(bundle.files["c17/model_core.rs"].contains("#![forbid(unsafe_code)]"));
    let manifest: serde_json::Value =
        serde_json::from_str(&bundle.files["c17/manifest.json"]).unwrap();
    assert_eq!(
        manifest["rust_source_sha256"],
        feat_math::artifact_manifest::sha256(pair.rust.as_bytes())
    );
    for file in manifest["files"].as_array().unwrap() {
        let source = &bundle.files[file["path"].as_str().unwrap()];
        assert_eq!(
            file["sha256"],
            feat_math::artifact_manifest::sha256(source.as_bytes())
        );
        assert_eq!(file["bytes"], source.len());
    }
    for source in bundle.files.values() {
        assert!(transpile::lint_shipped_text(source).is_empty());
    }
    assert!(transpile::supports_c17(&g).is_ok());
    assert!(transpile::emit_c17(
        &g,
        &EmitOptions {
            math_kernel: transpile::MathKernel::Portable,
            ..opts
        }
    )
    .is_err());
}

#[test]
fn v3_manifest_binds_every_c_compilation_input_and_rejects_tampering() {
    let g = fixture();
    assert_eq!(Target::parse("c").unwrap(), Target::C17);
    assert!(Target::parse_list("c,c17").is_err());
    let bundle =
        transpile::emit_targets_v3(&g, &EmitOptions::default(), &[Target::Rust, Target::C17])
            .unwrap();
    assert!(bundle.manifest.all_supported());
    match &bundle
        .manifest
        .targets
        .iter()
        .find(|r| r.target == Target::C17)
        .unwrap()
        .state
    {
        TargetState::Supported { files, build, .. } => {
            assert_eq!(files.len(), 4);
            assert_eq!(build.abi, "c17-v1");
        }
        _ => panic!("C17 must be supported"),
    }
    let root = dir("v3").join("artifact");
    // The publication API refuses replacing prior evidence; use a fresh child each run.
    let root = root.with_extension(format!(
        "{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    use feat_math::artifact_manifest::{Domain, ExpectedArtifactV3};
    fs::create_dir_all(&root).unwrap();
    for (path, source) in &bundle.sources {
        let dest = root.join(path);
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(dest, source).unwrap();
    }
    let bytes = bundle.manifest.envelope_bytes(Domain::Native).unwrap();
    let expected = ExpectedArtifactV3::from_emission(&bytes, &bundle.manifest);
    fs::write(root.join("manifest.json"), bytes).unwrap();
    feat_math::artifact_manifest::verify_directory(&root, "manifest.json", &expected).unwrap();
    fs::write(root.join("c17/model_ffi.rs"), "changed").unwrap();
    assert!(
        feat_math::artifact_manifest::verify_directory(&root, "manifest.json", &expected).is_err()
    );
}

#[test]
fn c_abi_checks_shapes_handles_lifetime_and_concurrent_predictions() {
    let root = dir("abi");
    c17::build(&root, &fixture(), &EmitOptions::default());
    let source = include_str!("c17_contract.c");
    assert_eq!(
        c17::consumer(&root, "gcc", source).trim(),
        "C17 ABI contracts passed"
    );
    if Command::new("clang").arg("--version").output().is_ok() {
        assert_eq!(
            c17::consumer(&root, "clang", source).trim(),
            "C17 ABI contracts passed"
        );
    } else {
        assert_ne!(
            std::env::var("JOPE_PARITY_STRICT").as_deref(),
            Ok("1"),
            "missing required clang"
        );
        eprintln!("clang C17 caller not executed: compiler unavailable; strict CI requires it");
    }
    // C++ consumes the same public header with extern-C linkage.
    fs::write(
        root.join("cpp.cpp"),
        "#include \"model.h\"\nint main(){return jope_model_abi_version()!=1;}\n",
    )
    .unwrap();
    c17::run(Command::new("g++").current_dir(&root).args([
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "cpp.cpp",
        "libmodel.a",
        "-ldl",
        "-lpthread",
        "-lm",
        "-o",
        "cpp",
    ]));
    c17::run(&mut Command::new(root.join("cpp")));
}

#[test]
fn c17_preserves_input_bits_and_explicit_fast_runtime_options() {
    let root = dir("options");
    let g = GenomeIr::parse("(sym raw \"c0\")").unwrap();
    let opts = EmitOptions {
        fast_runtime: true,
        guard_zero_width_roll: true,
        partition_point_quantile_bounds: true,
        ..EmitOptions::default()
    };
    let train = vec![vec![0.0], vec![1.0]];
    let y = vec![0.0, 1.0];
    let test: Vec<_> = (0..130)
        .map(|i| {
            vec![match i % 4 {
                0 => -0.0,
                1 => f64::from_bits(1),
                2 => 1e100,
                _ => -1.25,
            }]
        })
        .collect();
    let got = c17::predictions(&root, &g, &opts, &train, &y, &test);
    let expected = feat_math::fit_predict(&g, &train, &y, &test).unwrap();
    assert_eq!(
        got,
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let m: serde_json::Value =
        serde_json::from_str(&transpile::emit_c17(&g, &opts).unwrap().files["c17/manifest.json"])
            .unwrap();
    assert_eq!(m["options"]["fast_runtime"], true);
    assert_eq!(m["options"]["partition_point_quantile_bounds"], true);
}

#[test]
fn c17_unwinding_is_contained_and_shared_library_is_callable() {
    let root = dir("panic");
    c17::build(&root, &fixture(), &EmitOptions::default());
    // Inject numerical faults into the safe generated facade solely for this boundary test.
    let core = root.join("model_core.rs");
    let source = fs::read_to_string(&core).unwrap();
    let source = source.replace(
        "model: Model::fit(x, y),",
        "model: { if y[0] == -9999.0 { panic!(\"fit probe\"); } Model::fit(x, y) },",
    );
    let source = source.replace(
        "self.model.predict(x)",
        "{ if x[0][0] == -9999.0 { panic!(\"predict probe\"); } self.model.predict(x) }",
    );
    let source = source.replace(
        "self.model.score(x, y)",
        "{ if x[0][0] == -9999.0 { panic!(\"score probe\"); } self.model.score(x, y) }",
    );
    assert!(
        source.contains("fit probe")
            && source.contains("predict probe")
            && source.contains("score probe")
    );
    fs::write(core, source).unwrap();
    c17::compile(&root);
    let c = r#"#include "model.h"
#include <assert.h>
int main(void) {
 double x[]={0,1,1,2,2,3,3,4},y[]={-9999,1,2,3},out=-77; uint64_t h=999;
 assert(jope_model_fit(x,8,y,4,4,2,&h)==JOPE_MODEL_PANIC && h==999);
 y[0]=0; assert(jope_model_fit(x,8,y,4,4,2,&h)==0);
 x[0]=-9999;
 assert(jope_model_predict(h,x,2,1,2,&out,1)==JOPE_MODEL_PANIC && out==-77);
 assert(jope_model_score(h,x,8,y,4,4,2,&out)==JOPE_MODEL_PANIC && out==-77);
 x[0]=0; assert(jope_model_predict(h,x,2,1,2,&out,1)==0);
 return jope_model_free(h);
}"#;
    c17::consumer(&root, "gcc", c);
    c17::run(Command::new("rustc").current_dir(&root).args([
        "--edition=2021",
        "-O",
        "-C",
        "panic=unwind",
        "--crate-type=cdylib",
        "model_ffi.rs",
        "-o",
        "libmodel.so",
    ]));
    c17::run(Command::new("python3").current_dir(&root).args(["-c", "import ctypes; m=ctypes.CDLL('./libmodel.so'); assert m.jope_model_abi_version()==1; assert m.jope_model_free(0)==2"]));
}

#[test]
fn title_line_breaks_remain_metadata_and_native_callers_execute() {
    let root = dir("title-framing");
    let genome = fixture();
    let options = EmitOptions {
        title: Some("C17\r\ncompile_error!(\"injected title\");\u{2028}\u{2029}tail".into()),
        ..EmitOptions::default()
    };
    let bundle = transpile::emit_c17(&genome, &options).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(&bundle.files["c17/manifest.json"]).unwrap();
    assert_eq!(
        manifest["options"]["title"],
        options.title.as_deref().unwrap()
    );
    let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let y = vec![1.0, 3.0, 5.0, 7.0];
    let test = vec![vec![0.5], vec![2.5]];
    let expected = feat_math::fit_predict(&genome, &x, &y, &test).unwrap();
    let got = c17::predictions(&root, &genome, &options, &x, &y, &test);
    assert_eq!(
        got,
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    let caller = fs::read_to_string(root.join("consumer.c")).unwrap();
    let clang: Vec<u64> = c17::consumer(&root, "clang", &caller)
        .lines()
        .map(|line| u64::from_str_radix(line, 16).unwrap())
        .collect();
    assert_eq!(clang, got);
}
