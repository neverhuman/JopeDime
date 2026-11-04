use super::*;

use serde_json::json;
use std::process::Command;

fn typescript_header() -> Header {
    serde_json::from_value(json!({
        "schema": "jope.appliance-fitted/v1",
        "minimum_reader_version": 1,
        "product": "jopedime",
        "product_version": "1.0.0",
        "runtime_abi": "jope.appliance-runtime/v1",
        "method": {"id":"ridge", "family":"linear", "configuration":"default", "recipe":"ridge", "implementation":"fixture"},
        "task":"regression", "metric":"rmse", "class_order":[], "target":"target",
        "input_schema":[
            {"name":"age", "kind":"numeric"},
            {"name":"region", "kind":"categorical"},
            {"name":"quote\"field", "kind":"numeric"}
        ],
        "preprocessing":{"raw_cell_encoding":"jope.raw-cell-encoding/v1", "input_policy":"named-columns"},
        "recipe_sha256":"00", "required_asset_hashes":[], "fitted_sections":[],
        "training_procedure":{"rows":1, "fitting_input_sha256":"00", "training_file_sha256":"00", "method_choice":"requested", "selection":null, "budget":{}, "final_refit":true},
        "retained_training_context":{"neural_support_rows":0, "note":"fixture"},
        "created_by":{"source_sha40":"0000000000", "package_manifest_sha256":null, "package_profile":null}
    }))
    .unwrap()
}

fn sources() -> BTreeMap<LanguageKey, LanguageFiles> {
    BTreeMap::from([
        (
            LanguageKey::Javascript,
            LanguageFiles {
                predictor: b"export function predict() {}".to_vec(),
                declaration: None,
            },
        ),
        (
            LanguageKey::Python,
            LanguageFiles {
                predictor: b"def predict(row): return 0".to_vec(),
                declaration: None,
            },
        ),
        (
            LanguageKey::Rust,
            LanguageFiles {
                predictor: b"pub fn predict() {}".to_vec(),
                declaration: None,
            },
        ),
        (
            LanguageKey::Typescript,
            LanguageFiles {
                predictor: b"export function predict() {}".to_vec(),
                declaration: Some(b"export declare function predict(): number;".to_vec()),
            },
        ),
    ])
}

#[test]
fn manifest_round_trips_and_hashes_every_file() {
    let requested = [Language::Javascript, Language::Typescript];
    let a = assemble(
        b"model object",
        &requested,
        &sources(),
        b"{\"checks\":[]}",
        false,
    )
    .unwrap();
    let b = assemble(
        b"model object",
        &requested,
        &sources(),
        b"{\"checks\":[]}",
        false,
    )
    .unwrap();
    assert_eq!(a.entries, b.entries);
    let archive = zip_bytes(&a).unwrap();
    assert_eq!(archive, zip_bytes(&b).unwrap());
    let mut zip = zip::ZipArchive::new(Cursor::new(archive)).unwrap();
    let names = (0..zip.len())
        .map(|index| zip.by_index(index).unwrap().name().unwrap().into_owned())
        .collect::<Vec<_>>();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted);
    for index in 0..zip.len() {
        let file = zip.by_index(index).unwrap();
        assert_eq!(
            file.last_modified(),
            Some(zip::DateTime::from_date_and_time(1980, 1, 1, 0, 0, 0).unwrap())
        );
        assert_eq!(file.compression(), zip::CompressionMethod::Deflated);
    }
    let parsed: BundleManifest = serde_json::from_slice(&a.entries["manifest.json"]).unwrap();
    parsed.validate().unwrap();
    assert_eq!(parsed, a.manifest);
    assert_eq!(parsed.entries["model.jope"], sha256_hex(b"model object"));
    assert!(a.entries.contains_key("src/typescript/predictor.d.ts"));
    assert!(a.entries.contains_key("src/typescript/README.md"));
}

#[test]
fn unsupported_required_language_fails_before_writer_publication() {
    struct Stub {
        calls: usize,
    }
    impl BundleWriter for Stub {
        type Published = ();
        fn publish(&mut self, _: &[u8]) -> Result<()> {
            self.calls += 1;
            Ok(())
        }
    }
    let absent = BTreeMap::new();
    let mut writer = Stub { calls: 0 };
    let result = assemble_and_publish(
        b"model",
        &[Language::Rust],
        &absent,
        b"{}",
        false,
        &mut writer,
    );
    assert!(result.is_err());
    assert_eq!(writer.calls, 0);
}

#[test]
fn store_publication_returns_content_identity_and_archive_bytes() {
    let root = tempfile::tempdir().unwrap();
    let store = crate::store::Store::open(root.path()).unwrap();
    let bundle = assemble(b"model object", &[Language::Rust], &sources(), b"{}", false).unwrap();
    let expected = zip_bytes(&bundle).unwrap();
    let published = publish(&bundle, &mut StoreBundleWriter { store: &store }).unwrap();
    assert_eq!(published.blob_id, format!("sha256:{}", published.sha256));
    assert_eq!(published.size, expected.len() as u64);
    let digest = (0..32)
        .map(|index| u8::from_str_radix(&published.sha256[index * 2..index * 2 + 2], 16).unwrap())
        .collect::<Vec<_>>();
    let reference = crate::store::BlobRef {
        sha256: digest.try_into().unwrap(),
        bytes: published.size,
    };
    assert_eq!(store.read_blob(&reference).unwrap(), expected);
}

#[test]
fn manifest_rejects_traversal_and_bad_hashes() {
    let mut manifest = BundleManifest {
        schema_version: 1,
        model_object_digest: "a".repeat(64),
        retains_training_data: false,
        entries: BTreeMap::from([("../evil".into(), "b".repeat(64))]),
    };
    assert!(manifest.validate().is_err());
    manifest.entries = BTreeMap::from([("model.jope".into(), "bad".into())]);
    assert!(manifest.validate().is_err());
}

#[test]
fn retention_flag_tracks_training_row_heads() {
    let model = |head: &str, value: serde_json::Value| -> Model {
        serde_json::from_value(serde_json::json!({
            "Pipeline": {"target": null, "ops": [], "head": {head: value}}
        }))
        .unwrap()
    };
    let krr = serde_json::json!({"means": [], "scales": [], "basis": [], "alpha": [], "gamma": 0.0, "y_mean": 0.0});
    let gpr = serde_json::json!({"means": [], "scales": [], "basis": [], "alpha": [], "ls": [], "sf2": 0.0, "y_mean": 0.0});
    let ksmooth = serde_json::json!({"means": [], "scales": [], "z": [], "ys": [], "k": 1, "gamma": 1.0, "y_mean": 0.0});
    let ridge = serde_json::json!({"means": [], "scales": [], "w": []});
    assert!(retains_training_data(&model("Krr", krr)));
    assert!(retains_training_data(&model("Gpr", gpr)));
    assert!(retains_training_data(&model("Ksmooth", ksmooth)));
    assert!(!retains_training_data(&model("Ridge", ridge)));
}

#[test]
fn retained_legacy_bound_model_reaches_all_fitted_emitters() {
    let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let bound = feat_math::advanced_v3::fit_legacy_numeric_bound(
        &genome,
        &[vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
        &[0.0, 1.0, 2.0, 3.0],
    )
    .unwrap();
    let model = legacy_model(&bound).unwrap();
    assert!(!retains_training_data(&model));
    assert!(
        !jope_runtime::fitted_js::emit_with_checks(&genome, &model, 1, &[])
            .unwrap()
            .is_empty()
    );
    assert!(
        !jope_runtime::fitted_py::emit_with_checks(&genome, &model, 1, &[])
            .unwrap()
            .is_empty()
    );
    assert!(
        !jope_runtime::fitted_rs::emit_with_checks(&genome, &model, 1, &[])
            .unwrap()
            .is_empty()
    );
}

#[test]
fn typescript_bundle_consumer_compiles_and_calls_named_predictors() {
    let Some(tsc) = std::env::var_os("JOPE_TSC") else {
        return;
    };
    let header = typescript_header();
    let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let bound = feat_math::advanced_v3::fit_legacy_numeric_bound(
        &genome,
        &[
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
        ],
        &[0.0, 1.0, 2.0],
    )
    .unwrap();
    let model = legacy_model(&bound).unwrap();
    let mut predictor = jope_runtime::fitted_js::emit_with_checks(&genome, &model, 3, &[]).unwrap();
    append_typescript_api(&mut predictor, &header).unwrap();
    let emitted = BTreeMap::from([(
        LanguageKey::Typescript,
        LanguageFiles {
            predictor: predictor.into_bytes(),
            declaration: Some(crate::export::dts::generate(&header).unwrap().into_bytes()),
        },
    )]);
    let bundle = assemble(
        b"fitted model fixture",
        &[Language::Typescript],
        &emitted,
        b"{}",
        false,
    )
    .unwrap();
    let archive_bytes = zip_bytes(&bundle).unwrap();
    let mut archive = zip::ZipArchive::new(Cursor::new(archive_bytes)).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let evidence_dir =
        std::env::var_os("JOPE_TS_BUNDLE_EVIDENCE_DIR").map(std::path::PathBuf::from);
    if let Some(path) = &evidence_dir {
        std::fs::create_dir_all(path).unwrap();
        std::fs::write(path.join("manifest.json"), &bundle.entries["manifest.json"]).unwrap();
    }
    for entry_name in [
        "src/typescript/predictor.mjs",
        "src/typescript/predictor.d.ts",
        "src/typescript/predictor.d.mts",
    ] {
        let mut entry = archive.by_name(entry_name).unwrap();
        let destination = dir
            .path()
            .join(std::path::Path::new(entry_name).file_name().unwrap());
        let mut output = std::fs::File::create(destination).unwrap();
        std::io::copy(&mut entry, &mut output).unwrap();
        if let Some(path) = &evidence_dir {
            std::fs::copy(
                dir.path()
                    .join(std::path::Path::new(entry_name).file_name().unwrap()),
                path.join(std::path::Path::new(entry_name).file_name().unwrap()),
            )
            .unwrap();
        }
    }
    let consumer = dir.path().join("fitted-consumer.mts");
    let consumer_source = include_str!("../../../../../deploy/ci/typescript/fitted-consumer.mts");
    std::fs::write(&consumer, consumer_source).unwrap();
    if let Some(path) = &evidence_dir {
        std::fs::write(path.join("fitted-consumer.mts"), consumer_source).unwrap();
    }
    let output = Command::new(tsc)
        .args([
            "--strict",
            "--noEmit",
            "--target",
            "ES2020",
            "--module",
            "NodeNext",
            "--moduleResolution",
            "NodeNext",
        ])
        .arg(&consumer)
        .output()
        .unwrap();
    if let Some(path) = &evidence_dir {
        let log = format!(
            "command: tsc --strict --noEmit --target ES2020 --module NodeNext --moduleResolution NodeNext fitted-consumer.mts\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::write(path.join("tsc.log"), log).unwrap();
    }
    assert!(
        output.status.success(),
        "tsc --strict failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let runtime = dir.path().join("exercise.mjs");
    std::fs::write(
        &runtime,
        "import {predict,predictBatch} from './predictor.mjs';\nconst row={age:1,region:2,'quote\\\"field':3};\nconst single=predict(row),batch=predictBatch([row]);\nif(!Number.isFinite(single)||batch.length!==1||batch[0]!==single)throw Error('predict API mismatch');\nfor(const bad of [{...row,region:'north'},{age:1,'quote\\\"field':3}]){let refused=false;try{predict(bad);}catch(error){refused=error instanceof TypeError;}if(!refused)throw Error('non-numeric or missing input was not refused: '+JSON.stringify(bad));}\n",
    )
    .unwrap();
    let output = Command::new("node").arg(runtime).output().unwrap();
    assert!(
        output.status.success(),
        "generated predictor runtime failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    if let Some(path) = evidence_dir {
        std::fs::write(
            path.join("node-runtime.log"),
            "node exercised exported predict and predictBatch with the same fitted row; result finite and equal\n",
        )
        .unwrap();
    }
}
