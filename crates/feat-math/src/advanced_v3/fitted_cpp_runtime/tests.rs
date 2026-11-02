use super::*;

use crate::advanced_v3::{fit_table, ColumnSpec, DataSchema, RowSemantics};
use crate::{GenomeIr, Objective, TaskSpec};

#[test]
fn embedded_runtime_loads_with_good_pin_and_rejects_mismatch() {
    let image_bytes = b"pinned fitted model image";
    let expected = sha256_hex(image_bytes);
    let image = sealed_image(image_bytes, &expected).unwrap();
    assert_eq!(
        fs::read(format!("/proc/self/fd/{}", image.as_raw_fd())).unwrap(),
        image_bytes
    );
    let error = sealed_image(b"different model image", &expected).unwrap_err();
    assert!(error.to_string().contains("SHA-256 mismatch"));
    assert_eq!(
        sha256_hex(RUNTIME_ORACLE_SOURCE.as_bytes()),
        RUNTIME_SOURCE_SHA256
    );
}

fn fixture() -> (BoundModel, RawTable) {
    let schema = DataSchema {
        columns: vec![ColumnSpec::numeric("x"), ColumnSpec::numeric("z")],
        row_semantics: RowSemantics::Iid,
    };
    let train = RawTable::from_csv("x,z\n0,1\n1,2\n2,3\n3,4\n4,5\n").unwrap();
    let test = RawTable::from_csv("x,z\n0.5,1.5\n,2.5\n").unwrap();
    let genome = GenomeIr::parse("(pipe X (input numeric=median) (ridge))").unwrap();
    let task = TaskSpec::regression(Objective::Rmse).unwrap();
    let model = fit_table(&genome, &task, &schema, &train, &[0.0, 0.2, 0.4, 0.6, 0.8]).unwrap();
    (model, test)
}

#[test]
fn codec_rejects_every_truncation_and_corruption() {
    let (model, _) = fixture();
    let codec = encode_fitted_cpp17(&model).unwrap();
    assert!(decode(&codec.bytes).is_ok());
    for length in 0..codec.bytes.len() {
        assert!(decode(&codec.bytes[..length]).is_err(), "length {length}");
    }
    for index in [0, 8, 16, HEADER_LEN, codec.bytes.len() - 1] {
        let mut corrupt = codec.bytes.clone();
        corrupt[index] ^= 0x80;
        assert!(decode(&corrupt).is_err(), "byte {index}");
    }
}

#[test]
fn historical_unsafe_expression_library_is_regenerated_before_loading() {
    let genome = GenomeIr::parse("(sym raw \"c9+1\")").unwrap();
    let model =
        crate::fit_legacy_numeric_bound(&genome, &[vec![1., 2.], vec![2., 3.]], &[0., 1.]).unwrap();
    let mut artifact = emit_fitted_cpp17(&model).unwrap();
    let GenomeIr::Symbolic { expr, .. } = genome else {
        unreachable!()
    };
    artifact.source = artifact
        .source
        .replace(&expr.to_cpp17(false).unwrap(), "(row[9]+1.0)");
    artifact.source_sha256 = text_hash(&artifact.source);
    let encoded = encode_artifact(&model, &artifact).unwrap();
    let old_metadata_len = read_u64(&encoded.bytes, 24).unwrap() as usize;
    let old_payload_at = HEADER_LEN + old_metadata_len;
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&encoded.bytes[HEADER_LEN..old_payload_at]).unwrap();
    let old_library_hash = metadata["model_library_sha256"]
        .as_str()
        .unwrap()
        .to_string();
    metadata.as_object_mut().unwrap().remove("emitter_revision");
    let metadata = serde_json::to_vec(&metadata).unwrap();
    let mut historical = encoded.bytes[..HEADER_LEN].to_vec();
    historical.extend(&metadata);
    historical.extend(&encoded.bytes[old_payload_at..encoded.bytes.len() - CHECKSUM_LEN]);
    let total = (historical.len() + CHECKSUM_LEN) as u64;
    historical[16..24].copy_from_slice(&total.to_le_bytes());
    historical[24..32].copy_from_slice(&(metadata.len() as u64).to_le_bytes());
    let checksum = Sha256::digest(&historical);
    historical.extend(checksum);
    assert_eq!(decode(&historical).unwrap().emitter_revision, 1);
    let original = historical.clone();
    let scratch = tempfile::Builder::new()
        .prefix("kernel-codec-upgrade-")
        .tempdir()
        .unwrap();
    let runtime = FittedCppRuntime::new(scratch.path()).unwrap();
    let loaded = runtime.load(&historical).unwrap();
    let upgraded = encode_fitted_cpp17(&decode(&historical).unwrap().model).unwrap();
    let upgraded_library = decode(&upgraded.bytes).unwrap().model_library;
    assert_ne!(sha256_hex(&upgraded_library), old_library_hash);
    assert_eq!(
        loaded
            .predict(&EncodedTable::from_numeric_rows(&[vec![1., 2.]]).unwrap())
            .unwrap(),
        vec![0.0]
    );
    assert_eq!(historical, original);
    drop(loaded);
    drop(runtime);
    scratch.close().unwrap();
}

#[test]
fn safe_runtime_matches_bound_predictions() {
    let (model, test) = fixture();
    let codec = encode_fitted_cpp17(&model).unwrap();
    let repeated = encode_fitted_cpp17(&model).unwrap();
    assert_eq!(codec.sha256, repeated.sha256);
    assert_eq!(codec.bytes, repeated.bytes);
    let decoded = decode(&codec.bytes).unwrap();
    assert_eq!(decoded.emitter_revision, 3);
    assert_eq!(decoded.model.genome_sha256, model.genome_sha256);
    let encoded = EncodedTable::from_bound_model(&model, &test).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("fitted-cpp-runtime-")
        .tempdir()
        .unwrap();
    let runtime = FittedCppRuntime::new(scratch.path()).unwrap();
    let loaded = runtime.load(&codec.bytes).unwrap();
    let actual = loaded.predict(&encoded).unwrap();
    let expected = model.predict_table(&test).unwrap();
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    drop(loaded);
}

#[test]
fn runtime_rejects_malformed_shapes_without_entering_model_code() {
    let (model, test) = fixture();
    let codec = encode_fitted_cpp17(&model).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("fitted-cpp-runtime-malformed-")
        .tempdir()
        .unwrap();
    let runtime = FittedCppRuntime::new(scratch.path()).unwrap();
    let loaded = runtime.load(&codec.bytes).unwrap();
    let mut wrong_columns = EncodedTable::from_bound_model(&model, &test).unwrap();
    wrong_columns.columns += 1;
    assert!(loaded.predict(&wrong_columns).is_err());
    let overflow = EncodedTable {
        values: Vec::new(),
        missing: Vec::new(),
        rows: usize::MAX,
        columns: 2,
    };
    assert!(loaded.predict(&overflow).is_err());
    let mut corrupt = codec.bytes.clone();
    corrupt[0] ^= 1;
    assert!(runtime.load(&corrupt).is_err());
    drop(loaded);
}

#[test]
fn runtime_c_abi_is_sanitizer_clean_and_fail_closed() {
    let scratch = tempfile::Builder::new()
        .prefix("fitted-cpp-runtime-sanitizer-")
        .tempdir()
        .unwrap();
    let root = scratch.path();
    let runtime_source = root.join("runtime.cpp");
    let driver_source = root.join("driver.cpp");
    let binary = root.join("runtime-sanitizer");
    fs::write(&runtime_source, RUNTIME_ORACLE_SOURCE).unwrap();
    fs::write(
            &driver_source,
            r#"
#include <cstddef>
#include <cstdint>
struct Error { std::int32_t code; char stage[32]; char message[256]; };
extern "C" int jope_runtime_model_load(const std::uint8_t*,std::size_t,const char*,void**,Error*) noexcept;
extern "C" int jope_runtime_model_predict(void*,const double*,std::size_t,const std::uint8_t*,std::size_t,std::size_t,std::size_t,double*,std::size_t,Error*) noexcept;
extern "C" void jope_runtime_model_free(void*) noexcept;
int main() {
  Error error{}; void* model=nullptr; std::uint8_t short_codec[64]{};
  if (jope_runtime_model_load(nullptr,0,"/tmp/unused",&model,&error)==0) return 1;
  if (jope_runtime_model_load(short_codec,sizeof(short_codec),"/tmp/unused",&model,&error)==0) return 2;
  short_codec[0]='J'; short_codec[1]='D'; short_codec[2]='F'; short_codec[3]='C';
  short_codec[4]='P'; short_codec[5]='P'; short_codec[6]='3';
  short_codec[8]=2; short_codec[10]=56; short_codec[16]=64;
  for (int i=24;i<56;i++) short_codec[i]=0xff;
  if (jope_runtime_model_load(short_codec,sizeof(short_codec),"/tmp/unused",&model,&error)==0) return 3;
  if (jope_runtime_model_predict(nullptr,nullptr,0,nullptr,0,0,0,nullptr,0,&error)==0) return 4;
  jope_runtime_model_free(nullptr);
  return 0;
}
"#,
        )
        .unwrap();
    let compiled = crate::cxx::command(None)
        .args([
            "-O1",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=undefined",
        ])
        .arg(&runtime_source)
        .arg(&driver_source)
        .args(["-ldl", "-o"])
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "sanitized runtime compilation failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let run = Command::new(&binary).output().unwrap();
    assert!(
        run.status.success(),
        "sanitized runtime rejected fail-closed probe: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    scratch.close().unwrap();
}

/// The source that emitter revision 2 wrote for `fixture()`'s model, written by
/// `emit_fitted_cpp17` at `8676953c1`, the base of kernel2 PR 07. It is kept so that a stored
/// revision-2 container can still be built, and loaded, once the emitter moves on.
const REVISION_TWO_SOURCE: &str =
    include_str!("../../tests/fixtures/fitted_cpp/revision2-ridge.cpp");

/// `REVISION_TWO_SOURCE` and the current source of the same model differ only in the lines
/// that revision 3 changed: the marker, the width error and its index check, `piece`,
/// `row_mean`, `row_std` and the catch of status 8. Every other line, the model's own code
/// included, is the same, so the fixture stays revision 2's source for this very model.
#[test]
fn revision_two_source_differs_from_the_current_one_only_in_revision_three_lines() {
    /// The lines of `left` that `right` lacks, counting repeats, in `left`'s order.
    fn only_in<'a>(left: &'a str, right: &str) -> Vec<&'a str> {
        let mut counts = std::collections::BTreeMap::new();
        for line in right.lines() {
            *counts.entry(line).or_insert(0usize) += 1;
        }
        left.lines()
            .filter(|line| match counts.get_mut(line) {
                Some(count) if *count > 0 => {
                    *count -= 1;
                    false
                }
                _ => true,
            })
            .collect()
    }
    fn starts_with_each(lines: &[&str], prefixes: &[&str]) -> bool {
        lines.len() == prefixes.len()
            && lines
                .iter()
                .zip(prefixes)
                .all(|(line, prefix)| line.starts_with(prefix))
    }
    let (model, _) = fixture();
    let current = emit_fitted_cpp17(&model).unwrap().source;
    let removed = only_in(REVISION_TWO_SOURCE, &current);
    assert!(
        starts_with_each(
            &removed,
            &[
                "// jope-dime fitted-cpp-typed-table-v1 emitter-revision=2",
                "[[maybe_unused]] static double piece(",
                "[[maybe_unused]] static double row_mean(",
                "[[maybe_unused]] static double row_std(",
                "  } catch(...) { return 6; }",
            ]
        ),
        "{removed:#?}"
    );
    let added = only_in(&current, REVISION_TWO_SOURCE);
    assert!(
        starts_with_each(
            &added,
            &[
                "// jope-dime fitted-cpp-typed-table-v1 emitter-revision=3",
                "struct FittedWidthError {};",
                "[[maybe_unused]] static std::size_t fitted_index(",
                "[[maybe_unused]] static double piece(",
                "[[maybe_unused]] static double row_mean(",
                "[[maybe_unused]] static double row_std(",
                "  } catch(const FittedWidthError&) { return 8; } catch(...) { return 6; }",
            ]
        ),
        "{added:#?}"
    );
}

/// `bytes` with its metadata section replaced by `edit` of it, and the header's lengths and
/// the checksum recomputed.
fn with_metadata(bytes: &[u8], edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    let payload_at = HEADER_LEN + read_u64(bytes, 24).unwrap() as usize;
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&bytes[HEADER_LEN..payload_at]).unwrap();
    edit(&mut metadata);
    let metadata = serde_json::to_vec(&metadata).unwrap();
    let mut rebuilt = bytes[..HEADER_LEN].to_vec();
    rebuilt.extend(&metadata);
    rebuilt.extend(&bytes[payload_at..bytes.len() - CHECKSUM_LEN]);
    let total = (rebuilt.len() + CHECKSUM_LEN) as u64;
    rebuilt[16..24].copy_from_slice(&total.to_le_bytes());
    rebuilt[24..32].copy_from_slice(&(metadata.len() as u64).to_le_bytes());
    let checksum = Sha256::digest(&rebuilt);
    rebuilt.extend(checksum);
    rebuilt
}

/// The metadata section of a container.
fn metadata_of(bytes: &[u8]) -> serde_json::Value {
    let payload_at = HEADER_LEN + read_u64(bytes, 24).unwrap() as usize;
    serde_json::from_slice(&bytes[HEADER_LEN..payload_at]).unwrap()
}

#[test]
fn stored_revision_two_container_validates_and_load_regenerates_it() {
    let (model, test) = fixture();
    let encoded = EncodedTable::from_bound_model(&model, &test).unwrap();
    let bits = |values: &[f64]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };

    // A container as revision 2 stored it: that revision's source, the library compiled from
    // it, and metadata that records revision 2.
    let revision_two = FittedCppArtifact {
        source: REVISION_TWO_SOURCE.to_string(),
        source_sha256: text_hash(REVISION_TWO_SOURCE),
        codec: String::new(),
        codec_sha256: String::new(),
    };
    let stored = with_metadata(
        &encode_artifact(&model, &revision_two).unwrap().bytes,
        |metadata| metadata["emitter_revision"] = serde_json::json!(2),
    );
    let stored_library = metadata_of(&stored)["model_library_sha256"]
        .as_str()
        .unwrap()
        .to_string();

    // It still decodes and validates: revision 2's source is not compared with the emitter's.
    validate_fitted_cpp_codec(&stored).unwrap();
    let decoded = decode(&stored).unwrap();
    assert_eq!(decoded.emitter_revision, 2);
    assert_eq!(
        serde_json::to_vec(&decoded.model).unwrap(),
        serde_json::to_vec(&model).unwrap()
    );

    // Its own library, loaded as stored (without `load`'s regeneration), gives revision 2's
    // predictions.
    let scratch = tempfile::Builder::new()
        .prefix("fitted-cpp-revision-two-")
        .tempdir()
        .unwrap();
    let runtime = FittedCppRuntime::new(scratch.path()).unwrap();
    let as_stored = load_model_library(&decoded.model_library).unwrap();
    assert_eq!(sha256_hex(&decoded.model_library), stored_library);
    let revision_two_predictions = as_stored.predict(&encoded).unwrap();
    drop(as_stored);

    // `load` regenerates it from the embedded model, without refitting: the library it loads
    // is the one revision 3 compiles for the same model, not the stored one, and the
    // predictions are revision 2's, bit for bit, and the runtime's.
    let loaded = runtime.load(&stored).unwrap();
    let library = sha256_hex(
        &decode(&encode_fitted_cpp17(&model).unwrap().bytes)
            .unwrap()
            .model_library,
    );
    assert_ne!(library, stored_library);
    let current = encode_fitted_cpp17(&model).unwrap();
    assert_eq!(
        library,
        metadata_of(&current.bytes)["model_library_sha256"]
            .as_str()
            .unwrap()
    );
    assert_eq!(
        metadata_of(&current.bytes)["emitter_revision"],
        serde_json::json!(3)
    );
    let predictions = loaded.predict(&encoded).unwrap();
    assert_eq!(bits(&predictions), bits(&revision_two_predictions));
    assert_eq!(
        bits(&predictions),
        bits(&model.predict_table(&test).unwrap())
    );
    drop(loaded);

    // A revision-3 container is held to the current emitter's source, and a later revision
    // is refused.
    let relabelled = with_metadata(&stored, |metadata| {
        metadata["emitter_revision"] = serde_json::json!(3)
    });
    let error = decode(&relabelled).unwrap_err().to_string();
    assert!(
        error.contains("does not match the current emitter"),
        "{error}"
    );
    let later = with_metadata(&stored, |metadata| {
        metadata["emitter_revision"] = serde_json::json!(4)
    });
    let error = decode(&later).unwrap_err().to_string();
    assert!(
        error.contains("metadata or runtime identity mismatch"),
        "{error}"
    );
    drop(runtime);
    scratch.close().unwrap();
}
