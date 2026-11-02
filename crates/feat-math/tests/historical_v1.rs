//! B00 historical v1 corpus. The fixture is read only: there is no regeneration mode.
//! Run this test together with `compat_freeze`, which executes the pre-existing fitted
//! witnesses whose names and exact prediction bits are indexed here.

use feat_math::dsl::{DslExpr, Stage};
use feat_math::kernel::{pipeline::PipelineSlots, Program};
use feat_math::math_expr::{Expr, Func, MathExpr, Var};
use feat_math::ops::{apply_op, fit_op};
use feat_math::registry::operator_registry;
use feat_math::transpile::{emit, emit_cpp17, emit_javascript, EmitOptions};
use feat_math::{emit_fitted_cpp17, fit, fit_legacy_numeric_bound, GenomeIr, Model};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const BASE: &str = "e709ed90e0890a619b91fdc2fe395439b20ffb65";

#[path = "support/identity_transitions.rs"]
mod identity_transitions;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/compat_freeze/historical_v1")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn outcome(result: anyhow::Result<f64>) -> String {
    match result {
        Ok(value) => bits(value),
        Err(error) => format!("ERR {error}"),
    }
}

fn old_fixture(name: &str) -> Value {
    let path = root().parent().unwrap().join(name);
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn program_bytes(genome: &GenomeIr) -> String {
    let bytes = Program::import_legacy(genome).to_bytes().unwrap();
    assert_eq!(
        Program::from_bytes(&bytes).unwrap().to_bytes().unwrap(),
        bytes
    );
    let text = Program::from_bytes(&bytes)
        .unwrap()
        .to_lossless_text()
        .unwrap();
    assert_eq!(
        Program::parse_lossless_text(&text)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    hex(&bytes)
}

fn data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let rows: Vec<Vec<f64>> = (0..24)
        .map(|i| {
            let x = (i as f64 - 11.0) / 7.0;
            vec![x, x * x + 0.25, (i % 5) as f64 - 2.0, x * 0.5 + 1.0]
        })
        .collect();
    let target = rows
        .iter()
        .map(|row| 1.5 * row[0] - 0.3 * row[1] + row[2] * 0.1)
        .collect();
    (rows, target)
}

fn alias_records() -> Vec<Value> {
    let (rows, target) = data();
    let pairs = [
        (
            3,
            "norm_features",
            "standardize",
            "(pipe X (norm_features) (ridge))",
            "(pipe X (standardize) (ridge))",
        ),
        (
            11,
            "winsor_moments",
            "winsor_clip",
            "(pipe X (winsor_moments) (ridge))",
            "(pipe X (winsor_clip) (ridge))",
        ),
        (
            39,
            "gbm",
            "histgbm",
            "(pipe X (gbm rounds=5) )",
            "(pipe X (histgbm rounds=5) )",
        ),
        (
            41,
            "ensemble_ridge_gbm",
            "ensemble",
            "(pipe X (ensemble_ridge_gbm rounds=5) )",
            "(pipe X (ensemble rounds=5) )",
        ),
    ];
    pairs
        .into_iter()
        .map(|(id, name, canonical_name, source, canonical_source)| {
            let alias = GenomeIr::parse(source).unwrap_or_else(|error| panic!("{name}: {error}"));
            let canonical = GenomeIr::parse(canonical_source).unwrap();
            let left = fit(&alias, &rows, &target).unwrap_or_else(|error| panic!("{name}: {error}"));
            let right = fit(&canonical, &rows, &target).unwrap();
            let alias_bits: Vec<_> = left.predict(&rows).unwrap().into_iter().map(bits).collect();
            let canonical_bits: Vec<_> = right.predict(&rows).unwrap().into_iter().map(bits).collect();
            assert_eq!(alias_bits, canonical_bits, "wire alias {id} must preserve its semantic key");
            json!({"id": id, "name": name, "semantic_key": canonical_name,
                "source": source, "program_hex": program_bytes(&alias), "prediction_bits": alias_bits})
        })
        .collect()
}

fn registry_records() -> Vec<Value> {
    let emitted = old_fixture("emitted_bytes.v1.json");
    let models = old_fixture("model_serde.v1.json");
    let programs = emitted["programs"].as_array().unwrap();
    let model_rows = models["models"].as_array().unwrap();
    assert_eq!(operator_registry().len(), 53);
    operator_registry()
        .iter()
        .enumerate()
        .map(|(id, op)| {
            assert_eq!(usize::from(op.id.0), id);
            let witness = programs.iter().find(|program| {
                let canonical = program["canonical"].as_str().unwrap();
                canonical.contains(&format!("({} ", op.name))
                    || canonical.contains(&format!("({})", op.name))
            });
            let evidence =
                if let Some(witness) = witness {
                    let name = witness["name"].as_str().unwrap();
                    let model = model_rows.iter().find(|row| row["name"] == name).unwrap();
                    let predicted = model["predictions"].as_str().unwrap();
                    assert!(
                        predicted.split_whitespace().all(|token| token.len() == 16
                            && token.bytes().all(|b| b.is_ascii_hexdigit())),
                        "wire {id} witness {name} must be a bit-frozen executable model"
                    );
                    let genome = GenomeIr::parse(witness["canonical"].as_str().unwrap()).unwrap();
                    json!({"kind": "fitted_existing", "name": name,
                    "program_hex": program_bytes(&genome), "prediction_bits": predicted})
                } else if [3, 11, 39, 41].contains(&id) {
                    json!({"kind": "alias_case"})
                } else {
                    assert!(
                        op.lifecycle.quarantined,
                        "executable wire {id} lacks a witness"
                    );
                    let pipe = DslExpr {
                        stages: vec![Stage::new(op.name), Stage::new("ridge")],
                    };
                    let refusal = PipelineSlots::from_pipeline(&pipe).unwrap_err().to_string();
                    let source = format!("(pipe X ({}) (ridge))", op.name);
                    let parsed = Program::parse_legacy(&source)
                        .map(|program| program.to_bytes().unwrap().len().to_string())
                        .unwrap_or_else(|error| format!("ERR {error}"));
                    json!({"kind": "quarantined_refusal", "parse": parsed, "error": refusal})
                };
            json!({"id": id, "name": op.name, "semantic_key": op.semantic_key,
                "parseable": op.lifecycle.parseable, "executable": op.lifecycle.executable,
                "generatable": op.lifecycle.generatable, "quarantined": op.lifecycle.quarantined,
                "evidence": evidence})
        })
        .collect()
}

fn func_args(func: Func) -> Vec<Expr> {
    let x = Expr::Var(Var::X);
    match func {
        Func::Min | Func::Max | Func::Step => vec![x, Expr::Const(0.5)],
        Func::Clip => vec![x, Expr::Const(-1.0), Expr::Const(1.0)],
        _ => vec![x],
    }
}

fn exact_records() -> Value {
    let edges: [(&str, u64); 12] = [
        ("positive_zero", 0),
        ("negative_zero", 0x8000_0000_0000_0000),
        ("nan_payload", 0x7ff8_0000_0000_0001),
        ("positive_infinity", 0x7ff0_0000_0000_0000),
        ("negative_infinity", 0xfff0_0000_0000_0000),
        ("positive_subnormal", 1),
        ("negative_subnormal", 0x8000_0000_0000_0001),
        ("small", 1e-300f64.to_bits()),
        ("large", 1e200f64.to_bits()),
        ("near_max", 1e308f64.to_bits()),
        ("negative_one", (-1.0f64).to_bits()),
        ("half", 0.5f64.to_bits()),
    ];
    let funcs = [
        Func::Abs,
        Func::Sqrt,
        Func::Log1p,
        Func::Sin,
        Func::Cos,
        Func::Tanh,
        Func::Sigmoid,
        Func::Min,
        Func::Max,
        Func::Clip,
        Func::Step,
        Func::Exp,
        Func::Log,
        Func::Asinh,
        Func::Atan,
        Func::Erf,
        Func::Sign,
        Func::Floor,
        Func::Round,
    ];
    let vars = [
        Var::X,
        Var::RowMean,
        Var::RowSum,
        Var::RowL1,
        Var::RowL2,
        Var::RowMin,
        Var::RowMax,
        Var::RowStd,
    ];
    let func_rows: Vec<_> = funcs.into_iter().enumerate().map(|(tag, func)| {
        let expr = MathExpr::from_ast(Expr::Call(func, func_args(func))).unwrap();
        let genome = GenomeIr::Symbolic { source: format!("historical_func_{tag}"), expr: expr.clone(), affine: false };
        let outcomes: Vec<_> = edges.iter().map(|(name, raw)| {
            let x = f64::from_bits(*raw);
            json!({"edge": name, "input_bits": format!("{raw:016x}"),
                "outcome": outcome(expr.eval_element(x, &[x, -0.0, 1.0]))})
        }).collect();
        json!({"tag": tag, "name": format!("{func:?}"), "program_hex": program_bytes(&genome), "outcomes": outcomes})
    }).collect();
    let var_rows: Vec<_> = vars.into_iter().enumerate().map(|(tag, var)| {
        let expr = MathExpr::from_ast(Expr::Var(var)).unwrap();
        let genome = GenomeIr::Symbolic { source: format!("historical_var_{tag}"), expr: expr.clone(), affine: false };
        let outcomes: Vec<_> = edges.iter().map(|(name, raw)| {
            let x = f64::from_bits(*raw);
            json!({"edge": name, "input_bits": format!("{raw:016x}"),
                "outcome": outcome(expr.eval_element(x, &[x, -0.0, 1.0]))})
        }).collect();
        json!({"tag": tag, "name": format!("{var:?}"), "program_hex": program_bytes(&genome), "outcomes": outcomes,
            "empty_row": outcome(expr.eval_element(0.5, &[]))})
    }).collect();
    json!({"functions": func_rows, "variables": var_rows})
}

fn emitted_bytes() -> Value {
    let genome = GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap();
    let options = EmitOptions::default();
    let artifact = emit(&genome, &options).unwrap();
    let cpp = emit_cpp17(&genome, &options).unwrap();
    let js = emit_javascript(&genome, &options).unwrap();
    let (rows, target) = data();
    let bound = fit_legacy_numeric_bound(&genome, &rows, &target).unwrap();
    let fitted = emit_fitted_cpp17(&bound).unwrap();
    let sources = [
        ("rust", artifact.rust),
        ("python", artifact.python),
        ("manifest", artifact.manifest),
        ("cpp17", cpp),
        ("javascript", js),
        ("fitted_cpp17_source", fitted.source),
        ("fitted_cpp17_codec", fitted.codec),
    ];
    Value::Array(sources.into_iter().map(|(language, text)| {
        json!({"language": language, "bytes": text.len(), "sha256": sha256(text.as_bytes()), "source": text})
    }).collect())
}

fn stored_models() -> Vec<Value> {
    let models = old_fixture("model_serde.v1.json");
    models["stored"].as_array().unwrap().iter().map(|row| {
        let name = row["name"].as_str().unwrap();
        let text = row["json"].as_str().unwrap();
        let _: Model = serde_json::from_str(text).unwrap();
        json!({"name": name, "bytes": text.len(), "sha256": sha256(text.as_bytes()), "json": text})
    }).collect()
}

fn negative_cases() -> Value {
    let genome = GenomeIr::parse("(sym raw \"x\")").unwrap();
    let mut bytes = Program::import_legacy(&genome).to_bytes().unwrap();
    bytes[7] = 2;
    let future_envelope = Program::from_bytes(&bytes).unwrap_err().to_string();
    bytes[7] = 1;
    bytes[8] = 255;
    let future_variant = Program::from_bytes(&bytes).unwrap_err().to_string();
    let roll = Stage::new("roll");
    let fitted = fit_op(&roll, &vec![vec![1.0, 2.0]], &[1.0]).unwrap();
    let empty = apply_op(&fitted, &vec![vec![]]).unwrap_err();
    json!({"future_envelope": future_envelope, "future_variant": future_variant,
        "roll_zero_width": format!("refused: {empty}")})
}

fn snapshot() -> Value {
    json!({"schema": "feat-math.historical-v1-golden.v1", "base_sha": BASE,
        "registry": registry_records(), "aliases": alias_records(), "exact_tags": exact_records(),
        "emitted": emitted_bytes(), "stored_models": stored_models(),
        "negative_cases": negative_cases()})
}

#[test]
fn historical_v1_golden_matches() {
    let actual = snapshot();
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(root().join("manifest.json")).unwrap()).unwrap();
    let golden_sha = manifest["golden_sha256"].as_str().unwrap();
    let path = root()
        .join("objects/sha256")
        .join(format!("{golden_sha}.json"));
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        sha256(&bytes),
        golden_sha,
        "historical_v1 golden content address"
    );
    let mut expected: Value = serde_json::from_slice(&bytes).unwrap();
    let ledger: Value = serde_json::from_slice(
        &std::fs::read(
            root()
                .parent()
                .unwrap()
                .join("transitions/b02_roll_zero_width.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(ledger["schema"], "feat-math.historical-v1-transition.v1");
    assert_eq!(ledger["path"], "negative_cases.roll_zero_width");
    assert_eq!(ledger["before"], "panic");
    assert_eq!(
        expected["negative_cases"]["roll_zero_width"],
        ledger["before"]
    );
    assert_eq!(actual["negative_cases"]["roll_zero_width"], ledger["after"]);
    expected["negative_cases"]["roll_zero_width"] = ledger["after"].clone();
    assert_eq!(
        actual, expected,
        "historical_v1 byte and bit drift outside B02's one-case ledger"
    );
}

#[test]
fn historical_source_manifest_and_old_fixtures_are_pinned() {
    let index: Value =
        serde_json::from_slice(&std::fs::read(root().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(index["schema"], "feat-math.historical-v1-index.v1");
    assert_eq!(index["base_sha"], BASE);
    let bytes = std::fs::read(root().join("source-manifest.json")).unwrap();
    assert_eq!(sha256(&bytes), index["source_manifest_sha256"]);
    let source: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(source["schema"], "feat-math.historical-source-inventory.v1");
    assert_eq!(source["base_sha"], BASE);
    let rows = source["sources"].as_array().unwrap();
    let transition = old_fixture("transitions/b02_identity_rebaseline.v1.json");
    assert_eq!(
        transition["schema"],
        "feat-math.compat-freeze.identity-rebaseline.v1"
    );
    let replacements = transition["fixtures"].as_array().unwrap();
    assert_eq!(replacements.len(), 2);
    let changed_fixtures = [
        "crates/feat-math/tests/fixtures/compat_freeze/evaluator_identity.v1.json",
        "crates/feat-math/tests/fixtures/compat_freeze/lineage_sources.v1.json",
    ];
    for (entry, path) in replacements.iter().zip(changed_fixtures) {
        assert_eq!(entry["path"], path);
    }
    let successor = old_fixture("transitions/e3_identity_rebaseline.v1.json");
    assert_eq!(successor["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(successor["fixtures"][0]["path"], changed_fixtures[0]);
    // B12: hash-bound source identity successor; behavioral pins stay strict.
    let b12 = old_fixture("transitions/b12_identity_rebaseline.v1.json");
    assert_eq!(b12["fixtures"].as_array().unwrap().len(), 2);
    // B13: the same for the wire-61 bank's planner and emitter arms.
    let b13 = old_fixture("transitions/b13_identity_rebaseline.v1.json");
    assert_eq!(b13["fixtures"].as_array().unwrap().len(), 2);
    // B10: the same for the opt-in parent-selection seam in the search.
    let b10 = old_fixture("transitions/b10_identity_rebaseline.v1.json");
    assert_eq!(b10["fixtures"].as_array().unwrap().len(), 2);
    // B11: the same for the bit-identical ridge fit and the certified scoring seam.
    let b11 = old_fixture("transitions/b11_identity_rebaseline.v1.json");
    assert_eq!(b11["fixtures"].as_array().unwrap().len(), 2);
    // B05–B07: typed scalar planner and three-language recipe/fitted adapters.
    let scalar_recipes = old_fixture("transitions/b05_b07_recipes_identity_rebaseline.v1.json");
    assert_eq!(scalar_recipes["fixtures"].as_array().unwrap().len(), 2);
    // Owner-approved manifest cmp_owned change: identity-only, no behavior drift.
    let manifest_cmp = old_fixture("transitions/manifest_cmp_owned_identity_rebaseline.v1.json");
    assert_eq!(manifest_cmp["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(manifest_cmp["fixtures"][0]["path"], changed_fixtures[1]);
    // Synthetic arbitration clock: source identity moves; runtime behavior stays pinned.
    let authority_clock =
        old_fixture("transitions/authority_synthetic_clock_identity_rebaseline.v1.json");
    assert_eq!(authority_clock["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(authority_clock["fixtures"][0]["path"], changed_fixtures[1]);
    // Clippy remediation changes source identity; frozen and behavioral pins remain strict.
    let clippy_floor = old_fixture("transitions/required_clippy_floor_identity_rebaseline.v1.json");
    assert_eq!(clippy_floor["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(clippy_floor["fixtures"][0]["path"], changed_fixtures[1]);
    // Checked corpus mapping bounds mint new teaching and scan-policy source identities.
    let corpus_bounds = old_fixture("transitions/corpus_bounds_identity_rebaseline.v1.json");
    assert_eq!(corpus_bounds["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(corpus_bounds["fixtures"][0]["path"], changed_fixtures[1]);
    // Continuously held executable digest reuse: source identity only.
    let executable_cache =
        old_fixture("transitions/executable_verification_identity_rebaseline.v1.json");
    assert_eq!(executable_cache["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(executable_cache["fixtures"][0]["path"], changed_fixtures[1]);
    // Sealed embedded runtime and libc dependency: evaluator identity only.
    let sealed_runtime =
        old_fixture("transitions/fitted_runtime_sealed_identity_rebaseline.v1.json");
    assert_eq!(sealed_runtime["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(sealed_runtime["fixtures"][0]["path"], changed_fixtures[0]);
    // Evaluator child isolation (J24): teaching-store lineage source identity only.
    let eval_isolation =
        old_fixture("transitions/eval_child_isolation_identity_rebaseline.v1.json");
    assert_eq!(eval_isolation["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(eval_isolation["fixtures"][0]["path"], changed_fixtures[1]);
    // Workspace version bump to 0.3.0 (just bump 0.3.0): evaluator identity only.
    let release_0_3_0 = old_fixture("transitions/release_0_3_0_identity_rebaseline.v1.json");
    assert_eq!(release_0_3_0["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(release_0_3_0["fixtures"][0]["path"], changed_fixtures[0]);
    // EXP-NEXT-04 opt-in partition_point quantile emitters: evaluator identity only.
    let exp_next_04 = old_fixture("transitions/exp_next_04_identity_rebaseline.v1.json");
    assert_eq!(exp_next_04["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(exp_next_04["fixtures"][0]["path"], changed_fixtures[0]);
    // I09 opt-in v3 emitted artifact target binding: evaluator identity only.
    let i09_artifact_target_binding =
        old_fixture("transitions/i09_artifact_target_binding_identity_rebaseline.v1.json");
    assert_eq!(
        i09_artifact_target_binding["fixtures"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        i09_artifact_target_binding["fixtures"][0]["path"],
        changed_fixtures[0]
    );
    // Batch 4 algorithmic self-play additive feat-math modules: evaluator identity only.
    let batch4_selfplay = old_fixture("transitions/batch4_selfplay_identity_rebaseline.v1.json");
    assert_eq!(batch4_selfplay["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(batch4_selfplay["fixtures"][0]["path"], changed_fixtures[0]);
    // MR06 shard directory payload closure verifiers in label_admission.rs: teaching-store lineage source identity only.
    let mr06_label_admission =
        old_fixture("transitions/mr06_label_admission_identity_rebaseline.v1.json");
    assert_eq!(
        mr06_label_admission["fixtures"].as_array().unwrap().len(),
        1
    );
    assert_eq!(
        mr06_label_admission["fixtures"][0]["path"],
        changed_fixtures[1]
    );
    // MR19 shared-node feature DAG: exporting `combinators` moves the evaluator identity only.
    let mr19_shared_dag = old_fixture("transitions/mr19_shared_dag_identity_rebaseline.v1.json");
    assert_eq!(mr19_shared_dag["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(mr19_shared_dag["fixtures"][0]["path"], changed_fixtures[0]);
    // PR384 observatory workspace lock change: announced identity only.
    let observatory_workspace_lock =
        old_fixture("transitions/observatory_workspace_lock_identity_rebaseline.v1.json");
    let observatory_fixtures = observatory_workspace_lock["fixtures"].as_array().unwrap();
    assert_eq!(observatory_fixtures.len(), 2);
    for (entry, path) in observatory_fixtures.iter().zip(changed_fixtures) {
        assert_eq!(entry["path"], path);
    }
    // GP complete-source capture and lock successor: announced identity only.
    let gp_source_stream = old_fixture("transitions/gp_source_stream_identity_rebaseline.v1.json");
    let gp_fixtures = gp_source_stream["fixtures"].as_array().unwrap();
    assert_eq!(gp_fixtures.len(), 2);
    for (entry, path) in gp_fixtures.iter().zip(changed_fixtures) {
        assert_eq!(entry["path"], path);
    }
    // PR392 jope-automl workspace lock addition: announced identity only.
    let gpt6p1_sol_max = old_fixture("transitions/gpt6p1_sol_max_lock_identity_rebaseline.v1.json");
    let automl_fixtures = gpt6p1_sol_max["fixtures"].as_array().unwrap();
    assert_eq!(automl_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(
            automl_fixtures.iter().any(|entry| entry["path"] == path),
            "PR392 transition must name {path}"
        );
    }
    // PR394 follows the merged PR392 identity, rather than its original parallel baseline.
    let appliance_after_automl =
        old_fixture("transitions/appliance_after_automl_identity_rebaseline.v1.json");
    let appliance_fixtures = appliance_after_automl["fixtures"].as_array().unwrap();
    assert_eq!(appliance_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(appliance_fixtures.iter().any(|entry| entry["path"] == path));
    }
    // PR395 runtime audit repairs mint source-bound engine lineage only.
    let jankurai_runtime_safety =
        old_fixture("transitions/jankurai_runtime_safety_lineage_rebaseline.v1.json");
    let audit_fixtures = jankurai_runtime_safety["fixtures"].as_array().unwrap();
    assert_eq!(audit_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(audit_fixtures.iter().any(|entry| entry["path"] == path));
    }
    // Dependency/security successor preserves prior runtime and behavioral floors.
    let jankurai_dependency_security =
        old_fixture("transitions/jankurai_dependency_security_rebaseline.v1.json");
    let dependency_fixtures = jankurai_dependency_security["fixtures"].as_array().unwrap();
    assert_eq!(dependency_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(dependency_fixtures
            .iter()
            .any(|entry| entry["path"] == path));
    }
    let hpc_cache = old_fixture("transitions/hpc_cache_identity_rebaseline.v1.json");
    let hpc_fixtures = hpc_cache["fixtures"].as_array().unwrap();
    assert_eq!(hpc_fixtures.len(), 1);
    assert_eq!(hpc_fixtures[0]["path"], changed_fixtures[0]);
    // Preserve the original PR396 parallel edge without applying it after later identities.
    identity_transitions::verify_fixture(
        changed_fixtures[0],
        hpc_fixtures[0]["before"].as_str().unwrap(),
        hpc_fixtures[0]["after"].as_str().unwrap(),
        std::slice::from_ref(&hpc_cache),
    )
    .expect("preserved PR396 parallel identity announcement");
    let pr396_combined =
        old_fixture("transitions/pr396_after_jankurai_dependency_security_rebaseline.v1.json");
    let combined_fixtures = pr396_combined["fixtures"].as_array().unwrap();
    assert_eq!(combined_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(combined_fixtures.iter().any(|edge| edge["path"] == path));
    }
    // Preserve original PR397 parallel edge separately from the combined linear successor.
    let pr397_parallel = old_fixture("transitions/hpc_memory_threads_identity_rebaseline.v1.json");
    for edge in pr397_parallel["fixtures"].as_array().unwrap() {
        identity_transitions::verify_fixture(
            edge["path"].as_str().unwrap(),
            edge["before"].as_str().unwrap(),
            edge["after"].as_str().unwrap(),
            std::slice::from_ref(&pr397_parallel),
        )
        .expect("preserved PR397 parallel identity announcement");
    }
    // Preserve original PR398 parallel announcement separately from the combined successor.
    let pr398_parallel = old_fixture("transitions/hpc_safety_cleanup_identity_rebaseline.v1.json");
    assert_eq!(pr398_parallel["fixtures"].as_array().unwrap().len(), 2);
    for edge in pr398_parallel["fixtures"].as_array().unwrap() {
        identity_transitions::verify_fixture(
            edge["path"].as_str().unwrap(),
            edge["before"].as_str().unwrap(),
            edge["after"].as_str().unwrap(),
            std::slice::from_ref(&pr398_parallel),
        )
        .expect("preserved PR398 parallel identity announcement");
    }
    let pr397_combined =
        old_fixture("transitions/pr397_after_pr396_jankurai_security_rebaseline.v1.json");
    let combined_fixtures = pr397_combined["fixtures"].as_array().unwrap();
    assert_eq!(combined_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(combined_fixtures.iter().any(|edge| edge["path"] == path));
    }
    // PR398 follows the merged PR397 identity, rather than its original parallel baseline.
    let pr398_combined =
        old_fixture("transitions/pr398_after_pr397_hpc_safety_cleanup_rebaseline.v1.json");
    let combined_fixtures = pr398_combined["fixtures"].as_array().unwrap();
    assert_eq!(combined_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(combined_fixtures.iter().any(|edge| edge["path"] == path));
    }
    // Workspace version bump to 0.4.0 (just bump 0.4.0): announced identity only.
    let release_0_4_0 = old_fixture("transitions/release_0_4_0_identity_rebaseline.v1.json");
    let release_fixtures = release_0_4_0["fixtures"].as_array().unwrap();
    assert_eq!(release_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(release_fixtures.iter().any(|edge| edge["path"] == path));
    }
    // Model-object decoder dependency: announced identity-only Cargo.lock transition.
    let model_objects =
        old_fixture("transitions/model_object_contract_identity_rebaseline.v1.json");
    let model_object_fixtures = model_objects["fixtures"].as_array().unwrap();
    assert_eq!(model_object_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(model_object_fixtures
            .iter()
            .any(|edge| edge["path"] == path));
    }
    // TypeScript/Wasm target: identity-only successor after model-object integration.
    let typescript = old_fixture("transitions/typescript_wasm_identity_rebaseline.v1.json");
    let typescript_fixtures = typescript["fixtures"].as_array().unwrap();
    assert_eq!(typescript_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(typescript_fixtures.iter().any(|edge| edge["path"] == path));
    }
    // Preserve each published Go announcement as its original closed two-fixture edge.
    for name in [
        "transitions/go_wasm_identity_rebaseline.v1.json",
        "transitions/go_after_typescript_identity_rebaseline.v1.json",
        "transitions/typescript_after_go_identity_rebaseline.v1.json",
    ] {
        let record = old_fixture(name);
        let fixtures = record["fixtures"].as_array().unwrap();
        assert_eq!(fixtures.len(), 2);
        for edge in fixtures {
            identity_transitions::verify_fixture(
                edge["path"].as_str().unwrap(),
                edge["before"].as_str().unwrap(),
                edge["after"].as_str().unwrap(),
                std::slice::from_ref(&record),
            )
            .expect("preserved original Go identity announcement");
        }
    }
    // Preserve the original parallel C17 announcement; apply the combined edge below.
    let c17_parallel = old_fixture("transitions/c17_native_identity_rebaseline.v1.json");
    assert_eq!(c17_parallel["fixtures"].as_array().unwrap().len(), 2);
    for edge in c17_parallel["fixtures"].as_array().unwrap() {
        identity_transitions::verify_fixture(
            edge["path"].as_str().unwrap(),
            edge["before"].as_str().unwrap(),
            edge["after"].as_str().unwrap(),
            std::slice::from_ref(&c17_parallel),
        )
        .expect("preserved original C17 parallel identity announcement");
    }
    let c17_combined =
        old_fixture("transitions/pr414_after_typescript_c17_identity_rebaseline.v1.json");
    let c17_fixtures = c17_combined["fixtures"].as_array().unwrap();
    assert_eq!(c17_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(c17_fixtures.iter().any(|edge| edge["path"] == path));
    }
    // ctutils security patch: identity-only successor after combined C17 integration.
    let ctutils_security =
        old_fixture("transitions/pr414_ctutils_security_identity_rebaseline.v1.json");
    let security_fixtures = ctutils_security["fixtures"].as_array().unwrap();
    assert_eq!(security_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(security_fixtures.iter().any(|edge| edge["path"] == path));
    }
    // Workspace version bump to 0.4.1 (just bump 0.4.1): announced identity only.
    let release_0_4_1 = old_fixture("transitions/release_0_4_1_identity_rebaseline.v1.json");
    let release_0_4_1_fixtures = release_0_4_1["fixtures"].as_array().unwrap();
    assert_eq!(release_0_4_1_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(release_0_4_1_fixtures
            .iter()
            .any(|edge| edge["path"] == path));
    }
    // tokio 1.53.3 security patch (RUSTSEC-2026-0386): announced identity only.
    let tokio_security = old_fixture("transitions/tokio_security_identity_rebaseline.v1.json");
    let tokio_security_fixtures = tokio_security["fixtures"].as_array().unwrap();
    assert_eq!(tokio_security_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(tokio_security_fixtures
            .iter()
            .any(|edge| edge["path"] == path));
    }
    let integration_join =
        old_fixture("transitions/emergency_20261023_after_go_identity_join.v1.json");
    // Signature dependency security update: extend only the two announced identity files.
    let signature_security =
        old_fixture("transitions/pr429_signature_security_identity_rebaseline.v1.json");
    let signature_fixtures = signature_security["fixtures"].as_array().unwrap();
    assert_eq!(signature_fixtures.len(), 2);
    for path in changed_fixtures {
        assert!(signature_fixtures.iter().any(|edge| edge["path"] == path));
    }
    // Keep all committed optional-landmark identity successors as closed edges.
    let optional_landmarks_dependency =
        old_fixture("transitions/optional_landmarks_dependency_identity_rebaseline.v1.json");
    let optional_landmarks_pin =
        old_fixture("transitions/optional_landmarks_published_pin_identity_rebaseline.v1.json");
    let optional_landmarks_product =
        old_fixture("transitions/optional_landmarks_product_scope_identity_rebaseline.v1.json");
    let optional_landmarks_p1p2 =
        old_fixture("transitions/optional_landmarks_p1p2_identity_rebaseline.v1.json");
    let dope_main_repin = old_fixture("transitions/dope_main_repin_identity_rebaseline.v1.json");
    let dope_engine_v1_1_0_repin =
        old_fixture("transitions/dope_engine_v1_1_0_repin_identity_rebaseline.v1.json");
    let search_narration = old_fixture("transitions/search_narration_identity_rebaseline.v1.json");
    assert_eq!(
        search_narration["source_commit"],
        "4704b6197a49fe532e61915fac96b22f06b9e4f1"
    );
    let feat_math_hpc_w1_w9 =
        old_fixture("transitions/feat_math_hpc_w1_w9_identity_rebaseline.v1.json");
    assert_eq!(
        feat_math_hpc_w1_w9["source_commit"],
        "da2d02e8a0b54d615ad08e9509f20d2f2e1532dd"
    );
    let telemetry_lockfile =
        old_fixture("transitions/telemetry_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        telemetry_lockfile["source_commit"],
        "c4974e5a77fff40d135809ab5d2f49a8d9b353e2"
    );
    let rnk11_lockfile =
        old_fixture("transitions/rnk11_cpu_closure_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        rnk11_lockfile["source_commit"],
        "b54ad345e08a395c5e496cfbc625f376d1cdd51d"
    );
    let api01_lockfile =
        old_fixture("transitions/api01_jope_api_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        api01_lockfile["source_commit"],
        "6fb55261b22712fa011a9b8e96e3174c229ada61"
    );
    let dsc01_api = old_fixture("transitions/v1_dsc01_identity_rebaseline.v1.json");
    assert_eq!(
        dsc01_api["source_commit"],
        "d053acf5a84990caa3fd5a277666840e7991a01a"
    );
    assert_eq!(dsc01_api["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(dsc01_api["fixtures"][0]["path"], changed_fixtures[0]);
    let pub02_identity = old_fixture("transitions/pub02_identity_rebaseline.v1.json");
    assert_eq!(
        pub02_identity["source_commit"],
        "f606491e30f007786ac79975cd40c552f457b86d"
    );
    let api06_lockfile = old_fixture(
        "transitions/api06_workspace_registration_lockfile_identity_rebaseline.v1.json",
    );
    assert_eq!(
        api06_lockfile["source_commit"],
        "f11f6f474a5f457692d9e4ab6b4fb31860575a91"
    );
    assert_eq!(api06_lockfile["fixtures"].as_array().unwrap().len(), 2);
    assert_eq!(api06_lockfile["fixtures"][0]["path"], changed_fixtures[0]);
    assert_eq!(api06_lockfile["fixtures"][1]["path"], changed_fixtures[1]);
    let tui10_identity =
        old_fixture("transitions/tui10_portable_pty_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        tui10_identity["source_commit"],
        "e1c814b5885f5993b101ddf9d6f4c4eb4822040a"
    );
    assert_eq!(tui10_identity["fixtures"].as_array().unwrap().len(), 2);
    assert_eq!(tui10_identity["fixtures"][0]["path"], changed_fixtures[0]);
    assert_eq!(tui10_identity["fixtures"][1]["path"], changed_fixtures[1]);
    let gov04_portable_pty =
        old_fixture("transitions/gov04_pub06_portable_pty_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        gov04_portable_pty["source_commit"],
        "4b51c5528129c3fc4c170f2c4074a1a176c25eb3"
    );
    assert_eq!(gov04_portable_pty["fixtures"].as_array().unwrap().len(), 2);
    assert_eq!(
        gov04_portable_pty["fixtures"][0]["path"],
        changed_fixtures[0]
    );
    assert_eq!(
        gov04_portable_pty["fixtures"][1]["path"],
        changed_fixtures[1]
    );
    let api03_id_newtypes =
        old_fixture("transitions/api03_id_newtypes_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        api03_id_newtypes["source_commit"],
        "f99d5d09759f2a27609684d27f0f7ab9679bdff1"
    );
    assert_eq!(
        api03_id_newtypes["base_commit"],
        "8f417243b4073cd40eb2d80aaeeb491411ec245a"
    );
    assert_eq!(api03_id_newtypes["fixtures"].as_array().unwrap().len(), 2);
    assert_eq!(
        api03_id_newtypes["fixtures"][0]["path"],
        changed_fixtures[0]
    );
    assert_eq!(
        api03_id_newtypes["fixtures"][1]["path"],
        changed_fixtures[1]
    );
    let dsc02_lockfile =
        old_fixture("transitions/dsc02_service_sha2_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        dsc02_lockfile["source_commit"],
        "f19e71d6b6301e44017f4cd6594d11c01cfa326b"
    );
    assert_eq!(
        dsc02_lockfile["base_commit"],
        "9d4d7d6b8a941427a98ebb31616f1344ea3ee168"
    );
    assert_eq!(dsc02_lockfile["fixtures"].as_array().unwrap().len(), 2);
    assert_eq!(dsc02_lockfile["fixtures"][0]["path"], changed_fixtures[0]);
    assert_eq!(dsc02_lockfile["fixtures"][1]["path"], changed_fixtures[1]);
    // Preserve the original parallel DSC-02 edge separately; the queued re-chain below
    // starts from the merged-main identity and joins that source change into the linear history.
    for edge in dsc02_lockfile["fixtures"].as_array().unwrap() {
        identity_transitions::verify_fixture(
            edge["path"].as_str().unwrap(),
            edge["before"].as_str().unwrap(),
            edge["after"].as_str().unwrap(),
            std::slice::from_ref(&dsc02_lockfile),
        )
        .expect("preserved original parallel DSC-02 identity announcement");
    }
    let svc07_lockfile =
        old_fixture("transitions/svc07_config_loader_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        svc07_lockfile["source_commit"],
        "2372c2cf316e001347d01e7d494371446c1330f7"
    );
    assert_eq!(
        svc07_lockfile["base_commit"],
        "f99d5d09759f2a27609684d27f0f7ab9679bdff1"
    );
    assert_eq!(svc07_lockfile["fixtures"].as_array().unwrap().len(), 2);
    assert_eq!(svc07_lockfile["fixtures"][0]["path"], changed_fixtures[0]);
    assert_eq!(svc07_lockfile["fixtures"][1]["path"], changed_fixtures[1]);
    let wrk02_identity =
        old_fixture("transitions/wrk02_frame_codec_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        wrk02_identity["source_commit"],
        "b22603d45adc7260b3ab217569a53eaf55093c4e"
    );
    assert_eq!(
        wrk02_identity["base_commit"],
        "4158c237bcaf23ada68957206b417344ab1075a3"
    );
    assert_eq!(wrk02_identity["fixtures"].as_array().unwrap().len(), 2);
    assert_eq!(wrk02_identity["fixtures"][0]["path"], changed_fixtures[0]);
    assert_eq!(wrk02_identity["fixtures"][1]["path"], changed_fixtures[1]);
    let dsc04_lockfile =
        old_fixture("transitions/dsc04_elite_pool_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        dsc04_lockfile["source_commit"],
        "17c78f4e650112b19f1390971d4af11e28a14e8a"
    );
    let dsc02_rechain =
        old_fixture("transitions/dsc02_after_009de907_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        dsc02_rechain["source_commit"],
        "85aef0634193e54e607e29878185ab595da7cdb6"
    );
    assert_eq!(
        dsc02_rechain["base_commit"],
        "009de907e9c1c829f8014aac7974e6c283460f56"
    );
    let tui11_fake_transport =
        old_fixture("transitions/tui11_fake_transport_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        tui11_fake_transport["source_commit"],
        "b760efc14de92a14940851b767c0a3432f7173a9"
    );
    assert_eq!(
        tui11_fake_transport["base_commit"],
        "0462cd418f733269ba69270402fc7c97cf30eaa2"
    );
    let tui03_command_parser =
        old_fixture("transitions/tui03_command_parser_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        tui03_command_parser["source_commit"],
        "fb9925fe9c85294c80214625625abda19e31b891"
    );
    assert_eq!(
        tui03_command_parser["base_commit"],
        "13dfc08c15ae7e86a77229c6d429dffad7db5512"
    );
    let svc03_identity =
        old_fixture("transitions/svc03_run_controller_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        svc03_identity["source_commit"],
        "a1ba0f6cdda630c22d82146db24409232db61e8d"
    );
    assert_eq!(
        svc03_identity["base_commit"],
        "123349f752780968a8750e77c44ea4da4016c45a"
    );
    let wrk03_sampler =
        old_fixture("transitions/wrk03_sampler_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        wrk03_sampler["source_commit"],
        "eb170b9e22605de75bff6d5a0d6727a2509f394c"
    );
    assert_eq!(
        wrk03_sampler["base_commit"],
        "47a35eda5a2e87d2e46fea327f526d4a0857e4c9"
    );
    let obj01_runtime_fitness =
        old_fixture("transitions/obj01_runtime_fitness_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        obj01_runtime_fitness["source_commit"],
        "89b5ffe71abf9421f8b995ba3712d7765cc40f7e"
    );
    assert_eq!(
        obj01_runtime_fitness["base_commit"],
        "771755480af1d8c18ebe787f362262a70a343cfe"
    );
    let svc05_http_router =
        old_fixture("transitions/svc05_http_router_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        svc05_http_router["source_commit"],
        "899261ff31bd1be314e49dd18ea1b566f1cbb2ff"
    );
    assert_eq!(
        svc05_http_router["base_commit"],
        "80b954a2bf83e0ff3e8ed9fd6fbb72457b4d0940"
    );
    let bat05_cli_ulid =
        old_fixture("transitions/bat05_cli_ulid_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        bat05_cli_ulid["source_commit"],
        "942a9fd7e59022fbf0faedd14129818604bd5fcb"
    );
    assert_eq!(
        bat05_cli_ulid["base_commit"],
        "cd3bf1f432ece683eba47984e788225dc2a17ff9"
    );
    let svc06_daemon_file =
        old_fixture("transitions/svc06_daemon_file_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        svc06_daemon_file["source_commit"],
        "2012b7637a28f7252e84431a013a1ab1c2e7e68d"
    );
    assert_eq!(
        svc06_daemon_file["base_commit"],
        "0fb9dc3ab27714f42f3385d109d68be2bde5ccd5"
    );
    let svc01_store = old_fixture("transitions/svc01_store_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        svc01_store["source_commit"],
        "f2baa1a33814ef92bb3623e32dc8b1aed1746ecb"
    );
    assert_eq!(
        svc01_store["base_commit"],
        "cf6078317fbcb8afdb5a235490521cb4c98e0566"
    );
    // Preserve TUI-07's original parallel identity announcement from its pre-merge base.
    let tui07_lockfile =
        old_fixture("transitions/tui07_status_lines_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        tui07_lockfile["source_commit"],
        "7804bc3dcb898269572805722404c1dd8c026657"
    );
    assert_eq!(
        tui07_lockfile["base_commit"],
        "4158c237bcaf23ada68957206b417344ab1075a3"
    );
    assert_eq!(tui07_lockfile["fixtures"].as_array().unwrap().len(), 2);
    for edge in tui07_lockfile["fixtures"].as_array().unwrap() {
        identity_transitions::verify_fixture(
            edge["path"].as_str().unwrap(),
            edge["before"].as_str().unwrap(),
            edge["after"].as_str().unwrap(),
            std::slice::from_ref(&tui07_lockfile),
        )
        .expect("preserved original parallel TUI-07 identity announcement");
    }
    let tui07_rechain = old_fixture("transitions/tui07_after_eeabea_identity_rechain.v1.json");
    assert_eq!(
        tui07_rechain["source_commit"],
        "c23c022b9e6f9c266b70633a1c5d9bad3ed137fe"
    );
    assert_eq!(
        tui07_rechain["base_commit"],
        "eeabea7fa8fb8f156296511459058b2677922263"
    );
    assert_eq!(
        tui07_rechain["before_identity_digest"],
        svc01_store["after_identity_digest"]
    );
    assert_eq!(
        tui07_rechain["after_identity_digest"],
        svc01_store["after_identity_digest"]
    );
    assert_eq!(tui07_rechain["fixtures"].as_array().unwrap().len(), 2);
    assert_eq!(tui07_rechain["fixtures"][0]["path"], changed_fixtures[0]);
    assert_eq!(tui07_rechain["fixtures"][1]["path"], changed_fixtures[1]);
    let pub05_identity =
        old_fixture("transitions/pub05_phrase_renderer_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        pub05_identity["source_commit"],
        "7e55a0f9d5f3e5bdda33ff0eff67500dcdc6807f"
    );
    assert_eq!(
        pub05_identity["base_commit"],
        "d60ec583c5fd12b357722f5fe6d30e9f8dc9b25d"
    );
    assert_eq!(
        pub05_identity["before_identity_digest"],
        tui07_rechain["after_identity_digest"]
    );
    assert_eq!(pub05_identity["fixtures"].as_array().unwrap().len(), 2);

    let svc11_jmat = old_fixture("transitions/svc11_jmat_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        svc11_jmat["source_commit"],
        "187f02bbb196f208e83d1b3aa517340a47458522"
    );
    assert_eq!(
        svc11_jmat["before_identity_digest"],
        pub05_identity["after_identity_digest"]
    );
    let svc04_identity =
        old_fixture("transitions/svc04_event_journal_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        svc04_identity["source_commit"],
        "08571c00178e91ded00e9df08feaf936f4124954"
    );
    assert_eq!(
        svc04_identity["base_commit"],
        "2a7c4cdd266ae7d441c81f83e52bb78ce9b782c0"
    );
    let clt01_client_transports =
        old_fixture("transitions/clt01_client_transports_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        clt01_client_transports["source_commit"],
        "438631e619f67479eaf89a60bedf5e8d595b9a8e"
    );
    assert_eq!(
        clt01_client_transports["base_commit"],
        "d112464133c9fa4454f39f617554d119dfed0356"
    );
    let tui01_terminal =
        old_fixture("transitions/tui01_terminal_skeleton_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        tui01_terminal["source_commit"],
        "8042ebbacab435e91457caeffbb1400d37291cec"
    );
    assert_eq!(
        tui01_terminal["base_commit"],
        "5614f6fed6154e82a8033a96ba2aac9ff5475be0"
    );
    assert_eq!(
        tui01_terminal["before_identity_digest"],
        clt01_client_transports["after_identity_digest"]
    );

    let wrk01_worker_pool =
        old_fixture("transitions/wrk01_worker_pool_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        wrk01_worker_pool["source_commit"],
        "4c09aea83b200ee0c27e87bb63963e8cfa8d97f2"
    );
    assert_eq!(
        wrk01_worker_pool["base_commit"],
        "fd3013c753ab3eee30ce6d54a3485dd7f289bd06"
    );
    let rnk01_bundle_loader =
        old_fixture("transitions/rnk01_bundle_loader_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        rnk01_bundle_loader["source_commit"],
        "a6d96eeca2a1bace9b3dd323d10816a9d1356439"
    );
    assert_eq!(
        rnk01_bundle_loader["base_commit"],
        "993ce5f700a45a25f668aecfd461b214013e4d85"
    );
    assert_eq!(rnk01_bundle_loader["fixtures"].as_array().unwrap().len(), 2);
    assert_eq!(
        rnk01_bundle_loader["fixtures"][0]["path"],
        changed_fixtures[0]
    );
    assert_eq!(
        rnk01_bundle_loader["fixtures"][1]["path"],
        changed_fixtures[1]
    );
    let mcp02_rmcp = old_fixture("transitions/mcp02_rmcp_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        mcp02_rmcp["source_commit"],
        "6790a10355c67f2993a4c00c43c96015166be3af"
    );
    assert_eq!(
        mcp02_rmcp["base_commit"],
        "dee223831e1462d9d097389ee45f4cc829c304fa"
    );
    assert_eq!(
        mcp02_rmcp["before_identity_digest"],
        rnk01_bundle_loader["after_identity_digest"]
    );
    let clt03_files = old_fixture("transitions/clt03_files_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        clt03_files["source_commit"],
        "26aaf1f01f412426fd2c7f337b9758e2b4ec0a06"
    );
    assert_eq!(
        clt03_files["base_commit"],
        "ff127e4d61cb436903b83c59d09afa20d120181a"
    );
    assert_eq!(
        clt03_files["before_identity_digest"],
        mcp02_rmcp["after_identity_digest"]
    );
    assert_eq!(clt03_files["fixtures"].as_array().unwrap().len(), 2);
    assert_eq!(clt03_files["fixtures"][0]["path"], changed_fixtures[0]);
    assert_eq!(clt03_files["fixtures"][1]["path"], changed_fixtures[1]);
    let exp09_bundle = old_fixture("transitions/exp09_bundle_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        exp09_bundle["source_commit"],
        "acb0f4ef374498745a7ea9560ee98c2c11fc6521"
    );
    assert_eq!(
        exp09_bundle["base_commit"],
        "62327dd06f1f5eb2d8504be73fcdc593349a601d"
    );
    assert_eq!(
        exp09_bundle["before_identity_digest"],
        clt03_files["after_identity_digest"]
    );
    assert_eq!(exp09_bundle["fixtures"].as_array().unwrap().len(), 2);
    assert_eq!(exp09_bundle["fixtures"][0]["path"], changed_fixtures[0]);
    assert_eq!(exp09_bundle["fixtures"][1]["path"], changed_fixtures[1]);
    let tel04f01_nvml =
        old_fixture("transitions/tel04f01_nvml_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        tel04f01_nvml["source_commit"],
        "decef104718a27e235d641bd2f5a91ec0bdced81"
    );
    assert_eq!(
        tel04f01_nvml["base_commit"],
        "b22b932f869d1f9d87882ac7bf506299fd5dce35"
    );
    let svc09_engine = old_fixture("transitions/svc09_engine_lock_identity_rebaseline.v1.json");
    assert_eq!(
        svc09_engine["source_commit"],
        "8ff0c4476957e500349792f4f87484ed7db6d775"
    );
    assert_eq!(
        svc09_engine["base_commit"],
        "9fe8c6753919cfaaac05ccb0d7ad125ff281dac4"
    );
    assert_eq!(
        svc09_engine["before_identity_digest"],
        tel04f01_nvml["after_identity_digest"]
    );
    assert_eq!(svc09_engine["fixtures"].as_array().unwrap().len(), 2);
    assert_eq!(svc09_engine["fixtures"][0]["path"], changed_fixtures[0]);
    assert_eq!(svc09_engine["fixtures"][1]["path"], changed_fixtures[1]);
    let bat02_jope_paths =
        old_fixture("transitions/bat02_jope_paths_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        bat02_jope_paths["source_commit"],
        "a8edb0d6ea06ad6d8df7ef1e65ee5af399ab430d"
    );
    assert_eq!(
        bat02_jope_paths["base_commit"],
        "57693b7712aff9ffdf6caa689769e93bc5b8b3db"
    );
    assert_eq!(
        bat02_jope_paths["before_identity_digest"],
        svc09_engine["after_identity_digest"]
    );
    assert_eq!(bat02_jope_paths["fixtures"].as_array().unwrap().len(), 2);
    assert_eq!(bat02_jope_paths["fixtures"][0]["path"], changed_fixtures[0]);
    assert_eq!(bat02_jope_paths["fixtures"][1]["path"], changed_fixtures[1]);
    let dsc08_heuristic =
        old_fixture("transitions/dsc08_heuristic_loop_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        dsc08_heuristic["source_commit"],
        "62473fb5ed12333e705ed1b0ce332051e0960c21"
    );
    assert_eq!(
        dsc08_heuristic["base_commit"],
        "6697c1402249ede99db9b290a50b3808c1e8bba6"
    );
    assert_eq!(
        dsc08_heuristic["before_identity_digest"],
        bat02_jope_paths["after_identity_digest"]
    );
    let clienttmp_tempfile =
        old_fixture("transitions/clienttmp_tempfile_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        clienttmp_tempfile["source_commit"],
        "900ea63e88a1af154fa3bcbc6dc6ef559e54a179"
    );
    assert_eq!(
        clienttmp_tempfile["base_commit"],
        "4e6a834794d0bacb7d792a6ff3e1b9172fa435af"
    );
    let rnk09_ruzstd = old_fixture("transitions/rnk09_ruzstd_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        rnk09_ruzstd["source_commit"],
        "d3b589181be62f2b99928be419f359c0b77e7626"
    );
    assert_eq!(
        rnk09_ruzstd["base_commit"],
        "31ba8225da6c8c0128dd8c58fbb2839920fb102f"
    );
    let res09_candle = old_fixture("transitions/res09_candle_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        res09_candle["source_commit"],
        "27b292c7e76b28b15cddebea15f0683a5baab9ae"
    );
    assert_eq!(
        res09_candle["base_commit"],
        "b452594f63fb44d607d558a5116643a2db51c3a2"
    );
    let gov03_axum = old_fixture("transitions/gov03_axum_dev_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        gov03_axum["source_commit"],
        "aacd29a1555e9bdc02be829ca1181460f29dc58f"
    );
    assert_eq!(
        gov03_axum["base_commit"],
        "1e1229abcdc50f360721c5c3cc6912b58734e30e"
    );
    // GOV04 already records the approved metadata-only successor on main. Every historical
    // chain must reach it; otherwise the frozen current fixture is incorrectly called unledgered.
    let gov04_portable_pty =
        old_fixture("transitions/gov04_pub06_portable_pty_lockfile_identity_rebaseline.v1.json");
    assert_eq!(
        gov04_portable_pty["source_commit"],
        "4b51c5528129c3fc4c170f2c4074a1a176c25eb3"
    );
    assert_eq!(
        gov04_portable_pty["base_commit"],
        "3df2979abf861502f69f3fa70e7871952bbf0c06"
    );
    for record in [
        &optional_landmarks_dependency,
        &optional_landmarks_pin,
        &optional_landmarks_product,
        &optional_landmarks_p1p2,
        &dope_main_repin,
        &dope_engine_v1_1_0_repin,
        &search_narration,
        &feat_math_hpc_w1_w9,
        &telemetry_lockfile,
        &rnk11_lockfile,
        &api01_lockfile,
        &pub02_identity,
        &api06_lockfile,
        &tui10_identity,
        &api03_id_newtypes,
        &dsc02_lockfile,
        &svc07_lockfile,
        &wrk02_identity,
        &dsc04_lockfile,
        &dsc02_rechain,
        &tui11_fake_transport,
        &tui03_command_parser,
        &svc03_identity,
        &wrk03_sampler,
        &obj01_runtime_fitness,
        &svc05_http_router,
        &bat05_cli_ulid,
        &svc06_daemon_file,
        &svc01_store,
        &tui07_lockfile,
        &tui07_rechain,
        &pub05_identity,
        &svc11_jmat,
        &svc04_identity,
        &clt01_client_transports,
        &tui01_terminal,
        &wrk01_worker_pool,
        &rnk01_bundle_loader,
        &mcp02_rmcp,
        &clt03_files,
        &exp09_bundle,
        &tel04f01_nvml,
        &svc09_engine,
        &bat02_jope_paths,
        &dsc08_heuristic,
        &clienttmp_tempfile,
        &rnk09_ruzstd,
        &res09_candle,
        &gov03_axum,
        &gov04_portable_pty,
    ] {
        let fixtures = record["fixtures"].as_array().unwrap();
        assert_eq!(fixtures.len(), 2);
        for (edge, path) in fixtures.iter().zip(changed_fixtures) {
            assert_eq!(edge["path"], path);
        }
    }
    let shared_feature_join =
        old_fixture("transitions/pr416_after_shared_capacity_release_identity_join.v1.json");
    let feature_shared_join = old_fixture(
        "transitions/pr416_shared_capacity_release_after_feature_identity_join.v1.json",
    );
    for record in [&shared_feature_join, &feature_shared_join] {
        assert_eq!(record["fixtures"].as_array().unwrap().len(), 2);
    }
    // The capacity branch announced the same Tokio fix on the old workspace version.
    // Keep its original edge and verify the parallel join to the release identity.
    let capacity_tokio =
        old_fixture("transitions/pr420_tokio_security_identity_rebaseline.v1.json");
    let capacity_release_join =
        old_fixture("transitions/pr420_after_release_0_4_1_identity_join.v1.json");
    for record in [&capacity_tokio, &capacity_release_join] {
        assert_eq!(record["fixtures"].as_array().unwrap().len(), 2);
    }
    for edge in capacity_tokio["fixtures"].as_array().unwrap() {
        let path = edge["path"].as_str().unwrap();
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let current = sha256(&std::fs::read(repo.join(path)).unwrap());
        identity_transitions::verify_fixture(
            path,
            edge["before"].as_str().unwrap(),
            &current,
            &[
                capacity_tokio.clone(),
                capacity_release_join.clone(),
                shared_feature_join.clone(),
                integration_join.clone(),
                signature_security.clone(),
                optional_landmarks_dependency.clone(),
                optional_landmarks_pin.clone(),
                optional_landmarks_product.clone(),
                optional_landmarks_p1p2.clone(),
                dope_main_repin.clone(),
                dope_engine_v1_1_0_repin.clone(),
                search_narration.clone(),
                feat_math_hpc_w1_w9.clone(),
                telemetry_lockfile.clone(),
                rnk11_lockfile.clone(),
                api01_lockfile.clone(),
                dsc01_api.clone(),
                pub02_identity.clone(),
                api06_lockfile.clone(),
                tui10_identity.clone(),
                api03_id_newtypes.clone(),
                svc07_lockfile.clone(),
                wrk02_identity.clone(),
                dsc04_lockfile.clone(),
                dsc02_rechain.clone(),
                tui11_fake_transport.clone(),
                tui03_command_parser.clone(),
                svc03_identity.clone(),
                wrk03_sampler.clone(),
                obj01_runtime_fitness.clone(),
                svc05_http_router.clone(),
                bat05_cli_ulid.clone(),
                svc06_daemon_file.clone(),
                svc01_store.clone(),
                tui07_rechain.clone(),
                pub05_identity.clone(),
                svc11_jmat.clone(),
                svc04_identity.clone(),
                clt01_client_transports.clone(),
                tui01_terminal.clone(),
                wrk01_worker_pool.clone(),
                rnk01_bundle_loader.clone(),
                mcp02_rmcp.clone(),
                clt03_files.clone(),
                exp09_bundle.clone(),
                tel04f01_nvml.clone(),
                svc09_engine.clone(),
                bat02_jope_paths.clone(),
                dsc08_heuristic.clone(),
                clienttmp_tempfile.clone(),
                rnk09_ruzstd.clone(),
                res09_candle.clone(),
                gov03_axum.clone(),
                gov04_portable_pty.clone(),
            ],
        )
        .expect("capacity Tokio history joins the release0.4.1 security identity");
    }
    let go_after_c17 = old_fixture("transitions/pr416_after_c17_identity_rebaseline.v1.json");
    let go_parallel_join =
        old_fixture("transitions/pr416_c17_after_go_identity_rebaseline.v1.json");
    let published_go = old_fixture("transitions/go_after_typescript_identity_rebaseline.v1.json");
    for record in [&go_after_c17, &go_parallel_join] {
        assert_eq!(record["fixtures"].as_array().unwrap().len(), 2);
    }
    for edge in published_go["fixtures"].as_array().unwrap() {
        let path = edge["path"].as_str().unwrap();
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let current = sha256(&std::fs::read(repo.join(path)).unwrap());
        identity_transitions::verify_fixture(
            path,
            edge["after"].as_str().unwrap(),
            &current,
            &[
                go_parallel_join.clone(),
                feature_shared_join.clone(),
                integration_join.clone(),
                signature_security.clone(),
                optional_landmarks_dependency.clone(),
                optional_landmarks_pin.clone(),
                optional_landmarks_product.clone(),
                optional_landmarks_p1p2.clone(),
                dope_main_repin.clone(),
                dope_engine_v1_1_0_repin.clone(),
                search_narration.clone(),
                feat_math_hpc_w1_w9.clone(),
                telemetry_lockfile.clone(),
                rnk11_lockfile.clone(),
                api01_lockfile.clone(),
                dsc01_api.clone(),
                pub02_identity.clone(),
                api06_lockfile.clone(),
                tui10_identity.clone(),
                api03_id_newtypes.clone(),
                svc07_lockfile.clone(),
                wrk02_identity.clone(),
                dsc04_lockfile.clone(),
                dsc02_rechain.clone(),
                tui11_fake_transport.clone(),
                tui03_command_parser.clone(),
                svc03_identity.clone(),
                wrk03_sampler.clone(),
                obj01_runtime_fitness.clone(),
                svc05_http_router.clone(),
                bat05_cli_ulid.clone(),
                svc06_daemon_file.clone(),
                svc01_store.clone(),
                tui07_rechain.clone(),
                pub05_identity.clone(),
                svc11_jmat.clone(),
                svc04_identity.clone(),
                clt01_client_transports.clone(),
                tui01_terminal.clone(),
                wrk01_worker_pool.clone(),
                rnk01_bundle_loader.clone(),
                mcp02_rmcp.clone(),
                clt03_files.clone(),
                exp09_bundle.clone(),
                tel04f01_nvml.clone(),
                svc09_engine.clone(),
                bat02_jope_paths.clone(),
                dsc08_heuristic.clone(),
                clienttmp_tempfile.clone(),
                rnk09_ruzstd.clone(),
                res09_candle.clone(),
                gov03_axum.clone(),
                gov04_portable_pty.clone(),
            ],
        )
        .expect("published Go fixtures join the combined C17/TypeScript/Go identity");
    }
    for edge in go_after_c17["fixtures"].as_array().unwrap() {
        let path = edge["path"].as_str().unwrap();
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let current = sha256(&std::fs::read(repo.join(path)).unwrap());
        identity_transitions::verify_fixture(
            path,
            edge["before"].as_str().unwrap(),
            &current,
            &[
                go_after_c17.clone(),
                feature_shared_join.clone(),
                integration_join.clone(),
                signature_security.clone(),
                optional_landmarks_dependency.clone(),
                optional_landmarks_pin.clone(),
                optional_landmarks_product.clone(),
                optional_landmarks_p1p2.clone(),
                dope_main_repin.clone(),
                dope_engine_v1_1_0_repin.clone(),
                search_narration.clone(),
                feat_math_hpc_w1_w9.clone(),
                telemetry_lockfile.clone(),
                rnk11_lockfile.clone(),
                api01_lockfile.clone(),
                dsc01_api.clone(),
                pub02_identity.clone(),
                api06_lockfile.clone(),
                tui10_identity.clone(),
                api03_id_newtypes.clone(),
                svc07_lockfile.clone(),
                wrk02_identity.clone(),
                dsc04_lockfile.clone(),
                dsc02_rechain.clone(),
                tui11_fake_transport.clone(),
                tui03_command_parser.clone(),
                svc03_identity.clone(),
                wrk03_sampler.clone(),
                obj01_runtime_fitness.clone(),
                svc05_http_router.clone(),
                bat05_cli_ulid.clone(),
                svc06_daemon_file.clone(),
                svc01_store.clone(),
                tui07_rechain.clone(),
                pub05_identity.clone(),
                svc11_jmat.clone(),
                svc04_identity.clone(),
                clt01_client_transports.clone(),
                tui01_terminal.clone(),
                wrk01_worker_pool.clone(),
                rnk01_bundle_loader.clone(),
                mcp02_rmcp.clone(),
                clt03_files.clone(),
                exp09_bundle.clone(),
                tel04f01_nvml.clone(),
                svc09_engine.clone(),
                bat02_jope_paths.clone(),
                dsc08_heuristic.clone(),
                clienttmp_tempfile.clone(),
                rnk09_ruzstd.clone(),
                res09_candle.clone(),
                gov03_axum.clone(),
                gov04_portable_pty.clone(),
            ],
        )
        .expect("published feature history joins shared capacity and release fixes");
    }
    // Preserve CUDA's independently announced history through the combined identity.
    let cuda_cpp = old_fixture("transitions/cuda_cpp_identity_rebaseline.v1.json");
    assert_eq!(cuda_cpp["fixtures"].as_array().unwrap().len(), 2);
    for path in changed_fixtures {
        assert!(cuda_cpp["fixtures"]
            .as_array()
            .unwrap()
            .iter()
            .any(|edge| edge["path"] == path));
    }
    let cuda_local_only = old_fixture("transitions/cuda_local_only_ci_identity_rebaseline.v1.json");
    assert_eq!(cuda_local_only["fixtures"].as_array().unwrap().len(), 2);
    for path in changed_fixtures {
        assert!(cuda_local_only["fixtures"]
            .as_array()
            .unwrap()
            .iter()
            .any(|edge| edge["path"] == path));
    }
    let cuda_feature_shared_join = old_fixture(
        "transitions/pr422_shared_capacity_release_after_feature_identity_join.v1.json",
    );
    let cuda_shared_feature_join =
        old_fixture("transitions/pr422_after_shared_capacity_release_identity_join.v1.json");
    let cuda_integration_join =
        old_fixture("transitions/emergency_20261023_after_cuda_identity_join.v1.json");
    for record in [
        &cuda_cpp,
        &cuda_local_only,
        &cuda_feature_shared_join,
        &cuda_shared_feature_join,
        &cuda_integration_join,
    ] {
        assert_eq!(record["fixtures"].as_array().unwrap().len(), 2);
    }
    for (origin, chain) in [
        (
            &cuda_cpp,
            vec![
                cuda_cpp.clone(),
                cuda_local_only.clone(),
                cuda_feature_shared_join.clone(),
                cuda_integration_join.clone(),
                signature_security.clone(),
                optional_landmarks_dependency.clone(),
                optional_landmarks_pin.clone(),
                optional_landmarks_product.clone(),
                optional_landmarks_p1p2.clone(),
                dope_main_repin.clone(),
                dope_engine_v1_1_0_repin.clone(),
                search_narration.clone(),
                feat_math_hpc_w1_w9.clone(),
                telemetry_lockfile.clone(),
                rnk11_lockfile.clone(),
                api01_lockfile.clone(),
                dsc01_api.clone(),
                pub02_identity.clone(),
                api06_lockfile.clone(),
                tui10_identity.clone(),
                api03_id_newtypes.clone(),
                svc07_lockfile.clone(),
                wrk02_identity.clone(),
                dsc04_lockfile.clone(),
                dsc02_rechain.clone(),
                tui11_fake_transport.clone(),
                tui03_command_parser.clone(),
                svc03_identity.clone(),
                wrk03_sampler.clone(),
                obj01_runtime_fitness.clone(),
                svc05_http_router.clone(),
                bat05_cli_ulid.clone(),
                svc06_daemon_file.clone(),
                svc01_store.clone(),
                tui07_rechain.clone(),
                pub05_identity.clone(),
                svc11_jmat.clone(),
                svc04_identity.clone(),
                clt01_client_transports.clone(),
                tui01_terminal.clone(),
                wrk01_worker_pool.clone(),
                rnk01_bundle_loader.clone(),
                mcp02_rmcp.clone(),
                clt03_files.clone(),
                exp09_bundle.clone(),
                tel04f01_nvml.clone(),
                svc09_engine.clone(),
                bat02_jope_paths.clone(),
                dsc08_heuristic.clone(),
                clienttmp_tempfile.clone(),
                rnk09_ruzstd.clone(),
                res09_candle.clone(),
                gov03_axum.clone(),
                gov04_portable_pty.clone(),
            ],
        ),
        (
            &cuda_shared_feature_join,
            vec![
                cuda_shared_feature_join.clone(),
                cuda_integration_join.clone(),
                signature_security.clone(),
                optional_landmarks_dependency.clone(),
                optional_landmarks_pin.clone(),
                optional_landmarks_product.clone(),
                optional_landmarks_p1p2.clone(),
                dope_main_repin.clone(),
                dope_engine_v1_1_0_repin.clone(),
                search_narration.clone(),
                feat_math_hpc_w1_w9.clone(),
                telemetry_lockfile.clone(),
                rnk11_lockfile.clone(),
                api01_lockfile.clone(),
                dsc01_api.clone(),
                pub02_identity.clone(),
                api06_lockfile.clone(),
                tui10_identity.clone(),
                api03_id_newtypes.clone(),
                svc07_lockfile.clone(),
                wrk02_identity.clone(),
                dsc04_lockfile.clone(),
                dsc02_rechain.clone(),
                tui11_fake_transport.clone(),
                tui03_command_parser.clone(),
                svc03_identity.clone(),
                wrk03_sampler.clone(),
                obj01_runtime_fitness.clone(),
                svc05_http_router.clone(),
                bat05_cli_ulid.clone(),
                svc06_daemon_file.clone(),
                svc01_store.clone(),
                tui07_rechain.clone(),
                pub05_identity.clone(),
                svc11_jmat.clone(),
                svc04_identity.clone(),
                clt01_client_transports.clone(),
                tui01_terminal.clone(),
                wrk01_worker_pool.clone(),
                rnk01_bundle_loader.clone(),
                mcp02_rmcp.clone(),
                clt03_files.clone(),
                exp09_bundle.clone(),
                tel04f01_nvml.clone(),
                svc09_engine.clone(),
                bat02_jope_paths.clone(),
                dsc08_heuristic.clone(),
                clienttmp_tempfile.clone(),
                rnk09_ruzstd.clone(),
                res09_candle.clone(),
                gov03_axum.clone(),
                gov04_portable_pty.clone(),
            ],
        ),
    ] {
        for edge in origin["fixtures"].as_array().unwrap() {
            let path = edge["path"].as_str().unwrap();
            let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
            let current = sha256(&std::fs::read(repo.join(path)).unwrap());
            identity_transitions::verify_fixture(
                path,
                edge["before"].as_str().unwrap(),
                &current,
                &chain,
            )
            .expect("preserved CUDA history joins combined integration identity");
        }
    }
    let identity_chain = [
        transition,
        successor,
        b12,
        b13,
        b10,
        b11,
        scalar_recipes,
        manifest_cmp,
        authority_clock,
        clippy_floor,
        corpus_bounds,
        executable_cache,
        sealed_runtime,
        eval_isolation,
        release_0_3_0,
        exp_next_04,
        i09_artifact_target_binding,
        batch4_selfplay,
        mr06_label_admission,
        mr19_shared_dag,
        observatory_workspace_lock,
        gp_source_stream,
        gpt6p1_sol_max,
        appliance_after_automl,
        jankurai_runtime_safety,
        jankurai_dependency_security,
        pr396_combined,
        pr397_combined,
        pr398_combined,
        release_0_4_0,
        model_objects,
        typescript,
        c17_combined,
        ctutils_security,
        release_0_4_1,
        tokio_security,
        shared_feature_join,
        integration_join,
        signature_security,
        optional_landmarks_dependency,
        optional_landmarks_pin,
        optional_landmarks_product,
        optional_landmarks_p1p2,
        dope_main_repin,
        dope_engine_v1_1_0_repin,
        search_narration,
        feat_math_hpc_w1_w9,
        telemetry_lockfile,
        rnk11_lockfile,
        api01_lockfile,
        dsc01_api,
        pub02_identity,
        api06_lockfile,
        tui10_identity,
        api03_id_newtypes,
        svc07_lockfile,
        wrk02_identity,
        dsc04_lockfile,
        dsc02_rechain,
        tui11_fake_transport,
        tui03_command_parser,
        svc03_identity,
        wrk03_sampler,
        obj01_runtime_fitness,
        svc05_http_router,
        bat05_cli_ulid,
        svc06_daemon_file,
        svc01_store,
        tui07_rechain,
        pub05_identity,
        svc11_jmat,
        svc04_identity,
        clt01_client_transports,
        tui01_terminal,
        wrk01_worker_pool,
        rnk01_bundle_loader,
        mcp02_rmcp,
        clt03_files,
        exp09_bundle,
        tel04f01_nvml,
        svc09_engine,
        bat02_jope_paths,
        dsc08_heuristic,
        clienttmp_tempfile,
        rnk09_ruzstd,
        res09_candle,
        gov03_axum,
        gov04_portable_pty,
    ];
    assert_eq!(rows.len(), 49);
    assert_eq!(index["source_count"], 49);
    let mut old_fixtures = 0;
    let mut release = 0;
    let mut bundles = 0;
    for row in rows {
        let uri = row["uri"].as_str().unwrap();
        let hash = row["sha256"].as_str().unwrap();
        assert_eq!(hash.len(), 64, "{uri}");
        if row["format"] == "release asset" {
            release += 1;
            let name = uri.rsplit('/').next().unwrap();
            assert_eq!(index["release_asset_sha256"][name], hash, "{uri}");
        }
        if uri.starts_with("s3://") && uri.ends_with("/manifest.json") {
            bundles += 1;
            assert_eq!(index["s3_bundle_manifest_sha256"][uri], hash, "{uri}");
        }
        if let Some(path) = uri.strip_prefix(&format!(
            "git+https://github.com/neverhuman/JopeDime@{BASE}:"
        )) {
            if path.starts_with("crates/feat-math/tests/fixtures/compat_freeze/") {
                old_fixtures += 1;
                let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
                let current = sha256(&std::fs::read(repo.join(path)).unwrap());
                identity_transitions::verify_fixture(path, hash, &current, &identity_chain)
                    .unwrap_or_else(|error| panic!("{path}: {error}"));
            }
        }
    }
    assert_eq!(old_fixtures, 10);
    assert_eq!(release, 8);
    assert_eq!(bundles, 3);
    assert!(source["coverage_gaps"]
        .as_array()
        .unwrap()
        .iter()
        .any(|gap| {
            gap["family"] == "shipped feat-math fitted Model/BoundModel" && gap["status"] == "held"
        }));
}

#[test]
fn unknown_exact_tags_refuse_before_evaluation() {
    let expression = MathExpr::from_ast(Expr::Var(Var::X)).unwrap();
    let genome = GenomeIr::Symbolic {
        source: "x".to_string(),
        expr: expression,
        affine: false,
    };
    let mut bytes = Program::import_legacy(&genome).to_bytes().unwrap();
    let tag_offset = 8 + 1 + 1 + 1 + 8 + 1 + 1;
    assert_eq!(
        bytes[tag_offset - 1],
        2,
        "Var exact tag follows AST discriminator"
    );
    bytes[tag_offset] = 255;
    assert_eq!(
        Program::from_bytes(&bytes).unwrap_err().to_string(),
        "Syntax(\"math operation tag\")"
    );

    let expression = MathExpr::from_ast(Expr::Call(Func::Abs, vec![Expr::Var(Var::X)])).unwrap();
    let genome = GenomeIr::Symbolic {
        source: "x".to_string(),
        expr: expression,
        affine: false,
    };
    let mut bytes = Program::import_legacy(&genome).to_bytes().unwrap();
    assert_eq!(
        bytes[tag_offset - 1],
        5,
        "Func exact tag follows AST discriminator"
    );
    bytes[tag_offset] = 255;
    assert_eq!(
        Program::from_bytes(&bytes).unwrap_err().to_string(),
        "Syntax(\"math operation tag\")"
    );
}
