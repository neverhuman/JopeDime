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
