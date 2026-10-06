//! Capability admission and native faults for the opt-in scalar lifecycle.
use feat_math::kernel::capabilities::{self, Class, Grammar, Manifest, Tier};
use feat_math::kernel::inverse_links::InverseLink;
use feat_math::kernel::numerical_policy::{asinh_v2, reduce_v2, NumericalPolicy, Reduction};
use feat_math::kernel::stable_exponential::StableExponential;
use feat_math::kernel::wire_v2::Domain;
use feat_math::math_expr::Func;

#[test]
fn explicit_scalar_capabilities_admit_experimental_without_promoting_frozen_legacy() {
    let frozen = capabilities::manifest_json();
    for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
        for id in [13, 19, 20, 21, 22] {
            let row = capabilities::scalar_function_v2_capability(id, policy);
            if id == 13 && policy == NumericalPolicy::LegacyV1 {
                assert!(row.is_none());
                continue;
            }
            let row = row.unwrap();
            assert_eq!(
                row.class,
                Class::Searchable {
                    since: Grammar::ScalarV2,
                    tier: Tier::Experimental
                }
            );
            assert!(row.class.admits(Tier::Experimental));
            assert!(!row.class.admits(Tier::Baseline));
            assert!(row.emitters.rust_recipe.accepts());
            assert!(row.emitters.python_recipe.accepts());
            assert!(row.emitters.cpp17_recipe.accepts());
            assert!(row.emitters.cpp17_fitted.accepts());
            assert!(!row.emitters.javascript.accepts());
            assert_eq!(row.portable_twin, matches!(id, 21 | 22));
            assert!(
                feat_math::transpile::lint_shipped_text(&serde_json::to_string(&row).unwrap())
                    .is_empty()
            );
        }
        for id in [0, 18, 23, u16::MAX] {
            assert!(capabilities::scalar_function_v2_capability(id, policy).is_none());
        }
    }
    let numerical = capabilities::scalar_policy_v2_capability();
    assert!(numerical.class.admits(Tier::Experimental));
    assert!(!numerical.class.admits(Tier::Baseline));
    assert!(!numerical.portable_twin);
    assert!(!capabilities::func_capability(Func::Asinh)
        .class
        .admits(Tier::Baseline));
    for row in Manifest::current().v2_reservations {
        if row.domain == Domain::Function && (19..=22).contains(&row.id) {
            assert_eq!(row.lifecycle, "reserved");
            assert!(!row.generatable);
        }
    }
    assert_eq!(capabilities::manifest_json(), frozen);
}

#[test]
fn native_boundary_faults_are_covered_by_the_separate_capability_declaration() {
    let inputs = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        -1.0e308,
        1.0e308,
        -31.0,
        31.0,
        -0.5,
        0.5,
        1.0,
    ];
    for id in [13, 19, 20, 21, 22] {
        let row =
            capabilities::scalar_function_v2_capability(id, NumericalPolicy::StableV2).unwrap();
        for value in inputs {
            let (_, faults) = match id {
                13 => asinh_v2(value),
                19 | 20 => InverseLink::from_versioned_id(2, id)
                    .unwrap()
                    .evaluate(value),
                _ => StableExponential::from_versioned_id(2, id)
                    .unwrap()
                    .evaluate(value),
            };
            assert_eq!(
                faults.bits() & !row.fault_classes.bits(),
                0,
                "id{id} {:016x}",
                value.to_bits()
            );
        }
    }
    let row = capabilities::scalar_policy_v2_capability();
    let inputs: &[&[f64]] = &[
        &[],
        &[0.0, -0.0],
        &[f64::NAN],
        &[f64::INFINITY],
        &[1.0e308, -1.0e308, 1.0],
        &[1.0e308, 1.0e308],
        &[f64::from_bits(1)],
    ];
    for reduction in [
        Reduction::Sum,
        Reduction::Mean,
        Reduction::L1,
        Reduction::L2,
    ] {
        for input in inputs {
            let (_, faults) = reduce_v2(reduction, input);
            assert_eq!(
                faults.bits() & !row.fault_classes.bits(),
                0,
                "{reduction:?} {input:?}"
            );
        }
    }
}

#[test]
fn capability_decisions_bind_receipt_hashes_and_failed_promotion_gates() {
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use std::collections::BTreeSet;
    use std::path::Path;

    let document: Value =
        serde_json::from_str(capabilities::scalar_capability_decisions_json()).unwrap();
    assert_eq!(
        document["schema"],
        "feat-math.scalar-capability-decisions.v1"
    );
    assert_eq!(document["promotion_authorized"], false);
    let cards = document["cards"].as_array().unwrap();
    assert_eq!(cards.len(), 3);
    let mut names = BTreeSet::new();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for card in cards {
        let name = card["card"].as_str().unwrap();
        assert!(matches!(name, "B05" | "B06" | "B07"));
        assert!(names.insert(name));
        assert_eq!(card["class"], "experimental");
        assert_eq!(card["decision"], "keep-experimental");
        assert_eq!(card["default_enabled"], false);
        assert_eq!(card["parser_ids_retained"], true);
        let filename = format!("{}-scalar-v2", name.to_lowercase());
        for (path_key, hash_key, extension) in [
            ("receipt_path", "receipt_sha256", "json"),
            ("markdown_path", "markdown_sha256", "md"),
        ] {
            let expected = format!("docs/internal/gpms/receipts/{filename}.{extension}");
            assert_eq!(card[path_key], expected);
            let bytes = std::fs::read(root.join(expected)).unwrap();
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), card[hash_key]);
            if extension == "json" {
                let receipt: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(receipt["schema"], "gpms.ladder_receipt.v1");
                assert_eq!(receipt["rung"], format!("{name}-scalar-v2"));
                assert_eq!(receipt["decision"], card["decision"]);
                assert_eq!(receipt["promotion_authorized"], false);
                assert_eq!(
                    receipt["measurement_source_sha"],
                    document["measurement_source_sha"]
                );
                assert_eq!(receipt["gates"]["G-B"], "pass");
                assert_eq!(receipt["compiled_parity"], "PASS");
                assert_eq!(receipt["sanitizer_qualification"], "PASS");
                assert_eq!(receipt["decline_reasons"], card["decline_reasons"]);
                for gate in ["G-A", "G-C"] {
                    assert_eq!(receipt["gates"][gate], "fail");
                    assert!(receipt["decline_reasons"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|reason| reason["gate"] == gate
                            && reason["reason"].as_str().is_some_and(|v| !v.is_empty())));
                }
                assert_eq!(receipt["caps"]["max_nodes"], 15);
                assert_eq!(receipt["caps"]["max_depth"], 4);
                assert_eq!(receipt["budget_evals"], 72);
                assert_eq!(receipt["datasets"].as_array().unwrap().len(), 8);
                assert_eq!(receipt["seeds"], serde_json::json!([3, 17, 29, 101, 701]));
                assert_eq!(receipt["native_evidence"]["unknown_internal_passes"], 0);
            }
        }
    }
    assert_eq!(names, BTreeSet::from(["B05", "B06", "B07"]));
}
