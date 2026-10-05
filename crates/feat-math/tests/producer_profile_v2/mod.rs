use feat_math::kernel::scalar_program::{ScalarCarrier, ScalarProducerKind, ScalarProducerProfile};
use serde_json::{json, Value};

#[test]
fn producer_controls_are_canonical_bound_and_component_isolated() {
    for kind in [
        ScalarProducerKind::InverseLinks,
        ScalarProducerKind::StableExponentials,
        ScalarProducerKind::NumericalPolicy,
        ScalarProducerKind::All,
    ] {
        for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
            let profile = ScalarProducerProfile::new(kind, carrier, 2).unwrap();
            for profile in [
                profile.clone(),
                profile.clone().generation_disabled(),
                profile.clone().inverse_links_disabled(),
                profile.clone().stable_exponentials_disabled(),
                profile.numerical_policy_disabled(),
            ] {
                let declaration = profile.canonical_json();
                let restored = ScalarProducerProfile::from_canonical(&declaration).unwrap();
                assert_eq!(restored.sha256(), profile.sha256());
                assert_eq!(restored.carrier(), carrier);
                assert_eq!(restored.input_width(), 2);
                assert_eq!(restored.base_profile().id, "kernel2-baseline");
                assert_eq!(restored.base_profile().revision, 3);
                assert_eq!(
                    restored.base_profile().sha256(),
                    profile.base_profile().sha256()
                );
                let mut forged: Value = serde_json::from_str(&declaration).unwrap();
                for (key, value) in [
                    ("schema", json!("unknown")),
                    ("revision", json!(2)),
                    ("class", json!("baseline")),
                    ("birth_stream", json!(17)),
                    ("site_stream", json!(17)),
                    ("opt_in_edit_stream", json!(17)),
                    ("opt_in_numerator", json!(2)),
                    ("opt_in_denominator", json!(5)),
                    ("function_catalog", json!([99])),
                    ("reduction_catalog", json!(["unknown"])),
                    ("policy_birth_choices", json!(["unknown"])),
                    ("admission_contract", json!("loosened")),
                    ("rejection", json!("redraw")),
                    ("base_profile", json!("{}")),
                    ("max_nodes", json!(16)),
                    ("max_depth", json!(5)),
                    ("base_profile_sha256", json!("unknown")),
                    ("unknown", json!(true)),
                ] {
                    let before = forged.clone();
                    forged[key] = value;
                    assert!(
                        ScalarProducerProfile::from_canonical(
                            &serde_json::to_string(&forged).unwrap()
                        )
                        .is_err(),
                        "{key}"
                    );
                    forged = before;
                }
                assert!(ScalarProducerProfile::from_canonical(&format!(" {declaration}")).is_err());
            }
        }
    }
    assert!(ScalarProducerProfile::new(ScalarProducerKind::All, ScalarCarrier::Row, 0).is_err());
    assert!(ScalarProducerProfile::new(ScalarProducerKind::All, ScalarCarrier::Row, 513).is_err());
}

#[test]
fn producer_declarations_preserve_all_existing_profiles_and_width_limits() {
    use feat_math::kernel::profile;
    let before: Vec<_> = profile::profiles()
        .into_iter()
        .map(|p| p.canonical_json())
        .collect();
    let shipped_before: Vec<_> = profile::shipped_kernel2_profiles()
        .into_iter()
        .map(|p| p.canonical_json())
        .collect();
    for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
        for width in [1, 512] {
            let p = ScalarProducerProfile::new(ScalarProducerKind::All, carrier, width).unwrap();
            let text = p.canonical_json();
            let restored = ScalarProducerProfile::from_canonical(&text).unwrap();
            assert_eq!(restored.input_width(), width);
            assert_eq!(restored.canonical_json(), text);
            let value: Value = serde_json::from_str(&text).unwrap();
            assert_eq!(value["class"], "experimental");
            assert_eq!(
                value["base_profile"],
                profile::profile("kernel2-baseline", 3)
                    .unwrap()
                    .canonical_json()
            );
            assert_eq!(
                value["base_profile_sha256"],
                profile::profile("kernel2-baseline", 3).unwrap().sha256()
            );
        }
    }
    assert_eq!(
        before,
        profile::profiles()
            .into_iter()
            .map(|p| p.canonical_json())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        shipped_before,
        profile::shipped_kernel2_profiles()
            .into_iter()
            .map(|p| p.canonical_json())
            .collect::<Vec<_>>()
    );
    let p = ScalarProducerProfile::new(ScalarProducerKind::InverseLinks, ScalarCarrier::Row, 2)
        .unwrap();
    let value: Value = serde_json::from_str(&p.canonical_json()).unwrap();
    for key in ["enable_stable_exponentials", "enable_numerical_policy"] {
        let mut forged = value.clone();
        forged[key] = json!(true);
        assert!(
            ScalarProducerProfile::from_canonical(&serde_json::to_string(&forged).unwrap())
                .is_err(),
            "{key}"
        );
    }
}
