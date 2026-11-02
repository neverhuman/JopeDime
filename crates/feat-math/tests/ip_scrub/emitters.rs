use super::*;

#[test]
fn every_emitter_is_clean_on_every_fixture() {
    let mut emitted = 0;
    let (mut emitted_cpp, mut emitted_js) = (0, 0);
    for src in every_fixture() {
        let genome = GenomeIr::parse(&src).unwrap_or_else(|e| panic!("{src}: {e}"));
        for opts in option_sets() {
            let artifact = emit(&genome, &opts).unwrap_or_else(|e| panic!("{src}: {e}"));
            assert_clean(&format!("python of {src}"), &artifact.python);
            assert_clean(&format!("rust of {src}"), &artifact.rust);
            emitted += 2;

            // Manifest: every key, and every value except the program text itself.
            let manifest: serde_json::Value = serde_json::from_str(&artifact.manifest).unwrap();
            let object = manifest.as_object().expect("manifest is an object");
            for (key, value) in object {
                if key != "dsl" && key != "genome_sha256" {
                    assert_clean(&format!("manifest key {key:?} of {src}"), key);
                }
                if key != "program" && key != "dsl" {
                    assert_clean(
                        &format!("manifest value {key:?} of {src}"),
                        &value.to_string(),
                    );
                }
            }
            assert_eq!(
                object["program"], object["dsl"],
                "{src}: program mirrors dsl"
            );
            assert_eq!(
                object["program_sha256"], object["genome_sha256"],
                "{src}: program_sha256 mirrors genome_sha256"
            );
            assert_eq!(object["program"].as_str().unwrap(), genome.canonical());

            if supports_cpp17(&genome).is_ok() {
                let cpp = emit_cpp17(&genome, &opts).unwrap_or_else(|e| panic!("{src}: {e}"));
                assert_clean(&format!("c++17 of {src}"), &cpp);
                emitted_cpp += 1;
            }
            if supports_javascript(&genome).is_ok() {
                let js = emit_javascript(&genome, &opts).unwrap_or_else(|e| panic!("{src}: {e}"));
                assert_clean(&format!("javascript of {src}"), &js);
                emitted_js += 1;
            }
        }
    }
    // Python + Rust on every fixture under both option sets, plus every C++ and JavaScript
    // artifact the gates admit: the four-language fixture alone yields two of each.
    assert!(
        emitted >= 2 * 2 * 20,
        "only {emitted} Python/Rust artifacts were checked"
    );
    assert!(
        emitted_cpp >= 2,
        "only {emitted_cpp} C++17 artifacts were checked"
    );
    assert!(
        emitted_js >= 2,
        "only {emitted_js} JavaScript artifacts were checked"
    );
    assert_eq!(EMITTER_VERSION, "2.2.0");
}

/// Numeric schema `c0..` and a shortest-round-trip string table, as the campaign lanes build them.
#[test]
fn bound_and_fitted_cpp_emitters_are_clean() {
    let mut rng = feat_math::Pcg32::new(17, 3);
    let mut x = Vec::new();
    let mut y_regression = Vec::new();
    let mut y_binary = Vec::new();
    for _ in 0..160 {
        let a = rng.next_f64();
        let b = rng.next_f64();
        let c = rng.next_f64();
        x.push(vec![a, b, c]);
        y_regression.push(2.0 * a - b + 0.3 * c);
        y_binary.push(if 3.0 * (a - b) + (c - 0.5) > 0.0 {
            1.0
        } else {
            0.0
        });
    }
    let (schema, table) = numeric_table(&x);
    let regression = TaskSpec::regression(Objective::R2).unwrap();
    let binary = TaskSpec::binary(Objective::RocAuc).unwrap();
    let cases = [
        ("(pipe X (ridge))", &regression, &y_regression),
        (
            "(pipe X (standardize) (tab_boost))",
            &regression,
            &y_regression,
        ),
        ("(pipe X (logit_ridge))", &binary, &y_binary),
        ("(pipe X (tab_boost))", &binary, &y_binary),
        ("(pipe X (extra_trees))", &binary, &y_binary),
        ("(pipe X (gam_boost))", &binary, &y_binary),
        ("(pipe X (portfolio))", &binary, &y_binary),
    ];
    let (mut bound, mut fitted) = (0, 0);
    for (src, task, y) in cases {
        let genome = GenomeIr::parse(src).unwrap();
        let model =
            fit_table(&genome, task, &schema, &table, y).unwrap_or_else(|e| panic!("{src}: {e:#}"));
        let artifact = emit_bound(&model, &EmitOptions::default())
            .unwrap_or_else(|e| panic!("bound {src}: {e:#}"));
        assert_clean(&format!("bound python of {src}"), &artifact.python);
        assert_clean(&format!("bound rust of {src}"), &artifact.rust);
        assert_clean(&format!("bound manifest of {src}"), &artifact.manifest);
        let manifest: serde_json::Value = serde_json::from_str(&artifact.manifest).unwrap();
        assert_eq!(manifest["emitter_version"], EMITTER_VERSION);
        assert!(
            manifest.get("program_sha256").is_some(),
            "{src}: manifest names the program hash"
        );
        bound += 1;
        match emit_fitted_cpp17(&model) {
            Ok(cpp) => {
                assert_clean(&format!("fitted c++17 of {src}"), &cpp.source);
                assert_clean(&format!("fitted c++17 codec of {src}"), &cpp.codec);
                fitted += 1;
            }
            Err(error) => eprintln!("fitted C++17 not supported for {src}: {error:#}"),
        }
    }
    assert_eq!(bound, 7);
    assert_eq!(
        fitted, 7,
        "every one of the seven heads emits fitted C++17 today; a head losing it is a change to review"
    );
}
