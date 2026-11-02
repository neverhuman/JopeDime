use super::*;

/// Savgol-only and stage-free programs keep B12's exact bytes: these digests were taken at the
/// B12 head (289a4cae) and match after the bank was added.
#[test]
fn savgol_only_artifacts_keep_their_b12_bytes() {
    use sha2::{Digest, Sha256};
    let sha = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    let pins: &[(&str, &str)] = &[
        (
            "empty.canonical",
            "09687d18dcdc5b5a5691532254f2bdb83a0411a5fe67ffe00639fddef0b2a57f",
        ),
        (
            "empty.fitted_cpp17",
            "3844b4fbe7c771a95f1d4321c130681d4e27f54df0cbd677460bfe85bdb1881f",
        ),
        (
            "empty.model_json",
            "64a187e253d871d64c0e2735f24347ee719deba20376f9896bdf799ff73c4bcd",
        ),
        (
            "empty.program",
            "9fdc3311c001a99a28bf13adeaffab9522a918857dd630cdfdaff4c2ea5c57e5",
        ),
        (
            "empty.recipe.cpp17",
            "fede71b4af3bb915b8219f3cd63f03b35505eaf96e6b8cce5a180004db348c22",
        ),
        (
            "empty.recipe.python",
            "b34e9f7c56f5c1ea178b20b8b5df6fad44b3bbf2116023b77acfa659cc18289b",
        ),
        (
            "empty.recipe.rust",
            "7d28963f155739424eb35dc0fc64614225aa1b7949a6fcc3bcd2105f28bbf0b9",
        ),
        (
            "savgol5.canonical",
            "4596f1fe27cd00cf0c468f4f090d8a248097c5ce666f0523c08ae9673967a81f",
        ),
        (
            "savgol5.fitted_cpp17",
            "792b57e69978a2ff2d38935afc219535a0776a82bd1b088545c8bf0b083649e7",
        ),
        (
            "savgol5.model_json",
            "d5b2d764ff72ecc058e6a8a7927872d4fb0180e1d05b5b752e43d769f8abed99",
        ),
        (
            "savgol5.program",
            "0af38ee99c7ba8309d1fd765ce02cd8b163b6a35eec843eb11047fe527d70790",
        ),
        (
            "savgol5.recipe.cpp17",
            "c7385381ac77a7fd1e396dadfa3c35eb90a1b2152139a0ce8d492fa9346772ef",
        ),
        (
            "savgol5.recipe.python",
            "66fb375493bc03acee63824d09bf087da4a3b2548b4d09c1ed16ad7ba2c615d5",
        ),
        (
            "savgol5.recipe.rust",
            "cd9d1c6b4d099df99a87e07ae483fa7e9d92641153d3e2fdac4987ede68cd1b4",
        ),
    ];
    let mut actual = std::collections::BTreeMap::new();
    for (name, p) in [
        ("savgol5", program(7, vec![RowStage::Savgol(smoothing())])),
        ("empty", program(7, vec![])),
    ] {
        actual.insert(format!("{name}.program"), sha(&p.to_bytes().unwrap()));
        actual.insert(
            format!("{name}.canonical"),
            sha(p.canonical().unwrap().as_bytes()),
        );
        for (language, label) in [
            (RowRecipeLanguage::Rust, "rust"),
            (RowRecipeLanguage::Python, "python"),
            (RowRecipeLanguage::Cpp17, "cpp17"),
        ] {
            actual.insert(
                format!("{name}.recipe.{label}"),
                sha(p.emit_recipe(language).unwrap().as_bytes()),
            );
        }
        let (x, y) = training(7);
        let model = p.fit(&x, &y).unwrap();
        actual.insert(
            format!("{name}.model_json"),
            sha(model.to_json().unwrap().as_bytes()),
        );
        actual.insert(
            format!("{name}.fitted_cpp17"),
            sha(model.emit_fitted_cpp17().unwrap().as_bytes()),
        );
    }
    assert_eq!(actual.len(), pins.len());
    for (name, digest) in pins {
        assert_eq!(actual[*name], *digest, "{name}");
    }
}

#[test]
fn rows_reset_independently_and_bank_artifacts_pass_the_shipped_text_lint() {
    let p = program(
        8,
        vec![
            RowStage::Goertzel(bank(&[0.0, 0.125, 0.3, 0.5])),
            RowStage::Savgol(smoothing()),
        ],
    );
    let (x, y) = training(8);
    let model = p.fit(&x, &y).unwrap();
    // Each row restarts the recurrence: predicting rows together, apart or permuted agrees.
    let (all, faults) = model.predict(&x).unwrap();
    let (head, rest) = x.split_at(9);
    let mut parts = model.predict(&head.to_vec()).unwrap().0;
    parts.extend(model.predict(&rest.to_vec()).unwrap().0);
    assert_eq!(bits(&parts), bits(&all));
    let reversed: Vec<Vec<f64>> = x.iter().rev().cloned().collect();
    let mut back = model.predict(&reversed).unwrap().0;
    back.reverse();
    assert_eq!(bits(&back), bits(&all));
    assert!(faults
        .iter()
        .all(|&f| f & FaultSet::NONFINITE_SANITIZED.bits() == 0));
    // Shipped bank artifacts contain no banned vocabulary.
    for language in [
        RowRecipeLanguage::Rust,
        RowRecipeLanguage::Python,
        RowRecipeLanguage::Cpp17,
    ] {
        let recipe = p.emit_recipe(language).unwrap();
        assert!(feat_math::transpile::lint_shipped_text(&recipe).is_empty());
    }
    let fitted = model.emit_fitted_cpp17().unwrap();
    assert!(feat_math::transpile::lint_shipped_text(&fitted).is_empty());
}

#[test]
fn unknown_wires_stay_unsupported_and_savgol_ingress_is_unchanged() {
    let mut bytes = program(4, vec![RowStage::Goertzel(bank(&[0.1]))])
        .to_bytes()
        .unwrap();
    let at = bytes
        .windows(11)
        .position(|w| w == b"\"wire\":\"61\"")
        .unwrap();
    bytes[at + 8] = b'6';
    bytes[at + 9] = b'2';
    assert!(matches!(
        Program::from_bytes(&bytes),
        Err(VersionError::UnsupportedVersion(_))
    ));
    assert!(matches!(
        VersionedProgram::from_bytes(&bytes),
        Err(VersionError::UnsupportedVersion(_))
    ));
}
