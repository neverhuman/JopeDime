use super::*;

#[test]
fn exact_tensor_names_dtypes_and_buffers_are_mandatory() {
    for change in [0, 1, 2, 3] {
        let mut fixture = Fixture::new();
        let specs = fixture
            .release
            .assets
            .get_mut("jope.safetensors")
            .unwrap()
            .tensors
            .as_mut()
            .unwrap();
        match change {
            0 => {
                specs.remove("running_buffer");
            }
            1 => {
                let spec = specs.remove("running_buffer").unwrap();
                specs.insert("other_buffer".into(), spec);
            }
            2 => {
                specs.get_mut("projection.weight").unwrap().dtype = Dtype::F16;
            }
            _ => {
                fixture
                    .release
                    .assets
                    .get_mut("jope.safetensors")
                    .unwrap()
                    .tensors = None;
            }
        }
        let error = fixture.load(load_limits()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::ModelIntegrityFailed);
        assert_eq!(error.exit_code(), 3);
    }
}

#[test]
fn signature_hash_key_footer_and_high_s_are_mandatory() {
    let mut fixture = Fixture::new();
    fixture.signature[0] ^= 1;
    refuses(fixture.load(load_limits()), "signature");
    fixture.seal();
    fixture.signed_manifest[0] ^= 1;
    refuses(fixture.load(load_limits()), "catalog SHA-256 mismatch");
    fixture.seal();
    fixture.footer[64] ^= 1;
    refuses(fixture.load(load_limits()), "untrusted package signer");
    fixture.seal();
    fixture.footer[60] = 2;
    refuses(fixture.load(load_limits()), "footer");
    fixture.seal();
    let low = Signature::from_slice(&fixture.signature).unwrap();
    let (r, s) = low.split_scalars();
    fixture.signature = Signature::from_scalars(r, -*s).unwrap().to_bytes().into();
    refuses(
        fixture.load(load_limits()),
        "noncanonical package signature",
    );
}

#[test]
fn unknown_architecture_and_changed_release_pins_are_refused() {
    let mut fixture = Fixture::new();
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fixture.files["manifest.json"]).unwrap();
    manifest["architecture"] = "unknown-architecture".into();
    fixture.files.insert(
        "manifest.json".into(),
        serde_json::to_vec(&manifest).unwrap(),
    );
    fixture.seal();
    refuses(fixture.load(load_limits()), "architecture");
    let mut fixture = Fixture::new();
    fixture.release.bundle_digest = BundleDigest([7; 32]);
    refuses(fixture.load(load_limits()), "release identities");
    fixture.release.dope_logical_composite[0] ^= 1;
    refuses(fixture.load(load_limits()), "frozen DOPE");
    let mut fixture = Fixture::new();
    fixture.release.dope_weights_sha256[0] ^= 1;
    refuses(fixture.load(load_limits()), "custody pin");
}
