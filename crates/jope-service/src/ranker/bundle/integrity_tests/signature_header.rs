use super::*;

#[test]
fn signed_embedded_and_development_directory_load_exact_bytes() {
    let fixture = Fixture::new();
    let embedded = fixture.load(load_limits()).unwrap();
    assert_eq!(embedded.digest(), fixture.release.bundle_digest);
    assert_eq!(
        embedded.declared_qualifications()[0].task,
        QualifiedTask::Regression
    );
    assert_eq!(
        embedded.asset("dope.safetensors").unwrap(),
        fixture.files["dope.safetensors"]
    );
    let directory = fixture.directory();
    let loaded = fixture.load_directory(directory.path()).unwrap();
    assert_eq!(loaded.digest(), embedded.digest());
    for (path, bytes) in &fixture.files {
        assert_eq!(loaded.asset(path).unwrap(), bytes);
    }
    // The production API cannot opt into the fixture signer.
    refuses(
        load_bundle(
            BundleInput::Embedded(&fixture.embedded()),
            fixture.context(),
            &fixture.release,
            load_limits(),
        ),
        "untrusted package signer",
    );
}

#[test]
fn m4_wrong_hash_is_refused_with_integrity_exit3() {
    let mut fixture = Fixture::new();
    fixture.files.get_mut("jope.safetensors").unwrap()[8] ^= 1;
    refuses(fixture.load(load_limits()), "SHA-256 mismatch");
}

#[test]
fn m4_signed_corrupt_header_is_refused_with_integrity_exit3() {
    let mut fixture = Fixture::new();
    fixture.files.get_mut("jope.safetensors").unwrap()[..8]
        .copy_from_slice(&u64::MAX.to_le_bytes());
    fixture.seal();
    refuses(fixture.load(load_limits()), "header exceeds");
}

#[test]
fn signed_duplicate_tensor_header_keys_are_refused() {
    let mut header = br#"{"projection.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"running_buffer":{"dtype":"F64","shape":[],"data_offsets":[8,16]},"projection.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#.to_vec();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(1.0f32.to_le_bytes());
    bytes.extend(2.0f32.to_le_bytes());
    bytes.extend(0.5f64.to_le_bytes());
    // The library's map projection alone collapses the duplicate name.
    assert!(SafeTensors::deserialize(&bytes).is_ok());
    let mut fixture = Fixture::new();
    fixture.files.insert("jope.safetensors".into(), bytes);
    fixture.seal();
    refuses(fixture.load(load_limits()), "duplicate");
}

#[test]
fn m4_signed_nan_weight_and_infinite_buffer_are_refused_with_integrity_exit3() {
    for (values, buffer) in [
        (vec![f32::NAN, 1.0], 0.5),
        (vec![1.0, 2.0], f64::INFINITY),
        (vec![f32::NEG_INFINITY, 2.0], 0.5),
    ] {
        let mut fixture = Fixture::new();
        fixture
            .files
            .insert("jope.safetensors".into(), weights(&values, buffer));
        fixture.seal();
        refuses(fixture.load(load_limits()), "nonfinite");
    }
}

#[test]
fn m4_signed_wrong_shape_and_oversize_are_refused_with_integrity_exit3() {
    let mut fixture = Fixture::new();
    fixture
        .files
        .insert("jope.safetensors".into(), weights(&[1.0], 0.5));
    fixture.seal();
    refuses(fixture.load(load_limits()), "shape or dtype mismatch");
    let fixture = Fixture::new();
    refuses(
        fixture.load(BundleLimits {
            file_bytes: 16,
            ..load_limits()
        }),
        "file byte cap",
    );
    refuses(
        fixture.load(BundleLimits {
            total_bytes: 16,
            ..load_limits()
        }),
        "total byte cap",
    );
}
