use super::*;

#[test]
fn float_masks_refuse_nan_inf_in_every_supported_width() {
    for (dtype, finite, bad) in [
        (
            Dtype::F64,
            1.0f64.to_le_bytes().to_vec(),
            f64::NAN.to_le_bytes().to_vec(),
        ),
        (
            Dtype::F32,
            1.0f32.to_le_bytes().to_vec(),
            f32::INFINITY.to_le_bytes().to_vec(),
        ),
        (
            Dtype::F16,
            0x3c00u16.to_le_bytes().to_vec(),
            0x7e00u16.to_le_bytes().to_vec(),
        ),
        (
            Dtype::BF16,
            0x3f80u16.to_le_bytes().to_vec(),
            0xff80u16.to_le_bytes().to_vec(),
        ),
    ] {
        check_finite(&finite, dtype).unwrap();
        assert_eq!(check_finite(&bad, dtype).unwrap_err().exit_code(), 3);
    }
    assert!(check_finite(&[0xff; 8], Dtype::I64).is_ok());
    assert!(check_finite(&[0], Dtype::F8_E5M2).is_err());
}

#[test]
fn missing_extra_duplicate_or_escaping_assets_are_refused() {
    let fixture = Fixture::new();
    let mut embedded = fixture.embedded();
    embedded.pop();
    refuses(
        load_with_keys(
            BundleInput::Embedded(&embedded),
            fixture.context(),
            &fixture.release,
            load_limits(),
            &[*fixture_key().verifying_key()],
        ),
        "inventory",
    );
    for path in [
        fixture.files.first_key_value().unwrap().0.as_str(),
        "unlisted.json",
    ] {
        let mut embedded = fixture.embedded();
        embedded.push(EmbeddedAsset { path, bytes: b"{}" });
        refuses(
            load_with_keys(
                BundleInput::Embedded(&embedded),
                fixture.context(),
                &fixture.release,
                load_limits(),
                &[*fixture_key().verifying_key()],
            ),
            "inventory",
        );
    }
    let mut fixture = Fixture::new();
    let spec = fixture.release.assets.remove("featurizer.json").unwrap();
    fixture.release.assets.insert("../outside".into(), spec);
    refuses(fixture.load(load_limits()), "required asset");
    for path in [
        "../outside",
        "a//b",
        "a/./b",
        "a/../b",
        "/absolute",
        "a\\b",
        "a:b",
    ] {
        assert!(!safe_path(path));
    }
}

#[cfg(unix)]
#[test]
fn directory_refuses_symlinks_and_fifo_without_a_writer_promptly() {
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new();
    let directory = fixture.directory();
    let path = directory.path().join("featurizer.json");
    std::fs::remove_file(&path).unwrap();
    symlink("manifest.json", &path).unwrap();
    refuses(fixture.load_directory(directory.path()), "without links");
    std::fs::remove_file(&path).unwrap();
    nix::unistd::mkfifo(
        &path,
        nix::sys::stat::Mode::S_IRUSR | nix::sys::stat::Mode::S_IWUSR,
    )
    .unwrap();
    let start = std::time::Instant::now();
    refuses(fixture.load_directory(directory.path()), "nonregular");
    assert!(start.elapsed() < std::time::Duration::from_secs(1));
    let linked = directory.path().join("root-link");
    symlink(directory.path().canonicalize().unwrap(), &linked).unwrap();
    assert!(DirectoryReader::open(&linked).is_err());
    let golden = directory.path().join("golden");
    std::fs::rename(&golden, directory.path().join("saved-golden")).unwrap();
    symlink("saved-golden", &golden).unwrap();
    // Restore the earlier file so traversal reaches the intermediate link.
    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path, &fixture.files["featurizer.json"]).unwrap();
    refuses(fixture.load_directory(directory.path()), "without links");
}
