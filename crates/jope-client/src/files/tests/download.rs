use super::*;

pub(super) fn digest_format_must_be_exact_lowercase_sha256() {
    assert!(matches!(
        parse_digest("sha256:ABC"),
        Err(FileError::InvalidDigest)
    ));
    assert!(matches!(
        parse_digest(&format!("sha256:{}", "A".repeat(64))),
        Err(FileError::InvalidDigest)
    ));
}

pub(super) fn export_retry_checks_every_existing_byte_and_refuses_tampering() {
    let temp = temp_dir();
    let out = temp.join("export");
    fs::create_dir(&out).unwrap(); // Existing empty directories are valid destinations.
    let transport = Fake {
        bytes: zip(&[("model/data", b"frozen"), ("manifest.json", b"manifest")]),
    };
    download_export_artifact(&transport, &artifact_id(), &out, false).unwrap();
    download_export_artifact(&transport, &artifact_id(), &out, false).unwrap();
    fs::write(out.join("model/data"), b"edited").unwrap();
    assert!(matches!(
        download_export_artifact(&transport, &artifact_id(), &out, false),
        Err(FileError::OutputExists)
    ));
    assert_eq!(fs::read(out.join("model/data")).unwrap(), b"edited");
    fs::write(out.join("model/data"), b"frozen").unwrap();
    fs::write(out.join("extra"), b"keep").unwrap();
    assert!(matches!(
        download_export_artifact(&transport, &artifact_id(), &out, false),
        Err(FileError::OutputExists)
    ));
    assert_eq!(fs::read(out.join("extra")).unwrap(), b"keep");
    #[cfg(unix)]
    {
        fs::remove_file(out.join("extra")).unwrap();
        fs::remove_file(out.join("model/data")).unwrap();
        std::os::unix::fs::symlink("../manifest.json", out.join("model/data")).unwrap();
        assert!(matches!(
            download_export_artifact(&transport, &artifact_id(), &out, false),
            Err(FileError::OutputExists)
        ));
        assert!(fs::symlink_metadata(out.join("model/data"))
            .unwrap()
            .file_type()
            .is_symlink());
    }
}

pub(super) fn download_collision_preserves_existing_directory() {
    const CHILD: &str = "JOPE_CLIENT_DOWNLOAD_COLLISION_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // A fresh process isolates the staging counter from parallel tests.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "files::tests::download_collision_preserves_existing_directory",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "collision child failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let root = temp_dir();
    let existing = root.join(format!(".out.download-{}-0", std::process::id()));
    fs::create_dir(&existing).unwrap();
    fs::write(existing.join("keep"), b"existing data").unwrap();
    let bytes = zip(&[("new", b"downloaded data")]);
    let result = download_artifact(
        &Fake {
            bytes: bytes.clone(),
        },
        &artifact_id(),
        &digest(&bytes),
        &root.join("out"),
        false,
    );
    assert!(
        matches!(result, Err(FileError::Io(error)) if error.kind() == io::ErrorKind::AlreadyExists)
    );
    assert_eq!(
        fs::read(existing.join("keep")).ok().as_deref(),
        Some(b"existing data".as_slice()),
        "exclusive-create failure removed a pre-existing directory"
    );
    fs::remove_dir_all(root).unwrap();
}

pub(super) fn download_archive_is_cleaned_after_success_and_publish_refusal() {
    let root = temp_dir();
    let out = root.join("out");
    let bytes = zip(&[("new", b"downloaded data")]);
    let transport = Fake {
        bytes: bytes.clone(),
    };
    let expected = digest(&bytes);
    download_artifact(&transport, &artifact_id(), &expected, &out, false).unwrap();
    let download_archives = || {
        fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .filter(|name| name.to_string_lossy().contains(".download-"))
            .count()
    };
    assert_eq!(download_archives(), 0, "successful download leaked its ZIP");
    assert!(matches!(
        download_artifact(&transport, &artifact_id(), &expected, &out, false),
        Err(FileError::OutputExists)
    ));
    assert_eq!(download_archives(), 0, "refused publication leaked its ZIP");
    assert_eq!(fs::read(out.join("new")).unwrap(), b"downloaded data");
    fs::remove_dir_all(root).unwrap();
}

pub(super) fn verified_zip_is_extracted_and_force_backs_up_existing_output() {
    let root = temp_dir();
    let out = root.join("out");
    fs::create_dir(&out).unwrap();
    fs::write(out.join("old"), "old").unwrap();
    let bytes = zip(&[("nested/a.txt", b"new")]);
    download_artifact(
        &Fake {
            bytes: bytes.clone(),
        },
        &artifact_id(),
        &digest(&bytes),
        &out,
        true,
    )
    .unwrap();
    assert_eq!(fs::read(out.join("nested/a.txt")).unwrap(), b"new");
    let backup = fs::read_dir(&root)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name())
        .find(|n| n.to_string_lossy().contains(".bak-"))
        .unwrap();
    assert_eq!(fs::read(root.join(backup).join("old")).unwrap(), b"old");
    fs::remove_dir_all(root).unwrap();
}

pub(super) fn mismatch_existing_output_and_interrupted_archive_leave_output_untouched() {
    let root = temp_dir();
    let out = root.join("out");
    fs::create_dir(&out).unwrap();
    fs::write(out.join("keep"), "same").unwrap();
    let bytes = zip(&[("new", b"data")]);
    assert!(matches!(
        download_artifact(
            &Fake {
                bytes: bytes.clone()
            },
            &artifact_id(),
            &digest(b"wrong"),
            &out,
            true
        ),
        Err(FileError::DigestMismatch)
    ));
    assert_eq!(fs::read(out.join("keep")).unwrap(), b"same");
    assert!(matches!(
        download_artifact(
            &Fake {
                bytes: bytes.clone()
            },
            &artifact_id(),
            &digest(&bytes),
            &out,
            false
        ),
        Err(FileError::OutputExists)
    ));
    assert_eq!(fs::read(out.join("keep")).unwrap(), b"same");
    let broken = b"not a zip".to_vec();
    assert!(download_artifact(
        &Fake {
            bytes: broken.clone()
        },
        &artifact_id(),
        &digest(&broken),
        &out,
        true
    )
    .is_err());
    assert_eq!(fs::read(out.join("keep")).unwrap(), b"same");
    fs::remove_dir_all(root).unwrap();
}

pub(super) fn refuses_parent_traversal_and_symlink_entries() {
    let root = temp_dir();
    let dest = root.join("stage");
    fs::create_dir(&dest).unwrap();
    assert!(matches!(
        extract_zip(Cursor::new(zip(&[("../escape", b"x")])), &dest),
        Err(FileError::UnsafeArchive(_))
    ));
    assert!(matches!(
        extract_zip(Cursor::new(zip(&[("C:/escape", b"x")])), &dest),
        Err(FileError::UnsafeArchive(_))
    ));
    let cursor = Cursor::new(Vec::new());
    let mut writer = ZipWriter::new(cursor);
    writer
        .start_file("link", SimpleFileOptions::default().unix_permissions(0o777))
        .unwrap();
    writer.write_all(b"target").unwrap();
    let mut symlink = writer.finish().unwrap().into_inner();
    let central = symlink
        .windows(4)
        .position(|bytes| bytes == b"PK\x01\x02")
        .unwrap();
    symlink[central + 38..central + 42].copy_from_slice(&(0o120777u32 << 16).to_le_bytes());
    assert!(matches!(
        extract_zip(Cursor::new(symlink), &dest),
        Err(FileError::UnsafeArchive("symlink entry"))
    ));
    fs::remove_dir_all(root).unwrap();
}
