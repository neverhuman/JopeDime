use super::*;

pub(super) fn generic_csv_upload_preserves_name_key_and_bytes() {
    let transport = UploadCapture::default();
    upload_csv(
        &transport,
        "input.csv",
        Cursor::new(b"x,y\n1,2\n"),
        "retry-key",
    )
    .unwrap();
    assert_eq!(
        transport.0.lock().unwrap().as_ref().unwrap(),
        &(
            "input.csv".into(),
            "retry-key".into(),
            b"x,y\n1,2\n".to_vec()
        )
    );
}

pub(super) fn atomic_csv_write_publishes_complete_file_and_refuses_overwrite() {
    let temp = temp_dir();
    let out = temp.join("prediction.csv");
    assert_eq!(
        write_csv_atomic(&out, Cursor::new(b"prediction\n0.25\n".to_vec())).unwrap(),
        16
    );
    assert_eq!(fs::read(&out).unwrap(), b"prediction\n0.25\n");
    assert!(matches!(
        write_csv_atomic(&out, Cursor::new(b"replacement".to_vec())),
        Err(FileError::OutputExists)
    ));
    assert_eq!(fs::read(&out).unwrap(), b"prediction\n0.25\n");
    assert_eq!(fs::read_dir(&*temp).unwrap().count(), 1);
}

pub(super) fn atomic_csv_retry_verifies_existing_bytes_without_replacing_the_file() {
    let temp = temp_dir();
    let out = temp.join("prediction.csv");
    let csv = b"prediction\n0.25\n0.5\n";
    write_csv_atomic(&out, Cursor::new(csv)).unwrap();
    // Keeping this handle open also proves that a successful replay does not
    // replace the original file with a newly staged one.
    let mut original = File::open(&out).unwrap();
    let before = original.metadata().unwrap();
    assert_eq!(
        write_csv_atomic(&out, Cursor::new(csv)).unwrap(),
        csv.len() as u64
    );
    assert_eq!(
        fs::metadata(&out).unwrap().modified().unwrap(),
        before.modified().unwrap()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(fs::metadata(&out).unwrap().ino(), before.ino());
    }
    let mut retained = Vec::new();
    original.read_to_end(&mut retained).unwrap();
    assert_eq!(retained, csv);
    for different in [
        &b"prediction\n0.25\n0.6\n"[..],
        &csv[..csv.len() - 1],
        b"prediction\n0.25\n0.5\n0\n",
    ] {
        assert!(matches!(
            write_csv_atomic(&out, Cursor::new(different)),
            Err(FileError::OutputExists)
        ));
        assert_eq!(fs::read(&out).unwrap(), csv);
    }
    assert_eq!(fs::read_dir(&*temp).unwrap().count(), 1);
}

pub(super) fn atomic_csv_retry_refuses_directories_and_symlinks_even_to_identical_bytes() {
    let temp = temp_dir();
    let out = temp.join("prediction.csv");
    fs::create_dir(&out).unwrap();
    assert!(matches!(
        write_csv_atomic(&out, Cursor::new(b"prediction\n0.25\n")),
        Err(FileError::OutputExists)
    ));
    assert!(out.is_dir());
    #[cfg(unix)]
    {
        let original = temp.join("original.csv");
        fs::write(&original, b"prediction\n0.25\n").unwrap();
        let link = temp.join("linked.csv");
        std::os::unix::fs::symlink(&original, &link).unwrap();
        assert!(matches!(
            write_csv_atomic(&link, Cursor::new(b"prediction\n0.25\n")),
            Err(FileError::OutputExists)
        ));
        assert!(fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read(&original).unwrap(), b"prediction\n0.25\n");
    }
}

struct FailsAfterPrefix(bool);
impl Read for FailsAfterPrefix {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if !self.0 {
            self.0 = true;
            output[..7].copy_from_slice(b"partial");
            return Ok(7);
        }
        Err(io::Error::other("fixture read failure"))
    }
}

pub(super) fn atomic_csv_write_removes_partial_temporary_after_read_error() {
    let temp = temp_dir();
    let out = temp.join("prediction.csv");
    assert!(write_csv_atomic(&out, FailsAfterPrefix(false)).is_err());
    assert!(!out.exists());
    assert_eq!(fs::read_dir(&*temp).unwrap().count(), 0);
}

pub(super) fn atomic_csv_failed_retry_preserves_existing_file_and_removes_staging() {
    let temp = temp_dir();
    let out = temp.join("prediction.csv");
    fs::write(&out, b"prediction\n0.25\n").unwrap();
    assert!(matches!(
        write_csv_atomic(&out, FailsAfterPrefix(false)),
        Err(FileError::Io(_))
    ));
    assert_eq!(fs::read(&out).unwrap(), b"prediction\n0.25\n");
    assert_eq!(fs::read_dir(&*temp).unwrap().count(), 1);
}

pub(super) fn atomic_csv_publication_verifies_a_concurrent_creator_without_overwriting() {
    struct PublishDuringRead<'a> {
        out: &'a Path,
        concurrent: Option<&'a [u8]>,
        response: Cursor<&'a [u8]>,
    }
    impl Read for PublishDuringRead<'_> {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            if let Some(concurrent) = self.concurrent.take() {
                fs::write(self.out, concurrent)?;
            }
            self.response.read(bytes)
        }
    }
    let csv = b"prediction\n0.25\n";
    for concurrent in [&csv[..], &b"prediction\n0.75\n"[..]] {
        let temp = temp_dir();
        let out = temp.join("prediction.csv");
        let result = write_csv_atomic(
            &out,
            PublishDuringRead {
                out: &out,
                concurrent: Some(concurrent),
                response: Cursor::new(csv),
            },
        );
        if concurrent == csv {
            assert_eq!(result.unwrap(), csv.len() as u64);
        } else {
            assert!(matches!(result, Err(FileError::OutputExists)));
        }
        assert_eq!(fs::read(&out).unwrap(), concurrent);
        assert_eq!(fs::read_dir(&*temp).unwrap().count(), 1);
    }
}

pub(super) fn csv_upload_streams_to_dataset_route_with_auth_and_idempotency() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut input = BufReader::new(socket.try_clone().unwrap());
        let mut headers = String::new();
        loop {
            let mut line = String::new();
            input.read_line(&mut line).unwrap();
            headers.push_str(&line);
            if line == "\r\n" {
                break;
            }
        }
        let mut body = Vec::new();
        loop {
            let mut size = String::new();
            input.read_line(&mut size).unwrap();
            let size = usize::from_str_radix(size.trim(), 16).unwrap();
            if size == 0 {
                let mut trailer = String::new();
                input.read_line(&mut trailer).unwrap();
                break;
            }
            let start = body.len();
            body.resize(start + size, 0);
            input.read_exact(&mut body[start..]).unwrap();
            let mut crlf = [0; 2];
            input.read_exact(&mut crlf).unwrap();
            assert_eq!(&crlf, b"\r\n");
        }
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .unwrap();
        (headers.to_ascii_lowercase(), body)
    });
    let http = Http::new(&endpoint, &"ab".repeat(32)).unwrap();
    upload_csv(
        &http,
        "train data.csv",
        Cursor::new(b"x,y\n1,2\n".to_vec()),
        "upload-1",
    )
    .unwrap();
    let (headers, body) = server.join().unwrap();
    assert!(headers.starts_with("post /v1/datasets?name=train%20data.csv http/1.1\r\n"));
    assert!(headers.contains("content-type: text/csv\r\n"));
    assert!(headers.contains("idempotency-key: upload-1\r\n"));
    assert!(headers.contains(&format!("authorization: bearer {}\r\n", "ab".repeat(32))));
    assert_eq!(body, b"x,y\n1,2\n");
}
