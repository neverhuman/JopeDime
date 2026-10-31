use super::*;
use std::io::Read;
use std::sync::{Arc, Barrier};

#[test]
fn poisoned_injected_registry_does_not_panic_store_backend() {
    let _ = std::panic::catch_unwind(|| {
        let _guard = injected_registry().lock().unwrap();
        panic!("poison the test-only backend registry");
    });

    let store = Store::open("memory:///").unwrap();
    let expected = StoreLocation::parse("memory:///").unwrap().backend;
    assert_eq!(store.backend(), expected);
}

fn exercise_preconditions(store: &Store) {
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let key = "hwm/ledger/HEAD.json";
        assert!(store.get_observed(key).await.unwrap().is_none());
        store
            .put_conditional(key, b"first".to_vec(), None)
            .await
            .unwrap();
        let first = store.get_observed(key).await.unwrap().unwrap();
        let second_reader = store.get_observed(key).await.unwrap().unwrap();
        store
            .put_conditional(key, b"second".to_vec(), Some(&first))
            .await
            .unwrap();
        assert!(store
            .put_conditional(key, b"lost-update".to_vec(), Some(&second_reader))
            .await
            .is_err());
        assert!(store
            .put_conditional(key, b"late-create".to_vec(), None)
            .await
            .is_err());
        assert_eq!(store.get_bytes(key).await.unwrap().unwrap(), b"second");
        // A token read for one key cannot replace another key.
        assert!(store
            .put_conditional("other.json", b"wrong-key".to_vec(), Some(&first))
            .await
            .is_err());
        assert!(store.get_bytes("other.json").await.unwrap().is_none());
    });
}

#[test]
fn memory_pointer_versions_reject_stale_updates_and_creates() {
    exercise_preconditions(&Store::open("memory:///").unwrap());
}

#[cfg(unix)]
#[test]
fn local_pointer_versions_reject_stale_updates_and_creates() {
    let dir = tempfile::tempdir().unwrap();
    exercise_preconditions(&Store::open(dir.path().to_str().unwrap()).unwrap());
}

#[cfg(unix)]
#[test]
fn independent_local_publishers_have_one_winner() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    for exists in [false, true] {
        let key = if exists { "current.json" } else { "HEAD.json" };
        if exists {
            std::fs::write(root.join(key), b"parent").unwrap();
        }
        let barrier = Arc::new(Barrier::new(2));
        let results: Vec<_> = std::thread::scope(|scope| {
            let writers: Vec<_> = (0..2)
                .map(|rank| {
                    let barrier = barrier.clone();
                    let root = &root;
                    scope.spawn(move || {
                        let store = Store::open(root.to_str().unwrap()).unwrap();
                        let rt = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .unwrap();
                        let observed = rt.block_on(store.get_observed(key)).unwrap();
                        barrier.wait(); // Both readers have the original version before either writes.
                        let bytes = format!("writer-{rank}").into_bytes();
                        let result = rt.block_on(store.put_conditional(
                            key,
                            bytes.clone(),
                            observed.as_ref(),
                        ));
                        (bytes, result.is_ok())
                    })
                })
                .collect();
            writers
                .into_iter()
                .map(|writer| writer.join().unwrap())
                .collect()
        });
        let winners: Vec<_> = results.iter().filter(|(_, ok)| *ok).collect();
        assert_eq!(
            winners.len(),
            1,
            "exactly one publisher must succeed: {results:?}"
        );
        assert_eq!(std::fs::read(root.join(key)).unwrap(), winners[0].0);
    }
    // Locking must not create fake objects in the mirrored ledger namespace.
    assert_eq!(std::fs::read_dir(root).unwrap().count(), 2);
}

/// A loopback S3 protocol fixture, not a cloud write. It accepts an unconditional
/// overwrite just like S3: the test only passes if the client sends preconditions.
#[test]
fn s3_pointer_writes_send_server_enforced_preconditions() {
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    let done = Arc::new(AtomicBool::new(false));
    let stop = done.clone();
    let server = std::thread::spawn(move || {
        let mut value: Option<Vec<u8>> = None;
        let mut revision = 0;
        let mut writes = Vec::new();
        while !stop.load(Ordering::SeqCst) {
            let (mut stream, _) = match listener.accept() {
                Ok(pair) => pair,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    continue;
                }
                Err(error) => panic!("accept: {error}"),
            };
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let method = line.split_whitespace().next().unwrap().to_string();
            let path = line.split_whitespace().nth(1).unwrap().to_string();
            let mut headers = std::collections::BTreeMap::new();
            loop {
                line.clear();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                let (key, value) = line.trim_end().split_once(':').unwrap();
                headers.insert(key.to_ascii_lowercase(), value.trim().to_string());
            }
            let mut request = vec![
                0;
                headers
                    .get("content-length")
                    .map_or(0, |v| v.parse().unwrap())
            ];
            reader.read_exact(&mut request).unwrap();
            let mut status = "200 OK";
            let mut body = Vec::new();
            match method.as_str() {
                "GET" if path.ends_with("/other.json") => status = "404 Not Found",
                "GET" => match &value {
                    Some(bytes) => body = bytes.clone(),
                    None => status = "404 Not Found",
                },
                "PUT" => {
                    let matched = headers.get("if-match").cloned();
                    let absent = headers.get("if-none-match").cloned();
                    let conflict = (absent.as_deref() == Some("*") && value.is_some())
                        || matched
                            .as_ref()
                            .is_some_and(|tag| tag != &format!("\"{revision}\""));
                    writes.push((matched, absent, conflict));
                    if conflict {
                        status = "412 Precondition Failed";
                    } else {
                        value = Some(request);
                        revision += 1;
                    }
                }
                other => panic!("unexpected S3 fixture method {other}"),
            }
            write!(stream, "HTTP/1.1 {status}\r\nContent-Length: {}\r\nETag: \"{revision}\"\r\nLast-Modified: Fri, 25 Sep 2026 00:00:00 GMT\r\nConnection: close\r\n\r\n", body.len()).unwrap();
            stream.write_all(&body).unwrap();
        }
        (value, writes)
    });
    let store = Store {
        inner: Box::new(
            object_store::aws::AmazonS3Builder::new()
                .with_bucket_name("pointer-test")
                .with_region("us-east-2")
                .with_endpoint(endpoint)
                .with_allow_http(true)
                .with_skip_signature(true)
                .build()
                .unwrap(),
        ),
        prefix: ObjPath::default(),
        url: "s3://pointer-test".into(),
        local_root: None,
    };
    // Always stop the fixture, including after an assertion failure.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        exercise_preconditions(&store)
    }));
    done.store(true, Ordering::SeqCst);
    let (value, writes) = server.join().unwrap();
    if let Err(error) = result {
        std::panic::resume_unwind(error);
    }
    assert_eq!(value.unwrap(), b"second");
    assert_eq!(writes.len(), 4);
    assert_eq!(writes[0], (None, Some("*".into()), false));
    assert_eq!(writes[1], (Some("\"1\"".into()), None, false));
    assert_eq!(writes[2], (Some("\"1\"".into()), None, true));
    assert_eq!(writes[3], (None, Some("*".into()), true));
}

#[cfg(unix)]
#[test]
fn historical_compare_then_put_reproduces_two_successful_publishers() {
    // Reproduce the previous Mirror::put_pointer sequence against the real local
    // Store. The barrier makes the GET/PUT gap deterministic, without timing races.
    let dir = tempfile::tempdir().unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let root = dir.path();
    std::fs::write(root.join("HEAD.json"), b"parent").unwrap();
    let successful = std::thread::scope(|scope| {
        let writers: Vec<_> = (0..2)
            .map(|rank| {
                let barrier = barrier.clone();
                scope.spawn(move || {
                    let store = Store::open(root.to_str().unwrap()).unwrap();
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();
                    let current = rt.block_on(store.get_bytes("HEAD.json")).unwrap();
                    assert_eq!(current.as_deref(), Some(b"parent".as_slice()));
                    barrier.wait();
                    rt.block_on(store.put("HEAD.json", format!("writer-{rank}").into_bytes()))
                        .is_ok()
                })
            })
            .collect();
        writers
            .into_iter()
            .map(|writer| usize::from(writer.join().unwrap()))
            .sum::<usize>()
    });
    assert_eq!(
        successful, 2,
        "historical code silently accepted both publishers"
    );
    // independent_local_publishers_have_one_winner applies the same schedule to
    // the corrected API and requires exactly one committed update instead.
}

/// PR #114 review round 2: a `StoreLocation` names the backend and prefix object_store's own
/// classifier (`ObjectStoreScheme::parse`, then the prefix parse `object_store::parse_url`
/// applies) names, for every URL `Store::open_with` opens through `parse_url` and for the
/// well-formed `s3://`/`s3a://` URLs it opens on its own S3 route. The HTTPS URLs object_store
/// sends to S3 (AWS virtual-host and path-style, R2) are S3 locations at the prefix the store
/// would write under, which the W8 publishing preflight refuses at the bucket root. HTTPS URLs
/// still parse for the read path, which opens them through `parse_url` as before. Pure URL
/// parsing: no store is opened, and no credential or network is used.
#[test]
fn a_store_location_names_the_backend_and_prefix_object_store_opens() {
    use object_store::ObjectStoreScheme as S;
    let bucket = Some("example-bucket");
    let cases: [(&str, S, &str, Option<&str>); 20] = [
        ("s3://example-bucket", S::AmazonS3, "", bucket),
        (
            "s3://example-bucket/run-1/a",
            S::AmazonS3,
            "run-1/a",
            bucket,
        ),
        ("s3a://example-bucket/run-1", S::AmazonS3, "run-1", bucket),
        (
            "S3://example-bucket/runs/../run-1",
            S::AmazonS3,
            "run-1",
            bucket,
        ),
        (
            "https://example-bucket.s3.us-east-2.amazonaws.com",
            S::AmazonS3,
            "",
            bucket,
        ),
        (
            "https://example-bucket.s3.us-east-2.amazonaws.com/run/..",
            S::AmazonS3,
            "",
            bucket,
        ),
        (
            "https://example-bucket.s3.us-east-2.amazonaws.com/run?q=1#x",
            S::AmazonS3,
            "run",
            bucket,
        ),
        (
            "https://s3.us-east-2.amazonaws.com/example-bucket",
            S::AmazonS3,
            "",
            bucket,
        ),
        (
            "https://s3.us-east-2.amazonaws.com/example-bucket/run",
            S::AmazonS3,
            "run",
            bucket,
        ),
        (
            "https://account.r2.cloudflarestorage.com/example-bucket",
            S::AmazonS3,
            "",
            bucket,
        ),
        (
            "https://account.r2.cloudflarestorage.com/example-bucket/run-1",
            S::AmazonS3,
            "run-1",
            bucket,
        ),
        (
            "gs://example-bucket/run-1",
            S::GoogleCloudStorage,
            "run-1",
            None,
        ),
        ("az://container/run-1", S::MicrosoftAzure, "run-1", None),
        (
            "abfss://container@account.dfs.core.windows.net/run-1",
            S::MicrosoftAzure,
            "run-1",
            None,
        ),
        (
            "https://account.blob.core.windows.net/container/run-1",
            S::MicrosoftAzure,
            "run-1",
            None,
        ),
        (
            "https://example.com/progress/run-1",
            S::Http,
            "progress/run-1",
            None,
        ),
        (
            "http://example.com/progress/run-1",
            S::Http,
            "progress/run-1",
            None,
        ),
        (
            "file:///var/lib/jope/progress",
            S::Local,
            "var/lib/jope/progress",
            None,
        ),
        (
            "file:///var/lib/jope/run%201",
            S::Local,
            "var/lib/jope/run 1",
            None,
        ),
        ("memory:///", S::Memory, "", None),
    ];
    for (url, backend, prefix, bucket) in cases {
        let location = StoreLocation::parse(url).unwrap();
        // The oracle: object_store's classifier and parse_url's prefix, on the same URL.
        let (oracle, path) = S::parse(&Url::parse(url).unwrap()).unwrap();
        let oracle_prefix = ObjPath::parse(path).unwrap();
        assert_eq!(
            (&location.backend, &location.prefix),
            (&oracle, &oracle_prefix),
            "{url}"
        );
        assert_eq!(
            (&location.backend, location.prefix.as_ref()),
            (&backend, prefix),
            "{url}"
        );
        let route = if url.to_ascii_lowercase().starts_with("s3") {
            StoreRoute::S3
        } else {
            StoreRoute::ParseUrl
        };
        assert_eq!(location.route, route, "{url}");
        assert_eq!(
            location
                .s3
                .as_ref()
                .map(|s3| (s3.bucket.as_str(), s3.prefix.as_ref())),
            bucket.map(|bucket| (bucket, prefix)),
            "{url}"
        );
    }
    // A bare path is a local store at its absolute path.
    let bare = StoreLocation::parse("progress-store").unwrap();
    let (oracle, path) = S::parse(&Url::parse(&bare.url).unwrap()).unwrap();
    assert_eq!(
        (bare.route, &bare.backend, &bare.prefix, &bare.s3),
        (
            StoreRoute::ParseUrl,
            &oracle,
            &ObjPath::parse(path).unwrap(),
            &None
        )
    );
    assert_eq!(bare.backend, S::Local);
    // Neither object_store nor the store opens these.
    for url in [
        "ftp://example.com/run-1",
        "sftp://example.com/run-1",
        "file://remote-host/run-1",
        "memory://remote/",
    ] {
        assert!(S::parse(&Url::parse(url).unwrap()).is_err(), "{url}");
        assert!(StoreLocation::parse(url).is_err(), "{url}");
    }
}
