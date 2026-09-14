use jope_parallel::{corpus::Records,manifest::*};
use std::{fs,path::PathBuf,process::{Command,Stdio},time::{Duration,Instant}};

#[test]
#[ignore = "subprocess fixture isolates cache environment"]
fn shared_cache_child() {
    let root=PathBuf::from(std::env::var_os("JOPE_CACHE_TEST_ROOT").unwrap());
    let path=root.join("records.bin"); let expected=fs::read_to_string(root.join("expected")).unwrap();
    if std::env::var_os("JOPE_CACHE_EXPECT_REJECTION").is_some() {
        assert!(Records::open_verified(&path,1,&expected).is_err()); return;
    }
    let first=Records::open_verified(&path,1,&expected).unwrap();
    assert_eq!(first.get(0).tokens[0],7);
    fs::write(&path,vec![0u8;532]).unwrap();
    let second=Records::open_verified(&path,1,&expected).unwrap();
    assert_eq!(second.get(0).tokens[0],7);
    assert_eq!(first.get(0).tokens[0],7);
}

#[test]
fn native_ranks_share_a_sealed_identity_and_server_restart_reverifies() {
    let d=tempfile::tempdir().unwrap(); let root=d.path();
    let pool=root.join("runs/pool");fs::create_dir_all(&pool).unwrap();
    let mut bytes=vec![0u8;532];bytes[0]=7;
    fs::write(root.join("records.bin"),&bytes).unwrap();
    fs::write(root.join("expected"),hash(&bytes)).unwrap();
    for restart in [false,true] {
        let mut server=Command::new("python3").args(["-B"])
            .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/shared_teaching_cache.py"))
            .arg("--root").arg(root).arg("--pool").arg(&pool)
            .stdout(Stdio::null()).stderr(Stdio::inherit()).spawn().unwrap();
        let deadline=Instant::now()+Duration::from_secs(10);
        while !pool.join("teaching.sock").exists() {
            assert!(Instant::now()<deadline && server.try_wait().unwrap().is_none());
            std::thread::sleep(Duration::from_millis(20));
        }
        let mut child=Command::new(std::env::current_exe().unwrap());
        child.args(["--ignored","--exact","shared_cache_child"])
            .env("JOPE_CACHE_TEST_ROOT",root).env("JOPE_TEACHING_CACHE",pool.join("teaching.sock"))
            .env("JOPE_GPU_TASK_DEADLINE_MS", (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() + 60_000).to_string());
        if restart {child.env("JOPE_CACHE_EXPECT_REJECTION","1");}
        let result=child.status().unwrap();
        if !restart {
            // Cache counters are reported every ten seconds; content/seal
            // verification itself still precedes each successful response.
            let deadline=Instant::now()+Duration::from_secs(15);
            loop {
                let receipt:serde_json::Value=read_json(&pool.join("teaching-cache.json")).unwrap_or_default();
                if receipt["verifications"]==1 && receipt["hits"]==1 {break;}
                assert!(Instant::now()<deadline && server.try_wait().unwrap().is_none(),"cache telemetry did not arrive");
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        server.kill().unwrap();server.wait().unwrap();
        fs::remove_file(pool.join("teaching.sock")).unwrap();
        assert!(result.success());
        if !restart {
            let receipt:serde_json::Value=read_json(&pool.join("teaching-cache.json")).unwrap();
            assert_eq!(receipt["verifications"],1);assert_eq!(receipt["hits"],1);
        }
    }
}
