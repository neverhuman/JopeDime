use super::*;

/// Runs nothing: these fixtures record no GPU.
struct NoPrograms;
impl CommandRunner for NoPrograms {
    fn run(&self, program: &str, _: &[&str]) -> Result<Option<CommandOutput>> {
        bail!("this fixture runs no {program}")
    }
}
fn loopback() -> Result<BTreeSet<IpAddr>> {
    Ok(BTreeSet::from([IpAddr::V4(Ipv4Addr::LOCALHOST)]))
}
/// A host whose identity files are `dir/{hostname,machine-id,boot_id}`, installed as a CPU-only
/// collector would see it. Every source is a file or function of this fixture.
fn host(dir: &Path) -> (CollectSpec, HostSources<'static>) {
    let files: [(&str, &[u8]); 8] = [
        ("hostname", b"fixture-host\n"),
        ("machine-id", b"0123456789abcdef0123456789abcdef\n"),
        ("boot_id", b"01234567-89ab-cdef-0123-456789abcdef\n"),
        ("worker", b"fixture worker"),
        ("runtime.so", b"fixture runtime"),
        ("compiler", b"fixture compiler"),
        ("policy.json", b"{}"),
        ("host.der", &[0x30, 0x05, 0x30, 0x03, 0x02, 0x01, 0x07]),
    ];
    for (name, bytes) in files {
        fs::write(dir.join(name), bytes).unwrap();
    }
    let spec = CollectSpec {
        address: "127.0.0.1".into(),
        certificate: dir.join("host.der"),
        source_commit: "0".repeat(40),
        executable: dir.join("worker"),
        runtime_libraries: vec![dir.join("runtime.so")],
        compiler: dir.join("compiler"),
        cuda_libraries: Vec::new(),
        nccl_libraries: Vec::new(),
        installed_policies: vec![dir.join("policy.json")],
        cuda_visible_devices: Some(OsString::new()),
    };
    let sources = HostSources {
        host_name: dir.join("hostname"),
        machine_id: dir.join("machine-id"),
        boot_id: dir.join("boot_id"),
        interfaces: &loopback,
        commands: &NoPrograms,
    };
    (spec, sources)
}
fn refusal<T: std::fmt::Debug>(r: Result<T>) -> String {
    format!("{:#}", r.expect_err("expected a refusal"))
}

/// The runner container's `/etc/machine-id` is empty (full-main run 36449887665). An empty,
/// malformed or absent machine id refuses both the collection and a participant's
/// observation, where it is read and naming the file; nothing is derived in its place.
#[test]
fn an_empty_malformed_or_absent_machine_id_is_refused_where_it_is_read() {
    let t = tempfile::tempdir().unwrap();
    let dir = fs::canonicalize(t.path()).unwrap();
    let (spec, sources) = host(&dir);
    let path = dir.join("machine-id");
    let (census, _) = collect(&spec, &sources, 1).unwrap();
    assert_eq!(
        census.hosts[0].identity.machine_id,
        "0123456789abcdef0123456789abcdef"
    );
    for (content, read) in [
        ("", ""),
        ("\n", ""),
        (
            "0123456789ABCDEF0123456789ABCDEF\n",
            "0123456789ABCDEF0123456789ABCDEF",
        ),
        (
            "0123456789abcdef0123456789abcde\n",
            "0123456789abcdef0123456789abcde",
        ),
        (
            "0123456789abcdef0123456789abcdef0\n",
            "0123456789abcdef0123456789abcdef0",
        ),
        (
            "0123456789abcdef 0123456789abcdef",
            "0123456789abcdef 0123456789abcdef",
        ),
        ("uninitialized\n", "uninitialized"),
    ] {
        fs::write(&path, content).unwrap();
        let expected = format!(
            "machine id `{read}` read from {} is not 32 lowercase hex",
            path.display()
        );
        assert_eq!(refusal(collect(&spec, &sources, 1)), expected);
        assert_eq!(refusal(observe_identity(&spec, &sources)), expected);
    }
    fs::remove_file(&path).unwrap();
    let absent = refusal(collect(&spec, &sources, 1));
    assert!(
        absent.starts_with(&format!("resolve {}: ", path.display())),
        "{absent}"
    );
}

#[test]
fn an_empty_or_oversized_boot_id_and_a_bad_host_name_are_refused_where_read() {
    let t = tempfile::tempdir().unwrap();
    let dir = fs::canonicalize(t.path()).unwrap();
    let (spec, sources) = host(&dir);
    let boot = dir.join("boot_id");
    for content in ["", "\n", &"b".repeat(65)] {
        fs::write(&boot, content).unwrap();
        assert_eq!(
            refusal(collect(&spec, &sources, 1)),
            format!(
                "boot id `{}` read from {} is empty or longer than 64 bytes",
                content.trim(),
                boot.display()
            )
        );
    }
    fs::write(&boot, "b".repeat(64)).unwrap();
    collect(&spec, &sources, 1).unwrap();
    let name = dir.join("hostname");
    for content in ["", "not a host\n", "host/name"] {
        fs::write(&name, content).unwrap();
        assert_eq!(
            refusal(observe_identity(&spec, &sources)),
            format!(
                "host name `{}` read from {} is not a census host name",
                content.trim(),
                name.display()
            )
        );
    }
}

/// The production provider names exactly the real host's identity files, so it reads them
/// through the checked reader the refusals above exercise: an empty or malformed
/// `/etc/machine-id` refuses a production collection and a production participant alike.
/// Nothing here reads the real files.
#[test]
fn the_production_provider_reads_the_real_host_identity_files() {
    let system = HostSources::system();
    assert_eq!(system.host_name, Path::new("/proc/sys/kernel/hostname"));
    assert_eq!(system.machine_id, Path::new("/etc/machine-id"));
    assert_eq!(system.boot_id, Path::new("/proc/sys/kernel/random/boot_id"));
    assert_eq!(
        (SYSTEM_HOST_NAME, SYSTEM_MACHINE_ID, SYSTEM_BOOT_ID),
        (
            "/proc/sys/kernel/hostname",
            "/etc/machine-id",
            "/proc/sys/kernel/random/boot_id"
        )
    );
}

#[test]
fn base64_and_pem_framing_are_strict() {
    assert_eq!(base64_decode("TWFu").unwrap(), b"Man");
    assert_eq!(base64_decode("TWE=").unwrap(), b"Ma");
    assert_eq!(base64_decode("TQ==\n").unwrap(), b"M");
    for bad in ["TQ=", "T===", "TQ==TWFu", "TW!u", ""] {
        assert!(base64_decode(bad).is_err(), "{bad}");
    }
    let der = [0x30, 0x03, 0x30, 0x01, 0x00];
    assert_eq!(certificate_der(&der).unwrap(), der);
    let pem = "-----BEGIN CERTIFICATE-----\nMAMwAQA=\n-----END CERTIFICATE-----\n";
    assert_eq!(certificate_der(pem.as_bytes()).unwrap(), der);
    assert!(certificate_der(format!("{pem}{pem}").as_bytes()).is_err());
    assert!(certificate_der(&[0x30, 0x03, 0x02, 0x01, 0x00]).is_err());
    assert!(certificate_der(&[0x30, 0x03, 0x30, 0x01]).is_err());
    assert!(certificate_der(&[0x30, 0x03, 0x30, 0x01, 0x00, 0x00]).is_err());
}
