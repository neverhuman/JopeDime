use super::*;

/// Read a small local file (bounded, regular). A symlink resolves to its installed target,
/// as `/etc/machine-id` or a certificate may legitimately be one.
fn read_small(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let canonical =
        fs::canonicalize(path).with_context(|| format!("resolve {}", path.display()))?;
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(&canonical)
        .with_context(|| format!("open {}", path.display()))?;
    let meta = f.metadata()?;
    // procfs files report length 0; they are regular files all the same.
    ensure!(meta.is_file(), "{} is not a regular file", path.display());
    let mut b = Vec::new();
    f.take(limit + 1).read_to_end(&mut b)?;
    ensure!(
        b.len() as u64 <= limit,
        "{} exceeds {limit} bytes",
        path.display()
    );
    Ok(b)
}
fn read_line(path: &Path) -> Result<String> {
    let text = String::from_utf8(read_small(path, 256)?)
        .with_context(|| format!("{} is not UTF-8", path.display()))?;
    Ok(text.trim().to_string())
}

/// SHA-256 of one installed file, streamed, with its size and identity re-checked afterwards.
/// Symlinks resolve to the installed file, whose canonical path is returned.
pub fn hash_installed_file(path: &Path) -> Result<(PathBuf, String, u64)> {
    ensure!(path.is_absolute(), "{} must be absolute", path.display());
    let canonical =
        fs::canonicalize(path).with_context(|| format!("resolve {}", path.display()))?;
    let mut f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(&canonical)
        .with_context(|| format!("open {}", canonical.display()))?;
    let before = f.metadata()?;
    ensure!(
        before.is_file() && before.len() <= MAX_HASHED_FILE_BYTES,
        "{} is not a bounded regular file",
        canonical.display()
    );
    let mut state = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut count = 0u64;
    loop {
        let n = f.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        count += n as u64;
        ensure!(
            count <= before.len(),
            "{} grew while hashing",
            canonical.display()
        );
        state.update(&buffer[..n]);
    }
    let after = f.metadata()?;
    ensure!(
        count == before.len()
            && after.len() == before.len()
            && after.ino() == before.ino()
            && after.mtime() == before.mtime()
            && after.mtime_nsec() == before.mtime_nsec(),
        "{} changed while hashing",
        canonical.display()
    );
    Ok((canonical, hex::encode(state.finalize()), count))
}
/// Digest of a set of installed files: `(canonical path, sha256, bytes)` sorted by path.
pub fn file_closure_digest(paths: &[PathBuf]) -> Result<String> {
    ensure!(
        !paths.is_empty() && paths.len() <= MAX_FILES_PER_CLOSURE,
        "file closure must name 1..={MAX_FILES_PER_CLOSURE} files"
    );
    let mut entries = BTreeMap::new();
    for p in paths {
        let (canonical, sha, bytes) = hash_installed_file(p)?;
        ensure!(
            entries.insert(canonical.clone(), (sha, bytes)).is_none(),
            "{} named twice",
            canonical.display()
        );
    }
    let entries: Vec<_> = entries.into_iter().map(|(p, (s, b))| (p, s, b)).collect();
    digest(&(FILE_CLOSURE_SCHEMA, entries))
}

/// Strict standard base64 (RFC 4648 with padding), ignoring ASCII whitespace.
pub(crate) fn base64_decode(text: &str) -> Result<Vec<u8>> {
    let clean: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    ensure!(
        !clean.is_empty() && clean.len().is_multiple_of(4),
        "base64 length is not a positive multiple of 4"
    );
    let value = |c: u8| -> Result<u32> {
        Ok(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => bail!("invalid base64 byte"),
        } as u32)
    };
    let mut out = Vec::with_capacity(clean.len() / 4 * 3);
    let chunks = clean.chunks(4).count();
    for (i, chunk) in clean.chunks(4).enumerate() {
        let pad = chunk.iter().rev().take_while(|&&c| c == b'=').count();
        ensure!(
            pad <= 2 && (pad == 0 || i + 1 == chunks),
            "misplaced base64 padding"
        );
        let mut n = 0u32;
        for &c in &chunk[..4 - pad] {
            n = (n << 6) | value(c)?;
        }
        n <<= 6 * pad as u32;
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&bytes[..3 - pad]);
    }
    Ok(out)
}
/// DER bytes of one certificate: DER as given, or the single `CERTIFICATE` block of a PEM file.
/// Only the outer framing is checked here (a SEQUENCE whose first element is a SEQUENCE); the
/// transport verifies the certificate itself.
pub fn certificate_der(bytes: &[u8]) -> Result<Vec<u8>> {
    const BEGIN: &str = "-----BEGIN CERTIFICATE-----";
    const END: &str = "-----END CERTIFICATE-----";
    let der = if bytes.starts_with(b"-----BEGIN") {
        let text = std::str::from_utf8(bytes).context("PEM certificate is not UTF-8")?;
        ensure!(
            text.matches(BEGIN).count() == 1 && text.matches("-----BEGIN").count() == 1,
            "PEM file must hold exactly one CERTIFICATE block"
        );
        let body = text
            .split_once(BEGIN)
            .and_then(|(_, rest)| rest.split_once(END))
            .map(|(body, _)| body)
            .context("PEM CERTIFICATE block is not terminated")?;
        base64_decode(body)?
    } else {
        bytes.to_vec()
    };
    let (outer, rest) = der_element(&der)?;
    ensure!(
        rest.is_empty() && outer.0 == 0x30,
        "certificate is not a single DER SEQUENCE"
    );
    let (inner, _) = der_element(outer.1)?;
    ensure!(
        inner.0 == 0x30,
        "certificate does not start with a TBSCertificate SEQUENCE"
    );
    Ok(der)
}
/// One DER TLV: `((tag, contents), remainder)`, definite lengths only.
type DerElement<'a> = ((u8, &'a [u8]), &'a [u8]);
pub(crate) fn der_element(b: &[u8]) -> Result<DerElement<'_>> {
    ensure!(b.len() >= 2, "truncated DER element");
    let tag = b[0];
    let (len, header) = match b[1] {
        n if n < 0x80 => (n as usize, 2),
        0x81..=0x84 => {
            let k = (b[1] & 0x7f) as usize;
            ensure!(b.len() >= 2 + k, "truncated DER length");
            let mut len = 0usize;
            for &x in &b[2..2 + k] {
                len = (len << 8) | x as usize;
            }
            ensure!(len >= 0x80, "non-minimal DER length");
            (len, 2 + k)
        }
        _ => bail!("unsupported DER length form"),
    };
    ensure!(b.len() >= header + len, "truncated DER contents");
    Ok(((tag, &b[header..header + len]), &b[header + len..]))
}

/// Every IP address assigned to a local interface.
pub fn local_addresses() -> Result<BTreeSet<IpAddr>> {
    let mut head: *mut libc::ifaddrs = std::ptr::null_mut();
    ensure!(
        // SAFETY: `head` is a live local out-pointer; on success the list is freed once below.
        unsafe { libc::getifaddrs(&mut head) } == 0,
        "getifaddrs: {}",
        std::io::Error::last_os_error()
    );
    let mut out = BTreeSet::new();
    let mut cursor = head;
    while !cursor.is_null() {
        // SAFETY: getifaddrs returned a valid list that is freed only below.
        let ifa = unsafe { &*cursor };
        if !ifa.ifa_addr.is_null() {
            match i32::from(unsafe { (*ifa.ifa_addr).sa_family }) {
                libc::AF_INET => {
                    // SAFETY: sa_family is AF_INET, so getifaddrs stored a sockaddr_in at
                    // `ifa_addr`; it lives in the list until freeifaddrs below.
                    let sa = unsafe { &*(ifa.ifa_addr as *const libc::sockaddr_in) };
                    out.insert(IpAddr::V4(Ipv4Addr::from(u32::from_be(sa.sin_addr.s_addr))));
                }
                libc::AF_INET6 => {
                    // SAFETY: sa_family is AF_INET6, so getifaddrs stored a sockaddr_in6 at
                    // `ifa_addr`; it lives in the list until freeifaddrs below.
                    let sa = unsafe { &*(ifa.ifa_addr as *const libc::sockaddr_in6) };
                    out.insert(IpAddr::V6(Ipv6Addr::from(sa.sin6_addr.s6_addr)));
                }
                _ => {}
            }
        }
        cursor = ifa.ifa_next;
    }
    // SAFETY: `head` is the list the successful getifaddrs allocated, freed exactly once here;
    // no reference into it (`ifa`, `sa`) outlives the loop above.
    unsafe { libc::freeifaddrs(head) };
    Ok(out)
}

fn online_cpu_millicores() -> Result<u32> {
    // SAFETY: sysconf takes a constant by value and has no preconditions; -1 is checked below.
    let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    ensure!(n > 0, "online CPU count unavailable");
    u32::try_from(n)
        .ok()
        .and_then(|n| n.checked_mul(1000))
        .context("online CPU count out of range")
}
/// `MemAvailable` from `/proc/meminfo`: memory available now, not total capacity.
fn available_memory_bytes() -> Result<u64> {
    let text = String::from_utf8(read_small(Path::new("/proc/meminfo"), 64 * 1024)?)?;
    let kib: u64 = text
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().strip_suffix("kB"))
        .context("MemAvailable absent from /proc/meminfo")?
        .trim()
        .parse()?;
    kib.checked_mul(1024).context("MemAvailable overflow")
}

fn csv_fields(line: &str, n: usize) -> Result<Vec<&str>> {
    let fields: Vec<&str> = line.splitn(n, ',').map(str::trim).collect();
    ensure!(
        fields.len() == n && fields.iter().all(|f| !f.is_empty()),
        "unexpected nvidia-smi record `{line}`"
    );
    ensure!(
        !fields.iter().any(|f| f.starts_with('[')),
        "nvidia-smi reports an unavailable value in `{line}`; nothing is invented"
    );
    Ok(fields)
}
fn probe_text(out: &CommandOutput, what: &str) -> Result<String> {
    ensure!(
        out.success,
        "nvidia-smi {what} failed: {}",
        String::from_utf8_lossy(&out.stderr)
            .chars()
            .take(400)
            .collect::<String>()
    );
    String::from_utf8(out.stdout.clone()).with_context(|| format!("nvidia-smi {what} output"))
}
/// GPUs, the common driver version and device-owning processes, host-wide.
type GpuInventory = (
    GpuProbe,
    Vec<GpuObservation>,
    Option<String>,
    Vec<Ownership>,
);
fn gpu_inventory(runner: &dyn CommandRunner, visible: Option<&OsString>) -> Result<GpuInventory> {
    match visible {
        Some(v) if v.is_empty() => {
            return Ok((
                GpuProbe::NotQueriedCudaVisibleDevicesEmpty,
                Vec::new(),
                None,
                Vec::new(),
            ))
        }
        Some(_) => bail!(
            "the census is host-wide: unset CUDA_VISIBLE_DEVICES, or set it to the empty string \
             for a CPU-only census that records no GPU"
        ),
        None => {}
    }
    let Some(out) = runner.run("nvidia-smi", &GPU_QUERY)? else {
        return Ok((GpuProbe::NvidiaSmiAbsent, Vec::new(), None, Vec::new()));
    };
    let mut gpus = Vec::new();
    let mut drivers = BTreeSet::new();
    for line in probe_text(&out, "GPU query")?
        .lines()
        .filter(|l| !l.trim().is_empty())
    {
        let f = csv_fields(line, 6)?;
        let uuid = f[1].to_string();
        ensure!(
            uuid.starts_with("GPU-") && is_name(&uuid),
            "nvidia-smi GPU UUID `{uuid}` is not in GPU-<uuid> form"
        );
        ensure!(
            f[3].len() <= 64 && f[3].bytes().all(|b| b.is_ascii_digit() || b == b'.'),
            "nvidia-smi driver version `{}`",
            f[3]
        );
        drivers.insert(f[3].to_string());
        let mib = |s: &str| -> Result<u64> {
            s.parse::<u64>()
                .ok()
                .and_then(|m| m.checked_mul(MIB))
                .with_context(|| format!("nvidia-smi memory value `{s}`"))
        };
        gpus.push(GpuObservation {
            uuid,
            model: f[2].to_string(),
            local_index: f[0]
                .parse()
                .with_context(|| format!("nvidia-smi index `{}`", f[0]))?,
            total_vram_bytes: mib(f[4])?,
            available_vram_bytes: mib(f[5])?,
        });
    }
    ensure!(
        drivers.len() <= 1,
        "GPUs on one host report different driver versions"
    );
    let known: BTreeSet<&str> = gpus.iter().map(|g| g.uuid.as_str()).collect();
    let apps = runner
        .run("nvidia-smi", &APPS_QUERY)?
        .context("nvidia-smi disappeared between the GPU and process queries")?;
    let mut owners = Vec::new();
    for line in probe_text(&apps, "compute-apps query")?
        .lines()
        .filter(|l| !l.trim().is_empty() && l.trim() != "No running processes found")
    {
        let f = csv_fields(line, 3)?;
        ensure!(
            known.contains(f[0]),
            "a device process names GPU `{}`, which the GPU query did not report",
            f[0]
        );
        let pid: u32 = f[1]
            .parse()
            .with_context(|| format!("nvidia-smi pid `{}`", f[1]))?;
        // An observed device process is an owner the collector cannot attribute to a managed
        // run; it is recorded as a non-managed owner, which conflicts with any plan for it.
        owners.push(Ownership {
            owner: format!("gpu-process-{pid}"),
            kind: OwnerKind::User,
            device_uuid: Some(f[0].to_string()),
            cpu_millicores: 0,
            ram_bytes: 0,
            run: None,
            attempt: None,
        });
    }
    Ok((
        GpuProbe::NvidiaSmi,
        gpus,
        drivers.into_iter().next(),
        owners,
    ))
}

/// The local host's identity exactly as the collector records it. A rendezvous participant
/// reports this in its hello, with its own running binary as `spec.executable`, so a host's
/// census and its participants observe the same values the same way, through the same sources.
pub fn observe_identity(spec: &CollectSpec, sources: &HostSources) -> Result<HostIdentity> {
    Ok(observe(spec, sources)?.0)
}
type Observed = (HostIdentity, GpuProbe, Vec<GpuObservation>, Vec<Ownership>);
fn observe(spec: &CollectSpec, sources: &HostSources) -> Result<Observed> {
    let (host, machine_id, boot_id) = observe_host(sources)?;
    let address: IpAddr = spec
        .address
        .parse()
        .with_context(|| format!("address `{}` is not an IP address", spec.address))?;
    ensure!(
        (sources.interfaces)()?.contains(&address),
        "address {address} is not assigned to a local interface"
    );
    ensure!(
        is_commit(&spec.source_commit),
        "source commit must be 40 lowercase hex"
    );
    let certificate = read_small(&spec.certificate, MAX_CERTIFICATE_BYTES)
        .and_then(|b| certificate_der(&b))
        .with_context(|| format!("certificate {}", spec.certificate.display()))?;
    let closure = |paths: &[PathBuf]| -> Result<Option<String>> {
        if paths.is_empty() {
            Ok(None)
        } else {
            file_closure_digest(paths).map(Some)
        }
    };
    ensure!(
        !spec.runtime_libraries.is_empty() && !spec.installed_policies.is_empty(),
        "at least one runtime library and one installed policy file are required"
    );
    let (gpu_probe, gpus, driver_version, ownership) =
        gpu_inventory(sources.commands, spec.cuda_visible_devices.as_ref())?;
    let identity = HostIdentity {
        host,
        machine_id,
        boot_id,
        address: address.to_string(),
        certificate_sha256: hash(&certificate),
        source_commit: spec.source_commit.clone(),
        executable_sha256: hash_installed_file(&spec.executable)?.1,
        installed_policy_sha256: file_closure_digest(&spec.installed_policies)?,
        runtime: RuntimeIdentity {
            runtime_sha256: file_closure_digest(&spec.runtime_libraries)?,
            compiler_sha256: hash_installed_file(&spec.compiler)?.1,
            cuda_sha256: closure(&spec.cuda_libraries)?,
            nccl_sha256: closure(&spec.nccl_libraries)?,
            driver_version,
        },
    };
    Ok((identity, gpu_probe, gpus, ownership))
}

/// Observe the local host through `sources`. Refuses rather than inventing any value.
pub fn collect(
    spec: &CollectSpec,
    sources: &HostSources,
    now_unix_ms: u64,
) -> Result<(HostCensus, CollectionReport)> {
    let (identity, gpu_probe, gpus, ownership) = observe(spec, sources)?;
    let census = HostCensus {
        schema: CENSUS_SCHEMA.into(),
        captured_unix_ms: now_unix_ms,
        hosts: vec![HostObservation {
            identity,
            observed_unix_ms: now_unix_ms,
            cpu_millicores_allocatable: online_cpu_millicores()?,
            ram_bytes_allocatable: available_memory_bytes()?,
            gpus,
            ownership,
        }],
    };
    census.validate()?;
    let bytes = serde_json::to_vec(&census)?;
    ensure!(bytes.len() <= MAX_CENSUS_BYTES, "collected census bound");
    let h = &census.hosts[0];
    let report = CollectionReport {
        schema: COLLECTION_SCHEMA.into(),
        census_sha256: hash(&bytes),
        host: h.identity.host.clone(),
        machine_id: h.identity.machine_id.clone(),
        boot_id: h.identity.boot_id.clone(),
        address: h.identity.address.clone(),
        observed_unix_ms: now_unix_ms,
        gpu_probe,
        gpus: h.gpus.len() as u32,
        device_owners: h.ownership.len() as u32,
        allocation_held: false,
    };
    Ok((census, report))
}
