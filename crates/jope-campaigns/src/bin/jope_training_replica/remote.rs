use super::*;

// ssh transport
// ---------------------------------------------------------------------------

pub(super) fn ssh_base(target: &Target) -> Command {
    let mut cmd = Command::new("ssh");
    cmd.arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg(format!("ConnectTimeout={SSH_CONNECT_TIMEOUT}"))
        .arg("-o")
        .arg("ServerAliveInterval=15")
        .arg("-o")
        .arg("ServerAliveCountMax=4")
        .arg("-C")
        .arg("-p")
        .arg(target.port.to_string())
        .arg(&target.node);
    cmd
}

pub(super) fn ssh_run(target: &Target, script: &str) -> Result<String> {
    let out = ssh_base(target)
        .arg(script)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("ssh {}", target.label()))?;
    if !out.status.success() {
        bail!(
            "ssh {} exit {:?}: {}",
            target.label(),
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

pub(super) fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// Push bytes to `incoming/<tmp>` and return the hash the *receiving node*
/// computed over what actually landed. The caller compares it to the local hash
/// before promoting the object; nothing is ever renamed on trust.
pub(super) fn push_stream(
    target: &Target,
    rroot: &str,
    tmp: &str,
    source: Stdio,
) -> Result<String> {
    let dest = format!("{rroot}/incoming/{tmp}");
    let script = format!(
        "set -e; mkdir -p {} ; cat > {} ; sha256sum {}",
        quote(&format!("{rroot}/incoming")),
        quote(&dest),
        quote(&dest)
    );
    let child = ssh_base(target)
        .arg(&script)
        .stdin(source)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("ssh {}", target.label()))?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!(
            "upload to {} exit {:?}: {}",
            target.label(),
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let sha = text
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string();
    if !is_sha256(&sha) {
        bail!("remote sha256sum returned {text:?}");
    }
    Ok(sha)
}

pub(super) fn upload_object(target: &Target, rroot: &str, sha: &str, path: &Path) -> Result<()> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let tmp = format!("{sha}.{}.part", std::process::id());
    let observed = push_stream(target, rroot, &tmp, Stdio::from(file))?;
    if observed != sha {
        let _ = ssh_run(
            target,
            &format!("rm -f {}", quote(&format!("{rroot}/incoming/{tmp}"))),
        );
        bail!("corrupt transfer of {sha}: target computed {observed}");
    }
    let rel = object_rel(sha);
    ssh_run(
        target,
        &format!(
            "set -e; mkdir -p {}; mv -f {} {}",
            quote(&format!("{rroot}/objects/{}", &sha[..2])),
            quote(&format!("{rroot}/incoming/{tmp}")),
            quote(&format!("{rroot}/objects/{rel}"))
        ),
    )?;
    Ok(())
}

pub(super) fn upload_bytes(target: &Target, rroot: &str, bytes: &[u8]) -> Result<String> {
    let sha = sha256_hex(bytes);
    let tmp = format!("{sha}.{}.part", std::process::id());
    let dest = format!("{rroot}/incoming/{tmp}");
    let script = format!(
        "set -e; mkdir -p {} ; cat > {} ; sha256sum {}",
        quote(&format!("{rroot}/incoming")),
        quote(&dest),
        quote(&dest)
    );
    let mut child = ssh_base(target)
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child.stdin.as_mut().context("stdin")?.write_all(bytes)?;
    drop(child.stdin.take());
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!(
            "manifest upload to {} exit {:?}: {}",
            target.label(),
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let observed = text
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string();
    if observed != sha {
        bail!("corrupt manifest transfer: target computed {observed:?}");
    }
    Ok(tmp)
}

/// What a target actually holds, by content rather than by filename.
pub(super) struct RemoteObjects {
    /// Objects whose bytes still hash to their own name.
    pub(super) verified: BTreeSet<String>,
    /// Object names present but hashing to something else — bit rot, a truncated
    /// filesystem, or tampering.
    pub(super) corrupt: BTreeSet<String>,
}

impl RemoteObjects {
    fn all(&self) -> BTreeSet<String> {
        self.verified.union(&self.corrupt).cloned().collect()
    }
}

/// Re-hash every object on the target and keep only the ones that still match
/// their own name.
///
/// Listing filenames is not enough, and this was a real hole rather than a
/// theoretical one: a corrupted object keeps its correct name, so a
/// name-only survey reports it "present", never re-uploads it, and the replica
/// stays quietly unusable until someone attempts a restore. Anything that fails
/// to verify is excluded from `verified`, which makes the ordinary upload path
/// repair it on this very cycle.
///
/// `nice` keeps the scrub off whatever the target is really doing.
pub(super) fn remote_objects(target: &Target, rroot: &str) -> Result<RemoteObjects> {
    let text = ssh_run(
        target,
        &format!(
            "set -e; mkdir -p {} {} {}; cd {}; \
             find . -type f -print0 | xargs -0 -r nice -n 10 sha256sum 2>/dev/null || true",
            quote(&format!("{rroot}/objects")),
            quote(&format!("{rroot}/incoming")),
            quote(&format!("{rroot}/manifests")),
            quote(&format!("{rroot}/objects"))
        ),
    )?;
    let mut verified = BTreeSet::new();
    let mut corrupt = BTreeSet::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let (Some(hash), Some(path)) = (parts.next(), parts.next()) else {
            continue;
        };
        let name = path.rsplit('/').next().unwrap_or_default();
        if !is_sha256(name) {
            continue;
        }
        if hash == name {
            verified.insert(name.to_string());
        } else {
            corrupt.insert(name.to_string());
        }
    }
    Ok(RemoteObjects { verified, corrupt })
}

// ---------------------------------------------------------------------------
