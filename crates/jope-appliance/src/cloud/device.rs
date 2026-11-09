//! The `X-Jain-Device` pseudonym (ai-veox-app `docs/connected-cli.md`, *The CLI device id*):
//!
//! `device_id = lowercase_hex(HMAC-SHA256(key = "jain-device-v1", message = OS machine id))[0..32]`
//!
//! The raw machine id never leaves this file: it is read, hashed and dropped. Without a readable
//! machine id a random 128-bit id, generated once and kept in a private file, stands in. The
//! header is not sent when telemetry is off (`DO_NOT_TRACK`, `JAIN_TELEMETRY=0`,
//! `--telemetry off` or the first-run "off" choice).
use super::*;

const KEY: &[u8] = b"jain-device-v1";

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let hex = jope_runtime::sha256_hex(bytes);
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).expect("sha256_hex is hex");
    }
    out
}

/// RFC 2104 HMAC over SHA-256, from the runtime's SHA-256 (no extra dependency).
pub(crate) fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut block = [0u8; 64];
    if key.len() > 64 {
        block[..32].copy_from_slice(&sha256(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut inner = Vec::with_capacity(64 + message.len());
    inner.extend(block.iter().map(|b| b ^ 0x36));
    inner.extend_from_slice(message);
    let inner = sha256(&inner);
    let mut outer = Vec::with_capacity(96);
    outer.extend(block.iter().map(|b| b ^ 0x5c));
    outer.extend_from_slice(&inner);
    sha256(&outer)
}

/// The 32-hex device id for a machine id.
pub(crate) fn device_digest(machine_id: &[u8]) -> String {
    hmac_sha256(KEY, machine_id)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()[..32]
        .to_owned()
}

fn machine_id() -> Option<Vec<u8>> {
    #[cfg(target_os = "linux")]
    {
        for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"] {
            if let Ok(text) = fs::read_to_string(path) {
                let text = text.trim();
                if !text.is_empty() && text.len() <= 256 {
                    return Some(text.as_bytes().to_vec());
                }
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("ioreg")
            .args(["-rd1", "-c", "IOPlatformExpertDevice"])
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&output.stdout);
        let line = text.lines().find(|l| l.contains("\"IOPlatformUUID\""))?;
        let value = line.rsplit('"').nth(1)?.trim();
        (!value.is_empty()).then(|| value.as_bytes().to_vec())
    }
    #[cfg(windows)]
    {
        let output = std::process::Command::new("reg")
            .args([
                "query",
                r"HKLM\SOFTWARE\Microsoft\Cryptography",
                "/v",
                "MachineGuid",
            ])
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&output.stdout);
        let value = text
            .lines()
            .find(|l| l.contains("MachineGuid"))?
            .split_whitespace()
            .last()?
            .to_owned();
        (!value.is_empty()).then(|| value.into_bytes())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        None
    }
}

fn fallback() -> Result<Vec<u8>> {
    let path = cloud_root()?.join("device-fallback-id");
    if path.exists() {
        let value = read_private(&path)?;
        if value.len() == 32 && value.iter().all(u8::is_ascii_hexdigit) {
            return Ok(value);
        }
    }
    let value = format!("{:032x}", rand::random::<u128>());
    write_private(&path, value.as_bytes())?;
    Ok(value.into_bytes())
}

/// Whether the device header may be sent at all.
pub(crate) fn allowed() -> bool {
    if crate::terminal::onboarding::telemetry_forced_off_from_env() {
        return false;
    }
    crate::terminal::onboarding::load(&jope_paths::jopedime_home()).telemetry != Some(false)
}

/// The value of `X-Jain-Device`, or `None` when telemetry is off.
pub(crate) fn header() -> Option<String> {
    static ID: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    if !allowed() {
        return None;
    }
    ID.get_or_init(|| {
        let id = machine_id().map(Ok).unwrap_or_else(fallback).ok()?;
        Some(device_digest(&id))
    })
    .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_matches_rfc_4231() {
        let hex = |b: [u8; 32]| b.iter().map(|b| format!("{b:02x}")).collect::<String>();
        // RFC 4231 test case 2.
        assert_eq!(
            hex(hmac_sha256(b"Jefe", b"what do ya want for nothing?")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        // Test case 6: a key longer than the block is hashed first.
        assert_eq!(
            hex(hmac_sha256(
                &[0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            )),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn the_device_id_is_32_lowercase_hex_and_never_the_machine_id() {
        let id = device_digest(b"0123456789abcdef0123456789abcdef");
        assert_eq!(id.len(), 32);
        assert!(id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        assert_ne!(id, "0123456789abcdef0123456789abcdef");
        assert_eq!(id, device_digest(b"0123456789abcdef0123456789abcdef"));
        assert_ne!(id, device_digest(b"another machine"));
    }
}
