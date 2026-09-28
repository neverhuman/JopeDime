//! In-test certificates for the bounded transport fixtures: self-signed Ed25519 X.509 v3
//! certificates and PKCS#8 v1 keys, encoded from DER by hand and signed with ed25519-dalek.
//! No certificate crate is added to the workspace. Every identity here is synthetic.
#![allow(dead_code)]
use ed25519_dalek::{Signer, SigningKey};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// How long a trickling peer keeps sending. Before the bounded transport enforced absolute
/// deadlines, a blocked read lasted this long (review 5339604155).
pub const TRICKLE: Duration = Duration::from_millis(2_500);
/// Scheduling slack over a deadline on a loaded CI host. A 300 ms deadline plus this is still
/// under the 1.5 s the review observed, and far under [`TRICKLE`].
pub const SLACK: Duration = Duration::from_millis(500);
/// A TLS handshake record header that announces 16 KiB of body, none of which follows in time.
pub const PARTIAL_RECORD: [u8; 5] = [22, 3, 3, 0x40, 0];
/// Send `head`, then one byte every 75 ms (inside every per-read timeout these tests use)
/// until [`TRICKLE`] has passed or the other side has gone away.
pub fn trickle(mut out: impl Write, head: &[u8]) {
    let start = Instant::now();
    let mut next = head.to_vec();
    while start.elapsed() < TRICKLE {
        if out.write_all(&next).and_then(|()| out.flush()).is_err() {
            return;
        }
        next = vec![0];
        std::thread::sleep(Duration::from_millis(75));
    }
}
/// The operation ended at its deadline: not before it (less 50 ms of clock rounding) and at
/// most [`SLACK`] after it.
pub fn ended_at_deadline(elapsed: Duration, deadline: Duration, what: &str) {
    assert!(
        elapsed + Duration::from_millis(50) >= deadline && elapsed <= deadline + SLACK,
        "{what} ended after {elapsed:?}; its deadline was {deadline:?}"
    );
}

pub struct TestCertificate {
    pub der: Vec<u8>,
    pub pkcs8: Vec<u8>,
    pub sha256: String,
}
pub fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}
fn tlv(tag: u8, contents: &[u8]) -> Vec<u8> {
    let n = contents.len();
    let mut out = vec![tag];
    if n < 0x80 {
        out.push(n as u8);
    } else if n < 0x100 {
        out.extend([0x81, n as u8]);
    } else {
        out.extend([0x82, (n >> 8) as u8, n as u8]);
    }
    out.extend_from_slice(contents);
    out
}
fn seq(parts: &[Vec<u8>]) -> Vec<u8> {
    tlv(0x30, &parts.concat())
}
fn ed25519() -> Vec<u8> {
    seq(&[vec![0x06, 0x03, 0x2b, 0x65, 0x70]])
}
fn name(common_name: &str) -> Vec<u8> {
    let cn = seq(&[
        vec![0x06, 0x03, 0x55, 0x04, 0x03],
        tlv(0x0c, common_name.as_bytes()),
    ]);
    seq(&[tlv(0x31, &cn)])
}
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}
/// UTCTime for 1950..2050.
fn utc_time(unix: i64) -> Vec<u8> {
    let (year, month, day) = civil_from_days(unix.div_euclid(86_400));
    assert!((1950..2050).contains(&year));
    let s = unix.rem_euclid(86_400);
    let text = format!(
        "{:02}{month:02}{day:02}{:02}{:02}{:02}Z",
        year % 100,
        s / 3_600,
        s % 3_600 / 60,
        s % 60
    );
    tlv(0x17, text.as_bytes())
}
/// A self-signed Ed25519 certificate for `common_name` valid from `not_before` to
/// `not_after` (Unix seconds), with its PKCS#8 key. `seed` (1..128) selects the key.
pub fn certificate(
    common_name: &str,
    seed: u8,
    not_before: i64,
    not_after: i64,
) -> TestCertificate {
    assert!((1..128).contains(&seed));
    let key = SigningKey::from_bytes(&[seed; 32]);
    let public = [&[0u8][..], key.verifying_key().as_bytes()].concat();
    let spki = seq(&[ed25519(), tlv(0x03, &public)]);
    let tbs = seq(&[
        tlv(0xa0, &tlv(0x02, &[2])),
        tlv(0x02, &[seed]),
        ed25519(),
        name(common_name),
        seq(&[utc_time(not_before), utc_time(not_after)]),
        name(common_name),
        spki,
    ]);
    let signature = [&[0u8][..], &key.sign(&tbs).to_bytes()[..]].concat();
    let der = seq(&[tbs, ed25519(), tlv(0x03, &signature)]);
    let pkcs8 = [
        &[
            0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22,
            0x04, 0x20,
        ][..],
        &[seed; 32][..],
    ]
    .concat();
    TestCertificate {
        sha256: hex::encode(Sha256::digest(&der)),
        der,
        pkcs8,
    }
}
/// PEM armour for `der` under `label` (standard base64, 64-column lines).
pub fn pem(label: &str, der: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut b64 = String::new();
    for chunk in der.chunks(3) {
        let byte = |i: usize| u32::from(*chunk.get(i).unwrap_or(&0));
        let n = (byte(0) << 16) | (byte(1) << 8) | byte(2);
        for i in 0..4 {
            if i <= chunk.len() {
                b64.push(T[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                b64.push('=');
            }
        }
    }
    let lines: Vec<&str> = b64
        .as_bytes()
        .chunks(64)
        .map(|l| std::str::from_utf8(l).unwrap())
        .collect();
    format!(
        "-----BEGIN {label}-----\n{}\n-----END {label}-----\n",
        lines.join("\n")
    )
}
/// A certificate valid for an hour either side of now.
pub fn current(common_name: &str, seed: u8) -> TestCertificate {
    let now = unix_now();
    certificate(common_name, seed, now - 3_600, now + 3_600)
}
