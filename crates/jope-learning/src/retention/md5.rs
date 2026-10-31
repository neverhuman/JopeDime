//! MD5 (RFC 1321), hand-written for one job: S3 Inventory's `manifest.checksum` and the manifest's
//! per-file `MD5checksum` are MD5 hex digests. Adding the `md-5` crate as a jope-learning
//! dependency would change `Cargo.lock` and so the ADR-0004 evaluator identity. MD5 is used only
//! to check that a downloaded report is the one S3 described, never as a content identity
//! (retention identities are sha256).

const S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

fn k(i: usize) -> u32 {
    // K[i] = floor(2^32 * |sin(i + 1)|), RFC 1321 §3.4.
    ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32
}

fn block(st: &mut [u32; 4], chunk: &[u8]) {
    let m: Vec<u32> = chunk
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    let [mut a, mut b, mut c, mut d] = *st;
    for (i, s) in S.iter().enumerate() {
        let (f, g) = match i / 16 {
            0 => ((b & c) | (!b & d), i),
            1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
            2 => (b ^ c ^ d, (3 * i + 5) % 16),
            _ => (c ^ (b | !d), (7 * i) % 16),
        };
        let t = d;
        d = c;
        c = b;
        b = b.wrapping_add(
            a.wrapping_add(f)
                .wrapping_add(k(i))
                .wrapping_add(m[g])
                .rotate_left(*s),
        );
        a = t;
    }
    st[0] = st[0].wrapping_add(a);
    st[1] = st[1].wrapping_add(b);
    st[2] = st[2].wrapping_add(c);
    st[3] = st[3].wrapping_add(d);
}

/// Incremental MD5, so multi-GB inventory files need not be hashed in one buffer.
pub struct Md5 {
    st: [u32; 4],
    buf: Vec<u8>,
    len: u64,
}

impl Default for Md5 {
    fn default() -> Self {
        Md5 {
            st: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            buf: Vec::with_capacity(64),
            len: 0,
        }
    }
}

impl Md5 {
    /// Performs the update operation and returns an error if it cannot be completed.
    pub fn update(&mut self, mut data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u64);
        if !self.buf.is_empty() {
            let take = (64 - self.buf.len()).min(data.len());
            self.buf.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.buf.len() == 64 {
                let b = std::mem::take(&mut self.buf);
                block(&mut self.st, &b);
            }
        }
        let mut chunks = data.chunks_exact(64);
        for c in &mut chunks {
            block(&mut self.st, c);
        }
        self.buf.extend_from_slice(chunks.remainder());
    }

    /// Performs the hex operation and returns an error if it cannot be completed.
    pub fn hex(mut self) -> String {
        let bits = self.len.wrapping_mul(8);
        let mut tail = std::mem::take(&mut self.buf);
        tail.push(0x80);
        while tail.len() % 64 != 56 {
            tail.push(0);
        }
        tail.extend_from_slice(&bits.to_le_bytes());
        for c in tail.chunks_exact(64) {
            block(&mut self.st, c);
        }
        self.st
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

/// Performs the md5 hex operation and returns an error if it cannot be completed.
pub fn md5_hex(data: &[u8]) -> String {
    let mut h = Md5::default();
    h.update(data);
    h.hex()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc1321_test_suite() {
        // RFC 1321 Appendix A.5.
        for (input, want) in [
            ("", "d41d8cd98f00b204e9800998ecf8427e"),
            ("a", "0cc175b9c0f1b6a831c399e269772661"),
            ("abc", "900150983cd24fb0d6963f7d28e17f72"),
            ("message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            (
                "abcdefghijklmnopqrstuvwxyz",
                "c3fcd3d76192e4007dfb496cca67e13b",
            ),
            (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                "d174ab98d277d9f5a5611c2c9f419d9f",
            ),
            (
                "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                "57edf4a22be3c955ac49da2e2107b67a",
            ),
        ] {
            assert_eq!(md5_hex(input.as_bytes()), want, "{input:?}");
        }
    }

    #[test]
    fn incremental_equals_one_shot() {
        let data: Vec<u8> = (0..10_000u32).map(|i| (i * 31 % 256) as u8).collect();
        for step in [1usize, 7, 63, 64, 65, 1000] {
            let mut h = Md5::default();
            for c in data.chunks(step) {
                h.update(c);
            }
            assert_eq!(h.hex(), md5_hex(&data), "step {step}");
        }
    }
}
