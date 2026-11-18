//! Pure-Rust, constant-time NIST SP 800-38D `AES-256-GCM` and RFC 5869 `HKDF-SHA256` primitives
//! built on `aes` + `sha2` with `Zeroizing` key material (`wasm32-unknown-unknown` compatible).

use aes::Aes256;
use cipher::{Array, BlockCipherEncrypt, KeyInit};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

/// Fixed 96-bit nonce length for `AES-256-GCM`.
pub const AES_GCM_NONCE_BYTES: usize = 12;
/// Fixed 128-bit authentication tag length for `AES-256-GCM`.
pub const AES_GCM_TAG_BYTES: usize = 16;

/// Compute RFC 2104 `HMAC-SHA256(key, data)`.
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut k_block = Zeroizing::new([0u8; 64]);
    if key.len() > 64 {
        let hashed = Sha256::digest(key);
        k_block[..32].copy_from_slice(&hashed);
    } else {
        k_block[..key.len()].copy_from_slice(key);
    }
    let mut ipad = Zeroizing::new([0x36u8; 64]);
    let mut opad = Zeroizing::new([0x5cu8; 64]);
    for i in 0..64 {
        ipad[i] ^= k_block[i];
        opad[i] ^= k_block[i];
    }
    let mut inner = Sha256::new();
    inner.update(ipad.as_slice());
    inner.update(data);
    let inner_digest = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(opad.as_slice());
    outer.update(inner_digest);
    outer.finalize().into()
}

/// Derive a 32-byte key via RFC 5869 `HKDF-SHA256(salt, ikm, info)`.
pub fn hkdf_sha256_32(salt: &[u8], ikm: &[u8], info: &[u8]) -> Zeroizing<[u8; 32]> {
    let prk = Zeroizing::new(hmac_sha256(salt, ikm));
    let mut expand_in = Vec::with_capacity(info.len() + 1);
    expand_in.extend_from_slice(info);
    expand_in.push(0x01);
    Zeroizing::new(hmac_sha256(&*prk, &expand_in))
}

/// Constant-time multiplication in `GF(2^128)` (`x^128 + x^7 + x^2 + x + 1`, NIST SP 800-38D §6.3).
fn ghash_mul(x: u128, y: u128) -> u128 {
    let mut z: u128 = 0;
    let mut v: u128 = y;
    for i in 0..128 {
        let xi = (x >> (127 - i)) & 1;
        let mask = 0u128.wrapping_sub(xi);
        z ^= v & mask;
        let lsb = v & 1;
        let red_mask = 0u128.wrapping_sub(lsb);
        v = (v >> 1) ^ (0xe100_0000_0000_0000_0000_0000_0000_0000u128 & red_mask);
    }
    z
}

fn ghash(h: u128, aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
    let mut acc: u128 = 0;
    let mut absorb = |data: &[u8]| {
        let mut chunks = data.chunks_exact(16);
        for blk in &mut chunks {
            let mut arr = [0u8; 16];
            arr.copy_from_slice(blk);
            acc = ghash_mul(acc ^ u128::from_be_bytes(arr), h);
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            let mut padded = [0u8; 16];
            padded[..rem.len()].copy_from_slice(rem);
            acc = ghash_mul(acc ^ u128::from_be_bytes(padded), h);
        }
    };
    absorb(aad);
    absorb(ciphertext);
    let aad_bits = (aad.len() as u64).wrapping_mul(8);
    let ct_bits = (ciphertext.len() as u64).wrapping_mul(8);
    let mut len_blk = [0u8; 16];
    len_blk[..8].copy_from_slice(&aad_bits.to_be_bytes());
    len_blk[8..].copy_from_slice(&ct_bits.to_be_bytes());
    acc = ghash_mul(acc ^ u128::from_be_bytes(len_blk), h);
    acc.to_be_bytes()
}

fn aes256_encrypt_block(cipher: &Aes256, input: &[u8; 16]) -> [u8; 16] {
    let mut block = Array::from(*input);
    cipher.encrypt_block(&mut block);
    block.into()
}

fn aes256_gctr_xor(
    cipher: &Aes256,
    nonce: &[u8; AES_GCM_NONCE_BYTES],
    start_ctr: u32,
    buf: &mut [u8],
) {
    let mut ctr = start_ctr;
    for chunk in buf.chunks_mut(16) {
        let mut cb = [0u8; 16];
        cb[..12].copy_from_slice(nonce);
        cb[12..].copy_from_slice(&ctr.to_be_bytes());
        ctr = ctr.wrapping_add(1);
        let ks = aes256_encrypt_block(cipher, &cb);
        for (b, k) in chunk.iter_mut().zip(ks.iter()) {
            *b ^= *k;
        }
    }
}

fn constant_time_eq_16(a: &[u8; 16], b: &[u8; 16]) -> bool {
    let mut diff = 0u8;
    for i in 0..16 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Seal `plaintext` with `AES-256-GCM` under `key`, `nonce`, and `aad`, returning `ciphertext || tag`.
pub fn aes256_gcm_seal(
    key: &[u8; 32],
    nonce: &[u8; AES_GCM_NONCE_BYTES],
    aad: &[u8],
    plaintext: &[u8],
) -> Vec<u8> {
    let key_arr = Array::from(*key);
    let cipher = Aes256::new(&key_arr);
    let h = u128::from_be_bytes(aes256_encrypt_block(&cipher, &[0u8; 16]));

    let mut out = Vec::with_capacity(plaintext.len() + AES_GCM_TAG_BYTES);
    out.extend_from_slice(plaintext);
    aes256_gctr_xor(&cipher, nonce, 2, &mut out);

    let s = ghash(h, aad, &out);
    let mut j0 = [0u8; 16];
    j0[..12].copy_from_slice(nonce);
    j0[12..].copy_from_slice(&1u32.to_be_bytes());
    let j0_enc = aes256_encrypt_block(&cipher, &j0);
    let mut tag = [0u8; AES_GCM_TAG_BYTES];
    for i in 0..16 {
        tag[i] = s[i] ^ j0_enc[i];
    }
    out.extend_from_slice(&tag);
    out
}

/// Open and authenticate `ciphertext_and_tag` (`AES-256-GCM`) in constant time.
pub fn aes256_gcm_open(
    key: &[u8; 32],
    nonce: &[u8; AES_GCM_NONCE_BYTES],
    aad: &[u8],
    ciphertext_and_tag: &[u8],
) -> Result<Zeroizing<Vec<u8>>, &'static str> {
    if ciphertext_and_tag.len() < AES_GCM_TAG_BYTES {
        return Err("ciphertext shorter than 16-byte AES-256-GCM tag");
    }
    let ct_len = ciphertext_and_tag.len() - AES_GCM_TAG_BYTES;
    let (ct, tag_slice) = ciphertext_and_tag.split_at(ct_len);
    let mut received_tag = [0u8; AES_GCM_TAG_BYTES];
    received_tag.copy_from_slice(tag_slice);

    let key_arr = Array::from(*key);
    let cipher = Aes256::new(&key_arr);
    let h = u128::from_be_bytes(aes256_encrypt_block(&cipher, &[0u8; 16]));

    let s = ghash(h, aad, ct);
    let mut j0 = [0u8; 16];
    j0[..12].copy_from_slice(nonce);
    j0[12..].copy_from_slice(&1u32.to_be_bytes());
    let j0_enc = aes256_encrypt_block(&cipher, &j0);
    let mut expected_tag = [0u8; AES_GCM_TAG_BYTES];
    for i in 0..16 {
        expected_tag[i] = s[i] ^ j0_enc[i];
    }
    if !constant_time_eq_16(&expected_tag, &received_tag) {
        return Err("AES-256-GCM authentication tag verification failed");
    }

    let mut pt = Zeroizing::new(ct.to_vec());
    aes256_gctr_xor(&cipher, nonce, 2, pt.as_mut_slice());
    Ok(pt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc5869_hkdf_sha256_test_case_1_matches() {
        let ikm = hex::decode("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b").unwrap();
        let salt = hex::decode("000102030405060708090a0b0c").unwrap();
        let info = hex::decode("f0f1f2f3f4f5f6f7f8f9").unwrap();
        let okm = hkdf_sha256_32(&salt, &ikm, &info);
        assert_eq!(
            hex::encode(*okm),
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
        );
    }

    #[test]
    fn nist_sp800_38d_aes256_gcm_test_case_16_matches() {
        let key: [u8; 32] =
            hex::decode("feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308")
                .unwrap()
                .try_into()
                .unwrap();
        let nonce: [u8; 12] = hex::decode("cafebabefacedbaddecaf888")
            .unwrap()
            .try_into()
            .unwrap();
        let pt = hex::decode(
            "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72\
             1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39",
        )
        .unwrap();
        let aad = hex::decode("feedfacedeadbeeffeedfacedeadbeefabaddad2").unwrap();
        let sealed = aes256_gcm_seal(&key, &nonce, &aad, &pt);
        let expected_ct_tag = hex::decode(
            "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa\
             8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662\
             76fc6ece0f4e1768cddf8853bb2d551b",
        )
        .unwrap();
        assert_eq!(sealed, expected_ct_tag);

        let opened = aes256_gcm_open(&key, &nonce, &aad, &sealed).unwrap();
        assert_eq!(&*opened, &pt);

        let mut tampered = sealed.clone();
        tampered[0] ^= 0x01;
        assert!(aes256_gcm_open(&key, &nonce, &aad, &tampered).is_err());
    }
}
