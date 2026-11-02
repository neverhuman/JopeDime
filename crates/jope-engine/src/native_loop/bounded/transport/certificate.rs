use super::*;

pub(crate) fn provider() -> Arc<CryptoProvider> {
    Arc::new(aws_lc_rs::default_provider())
}

/// This host's certificate and PKCS#8 private key.
pub struct HostCredential {
    pub(crate) certificate: CertificateDer<'static>,
    pub(crate) key: PrivateKeyDer<'static>,
    certificate_sha256: String,
}
impl HostCredential {
    /// A certificate (DER, or PEM with one certificate) and a DER PKCS#8 private key.
    pub fn from_der(certificate: Vec<u8>, pkcs8_key: Vec<u8>) -> Result<Self> {
        let certificate = certificate_der(&certificate)?;
        certificate_validity(&certificate)?;
        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(pkcs8_key));
        provider()
            .key_provider
            .load_private_key(key.clone_key())
            .map_err(|e| anyhow!("unsupported private key: {e}"))?;
        Ok(Self {
            certificate_sha256: hash(&certificate),
            certificate: CertificateDer::from(certificate),
            key,
        })
    }
    /// Load the certificate (DER, or PEM with one certificate) and the PKCS#8 private key
    /// (PEM `PRIVATE KEY` or DER). Neither may be a symlink; the key must be readable by its
    /// owner only.
    pub fn load(certificate: &Path, key: &Path) -> Result<Self> {
        let certificate = read_credential(certificate, false)?;
        let key = read_credential(key, true)?;
        let key = if key.starts_with(b"-----BEGIN") {
            match PrivateKeyDer::from_pem_slice(&key)
                .map_err(|e| anyhow!("private key PEM: {e:?}"))?
            {
                PrivateKeyDer::Pkcs8(k) => k.secret_pkcs8_der().to_vec(),
                _ => bail!("the private key must be PKCS#8 (`PRIVATE KEY`)"),
            }
        } else {
            key
        };
        Self::from_der(certificate, key)
    }
    pub fn certificate_sha256(&self) -> &str {
        &self.certificate_sha256
    }
}
fn read_credential(path: &Path, private: bool) -> Result<Vec<u8>> {
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let meta = f.metadata()?;
    ensure!(meta.is_file(), "{} is not a regular file", path.display());
    ensure!(
        !private || meta.mode() & 0o077 == 0,
        "private key {} must be readable by its owner only",
        path.display()
    );
    let mut b = Vec::new();
    f.take(MAX_CREDENTIAL_BYTES + 1).read_to_end(&mut b)?;
    ensure!(
        b.len() as u64 <= MAX_CREDENTIAL_BYTES,
        "{} exceeds {MAX_CREDENTIAL_BYTES} bytes",
        path.display()
    );
    Ok(b)
}

fn next_field<'a>(fields: &mut &'a [u8]) -> Result<(u8, &'a [u8])> {
    let ((tag, contents), rest) = der_element(fields)?;
    *fields = rest;
    Ok((tag, contents))
}
/// `(not_before, not_after)` of a DER certificate, in Unix seconds.
pub fn certificate_validity(der: &[u8]) -> Result<(i64, i64)> {
    let ((tag, certificate), rest) = der_element(der)?;
    ensure!(tag == 0x30 && rest.is_empty(), "certificate framing");
    let ((tag, tbs), _) = der_element(certificate)?;
    ensure!(tag == 0x30, "TBSCertificate framing");
    let mut fields = tbs;
    let (mut tag, _) = next_field(&mut fields)?;
    if tag == 0xa0 {
        tag = next_field(&mut fields)?.0; // the explicit version, then the serial number
    }
    ensure!(tag == 0x02, "certificate serial number");
    ensure!(
        next_field(&mut fields)?.0 == 0x30,
        "certificate signature algorithm"
    );
    ensure!(next_field(&mut fields)?.0 == 0x30, "certificate issuer");
    let (tag, validity) = next_field(&mut fields)?;
    ensure!(tag == 0x30, "certificate validity");
    let ((t0, before), rest) = der_element(validity)?;
    let ((t1, after), rest) = der_element(rest)?;
    ensure!(rest.is_empty(), "certificate validity holds two times");
    let (not_before, not_after) = (der_time(t0, before)?, der_time(t1, after)?);
    ensure!(not_before <= not_after, "certificate validity is inverted");
    Ok((not_before, not_after))
}
/// UTCTime (`YYMMDDHHMMSSZ`) or GeneralizedTime (`YYYYMMDDHHMMSSZ`) to Unix seconds.
fn der_time(tag: u8, bytes: &[u8]) -> Result<i64> {
    let s = std::str::from_utf8(bytes).context("certificate time is not ASCII")?;
    ensure!(
        s.len() >= 2 && s.ends_with('Z') && s[..s.len() - 1].bytes().all(|b| b.is_ascii_digit()),
        "certificate time `{s}` is not a UTC time"
    );
    let n = |r: std::ops::Range<usize>| -> i64 { s[r].parse().unwrap_or(-1) };
    let (year, at) = match (tag, s.len()) {
        (0x17, 13) => {
            let yy = n(0..2);
            (if yy >= 50 { 1900 + yy } else { 2000 + yy }, 2)
        }
        (0x18, 15) => (n(0..4), 4),
        _ => bail!("unsupported certificate time encoding"),
    };
    let (month, day) = (n(at..at + 2), n(at + 2..at + 4));
    let (hour, minute, second) = (n(at + 4..at + 6), n(at + 6..at + 8), n(at + 8..at + 10));
    ensure!(
        (1..=12).contains(&month)
            && (1..=31).contains(&day)
            && (0..24).contains(&hour)
            && (0..60).contains(&minute)
            && (0..60).contains(&second),
        "certificate time `{s}` is out of range"
    );
    Ok(days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second)
}
/// Days since 1970-01-01 of a proleptic Gregorian date.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The certificates one side accepts (`certificate SHA-256 -> host`).
#[derive(Debug)]
pub(crate) struct Pinned {
    pub(crate) accepted: BTreeMap<String, String>,
    pub(crate) provider: Arc<CryptoProvider>,
}
impl Pinned {
    fn check(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        now: UnixTime,
    ) -> std::result::Result<(), rustls::Error> {
        let bad = rustls::Error::InvalidCertificate;
        if !intermediates.is_empty() {
            return Err(rustls::Error::General(
                "a pinned host certificate carries no chain".into(),
            ));
        }
        if !self.accepted.contains_key(&hash(end_entity.as_ref())) {
            return Err(bad(CertificateError::ApplicationVerificationFailure));
        }
        let (not_before, not_after) = certificate_validity(end_entity.as_ref())
            .map_err(|_| bad(CertificateError::BadEncoding))?;
        let now = now.as_secs() as i64;
        if now < not_before {
            return Err(bad(CertificateError::NotValidYet));
        }
        if now > not_after {
            return Err(bad(CertificateError::Expired));
        }
        Ok(())
    }
    fn tls12(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }
    fn tls13(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }
    fn schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}
impl ServerCertVerifier for Pinned {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, rustls::Error> {
        self.check(end_entity, intermediates, now)
            .map(|()| ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        self.tls12(message, cert, dss)
    }
    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        self.tls13(message, cert, dss)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.schemes()
    }
}
impl ClientCertVerifier for Pinned {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }
    fn client_auth_mandatory(&self) -> bool {
        true
    }
    fn verify_client_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        now: UnixTime,
    ) -> std::result::Result<ClientCertVerified, rustls::Error> {
        self.check(end_entity, intermediates, now)
            .map(|()| ClientCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        self.tls12(message, cert, dss)
    }
    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        self.tls13(message, cert, dss)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.schemes()
    }
}
