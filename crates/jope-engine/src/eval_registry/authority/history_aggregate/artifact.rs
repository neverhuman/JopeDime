use super::*;

pub(crate) fn open(a: &HistoryArtifact) -> Result<File> {
    ensure!(
        a.path.is_absolute() && valid_sha(&a.sha256) && fs::canonicalize(&a.path)? == a.path,
        "history artifact requires physical hash-bound path"
    );
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&a.path)?;
    ensure!(
        f.metadata()?.is_file() && f.metadata()?.len() == a.bytes,
        "history artifact type/size changed"
    );
    Ok(f)
}
pub(crate) fn digest_file(a: &HistoryArtifact, whole: Option<&mut Sha256>) -> Result<()> {
    let mut f = open(a)?;
    let mut h = Sha256::new();
    let mut whole = whole;
    let mut n = 0u64;
    let mut buf = [0u8; 1 << 20];
    loop {
        let got = f.read(&mut buf)?;
        if got == 0 {
            break;
        };
        n = n
            .checked_add(got as u64)
            .context("history length overflow")?;
        ensure!(n <= a.bytes, "history payload grew");
        h.update(&buf[..got]);
        if let Some(w) = whole.as_deref_mut() {
            w.update(&buf[..got]);
        }
    }
    ensure!(
        n == a.bytes && hex::encode(h.finalize()) == a.sha256,
        "history payload content changed"
    );
    Ok(())
}
pub(crate) fn metadata(a: &HistoryArtifact) -> Result<Vec<u8>> {
    ensure!(a.bytes <= META, "history metadata requires streamed index");
    let mut f = open(a)?;
    let mut b = Vec::new();
    Read::by_ref(&mut f).take(META + 1).read_to_end(&mut b)?;
    ensure!(
        b.len() as u64 == a.bytes && hash(&b) == a.sha256,
        "history metadata changed"
    );
    Ok(b)
}
