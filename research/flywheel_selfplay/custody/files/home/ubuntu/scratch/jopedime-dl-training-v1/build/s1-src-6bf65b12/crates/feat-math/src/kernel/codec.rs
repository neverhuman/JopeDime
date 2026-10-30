//! Canonical little-endian encoding. Lengths and allocation limits are checked
//! before slicing or allocating. No floats pass through decimal formatting.
use super::{Error, Result};
use sha2::{Digest, Sha256};

pub(crate) const MAX_BYTES: usize = 16 * 1024 * 1024;

#[derive(Default)]
pub(crate) struct Writer(Vec<u8>);
impl Writer {
    pub fn raw(&mut self, value: &[u8]) -> Result<()> {
        let end = self
            .0
            .len()
            .checked_add(value.len())
            .filter(|&n| n <= MAX_BYTES)
            .ok_or_else(|| Error::Resource("codec bytes".into()))?;
        if end > self.0.capacity() {
            let capacity = end
                .max(self.0.capacity().saturating_mul(2).max(64))
                .min(MAX_BYTES);
            self.0
                .try_reserve_exact(capacity - self.0.len())
                .map_err(|_| Error::Resource("codec allocation".into()))?;
        }
        self.0.extend_from_slice(value);
        Ok(())
    }
    pub fn byte(&mut self, value: u8) -> Result<()> {
        self.raw(&[value])
    }
    pub fn word(&mut self, value: u64) -> Result<()> {
        self.raw(&value.to_le_bytes())
    }
    pub fn text(&mut self, value: &str) -> Result<()> {
        self.word(value.len() as u64)?;
        self.raw(value.as_bytes())
    }
    pub fn finish(self) -> Vec<u8> {
        self.0
    }
}

pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() > MAX_BYTES {
            return Err(Error::Resource("codec bytes".into()));
        }
        Ok(Self { bytes, position: 0 })
    }
    pub fn take(&mut self, size: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(size)
            .ok_or_else(|| Error::Bounds("codec length overflow".into()))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| Error::Syntax("truncated codec".into()))?;
        self.position = end;
        Ok(value)
    }
    pub fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn word(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn length(&mut self) -> Result<usize> {
        let n =
            usize::try_from(self.word()?).map_err(|_| Error::Resource("codec length".into()))?;
        // Every encoded item consumes at least one byte. Also prevents an
        // attacker-controlled capacity allocation from a truncated payload.
        if n > self.bytes.len() - self.position {
            return Err(Error::Bounds("codec count exceeds remaining bytes".into()));
        }
        Ok(n)
    }
    pub fn text(&mut self) -> Result<String> {
        let n = self.length()?;
        std::str::from_utf8(self.take(n)?)
            .map(str::to_owned)
            .map_err(|_| Error::Syntax("codec UTF-8".into()))
    }
    pub fn finish(self) -> Result<()> {
        if self.position == self.bytes.len() {
            Ok(())
        } else {
            Err(Error::Syntax("trailing codec bytes".into()))
        }
    }
}

/// Domain separation prevents programs, bindings, states, evaluations and
/// emitted artifacts from accidentally sharing identities for identical data.
pub fn identity(domain: &str, bytes: &[u8]) -> String {
    // Hash the same length-prefixed encoding without copying the payload.
    let mut hash = Sha256::new();
    for value in [
        b"feat-math.identity.v1".as_slice(),
        domain.as_bytes(),
        bytes,
    ] {
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value);
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_rejects_growth_before_exceeding_budget() {
        let mut writer = Writer::default();
        writer.raw(&vec![0; MAX_BYTES]).unwrap();
        assert!(matches!(writer.byte(0), Err(Error::Resource(_))));
        assert_eq!(writer.0.len(), MAX_BYTES);
        assert!(writer.0.capacity() <= MAX_BYTES);
    }
}
