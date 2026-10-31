use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::canon::{canonical_sha256, sha256_hex};

const NAMES: [&str; 26] = [
    "PAD",
    "BOS",
    "EOS",
    "OP_BEGIN",
    "OP_END",
    "KEY_BEGIN",
    "KEY_END",
    "STR_BEGIN",
    "STR_END",
    "ID_BEGIN",
    "ID_END",
    "RAW_BEGIN",
    "RAW_END",
    "PIPE_OPEN",
    "STAGE_OPEN",
    "CLOSE",
    "SYM_AFFINE_OPEN",
    "SYM_RAW_OPEN",
    "SYM_CLOSE",
    "PORTFOLIO_OPEN",
    "MEMBER_OPEN",
    "BLEND_AUTO",
    "BLEND_ONE",
    "TRUE",
    "FALSE",
    "COL",
];
const PIECES: [&str; 26] = [
    "",
    "",
    "",
    "",
    "",
    " ",
    "=",
    "",
    "",
    "",
    "",
    "",
    "",
    "(pipe X",
    " (",
    ")",
    "(sym affine \"",
    "(sym raw \"",
    "\")",
    "(portfolio",
    " (member ",
    " (blend folds=5 guard=clip kind=nnls repeats=auto))",
    " (blend folds=5 guard=clip kind=nnls repeats=1))",
    "true",
    "false",
    "c",
];

/// The deployed 282-token byte-piece vocabulary. Data-name vocabularies are refused;
/// supporting a new model vocabulary requires a separately pinned implementation.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Vocabulary {
    /// Sha-256 recorded for this vocabulary.
    pub sha256: String,
    /// Upper bound on seq.
    pub max_seq: usize,
}

/// Performs the fixed vocabulary SHA-256 operation and returns an error if it cannot be completed.
pub fn fixed_vocabulary_sha256() -> Result<String> {
    let mut table = Vec::with_capacity(282);
    for (id, name) in NAMES.iter().enumerate() {
        let class = if id < 3 {
            "SPECIAL"
        } else if id < 13 {
            "MARKER"
        } else {
            "STRUCT"
        };
        table.push(json!({"class":class,"name":name,"piece_hex":hex::encode(PIECES[id])}));
    }
    for byte in 0u16..256 {
        table.push(json!({"class":"BYTE","name":format!("0x{byte:02x}"),"piece_hex":format!("{byte:02x}")}));
    }
    canonical_sha256(&table)
}

impl Vocabulary {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.sha256 == fixed_vocabulary_sha256()?,
            "deployed vocabulary identity mismatch"
        );
        ensure!(
            (3..=640).contains(&self.max_seq),
            "model sequence cap outside 3..640"
        );
        Ok(())
    }

    /// Performs the detokenize operation and returns an error if it cannot be completed.
    pub fn detokenize(&self, tokens: &[u16]) -> Result<String> {
        let mut bytes = Vec::new();
        for &token in tokens {
            ensure!(token < 282, "token outside deployed vocabulary");
            if token < 26 {
                bytes.extend_from_slice(PIECES[token as usize].as_bytes());
            } else {
                bytes.push((token - 26) as u8);
            }
        }
        Ok(String::from_utf8(bytes)?)
    }

    /// Native Rust parser/printer supplies canonicality. The spelling below preserves
    /// bytes and numbers and matches the pinned PolicyLM tokenizer's pipeline form.
    pub fn tokenize(&self, dsl: &str) -> Result<Vec<u16>> {
        self.validate()?;
        ensure!(dsl.len() <= 8192, "bounded pipeline text required");
        ensure!(
            jope_pipeline::seeds::canonical_pipeline(dsl)? == dsl,
            "noncanonical pipeline"
        );
        let bytes = dsl.as_bytes();
        ensure!(dsl.starts_with("(pipe X"), "pipeline form required");
        let mut tokens = vec![1, 13];
        let mut pos = 7;
        while bytes.get(pos..pos + 2) == Some(b" (") {
            tokens.push(14);
            pos += 2;
            let end = atom_end(bytes, pos)?;
            spell(&mut tokens, 3, &bytes[pos..end], 4);
            pos = end;
            while bytes.get(pos) == Some(&b' ') {
                pos += 1;
                let end = atom_end(bytes, pos)?;
                let atom = &bytes[pos..end];
                let equal = atom
                    .iter()
                    .position(|x| *x == b'=')
                    .ok_or_else(|| anyhow::anyhow!("parameter without equals"))?;
                ensure!(equal > 0, "empty parameter key");
                spell(&mut tokens, 5, &atom[..equal], 6);
                let value = &atom[equal + 1..];
                if value == b"true" {
                    tokens.push(23);
                } else if value == b"false" {
                    tokens.push(24);
                } else if is_plain_number(value) {
                    tokens.extend(value.iter().map(|b| 26 + u16::from(*b)));
                } else {
                    spell(&mut tokens, 7, value, 8);
                }
                pos = end;
            }
            ensure!(bytes.get(pos) == Some(&b')'), "stage terminator required");
            tokens.push(15);
            pos += 1;
        }
        ensure!(
            bytes.get(pos) == Some(&b')') && pos + 1 == bytes.len(),
            "complete pipeline required"
        );
        tokens.extend([15, 2]);
        ensure!(
            tokens.len() <= self.max_seq,
            "program exceeds model sequence cap"
        );
        ensure!(self.detokenize(&tokens)? == dsl, "token roundtrip mismatch");
        Ok(tokens)
    }

    /// Performs the program operation and returns an error if it cannot be completed.
    pub fn program(&self, dsl: &str) -> Result<super::Program> {
        Ok(super::Program {
            dsl: dsl.into(),
            program_sha256: sha256_hex(dsl.as_bytes()),
            tokens: self.tokenize(dsl)?,
        })
    }
}

fn spell(tokens: &mut Vec<u16>, begin: u16, text: &[u8], end: u16) {
    tokens.push(begin);
    tokens.extend(text.iter().map(|b| 26 + u16::from(*b)));
    tokens.push(end);
}

fn atom_end(bytes: &[u8], start: usize) -> Result<usize> {
    let mut pos = start;
    let mut quoted = false;
    while let Some(&byte) = bytes.get(pos) {
        if !quoted && matches!(byte, b' ' | b'(' | b')') {
            break;
        }
        if quoted && byte == b'\\' {
            pos += 2;
            continue;
        }
        if byte == b'"' {
            quoted = !quoted;
        }
        pos += 1;
    }
    ensure!(
        pos > start && pos <= bytes.len() && !quoted,
        "complete parameter atom required"
    );
    Ok(pos)
}

fn is_plain_number(value: &[u8]) -> bool {
    let value = value.strip_prefix(b"-").unwrap_or(value);
    if value.is_empty() {
        return false;
    }
    let mut parts = value.split(|x| *x == b'.');
    let first = parts.next().unwrap_or_default();
    if first.is_empty() || !first.iter().all(u8::is_ascii_digit) {
        return false;
    }
    if let Some(second) = parts.next() {
        if second.is_empty() || !second.iter().all(u8::is_ascii_digit) {
            return false;
        }
    }
    parts.next().is_none()
}

pub(super) fn validate_program(vocab: &Vocabulary, program: &super::Program) -> Result<()> {
    if program.program_sha256 != sha256_hex(program.dsl.as_bytes()) {
        bail!("canonical program SHA mismatch");
    }
    ensure!(
        vocab.tokenize(&program.dsl)? == program.tokens,
        "noncanonical token sequence"
    );
    Ok(())
}
