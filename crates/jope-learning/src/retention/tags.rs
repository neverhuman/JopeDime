//! Write-time S3 object tags (strategy §4 "Tag at write time"): `tier`, `run_id`, `family`,
//! `qualified`, `superseded_by`, `kind`.
//!
//! Tags cost $0.01 per 10k tags per month, so writers stamp packs, pack indexes and manifests, not
//! every small result object. The constructor enforces the retention invariants rather than
//! trusting each writer:
//! - `qualified=true`, and kinds `split` and `evidence`, are always `tier=T0`;
//! - nothing is tagged `T4` at write time: expiry only comes from an owner-approved plan;
//! - `superseded_by` names a champion only when that champion is qualified (§8.5); otherwise it
//!   is `none`.

use anyhow::{bail, ensure, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    T0,
    T1,
    T2,
    T3,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::T0 => "T0",
            Tier::T1 => "T1",
            Tier::T2 => "T2",
            Tier::T3 => "T3",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectKind {
    Result,
    Log,
    Checkpoint,
    Split,
    Evidence,
    Pack,
    PackIndex,
    Manifest,
}

impl ObjectKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ObjectKind::Result => "result",
            ObjectKind::Log => "log",
            ObjectKind::Checkpoint => "checkpoint",
            ObjectKind::Split => "split",
            ObjectKind::Evidence => "evidence",
            ObjectKind::Pack => "pack",
            ObjectKind::PackIndex => "pack-index",
            ObjectKind::Manifest => "manifest",
        }
    }
}

/// The champion a result is superseded by; it only counts when the champion is qualified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Supersession {
    pub champion_run_id: String,
    pub champion_qualified: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteTags {
    pairs: Vec<(String, String)>,
}

/// S3 tag rules: key 1..=128 and value 0..=256 characters from letters, digits, spaces and
/// `+ - = . _ : / @`; no `aws:` prefix; at most 10 tags per object.
fn check_tag(k: &str, v: &str) -> Result<()> {
    let ok = |s: &str| {
        s.chars()
            .all(|c| c.is_ascii_alphanumeric() || " +-=._:/@".contains(c))
    };
    ensure!(
        !k.is_empty() && k.len() <= 128 && ok(k),
        "tag key {k:?} is not a valid S3 tag key"
    );
    ensure!(
        !k.starts_with("aws:"),
        "tag key {k:?} uses the reserved aws: prefix"
    );
    ensure!(
        v.len() <= 256 && ok(v),
        "tag {k}: value {v:?} is not a valid S3 tag value"
    );
    Ok(())
}

impl WriteTags {
    pub fn new(
        tier: Tier,
        run_id: &str,
        family: &str,
        qualified: bool,
        superseded_by: Option<&Supersession>,
        kind: ObjectKind,
    ) -> Result<WriteTags> {
        ensure!(
            !run_id.is_empty() && !family.is_empty(),
            "run_id and family are required"
        );
        if (qualified || matches!(kind, ObjectKind::Split | ObjectKind::Evidence))
            && tier != Tier::T0
        {
            bail!(
                "a {} object{} must be tier T0, not {}",
                kind.as_str(),
                if qualified {
                    " with qualified=true"
                } else {
                    ""
                },
                tier.as_str()
            );
        }
        let superseded = match superseded_by {
            None => "none".to_string(),
            Some(s) if s.champion_qualified => {
                ensure!(
                    !s.champion_run_id.is_empty(),
                    "superseded_by needs a champion run_id"
                );
                s.champion_run_id.clone()
            }
            Some(s) => bail!(
                "superseded_by {} refused: the champion is not qualified (strategy §8.5)",
                s.champion_run_id
            ),
        };
        let pairs = vec![
            ("tier".to_string(), tier.as_str().to_string()),
            ("run_id".to_string(), run_id.to_string()),
            ("family".to_string(), family.to_string()),
            ("qualified".to_string(), qualified.to_string()),
            ("superseded_by".to_string(), superseded),
            ("kind".to_string(), kind.as_str().to_string()),
        ];
        for (k, v) in &pairs {
            check_tag(k, v)?;
        }
        Ok(WriteTags { pairs })
    }

    pub fn pairs(&self) -> &[(String, String)] {
        &self.pairs
    }

    /// The `x-amz-tagging` header value: URL-encoded `k=v` pairs joined by `&`.
    pub fn header_value(&self) -> String {
        let enc = |s: &str| {
            s.bytes()
                .map(|b| match b {
                    b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                        (b as char).to_string()
                    }
                    _ => format!("%{b:02X}"),
                })
                .collect::<String>()
        };
        self.pairs
            .iter()
            .map(|(k, v)| format!("{}={}", enc(k), enc(v)))
            .collect::<Vec<_>>()
            .join("&")
    }
}
