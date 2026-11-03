use super::*;

/// Read a corpus's `vocabulary.json` once, require exactly those bytes to be
/// the digest its index binds (`vocabulary_sha256`), and parse them (W1 route
/// R8). A reader holding the index never reopens the vocabulary by path.
pub fn read_bound_vocabulary(corpus: &Path, vocabulary_sha256: &str) -> Result<Vocab> {
    let bytes = read_verified(
        &corpus.join("vocabulary.json"),
        vocabulary_sha256,
        "corpus vocabulary",
    )?;
    serde_json::from_slice(&bytes).context("parse corpus vocabulary")
}

/// A vocabulary is frozen for the whole build, so exact-genome tokenization
/// and its rejection reason can be reused across cuts. Limit both entry count
/// and retained payload; cache exhaustion affects speed only, never labels.
#[derive(Default)]
pub(crate) struct TokenCache {
    entries: HashMap<String, std::result::Result<Vec<usize>, String>>,
    payload_bytes: usize,
}
impl TokenCache {
    #[cfg(test)]
    fn encode(&mut self, vocab: &Vocab, genome: &str) -> std::result::Result<Vec<usize>, String> {
        self.encode_for_task(vocab, genome, "regression")
    }
    pub(super) fn encode_for_task(
        &mut self,
        vocab: &Vocab,
        genome: &str,
        task: &str,
    ) -> std::result::Result<Vec<usize>, String> {
        let key = format!("{task}\0{genome}");
        if let Some(result) = self.entries.get(&key) {
            return result.clone();
        }
        let result = match vocab.encode_for_task(genome, task) {
            Ok(tokens) if tokens.len() <= SEQUENCE => Ok(tokens),
            Ok(_) => Err("sequence_too_long".into()),
            Err(e) => Err(format!(
                "unrepresentable:{}",
                format!("{e:#}").split(':').next().unwrap_or("unknown")
            )),
        };
        let bytes = genome.len()
            + match &result {
                Ok(tokens) => tokens.len() * std::mem::size_of::<usize>(),
                Err(reason) => reason.len(),
            };
        if self.entries.len() < 65_536 && self.payload_bytes + bytes <= 32 * 1024 * 1024 {
            self.payload_bytes += bytes;
            self.entries.insert(key, result.clone());
        }
        result
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn cache_reuses_exact_tokens_and_preserves_unsnapped_rejections() {
        let vocab: Vocab =
            read_json(&Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"))
                .unwrap();
        let accepted = "(pipe X (standardize) (ridge))";
        let rejected = "(pipe X (rff_cos n=17 gamma=0.513 seed=7) (ridge))";
        let expected = vocab.encode(accepted).unwrap();
        let mut cache = TokenCache::default();
        assert_eq!(cache.encode(&vocab, accepted).unwrap(), expected);
        let mut returned = cache.encode(&vocab, accepted).unwrap();
        returned[0] = usize::MAX;
        assert_eq!(cache.encode(&vocab, accepted).unwrap(), expected);
        let reason = cache.encode(&vocab, rejected).unwrap_err();
        assert_eq!(reason, "unrepresentable:exact parameter has no frozen token; scores cannot be attached to a snapped program");
        assert_eq!(cache.encode(&vocab, rejected).unwrap_err(), reason);
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn exhausted_cache_preserves_results_without_retaining_more_payload() {
        let vocab: Vocab =
            read_json(&Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"))
                .unwrap();
        let genome = "(pipe X (standardize) (ridge))";
        let mut cache = TokenCache {
            payload_bytes: 32 * 1024 * 1024,
            ..Default::default()
        };
        assert_eq!(
            cache.encode(&vocab, genome).unwrap(),
            vocab.encode(genome).unwrap()
        );
        assert!(cache.entries.is_empty());
        assert_eq!(cache.payload_bytes, 32 * 1024 * 1024);
    }
}
