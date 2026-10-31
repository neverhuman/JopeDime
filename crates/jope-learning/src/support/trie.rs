use super::{FrozenSupport, Program};
use anyhow::{ensure, Result};

/// Sorted complete leaves use O(total token bytes) storage. Prefix child queries
/// skip subtrees by binary search, with no eager node cap or cross-context KV.
pub struct Trie<'a> {
    /// SHA-256 digest identifying the associated support.
    pub support_sha256: &'a str,
    /// SHA-256 digest identifying the associated vocabulary.
    pub vocabulary_sha256: &'a str,
    /// Upper bound on seq.
    pub max_seq: usize,
    leaves: Vec<&'a Program>,
}

impl<'a> Trie<'a> {
    /// Constructs or opens a new value from the supplied inputs.
    pub fn new(support: &'a FrozenSupport) -> Result<Self> {
        support.verify()?;
        let mut leaves: Vec<_> = support.programs.iter().collect();
        leaves.sort_by(|a, b| a.tokens.cmp(&b.tokens));
        ensure!(
            leaves.windows(2).all(|p| p[0].tokens != p[1].tokens),
            "duplicate token leaf"
        );
        Ok(Self {
            support_sha256: &support.support_sha256,
            vocabulary_sha256: &support.vocabulary.sha256,
            max_seq: support.vocabulary.max_seq,
            leaves,
        })
    }

    /// Validate exact complete-leaf identity, including BOS/EOS and the deployed length bound.
    pub fn validate_leaf(&self, tokens: &[u16]) -> Result<()> {
        ensure!(
            tokens.len() >= 3
                && tokens.len() <= self.max_seq
                && tokens.first() == Some(&1)
                && tokens.last() == Some(&2)
                && self.contains(tokens),
            "complete registered BOS/EOS leaf within the frozen sequence cap required"
        );
        Ok(())
    }

    /// Performs the leaf count operation.
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    /// Performs the contains operation.
    pub fn contains(&self, tokens: &[u16]) -> bool {
        self.leaves
            .binary_search_by(|p| p.tokens.as_slice().cmp(tokens))
            .is_ok()
    }

    /// Performs the children operation and returns an error if it cannot be completed.
    pub fn children(&self, prefix: &[u16]) -> Result<Vec<u16>> {
        ensure!(
            prefix.first() == Some(&1) && !prefix[1..].contains(&1),
            "exact BOS prefix required"
        );
        ensure!(prefix.len() <= self.max_seq, "prefix exceeds model cap");
        if prefix.contains(&2) {
            ensure!(
                prefix.last() == Some(&2) && self.contains(prefix),
                "only registered complete EOS leaf allowed"
            );
            return Ok(Vec::new());
        }
        let mut pos = self
            .leaves
            .partition_point(|p| p.tokens.as_slice() < prefix);
        let mut children = Vec::new();
        while let Some(program) = self.leaves.get(pos) {
            if !program.tokens.starts_with(prefix) {
                break;
            }
            ensure!(program.tokens.len() > prefix.len(), "nonterminal leaf");
            let token = program.tokens[prefix.len()];
            children.push(token);
            let mut upper = prefix.to_vec();
            upper.push(token + 1);
            pos += self.leaves[pos..].partition_point(|p| p.tokens.as_slice() < upper.as_slice());
        }
        ensure!(
            !children.is_empty() && prefix.len() < self.max_seq,
            "prefix outside frozen support"
        );
        Ok(children)
    }
}
