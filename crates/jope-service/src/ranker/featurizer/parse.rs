//! Match rank_prepare's leaf-group extraction, preserving the original source.
//! This permissive feature recipe is separate from executable-genome validation.

use anyhow::{Context, Result};
use feat_math::genome::GenomeIr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParserKind {
    GenomePipeline,
    ReferenceLeafGroups,
}

pub struct ParsedProgram<'a> {
    source: &'a str,
    operations: Vec<String>,
    parser: ParserKind,
}

impl ParsedProgram<'_> {
    pub fn source(&self) -> &str {
        self.source
    }

    pub fn operations(&self) -> &[String] {
        &self.operations
    }

    pub fn parser(&self) -> ParserKind {
        self.parser
    }

    /// Raw tokens retain duplicates and numeric spelling for RNK-04. A typed
    /// genome parameter map would lose the reference's repeated-token counts.
    pub fn raw_steps(&self) -> impl Iterator<Item = &str> {
        leaf_groups(self.source)
    }
}

pub fn parse(source: &str) -> Result<ParsedProgram<'_>> {
    // A parsed, unquoted ASCII flat pipeline has exactly the same leaf stages
    // as the reference regex. Quoted parentheses and Python's extra control
    // whitespace do not share that grammar and must retain the reference path.
    let common = source.is_ascii()
        && !source.contains(['"', '\''])
        && !source.chars().any(|c| matches!(c, '\u{1c}'..='\u{1f}'))
        && source.trim_start().starts_with("(pipe");
    if common {
        if let Ok(GenomeIr::Pipeline(pipeline)) = GenomeIr::parse(source) {
            return Ok(ParsedProgram {
                source,
                operations: pipeline.stages.into_iter().map(|s| s.op).collect(),
                parser: ParserKind::GenomePipeline,
            });
        }
    }
    let operations = leaf_groups(source)
        .map(|group| {
            tokens(group)
                .next()
                .map(str::to_owned)
                .context("rank_prepare leaf group has no operation")
        })
        .collect::<Result<_>>()?;
    Ok(ParsedProgram {
        source,
        operations,
        parser: ParserKind::ReferenceLeafGroups,
    })
}

/// Python str.split() also treats U+001C..U+001F as whitespace.
pub fn tokens(group: &str) -> impl Iterator<Item = &str> {
    group
        .split(|c: char| c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}'))
        .filter(|s| !s.is_empty())
}

fn leaf_groups(source: &str) -> impl Iterator<Item = &str> {
    let mut start = None;
    source.char_indices().filter_map(move |(index, c)| match c {
        '(' => {
            start = Some(index + 1);
            None
        }
        ')' => start
            .take()
            .filter(|&s| s != index)
            .map(|s| &source[s..index]),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_genome_parser_keeps_original_spelling_and_duplicate_tokens() {
        let source = " (pipe X (ridge lambda_frac=1e-3 lambda_frac=2 seed=9)) \n";
        let parsed = parse(source).unwrap();
        assert_eq!(parsed.parser(), ParserKind::GenomePipeline);
        assert_eq!(parsed.source(), source);
        assert_eq!(parsed.operations(), ["ridge"]);
        assert_eq!(
            tokens(parsed.raw_steps().next().unwrap()).collect::<Vec<_>>(),
            ["ridge", "lambda_frac=1e-3", "lambda_frac=2", "seed=9"]
        );
    }

    #[test]
    fn deeply_nested_reference_groups_need_no_recursive_parser() {
        let source = format!("{}ridge{}", "(".repeat(10000), ")".repeat(10000));
        let parsed = parse(&source).unwrap();
        assert_eq!(parsed.operations(), ["ridge"]);
        assert_eq!(parsed.parser(), ParserKind::ReferenceLeafGroups);
    }
}
