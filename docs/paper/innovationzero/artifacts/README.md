# InnovationZero archival supplement

These files preserve the complete engineering challenge, four complete frozen Rust
submissions, and the supporting receipts and exploratory benchmark records supplied
with the paper specification. They are source material for the publication, not new
InnovationZero experimental results.

The [artifact manifest](../evidence/artifact-manifest.json) records every payload's
byte length, SHA-256, original repository path and immutable source commit. Each
payload was extracted using the specification's declared byte length, checked against
its declared hash, and compared with its pinned Git object. Eighteen payloads match
exactly. The seven benchmark payloads each contain one extra trailing line feed in
the supplied appendix; all other bytes match upstream. Both identities are recorded,
and the uploaded bytes are preserved. The participant packet matches the exact
inter-marker span of the master. No original source or receipt was edited,
reformatted, repaired or normalized during this publication build.

The local Git attributes disable line-ending conversion and retain intentional
trailing boundary newlines. This narrow archival exception leaves source hashes
authoritative while ordinary authored prose and generated figures follow normal
whitespace checks.

## Reading order

| Material | Location | Interpretation |
| --- | --- | --- |
| Complete organizer master | [challenge/master.md](challenge/master.md) | Participant challenge plus organizer material; not the neutral participant input |
| Complete participant packet | [challenge/participant.md](challenge/participant.md) | Exact inter-marker bytes; campaign-specific cards would be separate artifacts |
| Opus source | [sources/Opus5p5_Extra.rs](sources/Opus5p5_Extra.rs) | Complete frozen historical source |
| Astra source | [sources/GPT6_Astra_Max.rs](sources/GPT6_Astra_Max.rs) | Complete frozen historical source |
| Sol source | [sources/GPT6p1_Sol_Max.rs](sources/GPT6p1_Sol_Max.rs) | Complete frozen historical source |
| Gemini-labelled source | [sources/Gemini4_argon_high.rs](sources/Gemini4_argon_high.rs) | Complete owner-supplied PR #407 source; provider/session identity remains unverified |
| Authoring and verification records | [receipts/](receipts/) | Historical reports; estimates and missing fields retain their original status |
| Four-artifact exploratory comparison | [benchmark/](benchmark/) | Training-family inner-CV evidence, not prospective InnovationZero synthesis evidence |

The printable source supplement includes both challenge files and all four complete
Rust files. The exact machine-readable originals remain authoritative: typesetting
can wrap lines and escape glyphs without changing the archived files.

## Exact source identities

The master, Opus, Astra, Sol and their receipts are pinned to main snapshot
`bdf9441aaa17f87dcb37053d33b185ba2217bf17`. Gemini and the four-artifact benchmark
records are pinned separately to PR #407 head
`584713ed3f512057361460567bb56d620162286a`. The supplied specification described that
PR as open and unmerged at its audit; this archive does not assert its current status
or transplant the branch's integration code into main.

The participant payload is the bytes strictly between
`<!-- BEGIN MASTER PROMPT -->` and `<!-- END MASTER PROMPT -->`, including its
boundary newlines. It has **39,747 bytes** and SHA-256
`96a9d0c4625bd9840fd1d6ed9afad4fca07dc832fa9fe7c1f3fda3a21cba3d39`.
Trimming its first newline creates a different packet and must not be done silently.

The archive reproduces public challenge permissions and emitted source outputs. It
does not add the protected GP grammar, production inventories, decomposition maps,
internal compiler transformations or reversible solution representations.

## What the receipts establish

Byte identity verifies preservation. It does not authenticate an authoring provider,
reproduce an experiment, establish challenge eligibility or demonstrate algorithmic
novelty. In particular:

- Astra and Sol disclose prior project context. Their token figures are estimates,
  not recovered provider counters. Report C is matched to the Opus registry prose;
  report D remains unattributed and is not attached to the Gemini source.
- The Gemini receipt records an owner-supplied filename and source. Provider model,
  revision, original session and authoring telemetry remain unavailable.
- The verification files report historical compilation and tests. Those experiments
  were not rerun merely to package this archival supplement.
- `benchmark/gemini4-argon-high-cells.csv` preserves 576 reported fit cells.
  It is a raw cell table within this record, not the underlying dataset, prediction
  archive, original provider telemetry or a complete independent rerun package.
- `RESULTS.json`, `SOURCE_COUNTS.json`, `GEMINI_COSTS.json` and `match-matrix.txt`
  contain derived summaries. Source counts are lexical counts, not model tokens.
- Narrative receipts retain original prose, including stale status and known wording
  errors. The manuscript supplies the current interpretation without rewriting the
  historical evidence.

The exploratory benchmark uses 64 training families, three folds and three seeds.
Canonical preprocessing preceded the original split, and pilot outcomes were seen
before successor seeds were fixed. Three entrants reuse historical fits; Gemini's
fits are new. Some inputs, predictions and analysis implementation remain outside
the committed record. Bootstrap intervals do not remove these limitations.

Two corrections matter when interpreting the untouched receipts. Gemini's 65.625%
score against Sol is a seed-averaged fractional match score; majority-family wins
are 45/64, or 70.3125%. The geometric RMSE ratio Sol/Gemini is 0.846826: its
reciprocal implies Gemini's point-estimate error is 18.1% higher, not 15% higher.
The reported interval includes parity, and the reported adjusted comparisons do
not establish an RMSE advantage. Historical and Gemini fits also ran on different
hosts; their time totals are not a controlled hardware speed comparison.
