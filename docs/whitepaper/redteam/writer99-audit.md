# Writer red-team and bibliography audit — t2150

Reviewed main `15bef0ac57f7f194da80f08d5d2b214976f655a0` on 1 November 2026 (refreshed from `9999190e407c1249e727924b6d3350060cac44f8`; all inspected source and data pins unchanged). This is a same-agent source and committed-aggregate review, not independent review or runtime qualification. Sections 4–8 and Appendices A–H were read; Results, Discussion, the abstract and every TBD remain pending. No model weights, sealed/holdout payloads or new run outcomes were opened. Builds run serially at nice 19.

## Corrections

1. Section 4.2 now says incumbent code and score **must** stay outside the writer input. This is the reference input contract, not a claim that an artifact's finite context values prove input closure. `PolicyContext` validates shape, finiteness and bounds; provenance/closure audits remain required (`crates/jope-policy/src/context.rs`, `sections/07_results.tex`).
2. Section 6.1 ties writer/map separation to the **native implementation** and map omission to the **proposed game-day protocol**. Native next-token decoding and reward scoring are separate methods; implemented pool sampling does not establish a qualified single-draw deployment (`crates/jope-policy/src/model.rs`, `decode.rs`, `seeds.rs`).
3. Section 6.4 attributes separate context networks and token embeddings to the **native implementation**, rather than asserting a binding for an evaluated checkpoint. The loader and ranking bindings check identities; acceptance and deployment receipts remain separate evidence (`crates/jope-policy/src/model.rs`, `config.rs`, `ranking.rs`).
4. The Section 6 source comment pointed to nonexistent `06_method.notes.md`. It now points to the supplied notes under `06_method/sources/`. This comment is not rendered.
5. Consolidate the AlphaZero preprint key `silver2017chess` into `silver2018general`, and replace their one compiled co-citation in Section 5. The published supplement's Table S3 contains both chess training statistics used there, so neither figure depends on keeping a second bibliography entry. Retain the earlier-version arXiv URL and an explicit version note in the canonical entry. All scientific numbers remain unchanged.

## Claim evidence and remaining boundaries

| Claim reviewed | Source inspected on the pinned main | Disposition |
| --- | --- | --- |
| Reference context is 2,048 finite values; incumbent/final targets are excluded by contract | `crates/jope-policy/src/context.rs`, `semantic.rs`; Section 6 supplied notes | Shape/ABI checks are implemented; semantic input closure still needs its evaluation receipt. Correction 1 makes this boundary explicit. |
| Grammar-constrained writer, token budget and parse gate | `crates/jope-policy/src/decode.rs`, `grammar.rs` | Legal-token masking, bounded decoding and decode/parse admission exist. They do not prove runtime validity, useful composition or transfer. |
| First 64 distinct valid drafts, grammar-uniform fill and 16 exploration seeds | `crates/jope-policy/src/decode.rs`, `seeds.rs` | Pool construction is implemented; proposed one-draw deployment is a separate protocol. |
| Map pools token embeddings without order, combines context three ways and averages four ranking heads | `crates/jope-policy/src/model.rs:reward_runtime`, `crates/jope-engine/src/neural.rs:reward_runtime_encoded` | Matches implementation; scores are within-context logits, not calibrated errors or win probabilities. |
| Map does not screen or condition writer draws, but ranks influence search parents | `crates/jope-policy/src/seeds.rs:assemble_population`, `crates/jope-engine/src/gp.rs:policy_weighted_parent` | All pool members retained, ranks attached, parent-selection route present. Causal first-proposal benefit remains TBD. |
| Writer objective, listwise/pairwise terms and eight-program comparisons | `crates/jope-engine/src/training/objective.rs`, `training.rs:sample` | Coefficients, rank weights, grammar smoothing and normalized token/program losses match the existing qualified description. Exact objective/partition identity remains part of checkpoint provenance. |
| Historical reference size, corpus, optimizer and hardware | supplied `sections/06_method/sources/01M674E8XA0S4ZS5BT6FVH5QKX.md`; native model/config and objective | Historical notes and compatible source are not acceptance or operational evidence. Existing text retains that qualification. |
| Reference five-fold and 16-cut/three-seed ratchets | `docs/adr/ADR-0005-canonical-learning-loop.md`, supplied method notes | Reference rules; not attributed to the later adapter loop. |
| Inspected flywheel generates programs heuristically and optimizes a ranking adapter | `research/flywheel_selfplay/run_fused_closed_loop.py`, `rust_evaluator.py` | Candidate generation and adapter optimization do not train a neural program writer. |
| First-four revealed outcomes and differing train/evaluation order | `run_fused_closed_loop.py:predict`, `utility`; `automatic_promotion.py:training_indices` | Stored outcomes enter ranking; it is a conditional selection task. No independent first-proposal inference is claimed. |
| Logged evaluation differs from pool admission | `run_fused_closed_loop.py`; `automatic_promotion.py:promote` | Ordinary pool admission requires completed passing evaluation and strict incumbent improvement; historical admission is separate. |
| Default adapter gate accepts ties; rejection restores model and optimizer | `research/flywheel_selfplay/dev_incumbent.py:DevIncumbent` | Matches source. Existing logs distinguish acceptance flags from strict best increases. |
| Flywheel counts, gaps, rates and source binding | committed `figures/data/` CSVs and provenance; existing figure scripts | Development aggregates only; 0 new observations. Qualification waits and unobserved gaps retain their stated scope; exact runtime binding remains TBD. |
| Results, Discussion and conclusions | Sections 7–8, abstract, Appendices A–H | No new outcomes supplied. All 28 TBD bodies and source locations, scientific numbers and equations remain intact. |

All 20 inspected source/note files and all 24 committed data/provenance files are SHA-256 pinned in the durable pass receipt. The review distinguishes implementation, supplied historical specifications, logged development observations and unqualified deployment/evaluation claims.

## Bibliography identity and metadata

The master now has **158 entries under 158 unique keys**. DOI, arXiv identifiers (including version notes/URLs) and normalized titles were checked, followed by manual review of title/author near-pairs. The AlphaZero preliminary and published versions are consolidated as above; no other duplicate work was found. Distinct AlphaGo/AlphaGo Zero studies, KumoRFM model versions and separate works by the same author remain separate.

All entries have a nonempty year and a publication venue or source (`journal`, `booktitle`, `publisher`, `institution`, `school` or `howpublished`). Preprints, books and web sources retain their actual source type; no venue was invented. The 124 works cited by the compiled paper resolve to entries. The archived source drafts are historical inputs and are not rewritten.

The [authors' publication announcement](https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) links both versions and the [published paper with supplement](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf). Its Table S3 (PDF page 27) reports the chess training-game count and duration used in Section 5. This supersedes the reason for retaining two keys in the dated layout pass 15; that historical report is preserved.

### Uncited entries

These **34 entries are uncited by the current compiled inputs**. They are retained and flagged here, not inserted as irrelevant citations or silently deleted. This inventory is dated to this pass; future citation edits should refresh it.

- `agarwal2021precipice`
- `amos2023amortized`
- `arthur2009nature`
- `carlini2023quantifying`
- `chen2023teaching`
- `chiang2024arena`
- `doit2026c7i`
- `einstein1938evolution`
- `epoch2024data`
- `hong2021latent`
- `jaech2024openai`
- `kahneman2011thinking`
- `klein1998sources`
- `kumorfm2026v2`
- `le2022coderl`
- `li2022competition`
- `liu2024lost`
- `ma2024eureka`
- `openai2025gptoss`
- `ornn2026ocpi`
- `rahman2025over30`
- `recursive2026research`
- `runco2012standard`
- `shao2024deepseekmath`
- `shinn2023reflexion`
- `simon1992explanation`
- `snell2024scaling`
- `stanley2015greatness`
- `thunder2026gpu`
- `tirumala2022memorization`
- `together2026pricing`
- `tomasello1999cultural`
- `trivedi2021learning`
- `weitzman1998recombinant`

## Formatting and self-review

Cross-references, captions, graphic imports and all existing exhibit numbers are retained. Figure/table order is checked in source and rendered column order, allowing intentional forward citations and appendix figures. Opus's Sections 1–3, Figures 11–12 definitions, title, layout settings and layout report are outside this edit. The rebuilt PDF and paired page proofs accompany the change; warning/box/reference counts and actual page positions are recorded in the build receipt.

The seven review dimensions are assessed by the implementing agent: claim/evidence 4, flow 4, notation 4, figures/tables 4, citations 4, concision 4, build hygiene 5 (clean 27-page build, no TeX/BibTeX warnings, undefined references or box warnings). These are a self-review disposition, not an independent or human score. Missing scientific inputs remain explicitly pending.

Fresh baseline and candidate builds place Figures 11–12 on page 16 and Figure 22 on page 22; every existing section/exhibit page is unchanged. The earlier t2055 coordination note saying page 15 was incorrect. Source float definitions are unchanged.
