# Literature verification record — EurekaZero north-star draft

Prepared 23 October 2026. This record says how each external fact the draft relies on was checked.
It is an editorial audit, not a systematic review. Three levels are distinguished:

- **verified** — the primary source or its official record was looked up while drafting;
- **inherited** — verified for the InnovationZero paper and recorded in
  [`../../innovationzero/evidence/literature.md`](../../innovationzero/evidence/literature.md); the
  qualifications recorded there apply here too;
- **standard** — a well-known reference whose bibliographic metadata comes from the authors'
  knowledge and was not re-fetched; re-check before any submission.

## Facts the argument depends on

| Fact used in the draft | Key | Level | Qualification |
| --- | --- | --- | --- |
| AlphaGo Zero's policy-only network rated about 3,000 Elo against 5,185 with search | `silver2017zero` | inherited | read from the publisher figure; never print 3,055 |
| AlphaGo's move in the 2016 Seoul match and the human-play probability estimate | `hassabis2016seoul` | inherited | attribute the estimate to DeepMind |
| AlphaTensor's 47-multiplication 4 × 4 scheme | `fawzi2022alphatensor` | inherited | arithmetic modulo 2 |
| AlphaDev sorting routines incorporated into LLVM's C++ library | `mankowitz2023alphadev` | inherited | — |
| FunSearch improved cap-set bounds with an evaluator loop | `romeraparedes2024funsearch` | inherited | online 2023, volume 2024 |
| AlphaEvolve impact across matrix multiplication, scheduling, caches, circuits and commercial optimization | `novikov2025alphaevolve`, `deepmind2026alphaevolve` | verified | impact post dated 7 May 2026; company-reported |
| Recursive, Socher's lab, and its Eureka Machine framing | `socher2026eureka`, `recursive2026firststeps`, `recursive2026flint` | inherited | company-reported; book on sale 22 September 2026 |
| USPTO revised inventorship guidance: only natural persons are inventors; AI systems are tools | `uspto2025inventorship` | verified | released 26 November 2025; replaces the February 2024 guidance; not legal advice |
| Text2CAD-Bench allows repair attempts | `wang2026text2cadbench` | verified | arXiv:2605.18430; the benchmark permits up to three retries |
| CADFS | `pyatov2026cadfs` | verified | arXiv:2605.01925 |
| ZeroShotOpt pretrained optimizer | `meindl2025zeroshotopt` | verified | arXiv:2510.03051 |
| RIBBO | `song2025ribbo` | verified | IJCAI 2025 |
| POM pretrained optimization model | `li2024pom` | verified | arXiv:2405.03728 |
| MM-Eureka uses the Eureka name | `meng2025mmeureka` | verified | arXiv:2503.07365 |
| Eureka, LLM reward design | `ma2024eureka` | standard | ICLR 2024, arXiv:2310.12931 |
| Absolute Zero uses the Zero name for self-play reasoning without curated data | `zhao2025absolutezero` | verified | NeurIPS 2025 poster; arXiv:2505.03335 |
| TabArena | `erickson2025tabarena` | verified | Erickson et al.; arXiv:2506.16791 |
| TabICL | `qu2025tabicl` | verified | ICML 2025, PMLR 267 |
| TabPFN | `hollmann2025tabpfn` | inherited | Nature 2025 |
| SimJEB | `whalen2021simjeb` | verified | Computer Graphics Forum 40(5), 2021 |
| Latent Programmer predicts a discrete latent code, then a program | `hong2021latent` | inherited | ICML 2021; uses a two-level search, so not a no-search baseline |
| HyperFast generates a dataset-specific network in a forward pass | `bonet2024hyperfast` | inherited | a fitted model, not a reusable procedure |
| AutoML-Zero discovered learning algorithms from basic operations | `real2020automlzero` | inherited | — |

## Standard references (re-check before submission)

`anthony2017exit` (NeurIPS 2017), `laterre2018ranked` (arXiv:1807.01672), `amos2023amortized`
(Foundations and Trends in ML 16(5)), `chen2022optformer` (NeurIPS 2022), `muller2023pfns4bo`
(ICML 2023), `laskin2023ad` (ICLR 2023), `pineda2021hpob` (NeurIPS Datasets and Benchmarks 2021),
`hansen2021coco` (Optimization Methods and Software 36(1)), `hansen2019pycma` (the CMA-ES reference
implementation, Zenodo DOI 10.5281/zenodo.2559634), `bergstra2011tpe` (NeurIPS 2011), `lindauer2022smac` (JMLR 23(54)),
`balandat2020botorch` (NeurIPS 2020), `snoek2012bo` (NeurIPS 2012), `shields2021reaction`
(Nature 590), `ahneman2018cn` (Science 360), `sigmund2001topopt` (Structural and Multidisciplinary
Optimization 21(2)), `bendsoe2003topology` (Springer 2003), `wu2021deepcad` (ICCV 2021),
`khan2024text2cad` (NeurIPS 2024), `geuzaine2009gmsh` (IJNME 79(11)), `cadquery2024` (software),
`lu2024aiscientist` (arXiv:2408.06292), `urbina2022dualuse` (Nature Machine Intelligence 4),
`gao2022pmo` (NeurIPS Datasets and Benchmarks 2022), `jevons1865` (*The Coal Question*).

## Program records

Facts about the program itself come from the repository at commit
`f9543b85b356675d377e38d75c4e3ecea8b2d302` and are cited as `jopedime2026repo`,
`jopedime2026innovationzero`, `jopedime2026challenge`, `jopedime2026writeup`, `jopedime2026h27` and
`jopedime2026experiments`. Appendix I gives each with its EST or PRE status.

## Rhetoric excluded

- No claim of "zero priors", "without human knowledge" or "from scratch" for EurekaZero.
- No claim that the approach generalizes to all innovation (claim EZ-C12, *do not assert*).
- No claim that a first proposal replaces later experiments, or that archive novelty is novelty.
- No numerical result from any cited system is imported as a comparator value; every comparator
  number in the draft is synthetic until measured under this paper's protocol.
