# Primary-source verification record

Verified 22 October 2026. This record distinguishes scholarly evidence, author
accounts, publisher metadata, and company reports. The bibliography is the
publication-facing reference list. This is an editorial audit, not an exhaustive
systematic literature review. No web source text or figure is reproduced in full.

| Key | Primary evidence inspected | Permitted use / qualification |
| --- | --- | --- |
| `silver2016alphago` | [Nature article](https://www.nature.com/articles/nature16961), DOI 10.1038/nature16961 | Neural networks and tree search; Nature 529, 484–489 (2016). |
| `hassabis2016seoul` | [Hassabis's account](https://blog.google/innovation-and-ai/products/what-we-learned-in-seoul-with-alphago/) | Attribute the human-play probability estimate to DeepMind; do not identify it with objective creativity. |
| `silver2017zero` | [Publisher page](https://www.nature.com/articles/nature24270), [author manuscript record](https://discovery.ucl.ac.uk/id/eprint/10045895/), [authors' explanation](https://deepmind.google/blog/alphago-zero-starting-from-scratch/), publisher Figure 6 | Search-guided self-play and policy-only distinction. Figure 6 inspected visually: policy-only bar approximately 3,000; no exact printed policy-only value. Do not claim 3,055 from this evidence. |
| `fawzi2022alphatensor` | [Nature article](https://www.nature.com/articles/s41586-022-05172-4), [authors' explanation](https://deepmind.google/blog/discovering-novel-algorithms-with-alphatensor/) | Matrix multiplication discovery by RL. Distinguish arithmetic modulo 2 from ordinary real-valued multiplication when discussing the 47-multiplication result. |
| `mankowitz2023alphadev` | [Full primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10247365/) | RL search for assembly programs and measured latency; sorting components incorporated into LLVM. |
| `romeraparedes2024funsearch` | [Nature article](https://www.nature.com/articles/s41586-023-06924-6), [authors' explanation](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/) | LLM plus evaluator in an iterative feedback loop. Online publication 2023, journal volume 2024. |
| `novikov2025alphaevolve` | [arXiv:2506.13131](https://arxiv.org/abs/2506.13131) | Iterative evaluator feedback. Its reported 48-multiplication result concerns 4 by 4 complex-valued matrices. |
| `balog2017deepcoder` | [arXiv:1611.01989](https://arxiv.org/abs/1611.01989), [Microsoft Research](https://www.microsoft.com/en-us/research/publication/deepcoder-learning-write-programs/) | Neural properties guide enumerative/SMT search; not a no-search baseline. |
| `ellis2021dreamcoder` | [Author institution record](https://dspace.mit.edu/entities/publication/86d768ab-558f-4fb1-b92e-9288de643ccb), [primary preprint](https://arxiv.org/abs/2006.08381) | Learned abstractions and neural search. PLDI title differs from the earlier preprint title; bibliography uses the PLDI title. |
| `kamienny2022e2e` | [Primary preprint](https://arxiv.org/abs/2204.10532), [NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2022/hash/42eb37cdbefd7abae0835f4b67548c39-Abstract-Conference.html) | Directly predicts expressions and constants; optional refinement is distinct. This antecedent precludes broad first-ever data-to-program claims. |
| `ha2017hypernetworks` | [Primary preprint](https://arxiv.org/abs/1609.09106) | A network generating another network's weights; distinguish parameters from a reusable learning procedure. |
| `real2020automlzero` | [ICML proceedings](https://proceedings.mlr.press/v119/real20a.html) | Discovery of complete learning algorithms from basic mathematical operations. |
| `socher2026eureka` | [Publisher metadata](https://www.hachettebookgroup.com/titles/richard-socher/the-eureka-machine/9781541705708/) | Richard Socher; PublicAffairs; on-sale date 22 September 2026; 288 pages; ISBN 9781541705708. Page also retains preorder boilerplate. Do not infer sales status. Book contents were not read in full. |
| `socher2026bookbite` | [Author-written excerpt](https://nextbigideaclub.com/magazine/ais-biggest-breakthrough-wont-chatbot-bookbite/61534/) | Four foundations and coordinating agents, stated as Socher's vision. Attribute book-content summary to this excerpt rather than claiming full-book access. |
| `recursive2026firststeps` | [Company research report](https://recursive.com/updates/first-steps-toward-automated-ai-research) | Parallel research threads with propose/implement/run/validate/update feedback. Company-reported results. |
| `recursive2026flint` | [Company announcement](https://recursive.com/updates/introducing-recursive-flint), [update index](https://recursive.com/updates) | Index dates release 21 October 2026. Company attributes much of model-building stack to its Eureka Machine. No inference that physical labs or every book pillar participated. |
| `typesafe2026jev` | [Almeida's announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [creator interview](https://www.latent.space/p/jev) | Jev is the model, Diogo Almeida the person. Company objective: calibrated typed decisions. Announcement explicitly acknowledges RLVR; do not reduce modern LLMs to RLHF alone. Type safety does not establish factual correctness. |
| `dupoux2026autonomous` | [Dupoux, LeCun and Malik preprint](https://arxiv.org/abs/2603.15381) | Primary architecture argument about autonomous learning; not an impossibility theorem about novelty. |
| `christiano2017rlhf` | [Christiano et al. primary paper](https://arxiv.org/abs/1706.03741) | Learning from preferences over trajectories. |
| `deepseek2025r1` | [DeepSeek-R1 primary preprint](https://arxiv.org/abs/2501.12948v1) | Outcome-based RL is already used in language reasoning; do not infer undocumented recipes for the four repository entrants. |

## Additional close antecedents verified during review

| Key | Primary evidence inspected | Permitted use / qualification |
| --- | --- | --- |
| `hong2021latent` | [ICML proceedings](https://proceedings.mlr.press/v139/hong21a.html) | Hong, Dohan, Singh, Sutton and Zaheer; ICML 2021, PMLR 139:4308–4318. Predicts a discrete latent code from input-output examples and then generates a program; uses a two-level search. Close prior work for solution representations. |
| `bonet2024hyperfast` | [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/28988) | Bonet, Mas Montserrat, Giró-i-Nieto and Ioannidis; AAAI 38(10):11114–11123, published 24 March 2024; DOI 10.1609/aaai.v38i10.28988. Meta-trained hypernetwork generates a dataset-specific classification network in a forward pass. This is a fitted model, distinct from a reusable fitting procedure. No numerical performance comparison is imported. |
| `meng2026swarmrsi` | [Versioned primary preprint](https://arxiv.org/abs/2610.20431v2) | Qianru Meng et al., *SwarmRSI* (full title in `references.bib`). Submission history read directly: v1 14 October 2026, v2 19 October 2026. It is available before this manuscript's 22 October cutoff. Experimental feedback guides cross-agent variation and recombination. Not independent evidence of unrestricted RSI or zero-feedback discovery. No numerical result imported. |

## Excluded or corrected rhetoric

- Do not say every machine discovery uses test-time iteration, or that no prior
  work has learned to map data to programs. This is stronger than the evidence
  and ignores neural program synthesis and end-to-end expression prediction.
- Do not treat the AlphaGo Zero title's “without human knowledge” as a literal
  property of an algorithm system containing designed operator families.
- Do not print the proposed LeCun podcast quotation as audio-verified. The March
  19, 2025 Big Technology episode is identifiable, but this audit did not verify
  its original audio. Third-party automatic transcripts are insufficient.
- “LLMs are chasing the wrong objective” is a proposed synthesis, not a verified
  quotation from a person named Jev.
- Company reports about Flint and Jev are attributed descriptions, not
  independently reproduced benchmarks or proof of unrestricted capabilities.
- Socher's broad scientific vision and Recursive's deployed research system are
  related but not interchangeable evidence objects.

The historical comparison deliberately uses no directly reproduced quotations.
