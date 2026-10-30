The economic promise of algorithmic self-play is that an experiment can improve both an algorithm and the system that proposes the next one. Realizing that promise requires two accounts: the cost of acquiring useful experience, and the cost of deploying the capability learned from it. Cheap generation helps, but the decisive quantity is how much verified progress a complete experimental budget buys.

## 5.1 The unit of useful experience

A proposal incurs more than a token bill. Its cost includes input processing, reasoning, code generation, validity checks, execution, and evaluation. A training batch also incurs learning updates and operating overhead. Let C_batch include all these charges, N count every candidate attempt, and W > 0 count distinct improvements accepted under the game’s verification rule. The observed cost per accepted improvement is

```math
C_{\mathrm{win}}=\frac{C_{\mathrm{batch}}}{W}
=\frac{\bar c}{\hat q},
\qquad \bar c=\frac{C_{\mathrm{batch}}}{N},
\quad \hat q=\frac{W}{N}. \tag{2}
```

This is an accounting identity, not an independence assumption. If one attempt in ten succeeds, each accepted improvement carries ten attempts’ average cost. Forecasting with expected cost per attempt divided by q additionally requires a specified, sufficiently stable proposal and evaluation regime. As the policy and incumbent change, so can q.

Execution success and improvement are different events. Writing q = v × s, with v the probability of obtaining an admissible score and s the probability of acceptance after the prescribed confirmation conditional on that score, exposes both bottlenecks. A program can execute perfectly and still lose. Conversely, a losing program can supply useful training information: immediate cost per win does not measure the entire value of experience. Cost per win is comparable only under a common problem mix, incumbent standard, practical improvement threshold, and verification rule.

## 5.2 Reliability repair and the reasoning budget

We call proposals that cannot produce an admissible evaluated result *stillborn algorithms*. Their failure modes include parse or type errors, runtime exceptions, invalid numerical outputs, and memory or time limits. Structured representations can exclude some invalid constructions; execution and competitive performance still require separate tests.

A repair consumes another candidate attempt whenever it emits a revised algorithm. Its longer context, reasoning, checks, and execution belong in the ledger. Repair may be worthwhile while collecting training experience, but it cannot be hidden inside deployment at search = 1. A failed first proposal remains a failed first proposal.

The reasoning budget should likewise be chosen by its effect on cost per accepted improvement. Increasing per-attempt cost from c₀ to c₁ is economical when the corresponding success probabilities satisfy the inequality below, for positive initial cost and success probability. Work on inference-time computation finds that effective allocation depends on problem difficulty [1]; it does not establish a universal reasoning budget for algorithm discovery. Small models, long reasoning traces, and repair loops must therefore be compared through their complete costs and outcomes.

```math
\frac{q_1}{q_0}>\frac{c_1}{c_0}.
```

## 5.3 A batch priced

Table 5 prices a common workload across six capacity scenarios. Each batch contains 1,000 candidate attempts, each with 2,048 fresh input tokens and 5,000 generated tokens, including reasoning. The hardware allocations and throughput values are assumptions; the price is a dated public quote. Throughput is aggregate generated tokens divided by billed generation time, including input processing, initialization, communication, and idle time. It is not single-request decoding speed.

**Table 5. Illustrative generation budgets under explicitly assumed serving throughput.** USD per 1,000 attempts at Together AI’s quoted HGX H100 GPU Clusters on-demand rate of $3.99 per GPU-hour, accessed 30 October 2026 [2]. All hardware counts and throughput values are assumed. Devices have an assumed 80 GB each; weights use two bytes per parameter without offloading. “Low” and “high” costs use twice and half the stated throughput, respectively; these are sensitivity cases, not confidence intervals. The table excludes evaluation, additional repair proposals, learner updates, and separately billed operations. Configuration availability and feasibility are not established by the unit-price quote.

| Generator scenario | Total / active parameters | GPUs | Assumed aggregate output tokens/s | Low USD | Base USD | High USD |
| - | -: | -: | -: | -: | -: | -: |
| Small dense | 8B / 8B | 1 | 3,000 | 0.92 | 1.85 | 3.69 |
| Medium dense | 32B / 32B | 2 | 1,500 | 3.69 | 7.39 | 14.78 |
| Large dense | 70B / 70B | 4 | 1,200 | 9.24 | 18.47 | 36.94 |
| DeepSeek-V3 | 671B / 37B | 32 | 2,000 | 44.33 | 88.67 | 177.33 |
| Hypothetical 1.5T MoE | 1,500B / 100B | 64 | 2,000 | 88.67 | 177.33 | 354.67 |
| Hypothetical 1.5T dense | 1,500B / 1,500B | 64 | 600 | 295.56 | 591.11 | 1,182.22 |

The table asks what generation would cost if a deployment sustained the stated throughput. Its ordering follows from those assumptions; it does not establish the cheapest model per verified improvement. Actual throughput can lie outside the sensitivity cases.

For comparison, Together AI lists Llama 3 8B Instruct Lite at $0.14 per million input and output tokens on 30 October 2026 [2]. The same workload costs $0.987 through that hosted endpoint before evaluation or learning. This is a conditional token bill, not evidence of the model’s success rate. The hosted product is not a measured counterpart to the generic self-hosted 8B scenario. Hosted inference and self-hosted GPU-hours are alternative charges for generation; adding both would count the same work twice.

The referee can dominate a small generator’s bill. At an explicitly assumed $0.04 per CPU-core-hour, 100 CPU-core-seconds of evaluation per attempt add $1.11 per 1,000 attempts. Eight cores running for 100 seconds add $8.89 instead. Failed fits, repeated folds, and confirmation runs count too. Appendix D gives the full accounting and scaling calculations.

## 5.4 Exploration buys information

Inherited algorithmic patterns provide competence; exploration tests alternatives those patterns may underrepresent. Quality-diversity methods illustrate why retaining several kinds of strong solution can support exploration [3]. This motivates a budget for speculative proposals, rather than an assumption that unfamiliarity itself predicts value. Any exploration incentive must remain separate from the criterion for accepting an improvement.

> **Box 3. A referee makes speculation testable.** A “useful hallucination” is a conjectural artifact whose value survives evaluation. It is not permission to invent facts. An unusual program earns promotion through the same validity and confirmation rules as a familiar one. The referee supplies evidence within those rules; it does not establish universal correctness or safety.

## 5.5 Scale is a workload not a learning guarantee

A planning allocation of 10,000 problems, 100 rounds, and eight candidate attempts per round contains eight million attempts. At the lengths above, it requires 40 billion generated tokens and 16.384 billion fresh input tokens. That specifies work; it does not establish how much experience a transferable policy needs. Learning curves on development problems must determine whether additional expenditure continues to improve first-proposal performance.

The leading parameter-multiply approximation for output generation is

```math
F_{\mathrm{out}}\approx2P_{\mathrm{active}}T_{\mathrm{out}}.
```

This adapts the standard Transformer compute approximation [4]. Applied to the illustrative workload, it gives 1.2 × 10²³ FLOPs for a hypothetical 1.5T dense model, or 8 × 10²¹ FLOPs for a hypothetical 1.5T mixture-of-experts model with 100B active parameters. Neither figure includes input processing, attention corrections, communication, evaluation, or learning updates. Both models require about 3 TB for two-byte weights alone. Active parameters estimate arithmetic; total parameters constrain storage. DeepSeek-V3’s reported 671B total and 37B active parameters provide a concrete example of this distinction [5]. A sparse arithmetic count is not a rental quote.

## 5.6 From experiments to reusable judgment

Learning can pay an up-front cost and reuse the resulting capability across related problems, the principle of amortized optimization [6]. The relevant comparison is lifetime cost at comparable delivered quality: development and training, plus encoding, generation, fitting, and use on new tasks. A frontier-designed general algorithm can also be developed once and reused; the same accounting applies to every entrant.

EurekaZero’s economic opportunity is therefore to convert experimental feedback into a better first proposal. An informative problem representation can guide that proposal, while an executable solution representation can reduce avoidable failures. Whether these benefits repay training is an empirical question. Section 6 develops the mechanism: a structured mapping from the state of a problem to a proposed solution.

# Appendix D Economics assumptions and calculations

The numerical examples below are workload scenarios. They do not estimate a measured EurekaZero cost, failure rate, learning curve, or win rate. Public price quotes and published architecture facts are distinguished from assumptions.

## D.1 The complete ledger

```math
C_{\mathrm{batch}}=C_{\mathrm{generation}}+C_{\mathrm{repair}}
+C_{\mathrm{evaluation}}+C_{\mathrm{updates}}+C_{\mathrm{operations}}.
```

Here generation excludes separately recorded repair generations, and evaluation includes their checks and evaluations. Repairs also contribute to the candidate count. If a ledger already includes repair generations in its generation total, the repair subtotal must not be added again. Shared services and learner updates are allocated once.

For hosted generation, the uncached token charge is

```math
C_{\mathrm{API}}=
\frac{T_{\mathrm{in}}p_{\mathrm{in}}+
(T_{\mathrm{reason}}+T_{\mathrm{code}})p_{\mathrm{out}}}{10^6}.
```

Prices are in USD per million billed tokens. Apply separate cache or reasoning rates if the provider specifies them. The example assumes fresh inputs, no discount, and all 5,000 generated tokens billed at the output rate. Actual provider billing determines whether non-visible reasoning is included.

For a fixed self-hosted allocation of g GPUs at a price p_g in USD per GPU-hour, aggregate effective throughput R gives

```math
H_{\mathrm{GPU}}=\frac{gNT_{\mathrm{out,per}}}{3{,}600R},
\qquad C_{\mathrm{GPU}}=p_gH_{\mathrm{GPU}}.
```

GPU-hours are summed across devices, not elapsed wall-hours. R uses the entire billed generation interval; input processing is included in elapsed time even though the numerator counts generated tokens. For changing allocations, sum charges over allocation intervals. Model downloads, warm-up, reservation idle time, and failed jobs must either enter this interval or be added explicitly. Separately billed storage, networking, CPU services, and taxes are outside Table 5.

## D.2 Evidence status and capacity

| Input | Value and scope | Evidence status |
| - | - | - |
| GPU unit price | Together AI GPU Clusters, HGX H100 on-demand, $3.99/GPU-hour. Dedicated inference is a different product. | Public price accessed 30 October 2026 [2] |
| Hosted token price | Llama 3 8B Instruct Lite, $0.14/million input and $0.14/million output tokens. | Public price accessed 30 October 2026 [2] |
| Hardware and precision | H100-class 80 GB devices; two-byte resident weights; no weight offload. GPU counts and cross-node deployment are assumed. | Capacity scenario |
| Throughput | Table 5; low cost uses twice base throughput, high cost uses half. | Assumed sensitivity |
| DeepSeek-V3 size | 671B total; 37B active per token. | Reported architecture [5] |
| Other model tiers | Dense 8B, 32B, 70B; hypothetical 1.5T dense and 1.5T MoE with 100B active. | Representative scenarios |
| Workload and evaluation | 2,048 input and 5,000 output tokens per attempt; CPU price $0.04/core-hour. | Planning assumptions |
| Validity and win rates | No model-specific values assigned. | Unmeasured |

The chosen allocations pass a weights-only capacity check: 16, 64, 140, 1,342, 3,000, and 3,000 GB of two-byte weights respectively. Aggregate device capacities are 80, 160, 320, 2,560, 5,120, and 5,120 GB. These are decimal units. Positive aggregate headroom is necessary but insufficient for a feasible implementation: partitioning, attention caches, workspaces, and routing buffers can impose stricter limits. Training adds optimizer and gradient storage. Neither an 80 GB designation nor a quoted GPU price guarantees a feasible multi-node serving configuration.

## D.3 Scaling the batch

The eight-million-attempt scenario is 8,000 batches of 1,000 attempts. At unchanged workload, price, and effective throughput, Table 5’s base generation charges scale to approximately $14,800, $59,100, $147,800, $709,300, $1.419 million, and $4.729 million. The corresponding low/high cases are one-half/twice these amounts. This is a conditional multiplication, not a prediction of an affordable or successful training run.

The hosted 8B example instead gives

```math
\frac{(2{,}048{,}000+5{,}000{,}000)\times0.14}{10^6}
=\$0.98672
```

per 1,000 attempts, or $7,893.76 for eight million attempts. The provider quote covers generation, not researcher access to gradients or a custom reinforcement-learning service. Local policy learning therefore needs a separate implementation and budget.

At 100 CPU-core-seconds per attempt and $0.04 per core-hour, evaluation of eight million attempts costs $8,888.89. At eight cores for 100 seconds per attempt, it costs $71,111.11. Memory, storage, orchestration, and any resource reservation not captured by these core-hours are additional. Work already paid for inside a rented instance must not also be charged as an external CPU service.

For an illustrative batch with $10 total cost, 1,000 attempts, 800 admissible scores, and 40 accepted improvements, the raw-attempt cost is $0.01, the cost per admissible score is $0.0125, and the cost per accepted improvement is $0.25. Here v = 0.8, s = 0.05, and q = 0.04. The example applies no model-specific failure assumption. If no improvement is accepted, report the cost and zero accepted improvements; a finite observed cost-per-win value is unavailable.

## D.4 Compute scale and the 1.5T cases

Kaplan et al. approximate a Transformer forward pass by twice its non-embedding parameter count per token, plus an attention term; the corresponding training approximation is six times that count [4]. Using active parameters extends the leading parameter-multiply term to sparse models as a planning proxy, not an exact architecture-specific FLOP count.

```math
\begin{aligned}
F_{\mathrm{out,dense}}&\approx2(1.5\times10^{12})(4\times10^{10})
=1.2\times10^{23}\ \mathrm{FLOPs},\\
F_{\mathrm{out,MoE}}&\approx2(10^{11})(4\times10^{10})
=8\times10^{21}\ \mathrm{FLOPs}.
\end{aligned}
```

Fresh inputs add a leading parameter-multiply term proportional to 16.384 billion tokens. Attention costs also depend on sequence length. These effects are separate from routing, memory traffic, communication, scheduling, and the referee’s computation. Dividing these FLOP counts by peak hardware FLOP/s does not yield observed throughput.

If an additional, explicitly assumed learning workload processes four billion tokens once—10% of the generated count—the full-training 6PT proxy adds 3.6 × 10²² FLOPs for the dense case and 2.4 × 10²¹ FLOPs for the 100B-active case. More epochs or replay increase processed tokens. Reference-policy scoring, auxiliary models, optimizer work, and algorithm-specific passes need separate accounting. Parameter-efficient methods such as LoRA reduce trainable state but still execute the base model [7]; their training FLOPs must be profiled rather than inferred solely from the adapter parameter count.

The dense 1.5T-to-8B parameter ratio is 187.5. It is a ratio of leading arithmetic at equal token work, not a general 187.5-fold cost advantage. Differences in utilization, generation length, failure rates, evaluation, and repair can change the cost-per-win comparison. No claimed frontier pretraining cost is needed for this argument.

## D.5 Experience value and reuse

Equation 2 describes a realized batch even when outcomes are correlated. For prediction, the familiar geometric waiting time 1/q additionally assumes repeated independent trials with fixed success probability. Adaptive repair, shared task families, and a changing incumbent violate that simplification. Likewise, repeated copies of the same accepted artifact do not constitute distinct improvements.

A pilot should record candidate and parent identifiers; checkpoint and task-family identifiers; fresh and cached input tokens; reasoning and output tokens; billed resource time; validity failures; evaluation and confirmation outcomes; and update costs. Confirmation during training uses the game’s designated confirmation data, not the final sealed tournament. These records support the intended survival funnel without inventing counts in advance. Distinguish proposals, completed scores, provisional wins, and confirmed promotions.

For M new deployments, define lifetime cost as

```math
C_{\mathrm{life},j}(M)=C_{\mathrm{build},j}+M\,c_{\mathrm{deploy},j}.
```

The index j identifies an entrant. Deployment cost includes the declared encoding, generation, fitting, and prediction workload. Report common underlying pretraining separately from incremental method development, using the same inclusion rule for all entrants. If entrant A costs more to build but less to deploy than B, their cost break-even point is

```math
M_*=\frac{C_{\mathrm{build},A}-C_{\mathrm{build},B}}
{c_{\mathrm{deploy},B}-c_{\mathrm{deploy},A}},
```

provided both numerator and denominator are positive. This comparison is meaningful only at comparable output quality, reliability, and constraints. Search = 1 specifies a deployment procedure; by itself it proves neither a cost saving nor a particular break-even volume.

# References

[1] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters*. arXiv:2408.03314, 2024. [Primary source](https://arxiv.org/abs/2408.03314).

[2] Together AI. *Together AI Pricing*. Accessed 30 October 2026. [Official pricing](https://www.together.ai/pricing).

[3] Jean-Baptiste Mouret and Jeff Clune. *Illuminating Search Spaces by Mapping Elites*. arXiv:1504.04909, 2015. [Primary source](https://arxiv.org/abs/1504.04909).

[4] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. *Scaling Laws for Neural Language Models*. arXiv:2001.08361, 2020. [Primary source](https://arxiv.org/abs/2001.08361).

[5] DeepSeek-AI. *DeepSeek-V3 Technical Report*. arXiv:2412.19437, 2024; version 2 revised 18 February 2025. [Primary source](https://arxiv.org/abs/2412.19437v2).

[6] Brandon Amos. *Tutorial on Amortized Optimization*. arXiv:2202.00665, 2022. [Primary source](https://arxiv.org/abs/2202.00665).

[7] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685, 2021. [Primary source](https://arxiv.org/abs/2106.09685).

# BibTeX source

Save the following bibliography as references.bib. Entries use the verified arXiv records; the Amos key identifies the 2022 preprint rather than assigning unverified journal metadata.

```bibtex
@misc{snell2024scaling,
  author = {Snell, Charlie and Lee, Jaehoon and Xu, Kelvin and Kumar, Aviral},
  title = {Scaling {LLM} Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters},
  year = {2024},
  eprint = {2408.03314},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2408.03314}
}

@misc{together2026pricing,
  author = {{Together AI}},
  title = {{Together AI} Pricing},
  year = {2026},
  url = {https://www.together.ai/pricing},
  urldate = {2026-10-30},
  note = {Accessed October 30, 2026. GPU Clusters HGX H100 on-demand: USD 3.99 per GPU-hour. Serverless Llama 3 8B Instruct Lite: USD 0.14 per million input and output tokens.}
}

@misc{mouret2015mapelites,
  author = {Mouret, Jean-Baptiste and Clune, Jeff},
  title = {Illuminating Search Spaces by Mapping Elites},
  year = {2015},
  eprint = {1504.04909},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/1504.04909}
}

@misc{kaplan2020scaling,
  author = {Kaplan, Jared and McCandlish, Sam and Henighan, Tom and Brown, Tom B. and Chess, Benjamin and Child, Rewon and Gray, Scott and Radford, Alec and Wu, Jeffrey and Amodei, Dario},
  title = {Scaling Laws for Neural Language Models},
  year = {2020},
  eprint = {2001.08361},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2001.08361}
}

@misc{deepseekv3,
  author = {{DeepSeek-AI}},
  title = {{DeepSeek-V3} Technical Report},
  year = {2024},
  eprint = {2412.19437},
  archivePrefix = {arXiv},
  note = {Version 2, revised February 18, 2025},
  url = {https://arxiv.org/abs/2412.19437v2}
}

@misc{amos2022tutorial,
  author = {Amos, Brandon},
  title = {Tutorial on Amortized Optimization},
  year = {2022},
  eprint = {2202.00665},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2202.00665}
}

@misc{hu2021lora,
  author = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  title = {{LoRA}: Low-Rank Adaptation of Large Language Models},
  year = {2021},
  eprint = {2106.09685},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2106.09685}
}
```
