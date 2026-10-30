# Section 5 (`sections/05_price.tex`) — Editor Handoff Note & Claim Ledger

**Section Title:** The Price of Play (`\label{sec:price}`, alias `\label{sec:economics}`)  
**Word Count:** ~1,180 words of main prose (within the 2-page / 1,200-word target budget).  
**Subsections:** 6 (`\ref{sub:price-bill}` through `\ref{sub:price-batch}`), matching all speaking points in order.

---

## 1. Core Points & Speaking Points Advanced

1. **Core Point 2 (Novelty is free; judgment is the inventor):** Connects the search-length law $\mathbb{E}[N_{\text{tries}}] = 1/q$ (`\cref{eq:taste}`) directly to the unit economics of discovery in Equation 2 (`\cref{eq:bill}`): $C_{\text{win}} = (C_{\text{attempt}} + \mathbb{E}[r]\,C_{\text{repair}}) / q$.
2. **Core Point 3 ($\text{Search} = 1$):** Establishes the two-ledger accounting framework (training experience acquisition vs. single-generation deployment at $B = 1$), applying the same amortization logic to both EurekaZero and the frontier-invented general methods.
3. **Speaking Point 7 (Waste, not tokens, sets the price of play):** Shows that stillborn moves ($\tilde{w}$) and also-rans ($w^-$)—not raw token prices—dominate discovery costs, and that an 8-million-attempt frontier run costs roughly 1% of a $10^{25}$-FLOP pretraining run.

---

## 2. Inventory of `\TBD{}` Placeholders Left for Final Data Integration

Every unmeasured pilot or sealed-arena quantity is explicitly marked with `\TBD{}` so no synthetic numbers enter the manuscript:

1. `\TBD{pilot waste reduction from X\% to Y\%}` (in `§5.2` prose): Pilot measurement of total waste ($1 - q$) before and after training with the problem–program map.
2. `\TBD{N}` and `\TBD{seed count}` (in `Figure 4` caption and schematic panels): Number of pilot attempts and random seeds used to populate the Figure 4a survival funnel and Figure 4b scale $\times$ reasoning heatmap.
3. `\TBD{E\%}` (in `§5.4` prose): Exact percentage of each self-play round budgeted for high-temperature speculative exploration games.
4. `\TBD{\$}` (in `Table 5` takeaway caption): Final headline dollar figure for the chosen production model tier once pilot calibration is locked.
5. `\TBD{median sealed win problem}`, `\TBD{\$X}`, `\TBD{Y}`, and `\TBD{\$Z}` (in `§5.6` prose): Running-example cost metrics—training cost per attempt (`$X`), referee cross-validation time (`Y` seconds), and game-day $\text{search} = 1$ inference cost (`$Z`).

---

## 3. Citation Check (`\citeneeded{}` Status)

- **Zero `\citeneeded{}` placeholders remain.** Every citation required by the brief and style guides (`kaplan2020scaling`, `hoffmann2022chinchilla`, `epoch2024data`, `deepseekv3`, `guo2025deepseekr1`, `silver2016mastering`, `silver2017mastering`, `silver2018general`, `lehman2011novelty`, `mouret2015mapelites`, `stanley2015greatness`, `metz2016twomoves`, `amos2023amortized`, `koza1992genetic`, `kusner2017grammar`, `agarwal2021precipice`) has been verified against primary records and included in `refs.bib`.

---

## 4. Claim Ledger (per Shared Writing Strategy §5 & §11)

| Subsection | Claim | Category | Source / Evidence Identifier |
| :--- | :--- | :--- | :--- |
| §5.1 | Autoregressive generation uses $\approx 2 P_{\text{active}}$ FLOPs/token; training uses $\approx 6 P$ FLOPs/token. | Cited / Analytical | `\citep{kaplan2020scaling,hoffmann2022chinchilla}` |
| §5.1 | Expected cost per verified win is $C_{\text{win}} = (C_{\text{attempt}} + \mathbb{E}[r] C_{\text{repair}}) / q$. | Definition / Analytical | `\cref{eq:bill}` (derived from `\cref{eq:taste}`) |
| §5.1 | 1,000 proposals $\times$ 5,000 output tokens at $\$1/\text{M}$ tokens costs $\$5.00$ in raw generation output. | Illustrative Scenario | Explicitly labeled illustrative arithmetic (Oct 2026 basis) |
| §5.2 | Unsuccessful attempts in self-play still contribute negative examples to the problem–program map. | Definition / Mechanism | Cross-referenced to Section 6 (`\cref{eq:mapLoss}`) |
| §5.3 | Dense generation compute per token drops $187.5\times$ ($\approx 190\times$) from 1.5T to 8B parameters. | Analytical | Ratio $1500 / 8 = 187.5$ via `\citep{kaplan2020scaling}` |
| §5.3 | Whether $C_{\text{win}}$ is U-shaped across model scale and reasoning effort is an empirical hypothesis. | Hypothesis / Measured | Tested in `\cref{fig:price}b` (pilot measurement surface) |
| §5.4 | AlphaGo estimated Move 37 at a 1-in-10,000 human probability. | Cited | `\citep{silver2016mastering,metz2016twomoves}` (Box 3) |
| §5.5 | AlphaGo Zero played 4.9M games; AlphaZero played ~44M chess and ~140M Go games. | Cited | `\citep{silver2017mastering,silver2018general}` |
| §5.5 | ~150 independent unseen problems yield a $\pm 8$-point 95% interval around a 60% win rate. | Analytical / Cited | Binomial CI: $1.96\sqrt{0.24/150} = 7.84\%$; `\citep{agarwal2021precipice}` |
| §5.6 | A 1.5T dense 8M-attempt run ($1.38 \times 10^{23}$ FLOPs) is $\approx 1\%$ of a $10^{25}$-FLOP pretraining run. | Projection / Cited | `\cref{tab:price}` and `\citep{epoch2024data}` |
| §5.6 | Holding 1.5T weights at 2 bytes/parameter requires $\approx 3\text{ TB}$ of memory before KV-cache. | Analytical | $1.5 \times 10^{12} \times 2\text{ B} = 3.0\text{ TB}$ (decimal) |

---

## 5. Exhibit Specifications & Required Data Inputs

1. **Figure 4 (`figures/fig4_price.pdf`, `\label{fig:price}`):**
   - **Panel (a):** Horizontal survival funnel over logged pilot attempts showing percentage surviving and cumulative GPU/CPU-hours at each stage: (1) Generated, (2) Parses & type-checks, (3) Executes within sandbox limits, (4) Scored on hidden folds by referee $V$, (5) Beats incumbent ($\Delta > 0$), (6) Passes independent champion re-verification. Use `ezaccent` (`#0072B2`) for surviving winners and greyscale for stillborn/also-ran drop-off.
   - **Panel (b):** 2D heatmap of measured pilot $C_{\text{win}}$ (USD, October 2026 basis) with generator tier on the vertical axis (`8B`, `32B`, `70B`, `1T MoE`) and reasoning token budget $t_{\text{think}}$ on the horizontal axis (`0`, `1k`, `2k`, `4k` tokens). Annotate cells with mean $[95\% \text{ bootstrap CI}]$.
2. **Table 5 (`\label{tab:price}`):**
   - Rendered in `sections/05_price.tex` using `booktabs`. All numbers match the planning brief's assumptions ($5{,}000$ output tokens/try, $\$2.50/\text{H100-hour}$, $20\%$ rollout MFU, $40\%$ update MFU on $10\%$ of tokens).

---

## 6. Hand-off & Open Questions for the Team

- **Closing Hand-off Sentence:** *"The method's job is now clear: raise $q$, so fewer moves are wasted."* (Matches the Section 5 brief verbatim and sets up Section 6, *EurekaZero: Learning Taste*).
- **Open Questions for Engineering/Experiments Team:**
  1. In the final pilot run for Figure 4b, is multi-turn repair completely disabled ($\mathbb{E}[r] = 0$, matching strict one-move generation in training as well as deployment), or do we log a separate $\mathbb{E}[r] > 0$ comparison column for the small-model correction tax?
  2. Which specific sealed problem is the median arena win so we can fill in the three running-example cost numbers in `§5.6`?