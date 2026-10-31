# H73 — family-balanced sampling diagnostic and follow-up preregistration

Status: CPU diagnostic preregistration. No GPU cell is authorized or queued by this file.

## Question

Do large DL-Train families contribute a disproportionate share of the frozen r882 adapter's context-level training objective, and do those contributions include many contexts whose predicted best program fails the gate and therefore receives utility `-2.0`?

The CPU diagnostic uses the immutable offline copy of round 882, evaluates its H48 `CONTROL` row objective over the 1,219 DL-Train contexts, and reports each of the five seed-4900 whole-family folds separately. Since r882 itself was trained on all DL-Train families, the folds are descriptive partitions only. This is not an out-of-sample estimate, evidence that family size caused a training effect, or a model admission result.

## CPU diagnostic input and outcome rules

- Input cohort: the admitted H50 real-S3-derived packet, DL-Train only (1,219 contexts, 205 resolved family groups); no validation, outer, fold-B, holdout, or sealed payloads.
- The reader consumes only the first `train` array in `selected.json` and stops at its closing bracket. It opens pool and JTT1 table files only for those train context hashes, checking each against the row's recorded SHA256. The committed H67 fold manifest supplies the five family-group assignments at seed 4900.
- Checkpoint: read-only offline H51 copy `round-882.pt`, SHA256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`; verify round 882, DOPE SHA256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, and exactly 29,211 trainable adapter parameters. No live-loop path is opened or modified.
- Per-context objective: H48 `CONTROL`: target cost is `fitness` for gate-passing candidates and `+2.0` for gate failures; row loss is mean smooth-L1 plus `0.1 × KL(softmax(-target/0.2) || softmax(-prediction/0.2))`. Report raw context-uniform loss mass and the same mass under H48's full-cohort family-balanced weights.
- Also report the predicted argmin utility for each context (`-fitness` when the selected candidate passes, otherwise `-2.0`), the number and loss mass of selected `-2.0` contexts, and the signed loss contribution from failed-candidate `+2.0` targets. No candidate is fitted or trained.
- Stop if host CPU exceeds 85%, RSS reaches 8 GiB, or `MemAvailable` falls below 16 GiB. Run at nice 19 with OMP/MKL/OpenBLAS capped at four threads.

## GPU follow-up (queued behind H57, H59, and H61)

The following preregistered comparison may start only after H57, H59, and H61 are complete and a new explicit owner launch authorization is recorded. Do not create a lock waiter. This is a future plan, not authorization to run now.

- Arms: `CONTROL` (H48 loss and context-uniform sampling), inverse-square-root family weighting, and uniform-per-family context sampling. Every arm receives the same optimizer, batch size, and exact total optimizer-step count. The treatment changes only its family weighting or context sampler.
- Seeds: 12, 13, and 14, all retained and reported. No best seed, fold, or endpoint selection.
- Budget: three arms × three seeds = nine equal cells. Set each to four of the eight H57 timing-equivalent generations, giving the same fixed steps in every cell. At the conservative H57 projection of about 600 seconds per eight-generation cell, each four-generation cell projects to 300 seconds; nine cells project to 2,700 seconds. A 10% setup/measurement reserve yields 2,970 seconds, below the 3,600-second total cap. If the prelaunch timing receipt projects above 3,600 seconds, defer for owner re-budget; do not shorten or selectively drop cells after outcomes.
- Inputs and interpretation: DL-Train only; frozen DOPE `c51a56d0`; adapter-only (29,211 parameters). Any score is conditional on the r882 initialization, which already saw this cohort. It cannot establish untouched-family or external generalization.
- Primary report order: family-grouped Bradley–Terry Elo against r882 first, using the repository's fixed-panel helper, r882 anchored at 1500, tie margin `1e-4`, prior `.01`, 400 iterations, followed by W/T/L. Then report paired per-family utility gains versus r882; family mean, median, fraction below `-0.01`, and a 10,000-draw paired family-cluster bootstrap (seed `20261031`) with a one-sided mean LCB95 and two-sided median 95% interval. Elo is not AUC and does not replace the raw-utility gate.
- Advancement requires mean LCB95 `> 0.001`, family median `>= 0`, and no more than 10% of families below `-0.01`; ties fail. A pass is incremental development evidence only; serving or broad qualification is out of scope.
- Queue this comparison only after H57, H59, and H61. No S3 writes, holdout reads, live-loop access, or DOPE changes.
