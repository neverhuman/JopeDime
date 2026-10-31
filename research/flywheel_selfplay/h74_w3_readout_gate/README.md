# H74 — W3 paired-family readout gate

This CPU-only component implements the two preregistered W3 primary contrasts as a
fail-closed gate. It consumes an explicit summary from a separately admitted W3
export; it does not collect rows, fit or score models, establish provenance, or
authorize a W3 launch.

## Registered comparisons

The W3 proposal fixes `top_resolved_vs_h58_d` and
`decomposed_vs_smoothl1_kl` as the two primaries. The gate requires both to pass
the common thresholds: adjusted family-mean lower bound strictly above `+0.001`,
median family delta at least zero, and no more than 10% of the 205 families below
`-0.01`. It also requires no increase in the primary gate-failure rate for either
intervention against its matched control. The same 10,000 paired family bootstrap
draws resample both contrasts together; the two one-sided bounds use Bonferroni
alpha `0.05 / 2` and the nearest-rank percentile. This is a conservative
implementation choice for the preregistered within-wave interval adjustment, not
a claim of exact error control.

Three seed deltas are averaged within each family before resampling. The input
must identify all 1,219 DL-Train contexts, 205 families, the five-fold seed 4900,
the frozen DOPE digest, and SHA256 identities for the admitted W3 export, fold
manifest, W1 winner config, and H59 disposition. It must state zero DL-Val, outer,
and sealed rows. These are consistency checks on caller-supplied metadata; the
tool cannot verify the objects or establish that an export is eligible.

The gate refuses incomplete or duplicate family rosters, mismatched contrasts,
non-finite deltas, invalid failure counts, or a changed bootstrap contract. The
registered kill checks must be present: auxiliary-loss-only gain, shuffled-target
gain matching, or promotion of near-pass failures through changed evaluation
semantics blocks advancement; zero-update choice preservation is required.

## Run

```sh
nice -n 10 python3 -m unittest research.flywheel_selfplay.h74_w3_readout_gate.w3_gate
nice -n 10 python3 research/flywheel_selfplay/h74_w3_readout_gate/w3_gate.py --evidence PATH
```

This PR contains only synthetic tests and no W3 evidence. W1 winner/config,
admitted W3 outcomes, and the H59 disposition remain prerequisites to a study.
No result, Elo, or launch decision is reported here.
