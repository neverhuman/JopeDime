# Audited per-step recovery telemetry

This packet preserves selected original telemetry from21 raw metrics.jsonl logs, cycles000001–000021 of eight-a100-recovery-20260912. The source manifest binds full original file paths, SHA256, byte sizes, every block's corpus and parent/final checkpoint identity. Raw sources stay in retained custody; model files and sealed targets were not opened.

`a100-step-metrics.csv.gz` is deterministic gzip (mtime0, no stored filename). It contains63,000 rows, one for every step6001–69000. The six metric columns preserve378,000 original JSON numeric lexemes exactly, without rounding: composite `loss`, reported objective components `sequence_nll`, `ranking_bt`, `ranking_listwise`, `runtime_huber`, and `gradient_norm`. Step, local-step and timestamp tokens are also preserved. Cycle plus one-based source_line locates each original row; constant corpus, parent, world/rank and batch fields resolve through source-manifest.json. Other raw telemetry fields are deliberately not copied. Gradient norm is a reported diagnostic, not per-term gradient attribution. Missing contextfree_bt and LR are not zeros.

`a100-block-statistics.csv` contains126 metric/block rows: first/last sample, mean, first/last100 means, descriptive OLS slope/R² and fitted-span delta. `a100-step-bins.csv` contains3780 bins (630 per metric), each exactly100 contiguous rows inside one corpus. `a100-corpus-boundaries.csv` contains20 transitions. No line connects corpora in the figure. There are21 negative composite slopes;20 positive single-sample boundary jumps but19 positive100-mean jumps.

OLS is ordinary arithmetic over dependent, changing-population observations. No confidence interval, p-value, causal effect, independent generalization or downstream GP benefit is claimed. Adaptive checkpoint validation is the already audited78-state series (22corpora,21panels);296 overlapping validation cells in74 states cross-match these raw logs. Earlier fixed2048-example recovery/cycle25/cycle107 monitoring remains separate.

The data neither promotes69000 nor replaces accepted60000. Native model hashes are not reloaded or refitted. The retained world1 hardware qualification60100 is a separate run. Current native GPU dispatch/admission and the bounded training64 proof remain pending as described in chapters14/16.

Recreate extraction into a NEW custody output directory; it intentionally refuses an existing directory:

```sh
python3 docs/internal/asm-manual/tools/extract_per_step.py \
  --source /home/ubuntu/jope-migration-evidence/claude/node-high-value/files/home/ubuntu/jope-dime/parallel/runs/eight-a100-recovery-20260912/learning \
  --history docs/internal/asm-manual/data/training-history \
  --output /YOUR_CUSTODY/new-per-step-extraction
```

This checks each selected numeric token on a second read, rehashes each source, checks every row's parent/corpus/world/rank/batch, validates contiguous steps, and cross-matches retained checkpoint validation. It reads telemetry, not model bodies. Regenerate the compact SVG from shipped data with `python3 docs/internal/asm-manual/tools/make_per_step_figure.py`; that helper rechecks all630 plotted means against the full compressed CSV. `--preview` optionally writes a PNG for local visual review and is not needed for the manual. Both helpers leave original sources unchanged.

Source recovery boundary: inspected cycle000000/train-2 has metadata but no metrics.jsonl. Steps3001–6000 lack per-step records there; antecedent1–3000 is outside this source set. This is not a global absence claim. The older108-block capture still has only endpoints, and gen31's log has no embedded event timestamp.
