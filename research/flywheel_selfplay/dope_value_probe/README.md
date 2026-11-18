# DOPE value-probe contract

Status: scored probe **NOT_RUN**.

This directory records the 629 layout and the arm widths from the DOPE value
specs. It does not load weights, fit a ranker, read val91 or sealed rows, or
take an H id.

Base at authoring: JopeDime `5ed0e108e6aece4d15460db3f55732448fbc668f`.
`crates/jope-encoders`, the fused ranker, and the freeze guard are unchanged.
H468's worktree is not used.

## What is fixed here

- Released `dope629` blocks, and the Dope512 prefix of the first 512 coordinates.
  Dope512 remains the separate product recipe
  `dope-mlcv3-high-f32-descriptor256-task128-dist128/v1`.
- Fine landmarks masked at level `high`: `gbdt`, `knn1`, `knn16`, `net`,
  `gbdt_quarter`.
- Arm widths for `R_none`, `R_shape`, `R_desc`, `R_learned` (`dope629[256:]`,
  373), `R_dope`, `R_hidden`, `R_tok`, `R_randrouter`, `R_shuffled`, and
  `R_noise`.
- `R_packet` is the global token `dope.packet-global/v0` only (24 values). It
  is not the full pooled column, pair, and tree packet.
- `R_desc_ultra` and `R_dope_ultra` use the same widths as `R_desc` and `R_dope`.
- `profile_oracle` is a ceiling readout, not a feature width.
- Contrasts C1–C5. The two C4 placebos must include 0. That check has not been run.

## What this is not

No scientific result. No new safetensors identity. No product recipe
`dope629-ultra/v1`. No program grammar leaves JopeDime.

Run the contract with:

```sh
python3 -B research/flywheel_selfplay/dope_value_probe/test_dope_value_probe.py
```
