# H256: Gradient Centralization and Adaptive Gradient Clipping for the r882 adapter continuation

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before anything runs.

## Hypothesis

**What has been tried.** Every gradient continuation of r882 tied with or lost to r882. None filtered each unit's
gradient relative to its own weights.

**What is new.** Two gradient filters on the exact H170 CTRL AdamW recipe: Gradient Centralization (GC; Yong et al.
2020), which removes each output row's mean gradient, and unit-wise Adaptive Gradient Clipping (AGC; Brock et al. 2021,
λ 0.01), which bounds each unit's gradient to a fraction of its weight norm. The question is whether they lift DL-Val91
over r882.

## Setup

- **Arms.** All use the same CTRL AdamW, loss, batches, data order, clip 1.0 and 40 updates from a read-only r882 copy:
  - **CONTROL:** exact H170 CTRL AdamW.
  - **GC_AGC (primary):** GC on every ≥ 2-D adapter gradient, then AGC (λ 0.01, ε 1e-3) on all adapter tensors.
  - **AGC_ONLY (mechanism control):** AGC only.
- **Hypothesis family.** GC_AGC and AGC_ONLY against r882, with Holm m = 2. GC_AGC − AGC_ONLY tests GC.
- **Design.** One fresh fold seed 256140, seeds 256101–256103, 15 cells per arm, the strict inner selector, and
  selections locked before assessment.
- **Decision.** It is made on DL-Val91: median gain > +0.002, LCB95 > 0, Holm p < .05, 3/3 seeds positive.
- **Also reported:** the step-40 unconstrained diagnostic, GC_AGC − CONTROL, GC_AGC − AGC_ONLY, drift, harm families,
  AGC clip fractions and the cliff context #27.
- **Host.** xbabe2 CPU cores 44–47, nice 19, under the H239 A1 memory guard. The compute stop is 16:00Z.

## Files

| File | Contents |
|---|---|
| [runner.py](runner.py) | the study runner |
| [treatment.py](treatment.py) | GC, AGC and CTRL |
| [analysis.py](analysis.py) | selection and inference |
| [monitor.py](monitor.py) | the A1 guard |
| [test_h256.py](test_h256.py) | data-free controls |
| [schedules.json](schedules.json) | the frozen settings |
| [manifest.json](manifest.json) | input, asset and source pins |
| [fold_manifest.json](fold_manifest.json) | the folds |
| [provenance.json](provenance.json) | the real-S3 closure |
| [metrics.json](metrics.json) | the metrics schema placeholder |
