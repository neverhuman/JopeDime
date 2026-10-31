# H131 — family-bootstrap adapter soup

**Preregistered; not run.** [PREREG.md](PREREG.md) freezes the hypothesis, family bootstrap, seeds97–99, fold131097, exact recipe and three-arm Holm admission gates. CONTROL uses40 updates; SOUP8 averages eight40-update adapters; SOUP8_HALF applies fixed half shrinkage toward r882. SOUP costs8x CONTROL compute; optional CONTROL_320 is omitted.

All outcomes remain unknown. [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) and [bootstrap_manifest.json](bootstrap_manifest.json) bind source, inputs and resampling before scoring. The source [runner.py](runner.py) runs only through [monitor.py](monitor.py) after the preregistration merges; numerical artifacts stay in external custody/S3, code stays in Git.

Red-team requirements: assessment closed through all training/candidate freezing; whole-family duplicate multiplicity retained; all eight members required; no cherry-picking or weight tuning; exact r882 ties never admitted; only three seeds/one assessment fold; prior DL-Train exposure and8x compute disclosed; all non-adapter parameters/buffers and DOPE checked unchanged. No unseen-family, local-optimum or serving claim.

Next step: merge preregistration, run the single bounded1,080-update CPU wave under resource guards, then publish the complete readout and verified numerical custody. No automatic restart or live-loop action.
