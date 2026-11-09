# H329 evaluator field correction

Pre-score code review found the H325-derived evaluator wrote `XT` in chosen-config
records but read `DART` when counting config agreement. The original raises
`KeyError: DART` before producing metrics. A data-free end-to-end summary fixture
reproduces the failure and passes after changing the written field to `DART`.
The raw-family-gains NPZ field is also named `DART` rather than inherited `XT`.

This is a report field correction only. No fit, model, prediction, selected
configuration, numerical function, bootstrap RNG, threshold, input closure,
preregistration, or resource/deadline is changed. The running fit workers keep
their original source in memory; their source bytes are unchanged. The corrected
evaluator and manifest are installed before the fresh once-only score process
starts. Existing fits and synthetic timing controls are never restarted.
Original evaluator/manifest bytes and the failing fixture log are retained in
local custody. CI NOT_RUN under CI_FREEZE and MASTER_MERGE_HOLD.
