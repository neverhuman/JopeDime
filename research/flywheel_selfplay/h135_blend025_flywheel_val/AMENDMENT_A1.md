# H135 Amendment A1 — enforce the four-total-thread ceiling

Prospective operational amendment, after preregistration PR #1191 merged and **before any H135 scientific execution, checkpoint load, baseline prediction, fit or blend outcome**. Original [PREREG.md](PREREG.md), runner.py and science.py remain byte-identical. This is not a retry: the scientific attempt has not started.

A separate synthetic native-runtime check found an otherwise hidden PyArrow jemalloc background thread: scikit-learn import used2 threads, and a synthetic HistGB fit with OpenMP4 used5. A Torch-only matrix fixture used4. The initial coordination note misattributed the failing phase to Torch; the recorded phase-specific probe corrects it. Generic MALLOC_CONF and ARROW_DEFAULT_MEMORY_POOL did not remove the helper on this host.

Set **JE_ARROW_MALLOC_CONF=background_thread:false** in each worker's environment before Python imports. The same synthetic HistGB fixture then uses1 thread after import and4 after fitting. This disables allocator background work; it changes no estimator parameter, training sample, model feature, target, calibration, prediction, baseline protocol or statistic. The monitor records the setting in its receipt. The external ceiling stays four total worker threads, with all other resource and read boundaries unchanged.

A new subprocess regression uses the real installed sklearn/Arrow/OpenMP libraries and synthetic arrays only. It fails under the original environment (five threads) and passes with the amended environment (four). It complements the original nine data-free controls; it is not a scientific result or GPU qualification.

The updated monitor, test and manifest must merge before launch. Verify their exact merged bytes, original prereg/runner/science identity, and a fresh seed-use audit. Then permit exactly the original one H135 attempt. If that attempt fails its r882 reproduction gate, stop before blend reading and record a separate prospective amendment; this amendment does not authorize automatic retries or any guard relaxation.
