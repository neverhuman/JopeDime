# Core IP technical reference

Maintained twelve-chapter monograph, evidence inventory and publication renderer.
The public PDF and Markdown are generated together; source is reviewed through the
normal PR lifecycle. Historical data retain original dates and source identities.

The [regression remediation register](findings/README.md) preserves all ten audit
shards and their 195 individual entries. Inventory verification does not close
the findings or qualify the learning loop.

The [W1–W12 work orders](work-orders/README.md) expand the remediation plan into
numbered implementation tasks, dependencies, acceptance cases, evidence requirements
and gated execution phases. They are plans, not completed capability claims.

```sh
python3 docs/internal/core-ip/tools/build.py \
  --output /home/ubuntu/xbabe2_kicker/docs \
  --cutoff UTC_ISO_TIMESTAMP --study /ABSOLUTE/REPORT/summary.json
python3 docs/internal/core-ip/tools/verify_inventory.py
python3 docs/internal/core-ip/tools/verify.py /home/ubuntu/xbabe2_kicker/docs
```

Dependencies: Python3, Markdown, BeautifulSoup4, WeasyPrint, matplotlib, Pillow,
and Poppler (`pdfinfo`, `pdftotext`, `pdftoppm`). The renderer reuses the maintained
ASM manual slug policy and retained historical figure/data inputs. It performs no
network fetch, training, fitting or benchmark execution.

The original admitted v2 study failed before CUDA qualification and its staging parsed outer development labels before projection. Both findings are retained in chapter10. The repaired v3 planner requires a pinned fitting-only manifest and label-free split assignments; v2 plans remain reportable but cannot be executed by v3. No corrected-source GPU rerun is claimed.

GPU tooling lives in `crates/jope-bench/src/panel/inference_{benchmark,cuda,export,gp}.rs`
and the adjacent prediction/report Python modules. Chapter10 contains command
forms. Exact argument arrays, immutable binaries and staged fitting-only inputs
are retained in the study custody manifest. Build with `inference-cuda` and a
compiled `JOPE_SOURCE_SHA`; CPU fallback is explicitly refused. The run command
launches the existing systemd supervisor, CPU reservation and GPU lock; do not
wrap it in a second exclusive CPU lock. Drain the idle full runner only after
checking its API busy state, and arm a restoration trap and independent deadman.

The execution deadline belongs to the first admitted journal window. Neither a
new process nor a resume grants a new budget. Preserve failures and unexecuted
attempts. An unsupported original selected program is never replaced for export.
The timing-only paired backend cannot establish learning admission or promotion.

Focused verification:

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo test -p jope-bench --lib -j4
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo test -p jope-bench --features inference-cuda inference_benchmark --lib -j4
python3 -m unittest discover -s crates/jope-bench/src/panel -p test_inference_report.py
git diff --check
```

`verify.py` checks export hashes, page count, text extraction and PDF bookmarks,
and renders every page into contact sheets. A human/agent must inspect those
sheets and retain a separate signed-by-identity inspection receipt; creating
images alone is not visual verification. The 50–80-page bound applies to the
monograph, with the exact tensor inventory appended separately.
