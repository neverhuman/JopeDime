#!/usr/bin/env python3
"""Publish the optional GPR experiment including failed parity and startup cost."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--run',type=Path,required=True)
a=p.parse_args()
out=a.run/'report'
assert not out.exists()
warm=json.loads((a.run/'gpu-warm-batch4/summary.json').read_text())
cold=json.loads((a.run/'gpu-batch4/summary.json').read_text())
cpu=json.loads((a.run/'reference/summary.json').read_text())
prefix=json.loads((a.run/'periodic-prefix-gpu/summary.json').read_text())
prefix_cpu={r['id']:r for r in json.loads((a.run/'periodic-prefix-reference/summary.json').read_text())['rows']}
groups=defaultdict(list)
for row in warm['rows']: groups[row['context']['kernel']].append(row)
results=[]
for kernel,rows in groups.items():
    results.append(dict(kernel=kernel,count=len(rows),passed=sum(r['status']=='parity_pass' for r in rows),maximum_absolute_prediction_error=max(r['max_absolute_prediction_error'] for r in rows),aggregate_full_cost_speedup=sum(r['reference_end_to_end_seconds'] for r in rows)/sum(r['end_to_end_seconds'] for r in rows)))
trace=defaultdict(list)
for row in prefix['rows']:
    reference=prefix_cpu[row['id']]
    trace[row['context']['steps']].append(dict(prediction_error=row['max_absolute_prediction_error'],theta_error=max(abs(x-y) for x,y in zip(row['theta'],reference['theta'])),passed=row['status']=='parity_pass'))
trace=[dict(steps=steps,count=len(rows),passed=sum(r['passed'] for r in rows),max_prediction_error=max(r['prediction_error'] for r in rows),max_theta_error=max(r['theta_error'] for r in rows)) for steps,rows in sorted(trace.items())]
report=dict(schema='parallel.experimental_gpr_report.v1',production_enabled=False,reward_admission=False,all_kernels_qualified=False,actual_programs=32,actual_programs_passing=28,world_size=8,evaluation_bias='adaptive_cut_heldout_teacher',final_tests_opened=False,cold_gpu_elapsed_seconds=cold['elapsed_seconds'],cpu_reference_elapsed_seconds=cpu['elapsed_seconds'],warm_inclusive_of_startup_elapsed_seconds=warm['elapsed_seconds'],kernels=results,periodic_step_prefix_diagnostics=trace,limitations=['Only four actual programs per kernel expression were tested; source-family coverage is not representative of the full corpus.','Reported warm ratios include CPU preprocessing, upload, initialization of model hyperparameters, optimization, prediction, and download, but exclude the separately timed one-time CUDA process initialization.','Cold elapsed GPU execution was slower than the CPU reference.','Pure periodic full-step predictions failed tolerance; some periodic compositions approach tolerance. No GPU rewards or predictors were admitted.','No production scheduler, persistent queue, safe CPU fallback integration, or eight-GPU saturation claim is provided by this prototype.'])
lines=['# Optional FP64 GPR prototype: measured qualification','',
    'The Rust/LibTorch prototype runs all eight existing GPR kernel expressions on eight A100 GPUs. **28 of 32 real programs pass prediction parity; the four pure-periodic programs fail. Production remains disabled and no teaching rewards or predictors changed.**','',
    'The comparison uses the pinned CPU evaluator on the same training-cut transformations and heldout feature rows. The declared tolerance is `1e-5 + 1e-5 × abs(reference prediction)` in normalized target units. Original final tests remain sealed.','',
    '| Kernel | Passing | Maximum prediction error | Warm full-cost speedup |','|---|---:|---:|---:|']
for r in results: lines.append(f"| {r['kernel']} | {r['passed']}/{r['count']} | {r['maximum_absolute_prediction_error']:.3g} | {r['aggregate_full_cost_speedup']:.2f}× |")
lines += ['',
    'Warm ratios include CPU cut loading/preprocessing, fitted feature transforms, hyperparameter initialization, upload, optimization, prediction, and download. They compare aggregate CPU time with aggregate per-program GPU time amortized across batches of four. They exclude one-time CUDA process initialization, which was measured separately at roughly 5.0–5.5 seconds per rank. Total elapsed time still includes that startup.', '',
    f"The cold eight-GPU run took {cold['elapsed_seconds']:.2f} seconds versus {cpu['elapsed_seconds']:.2f} seconds for the eight-process CPU reference. The warm-compute run, including its separately measured process startup, took {warm['elapsed_seconds']:.2f} seconds. A persistent queue is therefore required for any potential benefit; this short experiment does not establish campaign-level speedup or GPU saturation.", '',
    'The pure-periodic diagnostic changes only the step count of four actual programs, preserving parent program hashes and cut pointers. It shows near-identical early updates followed by strong numerical trajectory sensitivity:', '',
    '| Steps | Passing | Maximum parameter error | Maximum prediction error |','|---|---:|---:|---:|']
for r in trace: lines.append(f"| {r['steps']} | {r['passed']}/{r['count']} | {r['max_theta_error']:.3g} | {r['max_prediction_error']:.3g} |")
lines += ['',
    'The first-step parameters match exactly. Differences at steps 2–10 remain tiny, then grow into different optimization trajectories by step 20. Full-step errors are too large for numerical parity. Changing the tolerance would not make these equivalent teaching evaluations.', '',
    'The prototype preserves the original CPU inducing-row recipe, feature standardization, parameter initialization/layout, sum/product kernel registry, optimizer update clamps, and prediction clamp. LibTorch FP64 matrix operations and gradients use a different arithmetic order. Cholesky failures and the CPU pivot-floor regime return explicit backend errors.', '',
    'Recommended next step: extend independent numerical coverage for non-periodic kernels and benchmark larger homogeneous batches in persistent workers. Retain the pinned CPU path for unqualified cases. The active GP service, corpus, and fitted predictors remain unchanged. Any production integration needs explicit backend identities, queue ownership, resource admission, and exact fallback receipts.', '',
    'Source files: `src/experimental_gpr.rs`, `examples/gpr_cuda_bench.rs`, and `tools/{select_gpr_benchmark,run_gpr_benchmark,report_gpr_benchmark}.py`. The example is optional (`cargo build --release --features neural --example gpr_cuda_bench`) and is not linked into the production GP path.', '',
    'No new CSV tables or transformed matrices were stored. Persistent artifacts contain cut/program pointers, CPU/GPU prediction and parameter comparisons, timing receipts, logs, and hashes. Frozen encoder outputs remain unchanged.','']
out.mkdir()
(out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
(out/'report.md').write_text('\n'.join(lines))
sources=[a.root/name for name in ['src/experimental_gpr.rs','examples/gpr_cuda_bench.rs','tools/audit_gpu_ridge.py','tools/select_gpr_benchmark.py','tools/run_gpr_benchmark.py','tools/report_gpr_benchmark.py','assets/source/jope-dime/vendor/feat-math/src/gpr.rs','assets/source/jope-dime/vendor/feat-math/src/heads.rs','assets/source/jope-dime/vendor/feat-math/src/ops.rs','bin/gpr-cuda-bench-v2','bin/gpr-cuda-bench-v3']]
files={str(f.relative_to(a.root)):dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size) for f in sources+list(a.run.rglob('*.json'))+list(a.run.rglob('*.log'))+[out/'report.md'] if f.is_file()}
(out/'receipt.json').write_text(json.dumps(dict(schema='parallel.experimental_gpr_report_receipt.v1',files=files,production_enabled=False,reward_admission=False,final_tests_opened=False),indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(output=str(out),files=len(files),production_enabled=False,parity='28/32 actual programs; pure-periodic rejected'),indent=2))
