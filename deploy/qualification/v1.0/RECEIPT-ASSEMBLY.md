# REL-07 final receipt assembly

This is a non-executing receipt assembly procedure: it reads gate results and
never launches providers, named clients, models, training or signing. Use the
merged assembler with #1724's M3 disposition and #1737's same-candidate D2 binding.
Runtime source remains RC4 even though this tooling branch starts from newer main.

The following command is the exact partial-run plumbing check. Its expected
result is refusal / NOT_QUALIFIED with no final receipt. All output belongs in
external evidence custody; do not commit the publisher's partial receipts.

```bash
repo=$(git rev-parse --show-toplevel)
plan_root=${JOPE_PLAN_ROOT:-"$HOME/jope_plan"}
run_dir="$plan_root/evidence/REL-07/pub-t0612-28-v3"
inputs="$plan_root/evidence/REL-07/inputs-t0545/candidate-inputs-v3.json"
candidate_dir="$plan_root/evidence/REL-07/res08-t1955/candidate-ccaf14b210b3"
ruling="$plan_root/master/status/REL-07-M3-RULING-v1.0.0-WAIVER-t1955.md"
out="$plan_root/evidence/REL-07/dsc01-t0115/dry-run/assembled.json"

printf '%s  %s\n' \
  24cf732aff5bd18bdaf6363bc531ad728aac8ad9334289805f85470dc53d65ff \
  "$candidate_dir/rc4-manifest.json" \
  30767897219dcfba678bab550326e36fedb34f1bc202cc94001a1a01163d5b20 \
  "$candidate_dir/full-linux-x86_64/jopedime" \
  5c45cf3b6fea741d753de5be0eaec3ce9443602da9bf2258a96393705f9b8ddf \
  "$ruling" \
  e18be03525e066ed2cd6e16453aadbe2fd96dd27a981d23ac4c1a7bba551a84b \
  "$inputs" | sha256sum --check || exit 1

args=(
  --scaffold "$repo/deploy/qualification/v1.0/release-command-scaffold.json"
  --artifact "$candidate_dir/full-linux-x86_64/jopedime"
  --candidate-manifest "$candidate_dir/rc4-manifest.json"
  --controller-run "$run_dir"
  --controller-inputs "$inputs"
  --source-sha ccaf14b210b3b55a350218e60f6b49ef76524cf3
  --m3-waiver-ruling "$ruling"
  --release-version v1.0.0
  --output "$out"
)
for gate in M1 M2 M3 M4 M5 S1 S2 S3 S4 R1 R2 R3 R4 A1 A2 A3 A4 A5 \
            E1 E2 E3 U1 U2 T1 T2 T3 D1 D2; do
  args+=(--gate-receipt "$run_dir/$gate/result.json")
done
nice -n 10 python3 -B "$repo/deploy/qualification/v1.0/gates_receipt.py" "${args[@]}"
```

For final assembly, use the publisher's **completed run with all five external
provider bindings**, its exact immutable input file/pin, and a new external
`out` path. PUB's prepared next run is `pub-t0115-final-v4`; its final A5-bound
input path/hash must come from the inputs owner. Do not populate absent result
files or reuse another run's passing subsets. Final receipt admission requires
all 27 non-M3 gates PASS plus a raw M3 FAIL accepted as WAIVED_V1_0_0 under
the ruling. A draft preparation PR may precede that admission under t0115.
The assembler checks source/artifact/counts, the unwaived clauses,
integer guards and expiry; failure is not a partial success. `all_pass` remains
false when raw M3 is FAIL, even after its separate disposition is admitted.

The paired controller options retain `execution.json` machine/commands/exits/times,
runner/command-manifest hashes, the versioned input file and bindings, and all
result/log path/hash identities in `provenance`. All 28 receipts must belong to
that controller directory; missing bindings, interrupted/incomplete commands,
changed output hashes and metadata changed during assembly cause refusal. The
frozen controller's exit 1 for M3 is admitted only if the raw result separately
satisfies the waiver guards. A preflight refusal cannot substitute for that result.

The final delivery also needs the exact assembly argv/tool commit/hash and the
recorded Q1/default decision. These lifecycle records are not invented by the
assembler. Their absence keeps this a draft.
Preserve the assembled gate rows and raw counts verbatim. Capture and recheck
all input hashes around generation; refuse a controller run that changes.
Never claim OWNER release approval or aggregate signing from assembly alone.

Only after complete admission and provenance verification should the final
generated receipt enter this directory. Keep failed attempts, partial snapshots
and raw private inputs outside Git. Hand ENG the concrete receipt, checks and
source head before the gated release lifecycle proceeds.
