"""Compare every `pred-*.txt` in a directory as IEEE-754 bit patterns.

Used by both `deploy/ci/emitters.sh` and `deploy/ci/e2e-smoke.sh`: an emitted
program is only equivalent to its siblings when the bits agree, not the decimal
text. Prints the agreed bit stream on stdout; exits 1 naming the first file that
disagrees.
"""
import glob, os, struct, sys
out = sys.argv[1]
bits = {}
for path in sorted(glob.glob(os.path.join(out, "pred-*.txt"))):
    rows = [line.strip() for line in open(path) if line.strip()]
    bits[os.path.basename(path)] = [struct.pack("<d", float(v)).hex() for v in rows]
first = next(iter(bits.values()))
for name, value in bits.items():
    if value != first:
        print(f"MISMATCH {name}", file=sys.stderr); sys.exit(1)
print("\n".join(first))
