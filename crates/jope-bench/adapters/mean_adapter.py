#!/usr/bin/env python3
"""jope-bench adapter (contract jope.bench-adapter/v1): the constant baseline, standard library only.

Regression predicts the training mean, binary the training positive rate, multiclass the majority
training class. It exists to exercise the adapter contract in tests and as the floor every other
adapter must clear.

    mean_adapter.py --version
    mean_adapter.py <train.csv> <predict.csv> <out.csv> <task>
"""
import sys
from collections import Counter


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--version":
        print("mean 1")
        return 0
    if len(sys.argv) != 5:
        sys.stderr.write(__doc__)
        return 2
    train, predict, out, task = sys.argv[1:5]
    targets = [float(line.rsplit(",", 1)[1]) for line in open(train) if line.strip()]
    rows = sum(1 for line in open(predict) if line.strip())
    if not targets:
        sys.stderr.write("no training rows\n")
        return 1
    if task == "regression":
        value = sum(targets) / len(targets)
    elif task == "binary":
        value = sum(1.0 for t in targets if t == 1.0) / len(targets)
    elif task == "multiclass":
        value = float(Counter(int(t) for t in targets).most_common(1)[0][0])
    else:
        sys.stderr.write(f"unknown task {task!r}\n")
        return 2
    with open(out, "w") as handle:
        for _ in range(rows):
            handle.write(f"{value!r}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
