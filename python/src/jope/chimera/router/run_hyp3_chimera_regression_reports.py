"""Run HYP3-augmented FINAL Chimera regression reports."""
from __future__ import annotations

from python.router.run_chimera_regression_reports import main


if __name__ == "__main__":
    raise SystemExit(main(default_feature_augmentation="hyp3"))
