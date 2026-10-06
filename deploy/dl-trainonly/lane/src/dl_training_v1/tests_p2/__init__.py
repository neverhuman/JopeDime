"""Offline P2 tests (PLAN §5 P2): family keys, near-duplicate screen, pin and the core family registry build.

Run each module only inside the lane slice, with TMPDIR=RUN/tmp, e.g.
  systemd-run --user --scope --quiet --slice=jopedime-dltrain.slice ... python -m unittest discover -s tests_p2 -t . -v
No test touches the network, the live registry or S3; every write goes under TMPDIR.
"""
