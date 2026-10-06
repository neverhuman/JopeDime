"""P4 tests (PLAN §5 P4), kept apart from tests/ so the concurrent P7 build is unaffected.

Run each module only inside the lane slice, e.g.
  systemd-run --user --scope --quiet --slice=jopedime-dltrain.slice ... python -m unittest tests_p4.test_oracle -v
"""
