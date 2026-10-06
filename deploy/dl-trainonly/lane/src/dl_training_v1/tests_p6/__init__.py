"""P6 tests (PLAN §5 P6, §6.4): train-only context encodings: the A0 meta-features (dltrain.metafeatures), the
train_fit normalizers (dltrain.normalizers) and the encodings CLI (run, adopt, verify, sample, fake-partition-map).

Offline only: synthetic JTT1 bytes and vectors, no S3 client, no GPU, no encoder inference. The test classes subclass
tests_p3.fixtures_p3.P3Case, whose armed audit hook fails any test that writes outside TMPDIR (RUN/tmp). Run inside
the lane slice, with zero skips expected:
  systemd-run --user --scope --quiet --slice=jopedime-dltrain.slice ... python -m unittest discover -s tests_p6 -t . -v
"""
