"""Shared read/write boundary audit for CPU-only historical flywheel runners."""

import os
from pathlib import Path

class Audit:
    def __init__(self, allowed, output):
        self.allowed = {str(Path(p).resolve()) for p in allowed}
        self.output = Path(output).resolve()
        self.opened = set()
        self.refused = []

    def check(self, event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        s = str(p)
        mode = args[1]
        flags = args[2] if len(args) > 2 else 0
        write = (isinstance(mode, str) and any(x in mode for x in "wax+")) or (
            isinstance(flags, int)
            and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        )
        protected = any(
            x in s.lower()
            for x in [
                "/sealed",
                "/holdout",
                "meta-test",
                "meta_test",
                "qualification.lock",
                "/data/flywheel_selfplay/",
            ]
        )
        data = p.is_relative_to(Path("/home/ubuntu/jope-data"))
        if (
            protected
            or (data and s not in self.allowed and not p.is_relative_to(self.output))
            or (write and s in self.allowed)
        ):
            self.refused.append(s)
            raise RuntimeError("input/output boundary refusal: " + s)
        if data or s.startswith("/home/ubuntu/JopeDimeMicro/"):
            self.opened.add(s)

    def receipt(self):
        return dict(opened=sorted(self.opened), refused=self.refused, protected_reads=0)
