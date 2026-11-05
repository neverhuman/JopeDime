"""H357 data-free tests (synthetic arrays only). Run from the staged code root (stage_h357.sh), which also holds the
H354 modules: PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest test_h357 test_h354 -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h354_core as K  # noqa: E402
import phase_c_h357 as PC  # noqa: E402

ST, E = K.ST, K.E


class H357Tests(unittest.TestCase):
    def test_fixed_design(self):
        self.assertEqual(PC.ARM, "X_CL")
        self.assertEqual(tuple(PC.LIBRARY), ("catboost.v1", "lightgbm.v1", "xgboost.v1", "random_forest.v1"))
        self.assertEqual((PC.SEED_C, PC.STAGE_C, PC.BOOT_SEED), (357101, "C_AGG", 335150))
        self.assertEqual(K.STACK_KEY, "AGG.g50.CV3@12")
        self.assertEqual(K.C_LIB, 0.0)

    def test_verdict_map(self):
        ok, no = {"gate": {"pass": True}}, {"gate": {"pass": False}}
        self.assertEqual(PC.verdict({"pass": True, "inferior": False}, ok), "XSTACK_CL_CONFIRMED_ADOPTION_GRADE")
        self.assertEqual(PC.verdict({"pass": True, "inferior": False}, no), "XSTACK_CL_CONFIRMED")
        self.assertEqual(PC.verdict({"pass": False, "inferior": True}, ok), "XSTACK_CL_INFERIOR")
        self.assertEqual(PC.verdict({"pass": False, "inferior": False}, ok), "XSTACK_CL_NOT_CONFIRMED")

    def test_x_cl_ignores_llm_and_audit(self):
        """X_CL weights use the 4 classical SEARCH OOF predictions only: LLM entries and AUDIT arrays change nothing."""
        import test_h354 as T

        ctx = T.synth_ctx(7)
        ls = {n: K.lib_search_result(ctx, n, ctx.fd.y0 + 0.2, ctx.fd.y1 - 0.1) for n in K.LIBRARY}
        a = K.stack_arm(ctx, ls, PC.ARM)
        for n in K.LLM:
            ls[n] = K.lib_search_result(ctx, n, ctx.fd.y0, ctx.fd.y1)  # a perfect LLM member is still not a candidate
        ctx.ya = ctx.ya * 3.0
        b = K.stack_arm(ctx, ls, PC.ARM)
        np.testing.assert_array_equal(a.w_audit, b.w_audit)
        self.assertEqual(a.members, b.members)
        self.assertFalse(any(m[4:] in K.LLM for m in b.members if m.startswith("lib:")))

    def test_amendment1_missing_audit_members(self):
        """A GP member that X0 leaves at zero weight but X_CL uses has no AUDIT record: it is listed for the
        Amendment-1 audit-extra run; library members and members with AUDIT records are not."""
        import test_h354 as T

        ctx = T.synth_ctx(11)
        ls = {n: K.lib_search_result(ctx, n, ctx.fd.y0 + 0.4, ctx.fd.y1 - 0.3) for n in K.LIBRARY}
        spec = K.stack_arm(ctx, ls, PC.ARM)
        declared = spec.declared()[0]
        gp = [s for s in declared if not s.startswith("lib:")]
        self.assertEqual(PC.gp_members_without_audit(spec, ctx.audit_by_sha), [])
        drop = dict(ctx.audit_by_sha)
        del drop[gp[0]]
        self.assertEqual(PC.gp_members_without_audit(spec, drop), [gp[0]])
        # weights are unchanged by which AUDIT records exist
        np.testing.assert_array_equal(K.stack_arm(ctx, ls, PC.ARM).w_audit, spec.w_audit)

    def test_panel_pin(self):
        self.assertEqual(PC.P201C_CIDS_SHA256, "36fbf6a92204487a31f2533905a57393e9c317fa7484ab7cbde5026e34468f6d")


if __name__ == "__main__":
    unittest.main()
