"""P6 spec tests T13-T20 (+ the T39 hygiene scan) for dltrain.normalizers: train_fit-only refits (DL-D4a), canonical
serialization, PCA widths and the deterministic Jacobi eigendecomposition. Synthetic vectors and partition maps only;
every write goes under TMPDIR (P3 write guard)."""
import ast
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np

from dltrain import common, programs
from dltrain import normalizers as nz
from tests_p3 import fixtures_p3 as fx

SRC = Path(nz.__file__)
LANE = Path(__file__).resolve().parents[1]
EPOCH_SHA = hashlib.sha256(b"p6-normalizer-fixture:epoch").hexdigest()
PMAP_SHA = hashlib.sha256(b"p6-normalizer-fixture:partition-map").hexdigest()
FEATURE_LIST_SHA = hashlib.sha256(b"p6-normalizer-fixture:feature-list").hexdigest()
FORBIDDEN_MODULES = {"jope", "gp_worker", "local_gateway", "reuse_cache", "intake_registry"}
FORBIDDEN_STRING = "jopedime-regression-research-2k-" + "tabpfn35-unseen-20261001"
CONST_DIMS = (4, 5)
D = 88


def cid(tag: str) -> str:
    return hashlib.sha256(f"p6-normalizer-fixture:{tag}".encode()).hexdigest()


def build_map() -> dict:
    """8 train_fit families of 5 contexts (one diag context in family tf2), 2 dev, 2 heldout, 1 sealed and
    1 prohibited family."""
    pmap = {}
    for f in range(8):
        for c in range(5):
            pmap[cid(f"tf{f}:{c}")] = {"family_id": f"fam-tf{f}", "split_role": "train_fit"}
    pmap[cid("tf2:4")]["split_role"] = "diag"
    for role, tag, fams, per in (("dev", "dv", 2, 3), ("heldout", "ho", 2, 3), ("sealed", "se", 1, 2),
                                 ("prohibited", "pr", 1, 2)):
        for f in range(fams):
            for c in range(per):
                pmap[cid(f"{tag}{f}:{c}")] = {"family_id": f"fam-{tag}{f}", "split_role": role}
    return pmap


def build_vectors(pmap, seed=0, d=D) -> dict:
    """Correlated vectors (rank well above N-1 is impossible here, so the fit is rank-deficient like the sample)
    with constant dims 4 and 5 on every context."""
    rng = np.random.default_rng(seed)
    mix = rng.standard_normal((10, d))
    out = {}
    for ctx in sorted(pmap):
        v = rng.standard_normal(10) @ mix + 0.3 * rng.standard_normal(d) + np.linspace(-2, 2, d)
        v[list(CONST_DIMS)] = (0.1, 5.0)
        out[ctx] = v
    return out


def all_rows(pmap, vecs):
    return [nz.FitRow(c, pmap[c]["family_id"], pmap[c]["split_role"], vecs[c]) for c in sorted(vecs)]


def train_rows(pmap, vecs):
    return [r for r in all_rows(pmap, vecs) if pmap.get(r.context_sha256, {}).get("split_role") == "train_fit"]


def fit(rows, pmap, **kw):
    args = dict(partition_map=pmap, arm="A0", block_version="jope.dl-a0-meta/v1", feature_list_sha256=FEATURE_LIST_SHA,
                epoch_sha256=EPOCH_SHA, partition_map_sha256=PMAP_SHA, partition_map_kind="fake")
    args.update(kw)
    return nz.fit_block("m", rows, **args)


class NormCase(fx.P3Case):
    def setUp(self):
        super().setUp()
        self.pmap = build_map()
        self.vecs = build_vectors(self.pmap)
        self.rows = train_rows(self.pmap, self.vecs)
        self.n = fit(self.rows, self.pmap)


# ------------------------------------------------------------------------------------------------ T13
class T13FitRefusal(NormCase):
    def first(self, role):
        return next(c for c in sorted(self.pmap) if self.pmap[c]["split_role"] == role)

    def test_given_role_refused_with_context_and_role(self):
        for role in ("dev", "diag", "heldout", "sealed", "prohibited"):
            ctx = self.first(role)
            row = nz.FitRow(ctx, self.pmap[ctx]["family_id"], role, self.vecs[ctx])
            with self.assertRaises(nz.NormalizerFitRefused) as cm:
                fit(self.rows + [row], self.pmap)
            self.assertIn(ctx, str(cm.exception))
            self.assertIn(role, str(cm.exception))
            self.assertEqual((cm.exception.context_sha256, cm.exception.split_role), (ctx, role))

    def test_map_role_refused_even_when_the_row_claims_train_fit(self):
        for role in ("dev", "diag", "heldout", "sealed", "prohibited"):
            ctx = self.first(role)
            row = nz.FitRow(ctx, self.pmap[ctx]["family_id"], "train_fit", self.vecs[ctx])
            with self.assertRaises(nz.NormalizerFitRefused) as cm:
                fit(self.rows + [row], self.pmap)
            self.assertIn(ctx, str(cm.exception))
            self.assertIn(role, str(cm.exception))
            self.assertEqual((cm.exception.reason, cm.exception.split_role), ("map_role", role))

    def test_unmapped_and_family_mismatch(self):
        stray = cid("nowhere")
        with self.assertRaises(nz.NormalizerFitRefused) as cm:
            fit(self.rows + [nz.FitRow(stray, "fam-tf0", "train_fit", np.zeros(D))], self.pmap)
        self.assertIn(stray, str(cm.exception))
        self.assertIn("unmapped", str(cm.exception))
        r0 = self.rows[0]
        with self.assertRaises(nz.NormalizerFitRefused) as cm:
            fit([r0._replace(family_id="fam-other")] + self.rows[1:], self.pmap)
        self.assertIn(r0.context_sha256, str(cm.exception))
        self.assertEqual(cm.exception.reason, "family_mismatch")

    def test_shape_value_and_size_refusals(self):
        r = self.rows
        cases = {
            "empty": [],
            "duplicate": r + [r[3]],
            "ragged": r[:-1] + [r[-1]._replace(vector=np.zeros(D - 1))],
            "nonfinite_nan": r[:-1] + [r[-1]._replace(vector=np.where(np.arange(D) == 9, np.nan, 1.0))],
            "nonfinite_inf": r[:-1] + [r[-1]._replace(vector=np.where(np.arange(D) == 9, np.inf, 1.0))],
            "too_few": r[:1],
        }
        for name, rows in cases.items():
            with self.assertRaises(nz.NormalizerFitRefused, msg=name):
                fit(rows, self.pmap)

    def test_no_arithmetic_runs_before_a_refusal(self):
        ctx = self.first("heldout")
        bad_sets = [
            self.rows + [nz.FitRow(ctx, self.pmap[ctx]["family_id"], "heldout", self.vecs[ctx])],
            self.rows + [nz.FitRow(ctx, self.pmap[ctx]["family_id"], "train_fit", self.vecs[ctx])],
            self.rows + [nz.FitRow(cid("nowhere"), "fam-tf0", "train_fit", np.zeros(D))],
            self.rows + [self.rows[0]],
            self.rows[:-1] + [self.rows[-1]._replace(vector=np.zeros(3))],
            self.rows[:-1] + [self.rows[-1]._replace(vector=np.full(D, np.nan))],
            self.rows[:1],
            [],
        ]
        with mock.patch.object(nz.np, "einsum", side_effect=AssertionError("einsum ran")) as einsum, \
                mock.patch.object(nz, "jacobi_eigh", side_effect=AssertionError("eigh ran")) as eigh, \
                mock.patch.object(nz, "_compute", side_effect=AssertionError("arithmetic ran")) as compute:
            for rows in bad_sets:
                with self.assertRaises(nz.NormalizerFitRefused):
                    fit(rows, self.pmap)
        self.assertEqual((einsum.call_count, eigh.call_count, compute.call_count), (0, 0, 0))

    def test_argument_refusals(self):
        for kw in ({"partition_map_kind": "real"}, {"epoch_sha256": "abc"}, {"partition_map_sha256": None},
                   {"holdout_binding_sha256": "x" * 64}, {"max_components": 0}, {"max_components": True},
                   {"arm": ""}, {"feature_list_sha256": "F" * 64},
                   # TOL-01: p3 is bound (both binding hashes); a fake map carries neither
                   {"partition_map_kind": "p3"}, {"partition_map_kind": "p3", "holdout_binding_sha256": "1" * 64},
                   {"partition_map_kind": "p3", "partition_map_binding_sha256": "2" * 64},
                   {"holdout_binding_sha256": "1" * 64}, {"partition_map_binding_sha256": "2" * 64},
                   {"partition_map_kind": "p3", "holdout_binding_sha256": "1" * 64, "partition_map_binding_sha256": "Z" * 64}):
            with self.assertRaises(nz.NormalizerError, msg=str(kw)):
                fit(self.rows, self.pmap, **kw)

    def test_p3_provenance_is_recorded_and_decoded(self):
        n = fit(self.rows, self.pmap, partition_map_kind="p3", holdout_binding_sha256="1" * 64,
                partition_map_binding_sha256="2" * 64)
        f = n.header["fit"]
        self.assertEqual((f["partition_map_kind"], f["holdout_binding_sha256"], f["partition_map_binding_sha256"]),
                         ("p3", "1" * 64, "2" * 64))
        self.assertEqual(nz.from_bytes(*nz.to_bytes(n)), n)
        self.assertNotEqual(n.sha256, self.n.sha256)
        self.assertEqual(n.blob, self.n.blob)                             # provenance changes the header, never the arrays
        for edit in ({"partition_map_binding_sha256": None}, {"holdout_binding_sha256": None}):
            doc = json.loads(n.header_bytes)
            doc["fit"].update(edit)
            with self.assertRaises(nz.NormalizerError, msg=str(edit)):
                nz.from_bytes(common.canonical(doc), n.blob)
        doc = json.loads(self.n.header_bytes)
        doc["fit"]["partition_map_binding_sha256"] = "2" * 64             # a fake map claiming a binding
        with self.assertRaises(nz.NormalizerError):
            nz.from_bytes(common.canonical(doc), self.n.blob)


# ------------------------------------------------------------------------------------------------ T14
class T14FitSetHashes(NormCase):
    def test_hashes_recomputed_independently(self):
        ctxs = sorted(r.context_sha256 for r in self.rows)
        fams = sorted({r.family_id for r in self.rows})

        def h(doc):
            return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                                  .encode()).hexdigest()
        fit_doc = self.n.header["fit"]
        self.assertEqual(fit_doc["fit_context_set_sha256"], h({"schema": "jope.dl-fit-context-set/v1", "contexts": ctxs}))
        self.assertEqual(fit_doc["fit_family_set_sha256"], h({"schema": "jope.dl-fit-family-set/v1", "families": fams}))
        self.assertEqual((fit_doc["n_contexts"], fit_doc["n_families"]), (len(ctxs), len(fams)))
        self.assertEqual(nz.fit_context_set_sha256(list(reversed(ctxs))), fit_doc["fit_context_set_sha256"])
        self.assertEqual(fit_doc["partition_role"], "train_fit")
        self.assertEqual((fit_doc["epoch_sha256"], fit_doc["partition_map_sha256"], fit_doc["partition_map_kind"],
                          fit_doc["holdout_binding_sha256"], fit_doc["partition_map_binding_sha256"]),
                         (EPOCH_SHA, PMAP_SHA, "fake", None, None))
        train_families = {f for f in {e["family_id"] for e in self.pmap.values()}
                          if {e["split_role"] for e in self.pmap.values() if e["family_id"] == f} <= {"train_fit", "diag"}}
        self.assertTrue(set(fams) <= train_families)
        self.assertNotIn(cid("tf2:4"), ctxs)                    # the diag context of a train family is not fitted

    def test_assert_fit_closure(self):
        present = [r.context_sha256 for r in self.rows]
        got = nz.assert_fit_closure(self.n, self.pmap, reversed(present))
        self.assertEqual(got["fit_context_set_sha256"], self.n.header["fit"]["fit_context_set_sha256"])
        with self.assertRaises(nz.NormalizerFitRefused):
            nz.assert_fit_closure(self.n, self.pmap, present[1:])
        dev = next(c for c in self.pmap if self.pmap[c]["split_role"] == "dev")
        with self.assertRaises(nz.NormalizerFitRefused) as cm:
            nz.assert_fit_closure(self.n, self.pmap, present + [dev])
        self.assertEqual(cm.exception.split_role, "dev")
        leaky = {k: dict(v) for k, v in self.pmap.items()}
        leaky[cid("held-in-train-family")] = {"family_id": "fam-tf0", "split_role": "heldout"}
        with self.assertRaises(nz.NormalizerFitRefused) as cm:
            nz.assert_fit_closure(self.n, leaky, present)
        self.assertEqual(cm.exception.reason, "family_not_train_fit")
        doc = json.loads(self.n.header_bytes)
        doc["fit"]["fit_family_set_sha256"] = hashlib.sha256(b"tampered").hexdigest()
        doc["arrays"] = self.n.header["arrays"]
        tampered = nz.from_bytes(common.canonical(doc), self.n.blob)
        with self.assertRaises(nz.NormalizerFitRefused):
            nz.assert_fit_closure(tampered, self.pmap, present)


# ------------------------------------------------------------------------------------------------ T15
class T15Serialization(NormCase):
    def test_canonical_header_and_array_table(self):
        hb, blob = nz.to_bytes(self.n)
        doc = json.loads(hb)
        self.assertEqual(json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                    allow_nan=False).encode(), hb)
        keys = []

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    keys.append(k.lower())
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(doc)
        self.assertEqual([k for k in keys if any(t in k for t in ("utc", "time", "host", "pid", "date", "user"))], [])
        self.assertEqual(self.n.sha256, hashlib.sha256(hb).hexdigest())
        d, r = doc["dims"]["in"], doc["dims"]["width_full"]
        shapes = {"mean": [d], "scale": [d], "rotation": [d, r], "proj_mean": [r], "proj_scale": [r], "eigenvalues": [r]}
        offset = 0
        for entry, name in zip(doc["arrays"], nz.ARRAY_ORDER):
            self.assertEqual(entry, {"name": name, "dtype": "<f4", "shape": shapes[name], "offset": offset,
                                     "bytes": 4 * int(np.prod(shapes[name]))})
            offset += entry["bytes"]
        self.assertEqual(offset, len(blob))
        self.assertEqual(doc["blob"], {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)})
        self.assertEqual(doc["schema"], "jope.dl-normalizer/v1")
        self.assertEqual(doc["experiment_id"], "gp-policy-dl-v1-e0")
        self.assertEqual((doc["method"]["dtype_compute"], doc["method"]["dtype_stored"], doc["method"]["gram"],
                          doc["method"]["clip_abs"], doc["method"]["rank_rtol"]),
                         ("float64", "<f4", "einsum(optimize=False)", 32.0, 1e-10))

    def test_round_trip_and_tamper_refusals(self):
        hb, blob = nz.to_bytes(self.n)
        again = nz.from_bytes(hb, blob, normalizer_sha256=self.n.sha256)
        self.assertEqual(again, self.n)
        self.assertEqual(nz.to_bytes(again), (hb, blob))
        flipped = bytearray(blob)
        flipped[len(blob) // 2] ^= 0x01
        doc = json.loads(hb)
        doc_arm = dict(doc, arm="A1")
        doc_dims = json.loads(hb)
        doc_dims["dims"]["width_full"] += 1
        doc_rank = json.loads(hb)
        doc_rank["dims"]["constant_dims"] = [4, 5, 6]                  # dim 6 has a real scale and rotation row
        refusals = {
            "flipped blob byte": (hb, bytes(flipped), None),
            "edited header field (expected sha)": (common.canonical(doc_arm), blob, self.n.sha256),
            "edited dims": (common.canonical(doc_dims), blob, None),
            "edited constant dims": (common.canonical(doc_rank), blob, None),
            "short blob": (hb, blob[:-4], None),
            "long blob": (hb, blob + b"\0\0\0\0", None),
            "non-canonical header": (json.dumps(doc, indent=1, sort_keys=True).encode(), blob, None),
            "wrong schema": (common.canonical(dict(doc, schema="jope.dl-normalizer/v2")), blob, None),
        }
        for name, (h, b, sha) in refusals.items():
            with self.assertRaises(nz.NormalizerError, msg=name):
                nz.from_bytes(h, b, normalizer_sha256=sha)

    def test_create_only_files(self):
        out = self.tmp / "normalizers"
        res = nz.write(self.n, out)
        jpath, bpath = out / f"{self.n.sha256}.json", out / f"{self.n.blob_sha256}.f32le"
        self.assertEqual((res["header"]["existed"], res["blob"]["existed"]), (False, False))
        for path in (jpath, bpath):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
        self.assertEqual(jpath.read_bytes(), self.n.header_bytes)
        self.assertEqual(nz.load(out, self.n.sha256), self.n)
        res2 = nz.write(self.n, out)
        self.assertEqual((res2["header"]["existed"], res2["blob"]["existed"]), (True, True))
        other = self.tmp / "tampered"
        common.immutable_write(other / f"{self.n.blob_sha256}.f32le", b"not the blob")
        with self.assertRaises(common.ImmutableConflict):
            nz.write(self.n, other)


# ------------------------------------------------------------------------------------------------ T16
class T16Standardization(NormCase):
    def test_fit_rows_standardized_constant_dims_and_clip(self):
        x = np.stack([np.asarray(r.vector, dtype=np.float64) for r in self.rows])
        mean = self.n.arrays["mean"].astype(np.float64)
        scale = self.n.arrays["scale"].astype(np.float64)
        self.assertEqual(self.n.constant_dims, list(CONST_DIMS))
        self.assertTrue(np.all(scale[list(CONST_DIMS)] == 1.0))
        z = (x - mean) / scale
        live = [i for i in range(D) if i not in CONST_DIMS]
        self.assertLess(float(np.max(np.abs(z[:, live].mean(axis=0)))), 1e-5)
        self.assertLess(float(np.max(np.abs(z[:, live].std(axis=0) - 1.0))), 1e-5)
        probe = x[0].copy()
        base = nz.apply(self.n, probe)
        probe[4], probe[5] = 1e3, -7.0
        self.assertTrue(np.array_equal(nz.apply(self.n, probe), base), "a constant dim contributed")
        far = x[0].copy()
        far[live[0]] += 1e4 * scale[live[0]]
        out, clipped = nz.apply_with_stats(self.n, far)
        self.assertTrue(np.all(np.abs(out) <= 32.0))
        self.assertGreater(clipped, 0)
        self.assertEqual(clipped, int(np.count_nonzero(np.abs(out) == 32.0)))
        self.assertEqual(out.dtype, np.float32)
        _, none_clipped = nz.apply_with_stats(self.n, x[1])
        self.assertEqual(none_clipped, 0)
        for bad in (np.zeros(D - 1), np.where(np.arange(D) == 0, np.nan, 0.0)):
            with self.assertRaises(nz.NormalizerError):
                nz.apply(self.n, bad)


# ------------------------------------------------------------------------------------------------ T17
class T17PCA(NormCase):
    def test_orthonormal_descending_signed(self):
        rot = self.n.arrays["rotation"].astype(np.float64)
        r = self.n.width_full
        self.assertLess(float(np.max(np.abs(rot.T @ rot - np.eye(r)))), 1e-5)
        eig = self.n.arrays["eigenvalues"]
        self.assertTrue(np.all(np.diff(eig) <= 0))
        for k in range(r):
            self.assertGreater(rot[int(np.argmax(np.abs(rot[:, k]))), k], 0.0)
        self.assertFalse(np.any(rot[list(CONST_DIMS), :]))
        self.assertEqual(r, min(self.n.rank, self.n.header["fit"]["n_contexts"] - 1))
        x = np.stack([np.asarray(row.vector, dtype=np.float64) for row in self.rows])
        z = (x - x.mean(axis=0)) / np.where(np.isin(np.arange(D), CONST_DIMS), 1.0, x.std(axis=0))
        z[:, list(CONST_DIMS)] = 0.0
        ref = np.sort(np.linalg.eigh(z.T @ z / len(x))[0])[::-1][:r]
        self.assertLess(float(np.max(np.abs(eig - ref) / ref[0])), 1e-6)

    def test_rank_cut_and_width_caps(self):
        rng = np.random.default_rng(5)
        pmap, rows = {}, []
        latent, mix = rng.standard_normal((40, 5)), rng.standard_normal((5, 20))
        x = latent @ mix
        x[:, 7] = 3.0
        for i in range(40):
            c = cid(f"lowrank:{i}")
            pmap[c] = {"family_id": f"fam-lr{i % 9}", "split_role": "train_fit"}
            rows.append(nz.FitRow(c, f"fam-lr{i % 9}", "train_fit", x[i]))
        n = fit(rows, pmap)
        self.assertEqual((n.rank, n.width_full, n.constant_dims), (5, 5, [7]))
        n3 = fit(rows, pmap, max_components=3)
        self.assertEqual((n3.rank, n3.width_full), (5, 3))
        self.assertTrue(np.array_equal(n3.arrays["rotation"], n.arrays["rotation"][:, :3]))
        few = fit(rows[:4], pmap)
        self.assertEqual(few.width_full, 3)                          # N - 1 caps the width

    def test_prefix_property_and_width_refusals(self):
        probes = [np.asarray(self.rows[0].vector, dtype=np.float64),
                  build_vectors(self.pmap, seed=9)[cid("ho0:1")], np.linspace(-5, 5, D)]
        full = self.n.width_full
        for x in probes:
            whole = nz.apply(self.n, x)
            self.assertEqual(whole.shape, (full,))
            for k in range(1, full + 1):
                self.assertTrue(np.array_equal(nz.apply(self.n, x, k), whole[:k]), k)
        for bad in (0, -1, full + 1, 1.5, True, "3"):
            with self.assertRaises(nz.NormalizerError, msg=repr(bad)):
                nz.apply(self.n, probes[0], bad)

    def test_eigh_is_not_used(self):
        with mock.patch.object(nz.np.linalg, "eigh", side_effect=AssertionError("numpy eigh used")):
            again = fit(self.rows, self.pmap)
        self.assertEqual(again, self.n)
        self.assertIn("jacobi", self.n.header["method"]["eigh"])


class JacobiEigh(fx.P3Case):
    def test_matches_lapack_on_full_and_rank_deficient_matrices(self):
        rng = np.random.default_rng(31)
        for d, k in ((1, 1), (2, 2), (3, 1), (17, 17), (30, 4), (88, 19), (64, 64)):
            b = rng.standard_normal((k, d))
            g = b.T @ b / k
            w, v = nz.jacobi_eigh(g)
            ref = np.linalg.eigh(g)[0]
            scale = max(1.0, float(np.max(np.abs(ref))))
            self.assertLess(float(np.max(np.abs(np.sort(w) - ref))) / scale, 1e-12, (d, k))
            self.assertLess(float(np.max(np.abs(v.T @ v - np.eye(d)))), 1e-12)
            self.assertLess(float(np.max(np.abs((v * w) @ v.T - g))) / scale, 1e-12)
        w0, v0 = nz.jacobi_eigh(np.zeros((5, 5)))
        self.assertTrue(np.array_equal(w0, np.zeros(5)) and np.array_equal(v0, np.eye(5)))
        with self.assertRaises(nz.NormalizerError):
            nz.jacobi_eigh(np.zeros((3, 4)))

    def test_schedule_covers_every_pair_once(self):
        for d in (2, 5, 8, 87, 88):
            pairs = [(int(p), int(q)) for ps, qs in nz._jacobi_schedule(d) for p, q in zip(ps, qs)]
            self.assertEqual(sorted(pairs), [(p, q) for p in range(d) for q in range(p + 1, d)])
            for ps, qs in nz._jacobi_schedule(d):
                touched = list(ps) + list(qs)
                self.assertEqual(len(touched), len(set(touched)))


# ------------------------------------------------------------------------------------------------ T18
class T18AllocateWidths(fx.P3Case):
    def test_proportional_floor_and_sum(self):
        blocks = [("hyperion", 512), ("m", 85)]
        self.assertEqual(nz.allocate_widths(256, blocks), {"hyperion": 220, "m": 36})
        self.assertEqual(nz.allocate_widths(597, blocks), {"hyperion": 512, "m": 85})
        three = [("hyperion", 512), ("foundation", 1024), ("m", 85)]
        self.assertEqual(nz.allocate_widths(1621, three), dict(three))
        proportional_cases = 0
        for blk in (blocks, three):
            full = sum(d for _, d in blk)
            for w_total in range(16 * len(blk), full + 1):
                got = nz.allocate_widths(w_total, blk)
                self.assertEqual(list(got), [b for b, _ in blk])
                self.assertEqual(sum(got.values()), w_total)
                self.assertTrue(all(16 <= got[b] <= d for b, d in blk), (w_total, got))
                if all(w_total * d / full >= 16 for _, d in blk):     # no block on the floor: largest remainder
                    proportional_cases += 1
                    for b, d in blk:
                        self.assertLess(abs(got[b] - w_total * d / full), 1.0, (w_total, got))
        self.assertGreater(proportional_cases, 1000)

    def test_ties_follow_block_order_and_floor_adjustment(self):
        self.assertEqual(nz.allocate_widths(200, [("a", 100), ("b", 100), ("c", 100)]), {"a": 67, "b": 67, "c": 66})
        self.assertEqual(nz.allocate_widths(200, [("c", 100), ("b", 100), ("a", 100)]), {"c": 67, "b": 67, "a": 66})
        self.assertEqual(nz.allocate_widths(100, [("a", 1000), ("b", 16), ("c", 16)]), {"a": 68, "b": 16, "c": 16})
        self.assertEqual(nz.allocate_widths(np.int64(256), [("hyperion", np.int64(512)), ("m", 85)]),
                         {"hyperion": 220, "m": 36})

    def test_refusals(self):
        blocks = [("hyperion", 512), ("m", 85)]
        for w_total, blk in ((598, blocks), (31, blocks), (40, [("a", 10), ("b", 100)]), (64, [("a", 64), ("a", 64)]),
                             (64.0, blocks), (64, [])):
            with self.assertRaises(nz.NormalizerError, msg=str((w_total, blk))):
                nz.allocate_widths(w_total, blk)


class AbiIdAndArms(NormCase):
    def test_abi_id(self):
        widths = {"m": self.n.width_full}
        got = nz.abi_id("A0", [self.n], widths)
        digest = hashlib.sha256(json.dumps({"arm": "A0", "normalizers": [self.n.sha256], "widths": widths},
                                           sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(got, "dl-v1-e0:A0:" + digest[:16])
        self.assertEqual(nz.abi_id("A0", [self.n.sha256], widths), got)
        self.assertNotEqual(nz.abi_id("A0", [self.n], {"m": self.n.width_full - 1}), got)
        with self.assertRaises(nz.NormalizerError):
            nz.abi_id("A0", ["not-a-sha"], widths)

    def test_arm_table(self):
        self.assertEqual(nz.arm_blocks("A0"), ("m",))
        self.assertEqual(nz.ARMS["A1"], ("hyperion", "m"))
        self.assertNotIn("target", [b for arm in ("A0", "A1", "A2", "A3") for b in nz.ARMS[arm]])
        self.assertIn("target", nz.ARMS["LB-replica"])
        self.assertEqual(nz.BLOCKS["foundation"]["max_components"], 1024)
        self.assertEqual(self.n.manifest_entry()["normalizer_sha256"], self.n.sha256)
        with self.assertRaises(nz.NormalizerError):
            nz.arm_blocks("A9")


# ------------------------------------------------------------------------------------------------ T19
class T19Poison(NormCase):
    def refit(self, rows, pmap):
        return fit([r for r in rows if pmap.get(r.context_sha256, {}).get("split_role") == "train_fit"], pmap)

    def test_non_train_rows_never_move_the_normalizer(self):
        everything = all_rows(self.pmap, self.vecs)
        base = self.refit(everything, self.pmap)
        self.assertEqual(base, self.n)
        rng = np.random.default_rng(77)
        poisoned = [r if self.pmap[r.context_sha256]["split_role"] == "train_fit"
                    else r._replace(vector=np.asarray(r.vector) * 1e3 + rng.standard_normal(D)) for r in everything]
        self.assertEqual(self.refit(poisoned, self.pmap).header_bytes, base.header_bytes)
        added_map = dict(self.pmap)
        extra = cid("extra-heldout")
        added_map[extra] = {"family_id": "fam-ho0", "split_role": "heldout"}
        added = everything + [nz.FitRow(extra, "fam-ho0", "heldout", rng.standard_normal(D))]
        self.assertEqual(self.refit(added, added_map).header_bytes, base.header_bytes)
        dropped = [r for r in everything if r.context_sha256 != cid("dv0:1")]
        self.assertEqual(self.refit(dropped, self.pmap).header_bytes, base.header_bytes)
        shuffled = [everything[i] for i in np.random.default_rng(5).permutation(len(everything))]
        self.assertEqual(self.refit(shuffled, self.pmap), base)

    def test_one_train_row_moves_it(self):
        everything = all_rows(self.pmap, self.vecs)
        target = cid("tf3:2")
        moved = [r._replace(vector=np.asarray(r.vector) + np.where(np.arange(D) == 11, 1e-3, 0.0))
                 if r.context_sha256 == target else r for r in everything]
        self.assertNotEqual(self.refit(moved, self.pmap).header_bytes, self.n.header_bytes)


# ------------------------------------------------------------------------------------------------ T20
CHILD = r"""
import hashlib, os
import numpy as np
from dltrain import normalizers as nz
rng = np.random.default_rng(20261005)
n_rows, d = 300, 88
x = np.einsum("nk,kd->nd", rng.standard_normal((n_rows, 12)), rng.standard_normal((12, d)), optimize=False)
x = x + 0.1 * rng.standard_normal((n_rows, d))
x[:, 7] = 0.25
pmap, rows = {}, []
for i in range(n_rows):
    ctx = hashlib.sha256(f"t20:{i}".encode()).hexdigest()
    pmap[ctx] = {"family_id": f"fam-{i // 3}", "split_role": "train_fit"}
    rows.append(nz.FitRow(ctx, f"fam-{i // 3}", "train_fit", x[i]))
n = nz.fit_block("m", rows, partition_map=pmap, arm="A0", block_version="v1", epoch_sha256="0" * 64,
                 partition_map_sha256="1" * 64, partition_map_kind="fake")
print(os.environ["OMP_NUM_THREADS"], os.environ["OPENBLAS_NUM_THREADS"], n.sha256, n.width_full)
"""


class T20ThreadIndependence(fx.P3Case):
    def test_normalizer_bytes_do_not_depend_on_thread_count(self):
        seen = {}
        for threads in ("1", "4"):
            env = dict(os.environ, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
            out = subprocess.run([sys.executable, "-c", CHILD], env=env, cwd=str(LANE), capture_output=True, text=True,
                                 timeout=900)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            t1, t2, sha, width = out.stdout.split()
            self.assertEqual((t1, t2), (threads, threads))
            self.assertGreater(int(width), 1)
            seen[threads] = sha
        self.assertEqual(seen["1"], seen["4"])


# ------------------------------------------------------------------------------------------------ T39 / constants
class T39HygieneNormalizers(fx.P3Case):
    def test_forbidden_imports_top_level_s3_and_string(self):
        text = SRC.read_text()
        tree = ast.parse(text)
        imported, top = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported |= {node.module or ""} | {f"{node.module}.{a.name}" for a in node.names}
        for node in tree.body:
            if isinstance(node, ast.Import):
                top |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                top |= {node.module or ""} | {f"{node.module}.{a.name}" for a in node.names}
        self.assertEqual({m.split(".")[0] for m in imported} & FORBIDDEN_MODULES, set())
        self.assertEqual({t for t in top if t.split(".")[0] in {"boto3", "botocore"} or t == "dltrain.s3io"}, set())
        self.assertEqual({m for m in imported if m.startswith("dltrain.")}, {"dltrain.common"})
        self.assertNotIn(FORBIDDEN_STRING, text)

    def test_partition_constants_match_programs(self):
        self.assertEqual(nz.PARTITIONS, programs.PARTITIONS)
        self.assertEqual(nz.EXCLUDED_ROLES, programs.EXCLUDED_ROLES)
        self.assertEqual((nz.SCHEMA, nz.CLIP_ABS, nz.RANK_RTOL, nz.MIN_BLOCK_WIDTH),
                         ("jope.dl-normalizer/v1", 32.0, 1e-10, 16))
